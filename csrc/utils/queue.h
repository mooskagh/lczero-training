#pragma once

#include <algorithm>
#include <stdexcept>

#include "absl/base/thread_annotations.h"
#include "absl/container/fixed_array.h"
#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"

namespace lczero {

// Virtual base class for type-erased handling of queues.
class QueueBase {
 public:
  virtual ~QueueBase() = default;
  virtual size_t Size() const = 0;
  virtual size_t Capacity() const = 0;
  virtual bool IsClosed() const = 0;
  virtual void Close() = 0;
  virtual size_t GetTotalPutCount(bool reset = false) = 0;
  virtual size_t GetTotalGetCount(bool reset = false) = 0;
  virtual size_t GetTotalDropCount(bool reset = false) = 0;
};

// Exception thrown when queue operations are attempted on a closed queue.
class QueueClosedException : public std::runtime_error {
 public:
  QueueClosedException() : std::runtime_error("Queue is closed") {}
};

enum class OverflowBehavior { BLOCK, DROP_NEW, KEEP_NEWEST };

// Thread-safe fixed-size circular buffer queue with blocking operations.
// Supports both single and batch put/get operations.
// The queue automatically closes when all Producer tokens are destroyed.
// When closed, Put operations throw immediately, but Get operations only throw
// when the queue becomes empty - allowing consumption of remaining elements.
template <typename T>
class Queue : public QueueBase {
 public:
  // Backwards-compatible alias to support code referring to
  // Queue<T>::OverflowBehavior.
  using OverflowBehavior = ::lczero::OverflowBehavior;

  explicit Queue(size_t capacity,
                 OverflowBehavior overflow_behavior = OverflowBehavior::BLOCK);

  // RAII token for producers. Queue automatically closes when all producers
  // are destroyed. All Put operations must go through this class.
  class Producer {
   public:
    explicit Producer(Queue<T>& queue);
    ~Producer();

    // Move constructor and assignment
    Producer(Producer&& other) noexcept;
    Producer& operator=(Producer&& other) noexcept;

    // Disable copy to maintain RAII semantics
    Producer(const Producer&) = delete;
    Producer& operator=(const Producer&) = delete;

    // Puts a single element into the queue. Blocks if queue is full.
    void Put(const T& item);
    void Put(T&& item);

    // Puts multiple elements into the queue. Blocks if not enough space.
    void Put(absl::Span<const T> items);
    void Put(absl::Span<T> items);

    // Explicitly close this producer, decrementing the producer count
    void Close();

   private:
    Queue<T>* queue_;
  };

  // Creates a new producer token for this queue.
  Producer CreateProducer();

  // Gets a single element from the queue. Blocks if queue is empty.
  T Get();

  // Gets exactly count elements from the queue. Blocks until count elements
  // available.
  absl::FixedArray<T> Get(size_t count);

  // Returns the current size of the queue.
  size_t Size() const override;

  // Returns the capacity of the queue.
  size_t Capacity() const override;

  // Explicitly close the queue, preventing further Put operations.
  void Close() override;

  // Returns true if the queue is closed.
  bool IsClosed() const override;

  // Wait until queue has at least the specified amount of free space.
  void WaitForRoomAtLeast(size_t room);

  // Wait until queue has at most the specified amount of free space.
  void WaitForRoomAtMost(size_t room);

  // Wait until queue has at least the specified number of elements.
  void WaitForSizeAtLeast(size_t size);

  // Wait until queue has at most the specified number of elements.
  void WaitForSizeAtMost(size_t size);

  // Returns the total number of elements that have been put into the queue.
  // If reset is true, resets the counter to 0 after returning the value.
  size_t GetTotalPutCount(bool reset = false) override;

  // Returns the total number of elements that have been retrieved from the
  // queue. If reset is true, resets the counter to 0 after returning the value.
  size_t GetTotalGetCount(bool reset = false) override;

  // Returns the total number of elements that have been dropped from the queue.
  // If reset is true, resets the counter to 0 after returning the value.
  size_t GetTotalDropCount(bool reset = false) override;

 private:
  friend class Producer;

  const size_t capacity_;
  const OverflowBehavior overflow_behavior_;
  absl::FixedArray<T> buffer_ ABSL_GUARDED_BY(mutex_);
  size_t head_ ABSL_GUARDED_BY(mutex_) = 0;
  size_t tail_ ABSL_GUARDED_BY(mutex_) = 0;
  size_t size_ ABSL_GUARDED_BY(mutex_) = 0;
  size_t producer_count_ ABSL_GUARDED_BY(mutex_) = 0;
  bool closed_ ABSL_GUARDED_BY(mutex_) = false;
  size_t total_put_count_ ABSL_GUARDED_BY(mutex_) = 0;
  size_t total_get_count_ ABSL_GUARDED_BY(mutex_) = 0;
  size_t total_drop_count_ ABSL_GUARDED_BY(mutex_) = 0;

  mutable absl::Mutex mutex_;

  // Internal methods for producer management
  void RemoveProducer();

  // Internal Put methods (called by Producer)
  void PutInternal(const T& item);
  void PutInternal(T&& item);
  void PutInternal(absl::Span<const T> items);
  void PutInternal(absl::Span<T> items);

  // Condition predicates for blocking operations
  bool CanPutOne() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  bool CanGet() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Additional condition predicates for wait functions
  bool HasRoomAtLeast(size_t room) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  bool HasRoomAtMost(size_t room) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  bool HasSizeAtLeast(size_t size) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  bool HasSizeAtMost(size_t size) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
};

// Implementation

template <typename T>
Queue<T>::Queue(size_t capacity, OverflowBehavior overflow_behavior)
    : capacity_(capacity),
      overflow_behavior_(overflow_behavior),
      buffer_(capacity) {}

// Producer implementation
template <typename T>
Queue<T>::Producer::Producer(Queue<T>& queue) : queue_(&queue) {
  // Producer count is incremented in CreateProducer()
  LOG(INFO) << "Queue@" << static_cast<const void*>(queue_) << " producer@"
            << static_cast<const void*>(this) << " constructed.";
}

template <typename T>
Queue<T>::Producer::~Producer() {
  if (queue_) {
    LOG(INFO) << "Queue@" << static_cast<const void*>(queue_) << " producer@"
              << static_cast<const void*>(this) << " destructing.";
    queue_->RemoveProducer();
  }
}

template <typename T>
Queue<T>::Producer::Producer(Producer&& other) noexcept : queue_(other.queue_) {
  other.queue_ = nullptr;
}

template <typename T>
typename Queue<T>::Producer& Queue<T>::Producer::operator=(
    Producer&& other) noexcept {
  if (this != &other) {
    if (queue_) {
      queue_->RemoveProducer();
    }
    queue_ = other.queue_;
    other.queue_ = nullptr;
  }
  return *this;
}

template <typename T>
void Queue<T>::Producer::Put(const T& item) {
  queue_->PutInternal(item);
}

template <typename T>
void Queue<T>::Producer::Put(T&& item) {
  queue_->PutInternal(std::move(item));
}

template <typename T>
void Queue<T>::Producer::Put(absl::Span<const T> items) {
  queue_->PutInternal(items);
}

template <typename T>
void Queue<T>::Producer::Put(absl::Span<T> items) {
  queue_->PutInternal(items);
}

template <typename T>
void Queue<T>::Producer::Close() {
  if (queue_) {
    LOG(INFO) << "Queue@" << static_cast<const void*>(queue_) << " producer@"
              << static_cast<const void*>(this) << " close invoked.";
    queue_->RemoveProducer();
    queue_ = nullptr;
  }
}

// Queue implementation
template <typename T>
typename Queue<T>::Producer Queue<T>::CreateProducer() {
  absl::MutexLock lock(&mutex_);
  if (closed_) throw QueueClosedException();
  ++producer_count_;
  return Producer(*this);
}

template <typename T>
void Queue<T>::RemoveProducer() {
  absl::MutexLock lock(&mutex_);
  --producer_count_;
  if (producer_count_ == 0 && !closed_) {
    closed_ = true;
    LOG(INFO) << "Queue@" << static_cast<const void*>(this)
              << " closed after last producer removed.";
  }
}

template <typename T>
void Queue<T>::PutInternal(const T& item) {
  absl::MutexLock lock(&mutex_);
  if (closed_) {
    LOG(INFO) << "Queue@" << static_cast<const void*>(this)
              << " PutInternal(const&) throwing QueueClosedException;"
              << " producers=" << producer_count_;
    throw QueueClosedException();
  }
  ++total_put_count_;

  switch (overflow_behavior_) {
    case OverflowBehavior::BLOCK:
      mutex_.Await(absl::Condition(this, &Queue<T>::CanPutOne));
      // Second check needed: queue might have closed while waiting.
      if (closed_) throw QueueClosedException();
      break;
    case OverflowBehavior::DROP_NEW:
      // No blocking: return immediately if full.
      if (size_ >= capacity_) {
        ++total_drop_count_;
        return;
      }
      break;
    case OverflowBehavior::KEEP_NEWEST:
      // No blocking: make room by dropping oldest item.
      if (size_ >= capacity_) {
        head_ = (head_ + 1) % capacity_;
        --size_;
        ++total_drop_count_;
      }
      break;
  }

  buffer_[tail_] = item;
  tail_ = (tail_ + 1) % capacity_;
  ++size_;
}

template <typename T>
void Queue<T>::PutInternal(T&& item) {
  absl::MutexLock lock(&mutex_);
  if (closed_) {
    LOG(INFO) << "Queue@" << static_cast<const void*>(this)
              << " PutInternal(T&&) throwing QueueClosedException;"
              << " producers=" << producer_count_;
    throw QueueClosedException();
  }
  ++total_put_count_;

  switch (overflow_behavior_) {
    case OverflowBehavior::BLOCK:
      mutex_.Await(absl::Condition(this, &Queue<T>::CanPutOne));
      // Second check needed: queue might have closed while waiting.
      if (closed_) throw QueueClosedException();
      break;
    case OverflowBehavior::DROP_NEW:
      // No blocking: return immediately if full.
      if (size_ >= capacity_) {
        ++total_drop_count_;
        return;
      }
      break;
    case OverflowBehavior::KEEP_NEWEST:
      // No blocking: make room by dropping oldest item.
      if (size_ >= capacity_) {
        head_ = (head_ + 1) % capacity_;
        --size_;
        ++total_drop_count_;
      }
      break;
  }

  buffer_[tail_] = std::move(item);
  tail_ = (tail_ + 1) % capacity_;
  ++size_;
}

template <typename T>
void Queue<T>::PutInternal(absl::Span<const T> items) {
  if (items.empty()) return;

  size_t remaining = items.size();
  size_t offset = 0;

  while (remaining > 0) {
    absl::MutexLock lock(&mutex_);
    if (closed_) {
      LOG(INFO) << "Queue@" << static_cast<const void*>(this)
                << " PutInternal(span const) throwing QueueClosedException;"
                << " producers=" << producer_count_;
      throw QueueClosedException();
    }

    size_t batch_size;
    switch (overflow_behavior_) {
      case OverflowBehavior::BLOCK:
        mutex_.Await(absl::Condition(this, &Queue<T>::CanPutOne));
        // Second check needed: queue might have closed while waiting.
        if (closed_) throw QueueClosedException();
        batch_size = std::min(remaining, capacity_ - size_);
        break;
      case OverflowBehavior::DROP_NEW:
        // No blocking: process only what fits.
        batch_size = std::min(remaining, capacity_ - size_);
        if (batch_size == 0) {
          total_put_count_ += remaining;
          total_drop_count_ += remaining;
          return;
        }
        break;
      case OverflowBehavior::KEEP_NEWEST:
        // No blocking: make room by dropping oldest items.
        batch_size = std::min(remaining, capacity_);
        while (size_ + batch_size > capacity_) {
          head_ = (head_ + 1) % capacity_;
          --size_;
          ++total_drop_count_;
        }
        break;
    }

    for (size_t i = 0; i < batch_size; ++i) {
      buffer_[tail_] = items[offset + i];
      tail_ = (tail_ + 1) % capacity_;
      ++size_;
    }
    total_put_count_ += batch_size;

    offset += batch_size;
    remaining -= batch_size;
  }
}

template <typename T>
void Queue<T>::PutInternal(absl::Span<T> items) {
  if (items.empty()) return;

  size_t remaining = items.size();
  size_t offset = 0;

  while (remaining > 0) {
    absl::MutexLock lock(&mutex_);
    if (closed_) {
      LOG(INFO) << "Queue@" << static_cast<const void*>(this)
                << " PutInternal(span) throwing QueueClosedException;"
                << " producers=" << producer_count_;
      throw QueueClosedException();
    }

    size_t batch_size;
    switch (overflow_behavior_) {
      case OverflowBehavior::BLOCK:
        mutex_.Await(absl::Condition(this, &Queue<T>::CanPutOne));
        // Second check needed: queue might have closed while waiting.
        if (closed_) throw QueueClosedException();
        batch_size = std::min(remaining, capacity_ - size_);
        break;
      case OverflowBehavior::DROP_NEW:
        // No blocking: process only what fits.
        batch_size = std::min(remaining, capacity_ - size_);
        if (batch_size == 0) {
          total_put_count_ += remaining;
          total_drop_count_ += remaining;
          return;
        }
        break;
      case OverflowBehavior::KEEP_NEWEST:
        // No blocking: make room by dropping oldest items.
        batch_size = std::min(remaining, capacity_);
        while (size_ + batch_size > capacity_) {
          head_ = (head_ + 1) % capacity_;
          --size_;
          ++total_drop_count_;
        }
        break;
    }

    for (size_t i = 0; i < batch_size; ++i) {
      buffer_[tail_] = std::move(items[offset + i]);
      tail_ = (tail_ + 1) % capacity_;
      ++size_;
    }
    total_put_count_ += batch_size;

    offset += batch_size;
    remaining -= batch_size;
  }
}

template <typename T>
T Queue<T>::Get() {
  absl::MutexLock lock(&mutex_);
  mutex_.Await(absl::Condition(this, &Queue<T>::CanGet));
  if (closed_ && size_ == 0) {
    LOG(INFO) << "Queue@" << static_cast<const void*>(this)
              << " Get() throwing QueueClosedException; producers="
              << producer_count_;
    throw QueueClosedException();
  }

  T item = std::move(buffer_[head_]);
  head_ = (head_ + 1) % capacity_;
  --size_;
  ++total_get_count_;

  return item;
}

template <typename T>
absl::FixedArray<T> Queue<T>::Get(size_t count) {
  if (count == 0) return absl::FixedArray<T>(0);

  absl::FixedArray<T> result(count);
  size_t remaining = count;
  size_t offset = 0;

  while (remaining > 0) {
    absl::MutexLock lock(&mutex_);
    mutex_.Await(absl::Condition(this, &Queue<T>::CanGet));
    if (closed_ && size_ == 0) {
      LOG(INFO) << "Queue@" << static_cast<const void*>(this) << " Get("
                << count << ") throwing QueueClosedException; producers="
                << producer_count_;
      throw QueueClosedException();
    }

    // Get as many items as possible in this batch
    size_t batch_size = std::min(remaining, size_);

    for (size_t i = 0; i < batch_size; ++i) {
      result[offset + i] = std::move(buffer_[head_]);
      head_ = (head_ + 1) % capacity_;
      --size_;
      ++total_get_count_;
    }

    offset += batch_size;
    remaining -= batch_size;
  }

  return result;
}

template <typename T>
size_t Queue<T>::Size() const {
  absl::MutexLock lock(&mutex_);
  return size_;
}

template <typename T>
size_t Queue<T>::Capacity() const {
  return capacity_;
}

template <typename T>
void Queue<T>::Close() {
  absl::MutexLock lock(&mutex_);
  if (!closed_) {
    closed_ = true;
    LOG(INFO) << "Queue@" << static_cast<const void*>(this)
              << " closed explicitly; producers=" << producer_count_;
  }
}

template <typename T>
bool Queue<T>::IsClosed() const {
  absl::MutexLock lock(&mutex_);
  return closed_;
}

template <typename T>
void Queue<T>::WaitForRoomAtLeast(size_t room) {
  struct Args {
    Queue<T>* queue;
    size_t room;
  };
  Args args{this, room};
  absl::MutexLock lock(&mutex_);
  mutex_.Await(absl::Condition(
      +[](void* data) -> bool {
        auto* args = static_cast<Args*>(data);
        return args->queue->HasRoomAtLeast(args->room);
      },
      &args));
}

template <typename T>
void Queue<T>::WaitForRoomAtMost(size_t room) {
  struct Args {
    Queue<T>* queue;
    size_t room;
  };
  Args args{this, room};
  absl::MutexLock lock(&mutex_);
  mutex_.Await(absl::Condition(
      +[](void* data) -> bool {
        auto* args = static_cast<Args*>(data);
        return args->queue->HasRoomAtMost(args->room);
      },
      &args));
}

template <typename T>
void Queue<T>::WaitForSizeAtLeast(size_t size) {
  struct Args {
    Queue<T>* queue;
    size_t size;
  };
  Args args{this, size};
  absl::MutexLock lock(&mutex_);
  mutex_.Await(absl::Condition(
      +[](void* data) -> bool {
        auto* args = static_cast<Args*>(data);
        return args->queue->HasSizeAtLeast(args->size);
      },
      &args));
}

template <typename T>
void Queue<T>::WaitForSizeAtMost(size_t size) {
  struct Args {
    Queue<T>* queue;
    size_t size;
  };
  Args args{this, size};
  absl::MutexLock lock(&mutex_);
  mutex_.Await(absl::Condition(
      +[](void* data) -> bool {
        auto* args = static_cast<Args*>(data);
        return args->queue->HasSizeAtMost(args->size);
      },
      &args));
}

template <typename T>
bool Queue<T>::CanPutOne() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
  return closed_ || size_ < capacity_;
}

template <typename T>
bool Queue<T>::CanGet() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
  return closed_ || size_ > 0;
}

template <typename T>
bool Queue<T>::HasRoomAtLeast(size_t room)
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
  return capacity_ - size_ >= room;
}

template <typename T>
bool Queue<T>::HasRoomAtMost(size_t room)
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
  return capacity_ - size_ <= room;
}

template <typename T>
bool Queue<T>::HasSizeAtLeast(size_t size)
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
  return size_ >= size;
}

template <typename T>
bool Queue<T>::HasSizeAtMost(size_t size)
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
  return size_ <= size;
}

template <typename T>
size_t Queue<T>::GetTotalPutCount(bool reset) {
  absl::MutexLock lock(&mutex_);
  size_t count = total_put_count_;
  if (reset) total_put_count_ = 0;
  return count;
}

template <typename T>
size_t Queue<T>::GetTotalGetCount(bool reset) {
  absl::MutexLock lock(&mutex_);
  size_t count = total_get_count_;
  if (reset) total_get_count_ = 0;
  return count;
}

template <typename T>
size_t Queue<T>::GetTotalDropCount(bool reset) {
  absl::MutexLock lock(&mutex_);
  size_t count = total_drop_count_;
  if (reset) total_drop_count_ = 0;
  return count;
}

}  // namespace lczero
