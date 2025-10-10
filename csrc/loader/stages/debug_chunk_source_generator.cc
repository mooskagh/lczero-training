#include "loader/stages/debug_chunk_source_generator.h"

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/random/random.h"
#include "absl/random/seed_sequences.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "loader/data_loader_metrics.h"

namespace lczero {
namespace training {

namespace {
constexpr uint64_t kDefaultQueueCapacity = 16;
constexpr uint64_t kInitialShuffleSeed = 0xC0FFEEull;
constexpr absl::Duration kStopPollInterval = absl::Milliseconds(10);
}  // namespace

DebugChunkSourceGenerator::DebugChunkSourceGenerator(
    const DebugChunkSourceGeneratorConfig& config,
    const Stage::StageList& existing_stages)
    : config_(config),
      output_queue_(static_cast<size_t>(std::max<uint64_t>(
          config.initial_chunk_sources(), kDefaultQueueCapacity))),
      mean_chunk_count_(std::max(1.0, config.mean_chunks_per_chunk_source())) {
  (void)existing_stages;
  if (config.mean_chunks_per_chunk_source() <= 0.0) {
    LOG(WARNING) << "DebugChunkSourceGenerator mean chunk count not positive."
                 << " Using 1.";
  }
}

DebugChunkSourceGenerator::~DebugChunkSourceGenerator() { Stop(); }

void DebugChunkSourceGenerator::Start() {
  if (worker_.joinable()) {
    return;
  }
  worker_ = std::jthread(
      [this](std::stop_token stop_token) { Run(std::move(stop_token)); });
}

void DebugChunkSourceGenerator::Stop() {
  if (!worker_.joinable()) return;
  worker_.request_stop();
  worker_.join();
}

Queue<DebugChunkSourceGenerator::OutputType>*
DebugChunkSourceGenerator::output() {
  return &output_queue_;
}

QueueBase* DebugChunkSourceGenerator::GetOutput(std::string_view name) {
  (void)name;
  return &output_queue_;
}

StageMetricProto DebugChunkSourceGenerator::FlushMetrics() {
  StageMetricProto metric;
  metric.set_stage_type("debug_chunk_source_generator");
  *metric.add_queue_metrics() = MetricsFromQueue("output", output_queue_);
  auto* count_metric = metric.add_count_metrics();
  count_metric->set_name("chunk_sources_generated");
  count_metric->set_count(generated_sources_.load(std::memory_order_relaxed));
  return metric;
}

void DebugChunkSourceGenerator::Run(std::stop_token stop_token) {
  try {
    auto producer = output_queue_.CreateProducer();
    absl::Cleanup close_queue = [&] { output_queue_.Close(); };

    std::vector<uint64_t> initial_ids(config_.initial_chunk_sources());
    std::iota(initial_ids.begin(), initial_ids.end(), 0);
    if (!initial_ids.empty()) {
      absl::SeedSeq seed({static_cast<uint32_t>(kInitialShuffleSeed),
                          static_cast<uint32_t>(kInitialShuffleSeed >> 32)});
      absl::BitGen bitgen(seed);
      absl::c_shuffle(initial_ids, bitgen);
    }

    auto emit_source = [&](uint64_t id) {
      auto source = std::make_unique<DebugChunkSource>(id, mean_chunk_count_);
      producer.Put({.source = std::move(source),
                    .message_type = FilePathProvider::MessageType::kFile});
      generated_sources_.fetch_add(1, std::memory_order_relaxed);
    };

    for (uint64_t id : initial_ids) {
      if (stop_token.stop_requested()) return;
      emit_source(id);
    }

    if (stop_token.stop_requested()) return;

    producer.Put(
        {.source = nullptr,
         .message_type = FilePathProvider::MessageType::kInitialScanComplete});

    const double per_minute = config_.chunk_sources_per_minute();
    if (per_minute <= 0.0) return;

    const absl::Duration cadence = absl::Seconds(60.0 / per_minute);
    uint64_t next_id = config_.initial_chunk_sources();
    absl::Time next_deadline = absl::Now();

    while (!stop_token.stop_requested()) {
      emit_source(next_id++);
      next_deadline += cadence;
      while (!stop_token.stop_requested()) {
        const absl::Duration wait = next_deadline - absl::Now();
        if (wait <= absl::ZeroDuration()) break;
        const absl::Duration sleep =
            wait < kStopPollInterval ? wait : kStopPollInterval;
        absl::SleepFor(sleep);
      }
    }
  } catch (const QueueClosedException&) {
    LOG(INFO) << "DebugChunkSourceGenerator stopping due to closed queue.";
  }
}

}  // namespace training
}  // namespace lczero
