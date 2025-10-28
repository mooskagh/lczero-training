#include "loader/stages/simple_chunk_extractor.h"

#include <absl/algorithm/container.h>
#include <absl/log/log.h>

#include <numeric>

#include "loader/data_loader_metrics.h"

namespace lczero {
namespace training {

SimpleChunkExtractor::SimpleChunkExtractor(
    const SimpleChunkExtractorConfig& config,
    const StageRegistry& existing_stages)
    : SingleInputStage<SimpleChunkExtractorConfig, ChunkSourceWithPhase>(
          config, existing_stages),
      SingleOutputStage<TrainingChunk>(config.output()),
      bitgen_(absl::MakeSeedSeq()) {}

SimpleChunkExtractor::~SimpleChunkExtractor() { Stop(); }

void SimpleChunkExtractor::Start() {
  worker_thread_ = std::jthread([this]() { Worker(); });
}

void SimpleChunkExtractor::Stop() {
  if (stop_requested_.exchange(true)) return;
  input_queue()->Close();
  output_queue()->Close();
}

void SimpleChunkExtractor::Worker() {
  auto producer = output_queue()->CreateProducer();

  try {
    while (true) {
      auto item = input_queue()->Get();
      if (item.message_type != FilePathProvider::MessageType::kFile ||
          !item.source) {
        continue;
      }

      ProcessSource(producer, std::move(item.source));
    }
  } catch (const QueueClosedException&) {
  }
}

void SimpleChunkExtractor::ProcessSource(
    Queue<TrainingChunk>::Producer& producer,
    std::unique_ptr<ChunkSource> source) {
  const size_t chunk_count = source->GetChunkCount();
  if (chunk_count == 0) return;

  std::vector<size_t> indices(chunk_count);
  std::iota(indices.begin(), indices.end(), 0);
  absl::c_shuffle(indices, bitgen_);

  const std::string sort_key = source->GetChunkSortKey();
  for (size_t idx : indices) {
    if (auto chunk = LoadChunk(*source, sort_key, idx)) {
      producer.Put(std::move(*chunk));
      ++chunks_processed_;
    }
  }
  ++sources_processed_;
}

std::optional<TrainingChunk> SimpleChunkExtractor::LoadChunk(
    ChunkSource& source, const std::string& sort_key, size_t index) {
  auto data = source.GetChunkData(index);
  if (!data || data->empty() || data->size() % sizeof(FrameType) != 0) {
    if (data && !data->empty()) {
      LOG(WARNING) << "Invalid chunk size " << data->size() << " from "
                   << sort_key << " at index " << index;
    }
    ++chunks_dropped_;
    return std::nullopt;
  }

  TrainingChunk chunk;
  chunk.sort_key = sort_key;
  chunk.index_within_sort_key = index;
  chunk.global_index = chunks_processed_;
  chunk.use_count = 0;

  const auto* frames_begin = reinterpret_cast<const FrameType*>(data->data());
  const auto* frames_end = frames_begin + data->size() / sizeof(FrameType);
  chunk.frames.assign(frames_begin, frames_end);

  return chunk;
}

StageMetricProto SimpleChunkExtractor::FlushMetrics() {
  StageMetricProto metric;
  metric.set_stage_type("simple_chunk_extractor");

  auto add_count = [&](const char* name, std::atomic<uint64_t>& counter) {
    auto* m = metric.add_count_metrics();
    m->set_name(name);
    m->set_count(counter.exchange(0));
  };

  add_count("chunks_processed", chunks_processed_);
  add_count("chunks_dropped", chunks_dropped_);
  add_count("sources_processed", sources_processed_);

  *metric.add_queue_metrics() = MetricsFromQueue("output", *output_queue());
  return metric;
}

}  // namespace training
}  // namespace lczero
