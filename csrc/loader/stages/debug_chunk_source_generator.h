#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <stop_token>
#include <string_view>
#include <thread>

#include "loader/chunk_source/chunk_source.h"
#include "loader/chunk_source/debug_chunk_source.h"
#include "loader/stages/chunk_source_loader.h"
#include "loader/stages/stage.h"
#include "proto/data_loader_config.pb.h"
#include "proto/training_metrics.pb.h"
#include "utils/queue.h"

namespace lczero {
namespace training {

// DebugChunkSourceGenerator emits deterministic DebugChunkSource instances.
// It is intended for loader bring-up and testing without filesystem input.
class DebugChunkSourceGenerator : public Stage {
 public:
  using OutputType = ChunkSourceWithPhase;

  explicit DebugChunkSourceGenerator(
      const DebugChunkSourceGeneratorConfig& config,
      const Stage::StageList& existing_stages = {});
  ~DebugChunkSourceGenerator() override;

  void Start() override;
  void Stop() override;

  StageMetricProto FlushMetrics() override;

  QueueBase* GetOutput(std::string_view name = "") override;
  Queue<OutputType>* output();

 private:
  void Run(std::stop_token stop_token);

  const DebugChunkSourceGeneratorConfig config_;
  Queue<OutputType> output_queue_;
  std::jthread worker_;
  std::atomic<uint64_t> generated_sources_{0};
  const double mean_chunk_count_;
};

}  // namespace training
}  // namespace lczero
