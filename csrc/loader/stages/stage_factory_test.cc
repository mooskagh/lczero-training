#include "loader/stages/stage_factory.h"

#include <gtest/gtest.h>

#include <stdexcept>

namespace lczero {
namespace training {

TEST(StageFactoryTest, CreatesFilePathProviderStage) {
  StageConfig config;
  config.mutable_file_path_provider()->set_directory(".");

  auto stage = CreateStage(config, {});

  ASSERT_NE(stage, nullptr);
  EXPECT_NE(stage->GetOutput(), nullptr);
}

TEST(StageFactoryTest, ThrowsWhenNoStageConfigSet) {
  StageConfig config;

  EXPECT_THROW(CreateStage(config, {}), std::runtime_error);
}

TEST(StageFactoryTest, ThrowsWhenMultipleStageConfigsSet) {
  StageConfig config;
  config.mutable_file_path_provider()->set_directory(".");
  config.mutable_tensor_generator();

  EXPECT_THROW(CreateStage(config, {}), std::runtime_error);
}

TEST(StageFactoryTest, CreatesDebugChunkSourceGeneratorStage) {
  StageConfig config;
  auto* debug_config = config.mutable_debug_chunk_source_generator();
  debug_config->set_mean_chunks_per_chunk_source(10.0);
  debug_config->set_initial_chunk_sources(2);
  debug_config->set_chunk_sources_per_minute(60.0);

  auto stage = CreateStage(config, {});

  ASSERT_NE(stage, nullptr);
  EXPECT_NE(stage->GetOutput(), nullptr);
}

}  // namespace training
}  // namespace lczero
