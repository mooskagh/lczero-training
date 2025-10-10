#include "loader/stages/debug_chunk_source_generator.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace lczero {
namespace training {

TEST(DebugChunkSourceGeneratorTest, EmitsInitialSourcesAndMarker) {
  DebugChunkSourceGeneratorConfig config;
  config.set_mean_chunks_per_chunk_source(10.0);
  config.set_initial_chunk_sources(3);
  config.set_chunk_sources_per_minute(6000.0);

  DebugChunkSourceGenerator generator(config);
  generator.Start();

  auto* queue = generator.output();
  std::vector<uint64_t> initial_ids;
  for (int i = 0; i < 3; ++i) {
    auto item = queue->Get();
    ASSERT_NE(item.source, nullptr);
    EXPECT_EQ(item.message_type, FilePathProvider::MessageType::kFile);
    uint64_t id = 0;
    ASSERT_NO_THROW(id = std::stoull(item.source->GetChunkSortKey()));
    initial_ids.push_back(id);
  }
  std::sort(initial_ids.begin(), initial_ids.end());
  EXPECT_EQ(initial_ids, (std::vector<uint64_t>{0, 1, 2}));

  auto marker = queue->Get();
  EXPECT_EQ(marker.source, nullptr);
  EXPECT_EQ(marker.message_type,
            FilePathProvider::MessageType::kInitialScanComplete);

  auto next = queue->Get();
  ASSERT_NE(next.source, nullptr);
  EXPECT_EQ(next.message_type, FilePathProvider::MessageType::kFile);
  uint64_t next_id = 0;
  ASSERT_NO_THROW(next_id = std::stoull(next.source->GetChunkSortKey()));
  EXPECT_EQ(next_id, 3);

  generator.Stop();
}

}  // namespace training
}  // namespace lczero
