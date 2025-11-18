#include "loader/chunk_source/rawfile_chunk_source.h"

#include <absl/log/log.h>

#include <fstream>
#include <stdexcept>

#include "utils/files.h"
#include "utils/gz.h"

namespace lczero {
namespace training {

RawFileChunkSource::RawFileChunkSource(const std::filesystem::path& filename)
    : filename_(filename) {}

RawFileChunkSource::~RawFileChunkSource() = default;

std::string RawFileChunkSource::GetChunkSortKey() const {
  return std::filesystem::path(filename_).filename().string();
}

size_t RawFileChunkSource::GetChunkCount() const { return 1; }

std::optional<std::vector<V6TrainingData>> RawFileChunkSource::GetChunkData(
    size_t index) {
  if (index != 0) return std::nullopt;
  std::string data = ReadFileToString(filename_);
  if (data.empty()) return std::nullopt;
  if (data.size() % sizeof(V6TrainingData) != 0) {
    LOG(WARNING) << "File " << filename_ << " size " << data.size()
                 << " is not a multiple of V6TrainingData size "
                 << sizeof(V6TrainingData);
    return std::nullopt;
  }
  std::vector<V6TrainingData> result(data.size() / sizeof(V6TrainingData));
  std::memcpy(result.data(), data.data(), data.size());
  return result;
}

}  // namespace training
}  // namespace lczero
