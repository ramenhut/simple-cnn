
#include "ensemble.h"

#include <fstream>
#include <iostream>

namespace base {

#pragma pack(push)
#pragma pack(2)

typedef struct {
  uint16 magic;  // must be 0x4244
  uint32 network_count;
} NEURAL_ENSEMBLE_HEADER;

#pragma pack(pop)

Ensemble::Ensemble(const std::string &filename) {
  std::string error;
  Load(filename, &error);
  if (error.length()) {
    std::cout << error << std::endl;
  }
}

Ensemble::Ensemble(const std::vector<Network> &networks) {
  for (auto &nn : networks) {
    if (ensemble_.empty()) {
      ensemble_.push_back(nn);
    } else {
      if (nn.GetInputLayerSize() == GetInputLayerSize() &&
          nn.GetOutputLayerSize() == GetOutputLayerSize()) {
        ensemble_.push_back(nn);
      }
    }
  }
}

uint32 Ensemble::GetInputLayerSize() const {
  assert(!ensemble_.empty());
  return ensemble_[0].GetInputLayerSize();
}

uint32 Ensemble::GetOutputLayerSize() const {
  assert(!ensemble_.empty());
  return ensemble_[0].GetOutputLayerSize();
}

Vectorf Ensemble::FeedForward(const Vectorf &sample) {
  Vectorf result(GetOutputLayerSize());
  for (auto &net : ensemble_) {
    result += net.FeedForward(sample);
  }
  result = result / ensemble_.size();
  return result;
}

float64 Ensemble::Evaluate(std::vector<SampleSet> *test_data) {
  float64 total_correct = 0.0f;
  for (auto &data : *test_data) {
    Vectorf result = FeedForward(data.sample);
    uint32 network_guess = result.GetMaxIndex();
    uint32 ground_truth = data.label.GetMaxIndex();
    if (network_guess == ground_truth) {
      total_correct += 1.0;
    }
  }
  return total_correct;
}

void Ensemble::Save(const std::string &filename, std::string *error) const {
  std::ofstream out_stream(filename, std::ios::out | std::ios::binary);

  NEURAL_ENSEMBLE_HEADER header;
  header.magic = 0x4242;
  header.network_count = ensemble_.size();

  if (!out_stream.write((char *)&header, sizeof(header))) {
    if (error) {
      *error = "Failed to write neural network ensemble header to disk.";
    }
    return;
  }

  for (auto &net : ensemble_) {
    std::string network_write_error;
    net.Save(&out_stream, &network_write_error);
    if (network_write_error.length()) {
      if (error) {
        *error = network_write_error;
      }
      return;
    }
  }
}

void Ensemble::Load(const std::string &filename, std::string *error) {
  NEURAL_ENSEMBLE_HEADER header;
  std::ifstream in_stream(filename, ::std::ios::in | ::std::ios::binary);

  if (!in_stream.read((char *)&header, sizeof(header))) {
    if (error) {
      *error = "Failed to read neural network ensemble header from disk.";
    }
    return;
  }

  if (0x4242 != header.magic || 0 == header.network_count) {
    if (error) {
      *error = "Invalid neural network ensemble file format detected.";
    }
    return;
  }

  for (uint32 i = 0; i < header.network_count; i++) {
    ensemble_.emplace_back(&in_stream);
  }
}

}  // namespace base
