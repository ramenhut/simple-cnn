
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "ensemble.h"
#include "image.h"
#include "network.h"
#include "time.h"

#if !__cplusplus > 201402L
#error "This build requires C++14 or higher."
#endif

using namespace base;

const std::string mnist_training_images = "train-images.idx3-ubyte";
const std::string mnist_training_labels = "train-labels.idx1-ubyte";
const std::string mnist_classify_images = "t10k-images.idx3-ubyte";
const std::string mnist_classify_labels = "t10k-labels.idx1-ubyte";

namespace base {

uint64 GetSystemTime() {
  return uint64(double(clock()) / CLOCKS_PER_SEC * 1000);
}

uint64 GetElapsedTimeMs(uint64 from_time) {
  return (GetSystemTime() - from_time);
}

}  // namespace base

void PrintUsage(const char* programName) {
  std::cout << "Usage: " << programName << " [options]" << std::endl;
  std::cout << "  --train  [output neural network filename]\t\t\t"
            << "Trains a single neural network based on the MNIST dataset."
            << std::endl;
  std::cout << "  --combine [output ensemble filename] <network "
               "filenames>\t"
            << "Combines a set of network files into a single ensemble file."
            << std::endl;
  std::cout
      << "  --verify [input ensemble filename] \t\t\t\tTests the accuracy of a "
         "neural network ensemble against the "
      << "MNIST test set." << std::endl;
}

void ExecuteCombine(const std::string& output_filename,
                    const std::vector<std::string>& networks) {
  std::string error;
  std::vector<Network> ensemble_networks;

  for (auto& s : networks) {
    ensemble_networks.emplace_back(s);
  }

  Ensemble ensemble(ensemble_networks);
  ensemble.Save(output_filename, &error);
  if (error.length()) {
    std::cout << error << std::endl;
  }
}

void ExecuteTraining(const std::string& output_filename) {
  std::string error;
  uint32 training_label_count = 0;
  uint32 classify_label_count = 0;
  std::vector<SampleSet> training_data;
  std::vector<SampleSet> classify_data;

  if (output_filename.empty()) {
    std::cout
        << "You must specify a valid network filename to save the trained "
           "network."
        << std::endl;
    return;
  }

  std::cout << "Loading training data..." << std::endl;

  if (!LoadImageSet(mnist_training_images, mnist_training_labels,
                    &training_data, &error)) {
    std::cout << "Error detected during data load: " << error << std::endl;
    return;
  }

  std::cout << "Loaded " << training_data.size() << " training samples."
            << std::endl;

  std::cout << "Loading verification data..." << std::endl;

  if (!LoadImageSet(mnist_classify_images, mnist_classify_labels,
                    &classify_data, &error)) {
    std::cout << "Error detected during data load: " << error << std::endl;
    return;
  }

  std::cout << "Loaded " << classify_data.size() << " verification samples."
            << std::endl;

  if (training_label_count != classify_label_count) {
    std::cout << "Error: training and verification data do not have the same"
                 "label cardinality. Unable to continue."
              << std::endl;
    return;
  }

  std::cout << "Initiating training sequence." << std::endl;

  uint64 start_time = GetSystemTime();

  uint32 feature_count = 1;
  std::vector<LAYER_FORMAT> layer_formats = {
      {LAYER_INPUT, 784, 28, 28, 0, 0},
      {LAYER_CONVOLUTIONAL, feature_count * 576, 24, 24, feature_count, 5},
      {LAYER_MAX_POOLING, feature_count * 144, 12, 12, feature_count, 2},
      {LAYER_FULLY_CONNECTED, 30, 0, 0, 0, 0},
      {LAYER_FULLY_CONNECTED, 10, 0, 0, 0, 0}};

  Network neural_net(output_filename, 300, 10, layer_formats, 0.5, 1,
                     &training_data, &classify_data);

  uint64 elapsed_time = GetElapsedTimeMs(start_time);

  std::cout << "Training took " << elapsed_time / 1000.0f << " seconds."
            << std::endl;
}

void ExecuteVerification(const std::string& input_filename) {
  std::string error;
  std::vector<SampleSet> classify_data;

  if (input_filename.empty()) {
    std::cout << "You must specify a valid network filename to load."
              << std::endl;
    return;
  }

  std::cout << "Loading verification data..." << std::endl;

  if (!LoadImageSet(mnist_classify_images, mnist_classify_labels,
                    &classify_data, &error)) {
    std::cout << "Error detected during data load: " << error << std::endl;
    return;
  }

  std::cout << "Loaded " << classify_data.size() << " verification samples."
            << std::endl;

  Ensemble ens(input_filename);

  std::cout << "Initiating verification sequence." << std::endl;

  uint64 start_time = GetSystemTime();
  float64 total_correct = ens.Evaluate(&classify_data);
  uint64 elapsed_time = GetElapsedTimeMs(start_time);

  std::cout << "Total correct: " << total_correct << std::endl;
  std::cout << "Training accuracy: "
            << 100.0f * total_correct / classify_data.size() << "."
            << std::endl;
  std::cout << "Verification took " << elapsed_time / 1000.0f << " seconds."
            << std::endl;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    PrintUsage(argv[0]);
    return 0;
  }

  for (int i = 1; i < argc; i++) {
    char* optBegin = argv[i];
    for (int j = 0; j < 2; j++) (optBegin[0] == '-') ? optBegin++ : optBegin;

    switch (optBegin[0]) {
      case 't':
        ExecuteTraining(argv[++i]);
        break;
      case 'v':
        ExecuteVerification(argv[++i]);
        break;
      case 'c':
        std::string output(argv[++i]);
        std::vector<std::string> networks;
        while (i < argc - 1) {
          networks.emplace_back(argv[++i]);
        }
        ExecuteCombine(output, networks);
    }
  }

  return 0;
}
