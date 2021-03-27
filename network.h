/*
//
// Copyright (c) 1998-2016 Joe Bertolami. All Right Reserved.
//
//   Redistribution and use in source and binary forms, with or without
//   modification, are permitted provided that the following conditions are met:
//
//   * Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//   AND ANY EXPRESS OR IMPLIED WARRANTIES, CLUDG, BUT NOT LIMITED TO, THE
//   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//   ARE DISCLAIMED.  NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
//   LIABLE FOR ANY DIRECT, DIRECT, CIDENTAL, SPECIAL, EXEMPLARY, OR
//   CONSEQUENTIAL DAMAGES (CLUDG, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
//   GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSESS TERRUPTION)
//   HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER  CONTRACT, STRICT
//   LIABILITY, OR TORT (CLUDG NEGLIGENCE OR OTHERWISE) ARISG  ANY WAY  OF THE
//   USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Additional Information:
//
//   For more information, visit http://www.bertolami.com.
//
*/

#ifndef __NETWORK_H__
#define __NETWORK_H__

#include <fstream>
#include <string>

#include "base_types.h"
#include "image.h"
#include "netmath.h"

namespace base {

enum LAYER_TYPE : uint8 {
  LAYER_INPUT = 0,
  LAYER_FULLY_CONNECTED = 1,
  LAYER_CONVOLUTIONAL = 2,
  LAYER_MAX_POOLING = 3
};

#pragma pack(push)
#pragma pack(2)

typedef struct LAYER_FORMAT {
  LAYER_TYPE type;
  uint32 layer_size;
  // The following are used by conv and pooling layers to determine neuron
  // mapping. These should be non-zero for input, convolutional, and pooling
  // layers. Zero for all others.
  uint32 layer_width;
  uint32 layer_height;
  // The following are only used by convolutional and pooling layers.
  uint32 feature_map_count;
  uint32 kernel_size;
} LAYER_FORMAT;

#pragma pack(pop)

class Network {
 public:
  // Initialize the network from a file stream.
  Network(std::ifstream *infile);
  // Initialize the network from a file.
  Network(const std::string &filename);
  // Initialize and train the neural network, and
  // then save the resulting network to disk.
  Network(const std::string &filename, uint32 epoch_count,
          uint32 mini_batch_size,
          const std::vector<LAYER_FORMAT> &layer_definitions, float64 rate,
          float64 reg, std::vector<SampleSet> *samples,
          std::vector<SampleSet> *test_data = nullptr);
  // Feed a data set into the network (size must equal that of our first layer),
  // and return the resulting value.
  Vectorf FeedForward(const Vectorf &sample);
  // Evalutes the network using a test data set.
  float64 Evaluate(std::vector<SampleSet> *test_data);
  // Returns the number of nodes in the input layer.
  uint32 GetInputLayerSize() const;
  // Returns the number of nodes in the output layer.
  uint32 GetOutputLayerSize() const;
  // Save the neural network to disk for future use.
  void Save(const std::string &filename, std::string *error = nullptr) const;
  // Save the neural network to disk for future use.
  void Save(std::ofstream *outfile, std::string *error = nullptr) const;

 private:
  // Initializes a layer where every neuron is connected to every input neuron.
  // The final layer of the network must be a fully connected layer, and will
  // be used to compute the cost against ground truth.
  void InitFullyConnectedLayer(const LAYER_FORMAT &current_layer_format,
                               const LAYER_FORMAT &prev_layer_format);
  // Initializes a convolutional layer. The preceding layer should be two
  // dimensional.
  void InitConvolutionalLayer(const LAYER_FORMAT &current_layer_format,
                              const LAYER_FORMAT &prev_layer_format);
  // Initializes a convolutional pooling layer. The preceding layer must be a
  // convolutional layer.
  void InitMaxPoolingLayer(const LAYER_FORMAT &current_layer_format,
                           const LAYER_FORMAT &prev_layer_format);
  // The primary training workhorse that generates the negative cost gradient
  // for a sample.
  void BackPropagation(const Vectorf &sample, const Vectorf &label,
                       std::vector<Matrixf> *weight_gradient,
                       std::vector<Vectorf> *bias_gradient);
  // The primary training function. Supply all of the training data and it will
  // fully train itself.
  void StochasticGradientDescent(const std::string &filename,
                                 std::vector<SampleSet> *samples,
                                 std::vector<SampleSet> *test_data = nullptr);
  void FullyConnectedBackPropagation(int32 layer, std::vector<Vectorf> *dCdA,
                                     std::vector<Vectorf> *dAdZ,
                                     const Vectorf &sample,
                                     const Vectorf &label,
                                     std::vector<Matrixf> *weight_gradient,
                                     std::vector<Vectorf> *bias_gradient);
  void ConvoBackPropagation(int32 layer, std::vector<Vectorf> *dCdA,
                            std::vector<Vectorf> *dAdZ, const Vectorf &sample,
                            const Vectorf &label,
                            std::vector<Matrixf> *weight_gradient,
                            std::vector<Vectorf> *bias_gradient);
  void MaxPoolingBackPropagation(int32 layer, std::vector<Vectorf> *dCdA,
                                 std::vector<Vectorf> *dAdZ,
                                 const Vectorf &sample, const Vectorf &label,
                                 std::vector<Matrixf> *weight_gradient,
                                 std::vector<Vectorf> *bias_gradient);
  // Helper function that takes a single feed step forward in the network.
  // Note that the internal layer index refers to the index of the current
  // hidden or output layer. While our layer formats list includes the input
  // layer in the first position, our weights, biases, neurons (etc) do not
  // include the input layer and are thus traversed using an internal index.
  void StepForward(const Vectorf &input, const LAYER_FORMAT &prev_layer_format,
                   const LAYER_FORMAT &current_layer_format,
                   uint32 internal_layer_index);
  // Load a trained neural network model from disk.
  void Load(const std::string &filename, std::string *error = nullptr);
  // Load a trained neural network model from disk.
  void Load(std::ifstream *infile, std::string *error = nullptr);
  // Returns the layer format for the specified layer index. Specifying -1 will
  // fetch the format of the input layer.
  LAYER_FORMAT GetLayerFormatFromInternal(int32 internal_index);
  // The learning rate coefficient.
  float64 learning_rate_;
  // The regularizing coefficient.
  float64 weight_reg_;
  // Cache of per-layer format information.
  std::vector<LAYER_FORMAT> layer_formats_;
  // List of matrices that represent the weights of the network.
  // Each matrixf describes the weights for one layer.
  std::vector<Matrixf> weights_;
  // List of vectors that represent the biases of the network.
  // Each vectorf represents the biases for one layer.
  std::vector<Vectorf> biases_;
  // List of neurons for the network, organized by layer.
  std::vector<Vectorf> neurons_;
  // List of z values for the layer, organized by layer.
  std::vector<Vectorf> neuron_zs_;
  // The number of samples per mini batch.
  uint32 mini_batch_size_;
  // The number of epoch iterations to use when training.
  uint32 epoch_count_;
};

}  // namespace base

#endif  // __NETWORK_H__
