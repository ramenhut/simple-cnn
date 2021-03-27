
#include "network.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>

#define QUADRATIC_COST (0)
#define L2_REGULARIZATION (1)
#define activation sigmoid
#define activation_prime sigmoid_prime

namespace base {

#pragma pack(push)
#pragma pack(2)

typedef struct {
  uint16 magic;  // must be 0x4244
  float64 learning_rate;
  float64 weight_reg_coeff;
  uint32 mini_batch_size;
  uint32 epoch_count;
  uint64 layer_count;
} NEURAL_NETWORK_HEADER;

#pragma pack(pop)

Network::Network(std::ifstream *infile) { Load(infile); }

Network::Network(const std::string &filename) {
  std::string error;
  Load(filename, &error);
  if (error.length()) {
    std::cout << error << std::endl;
  }
}

Network::Network(const std::string &filename, uint32 epoch_count,
                 uint32 mini_batch_size,
                 const std::vector<LAYER_FORMAT> &layer_definitions,
                 float64 rate, float64 reg, std::vector<SampleSet> *samples,
                 std::vector<SampleSet> *test_data) {
  assert(layer_definitions.size() > 1);

  epoch_count_ = epoch_count;
  mini_batch_size_ = mini_batch_size;
  learning_rate_ = rate;
  weight_reg_ = reg;
  layer_formats_ = layer_definitions;

  for (uint32 i = 1; i < layer_definitions.size(); i++) {
    switch (layer_definitions[i].type) {
      case LAYER_INPUT: {
      } break;
      case LAYER_FULLY_CONNECTED: {
        InitFullyConnectedLayer(layer_definitions[i], layer_definitions[i - 1]);
      } break;
      case LAYER_CONVOLUTIONAL: {
        InitConvolutionalLayer(layer_definitions[i], layer_definitions[i - 1]);
      } break;
      case LAYER_MAX_POOLING: {
        InitMaxPoolingLayer(layer_definitions[i], layer_definitions[i - 1]);
      } break;
      default: {
        assert(false);
        return;
      }
    }
  }

  // Train our network, saving the result to disk. If test_data is provided,
  // we'll save the best epoch. Otherwise, we'll save the last epoch.
  StochasticGradientDescent(filename, samples, test_data);
}

LAYER_FORMAT Network::GetLayerFormatFromInternal(int32 internal_index) {
  return layer_formats_[internal_index + 1];
}

void Network::InitFullyConnectedLayer(const LAYER_FORMAT &current_layer_format,
                                      const LAYER_FORMAT &prev_layer_format) {
  // The size of each layer matrix equals the W x H, where
  // W is the size of the prevous layer, and H is the size
  // of the current layer.
  weights_.push_back(
      Matrixf(prev_layer_format.layer_size, current_layer_format.layer_size));
  biases_.push_back(Vectorf(current_layer_format.layer_size));
  neurons_.push_back(Vectorf(current_layer_format.layer_size));
  neuron_zs_.push_back(Vectorf(current_layer_format.layer_size));

  biases_[biases_.size() - 1].Randomize();
  weights_[weights_.size() - 1].Randomize();
  neurons_[neurons_.size() - 1].Zero();
  neuron_zs_[neuron_zs_.size() - 1].Zero();
}

void Network::InitConvolutionalLayer(const LAYER_FORMAT &current_layer_format,
                                     const LAYER_FORMAT &prev_layer_format) {
  assert(prev_layer_format.layer_width && prev_layer_format.layer_height);
  assert(prev_layer_format.type == LAYER_INPUT ||
         (prev_layer_format.feature_map_count ==
          current_layer_format.feature_map_count));
  assert(current_layer_format.layer_size ==
         current_layer_format.feature_map_count *
             current_layer_format.layer_width *
             current_layer_format.layer_height);

  uint32 layer_feature_width =
      prev_layer_format.layer_width - (current_layer_format.kernel_size - 1);
  uint32 layer_feature_height =
      prev_layer_format.layer_height - (current_layer_format.kernel_size - 1);

  assert(layer_feature_width == current_layer_format.layer_width);
  assert(layer_feature_height == current_layer_format.layer_height);

  weights_.push_back(Matrixf(
      current_layer_format.kernel_size * current_layer_format.kernel_size,
      current_layer_format.feature_map_count));

  biases_.push_back(Vectorf(current_layer_format.feature_map_count));
  neurons_.push_back(Vectorf(layer_feature_width * layer_feature_height *
                             current_layer_format.feature_map_count));
  neuron_zs_.push_back(Vectorf(layer_feature_width * layer_feature_height *
                               current_layer_format.feature_map_count));

  biases_[biases_.size() - 1].Randomize();
  weights_[weights_.size() - 1].Randomize();
  neurons_[neurons_.size() - 1].Zero();
  neuron_zs_[neuron_zs_.size() - 1].Zero();
}

void Network::InitMaxPoolingLayer(const LAYER_FORMAT &current_layer_format,
                                  const LAYER_FORMAT &prev_layer_format) {
  assert(prev_layer_format.layer_width && prev_layer_format.layer_height);
  assert(prev_layer_format.feature_map_count ==
         current_layer_format.feature_map_count);
  assert(current_layer_format.layer_size ==
         current_layer_format.feature_map_count *
             current_layer_format.layer_width *
             current_layer_format.layer_height);

  // Max pooling layer has fewer neurons than the convolutional layer.
  // We have a weight for each of our input neurons that gets set *during*
  // feed forward, according to which input neuron was the maximum value.
  // This enables back propagation to traverse this layer. Back prop will not
  // change this layer's weights (it will simply pass them to the gradient).

  uint32 layer_feature_width =
      prev_layer_format.layer_width / current_layer_format.kernel_size;
  uint32 layer_feature_height =
      prev_layer_format.layer_height / current_layer_format.kernel_size;

  assert(layer_feature_width == current_layer_format.layer_width);
  assert(layer_feature_height == current_layer_format.layer_height);

  weights_.push_back(Matrixf(
      current_layer_format.kernel_size * current_layer_format.kernel_size,
      layer_feature_width * layer_feature_height *
          current_layer_format.feature_map_count));

  biases_.push_back(Vectorf(layer_feature_width * layer_feature_height *
                            current_layer_format.feature_map_count));
  neurons_.push_back(Vectorf(layer_feature_width * layer_feature_height *
                             current_layer_format.feature_map_count));
  neuron_zs_.push_back(Vectorf(layer_feature_width * layer_feature_height *
                               current_layer_format.feature_map_count));

  biases_[biases_.size() - 1].Zero();
  weights_[weights_.size() - 1].Zero();
  neurons_[neurons_.size() - 1].Zero();
  neuron_zs_[neuron_zs_.size() - 1].Zero();
}

uint32 Network::GetInputLayerSize() const {
  assert(!weights_.empty());
  return weights_[0].GetWidth();
}

uint32 Network::GetOutputLayerSize() const {
  assert(!biases_.empty());
  return biases_[biases_.size() - 1].GetWidth();
}

bool SaveVector(std::ofstream *out_stream, const Vectorf &v) {
  if (!out_stream || 0 == v.GetWidth()) {
    return false;
  }
  for (uint32 i = 0; i < v.GetWidth(); i++) {
    if (!out_stream->write((char *)&v[i], sizeof(float64))) {
      return false;
    }
  }
  return true;
}

bool LoadVector(std::ifstream *in_stream, Vectorf *v) {
  if (!in_stream || !v || 0 == v->GetWidth()) {
    return false;
  }
  for (uint32 i = 0; i < v->GetWidth(); i++) {
    if (!in_stream->read((char *)&(*v)[i], sizeof(float64))) {
      return false;
    }
  }
  return true;
}

void Network::Save(const std::string &filename, std::string *error) const {
  std::ofstream out_stream(filename, std::ios::out | std::ios::binary);
  Save(&out_stream, error);
}

void Network::Save(std::ofstream *outfile, std::string *error) const {
  NEURAL_NETWORK_HEADER header = {
      0x4244,  // magic number
      learning_rate_, weight_reg_, mini_batch_size_, epoch_count_, 0};

  // Add one layer to account for the input layer.
  header.layer_count = weights_.size() + 1;

  if (!outfile->write((char *)&header, sizeof(header))) {
    if (error) {
      *error = "Failed to write neural network header to disk.";
    }
    return;
  }

  // Write out the formats of each layer, beginning with the
  // input layer, and following with the rest of the layers.
  for (auto &layer : layer_formats_) {
    if (!outfile->write((char *)&layer, sizeof(LAYER_FORMAT))) {
      if (error) {
        *error = "Failed to write layer format to disk.";
      }
      return;
    }
  }

  for (auto &layer_weights : weights_) {
    uint32 layer_height = layer_weights.GetHeight();
    for (uint32 row = 0; row < layer_height; row++) {
      if (!SaveVector(outfile, layer_weights[row])) {
        if (error) {
          *error = "Failed to write layer weights to disk.";
        }
        return;
      }
    }
  }

  for (auto &layer_biases : biases_) {
    if (!SaveVector(outfile, layer_biases)) {
      if (error) {
        *error = "Failed to write layer biases to disk.";
      }
      return;
    }
  }
}

void Network::Load(const std::string &filename, std::string *error) {
  std::ifstream in_stream(filename, ::std::ios::in | ::std::ios::binary);
  Load(&in_stream, error);
}

void Network::Load(std::ifstream *infile, std::string *error) {
  NEURAL_NETWORK_HEADER header;
  if (!infile->read((char *)&header, sizeof(header))) {
    if (error) {
      *error = "Failed to read neural network header from disk.";
    }
    return;
  }

  if (header.magic != 0x4244) {
    if (error) {
      *error = "Invalid neural network file format detected.";
    }
    return;
  }

  learning_rate_ = header.learning_rate;
  weight_reg_ = header.weight_reg_coeff;
  mini_batch_size_ = header.mini_batch_size;
  epoch_count_ = header.epoch_count;

  for (uint32 i = 0; i < header.layer_count; i++) {
    LAYER_FORMAT format;
    if (!infile->read((char *)&format, sizeof(LAYER_FORMAT))) {
      if (error) {
        *error = "Failed to read neural network layer format from disk.";
      }
      return;
    }
    layer_formats_.push_back(format);
  }

  for (uint32 i = 0; i < layer_formats_.size(); i++) {
    switch (layer_formats_[i].type) {
      case LAYER_INPUT: {
        // No initialization needed.
      } break;
      case LAYER_FULLY_CONNECTED: {
        InitFullyConnectedLayer(layer_formats_[i], layer_formats_[i - 1]);
      } break;
      case LAYER_CONVOLUTIONAL: {
        InitConvolutionalLayer(layer_formats_[i], layer_formats_[i - 1]);
      } break;
      case LAYER_MAX_POOLING: {
        InitMaxPoolingLayer(layer_formats_[i], layer_formats_[i - 1]);
      } break;
      default: {
        assert(false);
        return;
      }
    }
  }

  for (auto &layer_weights : weights_) {
    uint32 layer_height = layer_weights.GetHeight();
    for (uint32 row = 0; row < layer_height; row++) {
      if (!LoadVector(infile, &layer_weights[row])) {
        if (error) {
          *error = "Failed to read layer weights from disk.";
        }
        return;
      }
    }
  }

  for (auto &layer_biases : biases_) {
    if (!LoadVector(infile, &layer_biases)) {
      if (error) {
        *error = "Failed to read layer biases from disk.";
      }
      return;
    }
  }
}

void Network::StepForward(const Vectorf &input,
                          const LAYER_FORMAT &prev_layer_format,
                          const LAYER_FORMAT &current_layer_format,
                          uint32 internal_layer_index) {
  switch (current_layer_format.type) {
    case LAYER_FULLY_CONNECTED: {
      neuron_zs_[internal_layer_index] =
          weights_[internal_layer_index] * input +
          biases_[internal_layer_index];
      neurons_[internal_layer_index] =
          activation(neuron_zs_[internal_layer_index]);
    } break;
    case LAYER_CONVOLUTIONAL: {
      uint32 feature_size =
          current_layer_format.layer_width * current_layer_format.layer_height;

      for (uint32 feature = 0; feature < current_layer_format.feature_map_count;
           feature++) {
        for (uint32 y = 0; y < current_layer_format.layer_height; y++) {
          for (uint32 x = 0; x < current_layer_format.layer_width; x++) {
            float64 total = 0.0;
            uint64 dest_node_index = feature * feature_size +
                                     y * current_layer_format.layer_width + x;
            for (uint32 kx = 0; kx < current_layer_format.kernel_size; kx++) {
              for (uint32 ky = 0; ky < current_layer_format.kernel_size; ky++) {
                total +=
                    weights_[internal_layer_index][feature]
                            [ky * current_layer_format.kernel_size + kx] *
                    input[(y + ky) * prev_layer_format.layer_width + (x + kx)];
              }
            }
            total += biases_[internal_layer_index][feature];
            neuron_zs_[internal_layer_index][dest_node_index] = total;
            neurons_[internal_layer_index][dest_node_index] = activation(total);
          }
        }
      }
    } break;
    case LAYER_MAX_POOLING: {
      assert(prev_layer_format.layer_width && prev_layer_format.layer_height);
      weights_[internal_layer_index].Zero();

      uint32 feature_size =
          current_layer_format.layer_width * current_layer_format.layer_height;

      for (uint32 feature = 0; feature < current_layer_format.feature_map_count;
           feature++) {
        for (uint32 y = 0; y < current_layer_format.layer_height; y++) {
          for (uint32 x = 0; x < current_layer_format.layer_width; x++) {
            uint64 dest_node_index = feature * feature_size +
                                     y * current_layer_format.layer_width + x;
            float64 max_value = -BASE_INFINITY;
            uint64 max_node_index = 0;
            for (uint32 kx = 0; kx < current_layer_format.kernel_size; kx++) {
              for (uint32 ky = 0; ky < current_layer_format.kernel_size; ky++) {
                uint64 test_index =
                    (y * current_layer_format.kernel_size + ky) *
                        prev_layer_format.layer_width +
                    (x * current_layer_format.kernel_size + kx);
                float64 input_value = input[test_index];

                if (input_value >= max_value) {
                  max_value = input_value;
                  max_node_index = ky * current_layer_format.kernel_size + kx;
                }
              }
            }
            // Select the max value and set it's weight to 1.
            weights_[internal_layer_index][dest_node_index][max_node_index] =
                1.0;
            neuron_zs_[internal_layer_index][dest_node_index] = max_value;
            neurons_[internal_layer_index][dest_node_index] = max_value;
          }
        }
      }
    } break;
    default: {
      std::cout << "Invalid layer detected during forward step." << std::endl;
      assert(false);
      return;
    }
  }
}

Vectorf Network::FeedForward(const Vectorf &sample) {
  // Prime the first iteration using our input layer.
  StepForward(sample, GetLayerFormatFromInternal(-1),
              GetLayerFormatFromInternal(0), 0);
  // Propagate our neural activations to the output layer.
  for (uint32 i = 1; i < weights_.size(); i++) {
    // The layer_formats_ list includes the input layer, so layer_formats_[i] is
    // actually the previous layer format, while layer_formats_[i+1] is the
    // current layer format.
    StepForward(neurons_[i - 1], GetLayerFormatFromInternal(i - 1),
                GetLayerFormatFromInternal(i), i);
  }

  return neurons_[neurons_.size() - 1];
}

float64 Network::Evaluate(std::vector<SampleSet> *test_data) {
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

void Network::StochasticGradientDescent(const std::string &filename,
                                        std::vector<SampleSet> *samples,
                                        std::vector<SampleSet> *test_data) {
  float64 total_samples_trained = 0.0;
  float64 highest_accuracy = -1.0;

  for (uint32 epoch = 0; epoch < epoch_count_; epoch++) {
    // For each epoch, shuffle our input samples, then operate in mini batch
    // sizes
    uint32 random_seed =
        std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(samples->begin(), samples->end(),
                 std::default_random_engine(random_seed));
    uint32 mini_batch_count = samples->size() / mini_batch_size_ +
                              !!(samples->size() % mini_batch_size_);

    // Allocate our gradient accumulators according to the shape of our
    // layer weights and biases.
    std::vector<Matrixf> grad_accum_weights = weights_;
    std::vector<Vectorf> grad_accum_biases = biases_;
    std::vector<Matrixf> grad_impulse_weights = weights_;
    std::vector<Vectorf> grad_impulse_biases = biases_;

    for (uint32 batch = 0; batch < mini_batch_count; batch++) {
      uint32 start_sample = batch * mini_batch_size_;
      uint32 end_sample = ((batch + 1) * mini_batch_size_ < samples->size())
                              ? (batch + 1) * mini_batch_size_
                              : samples->size();
      uint32 mini_batch_sample_count = end_sample - start_sample;

      for (auto &v : grad_accum_biases) v.Zero();
      for (auto &m : grad_accum_weights) m.Zero();

      // For each mini-batch, feed it through our network (feed forward), and
      // then compute the cost gradient (back propagation).
      for (uint32 s = start_sample; s < end_sample; s++) {
        FeedForward(samples->at(s).sample);

        for (auto &v : grad_impulse_biases) v.Zero();
        for (auto &m : grad_impulse_weights) m.Zero();

        BackPropagation(samples->at(s).sample, samples->at(s).label,
                        &grad_impulse_weights, &grad_impulse_biases);

        // Add our impulse gradient to our accumulation.
        for (uint32 layer = 0; layer < weights_.size(); layer++) {
          grad_accum_weights[layer] += grad_impulse_weights[layer];
          grad_accum_biases[layer] += grad_impulse_biases[layer];
        }
      }

      // Compute the average weights and biases of our negative gradient and
      // apply them to our actual network weights and biases.
      float64 divisor = (learning_rate_ / mini_batch_sample_count);
      // float64 reg_factor =
      //    (1.0 - learning_rate_ * weight_reg_ / mini_batch_sample_count);
      for (uint32 layer = 0; layer < weights_.size(); layer++) {
#if QUADRATIC_COST
#if L2_REGULARIZATION
        weights_[layer] =
            weights_[layer] * weight_reg_ - grad_accum_weights[layer] * divisor;
#else
        weights_[layer] = weights_[layer] - grad_accum_weights[layer] * divisor;
#endif
        biases_[layer] = biases_[layer] - grad_accum_biases[layer] * divisor;
#else
#if L2_REGULARIZATION
        weights_[layer] = weights_[layer] * weight_reg_ -
                          grad_accum_weights[layer] * divisor * -1.0;
#else
        weights_[layer] =
            weights_[layer] - grad_accum_weights[layer] * divisor * -1.0;
#endif
        biases_[layer] =
            biases_[layer] - grad_accum_biases[layer] * divisor * -1.0;
#endif
      }

      total_samples_trained += mini_batch_sample_count;
    }

    if (test_data) {
      float64 total_correct = Evaluate(test_data);
      std::cout << "Epoch " << epoch
                << " accuracy: " << 100.0f * total_correct / test_data->size()
                << "." << std::endl;
      if (total_correct > highest_accuracy) {
        highest_accuracy = total_correct;
        Save(filename);
      }
    } else {
      // If we have no test criteria then we simply write out the results
      // of every epoch.
      Save(filename);
    }
  }
  std::cout << std::endl;
}

void Network::BackPropagation(const Vectorf &sample, const Vectorf &label,
                              std::vector<Matrixf> *weight_gradient,
                              std::vector<Vectorf> *bias_gradient) {
  assert(bias_gradient->size() == biases_.size());
  assert(weight_gradient->size() == weights_.size());
  assert(weights_.size() == biases_.size());

  // Beginning with the final layer, compute gradient and store in
  // weight_gradient[last] and bias_gradient[last] Iterate backwards, using the
  // previous layer neurons, the weights and biases, and the derivatives from
  // the next layer.

  std::vector<Vectorf> dCdA = biases_;
  std::vector<Vectorf> dAdZ = biases_;

  for (auto &v : dCdA) {
    v.Zero();
  }
  for (auto &v : dAdZ) {
    v.Zero();
  }

  for (int32 layer = dCdA.size() - 1; layer > 0; layer--)
    for (int32 layer = dCdA.size() - 1; layer > 0; layer--) {
      switch (layer_formats_[layer + 1].type) {
        case LAYER_FULLY_CONNECTED: {
          FullyConnectedBackPropagation(layer, &dCdA, &dAdZ, sample, label,
                                        weight_gradient, bias_gradient);
        } break;
        case LAYER_CONVOLUTIONAL: {
          ConvoBackPropagation(layer, &dCdA, &dAdZ, sample, label,
                               weight_gradient, bias_gradient);
        } break;
        case LAYER_MAX_POOLING: {
          MaxPoolingBackPropagation(layer, &dCdA, &dAdZ, sample, label,
                                    weight_gradient, bias_gradient);
        } break;
        default:
          std::cout << "Invalid layer detected during back propagation."
                    << std::endl;
          assert(false);
          return;
      }
    }
}

void Network::ConvoBackPropagation(int32 layer, std::vector<Vectorf> *dCdA,
                                   std::vector<Vectorf> *dAdZ,
                                   const Vectorf &sample, const Vectorf &label,
                                   std::vector<Matrixf> *weight_gradient,
                                   std::vector<Vectorf> *bias_gradient) {
  // Convolutional layers cannot be the final layer in the network.
  assert(layer != dCdA->size() - 1);
  LAYER_FORMAT current_layer = GetLayerFormatFromInternal(layer);
  LAYER_FORMAT next_layer = GetLayerFormatFromInternal(layer + 1);
  LAYER_FORMAT prev_layer = GetLayerFormatFromInternal(layer - 1);

  assert(prev_layer.type == LAYER_INPUT);
  assert(next_layer.type == LAYER_FULLY_CONNECTED ||
         next_layer.type == LAYER_MAX_POOLING);

  for (uint32 node = 0; node < (*dCdA)[layer].GetWidth(); node++) {
    uint64 feature =
        node / (current_layer.layer_width * current_layer.layer_height);
    uint64 feature_node =
        node % (current_layer.layer_width * current_layer.layer_height);
    uint64 x = feature_node % current_layer.layer_width;
    uint64 y = feature_node / current_layer.layer_width;
    // We're evaluating a hidden layer of our network. compute dCdA[layer][node]
    // as the sum of weights * dCdZ of the next layer.
    (*dCdA)[layer][node] = 0.0;
    switch (GetLayerFormatFromInternal(layer + 1).type) {
      case LAYER_FULLY_CONNECTED: {
        for (uint32 next_layer_node = 0;
             next_layer_node < (*dCdA)[layer + 1].GetWidth();
             next_layer_node++) {
          (*dCdA)[layer][node] += weights_[layer + 1][next_layer_node][node] *
                                  (*dAdZ)[layer + 1][next_layer_node] *
                                  (*dCdA)[layer + 1][next_layer_node];
        }
      } break;
      case LAYER_MAX_POOLING: {
        assert(current_layer.feature_map_count == next_layer.feature_map_count);
        // Only include the value that our current layer node actually
        // contributes to. For a pooling layer this means we only influence a
        // subset of one feature map.
        uint64 pool_feature_size =
            next_layer.layer_width * next_layer.layer_height;
        uint64 pool_node_index =
            feature * pool_feature_size +
            (y / next_layer.kernel_size) * next_layer.layer_width +
            (x / next_layer.kernel_size);
        uint32 kernel_node =
            (y % next_layer.kernel_size) * next_layer.kernel_size +
            (x % next_layer.kernel_size);
        (*dCdA)[layer][node] =
            weights_[layer + 1][pool_node_index][kernel_node] *
            (*dAdZ)[layer + 1][pool_node_index] *
            (*dCdA)[layer + 1][pool_node_index];
      } break;
      case LAYER_CONVOLUTIONAL: {
        // Only sum the values that our current layer node actually
        // contributes to. For a convo layer this means we only influence a
        // subset of one feature map.
        std::cout << "Convolutional layers cannot follow convolutional layers."
                  << std::endl;
        assert(false);
        return;
      } break;
      default: {
        std::cout << "Invalid layer hierarchy detected." << std::endl;
        assert(false);
        return;
      }
    }

    // Compute the Z gradient and cache it
    (*dAdZ)[layer][node] = activation_prime(neuron_zs_[layer][node]);

    Vectorf *input =
        (layer == 0) ? &const_cast<Vectorf &>(sample) : &(neurons_[layer - 1]);
    uint32 current_weight_count =
        current_layer.kernel_size * current_layer.kernel_size;

    // We are on one particular node of our layer, which means we accumulate the
    // influence of our current node on the weights for the current feature map.
    float64 denom =
        1.0 / (current_layer.layer_width * current_layer.layer_height);
    for (uint32 weight = 0; weight < current_weight_count; weight++) {
      uint32 wx = weight % current_layer.kernel_size;
      uint32 wy = weight / current_layer.kernel_size;
      // Average the contributions from each of the input values.
      for (uint32 y = 0; y < current_layer.layer_height; y++) {
        for (uint32 x = 0; x < current_layer.layer_width; x++) {
          weight_gradient->at(layer)[feature][weight] +=
              (*input)[(y + wy) * prev_layer.layer_width + (x + wx)] *
              (*dAdZ)[layer][node] * (*dCdA)[layer][node] * denom;
        }
      }
    }

    bias_gradient->at(layer)[node] =
        (*dAdZ)[layer][node] * (*dCdA)[layer][node];
  }
}

void Network::MaxPoolingBackPropagation(int32 layer, std::vector<Vectorf> *dCdA,
                                        std::vector<Vectorf> *dAdZ,
                                        const Vectorf &sample,
                                        const Vectorf &label,
                                        std::vector<Matrixf> *weight_gradient,
                                        std::vector<Vectorf> *bias_gradient) {
  // Max pooling layers cannot be the final layer in the network.
  assert(layer != dCdA->size() - 1);
  LAYER_FORMAT current_layer = GetLayerFormatFromInternal(layer);
  LAYER_FORMAT next_layer = GetLayerFormatFromInternal(layer + 1);
  LAYER_FORMAT prev_layer = GetLayerFormatFromInternal(layer - 1);

  assert(prev_layer.type == LAYER_MAX_POOLING ||
         prev_layer.type == LAYER_CONVOLUTIONAL);
  assert(next_layer.type == LAYER_FULLY_CONNECTED ||
         next_layer.type == LAYER_MAX_POOLING);
  assert(prev_layer.feature_map_count == current_layer.feature_map_count);

  for (uint32 node = 0; node < (*dCdA)[layer].GetWidth(); node++) {
    uint64 feature =
        node / (current_layer.layer_width * current_layer.layer_height);
    uint64 feature_node =
        node % (current_layer.layer_width * current_layer.layer_height);
    uint64 x = feature_node % current_layer.layer_width;
    uint64 y = feature_node / current_layer.layer_width;
    // We're evaluating a hidden layer of our network. compute dCdA[layer][node]
    // as the sum of weights * dCdZ of the next layer.
    (*dCdA)[layer][node] = 0.0;
    switch (GetLayerFormatFromInternal(layer + 1).type) {
      case LAYER_FULLY_CONNECTED: {
        for (uint32 next_layer_node = 0;
             next_layer_node < (*dCdA)[layer + 1].GetWidth();
             next_layer_node++) {
          (*dCdA)[layer][node] += weights_[layer + 1][next_layer_node][node] *
                                  (*dAdZ)[layer + 1][next_layer_node] *
                                  (*dCdA)[layer + 1][next_layer_node];
        }
      } break;
      case LAYER_MAX_POOLING: {
        assert(current_layer.feature_map_count == next_layer.feature_map_count);
        // Only include the value that our current layer node actually
        // contributes to. For a pooling layer this means we only influence a
        // subset of one feature map.
        uint64 pool_feature_size =
            next_layer.layer_width * next_layer.layer_height;
        uint64 pool_node_index =
            feature * pool_feature_size +
            (y / next_layer.kernel_size) * next_layer.layer_width +
            (x / next_layer.kernel_size);
        uint32 kernel_node =
            (y % next_layer.kernel_size) * next_layer.kernel_size +
            (x % next_layer.kernel_size);
        (*dCdA)[layer][node] =
            weights_[layer + 1][pool_node_index][kernel_node] *
            (*dAdZ)[layer + 1][pool_node_index] *
            (*dCdA)[layer + 1][pool_node_index];
      } break;
      case LAYER_CONVOLUTIONAL: {
        // Only sum the values that our current layer node actually
        // contributes to. For a convo layer this means we only influence a
        // subset of one feature map.
        std::cout << "Convolutional layers cannot follow max pooling layers."
                  << std::endl;
        assert(false);
        return;
      } break;
      default: {
        std::cout << "Invalid layer hierarchy detected." << std::endl;
        assert(false);
        return;
      }
    }

    // Compute the Z gradient and cache it
    (*dAdZ)[layer][node] = activation_prime(neuron_zs_[layer][node]);

    Vectorf *input =
        (layer == 0) ? &const_cast<Vectorf &>(sample) : &(neurons_[layer - 1]);
    uint32 current_weight_count =
        current_layer.kernel_size * current_layer.kernel_size;

    // Each pooling node is driven by a single input neuron (the max value in
    // the kernel).
    // TODO TODO TODO
    for (uint32 weight = 0; weight < current_weight_count; weight++) {
      uint32 wx = weight % current_layer.kernel_size;
      uint32 wy = weight / current_layer.kernel_size;
      uint32 input_feature_offset =
          (prev_layer.type == LAYER_CONVOLUTIONAL ||
           prev_layer.type == LAYER_MAX_POOLING)
              ? feature * prev_layer.layer_width * prev_layer.layer_height
              : 0;
      // x and y are the pooling node coordinates. We multiply the factor by the
      // current weight in order to isolate the influence of the max input (all
      // other weights are zero).
      weight_gradient->at(layer)[node][weight] =
          (*input)[input_feature_offset +
                   (y * current_layer.kernel_size + wy) *
                       prev_layer.layer_width +
                   (x * current_layer.kernel_size + wx)] *
          weights_[layer][node][weight] * (*dAdZ)[layer][node] *
          (*dCdA)[layer][node];
    }

    bias_gradient->at(layer)[node] =
        (*dAdZ)[layer][node] * (*dCdA)[layer][node];
  }
}

void Network::FullyConnectedBackPropagation(
    int32 layer, std::vector<Vectorf> *dCdA, std::vector<Vectorf> *dAdZ,
    const Vectorf &sample, const Vectorf &label,
    std::vector<Matrixf> *weight_gradient,
    std::vector<Vectorf> *bias_gradient) {
  for (uint32 node = 0; node < (*dCdA)[layer].GetWidth(); node++) {
    if (layer == dCdA->size() - 1) {
      // We're evaluating the final (output) layer of our network.
#if QUADRATIC_COST
      (*dCdA)[layer][node] = 2 * (neurons_[layer][node] - label[node]);
#else
      (*dCdA)[layer][node] = label[node] / (neurons_[layer][node]) -
                             (1.0 - label[node]) / (1 - neurons_[layer][node]);
#endif
    } else {
      // We're evaluating a hidden layer of our network. Compute
      // dCdA[layer][node] as the sum of weights * dCdZ of the next layer.
      (*dCdA)[layer][node] = 0.0;
      for (uint32 next_layer_node = 0;
           next_layer_node < (*dCdA)[layer + 1].GetWidth(); next_layer_node++) {
        (*dCdA)[layer][node] += weights_[layer + 1][next_layer_node][node] *
                                (*dAdZ)[layer + 1][next_layer_node] *
                                (*dCdA)[layer + 1][next_layer_node];
      }
    }

    // Compute the Z gradient and cache it
    (*dAdZ)[layer][node] = activation_prime(neuron_zs_[layer][node]);

    // Compute the weight and bias gradients off of the dAdZ we just computed
    // For all weights for this node, compute weight gradient = a(l-1) * dAdZ
    // of this node * dCdA of this node
    Vectorf *input =
        (layer == 0) ? &const_cast<Vectorf &>(sample) : &(neurons_[layer - 1]);
    uint32 node_weights = input->GetWidth();
    for (uint32 weight = 0; weight < node_weights; weight++) {
      weight_gradient->at(layer)[node][weight] =
          (*input)[weight] * (*dAdZ)[layer][node] * (*dCdA)[layer][node];
    }

    // For the sole bias for this node, compute bias gradient = 1 * dAdZ of
    // this node * dCdA of this node
    bias_gradient->at(layer)[node] =
        (*dAdZ)[layer][node] * (*dCdA)[layer][node];
  }
}

}  // namespace base
