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

#ifndef __ENSEMBLE_H__
#define __ENSEMBLE_H__

#include <string>

#include "base_types.h"
#include "network.h"

namespace base {

class Ensemble {
 public:
  // Initialize an ensemble from a list of networks. Each network may vary in
  // shape, but their input and output layers must have the same size.
  Ensemble(const std::vector<Network> &networks);
  // Initialize an ensemble of neural networks from a file.
  Ensemble(const std::string &filename);
  // Feed a sample into the network ensemble and return the final layer
  // activations.
  Vectorf FeedForward(const Vectorf &sample);
  // Evaluate the accuracy of the network against test data.
  float64 Evaluate(std::vector<SampleSet> *test_data);
  // Returns the number of nodes in the input layer.
  uint32 GetInputLayerSize() const;
  // Returns the number of nodes in the output layer.
  uint32 GetOutputLayerSize() const;
  // Save an ensemble to disk.
  void Save(const std::string &filename, std::string *error = nullptr) const;

 private:
  // Load an ensemble from disk.
  void Load(const std::string &filename, std::string *error = nullptr);
  // List of neural networks in our ensemble.
  std::vector<Network> ensemble_;
};

}  // namespace base

#endif  // __ENSEMBLE_H__
