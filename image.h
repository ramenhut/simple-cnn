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

#ifndef __IMAGE_H__
#define __IMAGE_H__

#include "base_types.h"
#include "netmath.h"

namespace base {

// TrainingSet pairs a data sample with a label.
typedef struct SampleSet {
  SampleSet(uint32 sample_width, uint32 label_width);
  // Source image converted to a float vector.
  Vectorf sample;
  //  Label for the sample, used as ground truth
  // during network training and verification.
  Vectorf label;
} SampleSet;

#pragma pack(push)
#pragma pack(2)

typedef struct MNIST_LABEL_FILE_HEADER {
  // Magic number, must be 2049.
  uint32 magic;
  // Number of labels in the file.
  uint32 label_count;
} MNIST_LABEL_FILE_HEADER;

typedef struct MNIST_IMAGE_FILE_HEADER {
  // Magic number, must be 2051.
  uint32 magic;
  // Number of images in the file.
  uint32 image_count;
  // Width of each image in the file.
  uint32 width;
  // Height of each image in the file.
  uint32 height;
} MNIST_IMAGE_FILE_HEADER;

#pragma pack(pop)

inline uint32 EndianSwap8in32(uint32 input) {
  return (input >> 24) | ((input >> 8) & 0xFF00) | ((input << 8) & 0xFF0000) |
         (input << 24);
}

bool LoadImageSet(const std::string& images_filename,
                  const std::string& labels_filename,
                  std::vector<SampleSet>* output, std::string* error = nullptr);

}  // namespace base

#endif  // __IMAGE_H__
