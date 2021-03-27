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

#ifndef __NETMATH_H__
#define __NETMATH_H__

#include "base_types.h"

namespace base {

class Vectorf {
 public:
  // Vectors are zeroed out at init by default.
  Vectorf(uint32 width);
  // Returns the width of the vector.
  uint32 GetWidth() const;
  // Resizes and zeroes out the vector.
  void Resize(uint32 new_size);
  // Returns the index with the higest activation.
  uint32 GetMaxIndex() const;
  // Zeroes out the vector.
  void Zero();
  // Sets the values in the vector to a random normal distribution.
  void Randomize();
  // Returns a reference to the item at index.
  float64 &operator[](uint32 index) const;
  // Returns the average of the values in the vector.
  float64 Average() const;
  // Computes the dot product with another vector.
  float64 Dot(const Vectorf &rhs) const;

 private:
  std::vector<float64> v_;
};

inline const Vectorf operator+(const Vectorf &lhs, const Vectorf &rhs) {
  assert(lhs.GetWidth());
  assert(lhs.GetWidth() == rhs.GetWidth());
  Vectorf output(lhs.GetWidth());
  for (uint32 i = 0; i < lhs.GetWidth(); i++) {
    output[i] = lhs[i] + rhs[i];
  }
  return output;
}

inline const Vectorf &operator+=(Vectorf &lhs, const Vectorf &rhs) {
  assert(lhs.GetWidth());
  assert(lhs.GetWidth() == rhs.GetWidth());
  for (uint32 i = 0; i < lhs.GetWidth(); i++) {
    lhs[i] += rhs[i];
  }
  return lhs;
}

inline const Vectorf operator-(const Vectorf &lhs, const Vectorf &rhs) {
  assert(lhs.GetWidth());
  assert(lhs.GetWidth() == rhs.GetWidth());
  Vectorf output(lhs.GetWidth());
  for (uint32 i = 0; i < lhs.GetWidth(); i++) {
    output[i] = lhs[i] - rhs[i];
  }
  return output;
}

inline const Vectorf &operator-=(Vectorf &lhs, const Vectorf &rhs) {
  assert(lhs.GetWidth());
  assert(lhs.GetWidth() == rhs.GetWidth());
  for (uint32 i = 0; i < lhs.GetWidth(); i++) {
    lhs[i] -= rhs[i];
  }
  return lhs;
}

inline const Vectorf operator/(const Vectorf &lhs, const Vectorf &rhs) {
  assert(lhs.GetWidth());
  assert(lhs.GetWidth() == rhs.GetWidth());
  Vectorf output(lhs.GetWidth());
  for (uint32 i = 0; i < lhs.GetWidth(); i++) {
    output[i] = lhs[i] / rhs[i];
  }
  return output;
}

inline const Vectorf &operator/=(Vectorf &lhs, const Vectorf &rhs) {
  assert(lhs.GetWidth());
  assert(lhs.GetWidth() == rhs.GetWidth());
  for (uint32 i = 0; i < lhs.GetWidth(); i++) {
    lhs[i] /= rhs[i];
  }
  return lhs;
}

inline const Vectorf operator*(const Vectorf &lhs, float64 rhs) {
  assert(lhs.GetWidth());
  Vectorf output(lhs.GetWidth());
  for (uint32 i = 0; i < lhs.GetWidth(); i++) {
    output[i] = lhs[i] * rhs;
  }
  return output;
}

inline const Vectorf &operator*=(const Vectorf &lhs, float64 rhs) {
  assert(lhs.GetWidth());
  for (uint32 i = 0; i < lhs.GetWidth(); i++) {
    lhs[i] *= rhs;
  }
  return lhs;
}

inline const Vectorf operator/(const Vectorf &lhs, float64 rhs) {
  assert(lhs.GetWidth());
  Vectorf output(lhs.GetWidth());
  for (uint32 i = 0; i < lhs.GetWidth(); i++) {
    output[i] = lhs[i] / rhs;
  }
  return output;
}

inline const Vectorf &operator/=(Vectorf &lhs, float64 rhs) {
  assert(lhs.GetWidth());
  for (uint32 i = 0; i < lhs.GetWidth(); i++) {
    lhs[i] /= rhs;
  }
  return lhs;
}

class Matrixf {
 public:
  // Matrices are zeroed out at init by default.
  Matrixf(uint32 width, uint32 height);
  // Returns the dot product of a vector and a row in the matrix.
  float64 Dot(const Vectorf &input, uint32 row) const;
  // Sets a value in the matrix.
  void SetValue(uint32 x, uint32 y, float64 value);
  // Returns a value from the matrix.
  float GetValue(uint32 x, uint32 y) const;
  // Zeroes out the matrix.
  void Zero();
  // Returns the width of the matrix.
  uint32 GetWidth() const;
  // Returns the height of the matrix.
  uint32 GetHeight() const;
  // Sets the values in the matrix to a random normal distribution.
  void Randomize();
  // Returns a reference to the vector at index.
  Vectorf &operator[](uint32 index) const;

 private:
  std::vector<Vectorf> m_;
  uint32 width_;
  uint32 height_;
};

inline const Matrixf operator+(const Matrixf &lhs, const Matrixf &rhs) {
  assert(lhs.GetWidth() && lhs.GetHeight());
  assert(lhs.GetWidth() == rhs.GetWidth());
  assert(lhs.GetHeight() == rhs.GetHeight());
  Matrixf output(lhs.GetWidth(), lhs.GetHeight());
  for (uint32 i = 0; i < output.GetHeight(); i++) {
    output[i] = lhs[i] + rhs[i];
  }
  return output;
}

inline const Matrixf &operator+=(Matrixf &lhs, const Matrixf &rhs) {
  assert(lhs.GetWidth() && lhs.GetHeight());
  assert(lhs.GetWidth() == rhs.GetWidth());
  assert(lhs.GetHeight() == rhs.GetHeight());
  for (uint32 i = 0; i < lhs.GetHeight(); i++) {
    lhs[i] += rhs[i];
  }
  return lhs;
}

inline const Matrixf operator-(const Matrixf &lhs, const Matrixf &rhs) {
  assert(lhs.GetWidth() && lhs.GetHeight());
  assert(lhs.GetWidth() == rhs.GetWidth());
  assert(lhs.GetHeight() == rhs.GetHeight());
  Matrixf output(lhs.GetWidth(), lhs.GetHeight());
  for (uint32 i = 0; i < output.GetHeight(); i++) {
    output[i] = lhs[i] - rhs[i];
  }
  return output;
}

inline const Matrixf &operator-=(Matrixf &lhs, const Matrixf &rhs) {
  assert(lhs.GetWidth() && lhs.GetHeight());
  assert(lhs.GetWidth() == rhs.GetWidth());
  assert(lhs.GetHeight() == rhs.GetHeight());
  for (uint32 i = 0; i < lhs.GetHeight(); i++) {
    lhs[i] -= rhs[i];
  }
  return lhs;
}

inline const Matrixf operator*(const Matrixf &lhs, float64 rhs) {
  assert(lhs.GetHeight() && lhs.GetWidth());
  Matrixf output(lhs.GetWidth(), lhs.GetHeight());
  for (uint32 i = 0; i < output.GetHeight(); i++) {
    output[i] = lhs[i] * rhs;
  }
  return output;
}

inline const Matrixf &operator*=(const Matrixf &lhs, float64 rhs) {
  assert(lhs.GetHeight() && lhs.GetWidth());
  for (uint32 i = 0; i < lhs.GetHeight(); i++) {
    lhs[i] *= rhs;
  }
  return lhs;
}

inline const Vectorf operator*(const Matrixf &lhs, const Vectorf &rhs) {
  assert(lhs.GetHeight());
  assert(rhs.GetWidth() == lhs.GetWidth());
  Vectorf output(lhs.GetHeight());
  for (uint32 i = 0; i < output.GetWidth(); i++) {
    output[i] = lhs.Dot(rhs, i);
  }
  return output;
}

inline float64 sigmoid(float64 input) { return 1.0 / (1.0 + exp(-input)); }

inline Vectorf sigmoid(const Vectorf &input) {
  Vectorf output(input.GetWidth());
  for (uint32 i = 0; i < output.GetWidth(); i++) {
    output[i] = sigmoid(input[i]);
  }
  return output;
}

inline float64 sigmoid_prime(float64 input) {
  return sigmoid(input) * (1.0 - sigmoid(input));
}

inline Vectorf sigmoid_prime(const Vectorf &input) {
  Vectorf output(input.GetWidth());
  for (uint32 i = 0; i < output.GetWidth(); i++) {
    output[i] = sigmoid_prime(input[i]);
  }
  return output;
}

inline float64 relu(float64 input) { return input > 0.0 ? input : 0.0; }

inline Vectorf relu(const Vectorf &input) {
  Vectorf output(input.GetWidth());
  for (uint32 i = 0; i < output.GetWidth(); i++) {
    output[i] = relu(input[i]);
  }
  return output;
}

inline float64 relu_prime(float64 input) { return input > 0 ? 1.0 : 0.0; }

inline Vectorf relu_prime(const Vectorf &input) {
  Vectorf output(input.GetWidth());
  for (uint32 i = 0; i < output.GetWidth(); i++) {
    output[i] = relu_prime(input[i]);
  }
  return output;
}

inline float64 tanh(float64 input) {
  return 2.0 / (1.0 + exp(-2.0 * input)) - 1.0;
}

inline Vectorf tanh(const Vectorf &input) {
  Vectorf output(input.GetWidth());
  for (uint32 i = 0; i < output.GetWidth(); i++) {
    output[i] = tanh(input[i]);
  }
  return output;
}

inline float64 tanh_prime(float64 input) {
  return 1.0 - tanh(input) * tanh(input);
}

inline Vectorf tanh_prime(const Vectorf &input) {
  Vectorf output(input.GetWidth());
  for (uint32 i = 0; i < output.GetWidth(); i++) {
    output[i] = tanh_prime(input[i]);
  }
  return output;
}

}  // namespace base

#endif  // __NETMATH_H__
