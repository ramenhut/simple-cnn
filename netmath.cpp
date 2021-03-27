#include "netmath.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <random>

namespace base {

Vectorf::Vectorf(uint32 width) {
  v_.resize(width);
  Zero();
}

uint32 Vectorf::GetWidth() const { return v_.size(); }

void Vectorf::Zero() {
  for (auto& v : v_) v = 0.0;
}

void Vectorf::Resize(uint32 new_size) {
  assert(new_size);
  v_.resize(new_size);
  Zero();
}

uint32 Vectorf::GetMaxIndex() const {
  assert(v_.size());
  uint32 max_index = 0;
  float64 max_value = v_[0];
  for (uint32 i = 1; i < v_.size(); i++) {
    if (v_[i] > max_value) {
      max_value = v_[i];
      max_index = i;
    }
  }
  return max_index;
}

void Vectorf::Randomize() {
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::normal_distribution<float64> distribution;

  for (auto& v : v_) {
    v = distribution(generator);
  }
}

float64 Vectorf::Average() const {
  float64 avg = 0.0;
  for (auto v : v_) {
    avg += v;
  }
  return avg / v_.size();
}

float64 Vectorf::Dot(const Vectorf& rhs) const {
  assert(rhs.GetWidth() == GetWidth());
  float64 accum = 0.0f;
  for (uint32 i = 0; i < v_.size(); i++) {
    accum += rhs[i] * v_[i];
  }
  return accum;
}

float64& Vectorf::operator[](uint32 index) const {
  return const_cast<float64&>(v_.at(index));
}

Matrixf::Matrixf(uint32 width, uint32 height) {
  assert(width);
  assert(height);
  width_ = width;
  height_ = height;

  for (uint32 h = 0; h < height; h++) {
    m_.push_back(Vectorf(width));
  }
}

float64 Matrixf::Dot(const Vectorf& input, uint32 row) const {
  assert(input.GetWidth() == width_);
  return m_[row].Dot(input);
}

void Matrixf::SetValue(uint32 x, uint32 y, float64 value) { m_[y][x] = value; }

float Matrixf::GetValue(uint32 x, uint32 y) const { return m_[y][x]; }

uint32 Matrixf::GetWidth() const { return width_; }

uint32 Matrixf::GetHeight() const { return height_; }

void Matrixf::Zero() {
  for (uint32 y = 0; y < height_; y++) m_[y].Zero();
}

void Matrixf::Randomize() {
  for (uint32 y = 0; y < height_; y++) m_[y].Randomize();
}

Vectorf& Matrixf::operator[](uint32 index) const {
  return const_cast<Vectorf&>(m_.at(index));
}

}  // namespace base
