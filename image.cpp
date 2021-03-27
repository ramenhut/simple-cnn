
#include "image.h"

#include <fstream>
#include <utility>

namespace base {

const uint32 kBackgroundClassLabel = 10;

SampleSet::SampleSet(uint32 sample_width, uint32 label_width)
    : sample(sample_width), label(label_width) {}

bool LoadImageSet(const std::string& images_filename,
                  const std::string& labels_filename,
                  std::vector<SampleSet>* output, std::string* error) {
  if (images_filename.empty() || labels_filename.empty() || !output) {
    if (error) {
      *error = "Invalid inputs to LoadImageSet.";
    }
    return false;
  }

  MNIST_IMAGE_FILE_HEADER image_header;
  MNIST_LABEL_FILE_HEADER label_header;

  std::ifstream image_source(images_filename, std::ios::in | std::ios::binary);
  std::ifstream label_source(labels_filename, std::ios::in | std::ios::binary);

  if (!image_source || !label_source) {
    if (error) {
      *error = "Failed to open data files.";
    }
    return false;
  }

  // Read in both headers. Data counts must match.
  if (!image_source.read((char*)&image_header,
                         sizeof(MNIST_IMAGE_FILE_HEADER)) ||
      !label_source.read((char*)&label_header,
                         sizeof(MNIST_LABEL_FILE_HEADER))) {
    if (error) {
      *error = "Failed to read MNIST image and/or label file headers.";
    }
    return false;
  }

  // Parse the image header, swapping the dwords into little endian.
  image_header.magic = EndianSwap8in32(image_header.magic);
  image_header.image_count = EndianSwap8in32(image_header.image_count);
  image_header.width = EndianSwap8in32(image_header.width);
  image_header.height = EndianSwap8in32(image_header.height);

  // Parse the label header, swapping the dwords into little endian.
  label_header.magic = EndianSwap8in32(label_header.magic);
  label_header.label_count = EndianSwap8in32(label_header.label_count);

  if (image_header.magic != 2051 || label_header.magic != 2049) {
    if (error) {
      *error = "Invalid MNIST data file(s) detected.";
    }
    return false;
  }

  if (image_header.image_count != label_header.label_count) {
    if (error) {
      *error = "Image and label count mismatch.";
    }
    return false;
  }

  // File reads have a relatively high fixed cost, so we load the entire
  //   data set into memory and then scatter afterwards.
  std::vector<uint8> image_file_buffer(
      image_header.image_count * image_header.width * image_header.height);

  std::vector<uint8> label_file_buffer(label_header.label_count);

  // Read in both sets of data.
  if (!image_source.read((char*)&image_file_buffer.at(0),
                         image_header.image_count * image_header.width *
                             image_header.height)) {
    if (error) {
      *error = "Failed to read image data from disk.";
    }
    return false;
  }

  if (!label_source.read((char*)&label_file_buffer.at(0),
                         label_header.label_count)) {
    if (error) {
      *error = "Failed to read label data from disk.";
    }
    return false;
  }

  for (uint32 i = 0; i < image_header.image_count; i++) {
    output->push_back(SampleSet(image_header.width * image_header.height,
                                kBackgroundClassLabel));
    SampleSet* training_set = &output->at(output->size() - 1);

    // Populate our image data.
    for (uint32 offset = 0; offset < image_header.width * image_header.height;
         offset++) {
      training_set->sample[offset] =
          image_file_buffer.at(image_header.width * image_header.height * i +
                               offset) /
          255.0;
    }

    training_set->label.Zero();
    training_set->label[label_file_buffer.at(i)] = 1.0;
  }

  return true;
}

}  // namespace base
