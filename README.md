## Convolutional Neural Networks
This is a simple and flexible implementation of a convolutional neural network ensemble that enables classification of the standard MNIST handwritten digit set, with a maximum observed accuracy of 97.68%.

### Features:
-   **Configuration**: control epoch size, learning rate, mini-batch size, feature map count, and layer dimensions (size and types).
-   **Tuning**: quadratic and cross-entropy cost, L2 regularization, multiple activation functions (tanh, relu, sigmoid).
-   **Serialization**: save and load network ensembles for quick reuse.

## Usage:
If you wish to use the source *as-is*, then you'll need to download the MNIST [training and test data](bertolami.com/files/mnist_dataset.zip) and unzip it into the same folder as the program. You can train your own ensemble, but you can save yourself some time and use [my prepared 3 network ensemble](bertolami.com/files/cnn_ensemble.zip) by unzipping it to the same folder as the program.

The default build environment is [http://bazel.build](bazel), however, the source is fully cross platform (C++14) and should compile with minimal effort in most other build environments.

Running the program (on Windows) without any parameters will display the prompt below. You train one network at a time and then combine them into an ensemble that you can use for verification. If you'd like to modify any of the hyper-parameters then you'll need to download, modify, and build the source code.

```
Usage: simple-cnn-x64.exe [options]
  --train  [output network filename]                            Trains a single neural network based on the MNIST dataset.
  --combine [output ensemble filename] [network filenames]      Combines a set of network files into a single ensemble file.
  --verify [input ensemble filename]                            Tests the accuracy of a neural network ensemble against the MNIST test set.
```

## Details

This software is released under the terms of the BSD 2-Clause “Simplified” License.
For  more information visit [http://www.bertolami.com](http://bertolami.com).