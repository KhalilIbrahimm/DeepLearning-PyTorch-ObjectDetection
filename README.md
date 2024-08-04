# Deep Learning Object Detection with PyTorch

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Project Overview

This project implements advanced object localization and detection techniques using Convolutional Neural Networks (CNNs) in PyTorch. We work with an augmented version of the MNIST dataset to tackle two main challenges:

1. **Object Localization**: Classifying a single object in an image and predicting its bounding box.
2. **Object Detection**: Identifying and classifying multiple objects in an image, with bounding boxes for each.

## Key Features

- Custom CNN architectures optimized for object localization and detection
- Specialized loss functions for accurate bounding box prediction
- Performance evaluation using accuracy and Intersection over Union (IoU)
- Visualization tools for predicted vs. ground truth bounding boxes
- Grid-based approach for multi-object detection

## Dataset

We use an enhanced version of MNIST with the following modifications:
- Image dimensions: 48 x 60 pixels
- Randomly positioned, rotated, and resized digits
- Added background noise for increased complexity

## Technologies

- Python 3.7+
- PyTorch 1.8+
- torchvision
- NumPy
- Matplotlib
- Jupyter Notebook

## Getting Started
1. Clone the repository:
```bash
git clone git@github.com:KhalilIbrahimm/DeepLearning-PyTorch-ObjectDetection.git
```

3. Set up a virtual environment and activate it:
```bash
python -m venv venv
```

```bash
source venv/bin/activate  # On Windows, use venv\Scripts\activate
```
5. Install the required dependencies:
```bash
pip install -r requirements.txt
```

6. Launch Jupyter Notebook:
jupyter notebook

5. Open and run the notebooks in the `notebooks/` directory.

# Results
Our best models achieved:

### Object Localization:
  - Accuracy: 91.63%
  - IoU: 0.4808
  - Mean Performance: 0.6985

### Object Detection:
  - Accuracy: 33.46%
  - IoU: 0.7322
  - Mean Performance: 0.5334

Note: The object detection accuracy is lower due to class imbalance in the grid-based approach, with empty grid cells dominating. The high IoU suggests good bounding box prediction despite the accuracy metric limitations.

## Future Improvements

- Implement more advanced architectures like YOLO or SSD
- Experiment with different data augmentation techniques
- Extend the model to work with more complex, real-world datasets
- Optimize for real-time detection on video streams

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## Contact

Khalil Ibrahim - [GitHub](https://github.com/KhalilIbrahimm)

Project Link: [https://github.com/KhalilIbrahimm/DeepLearning-PyTorch-ObjectDetection](https://github.com/KhalilIbrahimm/DeepLearning-PyTorch-ObjectDetection)
