# WildAnimalDetector

WildAnimalDetector is a machine learning project designed to detect wild animals, such as elephants, using OpenCV and TensorFlow. This project aims to help in monitoring and protecting areas from potential animal intrusions by accurately identifying specific animals. Right now the model is trained only for detecting elephants but the model will be updated frequently.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model and Labels](#model-and-labels)
- [Code](#code)
- [Contributing](#contributing)
- [License](#license)

## Introduction
The WildAnimalDetector project uses computer vision and deep learning techniques to detect the presence of wild animals in real-time. It leverages OpenCV for image processing and TensorFlow for the machine learning model. The primary goal is to provide a reliable system for wildlife monitoring and prevention of human-wildlife conflicts.

## Features
- Real-time animal detection
- High accuracy with pre-trained TensorFlow model
- Easy integration with camera systems
- Supports multiple animal species detection

## Installation
To get started with WildAnimalDetector, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/memidhun/Elephant-Detection-using-ML
    cd WildAnimalDetector
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

### requirements.txt
```
opencv-python-headless==4.5.5.64
tensorflow==2.9.1
numpy==1.21.6
Pillow==9.1.1
imutils==0.5.4
```

## Usage
To run the animal detection model, use the following command:
```sh
python main.py
```

Ensure that your camera is properly connected and configured.

## Model and Labels
The model and label files are included in the `model` directory. If you wish to use a different model, replace the existing files with your own.

- `model/saved_model.pb` - The pre-trained TensorFlow model.
- `model/labels.txt` - The label file containing the names of the detected animals.

## Code
The main script to run the detection is `detect_animals.py`. This script loads the model, processes the video feed from the camera, and performs the detection in real-time.

## Contributing
Contributions are welcome! If you have any suggestions or improvements, feel free to open an issue or submit a pull request.

## License
This project has been applied to patents.
