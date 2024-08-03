"""
WildAnimalDetector - A Machine Learning project for detecting wild animals using OpenCV and TensorFlow.

Author: Midhun Mathew
Date: April 15,2024
License: Applied for Patents

Description:
This script loads the pre-trained model and processes the video feed from the camera to detect wild animals in real-time.

GitHub Repository: https://github.com/memidhun/Elephant-Detection-using-ML
"""

# Midhun Mathew , Anshu Mohanan , Sarin C ROY
from keras.models import load_model
import cv2
import numpy as np

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Define camera object (0 for default camera)
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcam image
    ret, image = camera.read()

    # Resize the image to model input shape
    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Normalize the image
    image_norm = (image_resized / 255.0).reshape(1, 224, 224, 3)

    # Predict the class probabilities
    prediction = model.predict(image_norm)

    # Get the predicted class index and confidence score
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Draw a rectangle around the identified object if it's an elephant
    if class_name == "0 ELEPHANT":
        cv2.rectangle(image, (0, 0), (image.shape[1], image.shape[0]), (0, 0, 255), 2)
        cv2.putText(
            image,
            "Alert - Elephant Detected!",
            (20, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (0, 0, 255),
            2,
        )
    if class_name == "1 OTHERS":
        cv2.rectangle(image, (0, 0), (image.shape[1], image.shape[0]), (0, 255, 0), 2)

    # Display class and confidence score on the image
    cv2.putText(
        image,
        f"Class: {class_name}, Confidence: {confidence_score:.2f}",
        (10, image.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )

    # Show the image with predictions
    cv2.imshow("Webcam Image", image)

    # Listen to the keyboard for presses
    keyboard_input = cv2.waitKey(1)

    # Check if the ESC key is pressed
    if keyboard_input == 27:
        break

# Release the camera and close all windows
camera.release()
cv2.destroyAllWindows()
