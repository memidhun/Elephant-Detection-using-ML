from keras.models import load_model
import cv2
import numpy as np
import os
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = ["0 ELEPHANT", "1 OTHERS"]

# Define the test data paths
elephant_path = (
    r"C:\Users\Midhun Mathew\Desktop\ELEPHANT PROJECT CODE\ML MODEL\MAIN\0 ELEPHANT"
)
others_path = (
    r"C:\Users\Midhun Mathew\Desktop\ELEPHANT PROJECT CODE\ML MODEL\MAIN\1 OTHERS"
)

# Initialize lists to hold images and labels
test_images = []
test_labels = []

# Read the elephant images
for img_name in os.listdir(elephant_path):
    img_path = os.path.join(elephant_path, img_name)
    image = cv2.imread(img_path)
    if image is not None:
        image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image_norm = image_resized / 255.0
        test_images.append(image_norm)
        test_labels.append("0 ELEPHANT")  # Label all images as "0 ELEPHANT"

# Read the other images
for img_name in os.listdir(others_path):
    img_path = os.path.join(others_path, img_name)
    image = cv2.imread(img_path)
    if image is not None:
        image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image_norm = image_resized / 255.0
        test_images.append(image_norm)
        test_labels.append("1 OTHERS")  # Label all images as "1 OTHERS"

test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Encode labels to integers
label_encoder = LabelEncoder()
test_labels_encoded = label_encoder.fit_transform(test_labels)

# Predict using the model
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)

# Calculate accuracy
accuracy = accuracy_score(test_labels_encoded, predicted_classes)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Print detailed classification report
report = classification_report(
    test_labels_encoded, predicted_classes, target_names=class_names
)
print("Classification Report:\n", report)
