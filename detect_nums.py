import cv2
import numpy as np
import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Prepare the training data from labeled images
X = []
y = []

# Loop through all labeled images in the 'labeled_rois' folder
for filename in os.listdir('labeled_rois'):
    if filename.endswith('.jpg'):
        # Extract the label from the filename (e.g., 'roi_1_5.jpg' -> '5')
        label = filename.split('_')[-1].split('.')[0]

        # Load the labeled RoI
        roi_path = os.path.join('labeled_rois', filename)
        roi = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)

        # Resize the RoI to a fixed size (e.g., 20x20)
        roi_resized = cv2.resize(roi, (20, 20))

        # Flatten the RoI into a feature vector
        X.append(roi_resized.flatten())
        y.append(label)

# Convert the data to numpy arrays
X = np.array(X)
y = np.array(y)

# Step 2: Train a k-Nearest Neighbors (k-NN) classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluate the classifier
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Classifier accuracy: {accuracy * 100:.2f}%")

# Step 3: Detect and recognize characters in the image
# Load the image
image = cv2.imread('rottilt.jpg')

# Preprocess the image (same as before)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(binary, kernel, iterations=1)
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on width and height (same as before)
min_width = 40
max_width = 1000
min_height = 70
max_height = 1000

filtered_contours = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if min_width <= w <= max_width and min_height <= h <= max_height:
        filtered_contours.append(contour)

# Step 4: Detect and recognize characters
detected_characters = []
for contour in filtered_contours:
    x, y, w, h = cv2.boundingRect(contour)
    roi = binary[y:y + h, x:x + w]
    roi_resized = cv2.resize(roi, (20, 20))
    roi_flattened = roi_resized.flatten().reshape(1, -1)

    # Predict the label using the trained classifier
    label = knn.predict(roi_flattened)[0]
    detected_characters.append((x, y, w, h, label))

    # Draw the bounding box and label on the original image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)  # Larger font size


# Step 5: Display the final image with detected characters
def resize_to_fit(image, max_width=1200, max_height=900):
    """Resize an image to fit within the specified dimensions while maintaining aspect ratio."""
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)
    return cv2.resize(image, (int(width * scale), int(height * scale)))


final_resized = resize_to_fit(image)
cv2.imshow('Detected Characters', final_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 6: Save the results
detected_data = pd.DataFrame({
    'x': [x for (x, y, w, h, label) in detected_characters],
    'y': [y for (x, y, w, h, label) in detected_characters],
    'width': [w for (x, y, w, h, label) in detected_characters],
    'height': [h for (x, y, w, h, label) in detected_characters],
    'label': [label for (x, y, w, h, label) in detected_characters]
})
detected_data.to_csv('detected_data.csv', index=False)

print("Detection complete. Detected characters saved in 'detected_data.csv'.")