import cv2
import numpy as np
import pandas as pd
import os

# Load the image
image = cv2.imread('rottilt.jpg')

# Step 1: Preprocess the image
# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply adaptive thresholding to handle uneven lighting
binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Step 2: Separate connected characters
# Use morphological dilation to separate characters
kernel = np.ones((3, 3), np.uint8)  # Adjust kernel size as needed
dilated = cv2.dilate(binary, kernel, iterations=1)

# Step 3: Find contours
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 4: Filter contours based on width and height
min_width = 40  # Minimum width for a contour to be considered a character
max_width = 1000  # Maximum width for a contour to be considered a character
min_height = 70  # Minimum height for a contour to be considered a character
max_height = 1000  # Maximum height for a contour to be considered a character

filtered_contours = []
for contour in contours:
    # Get the bounding box for the contour
    x, y, w, h = cv2.boundingRect(contour)
    # Filter based on width and height
    if min_width <= w <= max_width and min_height <= h <= max_height:
        filtered_contours.append(contour)

# Step 5: Store bounding boxes and their coordinates
bounding_boxes = []
for contour in filtered_contours:
    # Get the bounding box for the contour
    x, y, w, h = cv2.boundingRect(contour)
    bounding_boxes.append((x, y, w, h))

# Step 6: Create a folder to save labeled RoIs
if not os.path.exists('labeled_rois'):
    os.makedirs('labeled_rois')

# Step 7: Function to resize image to fit the screen
def resize_to_fit(image, max_width=800, max_height=600):
    """Resize an image to fit within the specified dimensions while maintaining aspect ratio."""
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)
    return cv2.resize(image, (int(width * scale), int(height * scale)))

# Step 8: Label each RoI
labels = []
for i, (x, y, w, h) in enumerate(bounding_boxes):
    # Extract the ROI from the binary image
    roi = binary[y:y + h, x:x + w]

    # Display the original image with the bounding box
    original_with_box = image.copy()
    cv2.rectangle(original_with_box, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green bounding box
    cv2.putText(original_with_box, f"RoI {i + 1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Label above the box

    # Display the binary image with the bounding box
    binary_with_box = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)  # Convert binary image to 3 channels for drawing
    cv2.rectangle(binary_with_box, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green bounding box
    cv2.putText(binary_with_box, f"RoI {i + 1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Label above the box

    # Resize the images to fit the screen
    original_resized = resize_to_fit(original_with_box)
    binary_resized = resize_to_fit(binary_with_box)

    # Combine the original and binary images for display
    combined = np.hstack((original_resized, binary_resized))
    cv2.imshow('Original (Left) vs Binary (Right)', combined)
    cv2.waitKey(1)  # Wait for the window to update

    # Prompt the user to label the RoI
    label = input(f"Enter label for RoI {i + 1} (0-9 or press 'a' for NaN): ").strip().upper()
    if label == "A":
        label = "NaN"
    while label not in [str(i) for i in range(10)] + ['NaN']:
        print("Invalid input. Please enter a digit (0-9) or press 'a' for 'NaN'.")
        label = input(f"Enter label for RoI {i + 1} (0-9 or press 'a' for NaN): ").strip().upper()
        if label == "A":
            label = "NaN"
    labels.append(label)

    # Save the labeled RoI
    cv2.imwrite(f'labeled_rois/roi_{i + 1}_{label}.jpg', roi)

# Step 9: Save the labeled data
labeled_data = pd.DataFrame({
    'roi_id': [f"roi_{i + 1}" for i in range(len(bounding_boxes))],
    'label': labels,
    'x': [x for (x, y, w, h) in bounding_boxes],
    'y': [y for (x, y, w, h) in bounding_boxes],
    'width': [w for (x, y, w, h) in bounding_boxes],
    'height': [h for (x, y, w, h) in bounding_boxes]
})
labeled_data.to_csv('labeled_data.csv', index=False)

# Step 10: Display the final image with labeled RoIs
final_image = image.copy()
for i, (x, y, w, h) in enumerate(bounding_boxes):
    cv2.rectangle(final_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green bounding box
    cv2.putText(final_image, labels[i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Label above the box

# Resize the final image to fit the screen
final_resized = resize_to_fit(final_image)
cv2.imshow('Final Labeled Image', final_resized)

# Wait for a key press to close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Labeling complete. Labeled RoIs saved in the 'labeled_rois' folder.")