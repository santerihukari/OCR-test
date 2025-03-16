import cv2
import numpy as np
import pandas as pd
import pytesseract

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

# Step 4: Filter contours
min_area = 100  # Minimum area for a contour to be considered a character
max_area = 13000  # Maximum area for a contour to be considered a character
filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]

# Step 5: Create a 12x20x2 DataFrame
# Initialize the DataFrame with empty key-value pairs
df = pd.DataFrame(
    [[{"key": "0000", "value": "0000"} for _ in range(12)] for _ in range(20)],
    columns=[f"Col_{i+1}" for i in range(12)],
    index=[f"Row_{i+1}" for i in range(20)]
)

# Step 6: Store bounding boxes and their coordinates
bounding_boxes = []
for contour in filtered_contours:
    # Get the bounding box for the contour
    x, y, w, h = cv2.boundingRect(contour)
    bounding_boxes.append((x, y, w, h))

    # Draw bounding box and label
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green bounding box
    cv2.putText(image, "Click to detect", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Label above the box

# Step 7: Mouse callback function
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button clicked
        for i, (bx, by, bw, bh) in enumerate(bounding_boxes):
            if bx <= x <= bx + bw and by <= y <= by + bh:  # Check if click is inside a bounding box
                # Extract the ROI
                roi = binary[by:by + bh, bx:bx + bw]

                # Use Tesseract to detect the number
                custom_config = r'--oem 1 --psm 10 -c tessedit_char_whitelist=0123456789'  # Treat ROI as a single character, whitelist digits
                detected_text = pytesseract.image_to_string(roi, config=custom_config).strip()

                # Print the detected number
                print(f"Detected number: {detected_text}")

                # Update the DataFrame (example: update column 3, row 5)
                update_dataframe(df, col=3, row=5, key="1234", value=detected_text)

                # Highlight the clicked bounding box
                cv2.rectangle(image, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)  # Red bounding box
                cv2.putText(image, detected_text, (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)  # Label above the box
                cv2.imshow('Processed Image', image)  # Refresh the display

# Step 8: Function to update the DataFrame
def update_dataframe(df, col, row, key, value):
    """Update the DataFrame with the detected key-value pair."""
    if 1 <= col <= 12 and 1 <= row <= 20:
        df.at[f"Row_{row}", f"Col_{col}"] = {"key": key, "value": value}
        print(f"Updated DataFrame at Column {col}, Row {row}: Key = {key}, Value = {value}")
    else:
        print("Invalid column or row index. Column must be between 1 and 12, and row must be between 1 and 20.")

# Step 9: Set up the mouse callback
cv2.namedWindow('Processed Image', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Processed Image', mouse_callback)

# Step 10: Display the processed image
cv2.resizeWindow('Processed Image', 800, 600)  # Width = 800, Height = 600
cv2.imshow('Processed Image', image)

# Wait for a key press to close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the final DataFrame
print("Final DataFrame:")
print(df)