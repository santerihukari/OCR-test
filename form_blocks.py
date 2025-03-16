import pandas as pd
import numpy as np
import cv2
from sklearn.cluster import DBSCAN

# Load the detected data
data = pd.read_csv("detected_data.csv")

# Filter out rows with NaN labels
data = data.dropna(subset=["label"])

# Extract (x, y) coordinates for clustering
coordinates = data[["x", "y"]].values

# Use DBSCAN to cluster the detected numbers into blocks
# eps: The maximum distance between two samples for them to be considered as in the same neighborhood.
# min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
dbscan = DBSCAN(eps=100, min_samples=4)  # Adjust eps and min_samples as needed
data["block"] = dbscan.fit_predict(coordinates)

# Load the original image
image = cv2.imread("rottilt.jpg")

# Draw bounding boxes around each block
for block_id in data["block"].unique():
    if block_id == -1:
        continue  # Skip noise points (outliers)

    # Get the coordinates of numbers in this block
    block_data = data[data["block"] == block_id]
    x_min = block_data["x"].min()
    x_max = block_data["x"].max()
    y_min = block_data["y"].min()
    y_max = block_data["y"].max()

    # Draw the bounding box on the image
    cv2.rectangle(
        image,
        (int(x_min), int(y_min)),
        (int(x_max), int(y_max)),
        (0, 255, 0),  # Green color for the bounding box
        2,  # Thickness of the bounding box
    )

# Resize the image to fit the screen
screen_width = 1920  # Adjust this to your screen width
screen_height = 1080  # Adjust this to your screen height
scale_width = screen_width / image.shape[1]
scale_height = screen_height / image.shape[0]
scale = min(scale_width, scale_height)
window_width = int(image.shape[1] * scale)
window_height = int(image.shape[0] * scale)
resized_image = cv2.resize(image, (window_width, window_height))

# Display the image in a resizable window
cv2.namedWindow("Block Visualization", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Block Visualization", window_width, window_height)
cv2.imshow("Block Visualization", resized_image)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()