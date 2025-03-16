import cv2
import numpy as np

# Create a folder to save the templates
import os
if not os.path.exists('templates'):
    os.makedirs('templates')

# Define the size of the template images
template_size = (20, 20)  # Width = 20, Height = 20

# Define the font and scale for drawing digits
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8
font_thickness = 2

# Generate template images for digits 0-9
for digit in range(10):
    # Create a blank white image
    template = np.ones((template_size[1], template_size[0]), dtype=np.uint8) * 255

    # Draw the digit in the center of the image
    text = str(digit)
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = (template_size[0] - text_size[0]) // 2
    text_y = (template_size[1] + text_size[1]) // 2
    cv2.putText(template, text, (text_x, text_y), font, font_scale, 0, font_thickness)

    # Save the template image
    cv2.imwrite(f'templates/{digit}.jpg', template)

    # Display the template (optional)
    cv2.imshow(f'Digit {digit}', template)
    cv2.waitKey(100)  # Display each template for 100 ms

# Close all OpenCV windows
cv2.destroyAllWindows()

print("Template images generated and saved in the 'templates' folder.")