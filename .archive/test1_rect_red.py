import cv2
import numpy as np

image_path = './photo.jpg'

# Read the image
image = cv2.imread(image_path)

# Convert the image to the HSV color space for better color segmentation
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define lower and upper bounds for the red color in HSV space
# Red color in HSV can wrap around, so we have two ranges
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

# Create two masks for the red color (both parts of the red spectrum)
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

# Combine the two masks
red_mask = mask1 + mask2

# Optional: Apply some morphological operations to clean up the mask
kernel = np.ones((5, 5), np.uint8)
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

# Create a masked image that only contains the red areas
result = cv2.bitwise_and(image, image, mask=red_mask)

# Resize the result (if needed)
result = cv2.resize(result, (3480, 2160))  # Resize image to your preferred size

# Display the result
cv2.imshow("Red Masked Image", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit(0)
