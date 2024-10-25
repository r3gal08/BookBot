# Python program to explain cv2.imshow() method

# importing cv2
import cv2
import time
import numpy as np
import pytesseract



# path
path = r'photo.jpg'

# # Reading an image in grayscale mode
# image = cv2.imread(path, 0)
#
# # Convert image to HSV color space
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#
# # Window name in which image is displayed
# window_name = 'image'
#
# # Using cv2.imshow() method
# # Displaying the image
# cv2.imshow(window_name, image)
#
# # waits for user to press any key
# # (this is necessary to avoid Python kernel form crashing)
# cv2.waitKey(0)
#
# # closing all open windows
# cv2.destroyAllWindows()

# Python program to explain cv2.cvtColor() method


# Window name in which image is displayed
# window_name = 'Image'
#
# # Using cv2.cvtColor() method
# # Using cv2.COLOR_BGR2GRAY color space
# # conversion code
# image = cv2.imread(path)
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#
# cv2.imshow(window_name, hsv)
#
# # waits for user to press any key
# # (this is necessary to avoid Python kernel form crashing)
# cv2.waitKey(0)
#
# # closing all open windows
# cv2.destroyAllWindows()


# Read the image
color = 'red'
image = cv2.imread(path)


# Convert image to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color ranges for red and gray in HSV
if color == 'red':
    # Red has two ranges in HSV space
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red color in both ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2  # Combine both masks
elif color == 'gray':
    # Define range for gray color in HSV
    lower_gray = np.array([0, 0, 50])  # Low saturation, varying brightness
    upper_gray = np.array([180, 50, 200])
    mask = cv2.inRange(hsv, lower_gray, upper_gray)
else:
    print("Unsupported color.")

# Use the mask to isolate colored regions in the image
result = cv2.bitwise_and(image, image, mask=mask)

# Convert masked image to grayscale and find contours
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# Detect contours of the circles
contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

extracted_text = ""

# Loop over the contours to find circles
for contour in contours:
    # Get the circularity of the contour
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if perimeter == 0:
        continue
    circularity = 4 * np.pi * (area / (perimeter ** 2))

    # Filter for circular shapes
    if 0.7 < circularity < 10:  # Approximate range for circularity
        # Get bounding box of the circular region
        x, y, w, h = cv2.boundingRect(contour)

        # Crop the circular region
        cropped_circle = gray[y:y+h, x:x+w]
        # Display the result

        # Use Tesseract to extract text from the cropped region
        text = pytesseract.image_to_string(cropped_circle)
        extracted_text += text + "\n"

        # Draw the detected circle (for visualization)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

print("Text: \n")
print(extracted_text)