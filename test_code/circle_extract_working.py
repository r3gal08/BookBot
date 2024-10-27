import cv2
import pytesseract
import numpy as np

image_path = '../photos/line_bracket_test.jpg'
extracted_text = ""

# Read the image
image = cv2.imread(image_path)

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to remove noise and improve contour detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Using underscore as the first return value is something I don't care about
# Apply threshold to get a binary image (black and white)
_, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)

# Find contours in the image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a mask of the same size as the original image, initially all black
mask = np.zeros_like(image)

# Loop over the contours to find circular-like shapes or closed loops
for contour in contours:
    # Get the perimeter of the contour
    perimeter = cv2.arcLength(contour, True)
    # print("Perimeter: " + str(round(perimeter,1)) + "\n")

    # Approximate the shape of the contour (with accuracy proportional to perimeter)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    # TODO: Trial and error shows that 5000 is a decent value for this specifc image, but there should be a better way to do this
    # Check if the contour is large enough to be relevant
    if len(approx) >= 4 and perimeter > 5000:  # Filter out small/irrelevant shapes
        # Fill the contour on the mask with white (255)
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

# Apply the mask to the original image
# Use the mask to keep only the region inside the contours (everything else is black)
result = cv2.bitwise_and(image, mask)
# result = cv2.resize(result, (3480, 2160))                # Resize image


# Tesseract isn't playing nice, but realstically at this point we can send the image out to the model for text extract
# Use Tesseract to extract text from the cropped region
# Use Tesseract to extract text from the cropped region
# text = pytesseract.image_to_string(result)
# extracted_text += text + "\n"
# print(text)

# Display the result
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
cv2.imshow("Masked Image", gray)
# cv2.imshow("Masked Image", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
