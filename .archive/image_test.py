import cv2
import pytesseract
import numpy as np

image_path = './photo_out.jpg'

image = cv2.imread(image_path)
# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use Tesseract to extract text from the cropped region
# Use Tesseract to extract text from the cropped region
text = pytesseract.image_to_string(gray)
cleaned_text = text.strip()
print(cleaned_text)

# Display the result
cv2.imshow("Masked Image", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
