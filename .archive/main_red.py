import cv2
import numpy as np
import pytesseract
import ollama
import time

# Function to detect red circles and extract text from those regions
def extract_text_from_colored_circles(image_path, color='red'):
    # Read the image
    image = cv2.imread(image_path)

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
        return None, image

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
        if 0.7 < circularity < 1.3:  # Approximate range for circularity
            # Get bounding box of the circular region
            x, y, w, h = cv2.boundingRect(contour)

            # Crop the circular region
            cropped_circle = gray[y:y+h, x:x+w]

            # Use Tesseract to extract text from the cropped region
            text = pytesseract.image_to_string(cropped_circle)
            extracted_text += text + "\n"

            # Draw the detected circle (for visualization)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    print(extracted_text)

    return extracted_text, image

# Function to ask the Ollama model a question based on extracted text
def ask_ollama_question(text, question):
    prompt = f"""
    Here is some extracted text from an image: "{text}"

    Question: {question}
    """
    response = ollama.chat(
        model="llama2",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']

# Main function
if __name__ == "__main__":
    # Path to your image with red or gray circles
    image_path = 'photo.jpg'

    # Detect and extract text from red or gray circles
    extracted_text, output_image = extract_text_from_colored_circles(image_path, color='red')

    # Show the image with detected circles
    if extracted_text:
        print("Extracted Text: ", extracted_text)
        cv2.imshow("Detected Circles", output_image)
        cv2.waitKey(0)

        # Ask a question about the extracted text using the Ollama model
        question = "What does this mean?"
        answer = ask_ollama_question(extracted_text, question)

        print("Ollama's Response: ", answer)
    else:
        print("No text found in the specified color regions.")

    cv2.destroyAllWindows()
