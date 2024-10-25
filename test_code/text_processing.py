# TODO: Make better
def preprocess_image(img_path: str):
    # Read the image
    img = cv2.imread(img_path)

    # Apply binary threshold to make the text stand out more
    _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)

    # Apply morphological operations to reduce noise (optional but can improve quality)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Display the result
    cv2.imshow("Image", morph)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return morph

def postprocess_text(raw_text: str) -> str:
    # Remove unnecessary newlines (e.g., when the OCR mistakenly splits words)
    clean_text = re.sub(r'\n+', ' ', raw_text)

    # Strip leading/trailing whitespace and fix double spaces
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    return clean_text

def get_image_text(img_path: str) -> str:
    # Step 1: Preprocess the image
    processed_image = preprocess_image(img_path)
    # img = cv2.imread(img_path)

    # Step 2: Use Tesseract to extract text from the preprocessed image
    raw_text = pytesseract.image_to_string(processed_image, config='--psm 6')
    # raw_text = pytesseract.image_to_string(img, config='--psm 6')
    print("raw_text: ", raw_text)

    # Step 3: Postprocess the extracted text to remove noise
    clean_text = postprocess_text(raw_text)

    return clean_text
