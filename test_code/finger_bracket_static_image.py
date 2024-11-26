# The entire point of this test code is to find a way to detect a third boundary that splits the right and left
# side of text to ensure we crop the photo of only the left or right side

import cv2
import mediapipe as mp
import numpy as np

# Path to the static image you want to process
image_path = "../photos/finger_bracket_test/open_hand.jpg"
output_path = "../photos/finger_bracket/processed_photo.jpg"

# Initialize MediaPipe Hand and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load the static image
frame = cv2.imread(image_path)
if frame is None:
    print("Error: Could not load image.")
    exit()

# TODO: Does it though?
# Convert the frame to RGB (required for MediaPipe)
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Get frame dimensions
h, w, _ = frame.shape

# Initialize the MediaPipe Hands model
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    # Process the static image
    results = hands.process(frame_rgb)

    # Apply edge detection on the grayscale version of the image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Sum the vertical edges along the x-axis to find the spine position
    column_sums = np.sum(edges, axis=0)
    spine_x = np.argmax(column_sums)  # Get the x-coordinate with the highest edge density (spine)

    # Draw the detected book spine boundary as a red vertical line
    cv2.line(frame, (spine_x, 0), (spine_x, h), (0, 0, 255), 2)

    # Check if any hands are detected in the image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get index finger and thumb y-coordinates
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # Convert normalized coordinates to pixel coordinates
            index_finger_x, index_finger_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # Draw circles only at the index fingertip and thumb tip
            cv2.circle(frame, (index_finger_x, index_finger_y), 10, (0, 255, 0), -1)  # Green for index finger
            cv2.circle(frame, (thumb_x, thumb_y), 10, (255, 0, 0), -1)  # Blue for thumb

            # Draw the horizontal lines at thumb and index finger y-coordinates
            cv2.line(frame, (0, index_finger_y), (w, index_finger_y), (0, 255, 0), 2)
            cv2.line(frame, (0, thumb_y), (w, thumb_y), (255, 0, 0), 2)

            # Crop the frame between thumb and index finger lines if they exist
            y1, y2 = sorted((index_finger_y, thumb_y))
            cropped_frame = frame[y1:y2, :]

            # Save the cropped frame with left or right half
            cv2.imwrite(f"{output_path}_cropped.jpg", cropped_frame)

    # Display the result with spine line and landmarks
    cv2.imshow("Processed Image", frame)
    cv2.imwrite(output_path, frame)
    print(f"Saved processed image with spine boundary: {output_path}")

    # Hold the display until a key is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()
