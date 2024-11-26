# The entire point of this test code is to find a way to detect a third boundary that splits the right and left
# side of text to ensure we crop the photo of only the left or right side

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hand and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set up video capture (webcam)
cap = cv2.VideoCapture(0)

# Initialize variables for storing coordinates and photo count
index_finger_y, thumb_y = None, None
photo_num = 0

rel_photo_path = "../photos/finger_bracket/"

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB (required for MediaPipe)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Get frame dimensions
        h, w, _ = frame.shape


        # TODO: Implement something similar to this, IE ignoring a certain percentage of the frame width for edge detection
        # Define a region of interest (ROI) to ignore edges near the left and right borders
        # roi_width = int(frame.shape[1] * 0.2)  # Ignore the outer 20% of the frame width
        # edges[:, :roi_width] = 0               # Ignore left border
        # edges[:, -roi_width:] = 0              # Ignore right border

        # Convert to grayscale and apply Canny edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Sum the vertical edges along the x-axis to find the spine position
        column_sums = np.sum(edges, axis=0)
        spine_x = np.argmax(column_sums)  # Get the x-coordinate with the highest edge density (spine)

        # Draw the detected book spine boundary as a red vertical line
        cv2.line(frame, (spine_x, 0), (spine_x, h), (0, 0, 255), 2)

        # Check if any hands are detected
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

        # Display the frame with the red spine boundary
        cv2.imshow('Finger Tracking with Spine Detection', frame)

        # Check for key presses
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            # Capture and save photo when space bar is pressed
            if index_finger_y is not None and thumb_y is not None:
                # Ensure correct cropping direction
                y1, y2 = sorted((index_finger_y, thumb_y))

                # Crop the frame between thumb and index finger lines
                cropped_frame = frame[y1:y2, :]

                # Save the cropped frame
                photo_path = f'{rel_photo_path}cropped_photo_{photo_num}.png'
                cv2.imwrite(photo_path, cropped_frame)
                print(f"Saved cropped photo: {photo_path}")
                photo_num += 1

# Release resources
cap.release()
cv2.destroyAllWindows()
