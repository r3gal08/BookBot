import cv2
import mediapipe as mp
import time
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

# Function to focus the camera
def focus_camera(camera):
    # Attempt to set focus; you might need to adjust this depending on your camera
    focus_value = 50  # Example focus value (0-100), adjust as necessary
    camera.set(cv2.CAP_PROP_FOCUS, focus_value)
    time.sleep(1)  # Allow time for the camera to adjust focus

# min_detection_confidence: This is the minimum confidence level required for MediaPipe to initially detect a hand.
#                           A value of 0.7 (or 70%) means that if the confidence score of detection is below 70%, it won't consider the hand detected in that frame.
# min_tracking_confidence: This is the minimum confidence level required to continue tracking the hand landmarks once a hand is initially detected.
#                          A value of 0.7 means that if the tracking confidence drops below 70%, MediaPipe may consider the hand lost and reinitialize detection in the next frames.
# By using "with", we ensure that hands is properly closed after the code finishes, preventing memory leaks or other issues with unfreed resources.
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

        # Check if any hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get index finger and thumb y-coordinates
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                # Convert normalized coordinates to pixel coordinates
                index_finger_y = int(index_finger_tip.y * h)
                thumb_y = int(thumb_tip.y * h)

                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Draw the horizontal lines at thumb and index finger y-coordinates
                cv2.line(frame, (0, index_finger_y), (w, index_finger_y), (0, 255, 0), 2)
                cv2.line(frame, (0, thumb_y), (w, thumb_y), (255, 0, 0), 2)

        # Display the frame
        cv2.imshow('Finger Tracking', frame)

        # Check for key presses
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            # Capture and save photo when space bar is pressed
            if index_finger_y is not None and thumb_y is not None:
                # Ensure correct cropping direction
                y1, y2 = sorted((index_finger_y, thumb_y))
                focus_camera(cap)

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