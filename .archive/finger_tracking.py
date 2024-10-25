import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Hand and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set up video capture (webcam)
cap = cv2.VideoCapture(0)

# Initialize variables for trail and timer
finger_trail = []
start_time = time.time()  # Record the initial time

# Use MediaPipe hands model
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB (required for MediaPipe)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Check if any hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get the tip of the index finger (landmark 8 in MediaPipe Hands)
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Convert normalized coordinates to pixel coordinates
                h, w, _ = frame.shape
                finger_x = int(index_finger_tip.x * w)
                finger_y = int(index_finger_tip.y * h)

                # Add the finger position to the trail
                finger_trail.append((finger_x, finger_y))

                # Draw hand landmarks on the frame (for debugging)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Get the current time and check if 5 seconds have passed
        current_time = time.time()
        if current_time - start_time > 5:
            # Clear the trail after 5 seconds
            finger_trail = []
            start_time = current_time  # Reset the timer

        # Draw the finger trail on the frame
        for i in range(1, len(finger_trail)):
            if finger_trail[i-1] and finger_trail[i]:
                cv2.line(frame, finger_trail[i-1], finger_trail[i], (0, 255, 0), 3)

        # Show the frame
        cv2.imshow('Finger Tracking', frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
