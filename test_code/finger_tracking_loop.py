import cv2
import mediapipe as mp
import time
import numpy as np

# Initialize MediaPipe Hand and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set up video capture (webcam)
cap = cv2.VideoCapture(0)

# Initialize variables for trail, timer, and photo capture
finger_trail = []
start_time = time.time()  # Record the initial time
loop_completed = False

# Define a distance threshold for loop completion
LOOP_THRESHOLD = 50  # Pixels, adjust based on your needs

photo_num = 0

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

        # Draw the finger trail on the frame
        for i in range(1, len(finger_trail)):
            if finger_trail[i-1] and finger_trail[i]:
                cv2.line(frame, finger_trail[i-1], finger_trail[i], (0, 255, 0), 3)

        # TODO: Not exactly working how I intended it to......  Check with gpt4o tomorrow
        # Check for loop completion
        if len(finger_trail) > 1:
            # Calculate the distance between the last and first points in the trail
            last_point = np.array(finger_trail[-1])
            first_point = np.array(finger_trail[0])
            distance = np.linalg.norm(last_point - first_point)

            # Check if the distance is within the loop threshold
            if distance < LOOP_THRESHOLD and not loop_completed:
                # Take a photo
                cv2.imwrite('loop_photo.png', frame)
                photo_num+=1
                print("Loop completed! Photo #" +  str(photo_num) + " taken.")
                loop_completed = True  # Set the flag to avoid taking multiple photos
                # TODO: Extract text from circled area with OCR prog
                #       send OCR text to AI model
                #       By default, AI gives explanation
                #       future AGR, might present an option to the user, specific word, questions, etc...
                #       Eventually this data is saved in some form of DB... Tailoring to the user, tying thoughts together
                #       and creating discussion or even introducing to new communities?

        # Reset the trail and loop completion after a set time
        current_time = time.time()
        if current_time - start_time > 5:
            finger_trail = []
            start_time = current_time  # Reset the timer
            loop_completed = False  # Allow for new loop detection

        # Show the frame
        cv2.imshow('Finger Tracking', frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
