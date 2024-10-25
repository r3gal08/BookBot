import cv2

# Open the video capture with the USB camera (assuming it's device 0)
cap = cv2.VideoCapture(0)  # Change 0 to 1 or 2 if needed for different camera device IDs

if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    print("Camera is working! Press 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Display the resulting frame
        cv2.imshow('USB Camera Test', frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
