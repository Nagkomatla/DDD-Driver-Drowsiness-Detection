# Import required libraries
import cv2

# Open the default webcam (0)
video = cv2.VideoCapture(0)

# Continuously capture frames from webcam
while True:
    ret, frame = video.read()  # Read a new frame
    if not ret:
        break  # Exit if frame is not read properly

    # Display the captured frame in a window
    cv2.imshow("Real-Time Video Capture", frame)

    # Press 'q' key to exit the video window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
video.release()
cv2.destroyAllWindows()
