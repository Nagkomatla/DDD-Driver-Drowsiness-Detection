import mediapipe as mp #facial landmarks 

import cv2
import numpy as np

import threading
import pygame
import time

# Initialize pygame mixer
pygame.mixer.init()
sound = pygame.mixer.Sound("kavya.wav")  # Make sure this file exists!

def play_sound():
    if not pygame.mixer.get_busy():
        sound.play(-1)  # Loop the sound
def stop_sound():
    if pygame.mixer.get_busy():
        sound.stop()

def draw_progress_bar(frame, ear_value):
    bar_width = 250
    bar_height = 20
    bar_x = 10
    bar_y = 120
    min_ear = 0.23
    max_ear = 0.30
    percentage = (ear_value - min_ear) / (max_ear - min_ear)
    percentage = max(0, min(1, percentage))
    
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
    filled_width = int(bar_width * percentage)
    bar_color = (0, 0, 255) if percentage < 0.4 else (0, 255, 0)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), bar_color, -1)

    status_text = f"Eye Opening: {int(percentage * 100)}% (EAR: {ear_value:.2f})"
    cv2.putText(frame, status_text, (bar_x, bar_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

def calculate_ear(eye_points):
    v1 = np.linalg.norm(eye_points[1] - eye_points[5])
    v2 = np.linalg.norm(eye_points[2] - eye_points[4])
    h = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (v1 + v2) / (2.0 * h)
    return ear

# Setup Mediapipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    EAR_THRESHOLD = 0.28
    EAR_FRAMES = 40
    
    eye_closed = False
    start_time = None
    counter = 0

    video = cv2.VideoCapture(0)
    while True:
        ret, frame = video.read()
        if not ret:
            break

        H, W, _ = frame.shape
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb_img)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                left_eye = np.array([
                    [int(face_landmarks.landmark[33].x * W), int(face_landmarks.landmark[33].y * H)],
                    [int(face_landmarks.landmark[160].x * W), int(face_landmarks.landmark[160].y * H)],
                    [int(face_landmarks.landmark[158].x * W), int(face_landmarks.landmark[158].y * H)],
                    [int(face_landmarks.landmark[133].x * W), int(face_landmarks.landmark[133].y * H)],
                    [int(face_landmarks.landmark[153].x * W), int(face_landmarks.landmark[153].y * H)],
                    [int(face_landmarks.landmark[144].x * W), int(face_landmarks.landmark[144].y * H)]
                ])

                right_eye = np.array([
                    [int(face_landmarks.landmark[362].x * W), int(face_landmarks.landmark[362].y * H)],
                    [int(face_landmarks.landmark[387].x * W), int(face_landmarks.landmark[387].y * H)],
                    [int(face_landmarks.landmark[386].x * W), int(face_landmarks.landmark[386].y * H)],
                    [int(face_landmarks.landmark[263].x * W), int(face_landmarks.landmark[263].y * H)],
                    [int(face_landmarks.landmark[373].x * W), int(face_landmarks.landmark[373].y * H)],
                    [int(face_landmarks.landmark[380].x * W), int(face_landmarks.landmark[380].y * H)]
                ])

                left_ear = calculate_ear(left_eye)
                right_ear = calculate_ear(right_eye)
                ear = (left_ear + right_ear) / 2.0

                draw_progress_bar(frame, ear)

                if ear < EAR_THRESHOLD:
                    cv2.putText(frame, "Eyes Closed", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if start_time is None:
                        start_time = time.time()
                    counter += 1
                    if counter >= EAR_FRAMES:
                        elapsed = time.time() - start_time
                        cv2.putText(frame, f"Eyes Closed for {elapsed:.1f} sec", (10, 60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        threading.Thread(target=play_sound).start()
                else:
                    if eye_closed:
                        blink_count += 1
                    eye_closed = False
                    counter = 0
                    start_time = None
                    stop_sound()
                    cv2.putText(frame, "Eyes Open", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (139, 0, 0), 2)
                
                # Draw eye landmarks
                for point in left_eye:
                    cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)
                for point in right_eye:
                    cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)
                for i in range(3):
                    cv2.line(frame, tuple(left_eye[i]), tuple(left_eye[i+3]), (0, 255, 0), 1)
                    cv2.line(frame, tuple(right_eye[i]), tuple(right_eye[i+3]), (0, 255, 0), 1)

        cv2.imshow("Eye State Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows(0)