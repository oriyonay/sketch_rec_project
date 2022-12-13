import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
# set the setting of the webcam
cap.set(3, 640)    # width
cap.set(4, 420)    # height
cap.set(10, 100)   # brightness

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def getLandmarks(image):
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5,refine_landmarks=True)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = face_mesh.process(image)
    # print(results.multi_face_landmarks[0])
    landmarks = results.multi_face_landmarks[0].landmark
    return landmarks, results

while True:
    # Read camera frames
    success, frame = cap.read()
    if not success:
        print('Ignoring empty camera frame.')
        continue    # use break if reading from a video file

    # flip the video vertically and change the BGR to RGB
    frame = cv2.flip(frame, 1)

    landmarks, results = getLandmarks(frame)
    print("len landmarks",len(landmarks))
    iris_landmarks  = landmarks[-10:]
    # print("length:",len(getLandmarks(frame)))
    # print(getLandmarks(frame))
    # show webcame frames


    for landmark in iris_landmarks:
        shape = frame.shape
        x = landmark.x
        y = landmark.y
        relative_x = int(x * shape[1])
        relative_y = int(y * shape[0])

        frame = cv2.circle(frame, (relative_x, relative_y),radius=1, color=(0, 0, 255), thickness=-1)

    cv2.imshow("Webcam", frame)


    # press q to exit
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

# Finally release back the camera resources
cap.release()


# import cv2
# import itertools
# import numpy as np
# from time import time
# import mediapipe as mp
# import matplotlib.pyplot as plt
#
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,min_detection_confidence=0.5)
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
#
#
#
# sample_img = cv2.imread('sample.jpg')
# plt.figure(figsize = [10, 10])
# plt.title("Sample Image");plt.axis('off');plt.imshow(sample_img[:,:,::-1]);plt.show()
#
#
# face_mesh_results = face_mesh_images.process(sample_img[:,:,::-1])
#
# LEFT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYE)))
# RIGHT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYE)))
#
# if face_mesh_results.multi_face_landmarks:
#
#     for face_no, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
#
#         print(f'FACE NUMBER: {face_no+1}')
#         print('-----------------------')
#
#         print(f'LEFT EYE LANDMARKS:n')
#
#         for LEFT_EYE_INDEX in LEFT_EYE_INDEXES[:2]:
#
#             print(face_landmarks.landmark[LEFT_EYE_INDEX])
#
#         print(f'RIGHT EYE LANDMARKS:n')
#
#         for RIGHT_EYE_INDEX in RIGHT_EYE_INDEXES[:2]:
#
#             print(face_landmarks.landmark[RIGHT_EYE_INDEX])

# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_face_mesh = mp.solutions.face_mesh
#
# cap = cv2.VideoCapture(0)
# with mp_face_mesh.FaceMesh(
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as face_mesh:
#   while cap.isOpened():
#     print("Starting")
#     success, image = cap.read()
#     if not success:
#       print("Ignoring empty camera frame.")
#       # If loading a video, use 'break' instead of 'continue'.
#       continue
#
#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(image)
#
#     # Draw the face mesh annotations on the image.
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     if results.multi_face_landmarks:
#       for face_landmarks in results.multi_face_landmarks:
#         print(face_landmarks)
#         mp_drawing.draw_landmarks(
#             image=image,
#             landmark_list=face_landmarks,
#             connections=mp_face_mesh.FACEMESH_TESSELATION,
#             landmark_drawing_spec=None,
#             connection_drawing_spec=mp_drawing_styles
#             .get_default_face_mesh_tesselation_style())
#         mp_drawing.draw_landmarks(
#             image=image,
#             landmark_list=face_landmarks,
#             connections=mp_face_mesh.FACEMESH_CONTOURS,
#             landmark_drawing_spec=None,
#             connection_drawing_spec=mp_drawing_styles
#             .get_default_face_mesh_contours_style())
#         mp_drawing.draw_landmarks(
#             image=image,
#             landmark_list=face_landmarks,
#             connections=mp_face_mesh.FACEMESH_IRISES,
#             landmark_drawing_spec=None,
#             connection_drawing_spec=mp_drawing_styles
#             .get_default_face_mesh_iris_connections_style())
#     # Flip the image horizontally for a selfie-view display.
#     cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
#     if cv2.waitKey(5) & 0xFF == 27:
#       break
# cap.release()