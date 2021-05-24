import cv2
import numpy as np
import matplotlib.pyplot as plt
from ffpyplayer.player import MediaPlayer
# 1. Load files and pictures 2. Perform grayscale processing 3. Get haar features 4. Detect faces 5. Mark


def adaboost_method_png():
    face_xml = cv2.CascadeClassifier('adaboost/haarcascase_frontalface_default.xml')
    # eye_xml = cv2.CascadeClassifier('adaboost/haarcascase_eye.xml')
    img = cv2.imread('img1.jpg')
    cv2.imshow('img', img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 1. Gray image 2. Zoom factor 3. Target size
    faces = face_xml.detectMultiScale(gray, 1.3, 5)
    print('face = ', len(faces))
    print(faces)
    # Draw a face, draw a box for the face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_face = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        # eyes = eye_xml.detectMultiScale(roi_face)
        # print('eyes = ', len(eyes))
        # for (ex, ey, ew, eh) in eyes:
        #     cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    cv2.imshow('dat', img)
    cv2.waitKey(0)


def adaboost_method_video():
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    video_capture = cv2.VideoCapture("videoplayback.mp4")

    while True:

        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            roi_gray_mouth = gray[y + (int(h / 2)):y + h, x:x + w]
            roi_color_mouth = frame[y + (int(h / 2)):y + h, x:x + w]

            roi_gray_eye = gray[y - (int(h / 2)):y + h, x:x + w]
            roi_color_eye = frame[y - (int(h / 2)):y + h, x:x + w]

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # adaboost_method_png()
    adaboost_method_video()
