import cv2
import numpy as np

from os import path

cap = cv2.VideoCapture(0)

classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

name = input("Enter your name : ")

counter = 30

face_list = []

while True:
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = classifier.detectMultiScale(gray)

        areas = []
        for face in faces:
            x, y, w, h = face
            areas.append((w*h, face))

        if len(faces) > 0:
            face = max(areas)[1]
            x, y, w, h = face

            face_img = gray[y:y+h, x:x+w]

            face_img = cv2.resize(face_img, (100, 100))
            face_flatten = face_img.flatten()
            face_list.append(face_flatten)
            counter -= 1
            print("loaded with", 30 - counter)
            if counter <= 0:
                break

            cv2.imshow("video", face_img)

    key = cv2.waitKey(1)
    if key & 0xff == ord('q'):
        break


X = np.array(face_list)
y = np.full((len(X), 1), name)

data = np.hstack([y, X])

print(data.shape)
print(data.dtype)

cap.release()
cv2.destroyAllWindows()

if path.exists("face_data.npy"):
    face_data = np.load("face_data.npy")
    face_data = np.vstack([face_data, data])
    np.save("face_data.npy", face_data)
else:
    np.save("face_data.npy", data)

