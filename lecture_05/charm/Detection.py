import cv2

cap = cv2.VideoCapture(0)

classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()

    # print(type(frame))

    if ret:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = classifier.detectMultiScale(gray)

        areas = []

        for face in faces:
            x, y, w, h = face
            area = w*h
            areas.append((area, face))

        areas = sorted(areas, reverse=True)

        if len(areas) > 0:
            face = areas[0][1]
            x, y, w, h = face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

            img_face = frame[y:y+h, x:x+w]

            img_face = cv2.resize(img_face, (400, 400))

            shape = img_face.shape

            frame[0:shape[0], 0:shape[1]] = img_face

        cv2.imshow("cam", frame)

    key = cv2.waitKey(1)
    if key & 0xff == ord('q'):
        break
    elif key & 0xff == ord('c'):
        cv2.imwrite("classroom.jpg", frame)


cap.release()
cv2.destroyAllWindows()


