import cv2

cap = cv2.VideoCapture(0)

classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()

    # print(type(frame))

    if ret:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = classifier.detectMultiScale(gray)

        for face in faces:
            x, y, w, h = face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

        cv2.imshow("cam", frame)

    key = cv2.waitKey(1)
    if key & 0xff == ord('q'):
        break
    elif key & 0xff == ord('c'):
        cv2.imwrite("classroom.jpg", frame)


cap.release()
cv2.destroyAllWindows()


