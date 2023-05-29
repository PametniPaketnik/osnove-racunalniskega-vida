import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Read the input image
#img = cv2.imread('test.png')
cap = cv2.VideoCapture('IMG_8157.mov')

while cap.isOpened():
    _, img = cap.read() # Prebere naslednji okvir iz videja in ga shrani v img (_ shrani true/false če uspesno prebrano)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4) # Uporabi detektor obraza face_cascade
    # delectMultiScale vrne seznam pravokotnikov ki predstavljajo najdene obraze

    for (x, y , w ,h) in faces: #Skozi seznam najdenih obrazov
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0 , 0), 3) # Na vsak obraz izrise pravokotnik
        # x y zgornji levi kot pravokotnika w in h pa širino in višino

    # Display the output
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

