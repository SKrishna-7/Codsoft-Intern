import cv2

capture=cv2.VideoCapture(0)
Classifier=cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
while True:
    IsTrue,frames=capture.read()
    GrayImg=cv2.cvtColor(frames,cv2.COLOR_BGR2GRAY)
    faces=Classifier.detectMultiScale(GrayImg)
    # cv2.putText(frames,str(faces),(100,100),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)
    for x,y,w,h in faces:
        cv2.rectangle(frames,(x,y),(x+w,y+h),(0,255,255),1)
    cv2.imshow('video',frames)
    if cv2.waitKey(20) & 0xFF==ord('d'):
        break
capture.release()
cv2.destroyAllWindows()