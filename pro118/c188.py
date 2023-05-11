import cv2

image=cv2.VideoCapture("walking.avi")
fullbodyCascade=cv2.CascadeClassifier("haarcascade_fullbody.xml")


while True:
    dummy,frame=image.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    body=fullbodyCascade.detectMultiScale(gray,1.1,5)
    for(x,y,w,h) in body:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(3,40,252),1)
    cv2.imshow("window",frame)
    if cv2.waitKey(25)==32:
        break
               

image.release()
cv2.destroyAllWindows()
