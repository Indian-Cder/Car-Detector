import cv2
import numpy as np

cap = cv2.VideoCapture("DRONE-SURVEILLANCE-CONTEST-VIDEO.mp4")

min_width_rec = 80
min_height_rec = 80
line_pos = 500

algo = cv2.bgsegm.createBackgroundSubtractorMOG()

def center_handle(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    return cx, cy

detect = []
offset = 6
counter = 0

while True:
    ret, frame1 = cap.read()
    gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(3,3), 5)

    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatada = cv2.morphologyEx(dilat,cv2.MORPH_CLOSE,kernel)
    dilatada = cv2.morphologyEx(dilatada,cv2.MORPH_CLOSE,kernel)
    contour, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for (i,c) in enumerate(contour):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_counter = (w>= min_width_rec) and (h>=min_height_rec)
        if not validate_counter:
            continue
        cv2.rectangle(frame1,(x,y),(x+w, y+h),(0,0,255),2)

        center = center_handle(x,y, w, h)
        detect.append(center)

        for (x,y) in detect:
            if y< (line_pos+offset) and y> (line_pos-offset):
                counter += 1
            detect.remove((x,y))


    cv2.putText(frame1, str(counter),(1400,140),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)

    cv2.putText(frame1, "Indian Coder",(875,90),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)

    cv2.imshow("Car detection", frame1)

    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()
cap.release()
