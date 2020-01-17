import cv2
import numpy as np
import imutils
import dobot
import serial
from serial.tools import list_ports
co=0
port = list_ports.comports()[0].device
d1 = dobot.Dobot(port = port, verbose = True)

# d1.home()
# d1.delay(29)
d1.motor(20)
d1.movel(205, 0, 95, 112)
d1.delay(2)

cap = cv2.VideoCapture(0)
def blue():
    x1 = 191
    y1 = -104
    d1.movel(205,0,37,112)
    d1.delay(2)
    _, cam = cap.read()
    hsv = cv2.cvtColor(cam, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([88,70,50])
    upper_blue = np.array([130,255,255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    cv2.imshow("blue_frame",cam)
    
    cnts = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
   
    for c in cnts:
        area = cv2.contourArea(c)
        
        if area > 3000:
            cv2.drawContours(cam, [c], -1, (0,255,0), 2)
            
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(cam, (cX,cY), 7, (255,255,255), -1)
            cv2.putText(cam, "center", (cX-20, cY-20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1)
            cv2.imshow("blue_frame",cam)
            dX = x1 + ((480-cY)*0.355)
            dY = y1 + ((640-cX)*0.365)
            d1.movel(dX,dY,-122,112)
            d1.delay(1)
            d1.suck(True)
            d1.delay(0.5)
            d1.movel(205,0,95,112)
            d1.delay(1)
            d1.movebelt(0)
            d1.delay(3)
            break
            
            
       
            
            
def dobot(i):
    x1=191
    y1=-104
    
    d1.movebelt(500)
    d1.delay(4)
    d1.movel(205,0,37,112)
    d1.delay(1)
    blue()
    cX=i[0]
    cY=i[1]
    dX = x1 + ((480-cY)*0.365)
    dY = y1 + ((640-cX)*0.36)
    d1.movel(dX,dY,-25,112)
    d1.delay(1)
    d1.suck(False)
    d1.delay(0.5)
    d1.movel(205,0,95,112)
    d1.delay(1)
    
    

def crop(p,frame):
    for i in p:
        x = int((i[0]*2-50)/2)
        y = int((i[1]*2-50)/2)
        xr = range(i[0]-20,i[0]+20)
        yr = range(i[1]-20,i[1]+20)
        
        crop = frame[y:y+50,x:x+50]

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([88,50,80])#88,70,50
        upper_blue = np.array([130,255,255])
        
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        if mask.any():
            print("success")
        else:
            print("fail")
            dobot(i)


while(True):
    
    _, frame = cap.read()
    cap.set(3,640)
    cap.set(4,480)
    cap.set(15, 10.0)
    cap.set(10, -10.0)
    cap.set(11, 25.0)
    cap.set(13, 13.0)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    cv2.imshow("frame",frame)
    
    _, threshold = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)  #(gray, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("thresh",threshold)
    cnts = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        area = cv2.contourArea(c)

        if area > 50000:
            d1.delay(2)
            d1.motor(0)
                                             
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
            # white color is detected in lab lighting condition under lights L1 & L4
            
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            R2C2 = [cX,cY]
            R1C1 = [cX-90,cY+90]
            R1C2 = [cX,cY+90]
            R1C3 = [cX+90,cY+90]
            R2C1 = [cX-90,cY]
            R2C3 = [cX+90,cY]
            R3C1 = [cX-90,cY-90]
            R3C2 = [cX,cY-90]
            R3C3 = [cX+90,cY-90]
            cv2.circle(frame, (R2C2[0],R2C2[1]), 7, (0,0,0), -1)
            #cv2.putText(frame, "R2C2", (cX-20, cY-20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1)
            cv2.circle(frame, (R1C1[0],R1C1[1]), 7, (0,0,0), -1)
             #cv2.putText(frame, "R1C1", (cX-116, cY+76), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1)
            cv2.circle(frame, (R1C2[0],R1C2[1]), 7, (0,0,0), -1)
             #cv2.putText(frame, "R1C2", (cX-20, cY+76), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1)
            cv2.circle(frame, (R1C3[0],R1C3[1]), 7, (0,0,0), -1)
             #cv2.putText(frame, "R1C3", (cX+76, cY+76), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1)
            cv2.circle(frame, (R2C1[0],R2C1[1]), 7, (0,0,0), -1)
             #cv2.putText(frame, "R2C1", (cX-116, cY-20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1)
            cv2.circle(frame, (R2C3[0],R2C3[1]), 7, (0,0,0), -1)
             #cv2.putText(frame, "R2C3", (cX+76, cY-20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1)
            cv2.circle(frame, (R3C1[0],R3C1[1]), 7, (0,0,0), -1)
            # #cv2.putText(frame, "R3C1", (cX-116, cY-116), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1)
            cv2.circle(frame, (R3C2[0],R3C2[1]), 7, (0,0,0), -1)
            # #cv2.putText(frame, "R3C2", (cX-20, cY-116), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1)
            cv2.circle(frame, (R3C3[0],R3C3[1]), 7, (0,0,0), -1)
            # #cv2.putText(frame, "R3C3", (cX+76, cY-116), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1)
            cv2.imshow("frame",frame)
           
            p = [R1C1,R1C2,R1C3,R2C1,R2C2,R2C3,R3C1,R3C2,R3C3] 
            
            if co >=3:
                crop(p,frame)
                co=0
            co=co+1
           
    k = cv2.waitKey(1)
    if k == 27:
        break

d1.motor(0)
d1.close()
cv2.destroyAllWindows()        