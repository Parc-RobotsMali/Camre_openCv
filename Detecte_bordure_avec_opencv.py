#!/usr/bin/env python
#coding=utf-8
import rospy 
import time
import cv2
import matplotlib.pylab as plt
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError
#On pourrait changer les valeurs de ces variables pour avoir de resultat plus nette 
ymin=310
ymax=341
xmin1=90
xmax1=150
xmin2=410
xmax2=500
def Point(capteur):
    S1=len(capteur)-1
    S2=len(capteur)-1
    for i in range(len(capteur)):
        if capteur[i]!=0:
            S1=i
            break
    if S1!=len(capteur)-1:
        for i in range(len(capteur)-1, S1-1, -1):
            if capteur[i]!=0:
                S2=i
                break
        return int((S1+S2)/2)
    return -1
cv_b = CvBridge()
mn = np.array([0,0,120])
ma = np.array([0,0,130])
def Fonc(msg) :
    #th1,th2 sont les seuils on pourrait les changer aussi pour avoir de resulta plus nette.
    k=1
    th1 ,th2 = 75,117
    S1_old=0
    S2_old=0
    S1=0
    S2=0
    S1_time=0
    S2_time=0
    stop=0
    try :
        cv_image= cv_b.imgmsg_to_cv2(msg,"bgr8")
        image = cv_image.copy()
        #print(image.shape)
    except CvBridgeError as e :
        print(e)
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,mn,ma)
    cv2.imshow('mask',mask)
    cv2.imshow('hsv',hsv)
    gray1=cv2.cvtColor(image[ymin:ymax, xmin1:xmax1], cv2.COLOR_BGR2GRAY)
    if k!=1:
        gray1=cv2.blur(gray1, (k, k))
    capteur1=cv2.Canny(gray1, th1, th2)
    gray2=cv2.cvtColor(image[ymin:ymax, xmin2:xmax2], cv2.COLOR_BGR2GRAY)
    if k!=1:
        gray2=cv2.blur(gray2, (k, k))
    capteur2=cv2.Canny(gray2, th1, th2)
    cv2.rectangle(image, (xmin1, ymin), (xmax1, ymax), (0, 0, 255), 1)
    cv2.rectangle(image, (xmin2, ymin), (xmax2, ymax), (0, 0, 255), 1)
    #cv2.line(image, (xmin2, ymin), (xmax1, ymax), (0, 0, 255), 1)
    #cv2.line(image, (xmin2, ymin), (xmax2, ymax), (0, 0, 255), 1)
    S1=Point(capteur1[0])
    if S1!=-1:

        cv2.circle(image, (S1+xmin1, ymin), 3, (0, 255, 0), 3)

        S1_old=S1
        S1_time=time.time()
    else:
        if time.time()-S1_time<1:
            cv2.circle(image, (S1_old+xmin1, ymin), 3, (100, 255, 255), 3)
    
            S1=S1_old
        else:
            S1=-1
    S2=Point(capteur2[0])
    if S2!=-1:
        cv2.circle(image, (S2+xmin2, ymin), 3, (0, 255, 0), 3)

        S2_old=S2
        S2_time=time.time()
    else:
        if time.time()-S2_time<1:
            cv2.circle(image, (S2_old+xmin2, ymin), 3, (100, 255, 255), 3)
            S2=S2_old
        else:
            S2=-1
    if S1!=-1 and S2!=-1:
        S2_=abs(xmax2-xmin2-S2)
        if abs(S2_-S1)>20:
            c=(0, max(0, 255-10*int(abs(S1-S2_)/2)), min(255, 10*int(abs(S1-S2_)/2)))
            cv2.circle(image, (int((xmax2-xmin1)/2)+xmin1, ymax-25), 5, c, 7)
            cv2.arrowedLine(image, (int((xmax2-xmin1)/2)+xmin1, ymax-25), (int((xmax2-xmin1)/2)+xmin1+2*int((S1-S2_)/2), ymax-25), c, 3, tipLength=0.4)
        else:
            cv2.putText(image, "OK", (int((xmax2-xmin1)/2)+xmin1-15, ymax-16), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
    cv2.imshow("image", image)
    gray=cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    if k!=1:
        gray=cv2.blur(gray, (k, k))
    cv2.imshow("blur", gray)
    gray_canny=cv2.Canny(gray, th1, th2)
    cv2.imshow("canny", gray_canny)
    cv2.waitKey(1)
if __name__=="__main__" :
    rospy.init_node("image_process")
    rospy.Subscriber("camera/color/image_raw",Image,Fonc)
    rospy.spin()