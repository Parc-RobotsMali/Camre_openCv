#!/usr/bin/env python
#coding=utf-8
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError
import cv2
import numpy as np
import matplotlib.pylab as plt

img = cv2.imread("ros_image.png")

#Convertir image de BGR en RGB
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#2.Filtrage des couleurs
#L'un des espaces calorimétriques qui facilite vraiment le filtage des couleurs
#est l' espace calorimétrique HLS(HUE-LIGHTNESS-SATURATION)
#Pour obtenir les lignes blanches nous allons supprimer tous les pixels
#dont la valeur de luminosité est inférieure à 190

#Création de la fonction pour filter les lignes de voie
def color_filter(image) :
    hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    lower = np.array([0,120,0])
    upper = np.array([255,255,255])
    yellower = np.array([10,0,90])
    yelupper = np.array([50,255,255])
    whitemask = cv2.inRange(hls,lower,upper)
    yellowmask = cv2.inRange(hls,yellower,yelupper)
    mask = cv2.bitwise_or(whitemask,yellowmask)
    masked = cv2.bitwise_and(image,image,mask=mask)
    return masked
filtered_img = color_filter(img)
plt.imshow(filtered_img)
plt.show()
#3.Region d'intérêt
def roi(im) :
    x = int(im.shape[1])
    y = int(im.shape[0])
    shape = np.array([[int(0.003*x),int(0.7651*y)],[int(0.995*x),int(0.735*y)],[int(0.552*x),int(0.514*y)],[int(0.445*x),
    int(0.52*y)]])
    mask = np.zeros_like(im)
    if len(im.shape)>2 :
        channel1_count = im.shape[2]
        ignore_mask_color = (255,)*channel1_count
    else :
        ignore_mask_color = 255
    cv2.fillPoly(mask,np.int32([shape]),ignore_mask_color)
    masked_image = cv2.bitwise_and(filtered_img,mask)
    masked_image_whith_original_image=cv2.bitwise_and(img,mask)
    #remplacer masked_image par  masked_image_whith_original_image pour voir la region d'intérêt avec l'image originale
    return masked_image #masked_image_whith_original_image
roi_img = roi(img)
plt.imshow(roi_img)
plt.show()