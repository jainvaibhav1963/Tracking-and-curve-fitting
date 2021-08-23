# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 03:31:48 2021

@author: jain
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2


#extracting frames from video

vidcap = cv2.VideoCapture('video.mp4')
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite("image"+str(count)+".png", image)     # save frame as JPG file
    return hasFrames
sec = 0
frameRate = 0.5 #//it will capture image in each 0.5 second
count=1
success = getFrame(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)
    
###############################################################################
img = cv2.imread("image1.png")

# Convert image to grayscale
    # OpenCV reads the image as BGR instead of the intuitive RGB color scheme
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Display gray image
#cv2.imshow('gray_image', gray)
#cv2.waitKey(0)  # waits for ESC key 

threshold_level = 250

coords = np.column_stack(np.where(gray < threshold_level))

print(coords)

maxy = np.amax(coords, axis=0)
miny = np.amin(coords, axis=0)

maxy[1]=(maxy[1]+miny[1])/2
miny[1]=maxy[1]



###get coordinates of all the images
img = cv2.imread("image2.png")

# Convert image to grayscale
    # OpenCV reads the image as BGR instead of the intuitive RGB color scheme
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Display gray image
#cv2.imshow('gray_image', gray)
#cv2.waitKey(0)  # waits for ESC key 

coords = np.column_stack(np.where(gray < threshold_level))

maxy1 = np.amax(coords, axis=0)
miny1= np.amin(coords, axis=0)

maxy1[1]=(maxy1[1]+miny1[1])/2
miny1[1]=maxy1[1]

img = cv2.imread("image3.png")

# Convert image to grayscale
    # OpenCV reads the image as BGR instead of the intuitive RGB color scheme
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Display gray image
#cv2.imshow('gray_image', gray)
#cv2.waitKey(0)  # waits for ESC key 

coords = np.column_stack(np.where(gray < threshold_level))

maxy2 = np.amax(coords, axis=0)
miny2= np.amin(coords, axis=0)

maxy2[1]=(maxy2[1]+miny2[1])/2
miny2[1]=maxy2[1]

img = cv2.imread("image4.png")

# Convert image to grayscale
    # OpenCV reads the image as BGR instead of the intuitive RGB color scheme
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Display gray image
#cv2.imshow('gray_image', gray)
#cv2.waitKey(0)  # waits for ESC key 

coords = np.column_stack(np.where(gray < threshold_level))

maxy3 = np.amax(coords, axis=0)
miny3= np.amin(coords, axis=0)

maxy3[1]=(maxy3[1]+miny3[1])/2
miny3[1]=maxy3[1]

img = cv2.imread("image5.png")

# Convert image to grayscale
    # OpenCV reads the image as BGR instead of the intuitive RGB color scheme
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Display gray image
#cv2.imshow('gray_image', gray)
#cv2.waitKey(0)  # waits for ESC key 

coords = np.column_stack(np.where(gray < threshold_level))

maxy4 = np.amax(coords, axis=0)
miny4= np.amin(coords, axis=0)

maxy4[1]=(maxy4[1]+miny4[1])/2
miny4[1]=maxy4[1]

img = cv2.imread("image6.png")

# Convert image to grayscale
    # OpenCV reads the image as BGR instead of the intuitive RGB color scheme
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Display gray image
#cv2.imshow('gray_image', gray)
#cv2.waitKey(0)  # waits for ESC key 

coords = np.column_stack(np.where(gray < threshold_level))

maxy5 = np.amax(coords, axis=0)
miny5= np.amin(coords, axis=0)

maxy5[1]=(maxy5[1]+miny5[1])/2
miny5[1]=maxy5[1]

img = cv2.imread("image7.png")

# Convert image to grayscale
    # OpenCV reads the image as BGR instead of the intuitive RGB color scheme
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Display gray image
#cv2.imshow('gray_image', gray)
#cv2.waitKey(0)  # waits for ESC key 

coords = np.column_stack(np.where(gray < threshold_level))

maxy6 = np.amax(coords, axis=0)
miny6= np.amin(coords, axis=0)

maxy6[1]=(maxy6[1]+miny6[1])/2
miny6[1]=maxy6[1]

max_points = [maxy,maxy1,maxy2,maxy3,maxy4,maxy5,maxy6]
min_points = [miny,miny1,miny2,miny3,miny4,miny5,miny6]

#obtaining the centre points for the ball like object in each frame.
avg_points = (np.array(max_points) + np.array(min_points)) / 2.0
###############################################################################3
###############################################################################3

## Least square fitting
X = np.array([[(avg_points[0][1])**2, avg_points[0][1], 1],
              [(avg_points[1][1])**2, avg_points[1][1], 1],
              [(avg_points[2][1])**2, avg_points[2][1], 1],
              [(avg_points[3][1])**2, avg_points[3][1], 1],
              [(avg_points[4][1])**2, avg_points[4][1], 1],
              [(avg_points[5][1])**2, avg_points[5][1], 1],
              [(avg_points[6][1])**2, avg_points[6][1], 1]])

Y = np.array(avg_points[:,0])
Y = Y.transpose()

B = np.linalg.inv(X.transpose()@X)@(X.transpose()@Y)

## Equation is of the form y = ax^2 + bx + c,  B = [a, b, c]


# create 1000 equally spaced points between -10 and 10
x = np.linspace(0, 2400, 1000)

# calculate the y value for each element of the x vector
y = (x**2)*B[0] + B[1]*x + B[2]  

fig, ax = plt.subplots()
ax.plot(x, (1676-y))