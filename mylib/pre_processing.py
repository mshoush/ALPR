import copy
import cv2
import argparse
import os
from time import gmtime, strftime
import math
import numpy as np


def preprocess(plateFrame, se_shape, Show = False):
    # 1
    plate_origin = copy.copy(plateFrame)

    # apply some filters on the frame (Gaussian, Soble)
    plate_gray = enhance(cv2.cvtColor(plateFrame, cv2.COLOR_BGR2GRAY))
    plate_gaussian = cv2.GaussianBlur(plate_gray, (5,5), 0)
    plate_sobel = cv2.Sobel(plate_gaussian, -1, 1, 0)
    h, plate_threshold = cv2.threshold(plate_sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    se = cv2.getStructuringElement(cv2.MORPH_RECT, se_shape)
    # 2

    # Opening is just another name of erosion followed by dilation. It is useful in removing noise,
    # as we explained above. Here we use the function, cv2.morphologyEx()
    plate_morphEx = cv2.morphologyEx(plate_threshold, cv2.MORPH_CLOSE, se)
    # 3
    edge = np.copy(plate_threshold)

    if Show == True:
        cv2.imshow("1. Original frame", plate_origin)
        cv2.imshow("2. Gray frame", plate_gray)
        cv2.imshow("3. GaussianBlur frame", plate_gaussian)
        cv2.imshow("4. Sobel frame", plate_sobel)
        cv2.imshow("5. Threshold frame", plate_threshold)
        cv2.imshow("6. morphologyEx frame", plate_morphEx)
        pass
    return plate_origin, plate_morphEx, edge

def im_reshape(Image, Size, Spec = "imgBuffer.jpg"):
    old_size = Image.shape[:2]
    ratio = float(Size) / max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # Resize image
    Image = cv2.resize(Image, (new_size[1], new_size[0]))

    delta_w = Size - new_size[1]
    delta_h = Size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    # set border to image
    Image = cv2.copyMakeBorder(Image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    # Write image Spec = "imBuffer.jpg"
    cv2.imwrite(str(Spec), Image)
    # Read grey image
    Image = cv2.imread(str(Spec), 0)
    return Image


# Charcter segmentation 26 capital letters, 10 numbers, None
def code_to_char(model_out):
    # Letters (A-Z)
    if np.argmax(model_out) == 0:   return "A"
    elif np.argmax(model_out) == 1:   return "B"
    elif np.argmax(model_out) == 2:   return "C"
    elif np.argmax(model_out) == 3:   return "D"
    elif np.argmax(model_out) == 4:   return "E"
    elif np.argmax(model_out) == 5:   return "F"
    elif np.argmax(model_out) == 6:   return "G"
    elif np.argmax(model_out) == 7:   return "H"
    elif np.argmax(model_out) == 8:   return "I"
    elif np.argmax(model_out) == 9:   return "G"
    elif np.argmax(model_out) == 10:   return "K"
    elif np.argmax(model_out) == 11:   return "L"
    elif np.argmax(model_out) == 12:   return "M"
    elif np.argmax(model_out) == 13:   return "N"
    elif np.argmax(model_out) == 14:   return "O"
    elif np.argmax(model_out) == 15:   return "P"
    elif np.argmax(model_out) == 16:   return "Q"
    elif np.argmax(model_out) == 17:   return "R"
    elif np.argmax(model_out) == 18:   return "S"
    elif np.argmax(model_out) == 19:   return "T"
    elif np.argmax(model_out) == 20:   return "U"
    elif np.argmax(model_out) == 21:   return "V"
    elif np.argmax(model_out) == 22:   return "W"
    elif np.argmax(model_out) == 23:   return "X"
    elif np.argmax(model_out) == 24:   return "Y"
    elif np.argmax(model_out) == 25:   return "Z"

    # Numbers (0 - 9)
    elif np.argmax(model_out) == 26:   return "1"
    elif np.argmax(model_out) == 27:   return "2"
    elif np.argmax(model_out) == 28:   return "3"
    elif np.argmax(model_out) == 29:   return "4"
    elif np.argmax(model_out) == 30:   return "5"
    elif np.argmax(model_out) == 31:   return "6"
    elif np.argmax(model_out) == 32:   return "7"
    elif np.argmax(model_out) == 33:   return "8"
    elif np.argmax(model_out) == 34:   return "9"
    elif np.argmax(model_out) == 35:   return "0"


    elif np.argmax(model_out) == 36:   return "None"
    pass

def contour_vladition(contour, img, aspect_ratio_range, area_range):
    rect = cv2.minAreaRect(contour)
    img_width, img_height = img.shape[1], img.shape[0]
    box = cv2.boxPoints(rect);      box = np.int0(box)

    width, height = rect[1][0], rect[1][1]
    X, Y = rect[0][0], rect[0][1]
    angle = rect[2]

    angle = (angle + 180) if width < height else (angle + 90)
    output=False

    if (width > 0 and height > 0) and ((width < img_width/2.0) and (height < img_width/2.0)):
        aspect_ratio = float(width)/height if width > height else float(height)/width
        if (aspect_ratio >= aspect_ratio_range[0] and aspect_ratio <= aspect_ratio_range[1]):

            if((height*width > area_range[0]) and (height*width < area_range[1])):

                box_copy = list(box)
                point = box_copy[0]
                del(box_copy[0])
                dists = [((p[0]-point[0])**2 + (p[1]-point[1])**2) for p in box_copy]
                sorted_dists = sorted(dists)
                opposite_point = box_copy[dists.index(sorted_dists[1])]
                tmp_angle = 90

                if abs(point[0]-opposite_point[0]) > 0:
                    tmp_angle = abs(float(point[1]-opposite_point[1]))/abs(point[0]-opposite_point[0])
                    tmp_angle = rad_to_deg(math.atan(tmp_angle))

                    if tmp_angle <= 45:
                           output = True
    return output

def deg_to_rad(angle):  return angle*np.pi/180.0
def rad_to_deg(angle):  return angle*180/np.pi
def enhance(img):
	kernel = np.array([[-1,0,1],[-2,0,2],[1,0,1]])
	return cv2.filter2D(img, -1, kernel)
