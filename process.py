import cv2 as cv
import colorsys
import numpy as np
import matplotlib.pyplot as plt

def sum_area(x, y, im, n):
    
    colorsum = np.array([0, 0, 0])
    count = 0
    for i in range(x+5, x + 35):
        for j in range(y+5, y + 18):
            colorsum[0] = colorsum[0] + im[j, i, 0] # b
            colorsum[1] = colorsum[1] + im[j, i, 1] # g
            colorsum[2] = colorsum[2] + im[j, i, 2] # r
            # img[j,i] = np.array([255,255,0])
            count += 1

    colorsum[0] = int(colorsum[0] / count)
    colorsum[1] = int(colorsum[1] / count)
    colorsum[2] = int(colorsum[2] / count)
    imageFrame = cv.rectangle(img, (x, y), (x + 38, y + 23), (0, 0, 255), 1)
    cv.putText(img, ("{}".format(n)), (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
    return colorsum[0], colorsum[1], colorsum[2]

def sum_area_for_click_color_matching(x, y, im):
    colorsum = np.array([0, 0, 0])
    count = 0
    for i in range(x-15, x + 16):
        for j in range(y - 8, y + 9):
            colorsum[0] = colorsum[0] + im[j, i, 0] # b
            colorsum[1] = colorsum[1] + im[j, i, 1] # g
            colorsum[2] = colorsum[2] + im[j, i, 2] # r
            # img[j,i] = np.array([255,255,0])
            count += 1

    colorsum[0] = int(colorsum[0] / count)
    colorsum[1] = int(colorsum[1] / count)
    colorsum[2] = int(colorsum[2] / count)
    return colorsum[0], colorsum[1], colorsum[2]

def clickless_color_detection(coords, c):
    tobemasked = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    temp = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    (h, s, v) = sum_area(coords[0], coords[1], temp,c)

    hs = 5
    ss = 50
    vs = 50
    lb = 0 if h - hs <= 0 else h - hs
    lg = 0 if s - 8 <= 0 else s - 8
    lr = 0 if v - 11 <= 0 else v - 11
    ub = 180 if h + hs >= 180 else h + hs
    ug = 255 if s + ss >= 255 else s + ss
    ur = 255 if v + vs >= 255 else v + vs

    lower = np.array([lb, lg, lr], dtype = "uint8")
    upper = np.array([ub, ug, ur], dtype = "uint8")

    mask = cv.inRange(tobemasked, lower, upper)

    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    istrue = False

    for pic, contour in enumerate(contours):
        area = cv.contourArea(contour)
        if area > 300:
            istrue = True
            x, y, w, h = cv.boundingRect(contour)
            if x > 540:
                imageFrame = cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv.putText(img, str(c), (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
                break
            
    cv.imshow('image', img)

def click_event(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        tobemasked = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        temp = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        (h, s, v) = sum_area_for_click_color_matching(x, y, temp)

        hs = 5
        ss = 25
        vs = 26
        lb = 0 if h - hs <= 0 else h - hs
        lg = 0 if s - 8 <= 0 else s - 8
        lr = 0 if v - 11 <= 0 else v - 11
        ub = 180 if h + hs >= 180 else h + hs
        ug = 255 if s + ss >= 255 else s + ss
        ur = 255 if v + vs >= 255 else v + vs

        lower = np.array([lb, lg, lr], dtype = "uint8")

        upper = np.array([ub, ug, ur], dtype = "uint8")

        mask = cv.inRange(tobemasked, lower, upper)

        contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv.contourArea(contour)
            if area > 300:
                x, y, w, h = cv.boundingRect(contour)
                if(x > 540):
                    imageFrame = cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    # cv.putText(img2, ("{}".format(cnum)), (x, y), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255))
                    break

        cv.imshow('image', img)

img = cv.imread('images/twocolorsv1.jpeg')

img = cv.resize(img, (1000,750))

arr = [
    (565-498, 159),(744-498, 159),(860-498, 274),
    (565-498, 198),(744-498, 274),(860-498, 311),
    (564-498, 235),(744-498, 311),(860-498, 350),
    (564-498, 274),(742-498, 350),(859-498, 387),
    (563-498, 311),(742-498, 387),(858-497, 424),
    (563-498, 350),(742-498, 425),(858-498, 463),
    (563-500, 387),(742-498, 501),(858-497, 502),
    (563-499, 426),(802-498, 159),(918-498, 159),
    (563-499, 463),(802-498, 198),(918-498, 198),
    (563-499, 501),(802-498, 235),(919-498, 235),
    (624-498, 311),(802-498, 274),(919-498, 274),
    (624-498, 349),(802-498, 311),(919-498, 311),
    (624-499, 387),(802-498, 350),(919-499, 350),
    (684-498, 235),(801-498, 387),(918-498, 387),
    (684-498, 274),(800-498, 425),(918-498, 425),
    (684-498, 311),(800-498, 462),(918-498, 462),
    (683-498, 387),(800-498, 501),(918-498, 501),
    (683-498, 425),(860-498, 159),
    (683-498, 501),(860-498, 235)]

c = 0

for x in arr:
    c += 1
    clickless_color_detection(x, c)
cv.imshow('image', img)
cv.setMouseCallback('image', click_event)

cv.waitKey(0)