import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math

arr = [
    (565-498, 159), (565-498, 198), (564-498, 235), (564-498, 274),
    (563-498, 311), (563-498, 350), (563-500, 387), (563-499, 426),
    (563-499, 463), (563-499, 501), (624-498, 311), (624-498, 349),
    (624-499, 387), (684-498, 235), (684-498, 274), (684-498, 311),
    (683-498, 387), (683-498, 425), (683-498, 501), (744-498, 159),
    (744-498, 274), (744-498, 311), (742-498, 350), (742-498, 387),
    (742-498, 425), (742-498, 501), (802-498, 159), (802-498, 198),
    (802-498, 235), (802-498, 274), (802-498, 311), (802-498, 350),
    (801-498, 387), (800-498, 425), (800-498, 462), (800-498, 501),
    (860-498, 159), (860-498, 235), (860-498, 274), (860-498, 311),
    (860-498, 350), (859-498, 387), (858-497, 424), (858-498, 463),
    (858-497, 502), (918-498, 159), (918-498, 198), (919-498, 235),
    (919-498, 274), (919-498, 311), (919-499, 350), (918-498, 387),
    (918-498, 425), (918-498, 462), (918-498, 501),
    ]

arr1 = [
    (565, 159), (565, 198), (564, 235), (564, 274),
    (563, 311), (563, 350), (563, 387), (563, 426),
    (563, 463), (563, 501), (624, 311), (624, 349),
    (624, 387), (684, 235), (684, 274), (684, 311),
    (683, 387), (683, 425), (683, 501), (744, 159),
    (744, 274), (744, 311), (742, 350), (742, 387),
    (742, 425), (742, 501), (802, 159), (802, 198),
    (802, 235), (802, 274), (802, 311), (802, 350),
    (801, 387), (800, 425), (800, 462), (800, 501),
    (860, 159), (860, 235), (860, 274), (860, 311),
    (860, 350), (859, 387), (858, 424), (858, 463),
    (858, 502), (918, 159), (918, 198), (919, 235),
    (919, 274), (919, 311), (919, 350), (918, 387),
    (918, 425), (918, 462), (918, 501),
    ]

image_path = 'images/twocolorsv1.jpeg'

def get_area(x, y, im, n):
    cutted = im[y + 5 : y + 20, x + 5 : x + 35]
    # im = cv.cvtColor(im, cv.COLOR_BGR2HSV)
    # cutted = cv.cvtColor(cutted, cv.COLOR_BGR2HSV)
    minrect = -1
    smallestdist = 50000
    for d in range(len(arr1)):
        distsum = 0
        c = 0
        for k in range(cutted.shape[0]):
            for l in range(cutted.shape[1]):
                imx = arr1[d][0] + l + 5
                imy = arr1[d][1] + k + 5
                b1 = int(im[imy, imx][0])
                g1 = int(im[imy, imx][1])
                r1 = int(im[imy, imx][2])
                b2 = int(cutted[k, l][0])
                g2 = int(cutted[k, l][1])
                r2 = int(cutted[k, l][2])
                difB = b1 - b2
                difG = g1 - g2
                difR = r1 - r2
                rconst = (r1 + r2) / 2
                # distsum = math.sqrt((2 + rconst / 256)*(difR**2) + 4*(difG**2) + (2 + (255 - rconst) / 256)*(difB**2))
                distsum += math.sqrt((difR**2) + (difG**2) + (difB**2))
                c += 1
        distsum = distsum / c
        print(distsum)
        if distsum < smallestdist:
            smallestdist = distsum
            minrect = d
    
    x2 = arr1[minrect][0]
    y2 = arr1[minrect][1]
    w = 40
    h = 25
    cv.rectangle(img, (x2, y2), (x2 + w, y2 + h), (0, 0, 255), 2)
    cv.putText(img, str(n), (x + 12, y + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
    cv.putText(img, str(n), (x2 + 12, y2 + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255))
            

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
    im = cv.GaussianBlur(im,(3,3),0)
    for i in range(x-5, x + 6):
        for j in range(y - 5, y + 6):
            colorsum[0] = colorsum[0] + im[j, i, 0] # b
            colorsum[1] = colorsum[1] + im[j, i, 1] # g
            colorsum[2] = colorsum[2] + im[j, i, 2] # r
            img[j,i] = np.array([255,255,0])
            count += 1

    colorsum[0] = int(colorsum[0] / count)
    colorsum[1] = int(colorsum[1] / count)
    colorsum[2] = int(colorsum[2] / count)
    return colorsum[0], colorsum[1], colorsum[2]

def clickless_color_detection(coords, c):
    tobemasked = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    temp = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    (h, s, v) = sum_area(coords[0], coords[1], temp,c)

    hs = 4
    ss = 25
    vs = 25
    lb = 0 if h - hs <= 0 else h - hs
    lg = 0 if s - ss <= 0 else s - ss
    lr = 0 if v - vs <= 0 else v - vs
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

    # im = cv.GaussianBlur(img,(3,3),0)
    im = img

    if event == cv.EVENT_LBUTTONDOWN:
        tobemasked = cv.cvtColor(im, cv.COLOR_BGR2HSV)

        temp = cv.cvtColor(im, cv.COLOR_BGR2HSV)

        # (h, s, v) = sum_area_for_click_color_matching(x, y, temp)
        get_area(x, y, im)
        # print("{}--{}".format(x,y))
        # hs = 5
        # ss = 33
        # vs = 33
        # lb = 0 if h - hs <= 0 else h - hs
        # lg = 0 if s - ss <= 0 else s - ss
        # lr = 0 if v - vs <= 0 else v - vs
        # ub = 180 if h + hs >= 180 else h + hs
        # ug = 255 if s + ss >= 255 else s + ss
        # ur = 255 if v + vs >= 255 else v + vs

        # lower = np.array([lb, lg, lr], dtype = "uint8")

        # upper = np.array([ub, ug, ur], dtype = "uint8")

        # mask = cv.inRange(tobemasked, lower, upper)

        # contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # for pic, contour in enumerate(contours):
        #     area = cv.contourArea(contour)
        #     if area > 300:
        #         x, y, w, h = cv.boundingRect(contour)
        #         if(x > 580):
        #             imageFrame = cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        #             # cv.putText(img, str(params[0]), (x, y), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255))
        #             break

        cv.imshow('image', img)

img = cv.imread(image_path)
img = cv.resize(img, (1000,750))
# img = img[:,:498]
# img = cv.hconcat([img,img])

for i in range(len(arr)):
    get_area(arr[i][0], arr[i][1], img, i + 1)

cv.imshow('image', img)

# cv.setMouseCallback('image', click_event)

cv.waitKey(0)