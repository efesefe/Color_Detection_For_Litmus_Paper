import cv2 as cv
import colorsys
import numpy as np
import matplotlib.pyplot as plt

def sum_area(x, y, im, n=0):
    imageFrame = cv.rectangle(img, (x, y), (x + 38, y + 23), (0, 0, 255), 1)
    cv.putText(img, str(n), (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
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
    # print('{}-{}-{}'.format(h,s,v))
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

    istrue = False

    for pic, contour in enumerate(contours):
        area = cv.contourArea(contour)
        if area > 300:
            istrue = True
            x, y, w, h = cv.boundingRect(contour)
            if x > 540:
                imageFrame = cv.rectangle(img, (coords[0], coords[1]), (coords[0] + w, coords[1] + h), (0, 0, 255), 1)
                cv.putText(img, str(c), (coords[0], coords[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
                break
            
    cv.imshow('image', img)

def click_event(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        tobemasked = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        temp = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        (h, s, v) = sum_area_for_click_color_matching(x, y, temp)

        # print('{} {}'.format(x,y))
        # xx = 918
        # yy = 159
        # imageFrame = cv.rectangle(img, (xx, yy), (xx + 40, yy + 25), (0, 0, 255), 1)
        # cv.imshow('image', img)
        # return

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
# d = cv.cvtColor(img, cv.COLOR_BGR2HSV)
img2 = cv.imread('images/original.jpeg')

img = cv.resize(img, (1000,750))

# xx = 624
# yy = 311
# # sum_area(xx, yy, img)
# imageFrame = cv.rectangle(img, (xx, yy), (xx + 40, yy + 25), (0, 0, 255), 1)
# xx = 624
# yy = 349
# # sum_area(xx, yy, img)
# imageFrame = cv.rectangle(img, (xx, yy), (xx + 40, yy + 25), (0, 0, 255), 1)
# xx = 624
# yy = 387
# # sum_area(xx, yy, img)
# imageFrame = cv.rectangle(img, (xx, yy), (xx + 40, yy + 25), (0, 0, 255), 1)

dictate = {
    sum_area(565-498, 159, img): (565-498, 159), sum_area(744-498, 159, img): (744-498, 159), sum_area(860-498, 274, img): (860-498, 274),
    sum_area(565-498, 198, img): (565-498, 198), sum_area(744-498, 274, img): (744-498, 274), sum_area(860-498, 311, img): (860-498, 311),
    sum_area(564-498, 235, img): (564-498, 235), sum_area(744-498, 311, img): (744-498, 311), sum_area(860-498, 350, img): (860-498, 350),
    sum_area(564-498, 274, img): (564-498, 274), sum_area(742-498, 350, img): (742-498, 350), sum_area(859-498, 387, img): (859-498, 387),
    sum_area(563-498, 311, img): (563-498, 311), sum_area(742-498, 387, img): (742-498, 387), sum_area(858-497, 424, img): (858-497, 424),
    sum_area(563-498, 350, img): (563-498, 350), sum_area(742-498, 425, img): (742-498, 425), sum_area(858-498, 463, img): (858-498, 463),
    sum_area(563-500, 387, img): (563-500, 387), sum_area(742-498, 501, img): (742-498, 501), sum_area(858-497, 502, img): (858-497, 502),
    sum_area(563-499, 426, img): (563-499, 426), sum_area(802-498, 159, img): (802-498, 159), sum_area(918-498, 159, img): (918-498, 159),
    sum_area(563-499, 463, img): (563-499, 463), sum_area(802-498, 198, img): (802-498, 198), sum_area(918-498, 198, img): (918-498, 198),
    sum_area(563-499, 501, img): (563-499, 501), sum_area(802-498, 235, img): (802-498, 235), sum_area(919-498, 235, img): (919-498, 235),
    sum_area(624-498, 311, img): (624-498, 311), sum_area(802-498, 274, img): (802-498, 274), sum_area(919-498, 274, img): (919-498, 274),
    sum_area(624-498, 349, img): (624-498, 349), sum_area(802-498, 311, img): (802-498, 311), sum_area(919-498, 311, img): (919-498, 311),
    sum_area(624-499, 387, img): (624-499, 387), sum_area(802-498, 350, img): (802-498, 350), sum_area(919-499, 350, img): (919-499, 350),
    sum_area(684-498, 235, img): (684-498, 235), sum_area(801-498, 387, img): (801-498, 387), sum_area(918-498, 387, img): (918-498, 387),
    sum_area(684-498, 274, img): (684-498, 274), sum_area(800-498, 425, img): (800-498, 425), sum_area(918-498, 425, img): (918-498, 425),
    sum_area(684-498, 311, img): (684-498, 311), sum_area(800-498, 462, img): (800-498, 462), sum_area(918-498, 462, img): (918-498, 462),
    sum_area(683-498, 387, img): (683-498, 387), sum_area(800-498, 501, img): (800-498, 501), sum_area(918-498, 501, img): (918-498, 501),
    sum_area(683-498, 425, img): (683-498, 425), sum_area(860-498, 159, img): (860-498, 159),
    sum_area(683-498, 501, img): (683-498, 501), sum_area(860-498, 235, img): (860-498, 235),
}
c = 0
for x in dictate:
    c += 1
    clickless_color_detection(dictate.get(x), c)
cv.imshow('image', img)
cv.setMouseCallback('image', click_event)

cv.waitKey(0)