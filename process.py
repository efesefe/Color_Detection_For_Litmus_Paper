import cv2 as cv
import numpy as np
from numpy.ma import divide, mean
from matplotlib import pyplot as plt
import math
from os.path import exists
import xlsxwriter
from multiprocessing import Process
from generic_operations import *

def general_get_area(im, n, left, right, w, h, impaint):

    x = left[n - 1][0]
    y = left[n - 1][1]

    cutted = im[y : y + h, x : x + w]
    minrect = -1
    smallestdist = -1
    for d in range(len(right)):
        h1, h2, s1, s2, v1, v2 = 0, 0, 0, 0, 0, 0
        v1, s1, h1 = get_average_color_of_area(im, right[d][0], right[d][1], cutted.shape[1], cutted.shape[0])
        v2, s2, h2 = get_average_color_of_area(im, x, y, cutted.shape[1], cutted.shape[0])
        difH = h1 - h2
        difS = s1 - s2
        difV = v1 - v2
        ph = 2 + (h1 + h2) / 512
        ps = 4
        pv = 2 + (512 - h1 - h2) / 512
        distsum = math.sqrt(ph * (difH ** 2) + ps * (difS ** 2) + pv * (difV ** 2))
        # distsum = math.sqrt((difH ** 2) + (difS ** 2) + (difV ** 2))
        # distsum3 = math.sqrt((difS ** 2) + (difV ** 2)) + abs(difH) # should work better for LAB colorspace

        if distsum < smallestdist or d == 0:
            smallestdist = distsum
            minrect = d

    x2 = right[minrect][0]
    y2 = right[minrect][1]
    cv.rectangle(impaint, (x2, y2), (x2 + w, y2 + h), (0, 0, 255), 1)
    cv.putText(impaint, str(n), (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0))
    cv.putText(impaint, str(n), (x2 + 12, y2 + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255))
    return minrect + 1 == n, minrect, impaint

def test_range():
    procs = []
    for v in range(49, 40, -1):
        p = Process(target=f, args=(v,))
        procs.append(p)
        p.start()

    for c in procs:
        c.join()

def f(v):
    image_name = 'originalv' + str(v)
    img, b_img = original_bright_images(v)
    img2 = img = resize_smaller(img)
    b_img = resize_smaller(b_img)
    l, r = left_right_coordinates(v)

    w = 25
    h = 15
    
    workbook = xlsxwriter.Workbook('values/' + image_name + '_values.xlsx')
    worksheet = workbook.add_worksheet()

    worksheet.write('A1', 'Blur')
    worksheet.write('B1', 'Gamma')
    worksheet.write('C1', 'Luminance Correction')
    worksheet.write('D1', 'True Number (BGR)')

    worksheet.write('F1', 'Blur')
    worksheet.write('G1', 'Gamma')
    worksheet.write('H1', 'Luminance Correction')
    worksheet.write('I1', 'True Number (HSV)')

    worksheet.write('K1', 'Blur')
    worksheet.write('L1', 'Gamma')
    worksheet.write('M1', 'Luminance Correction')
    worksheet.write('N1', 'True Number (LAB)')

    worksheet.write('P1', 'Blur')
    worksheet.write('R1', 'Gamma')
    worksheet.write('S1', 'Luminance Correction')
    worksheet.write('T1', 'True Number (LUV)')

    t = 2
    blu = 0

    gammabgr = gammahsv = gammalab = gammaluv = 0
    blurbgr = blurhsv = blurlab = blurluv = 0
    besttruebgr = besttruehsv = besttruelab = besttrueluv = 0
    truetruesbgr = truetrueshsv = truetrueslab = truetruesluv = 0
    lumbgr = lumhsv = lumlab = lumluv = 0

    while blu <= 10:
        for gam in np.linspace(1.0,0.0,num=41):
            for tog in [0,1]:
                truetruesbgr = truetrueslab = truetrueshsv = truetruesluv = 0
                tm2 = tog % 2 == 0
                if tm2:
                    imglab = imghsv = imgbgr = imgluv = luminance_correction_with_bright_image(img, b_img)

                imglab = cv.cvtColor(imglab, cv.COLOR_BGR2LAB)
                imglab = apply_gamma_blur(imglab, gam, blu)
                
                imghsv = cv.cvtColor(imghsv, cv.COLOR_BGR2HSV)
                imghsv = apply_gamma_blur(imghsv, gam, blu)

                imgluv = cv.cvtColor(imgluv, cv.COLOR_BGR2LUV)
                imgluv = apply_gamma_blur(imgluv, gam, blu)

                imgbgr = apply_gamma_blur(imgbgr, gam, blu)

                for i in range(len(l)):
                    tempbgr, minrect, _ = general_get_area(imgbgr, i + 1, l, r, w,h, img2)
                    if tempbgr:
                        truetruesbgr += 1

                    temphsv, minrect, _ = general_get_area(imghsv, i + 1, l, r, w,h, img2)
                    if temphsv:
                        truetrueshsv += 1

                    templab, minrect, _ = general_get_area(imglab, i + 1, l, r, w,h, img2)
                    if templab:
                        truetrueslab += 1

                    templuv, minrect, _ = general_get_area(imgluv, i + 1, l, r, w,h, img2)
                    if templuv:
                        truetruesluv += 1

                if truetruesbgr > besttruebgr:
                    besttruebgr = truetruesbgr
                    gammabgr = gam
                    blurbgr = blu
                    lumbgr = 1 if tm2 else 0

                if truetrueshsv > besttruehsv:
                    besttruehsv = truetrueshsv
                    gammahsv = gam
                    blurhsv = blu
                    lumhsv = 1 if tm2 else 0

                if truetrueslab > besttruelab:
                    besttruelab = truetrueslab
                    gammalab = gam
                    blurlab = blu
                    lumlab = 1 if tm2 else 0

                if truetruesluv > besttrueluv:
                    besttrueluv = truetruesluv
                    gammaluv = gam
                    blurluv = blu
                    lumluv = 1 if tm2 else 0

                worksheet.write(('A' + str(t)), blu)
                worksheet.write(('B' + str(t)), gam)
                worksheet.write(('C' + str(t)), '+' if tm2 else '-')
                worksheet.write(('D' + str(t)), truetruesbgr)

                worksheet.write(('F' + str(t)), blu)
                worksheet.write(('G' + str(t)), gam)
                worksheet.write(('H' + str(t)), '+' if tm2 else '-')
                worksheet.write(('I' + str(t)), truetrueshsv)

                worksheet.write(('K' + str(t)), blu)
                worksheet.write(('L' + str(t)), gam)
                worksheet.write(('M' + str(t)), '+' if tm2 else '-')
                worksheet.write(('N' + str(t)), truetrueslab)

                worksheet.write(('P' + str(t)), blu)
                worksheet.write(('R' + str(t)), gam)
                worksheet.write(('S' + str(t)), '+' if tm2 else '-')
                worksheet.write(('T' + str(t)), truetruesluv)

                t += 1

        if blu == 0: blu += 1
        blu += 2
    workbook.close()
    print("-------originalv{}---------".format(v))
    print("best true for bgr: {}".format(besttruebgr))
    print("gamma for bgr: {}".format(gammabgr))
    print("blur for bgr: {}".format(blurbgr))
    print("luminance correction: {}".format(lumbgr))
    print("---------------------------")
    print("best true for hsv: {}".format(besttruehsv))
    print("gamma for hsv: {}".format(gammahsv))
    print("blur for hsv: {}".format(blurhsv))
    print("luminance correction: {}".format(lumhsv))
    print("---------------------------")
    print("best true for lab: {}".format(besttruelab))
    print("gamma for lab: {}".format(gammalab))
    print("blur for lab: {}".format(blurlab))
    print("luminance correction: {}".format(lumlab))
    print("---------------------------")
    print("best true for luv: {}".format(besttrueluv))
    print("gamma for luv: {}".format(gammaluv))
    print("blur for luv: {}".format(blurluv))
    print("luminance correction: {}".format(lumluv))
    print("---------------------------")

def singular_test(v, blur, gamma):
    img, im_b = original_bright_images(v)

    img = img2 = resize_smaller(img)

    # im_b = resize_smaller(im_b)

    img = custom_luminance_correction(img)
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    img = apply_gamma_blur(img, gamma, blur)
    l, r = left_right_coordinates(v)

    width = 25
    height = 15
    truetrues = overlap = 0
    maps = {}
    # test_coords(r, width, height, img2)
    # test_coords(l, width, height, img2)
    for i in range(len(l)):
        temp, minrect, img2 = general_get_area(img, i + 1, l, r, width,height, img2)
        if temp:
            truetrues += 1
        else:
            print('{} -> {}'.format(i + 1, minrect + 1))

        if maps.get(minrect) == 1:
            overlap += 1
        else:
            maps[minrect] = 1

    print('True Number: {}'.format(truetrues))
    cv.imshow('image', img2)
    cv.waitKey(0)

def custom_luminance_correction(im):
    
    imgr = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    maxi = 0
    # cv.imshow('before', im)
    for i in range(imgr.shape[0]):
        for j in range(imgr.shape[1]):
            if maxi < imgr[i,j]:
                maxi = imgr[i,j]
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            diff = maxi - imgr[i,j]
            im[i,j][0] = diff * 0.114 + im[i,j][0]
            im[i,j][1] = diff * 0.587 + im[i,j][1]
            im[i,j][2] = diff * 0.299 + im[i,j][2]
    return im

if __name__ == '__main__':
    # test_range()
    singular_test(40, 0, 0.1)
    # custom_luminance_correction()