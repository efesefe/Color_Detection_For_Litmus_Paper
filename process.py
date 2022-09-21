import cv2 as cv
import numpy as np
from numpy.ma import divide, mean
from matplotlib import pyplot as plt
import math
from os.path import exists

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
    for v in range(42, 41, -1):
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
    
    path = 'values/' + image_name + '_values.xlsx'
    cols = []
    datas = []
    color_spaces = ['HSV', 'LAB', 'LUV', 'BGR']
    col_char = 'A'
    for i in range(len(color_spaces)):
        cols.append(col_char + '1')
        col_char = next_char_for_excell(col_char)
        cols.append(col_char  + '1')
        col_char = next_char_for_excell(col_char)
        cols.append(col_char  + '1')
        col_char = next_char_for_excell(col_char)
        cols.append(col_char  + '1')
        col_char = next_char_for_excell(col_char)
        datas.append('Blur')
        datas.append('Gamma')
        datas.append('Custom Luminance Correction')
        datas.append('True Number ({})'.format(color_spaces[i]))
        col_char = next_char_for_excell(col_char)

    t = 2
    
    gammas = [0] * len(color_spaces)
    blurs = [0] * len(color_spaces)
    lums = [0] * len(color_spaces)
    besttrues = [0] * len(color_spaces)
    truetrues = [0] * len(color_spaces)

    imgs = []

    blu = 9
    while blu <= 11:
        for gam in np.linspace(2.0, 0.0, num=6):
            for tog in [0,1]:
                truetrues = [0] * len(color_spaces)
                tm2 = tog % 2 == 0
                temp_img = get_original_image(v)
                temp_img = resize_smaller(temp_img)
                if tm2:
                    temp_img = custom_luminance_correction(v)
                imgs = [temp_img] * len(color_spaces)

                cvt_imgs = []
                for i in range(len(color_spaces)):
                    tmp = color_converter(imgs[i], color_spaces[i])
                    ttt = apply_gamma_blur(tmp, gam, blu)
                    cvt_imgs.append(ttt)

                for k in range(len(color_spaces)):
                    for i in range(len(l)):
                        temp, _, _ = general_get_area(cvt_imgs[k], i + 1, l, r, w,h, img2)
                        if temp:
                            truetrues[k] = truetrues[k] + 1

                    if truetrues[k] > besttrues[k]:
                        besttrues[k] = truetrues[k]
                        gammas[k] = gam
                        blurs[k] = blu
                        lums[k] = 1 if tm2 else 0

                col_char = 'A'
                for i in range(len(color_spaces)):
                    cols.append(col_char + str(t))
                    col_char = next_char_for_excell(col_char)
                    cols.append(col_char  + str(t))
                    col_char = next_char_for_excell(col_char)
                    cols.append(col_char  + str(t))
                    col_char = next_char_for_excell(col_char)
                    cols.append(col_char  + str(t))
                    col_char = next_char_for_excell(col_char)
                    datas.append(str(blu))
                    datas.append(str(gam))
                    datas.append('+' if tm2 else '-')
                    datas.append(truetrues[i])
                    col_char = next_char_for_excell(col_char)

                t += 1

        if blu == 0: blu += 1
        blu += 2

    print("-------originalv{}---------".format(v))
    for i in range(len(color_spaces)):
        print(color_spaces[i] + ':')
        print("best true: {}".format(besttrues[i]))
        print("gamma: {}".format(gammas[i]))
        print("blur: {}".format(blurs[i]))
        print("luminance correction: {}".format(lums[i]))
        print("---------------------------")

    write_to_excell_file(cols, datas, path)

def custom_test(v):
    width = 20
    height = 10
    impath = 'images/referencev3.jpg'
    coordpath = 'referencev3'
    im = get_image_custom(impath)
    img2 = get_image_custom(impath)
    l = get_left_right_coords_custom(impath, coordpath)
    test_coords(l, width, height, img2)
    cv.imshow('image', img2)
    cv.waitKey(0)

def singular_test(v, blur, gamma, lum):
    # img, im_b = original_bright_images(v)
    img = get_original_image(v)
    img2 = get_original_image(v)
    img = resize_smaller(img)
    # img2 = resize_smaller(img)

    if lum:
        img = custom_luminance_correction(v)

    img = color_converter(img, 'HSV')

    img = apply_gamma_blur(img, gamma, blur)
    l, r = left_right_coordinates(v)

    width = 20
    height = 10
    truetrues = overlap = 0
    maps = {}
    # test_coords(r, width, height, img2)
    # test_coords(l, width, height, img2)
    for i in range(len(l)):
        temp, minrect, img2 = general_get_area(img, i + 1, l, r, width,height, img2)
        if temp:
            truetrues += 1
        # else:
        #     print('{} -> {}'.format(i + 1, minrect + 1))

        if maps.get(minrect) == 1:
            overlap += 1
        else:
            maps[minrect] = 1
    if lum:
        print('True Number (custom): {}'.format(truetrues))
    else:
        print('True Number: {}'.format(truetrues))
    # cv.imshow('image', img2)
    # cv.waitKey(0)

def custom_luminance_correction(n):
    
    im, imgr = original_bright_images(n)
    im = resize_smaller(im)
    imgr = resize_smaller(imgr)
    imgr = cv.cvtColor(imgr, cv.COLOR_BGR2HSV)
    # imgr = apply_blur(51, imgr)
    maxi = 0
    # cv.imshow('before', im)
    x = 0
    for i in range(imgr.shape[0]):
        for j in range(imgr.shape[1]):
            maxi += imgr[i,j][2]
            x += 1
    maxi /= x
    im = cv.cvtColor(im, cv.COLOR_BGR2HSV)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            diff = maxi - imgr[i,j][2]
            im[i,j][2] = np.add(im[i,j][2],diff)
    im = cv.cvtColor(im, cv.COLOR_HSV2BGR)
    return im

if __name__ == '__main__':
    # test_range()
    # singular_test(42, 9, 2.0, False)
    custom_test(3)
    # singular_test(41, 5, 1.9, False)
    # for i in range(4, 50):
    #     print("-----{}-----".format(i))
    #     singular_test(i, 0, 0, False)
    #     singular_test(i, 0, 0, True)
    # custom_luminance_correction()