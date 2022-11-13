import cv2 as cv
import numpy as np
import math
import csv

from matplotlib import pyplot as plt
from numpy.ma import divide, mean
from os.path import exists
from os import remove
from multiprocessing import Process
from generic_operations import *
from machine_learning import *

def reference_comparison(v):

    img = get_original_image(v)
    img2 = get_original_image(v)
    img = resize_smaller(img)
    reference_impath = 'images/referencev3.jpg'
    reference_coordpath = 'referencev3'
    pathcsv = 'ml_data/imgv' + str(v) + '_data.csv'
    ref_img = get_image_custom(reference_impath)

    ref_img = color_converter(ref_img, 'HSV')
    img = color_converter(img, 'HSV')

    lr = get_left_right_coords_custom(reference_impath, reference_coordpath)
    a = coord_set_operations(57, 3, [55,55,55])

    close = False
    if exists(pathcsv):
        remove(pathcsv) # deleting data if they already exist - testing only
    f = open(pathcsv, 'w')
    header = ['X', 'Y', 'H', 'S', 'V']
    writer = csv.writer(f)
    writer.writerow(header)
    close = True
    distsum = 0
    for i in range(len(lr)):
        for k in range(10):
            for l in range(10):
                x_ref = lr[i][0] + l
                y_ref = lr[i][1] + k

                x_imgl = a[0][i][0] + l
                y_imgl = a[0][i][1] + k
                x_imgr = a[2][i][0] + l
                y_imgr = a[2][i][1] + k
                (hr,sr,vr) = get_average_color_of_area(ref_img, x_ref, y_ref, 1, 1)

                (hil,sil,vil) = get_average_color_of_area(img, x_imgl, y_imgl, 1, 1)
                (hir,sir,vir) = get_average_color_of_area(img, x_imgr, y_imgr, 1, 1)
                difHl = int(hr - hil)
                difSl = int(sr - sil)
                difVl = int(vr - vil)

                difHr = int(hr - hir)
                difSr = int(sr - sir)
                difVr = int(vr - vir)

                img = set_new_color_for_area([(hil + difHl) , (sil + difSl) , (vil + difVl)], x_imgl, y_imgl, 1, 1, img)
                img = set_new_color_for_area([(hir + difHr) , (sir + difSr) , (vir + difVr)], x_imgr, y_imgr, 1, 1, img)

                if close:
                    dat = [x_imgl, y_imgl, difHl, difSl, difVl]
                    writer.writerow(dat)
                    dat = [x_imgr, y_imgr, difHr, difSr, difVr]
                    writer.writerow(dat)

    if close:
        f.close()

    img = cv.cvtColor(img, cv.COLOR_HSV2BGR)

    return img

def get_correct_image(version):
    img = reference_comparison(version)
    c = coord_set_operations(57, 3, [55,55,55])
    left_coords = c[0]
    right_coords = c[1]
    correction_ch_1, correction_ch_2, correction_ch_3 = svr_learning(version, right_coords)

    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    for right, ch1, ch2, ch3 in zip(right_coords, correction_ch_1, correction_ch_2, correction_ch_3):
        # for j in range(10):
        #     for i in range(10): 
                right_x = right[0]# + i
                right_y = right[1]# + j
                sumS = img[right_y,right_x][1] + ch2[0]
                sumV = img[right_y,right_x][2] + ch3[0]
                ch1_2 = img[right_y,right_x][0] + ch1[0] if img[right_y,right_x][0] + ch1[0] < 255 else img[right_y,right_x][0]
                ch2_2 = img[right_y,right_x][1] + ch2[0] if sumS < 255 else img[right_y,right_x][1]
                ch3_2 = img[right_y,right_x][2] + ch3[0] if sumV < 255 else img[right_y,right_x][2]
                col = [ch1_2, ch2_2, ch3_2]
                img = set_new_color_for_area(col, right_x, right_y, 10, 10, img)
    img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
    cv.imshow('correct',img)
    cv.waitKey(0)
    return img

def general_get_area(im, n, left, right, w, h, impaint, mark_image = False, dist_calculator = 1):

    x = left[n - 1][0]
    y = left[n - 1][1]

    cutted = im[y : y + h, x : x + w]
    minrect = -1
    smallestdist = -1
    for d in range(len(right)):
        ch1_1, ch1_2, ch1_3 = get_average_color_of_area(im, right[d][0], right[d][1], cutted.shape[1], cutted.shape[0])
        ch2_1, ch2_2, ch2_3 = get_average_color_of_area(im, x, y, cutted.shape[1], cutted.shape[0])
        dif_ch_1 = ch1_1 - ch2_1
        dif_ch_2 = ch1_2 - ch2_2
        dif_ch_3 = ch1_3 - ch2_3
        pch_1 = 2 + (ch1_1 + ch2_1) / 512
        pch_2 = 4
        pch_3 = 2 + (512 - ch1_1 - ch2_1) / 512
        distsum = 255
        if dist_calculator == 1:
            distsum = math.sqrt(pch_1 * (dif_ch_1 ** 2) + pch_2 * (dif_ch_2 ** 2) + pch_3 * (dif_ch_3 ** 2))

        elif dist_calculator == 2:
            distsum = math.sqrt((difH ** 2) + (difS ** 2) + (difV ** 2))
            
        elif dist_calculator == 3: # should work better for LAB colorspace
            distsum = math.sqrt((difS ** 2) + (difV ** 2)) + abs(difH)

        if distsum < smallestdist or d == 0:
            smallestdist = distsum
            minrect = d
    if mark_image:
        x2 = right[minrect][0]
        y2 = right[minrect][1]
        cv.rectangle(impaint, (x2, y2), (x2 + w, y2 + h), (0, 0, 255), 1)
        cv.putText(impaint, str(n), (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0))
        cv.putText(impaint, str(n), (x2 + 12, y2 + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255))
    return minrect + 1 == n, minrect, impaint

def test_range():
    procs = []
    for v in range(49, 48, -1):
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
    col_space_number = len(color_spaces)
    zeros = [0] * col_space_number
    gammas, blurs, lums, besttrues, truetrues = zeros[:], zeros[:], zeros[:], zeros[:], zeros[:]

    imgs = []
    blu = 21
    while blu <= 21:
        for gam in np.linspace(2.0, 0.0, num=3):
            for tog in [0,1]:
                truetrues = zeros[:]
                tm2 = tog % 2 == 0
                temp_img = get_original_image(v)
                temp_img = resize_smaller(temp_img)
                if tm2:
                    temp_img = custom_luminance_correction(v)
                imgs = [temp_img] * col_space_number

                cvt_imgs = []
                for i in range(col_space_number):
                    tmp = color_converter(imgs[i], color_spaces[i])
                    ttt = apply_gamma_blur(tmp, gam, blu)
                    cvt_imgs.append(ttt)

                for k in range(col_space_number):
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
                for i in range(col_space_number):
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

    print_scores(v, color_spaces, besttrues, gammas, blurs, lums)

    write_to_excell_file(cols, datas, path)

def custom_test(v):
    # width = 20
    # height = 10
    # impath = 'images/referencev3.jpg'
    # coordpath = 'referencev3'
    # im = get_image_custom(impath)
    # img2 = get_image_custom(impath)
    # l = get_left_right_coords_custom(impath, coordpath)
    # test_coords(l, width, height, img2)
    # cv.imshow('image', img2)
    # cv.waitKey(0)
    coord_set_operations('originalv' + str(v), 3, [55,55,55])

def singular_test(v, blur, gamma, lum):
    img2 = get_original_image(v)
    img2 = resize_smaller(img2)
    img = get_correct_image(v)
    if lum:
        img = custom_luminance_correction(v)
    img = color_converter(img, 'HSV')

    img = apply_gamma_blur(img, gamma, blur)
    c = coord_set_operations(v, 3, [55,55,55])
    l = c[0]
    r = c[1]

    width = 1
    height = 1
    truetrues = overlap = 0
    maps = {}
    # test_coords(r, 10, 10, img)
    # test_coords(l, 10, 10, img)
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
    img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
    # cv.imshow('image', img)
    # cv.waitKey(0)

def custom_luminance_correction(n):
    
    im, imgr = original_bright_images(n)
    im = resize_smaller(im)
    imgr = resize_smaller(imgr)
    imgr = cv.cvtColor(imgr, cv.COLOR_BGR2HSV)
    maxi = 0
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
    # reference_comparison(47)
    test_range()
    # for i in range(45,44,-1):
    #     print('-------{}-------'.format(i))
    #     singular_test(i, 0, 0, False)
    # custom_test(3)

    # singular_test(57, 0, 0.0, False)

    # for i in range(4, 50):
    #     print("-----{}-----".format(i))
    #     singular_test(i, 0, 0, False)
    #     singular_test(i, 0, 0, True)
    # custom_luminance_correction()
    # svr_learning(46)
    # c_im = get_correct_image(47)
    # cv.imshow('x', c_im)
    # cv.waitKey()
    # custom_test(57)

    # a = coord_set_operations(57, 3, [55,55,55])
    # img = get_original_image(57)
    # img = resize_smaller(img)

    # for i in range(3):
    #     test_coords(a[i], 10, 10, img)
    # cv.imshow('test', img)
    # cv.waitKey(0)