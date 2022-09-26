import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from numpy.ma import divide, mean
import math
from os.path import exists
from os import remove
import csv

from multiprocessing import Process
from generic_operations import *
from machine_learning import *

def reference_comparison(v):
    width = 5
    height = 5
    # img, im_b = original_bright_images(v)
    img = get_original_image(v)
    img2 = get_original_image(v)
    img = resize_smaller(img)
    impath = 'images/referencev3.jpg'
    coordpath = 'referencev3'
    ref_img = get_image_custom(impath)
    # cv.imshow('original', ref_img)
    ref_img = color_converter(ref_img, 'HSV')
    img = color_converter(img, 'HSV')

    lr = get_left_right_coords_custom(impath, coordpath)
    a = coord_set_operations(57, 3, [55,55,55])
    
    # li, ri = left_right_coordinates(v)

    cols = []
    datas = []
    col_char = 'A'
    cols.append(col_char + '1')
    col_char = next_char_for_excell(col_char)
    cols.append(col_char  + '1')
    col_char = next_char_for_excell(col_char)
    cols.append(col_char  + '1')
    col_char = next_char_for_excell(col_char)
    # cols.append(col_char  + '1')
    # col_char = next_char_for_excell(col_char)
    # cols.append(col_char  + '1')
    # col_char = next_char_for_excell(col_char)
    datas.append('X')
    datas.append('Y')
    # datas.append('H')
    # datas.append('S')
    datas.append('V')
    t = 2
    path = 'ml_data/imgv' + str(v) + '_data.xlsx'
    pathcsv = 'ml_data/imgv' + str(v) + '_data.csv'
    close = False
    if exists(pathcsv): # deleting data if they already exist - testing only
        remove(path)    # deleting data if they already exist - testing only
        remove(pathcsv) # deleting data if they already exist - testing only
    f = open(pathcsv, 'w')
    header = ['X', 'Y', 'V']# header = ['X', 'Y', 'H', 'S', 'V']
    writer = csv.writer(f)
    # write the header
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
                # difHl = int(hr - hil)
                # difSl = int(sr - sil)
                difVl = int(vr - vil)

                # difHr = int(hr - hir)
                # difSr = int(sr - sir)
                difVr = int(vr - vir)

                # img = set_new_color_for_area([(hil + difHl) , (sil + difSl) , (vil + difVl)], x_imgl, y_imgl, 10, 10, img)
                # img = set_new_color_for_area([(hir + difHr) , (sir + difSr) , (vir + difVr)], x_imgr, y_imgr, 10, 10, img)
                img = set_new_color_for_area([(hil) , (sil) , (vil + difVl)], x_imgl, y_imgl, 1, 1, img)
                img = set_new_color_for_area([(hir) , (sir) , (vir + difVr)], x_imgr, y_imgr, 1, 1, img)

                if not exists(path) and close:
                        
                    col_char = 'A'
                    cols.append(col_char + str(t))
                    col_char = next_char_for_excell(col_char)
                    cols.append(col_char  + str(t))
                    col_char = next_char_for_excell(col_char)
                    cols.append(col_char  + str(t))
                    col_char = next_char_for_excell(col_char)
                    # cols.append(col_char  + str(t))
                    # col_char = next_char_for_excell(col_char)
                    # cols.append(col_char  + str(t))
                    # col_char = next_char_for_excell(col_char)
                    datas.append(x_imgl)
                    datas.append(y_imgl)
                    # datas.append(difHl)
                    # datas.append(difSl)
                    datas.append(difVl)

                    t += 1
                    col_char = 'A'
                    cols.append(col_char + str(t))
                    col_char = next_char_for_excell(col_char)
                    cols.append(col_char  + str(t))
                    col_char = next_char_for_excell(col_char)
                    cols.append(col_char  + str(t))
                    col_char = next_char_for_excell(col_char)
                    # # cols.append(col_char  + str(t))
                    # # col_char = next_char_for_excell(col_char)
                    # # cols.append(col_char  + str(t))
                    # # col_char = next_char_for_excell(col_char)
                    datas.append(x_imgr)
                    datas.append(y_imgr)
                    # # datas.append(difHr)
                    # # datas.append(difSr)
                    datas.append(difVr)

                    # # write the data
                    dat = [x_imgl, y_imgl, difVl]# dat = [x_imgl, y_imgl, difHl, difSl, difVl]
                    writer.writerow(dat)
                    dat = [x_imgr, y_imgr, difVr]# dat = [x_imgr, y_imgr, difHr, difSr, difVr]
                    writer.writerow(dat)
                    t += 1

    if not exists(path):
        write_to_excell_file(cols, datas, path)
    if close:
        f.close()
    
    img = cv.cvtColor(img, cv.COLOR_HSV2BGR)

    return img

def set_new_color_for_area(color, x, y, w, h,img):
    img[y : y + h , x : x + w] = color
    return img

def get_correct_image(version):
    # image_name = 'originalv' + str(version)
    # img, b_img = original_bright_images(version)
    # img2 = img = resize_smaller(img)
    # b_img = resize_smaller(b_img)
    img = reference_comparison(version)
    # left_coords, right_coords = left_right_coordinates(version)
    c = coord_set_operations(57, 3, [55,55,55])
    left_coords = c[0]
    right_coords = c[1]
    # predict_coords = np.concatenate(img)
    # correctionH, correctionS, correctionV = svr_learning(version, right_coords)
    # print(predict_coords)
    # predict_coords = []
    # for a in range(img.shape[1]):
    #     for b in range(img.shape[0]):
    #         predict_coords.append([a,b])
    correctionV = svr_learning(version, right_coords)

    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    """cH,cS,"""
    for right,cV in zip(right_coords, correctionV):#zip(right_coords,correctionH,correctionS, correctionV):
        for j in range(10):
            for i in range(10): 
                right_x = right[0] + i
                right_y = right[1] + j
                H2 = img[right_y,right_x][0] #+ cH[0] if img[right_y,right_x][0] + cH[0] > 255 else img[right_y,right_x][0]
                S2 = img[right_y,right_x][1] #+ cS[0] if img[right_y,right_x][1] + cS[0] > 255 else img[right_y,right_x][1]
                V2 = img[right_y,right_x][2] + cV[0] if img[right_y,right_x][2] + cV[0] < 255 else img[right_y,right_x][2]
                col = [H2, S2, V2]
                img = set_new_color_for_area(col, right_x, right_y, 1, 1, img)
    img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
    cv.imshow('correct',img)
    cv.waitKey(0)
    return img

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

    # x2 = right[minrect][0]
    # y2 = right[minrect][1]
    # cv.rectangle(impaint, (x2, y2), (x2 + w, y2 + h), (0, 0, 255), 1)
    # cv.putText(impaint, str(n), (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0))
    # cv.putText(impaint, str(n), (x2 + 12, y2 + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255))
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
    blu = 0
    while blu <= 21:
        for gam in np.linspace(2.0, 0.0, num=41):
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
    # img, im_b = original_bright_images(v)
    # img = get_original_image(v)
    img2 = get_original_image(v)
    # img = resize_smaller(img)
    img2 = resize_smaller(img2)
    img = get_correct_image(v)
    if lum:
        img = custom_luminance_correction(v)
    # cv.imshow('corrected', img)
    # cv.waitKey(0)
    img = color_converter(img, 'HSV')

    img = apply_gamma_blur(img, gamma, blur)
    c = coord_set_operations(v, 3, [55,55,55])
    l = c[0]
    r = c[1]
    # l, r = left_right_coordinates(v)

    width = 10
    height = 10
    truetrues = overlap = 0
    maps = {}
    # test_coords(r, width, height, img)
    # test_coords(l, width, height, img)
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
    # img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
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
    # reference_comparison(47)
    # test_range()
    # for i in range(45,44,-1):
    #     print('-------{}-------'.format(i))
    #     singular_test(i, 0, 0, False)
    # custom_test(3)

    singular_test(57, 0, 0.0, False)

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