import cv2 as cv
import numpy as np
from numpy.ma import divide, mean
from os.path import exists
import xlsxwriter

def test_coords(arr,w,h,img):
    for i in range(len(arr)):
        x = arr[i][0]
        y = arr[i][1]
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv.putText(img, str(i + 1), (x + 12, y + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (50,0,255))

def luminance_correction_with_bright_image(f, b):
    f = f.astype(np.float32)
    b = b.astype(np.float32)
    C = calculate_C(f, b)
    g = divide(f, b) * C
    g = g.astype(np.uint8)
    return g

def coord_read_operations(read_coords_file, img):
    if exists("coords/" + read_coords_file + "left.txt") and exists("coords/" + read_coords_file + "right.txt"):
        a = open_file_then_read(read_coords_file + "left")
        b = open_file_then_read(read_coords_file + "right")
        return a, b

    elif exists("coords/" + read_coords_file + "left.txt") and not exists("coords/" + read_coords_file + "right.txt"):
        cv.imshow('image', img)
        cv.setMouseCallback('image', get_coords, [read_coords_file, 55])
        return [], []

    cv.imshow('image', img)
    cv.setMouseCallback('image', get_coords, [read_coords_file, 0])
    return [], []

def open_file_then_write(filename, x, y, n):
    f = open("coords/" + filename + ".txt","a")
    f.write("({},{})-\n".format(x,y))
    print("{} -> ({},{})".format(n + 1,x,y))
    f.close()

def open_file_then_read(filename):
    f = open("coords/" + filename + ".txt", "r")
    strf = f.read()
    splt = strf.split('-')
    splt = splt[:-1]
    arr = []

    for i in range(len(splt)):
        splt[i] = splt[i].replace('(', '')
        splt[i] = splt[i].replace(')', '')
        temp = splt[i].split(',')
        t1 = int(temp[0])
        t2 = int(temp[1])
        arr.append((t1,t2))

    f.close()
    return arr

def calculate_C(f, b):
    C = mean(f) / divide(f, b).mean()
    return C

def get_coords(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN and params[1] < 110:
        left_or_right(params[0], params[1],x,y)
        params[1] = params[1] + 1

def left_or_right(file_to_read_from, count,x,y):
    if count < 55:
        open_file_then_write((file_to_read_from + "left"), x, y, count)

    elif count >= 55 and count < 110:
        open_file_then_write((file_to_read_from + "right"), x, y, count)

def get_average_color_of_area(im,x, y, w, h):
    c = w * h
    t = im[y : y + h, x: x + w]
    b = np.sum(t[:,:,0])/ c
    g = np.sum(t[:,:,1])/ c
    r = np.sum(t[:,:,2])/ c
    return (b, g, r)

def gamma_correction(im, gamma=0.47):
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    return cv.LUT(im, lookUpTable)

def resize_smaller(im, width = 1000, height = 750):
    im = cv.resize(im, (width,height),interpolation = cv.INTER_AREA)
    return im

def original_bright_images(version, extension = 'jpg'):
    original_image = get_original_image(version)
    bright_image = cv.imread('images/brightv' + str(version) + '.' + extension)
    return original_image, bright_image

def get_original_image(version, extension = 'jpg'):
    original_image = cv.imread('images/originalv' + str(version) + '.' + extension)
    return original_image

def left_right_coordinates(version, extension = 'jpg'):
    img = get_original_image(version, extension)
    read_coords_file = 'originalv' + str(version) + '_coords'
    c1, c2 = coord_read_operations(read_coords_file, img)
    return c1, c2

def apply_gamma_blur(img, gamma, blur):
    img = apply_gamma(gamma, img)
    img = apply_blur(blur, img)
    return img

def apply_gamma(gamma, img):
    if gamma != 0:
        img = gamma_correction(img, gamma)
    return img

def apply_blur(blur, img):
    if blur != 0:
        img = cv.GaussianBlur(img,(blur,blur),0)
    return img

def write_to_excell_file(cols, datas, path):
    workbook = xlsxwriter.Workbook(path)
    worksheet = workbook.add_worksheet()
    for col, data in zip(cols, datas):
        worksheet.write(col, data)
    workbook.close()

def color_converter(img, space):
    if space == 'HSV':
        img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    elif space == 'LUV':
        img = cv.cvtColor(img, cv.COLOR_BGR2LUV)
    elif space == 'LAB':
        img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    return img

def clahe_applier(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    (H,S,V) = cv.split(img)

    clahe = cv.createCLAHE(clipLimit=2, tileGridSize=(8,8))
    final_img = clahe.apply(V) + 30
    final_img = cv.merge([H,S,V])
    final_img = cv.cvtColor(final_img, cv.COLOR_HSV2BGR)
    return final_img

def next_char_for_excell(char):
    if char[-1] == 'Z':
        for i in range(len(char) - 1, -1, -1,):
            if char[i] != 'Z':
                num = ord(char[i])
                num += 1
                char = char[:i] + chr(num) + char[i+1:]
                return char
            else:
                char = char[:i] + 'A' + char[i+1:]
        char += 'A'
        return char

    numeric = ord(char[-1])
    numeric += 1
    char = char[:len(char)-1] + chr(numeric)
    return char

def get_image_custom(path):
    reference_image = cv.imread(path)
    return reference_image

def get_left_right_coords_custom(impath, coordpath):
    img = get_image_custom(impath)
    read_coords_file = coordpath
    c1 = coord_read_operations_single(read_coords_file, img)
    return c1

def coord_read_operations_single(read_coords_file, img):
    if exists("coords/" + read_coords_file + ".txt"):
        a = open_file_then_read(read_coords_file)
        return a

    cv.imshow('image', img)
    cv.setMouseCallback('image', get_coords_single, [read_coords_file, 0])
    return [], []

def get_coords_single(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN and params[1] < 55:
        open_file_then_write((params[0]), x, y, params[1])
        params[1] = params[1] + 1
