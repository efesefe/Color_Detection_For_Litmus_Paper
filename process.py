import cv2 as cv
import numpy as np
from numpy.ma import divide, mean
from matplotlib import pyplot as plt
import math
from os.path import exists

arr=[(565-498,159),(565-498,198),(564-498,235),(564-498,274),(563-498,311),(563-498,350),(563-500,387),(563-499,426),(563-499,463),(563-499,501),(624-498,311),(624-498,349),(624-499,387),(684-498,235),(684-498,274),(684-498,311),(683-498,387),(683-498,425),(683-498,501),(744-498,159),(744-498,274),(744-498,311),(742-498,350),(742-498,387),(742-498,425),(742-498,501),(802-498,159),(802-498,198),(802-498,235),(802-498,274),(802-498,311),(802-498,350),(801-498,387),(800-498,425),(800-498,462),(800-498,501),(860-498,159),(860-498,235),(860-498,274),(860-498,311),(860-498,350),(859-498,387),(858-497,424),(858-498,463),(858-497,502),(918-498,159),(918-498,198),(919-498,235),(919-498,274),(919-498,311),(919-499,350),(918-498,387),(918-498,425),(918-498,462),(918-498,501),]

arr1=[(565,159),(565,198),(564,235),(564,274),(563,311),(563,350),(563,387),(563,426),(563,463),(563,501),(624,311),(624,349),(624,387),(684,235),(684,274),(684,311),(683,387),(683,425),(683,501),(744,159),(744,274),(744,311),(742,350),(742,387),(742,425),(742,501),(802,159),(802,198),(802,235),(802,274),(802,311),(802,350),(801,387),(800,425),(800,462),(800,501),(860,159),(860,235),(860,274),(860,311),(860,350),(859,387),(858,424),(858,463),(858,502),(918,159),(918,198),(919,235),(919,274),(919,311),(919,350),(918,387),(918,425),(918,462),(918,501),]

thisone_left=[(20,263),(25,300),(24,336),(26,375),(25,409),(27,448),(25,483),(28,520),(28,555),(32,593),(81,337),(81,372),(85,557),(135,260),(136,297),(138,333),(140,412),(139,484),(141,559),(194,263),(194,298),(196,336),(193,373),(198,410),(197,559),(196,595),(252,263),(252,300),(252,338),(253,373),(250,411),(251,447),(251,484),(251,520),(250,559),(253,596),(306,262),(306,298),(308,334),(306,374),(308,411),(308,449),(307,485),(306,561),(310,596),(364,263),(364,298),(363,335),(362,371),(363,410),(362,447),(364,487),(361,523),(362,561),(364,597),]

thisone_right=[(590,262),(591,300),(590,340),(588,374),(588,410),(593,449),(591,487),(593,523),(591,560),(593,599),(649,336),(651,374),(650,561),(710,260),(710,299),(711,334),(711,412),(710,487),(711,562),(766,261),(770,298),(769,336),(770,375),(767,410),(769,560),(770,600),(827,261),(826,296),(825,333),(823,369),(827,413),(827,452),(824,485),(822,522),(825,559),(826,598),(886,260),(887,295),(887,334),(885,375),(885,411),(884,448),(883,486),(882,561),(880,597),(943,261),(945,298),(945,335),(943,372),(943,410),(942,450),(940,488),(939,525),(939,565),(938,598),]

thisone_left2=[(66,258),(59,295),(58,333),(63,369),(60,406),(60,446),(63,483),(64,522),(63,559),(60,594),(123,327),(122,368),(126,558),(180,256),(183,293),(179,330),(179,405),(181,481),(182,558),(240,256),(241,294),(241,329),(240,369),(240,405),(240,555),(298,260),(299,293),(298,368),(297,444),(299,480),(297,518),(300,558),(302,594),(354,257),(356,294),(357,329),(353,366),(357,406),(353,442),(355,479),(357,554),(356,592),(411,259),(412,291),(412,332),(413,367),(412,406),(413,444),(414,477),(413,516),(411,554),(414,591),]

thisone_right2=[(635,266),(635,301),(631,335),(632,372),(632,407),(631,445),(632,480),(634,520),(633,557),(636,592),(695,336),(692,372),(690,555),(749,265),(752,299),(751,336),(748,408),(747,481),(749,556),(803,266),(802,302),(806,338),(805,372),(804,407),(801,550),(800,591),(857,267),(858,302),(859,335),(855,372),(858,409),(858,445),(855,483),(856,517),(859,556),(858,588),(913,269),(914,300),(912,336),(913,374),(913,409),(913,447),(911,480),(910,554),(911,588),(966,265),(965,304),(966,338),(965,370),(968,409),(964,446),(963,483),(964,513),(963,555),(965,588),]

globarr1 = []
globarr2 = []
nnn = 0
### --- CODE --- ###

def luminance_correction_with_bright_image(f, b):
    f = f.astype(np.float32)
    b = b.astype(np.float32)

    C = calculate_C(f, b)
    g = divide(f, b) * C
    g = g.astype(np.uint8)
    return g

def coord_read_operations(read_coords_file):
    if exists("coords/" + read_coords_file + "left.txt") and exists("coords/" + read_coords_file + "right.txt"):
        a = open_file_then_read(read_coords_file + "left")
        b = open_file_then_read(read_coords_file + "right")
        return a, b

    elif exists("coords/" + read_coords_file + "left.txt") and not exists("coords/" + read_coords_file + "right.txt"):
        cv.setMouseCallback('image', get_coords, [read_coords_file, 55])
    else:
        cv.setMouseCallback('image', get_coords, [read_coords_file, 0])
    return [], []

def open_file_then_write(filename, x, y):
    f = open("coords/" + filename + ".txt","a")
    f.write("({},{})-".format(x,y))
    print("({},{})-".format(x,y))
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

def clahe_on_bgr_image(bgr):

    lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)

    gridsize = 9

    clahe = cv.createCLAHE(clipLimit=0.2,tileGridSize=(gridsize,gridsize))

    lab[...,0] = clahe.apply(lab[...,0])

    bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)

    return bgr

def calculate_C(f, b):
    C = mean(f) / divide(f, b).mean()
    return C

def get_coords(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        if params[1] < 110:
            if params[1] < 55:
                open_file_then_write((params[0] + "left"), x, y)
            elif params[1] >= 55 and params[1] < 110:
                open_file_then_write((params[0] + "right"), x, y)
            params[1] = params[1] + 1
            if params[1] == 110:
                open_file_then_read(params[0] + "left", 1)
                open_file_then_read(params[0] + "right", 2)
                params[1] = params[1] + 1

def get_average_color_of_area(im,x, y, w, h):

    b, g, r, c = 0,0,0,0

    for i in range(y, y + h):
        for j in range(x, x + w):
            b += im[i, j][0]
            g += im[i, j][1]
            r += im[i, j][2]
            c += 1
    b = b / c
    g = g / c
    r = r / c
    return (b, g, r)

def histogram_equalization(colorimage):
    # For ease of understanding, we explicitly equalize each channel individually
    colorimage_b = cv.equalizeHist(colorimage[:,:,0])
    colorimage_g = cv.equalizeHist(colorimage[:,:,1])
    colorimage_r = cv.equalizeHist(colorimage[:,:,2])
 
    # Next we stack our equalized channels back into a single image
    colorimage_e = np.stack((colorimage_b,colorimage_g,colorimage_r), axis=2)
    return colorimage_e

def gamma_correction(im, gamma=0.47):
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    return cv.LUT(im, lookUpTable)

def general_get_area(x, y, im,b_im, n, left, right, gam):

    im = cv.cvtColor(im, cv.COLOR_BGR2HSV)

    im = gamma_correction(im, gam)
    fr = 9 # 15 is best
    im = cv.GaussianBlur(im,(fr,fr),0)

    referenceColorID = 4

    referenceColor = get_average_color_of_area(im, left[referenceColorID][0], left[referenceColorID][1], 20, 15)
    referenceColor2 = get_average_color_of_area(im, right[referenceColorID][0], right[referenceColorID][1], 20, 15)
    
    referenceColor = np.subtract(referenceColor2, referenceColor)
    
    cutted = im[y : y + 15, x : x + 25]
    # referenceColor[0] = 0
    # referenceColor[1] = 0
    # cutted = np.add(cutted, referenceColor)
    
    minrect = -1
    smallestdist = -1

    for d in range(len(right)):
        
        h1, h2, s1, s2, v1, v2 = 0, 0, 0, 0, 0, 0
        v1, s1, h1 = get_average_color_of_area(im, right[d][0], right[d][1], cutted.shape[1], cutted.shape[0])
        v2, s2, h2 = get_average_color_of_area(im, x, y, cutted.shape[1], cutted.shape[0])
        # v1 = v2 = maxx
        difH = h1 - h2
        difS = s1 - s2
        difV = v1 - v2

        ph = 2 + (h1 + h2) / 512
        ps = 4
        pv = 2 + (512 - h1 - h2) / 512

        distsum = math.sqrt(ph * (difH ** 2) + ps * (difS ** 2) + pv * (difV ** 2))
        
        if distsum < smallestdist or d == 0:
            smallestdist = distsum
            minrect = d

    x2 = right[minrect][0]
    y2 = right[minrect][1]
    w = 25
    h = 15

    cv.rectangle(img, (x2, y2), (x2 + w, y2 + h), (0, 0, 255), 2)
    cv.putText(img, str(n), (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0))
    cv.putText(img, str(n), (x2 + 12, y2 + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255))

    return minrect + 1 == n, minrect

image_name = 'originalv2'
bright_name = 'brightv5'
image_path = 'images/' + image_name + '.jpg'
bright_image_path = 'images/'+ bright_name +'.jpg'
read_coords_file = image_name + '_coords'

img = cv.imread(image_path)

img = cv.resize(img, (1000,750))

b_img = cv.imread(bright_image_path)

b_img = cv.resize(b_img, (1000,750))

cv.imshow('image', img)

globarr1, globarr2 = coord_read_operations(read_coords_file)

# img = luminance_correction_with_bright_image(img, b_img)

#######################################
truetrues = 0
overlap = 0
maps = {}
l = globarr1
r = globarr2
for i in range(len(l)):
    temp, minrect = general_get_area(l[i][0], l[i][1], img, b_img, i + 1, l, r, 0.47)
    if temp:
        truetrues += 1
    if maps.get(minrect) == 1:
        overlap += 1
        
    else:
        maps[minrect] = 1
    # print('{}->{}'.format(i +1, minrect+1))
print("true guesses: {}".format(truetrues))
print("overlapping: {}".format(overlap))

########################################
cv.imshow('image', img)


cv.waitKey(0)