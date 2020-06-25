import cv2
import numpy as np
import kociemba
import time
from operator import itemgetter, attrgetter

font = cv2.FONT_HERSHEY_COMPLEX

# tableau pour stocker le résultat
result = []

# Valeur haute et basse pour chaque couleur dans le spectre HSV
# rouge haut
lower_red = np.array([140, 100, 100])
upper_red = np.array([180, 255, 255])

# rouge bas
lower_red_low = np.array([0,100,100])
upper_red_low = np.array([5,255,255])

# bleu
lower_blue = np.array([101, 75, 75])
upper_blue = np.array([150, 255, 255])

lower_green = np.array([56, 75, 75])
upper_green = np.array([100, 255, 255])

lower_orange = np.array([5, 75, 75])
upper_orange = np.array([15, 255, 255])

lower_yellow = np.array([16, 75, 75])
upper_yellow = np.array([55, 255, 255])

lower_white = np.array([0,0,140])
upper_white = np.array([172,111,255])

def nothing():
    pass

def approx(c):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    return approx

def tri(struct):  
    s = sorted(struct, key=itemgetter(0)) 
    tri1 = []
    tri2 = []
    tri3 = []

    tri1.extend((s[0], s[1], s[2]))
    tri2.extend((s[3], s[4], s[5]))
    tri3.extend((s[6], s[7], s[8]))  

    tri1.sort(key=itemgetter(1))   
    tri2.sort(key=itemgetter(1))   
    tri3.sort(key=itemgetter(1))
    s = tri1+tri2+tri3         
    for i in range(len(s)):
        result.append(s[i][4])        
    return ""

def detect_face(img, trier):
    red_cnt =[]
    blue_cnt =[]
    yellow_cnt =[]
    white_cnt =[]
    orange_cnt =[]
    green_cnt =[]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Mask en fonction des couleurs
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_red_low = cv2.inRange(hsv, lower_red_low, upper_red_low)
    mask_red = mask_red + mask_red_low  
    mask_tot = mask_red + mask_red_low + mask_blue + mask_green + mask_orange + mask_yellow + mask_white
    # Permet d'erode les mask et donc d'avoir plus de précision
    kernel = np.ones((5, 5), np.uint8)
    mask_red = cv2.erode(mask_red, kernel)
    mask_blue = cv2.erode(mask_blue, kernel)
    mask_green = cv2.erode(mask_green, kernel)
    mask_orange = cv2.erode(mask_orange, kernel)
    mask_yellow = cv2.erode(mask_yellow, kernel)
    mask_white = cv2.erode(mask_white, kernel)
    mask_tot = cv2.erode(mask_tot, kernel)
    # Détection des contours pour chaque couleurs
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_orange, _ = cv2.findContours(mask_orange, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_white, _ = cv2.findContours(mask_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_tot, _ = cv2.findContours(mask_tot, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # for i in range(9):
        # Dessin des contours 
    color = []   
    for cnt_red in contours_red:                
        area = cv2.contourArea(cnt_red)
        approxim = approx(cnt_red)
        if area > 250 and area < 10000 and len(approxim) == 4:
            [x, y, w, h] = cv2.boundingRect(approxim)
            ar = w/ float(h)     
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, "rouge", (x, y), font, 1, (255, 255, 255))           
            color.append([x,y,w,h,"R"])
  
    for cnt_blue in contours_blue:             
        area = cv2.contourArea(cnt_blue)    
        approxim = approx(cnt_blue)
        if area > 250 and area < 10000 and len(approxim) == 4:
            x, y, w, h = cv2.boundingRect(approxim)
            ar = w/ float(h)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "bleu", (x, y), font, 1, (255, 255, 255))              
            color.append([x,y,w,h,"F"])                                            

    for cnt_green in contours_green:                  
        area = cv2.contourArea(cnt_green)
        approxim = approx(cnt_green)
        if area > 250 and area < 10000 and len(approxim) == 4:
            x, y, w, h = cv2.boundingRect(approxim)
            ar = w/ float(h)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "vert", (x, y), font, 1, (255, 255, 255))                   
            color.append([x,y,w,h,"B"])    
   
    for cnt_orange in contours_orange:                 
        area = cv2.contourArea(cnt_orange)
        approxim = approx(cnt_orange)
        if area > 250 and area < 10000 and len(approxim) == 4:            
            x, y, w, h = cv2.boundingRect(approxim)
            ar = w/ float(h)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 165, 255), 2)
            cv2.putText(img, "orange", (x, y), font, 1, (255, 255, 255))            
            color.append([x,y,w,h,"L"])             
 
    for cnt_yellow in contours_yellow:            
        area = cv2.contourArea(cnt_yellow)
        approxim = approx(cnt_yellow)
        if area > 250 and area < 10000 and len(approxim) == 4:
            x, y, w, h = cv2.boundingRect(approxim)
            ar = w/ float(h)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(img, "yellow", (x, y), font, 1, (0, 0, 0))                    
            color.append([x,y,w,h,"U"])             
  
    for cnt_white in contours_white:                    
        area = cv2.contourArea(cnt_white)
        approxim = approx(cnt_white)
        if area > 250 and area < 10000 and len(approxim) == 4:
            x, y, w, h = cv2.boundingRect(approxim)
            ar = w / float(h)
            if ar >=0.95 and ar <=1.05:
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
                cv2.putText(img, "white", (x, y), font, 1, (255, 255, 255))                         
                color.append([x,y,w,h,"D"])             
    if trier ==True:
        tri(color)  
    return img

def resize(img):
    scale_percent = 30 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
    return resized

# Dans l'orde:
# U : jaune
# R : rouge
# F : Bleu
# D : blanc
# L : orange
# B : vert

# Lecture des images
bleu = cv2.imread('./images/face_bleu_mel.JPG')
blanc = cv2.imread('./images/face_blanche_mel.JPG')
rouge = cv2.imread('./images/face_rouge_mel.JPG')
orange = cv2.imread('./images/face_orange_mel.JPG')
jaune = cv2.imread('./images/face_jaune_mel.JPG')
vert = cv2.imread('./images/face_verte_mel.JPG')
# détection des couleurs
bleu = resize(bleu)
blanc = resize(blanc) 
rouge = resize(rouge)
orange = resize(orange)
jaune = resize(jaune)
vert = resize(vert)
detect_face(jaune, True)
detect_face(rouge, True)
detect_face(bleu, True)
detect_face(blanc, True)
detect_face(orange, True)
detect_face(vert, True)
# Affichage de l'image
cv2.imshow("bleu", bleu)
cv2.imshow("blanc", blanc)
cv2.imshow("rouge", rouge)
cv2.imshow("orange", orange)
cv2.imshow("jaune", jaune)
cv2.imshow("vert", vert)

s = ''.join(result)
print(result)
solve = kociemba.solve(s)
print(solve)

test = cv2.imread('./my.png')
detect_face(test, False)
cv2.imshow('test',test)
# arret du programme
cv2.waitKey(0)
cv2.destroyAllWindows()
