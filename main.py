import cv2
import numpy as np


font = cv2.FONT_HERSHEY_COMPLEX

color =[]

# ok ~
lower_red = np.array([140, 100, 100])
upper_red = np.array([180, 255, 255])

lower_red_low = np.array([0,100,100])
upper_red_low = np.array([5,255,255])

# ok
lower_blue = np.array([101, 75, 75])
upper_blue = np.array([150, 255, 255])

# ok
lower_green = np.array([46, 75, 75])
upper_green = np.array([100, 255, 255])

# ok
lower_orange = np.array([5, 75, 75])
upper_orange = np.array([15, 255, 255])

# ok
lower_yellow = np.array([16, 75, 75])
upper_yellow = np.array([60, 255, 255])

lower_white = np.array([0,0,140])
upper_white = np.array([172,111,255])

def nothing():
    pass

def approx(c):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    return approx

def detect_face(img):
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
    # Permet d'erode les mask et donc d'avoir plus de précision
    kernel = np.ones((5, 5), np.uint8)
    mask_red = cv2.erode(mask_red, kernel)
    mask_blue = cv2.erode(mask_blue, kernel)
    mask_green = cv2.erode(mask_green, kernel)
    mask_orange = cv2.erode(mask_orange, kernel)
    mask_yellow = cv2.erode(mask_yellow, kernel)
    mask_white = cv2.erode(mask_white, kernel)
    # Détection des contours pour chaque couleurs
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_orange, _ = cv2.findContours(mask_orange, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_white, _ = cv2.findContours(mask_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Dessin des contours
    for cnt_red in contours_red:
        area = cv2.contourArea(cnt_red)
        approxim = approx(cnt_red)
        if area > 250 and area < 10000 and len(approxim) == 4:
            x, y, w, h = cv2.boundingRect(approxim)
            ar = w/ float(h)     
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, "rouge", (x, y), font, 1, (255, 255, 255))

    for cnt_blue in contours_blue:
        area = cv2.contourArea(cnt_blue)    
        approxim = approx(cnt_blue)
        if area > 250 and area < 10000 and len(approxim) == 4:
            x, y, w, h = cv2.boundingRect(approxim)
            ar = w/ float(h)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "bleu", (x, y), font, 1, (255, 255, 255))

    for cnt_green in contours_green:
        area = cv2.contourArea(cnt_green)
        approxim = approx(cnt_green)
        if area > 250 and area < 10000 and len(approxim) == 4:
            x, y, w, h = cv2.boundingRect(approxim)
            ar = w/ float(h)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "vert", (x, y), font, 1, (255, 255, 255))

    for cnt_orange in contours_orange:
        area = cv2.contourArea(cnt_orange)
        approxim = approx(cnt_orange)
        if area > 250 and area < 10000 and len(approxim) == 4:
            x, y, w, h = cv2.boundingRect(approxim)
            ar = w/ float(h)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 165, 255), 2)
            cv2.putText(img, "orange", (x, y), font, 1, (255, 255, 255))

    for cnt_yellow in contours_yellow:
        area = cv2.contourArea(cnt_yellow)
        approxim = approx(cnt_yellow)
        if area > 250 and area < 10000 and len(approxim) == 4:
            x, y, w, h = cv2.boundingRect(approxim)
            ar = w/ float(h)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(img, "yellow", (x, y), font, 1, (0, 0, 0))

    for cnt_white in contours_white:
        area = cv2.contourArea(cnt_white)
        approxim = approx(cnt_white)
        if area > 250 and area < 10000 and len(approxim) == 4:
            x, y, w, h = cv2.boundingRect(approxim)
            ar = w / float(h)
            if ar >=0.95 and ar <=1.05:
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
                cv2.putText(img, "white", (x, y), font, 1, (255, 255, 255))

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
bleu = cv2.imread('./images/face_bleu.JPG')
blanc = cv2.imread('./images/face_blanche.JPG')
rouge = cv2.imread('./images/face_rouge.JPG')
orange = cv2.imread('./images/face_orange.JPG')
jaune = cv2.imread('./images/face_jaune.JPG')
vert = cv2.imread('./images/face_verte.JPG')
# détection des couleurs
bleu = resize(bleu)
blanc = resize(blanc) 
rouge = resize(rouge)
orange = resize(orange)
jaune = resize(jaune)
vert = resize(vert)
detect_face(bleu)
detect_face(blanc)
detect_face(rouge)
detect_face(orange)
detect_face(jaune)
detect_face(vert)
# Affichage de l'image
cv2.imshow("bleu", bleu)
cv2.imshow("blanc", blanc)
cv2.imshow("rouge", rouge)
cv2.imshow("orange", orange)
cv2.imshow("jaune", jaune)
cv2.imshow("vert", vert)

print(color)
# arret du programme
cv2.waitKey(0)
cv2.destroyAllWindows()
