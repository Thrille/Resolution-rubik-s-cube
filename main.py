import cv2
import numpy as np


def nothing():
    pass


video = cv2.VideoCapture(0)

cv2.namedWindow("Frame")
cv2.createTrackbar("L-H", "Frame", 0, 180, nothing)
cv2.createTrackbar("L-S", "Frame", 100, 255, nothing)
cv2.createTrackbar("L-V", "Frame", 100, 255, nothing)
cv2.createTrackbar("U-H", "Frame", 180, 180, nothing)
cv2.createTrackbar("U-S", "Frame", 255, 255, nothing)
cv2.createTrackbar("U-V", "Frame", 255, 255, nothing)

font = cv2.FONT_HERSHEY_COMPLEX

# ok ~
lower_red = np.array([140, 100, 100])
upper_red = np.array([180, 255, 255])

# ok
lower_blue = np.array([101, 100, 100])
upper_blue = np.array([150, 255, 255])

# ok
lower_green = np.array([46, 100, 100])
upper_green = np.array([100, 255, 255])

# ok
lower_orange = np.array([0, 100, 100])
upper_orange = np.array([15, 255, 255])

# ok
lower_yellow = np.array([16, 100, 100])
upper_yellow = np.array([60, 255, 255])

lower_white = np.array([0, 0, 0])
upper_white = np.array([0, 0, 255])

while True:
    _, frame = video.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L-H", "Frame")
    l_s = cv2.getTrackbarPos("L-S", "Frame")
    l_v = cv2.getTrackbarPos("L-V", "Frame")
    u_h = cv2.getTrackbarPos("U-H", "Frame")
    u_s = cv2.getTrackbarPos("U-S", "Frame")
    u_v = cv2.getTrackbarPos("U-V", "Frame")
    # lower_red = np.array([l_h,l_s,l_v])
    # upper_red = np.array([u_h,u_s,u_v])

    # mask_red = cv2.inRange(hsv, lower_red, upper_red)
    # mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    # mask_green = cv2.inRange(hsv, lower_green, upper_green)
    # mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    # mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # mask_white = cv2.inRange(hsv, lower_white, upper_white)
    # kernel = np.ones((7, 7), np.uint8)
    # mask_red = cv2.erode(mask_red, kernel)
    # mask_blue = cv2.erode(mask_blue, kernel)
    # mask_green = cv2.erode(mask_green, kernel)
    # mask_orange = cv2.erode(mask_orange, kernel)
    # mask_yellow = cv2.erode(mask_yellow, kernel)
    # mask_white = cv2.erode(mask_white, kernel)
    # # contour
    # contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours_orange, _ = cv2.findContours(mask_orange, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours_white, _ = cv2.findContours(mask_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # for cnt_red in contours_red:
    #     area = cv2.contourArea(cnt_red)

    #     if area > 250:
    #         x, y, w, h = cv2.boundingRect(cnt_red)
    #         frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #         cv2.putText(frame, "rouge", (x, y), font, 1, (255, 255, 255))

    # for cnt_blue in contours_blue:
    #     area = cv2.contourArea(cnt_blue)

    #     if area > 250:
    #         x, y, w, h = cv2.boundingRect(cnt_blue)
    #         frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #         cv2.putText(frame, "bleu", (x, y), font, 1, (255, 255, 255))

    # for cnt_green in contours_green:
    #     area = cv2.contourArea(cnt_green)

    #     if area > 250:
    #         x, y, w, h = cv2.boundingRect(cnt_green)
    #         frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #         cv2.putText(frame, "vert", (x, y), font, 1, (255, 255, 255))

    # for cnt_orange in contours_orange:
    #     area = cv2.contourArea(cnt_orange)

    #     if area > 250:
    #         x, y, w, h = cv2.boundingRect(cnt_orange)
    #         frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
    #         cv2.putText(frame, "orange", (x, y), font, 1, (255, 255, 255))

    # for cnt_yellow in contours_yellow:
    #     area = cv2.contourArea(cnt_yellow)

    #     if area > 250:
    #         x, y, w, h = cv2.boundingRect(cnt_yellow)
    #         frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
    #         cv2.putText(frame, "yellow", (x, y), font, 1, (255, 255, 255))

    # for cnt_white in contours_white:
    #     area = cv2.contourArea(cnt_white)

    #     if area > 250:
    #         x, y, w, h = cv2.boundingRect(cnt_white)
    #         frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
    #         cv2.putText(frame, "white", (x, y), font, 1, (255, 255, 255))


    ## Test detection carré dev solo
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #floutté l'image
    blurred = cv2.GaussianBlur(gray, (5,5),0)
    #contour grisé
    canny = cv2.Canny(blurred, 20,40)
    #dilatation
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(canny, None)
    #contour a partir du dilaté
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    imcontour = cv2.drawContours(frame.copy(), contours, -1, (0,0,255), 1)

    imapprox = frame.copy()
    # approximation des contours
    for cnt in contours :        
        epsilon = cv2.arcLength(cnt, True)                
        approx = cv2.approxPolyDP(cnt, 0.1 * epsilon, True)
        (x,y,w,h) = cv2.boundingRect(approx)
        area = cv2.contourArea(approx)
        ar = w/float(h)
        if len(approx) == 4 and area > 200 and ar>=0.90 and ar<=1.10:
            # imapprox = cv2.drawContours(imapprox, [approx], -1, (0,255,0), 1)
            imapprox = cv2.rectangle(imapprox, (x, y), (x + w, y + h), (255, 255, 255), 2)


    cv2.imshow('exemple', frame)
    cv2.imshow('gris', gray)
    cv2.imshow('blur', blurred)
    cv2.imshow('canny', canny)
    cv2.imshow('dilated', dilated)
    cv2.imshow('contour', imcontour)
    cv2.imshow('approx', imapprox)
    # cv2.imshow("video", frame)
    # cv2.imshow("mask", mask_red)
    # cv2.imshow("mask_blue", mask_blue)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
