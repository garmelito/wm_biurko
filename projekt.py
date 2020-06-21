import cv2
import numpy
import math


# def passs(x) :
#     pass

# image = numpy.zeros((400,1000,3), numpy.uint8)
# cv2.namedWindow('image')

# cv2.createTrackbar('h', 'image', 0, 255, passs)
# cv2.createTrackbar('s', 'image', 0, 255, passs)
# cv2.createTrackbar('v', 'image', 0, 255, passs)
#
# while(True):
#     cv2.imshow('image', image)
#     key = cv2.waitKey(1)
#     if key == 27 :
#         break
#
#     h = cv2.getTrackbarPos('h', 'image')
#     s = cv2.getTrackbarPos('s', 'image')
#     v = cv2.getTrackbarPos('v', 'image')
#     image [ : ] = [h, s ,v]
#     image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)


table = cv2.imread("Photos/DSC_0344.jpg")
table = cv2.pyrDown(table)
table = cv2.pyrDown(table)
# cv2.imshow("original", table)
# table_HSV = cv2.cvtColor(table, cv2.COLOR_BGR2HSV)
#
# lower1 = numpy.array([0, 0, 0])
# upper1 = numpy.array([46, 219, 169])
# maska1 = cv2.inRange(table_HSV, lower1, upper1)
# lower2 = numpy.array([46, 0, 132])
# upper2 = numpy.array([176, 80, 234])
# maska2 = cv2.inRange(table_HSV, lower2, upper2)
# maska_or = cv2.bitwise_or(maska1, maska2)
# maska_not = cv2.bitwise_not(maska_or)
# table_masked = cv2.bitwise_and(table, table, mask=maska_not)
# # cv2.imshow("masked", table_masked)
#
# kernel = numpy.ones((3,3), numpy.uint8)
# kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
# maska_not = cv2.morphologyEx(maska_not, cv2.MORPH_CLOSE, kernel1, iterations=2)
# kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
# maska_not = cv2.morphologyEx(maska_not, cv2.MORPH_OPEN, kernel2, iterations=1)
# table_masked = cv2.bitwise_and(table, table, mask=maska_not)
# # cv2.imshow("opened", table_masked)
#
# contours, _ = cv2.findContours(maska_not, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# for i in range(len(contours)) :
#     area = cv2.contourArea(contours[i])
#     perimeter = cv2.arcLength(contours[i], True)
#     if area > 20 :
#         cv2.drawContours(table, contours, i, (255,0,0), 1)
#     cv2.imshow("contours", table)


MMs = [0]
MMs.clear()

for i in range(346,364) :
    path = "Photos/DSC_0" + str(i) + ".jpg"
    template = cv2.imread(path)             # 3920 x 2204 px
    template = cv2.pyrDown(template)        # 1960 x 1102 px
    template = template[250:750, 600:1280]  # 740 x 520 px

    template_masked = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    _, template_masked = cv2.threshold(template_masked, 125, 255, cv2.THRESH_BINARY_INV)
    template_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    template_masked = cv2.morphologyEx(template_masked, cv2.MORPH_CLOSE, template_kernel, iterations=2)
    template_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    template_masked = cv2.morphologyEx(template_masked, cv2.MORPH_OPEN, template_kernel, iterations=1)
    # cv2.imshow("template of " + path, template_masked)
    # cv2.imwrite("templatesOf" + path, template_masked)

    template_contour, _ = cv2.findContours(template_masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    maxArea = 0
    winner = 0
    for j in range(len(template_contour)) :
        if cv2.contourArea(template_contour[j]) > maxArea :
            maxArea = cv2.contourArea(template_contour[j])
            winner = j

    template_area = cv2.contourArea(template_contour[winner])
    template_perimeter = cv2.arcLength(template_contour[winner], True)
    MMs.append(template_perimeter / (2 * math.sqrt(math.pi * template_area)) - 1)

    cv2.drawContours(template, template_contour, winner, (0,255,0), 2)
    # cv2.imshow("template of " + path, template)
    # cv2.imwrite("contoursOf" + path, template)

for i in range(len(MMs)) :
    print(MMs[i])

cv2.waitKey(0)
cv2.destroyAllWindows()

