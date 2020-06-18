import cv2
import numpy

def passs(x) :
    pass

table = cv2.imread("Photos/DSC_0344.jpg")
table = cv2.pyrDown(table)
table = cv2.pyrDown(table)
cv2.imshow("original", table)
table_HSV = cv2.cvtColor(table, cv2.COLOR_BGR2HSV)

image = numpy.zeros((400,1000,3), numpy.uint8)
cv2.namedWindow('image')

cv2.createTrackbar('h', 'image', 0, 255, passs)
cv2.createTrackbar('s', 'image', 0, 255, passs)
cv2.createTrackbar('v', 'image', 0, 255, passs)

while(True):
    cv2.imshow('image', image)
    key = cv2.waitKey(1)
    if key == 27 :
        break

    h = cv2.getTrackbarPos('h', 'image')
    s = cv2.getTrackbarPos('s', 'image')
    v = cv2.getTrackbarPos('v', 'image')
    image [ : ] = [h, s ,v]
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    lower1 = numpy.array([0, 0, 0])
    upper1 = numpy.array([46, 219, 169])
    maska1 = cv2.inRange(table_HSV, lower1, upper1)
    lower2 = numpy.array([46, 0, 132])
    upper2 = numpy.array([176, 80, 234])
    maska2 = cv2.inRange(table_HSV, lower2, upper2)
    maska_or = cv2.bitwise_or(maska1, maska2)
    maska_not = cv2.bitwise_not(maska_or)
    table_masked = cv2.bitwise_and(table, table, mask=maska_not)
    cv2.imshow("masked", table_masked)
    cv2.imwrite("table_masked.jpg", table_masked)

    kernel = numpy.ones((3,3), numpy.uint8)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    maska_not = cv2.morphologyEx(maska_not, cv2.MORPH_CLOSE, kernel1, iterations=2)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    maska_not = cv2.morphologyEx(maska_not, cv2.MORPH_OPEN, kernel2, iterations=1)
    table_masked = cv2.bitwise_and(table, table, mask=maska_not)
    cv2.imshow("opened", table_masked)
    cv2.imwrite("table_morphed.jpg", table_masked)



# cv2.imwrite("table_morphed.jpg", table_masked)
cv2.waitKey(0)
cv2.destroyAllWindows()

