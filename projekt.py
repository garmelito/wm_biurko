import cv2                                      #obrobka obrazu
import math                                     #wspolczynnik Minkowskiej
import numpy                                    #kernel
from PIL import Image, ImageDraw, ImageFont     #tekst na obrazie


def przedmiot(liczba) :
    switcher = {
        0 : "cienkopis",
        1 : "zakreślacz",
        2 : "marker",
        3 : "dlugopis",
        4 : "cykriel",
        5 : "lupa",
        6 : "nozyczki",
        7 : "linijka",
        8 : "katomierz",
        9 : "zszywacz",
        10 : "ekierka",
        11 : "gumka",
        12 : "spinacz",
        13 : "spinacz",
        14 : "spinacz biurowy",
        15 : "spinacz biurowy",
        16 : "spinacz biurowy",
        17 : "spinacz biurowy"
    }
    return switcher.get(liczba)

#wczytanie, zmniejszenie i zamiana przestrzeni bard obrazu biurka na HSV
table = cv2.imread("Photos/DSC_0344.jpg")
table = cv2.pyrDown(table)
# cv2.imshow("original", table)
table_HSV = cv2.cvtColor(table, cv2.COLOR_BGR2HSV)

#maskowanie biurka po kolorze
#maskowanie pikseli o pierwszym kolorze biurka
lower1 = numpy.array([0, 0, 0])
upper1 = numpy.array([46, 219, 169])
maska1 = cv2.inRange(table_HSV, lower1, upper1)
#maskowanie pikseli o drugim drugi kolorze biurka
lower2 = numpy.array([46, 0, 132])
upper2 = numpy.array([176, 80, 234])
maska2 = cv2.inRange(table_HSV, lower2, upper2)
#maska zawierajaca stol
maska_or = cv2.bitwise_or(maska1, maska2)
#maska zawierajaca przedmioty
maska_not = cv2.bitwise_not(maska_or)
table_masked = cv2.bitwise_and(table, table, mask=maska_not)
# cv2.imshow("masked", table_masked)

#operacje morfologiczne
kernel = numpy.ones((3,3), numpy.uint8)
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
maska_not = cv2.morphologyEx(maska_not, cv2.MORPH_CLOSE, kernel1, iterations=2)
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
maska_not = cv2.morphologyEx(maska_not, cv2.MORPH_OPEN, kernel2, iterations=1)
table_masked = cv2.bitwise_and(table, table, mask=maska_not)
# cv2.imshow("opened", table_masked)

#kontury
contours, _ = cv2.findContours(maska_not, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
realContours = [0]
realContours.clear()
for i in range(len(contours)) :
    area = cv2.contourArea(contours[i])
    # perimeter = cv2.arcLength(contours[i], True)
    if area > 800 :
        realContours.append(contours[i])
cv2.drawContours(table, realContours, -1, (255,0,0), 1)
# cv2.imshow("contours", table)
cv2.imwrite("table_contours.jpg", table)

#parametry, ktore chce obliczyc i zapisach dla kazdego szablonu
areas = [0]
areas.clear()
perimeters = [0]
perimeters.clear()
MMs = [0]
MMs.clear()

#wczytywanie kolejno wszystkich szablonow
for i in range(346,364) :
    path = "Photos/DSC_0" + str(i) + ".jpg"
    template = cv2.imread(path)             # 3920 x 2204 px
    template = cv2.pyrDown(template)        # 1960 x 1102 px
    #wyciecie kartki
    template = template[250:750, 600:1280]  # 740 x 520 px

    #zamiana na obraz monochromatyczny i progowanie
    template_masked = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    _, template_masked = cv2.threshold(template_masked, 125, 255, cv2.THRESH_BINARY_INV)
    #operacje morfologiczne
    template_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    template_masked = cv2.morphologyEx(template_masked, cv2.MORPH_CLOSE, template_kernel, iterations=2)
    template_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    template_masked = cv2.morphologyEx(template_masked, cv2.MORPH_OPEN, template_kernel, iterations=1)
    # cv2.imshow("template of " + path, template_masked)
    # cv2.imwrite("templatesOf" + path, template_masked)

    #znalezienie dla kazdego szablonu tylko najwiekszego konturu
    template_contour, _ = cv2.findContours(template_masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    maxArea = 0
    winner = 0
    for j in range(len(template_contour)) :
        if cv2.contourArea(template_contour[j]) > maxArea :
            maxArea = cv2.contourArea(template_contour[j])
            winner = j

    #obliczenie parametrow dla najwiekszego konturu i przypisanie do list
    template_area = cv2.contourArea(template_contour[winner])
    areas.append(template_area)
    template_perimeter = cv2.arcLength(template_contour[winner], True)
    perimeters.append(template_perimeter)
    MMs.append(template_perimeter / (2 * math.sqrt(math.pi * template_area)) - 1)

    #rysowanie najwiekszego konturu
    cv2.drawContours(template, template_contour, winner, (0,255,0), 2)
    # cv2.imshow("template of " + path, template)
    # cv2.imwrite("contoursOf" + path, template)

#dla kazdego konturu na obrazie biurka bede znajdowal najmniej nieprawdopodobny przedmiot i srodek konturu
mostCertainIndexes = [0]
mostCertainIndexes.clear()
centers = [(0, 0)]
centers.clear()
#dla kazdego konturu na obrazie biurka znajduje parametry
for i in range(len(realContours)) :
    area = cv2.contourArea(realContours[i])
    perimeter = cv2.arcLength(realContours[i], True)
    M = perimeter / (2 * math.sqrt(math.pi * area)) - 1

    #dla kazdego konturu na obrazie biurka znajduje nieprawdopodobienstwo kazdego szablonu-przedmiotu (im wieksze tym wiekszy blad -> mniejsze prawdopodobienstwo ze to ten)
    uncertainty = [0]
    uncertainty.clear()
    for j in range(0,18) :
        uncertainty.append(abs(area - areas[j]) / areas[j] + abs(perimeter - perimeters[j]) / perimeters[j] + abs(M - MMs[j]) / MMs[j])

    #znajduje ten przedmiot ktorego nieprawdopodobienstwo jest najmniejsze
    mostCertain = uncertainty[0]
    mostCertainIndexes.append(0)
    for j in range(1,18) :
        if uncertainty[j] < mostCertain :
            mostCertain = uncertainty[j]
            mostCertainIndexes[i] = j

    #znajduje srodek badanego konturu na biurku
    mu = cv2.moments(realContours[i], False)
    x = int(mu["m10"] / mu["m00"])
    y = int(mu["m01"] / mu["m00"])
    centers.append((x,y))

#Do pisania po obrazie wykorzystałem bibliotek Pillow, nie cv2, dlatego musze otworzyc zapisane zdjecie
image = Image.open('table_contours.jpg')
font_type = ImageFont.truetype("arial.ttf", 28)
draw = ImageDraw.Draw(image)
#dla kazdego konturu wyswietlam na nim nazwe przedmiotu mu przeypisanego
for i in range(len(realContours)) :
    text = przedmiot(mostCertainIndexes[i])
    draw.text(xy=centers[i], text=text, fill=(0, 0, 255), font=font_type)

image.show("")
image.save("labeledContours3.jpg")

#przechodzac przez wszystkie kontury zliczam ilosc przedmiotow kazdego typu
count = numpy.zeros((18,1), numpy.uint8)
for i in range(len(realContours)) :
    count[mostCertainIndexes[i]] = count[mostCertainIndexes[i]] + 1

#wyswietlam ilosc przedmiotow kazdego typu
print("Ilosc elementow na biurku: ")
for j in range(0,18) :
    print(przedmiot(j), count[j])


cv2.waitKey(0)
cv2.destroyAllWindows()