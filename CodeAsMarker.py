import cv2 as cv
import numpy as np
import time as time

# Ziskavani bodu z aruco codu
def get_points(ids1, ids12, corners1, corners12):
    dest1 = []
    origin1 = []
    for i in range(len(ids12)):
        list_index = np.where(ids1 == ids12[i])[0]
        if len(list_index) > 0:
            index = list_index[0]
            dest1.append(corners12[i])
            origin1.append(corners1[index])

    shape = (int(np.prod(np.array(dest1).shape) / 2), 2)

    dest1 = np.array(dest1).reshape(shape).astype(int)
    origin1 = np.array(origin1).reshape(shape).astype(int)
    return origin1, dest1


# Funkce pro sjednocení obrazu: aruco kodu z videa a urceneho obrazku pro vlození
def merge_images(warped_img1, frame1):
    mix = warped_img1
    mix = np.where(mix == 0, 1, mix)
    mix = np.where(mix != 1, 0, mix)
    frame1 *= mix
    frame1 += warped_img
    np.clip(frame1, 0, 255, out=frame1)
    return frame1


# Otevreni webkamery
cap = cv.VideoCapture(0)
start = time.time()
fin = time.time()
pom = 20
ppom = True

# Nacteni obrazku
im = cv.imread('quokka.jpg')
img = cv.imread('code.png')
pic = cv.resize(im, (img.shape[1], img.shape[0]))

# Nastaveni aruco kodu
code_dict = cv.aruco.DICT_6X6_1000
retv = cv.aruco.getPredefinedDictionary(code_dict)
corners, ids, _ = cv.aruco.detectMarkers(img, retv)

# Snimanií videa a ziskani parametru pro zobrazovani
while ppom:
    ret, frame = cap.read()
    corners2, ids2, _ = cv.aruco.detectMarkers(frame, retv)
    if len(corners2) > 0:
        origin, dest = get_points(ids, ids2, corners, corners2)
        homography = cv.findHomography(origin, dest)[0]
        warped_img = cv.warpPerspective(pic, homography, (frame.shape[1], frame.shape[0]))
        frame = merge_images(warped_img, frame)
    # Zobrazení videa s vloženým obrázkem
    cv.namedWindow('frame', cv.WINDOW_NORMAL)
    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    stop = time.time()

    # Ukladani snimku kazde dve vteriny
    if (stop - start) > 2.0:
        filename = str(pom) + 'img.png'
        cv.imwrite(filename, frame)
        start = time.time()
        pom = pom + 1

    # Ukonceni algoritmu po uplynuti jedne minuty
    currTime = time.time() - fin
    if currTime > 60.0:
        ppom = False

cap.release()
cv.destroyAllWindows()
