import numpy as np
import cv2 as cv
import time

# Otevreni webkamery
cap = cv.VideoCapture(0)
# print(cap.isOpened())   -- Overeni, jestli je kamera pripojena

start = time.time()
fin = time.time()
pom = 0
ppom = True

while ppom:
    # Zaznamenavani snimek po snimku
    ret, frame = cap.read()

    # Zobrazeni snimku
    cv.imshow('frame', frame)
    if cv.waitKey(5) & 0xFF == ord('q'):
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
