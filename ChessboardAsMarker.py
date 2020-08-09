import numpy as np
import cv2 as cv
import glob

# Nacteni kalibracnich parametru kamery
cam_matrix = np.load('cam_matrix.npy')
calib_coeffs = np.load('calib_coeffs.npy')

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objectPoints = np.zeros((7*7, 3), np.float32)
objectPoints[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)

# Zadani osy pro vytvoreni krychle
axis = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0], [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])

# Funkce tvořici obrys krychle s modrou a cervenou uhloprickou na horni strane
def draw(img1, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    img1 = cv.drawContours(img, [imgpts[:4]], -1, (255, 0, 127), 2)
    img1 = cv.line(img, tuple(imgpts[4]), tuple(imgpts[6]), (0, 0, 255), 2)
    img1 = cv.line(img, tuple(imgpts[5]), tuple(imgpts[7]), (255, 0, 0), 2)
    for i, j in zip(range(4), range(4, 8)):
        img1 = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 127), 2)
        img1 = cv.drawContours(img, [imgpts[4:]], -1, (255, 0, 127), 2)
    return img1


for frame in glob.glob(r'imagesChess/*.png'):
    img = cv.imread(frame)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (7, 7), None)

    # Nalezeni rohu sachovnice a vykreslení kdychle do obrazu
    if ret:
        cornersNew = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        _, rvecs, tvecs, inliers = cv.solvePnPRansac(objectPoints, cornersNew, cam_matrix, calib_coeffs)
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, cam_matrix, calib_coeffs)

        img2 = draw(img, cornersNew, imgpts)

        # Zobrazeni vysledku a jeho ulozeni
        cv.imshow('img', img2)
        k = cv.waitKey(0) & 0xff
        if k == 's':
            cv.imwrite(frame[:6]+'.png', img)

cv.destroyAllWindows()


