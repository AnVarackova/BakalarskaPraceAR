import cv2
import os.path
import glob
import numpy as np
import time
import matplotlib.pyplot as plt


#  Hledani vyznamnych bobu pomoci FAST angoritmu
def featureDetection(img):
    fast_treshold = 20
    nonmaxSuppression = True
    fast = cv2.FastFeatureDetector_create(fast_treshold, nonmaxSuppression)
    kp = fast.detect(img, None)
    kp2 = np.array([x.pt for x in kp], dtype=np.float32)
    return kp2


# Funkce pro trackovani vyznamnych bodu
def featureTracking(img1, img2, points1):
    winSize = (21, 21)
    err = []
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

    # Funkce pro pocitani optickeho toku
    features, status, err = cv2.calcOpticalFlowPyrLK(img1, img2, points1, None, winSize, maxLevel=2, criteria=criteria,
                                                     flags=0, minEigThreshold=0.001)

    f = np.zeros([len(status[status == 1]), 1, 2], np.float32)  # prazdna pole, naplnovat, kdyz bude 1
    pf = np.zeros([len(status[status == 1]), 1, 2], np.float32)
    k = 0

    # Ulozeni puvodnich a novych hodnot
    for i in range(len(status)):
        if status[i] == 1:
            f[k, :] = features[i, :]
            pf[k, :] = points1[i, :]
            k += 1
    features = f
    points1 = pf

    return features, points1, err


# Funkce pro pocitani meritka
def getAbsoluteScale(annotations, fn):
    ss = annotations[fn-1].strip().split()
    x_prev = float(ss[3])
    y_prev = float(ss[7])
    z_prev = float(ss[11])
    ss = annotations[fn].strip().split()
    x = float(ss[3])
    y = float(ss[7])
    z = float(ss[11])
    trueX, trueY, trueZ = x, y, z
    return np.sqrt((x - x_prev) * (x - x_prev) + (y - y_prev) * (y - y_prev) + (z - z_prev) * (z - z_prev))


def main():
    min_num_feat = 1500
    # scale = 1

    ts = []  # translace
    Rs = []  # rotace
    Rf = np.identity(3, np.float)
    tf = np.zeros([3, 1], np.float)
    ts.append(tf)
    Rs.append(Rf)
    fin_tf = []

    # Nacitani obrazku a nastaveni parametru kamery
    fns = sorted(glob.glob("./image_0/*.png"))
    focal = 718.8560  # ohnisko
    pp = (607.1928, 185.2157)  # stred roviny promitani, stred obrazku
    cam_matrix = np.zeros([3, 3], dtype=np.float)  # matice kamery
    cam_matrix[0, 0] = focal
    cam_matrix[1, 1] = focal
    cam_matrix[0, 2] = pp[0]
    cam_matrix[1, 2] = pp[1]
    cam_matrix[2, 2] = 1

    img_1 = cv2.imread(fns[0], 0)
    points1 = featureDetection(img_1)

    p = len(fns)
    x = []
    y = []
    z = []

    pom = 0

    with open('00.txt') as f:
        annotations = f.readlines()

    traj = np.zeros((1000, 1000, 3), dtype=np.uint8)

    for fn in fns[1:]:
        pom = pom + 1
        img_2 = cv2.imread(fn, 0)
        points2, points1, err = featureTracking(img_1, img_2, points1)
        essMat, mask = cv2.findEssentialMat(points2, points1, focal, pp, cv2.RANSAC, 0.999, 1.0)
        retval, R, t, mask = cv2.recoverPose(essMat, points2, points1, cam_matrix)

        ts.append(t)
        Rs.append(R)

        scale = getAbsoluteScale(annotations, pom)

        if scale > 0.1 and t[2] > t[0] and t[2] > t[1]:
            tf = tf + scale*(np.dot(Rf, t))
            Rf = np.dot(R, Rf)
            fin_tf.append(tf)

        if points1.shape[0] < min_num_feat:
            fast = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
            points2 = fast.detect(img_2, None)
            points2 = np.array([x.pt for x in points2], dtype=np.float32)

        # Vykreslovani
        draw_x = int(tf[0]) + 450
        draw_y = int(-tf[2]) + 700
        cv2.circle(traj, (draw_x, draw_y), 1, (255, 255, 255), 1)
        cv2.imshow('Trajektorie', traj)
        cv2.waitKey(1)

        # Ukladani souradnic
        x.append(tf[0])
        y.append(tf[2])
        z.append(tf[1])

        if len(points1) < min_num_feat:
            points1 = featureDetection(img_1)
            points2, points1, err = featureTracking(img_1, img_2, points1)

        img_1 = img_2.copy()
        points1 = points2

        cv2.waitKey(1)
    # Vykresleni souradnic
    plt.plot(x, y, 'blueviolet')
    plt.show()

if __name__ == "__main__":
    main()
