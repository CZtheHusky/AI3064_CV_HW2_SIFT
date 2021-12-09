import numpy as np
import cv2
from collections import defaultdict
from scipy.ndimage.filters import *
from time import time


def posExtractor(idx, keyP):
    return (int(round(keyP[idx][0][0])), int(round(keyP[idx][0][1])))


def sizeExtractor(idx, keyP):
    return int(round(keyP[idx][0][2]))


def im_show(image):
    cv2.imshow('husky', image)
    cv2.waitKey()
    cv2.destroyAllWindows()


class keyPoints:
    def __init__(self):
        self.value = -1
        self.orientation = []
        self.o = -1
        self.s = -1


def gaussianKernelParameters(sigma=1.6, S=3):
    numsInOctave = S + 3
    sigmaAll = np.zeros(numsInOctave)
    k = 2 ** (1. / S)
    sigmaAll[0] = sigma
    for s in range(1, numsInOctave):
        sigmaPrev = k ** (s - 1) * sigma
        sigmaTot = k * sigmaPrev
        sigmaAll[s] = np.sqrt(sigmaTot ** 2 - sigmaPrev ** 2)
    return sigmaAll


def octaveNums(imShape):
    minShape = min(imShape)
    octaves = np.log(minShape) / np.log(2) - 1
    return int(round(octaves))


def baseImageGenerator(image, sigma=1.6, sigmaIn=0):
    image = cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    sigma_ = np.sqrt(max((sigma ** 2) - ((2 * sigmaIn) ** 2), 0))
    if sigma_ == 0:
        return image
    else:
        return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma_, sigmaY=sigma_)


def huskyGenerator(image, octaves, sigmaAll):
    imageBase = image
    husky = []
    gaussians = []
    for o in range(octaves):
        huskyOctave = [imageBase]
        for s in range(1, len(sigmaAll)):
            imageBase = cv2.GaussianBlur(imageBase, (0, 0), sigmaX=sigmaAll[s], sigmaY=sigmaAll[s])
            huskyOctave.append(imageBase)
            # im_show(imageBase)
        dogs = []
        for i in range(len(huskyOctave) - 1):
            dog = cv2.subtract(huskyOctave[i + 1], huskyOctave[i])
            dogs.append(dog)
        husky.append(dogs)
        gaussians.append(huskyOctave)
        baseNextOctave = huskyOctave[3]
        imageBase = cv2.resize(baseNextOctave, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    return husky, gaussians


def gradientCenter(array):
    dx = 0.5 * (array[1, 1, 2] - array[1, 1, 0])
    dy = 0.5 * (array[1, 2, 1] - array[1, 0, 1])
    dsigma = 0.5 * (array[2, 1, 1] - array[0, 1, 1])
    return [dx, dy, dsigma]


def hessianMatrix(array):
    center = array[1, 1, 1]
    dxx = array[1, 1, 2] - 2 * center + array[1, 1, 0]
    dyy = array[1, 2, 1] - 2 * center + array[1, 0, 1]
    dsigmas = array[2, 1, 1] - 2 * center + array[0, 1, 1]
    dxy = 0.25 * (array[1, 2, 2] - array[1, 2, 0] - array[1, 0, 2] + array[1, 0, 0])
    dxs = 0.25 * (array[2, 1, 2] - array[2, 1, 0] - array[0, 1, 2] + array[0, 1, 0])
    dys = 0.25 * (array[2, 2, 1] - array[2, 0, 1] - array[0, 2, 1] + array[0, 0, 1])
    return np.array([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dsigmas]])


def extremaLocating(j, i, s, o, S, dogs, sigma, cThres, borderWidth, r=10, iteration=5):
    imageShape = dogs[0].shape
    convergence = False
    # 极值点定位
    for iters in range(iteration):
        if not convergence:
            firstLayer = dogs[s - 1]
            secondLayer = dogs[s]
            thirdLayer = dogs[s + 1]
            extremaSpace = np.stack([firstLayer[j - 1:j + 2, i - 1:i + 2], secondLayer[j - 1:j + 2, i - 1:i + 2],
                                     thirdLayer[j - 1:j + 2, i - 1:i + 2]]).astype(float)
            gradient = gradientCenter(extremaSpace)
            hessian = hessianMatrix(extremaSpace)
            extremaDrift = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            if abs(extremaDrift).max() >= 0.5:
                start = [i, j, s]
                # print('from: ', i, j, s)
                j += int(round(extremaDrift[1]))
                i += int(round(extremaDrift[0]))
                s += int(round(extremaDrift[2]))
                # print('to: ', i, j, s)
                if j < borderWidth or j >= imageShape[0] - borderWidth or i < borderWidth or i >= imageShape[
                    1] - borderWidth or s < 1 or s > S:
                    return None
                else:
                    dogs[s][j][i] = dogs[start[2]][start[1]][start[0]] + 0.5 * np.matmul(gradient, extremaDrift)
            else:
                convergence = True
                break
    if not convergence:
        return None
    new_center = extremaSpace[1, 1, 1]
    # 去除边缘效应
    if abs(new_center) >= cThres / S:
        hessianXY = hessian[:2, :2]
        det = np.linalg.det(hessianXY)
        trace = np.trace(hessianXY)
        if trace ** 2 / det < (r + 1) ** 2 / r:
            octave = o
            coordinate = (round(i * (2 ** (o - 1))), round(j * (2 ** (o - 1))), sigma * (2 ** (s / S)) * (2 ** o))
            value = abs(new_center)
            return coordinate, value, octave
    return None


def orientationCompute(coordinate, o, gaussian, thres=0.8, nbFactor=1.5, r=3, numBins=36):
    orientations = []
    imShape = gaussian.shape
    size = coordinate[2]
    cornerY = coordinate[1]
    cornerX = coordinate[0]
    scale = nbFactor * size / (2 ** o)
    weightFactor = -0.5 / (scale ** 2)
    nbRadius = int(round(r * scale))
    origHis = np.zeros(numBins)
    smHis = np.zeros(numBins)
    rows = np.linspace(-nbRadius, nbRadius, 2 * nbRadius + 1)
    cols = np.linspace(-nbRadius, nbRadius, 2 * nbRadius + 1)
    mesh = np.meshgrid(rows, cols)
    coordinatesRel = np.stack((mesh[0].flatten(), mesh[1].flatten()), axis=1)
    coordinatesRel = coordinatesRel.T
    coordinateAbs = np.zeros(coordinatesRel.shape)
    coordinateAbs[1, :] = coordinatesRel[1, :] + int(round(cornerX / (2 ** (o - 1))))  # 奇数为col
    coordinateAbs[0, :] = coordinatesRel[0, :] + int(round(cornerY / (2 ** (o - 1))))  # 偶数为row
    coordinates = np.concatenate((coordinateAbs, coordinatesRel), axis=0)
    coordinates = np.delete(coordinates, np.where(coordinates[0, :] >= imShape[0] - 1), axis=1)
    coordinates = np.delete(coordinates, np.where(coordinates[0, :] <= 0), axis=1)
    coordinates = np.delete(coordinates, np.where(coordinates[1, :] <= 0), axis=1)  # 删除超过图像区域的点
    coordinates = np.delete(coordinates, np.where(coordinates[1, :] >= imShape[1] - 1), axis=1)
    gaussianPos = coordinates[0:2, :].astype(int)
    dx = gaussian[gaussianPos[0], gaussianPos[1] + 1] - gaussian[gaussianPos[0], gaussianPos[1] - 1]
    dy = gaussian[gaussianPos[0] - 1, gaussianPos[1]] - gaussian[gaussianPos[0] + 1, gaussianPos[1]]
    absGradient = np.sqrt(dx * dx + dy * dy)
    gradAngle = np.rad2deg(np.arctan2(dy, dx))
    weight = np.exp(weightFactor * (coordinates[2] ** 2 + coordinates[3] ** 2))
    hisIdx = np.round(gradAngle * numBins / 360).astype(int)
    hisIdx %= numBins
    magnet = weight * absGradient
    for i in range(len(hisIdx)):
        origHis[hisIdx[i]] += magnet[i]
    # histogram 平滑
    for n in range(numBins):
        smHis[n] = (6 * origHis[n] + 4 * (
                origHis[n - 1] + origHis[(n + 1) % numBins]) + origHis[n - 2] + origHis[(n + 2) % numBins]) / 16.
    angMax = max(smHis)
    angPeaks = np.where(np.logical_and(smHis > np.roll(smHis, 1), smHis > np.roll(smHis, -1)))[0]
    for idx in angPeaks:
        peak = smHis[idx]
        if peak >= thres * angMax:  # 保留大于全局最强 80% 的分量方向
            lVal = smHis[(idx - 1) % numBins]
            rVal = smHis[(idx + 1) % numBins]
            interpolation = (idx + 0.5 * (lVal - rVal) / (lVal - 2 * peak + rVal)) % numBins
            angle = 360 - interpolation * 360 / numBins
            if abs(angle - 360) < 1e-7:
                angle = 0
            orientations.append(angle)
    return orientations


def blobDetection(husky, gaussians, S=3, sigma=1.6, cThres=0.05):
    time1 = time()
    borderWidth = int(round(min(gaussians[0][0].shape) / 50))
    corners = defaultdict(keyPoints)
    thres = 0.5 * cThres / S
    counter = 0
    for o, dogs in enumerate(husky):
        dogs = np.array(dogs)
        footprint = np.ones((3, 3, 3))
        maximun = maximum_filter(dogs, footprint=footprint)
        minimun = minimum_filter(dogs, footprint=footprint)
        maximun = maximun[1:-1, :, :]
        minimun = minimun[1:-1, :, :]
        maxPos = np.where(dogs[1:-1, :, :] == maximun)
        minPos = np.where(dogs[1:-1, :, :] == minimun)
        maxPos = np.array(maxPos)
        minPos = np.array(minPos)
        extremaPos = np.concatenate((maxPos, minPos), axis=1)
        for i in range(extremaPos.shape[1]):
            s = extremaPos[0][i] + 1
            j = extremaPos[1][i]
            i = extremaPos[2][i]
            if dogs[s, j, i] > thres and borderWidth < j < dogs[s].shape[0] - borderWidth and borderWidth < i < \
                    dogs[s].shape[1] - borderWidth:
                pScale = sigma * (2 ** (s / S)) * (2 ** o)
                pos = (i * (2 ** (o - 1)), j * (2 ** (o - 1)), pScale)
                if pos not in corners:
                    extremaValidate = extremaLocating(j, i, s, o, S, dogs, sigma, cThres, borderWidth, r=10,
                                                      iteration=5)
                    if extremaValidate is not None:
                        coordinate, value, octave = extremaValidate
                        orientation = orientationCompute(coordinate, o, gaussians[o][s])
                        corners[coordinate].value = value
                        corners[coordinate].orientation = orientation
                        corners[coordinate].o = octave
                        corners[coordinate].s = s
                        counter += 1
    husky = sorted(corners.items(), key=lambda item: item[1].value, reverse=True)
    time2 = time()
    # print('find', counter, 'keypoints')
    # print('find keypoint:', time2 - time1, 's')
    return husky
