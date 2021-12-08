#!/usr/bin/env python3
import numpy as np
from tqdm import tqdm
from husky_assistant import *


def detect_blobs(image):
    """Laplacian blob detector.

    Args:
    - image (2D float64 array): A grayscale image.

    Returns:
    - corners (list of 2-tuples): A list of 2-tuples representing the locations
        of detected blobs. Each tuple contains the (x, y) coordinates of a
        pixel, which can be indexed by image[y, x].
    - scales (list of floats): A list of floats representing the scales of
        detected blobs. Has the same length as `corners`.
    - orientations (list of floats): A list of floats representing the dominant
        orientation of the blobs.
    """
    baseImage = baseImageGenerator(image)
    sigma_all = gaussianKernelParameters(sigma=1.6, S=3)
    octaves = octaveNums(baseImage.shape)
    husky, gaussians = huskyGenerator(baseImage, octaves, sigma_all)
    keyP = blobDetection(husky, gaussians)
    # for idx in range(len(keyP)):
    #     if idx <= 400:
    #         pos = posExtractor(idx, keyP)
    #         size = sizeExtractor(idx, keyP)
    #         cv2.circle(image, pos, size, color=(255, 0, 0))
    # im_show(image)
    return keyP, gaussians


def compute_descriptors(image, keyP):
    """Compute descriptors for corners at specified scales.

    Args:
    - image (2d float64 array): A grayscale image.
    - corners (list of 2-tuples): A list of (x, y) coordinates.
    - scales (list of floats): A list of scales corresponding to the corners.
        Must have the same length as `corners`.
    - orientations (list of floats): A list of floats representing the dominant
        orientation of the blobs.

    Returns:
    - descriptors (list of 1d array): A list of desciptors for each corner.
        Each element is an 1d array of length 128.
    """
    time1 = time()
    width = 4
    numBins = 8
    scaleRatio = 3
    maxDesVal = 0.2
    descriptors = []
    for idx, kp in enumerate(keyP):
        o = kp[1].o
        s = kp[1].s
        gaussian = image[o][s]
        r, c = gaussian.shape
        # 关键点坐标提取
        pos = (int(round(kp[0][0]) / (2 ** (o - 1))), int(round(kp[0][1] / 2 ** (o - 1))))
        # 权重因数
        weightFactor = -0.5 / ((0.5 * width) ** 2)
        # 扩展 tensor 长度为三线性插值做准备
        histogramTensor = np.zeros((width, width, numBins))
        # histogramTensor1 = np.zeros((width, width, numBins))
        # 基本采样区域 1/4 边长, 3sigma 为基本半径， sigma 需要缩放至当前 gaussian image 尺度
        histWidth = scaleRatio * kp[0][2] / 2 ** (o - 1)
        # 由于计算需要线性插值加旋转，故需要扩展图像采样点范围，https://www.cnblogs.com/JiePro/p/sift_4.html
        # 扩展个锤子
        halfWidth = int(round(histWidth * np.sqrt(2) * width * 0.5))
        # 采样范围不超过图像尺寸
        halfWidth = int(min(halfWidth, np.sqrt(r ** 2 + c ** 2) / 2))
        # 矩阵加速
        rows = np.linspace(-halfWidth, halfWidth, 2 * halfWidth + 1)
        cols = np.linspace(-halfWidth, halfWidth, 2 * halfWidth + 1)
        mesh = np.meshgrid(rows, cols)
        coordinatesRel = np.stack((mesh[0].flatten(), mesh[1].flatten()), axis=1)
        coordinatesRel = coordinatesRel.T
        # 旋转角
        for angle in kp[1].orientation:
            angle = 360 - angle
            # 旋转矩阵准备
            angleCos = np.cos(np.deg2rad(angle))
            angleSin = np.sin(np.deg2rad(angle))
            transMatrix = np.array([[angleCos, angleSin], [-angleSin, angleCos]])
            # 原相对坐标
            # 旋转后相对中心点的坐标
            coorRotate = np.dot(transMatrix, coordinatesRel)
            coorBin = coorRotate / histWidth + 0.5 * width
            # 每点的 bin 坐标
            coorBin = np.floor(coorBin).astype(int)
            # 每点绝对坐标
            coordinatesAbs = np.zeros(coordinatesRel.shape)
            coordinatesAbs[0] = coordinatesRel[0] + pos[1]  # 偶数为row
            coordinatesAbs[1] = coordinatesRel[1] + pos[0]  # 奇数为col
            # 组合
            coordinates = np.concatenate((coorRotate, coorBin, coordinatesAbs), axis=0)
            # 删除超过图像区域的点
            coordinates = np.delete(coordinates, np.where(coordinates[4, :] >= r - 1), axis=1)
            coordinates = np.delete(coordinates, np.where(coordinates[4, :] <= 0), axis=1)
            coordinates = np.delete(coordinates, np.where(coordinates[5, :] <= 0), axis=1)
            coordinates = np.delete(coordinates, np.where(coordinates[5, :] >= c - 1), axis=1)
            # 删除落在采样区 16 个 bin 之外的点
            coordinates = np.delete(coordinates, np.where(coordinates[2:4, :] >= width)[1], axis=1)
            coordinates = np.delete(coordinates, np.where(coordinates[2:4, :] < 0)[1], axis=1)
            gaussianPos = coordinates[4:6, :].astype(int)
            # 每点梯度
            dx = gaussian[gaussianPos[0], gaussianPos[1] + 1] - gaussian[gaussianPos[0], gaussianPos[1] - 1]
            dy = gaussian[gaussianPos[0] - 1, gaussianPos[1]] - gaussian[gaussianPos[0] + 1, gaussianPos[1]]
            absGradient = np.sqrt(dx * dx + dy * dy)
            gradAngle = np.rad2deg(np.arctan2(dy, dx)) % 360
            weight = np.exp(
                weightFactor * ((coordinates[0, :] / histWidth) ** 2 + (coordinates[1, :] / histWidth) ** 2))
            # 每点的orientation bin
            hisIdx = np.floor((gradAngle - angle) * numBins / 360).astype(int)
            hisIdx %= numBins
            magnitude = weight * absGradient
            for i in range(coordinates.shape[1]):
                histogramTensor[int(coordinates[2, i]), int(coordinates[3, i]), hisIdx[i]] += magnitude[i]
            # 三线性插值 https://en.wikipedia.org/wiki/Trilinear_interpolation
            # 插个锤子值
            # 展开成 128 维描述子
            descriptor_vector = histogramTensor.flatten()
            # 初次归一化，并以 0.2 为最大值进行截断
            threshold = np.linalg.norm(descriptor_vector) * maxDesVal
            descriptor_vector[descriptor_vector > threshold] = threshold
            # 二次归一化
            descriptor_vector /= max(np.linalg.norm(descriptor_vector), 1e-7)
            descriptors.append([descriptor_vector, idx])
    time2 = time()
    print('compute descriptor:', time2 - time1, 's')
    return descriptors


def match_descriptors(descriptors1, descriptors2):
    """Match descriptors based on their L2-distance and the "ratio test".

    Args:
    - descriptors1 (list of 1d arrays):
    - descriptors2 (list of 1d arrays):

    Returns:
    - matches (list of 2-tuples): A list of 2-tuples representing the matching
        indices. Each tuple contains two integer indices. For example, tuple
        (0, 42) indicates that corners1[0] is matched to corners2[42].
    """
    time1 = time()
    matches = defaultdict(int)
    for i, descriptor1 in enumerate(descriptors1):
        dist = np.zeros(len(descriptors2))
        for j, descriptor2 in enumerate(descriptors2):
            dist[j] = np.linalg.norm(descriptor1[0] - descriptor2[0])
        idxes = np.argsort(dist)
        if dist[idxes[0]] / (dist[idxes[1]] + 1e-10) < 0.8:
            matches[(descriptor1[1], descriptors2[idxes[0]][1])] = dist[idxes[0]]
    # matches = sorted(matches.keys(), key=lambda item: item[1], reverse=False)
    time2 = time()
    print('find', len(matches), 'matches')
    print('match descriptors:', time2 - time1, 's')
    return np.array(list(matches.keys()))


def draw_matches(image1, image2, keyP1, keyP2, matches,
                 outlier_labels=None):
    """Draw matched corners between images.

    Args:
    - matches (list of 2-tuples)
    - image1 (3D uint8 array): A color image having shape (H1, W1, 3).
    - image2 (3D uint8 array): A color image having shape (H2, W2, 3).
    - corners1 (list of 2-tuples)
    - corners2 (list of 2-tuples)
    - outlier_labels (list of bool)

    Returns:
    - match_image (3D uint8 array): A color image having shape
        (max(H1, H2), W1 + W2, 3).
    """
    # time1 = time()
    H1 = image1.shape[0]
    H2 = image2.shape[0]
    W1 = image1.shape[1]
    H = max(H1, H2)
    if H == H1:
        image2 = np.pad(image2, [(0, H - H2), (0, 0), (0, 0)])
    else:
        image1 = np.pad(image1, [(0, H - H1), (0, 0), (0, 0)])
    match_image = np.concatenate([image1, image2], axis=1)
    if outlier_labels is None:
        outlier_labels = np.zeros(len(matches))
    colors = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    rotate = 0
    for (a, b), i in zip(matches, outlier_labels):
        if i:
            color = (0, 0, 255)
        else:
            color = colors[rotate]
            rotate += 1
            rotate %= len(colors)
        pos1 = posExtractor(a, keyP1)
        pos2 = posExtractor(b, keyP2)
        pos2 = list(pos2)
        pos2[0] += W1
        pos2 = tuple(pos2)
        cv2.line(match_image, pos1, pos2, color=color, thickness=1)
        cv2.circle(match_image, pos1, radius=1, color=color, thickness=1)
        cv2.circle(match_image, pos2, radius=1, color=color, thickness=1)
    # im_show(match_image)
    # time2 = time()
    # print('draw matching image:', time2 - time1, 's')
    return match_image


def compute_affine_xform(keyP1, keyP2, matches):
    """Compute affine transformation given matched feature locations.

    Args:
    - corners1 (list of 2-tuples)
    - corners1 (list of 2-tuples)
    - matches (list of 2-tuples)

    Returns:
    - xform (2D float64 array): A 3x3 matrix representing the affine
        transformation that maps coordinates in image1 to the corresponding
        coordinates in image2.
    - outlier_labels (list of bool): A list of Boolean values indicating whether
        the corresponding match in `matches` is an outlier or not. For example,
        if `matches[42]` is determined as an outlier match after RANSAC, then
        `outlier_labels[42]` should have value `True`.
    """
    # time1 = time()
    if len(matches) == 0:
        return None, None, None
    iter = len(matches) * 12
    pos1 = []
    pos2 = []
    for i, j in matches:
        pos1.append(posExtractor(i, keyP1))
        pos2.append(posExtractor(j, keyP2))
    pos1 = np.array(pos1)
    pos2 = np.array(pos2)
    cat = np.ones((len(matches), 1))
    pos1 = np.concatenate((pos1, cat), axis=1)
    pos2 = np.concatenate((pos2, cat), axis=1)
    max_in = 0
    while iter:
        iter -= 1
        sample = np.random.randint(len(pos1), size=6)
        vector1 = pos1[sample]
        vector2 = pos2[sample]
        affMartix = np.linalg.lstsq(vector1, vector2, rcond=None)[0]
        dist = np.linalg.norm(pos1.dot(affMartix) - pos2, axis=1)
        var = np.var(dist)
        thres = np.sqrt(var * 12.59)
        inner_labels = dist < thres
        if inner_labels.sum() > max_in:
            outlier_labels = ~inner_labels
            max_in = inner_labels.sum()
            xform = np.linalg.lstsq(pos1[inner_labels], pos2[inner_labels], rcond=None)[0]
    if max_in == 0:
        return None, None, None
    dist = np.linalg.norm(pos1.dot(xform) - pos2, axis=1)
    idxes = np.argsort(dist)
    nums = min(len(idxes), 25)
    print('xform')
    print(xform)
    print('inliers:', max_in)
    print('outliers:', len(matches) - max_in)
    matches = matches[idxes[:nums]]
    outlier_labels = outlier_labels[idxes[:nums]]
    # time2 = time()
    # print('compute affine xform:', time2 - time1, 's')
    return xform, outlier_labels, matches


def stitch_images(image1, image2, xform):
    """Stitch two matched images given the transformation between them.

    Args:
    - image1 (3D uint8 array): A color image.
    - image2 (3D uint8 array): A color image.
    - xform (2D float64 array): A 3x3 matrix representing the transformation
        between image1 and image2. This transformation should map coordinates
        in image1 to the corresponding coordinates in image2.

    Returns:
    - image_stitched (3D uint8 array)
    """
    # time1 = time()
    affineMatrix = xform.T[:2]
    image1 = np.pad(image1, ((50, 50), (50, 50), (0, 0)))
    image2 = np.pad(image2, ((50, 50), (50, 50), (0, 0)))
    targetShape = (image2.shape[1], image2.shape[0])
    warped = cv2.warpAffine(image1, affineMatrix, targetShape)
    image_stitched = image2.astype('float32') + warped.astype('float32')
    image_stitched[(image2 > 0) & (warped > 0)] /= 2
    # time2 = time()
    # print('stitch image:', time2 - time1, 's')
    return image_stitched.astype('uint8')


def imageStitch(name, name1, img_path1, img_path2):
    dataPath = 'data/'
    matchPath = 'match_image/'
    stitchPath = 'image_stitched/'
    print('\nstitching', img_path1, 'and', img_path2)
    img1 = cv2.imread(dataPath + img_path1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(dataPath + img_path2, cv2.IMREAD_COLOR)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) / 255.0
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) / 255.0
    keyP1, gaussians1 = detect_blobs(gray1)
    descriptors1 = compute_descriptors(gaussians1, keyP1)
    keyP2, gaussians2 = detect_blobs(gray2)
    descriptors2 = compute_descriptors(gaussians2, keyP2)
    matches = match_descriptors(descriptors1, descriptors2)
    xform, outlier_labels, matches = compute_affine_xform(keyP1, keyP2, matches)
    if xform is not None:
        match_image = draw_matches(img1, img2, keyP1, keyP2, matches, outlier_labels)
        image_stitched = stitch_images(img1, img2, xform)
        cv2.imwrite(matchPath + name, match_image)
        cv2.imwrite(stitchPath + name1, image_stitched)


def main():
    imageStitch('bikes_01.png', 'bikes_11.png', 'bikes1.png', 'bikes2.png')
    imageStitch('bikes_02.png', 'bikes_22.png', 'bikes2.png', 'bikes3.png')
    imageStitch('bikes_03.png', 'bikes_33.png', 'bikes1.png', 'bikes3.png')
    imageStitch('graf_01.png', 'graf_11.png', 'graf1.png', 'graf2.png')
    imageStitch('graf_02.png', 'graf_22.png', 'graf2.png', 'graf3.png')
    imageStitch('graf_03.png', 'graf_33.png', 'graf1.png', 'graf3.png')
    imageStitch('leuven01.png', 'leuven11.png', 'leuven1.png', 'leuven2.png')
    imageStitch('leuven02.png', 'leuven22.png', 'leuven2.png', 'leuven3.png')
    imageStitch('leuven03.png', 'leuven33.png', 'leuven1.png', 'leuven3.png')
    imageStitch('wall01.png', 'wall11.png', 'wall1.png', 'wall2.png')
    imageStitch('wall02.png', 'wall22.png', 'wall2.png', 'wall3.png')
    imageStitch('wall03.png', 'wall33.png', 'wall1.png', 'wall3.png')


if __name__ == '__main__':
    main()
