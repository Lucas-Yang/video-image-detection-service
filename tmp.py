import cv2
from PIL import Image
import os
import numpy as np
import time


# 差异哈希算法
def dHash(image, leng=9, wid=8):
    image = np.array(image.resize((leng, wid), Image.ANTIALIAS).convert('L'), 'f')
    hash = []
    # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(wid):
        for j in range(wid):
            if image[i, j] > image[i, j + 1]:
                hash.append(1)
            else:
                hash.append(0)
    return hash


def aHash(image, leng=8, wid=8):
    image = np.array(image.resize((leng, wid), Image.ANTIALIAS).convert('L'), 'f')
    # 计算均值
    avreage = np.mean(image)
    hash = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] >= avreage:
                hash.append(1)
            else:
                hash.append(0)
    return hash


def pHash(image, leng=32, wid=32):
    image = np.array(image.resize((leng, wid), Image.ANTIALIAS).convert('L'), 'f')
    A = []
    for i in range(0, 32):
        for j in range(0, 32):
            if i == 0:
                a = np.sqrt(1 / 32)
            else:
                a = np.sqrt(2 / 32)
            A.append(a * np.cos(np.pi * (2 * j + 1) * i / (2 * 32)))
    dct = np.dot(np.dot(image, np.reshape(A, (32, 32))), np.transpose(image))
    b = dct[0:8][0:8]
    hash = []
    avreage = np.mean(b)
    for i in range(8):
        for j in range(8):
            if b[i, j] >= avreage:
                hash.append(1)
            else:
                hash.append(0)
    return hash


# 计算汉明距离
def Hamming_distance(hash1, hash2):
    num = 0
    for index in range(len(hash1)):
        if hash1[index] != hash2[index]:
            num += 1
    return num


if __name__ == "__main__":

    image1 = Image.open('1-src.png')
    # print(type(cv2.imread('2-src.png')))
    image2 = Image.open('1-target.png')

    start1 = time.time()
    d_dist = Hamming_distance(dHash(image1), dHash(image2))
    end1 = time.time()

    start2 = time.time()
    p_dist = Hamming_distance(pHash(image1), pHash(image2))
    end2 = time.time()

    start3 = time.time()
    a_dist = Hamming_distance(aHash(image1), aHash(image2))
    end3 = time.time()

    print('a_dist is ' + '%d' % a_dist + ', similarity is ' + '%f' % (1 - a_dist * 1.0 / 64) + ', time is ' + '%f' % (
                end3 - start3))
    print('p_dist is ' + '%d' % p_dist + ', similarity is ' + '%f' % (1 - p_dist * 1.0 / 64) + ', time is ' + '%f' % (
                end2 - start2))
    print('d_dist is ' + '%d' % d_dist + ', similarity is ' + '%f' % (1 - d_dist * 1.0 / 64) + ', time is ' + '%f' % (
                end1 - start1))