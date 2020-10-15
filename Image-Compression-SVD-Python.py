# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 13:46:03 2020

@author: elviswang
"""

import os
import matplotlib.pyplot as plt
import math
import numpy as np
import cv2
import tensorflow as tf

# img in GBR order same as OpenCV
def showPlt(img, title=""):
    im2 = img
    if im2.ndim == 3:
        im2 = im2[:,:,::-1]  # 轉 RGB Order
    plt.figure(figsize=(9, 6))
    plt.imshow(im2.astype(np.uint8), cmap ='gray');
    # plt.imshow(im2);
    plt.title(title)
    plt.show()    

# Reconstruct single channel, with k sigmas
def reconstructChannel(U, S, V, k):
    reconstChannel = np.matrix(U[:, :k]) * np.diag(S[:k]) * np.matrix(V[:k, :])
    return reconstChannel

# Reconstruct Image, with k sigmas
def reconstructImage(shape, U, S, V, k):

    reconstimg = np.zeros(shape)
    
    reconstimg = reconstructChannel(U, S, V, k)

    # 處理爆掉，value < 0 or value > 255
    # for j in range(3):
    #     ab = reconstimg[:,:,j] < 0
    #     ab = ab.astype(int) * -1
    #     ab[ab == 0] = 1
    #     reconstimg[:,:,j] = reconstimg[:,:,j] * ab
    # reconstimg[reconstimg > 255] = 255
    
    # print(aU[0][:,:k].shape, aU[0][:,:k].size, aV[0][:k,:].shape, aV[0][:k,:].size)
    # compression_size = aU[0][:,:k].size + aV[0][:k,:].size + k
    
    return reconstimg

def twoNorm(img1, img2):
    diff = img1 - img2
    print("norm.shape:", diff.shape)
    return np.linalg.norm(diff, 2)

def psnr(img1, img2):
    # mse = np.mean( (img1/1. - img2/1.) ** 2 )
    mse = np.mean( (img1/255. - img2/255.) ** 2 )
    if mse < 1.0e-10:
       return 100
    PIXEL_MAX = 1
    return 10 * math.log10(PIXEL_MAX / math.sqrt(mse))

def calculate_psnr(img1, img2, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))

def psnrTensorflow(sr, hr):
    y = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        y = sess.run(tf.image.psnr(hr, sr, max_val=255))
        print(y)
    return y

def main(img_file):
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    # print(img)
    img = img.astype(np.float64)
    print(img.dtype)
    print(img.shape, img.size)
    showPlt(img)

    U, sigma, V = np.linalg.svd(img)

    # 驗證 U, V 是否是 othogonal, 與自己的轉置相乘 = 單位矩陣
    # print(sigma)
    # othogonal = np.matrix(U * U.transpose())
    # print(othogonal)
    # othogonal = np.matrix(V * V.transpose())
    # print(othogonal)

    for i in range(1, 4):
        reconstimg = reconstructImage((img.shape[0], img.shape[1]), U, sigma, V, i)
        title = "k = %s" % i
        showPlt(reconstimg, title)
#         print('2 Norm:', twoNorm(img, reconstimg), 'sigma(k):', aS[0][i-1], aS[0][i], aS[0][i+1])
        print('2 Norm:', twoNorm(img, reconstimg), 'sigma(k):', sigma[i])
        print('PSNR:', psnr(img, reconstimg))
        print('PSNR 2:', calculate_psnr(img, reconstimg))
        # print('PSNR T:', psnrTensorflow(img, reconstimg))
        # print('compressed size: %d, original size: %d, Compression Ratio: %.2f' % (reconstSize, img.size, reconstSize/img.size*100))

        


print(os.getcwd())
os.chdir("C:\\Users\\elviswang\\workspaces_python_mac\\nchu\\2上_數據分析數學")

# main('queen1.jpg')
# main('queen2.jpg')
# main('tunghai_church.jpg')
# main('tunghai_church_2.jpg')

main('./boat.tif')

