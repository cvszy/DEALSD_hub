import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import bdcn

from pathlib import Path
import time
import scipy.io as sio
from tqdm import tqdm

import glob

import LineDetector.x64.Debug.LineDetector as LineDetetcor



def preprocess(image):
    image = np.asarray(image, dtype=np.float_)
    mean = np.array([104.00699, 116.66877, 122.67892])
    image -= mean
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image).float()
    return image[None]


def run(im_path, model,TH1=0.02,TH2=0.16):
    heatTh=TH1    #The preset threshold of heatmap
    anchorTh=TH2   #The preset threshold for anchor point selection
    ###heatmap
    image = Image.open(im_path).convert("RGB")
    image = preprocess(image)
    image = image.to("cuda")
    with torch.no_grad():
        output = model(image)
    output = output[-1].sigmoid().cpu()
    output = output.numpy()
    output = output[0, 0]
    output[output <= heatTh] = 0

    ###gradient
    img = cv.imread(img_path, 0)
    ksize = (3, 3)
    sigma = 1
    img_blur = cv.GaussianBlur(img, ksize, sigma)
    dx = cv.Sobel(img_blur, cv.CV_64F, 1, 0, ksize=3)
    dy = cv.Sobel(img_blur, cv.CV_64F, 0, 1, ksize=3)
    G = np.abs(dx) + np.abs(dy)
    G[G <= 30] = 0

    maxm = np.max(G)
    normal_G = G / maxm

    ### merge
    DEA_G = np.maximum(normal_G, output)
    DEA_G = DEA_G[None, None, :]
    DEA_G = torch.from_numpy(DEA_G)
    temp = DEA_G.clone()
    nonmax = temp != F.max_pool2d(temp, kernel_size=3, stride=1, padding=1)
    temp[nonmax] = 0
    temp[temp <= anchorTh] = 0
    DEA_G[temp > 0] = 1
    DEA_G = DEA_G.numpy()
    DEA_G = DEA_G[0, 0]

    return DEA_G


if __name__ == "__main__":
    #deep edge detector
    model = bdcn.BDCN().eval()
    checkpoint = torch.load("./model.pth")
    model.load_state_dict(checkpoint)
    model = model.to("cuda")

    ###test
    img_path = 'picture/P1020824.jpg'
    edge = run(img_path, model)
    out = np.zeros((30000, 4), dtype=np.float32)
    linenum = LineDetetcor.detect(img_path, edge, out)

    out = out.astype(np.uint32)
    img = cv.imread(img_path)
    for i in range(linenum):
        cv.line(img, (out[i, 0], out[i, 1]), (out[i, 2], out[i, 3]), (0, 0, 255), 1)

    cv.imshow("test", img)
    cv.waitKey(0)
