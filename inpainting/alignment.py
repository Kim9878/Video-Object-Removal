import os
import cv2
import numpy as np
import glob
import math
import torch
import torch.nn.functional as F
import tkinter as tk
from tkinter import filedialog
from torch.utils import data

def toTensor(img):
    assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0)

def alignment(txtFolder, curFolder, imgArray, maskArray):
    outputFramePath = txtFolder + '/' + str(curFolder) + '/alignment/'
    outputMaskPath = txtFolder + '/' + str(curFolder) + '/alignment_mask/'

    if not os.path.exists(outputFramePath):
            os.makedirs(outputFramePath)
    if not os.path.exists(outputMaskPath):
            os.makedirs(outputMaskPath)

    # alignment
    txtFile = txtFolder + '/' + str(curFolder) + '/aligned.txt'
    curFrame = -1
    with open(txtFile, "r") as f: 
        for (idx, fLine) in enumerate(f):
            data = fLine.split()
            if idx == 0:
                cv2.imwrite(outputFramePath + 'target.png', imgArray[int(data[0])])
                cv2.imwrite(outputMaskPath + 'target.png', maskArray[int(data[0])])
                curFrame = int(data[0])
            else:
                rf = int(data[0])
                num = "%05d" % (rf)
                img = toTensor(imgArray[rf].copy())
                mask = toTensor(maskArray[rf].copy())
                theta = np.array(data[1:]).astype('float')
                py_list = theta.reshape(1, 2, 3)
                theta_rt = torch.FloatTensor(py_list)
                grid_rt = F.affine_grid(theta_rt, img.size())
                aligned_r = F.grid_sample(img, grid_rt)
                aligned_m = F.grid_sample(mask, grid_rt)
                af = aligned_r[0].permute(1, 2, 0)
                af = np.array(af)
                af *= 255
                af = cv2.UMat(np.array(af, dtype = np.uint8))
                cv2.imwrite(outputFramePath + num + '.png', af)
                am = aligned_m[0].permute(1, 2, 0)
                am = np.array(am)
                am *= 255
                am = cv2.UMat(np.array(am, dtype = np.uint8))
                cv2.imwrite(outputMaskPath + num + '.png', am)
    return curFrame

            