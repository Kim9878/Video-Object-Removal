import os
import sys
sys.path.append("..")
import cv2
import numpy as np
import glob
import math
import tkinter as tk
from tqdm import tqdm, trange
from tkinter import filedialog
from tools import *
from alignment import alignment
from noWeightInpainting import noWeightInpainting
from oneLookInpainting import oneLookInpainting
from gaussianInpainting import gaussianInpainting

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    args = parseArgse()

    with open("../tmp.txt", 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            videoFolder = line.split(' ')[0]
            videoName = line.split(' ')[1]
            break

    framePath = videoFolder + '/frames'
    maskPath = videoFolder + '/masks/siamMasks'
    txtFolder = videoFolder + '/alignedFrame'

    imgNames = sorted(glob.glob(framePath + '/*.jpg'))
    maskNames = sorted(glob.glob(maskPath + '/*.png'))

    imgArray = []
    maskArray = []
    for imgName, maskName in zip(imgNames, maskNames):
        frame = cv2.imread(imgName)
        mask = cv2.imread(maskName)
        mask = (mask < 128) * 255
        imgArray.append(frame)
        maskArray.append(mask)
    
    alignedFolderNum = len(os.listdir(txtFolder))

    # save inpainting frames result
    outputInpaintPath = txtFolder + '/inpainting/'
    if not os.path.exists(outputInpaintPath):
        os.makedirs(outputInpaintPath)
    else:
        alignedFolderNum -= 1

    for i in trange(alignedFolderNum):

        # alignment
        curFrame = alignment(txtFolder, i, imgArray, maskArray)

        # inpainting
        if args.mode == 'N':
            newFrame, newMask = noWeightInpainting(txtFolder, outputInpaintPath, i, curFrame)
        elif args.mode == 'O':
            newFrame, newMask = oneLookInpainting(txtFolder, outputInpaintPath, i, curFrame)
        elif args.mode == 'G':
            newFrame, newMask = gaussianInpainting(txtFolder, outputInpaintPath, i, curFrame)
        else:
            print("Undefined inpainting mode! Use default inpainting mode.")
            newFrame, newMask = noWeightInpainting(txtFolder, outputInpaintPath, i, curFrame)
        imgArray[curFrame] = newFrame
        maskArray[curFrame] = newMask

    # generate video
    result_path = videoFolder + '/result/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    out = cv2.VideoWriter(result_path + videoName + '.mp4', fourcc, 30, (1280, 720))
    
    for i in range(len(imgArray)):
        imgArray[i] = cv2.resize(imgArray[i], (1280, 720), interpolation=cv2.INTER_CUBIC)
        out.write(imgArray[i])
    out.release()