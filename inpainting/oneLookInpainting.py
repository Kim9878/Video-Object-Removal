import os
import cv2
import numpy as np
import glob
import math
from bisect import bisect_left
import torch
import torch.nn.functional as F
import tkinter as tk
from tkinter import filedialog
from torch.utils import data

def oneLookInpainting(txtFolder, output_folder, curFolder, curFrame):

    tfName = txtFolder + '/' + str(curFolder) + '/alignment/target.png'
    tmName = txtFolder + '/' + str(curFolder) + '/alignment_mask/target.png'
    afNames = sorted(glob.glob(txtFolder + '/' + str(curFolder) + '/alignment/0*.png'))
    amNames = sorted(glob.glob(txtFolder + '/' + str(curFolder) + '/alignment_mask/0*.png'))

    target_frame = cv2.imread(tfName)
    target_mask = cv2.imread(tmName)
    target_frame = (target_mask > 128) * target_frame
    target_mask = (target_mask < 128) * 255

    total_visible = np.zeros(target_frame.shape, dtype = 'int64')
    
    curFrame = "%05d" % curFrame
    curIndex = bisect_left(afNames, tfName.replace('target', curFrame))
    l_index = curIndex -1
    r_index = curIndex

    while 1:
        if l_index < 0 and r_index >= len(afNames):
            break
        if l_index >= 0:
            afName = afNames[l_index]
            amName = amNames[l_index]
            frame = cv2.imread(afName)
            mask = cv2.imread(amName)
            visible_mask = (mask > 128) * (target_mask > 128) * 1
            visible_frame = visible_mask * frame
            total_visible += visible_frame
            target_mask -= (visible_mask * 255)
            l_index -= 1
        if r_index < len(afNames):
            afName = afNames[r_index]
            amName = amNames[r_index]
            frame = cv2.imread(afName)
            mask = cv2.imread(amName)
            visible_mask = (mask > 128) * (target_mask > 128) * 1
            visible_frame = visible_mask * frame
            total_visible += visible_frame
            target_mask -= (visible_mask * 255)
            r_index += 1

    total_visible += target_frame
    total_visible = total_visible.astype('uint8')
    target_mask = (target_mask < 128) * 255

    num = "%05d" % curFolder
    cv2.imwrite(output_folder + str(num) + '.png', total_visible)
    cv2.imwrite(output_folder + 'masks/' + str(num) + '.png', target_mask)
    
    return total_visible, target_mask