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

def noWeightInpainting(txtFolder, output_folder, curFolder, curFrame):

    tfName = txtFolder + '/' + str(curFolder) + '/alignment/target.png'
    tmName = txtFolder + '/' + str(curFolder) + '/alignment_mask/target.png'
    afNames = sorted(glob.glob(txtFolder + '/' + str(curFolder) + '/alignment/0*.png'))
    amNames = sorted(glob.glob(txtFolder + '/' + str(curFolder) + '/alignment_mask/0*.png'))

    target_frame = cv2.imread(tfName)
    target_mask = cv2.imread(tmName)
    target_frame = (target_mask > 128) * target_frame
    target_mask = (target_mask < 128) * 255

    total_weight = np.zeros(target_frame.shape, dtype = 'int64')
    total_visible = np.zeros(target_frame.shape, dtype = 'int64')

    for afName, amName in zip(afNames, amNames):
        frame = cv2.imread(afName)
        mask = cv2.imread(amName)
        visible_mask = (mask > 128) * (target_mask > 128) * 1
        visible_frame = visible_mask * frame
        total_weight += visible_mask
        total_visible += visible_frame

    total_visible = np.nan_to_num(total_visible / total_weight)
    total_visible += target_frame
    total_visible = total_visible.astype('uint8')

    total_weight = (total_weight > 0) * 255
    new_mask = total_weight + (target_mask < 128) * 255
    new_mask = np.remainder(new_mask, 256).astype('uint8')

    num = "%05d" % curFolder
    cv2.imwrite(output_folder + str(num) + '.png', total_visible)
    cv2.imwrite(output_folder + 'masks/' + str(num) + '.png', new_mask)

    
    return total_visible, new_mask