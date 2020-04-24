import os
import cv2
import sys
sys.path.append("..")
import glob
import numpy as np
import tkinter as tk
import siammask
from tools import *
from tkinter import filedialog



if __name__ == "__main__":

    root = tk.Tk()
    root.withdraw()
    args = parseArgse()

    videoName = args.videoPath.rsplit('/', 1)[1].replace('.mp4', '')
    bboxInfo_file = args.videoPath.replace('.mp4', '.txt')
    base_folder = args.videoPath.rsplit('/', 1)[0]
    framesFolder = base_folder + '/frames/'
    masksFolder = base_folder + '/masks/'

    if not os.path.isdir(masksFolder):
        os.mkdir(masksFolder)
    print('Start to generate masks...')
    framesInfo = readMask.generateMasks(bboxInfo_file, masksFolder)
    print('Generate masks finished.')
    
    # SiamMask (https://github.com/foolwood/SiamMask)
    print('Start to process SiamMask task...')
    siammask.mask(args, framesInfo)
    print('SiamMask task finished.')

    with open("../tmp.txt", 'w') as f:
        f.write(base_folder + ' ' + videoName)

    
    

        
