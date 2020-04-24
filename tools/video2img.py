import os
import cv2
import numpy as np
from .argument import parseArgse
import tkinter as tk
from tkinter import filedialog

def generateFrames(videoPath, framesFolder, outputSize=[1280, 720], imgType='jpg'):
    frameNum = 0
    width = -1
    height = -1

    #convert video to frames and count the total number of frames
    cap = cv2.VideoCapture(videoPath)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            if width == -1 or height == -1:
                height = frame.shape[0]
                width = frame.shape[1]

            frame = cv2.resize(frame, tuple(outputSize)) #downsampling

            idx = "%05d" % frameNum
            cv2.imwrite(framesFolder + str(idx) + '.' + imgType, frame)
            frameNum = frameNum + 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    return frameNum, width, height

if __name__ == "__main__":

    args = parseArgse()

    root = tk.Tk()
    root.withdraw()

    if args.videoPath is None:
        args.videoPath = filedialog.askopenfilename(initialdir = './', title = "Select video file", filetypes=(('mp4 files', '*.mp4'),('avi files', '*.avi')))
    # save path
    framesFolder = args.videoPath.rsplit('/', 1)[0] + '/frames/'
    if not os.path.isdir(framesFolder):
        os.mkdir(framesFolder)
    
    outputSize = [1280, 720]
    print('Start to split the video to frames...')
    framesNum, width, height = generateFrames(args.videoPath, framesFolder, outputSize)
    print('Generate frames finished. There are %d frames, video size is %d x %d' %(args.frameNum, width, height))