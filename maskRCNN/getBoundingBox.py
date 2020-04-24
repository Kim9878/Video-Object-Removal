import os
import sys
sys.path.append("..")
import cv2
import coco
import utils
import math
import random
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from tools import *
from tkinter import filedialog
from keras.preprocessing import image
from moviepy.editor import VideoFileClip
import model as modellib
import visualize
import visualize_car_detection

if __name__ == "__main__":

    args = argument.parseArgse()

    if args.videoPath is None:
        args.videoPath = filedialog.askopenfilename(initialdir = '/home', title = "Select video file", filetypes=(('mp4 files', '*.mp4'),))
    framesFolder = args.videoPath.rsplit('/', 1)[0] + '/frames/'
    
    if not os.path.isdir(framesFolder):
        os.mkdir(framesFolder)
    
    # save bounding box information
    bboxInfo_file = args.videoPath.replace('.mp4', '.txt')
    
    # choose detect classes
    targets = args.target.rsplit('-')

    print('Start to split the video to frames...')
    framesNum, width, height = video2img.generateFrames(args.videoPath, framesFolder, args.outputSize)
    print('Generate frames finished. There are %d frames, video size is %d x %d' %(framesNum, width, height))

    ROOT_DIR = os.getcwd() # Root directory of the project

    MODEL_DIR = os.path.join(ROOT_DIR, "logs") # Directory to save logs and trained model

    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5") # Path to trained weights file

    class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    #config.display()

    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config) # Create model object in inference mode.

    model.load_weights(COCO_MODEL_PATH, by_name=True) # Load weights trained on MS-COCO

    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                'bus', 'train', 'truck', 'boat', 'traffic_light',
                'fire_hydrant', 'stop_sign', 'parking_meter', 'bench', 'bird',
                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports_ball',
                'kite', 'baseball_bat', 'baseball_glove', 'skateboard',
                'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted_plant', 'bed',
                'dining_table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell_phone', 'microwave', 'oven', 'toaster',
                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy_bear', 'hair_drier', 'toothbrush']


    # Run Object Detection      
    filelist=os.listdir(framesFolder)
    filelist.sort()

    # record framesNum, image's width & height
    fp = open(bboxInfo_file, "w")
    image = skimage.io.imread(framesFolder +  '00000.jpg')
    fp.write("{} {} {}\n".format(len(filelist), image.shape[1], image.shape[0]))

    for fichier in filelist[:]: # filelist[:] makes a copy of filelist.
        
        image = skimage.io.imread(framesFolder + fichier)
        frame_num = fichier.split(".jpg")
        results = model.detect([image], verbose=1) # Run detection
        black_img = np.zeros(shape=(image.shape)) # Create  black image for mask

        # Visualize results
        r = results[0]
        image_mask = visualize_car_detection.display_instances2(fp, int(frame_num[0])+1, targets,  black_img, r['rois'], r['masks'], r['class_ids'], 
                                class_names, r['scores'], fp, frame_num)

    fp.close()




