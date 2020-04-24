# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import os
import glob
import sys
import numpy as np 
import shapely
from shapely.geometry import Polygon,MultiPoint
from get_mask.test import *
from get_mask.models.custom import Custom

class trackObject:
    def __init__(self, _box):
        self.isNew = True
        self.frame = 0
        self.x = _box[0]
        self.y = _box[1]
        self.w = _box[2]
        self.h = _box[3]
        self.x1 = _box[0]
        self.y1 = _box[1]
        self.x2 = _box[0] + _box[2]
        self.y2 = _box[1]
        self.x3 = _box[0] + _box[2]
        self.y3 = _box[1] + _box[3]
        self.x4 = _box[0]
        self.y4 = _box[1] + _box[3]
        self.color = tuple(255 * np.random.rand(3))
        self.iou = 0
    
    def setFourCorner(self, corners):
        self.x1 = corners[0][0][0][0]
        self.y1 = corners[0][0][0][1]
        self.x2 = corners[0][1][0][0]
        self.y2 = corners[0][1][0][1]
        self.x3 = corners[0][2][0][0]
        self.y3 = corners[0][2][0][1]
        self.x4 = corners[0][3][0][0]
        self.y4 = corners[0][3][0][1]

    def getFourCorner(self):
        return [self.x1, self.y1, self.x2, self.y2, self.x3, self.y3, self.x4, self.y4]
    
    def getValue(self):
        return [self.x, self.y, self.w, self.h]

def get_frames(video_name):
    if not video_name:
        print("video file not found!")
        sys.exit(0)

    elif video_name.endswith('mp4'):
        cap = cv2.VideoCapture(video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob.glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame

def calIoU(_bbox1, _bbox2):
    line1 = _bbox1.getFourCorner()
    a = np.array(line1).reshape(4, 2)
    poly1 = Polygon(a).convex_hull
    
    line2 = _bbox2.getFourCorner()
    b = np.array(line2).reshape(4, 2)
    poly2 = Polygon(b).convex_hull
    
    union_poly = np.concatenate((a,b))

    if not poly1.intersects(poly2):
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area
            union_area = MultiPoint(union_poly).convex_hull.area
            if union_area == 0:
                iou = 0

            iou = float(inter_area) / (poly1.area) 
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0

    _bbox1.iou = max(iou, _bbox1.iou)
    return iou

def mask(args, framesInfo):
    # Setup device
    args.config = 'get_mask/experiments/siammask/config_davis.json'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    np.random.seed(1)

    # Setup Model
    cfg = load_config(args)
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), '{} is not a valid file'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)

    # Parse Image file
    framesFolder = args.videoPath.rsplit('/', 1)[0] + '/frames/'
    img_files = get_frames(framesFolder)
    ims = [imf for imf in img_files]
    fgbg = cv2.createBackgroundSubtractorKNN()

    # Select ROI
    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    bboxes = []
    toc = 0
    maskedImg = []

    # read file
    for i in range(framesInfo[0].boxesNum):
        box = trackObject(framesInfo[0].boxes[i].getValue())
        bboxes.append(box)
    bboxes = sorted(bboxes, key = lambda trackObject: trackObject.x)

    tracking_folder = args.videoPath.rsplit('/', 1)[0] + '/tracking/'
    visualize_folder = args.videoPath.rsplit('/', 1)[0] + '/visualize/'
    siamMask_folder = args.videoPath.rsplit('/', 1)[0] + '/masks/siamMasks/'
    
    if not os.path.isdir(tracking_folder):
        os.mkdir(tracking_folder)

    if not os.path.isdir(visualize_folder):
        os.mkdir(visualize_folder)

    if not os.path.isdir(siamMask_folder):
        os.mkdir(siamMask_folder)

    def checkAlive(lastState, state, args):
        lastSize = lastState['target_sz'][0] * lastState['target_sz'][1]
        size = state['target_sz'][0] * state['target_sz'][1]
        lastPos = lastState['target_pos']
        pos = state['target_pos']

        if state['score'] < 0.3: # if the confidence model predicted is too low 
            return False
        if size / lastSize >= 1.5 or size / lastSize <= 0.5: # the size of bounding box changes rapidly
            return False
        if np.linalg.norm(pos-lastPos) > args.outputSize[0] * 0.05: # the center position of bounding box changes rapidly
            return False
        return True

    def detectShadow(frame):
        fgmask = fgbg.apply(frame)
        fgmask = np.uint8((fgmask > 0) * 255)
        return fgmask
    
    def findMaxConnectedComp(img):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations = 1) # closing
        ret, labels = cv2.connectedComponents(img)
        newImg = np.zeros_like(img)
        for val in np.unique(img)[1:]:
            mask = np.uint8(img == val)
            labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]
            largestLabel = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            newImg[labels == largestLabel] = val
        newImg = np.uint8(newImg)
        return newImg

    counter = 0
    for direction in range(2): # 0: forward, 1: backward
        if direction == 1:
            ims.reverse()
            bboxes = []
            for i in range(framesInfo[len(ims)-1].boxesNum):
                box = trackObject(framesInfo[len(ims)-1].boxes[i].getValue())
                bboxes.append(box)
            bboxes = sorted(bboxes, key = lambda trackObject: trackObject.x)
        states = []
        unwanted = []
        resetBoxes = False
    
        for f, im in enumerate(ims):
            tic = cv2.getTickCount()
            unwanted.clear()
            background = np.zeros((args.outputSize[1], args.outputSize[0], 3), np.uint8)
            visualize = im.copy()
            shadow = detectShadow(im.copy())

            for i in range(len(bboxes)):
                x, y, w, h = bboxes[i].getValue()
                if bboxes[i].isNew:  # init
                    target_pos = np.array([x + w / 2, y + h / 2])
                    target_sz = np.array([w, h])
                    state = siamese_init(im.copy(), target_pos, target_sz, siammask, cfg['hp'])  # init tracker
                    states.append(state)
                    cv2.rectangle(background, (x, y), (x + w, y + h), (255, 255, 255), -1)
                    top = visualize.copy()
                    cv2.rectangle(visualize, (x, y), (x + w, y + h), bboxes[i].color, -1)
                    visualize = cv2.addWeighted(visualize, 0.5, top, 0.5, 0)
                    bboxes[i].isNew = False
                    bboxes[i].frame += 1
                else:  # tracking
                    lastState = states[i].copy()
                    state = states[i]
                    state = siamese_track(state, im.copy(), mask_enable=True, refine_enable=True)  # track

                    if checkAlive(lastState.copy(), state.copy(), args) is False or (direction == 1 and bboxes[i].frame == 5):
                        unwanted.append(i)
                        if bboxes[i].frame > 1:
                            location = lastState['ploygon'].flatten()
                            mask = lastState['mask'] > lastState['p'].seg_thr
                        else:
                            location = state['ploygon'].flatten()
                            mask = state['mask'] > state['p'].seg_thr
                    else:
                        location = state['ploygon'].flatten()
                        mask = state['mask'] > state['p'].seg_thr

                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                    mask = cv2.morphologyEx(np.uint8(mask), cv2.MORPH_OPEN, kernel)
                    mask = cv2.dilate(np.uint8(mask), kernel, iterations = 1)

                    bboxes[i].frame += 1
                    bboxes[i].setFourCorner([np.int0(location).reshape((-1, 1, 2))])
                    size = [state['target_sz'][0] * 1.1 , state['target_sz'][1]]
                    pos = state['target_pos'] - [size[0]/2, size[1]/2]
                    size[1] *= 1.1
                    subShadow = shadow[int(max(0,pos[1])):int(pos[1]+size[1]), int(max(0, pos[0])):int(pos[0]+size[0])]
                    subShadow = findMaxConnectedComp(subShadow.copy())
                    subMask = mask[int(max(0,pos[1])):int(pos[1]+size[1]), int(max(0, pos[0])):int(pos[0]+size[0])]
                    subMask = (subShadow > 0) * 255 + (subShadow == 0) * subMask
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    subMask = cv2.morphologyEx(np.uint8(subMask), cv2.MORPH_OPEN, kernel, iterations = 1)
                    mask[int(max(0,pos[1])):int(pos[1]+size[1]), int(max(0, pos[0])):int(pos[0]+size[0])] = subMask
                    visualize[:, :, 0] = (mask > 0) * (bboxes[i].color[0] * 0.5 + visualize[:, :, 0] * 0.5) + (mask == 0) * visualize[:, :, 0]
                    visualize[:, :, 1] = (mask > 0) * (bboxes[i].color[1] * 0.5 + visualize[:, :, 1] * 0.5) + (mask == 0) * visualize[:, :, 1]
                    visualize[:, :, 2] = (mask > 0) * (bboxes[i].color[2] * 0.5 + visualize[:, :, 2] * 0.5) + (mask == 0) * visualize[:, :, 2]
                    background[:, :, 0] = (mask > 0) * 255 + (mask == 0) * background[:, :, 0]
                    background[:, :, 1] = (mask > 0) * 255 + (mask == 0) * background[:, :, 1]
                    background[:, :, 2] = (mask > 0) * 255 + (mask == 0) * background[:, :, 2]
                    states[i] = state

            for j in sorted(unwanted, reverse = True):
                states.pop(j)
                bboxes.pop(j)
                resetBoxes = True
            
            if f > 0 and (len(bboxes) == 0 or resetBoxes == True):
                resetBoxes = False
                if direction == 0:
                    frameIdx = f
                else:
                    frameIdx = len(ims) - f - 1
                for idx in range(framesInfo[frameIdx].boxesNum):
                    box = trackObject(framesInfo[frameIdx].boxes[idx].getValue())
                    overlap = False
                    for i in range(len(bboxes)):
                        if calIoU(box, bboxes[i]) > 0.3:
                            overlap = True
                            break
                    if overlap is False:
                        x, y, w, h = box.getValue()
                        target_pos = np.array([x + w / 2, y + h / 2])
                        target_sz = np.array([w, h])
                        state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'])  # init tracker
                        states.append(state)
                        box.isNew = False
                        box.frame += 1
                        bboxes.append(box)

            cv2.imshow('SiamMask', visualize)
            if direction == 0:
                maskedImg.append([im.copy(), background.copy(), visualize.copy(), bboxes.copy()])
            else:
                maskedImg[len(ims)-f-1][1] += background.copy()
                maskedImg[len(ims)-f-1][2] = cv2.addWeighted(maskedImg[len(ims)-f-1][2], 0.5, visualize, 0.5, 0)

            num = "%05d" % counter
            cv2.imwrite(tracking_folder + str(num) + '.png', visualize)
            counter += 1

            key = cv2.waitKey(1)
            if key > 0:
                break
            toc += cv2.getTickCount() - tic

    for i in range(len(maskedImg)):
        im = cv2.resize(maskedImg[i][2], tuple(args.outputSize))
        background = cv2.resize(maskedImg[i][1], tuple(args.outputSize))
        num = "%05d" % i
        cv2.imwrite(visualize_folder + str(num) + '.png', im)
        cv2.imwrite(siamMask_folder + str(num) + '.png', background)

    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
    cv2.destroyAllWindows()