import cv2
import numpy as np
import math

class boundingBox:
    def __init__(self): #(x1, y1):left upper point, (x2, y2): right lower point
        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0
        self.center = 0
        self.width = 0
        self.height = 0  

    def setBoundingBox(self, data):
        self.x1 = int(float(data[0]))
        self.y1 = int(float(data[1]))
        self.x2 = int(float(data[2]))
        self.y2 = int(float(data[3]))
        self.center = (int((float(data[0]) + float(data[2])) / 2.0), int((float(data[1]) + float(data[3])) / 2.0))
        self.width = self.x2 - self.x1
        self.height = self.y2 - self.y1

    def drawBoundingBox(self, canvas):
        white = (255, 255, 255)
        cv2.rectangle(canvas, (self.x1, self.y1), (self.x2, self.y2), white, -1) #-1 means solid

    def scaling(self, sa, sb):   #sa: width scaling, sb height scaling
        self.width = int(sa * self.width)
        self.height = int(sb * self.height)
        self.x1 = int(self.center[0] - self.width / 2)
        self.y1 = int(self.center[1] - self.height / 2)
        self.x2 = int(self.center[0] + self.width / 2)
        self.y2 = int(self.center[1] + self.height / 2)

class boxesInfo:
    def __init__(self):
        self.boxesNum = 0
        self.boxes = []
    
    def pushBack(self, _box):
        box = boxesInfo.boxInfo(_box)
        self.boxes.append(box)
        self.boxesNum += 1

    class boxInfo:
        def __init__(self, _box):
            ratioX = 1
            ratioY = 1

            self.x = int(_box.x1 * ratioX)
            self.y = int(_box.y1 * ratioY)
            self.w = int(_box.width * ratioX)
            self.h = int(_box.height * ratioY)
        
        def getValue(self):
            return[self.x, self.y, self.w, self.h]

def generateMasks(bboxInfo_file, masksFolder): 
    
    masks = []
    framesInfo = []
    framesNum = -1
    width = -1
    height = -1

    with open(bboxInfo_file, "r") as f: 
        for (idx, fLine) in enumerate(f):
            data = fLine.split()
            if idx == 0:
                framesNum = int(data[0])
                width = int(data[1])
                height = int(data[2])
                for i in range(framesNum):
                    background = np.zeros((height, width, 3), np.uint8)
                    frame = boxesInfo()
                    masks.append(background)
                    framesInfo.append(frame)
            elif int(data[0]) > framesNum: # current frame is out of our choosing video range
                break
            else:
                obj = boundingBox()
                obj.setBoundingBox(data[3:7])
                obj.scaling(1.1, 1.1)
                obj.drawBoundingBox(masks[int(data[0])-1])
                framesInfo[int(data[0])-1].pushBack(obj)

    for i in range(framesNum):
        idx = "%05d" % i
        cv2.imwrite(masksFolder + str(idx) + '.png', masks[i])

    return framesInfo
    