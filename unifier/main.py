# USAGE
# python main.py

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import math
import cv2


skipFrameNum = 20

# Parameters for grid conversion
subSecNum = 5

# Parameters for object detection
prototxtPath = 'MobileNetSSD_deploy.prototxt.txt'
modelPath = 'MobileNetSSD_deploy.caffemodel'

# Parameters for depth map
window_size = 5  
lmbda = 80000
sigma = 1.2
visual_multiplier = 1.0
 

# Initalizations
grid = [[False for i in range(subSecNum)] for j in range(subSecNum)]



def printGrid(grid):
    for row in grid:
        print(row)
    return

def convertNormToXPos(frame,currDimPer,numOfSec=5):
    # Converts normalized coords to X pos
    startX,startY,endX,endY = currDimPer
    startXSec = int(round(numOfSec*(startX)))
    #startYSec = round(numOfSec*(startY))
    endXSec = int(round(numOfSec*(endX)))
    #endYSec = round(numOfSec*(endY))

    return (startXSec, endXSec)

def normalizeCoord(frame,currDim):
    # Convert bounding box coords to normalized bounding boxCoord
    # Returns values between 0 ... 1
    startX,startY,endX,endY = currDim
    frameX = frame.shape[1]
    frameY = frame.shape[0]

    startXPer = startX/frameX
    startYPer = startY/frameY
    endXPer = endX/frameX
    endYPer = endY/frameY

    #print((startX/frameX , startY/frameY), '   ', (endX/frameX, endY/frameY))
    #print((5*(startX/frameX) , 5*(startY/frameY)), '   ', (5*(endX/frameX), 5*(endY/frameY)))
    return (startXPer, startYPer, endXPer, endYPer)


def convertCoordToEntireGrid(grid, frameNormal,frameDepth,coord):
    # Input full coordinates
    # Returns a matrix of True and False values

    # Determine row posttion (grid) for each object
    # Working with depth map
    startX,startY,endX,endY = coord

    # We need to split frames based on coordinates
    crop_depthimg = frameDepth[startY:endY, startX:endX]

    if crop_depthimg.shape[0] <= 0 or crop_depthimg.shape[1] <= 0:
        print('Nothing detected')
        return [[False for i in range(subSecNum)] for j in range(subSecNum)]
    cv2.imshow('Cropped depthmap',crop_depthimg)
    cv2.waitKey(1)


    meanVal = np.mean(crop_depthimg)

    # We get the Y grid pos of each object
    yGridPos = math.floor(5*(meanVal/255.0)+0.001)

    # We get the X grid pos of each object        
    normalizedCoord = normalizeCoord(frameL,coord)
    startXSec,endXSec = convertNormToXPos(frameL,normalizedCoord,5)

    for xGridPos in range(startXSec,endXSec):
        grid[yGridPos][xGridPos] = True    

    return grid

def initalizeCatcher():
    left_matcher = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities= 32,             # max_disp has to be dividable by 16 f. E. HH 192, 256
    blockSize=10,
    P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=0,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)
 
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)


    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    return left_matcher, right_matcher, wls_filter

def getDepthMap(imgL,imgR,wls_filter):
    # Maybe comment this out will work
    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)


    #imgL = cv2.resize(imgL,(100,100))
    #imgR = cv2.resize(imgR,(100,100))

    imgL = cv2.blur(imgL,(7,7))
    imgR = cv2.blur(imgR,(7,7))

     
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!
     
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)

    return filteredImg



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.2,
        help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxtPath, modelPath)

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter

################################################################################
vsL = VideoStream(src=0).start()
vsR = VideoStream(src=2).start()
################################################################################



time.sleep(2.0)
fps = FPS().start()

# loop over the frames from the video stream
loopCnt = 0
while True:
        grid = [[False for i in range(subSecNum)] for j in range(subSecNum)]
        begin = time.time()
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frameL = vsL.read()
        frameR = vsR.read()

        if frameL is None or frameR is None:
                print('Skip')
                continue


        frameL = imutils.resize(frameL, width=400)
        frameR = imutils.resize(frameR, width=400)

        # grab the frame dimensions and convert it to a blob
        (h, w) = frameL.shape[:2]
        blob = cv2.dnn.blobFromImage(frameL, 0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()


        left_matcher, right_matcher, wls_filter = initalizeCatcher()

        # Get depth
        frameDepth = getDepthMap(frameL,frameR,wls_filter)



        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence > args["confidence"]:
                        # extract the index of the class label from the
                        # `detections`, then compute the (x, y)-coordinates of
                        # the bounding box for the object
                        idx = int(detections[0, 0, i, 1])
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        print('-------------------------------------------')

                        # Get the grid
                        if loopCnt%skipFrameNum == 0:
                            grid = convertCoordToEntireGrid(grid, frameL,frameDepth,(startX,startY,endX,endY))




                        cv2.imshow('depth map',frameDepth)
                        cv2.waitKey(1)


                        # draw the prediction on the frame
                        label = "{}: {:.2f}%".format(CLASSES[idx],
                                confidence * 100)
                        cv2.rectangle(frameL, (startX, startY), (endX, endY),
                                COLORS[idx], 2)
                        y = startY - 15 if startY - 15 > 15 else startY + 15
                        cv2.putText(frameL, label, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        printGrid(grid)
        # show the output frame
        cv2.imshow("Frame", frameL)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
                break

        # update the FPS counter
        fps.update()
        print('Time: ', (time.time()-begin))
        loopCnt+= 0
# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
