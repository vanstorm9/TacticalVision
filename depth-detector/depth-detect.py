import numpy as np
from sklearn.preprocessing import normalize
import cv2
from time import time 

print('loading images...')

capL = cv2.VideoCapture(0)
capR = cv2.VideoCapture(2)

# SGBM Parameters -----------------
window_size = 5                     # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely


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
 
# FILTER Parameters
lmbda = 80000
sigma = 1.2
visual_multiplier = 1.0
 
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

i = 0
while(True):
    begin = time()
    retL,imgL = capL.read()
    retR,imgR = capR.read()

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

    print(filteredImg.shape)

    cv2.imshow('Disparity Map', filteredImg)
    cv2.waitKey(1)

    i+= 1
    print('Time: ', time()-begin)
cv2.destroyAllWindows()
