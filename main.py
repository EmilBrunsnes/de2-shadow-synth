import numpy as np
import cv2

background = None
hand = None

frames_elapsed = 0
FRAME_SIZE = (500, 500)

CALIBRATION_TIME = 30
BG_WEIGHT = 0.5
OBJ_THRESHOLD = 18

capture = cv2.VideoCapture(0)

class Hand:
    top = (0, 0)
    bottom = (0, 0)
    left = (0, 0)
    right = (0, 0)
    centerX = 0
    prevCenterX = 0
    isInFrame = False
    isWaving = False
    nFingers = 0
    
    def __init__(self, top, bottom, left, right, centerX):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.centerX = centerX
        self.prevCenterX = 0
        isInFrame = False # do i need tihs?
        isWaving = False
        
    def update(self, top, bottom, left, right):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

def write_on_image(frame, hand):
    text = "Searching..."
    
    if frames_elapsed < CALIBRATION_TIME:
        text = "Calibrating..."
    elif hand == None or hand.isInFrame == False:
        text = "No hand??"
    else:
        if hand.isWaving:
            text = "Waving"
        else:
            text = str(hand.nFingers)
            
    cv2.putText(frame, text, (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.4,( 0 , 0 , 0 ),2,cv2.LINE_AA)
    cv2.putText(frame, text, (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.4,(255,255,255),1,cv2.LINE_AA)
    cv2.rectangle(frame, (region_left, region_top), (region_right, region_bottom), (255,255,255), 2)
    
def get_region(frame):
    region = frame[region_top:region_bottom, region_left:region_right]
    region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    region = cv2.GaussianBlur(region, (5,5), 0)
    return region

def get_average(region):
    global background
    if background is None:
        background = region.copy().astype("float")
        return
    else:
        cv2.accumulateWeighted(region, background, BG_WEIGHT)

def segment(region):
    global Hand
    diff = cv2.absdiff(background.astype(np.uint8), region)
    
    thresholded_region = cv2.threshold(diff, OBJ_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    (contours, _) = cv2.findContours(thresholded_region.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        if Hand is not None:
            Hand.isInFrame = False
        return
    else:
        if Hand is not None:
            Hand.isInFrame = True
        segmented_region = max(contours, key = cv2.contourArea)     
        return (thresholded_region, segmented_region)
    
    
while True:
    region_top = 0
    region_bottom = int(2 * FRAME_SIZE[1] / 3)
    region_left = int(FRAME_SIZE[1] / 2)
    region_right = FRAME_SIZE[0]
    
    ret, frame = capture.read()
    frame = cv2.resize(frame, FRAME_SIZE)
    frame = cv2.flip(frame, 1)
    region = get_region(frame) #apply greyscale and gaussian blur
    
    if frames_elapsed < 30:
        get_average(region)
    else:
        region_pair = segment(region)
        if region_pair is not None:
            (thresholded_region, segmented_region) = region_pair
            blank_region = np.zeros_like(region)
            cv2.drawContours(blank_region, [segmented_region], -1, (255, 255, 255), thickness = cv2.FILLED)
            cv2.imshow("Segmented Image", blank_region)
        
    write_on_image(frame, hand)
    cv2.imshow("Camera", frame)
    frames_elapsed += 1
    if (cv2.waitKey(1) & 0xFF == ord('x')):
        break
    
capture.release()
cv2.destroyAllWindows()

