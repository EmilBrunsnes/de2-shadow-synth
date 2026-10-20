import numpy as np
from scipy.spatial.distance import cdist
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from synth import Synth

synth = Synth()

model_path = 'hand_landmarker.task'

FRAME_SIZE = (500,500)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    #print('hand landmarker result: {}'.format(result))
    pass

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands = 2
)

detector = vision.HandLandmarker.create_from_options(options)

videoFeed = cv2.VideoCapture(0)
frame_timestamp = 0

hand_locations = {
    'wrist': 0,
    'thumb_cmc': 1,
    'thumb_mcp': 2,
    'thumb_ip': 3,
    'thumb_tip': 4,
    'index_finger_mcp': 5,
    'index_finger_pip': 6,
    'index_finger_dip': 7,
    'index_finger_tip': 8,
    'middle_finger_mcp': 9,
    'middle_finger_pip': 10,
    'middle_finger_dip': 11,
    'middle_finger_tip': 12,
    'ring_finger_mcp': 13,
    'ring_finger_pip': 14,
    'ring_finger_dip': 15,
    'ring_finger_tip': 16,
    'pinky_mcp': 17,
    'pinky_pip': 18,
    'pinky_dip': 19,
    'pinky_tip': 20
}

class Hand:
    def __init__(self, hand_landmarks):
        self.hand_landmarks = hand_landmarks
        self.isInFrame = True
        self.update(self.hand_landmarks)
        
    def update(self, hand_landmarks):
        self.hand_landmarks = hand_landmarks
        self.points = np.array([[l.x, l.y, l.z] for l in hand_landmarks]).reshape(-1, 3)
        self.distance_matrix = cdist(self.points, self.points, metric = 'euclidean')
    
    def getDistance(self, landmark_1, landmark_2):
        global hand_locations
        return self.distance_matrix[hand_locations[landmark_1], hand_locations[landmark_2]] 
    
    def getSurfaceArea(self):
        if self.points is None or len(self.points) == 0:
            return 0.0
        
        npPoints = self.points[:, :2].astype(np.float32) #cast to np float 32 so openCV is happy and remove z coordinates
        hull = cv2.convexHull(npPoints)
        self.hand_area = cv2.contourArea(hull)
        return self.hand_area
        
    def getSpikiness(self):
        self.finger_distances = [
            self.getDistance('thumb_tip', 'index_finger_tip'),
            self.getDistance('index_finger_tip', 'middle_finger_tip'),
            self.getDistance('middle_finger_tip', 'ring_finger_tip'),
            self.getDistance('ring_finger_tip', 'pinky_tip')        
        ]
        return sum(self.finger_distances)/len(self.finger_distances) #return average of distances
        
    def getValues(self):
        return (self.getSurfaceArea(), self.getSpikiness())
           
while videoFeed.isOpened():
    success, frame = videoFeed.read()
    
    frame = cv2.flip(frame, 1) #mirror and change color 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data = rgb_frame)
    
    frame_timestamp += 33
    result = detector.detect_for_video(mp_image, frame_timestamp)
    
    if result.hand_landmarks:
        for hand_index, hand_landmarks in enumerate(result.hand_landmarks):
            for landmark in hand_landmarks: #draw on hand landmarks
                h, w, _ = frame.shape
                x, y = int(landmark.x * w), int (landmark.y * h)
                cv2.circle(frame, (x,y), 5, (0, 255, 0), -1)
            hand = Hand(hand_landmarks)
            
            vals = hand.getValues()
            fx1 = np.interp(float(vals[0]), [0.03, 0.4], [0,1])
            fx2 = np.interp(float(vals[1]), [0.03, 0.3], [0,1])
            synth.update(fx1, fx2)
            print(fx1, fx2)
              
                
    cv2.imshow('Hand track', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

videoFeed.release()
cv2.destroyAllWindows()
            