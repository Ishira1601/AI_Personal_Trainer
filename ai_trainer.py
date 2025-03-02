import cv2
import numpy as np
from enum import IntEnum
from pose_detector import PoseDetector, PoseLandmark

class Exercise(IntEnum):
    DEADLIFT = 0
    SQUAT = 1
    BENCH_PRESS = 2
    OVERHEAD_PRESS = 3
    BICEPS_CURL = 4
    TRICEPS_EXTENSION = 5
    LATERAL_RAISE = 6

class AI_Trainer:
    def __init__(self, exercise):
        self.detector = PoseDetector()
        self.exercise = exercise
        self.rep_complete = False
        self.count = 0

    def rep_counter(self):
        match self.exercise:
            case Exercise.DEADLIFT:
                self.deadlift(img)
            case default:
                print("Exercise not found")

    def deadlift(self, img):
        self.detector.get_landmarks(img)
        self.detector.draw(img, False)

        length = self.detector.get_length(img, PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_WRIST)*100
        if length == PoseLandmark.INVALID_POSE_LANDMARK:
            length = self.detector.get_length(img, PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_WRIST)*100

        if length < 5 and not self.rep_complete:
            self.rep_complete = True
            self.count += 1
        if length > 15 and self.rep_complete:
            self.rep_complete = False

        cv2.rectangle(img, (0, 0), (100, 100), (255, 0, 0), -1)
        cv2.putText(img, str(self.count), (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    def bench_press(self, img):
        self.detector.get_landmarks(img)
        self.detector.draw(img, False)


cap = cv2.VideoCapture('PoseVideos/bench_press.mp4')
ai_trainer = AI_Trainer(Exercise.DEADLIFT)

while True:
    success, img = cap.read()

    if not success:
        print("End of video")
        break

    ai_trainer.rep_counter()

    cv2.imshow('Image', img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
