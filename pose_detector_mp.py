from enum import IntEnum
import cv2
import mediapipe as mp
import math



class PoseLandmark(IntEnum):
    """
    @brief Enumeration of pose landmarks detected by MediaPipe
    @details Maps each body landmark to its corresponding index in the detection model
    """
    INVALID_POSE_LANDMARK = -1
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

class PoseDetector:
    """
    @brief A class for detecting and tracking human poses in images/video
    @details Uses MediaPipe's pose detection model to identify and track body landmarks
    """
    def __init__(self, detectionCon=0.5, trackingCon=0.5):
        """
        @brief Initialize the PoseDetector
        @param capture OpenCV video capture object
        @param detectionCon Minimum detection confidence threshold
        @param trackingCon Minimum tracking confidence threshold
        """
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon
        self.landmarks = None

    def get_landmarks(self, img):
        """
        @brief Detect pose landmarks in the given image
        @param img Input image in BGR format
        @return Detected pose landmarks
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)
        self.landmarks = results.pose_landmarks
        return self.landmarks

    def draw(self, img, drawPose=True):
        """
        @brief Draw detected pose landmarks on the image
        @param img Input image to draw on
        @param drawPose Boolean flag to enable/disable drawing
        """
        if self.landmarks is None:
            self.landmarks = self.get_landmarks(img)
        if drawPose:
            if self.landmarks:
                self.mpDraw.draw_landmarks(img, self.landmarks, self.mpPose.POSE_CONNECTIONS)

    def get_position(self, img, landmark, drawPose=True):
        """
        @brief Get the position of a specific landmark
        @param img Input image
        @param landmark PoseLandmark enum value for the desired landmark
        @param drawPose Boolean flag to enable/disable drawing the landmark
        @return Position of the specified landmark
        """
        if self.landmarks is None:
            self.landmarks = self.get_landmarks(img)
        position = None
        if self.landmarks:
            position = self.landmarks.landmark[landmark]
            if drawPose:
                cv2.circle(img, (int(position.x * img.shape[1]), int(position.y * img.shape[0])), 15, (0, 0, 255), 2)
                cv2.circle(img, (int(position.x * img.shape[1]), int(position.y * img.shape[0])), 10, (0, 0, 255), cv2.FILLED)
        return position

    def get_angle(self, img, landmark1, landmark2, landmark3, drawPose=True):
        """
        @brief Calculate the angle between three landmarks
        @param img Input image
        @param landmark1 PoseLandmark enum value for the first landmark
        @param landmark2 PoseLandmark enum value for the second landmark
        @param landmark3 PoseLandmark enum value for the third landmark
        """
        if self.landmarks is None:
            self.landmarks = self.get_landmarks(img)
        position1 = self.get_position(img, landmark1, drawPose)
        position2 = self.get_position(img, landmark2, drawPose)
        position3 = self.get_position(img, landmark3, drawPose)
        if position1 is None or position2 is None or position3 is None:
            return PoseLandmark.INVALID_POSE_LANDMARK
        if drawPose:
            self.draw_line(img, position1, position2)
            self.draw_line(img, position2, position3)

        angle = math.degrees(math.atan2(position3.y - position2.y, position3.x - position2.x) - math.atan2(position1.y - position2.y, position1.x - position2.x))

        return angle

    def get_length(self, img, landmark1, landmark2, drawPose=True):
        """
        @brief Calculate the length between two landmarks
        @param img Input image
        @param landmark1 PoseLandmark enum value for the first landmark
        @param landmark2 PoseLandmark enum value for the second landmark
        """
        if self.landmarks is None:
            self.landmarks = self.get_landmarks(img)
        position1 = self.get_position(img, landmark1, drawPose)
        position2 = self.get_position(img, landmark2, drawPose)
        if position1 is None or position2 is None:
            return PoseLandmark.INVALID_POSE_LANDMARK
        if drawPose:
            self.draw_line(img, position1, position2)

        length = math.sqrt((position2.x - position1.x)**2 + (position2.y - position1.y)**2)

        return length

    def draw_line(self, img, position1, position2):

        """
        @brief Draw a line between two landmarks
        @param img Input image
        @param position1 Position of the first landmark
        @param position2 Position of the second landmark
        """
        cv2.line(img, (int(position1.x * img.shape[1]), int(position1.y * img.shape[0])), (int(position2.x * img.shape[1]), int(position2.y * img.shape[0])), (0, 255, 0), 3)

def main():
    """
    @brief Main function for testing the PoseDetector class
    """
    cap = cv2.VideoCapture('PoseVideos/bench_press.mp4')
    detector = PoseDetector()

    while True:
        success, img = cap.read()

        if not success:
            print("End of video")
            break
        detector.get_landmarks(img)
        detector.draw(img)
        print(detector.get_position(img, PoseLandmark.LEFT_ELBOW))

        cv2.imshow('Image', img)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()