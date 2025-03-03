from enum import IntEnum
import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

class PoseLandmark(IntEnum):
    """
    @brief Enumeration of pose landmarks detected by MoveNet
    @details Maps each body landmark to its corresponding index in the detection model
    """
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

class PoseDetector:
    """
    @brief A class for detecting and tracking human poses using MoveNet
    @details Uses TensorFlow's MoveNet model for efficient pose estimation
    """
    def __init__(self, model_name="movenet_thunder"):
        """
        @brief Initialize the PoseDetector with MoveNet
        @param model_name Name of the MoveNet model to use (lightning or thunder)
        """
        if model_name == "movenet_thunder":
            model_url = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
        else:
            model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
            
        self.model = hub.load(model_url)
        self.movenet = self.model.signatures['serving_default']
        self.landmarks = None

    def get_landmarks(self, img):
        """
        @brief Detect pose landmarks in the given image
        @param img Input image in BGR format
        @return Detected pose landmarks
        """
        img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 256, 256)
        input_image = tf.cast(img, dtype=tf.int32)
        
        results = self.movenet(input_image)
        self.landmarks = results['output_0'].numpy()[0]
        return self.landmarks

    def draw(self, img, drawPose=True):
        """
        @brief Draw detected pose landmarks on the image
        @param img Input image to draw on
        @param drawPose Boolean flag to enable/disable drawing
        """
        if self.landmarks is None:
            self.landmarks = self.get_landmarks(img)

        if drawPose and self.landmarks is not None:
            h, w, _ = img.shape
            for landmark in self.landmarks:
                x = int(landmark[1][0] * w)
                y = int(landmark[0][0] * h)
                confidence = landmark[2][0]
                #if confidence > 0.3:  # Confidence threshold
                cv2.circle(img, (x, y), 3, (0, 255, 0), cv2.FILLED)

    def get_position(self, img, landmark_idx, drawPose=True):
        """
        @brief Get the position of a specific landmark
        @param img Input image
        @param landmark_idx Index of the desired landmark
        @param drawPose Boolean flag to enable/disable drawing
        @return Position and confidence of the specified landmark
        """
        if self.landmarks is None:
            self.landmarks = self.get_landmarks(img)
            
        landmark = self.landmarks[landmark_idx]
        if landmark[2] > 0.3:  # Confidence threshold
            h, w, _ = img.shape
            cx, cy = int(landmark[1] * w), int(landmark[0] * h)
            if drawPose:
                cv2.circle(img, (cx, cy), 10, (0, 0, 255), -1)
            return (cx, cy, landmark[2])
        return None

def main():
    """
    @brief Main function for testing the PoseDetector class
    """
    cap = cv2.VideoCapture('PoseVideos/deadlift.mp4')
    detector = PoseDetector()
    print("Press 'q' to exit")
    while True:
        success, img = cap.read()
        if not success:
            print("End of video")
            break

        detector.get_landmarks(img)
        detector.draw(img)
        # position = detector.get_position(img, PoseLandmark.LEFT_SHOULDER)
        # if position:
        #     print(f"Left shoulder position: {position}")

        cv2.imshow('Image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
