import cv2
import FaceDetectionModule as fdm
from time import time


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = fdm.FaceDetector()
    while True:
        success, img = cap.read()
        img, bboxs = detector.findFaces(img)
        cTime = time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.imshow("Face Detection", img)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
