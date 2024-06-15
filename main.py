import FaceDetectionModule as fmm
import cv2
from time import time


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = fmm.FaceMeshDetector(maxFaces=1)
    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        cTime = time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.imshow("Face Mesh", img)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
