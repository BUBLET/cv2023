import cv2
import numpy as np

class RoadImage:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.mask = None
        self.generate_road_mask()

    def generate_road_mask(self):
        blur = cv2.GaussianBlur(self.gray, (5, 5), 0)
        canny = cv2.Canny(blur, 30, 1)

        height, width = self.image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        roi_corners = np.array([[(0, height), (width / 2, height / 2), (width, height)]], dtype=np.int32)
        cv2.fillPoly(mask, roi_corners, 255)

        self.mask = cv2.bitwise_and(canny, mask)

    def show_road(self):
        road = cv2.bitwise_and(self.image, self.image, mask=self.mask)
        cv2.imshow('Road', road)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_mask(self):
        cv2.imshow('Road Mask', self.mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


road_image = RoadImage('Module_1a/img/map.jpg')
road_image.show_road()