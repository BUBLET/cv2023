import cv2
import numpy as np
import yaml

class RoadGrid:
    def __init__(self, camera_params_path, step):
        with open(camera_params_path) as f:
            calib_data = yaml.load(f, Loader=yaml.FullLoader)
            self.width = calib_data['image_width']
            self.height = calib_data['image_height']
            self.fx = calib_data['camera_matrix']['data'][0]
            self.fy = calib_data['camera_matrix']['data'][4]
            self.cx = calib_data['camera_matrix']['data'][2]
            self.cy = calib_data['camera_matrix']['data'][5]
            self.dist_coeffs = np.array(calib_data['distortion_coefficients']['data'])
        
        self.step = step
        self.grid = np.zeros((self.height, self.width), dtype=np.uint8)
        self.generate_grid()

    def generate_grid(self):
        for i in range(0, self.height, self.step):
            pt1 = ((i - self.cy) / self.fy, 0, 1)
            pt2 = ((i - self.cy) / self.fy, self.width, 1)
            pt1_dist = cv2.undistortPoints(np.array([pt1]), cameraMatrix=np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]]), distCoeffs=self.dist_coeffs)
            pt2_dist = cv2.undistortPoints(np.array([pt2]), cameraMatrix=np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]]), distCoeffs=self.dist_coeffs)
            pt1_pix = tuple(map(int, pt1_dist[0, 0]))
            pt2_pix = tuple(map(int, pt2_dist[0, 0]))
            cv2.line(self.grid, pt1_pix, pt2_pix, (255, 255, 255), 1)

        for j in range(0, self.width, self.step):
            pt1 = (0, (j - self.cx) / self.fx, 1)
            pt2 = (self.height, (j - self.cx) / self.fx, 1)
            pt1_dist = cv2.undistortPoints(np.array([pt1]), cameraMatrix=np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]]), distCoeffs=self.dist_coeffs)
            pt2_dist = cv2.undistortPoints(np.array([pt2]), cameraMatrix=np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]]), distCoeffs=self.dist_coeffs)
            pt1_pix = tuple(map(int, pt1_dist[0, 0]))
            pt2_pix = tuple(map(int, pt2_dist[0, 0]))
            cv2.line(self.grid, pt1_pix, pt2_pix, (255, 255, 255), 1)

    def show_grid(self):
        cv2.imshow('Road Grid', self.grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


road_grid = RoadGrid(camera_params_path='path/to/camera_params.yaml', step=50)
road_grid.show_grid()