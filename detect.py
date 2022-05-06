# import the necessary packages
import argparse
from cmath import inf
import numpy as np
import cv2
import sys
import time
import pandas as pd
import os

from utils import ARUCO_DICT, corners2center, plot_marker,  xywh2xyxy, scale_bbox, show_img, plot_box
from capture import RealSense


# construct the argument parser and parse the arguments
def parse_agrs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", type=str,
        default="DICT_ARUCO_ORIGINAL", help="type of ArUCo tag to generate")
    parser.add_argument("-f", "--filter_ids", nargs='+', default=[25], help="id filting")

    return parser.parse_args()


class Detector(object):
    def __init__(self, window_name):
        self.window_name = window_name
        self.debug = True
        self.box_type = 'min_area' # bbox or min_area
        self.contours_thres = (0, 50) # (25, 80) # select contours with number of pixels in (25, 80)
        self.view_result = True

        # Parameters for HSV
        # (36, 202, 59, 71, 255, 255)    # Green
        # (18, 0, 196, 36, 255, 255)  # Yellow
        # (89, 0, 0, 125, 255, 255)  # Blue
        self.icol = (71, 136, 157, 128, 230, 255)
        self.hsv_names = ('low_hue', 'low_sat', 'low_val', 'high_hue', 'high_sat', 'high_val')

        if self.debug:
            cv2.namedWindow(self.window_name)
            self.create_trackbar()

    def create_trackbar(self,):
        for i in range(len(self.hsv_names)):
            cv2.createTrackbar(self.hsv_names[i], self.window_name, self.icol[i], 255, lambda x: x)

    def get_trackbar_value(self,):
        hsv_thres_values = [] # ['low_hue', 'low_sat', 'low_val', 'high_hue', 'high_sat', 'high_val']
        for i in range(len(self.hsv_names)):
            hsv_thres_values.append(cv2.getTrackbarPos(self.hsv_names[i], self.window_name))

        return hsv_thres_values

    def color_detect(self, image, process_shape=(640, 480)):
        '''
        input: image
        output: dst, box(xyxy(bbox) or 4x2(min_area)), center(xy)
        '''
        if image is None or image is []:
            raise TypeError('No frame input')
        # Resize the frame
        shape = process_shape
        H, W, _ = image.shape
        scale_factor = np.array([shape[0]/W, shape[1]/H])

        image = cv2.resize(image, shape) # 300,400,3

        # Gaussian Blur
        image_gaussian = cv2.GaussianBlur(image, (7, 7), 0)

        # RGB to HSV
        image_hsv = cv2.cvtColor(image_gaussian, cv2.COLOR_BGR2HSV)

        # Get mask according to HSV
        hsv_thres_values = self.get_trackbar_value() if self.debug else list(self.icol)
        mask = cv2.inRange(image_hsv, np.array(hsv_thres_values[:3]), np.array(hsv_thres_values[3:]))

        # Median filter
        mask_f = cv2.medianBlur(mask, 5)

        # Morphology for three times
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_m = cv2.morphologyEx(mask_f, cv2.MORPH_CLOSE, kernel)
        mask_m = cv2.morphologyEx(mask_m, cv2.MORPH_OPEN, kernel)

        if self.debug:
            cv2.imshow('debug', mask_m)

        # Get Contours of The Mask
        box = None # xyxy
        center = None # xy
        contours, _= cv2.findContours(mask_m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt_list = [cnt for cnt in contours if self.contours_thres[0]<len(cnt)<self.contours_thres[1]]
        if cnt_list:
            cnt = max(contours, key=lambda x: x.shape[0]) # mutiple contours, choose the biggest one

            if self.box_type == 'bbox':
                # Get Bounding Box
                box = np.int0(scale_bbox(xywh2xyxy(cv2.boundingRect(cnt)), 1/scale_factor)) # xyxy
                center = np.int0(np.array([(box[0] + box[2])/2, (box[1] + box[3])/2]))
            elif self.box_type == 'min_area':
                # Get Minimum Area Box
                rect = cv2.minAreaRect(cnt) # center(x, y), (width, height), angle of rotation
                box = cv2.boxPoints(rect) # (4, 2)
                # scale box
                box = box / scale_factor
                center = np.sum(box, axis=0)/4
                box, center = np.int0(box), np.int0(center)
            else:
                raise TypeError('unsupported box type %s' % self.box_type)

        return box, center

    def aruco_detect(self, image, tag_type="DICT_ARUCO_ORIGINAL"):
        # verify that the supplied ArUCo tag exists and is supported by
        # OpenCV
        if ARUCO_DICT.get(tag_type, None) is None:
            print("[INFO] ArUCo tag of '{}' is not supported".format(tag_type))
            sys.exit(0)

        # load the ArUCo dictionary, grab the ArUCo parameters, and detect
        # the markers
        print("[INFO] detecting '{}' tags...".format(tag_type))
        arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[tag_type])
        arucoParams = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

        return corners, ids

    def post_process_color(self, image, depth_frame, box, center):
        coor = None
        if center is not None:
            x, y = center
            coor_cam = cap.cal_camera_coor(x, y, depth_frame)
            coor = cam_to_base_RT[:3,:3] @ np.array(coor_cam).reshape(-1, 1) + cam_to_base_RT[:3, 3:]
            coor = coor.reshape(3,)

            if self.view_result:
                image = plot_box(image, box, self.box_type)
                cv2.putText(image, 'Coor Cam: ({:.3f}, {:.3f}, {:.3f})'.format(*coor_cam), (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                cv2.putText(image, 'Coor Base: ({:.3f}, {:.3f}, {:.3f})'.format(*coor), (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        return coor, image

    def post_process_aurco(self, image, depth_frame, corners, ids, filterIDs=None):
        coors = {}

        # verify *at least* one ArUco marker was detected
        if len(corners) > 0:
            # flatten the ArUco IDs list
            ids = ids.flatten()

            new_corners, new_ids = [], []
            if filterIDs is not None:
                for markerCorner, markerID in zip(corners, ids):
                    if markerID in filterIDs:
                        new_corners.append(markerCorner)
                        new_ids.append(markerID)
                corners, ids = new_corners, new_ids
            if len(corners) > 0:
                # loop over the detected ArUCo corners
                for (markerCorner, markerID) in zip(corners, ids):
                    x, y = corners2center(markerCorner)
                    coor_cam = cap.cal_camera_coor(x, y, depth_frame)
                    coor = cam_to_base_RT[:3,:3] @ np.array(coor_cam).reshape(-1, 1) + cam_to_base_RT[:3, 3:]
                    coor = coor.reshape(3,)
                    coors[markerID] = coor

                    if self.view_result:
                        image = plot_marker(image, markerCorner, markerID)
                        cv2.putText(image, 'Coor Cam: ({:.3f}, {:.3f}, {:.3f})'.format(*coor_cam), (360, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                        cv2.putText(image, 'Coor Base: ({:.3f}, {:.3f}, {:.3f})'.format(*coor), (360, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        return coors, image


if __name__ == '__main__':
    args = parse_agrs()

    cap = RealSense()
    cam_to_base_RT = np.load('./calibration/camera2base.npy')

    window_name = 'RealSense'
    detector = Detector(window_name)
    coor_sequence = []
    for color_image, depth_frame in cap:
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(np.asanyarray(depth_frame.get_data()), alpha=0.03), cv2.COLORMAP_JET)

        corners, ids = detector.aruco_detect(color_image.copy(), args.type)
        aurco_coors, image = detector.post_process_aurco(color_image.copy(), depth_frame, corners, ids, filterIDs=args.filter_ids)

        box, center = detector.color_detect(color_image.copy())
        color_coor, image = detector.post_process_color(image, depth_frame, box, center)

        coor_cat = np.concatenate([aurco_coors[args.filter_ids[0]] if len(aurco_coors) else np.array([float(inf), float(inf), float(inf)]),
                                   color_coor if color_coor is not None else np.array([float(inf), float(inf), float(inf)])], axis=0)
        coor_sequence.append(coor_cat)

        image = np.hstack((image, depth_colormap))

        # show the output image
        cv2.imshow(window_name, image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    os.makedirs('./track', exist_ok=True)
    pd.DataFrame(np.stack(coor_sequence, axis=0)).to_csv(f'./track/{str(time.time())}.csv')
