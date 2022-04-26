import numpy as np
import transforms3d as tfs
import cv2
import math
import os
import argparse

from utils import ARUCO_DICT
from capture import RealSense
from detect import aruco_detect, post_process


# construct the argument parser and parse the arguments
def parse_agrs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", type=str,
        default="DICT_ARUCO_ORIGINAL", help="type of ArUCo tag to generate")
    parser.add_argument("-s", "--size", type=float,
        default=0.08, help="size of ArUCo tag to generate")
    parser.add_argument("--save", action='store_true')


    return parser.parse_args()

def cal_board2camera(args):
    cap = RealSense()
    intrix = cap.intric

    mtx = np.array([[intrix.fx, 0, intrix.ppx],
                   [0, intrix.fy, intrix.ppy],
                   [0, 0, 1]])

    count = -1
    poses = []
    for color_image, _ in cap:
        image = color_image.copy()
        corners, ids = aruco_detect(image, args.type)
        # verify *at least* one ArUco marker was detected
        if len(corners) > 0:
            # flatten the ArUco IDs list
            ids = ids.flatten()

            corners, ids = post_process(corners, ids, filterIDs=[25])

            if len(corners) > 0:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, args.size, mtx, np.array(intrix.coeffs))
                rvec, tvec = rvecs[0], tvecs[0]
                cv2.aruco.drawAxis(image, mtx, None, rvec[:, :], tvec[:, :], 0.03) # 绘制坐标轴
                cv2.aruco.drawDetectedMarkers(image, corners)
                print('tvec: ', tvec, 'rvec: ', rvec)

                if cv2.waitKey(1) & 0xFF == ord('c'):
                    count += 1
                    cv2.imwrite(f'./calibration/{count}.jpg', image)
                    poses.append(np.concatenate([tvec, rvec], axis=-1))

        # show the output image
        cv2.imshow('RealSense', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    if args.save:
        poses = np.concatenate(poses, axis=0) if len(poses) else np.zeros((0, 6))
        np.save('./calibration/board2camera.npy', poses)


def calibration(board_to_camera, end_to_base):
    board_to_cam_R, board_to_cam_T = [], []
    end_to_base_R, end_to_base_T = [], []
    base_to_end_R, base_to_end_T = [], []
    for board_cam in board_to_camera:
        bc = board_cam[3:6]
        bc_R, j = cv2.Rodrigues((bc[0], bc[1], bc[2]))
        board_to_cam_R.append(bc_R)
        board_to_cam_T.append(np.array(board_cam[0:3]).reshape(3, 1))
    for end_base in end_to_base:
        ed = end_base[3:6]
        eb_R = tfs.euler.euler2mat(ed[0], ed[1], ed[2], axes='sxyz')
        end_to_base_R.append(eb_R)
        base_to_end_R.append(eb_R.T)
        end_to_base_T.append(np.array(end_base[0:3]).reshape(3,1))
        base_to_end_T.append(-eb_R.T @ np.array(end_base[0:3]).reshape(3,1))

    print('board_to_cam_R', board_to_cam_R)
    print('board_to_cam_T', board_to_cam_T)
    print('base_to_end_R', end_to_base_R)
    print('base_to_end_T', end_to_base_T)
    cam_to_end_R, cam_to_end_T = cv2.calibrateHandEye(end_to_base_R, end_to_base_T, board_to_cam_R, board_to_cam_T,
                                                    method=cv2.CALIB_HAND_EYE_TSAI)
    print('cam_to_end_R', cam_to_end_R)
    print('cam_to_end_T', cam_to_end_T)
    cam_to_end_RT = tfs.affines.compose(np.squeeze(cam_to_end_T), cam_to_end_R, [1, 1, 1])
    print("cam_to_end_RT", cam_to_end_RT)
    np.save('./calibration/camera2end.npy', cam_to_end_RT)

    print("*"*80)
    N = len(board_to_cam_R)
    for i in range(0, N):
        RT_end_to_base = np.column_stack((end_to_base_R[i], end_to_base_T[i].reshape(3,1)))
        RT_end_to_base = np.row_stack((RT_end_to_base, np.array([0,0,0,1])))

        RT_board_to_cam = np.column_stack((board_to_cam_R[i], board_to_cam_T[i].reshape(3,1)))
        RT_board_to_cam = np.row_stack((RT_board_to_cam, np.array([0, 0, 0, 1])))

        RT_board_to_base = RT_end_to_base @ cam_to_end_RT @ RT_board_to_cam

        print(f'{i}', RT_board_to_base)


if __name__ == '__main__':
    args = parse_agrs()
    os.makedirs('./calibration', exist_ok=True)
    # cal_board2camera(args)
    board_to_camera = np.load('./calibration/board2camera.npy')
    board_to_camera = np.delete(board_to_camera, [2], axis=0)
    print(board_to_camera)
    end_to_base = np.load('./calibration/end2base_4.npy')
    print(end_to_base)

    board_to_camera = np.delete(board_to_camera, range(5, 7), axis=0)
    end_to_base = np.delete(end_to_base, range(5, 7), axis=0)
    calibration(board_to_camera, end_to_base)