# import the necessary packages
import cv2
import imutils
import numpy as np
import time


# define names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
#	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
#	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
#	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
#	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}


def load_image(path, width=600):
    print("[INFO] loading image...")
    image = cv2.imread(path)
    image = imutils.resize(image, width=width)

    return image


def load_stream(path=None, width=600, height=480):
    print("[INFO] starting video stream...")
    src = 0 if path is None or path=='0' else path
    cap = cv2.VideoCapture(src)

    # # set a lower resolution for speed up
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap


def plot_marker(image, markerCorner, markerID):
    # extract the marker corners (which are always returned in
    # top-left, top-right, bottom-right, and bottom-left order)
    corners = markerCorner.reshape((4, 2))
    (topLeft, topRight, bottomRight, bottomLeft) = corners

    # convert each of the (x, y)-coordinate pairs to integers
    topRight = (int(topRight[0]), int(topRight[1]))
    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
    topLeft = (int(topLeft[0]), int(topLeft[1]))

    # draw the bounding box of the ArUCo detection
    cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
    cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
    cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
    cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

    # compute and draw the center (x, y)-coordinates of the ArUco
    # marker
    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
    cY = int((topLeft[1] + bottomRight[1]) / 2.0)
    cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

    # if markerID == 15:
    #     cv2.circle(image, (bottomRight[0], bottomRight[1]), 4, (0, 0, 255), -1)
    # elif markerID == 20:
    #     cv2.circle(image, (topLeft[0], topLeft[1]), 4, (0, 0, 255), -1)
    # elif markerID == 25:
    #     cv2.circle(image, (topRight[0], topRight[1]), 4, (0, 0, 255), -1)
    # elif markerID == 30:
    #     cv2.circle(image, (bottomLeft[0], bottomLeft[1]), 4, (0, 0, 255), -1)

    # draw the ArUco marker ID on the image
    cv2.putText(image, str(markerID),
        (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (0, 255, 0), 2)

    return image


def plot_box(image, box, box_type):
    H, W, _ = image.shape
    line_width = int(0.005 * (H+W) / 2)
    if box is not None:
        if box_type == 'bbox':
            x1, y1, x2, y2 = box
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), line_width)
        elif box_type == 'min_area':
            image = cv2.drawContours(image, [box], 0, (0, 0,255), line_width)
        else:
            raise TypeError('unsupported box type %s' % box_type)

    return image


def corners2center(markerCorner):
    corners = markerCorner.reshape((4, 2))
    (topLeft, topRight, bottomRight, bottomLeft) = corners

    # convert each of the (x, y)-coordinate pairs to integers
    topRight = (int(topRight[0]), int(topRight[1]))
    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
    topLeft = (int(topLeft[0]), int(topLeft[1]))

    # compute and draw the center (x, y)-coordinates of the ArUco
    # marker
    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
    cY = int((topLeft[1] + bottomRight[1]) / 2.0)

    return cX, cY


video_format = ['mp4', 'avi']


def scale_bbox(x, scale):
    # Scale [4] box from [x1, y1, x2, y2] or [x1, y1, w, h]
    y = np.copy(x)
    y[0] *= scale[0]
    y[2] *= scale[0]
    y[1] *= scale[1]
    y[3] *= scale[1]

    return y


def xyxy2xywh(x):
    # Convert [4] box from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[0] = x[0]  # top-left
    y[1] = x[1]  # top-left
    y[2] = x[2] - x[0]  # width
    y[3] = x[3] - x[1]  # height

    return y


def xywh2xyxy(x):
    # Convert [4] box from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[0] = x[0]  # top left x
    y[1] = x[1]  # top left y
    y[2] = x[0] + x[2]  # bottom right x
    y[3] = x[1] + x[3]  # bottom right y

    return y


def show_img_debug(name, frame):
    print("The current window is occupied by image '" + name +
          "'. " + "Please press esc to shut down current window and move on.")
    start = time.time()
    while True:
        time_out = ((time.time() - start)) > 120
        cv2.imshow(name, frame)
        key = cv2.waitKey(33)
        if (key & 0xff) == 27 or time_out:
            if time_out:
                print("Time out! Shut down the window.")
            break


def show_img(name, frame):
    cv2.imshow(name, frame)
    if cv2.waitKey(1) == ord('q'):  # q to quit
        raise StopIteration