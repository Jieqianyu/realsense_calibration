import pyrealsense2 as rs
import numpy as np
import cv2
from threading import Thread

class RealSense:
    def __init__(self, viz=False):
        self.viz = viz
        pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        align = rs.align(rs.stream.color)

        pipeline.start(config)

        self.depth_frame, self.color_frame = self.once(pipeline, align)

        self.intric = self.depth_frame.profile.as_video_stream_profile().intrinsics

        self.thread = Thread(target=self.update, args=([pipeline, align]), daemon=not viz)
        self.thread.start()

    def once(self, pipeline, align):
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        return depth_frame, color_frame,

    def update(self, pipeline, align):
        try:
            while True:
                depth_frame, color_frame = self.once(pipeline, align)
                if not depth_frame or not color_frame:
                    continue
                self.depth_frame = depth_frame
                self.color_frame = color_frame
                self.intric = self.depth_frame.profile.as_video_stream_profile().intrinsics
                if self.viz:
                    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(np.asanyarray(self.depth_frame.get_data()), alpha=0.03), cv2.COLORMAP_JET)
                    images = np.hstack((np.asanyarray(self.color_frame.get_data()), depth_colormap))

                    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                    cv2.imshow('RealSense', images)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break
        finally:
            pipeline.stop()

    def __iter__(self):
        return self

    def __next__(self):
        color_image = np.asanyarray(self.color_frame.get_data())

        return color_image, self.depth_frame

    def cal_camera_coor(self, x, y, depth_frame):
        dis = depth_frame.get_distance(x, y)

        camera_coordinate = rs.rs2_deproject_pixel_to_point(self.intric, [x, y], dis)

        return camera_coordinate


if __name__ == '__main__':
    cap = RealSense(viz=False)
    for color_image, depth_frame in cap:
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(np.asanyarray(depth_frame.get_data()), alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((color_image, depth_colormap))

        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
