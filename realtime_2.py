import pyrealsense2 as rs
import os
import numpy as np
import cv2
from cv2 import aruco
from ultralytics import YOLO


class RealsensePose:
    def __init__(self, checkpoints_path, w=640, h=480):

        self.pipeline = rs.pipeline()

        self.pc = rs.pointcloud()

        self.align = rs.align(rs.stream.color)

        self.init_realsense(w, h)

        self.model = YOLO(r'E:\my_yolo\best.pt')

    def init_realsense(self, w, h):
        config = rs.config()
        config.enable_stream(rs.stream.depth, 320, 240, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        # config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 15)
        profile = self.pipeline.start(config)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: ", depth_scale)

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()

        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        return color_frame, depth_frame

    def get_vertices_1(self, color_frame, depth_frame):
        points = self.pc.calculate(depth_frame)
        self.pc.map_to(color_frame)
        vertices = points.get_vertices()
        vertices = np.asanyarray(vertices).view(np.float32).reshape(-1, 3)  # xyz
        return vertices

    def run(self):

        mean_time = 0
        count = 1

        while True:
            current_time = cv2.getTickCount()

            color_frame, depth_frame = self.get_frames()
            vertices = self.get_vertices_1(color_frame, depth_frame)

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            id = -1
            results = self.model(color_image, conf=0.5)
            boxes = results[0].boxes
            if len(boxes.conf) > 0:
                for index in range(len(boxes.conf)):
                    conf = boxes.conf[index]
                    id = boxes.cls[index]
                # print(results[0].boxes.xywhn[index])
                    xywhn_ = boxes.xywhn[index]
                    xywhn = [item.cpu().numpy() for item in xywhn_]
                    xyxy_ = boxes.xyxy[index]
                    xyxy = [int(item) for item in xyxy_]
                    p1 = (xyxy[0], xyxy[1])
                    p2 = (xyxy[2], xyxy[1])
                    p3 = (xyxy[0], xyxy[3])
                    p4 = (xyxy[2], xyxy[3])
                    cv2.line(color_image, p1, p2, color=(0, 255, 0), thickness=2)
                    cv2.line(color_image, p2, p4, color=(0, 255, 0), thickness=2)
                    cv2.line(color_image, p4, p3, color=(0, 255, 0), thickness=2)
                    cv2.line(color_image, p3, p1, color=(0, 255, 0), thickness=2)
                    # text = 'id: {}, conf: {:.2f}'.format(int(id), conf)
                    text = 'id: {}'.format(int(id))
                    cv2.putText(color_image, text, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
            if mean_time == 0:
                mean_time = current_time
            else:
                mean_time = mean_time * 0.95 + current_time * 0.05
            text = 'FPS: {}'.format(int(1 / mean_time * 10) / 10)
            cv2.putText(color_image, text, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # cv2.imshow('Depth', depth_colormap)
            cv2.imshow('Color', color_image)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break


if __name__ == "__main__":
    from config import OPENPOSE_PATH

    rs_pose = RealsensePose(OPENPOSE_PATH)
    rs_pose.run()
