import pyrealsense2 as rs
import os
import numpy as np
import cv2
from cv2 import aruco


class RealsensePose:
    def __init__(self, w=640, h=480):

        self.pipeline = rs.pipeline()

        self.pc = rs.pointcloud()

        self.align = rs.align(rs.stream.color)

        self.init_realsense(w, h)

    def init_realsense(self, w, h):
        config = rs.config()
        config.enable_stream(rs.stream.depth, 320, 240, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

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
        sign = 1

        while True:
            current_time = cv2.getTickCount()

            color_frame, depth_frame = self.get_frames()
            vertices = self.get_vertices_1(color_frame, depth_frame)

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            params = aruco.DetectorParameters()
            aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
            detector = aruco.ArucoDetector(aruco_dict, params)
            corners, ids, _ = detector.detectMarkers(gray)
            # print(corners)
            gcps = []
            if ids is not None:
                for i in range(ids.size):
                    j = ids[i][0]
                    # calculate center of aruco code
                    x = int(round(np.average(corners[i][0][:, 0])))
                    y = int(round(np.average(corners[i][0][:, 1])))
                    gcps.append((x, y, j, corners[i][0]))
                    # print(j)

            # print(gcps)
            # print(len(gcps))

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            points = []
            Oab = None
            Pxb = None
            Pyb = None
            test_point = None

            for gcp in gcps:
                point = (gcp[0], gcp[1])
                id = gcp[2]
                cv2.circle(color_image, point, 3, (0, 255, 0))
                x, y, z = vertices[640 * point[1] + point[0]]
                if id == 0:
                    Oab = np.array([x, y, z])
                elif id == 1:
                    Pxb = np.array([x, y, z])
                elif id == 2:
                    Pyb = np.array([x, y, z])
                elif id == 3:
                    test_point = np.array([x, y, z])
                points.append([x, y, z])
                # text = "({:+.2f}, {:+.2f}, {:+.2f}), id={}".format(x, y, z, gcp[2])
                text = "id={}".format(id)

                cv2.putText(color_image, text, point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # if len(points) == 2:
            #     print(np.sqrt(points[0][0]-points[1][0]))
            # if sign:
            #     print(points)
            #     sign = 0
            if Oab is not None and Pxb is not None and Pyb is not None:
                x1 = (Pxb - Oab) / np.linalg.norm(Pxb - Oab)
                y1 = (Pyb - Oab) / np.linalg.norm(Pyb - Oab)
                z1 = np.cross(x1, y1)

                length = np.linalg.norm(z1)
                z1 = z1 / length
                Rab = np.matrix([x1, y1, z1]).transpose()
                Tab = np.matrix(Oab).transpose()
                temp = np.hstack((Rab, Tab))
                RT_ab = np.vstack((temp, [0, 0, 0, 1]))
                # print(RT_ab)
                RT_ba = np.linalg.inv(RT_ab)
                if test_point is not None:
                    Oab_1 = np.hstack((Oab, [1])).reshape(4, 1)
                    test_point_1 = np.hstack((test_point, [1])).reshape(4, 1)
                print(RT_ba)
                print(np.dot(RT_ba, Oab_1))
                print(np.dot(RT_ba, test_point_1))

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
    rs_pose = RealsensePose()
    rs_pose.run()
