# Copyright (c) 2023 Cyberdog Team 11 (DA). All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http:#www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from motion_msgs.action import ChangeMode, ChangeGait, ExtMonOrder
from motion_msgs.msg import SE3VelocityCMD, Frameid
from cv_bridge import CvBridge, CvBridgeError
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy, QoSProfile
from sensor_msgs.msg import Image
import time
import cv2


class Camera_RGB(Node):
    """
    RGB相机节点，订阅RGB图像数据
    * callback: 将订阅到的数据存入msg
    * display: 将订阅的msg转换为cv2格式并显示
    * displayModifiedImage: 显示经过处理的图像
    """
    def __init__(self):
        super().__init__('Camera_RGB')
        self.msg = ()
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/mi1046939/camera/color/image_raw', self.callback, QoSProfile(depth = 10, reliability = ReliabilityPolicy.BEST_EFFORT, durability = DurabilityPolicy.VOLATILE))
        self.vw = cv2.VideoWriter('/home/mi/da26_ws/output/camera_rgb.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (640, 480))
        return

    def callback(self, my_img):
        print("camera_rgb_callback")
        self.msg = my_img
        return
    
    def display(self,mode = "r"):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(self.msg, 'rgb8')
        except CvBridgeError as e:
            print(e)
        cv2.rectangle(cv_img, [160, 220], [480, 260], 0x0000ff)
        cv2.imshow("Camera", cv_img)
        if mode == "w":
            self.vw.write(cv_img)
        cv2.waitKey(3)
        return
    
    def displayModifiedImage(self, img, mode = "r"):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(img, 'rgb8')
        except CvBridgeError as e:
            print(e)
        cv2.imshow("Camera", cv_img)
        if mode == "w":
            self.vw.write(cv_img)
        cv2.waitKey(3)
        return
    
class Camera_Depth(Node):
    """
    深度相机节点，订阅深度图像数据
    * callback: 回调函数，将订阅到的数据存入msg
    * display: 显示深度图像
    """
    def __init__(self):
        super().__init__('Camera_Depth')
        self.msg = ()
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/mi1046939/camera/depth/image_rect_raw', self.callback, QoSProfile(depth = 10, reliability = ReliabilityPolicy.BEST_EFFORT, durability = DurabilityPolicy.VOLATILE))
        self.vw = cv2.VideoWriter('/home/mi/da26_ws/output/camera_depth.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (640, 480))
        return

    def callback(self, my_img):
        print("camera_depth_callback")
        self.msg = my_img
        return
    
    def display(self, mode = "r"):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(self.msg, '16UC1')
        except CvBridgeError as e:
            print(e)
        cv2.imshow("Camera_Depth", cv_img)
        if mode == "w":
            self.vw.write(cv_img)
        cv2.waitKey(3)
        return

    
if __name__ == "__main__":
    print("TEST")