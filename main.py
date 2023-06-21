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
from rclpy.action import ActionClient
from rclpy.node import Node
from std_srvs.srv import SetBool
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy, QoSProfile
from motion_msgs.action import ChangeMode, ChangeGait, ExtMonOrder
from motion_msgs.msg import SE3VelocityCMD, Frameid
from .modules import Camera_RGB, Camera_Depth
import cv2
import sys
import time
import numpy as np
import math
from enum import Enum
import random


class Move(Node):
    """
    机器狗的运动节点，控制机器狗的步态、行动方向、行动速度、站立和趴下等动作
    * send_goal_mode: 初始化机器狗模式
    * send_goal_gait: 初始化机器狗步态
    * send_goal_order: 初始化机器狗指令
    * publish: 发布机器狗运动信息，使机器狗行动
    * mode_call: 模式回调函数
    * gait_call: 步态回调函数
    * order_call: 指令回调函数
    """
    def __init__(self):
        super().__init__('move')
        self.action_change_mode = ActionClient(
            self, ChangeMode, '/mi1047522/checkout_mode')
        self.action_change_gait = ActionClient(
            self, ChangeGait, '/mi1047522/checkout_gait')
        self.action_change_monorder = ActionClient(
            self, ExtMonOrder, '/mi1047522/exe_monorder')

        self.pub_vel_cmd = self.create_publisher(
            SE3VelocityCMD, '/mi1047522/body_cmd', 6)
        self.camera_rgb = Camera_RGB()
        self.camera_depth = Camera_Depth()

        self.bridge = CvBridge()

    def send_goal_mode(self, control_mode=3):
        goal_msg = ChangeMode.Goal()
        goal_msg.modestamped.timestamp = self.get_clock().now().to_msg()
        goal_msg.modestamped.control_mode = control_mode
        self.action_change_mode.wait_for_server()
        self.response = self.action_change_mode.send_goal_async(
            goal_msg, feedback_callback=self.mode_call)
        return

    def send_goal_gait(self, gait=6):
        goal_msg = ChangeGait.Goal()
        goal_msg.motivation = 253
        goal_msg.gaitstamped.timestamp = self.get_clock().now().to_msg()
        goal_msg.gaitstamped.gait = gait
        self.action_change_gait.wait_for_server()
        self.response = self.action_change_gait.send_goal_async(
            goal_msg, feedback_callback=self.gait_call)
        return

    def send_goal_order(self, order=13):
        goal_msg = ExtMonOrder.Goal()
        goal_msg.orderstamped.timestamp = self.get_clock().now().to_msg()
        goal_msg.orderstamped.id = order
        self.action_change_monorder.wait_for_server()
        self.response = self.action_change_monorder.send_goal_async(
            goal_msg, feedback_callback=self.order_call)
        return

    def publish(self, x, y, z, ang_x, ang_y, ang_z):
        my_msg = SE3VelocityCMD()
        my_frameid = Frameid()
        my_frameid.id = 1
        my_msg.sourceid = 2
        my_msg.velocity.frameid = my_frameid
        my_msg.velocity.timestamp = self.get_clock().now().to_msg()
        my_msg.velocity.linear_x = x
        my_msg.velocity.linear_y = y
        my_msg.velocity.linear_z = z
        my_msg.velocity.angular_x = ang_x
        my_msg.velocity.angular_y = ang_y
        my_msg.velocity.angular_z = ang_z
        self.pub_vel_cmd.publish(my_msg)
        # print(str(my_msg))

        return

    def mode_call(self, msg):
        print(msg)
        return

    def gait_call(self, msg):
        print(msg)
        return

    def order_call(self, msg):
        print(msg)
        return


class Perception():
    """
    机器狗的感知类，感知球的颜色信息、深度信息；感知障碍物的深度信息
    * mean_mask: 计算mask区域中深度的平均值
    * mean_circle: 计算识别到的外接圆区域中深度的平均值
    * depth_info_ball: 感知给定圆形区域的深度信息
    * depth_info_box: 感知机器狗前方的深度信息
    * rgb_info_ball: 感知机器狗前方空间中的蓝色信息
    """
    def __init__(self) -> None:
        self.camera_rgb = Camera_RGB()
        self.camera_depth = Camera_Depth()
        self.bridge = CvBridge()
        self.contour = None
        self.filter_depth1 = 100000
        self.filter_depth2 = 100000
        self.filter_depth3 = 100000


    def mean_mask(self, depth_img):
        mask = np.zeros(depth_img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, self.contour, -1, 1, -1)
        mean_val, _, _, _ = cv2.mean(depth_img, mask=mask)
        return mean_val

    def mean_circle(self, depth_img, center, radius):
        y, x = np.ogrid[-center[1]:depth_img.shape[0] -
                        center[1], -center[0]:depth_img.shape[1]-center[0]]
        mask = x*x + y*y <= radius*radius
        avg = np.mean(depth_img[mask])
        return avg

    def depth_info_ball(self, center, radius):
        rclpy.spin_once(self.camera_depth)
        try:
            depth_img = self.bridge.imgmsg_to_cv2(
                self.camera_depth.msg, '16UC1')
        except CvBridgeError as e:
            print(e)
        # cv2.imshow("Camera_Depth", depth_img)

        xl = int(max(0, center[0] - radius/2))
        xr = int(max(0, center[0] + radius/2))
        yl = int(max(0, center[1] - radius/2))
        yr = int(max(0, center[1] + radius/2))
        print("Depth_RECT => ", np.mean(depth_img[yl:yr, xl:xr]))

        filter_depth = self.mean_circle(depth_img, center, radius/4)
        print("Depth_Circle =>", filter_depth)
        filter_depth = self.mean_mask(depth_img)
        print("Depth_Mask => ", filter_depth)

        upper_threshold = 400
        lower_threshold = 100

        if filter_depth < upper_threshold and filter_depth > lower_threshold:
            print(
                f"Depth Ball => Less than {upper_threshold}, Greater than {lower_threshold}")
            return True

        return False

    def depth_info_box(self):
        rclpy.spin_once(self.camera_depth)
        try:
            depth_img = self.bridge.imgmsg_to_cv2(
                self.camera_depth.msg, '16UC1')
        except CvBridgeError as e:
            print(e)

        self.filter_depth1 = np.mean(depth_img[200:280, 00:240])
        self.filter_depth2 = np.mean(depth_img[200:280, 240:400])
        self.filter_depth3 = np.mean(depth_img[200:280, 400:640])

        print("Depth Box => ", [self.filter_depth1,
              self.filter_depth2, self.filter_depth3])

        if np.mean(depth_img[:, 240:400]) > 1300:
            print("GAP => GO GO GO")
            return False

        upper_threshold = 777
        lower_threshold = -1
        if self.filter_depth1 < upper_threshold + 100 or \
           self.filter_depth2 < upper_threshold or \
           self.filter_depth3 < upper_threshold + 100 :
            print(
                f"Depth Box => Less than {upper_threshold}, Greater than {lower_threshold}")
            return True

        return False

    def rgb_info_ball(self):
        """
        返回颜色区间内是否存在物体, 以及物体坐标和外接圆半径
        return True 代表存在物体
        return False 代表不存在物体
        """

        rclpy.spin_once(self.camera_rgb)
        try:
            frame = self.bridge.imgmsg_to_cv2(self.camera_rgb.msg, 'bgr8')
        except CvBridgeError as e:
            print(e)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([90, 68, 64])  # l_h, l_s, l_v
        upper_blue = np.array([144, 255, 255])  # u_h, u_s, u_v
        mask1 = cv2.inRange(hsv, lower_blue, upper_blue)
        mask = cv2.medianBlur(mask1, 9)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        contours, hierachy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        L = len(contours)  # contours轮廓数据是数组，用len()测数组长度，为了循环画点使用
        center, radius = (0, 0), 0
        max_idx = 0
        max_area = 0
        area_threshold = 500

        if L == 0:
            print("Region => None")
            return False, center, radius

        # 寻找最大面积区域
        for i in range(L):
            # cnt表示第i个色块的轮廓信息
            cnt = contours[i] 
            # 计算第i个色块的面积
            area = cv2.contourArea(cnt)  

            if area < area_threshold:
                continue
            if area > max_area:
                max_area = area
                max_idx = i

        # 筛选出来最大面积轮廓
        if max_area != 0:
            cnt = contours[max_idx]
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)

            area_predict = math.pi * radius*radius
            # 描述形状
            print("MAX_AREA => ", int(max_area), int(area_predict),
                  '{:.2f}'.format(max_area/area_predict))
            area_ratio = max_area/area_predict
            print("Circle =>", center, radius)

            # 画出拟合的外接圆
            cv2.circle(frame, center, radius, (0, 255, 0), thickness=2)
            cv2.drawContours(frame, cnt, -1, (0, 0, 255), thickness=2)

            self.contour = cnt
            if area_ratio > 0.50:
                return True, center, radius

        return False, center, radius


class State(Enum):
    """
    机器狗的状态枚举类
    * EMPTY: 寻球状态
    * BOX: 有障碍物状态
    * BALL: 找到球完成任务状态
    """
    EMPTY = 0
    BOX = 1
    BALL = 2


class Method():
    """
    机器狗的决策类
    * ini: 机器狗初始化，从趴下到站立状态
    * rest: 机器狗终止状态，从站立到趴下状态
    * update_para: 更新机器狗运动参数
    * rotate: 设置机器狗旋转参数
    * forward: 设置机器狗前进参数
    * horizon: 设置机器狗水平移动参数
    * sit_down: 设置机器狗趴下参数
    * follow_ball: PID寻球算法
    * rotate_action: 机器狗空间自旋转函数，用于跳出循环子空间
    * policy: 机器狗避障寻球策略函数
    * move_once: 机器狗根据参数行动一次函数
    * action: 机器狗运动控制函数
    * decision: 机器狗决策函数
    """
    def __init__(self) -> None:
        self.perception = Perception()
        self.move = Move()
        self.parameters = {
            'vx': 0,
            'vy': 0,
            'ang_z': 0,
            'gait': '',
            'delta': 0
        }

        self.flag = True
        self.time = time.time()
        self.state = State.EMPTY
        self.pid = PIDController(Kp=0.7, Ki=0.05, Kd=0.1, target=1.0)
        self.forward_begin_time = time.time()
        self.default_vx = 0.30

    def ini(self, gait=6):
        self.perception.rgb_info_ball()
        time.sleep(5)
        # 站立过程
        self.move.send_goal_mode(control_mode=3)
        time.sleep(5)
        print(f"MOVE => send_goal_gait(gait={gait})")
        self.move.send_goal_gait(gait=gait)
        time.sleep(5)
        print("MOVE => send_goal_order(order=13)")
        self.move.send_goal_order(order=13)

    def rest(self):
        time.sleep(2)
        print("MOVE => send_goal_gait(gait=2)")
        self.move.send_goal_gait(gait=2)
        time.sleep(5)
        # self.move.destroy_node()

    def update_para(self, vx, vy, ang_z, gait, delta):
        self.parameters['vx'] = vx
        self.parameters['vy'] = vy
        self.parameters['ang_z'] = ang_z
        self.parameters['gait'] = gait
        self.parameters['delta'] = delta

    def rotate(self, angle, clockwise=1):
        ratio = 2
        ang_z = -0.3 * clockwise * ratio
        # 负值是顺时针
        # 0.2 时 360° 90s
        # 0.3 时 360° 59s
        delta = angle / 6 / 180 * 100 / ratio
        self.update_para(vx=0.0, vy=0.0, ang_z=ang_z, gait='', delta=delta)

    def forward(self, vx=0.3, delta=0.1):
        vx = vx
        delta = delta
        self.update_para(vx=vx, vy=0.0, ang_z=0.0, gait='', delta=delta)

    def horizon(self, vy=0.3, delta=0.1):
        vy = vy
        delta = delta
        self.update_para(vx=0.0, vy=vy, ang_z=0.0, gait='', delta=delta)

    def sit_down(self):
        self.update_para(vx=0.0, vy=0.0, ang_z=0.0, gait='down', delta=5.0)

    def follow_ball(self):
        self.pid.reset()
        while True:
            # no obstacle, then pid_control to follow ball
            region, center, radius = self.perception.rgb_info_ball()

            if self.perception.depth_info_ball(center, radius):
                break

            input = center[0] / 320
            # 球丢失了(在走的过程中可能被箱子挡住了)，那就默认球在中央
            if not region:
                input = 1

            out = self.pid.update(input, dt=0.5)
            print("PID out => ", out, "========================================================")
            # rotate_ratio = 0.30
            rotate_ratio = 0.50
            vx_ratio = 0.19
            vy_ratio = 0.00
            self.update_para(vx=vx_ratio, vy=min(vy_ratio * out, vy_ratio),
                             ang_z=min(rotate_ratio * out, rotate_ratio), gait='', delta=0.1)
            print("Para =>", self.parameters)
            self.action()

    def rotate_action(self, angle):
        if random.random() < 0.5:
            clockwise = 1
        else:
            clockwise = -1
        for i in range(int(angle/1)):
            print("rotate_idx: ", i)
            self.rotate(1, clockwise=clockwise)
            self.forward_begin_time = time.time()
            self.action()
            region, center, r = self.perception.rgb_info_ball()
            # time.sleep(0.2)
            if (region):
                print("-----find ball from rotate-----")
                break

    def policy(self):
        """机器狗策略函数"""
        # 时刻观察视野里是否有球存在
        region, center, radius = self.perception.rgb_info_ball()
        # 视野里有球，重点在于找球
        if region:
            # 球的距离足够，则进入趴下终止状态
            if self.perception.depth_info_ball(center, radius):
                self.state = State.BALL
                self.sit_down()
                return self.parameters, self.state
            # 球的距离不够，进入寻球状态
            else:
                print("====FOLLOW_BALL_ING========================================================")
                self.follow_ball()
                print(self.parameters, self.state)
                return self.parameters, self.state

        # 视野里有盒子，重点在于避障
        if self.perception.depth_info_box():
            self.forward_begin_time = time.time()
            # 看到球了，平移避障
            if self.state == State.BALL:
                # 既有球又有盒子，优先保持为BALL状态
                print("-----horizon-----")
                self.horizon()
            # 还没看到球，旋转避障
            elif self.perception.filter_depth1 < 200 or \
                    self.perception.filter_depth2 < 200 or \
                    self.perception.filter_depth3 < 200:
                print("Pace Back => True")
                self.forward(vx=-0.17, delta=0.15)
            else:
                print("-----__rotate__------")
                self.rotate(15)
                # self.rotate(17)
                # self.horizon()
                self.state = State.BOX

        # 视野里没有盒子也没有球，直走
        else:
            self.forward()
            # self.rotate(30)
            self.state = State.EMPTY
            # 直走太久了(超过8s)，则进入空间自旋转寻球状态
            if time.time() - self.forward_begin_time > 8:
                # 顺时针旋转120度，看看有没有球
                self.rotate_action(80)

        return self.parameters, self.state

    def move_once(self, delta):
        """根据参数运动delta时间"""
        
        start_time = end_time = time.time()
        while (True):
            self.move.publish(x=self.parameters['vx'], y=self.parameters['vy'],
                              z=0.0, ang_x=0.0, ang_y=0.0,
                              ang_z=self.parameters['ang_z'])
            if end_time - start_time > delta:
                break
            end_time = time.time()

    def action(self):
        """action分两种：移动和趴下"""
        if self.parameters['gait'] == 'down':
            print("Action => gait_down")
            self.move.send_goal_gait(gait=2)
            self.parameters['delta'] = 5.0
            time.sleep(self.parameters['delta'])
    
        self.move_once(self.parameters['delta'])

    def decision(self):
        self.forward_begin_time = time.time()
        # 如果没有找到球，机器狗将一直处于决策状态
        while (True):
            para, state = self.policy()
            # 机器狗根据policy的返回值输出参数信息并运动
            print("Para => ", self.parameters)
            self.action(State.EMPTY)
            # 机器狗找到球，跳出决策状态
            if state == State.BALL:
                break


class PIDController:
    """
    PID控制类
    * update: PID信息更新
    * reset: PID信息重置
    """
    
    def __init__(self, Kp=0.7, Ki=0.05, Kd=0.1, target=1.0):
        """
        Args:
            kp (float): 比例系数
            ki (float): 比例积分系数
            kd (float): 比例微分系数
            target (float): 目标修正值 
        """
        
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.target = target

        self.integral = 0
        self.previous_error = 0

    def update(self, current_value, dt=1):
        error = self.target - current_value
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.previous_error = error
        return output

    def reset(self):
        self.integral = 0
        self.previous_error = 0


def main():
    print("Sys => INIT")
    rclpy.init(args=None)
    Dog = Method()
    Dog.ini(gait=7)
    while (True):
        Dog.decision()
        break

    Dog.rest()
    print("Sys => rcl shutdown")
    rclpy.shutdown()


if __name__ == '__main__':
    main()
