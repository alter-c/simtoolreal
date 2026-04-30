#!/usr/bin/env python3
import time
import threading
import numpy as np
import rospy
from sensor_msgs.msg import JointState

# 常量定义
PI = 3.14159265358979
LEFT_ARM_DOF = 7

# 左臂基准姿态 (对应 Bridge 中的前7位)
# 顺序通常为: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
POSE_BASE = np.array([0.0, PI/9, 0.0, PI/2, 0.0, 0.0, 0.0])

class ArmBridgeTester:
    def __init__(self, hz: float = 50.0):
        rospy.init_node("arm_bridge_tester", anonymous=True)
        self.rate = rospy.Rate(hz)
        self._current_state = None
        self._lock = threading.Lock()


        # 发布指令到 Bridge 监听的 Topic
        self._cmd_pub = rospy.Publisher("/unitree/joint_cmd", JointState, queue_size=1)
        self._hand_pub = rospy.Publisher("/linkerhand/joint_cmd", JointState, queue_size=1)
        
        # 订阅 Bridge 发布的状态 Topic
        self._state_sub = rospy.Subscriber("/unitree/joint_states", JointState, self._state_cb)

        rospy.loginfo("Test Node Initialized. Target: /unitree/joint_cmd, Feedback: /unitree/joint_states")

    def _state_cb(self, msg):
        with self._lock:
            # 获取当前反馈的关节角度
            self._current_state = np.array(msg.position)

    def get_feedback(self):
        with self._lock:
            return self._current_state.copy() if self._current_state is not None else None

    def send_command(self, pos_array):
        """构造并发送 JointState 消息"""
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.position = pos_array.tolist()
        self._cmd_pub.publish(msg)
    
    def send_hand_command(self, pos_array):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.position = pos_array.tolist()
        self._hand_pub.publish(msg)     

    def run_sine_test(self, duration=10.0):
        """左肩 Pitch 关节正弦摆动测试"""
        rospy.loginfo("Starting Sine Trajectory Test...")
        start_time = time.time()
        while not rospy.is_shutdown():
            elapsed = time.time() - start_time
            if elapsed > duration:
                break
            
            # 计算目标位置：基准姿态 + 第一个关节正弦波动
            target = POSE_BASE.copy()
            target[0] += (PI / 3) * np.sin(2 * PI * 0.5 * elapsed) # 0.5Hz, ±30度
            print(target[0])
            
            self.send_command(target)

            # 打印误差反馈
            fb = self.get_feedback()
            if fb is not None:
                # 确保维度一致再计算误差
                min_len = min(len(fb), len(target))
                error = np.abs(fb[:min_len] - target[:min_len])
                rospy.loginfo_throttle(0.2, f"Max Tracking Error: {np.max(error):.4f} rad")
            
            joint_max = np.array([
                1.3,
                0.58,
                1.6,
                1.6,
                1.6,
                1.6,
            ])
            target = joint_max * abs(np.sin(2 * PI * 0.05 * elapsed))
            self.send_hand_command(target)
            
            self.rate.sleep()

    def run_waypoint_test(self):
        """离散路点测试"""
        waypoints = [
            np.zeros(LEFT_ARM_DOF),               # 复位
            POSE_BASE,                            # 基准姿态
            np.array([0.0, PI/4, 0.0, PI/3, 0.0, 0.0, 0.0]) # 抬臂
        ]
        
        for i, wp in enumerate(waypoints):
            rospy.loginfo(f"Moving to Waypoint {i}")
            # 每个点持续 2 秒高频发布，确保 Bridge 稳定接收
            t0 = time.time()
            while time.time() - t0 < 2.0 and not rospy.is_shutdown():
                self.send_command(wp)
                self.rate.sleep()

if __name__ == "__main__":
    try:
        tester = ArmBridgeTester(hz=50.0)
        
        # 1. 缓慢移动到基准姿态
        # tester.run_waypoint_test()
        
        # 2. 执行正弦跟踪
        tester.run_sine_test(duration=10.0)
        
        # 3. 回归零位
        rospy.loginfo("Test complete. Returning to zero.")
        tester.send_command(POSE_BASE)

    except rospy.ROSInterruptException:
        pass