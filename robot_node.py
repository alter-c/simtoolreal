# ROS Bridge for simtoolreal (just left arm)
import numpy as np
import threading
import rospy
import sys
import copy
import signal
import time
from sensor_msgs.msg import JointState
from g1_arm_sdk import Custom
from linkerhand_sdk import O6_DirectJointController, scale, unscale, joint_max

class ArmROSBridge:
    """
    ROS通信, 与控制逻辑解耦。
    - 发布状态: 调用 arm_controller.get_state() → /unitree/joint_states
    - 监听动作: /unitree/joint_cmd → 调用 arm_controller.set_target()
    """

    def __init__(self, arm_controller: Custom, hand_controller: O6_DirectJointController, publish_hz: float = 100.0):
        rospy.init_node("arm_arm_controller", anonymous=False)
        self._ctrl = arm_controller
        self._ctrl.Init()
        self._hand_ctrl = hand_controller

        self.rate = rospy.Rate(publish_hz)
        self._arm_pub_thread = None
        self._hand_pub_thread = None

        self._arm_state_pub = rospy.Publisher("/unitree/joint_states", JointState, queue_size=1)
        self._arm_action_sub = rospy.Subscriber("/unitree/joint_cmd", JointState, self._arm_action_callback)

        self._hand_state_pub = rospy.Publisher("/linkerhand/joint_states", JointState, queue_size=1)
        self._hand_action_sub = rospy.Subscriber("/linkerhand/joint_cmd", JointState, self._hand_action_callback)

    def start(self):
        self._ctrl.Start()
        self._arm_pub_thread = threading.Thread(
            target=self._arm_publish_loop, 
            daemon=True
        )
        self._hand_pub_thread = threading.Thread(
            target=self._hand_publish_loop, 
            daemon=True
        )
        self._arm_pub_thread.start()
        self._hand_pub_thread.start()

    def stop(self):
        self._ctrl.Stop()

    def release(self):
        self._ctrl.Release()
        self._hand_ctrl.release_hand("left")
        time.sleep(2)

    def _arm_publish_loop(self):
        while not rospy.is_shutdown():
            state = self._ctrl.get_state()
            self._arm_publish_state(state)
            self.rate.sleep()

    def _arm_publish_state(self, state: dict):
        # only pub left arm joint
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.position = state["q"][:7].tolist()
        msg.velocity = state["dq"][:7].tolist()
        self._arm_state_pub.publish(msg)

    def _hand_publish_loop(self):
        while not rospy.is_shutdown():
            state = self._hand_ctrl.get_state()
            self._hand_publish_state(state)
            self.rate.sleep()

    def _hand_publish_state(self, state: dict):
        # only pub left hand joint
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.position = scale(state["q"][:6], joint_max).tolist()
        msg.velocity = state["dq"][:6].tolist()
        self._hand_state_pub.publish(msg)

    def _arm_action_callback(self, msg):
        action_msg = copy.copy(msg)
        target_joint = self._ctrl.ZERO_POSE 
        left_arm_joint = np.array(action_msg.position)
        rospy.loginfo(f"Target arm joint: {left_arm_joint}")
        target_joint[:7] = left_arm_joint
        self._ctrl.set_target(target_joint)
    
    def _hand_action_callback(self, msg):
        action_msg = copy.copy(msg)
        left_hand_joint = unscale(np.array(action_msg.position), joint_max)
        rospy.loginfo(f"Target hand joint: {left_hand_joint}")
        self._hand_ctrl.set_joints("left", left_hand_joint)

def signal_handler(sig, frame):
    rospy.loginfo("User interrupt")
    rospy.signal_shutdown("User interrupt.")

signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    custom = Custom()
    hand_control = O6_DirectJointController(left_can_port="can1", right_can_port=None, fps=50.0)
    ros_bridge = ArmROSBridge(custom, hand_control, publish_hz=50.0)
    def shutdown_bridge():
        ros_bridge.release()
        print("[Bridge] Safely stopped.")

    rospy.on_shutdown(shutdown_bridge)

    ros_bridge.start()
    custom.set_target(custom.INIT_POSE)
    rospy.spin()
