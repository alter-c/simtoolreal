#!/usr/bin/env python
import rospy
import math
from geometry_msgs.msg import Pose, PoseStamped, Quaternion

def talker():
    # 初始化节点
    rospy.init_node('pose_publisher_node', anonymous=True)
    
    # 定义发布者
    curr_pub = rospy.Publisher('/robot_frame/current_object_pose', PoseStamped, queue_size=1)
    goal_pub = rospy.Publisher('/robot_frame/goal_object_pose', Pose, queue_size=1)
    
    rate = rospy.Rate(10) # 10Hz 持续发布

    # 预设四元数：绕 x 轴旋转 90 度
    # 公式：[sin(theta/2), 0, 0, cos(theta/2)]
    q = Quaternion(
        x = -0.6652841055959126,
        y = 0.697490775612147,
        z = 0.20281342721883897,
        w = -0.17254098213804459
    )
    qc = Quaternion(
        x = 0.40564921392681963, 
        y = 0.36859424222420645,
        z = 0.6436580953764287, 
        w = 0.5341266292707232
    )
    qg = Quaternion(
        x = 0.5777317135190533,
        y = 0.5509842507191253,
        z = -0.4037743694673119,
        w = -0.44677587358215454
    )
    

    while not rospy.is_shutdown():
        # 1. 构造 current_object_pose (PoseStamped)
        current_pose = PoseStamped()
        current_pose.header.stamp = rospy.Time.now()
        current_pose.header.frame_id = "robot" # 根据你的实际 frame 修改
        current_pose.pose.position.x = 0.33
        current_pose.pose.position.y = 0.0
        current_pose.pose.position.z = -0.05
        current_pose.pose.orientation = qc

        # 2. 构造 goal_object_pose (Pose)
        goal_pose = Pose()
        goal_pose.position.x = 0.35
        goal_pose.position.y = 0.0
        goal_pose.position.z = 0.1
        goal_pose.orientation = q

        # 发布数据
        curr_pub.publish(current_pose)
        goal_pub.publish(goal_pose)

        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass