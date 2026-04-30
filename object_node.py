#!/usr/bin/env python
import rospy
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
    q = Quaternion(x=0.7071, y=-0.7071, z=0.0, w=0.0)
    qc = Quaternion(
        x= -0.6840873608321937,
        y = 0.7290931931342927, 
        z = 0.016322292491217295, 
        w = -0.013460358194198498
    )
    qg = Quaternion(
        x= -0.023853000486620587,
        y = -0.0031776704772307927,
        z = 0.9993195184429601,
        w = -0.027954191761773068
    )
    

    while not rospy.is_shutdown():
        # 1. 构造 current_object_pose (PoseStamped)
        current_pose = PoseStamped()
        current_pose.header.stamp = rospy.Time.now()
        current_pose.header.frame_id = "world" # 根据你的实际 frame 修改
        current_pose.pose.position.x = 0.4
        current_pose.pose.position.y = 0.1
        current_pose.pose.position.z = -0.1
        current_pose.pose.orientation = qc

        # 2. 构造 goal_object_pose (Pose)
        goal_pose = Pose()
        goal_pose.position.x = 0.4
        goal_pose.position.y = 0.15
        goal_pose.position.z = 0.1
        goal_pose.orientation = qc

        # 发布数据
        curr_pub.publish(current_pose)
        goal_pub.publish(goal_pose)

        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass