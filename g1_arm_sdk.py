import time
import sys
import threading
import numpy as np

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread


kPi = 3.141592654
NUM_MOTOR = 29
NUM_ARM   = 17


class G1JointIndex:
    # Left leg
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleB = 4
    LeftAnkleRoll = 5
    LeftAnkleA = 5

    # Right leg
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleB = 10
    RightAnkleRoll = 11
    RightAnkleA = 11

    WaistYaw = 12
    WaistRoll = 13  # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistA = 13     # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistPitch = 14 # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistB = 14     # NOTE: INVALID for g1 23dof/29dof with waist locked

    # Left arm
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20   # NOTE: INVALID for g1 23dof
    LeftWristYaw = 21     # NOTE: INVALID for g1 23dof

    # Right arm
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27  # NOTE: INVALID for g1 23dof
    RightWristYaw = 28    # NOTE: INVALID for g1 23dof

    kNotUsedJoint = 29 # NOTE: Weight


# Arm Control

class Custom:
    ARM_JOINTS = [
        G1JointIndex.LeftShoulderPitch,  G1JointIndex.LeftShoulderRoll,
        G1JointIndex.LeftShoulderYaw,    G1JointIndex.LeftElbow,
        G1JointIndex.LeftWristRoll,      G1JointIndex.LeftWristPitch,
        G1JointIndex.LeftWristYaw,
        G1JointIndex.RightShoulderPitch, G1JointIndex.RightShoulderRoll,
        G1JointIndex.RightShoulderYaw,   G1JointIndex.RightElbow,
        G1JointIndex.RightWristRoll,     G1JointIndex.RightWristPitch,
        G1JointIndex.RightWristYaw,
        G1JointIndex.WaistYaw,
        G1JointIndex.WaistRoll,
        G1JointIndex.WaistPitch,
    ]

    ZERO_POSE = np.array([
        0.0,  kPi/9, 0.0, kPi/2, 0.0, 0.0, 0.0,
        0.0, -kPi/9, 0.0, kPi/2, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ])

    INIT_POSE = np.array([
        0.0, 0.0,    0.0, 0.0, kPi/2, 0.0, 0.0,
        0.0, -kPi/9, 0.0, kPi/2, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ])

    def __init__(self):
        ChannelFactoryInitialize(0)

        self.control_dt_ = 0.02
        self.kp          = 60.0
        self.kd          = 1.5
        self.alpha        = 0.05 # 平滑滤波参数
        self.arrival_tol  = 0.01

        self.crc     = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()

        self._lock        = threading.Lock()
        # 当前关节角度及角速度
        self._motor_q     = np.zeros(NUM_MOTOR)
        self._motor_dq    = np.zeros(NUM_MOTOR)
        # 目标关节角度及当前发布指令角度
        self._target_q      = np.zeros(NUM_ARM)
        self._current_cmd = np.zeros(NUM_ARM)
        self._sdk_weight  = 1.0

        self._arrived     = threading.Event()
        self._state_ready = threading.Event()
        self._stop_event  = threading.Event()
        self._thread      = None

    def Init(self):
        self._pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self._pub.Init()
        self._sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._sub.Init(self._LowStateHandler, 10)
        print(f"[ArmController] G1 arm7 sdk init done, freq={1/self.control_dt_:.0f}Hz")

    ### 调用接口
    def get_state(self) -> dict:
        """返回当前手臂关节角与角速度, 顺序与索引基于ARM_JOINTS"""
        with self._lock:
            return {
                "q":  np.array([self._motor_q[j]  for j in self.ARM_JOINTS]),
                "dq": np.array([self._motor_dq[j] for j in self.ARM_JOINTS]),
            }

    def set_target(self, joint_angles) -> None:
        """设置目标关节角(弧度), 非阻塞用于连续动作轨迹"""
        target = np.asarray(joint_angles, dtype=float)
        assert target.shape == (NUM_ARM,), \
            f"[ArmController] Target shape error: {target.shape}, expected {NUM_ARM}."
        with self._lock:
            self._target_q = target.copy()
            self._arrived.clear()

    def control(self, joint_angles, timeout: float = 10.0) -> bool:
        """设置目标关节角(弧度), 阻塞等待适用于离散动作序列"""
        self.set_target(joint_angles)
        return self._arrived.wait(timeout=timeout)

    ### 生命周期 
    def Start(self):
        print("[ArmController] Wait for init LowState ...")
        self._state_ready.wait()
        self._stop_event.clear()
        self._arrived.clear()
        self._sdk_weight = 1.0

        with self._lock:
            init_q = np.array([self._motor_q[j] for j in self.ARM_JOINTS])
            self._current_cmd = init_q.copy()
            self._target_q      = init_q.copy()

        self._thread = RecurrentThread(
            interval=self.control_dt_, 
            target=self._LowCmdWrite, 
            name="control"
        )
        self._thread.Start()
        print("[ArmController] Control thread started.")

    def Stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.Wait()
        self._thread = None
        self._stop_event.clear()
        print("[ArmController] Control thread stop.")

    def Release(self, release_sec: float = 3.0):
        self.control(joint_angles=self.ZERO_POSE, timeout=release_sec)
        steps = int(release_sec / self.control_dt_)
        for i in range(steps, -1, -1):
            self._sdk_weight = i / steps
            time.sleep(self.control_dt_)
        self.Stop()
        print("[ArmController] Release Done!")

    ### 内部实现
    def _LowStateHandler(self, msg: LowState_):
        with self._lock:
            for i in range(NUM_MOTOR):
                self._motor_q[i]  = msg.motor_state[i].q
                self._motor_dq[i] = msg.motor_state[i].dq
        self._state_ready.set()

    def _LowCmdWrite(self):
        if self._stop_event.is_set():
            return

        # 指数平滑滤波
        with self._lock:
            target = self._target_q.copy()
        error = target - self._current_cmd
        self._current_cmd += self.alpha * error

        if not self._arrived.is_set() and np.max(np.abs(error)) < self.arrival_tol:
            self._arrived.set()

        self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = self._sdk_weight
        for i, joint in enumerate(self.ARM_JOINTS):
            self.low_cmd.motor_cmd[joint].q   = self._current_cmd[i]
            self.low_cmd.motor_cmd[joint].dq  = 0.0
            self.low_cmd.motor_cmd[joint].tau = 0.0
            self.low_cmd.motor_cmd[joint].kp  = self.kp
            self.low_cmd.motor_cmd[joint].kd  = self.kd

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self._pub.Write(self.low_cmd)

if __name__ == '__main__':
    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    custom = Custom()
    custom.Init()
    custom.Start()

    pose_zero = np.zeros(NUM_ARM)
    custom.control(pose_zero, timeout=10.0)

    pose_a = np.array([
        0.0,  kPi/3, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, -kPi/9, 0.0, kPi/2, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ])
    custom.control(pose_a, timeout=10.0)

    t0 = time.time()
    while time.time() - t0 < 5.0:
        t = time.time() - t0
        pose = pose_a.copy()
        pose[0] = (kPi / 4) * np.sin(2 * kPi * 0.5 * t)
        custom.set_target(pose)
        time.sleep(custom.control_dt_)

    input("Press Enter to exit...")
    custom.Release()