#!/usr/bin/env python3
import sys
import os
import time
import numpy as np
import threading  # 只需要 threading，不需要 multiprocessing
import pathlib

# --- SDK 路径配置 (保持不变) ---
linker_hand_sdk_path = './third_party/linkerhand-python-sdk'
sys.path.append(linker_hand_sdk_path)
sys.path.append(str(pathlib.Path(os.getcwd()).parent))
from LinkerHand.linker_hand_api import LinkerHandApi

O6_Num_Motors = 6  # 关节数量

class O6_DirectJointController:
    """直接关节控制版 Linker Hand O6 双手控制器 (多线程版)"""
    
    def __init__(
        self, 
        left_can_port: str = "can1", 
        right_can_port: str = "can0", 
        fps: float = 50.0
    ):
        print(f"[O6] Initializing (Threaded) — left_can:{left_can_port} right_can:{right_can_port} fps:{fps}")
        
        # --- 1. 初始化常量 ---
        self._POSE_RELEASE = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # 全部伸直
        self._POSE_OPEN = np.array([0.8, 0.0, 1.0, 1.0, 1.0, 1.0])     # 张开
        self._POSE_CLOSE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])   # 握拳
        self._HAND_SIDES = ("left", "right", "both")
        
        self.fps = fps
        self.left_can_port = left_can_port
        self.right_can_port = right_can_port
        
        # --- 2. 移除共享内存，直接使用普通 Numpy 数组 ---
        # 指令数组 [左手6个, 右手6个]
        self._joint_cmd = np.zeros(O6_Num_Motors * 2)
        self._joint_cmd[:] = np.tile(self._POSE_RELEASE, 2)  # 初始化为握拳
        
        # 状态数组 (用于存储硬件反馈)
        self._left_state = np.zeros(O6_Num_Motors)
        self._right_state = np.zeros(O6_Num_Motors)
        self._left_vel = np.zeros(O6_Num_Motors)
        self._right_vel = np.zeros(O6_Num_Motors)
        
        # --- 3. 初始化 API ---
        self._left_api, self._right_api = self._init_api(left_can_port, right_can_port)
        
        # --- 4. 启动线程 ---
        # 状态订阅线程 (保持 daemon=True)
        threading.Thread(target=self._subscribe_state, daemon=True).start()
        
        # 控制循环线程 (原来是 Process，现在改为 Thread)
        # 注意：target 直接指向 self._control_loop，不需要传 self
        threading.Thread(target=self._control_loop, daemon=True).start()
        
        print("[O6] Controller (Threaded) ready.\n")

    # ==================================================================
    # 公开接口 (Open/Close/Release/Set) - 这部分代码逻辑完全保持不变
    # ==================================================================
    def open_hand(self, side: str):
        self._apply_pose(side, self._POSE_OPEN)
        print(f"[O6] open_hand → {side}")

    def close_hand(self, side: str):
        self._apply_pose(side, self._POSE_CLOSE)
        print(f"[O6] close_hand → {side}")

    def release_hand(self, side: str = "both"):
        self._apply_pose(side, self._POSE_RELEASE)
        print(f"[O6] release_hand → {side}")

    def set_joints(self, side: str, pose: list):
        assert len(pose) == O6_Num_Motors, f"pose 长度必须为 {O6_Num_Motors}"
        self._apply_pose(side, np.array(pose))

    def get_state(self) -> dict:
        """读取当前状态"""
        return {
            # "left": self._left_state.copy(),
            # "right": self._right_state.copy()
            "q": np.concatenate((
                self._left_state.copy(),
                self._right_state.copy()
            )),
            "dq": np.concatenate((
                self._left_vel.copy(),
                self._right_vel.copy()
            )),
        }

    # ==================================================================
    # 内部实现 (Internal Implementation)
    # ==================================================================
    def _apply_pose(self, side: str, pose: np.ndarray):
        """将姿态写入指令槽"""
        assert side in self._HAND_SIDES, f"side 必须为 {self._HAND_SIDES}"
        
        # 这里直接操作 numpy 数组，不需要 get_lock()
        if side in ("right", "both"):
            self._joint_cmd[O6_Num_Motors:] = pose
        if side in ("left", "both"):
            self._joint_cmd[:O6_Num_Motors] = pose

    def _init_api(self, left_can_port, right_can_port):
        """初始化 API (改为实例方法或保持静态，这里改为实例方法更简洁)"""
        left_api = (LinkerHandApi(hand_joint='O6', hand_type="left", can=left_can_port) 
                   if left_can_port else None)
        right_api = (LinkerHandApi(hand_joint='O6', hand_type="right", can=right_can_port) 
                    if right_can_port else None)
        return left_api, right_api

    def _subscribe_state(self):
        """后台线程：持续从硬件读取状态"""
        print("[O6] State subscribe thread started.")
        last_arr = self._POSE_RELEASE
        while True:
            for api, state_arr, vel_arr, label in (
                (self._left_api, self._left_state, self._left_vel, "left"),
                (self._right_api, self._right_state, self._right_vel, "right")
            ):
                if api is None:
                    continue
                msg = api.get_state()
                msg_v = api.get_speed()
                if msg is not None and len(msg) == O6_Num_Motors:
                    # 直接更新 numpy 数组
                    state_arr[:] = np.array(msg) / 255.0
                    direct = np.sign(state_arr - last_arr)
                    vel_arr[:] = direct * np.array(msg_v) / 255.0
                    last_arr = state_arr.copy()
                elif msg is not None:
                    print(f"[O6] Unexpected {label} state length: {len(msg)}")
            time.sleep(0.02) # 高频轮询

    def _control_loop(self):
        """控制循环 (核心改动：直接作为实例方法)"""
        print("[O6] Control Thread started.")
        
        while True:
            t0 = time.time()
            
            # --- 1. 读取指令 (加锁或拷贝，防止读写冲突) ---
            # 使用 copy() 避免在读取过程中被外部修改
            cmd_copy = self._joint_cmd.copy()
            left_q = cmd_copy[:O6_Num_Motors]
            right_q = cmd_copy[O6_Num_Motors:]
            
            # --- 2. 发送指令 ---
            # 转换为 SDK 需要的 0-255 格式
            left_cmd = (left_q * 255).astype(int).tolist()
            right_cmd = (right_q * 255).astype(int).tolist()
            
            if self._left_api is not None:
                self._left_api.finger_move(pose=left_cmd)
            if self._right_api is not None:
                self._right_api.finger_move(pose=right_cmd)
            
            # --- 3. 节流 (Throttling) ---
            # 保持设定的频率
            sleep_time = max(0.0, 1.0 / self.fps - (time.time() - t0))
            time.sleep(sleep_time)

# --- 测试代码 (保持不变) ---

joint_max = [
    1.3,
    0.58,
    1.6,
    1.6,
    1.6,
    1.6,
]
def scale(q, j_max):
    real_joint = (1-q) * j_max
    return real_joint
def unscale(joint, j_max):
    q = 1 - joint / j_max
    return q

if __name__ == '__main__':
    ctrl = O6_DirectJointController(left_can_port="can1", right_can_port=None, fps=50.0)
    try:
        i = 0
        input()
        while True:
            # ctrl.close_hand("left")
            # time.sleep(2)
            # ctrl.open_hand("left")
            # time.sleep(2)
            pose = [abs(np.sin(2 * np.pi * i / 1000))] * 6
            ctrl.set_joints("left", pose)
            i += 1
            time.sleep(0.01)
    except KeyboardInterrupt:
        ctrl.release_hand("left")
        time.sleep(2)
        print("\n[INFO] 用户中断。")