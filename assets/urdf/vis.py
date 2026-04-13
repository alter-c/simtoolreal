import time
import numpy as np
import viser
from viser.extras import ViserUrdf
import yourdfpy

server = viser.ViserServer(
    host="0.0.0.0",
    port=8000,
    verbose=True,
)

urdf = yourdfpy.URDF.load("./g1_description/g1_29dof_with_linkerhand.urdf")

viser_urdf = ViserUrdf(
    server,
    urdf_or_path=urdf,
)

# 设置初始关节配置（全零）
num_joints = len(viser_urdf.get_actuated_joint_limits())
viser_urdf.update_cfg(np.zeros(num_joints))

# 添加地面网格
server.scene.add_grid("/grid", width=2, height=2)

# 保持服务器运行
while True:
    time.sleep(1.0)