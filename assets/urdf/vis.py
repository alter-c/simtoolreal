import time
import numpy as np
import viser
from viser.extras import ViserUrdf
import yourdfpy

server = viser.ViserServer(
    host="0.0.0.0",
    port=8050,
    verbose=True,
)

urdf = yourdfpy.URDF.load("./unitree_linkerhand_description/g1_29dof_left_linkerhand_adjusted.urdf")

viser_urdf = ViserUrdf(
    server,
    urdf_or_path=urdf,
)

num_joints = len(viser_urdf.get_actuated_joint_limits())
viser_urdf.update_cfg(np.zeros(num_joints))

server.scene.add_grid("/grid", width=2, height=2)

while True:
    time.sleep(1.0)