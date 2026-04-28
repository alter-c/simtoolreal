# Installation

## Prerequisites

Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install [torch for jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048):
```bash
wget "https://developer.download.nvidia.cn/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl"
uv pip install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
```

Install [torchvision](https://github.com/pytorch/vision):
```bash
uv pip install "torchvision @ git+https://github.com/pytorch/vision.git@v0.16.1"
cd ../ # for test
```

Install system packages:
```bash
sudo apt-get install -y libxml2-dev libxslt1-dev zlib1g-dev
```

## Main Environment

```bash
# Create a virtual environment with Python 3.8
uv venv --python 3.8

source .venv/bin/activate

# set CYCLONEDDS_HOME for install unitree-sdk
export CYCLONEDDS_HOME=/home/unitree/cyclonedds_ws/install/cyclonedds

# Install the project and all dependencies
uv pip install -e "[.real]"

# Install this repo's rl_games
cd rl_games
uv pip install -e .
cd -
```

## Sim2Real Env

For real-world deployment, we need ROS1 Noetic. This can either be installed globally or using RoboStack.

### Global ROS

Follow the instructions at https://wiki.ros.org/noetic/Installation/Ubuntu to install ROS1 Noetic globally. Then source the ROS environment variables in your shell:

```bash
source /opt/ros/noetic/setup.bash
```

Now, you can use this global ROS setup with the virtual environment above.

You may need to run:

```bash
uv pip install rospkg
```

### RoboStack

Note that the RoboStack option requires a separate environment because it requires Python 3.11+. Follow the instructions at https://robostack.github.io/noetic.html to create an environment with ROS1 Noetic. Then follow the installation instructions above for the main environment (skip the venv creation and Isaac Gym installation).
