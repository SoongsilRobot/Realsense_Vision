# D405 + YOLO11 ROS2 Pipeline

- /camera/color/image_raw (sensor_msgs/Image)
- /detected/center_distance_m (std_msgs/Float32)

## Setup

sudo apt install ros-jazzy-librealsense2 librealsense2-utils \
                 ros-jazzy-cv-bridge ros-jazzy-sensor-msgs

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt

## Run

source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
python3 main.py --onnx /path/to/yolo11n.onnx --annotated
