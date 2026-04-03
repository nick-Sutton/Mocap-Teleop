## Run Steps
> [!IMPORTANT]
> When Streaming ensure that the up-axis is set the y-axis in Motive. Coordinate transformations assume the y-axis is the up direction.

### Offline 
1. run `ros2 launch mocap_teleop mocap_teleop.launch.py`
2. run the camera node

### Online
1) Ensure that data streaming is enabled in Motive
2) run `ros2 launch mocap_teleop mocap_teleop.launch.py`
3) run the camera node



## Configuration
All Configuration values can be edited in the `config.toml` file located at `config/config.toml`

### Motion Capture 
```
[Mocap]
camera_frequency = 240

[Rigid Bodies]
root_name = Root
root_stream_id = 1
left_foot_name = LFoot
left_foot_stream_id = 2
right_foot_name = RFoot
right_foot_stream_id = 3
```
### Networking
```
[Networking]
use_multicast = false # Unicast = False / Multicast = True
server_address = "192.168.0.26" # Address in the Motive Software
local_address = "0.0.0.0" # Client Computers IP (Make sure they are on the same subnet)
multicast_address = "239.255.42.99"
command_port = 1510
data_port = 1511
```

### Whole-body Controller
```
[Controller]
controller_path = "../mpac/mpac_go2/"
interface_subpath = "../mpac/mpac_go2/atnmy/"
```

### Machine Learning Models
```
[Learning]
gait_classifier_path = "../mpac/mpac_go2/"
```

## Structure
The mocap_teleop package is broken into four nodes. Three of the nodes are hardware drivers that send data to the teleop node.

| Hardware Driver |     Data         | Teleop      |
| --------------- | ---------------- | ----------- | 
| motive_network  | Human Pose       |             |
| mpac_bridge     | Robot Telemetry  | teleop_node |
| realsense_camera| Camera and depth |             |