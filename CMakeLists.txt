cmake_minimum_required(VERSION 3.8)
project(mocap_teleop)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install Python package
ament_python_install_package(${PROJECT_NAME})

# Install Python node executables
install(PROGRAMS
  mocap_teleop/modes/online_node.py
  mocap_teleop/teleop_node.py
  mocap_teleop/modes/offline_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install config files
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

ament_package()