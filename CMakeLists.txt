cmake_minimum_required(VERSION 3.8)
project(tts_ros)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install the Python package
ament_python_install_package(${PROJECT_NAME})

# Install the node as an executable
install(PROGRAMS
  tts_ros/tts_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME tts_node
)

# Install the config/ directory to share/${PROJECT_NAME}/config/
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

ament_package()

