cmake_minimum_required(VERSION 2.8.3)
project(compressed_image_transport)

find_package(OpenCV REQUIRED)

find_package(CUDA 10.1 REQUIRED)

find_path(CUDA_NVJPEG_INCLUDE_DIR nvjpeg.h
    PATHS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES include)

find_library(CUDA_NVJPEG_LIBRARY nvjpeg
  PATHS ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64)

find_package(catkin REQUIRED COMPONENTS cv_bridge dynamic_reconfigure image_transport)

# generate the dynamic_reconfigure config file
generate_dynamic_reconfigure_options(cfg/CompressedPublisher.cfg cfg/CompressedSubscriber.cfg)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS cv_bridge dynamic_reconfigure image_transport
  DEPENDS OpenCV
)

include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${CUDA_NVJPEG_INCLUDE_DIR})

add_library(${PROJECT_NAME} src/compressed_publisher.cpp src/compressed_subscriber.cpp src/manifest.cpp src/NvJpegCodec.cc)
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_gencfg)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_NVJPEG_LIBRARY})

class_loader_hide_library_symbols(${PROJECT_NAME})

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

# add xml file
install(FILES compressed_plugins.xml
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
