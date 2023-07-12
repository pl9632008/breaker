cmake_minimum_required(VERSION 3.0)
project(dz)

aux_source_directory(./src SRC)

set(CMAKE_BUILD_TYPE Debug)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

include_directories(./include)

include_directories(/usr/local/cuda-11.6/include)
link_directories(/usr/local/cuda-11.6/lib64)

find_package (OpenCV REQUIRED)

include_directories(/wangjiadong/TensorRT-8.2.1.8/include)
link_directories(/wangjiadong/TensorRT-8.2.1.8/lib)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(daozha ${SRC})

target_link_libraries(daozha nvinfer cudart  ${OpenCV_LIBS})
