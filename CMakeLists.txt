cmake_minimum_required(VERSION 3.1.0)
project(cpp_flow)
add_compile_options(-std=c++11)

find_package(Boost REQUIRED)    # only found Boost_INCLUDE_DIRS, but not Boost_LIBRARIES
find_package(PythonLibs 2.7 REQUIRED)
find_package(OpenCV REQUIRED PATHS /usr/local NO_DEFAULT_PATH)

include_directories(src ${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/cpplib)  # set output lib path
add_library(${PROJECT_NAME} SHARED cppsrc/${PROJECT_NAME}.cpp)

# add_executable(${PROJECT_NAME} cppsrc/${PROJECT_NAME}.cpp)

set(Boost_LIBRARIES /usr/lib/x86_64-linux-gnu/libboost_python-py27.so.1.58.0)
set(BoostNumpy_LIBRARIES /usr/local/lib64/libboost_numpy.so)

target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES} ${BoostNumpy_LIBRARIES} ${OpenCV_LIBRARIES})
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")   # remove prefix
