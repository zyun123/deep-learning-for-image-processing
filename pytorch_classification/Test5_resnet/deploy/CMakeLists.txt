cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(restnet)
set(Torch_DIR /home/zy/Downloads/libtorch-cxx11-abi-shared-with-deps-1.9.1+cu111/libtorch/share/cmake/Torch)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TorchVision REQUIRED)

add_executable(resnet resnet18.cpp)

target_link_libraries(resnet "${TORCH_LIBRARIES}" "${OpenCV_LIBRARIES}" TorchVision::TorchVision)

set_property(TARGET resnet PROPERTY CXX_STANDARD 14 )