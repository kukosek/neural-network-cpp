cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 17)

PROJECT(neural-net)

# .. -> hint, that the mnist package is one directory level above.
# When using just "find_package(MNIST REQUIRED)", "MNIST_DIR"
#    cmake variable has to be set correctly.
find_package(MNIST PATHS mnist/)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP)
if(NOT MNIST_FOUND)
    message(FATAL_ERROR "MNIST loader could not be found. It is available under https://github.com/wichtounet/mnist")
endif(NOT MNIST_FOUND)


include_directories(${MNIST_INCLUDE_DIR})
add_executable(${PROJECT_NAME} main.cpp NeuralNetwork.cpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_range_for)

# Pass MNIST data directory to main.cpp
target_compile_definitions(${PROJECT_NAME} PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")
target_link_libraries (${PROJECT_NAME} Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
