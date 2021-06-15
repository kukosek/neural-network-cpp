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


add_executable(train main.cpp NeuralNetwork.cpp)
target_compile_features(train PRIVATE cxx_range_for)

target_link_libraries (train Eigen3::Eigen)


# Pass MNIST data directory to main.cpp
target_compile_definitions(train PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")
target_link_libraries (train Eigen3::Eigen)
target_link_libraries(train OpenMP::OpenMP_CXX)


# gui
find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK REQUIRED gtkmm-3.0)

add_executable(gui gui.cpp NeuralNetwork.cpp)
target_compile_features(gui PRIVATE cxx_range_for)

target_compile_definitions(gui PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")
target_link_libraries (gui Eigen3::Eigen)
target_link_libraries(gui OpenMP::OpenMP_CXX)

INCLUDE_DIRECTORIES(${GTK_INCLUDE_DIRS} ${MNIST_INCLUDE_DIR})
LINK_DIRECTORIES(${GTK_LIBRARY_DIRS})

# Add other flags to the compiler
ADD_DEFINITIONS(${GTK_CFLAGS_OTHER})

TARGET_LINK_LIBRARIES(gui ${GTK_LIBRARIES})
