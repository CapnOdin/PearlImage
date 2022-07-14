cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(Pearl-Image VERSION 0.1.0.0)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O2")

set(CMAKE_CXX_FLAGS_DEBUG "-g")

configure_file(PearlImageConfig.h.in PearlImageConfig.h)

add_library(Image
	src/Image/image.cpp
	src/Image/image.hpp
)

add_library(Json
	src/Json/json.cpp
	src/Json/json.hpp
	src/Json/nlohmann/json.hpp
)

# add the executable
add_executable(Pearl-Image src/main.cpp)

#SET("OpenCV_DIR" "C:/data/libs/vcpkg/packages/opencv_x64-windows")
SET("OpenCV_DIR" "C:/data/libs/vcpkg/packages/opencv4_x64-windows/share/opencv")
find_package(OpenCV REQUIRED)

MESSAGE(STATUS "Include dirs ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "LINK LIBRARIES ${OpenCV_LIBS}")

target_link_libraries(Image
	${OpenCV_LIBS}
)

target_link_libraries(Pearl-Image 
	PRIVATE Image
	PRIVATE Json
	PRIVATE ${OpenCV_LIBS}
)

target_include_directories(Image PUBLIC
	"${PROJECT_BINARY_DIR}"
	${OpenCV_INCLUDE_DIRS}
	"src/Json/"
	"src/Json/nlohmann/"
	"src/ThrPool/"
)

# add the binary tree to the search path for include files
# so that we will find LinkTransmitterConfig.h
target_include_directories(Pearl-Image PUBLIC
	"${PROJECT_BINARY_DIR}"
	"src/Image/"
	"src/Json/"
	"src/Json/nlohmann/"
	${OpenCV_INCLUDE_DIRS}
)

#install(TARGETS Pearl-Image RUNTIME DESTINATION bin)
