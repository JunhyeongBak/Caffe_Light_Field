cmake_minimum_required(VERSION 3.1.0)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

project(webcam)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

#hunter_add_package(Qt COMPONENTS qtmultimedia) # Qt5MultimediaWidgets
set(QT_LIBRARIES Qt5::Core Qt5::Widgets Qt5::Gui Qt5::Multimedia Qt5::MultimediaWidgets)

#Libraries required. Probably the same as above minus the '::'.     find_package() will be called on these

set(QT_LIBRARIES_REQUIRED Qt5Core Qt5Widgets Qt5Gui Qt5Multimedia Qt5MultimediaWidgets)

if(CMAKE_VERSION VERSION_LESS "3.7.0")
    set(CMAKE_INCLUDE_CURRENT_DIR ON)
endif()

find_package(OpenCL REQUIRED)
find_package(OpenCV REQUIRED)

#find_package(Qt5 COMPONENTS Core REQUIRED)
foreach(QT_LIBRARIES_REQUIRED ${QT_LIBRARIES_REQUIRED})
    find_package( ${QT_LIBRARIES_REQUIRED} REQUIRED )
endforeach()

include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

add_executable(webcam
    main2.cpp
    ocl.cpp
    ocl.h
    constrained_adaptive_entropy.cl
    wta_lf.cl
)

#target_link_libraries(webcam Qt5::Gui Qt5::Core)
target_link_libraries(webcam ${QT_LIBRARIES} ${OpenCV_LIBS} ${OpenCL_LIBRARY})
