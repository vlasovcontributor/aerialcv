project(coursework)

add_library (bigint bigint.h bigint.cpp)
target_include_directories (bigint PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


add_library (parameters parameters.h parameters.cpp)
target_include_directories (parameters PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library (curves curves.h curves.cpp)
target_include_directories (curves PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} "D:/lib/opencv_contrib/modules/ximgproc/include")
target_link_libraries(curves "D:/lib/opencv-4.5.2/build/lib/Release/opencv_ximgproc452.lib")
target_link_options(curves INTERFACE -lopencv_ximgproc452)

add_executable(coursework main.cpp)
target_link_libraries(coursework ${OpenCV_LIBS} bigint curves parameters "D:/lib/opencv_contrib/modules/ximgproc/include")
target_link_options(coursework INTERFACE -lopencv_ximgproc452)