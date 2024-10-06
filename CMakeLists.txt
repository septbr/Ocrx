cmake_minimum_required(VERSION 3.30)

project(Ocrx VERSION 1.2.0)


add_library(Ocrx)
target_link_directories(Ocrx PRIVATE "./onnxruntime/lib")
target_link_libraries(Ocrx PRIVATE onnxruntime onnxruntime_providers_shared)
add_custom_command(TARGET Ocrx POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/lib/onnxruntime.dll"
    "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/lib/onnxruntime_providers_shared.dll"
    $<TARGET_FILE_DIR:Ocrx>
)
target_include_directories(Ocrx PRIVATE "./onnxruntime/include")
target_sources(Ocrx PRIVATE "./Ocrx/OcrxOnnx.cpp")

add_executable(Example)
target_link_directories(Example PRIVATE "./onnxruntime/lib" "./example/opencv/lib")
target_link_libraries(Example PRIVATE Ocrx onnxruntime debug opencv_world4100d optimized opencv_world4100)
add_custom_command(TARGET Example POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/lib/onnxruntime.dll"
    "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/lib/onnxruntime_providers_shared.dll"
    "${CMAKE_CURRENT_SOURCE_DIR}/example/opencv/lib/opencv_world4100.dll"
    "${CMAKE_CURRENT_SOURCE_DIR}/example/opencv/lib/opencv_world4100d.dll"
    $<TARGET_FILE_DIR:Example>
)
target_include_directories(Example PRIVATE "./Ocrx" "./example/opencv/include")
target_sources(Example PRIVATE "./example/example.cpp")

