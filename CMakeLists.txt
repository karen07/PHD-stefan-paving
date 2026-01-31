cmake_minimum_required(VERSION 3.13)

project(paving)

enable_language(CUDA)
add_compile_options()
add_link_options()

file(GLOB SRC "src/*.cu")
add_executable(${PROJECT_NAME} ${SRC})

install(TARGETS ${PROJECT_NAME} DESTINATION /usr/bin)
