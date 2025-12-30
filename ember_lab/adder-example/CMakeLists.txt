cmake_minimum_required(VERSION 3.20)

# Find Package
find_package(ember CONFIG REQUIRED)

# Executable
add_executable(ember-adder-example ${CMAKE_CURRENT_LIST_DIR}/main.cpp)
target_link_libraries(ember-adder-example PRIVATE ember::ember)
target_include_directories(ember-adder-example PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_compile_definitions(ember-adder-example PRIVATE VERBOSE=1)
set_target_properties(ember-adder-example PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/..")