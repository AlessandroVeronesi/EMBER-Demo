cmake_minimum_required(VERSION 3.20)

# Find the EMBER package (CONFIG mode, build-tree or install-tree)
find_package(ember CONFIG REQUIRED)

# Project
project(ember_demo LANGUAGES CXX)
add_executable(ember_demo main.cpp)
target_link_libraries(ember_demo PRIVATE ember::ember)
