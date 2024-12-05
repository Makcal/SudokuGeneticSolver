cmake_minimum_required(VERSION 3.29)
project(lab1)

set(CMAKE_CXX_STANDARD 23)

add_compile_options(-O2)
add_executable(sudoku sudoku.cpp)
add_executable(dasha dasha.cpp)
add_executable(two_fit sudoku_two_fitness.cpp)
