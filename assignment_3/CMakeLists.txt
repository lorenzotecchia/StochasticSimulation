cmake_minimum_required(VERSION 3.12)
project(PolymerSimulation LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

pybind11_add_module(polymer_engine forces.cpp)  # or your .cpp source

target_link_libraries(polymer_engine PRIVATE Eigen3::Eigen pybind11::module)

# Optimization, flags etc.

if(APPLE)
    # On macOS: ensure correct flags for Python extension modules
    set_target_properties(polymer_engine PROPERTIES
        PREFIX ""  # prevent apple adding lib prefix
        SUFFIX ".so"  # ensure .so suffix
    )
    # Remove version properties (if any)
    set_target_properties(polymer_engine PROPERTIES
        VERSION ""
        SOVERSION ""
    )
		target_link_libraries(polymer_engine PRIVATE ${Python_LIBRARIES})
endif()
