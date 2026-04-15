include(FetchContent)

## == OSQP ==
FetchContent_Declare(
    osqp
    GIT_REPOSITORY https://github.com/osqp/osqp.git
    GIT_TAG v0.6.3
    OVERRIDE_FIND_PACKAGE
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release
)
FetchContent_MakeAvailable(osqp)
add_library(osqp::osqp ALIAS osqp) # needed because osqp-eigen uses osqp::osqp

# Set up OSQP include directories
set(OSQP_INCLUDE_DIRS 
    ${osqp_SOURCE_DIR}/include
    ${osqp_BINARY_DIR}
    CACHE INTERNAL "OSQP include directories"
)
include_directories(${OSQP_INCLUDE_DIRS})
message(STATUS "OSQP built: ${osqp_SOURCE_DIR}")
message(STATUS "OSQP include dirs: ${OSQP_INCLUDE_DIRS}")

## == OSQP-Eigen ==
FetchContent_Declare(
    osqp-eigen
    GIT_REPOSITORY https://github.com/robotology/osqp-eigen.git
    GIT_TAG v0.6.4
    OVERRIDE_FIND_PACKAGE
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release
        -DBUILD_SHARED_LIBS=ON
        -DCMAKE_POLICY_DEFAULT_CMP0002=NEW
)
set(__ADD_UNINSTALL_TARGET_INCLUDED TRUE) # prevent osqp-eigen from creating
FetchContent_MakeAvailable(osqp-eigen)
message(STATUS "OSQP-Eigen built: ${osqp-eigen_SOURCE_DIR}")

# Set up OSQP-Eigen include directories
set(OSQP_EIGEN_INCLUDE_DIRS 
    ${osqp-eigen_SOURCE_DIR}/include
    ${osqp-eigen_BINARY_DIR}
    CACHE INTERNAL "OSQP-Eigen include directories"
)
include_directories(${OSQP_EIGEN_INCLUDE_DIRS})

## == Unitree Legged SDK2 ==
FetchContent_Declare(
    unitree_sdk2
    GIT_REPOSITORY https://github.com/unitreerobotics/unitree_sdk2.git
    GIT_TAG eed0898b8d63d83406f7f460a827fa378dd3e631
    OVERRIDE_FIND_PACKAGE
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release
        -DBUILD_EXAMPLES=OFF
)
FetchContent_MakeAvailable(unitree_sdk2)
message(STATUS "Unitree SDK2 built: ${unitree_sdk2_SOURCE_DIR}")