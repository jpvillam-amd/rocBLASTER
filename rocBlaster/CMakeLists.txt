cmake_minimum_required(VERSION 3.22)

if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to which ROCM has been installed")
    elseif(DEFINED ENV{HIP_PATH})
        set(ROCM_PATH "$ENV{HIP_PATH}/.." CACHE PATH "Path to which ROCM has been installed")
    else()
        set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to which ROCM has been installed")
    endif()
endif()
set(CXX "${ROCM_PATH}/bin/hipcc")

project(Tunner)
# combined list
find_package(rocblas REQUIRED)
find_package(HIP REQUIRED)
find_package(Threads REQUIRED)

find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

#add_executable(rocBlasFinder rocBlasFinder.cpp)
pybind11_add_module(rocBlasFinder csrc/rocBlasFinder.cpp)
set( sample_list_all rocBlasFinder )

foreach( exe ${sample_list_all} )
  target_link_libraries( ${exe} PRIVATE roc::rocblas Threads::Threads )

  set_target_properties( ${exe} PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
  )

  target_compile_definitions( ${exe} PRIVATE ROCM_USE_FLOAT16 )

    if( CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # GCC or hip-clang needs specific flags to turn on f16c intrinsics
    target_compile_options( ${exe} PRIVATE -mf16c )
    target_compile_definitions( ${exe} PRIVATE ROCBLAS_INTERNAL_API )
  endif( )

endforeach( )

foreach( exe ${sample_list_all} )
  target_link_libraries( ${exe} PRIVATE hip::device )
  target_link_libraries( ${exe} PUBLIC pybind11::embed )
endforeach( )
