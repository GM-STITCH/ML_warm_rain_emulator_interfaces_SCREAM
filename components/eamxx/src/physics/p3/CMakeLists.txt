set(P3_SRCS
  eamxx_p3_process_interface.cpp
  eamxx_p3_run.cpp
)


##<++ LL 20260906
# TorchScript support is opt-in so standard EAMxx builds do not require a
# LibTorch installation. When enabled, set Torch_DIR to the directory
# containing TorchConfig.cmake from a compatible LibTorch package.
option(SCREAM_P3_ENABLE_TORCHSCRIPT
       "Enable LibTorch/TorchScript inference in P3"
       OFF)

option(SCREAM_P3_TORCHSCRIPT_SAMPLE_TEST 
       "Enable sample test in P3"
       OFF)

set(SCREAM_P3_TORCH_CUDA_ROOT "" CACHE PATH
    "CUDA toolkit root required by a CUDA-enabled LibTorch installation")

if (SCREAM_P3_ENABLE_TORCHSCRIPT)
  # A CUDA-enabled LibTorch package loads Caffe2's cuda.cmake while
  # find_package(Torch) runs.  Make an explicitly supplied toolkit available
  # through the names used by both modern CMake FindCUDAToolkit and older
  # FindCUDA-based package configurations.  This is intentionally scoped to
  # the optional TorchScript path; it does not turn P3 itself into a CUDA
  # build.
  if (SCREAM_P3_TORCH_CUDA_ROOT)
    if (NOT IS_DIRECTORY "${SCREAM_P3_TORCH_CUDA_ROOT}")
      message(FATAL_ERROR
        "SCREAM_P3_TORCH_CUDA_ROOT is not a directory: "
        "${SCREAM_P3_TORCH_CUDA_ROOT}")
    endif()

    set(CUDAToolkit_ROOT    "${SCREAM_P3_TORCH_CUDA_ROOT}")
    set(CUDA_TOOLKIT_ROOT_DIR "${SCREAM_P3_TORCH_CUDA_ROOT}")
    set(ENV{CUDA_HOME}      "${SCREAM_P3_TORCH_CUDA_ROOT}")

    # Caffe2 enables the CUDA language after locating the toolkit.  Supplying
    # nvcc explicitly prevents CMake from selecting an unrelated CUDA module
    # that may happen to appear first on PATH.
    if (EXISTS "${SCREAM_P3_TORCH_CUDA_ROOT}/bin/nvcc")
      set(CMAKE_CUDA_COMPILER
          "${SCREAM_P3_TORCH_CUDA_ROOT}/bin/nvcc")
    endif()
  endif()

  find_package(Torch REQUIRED)

  # Torch_CXX_FLAGS commonly carries the libstdc++ ABI selection required by
  # LibTorch. Keep it scoped to P3 rather than changing CMAKE_CXX_FLAGS for
  # all of EAMxx.
  separate_arguments(SCREAM_P3_TORCH_CXX_FLAGS
                     NATIVE_COMMAND
                     "${TORCH_CXX_FLAGS}")
endif()
##-- LL

# Add ETI source files if not on CUDA/HIP
if (NOT EAMXX_ENABLE_GPU OR Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE OR Kokkos_ENABLE_HIP_RELOCATABLE_DEVICE_CODE)
  list(APPEND P3_SRCS
    eti/p3_upwind.cpp
    eti/p3_cloud_sed.cpp
    eti/p3_ice_sed.cpp
    eti/p3_ice_collection.cpp
    eti/p3_ice_melting.cpp
    eti/p3_rain_sed.cpp
    eti/p3_table3.cpp
    eti/p3_table_ice.cpp
    eti/p3_dsd2.cpp
    eti/p3_find.cpp
    eti/p3_init.cpp
    eti/p3_update_prognostics.cpp
    eti/p3_get_time_space_phys_variables.cpp
    eti/p3_autoconversion.cpp
    eti/p3_ice_deposition_sublimation.cpp
    eti/p3_conservation.cpp
    eti/p3_rain_self_collection.cpp
    eti/p3_back_to_cell_average.cpp
    eti/p3_cloud_rain_acc.cpp
    eti/p3_calc_rime_density.cpp
    eti/p3_ice_classical_nucleation_impl.cpp
    eti/p3_cldliq_imm_freezing.cpp
    eti/p3_rain_imm_freezing.cpp
    eti/p3_droplet_self_coll.cpp
    eti/p3_evaporate_rain.cpp
    eti/p3_impose_max_total_ni.cpp
    eti/p3_calc_liq_relaxation_timescale.cpp
    eti/p3_ice_relaxation_timescale.cpp
    eti/p3_ice_nucleation.cpp
    eti/p3_ice_cldliq_wet_growth.cpp
    eti/p3_check_values.cpp
    eti/p3_incloud_mixingratios.cpp
    eti/p3_subgrid_variance_scaling.cpp
    eti/p3_main.cpp
    eti/p3_main_part1.cpp
    eti/p3_main_part2.cpp
    eti/p3_main_part3.cpp
    eti/p3_ice_supersat_conservation.cpp
    eti/p3_nc_conservation.cpp
    eti/p3_nr_conservation.cpp
    eti/p3_ni_conservation.cpp
    eti/p3_prevent_liq_supersaturation.cpp
  ) # P3 ETI SRCS
endif()

# List of dispatch source files if monolithic kernels are off
set(P3_SK_SRCS
    disp/p3_check_values_impl_disp.cpp
    disp/p3_ice_sed_impl_disp.cpp
    disp/p3_main_impl_part1_disp.cpp
    disp/p3_main_impl_part3_disp.cpp
    disp/p3_cloud_sed_impl_disp.cpp
    disp/p3_main_impl_disp.cpp
    disp/p3_main_impl_part2_disp.cpp
    disp/p3_rain_sed_impl_disp.cpp
    )

set(P3_LIBS "p3")
if (SCREAM_P3_SMALL_KERNELS)
  add_library(p3 ${P3_SRCS} ${P3_SK_SRCS})
else()
  add_library(p3 ${P3_SRCS})
  # If small kernels are ON, we don't need a separate executable to test them.
  # Also, we never want to generate baselines with this separate executable
  if (NOT SCREAM_LIB_ONLY AND NOT SCREAM_ONLY_GENERATE_BASELINES)
    add_library(p3_sk ${P3_SRCS} ${P3_SK_SRCS})
    # Always build p3_sk with SCREAM_P3_SMALL_KERNELS on
    target_compile_definitions(p3_sk PUBLIC "SCREAM_P3_SMALL_KERNELS")
    list(APPEND P3_LIBS "p3_sk")
  endif()
endif()

target_compile_definitions(p3 PUBLIC EAMXX_HAS_P3)

foreach (P3_LIB IN LISTS P3_LIBS)
  target_include_directories(${P3_LIB} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/impl
    ${SCREAM_BASE_DIR}/../eam/src/physics/cam
  )
  target_link_libraries(${P3_LIB} eamxx_physics_share scream_share)

  ##++ LL 2026/09
  if (SCREAM_P3_ENABLE_TORCHSCRIPT)
    target_compile_definitions(${P3_LIB} PUBLIC SCREAM_P3_ENABLE_TORCHSCRIPT)
    target_compile_options(${P3_LIB} PRIVATE ${SCREAM_P3_TORCH_CXX_FLAGS})
    target_include_directories(${P3_LIB} PRIVATE ${TORCH_INCLUDE_DIRS})
    # p3 is often static, so retain LibTorch on its consumers' final link line.
    target_link_libraries(${P3_LIB} ${TORCH_LIBRARIES})
  endif()
  ##-- LL
endforeach()

# Ensure tables are present in the data dir
if (SCREAM_DOUBLE_PRECISION)
  set (PRECISION_SUFFIX "8")
else()
  set (PRECISION_SUFFIX "4")
endif()
set (P3_TABLES
  scream/tables/p3_lookup_table_1.dat-v4.1.1
  scream/tables/mu_r_table_vals_v2.dat${PRECISION_SUFFIX}
  scream/tables/revap_table_vals_v2.dat${PRECISION_SUFFIX}
  scream/tables/vm_table_vals_v2.dat${PRECISION_SUFFIX}
  scream/tables/vn_table_vals_v2.dat${PRECISION_SUFFIX}
)

include (ScreamUtils)
foreach (file IN ITEMS ${P3_TABLES})
  GetInputFile(${file})
endforeach()

if (NOT SCREAM_LIB_ONLY)
  add_subdirectory(tests)
endif()

if (TARGET eamxx_physics)
  # Add this library to eamxx_physics
  target_link_libraries(eamxx_physics INTERFACE p3)
endif()
