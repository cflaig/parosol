cmake_minimum_required(VERSION 2.8)

PROJECT( cudaparosol )

FIND_PACKAGE(CUDA)

SET( cudaparosol_src
    ApplyKernels.cu
    VectorKernels.cu
    OtherKernels.cu
    VectorOperations.cpp
    CompactOffsetTableSizeGenerator.cpp
    FastOffsetTableSizeGenerator.cpp
    AOffsetTableSizeGenerator.cpp
    PsHash.cpp
    GReader.cpp
    GWriter.cpp
    JacobiSmoother.cpp
    Toolbox.cpp
    AsciiImage.cpp
    HDF5Image.cpp
    AsciiImageMirrored.cpp
    DispVectorN.cpp
    DispVectorS.cpp
    DiagonalN.cpp
    DiagonalS.cpp
    DiagonalNComp.cpp
    fem.cpp
    est_ev.cpp
    Chebyshev.cpp
    PCGSolver.cpp
    PCGSolverMC.cpp
    GPUMemoryManager.cpp
    main.cpp
    OPTIONS -gencode arch=compute_30,code=sm_30
)

CUDA_ADD_EXECUTABLE( cudaparosol ${cudaparosol_src} )

#Debuggin
#SET( CMAKE_VERBOSE_MAKEFILE on )
set(CMAKE_CXX_FLAGS "-Wall ${CMAKE_CXX_FLAGS} -fopenmp")

# HDF5
FIND_PACKAGE(HDF5 REQUIRED)
include_directories(${HDF5_INCLUDE_DIRS})
set(LIBS ${LIBS} ${HDF5_LIBRARIES})

# Doxygen
include(FindDoxygen)
IF (DOXYGEN_FOUND)
  SET(DOXYGEN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  SET(DOXYGEN_OUTPUT_DIR doc)
  CONFIGURE_FILE(Doxyfile ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
  ADD_CUSTOM_TARGET(doc ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
ELSE (DOXYGEN_FOUND)
    MESSAGE(STATUS "WARNING: Doxygen not found - Reference manual will not be created")
ENDIF (DOXYGEN_FOUND)

target_link_libraries(cudaparosol ${LIBS})
