PROJECT( cudaparosol )

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
)

add_executable( cudaparosol ${cudaparosol_src} )

target_compile_options(cudaparosol PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wall>")
target_link_libraries(cudaparosol PRIVATE MPI::MPI_CXX HDF5::HDF5 OpenMP::OpenMP_CXX CUDA::cudart)

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
