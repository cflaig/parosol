add_executable(global_node_ids_test global_node_ids_test.cpp)
target_compile_features(global_node_ids_test PRIVATE cxx_std_17)
target_include_directories(global_node_ids_test PRIVATE ../src)
target_link_libraries(global_node_ids_test PRIVATE Eigen3::Eigen)
add_test(NAME global_node_ids COMMAND global_node_ids_test)

add_executable(diagonal_mpi_test diagonal_mpi_test.cpp ../src/fem.cpp ../src/Toolbox.cpp)
target_compile_features(diagonal_mpi_test PRIVATE cxx_std_17)
target_include_directories(diagonal_mpi_test PRIVATE ../src)
target_link_libraries(diagonal_mpi_test PRIVATE Eigen3::Eigen MPI::MPI_CXX)
foreach(ranks 1 2 4)
    add_test(NAME diagonal_mpi_${ranks} COMMAND ${MPIEXEC_EXECUTABLE}
        ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:diagonal_mpi_test> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(diagonal_mpi_${ranks} PROPERTIES PROCESSORS ${ranks} TIMEOUT 60)
    add_test(NAME octree_empty_partitions_mpi_${ranks} COMMAND ${MPIEXEC_EXECUTABLE}
        ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:diagonal_mpi_test> ${MPIEXEC_POSTFLAGS} 8)
    set_tests_properties(octree_empty_partitions_mpi_${ranks} PROPERTIES PROCESSORS ${ranks} TIMEOUT 60)
endforeach()
