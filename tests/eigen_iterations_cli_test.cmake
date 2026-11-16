# Exercise the executable's parser before it opens an input file.
foreach(value 0 1 -1 1001 invalid 2x 999999999999999999999999999999999)
    execute_process(COMMAND "${PAROSOL}" --eigen-max-iterations "${value}"
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 10)
    if(NOT result STREQUAL "2" OR NOT output MATCHES "must be an integer from 2 to 1000")
        message(FATAL_ERROR "Invalid value '${value}': status=${result}\n${output}\n${error}")
    endif()
endforeach()

execute_process(COMMAND "${PAROSOL}" --eigen-max-iterations
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 10)
if(NOT result STREQUAL "2" OR NOT output MATCHES "usage:")
    message(FATAL_ERROR "Missing value: status=${result}\n${output}\n${error}")
endif()

foreach(value 2 100 1000)
    execute_process(COMMAND "${PAROSOL}" --eigen-max-iterations "${value}"
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 10)
    if(result STREQUAL "0" OR NOT output MATCHES "Eigenvalue estimation iteration limit: ${value}"
        OR NOT output MATCHES "No file given")
        message(FATAL_ERROR "Valid value '${value}' not accepted: status=${result}\n${output}\n${error}")
    endif()
endforeach()
