cmake_minimum_required(VERSION 3.21)

project(Forklift VERSION 1.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CHESS_BUILD_GUI "Build the SFML desktop application" ON)
option(CHESS_ENABLE_WARNINGS "Enable strict compiler warnings" ON)
option(CHESS_ENABLE_IPO "Enable interprocedural/link-time optimization" OFF)
option(CHESS_NATIVE_ARCH "Optimize for the CPU that performs the build" OFF)

find_package(Threads REQUIRED)

if(CHESS_ENABLE_IPO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT CHESS_IPO_SUPPORTED OUTPUT CHESS_IPO_ERROR)
  if(NOT CHESS_IPO_SUPPORTED)
    message(FATAL_ERROR "Interprocedural optimization is unavailable: ${CHESS_IPO_ERROR}")
  endif()
endif()

function(chess_apply_performance_options target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /EHsc)
  endif()
  if(CHESS_NATIVE_ARCH)
    if(MSVC)
      target_compile_options(${target} PRIVATE /arch:AVX2)
    else()
      target_compile_options(${target} PRIVATE -march=native)
    endif()
  endif()
  if(CHESS_ENABLE_IPO)
    set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
  endif()
endfunction()

add_library(chess_core INTERFACE)
target_include_directories(chess_core INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/src")

add_library(chess_uci STATIC src/uci.cpp)
target_link_libraries(chess_uci PUBLIC chess_core Threads::Threads)
chess_apply_performance_options(chess_uci)

add_executable(chess-engine-uci src/uci_main.cpp)
target_link_libraries(chess-engine-uci PRIVATE chess_uci)
chess_apply_performance_options(chess-engine-uci)

add_executable(chess-engine-tools src/tools_main.cpp)
target_link_libraries(chess-engine-tools PRIVATE chess_core Threads::Threads)
chess_apply_performance_options(chess-engine-tools)

if(CHESS_BUILD_GUI)
  find_package(SFML 2.6 COMPONENTS graphics window system audio REQUIRED)

  add_executable(gui
    src/main.cpp
    src/ui.cpp
  )
  set_target_properties(gui PROPERTIES OUTPUT_NAME "Forklift")
  if(WIN32)
    target_sources(gui PRIVATE resources/windows/forklift.rc)
  endif()
  target_link_libraries(gui PRIVATE chess_uci)
  chess_apply_performance_options(gui)

  if(TARGET sfml-graphics)
    target_link_libraries(gui PRIVATE sfml-graphics sfml-window sfml-system sfml-audio)
  elseif(TARGET SFML::Graphics)
    target_link_libraries(gui PRIVATE SFML::Graphics SFML::Window SFML::System SFML::Audio)
  else()
    message(FATAL_ERROR "SFML targets not found. Install SFML 2.6.x or configure with -DCHESS_BUILD_GUI=OFF.")
  endif()

  if(WIN32)
    get_filename_component(CHESS_SFML_BIN_DIR "${SFML_DIR}/../../../bin" ABSOLUTE)
    target_compile_definitions(gui PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
    target_link_libraries(gui PRIVATE Psapi)
  endif()

  add_custom_command(TARGET gui POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/assets"
            "$<TARGET_FILE_DIR:gui>/assets"
  )

  if(WIN32)
    add_custom_command(TARGET gui POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              $<TARGET_RUNTIME_DLLS:gui>
              $<TARGET_FILE_DIR:gui>
      COMMAND_EXPAND_LISTS
    )
    if(EXISTS "${CHESS_SFML_BIN_DIR}/openal32.dll")
      add_custom_command(TARGET gui POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CHESS_SFML_BIN_DIR}/openal32.dll"
                $<TARGET_FILE_DIR:gui>
      )
    endif()
  endif()
endif()

if(CHESS_ENABLE_WARNINGS)
  if(MSVC)
    set(CHESS_WARNING_FLAGS /W4 /permissive-)
  else()
    set(CHESS_WARNING_FLAGS -Wall -Wextra -Wpedantic -Wshadow)
  endif()
  target_compile_options(chess_uci PRIVATE ${CHESS_WARNING_FLAGS})
  target_compile_options(chess-engine-uci PRIVATE ${CHESS_WARNING_FLAGS})
  target_compile_options(chess-engine-tools PRIVATE ${CHESS_WARNING_FLAGS})
  if(TARGET gui)
    target_compile_options(gui PRIVATE ${CHESS_WARNING_FLAGS})
  endif()
endif()

include(CTest)
if(BUILD_TESTING)
  add_executable(chess-core-tests tests/core_tests.cpp)
  target_link_libraries(chess-core-tests PRIVATE chess_core Threads::Threads)
  chess_apply_performance_options(chess-core-tests)
  if(MSVC)
    target_compile_definitions(chess-core-tests PRIVATE _CRT_SECURE_NO_WARNINGS)
  endif()
  if(CHESS_ENABLE_WARNINGS)
    target_compile_options(chess-core-tests PRIVATE ${CHESS_WARNING_FLAGS})
  endif()
  add_test(NAME chess-core-tests COMMAND chess-core-tests)
  add_executable(chess-proof-budget-tests tests/search_proof_budget_tests.cpp)
  target_link_libraries(chess-proof-budget-tests PRIVATE chess_core Threads::Threads)
  chess_apply_performance_options(chess-proof-budget-tests)
  if(CHESS_ENABLE_WARNINGS)
    target_compile_options(chess-proof-budget-tests PRIVATE ${CHESS_WARNING_FLAGS})
  endif()
  add_test(NAME chess-proof-budget-tests COMMAND chess-proof-budget-tests)
  add_executable(chess-singular-tests tests/search_singular_tests.cpp)
  target_link_libraries(chess-singular-tests PRIVATE chess_core Threads::Threads)
  chess_apply_performance_options(chess-singular-tests)
  if(CHESS_ENABLE_WARNINGS)
    target_compile_options(chess-singular-tests PRIVATE ${CHESS_WARNING_FLAGS})
  endif()
  add_test(NAME chess-singular-tests COMMAND chess-singular-tests)
  add_test(NAME benchmark-production-default
    COMMAND chess-engine-tools --bench --bench-depth 1 --bench-tt 1)
  set_tests_properties(benchmark-production-default PROPERTIES
    PASS_REGULAR_EXPRESSION "probCut=off")
  add_test(NAME benchmark-probcut-opt-in
    COMMAND chess-engine-tools --bench --bench-depth 1 --bench-tt 1 --probcut)
  set_tests_properties(benchmark-probcut-opt-in PROPERTIES
    PASS_REGULAR_EXPRESSION "probCut=on")
  add_test(NAME benchmark-singular-default
    COMMAND chess-engine-tools --bench --bench-depth 1 --bench-tt 1)
  set_tests_properties(benchmark-singular-default PROPERTIES
    PASS_REGULAR_EXPRESSION "singular=off")
  add_test(NAME benchmark-singular-opt-in
    COMMAND chess-engine-tools --bench --bench-depth 1 --bench-tt 1 --singular)
  set_tests_properties(benchmark-singular-opt-in PROPERTIES
    PASS_REGULAR_EXPRESSION "singular=on")
  add_test(NAME benchmark-proof-budget-opt-in
    COMMAND chess-engine-tools --bench --bench-depth 1 --bench-tt 1 --singular --singular-budget 5)
  set_tests_properties(benchmark-proof-budget-opt-in PROPERTIES
    PASS_REGULAR_EXPRESSION "singularBudget=5")
  if(TARGET gui)
    add_test(NAME desktop-benchmark-production-default
      COMMAND gui --bench --bench-depth 1 --bench-tt 1)
    set_tests_properties(desktop-benchmark-production-default PROPERTIES
      PASS_REGULAR_EXPRESSION "probCut=off")
  endif()
  if(WIN32)
    find_program(CHESS_POWERSHELL_EXECUTABLE NAMES pwsh powershell REQUIRED)
    add_test(NAME uci-smoke
      COMMAND "${CHESS_POWERSHELL_EXECUTABLE}"
              -NoProfile
              -ExecutionPolicy Bypass
              -File "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_uci_smoke.ps1"
              $<TARGET_FILE:chess-engine-uci>)
  elseif(UNIX)
    add_test(NAME uci-smoke
      COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_uci_smoke.sh" $<TARGET_FILE:chess-engine-uci>)
  endif()
endif()

install(TARGETS chess-engine-uci chess-engine-tools RUNTIME DESTINATION bin)
if(TARGET gui)
  if(WIN32)
    install(TARGETS gui RUNTIME DESTINATION bin)
    install(FILES $<TARGET_RUNTIME_DLLS:gui> DESTINATION bin)
    if(EXISTS "${CHESS_SFML_BIN_DIR}/openal32.dll")
      install(FILES "${CHESS_SFML_BIN_DIR}/openal32.dll" DESTINATION bin)
    endif()
  else()
    install(TARGETS gui RUNTIME DESTINATION bin)
  endif()
  install(DIRECTORY assets DESTINATION bin)
  install(DIRECTORY assets DESTINATION share/forklift)
endif()
install(DIRECTORY web assets DESTINATION .
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE)
install(PROGRAMS scripts/run_web_gui.sh DESTINATION scripts)
install(FILES scripts/run_web_gui.ps1 DESTINATION scripts)
install(FILES README.md DESTINATION .)
install(DIRECTORY docs DESTINATION .)

set(CPACK_PACKAGE_NAME "Forklift")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_FILE_NAME "Forklift-${PROJECT_VERSION}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
set(CPACK_GENERATOR "ZIP")
include(CPack)
