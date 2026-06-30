cmake_minimum_required(VERSION 3.16)
project(llm_trace LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

include(FetchContent)

# JSON for the wire protocol.
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.zip
)
FetchContent_MakeAvailable(json)

# FTXUI for the terminal UI.
FetchContent_Declare(
    ftxui
    URL https://github.com/ArthurSonzogni/FTXUI/archive/refs/tags/v5.0.0.zip
)
FetchContent_MakeAvailable(ftxui)

add_executable(llm-trace
    src/main.cpp
    src/telemetry/event_bus.cpp
    src/telemetry/aggregator.cpp
    src/storage/ring_buffer.cpp
    src/anomaly/anomaly_detector.cpp
    src/instrumentation/device_monitor.cpp
    src/instrumentation/telemetry_server.cpp
    src/instrumentation/mock_tracer.cpp
    src/tui/panels.cpp
    src/tui/dashboard.cpp
)

target_include_directories(llm-trace PRIVATE include)
target_compile_options(llm-trace PRIVATE -Wall -Wextra)

target_link_libraries(llm-trace PRIVATE
    nlohmann_json::nlohmann_json
    ftxui::component  # transitively pulls in ftxui::dom and ftxui::screen
)

find_package(Threads REQUIRED)
target_link_libraries(llm-trace PRIVATE Threads::Threads)
