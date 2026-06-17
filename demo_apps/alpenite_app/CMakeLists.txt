project(alpine-renderer-alpenite_app LANGUAGES C CXX)

set(SOURCES
    main.cpp
    Renderer.h
    Renderer.cpp
    FrameworkSubsystem.h # needed for QTs moc
    Framework.cpp
    AlpUtils.h
    ClusteredMesh.cpp
)

qt_add_executable(alpenite_app
    ${SOURCES}
)

set_target_properties(alpenite_app PROPERTIES
    WIN32_EXECUTABLE FALSE  # Set to FALSE to keep console attached on Windows
    MACOSX_BUNDLE TRUE
)

target_link_libraries(alpenite_app PUBLIC webgpu Qt::Core)
target_include_directories(alpenite_app PRIVATE .)

# Copy necessary DLLs to the output directory on Windows
if (WIN32 AND NOT EMSCRIPTEN)
    # Copy SDL2 DLL
    if (EXISTS "${CMAKE_SOURCE_DIR}/${ALP_EXTERN_DIR}/sdl/bin/SDL2.dll")
        add_custom_command(TARGET alpenite_app POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/${ALP_EXTERN_DIR}/sdl/bin/SDL2.dll"
            "$<TARGET_FILE_DIR:alpenite_app>"
            COMMENT "Copying SDL2.dll"
        )
    endif()

    # Use Qt's windeployqt to copy all necessary Qt dependencies
    get_target_property(QMAKE_EXECUTABLE Qt6::qmake IMPORTED_LOCATION)
    get_filename_component(QT_BIN_DIR "${QMAKE_EXECUTABLE}" DIRECTORY)
    find_program(WINDEPLOYQT_EXECUTABLE windeployqt HINTS "${QT_BIN_DIR}")

    if(WINDEPLOYQT_EXECUTABLE)
        add_custom_command(TARGET alpenite_app POST_BUILD
            COMMAND "${WINDEPLOYQT_EXECUTABLE}" --no-translations "$<TARGET_FILE:alpenite_app>"
            COMMENT "Running windeployqt for alpenite_app"
        )
    endif()


    # === INSTALL CONFIGURATION FOR NATIVE WINDOWS BUILD ===
    install(TARGETS alpenite_app RUNTIME DESTINATION . BUNDLE DESTINATION .)
    if (EXISTS "${CMAKE_SOURCE_DIR}/${ALP_EXTERN_DIR}/sdl/bin/SDL2.dll")
        install(FILES "${CMAKE_SOURCE_DIR}/${ALP_EXTERN_DIR}/sdl/bin/SDL2.dll" DESTINATION .)
    endif()
    if(WINDEPLOYQT_EXECUTABLE)
        install(CODE "
            execute_process(
                COMMAND \"${WINDEPLOYQT_EXECUTABLE}\" --no-translations \"\${CMAKE_INSTALL_PREFIX}/alpenite_app.exe\"
            )
        ")
    endif()
    # ================================================
endif()
