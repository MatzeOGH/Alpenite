#include <QCoreApplication>
#include <iostream>
#include <SDL2/SDL.h>

// main entry for all platforms (WIN32 and WEB)
int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    QCoreApplication app(argc, argv);

    std::cout << "Hello World\n";

    SDL_SetMainReady();
    if (SDL_InitSubSystem(SDL_INIT_VIDEO) != 0) {
        qFatal("Could not initialize SDL2 video subsystem! SDL_Error: %s", SDL_GetError());
    }

    auto window = SDL_CreateWindow("weBIGeo - Geospatial Visualization Tool", // Window title
        SDL_WINDOWPOS_CENTERED, // Window position x
        SDL_WINDOWPOS_CENTERED, // Window position y
        500, // Window width
        500, // Window height
        SDL_WINDOW_RESIZABLE); // SDL_WINDOW_VULKAN

    if (!window) {
        SDL_Quit();
        qFatal("Could not create SDL window! SDL_Error: %s", SDL_GetError());
    }

    return 0;
}
