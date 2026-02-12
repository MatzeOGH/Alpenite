#include <QCoreApplication>
#include <iostream>
#include <SDL2/SDL.h>

#include "AlpUtils.h"


// main entry for all platforms (WIN32 and WEB)
int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    QCoreApplication app(argc, argv);

    SDL_SetMainReady();
    if (SDL_InitSubSystem(SDL_INIT_VIDEO) != 0) {
        qFatal("Could not initialize SDL2 video subsystem! SDL_Error: %s", SDL_GetError());
    }
    defer{
        std::cout << "SDL_Quit\n";
        SDL_Quit();
    };

    auto window = SDL_CreateWindow("Alpenite Appl", // Window title
        SDL_WINDOWPOS_CENTERED, // Window position x
        SDL_WINDOWPOS_CENTERED, // Window position y
        500, // Window width
        500, // Window height
        SDL_WINDOW_RESIZABLE); // SDL_WINDOW_VULKAN

    if (!window) {
        qFatal("Could not create SDL window! SDL_Error: %s", SDL_GetError());
        return -1;
    }
    defer{
        std::cout << "SDL_DestroyWindow\n";
        SDL_DestroyWindow(window);
    };

    while(true)
    {
        QCoreApplication::processEvents();
        SDL_Event event;
        while(SDL_PollEvent(&event))
        {
            if(event.type == SDL_QUIT)
            {
                return 0;
            }
        }

    }

    return 0;
}
