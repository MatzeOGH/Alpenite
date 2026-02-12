#include <QCoreApplication>
#include <SDL2/SDL.h>
#include <webgpu/webgpu.h>

#include "AlpUtils.h"


bool initWebGPUBackend(SDL_Window* window);
void drawFrame();

// main entry for all platforms (WIN32 and WEB)
int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    QCoreApplication app(argc, argv);

    SDL_SetMainReady();
    if (SDL_InitSubSystem(SDL_INIT_VIDEO) != 0) {
        qFatal("Could not initialize SDL2 video subsystem! SDL_Error: %s", SDL_GetError());
        return -1;
    }
    defer{
        SDL_Quit();
    };

    SDL_Window* window = SDL_CreateWindow("Alpenite App", // Window title
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
        SDL_DestroyWindow(window);
    };

    if(initWebGPUBackend(window) == false)
    {
        return -1;
    }

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

        drawFrame();

    }

    return 0;
}
