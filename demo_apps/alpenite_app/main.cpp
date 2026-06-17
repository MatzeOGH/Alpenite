#include <QCoreApplication>
#include <SDL2/SDL.h>
#include <webgpu/webgpu.h>

#include "AlpUtils.h"
#include "FrameworkSubsystem.h"
#include "Renderer.h"

bool initWebGPUBackend(SDL_Window* window);
void drawFrame(int32_t x, int32_t y);
void shutdownBacken();
void resize(uint32_t x, uint32_t y);


// main entry for all platforms (WIN32 and WEB)
int main(int argc, char* argv[])
{
    Framework framework(argc, argv);
    if(framework.init() == false) {
        return -1;
    }

    uint32_t width = 1024;
    uint32_t height = 768;

    auto window = framework.createWindow("Alpenite", width, height);
    if(window.isValid() == false){
        return -1;
    }
    defer{
        framework.closeWindow(window);
    };

    defer {
        shutdownBacken();
    };

    // WebGPU device is ready after createWindow — load mesh and set up clustered rendering
    loadMeshForRendering("C:/Users/matth/Documents/Aurora/Aurora/assets/roman_stone_capital_high/Roman_Stone_Capital_tfpvdgeda_High.gltf");

    framework.run();
    
    return 0;
}
