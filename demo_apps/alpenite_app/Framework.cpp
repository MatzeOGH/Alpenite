#include "FrameworkSubsystem.h"

#include <QCoreApplication>
#include <SDL2/SDL.h>
#include <cassert>
#include <string>
#include <string_view>

#include "Renderer.h"

Framework::Framework(int argc, char** argv)
{
    // Setup programm arguments
    ProgramArgs::argc = argc;
    ProgramArgs::argv = argv;
}

bool Framework::init() { 

    auto* qt = add<QTSubsystem>();
    assert(qt != nullptr && "QT is not an optional subsystem");

    auto* sdl = add<SDLSubsystem>();
    assert(sdl != nullptr && "SDL is not an optinal subsystem");

    auto* ren = add<IRenderer>();
    assert(ren != nullptr);

    return true; 
}

void Framework::run()
{
    IRenderer* ren = get<IRenderer>();

    while (shouldShutdown == false)
    {
        QTSubsystem* qt = get<QTSubsystem>();
        qt->update();

        SDLSubsystem* sdl = get<SDLSubsystem>();
        sdl->update(this);

        ren->render();
    }

}

void Framework::main_loop(Framework* framework) 
{ 

    
}

WindowHandel Framework::createWindow(std::string_view title, int width, int height) 
{ 
    auto* sdl = get<SDLSubsystem>();
    assert(sdl != nullptr);
    WindowHandel handle = sdl->createWindow(title, width, height);
    on_window_created(handle);
    return handle;
}

void Framework::closeWindow(WindowHandel window)
{
    auto* sdl = get<SDLSubsystem>();
    assert(sdl != nullptr);
    sdl->closeWindow(window);
}

bool QTSubsystem::register_subsystem(Framework* framework)
{
    QCoreApplication app(Framework::ProgramArgs::argc, Framework::ProgramArgs::argv);
    return true;
};

void QTSubsystem::unregister_subsystem(Framework* framework){};

void QTSubsystem::update()
{
    QCoreApplication::processEvents();
}

bool SDLSubsystem::register_subsystem(Framework* framework)
{
    // init SDL
    SDL_SetMainReady();
    if (SDL_InitSubSystem(SDL_INIT_VIDEO) != 0) 
    {
        qFatal("Could not initialize SDL2 video subsystem! SDL_Error: %s", SDL_GetError());
        return false;
    }

    //
    QObject::connect(framework, &Framework::on_window_resize, this, &SDLSubsystem::on_view_resized);

    active = true;
    return true;
};

void SDLSubsystem::unregister_subsystem(Framework* framework)
{
    SDL_Quit();
}

WindowHandel SDLSubsystem::createWindow(std::string_view title, int width, int height) {
    std::string tmp(title);
    SDL_Window* window = SDL_CreateWindow(
        tmp.c_str(), // Window title
        SDL_WINDOWPOS_CENTERED, // Window position x
        SDL_WINDOWPOS_CENTERED, // Window position y
        width, // Window width
        height, // Window height
        SDL_WINDOW_RESIZABLE); // SDL_WINDOW_VULKAN

    if(!window)
    {
        qFatal("Could not create SDL window! SDL_Error: %s", SDL_GetError());
    }

    return { static_cast<void*>(window) };
};

 
void SDLSubsystem::closeWindow(WindowHandel window)
{
    SDL_Window* handle = static_cast<SDL_Window*>(window.handle);
    SDL_DestroyWindow(handle);
}

void SDLSubsystem::update(Framework* framework)
{ 
    SDL_Event event;
    while(SDL_PollEvent(&event))
    {
        switch (event.type)
        {
            case SDL_QUIT: // all sdl subsystems are shut down
                framework->shutdown(0);
                break;
            case SDL_KEYDOWN:
            case SDL_KEYUP:
                framework->on_key_event(event.key.keysym.sym, event.type == SDL_KEYDOWN);
                break;
            case SDL_MOUSEMOTION:
                framework->on_mouse_move(event.motion.xrel, event.motion.yrel);
                break;
            case SDL_MOUSEBUTTONDOWN:
            case SDL_MOUSEBUTTONUP:
                framework->on_mouse_button(event.button.button, event.type == SDL_MOUSEBUTTONDOWN);
                break;
            case SDL_WINDOWEVENT:
                if(event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) // use this for the actual resize event because the other gets fired multiple times
                {
                    framework->on_window_resize(event.window.data1, event.window.data2);
                }
                break;
        }
    }
}

void SDLSubsystem::on_view_resized(int width, int height) {

    printf("window resized %d %d\n", width, height);
 }
