#pragma once

#include <QObject>

#include <vector>
#include <string_view>
#include <unordered_map>
#include <memory>
#include <typeindex>
#include <cassert>

class Framework;

// subsysem class to register with the framewokr
class IFrameworkSubsystem 
{
public:
    virtual ~IFrameworkSubsystem() = default;

    virtual bool register_subsystem(Framework* framework) = 0;
    virtual void unregister_subsystem(Framework* framework) = 0;
};

class QTSubsystem : public QObject, public IFrameworkSubsystem
{
    Q_OBJECT
public:
    QTSubsystem(QObject* parent) : QObject(parent) {}

    bool register_subsystem(Framework* framework) override;
    void unregister_subsystem(Framework* framework) override;
    void update();
};

struct WindowHandel
{
    void* handle{};
    bool isValid() const { return handle != nullptr; }
};

class IWindow
{
public: 
    virtual WindowHandel createWindow(std::string_view title, int width, int height) = 0;
    virtual void closeWindow(WindowHandel window) = 0;
};

class SDLSubsystem : public QObject, public IFrameworkSubsystem, public IWindow
{
    Q_OBJECT
public:
    SDLSubsystem(QObject* parent) : QObject(parent) {}

    bool register_subsystem(Framework* framework) override;
    void unregister_subsystem(Framework* framework) override ;

    WindowHandel createWindow(std::string_view title, int width, int height) override;
    void closeWindow(WindowHandel window) override;

    void update(Framework* framework);

    void on_view_resized(int, int);

private:
    bool active{false};
};


// TODO: handle optional subsystem
class Framework : public QObject, public IWindow
{
    Q_OBJECT
public:
    Framework(int argc, char** argv);

    virtual ~Framework()
    {
        for(auto& sub : subsystems)
        {
            sub.second->unregister_subsystem(this);
            delete sub.second;
        }
    }

    virtual bool init();
    void run();
    void main_loop(Framework* framework);

    template<typename T, typename... Args>
    T* add(Args&&... args)
    {
        static_assert(std::is_base_of_v<IFrameworkSubsystem, T>);

        auto type = std::type_index(typeid(T));
        assert(subsystems.find(type) == subsystems.end() && "Subsystem already registered");

        T* instance = new T(this, std::forward<Args>(args)...);
        if(instance->register_subsystem(this))
        {
            // Convert unique_ptr<Derived> to unique_ptr<Base>
            subsystems[type] = instance;
            return instance;
        }
        
        // registration of the system failed for some reason. The caller has to handle this case
        delete instance;
        return nullptr;
    }

    template<typename T>
    T* get() {
        static_assert(std::is_base_of_v<IFrameworkSubsystem, T>);

        auto it = subsystems.find(std::type_index(typeid(T)));
        assert(it != subsystems.end() && "Subsystem not found");

        return dynamic_cast<T*>(it->second);
    }

    WindowHandel createWindow(std::string_view title, int width, int height) override;
    void closeWindow(WindowHandel window) override;

    void shutdown(int code) {
        printf("shutting down wiht code %d", code);
        // notify systems of the shutdown
        emit on_framework_shutdown();
        shouldShutdown = true; 
    }

signals:

    void on_window_created(WindowHandel);
    void on_window_resize(int width, int height);
    void on_mouse_move(int dx, int dy);
    void on_mouse_button(int button, bool pressed);
    void on_key_event(int key, bool pressed);
    void on_framework_shutdown();


private:
    // container for registered subsystems
    // TODO: using a map here is more or less overkill 
    std::unordered_map<std::type_index, IFrameworkSubsystem*> subsystems;
    bool shouldShutdown{false};
public:
    // Helper to hold argc and argv. These are valid for the whole duration runtime
    struct ProgramArgs
    {
        static inline int argc;
        static inline char **argv;
    };
};

