#include <webgpu/webgpu.h>
#include <SDL2/SDL.h>
#include <QCoreApplication>
#include "webgpu/webgpu_interface.hpp"

WGPUInstance gInstance{};
WGPUAdapter gAdapter{};
WGPUDevice gDevice{};
WGPUQueue gQueue{};
WGPUSurface gSurface{};
WGPUTextureFormat gSurfaceFormat{};
WGPULimits gSupportedLimits{};


#define WEBGPU_STR(str) WGPUStringView{ .data = str, .length = sizeof(str) - 1 }

void OnDeviceLost(WGPUDeviceLostReason reason, char const* message, void* userdata);

static void OnDeviceError(const WGPUDevice* device, WGPUErrorType type, WGPUStringView message, void* userdata1, void* userdata2)
{
    qFatal("%.*s", message.length, message.data);
}

bool initWebGPUBackend(SDL_Window* window)
{

    const auto timed_wait_feature = WGPUInstanceFeatureName_TimedWaitAny;

    WGPUInstanceDescriptor instanceDesccriptor{
        .requiredFeatureCount = 1u,
        .requiredFeatures = &timed_wait_feature,
    };

#ifndef __EMSCRIPTEN__
    WGPUDawnTogglesDescriptor dawnToggles{};
    dawnToggles.chain.sType = WGPUSType_DawnTogglesDescriptor;

    std::vector<const char*> enabledToggles = { "allow_unsafe_apis" };
#if defined(QT_DEBUG)
    enabledToggles.push_back("use_user_defined_labels_in_backend");
    enabledToggles.push_back("enable_vulkan_validation");
    enabledToggles.push_back("disable_symbol_renaming");
#endif // defined(QT_DEBUG)
    instanceDesccriptor.nextInChain = &dawnToggles.chain;
#endif // __EMSCRIPTEN__

    WGPUInstance instance = wgpuCreateInstance(&instanceDesccriptor);
    if(instance == NULL)
    {
        qFatal("Failed to initalize WebGPU Instance! Aborting");
        return false;
    }

    WGPUSurface surface = SDL_GetWGPUSurface(instance, window);

    WGPURequestAdapterOptions adapterOpts{
        .powerPreference = WGPUPowerPreference_HighPerformance,
        .compatibleSurface = surface,
    };
    WGPUAdapter adapter = webgpu::requestAdapterSync(instance, adapterOpts);
    if(adapter == NULL)
    {
        qFatal("Faled to create WebGPU Adapters");
        return false;
    }

    {
        WGPUAdapterInfo info {};
        wgpuAdapterGetInfo(adapter, &info);

        const char* adapterTypes[] =
            {
            "NULL", // invalid index
            "Discrete GPU",
            "Integrated GPU",
            "CPU",
            "unknown",
            };

        char temp[1024];
        snprintf(temp, sizeof(temp),
            "Device        = %.*s\n"
            "Description   = %.*s\n"
            "Vendor        = %.*s\n"
            "Architecture  = %.*s\n"
            "Adapter Type  = %s\n",
            (int)info.device.length, info.device.data,
            (int)info.description.length, info.description.data,
            (int)info.vendor.length, info.vendor.data,
            (int)info.architecture.length, info.architecture.data,
            adapterTypes[info.adapterType]);

        qInfo(temp);
    }

    WGPULimits requiredLimits {};
    WGPULimits supportedLimits {};
    wgpuAdapterGetLimits(adapter, &supportedLimits);

    WGPUDeviceDescriptor deviceDescriptor{
        .label = WEBGPU_STR("Alpenite"),
        .uncapturedErrorCallbackInfo ={ .callback = &OnDeviceError}
        //.deviceLostCallbackInfo = {.callback = &OnDeviceLost,},

    };
    WGPUDevice device = webgpu::requestDeviceSync(instance, adapter, deviceDescriptor);

    if(device == NULL)
    {
        qFatal("Faled to get adapter");
        return false;
    }
    WGPUQueue queue = wgpuDeviceGetQueue(device);
    if(queue == NULL)
    {
        qFatal("Faled to get adapter");
        return false;
    }

    gInstance = instance;
    gAdapter = adapter;
    gDevice = device;
    gQueue = queue;
    gSurface = surface;

    WGPUSurfaceCapabilities surface_capabilities {};
    wgpuSurfaceGetCapabilities(surface, adapter, &surface_capabilities);
    if (surface_capabilities.formatCount < 1) {
        qFatal( "WebGPU surface formatCount is 0 - must support at least one format");
    }

    gSurfaceFormat = surface_capabilities.formats[0];
    WGPUSurfaceConfiguration config = {};
    config.nextInChain = nullptr;
    config.width = 500;
    config.height = 500;
    config.format = gSurfaceFormat;
    config.viewFormatCount = 0;
    config.viewFormats = nullptr;
    config.usage = WGPUTextureUsage_RenderAttachment;
    config.device = device;
    config.presentMode = WGPUPresentMode_Fifo;
    config.alphaMode = WGPUCompositeAlphaMode_Auto;

    wgpuSurfaceConfigure(surface, &config);

    return true;
}

void drawFrame()
{
    // process all internal async events or error callbacks
    wgpuInstanceProcessEvents(gInstance);

    static int frameIndex = 0;
    frameIndex++;

    WGPUSurfaceTexture surfaceTexture;
    wgpuSurfaceGetCurrentTexture(gSurface, &surfaceTexture);

    if(surfaceTexture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal)
    {
        qFatal("surface not readyr");
        return;// TODO: handle resize / lost surface
    }
    qInfo("Frame Index %d", frameIndex);

    WGPUTextureViewDescriptor view_desc =
    {
        .format = gSurfaceFormat,
        .dimension = WGPUTextureViewDimension_2D,
        .baseMipLevel = 0,
        .mipLevelCount = 1,
        .baseArrayLayer = 0,
        .arrayLayerCount = 1,
        .aspect = WGPUTextureAspect_All,
    };
    WGPUTextureView view = wgpuTextureCreateView(surfaceTexture.texture, &view_desc);

    WGPUCommandEncoder encoder =  wgpuDeviceCreateCommandEncoder(gDevice, NULL);

    WGPURenderPassColorAttachment colorAttachment = {0};
    colorAttachment.view = view;
    colorAttachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
    colorAttachment.loadOp = WGPULoadOp_Clear;
    colorAttachment.storeOp = WGPUStoreOp_Store;
    colorAttachment.clearValue = {0.1, 0.3, 0.6, 1.0};

    WGPURenderPassDescriptor renderPassDesc{};
    renderPassDesc.colorAttachmentCount = 1;
    renderPassDesc.colorAttachments = &colorAttachment;

    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &renderPassDesc);
    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);

    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, NULL);

    WGPUQueue queue = wgpuDeviceGetQueue(gDevice);
    wgpuQueueSubmit(queue, 1, &cmd);

    wgpuSurfacePresent(gSurface);

    wgpuTextureViewRelease(view);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);

}

void OnDeviceLost(WGPUDeviceLostReason reason, char const* message, void* userdata)
{
    /*
    qError() << "Device lost! Reason: " << reason << "\n";
    if (message) {
        qFatal() << "Message: " << message << "\n";
    }*/
}
