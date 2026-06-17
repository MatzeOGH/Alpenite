#include <webgpu/webgpu.h>
#include <SDL2/SDL.h>
#include <QCoreApplication>
#include <QDebug> // TODO: implement in PCH
#include "webgpu/webgpu_interface.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp> 

#include "AlpUtils.h"
#include "Renderer.h"
#include "ClusteredMesh.h"

#if _WIN64
// Enable NVIDIA Optimus discrete GPU on laptops for better perforance and power management
// Reference: https://developer.download.nvidia.com/devzone/devcenter/gamegraphics/files/OptimusRenderingPolicies.pdf
extern "C" _declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
#endif

WGPUInstance gInstance{};
WGPUAdapter gAdapter{};
WGPUDevice gDevice{};
WGPUQueue gQueue{};
WGPUSurface gSurface{};
WGPUTextureFormat gSurfaceFormat = WGPUTextureFormat_Undefined;
WGPULimits gSupportedLimits{};
WGPURenderPipeline  gPipeline;

// cluster culling compute resources
static WGPUComputePipeline gCullPipeline{};
static WGPUBindGroupLayout gCullBindGroupLayout{};
static WGPUBindGroup       gCullBindGroup{};
static WGPUBuffer          gVisibleClustersBuffer{};
static uint32_t            gCullClusterCount = 0;

// cluster index-expansion compute resources
static WGPUComputePipeline gExpandPipeline{};
static WGPUBindGroupLayout gExpandBindGroupLayout{};
static WGPUBindGroup       gExpandBindGroup{};
static WGPUBuffer          gOutputIndexBuffer{};   // u32 packed indices (cluster_id<<8 | local_vtx)
static WGPUBuffer          gDrawArgsBuffer{};      // DrawIndexedIndirectParams
static uint32_t            gExpandClusterCount = 0;

// clustered render pipeline — decodes packed indices in the vertex shader
static WGPURenderPipeline  gClusteredPipeline{};
static WGPUBindGroupLayout gClusteredRenderBGL{};
static WGPUBindGroup       gClusteredRenderBG{};

// depth buffer
static WGPUTexture gDepthTexture{};
static WGPUTextureView gDepthView{};

// construct an webgpu string view from a cstring
// (important!): only valid for const char* literals!!!!!
#define WEBGPU_STR(str) WGPUStringView{ .data = str, .length = sizeof(str) - 1 }

// ---------------------------------------------------------------------------
// Pass builder utilities
// ---------------------------------------------------------------------------

struct RenderPassBuilder
{
    static constexpr uint32_t kMaxColorAttachments = 8;
    WGPURenderPassColorAttachment colorAttachments[kMaxColorAttachments]{};
    uint32_t colorAttachmentCount = 0;
    WGPURenderPassDepthStencilAttachment depthStencilAttachment{};
    bool hasDepthStencil = false;

    RenderPassBuilder& color(WGPUTextureView view,
                             WGPULoadOp  load       = WGPULoadOp_Clear,
                             WGPUStoreOp store      = WGPUStoreOp_Store,
                             WGPUColor   clearValue = {0.0, 0.0, 0.0, 1.0})
    {
        assert(colorAttachmentCount < kMaxColorAttachments);
        colorAttachments[colorAttachmentCount++] = WGPURenderPassColorAttachment{
            .view       = view,
            .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
            .loadOp     = load,
            .storeOp    = store,
            .clearValue = clearValue,
        };
        return *this;
    }

    RenderPassBuilder& depth_stencil(WGPUTextureView view,
                                     WGPULoadOp  depthLoad    = WGPULoadOp_Clear,
                                     WGPUStoreOp depthStore   = WGPUStoreOp_Store,
                                     float       clearDepth   = 1.0f,
                                     WGPULoadOp  stencilLoad  = WGPULoadOp_Clear,
                                     WGPUStoreOp stencilStore = WGPUStoreOp_Discard,
                                     uint32_t    clearStencil = 0)
    {
        depthStencilAttachment = WGPURenderPassDepthStencilAttachment{
            .view              = view,
            .depthLoadOp       = depthLoad,
            .depthStoreOp      = depthStore,
            .depthClearValue   = clearDepth,
        };
        hasDepthStencil = true;
        return *this;
    }

    WGPURenderPassEncoder begin(WGPUCommandEncoder encoder) const
    {
        WGPURenderPassDescriptor desc{
            .colorAttachmentCount        = colorAttachmentCount,
            .colorAttachments            = colorAttachments,
            .depthStencilAttachment      = hasDepthStencil ? &depthStencilAttachment : nullptr,
        };
        return wgpuCommandEncoderBeginRenderPass(encoder, &desc);
    }
};

struct ComputePassBuilder
{
    WGPUStringView label{};

    ComputePassBuilder& set_label(const char* str)
    {
        label = WGPUStringView{ .data = str, .length = strlen(str) };
        return *this;
    }

    WGPUComputePassEncoder begin(WGPUCommandEncoder encoder) const
    {
        WGPUComputePassDescriptor desc{ .label = label };
        return wgpuCommandEncoderBeginComputePass(encoder, &desc);
    }
};

// ---------------------------------------------------------------------------


struct Camera {
    double latitude;   // φ
    double longitude;  // λ
    double height;     // meters above ellipsoid
    double heading;    // yaw angle
    double pitch;      // tilt angle
    double fov;        // field of view
};

glm::dvec3 geodeticToCartesian(double latitude, double longitude, double h);
glm::mat4 buildViewMatrix(const Camera &cam) ;

struct CameraUniforms {
    glm::mat4 view;
    glm::mat4 proj;
    glm::vec3 cameraPos;
    float     _pad0             = 0.f;
    float     clusterThreshold  = 1.0f;
    float     cameraProj        = 0.f;
    float     znear             = 0.01f;
    float     _pad1             = 0.f;
};
WGPUBuffer viewUniforms;
WGPUBindGroup cameraBindGroup;

// WebGPU indirect indexed draw parameters (matches GPU buffer layout)
struct DrawIndexedIndirectParams {
    uint32_t indexCount;
    uint32_t instanceCount;
    uint32_t firstIndex;
    int32_t  baseVertex;
    uint32_t firstInstance;
};

void OnDeviceLost(WGPUDeviceLostReason reason, char const* message, void* userdata);

struct RenderContext
{
    float ViewportWidth, viewportHeight;
    WGPUCommandEncoder encoder;
};

// defines a task for the renderer
class IRenderTask
{
public:
    virtual ~IRenderTask() = default;

    virtual void record_render_commands(RenderContext renderContex) = 0;
};

// view
struct AlpeniteView
{

};

// scene 
struct AlpeniteScene
{

};

class AlpeniteRenderTask : public IRenderTask
{
public:
    AlpeniteRenderTask()
    {
        WGPURenderPipelineDescriptor descriptor{
            .label = WEBGPU_STR("Alpenite Render Pipeline"),
            .vertex = {
                //.module = gVertexModule,
                .entryPoint = WEBGPU_STR("main")
            },
        };
        wgpuDeviceCreateRenderPipeline(gDevice, &descriptor);
    }

    // BEGIN IRenderTask
    virtual void record_render_commands(RenderContext renderContext) override
    {

    }
    // END IRenderTask

private:
    float x, y;

};

// utility function to create a shader module
WGPUShaderModule createShaderModule(WGPUDevice device, WGPUStringView shaderCode)
{
    WGPUShaderSourceWGSL wgslDesc {
        .chain = {.sType = WGPUSType_ShaderSourceWGSL},
        .code = shaderCode,
    };

    WGPUShaderModuleDescriptor shaderModuleDesc {
        .nextInChain = &wgslDesc.chain,
        .label = WEBGPU_STR("sample shader"),
    };

    WGPUShaderModule shaderModule = wgpuDeviceCreateShaderModule(device, &shaderModuleDesc);
    return shaderModule;
}


static void OnDeviceError(const WGPUDevice* device, WGPUErrorType type, WGPUStringView message, void* userdata1, void* userdata2)
{
    qFatal("%.*s", message.length, message.data);
}

static WGPUInstance createWebGPUInstance()
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
        return nullptr;
    }
    return instance;
}

bool initWebGPUBackend(SDL_Window* window)
{
    WGPUInstance instance = createWebGPUInstance();
    assert(instance);
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

    // pick the first possible format for the surface
    WGPUSurfaceCapabilities surfaceCapabilities{
    };
    wgpuSurfaceGetCapabilities(surface, adapter, &surfaceCapabilities);
    if (surfaceCapabilities.formatCount < 1) {
        qFatal( "WebGPU surface formatCount is 0 - must support at least one format");
    }
    for(int i = 0; i < surfaceCapabilities.formatCount; ++i)
    {
        qInfo("Surface format %d: %d", i, surfaceCapabilities.formats[i]);
    }
    gSurfaceFormat = surfaceCapabilities.formats[0];

    const char shaderCode[] = R"( 

struct Uniforms {
    view : mat4x4<f32>,
    proj : mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> u : Uniforms;


@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> @builtin(position) vec4f {
    var p = vec2f(0.0, 0.0);
    if (in_vertex_index == 0u) {
        p = vec2f(-0.5, -0.5);
    } else if (in_vertex_index == 1u) {
        p = vec2f(0.5, -0.5);
    } else {
        p = vec2f(0.0, 0.5);
    }
    
    var p4 = u.proj * u.view * vec4<f32>(p, 0.0, 1.0);

    return p4;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(1.0, 0.4, 1.0, 1.0);
}
    )";

    WGPUShaderModule shaderModule = createShaderModule(device, WEBGPU_STR(shaderCode));
    defer {
        wgpuShaderModuleRelease(shaderModule);
    };

    {
        WGPUBufferDescriptor bufferDesc{
            .label = WEBGPU_STR("camera uniform"),
            .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
            .size = sizeof(CameraUniforms),
        };
        viewUniforms = wgpuDeviceCreateBuffer(device, &bufferDesc);
    }

    WGPUBindGroupLayoutEntry layoutEntry{
        .binding = 0u,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = {.type = WGPUBufferBindingType_Uniform, .minBindingSize = sizeof(CameraUniforms)},
    };

    WGPUBindGroupLayoutDescriptor layoutDesc{
        .entryCount = 1,
        .entries = &layoutEntry
    };

    WGPUBindGroupLayout bindGroupLayout = wgpuDeviceCreateBindGroupLayout(device, &layoutDesc);

    WGPUBindGroupEntry bindEntry{};
    bindEntry.binding = 0;
    bindEntry.buffer = viewUniforms;
    bindEntry.offset = 0;
    bindEntry.size = sizeof(CameraUniforms);

    WGPUBindGroupDescriptor bindDesc{};
    bindDesc.layout = bindGroupLayout;
    bindDesc.entryCount = 1;
    bindDesc.entries = &bindEntry;

 cameraBindGroup = wgpuDeviceCreateBindGroup(device, &bindDesc);

    WGPUPipelineLayoutDescriptor pipelineLayoutDesc{};
    pipelineLayoutDesc.bindGroupLayoutCount = 1;
    pipelineLayoutDesc.bindGroupLayouts = &bindGroupLayout;

    WGPUPipelineLayout pipelineLayout = wgpuDeviceCreatePipelineLayout(device, &pipelineLayoutDesc);

    WGPUBlendState blendState{
	    .color {
            .operation = WGPUBlendOperation_Add,
            .srcFactor = WGPUBlendFactor_SrcAlpha,
            .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
        },
	    .alpha{
            .operation = WGPUBlendOperation_Add,
            .srcFactor = WGPUBlendFactor_Zero,
	        .dstFactor = WGPUBlendFactor_One,
        }
    };
	
	WGPUColorTargetState colorTarget{
	    .format = gSurfaceFormat,
        .blend = nullptr, // &blendState,
	    .writeMask = WGPUColorWriteMask_All, // We could write to only some of the color channels.
    };

    WGPUFragmentState fragment{
        .module = shaderModule,
        .entryPoint = WEBGPU_STR("fs_main"),
        .targetCount = 1,
        .targets = &colorTarget,
    };

    WGPURenderPipelineDescriptor pipelineDesc{
        .layout = pipelineLayout,
        .vertex { 
            .module = shaderModule,
            .entryPoint = WEBGPU_STR("vs_main"),
          },
        .primitive {
             .topology = WGPUPrimitiveTopology_TriangleList, 
             .stripIndexFormat = WGPUIndexFormat_Undefined,
             .frontFace = WGPUFrontFace_CCW,
             .cullMode = WGPUCullMode_Back,
            },
        .multisample {
            .count = 1,
            .mask = ~0u,
            .alphaToCoverageEnabled = false
        },
        .fragment = &fragment,
    };

    gPipeline = wgpuDeviceCreateRenderPipeline(device, &pipelineDesc);

    // ---------------------------------------------------------------------------
    // Cluster culling compute pipeline
    // ---------------------------------------------------------------------------
    {
        const char cullShaderCode[] = R"(
struct CameraUniforms {
    view             : mat4x4<f32>,
    proj             : mat4x4<f32>,
    cameraPos        : vec3<f32>,
    _pad0            : f32,
    clusterThreshold : f32,
    cameraProj       : f32,
    znear            : f32,
    _pad1            : f32,
}

struct ClusterN {
    refined              : i32,
    groupCenterX         : f32,
    groupCenterY         : f32,
    groupCenterZ         : f32,
    groupRadius          : f32,
    groupError           : f32,
    refinedCenterX       : f32,
    refinedCenterY       : f32,
    refinedCenterZ       : f32,
    refinedRadius        : f32,
    refinedError         : f32,
    meshletVertexOffset  : u32,
    meshletTriangleOffset: u32,
    vertexCount          : u32,
    triangleCount        : u32,
}

struct VisibleClusters {
    count : atomic<u32>,
    ids   : array<u32>,
}

@group(0) @binding(0) var<uniform>             camera   : CameraUniforms;
@group(0) @binding(1) var<storage, read>       clusters : array<ClusterN>;
@group(0) @binding(2) var<storage, read_write> visible  : VisibleClusters;

fn bounds_error(center: vec3<f32>, error: f32, radius: f32) -> f32 {
    let d = max(length(center - camera.cameraPos) - radius, camera.znear);
    return error / d * (camera.cameraProj * 0.5);
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&clusters)) { return; }

    let c             = clusters[idx];
    let groupCenter   = vec3<f32>(c.groupCenterX,   c.groupCenterY,   c.groupCenterZ);
    let refinedCenter = vec3<f32>(c.refinedCenterX, c.refinedCenterY, c.refinedCenterZ);

    let groupError   = bounds_error(groupCenter,   c.groupError,   c.groupRadius);
    let refinedError = bounds_error(refinedCenter, c.refinedError, c.refinedRadius);

    let render = groupError > camera.clusterThreshold &&
                (c.refined < 0 || refinedError <= camera.clusterThreshold);

    if (render) {
        let slot = atomicAdd(&visible.count, 1u);
        visible.ids[slot] = idx;
    }
}
        )";

        WGPUShaderModule cullShader = createShaderModule(device, WEBGPU_STR(cullShaderCode));
        defer { wgpuShaderModuleRelease(cullShader); };

        WGPUBindGroupLayoutEntry cullEntries[3] = {
            {
                .binding    = 0,
                .visibility = WGPUShaderStage_Compute,
                .buffer     = { .type = WGPUBufferBindingType_Uniform, .minBindingSize = sizeof(CameraUniforms) },
            },
            {
                .binding    = 1,
                .visibility = WGPUShaderStage_Compute,
                .buffer     = { .type = WGPUBufferBindingType_ReadOnlyStorage },
            },
            {
                .binding    = 2,
                .visibility = WGPUShaderStage_Compute,
                .buffer     = { .type = WGPUBufferBindingType_Storage },
            },
        };
        WGPUBindGroupLayoutDescriptor cullBglDesc{
            .entryCount = 3,
            .entries    = cullEntries,
        };
        gCullBindGroupLayout = wgpuDeviceCreateBindGroupLayout(device, &cullBglDesc);

        WGPUPipelineLayoutDescriptor cullPlDesc{
            .bindGroupLayoutCount = 1,
            .bindGroupLayouts     = &gCullBindGroupLayout,
        };
        WGPUPipelineLayout cullLayout = wgpuDeviceCreatePipelineLayout(device, &cullPlDesc);
        defer { wgpuPipelineLayoutRelease(cullLayout); };

        WGPUComputePipelineDescriptor cullCpDesc{
            .label   = WEBGPU_STR("cluster cull pipeline"),
            .layout  = cullLayout,
            .compute = { .module = cullShader, .entryPoint = WEBGPU_STR("cs_main") },
        };
        gCullPipeline = wgpuDeviceCreateComputePipeline(device, &cullCpDesc);
    }

    // ---------------------------------------------------------------------------
    // Cluster index-expansion compute pipeline
    // ---------------------------------------------------------------------------
    //
    // Takes the surviving cluster IDs from the cull pass and expands each cluster's
    // triangle list into a flat u32 index buffer ready for a single indexed draw call.
    //
    // Index encoding (TelluSim mesh-shader-emulation style):
    //   packed_index = (cluster_id << 8) | local_vertex_index
    //
    // The vertex shader decodes:
    //   cluster_id      = vertex_index >> 8
    //   local_vtx_index = vertex_index & 0xFF
    // then looks up:   meshletVertexBuffer[cluster.meshletVertexOffset + local_vtx_index]
    // to get the real vertex position.
    //
    // Triangle indices in meshletTriangleBuffer are uint8 packed into u32 words.
    // One workgroup (128 threads) handles one visible cluster; each thread = one triangle.
    // ---------------------------------------------------------------------------
    {
        const char expandShaderCode[] = R"(

// ------------------------------------------------------------------
// Structs
// ------------------------------------------------------------------

struct ClusterN {
    refined              : i32,
    groupCenterX         : f32,
    groupCenterY         : f32,
    groupCenterZ         : f32,
    groupRadius          : f32,
    groupError           : f32,
    refinedCenterX       : f32,
    refinedCenterY       : f32,
    refinedCenterZ       : f32,
    refinedRadius        : f32,
    refinedError         : f32,
    meshletVertexOffset  : u32,
    meshletTriangleOffset: u32,
    vertexCount          : u32,
    triangleCount        : u32,
}

// Read-only view of the cull pass output (count is non-atomic here)
struct VisibleClusters {
    count : u32,
    ids   : array<u32>,
}

// Mirrors DrawIndexedIndirectParams; index_count is atomic so the
// workgroups can each reserve a contiguous slice without a prefix-sum pass.
struct DrawArgs {
    index_count    : atomic<u32>,
    instance_count : u32,
    first_index    : u32,
    base_vertex    : i32,
    first_instance : u32,
}

@group(0) @binding(0) var<storage, read>       visible      : VisibleClusters;
@group(0) @binding(1) var<storage, read>       clusters     : array<ClusterN>;
@group(0) @binding(2) var<storage, read>       meshlet_tris : array<u32>;  // packed uint8 triangle indices
@group(0) @binding(3) var<storage, read_write> draw_args    : DrawArgs;
@group(0) @binding(4) var<storage, read_write> out_indices  : array<u32>;


var<workgroup> ws_base    : u32;       // base offset in out_indices reserved by this workgroup
var<workgroup> ws_cluster : ClusterN;  // cluster data loaded by thread 0

// Read a single uint8 packed inside a u32 word array
fn read_u8(buf: ptr<storage, array<u32>, read>, byte_offset: u32) -> u32 {
    let word  = (*buf)[byte_offset / 4u];
    let shift = (byte_offset % 4u) * 8u;
    return (word >> shift) & 0xFFu;
}


// one workgroup per visible cluster, one thread per triangle
@compute @workgroup_size(128)
fn cs_main(
    @builtin(workgroup_id)        wgid: vec3<u32>,
    @builtin(local_invocation_id) lid:  vec3<u32>
) {
    // guard: more workgroups may be dispatched than visible clusters
    if (wgid.x >= visible.count) { return; }

    let cluster_id = visible.ids[wgid.x];

    // thread 0: load cluster metadata and atomically reserve output slots
    if (lid.x == 0u) {
        ws_cluster = clusters[cluster_id];
        ws_base    = atomicAdd(&draw_args.index_count, ws_cluster.triangleCount * 3u);
    }
    workgroupBarrier();

    let tri_idx = lid.x;
    if (tri_idx >= ws_cluster.triangleCount) { return; }

    // unpack 3 uint8 local vertex indices for this triangle
    let tri_byte = ws_cluster.meshletTriangleOffset + tri_idx * 3u;

    let i0 = read_u8(&meshlet_tris, tri_byte + 0u);
    let i1 = read_u8(&meshlet_tris, tri_byte + 1u);
    let i2 = read_u8(&meshlet_tris, tri_byte + 2u);

    // encode: cluster_id in upper 24 bits, local vertex index in lower 8 bits
    let out_base = ws_base + tri_idx * 3u;
    out_indices[out_base + 0u] = (cluster_id << 8u) | i0;
    out_indices[out_base + 1u] = (cluster_id << 8u) | i1;
    out_indices[out_base + 2u] = (cluster_id << 8u) | i2;
}
        )";

        WGPUShaderModule expandShader = createShaderModule(device, WEBGPU_STR(expandShaderCode));
        defer { wgpuShaderModuleRelease(expandShader); };

        WGPUBindGroupLayoutEntry expandEntries[5] = {
            { .binding = 0, .visibility = WGPUShaderStage_Compute,
              .buffer = { .type = WGPUBufferBindingType_ReadOnlyStorage } },  // visible clusters
            { .binding = 1, .visibility = WGPUShaderStage_Compute,
              .buffer = { .type = WGPUBufferBindingType_ReadOnlyStorage } },  // cluster metadata
            { .binding = 2, .visibility = WGPUShaderStage_Compute,
              .buffer = { .type = WGPUBufferBindingType_ReadOnlyStorage } },  // meshlet triangle buffer
            { .binding = 3, .visibility = WGPUShaderStage_Compute,
              .buffer = { .type = WGPUBufferBindingType_Storage } },          // draw args (atomic count)
            { .binding = 4, .visibility = WGPUShaderStage_Compute,
              .buffer = { .type = WGPUBufferBindingType_Storage } },          // output index buffer
        };
        WGPUBindGroupLayoutDescriptor expandBglDesc{
            .entryCount = 5,
            .entries    = expandEntries,
        };
        gExpandBindGroupLayout = wgpuDeviceCreateBindGroupLayout(device, &expandBglDesc);

        WGPUPipelineLayoutDescriptor expandPlDesc{
            .bindGroupLayoutCount = 1,
            .bindGroupLayouts     = &gExpandBindGroupLayout,
        };
        WGPUPipelineLayout expandLayout = wgpuDeviceCreatePipelineLayout(device, &expandPlDesc);
        defer { wgpuPipelineLayoutRelease(expandLayout); };

        WGPUComputePipelineDescriptor expandCpDesc{
            .label   = WEBGPU_STR("cluster expand pipeline"),
            .layout  = expandLayout,
            .compute = { .module = expandShader, .entryPoint = WEBGPU_STR("cs_main") },
        };
        gExpandPipeline = wgpuDeviceCreateComputePipeline(device, &expandCpDesc);
    }

    return true;
}

void initClusterCullPass(const ClusteredMeshGPU& mesh)
{
    if (gVisibleClustersBuffer) { wgpuBufferRelease(gVisibleClustersBuffer); gVisibleClustersBuffer = {}; }
    if (gCullBindGroup)         { wgpuBindGroupRelease(gCullBindGroup);      gCullBindGroup = {};         }

    gCullClusterCount = mesh.clusterCount;

    // buffer layout: u32 atomic count + u32[clusterCount] ids
    const uint64_t bufSize = sizeof(uint32_t) + (uint64_t)mesh.clusterCount * sizeof(uint32_t);

    WGPUBufferDescriptor bufDesc{
        .label = WEBGPU_STR("visible clusters"),
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst,
        .size  = bufSize,
    };
    gVisibleClustersBuffer = wgpuDeviceCreateBuffer(gDevice, &bufDesc);

    WGPUBindGroupEntry entries[3] = {
        { .binding = 0, .buffer = viewUniforms,           .offset = 0, .size = sizeof(CameraUniforms) },
        { .binding = 1, .buffer = mesh.clusterBuffer,     .offset = 0, .size = (uint64_t)mesh.clusterCount * sizeof(ClusterN) },
        { .binding = 2, .buffer = gVisibleClustersBuffer, .offset = 0, .size = bufSize },
    };
    WGPUBindGroupDescriptor bgDesc{
        .layout     = gCullBindGroupLayout,
        .entryCount = 3,
        .entries    = entries,
    };
    gCullBindGroup = wgpuDeviceCreateBindGroup(gDevice, &bgDesc);
}

void initClusterExpandPass(const ClusteredMeshGPU& mesh)
{
    if (gOutputIndexBuffer) { wgpuBufferRelease(gOutputIndexBuffer); gOutputIndexBuffer = {}; }
    if (gDrawArgsBuffer)    { wgpuBufferRelease(gDrawArgsBuffer);    gDrawArgsBuffer    = {}; }
    if (gExpandBindGroup)   { wgpuBindGroupRelease(gExpandBindGroup); gExpandBindGroup  = {}; }

    gExpandClusterCount = mesh.clusterCount;

    // worst-case output: every triangle from every cluster is visible
    // meshletTriangleByteCount = total_triangles * 3 (one byte per local index)
    // output indices: total_triangles * 3 u32 values
    const uint32_t totalTriangles = mesh.meshletTriangleByteCount / 3u;
    const uint64_t indexBufSize   = (uint64_t)totalTriangles * 3u * sizeof(uint32_t);

    {
        WGPUBufferDescriptor desc{
            .label = WEBGPU_STR("output index buffer"),
            .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_Index,
            .size  = indexBufSize,
        };
        gOutputIndexBuffer = wgpuDeviceCreateBuffer(gDevice, &desc);
    }

    {
        // DrawIndexedIndirectParams layout (5 x u32 = 20 bytes):
        //   index_count, instance_count, first_index, base_vertex, first_instance
        // index_count is reset each frame; the rest are constant.
        WGPUBufferDescriptor desc{
            .label = WEBGPU_STR("draw args"),
            .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_Indirect | WGPUBufferUsage_CopyDst,
            .size  = sizeof(DrawIndexedIndirectParams),
        };
        gDrawArgsBuffer = wgpuDeviceCreateBuffer(gDevice, &desc);

        // pre-fill the constant fields; index_count will be reset per-frame
        DrawIndexedIndirectParams initArgs{ .instanceCount = 1 };
        wgpuQueueWriteBuffer(gQueue, gDrawArgsBuffer, 0, &initArgs, sizeof(initArgs));
    }

    // meshlet triangle buffer size padded to 4-byte alignment (as stored on GPU)
    const uint64_t trisBufSize = ((uint64_t)mesh.meshletTriangleByteCount + 3u) & ~3ull;

    WGPUBindGroupEntry expandEntries[5] = {
        { .binding = 0, .buffer = gVisibleClustersBuffer,       .offset = 0,
          .size = sizeof(uint32_t) + (uint64_t)mesh.clusterCount * sizeof(uint32_t) },
        { .binding = 1, .buffer = mesh.clusterBuffer,           .offset = 0,
          .size = (uint64_t)mesh.clusterCount * sizeof(ClusterN) },
        { .binding = 2, .buffer = mesh.meshletTriangleBuffer,   .offset = 0,
          .size = trisBufSize },
        { .binding = 3, .buffer = gDrawArgsBuffer,              .offset = 0,
          .size = sizeof(DrawIndexedIndirectParams) },
        { .binding = 4, .buffer = gOutputIndexBuffer,           .offset = 0,
          .size = indexBufSize },
    };
    WGPUBindGroupDescriptor bgDesc{
        .layout     = gExpandBindGroupLayout,
        .entryCount = 5,
        .entries    = expandEntries,
    };
    gExpandBindGroup = wgpuDeviceCreateBindGroup(gDevice, &bgDesc);
}

void initClusteredRenderPipeline(const ClusteredMeshGPU& mesh)
{
    if (gClusteredRenderBG)  { wgpuBindGroupRelease(gClusteredRenderBG);         gClusteredRenderBG  = {}; }
    if (gClusteredPipeline)  { wgpuRenderPipelineRelease(gClusteredPipeline);     gClusteredPipeline  = {}; }
    if (gClusteredRenderBGL) { wgpuBindGroupLayoutRelease(gClusteredRenderBGL);   gClusteredRenderBGL = {}; }

    const char clusteredShaderCode[] = R"(

struct CameraUniforms {
    view : mat4x4<f32>,
    proj : mat4x4<f32>,
}

struct ClusterN {
    refined              : i32,
    groupCenterX         : f32,
    groupCenterY         : f32,
    groupCenterZ         : f32,
    groupRadius          : f32,
    groupError           : f32,
    refinedCenterX       : f32,
    refinedCenterY       : f32,
    refinedCenterZ       : f32,
    refinedRadius        : f32,
    refinedError         : f32,
    meshletVertexOffset  : u32,
    meshletTriangleOffset: u32,
    vertexCount          : u32,
    triangleCount        : u32,
}

// must match MeshVertex in C++ (8 x f32 = 32 bytes)
struct MeshVertex {
    x:  f32, y:  f32, z:  f32,
    nx: f32, ny: f32, nz: f32,
    tu: f32, tv: f32,
}

@group(0) @binding(0) var<uniform>        camera        : CameraUniforms;
@group(0) @binding(1) var<storage, read>  clusters      : array<ClusterN>;
@group(0) @binding(2) var<storage, read>  meshlet_verts : array<u32>;
@group(0) @binding(3) var<storage, read>  vertices      : array<MeshVertex>;

struct VertexOut {
    @builtin(position) pos   : vec4<f32>,
    @location(0)       color : vec3<f32>,
    @location(1)       normal: vec3<f32>,
}

// low-cost hash-based per-cluster color for debug visualization
fn cluster_color(id: u32) -> vec3<f32> {
    let h = id * 2246822519u + 2654435761u;
    let r = f32((h              ) & 0xFFu) / 255.0;
    let g = f32((h >> 8u        ) & 0xFFu) / 255.0;
    let b = f32((h >> 16u       ) & 0xFFu) / 255.0;
    return vec3<f32>(r, g, b);
}

@vertex
fn vs_main(@builtin(vertex_index) packed: u32) -> VertexOut {
    // TelluSim decoding: cluster in upper 24 bits, local vertex index in lower 8
    let cluster_id = packed >> 8u;
    let local_idx  = packed & 0xFFu;

    let cluster = clusters[cluster_id];
    let global_vtx = meshlet_verts[cluster.meshletVertexOffset + local_idx];
    let v = vertices[global_vtx];

    var out: VertexOut;
    out.pos = camera.proj * camera.view * vec4<f32>(v.x, v.y, v.z, 1.0);
    out.color = cluster_color(cluster_id);
    out.normal = vec3<f32>(v.nx, v.ny, v.nz);
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // simple diffuse with a fixed light direction, tinted by cluster color
    let light_dir = normalize(vec3<f32>(0.4, 1.0, 0.6));
    let diffuse   = max(dot(normalize(in.normal), light_dir), 0.08);
    return vec4<f32>(in.color, 1.0);
}
    )";

    WGPUShaderModule shader = createShaderModule(gDevice, WEBGPU_STR(clusteredShaderCode));
    defer { wgpuShaderModuleRelease(shader); };

    WGPUBindGroupLayoutEntry entries[4] = {
        { .binding = 0, .visibility = WGPUShaderStage_Vertex,
          .buffer = { .type = WGPUBufferBindingType_Uniform, .minBindingSize = sizeof(CameraUniforms) } },
        { .binding = 1, .visibility = WGPUShaderStage_Vertex,
          .buffer = { .type = WGPUBufferBindingType_ReadOnlyStorage } },
        { .binding = 2, .visibility = WGPUShaderStage_Vertex,
          .buffer = { .type = WGPUBufferBindingType_ReadOnlyStorage } },
        { .binding = 3, .visibility = WGPUShaderStage_Vertex,
          .buffer = { .type = WGPUBufferBindingType_ReadOnlyStorage } },
    };
    WGPUBindGroupLayoutDescriptor bglDesc{ .entryCount = 4, .entries = entries };
    gClusteredRenderBGL = wgpuDeviceCreateBindGroupLayout(gDevice, &bglDesc);

    WGPUPipelineLayoutDescriptor plDesc{
        .bindGroupLayoutCount = 1,
        .bindGroupLayouts     = &gClusteredRenderBGL,
    };
    WGPUPipelineLayout layout = wgpuDeviceCreatePipelineLayout(gDevice, &plDesc);
    defer { wgpuPipelineLayoutRelease(layout); };

    WGPUBlendState blend{
        .color = { .operation = WGPUBlendOperation_Add,
                   .srcFactor = WGPUBlendFactor_SrcAlpha,
                   .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha },
        .alpha = { .operation = WGPUBlendOperation_Add,
                   .srcFactor = WGPUBlendFactor_Zero,
                   .dstFactor = WGPUBlendFactor_One },
    };
    WGPUDepthStencilState depthStencil{
        .format = WGPUTextureFormat_Depth24Plus,
        .depthWriteEnabled = WGPUOptionalBool_True,
        .depthCompare = WGPUCompareFunction_Less,
    };
    WGPUColorTargetState colorTarget{
        .format    = gSurfaceFormat,
        .blend     = nullptr,
        .writeMask = WGPUColorWriteMask_All,
    };
    WGPUFragmentState fragment{
        .module      = shader,
        .entryPoint  = WEBGPU_STR("fs_main"),
        .targetCount = 1,
        .targets     = &colorTarget,
    };
    WGPURenderPipelineDescriptor pipelineDesc{
        .layout   = layout,
        .vertex   = { .module = shader, .entryPoint = WEBGPU_STR("vs_main") },
        .primitive = {
            .topology         = WGPUPrimitiveTopology_TriangleList,
            .stripIndexFormat = WGPUIndexFormat_Undefined,
            .frontFace        = WGPUFrontFace_CCW,
            .cullMode         = WGPUCullMode_Back,
        },
        .depthStencil = &depthStencil,
        .multisample = { .count = 1, .mask = ~0u },
        .fragment    = &fragment,
    };
    gClusteredPipeline = wgpuDeviceCreateRenderPipeline(gDevice, &pipelineDesc);

    WGPUBindGroupEntry bgEntries[4] = {
        { .binding = 0, .buffer = viewUniforms,             .offset = 0, .size = sizeof(CameraUniforms) },
        { .binding = 1, .buffer = mesh.clusterBuffer,       .offset = 0, .size = (uint64_t)mesh.clusterCount      * sizeof(ClusterN)   },
        { .binding = 2, .buffer = mesh.meshletVertexBuffer, .offset = 0, .size = (uint64_t)mesh.meshletVertexCount * sizeof(uint32_t)   },
        { .binding = 3, .buffer = mesh.vertexBuffer,        .offset = 0, .size = (uint64_t)mesh.vertexCount        * sizeof(MeshVertex) },
    };
    WGPUBindGroupDescriptor bgDesc2{
        .layout     = gClusteredRenderBGL,
        .entryCount = 4,
        .entries    = bgEntries,
    };
    gClusteredRenderBG = wgpuDeviceCreateBindGroup(gDevice, &bgDesc2);
}

void loadMeshForRendering(const std::string& path)
{
    static ClusteredMeshGPU sMesh{};

    if (!loadGltfMeshToGPU(path, gDevice, gQueue, sMesh))
    {
        qWarning() << "loadMeshForRendering: failed to load" << path;
        return;
    }

    initClusterCullPass(sMesh);
    initClusterExpandPass(sMesh);
    initClusteredRenderPipeline(sMesh);

    qDebug() << "Mesh loaded and clustered render pipeline ready.";
}

static uint32_t gWidth = -1;
static uint32_t gHeight = -1;

void resize(uint32_t width, uint32_t height)
{
    gWidth = width;
    gHeight = height;
    assert(!(gWidth == -1 || gHeight == -1));

    WGPUSurfaceConfiguration config = {
        .device = gDevice,
        .format = gSurfaceFormat,
        .usage = WGPUTextureUsage_RenderAttachment,
        .width = width,
        .height = height,
        .alphaMode = WGPUCompositeAlphaMode_Auto,
        .presentMode = WGPUPresentMode_Fifo,
    };

    wgpuSurfaceConfigure(gSurface, &config);

    WGPUTextureDescriptor depthDesc{
        .usage = WGPUTextureUsage_RenderAttachment,
        .dimension = WGPUTextureDimension_2D,
        .size = { gWidth, gHeight, 1 },
        .format = WGPUTextureFormat_Depth24Plus,
        .mipLevelCount = 1,
        .sampleCount = 1,
    };

    gDepthTexture = wgpuDeviceCreateTexture(gDevice, &depthDesc);
    gDepthView = wgpuTextureCreateView(gDepthTexture, nullptr);
}

WGPUTextureView get_next_swapchain_view(WGPUSurface surface)
{
    WGPUSurfaceTexture surfaceTexture{};
    wgpuSurfaceGetCurrentTexture(surface, &surfaceTexture);

    // SuccessOptimal & SuccessSuboptimal are ok
    switch (surfaceTexture.status)
    {
    case WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal:
    case WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal:
        break;
    default:
        qDebug() << "Could not get current surface texture: surface_texture.status=" << surfaceTexture.status;
        return nullptr;
    }

    WGPUTextureViewDescriptor viewDesc =
    {
        .format = gSurfaceFormat, // TODO: handle format
        .dimension = WGPUTextureViewDimension_2D,
        .baseMipLevel = 0,
        .mipLevelCount = 1,
        .baseArrayLayer = 0,
        .arrayLayerCount = 1,
        .aspect = WGPUTextureAspect_All,
    };
    return wgpuTextureCreateView(surfaceTexture.texture, &viewDesc);
}




void drawFrame(int32_t lookDx, int32_t lookDy, float moveForward, float moveRight)
{
    // process all internal async events or error callbacks
    wgpuInstanceProcessEvents(gInstance);

    // skip rendering when no valid window size
    if(gWidth == 0 || gHeight == 0)
    {
        qInfo("No valid window size");
        return;
    }

    WGPUTextureView view = get_next_swapchain_view(gSurface);
    if(view == nullptr){
        return; // skip frame
    }

    // Free-flight camera state
    static glm::vec3 position = {0.0f, 0.0f, 3.0f};
    static float yaw = 0.0f;
    static float pitch = 0.0f;

    constexpr float kLookSensitivity = 0.003f;
    constexpr float kMoveSpeed = 0.3f;

    yaw += lookDx * kLookSensitivity;
    pitch += lookDy * kLookSensitivity;
    pitch = glm::clamp(pitch, -glm::half_pi<float>() + 0.01f, glm::half_pi<float>() - 0.01f);

    // forward vector from yaw/pitch
    glm::vec3 forward = glm::normalize(glm::vec3(
         glm::cos(pitch) * glm::sin(yaw),
        -glm::sin(pitch),
         glm::cos(pitch) * glm::cos(yaw)
    ));
    glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));

    position += forward * moveForward * kMoveSpeed;
    position += right * moveRight * kMoveSpeed;

    CameraUniforms cameraData{};
    cameraData.view = glm::lookAt(position, position + forward, glm::vec3(0.0f, 1.0f, 0.0f));
    cameraData.proj = glm::perspective(glm::radians(60.0f),
                                       (float)gWidth / (float)gHeight,
                                       0.01f, 1000.0f);
    cameraData.cameraPos        = glm::vec3(position);
    cameraData.clusterThreshold = 0.002f;
    cameraData.cameraProj       = cameraData.proj[1][1]; // 1/tan(fov/2)
    cameraData.znear             = 0.01f;

    wgpuQueueWriteBuffer(gQueue, viewUniforms, 0, & cameraData, sizeof(CameraUniforms));

    WGPUCommandEncoder encoder =  wgpuDeviceCreateCommandEncoder(gDevice, NULL);

    // cluster culling compute pass
    if (gCullBindGroup)
    {
        // reset atomic count to 0 before dispatch
        const uint32_t zero = 0;
        wgpuQueueWriteBuffer(gQueue, gVisibleClustersBuffer, 0, &zero, sizeof(zero));

        WGPUComputePassEncoder cpass = ComputePassBuilder{}
            .set_label("cluster cull")
            .begin(encoder);
        wgpuComputePassEncoderSetPipeline(cpass, gCullPipeline);
        wgpuComputePassEncoderSetBindGroup(cpass, 0, gCullBindGroup, 0, nullptr);
        const uint32_t groups = (gCullClusterCount + 63u) / 64u;
        wgpuComputePassEncoderDispatchWorkgroups(cpass, gCullClusterCount, 1, 1);
        wgpuComputePassEncoderEnd(cpass);
        wgpuComputePassEncoderRelease(cpass);
    }

    // cluster index-expansion compute pass
    if (gExpandBindGroup)
    {
        // reset draw_args.index_count to 0; keep instance_count=1 and other fields intact
        const uint32_t zero = 0;
        wgpuQueueWriteBuffer(gQueue, gDrawArgsBuffer, 0, &zero, sizeof(zero));

        WGPUComputePassEncoder epass = ComputePassBuilder{}
            .set_label("cluster expand")
            .begin(encoder);
        wgpuComputePassEncoderSetPipeline(epass, gExpandPipeline);
        wgpuComputePassEncoderSetBindGroup(epass, 0, gExpandBindGroup, 0, nullptr);
        // dispatch one workgroup per cluster (extras early-out via visible.count guard)
        wgpuComputePassEncoderDispatchWorkgroups(epass, gExpandClusterCount, 1, 1);
  
        wgpuComputePassEncoderEnd(epass);
        wgpuComputePassEncoderRelease(epass);
    }

    WGPURenderPassEncoder pass = RenderPassBuilder{}
        .color(view, WGPULoadOp_Clear, WGPUStoreOp_Store, {1.0, 1.0, 0.6, 1.0})
        .depth_stencil(gDepthView)
        .begin(encoder);

    if (gClusteredRenderBG && gClusteredPipeline && gOutputIndexBuffer && gDrawArgsBuffer)
    {
        // clustered indirect indexed draw
        wgpuRenderPassEncoderSetPipeline(pass, gClusteredPipeline);
        wgpuRenderPassEncoderSetBindGroup(pass, 0, gClusteredRenderBG, 0, nullptr);
        wgpuRenderPassEncoderSetIndexBuffer(pass, gOutputIndexBuffer, WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
        wgpuRenderPassEncoderDrawIndexedIndirect(pass, gDrawArgsBuffer, 0);
    }
    else
    {
        // fallback: single hardcoded triangle
        wgpuRenderPassEncoderSetPipeline(pass, gPipeline);
        wgpuRenderPassEncoderSetBindGroup(pass, 0, cameraBindGroup, 0, nullptr);
        wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    }

    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);

    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, NULL);

    wgpuQueueSubmit(gQueue, 1, &cmd);

    wgpuSurfacePresent(gSurface);

    // release swapchain texture view and command buffer
    wgpuTextureViewRelease(view);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);
}

void shutdownBacken()
{
    if (gClusteredRenderBG)      { wgpuBindGroupRelease(gClusteredRenderBG);          gClusteredRenderBG  = {}; }
    if (gClusteredPipeline)      { wgpuRenderPipelineRelease(gClusteredPipeline);      gClusteredPipeline  = {}; }
    if (gClusteredRenderBGL)     { wgpuBindGroupLayoutRelease(gClusteredRenderBGL);    gClusteredRenderBGL = {}; }

    if (gExpandBindGroup)        { wgpuBindGroupRelease(gExpandBindGroup);        gExpandBindGroup        = {}; }
    if (gOutputIndexBuffer)      { wgpuBufferRelease(gOutputIndexBuffer);          gOutputIndexBuffer      = {}; }
    if (gDrawArgsBuffer)         { wgpuBufferRelease(gDrawArgsBuffer);             gDrawArgsBuffer         = {}; }
    if (gExpandPipeline)         { wgpuComputePipelineRelease(gExpandPipeline);    gExpandPipeline         = {}; }
    if (gExpandBindGroupLayout)  { wgpuBindGroupLayoutRelease(gExpandBindGroupLayout); gExpandBindGroupLayout = {}; }

    if (gCullBindGroup)          { wgpuBindGroupRelease(gCullBindGroup);           gCullBindGroup          = {}; }
    if (gVisibleClustersBuffer)  { wgpuBufferRelease(gVisibleClustersBuffer);      gVisibleClustersBuffer  = {}; }
    if (gCullPipeline)           { wgpuComputePipelineRelease(gCullPipeline);      gCullPipeline           = {}; }
    if (gCullBindGroupLayout)    { wgpuBindGroupLayoutRelease(gCullBindGroupLayout); gCullBindGroupLayout  = {}; }

    wgpuSurfaceUnconfigure(gSurface);
    wgpuQueueRelease(gQueue);
    wgpuSurfaceRelease(gSurface);
    wgpuDeviceRelease(gDevice);
    wgpuAdapterRelease(gAdapter);
    wgpuInstanceRelease(gInstance);
}

static void OnDeviceLost(WGPUDeviceLostReason reason, char const* message, void* userdata)
{
    const char* deviceLostReasons[] =
        {
            "NULL", // invalid index
            "Unknown",
            "Destroyed",
            "CallbackCancelled",
            "FailedCreatio",
        };

    int idx = reason > countOf(deviceLostReasons) ? countOf(deviceLostReasons) : reason;
    qCritical("Device lost! Reason: %s\n", deviceLostReasons[idx]);
    if (message) {
        qFatal("Message: %s\n", message);
    }
}

bool IRenderer::register_subsystem(Framework* framework)
{
    // subscribe to events
    QObject::connect(framework, &Framework::on_window_created, this, &IRenderer::window_created);
    QObject::connect(framework, &Framework::on_window_resize,  this, &IRenderer::window_resized);
    QObject::connect(framework, &Framework::on_mouse_move,     this, &IRenderer::mouse_moved);
    QObject::connect(framework, &Framework::on_mouse_button,   this, &IRenderer::mouse_button);
    QObject::connect(framework, &Framework::on_key_event,      this, &IRenderer::key_event);

    return true;
}

void IRenderer::mouse_moved(int dx, int dy)
{    
    // only rotate camera if right mouse button is down
    if (!m_rmb) return;
    m_pendingDx += dx;
    m_pendingDy += dy;
}

void IRenderer::mouse_button(int button, bool pressed)
{
    if (button == SDL_BUTTON_RIGHT)
        m_rmb = pressed;
}

void IRenderer::key_event(int key, bool pressed)
{
    switch (key)
    {
        case SDLK_w: m_keyW = pressed; break;
        case SDLK_s: m_keyS = pressed; break;
        case SDLK_a: m_keyA = pressed; break;
        case SDLK_d: m_keyD = pressed; break;
    }
}

void IRenderer::render()
{
    float moveForward = (m_keyW ? 1.0f : 0.0f) - (m_keyS ? 1.0f : 0.0f);
    float moveRight   = (m_keyD ? 1.0f : 0.0f) - (m_keyA ? 1.0f : 0.0f);

    drawFrame(m_pendingDx, m_pendingDy, moveForward, moveRight);
    m_pendingDx = 0;
    m_pendingDy = 0;
}

void IRenderer::window_created(WindowHandel window) 
{ 
    // hardcoded sdl backend for now...
    SDL_Window* sdlWindow = static_cast<SDL_Window*>(window.handle);
    initWebGPUBackend(sdlWindow);

    int width{}, height{};
    SDL_GetWindowSizeInPixels(sdlWindow, &width, &height);
    resize((uint32_t)width, (uint32_t)height); // needs one resize event
}

void IRenderer::window_resized(int width, int height) 
{ 
    resize((uint32_t)width, (uint32_t)height);
}
