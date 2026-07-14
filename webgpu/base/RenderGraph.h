

// Immediate-mode render graph for WebGPU. Declare resources and passes per frame, then compile()
// and execute(). Pass bodies call wgpu* directly. The graph is only there for pass ordering and
// resource lifetime. Full documentation: docs/rendergraph.md.

#pragma once


#include <cstdint>
#include <initializer_list>
#include <new> // placement new
#include <string_view>
#include <type_traits>
#include <utility>
#include <QtAssert>
#include <webgpu/webgpu.h>

namespace webgpu::rg {

// Selects which encoder the pass records into and which access declarations are legal.
enum struct PassKind : uint8_t { None = 0, Graphics, Compute, Transfer };

// How a resource's size is specified.
enum struct SizeKind : uint8_t {
    Absolute, // resource uses absolute size
    Relative // resource is scaled relative to `relativeTo`
};

// Backing GPU object type of a resource.
enum struct ResourceKind : uint8_t {
    Texture,
    Buffer,
};

// Handle to a graph resource. Valid for one frame. generation catches stale handles.
struct ResourceHandle {
    uint32_t id {};
    ResourceKind kind {};
    uint64_t generation {};
};

// Ping-pong history pair. Two GPU objects rotate each frame: write .curr, read .prev.
struct HistoryResource {
    ResourceHandle curr; // used for writes
    ResourceHandle prev; // used for reads
};

namespace Internal {
struct ResourceNode;
struct PassNode;
}
struct GraphAllocator;
struct RenderGraph;

// Singly linked error chain returned by get_errors().
struct ErrorMessage {
    WGPUStringView message;
    ErrorMessage* next {};
};

// Pass body gets this during execute(). Resolves handles to GPU objects and holds the encoders.
struct PassContext {
    WGPUCommandEncoder encoder {}; // forwarded encoder, not owned by the graph
    WGPURenderPassEncoder render_pass {}; // set for Graphics passes otherwise null
    WGPUComputePassEncoder compute_pass {}; // set for Compute passes otherwise null
    WGPUQueue queue {}; // forwarded queue, not owned by the graph
    WGPUDevice device {}; // forwarded device, not owned by the graph. For creating body-local objects, e.g. a raii::BindGroup.
    RenderGraph* graph {};
    Internal::PassNode* pass {};

    // Texture view with the shape inferred from this pass's access declaration (including ViewRange).
    WGPUTextureView view(ResourceHandle h) const;
    // Single-subresource view.
    WGPUTextureView view(ResourceHandle h, uint32_t mip, uint32_t layer = 0) const;

    // Bind-group entry for a resource declared as an access in this pass. Textures get the same
    // view ctx.view(h) returns, buffers a whole-buffer binding at offset 0.
    WGPUBindGroupEntry bind(uint32_t binding, ResourceHandle h) const;
    // Single-subresource texture variant.
    WGPUBindGroupEntry bind(uint32_t binding, ResourceHandle h, uint32_t mip, uint32_t layer = 0) const;

    // Realized GPU objects and metadata. All assert the handle has a declared access in this pass.
    WGPUTexture texture(ResourceHandle h) const;
    WGPUBuffer buffer(ResourceHandle h) const;
    WGPUExtent3D texture_size(ResourceHandle h) const;
    WGPUExtent3D texture_size(ResourceHandle h, uint32_t mip) const;
    uint64_t buffer_size(ResourceHandle h) const;
    WGPUTextureFormat format(ResourceHandle h) const;
    uint32_t mip_count(ResourceHandle h) const;
    uint32_t sample_count(ResourceHandle h) const;

    // False on the frame a temporal resource's .prev was (re)created and cleared to 0 (first use,
    // resize, reset).
    bool history_valid(ResourceHandle h) const;

    // Copy-info resolvers for a copy declared in this pass, with the declared mip/layer already
    // applied. Each needs one unambiguous copy per handle and direction.
    WGPUTexelCopyTextureInfo copy_src_info(ResourceHandle h, WGPUOrigin3D origin = {}, WGPUTextureAspect aspect = WGPUTextureAspect_All) const;
    WGPUTexelCopyTextureInfo copy_dst_info(ResourceHandle h, WGPUOrigin3D origin = {}, WGPUTextureAspect aspect = WGPUTextureAspect_All) const;
    // copySize for h's texture copy: the declared subresource's width and height at its mip level.
    WGPUExtent3D copy_extent_src(ResourceHandle h) const;
    WGPUExtent3D copy_extent_dst(ResourceHandle h) const;
    // Buffer side of a texture<->buffer copy. layout is the caller's data layout. Returns {layout, buffer}.
    WGPUTexelCopyBufferInfo copy_src_buffer(ResourceHandle h, WGPUTexelCopyBufferLayout layout) const;
    WGPUTexelCopyBufferInfo copy_dst_buffer(ResourceHandle h, WGPUTexelCopyBufferLayout layout) const;

    // Resolved copy_buffer() declaration. size is already expanded and never 0.
    struct BufferCopyInfo {
        WGPUBuffer src {};
        uint64_t srcOffset {};
        WGPUBuffer dst {};
        uint64_t dstOffset {};
        uint64_t size {};
    };
    BufferCopyInfo buffer_copy_info(ResourceHandle src, ResourceHandle dst) const;
};

// Optional view shape for a sampled/storage access: dim override and mip/layer counts from
// baseMip/baseLayer. Attachments ignore it. A render target is always one mip and one layer.
struct ViewRange {
    WGPUTextureViewDimension dim = WGPUTextureViewDimension_Undefined; // Undefined = infer from texture
    uint32_t mipCount = 1; // subresource mips, from baseMip
    uint32_t layerCount = 1; // subresource layers, from baseLayer
    WGPUTextureAspect aspect = WGPUTextureAspect_All;
};

// Rounds a row's byte count up to the 256-byte alignment WebGPU requires for buffer<->texture copies.
inline uint32_t aligned_bytes_per_row(uint32_t widthTexels, uint32_t texelBlockBytes)
{
    constexpr uint32_t align = 256u;
    uint32_t raw = widthTexels * texelBlockBytes;
    return (raw + (align - 1u)) & ~(align - 1u);
}

// Access declaration interface handed to a pass's setup lambda. One call per resource use.
struct PassBuilder {
    // Color attachment.
    void color(ResourceHandle handle,
        WGPULoadOp load = WGPULoadOp_Clear,
        WGPUStoreOp store = WGPUStoreOp_Store,
        WGPUColor clear = { 0, 0, 0, 1 },
        uint32_t baseMip = 0,
        uint32_t baseLayer = 0);
    // Depth-stencil attachment.
    void depth_stencil(ResourceHandle handle,
        WGPULoadOp load = WGPULoadOp_Clear,
        WGPUStoreOp store = WGPUStoreOp_Store,
        float clearDepth = 1.0f,
        uint32_t baseMip = 0,
        uint32_t baseLayer = 0,
        WGPULoadOp stencilLoad = WGPULoadOp_Undefined,
        WGPUStoreOp stencilStore = WGPUStoreOp_Undefined,
        uint32_t stencilClear = 0);
    // Read-only depth-stencil attachment: test only, no write. WebGPU requires Undefined load/store.
    void depth_stencil_read_only(ResourceHandle handle, uint32_t baseMip = 0, uint32_t baseLayer = 0);
    // MSAA resolve target for the most recent color() in this pass. Declare it directly after that color().
    void resolve(ResourceHandle handle, uint32_t baseMip = 0, uint32_t baseLayer = 0);
    // Sampled texture, read-only in the shader. ViewRange selects a Cube/2DArray/range/aspect view.
    void sampled(ResourceHandle handle, uint32_t baseMip = 0, uint32_t baseLayer = 0, ViewRange range = {});
    // Storage texture/buffer, read-only.
    void storage_read(ResourceHandle handle, uint32_t baseMip = 0, uint32_t baseLayer = 0, ViewRange range = {});
    // Storage texture/buffer, write-only.
    void storage_write(ResourceHandle handle, uint32_t baseMip = 0, uint32_t baseLayer = 0, ViewRange range = {});
    // Storage read+write in one pass (WGSL var<storage, read_write>). In-dispatch races are the
    // shader's responsibility.
    void storage_read_write(ResourceHandle handle, uint32_t baseMip = 0, uint32_t baseLayer = 0, ViewRange range = {});
    // Uniform buffer.
    void uniform(ResourceHandle handle);
    // Buffer written from the host in this pass body (a wgpuQueueWriteBuffer).
    void host_write(ResourceHandle handle);
    // Declares CopySrc on src and CopyDst on dst.
    void copy_texture(ResourceHandle src, ResourceHandle dst, uint32_t srcMip = 0, uint32_t srcLayer = 0, uint32_t dstMip = 0, uint32_t dstLayer = 0);
    // Texture subresource to buffer, for readback. dst never counts as fully defined for aliasing.
    void copy_texture_to_buffer(ResourceHandle src, ResourceHandle dst, uint32_t srcMip = 0, uint32_t srcLayer = 0);
    // Buffer to texture subresource, for host-staged uploads.
    void copy_buffer_to_texture(ResourceHandle src, ResourceHandle dst, uint32_t dstMip = 0, uint32_t dstLayer = 0);
    // Buffer to buffer byte range. size 0 means the whole src buffer from srcOffset.
    void copy_buffer(ResourceHandle src, ResourceHandle dst, uint64_t srcOffset = 0, uint64_t dstOffset = 0, uint64_t size = 0);
    // Buffer-only: vertex, index, indirect args.
    void vertex_buffer(ResourceHandle handle);
    void index_buffer(ResourceHandle handle);
    void indirect_buffer(ResourceHandle handle);

    // Mark this pass as the initializer (bake pass) for a pool-backed resource. Runs only when the
    // target is stale: first frame, recreation, or hash mismatch (hash 0 bakes once). Only gates
    // execution.
    void initialize(ResourceHandle target, uint64_t hash = 0);

    // Exempt this pass from dead-pass culling. For side-effect-only passes: readback, profiling
    // resolve, indirect args consumed outside the graph.
    void force_keep();

    Internal::PassNode* m_pass;
    RenderGraph* m_graph {};
};

// Texture creation parameters. Size is absolute or scaled relative to another resource
struct TextureDesc {
    WGPUTextureDimension dimension = WGPUTextureDimension_Undefined;
    WGPUTextureFormat format = WGPUTextureFormat_Undefined;
    SizeKind sizeKind = SizeKind::Absolute;
    float scaleX = 1.0f;
    float scaleY = 1.0f;
    ResourceHandle relativeTo {};
    WGPUExtent3D absolute = WGPU_EXTENT_3D_INIT;
    uint32_t mipLevelCount = 1;
    uint32_t sampleCount = 1; // 1 or 4 (WebGPU 1.0 limit)
};

// Buffer creation parameters.
struct BufferDesc {
    uint64_t size {};
};

// Frame-scoped graph. Obtain via start_recording(). Invalid after the next begin_frame().
struct RenderGraph {
    // Per-frame transient GPU texture.
    ResourceHandle create_transient_texture(std::string_view id, const TextureDesc& desc);

    // Per-frame transient GPU buffer.
    ResourceHandle create_transient_buffer(std::string_view id, const BufferDesc& desc);

    // Import an external texture (e.g. swapchain). Caller owns lifetime. A null `texture` registers
    // the view only, good for sample/attach.
    ResourceHandle import_texture(std::string_view id, WGPUTextureView view, WGPUExtent3D size, WGPUTextureFormat format,
        WGPUTexture texture = nullptr, uint32_t mipCount = 1, uint32_t sampleCount = 1);

    // Import an external buffer. Caller owns lifetime.
    ResourceHandle import_buffer(std::string_view id, WGPUBuffer buffer);

    // Ping-pong texture pair that rotates each frame. hash != 0 enables hash-based invalidation.
    HistoryResource create_history_texture(std::string_view id, const TextureDesc& desc, uint64_t hash = 0);

    // Ping-pong buffer pair. Same rules as create_history_texture().
    HistoryResource create_history_buffer(std::string_view id, const BufferDesc& desc, uint64_t hash = 0);

    // Cross-frame persistent buffer.
    ResourceHandle create_persistent_buffer(std::string_view id, const BufferDesc& desc);

    // Persistent buffer initialized once from `data` (or zero-filled). The fallback for an optional binding.
    ResourceHandle create_initialized_buffer(std::string_view id, const BufferDesc& desc, const void* data = nullptr);

    // Cross-frame persistent texture.
    ResourceHandle create_persistent_texture(std::string_view id, const TextureDesc& desc);

    // Persistent texture cleared once to `fill`. The fallback for an optional sampled slot.
    ResourceHandle create_initialized_texture(std::string_view id, const TextureDesc& desc, WGPUColor fill);

    // Declare a pass. setup runs now and declares accesses. executeFn runs during execute() and
    // must be trivially destructible.
    template <typename BuilderFn, typename ExecuteFn>
    void add_pass(std::string_view id, PassKind kind, BuilderFn&& setup, ExecuteFn&& executeFn)
    {
        Q_ASSERT(!id.empty() && "must have name");
        Q_ASSERT(kind != PassKind::None);
        PassBuilder builder = begin_pass(id, kind);
        setup(builder);
        store_exec(builder, std::forward<ExecuteFn>(executeFn));
        end_pass(builder);
    }

    // Compile the declared graph. Call once after all passes are declared. Authoring errors land
    // in get_errors(). enableAlias packs disjoint-lifetime transients onto shared objects.
    void compile(bool enableAlias = true);

    // Record passes into a caller-owned encoder. enableProfiling adds per-pass GPU timestamps
    // (needs TimestampQuery). Multiple graphs per frame must be submitted in execute order on one
    // queue (transients may be shared).
    void execute(WGPUDevice device, WGPUCommandEncoder encoder, bool enableProfiling = false);
    // Kick async GPU timestamp readback. Call after queue submit. Results arrive via the instance
    // event pump a few frames later. No-op when profiling was off.
    void collect_gpu_timings();

    // Error chain produced by compile(). Non-null means the graph failed and execute() is a no-op.
    ErrorMessage* get_errors() const;

private:
    template <class F>
    void store_exec(PassBuilder& b, F&& f)
    {
        using D = std::decay_t<F>;
        static_assert(std::is_trivially_destructible_v<D>,
            "execute callback must be trivially destructible; the arena frees without running destructors, so capture handles or ids by value");
        void* m = alloc_exec(sizeof(D), alignof(D));
        Q_ASSERT_X( m, "RenderGraph", "Failed to alloc! GraphAllocator out of memory!");
        if (!m)
            return; // OOM
        ::new (m) D(std::forward<F>(f));
        set_exec(b, m, [](void* o, PassContext& c) { (*static_cast<D*>(o))(c); });
    }
    void* alloc_exec(size_t size, size_t align);
    void set_exec(PassBuilder& builder, void* obj, void (*fn)(void*, PassContext&));


    PassBuilder begin_pass(std::string_view id, PassKind kind);
    void end_pass(PassBuilder& builder);
};

// Create the allocator backing all graphs, arenas and the GPU object pool. One per device.
GraphAllocator* create_allocator();

// Destroy an allocator and everything it owns, including pooled GPU objects.
void destroy_allocator(GraphAllocator* allocator);

// Start a frame. All graphs and handles from the previous frame die here.
void begin_frame(GraphAllocator* allocator);

// Begin recording a graph. Several graphs per frame are allowed. See execute() for ordering rules.
RenderGraph* start_recording(GraphAllocator* allocator);

// End a frame. Ages the pools so unused GPU objects get evicted.
void end_frame(GraphAllocator* allocator);

} // namespace webgpu::rg
