

// Immediate-mode render graph for WebGPU. Declare resources and passes per frame; compile
// schedules them; realize and execute record the commands. No GPU-API abstraction. Pass
// bodies call wgpu* directly. The graph handles ordering and resource lifetime only.

#pragma once


#include <cstdint>
#include <new> // placement new
#include <string_view>
#include <type_traits>
#include <utility>
#include <QtAssert>
#include <webgpu/webgpu.h>

namespace webgpu {

enum struct PassKind : uint8_t { None = 0, Graphics, Compute, Transfer };

enum struct SizeKind : uint8_t {
    Absolute, // resource is uses absolute size
    Relative // resource is scaled relative to `relativeTo`
};

enum struct ResourceKind : uint8_t {
    Texture,
    Buffer,
};

// helper for const char* to fnv1a hash
inline constexpr uint64_t fnv1a(const char* s)
{
    uint64_t hash = 14695981039346656037ULL;
    while (*s) {
        hash ^= static_cast<uint8_t>(*s++);
        hash *= 1099511628211ULL;
    }
    return hash;
}

// `ResourceId` used for passes, textures and buffers
// Careful! the string internals are not owned
struct ResourceId {
    uint64_t value {};
    WGPUStringView name {};
};

// `ResourceId` literal
constexpr ResourceId operator""_rid(const char* s, size_t l) { return ResourceId(fnv1a(s), WGPUStringView { .data = s, .length = l }); }

constexpr bool operator==(ResourceId a, ResourceId b)
{
    return a.value == b.value && a.name.length == b.name.length &&
        std::string_view(a.name.data, a.name.length) == std::string_view(b.name.data, b.name.length);
}

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
} // namespace Internal
struct GraphAllocator;
struct RenderGraph;

// Error message chain for reporting errors to the client
struct ErrorMessage {
    WGPUStringView message;
    ErrorMessage* next {};
};

// Pass body gets this during execute(). Resolves handles to GPU objects and holds the encoders.
struct PassContext {
    WGPUCommandEncoder encoder {}; // forwarded encode. not owned by the graph
    WGPURenderPassEncoder render_pass {}; // set for Graphics passes otherwise null
    WGPUComputePassEncoder compute_pass {}; // set for Compute passes otherwise null
    WGPUQueue queue {}; // forwarded queue. not owned by the graph
    RenderGraph* graph {}; // not access directly
    Internal::PassNode* pass {}; // dont access directly

    // Returns a texture view for the given resource, in infered from the usage declaration
    WGPUTextureView view(ResourceHandle h) const;
    // Single-subresource view
    WGPUTextureView view(ResourceHandle h, uint32_t mip, uint32_t layer = 0) const;
    WGPUTexture texture(ResourceHandle h) const;
    WGPUBuffer buffer(ResourceHandle h) const;
    WGPUExtent3D texture_size(ResourceHandle h) const;
    WGPUExtent3D texture_size(ResourceHandle h, uint32_t mip) const;
    uint64_t buffer_size(ResourceHandle h) const;
    WGPUTextureFormat format(ResourceHandle h) const;
    uint32_t mip_count(ResourceHandle h) const;
    uint32_t sample_count(ResourceHandle h) const;

    // False on the frame a temporal resource's .prev was (re)created and cleared to 0 (first use,
    // resize, reset). Skip sampling from the .prev when this returns false.
    bool history_valid(ResourceHandle h) const;

    // Copy-info resolvers for a copy() declared in this pass. Each returns the WebGPU copy
    // descriptor for h on the given side, with the declared mip and layer already applied.
    // Declare one copy() per handle and direction in the pass, or build the infos by hand.
    WGPUTexelCopyTextureInfo copy_src_info(ResourceHandle h, WGPUOrigin3D origin = {}, WGPUTextureAspect aspect = WGPUTextureAspect_All) const;
    WGPUTexelCopyTextureInfo copy_dst_info(ResourceHandle h, WGPUOrigin3D origin = {}, WGPUTextureAspect aspect = WGPUTextureAspect_All) const;
    // copySize for h's copy(): the declared subresource's width and height at its mip level.
    WGPUExtent3D copy_extent_src(ResourceHandle h) const;
    WGPUExtent3D copy_extent_dst(ResourceHandle h) const;
    // Buffer side of a copy(). layout is the caller's data layout; returns {layout, buffer}.
    WGPUTexelCopyBufferInfo copy_src_buffer(ResourceHandle h, WGPUTexelCopyBufferLayout layout) const;
    WGPUTexelCopyBufferInfo copy_dst_buffer(ResourceHandle h, WGPUTexelCopyBufferLayout layout) const;
};

// Optional view shape for a sampled/storage read, beyond the baseMip/baseLayer the read methods
// already take. Set dim for a Cube or 2DArray view, or raise the counts for a mip/layer range.
// An Undefined dim infers 2DArray when layerCount > 1, 3D for a volume, else 2D. Attachments ignore
// this; a render target is always one mip and one layer.
struct ViewRange {
    WGPUTextureViewDimension dim = WGPUTextureViewDimension_Undefined; // Undefined = infer from texture
    uint32_t mipCount = 1; // subresource mips, from baseMip
    uint32_t layerCount = 1; // subresource layers, from baseLayer
    WGPUTextureAspect aspect = WGPUTextureAspect_All;
    WGPUTextureFormat format = WGPUTextureFormat_Undefined; // Undefined = the texture's format
};

// Rounds a row's byte count up to the 256-byte multiple WebGPU requires for a buffer/texture copy
inline uint32_t aligned_bytes_per_row(uint32_t widthTexels, uint32_t texelBlockBytes)
{
    uint32_t raw = widthTexels * texelBlockBytes;
    return (raw + 255u) & ~255u;
}

struct PassBuilder {
    // Color attachment
    void color(ResourceHandle handle,
        WGPULoadOp load = WGPULoadOp_Clear,
        WGPUStoreOp store = WGPUStoreOp_Store,
        WGPUColor clear = { 0, 0, 0, 1 },
        uint32_t baseMip = 0,
        uint32_t baseLayer = 0);
    // Depth-stencil attachment
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
    // MSAA resolve target. Call right after the color() it resolves. Must be single-sample with the
    // same format and size; Dawn validates. Can be imported, for example to resolve into the swapchain.
    void resolve(ResourceHandle handle, uint32_t baseMip = 0, uint32_t baseLayer = 0);
    // Sampled texture, read-only in the shader. Pass a ViewRange for a Cube/2DArray/range/aspect
    // view; ctx.view() then hands the body exactly that view.
    void sampled(ResourceHandle handle, uint32_t baseMip = 0, uint32_t baseLayer = 0, ViewRange range = {});
    void storage_read(ResourceHandle handle, uint32_t baseMip = 0, uint32_t baseLayer = 0, ViewRange range = {});
    void storage_write(ResourceHandle handle, uint32_t baseMip = 0, uint32_t baseLayer = 0, ViewRange range = {});
    // Storage read+write on one resource in a single pass. Maps to WGSL var<storage, read_write>.
    // In-dispatch races are the shader's responsibility via own-slot writes or atomics.
    void storage_read_write(ResourceHandle handle, uint32_t baseMip = 0, uint32_t baseLayer = 0, ViewRange range = {});
    // Uniform buffer
    void uniform(ResourceHandle handle);
    // Buffer written from the host in this pass body (a wgpuQueueWriteBuffer)
    void host_write(ResourceHandle handle);
    // Transfer passes only. Declares a copy in one call: CopySrc on src at srcMip/srcLayer, CopyDst
    // on dst at dstMip/dstLayer. mip/layer are texture-only. 0 for a buffer handle.
    // A pass may declare several copy() calls over one resource, but the ctx.copy_*_info resolvers
    // need one unambiguous handle+direction per pass; declare more and build the copy infos by hand.
    // src == dst is legal for a copy between non-overlapping subresources. A copy cannot resize, so this is not for mip-chain
    // generation; use a blit pass, sampled(baseMip=i) plus color()/storage_write(baseMip=i+1).
    void copy(ResourceHandle src, ResourceHandle dst, uint32_t srcMip = 0, uint32_t srcLayer = 0, uint32_t dstMip = 0, uint32_t dstLayer = 0);
    // Buffer-only: vertex, index, indirect args.
    void vertex_buffer(ResourceHandle handle);
    void index_buffer(ResourceHandle handle);
    void indirect_buffer(ResourceHandle handle);

    // Mark this pass as the initializer for a pool-backed resource, persistent or temporal .curr.
    // Runs only when a target needs baking: first frame, after pool eviction or recreation, or a
    // hash mismatch versus the last bake. hash 0 bakes once then skips permanently. Declare the
    // write normally with color() or storage_write(); initialize() only gates execution. Call once
    // per target to bake several outputs in one pass, up to a small fixed cap. Each target must be
    // written by the pass, and the pass re-bakes all of them if any one is stale.
    void initialize(ResourceHandle target, uint64_t hash = 0);

    // Prevent this pass from being culled even when no pass reads its output and it writes no
    // imported/persistent resource. For side-effect-only passes: GPU readback, profiling resolve,
    // indirect-arg gen consumed outside the graph.
    void force_keep();

    Internal::PassNode* m_pass;
    RenderGraph* m_graph {};
};

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

struct BufferDesc {
    uint64_t size {};
};

// Frame-scoped
struct RenderGraph {
    // Per-frame transient GPU texture.
    ResourceHandle create_transient_texture(ResourceId id, const TextureDesc& desc);
    // Import an external texture (e.g. swapchain). Caller owns lifetime.
    ResourceHandle importe_texture(ResourceId id, WGPUTextureView view, WGPUExtent3D size, WGPUTextureFormat format);
    // Per-frame transient GPU buffer.
    ResourceHandle create_transient_buffer(ResourceId id, const BufferDesc& desc);
    // Import an external buffer. Caller owns lifetime.
    ResourceHandle import_buffer(ResourceId id, WGPUBuffer buffer);

    // Ping-pong texture pair. Write .curr, read .prev; the two rotate each frame. hash != 0 enables
    // hash based invalidation. use history_valid to know if its legal to read from .prev
    HistoryResource create_history_texture(ResourceId id, const TextureDesc& desc, uint64_t hash = 0);

    // Ping-pong buffer pair. Write .curr, read .prev; the two rotate each frame. hash != 0 enables
    // hash based invalidation. use history_valid to know if its legal to read from .prev
    HistoryResource create_history_buffer(ResourceId id, const BufferDesc& desc, uint64_t hash = 0);

    // Cross-frame persistent buffer.
    ResourceHandle create_persistent_buffer(ResourceId id, const BufferDesc& desc);

    // Persistent buffer initialized from `data` (or zero-filled), reused across frames; the fallback for an optional binding.
    ResourceHandle create_initialized_buffer(ResourceId id, const BufferDesc& desc, const void* data = nullptr);

    // Creates a cross-frame persistent texture
    ResourceHandle create_persistent_texture(ResourceId id, const TextureDesc& desc);

    // Persistent texture cleared once to `fill`, reused across frames; the fallback for an optional sampled slot.
    ResourceHandle create_initialized_texture(ResourceId id, const TextureDesc& desc, WGPUColor fill);

    // Format a texture handle was declared with. Texture handles only.
    WGPUTextureFormat texture_format(ResourceHandle h) const;

    // Declare a pass
    template <typename BuilderFn, typename ExecuteFn>
    void add_pass(ResourceId id, PassKind kind, BuilderFn&& setup, ExecuteFn&& executeFn)
    {
        Q_ASSERT(id.name.length != 0 && "must have name");
        Q_ASSERT(kind != PassKind::None);
        PassBuilder builder = begin_pass(id, kind);
        setup(builder);
        store_exec(builder, std::forward<ExecuteFn>(executeFn));
        end_pass(builder);
    }

    // Compile the declared graph. Call once, after all passes are declared. On an authoring error,
    // reading a transient before its writer or a cyclic graph, it poisons the graph rather than
    // returning a status: the graph enters a Failed state, execute() becomes a no-op, and getErrors()
    // reports why.
    // EnableAlias packs disjoint-lifetime transients onto shared objects to cut peak VRAM.
    void compile(bool enableAlias = true);

    // Record passes into a caller-owned encoder. enableProfiling adds per-pass GPU timestamps and
    // needs the TimestampQuery feature; pair it with collect_gpu_timings() after submit.
    // With several graphs per frame, a later graph may reuse an earlier graph's transient objects;
    // that assumes graphs are submitted in execute order on one queue.
    void execute(WGPUDevice device, WGPUCommandEncoder encoder, WGPUQueue queue, bool enableProfiling = false);
    // Kick async GPU timestamp readback. Call after queue submit. Results arrive via the instance
    // event pump a few frames later. No-op if profiling was off.
    void collect_gpu_timings();

    // check for errors after compile(), execute becomes nulop when not nullptr
    ErrorMessage* get_errors();

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

    // Internal
    PassBuilder begin_pass(ResourceId id, PassKind kind);
    void end_pass(PassBuilder& builder);
};

// Create the `GraphAllocator`
GraphAllocator* create_allocator();

// Destroys a `GraphAllocator`
void destroy_allocator(GraphAllocator* allocator);

// Frame boundary
void begin_frame(GraphAllocator* allocator);

// Begin recording a render graph
RenderGraph* start_recording(GraphAllocator* allocator);

// Frame boundary
void end_frame(GraphAllocator* allocator);

} // namespace webgpu

// exposes the operator overload outside of the namespace
using webgpu::operator""_rid;
