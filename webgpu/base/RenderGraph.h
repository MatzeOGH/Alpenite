

// immediate-mode render graph for WebGPU. declare resources and passes per frame, then compile() and
// execute(). pass bodies call wgpu* directly, the graph only does pass ordering and resource lifetime.
// full documentation: docs/rendergraph.md.

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

// picks the encoder the pass records into, and which accesses are legal
enum struct PassKind : uint8_t {
    None = 0,
    // PassContext::render_pass, with the declared attachments already bound. the body must not begin a
    // render pass of its own.
    Graphics,
    // PassContext::compute_pass
    Compute,
    // no pass object, the body encodes copies on PassContext::encoder
    Transfer
};

enum struct SizeKind : uint8_t {
    Absolute,
    Relative // scaled relative to relativeTo
};

enum struct ResourceKind : uint8_t {
    Texture,
    Buffer,
};

// names a resource for this frame. cheap to copy, never carries the GPU object itself, so pass it
// between systems and capture it by value into pass lambdas. valid for one frame, generation catches
// stale handles.
struct ResourceHandle {
    uint32_t id {};
    ResourceKind kind {};
    uint64_t generation {};
};

// ping-pong pair, rotated each frame. write curr, read prev. this frame's curr is next frame's prev.
struct HistoryResource {
    ResourceHandle curr; // writes
    // reads. gate every read on PassContext::history_valid, prev holds nothing on the frame it was
    // (re)created.
    ResourceHandle prev;
};

namespace Internal {
struct ResourceNode;
struct PassNode;
}
struct GraphAllocator;
struct RenderGraph;

struct ErrorMessage {
    WGPUStringView message;
    ErrorMessage* next {};
};

// handed to the pass body during execute(), where handles turn into live GPU objects.
//
// call the resolvers from inside the body and let nothing they return escape it. the object behind a
// transient handle is picked at execute() and changes frame to frame, so a view or bind group built
// ahead of time, or cached for the next pass, points at another resource's memory. build bind groups
// here, from bind() entries. see "the two rules" in docs/rendergraph.md.
struct PassContext {
    WGPUCommandEncoder encoder {}; // borrowed
    WGPURenderPassEncoder render_pass {}; // Graphics passes, else null
    WGPUComputePassEncoder compute_pass {}; // Compute passes, else null
    WGPUQueue queue {}; // borrowed
    WGPUDevice device {}; // borrowed
    RenderGraph* graph {};
    Internal::PassNode* pass {};

    // shape inferred from this pass's access declaration, ViewRange included
    WGPUTextureView view(ResourceHandle h) const;
    WGPUTextureView view(ResourceHandle h, uint32_t mip, uint32_t layer = 0) const;

    // buffers bind whole, at offset 0
    WGPUBindGroupEntry bind(uint32_t binding, ResourceHandle h) const;
    WGPUBindGroupEntry bind(uint32_t binding, ResourceHandle h, uint32_t mip, uint32_t layer = 0) const;

    // all assert the handle has a declared access in this pass
    WGPUTexture texture(ResourceHandle h) const;
    WGPUBuffer buffer(ResourceHandle h) const;
    WGPUExtent3D texture_size(ResourceHandle h) const;
    WGPUExtent3D texture_size(ResourceHandle h, uint32_t mip) const;
    uint64_t buffer_size(ResourceHandle h) const;
    WGPUTextureFormat format(ResourceHandle h) const;
    uint32_t mip_count(ResourceHandle h) const;
    uint32_t sample_count(ResourceHandle h) const;

    // false on the frame .prev was (re)created and cleared to 0
    bool history_valid(ResourceHandle h) const;

    // resolvers for a copy declared in this pass, declared mip/layer applied. each needs one unambiguous
    // copy per handle and direction.
    WGPUTexelCopyTextureInfo copy_src_info(ResourceHandle h, WGPUOrigin3D origin = {}, WGPUTextureAspect aspect = WGPUTextureAspect_All) const;
    WGPUTexelCopyTextureInfo copy_dst_info(ResourceHandle h, WGPUOrigin3D origin = {}, WGPUTextureAspect aspect = WGPUTextureAspect_All) const;
    // the declared subresource's width and height at its mip level
    WGPUExtent3D copy_extent_src(ResourceHandle h) const;
    WGPUExtent3D copy_extent_dst(ResourceHandle h) const;
    // buffer side of a texture<->buffer copy. layout is the caller's data layout.
    WGPUTexelCopyBufferInfo copy_src_buffer(ResourceHandle h, WGPUTexelCopyBufferLayout layout) const;
    WGPUTexelCopyBufferInfo copy_dst_buffer(ResourceHandle h, WGPUTexelCopyBufferLayout layout) const;

    // size is already expanded, never 0
    struct BufferCopyInfo {
        WGPUBuffer src {};
        uint64_t srcOffset {};
        WGPUBuffer dst {};
        uint64_t dstOffset {};
        uint64_t size {};
    };
    BufferCopyInfo buffer_copy_info(ResourceHandle src, ResourceHandle dst) const;
};

// view shape for a sampled/storage access. attachments ignore it, a render target is always one mip and
// one layer. the default {} is one mip, one layer, all aspects, dimension inferred.
struct ViewRange {
    // Undefined infers 3D for a 3D texture, else 2DArray when the view covers more than one layer, else
    // 2D
    WGPUTextureViewDimension dim = WGPUTextureViewDimension_Undefined;
    uint32_t mipCount = 1; // counted from baseMip, not an index. 0 = all remaining.
    // counted from baseLayer, not an index. 0 = all remaining. layers come from a 2D texture's
    // absolute.depthOrArrayLayers.
    uint32_t layerCount = 1;
    WGPUTextureAspect aspect = WGPUTextureAspect_All; // picks one plane of a depth-stencil texture
};

// a cube needs exactly 6 layers, a cube array 6*cubes. sampling only, WebGPU storage textures cannot be
// cube. mipCount 0 means all mips from baseMip.
constexpr ViewRange cube(WGPUTextureAspect aspect = WGPUTextureAspect_All, uint32_t mipCount = 1)
{
    return ViewRange { WGPUTextureViewDimension_Cube, mipCount, 6, aspect };
}
constexpr ViewRange cube_array(uint32_t cubes, WGPUTextureAspect aspect = WGPUTextureAspect_All, uint32_t mipCount = 1)
{
    return ViewRange { WGPUTextureViewDimension_CubeArray, mipCount, 6 * cubes, aspect };
}

// all mips and layers from baseMip/baseLayer, dimension inferred
constexpr ViewRange whole(WGPUTextureAspect aspect = WGPUTextureAspect_All)
{
    return ViewRange { WGPUTextureViewDimension_Undefined, 0, 0, aspect };
}

// the 256-byte row alignment WebGPU requires for buffer<->texture copies
inline uint32_t aligned_bytes_per_row(uint32_t widthTexels, uint32_t texelBlockBytes)
{
    constexpr uint32_t align = 256u;
    uint32_t raw = widthTexels * texelBlockBytes;
    return (raw + (align - 1u)) & ~(align - 1u);
}

// handed to a pass's setup lambda, which runs immediately. one call per resource use.
//
// these declarations are the whole contract with the graph. it derives pass ordering from them (a
// sampled() runs after whatever wrote that handle), infers the WGPUTextureUsage/WGPUBufferUsage to
// create the resource with, and decides what memory a transient may share. no GPU work happens here.
//
// an access the body performs but never declares is invisible to all three: it gets no ordering edge,
// and the aliaser is free to hand that memory to another transient. declare every read and write.
struct PassBuilder {
    // attachmentIndex is the fragment shader @location. slots may be sparse, the same slot twice is an
    // error. a render target is always one mip and one layer, hence baseMip/baseLayer, not a range.
    void color(ResourceHandle handle,
        uint32_t attachmentIndex,
        WGPULoadOp load = WGPULoadOp_Clear,
        WGPUStoreOp store = WGPUStoreOp_Store,
        WGPUColor clear = { 0, 0, 0, 1 },
        uint32_t baseMip = 0,
        uint32_t baseLayer = 0);
    void depth_stencil(ResourceHandle handle,
        WGPULoadOp load = WGPULoadOp_Clear,
        WGPUStoreOp store = WGPUStoreOp_Store,
        float clearDepth = 1.0f,
        uint32_t baseMip = 0,
        uint32_t baseLayer = 0,
        WGPULoadOp stencilLoad = WGPULoadOp_Undefined,
        WGPUStoreOp stencilStore = WGPUStoreOp_Undefined,
        uint32_t stencilClear = 0);
    // test only, no write. WebGPU requires Undefined load/store.
    void depth_stencil_read_only(ResourceHandle handle, uint32_t baseMip = 0, uint32_t baseLayer = 0);
    // MSAA resolve of src, a color() declared earlier in this pass, into a single-sample target. src is
    // matched by handle, not by position, but its color() must already be declared. one resolve target
    // per color slot.
    void resolve(ResourceHandle src, ResourceHandle target, uint32_t baseMip = 0, uint32_t baseLayer = 0);
    void sampled(ResourceHandle handle, uint32_t baseMip = 0, uint32_t baseLayer = 0, ViewRange range = {});
    void storage_read(ResourceHandle handle, uint32_t baseMip = 0, uint32_t baseLayer = 0, ViewRange range = {});
    void storage_write(ResourceHandle handle, uint32_t baseMip = 0, uint32_t baseLayer = 0, ViewRange range = {});
    // in-dispatch races are the shader's responsibility
    void storage_read_write(ResourceHandle handle, uint32_t baseMip = 0, uint32_t baseLayer = 0, ViewRange range = {});
    void uniform(ResourceHandle handle);
    // a wgpuQueueWriteBuffer in this pass body
    void host_write(ResourceHandle handle);
    void copy_texture(ResourceHandle src, ResourceHandle dst, uint32_t srcMip = 0, uint32_t srcLayer = 0, uint32_t dstMip = 0, uint32_t dstLayer = 0);
    // dst is a transient buffer, or an imported CopyDst|MapRead buffer for CPU readback. the graph never
    // creates mappable buffers.
    void copy_texture_to_buffer(ResourceHandle src, ResourceHandle dst, uint32_t srcMip = 0, uint32_t srcLayer = 0);
    void copy_buffer_to_texture(ResourceHandle src, ResourceHandle dst, uint32_t dstMip = 0, uint32_t dstLayer = 0);
    // size 0 means the whole src buffer from srcOffset
    void copy_buffer(ResourceHandle src, ResourceHandle dst, uint64_t srcOffset = 0, uint64_t dstOffset = 0, uint64_t size = 0);
    void vertex_buffer(ResourceHandle handle);
    void index_buffer(ResourceHandle handle);
    void indirect_buffer(ResourceHandle handle);

    // bake pass for a pool-backed resource, a persistent one or a history curr. runs only when the target
    // is stale: first frame, recreation, or hash mismatch. hash 0 bakes once. still declare the write
    // itself (color(), storage_write(), a copy), this only gates whether the pass runs. on a skipped
    // frame the pass is culled, and readers bind the pooled result regardless.
    void initialize(ResourceHandle target, uint64_t hash = 0);

    // exempt this pass from dead-pass culling, for side-effect-only passes
    void force_keep();

    Internal::PassNode* m_pass;
    RenderGraph* m_graph {};
};

// usage flags are inferred from how passes declare the resource, so there is no usage field here
struct TextureDesc {
    WGPUTextureDimension dimension = WGPUTextureDimension_Undefined;
    WGPUTextureFormat format = WGPUTextureFormat_Undefined;
    SizeKind sizeKind = SizeKind::Absolute;
    // Relative only: size is this factor times relativeTo's, so a half-res target tracks its source
    // through resizes
    float scaleX = 1.0f;
    float scaleY = 1.0f;
    ResourceHandle relativeTo {}; // required for Relative. a cycle through relativeTo is a compile error.
    WGPUExtent3D absolute = WGPU_EXTENT_3D_INIT; // Absolute only
    uint32_t mipLevelCount = 1;
    uint32_t sampleCount = 1; // 1 or 4 (WebGPU 1.0 limit)
};

struct BufferDesc {
    uint64_t size {}; // bytes
};

// frame-scoped. obtain via start_recording(), invalid after the next begin_frame().
//
// every id is hashed and copied into the graph's arena during the call, so the string_view only has to
// be alive for that call. resource ids must be unique within a frame.
struct RenderGraph {
    // lives for this frame, drawn from a pool and possibly aliased with another transient. the default
    // choice, the others below exist for cross-frame memory.
    ResourceHandle create_transient_texture(std::string_view id, const TextureDesc& desc);

    ResourceHandle create_transient_buffer(std::string_view id, const BufferDesc& desc);

    // wraps an object the graph does not own, the swapchain being the usual one. caller owns lifetime and
    // guarantees it outlives the frame. writing an import is an output that leaves the frame, so those
    // passes are never culled.
    //
    // a null texture registers the view only, which rules out ctx.texture() and the copy family on this
    // handle. dimension is the source texture's own dimension, so an array or cube source must say 2D.
    ResourceHandle import_texture(std::string_view id, WGPUTextureView view, WGPUExtent3D size, WGPUTextureFormat format,
        WGPUTexture texture = nullptr, uint32_t mipCount = 1, uint32_t sampleCount = 1,
        WGPUTextureDimension dimension = WGPUTextureDimension_2D);

    // caller owns lifetime
    ResourceHandle import_buffer(std::string_view id, WGPUBuffer buffer);

    // cross-frame ping-pong for temporal effects. write .curr, read .prev, and see HistoryResource.
    // hash != 0 enables hash-based invalidation: history resets on the frame the hash changes, which is
    // how a camera cut drops reprojection.
    HistoryResource create_history_texture(std::string_view id, const TextureDesc& desc, uint64_t hash = 0);

    HistoryResource create_history_buffer(std::string_view id, const BufferDesc& desc, uint64_t hash = 0);

    // one graph-owned object kept across frames, for a cache or a LUT. content that is baked rather than
    // rewritten every frame wants PassBuilder::initialize() on the baking pass.
    ResourceHandle create_persistent_buffer(std::string_view id, const BufferDesc& desc);

    // initialized once from data, or zero-filled. the fallback for an optional binding.
    ResourceHandle create_initialized_buffer(std::string_view id, const BufferDesc& desc, const void* data = nullptr);

    ResourceHandle create_persistent_texture(std::string_view id, const TextureDesc& desc);

    // cleared once to fill. the fallback for an optional sampled slot.
    ResourceHandle create_initialized_texture(std::string_view id, const TextureDesc& desc, WGPUColor fill);

    // setup(PassBuilder&) runs now and declares this pass's accesses. executeFn(PassContext&) runs later,
    // inside execute(), and records the GPU commands.
    //
    // executeFn's closure lives in an arena that frees without running destructors, so it must be
    // trivially destructible: capture handles, raw WGPU objects, plain values and this, all by value.
    // a std::string, container, smart pointer or RAII wrapper fails the static_assert below.
    //
    // id must be non-empty. it labels the pass in the debug panel, GPU timings and error messages, and
    // is not checked for uniqueness.
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

    // called once, after every create_*, import_* and add_pass. the graph is frozen afterwards.
    //
    // the frame runs in this order:
    //   begin_frame() -> start_recording() -> declare resources and passes -> compile() -> get_errors()
    //   -> execute() -> submit the encoder yourself -> collect_gpu_timings() -> end_frame()
    //
    // compile() also culls: a pass survives only if it feeds a read by another surviving pass, an
    // imported resource, a history curr, or is marked force_keep(). a pass writing only a transient
    // nobody reads is dropped silently and leaves the panel and timings.
    //
    // authoring errors land in get_errors() rather than throwing. enableAlias packs disjoint-lifetime
    // transients onto shared objects, pass false to rule aliasing out while debugging.
    void compile(bool enableAlias = true);

    // records the surviving passes into encoder. does not submit, the caller owns that.
    //
    // device, queue and encoder are borrowed, never released. enableProfiling needs TimestampQuery.
    // multiple graphs per frame must be submitted in execute order on one queue.
    void execute(WGPUDevice device, WGPUQueue queue, WGPUCommandEncoder encoder, bool enableProfiling = false);
    // after queue submit. no-op when profiling was off. results arrive a few frames later, via the
    // instance event pump.
    void collect_gpu_timings();

    // the only signal that the frame is broken, so check it after every compile().
    //
    // an authoring error (a read of a transient nothing writes, a cycle, an out-of-range subresource)
    // poisons the graph instead of throwing, and execute() turns into a no-op. that shows up as a black
    // scene with everything outside the graph, the UI included, still running.
    //
    // non-null means the graph failed. the chain is freed by the next begin_frame().
    ErrorMessage* get_errors() const;

private:
    template <class F>
    void store_exec(PassBuilder& b, F&& f)
    {
        using D = std::decay_t<F>;
        static_assert(std::is_trivially_destructible_v<D>,
            "execute callback must be trivially destructible; the arena frees without running destructors, so capture handles or ids by value");
        void* m = alloc_exec(sizeof(D), alignof(D)); // never null, arena OOM aborts
        ::new (m) D(std::forward<F>(f));
        set_exec(b, m, [](void* o, PassContext& c) { (*static_cast<D*>(o))(c); });
    }
    void* alloc_exec(size_t size, size_t align);
    void set_exec(PassBuilder& builder, void* obj, void (*fn)(void*, PassContext&));


    PassBuilder begin_pass(std::string_view id, PassKind kind);
    void end_pass(PassBuilder& builder);
};

// backs all graphs, arenas and the GPU object pool. one per device.
GraphAllocator* create_allocator();

// destroys everything it owns, pooled GPU objects included
void destroy_allocator(GraphAllocator* allocator);

// all graphs and handles from the previous frame die here
void begin_frame(GraphAllocator* allocator);

// several graphs per frame are allowed. see execute() for ordering rules.
RenderGraph* start_recording(GraphAllocator* allocator);

// ages the pools so unused GPU objects get evicted. must run after the queue submit, the pooled objects
// are still referenced by the commands being submitted.
void end_frame(GraphAllocator* allocator);

} // namespace webgpu::rg
