
// Internal node and allocator layout for the render graph. Not part of the public API,
// may change without notice.

#pragma once

#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <utility>
#include <vector>
#include <QtAssert>
#include <QDebug>
#include <webgpu/webgpu.h>

#include "RenderGraph.h"

namespace webgpu::rg {

// FNV-1a hash of a name, the basis of ResourceId::value.
inline constexpr uint64_t fnv1a(std::string_view s)
{
    uint64_t hash = 14695981039346656037ULL;
    for (char c : s) {
        hash ^= static_cast<uint8_t>(c);
        hash *= 1099511628211ULL;
    }
    return hash;
}

// Name-based resource identity: the hash plus the name itself, so lookups stay cheap and
// a hash collision never conflates two names.
struct ResourceId {
    uint64_t value {};
    WGPUStringView name {};
};

// Build a borrowing ResourceId from a name. The name is not copied. An id that outlives the
// caller's string must intern the name into the arena (see intern_id in RenderGraph.cpp).
constexpr ResourceId make_resource_id(std::string_view s) {
    return ResourceId(fnv1a(s), WGPUStringView { .data = s.data(), .length = s.length() });
}

// Equality on hash and name content.
constexpr bool operator==(ResourceId a, ResourceId b)
{
    return a.value == b.value && a.name.length == b.name.length &&
        std::string_view(a.name.data, a.name.length) == std::string_view(b.name.data, b.name.length);
}

} // namespace webgpu::rg

namespace webgpu::rg::Internal {

// TransientAttachments feature id from the gpuweb transient-attachments proposal, a Dawn-native
// feature 0x00050006. The native headers name it, but the emdawnwebgpu web headers ship the usage
// bit without the feature enum, so we refer to it by value to compile on both. A browser never
// exposes it, so a web build always reports the feature off.
static constexpr WGPUFeatureName kTransientAttachmentsFeature = (WGPUFeatureName)0x00050006u;

// WebGPU spec limit on color attachments per render pass
static constexpr uint32_t kMaxColorAttachments = 8;

// How a pass touches a resource. Classifies read vs write for hazard edges and picks the
// required usage bits (columns: category, direction, usage).
enum struct AccessType : uint8_t {
    ColorAttachment, // attachment      write   tex RenderAttachment
    DepthStencilAttachment, // attachment      write   tex RenderAttachment  (depth/stencil test + write)
    DepthStencilReadOnly, // attachment-read read    tex RenderAttachment  (test only, depthReadOnly, no write hazard)
    ResolveAttachment, // attachment      write   tex RenderAttachment  (single-sample target an MSAA color resolves into)
    Sampled, // constant        read    tex TextureBinding
    StorageRead, // storage-read    read    tex StorageBinding / buf Storage
    StorageWrite, // storage         write   tex StorageBinding / buf Storage
    Uniform, // constant        read    buf Uniform (+CopyDst host-upload affordance)
    CopySrc, // copy            read    tex/buf CopySrc
    CopyDst, // copy            write   tex/buf CopyDst
    Vertex, // input           read    buf Vertex
    Index, // input           read    buf Index
    Indirect, // input           read    buf Indirect
};

// One recorded access on a pass. Fields are ordered wide-to-narrow to kill padding (~96B).
struct ResourceAccess {
    ResourceHandle handle {}; // 16B, 8-aligned -> first

    // byte range of a copy_buffer() on this buffer (offset of this side, shared size). 0/0 = whole
    // buffer. Texture and non-copy accesses leave both 0.
    uint64_t bufOffset {};
    uint64_t bufSize {};

    // attachment clear color. f32 (16B) not WGPUColor (32B doubles): f32 is lossless for every WebGPU
    // color format (none exceeds f32 component precision), so no ceiling. Reconstructed to WGPUColor at
    // the attachment-build site in execute().
    float clearColor[4] {};
    float clearDepth {};

    // attachment-only (ColorAttachment / DepthStencilAttachment)
    WGPULoadOp loadOp {};
    WGPUStoreOp storeOp {};
    // stencil aspect of a DepthStencilAttachment: set for a depth+stencil format
    WGPULoadOp stencilLoadOp {};
    WGPUStoreOp stencilStoreOp {};

    // view shape for a sampled/storage read, from PassBuilder::ViewRange
    WGPUTextureViewDimension viewDim { WGPUTextureViewDimension_Undefined }; // Undefined -> infer
    WGPUTextureAspect viewAspect { WGPUTextureAspect_All };

    // subresource this access touches + view range. Narrowed from u32, all within WebGPU limits.
    uint16_t baseLayer {};
    uint16_t layerCount { 1 };
    AccessType type {};
    uint8_t stencilClear {}; // stencil is always 8-bit in WebGPU, exact
    // buffer CopyDst only: this copy provably overwrites the whole buffer (copy_buffer with dstOffset
    // 0 and whole size). A copy_texture_to_buffer dst stays false: its coverage is only known at
    // encode time, so it must never enable aliasing's first-define takeover (see access_defines).
    bool bufFullDefine {};
    // NOTE(Huerbe): baseMip/mipCount are u8 (mip count bounded, ~14 for a 16K texture). mipCount==0 = all remaining.
    uint8_t baseMip {};
    uint8_t mipCount { 1 };
};

// Length of a string view, resolving the WGPU_STRLEN sentinel.
size_t sv_length(WGPUStringView s);

// Content equality of two string views.
bool sv_eq(WGPUStringView a, WGPUStringView b);

// Group label of a pass name: the span before the first '.', empty if none.
WGPUStringView group_prefix(WGPUStringView name);

// Per-physical-texture cache of subresource views, so per-mip and per-layer views persist
// instead of being recreated every pass.
struct SubviewCache {
    struct Slot {
        WGPUTextureViewDescriptor d; // cache key
        WGPUTextureView view;
    };
    std::vector<Slot> e; // distinct subviews of one physical texture (a mip chain is ~12), grows on demand

    // descriptor equality, the cache-key match
    static bool same(const WGPUTextureViewDescriptor& a, const WGPUTextureViewDescriptor& b);
    // cached view for this descriptor, created on first request
    WGPUTextureView get_or_create(WGPUTexture tex, const WGPUTextureViewDescriptor& d);
    // release every cached view
    void release_all();
};

// Owns GPU resources that must outlive the per-frame graph teardown: history textures and
// buffers such as accumulation, history feedback, and GPU-authored particle state. One Entry per
// logical history resource, keyed by name content. Each Entry holds two physical textures the
// graph ping-pongs: layer 0, current, maps to the opposite slot each frame, so last frame's current
// is this frame's previous for free.
struct PersistentResourcePool {
    static constexpr uint32_t kLayers = 2; // ping-pong: current + previous. N>2 deliberately unsupported.
    static constexpr uint64_t kRetain = 4; // free an entry no pass has touched for this many frames

    struct Entry {
        uint64_t idValue = 0; // to ResourceId::value
        std::string name; // identity across frames
        uint64_t frame = 0; // rotation counter
        uint64_t lastTouched = 0; // pool evictClock at the last touch
        uint64_t prevTouched = 0; // evictClock at the touch before this frame's
        uint32_t layers = kLayers; // physical instances: kLayers = history, 1 = single in-place
        ResourceKind kind = ResourceKind::Texture; // buffer or texture
        uint64_t initHash = 0; // initialize(): settings hash of the content baked in (recorded at execute)
        bool baked = false; // initialize(): true once an init pass baked content in, cleared on (re)create
        uint64_t historyHash = 0; // history invalidation: mismatch -> destroy+recreate (zeros .prev)

        WGPUTexture tex[kLayers] = {};
        WGPUTextureView view[kLayers] = {};
        SubviewCache sub[kLayers] = {};

        // texture
        bool created = false;
        uint64_t createdClock = UINT64_MAX; // evictClock at last (re)create, catches same-frame recreate
        WGPUExtent3D size = {};
        WGPUTextureFormat format = WGPUTextureFormat_Undefined;
        WGPUTextureDimension dim = WGPUTextureDimension_Undefined;
        uint32_t mipLevelCount = 1;
        uint32_t sampleCount = 1;
        WGPUTextureUsage usage = {}; // union of every layer's usage
        WGPUTextureUsage usageAtCreate = {};

        // buffer
        WGPUBuffer buf[kLayers] = {};
        uint64_t bufferSize = 0;
        WGPUBufferUsage bufUsage = {}; // running union across layers/frames
        WGPUBufferUsage bufUsageAtCreate = {};
    };
    std::vector<Entry> entries;
    uint64_t evictClock = 0;

    // entry with this id, null if absent
    Entry* find(ResourceId id);

    // realize-time, once per used entry per frame: ensure the entry exists and advance its rotation, so
    // rotation tracks use, not declaration. Idempotent within a frame: a history's two layer nodes share
    // one entry but only the first rotates (lastTouched == evictClock means "already touched this realize",
    // and evictClock advances once per frame, so a carried-over entry has lastTouched < evictClock and rotates).
    Entry* touch(ResourceId id, uint32_t layers, uint64_t hash, ResourceKind kind);

    // physical slot backing a layer under the entry's current rotation
    uint32_t slot(const Entry& e, uint32_t layerIndex) const;

    // true when the requested descriptor no longer matches the entry's textures, forcing a recreate
    static bool tex_descriptor_differs(
        const Entry& e, WGPUExtent3D size, WGPUTextureFormat format, WGPUTextureDimension dim, uint32_t mipLevelCount, uint32_t sampleCount);

    // (re)create the entry's `layers` textures when missing or the descriptor changed. Lazy: it needs
    // the device and the usage union, both known only at realize().
    void realize_texture_entry(
        Entry* e, WGPUDevice device, WGPUExtent3D size, WGPUTextureFormat format, WGPUTextureDimension dim, uint32_t mipLevelCount, uint32_t sampleCount);

    // (re)create the entry's `layers` buffers when missing or size/usage changed: 1 for a single
    // in-place buffer, kLayers for a ping-pong. usage is the union accumulated into e->bufUsage, since
    // each physical buffer cycles through every layer role across frames.
    void realize_buffer_entry(Entry* e, WGPUDevice device, uint64_t size);

    // free the entry's GPU objects
    void destroy(Entry* e);

    // per-frame teardown: free entries no pass has touched for kRetain frames, then advance the clock.
    // A history resource stops being touched when its feature goes inactive, so without this the pool
    // would hold its physical textures and buffers for the process lifetime. Call once per realized frame.
    void end_frame();

    ~PersistentResourcePool();
};

// Caches per-frame transient GPU objects across teardown so realize()/release_resources() stop
// churning the driver: one physical object per Entry, handed back next frame instead of recreated
// and destroyed. Descriptor-keyed: each Entry is a texture, matched by size/format/dim/mip/sample/usage,
// or a buffer, matched by size and usage, tagged by `kind`. Usage matches by superset (an object whose
// usage covers the request is reused), except restriction bits that must match exactly (see acquire).
// Objects left unclaimed for a few frames are evicted. Textures and buffers share the inUse claim,
// idle eviction, vector, and teardown. Only the match and create differ, in the two acquire overloads.
struct TransientResourcePool {
    static constexpr uint64_t kRetain = 4; // destroy an object left unclaimed this many frames

    struct Entry {
        ResourceKind kind = ResourceKind::Texture; // Texture = texture arm below, Buffer = buffer arm
        // texture arm
        WGPUExtent3D size = {};
        WGPUTextureFormat format = WGPUTextureFormat_Undefined;
        WGPUTextureDimension dim = WGPUTextureDimension_2D;
        uint32_t mipLevelCount = 1;
        uint32_t sampleCount = 1;
        WGPUTextureUsage usage = {};
        WGPUTexture tex = {};
        WGPUTextureView view = {};
        SubviewCache sub = {}; // subresource views of this texture, freed in destroy()
        // buffer arm
        uint64_t bufferSize = 0;
        WGPUBufferUsage bufUsage = {};
        WGPUBuffer buf = {};
        // shared
        bool inUse = false; // claimed right now, released in release_claims
        uint64_t lastUsedFrame = 0; // recency, eviction only
        uint64_t createdFrame = 0; // frame this object was created in, the supersede-eviction key (see superseded_by)
    };
    std::vector<Entry> entries;
    uint64_t frame = 0;
    uint32_t createdThisFrame = 0;

    // debug event log
    enum class Event : uint8_t { Create, Evict };
    struct LogRec {
        uint64_t frame;
        Event event;
        ResourceKind kind;
        WGPUExtent3D size; // texture
        WGPUTextureFormat format; // texture
        uint64_t bufferSize; // buffer
    };
    static constexpr uint32_t kLog = 128;
    LogRec eventLog[kLog] = {};
    uint64_t eventCount = 0;

    // append a create/evict record to the debug log
    void log_event(Event event, const Entry& e);

    // clear the debug log
    void log_reset();

    // hand out a free texture matching the descriptor, or create one
    void acquire(WGPUDevice device,
        WGPUExtent3D size,
        WGPUTextureFormat format,
        WGPUTextureDimension dim,
        uint32_t mipLevelCount,
        uint32_t sampleCount,
        WGPUTextureUsage usage,
        WGPUTexture& outTex,
        WGPUTextureView& outView);

    // hand out a free buffer matching size and usage, or create one
    void acquire(WGPUDevice device, uint64_t size, WGPUBufferUsage usage, WGPUBuffer& outBuf);

    // drop every inUse claim so the frame's objects become reusable
    void release_claims();

    // supersede-eviction predicate: true iff idle texture entry `e` is made redundant this frame by a
    // sibling `s` created this frame with an identical descriptor except the extent. Kills the resize
    // VRAM spike (old-size transients otherwise linger kRetain frames). Texture-only, exact usage match.
    // See the definition for the full rationale. Static so it is device-free unit-testable.
    static bool superseded_by(const Entry& e, const Entry& s, uint64_t frame);

    // destroy objects idle >= kRetain frames, advance the clock. Once per frame.
    void end_frame();

    // free the entry's GPU objects
    void destroy(Entry* e);

    ~TransientResourcePool();
};

// Off-by-default opt-in: also time Transfer/copy passes. They have no pass object, so their only
// timing path is the encoder-level wgpuCommandEncoderWriteTimestamp, which is off-spec and Dawn
// rejects unless the device was created with the "allow_unsafe_apis" toggle. Render and compute
// timing work regardless.
// #define RG_TIME_TRANSFER_PASSES

// Opt-in per-pass GPU timing. Two timestamp queries per executed pass, begin and end, resolved into a
// staging buffer and copied into a ring of mappable read-back buffers so a frame never stalls waiting
// on the GPU. Lives in GraphAllocator, a cross-frame survivor, so the query set and buffers outlive
// the per-frame arena reset. Only created when profiling is first enabled and the device has the
// TimestampQuery feature.
struct GpuProfiler {
    static constexpr uint32_t kMaxPasses = 64; // -> 2*64 timestamps in the set
    static constexpr uint32_t kRing = 3; // covers GPU read-back latency without stalling
    static constexpr uint32_t kMaxName = 48; // truncating owned copy of a pass name (dotted labels are short)

    // owned, NUL-terminated copy of a pass name. Pass names live in the per-frame arena and the
    // read-back lands frames later, after the arena rewound and the next graph overwrote those bytes,
    // so every name the profiler keeps must be its own copy, never a pointer into the graph.
    static void copy_name(char* dst, WGPUStringView s);

    WGPUQuerySet querySet {}; // type Timestamp, count 2*kMaxPasses
    WGPUBuffer resolveBuf {}; // 2*kMaxPasses*8, QueryResolve | CopySrc

    struct Slot {
        WGPUBuffer buf {}; // 2*kMaxPasses*8, MapRead | CopyDst
        bool pending {}; // mapAsync in flight -> not reusable until the cb fires
        uint32_t count {}; // passes captured into this fill
        char names[kMaxPasses][kMaxName] {}; // pass names, copied at capture (see copy_name)
    } ring[kRing];
    int pendingSlot { -1 }; // slot execute() filled this frame, awaiting the map kick

    // last completed read-back, read by the debug UI
    uint32_t resultCount {};
    char resultNames[kMaxPasses][kMaxName] {}; // own copies: the ring slot is rewritten on reuse
    float resultUs[kMaxPasses] {};

    bool initialized {};

    // --- timing history (recording), driven by the "Timings" tab ---
    static constexpr uint32_t kHistory = 256; // frames retained in the ring
    bool recording {};
    uint32_t resultId {}; // bumped by the read-back cb on each new sample, dedupes capture
    struct Series {
        char name[kMaxName] {};
        bool enabled { true };
        float v[kHistory] {};
    };
    Series series[kMaxPasses];
    uint32_t seriesCount {};
    uint32_t historyHead {}; // next write column
    uint32_t historyLen {}; // valid columns (<= kHistory)
    uint32_t lastSampledId {}; // resultId already appended

    // wipe the recorded history and all series
    void clear_history();

    // append one column iff recording and a fresh read-back arrived since last call. Match each result
    // to a series by name content and occurrence index, so repeated pass names keep distinct rows.
    // New names get a new series with older columns left 0. A missing series gets 0 this column.
    // NOTE(Huerbe): series keyed by name. Switching graphs adds new series and zero-fills old ones. Clear resets.
    void sample_history();

    // one-time creation of the query set, resolve buffer, and read-back ring
    void init(WGPUDevice device);

    // first ring slot not awaiting a map, -1 if all are in flight (skip this frame)
    int free_slot() const;
};

static constexpr size_t ARENA_DEFAULT_BLOCK_SIZE = 16 * 1024;
static constexpr size_t ARENA_SCRATCH_DEFAULT_BLOCK_SIZE = 2 * 1024;

// Chained-block bump allocator behind all graph storage. Frees nothing until reset() or
// free_all(), so per-frame nodes cost a pointer bump.
struct Arena {
    // Header + its payload share one malloc. `payload` is the usable region, rounded up to
    // alignof(max_align_t), the most any single allocation can ask for. this+1 is not that-aligned,
    // since the header is 12 bytes on wasm32 and 24 on x64, so the aligned start is computed once and stored.
    struct ArenaBlock {
        ArenaBlock* next; // chain link
        uint8_t* payload; // aligned payload start, inside this same malloc
        size_t capacity; // usable payload bytes
        size_t used; // bump cursor into payload
    };

    ArenaBlock* head {}; // first block, reset() rewinds to it and free_all() walks from it
    ArenaBlock* current {}; // block we bump from now (null until the first alloc)
    size_t blockSize = ARENA_DEFAULT_BLOCK_SIZE; // default payload size for new blocks

    // Debug-UI stats: peakUsage is the per-frame high-water (reset() zeros it), live_used() is the
    // current occupancy. totalCapacity/blockCount persist across reset().
    size_t totalCapacity {};
    size_t peakUsage {};
    size_t blockCount {};
    size_t liveBytes {}; // running occupancy, kept in sync by alloc_raw/extend_tail/reset/rewind so live_used() is O(1)

    // alignment must be a power of two
    static constexpr size_t align_up(size_t value, size_t alignment) { return (value + alignment - 1) & ~(alignment - 1); }

    // Malloc one block whose payload holds >= minPayload bytes (at least blockSize). Header, alignment
    // slack and payload come from a single malloc. Null on heap exhaustion.
    ArenaBlock* new_block(size_t minPayload);

    // ensure the first block exists ahead of the first alloc
    void reserve();

    // current occupancy in bytes, O(1)
    size_t live_used() const;

    // bump-allocate size bytes at the given alignment
    void* alloc_raw(size_t size, size_t align);

    // Grow allocation p in place when it is the tail of the current block. Null when it cannot.
    void* extend_tail(void* p, size_t oldSize, size_t addSize);

    // allocation-failure path, fatal
    void* oom(size_t size);

    // rewind every block for the next frame, keeping the memory
    void reset();

    // free the whole chain (destructor / destroy_allocator).
    void free_all();

    // Scoped scratch discipline (see ScopedScratch): mark a stack position, rewind back to it. Blocks
    // appended past the mark are kept (used=0) for reuse, never freed.
    struct Mark {
        ArenaBlock* block;
        size_t used;
    };
    Mark mark() const;
    void rewind(Mark m);

    // placement-construct T at m, null-safe
    template <typename T, typename... Args>
    static T* construct(void* m, Args&&... args)
    {
        return m ? ::new (m) T(std::forward<Args>(args)...) : nullptr;
    }
    // zero-fill m as count Ts, null-safe
    template <typename T>
    static T* zero(void* m, size_t count)
    {
        if (m)
            std::memset(m, 0, sizeof(T) * count);
        return static_cast<T*>(m);
    }

    // allocate and construct one T
    template <typename T, typename... Args>
    T* make(Args&&... args)
    {
        return construct<T>(alloc_raw(sizeof(T), alignof(T)), std::forward<Args>(args)...);
    }

    // Allocate zeroed POD storage (calloc replacement). Contiguous within one block.
    template <typename T>
    T* alloc(size_t count = 1)
    {
        return zero<T>(alloc_raw(sizeof(T) * count, alignof(T)), count);
    }

    // Copy a string view into arena-owned storage. Result is always null-terminated.
    WGPUStringView copy_string(WGPUStringView s);
};


} // namespace webgpu::rg::Internal

namespace webgpu::rg {

// The cross-frame survivor behind every graph: arenas for per-frame nodes plus the pools and
// profiler that outlive each frame's teardown. It is the public opaque handle (rg::GraphAllocator),
// so it stays in namespace rg while its members live in rg::Internal.
struct GraphAllocator {
    Internal::Arena front; // permanent per-frame nodes
    Internal::Arena scratch; // compile()-local temporaries, driven by ScopedScratch

    Internal::PersistentResourcePool pool; // name-keyed history textures + buffers
    Internal::TransientResourcePool transient; // descriptor-keyed per-frame texture + buffer cache (tagged by kind)
    Internal::GpuProfiler profiler; // opt-in per-pass GPU timestamps, created lazily on the first profiled frame

    // whether the device advertises WGPUFeatureName_TransientAttachments
    bool transientFeatureOn {};

    static constexpr size_t align_up(size_t value, size_t alignment) { return Internal::Arena::align_up(value, alignment); }

    // front-arena forwarders for per-frame allocations
    void* alloc_raw(size_t size, size_t align);
    template <typename T, typename... Args>
    T* make(Args&&... args)
    {
        return front.make<T>(std::forward<Args>(args)...);
    }
    template <typename T>
    T* alloc(size_t count = 1)
    {
        return front.alloc<T>(count);
    }
    // null-terminated front-arena copy
    WGPUStringView copy_string(WGPUStringView s);

    // per-frame teardown: rewind both arenas
    void reset();
};

} // namespace webgpu::rg

namespace webgpu::rg::Internal {

// RAII scratch scope: marks the arena on construction, rewinds on destruction.
struct ScopedScratch {
    Arena* arena;
    Arena::Mark mark;
    explicit ScopedScratch(Arena& a);
    ~ScopedScratch();
    ScopedScratch(const ScopedScratch&) = delete;
    ScopedScratch& operator=(const ScopedScratch&) = delete;

    // rewind to the mark early, before destruction
    void reset();

    template <typename T>
    T* alloc(size_t count = 1)
    {
        return arena->alloc<T>(count);
    }
    template <typename T, typename... Args>
    T* make(Args&&... args)
    {
        return arena->make<T>(std::forward<Args>(args)...);
    }
};

// Internal node for a texture or buffer resource.
struct ResourceNode {
    ResourceHandle handle {};
    ResourceId id {};
    ResourceKind kind {};

    // Clustered boolean flags, packed into one byte. None have their address taken (all read/written by
    // value in compile phases), so bitfields are safe. Usage sites:
    //   imported          - managed outside the graph, e.g. a swapchain texture
    //   persistent        - one layer of a PersistentResourcePool-backed resource (survives teardown)
    //   history           - ping-pong history: .curr written this frame, read next (persistent variant)
    //   historyValid      - .prev holds valid prev-frame history, false if the entry was recreated this frame
    //   resolving         - resolve_size(): on the relativeTo recursion stack -> re-entry is a cycle
    //   hasWriter         - phase-4 aliasing eligibility: some pass writes this (else bytes are host-owned)
    //   firstDefines      - phase-4 aliasing eligibility: first touch fully overwrites the previous occupant
    //   transientAttachment - detect_transient_attachments: graph-owned attachment that never leaves the GPU
    bool imported : 1 {};
    bool persistent : 1 {};
    bool history : 1 {};
    bool historyValid : 1 {};
    bool resolving : 1 {};
    bool hasWriter : 1 {};
    bool firstDefines : 1 {};
    bool transientAttachment : 1 {};

    uint32_t historyIndex {}; // 0 == current frame (write target), 1 == previous frame (read)
    uint64_t historyHash {}; // history invalidation hash

    // storage not owned by the per-frame graph (imported or pool-backed) -> realize()/release_resources()
    // skip it and lifetime aliasing excludes it.
    bool is_external() const;

    // texture fields
    WGPUTextureDimension dimension = WGPUTextureDimension_Undefined;
    WGPUTextureFormat format = WGPUTextureFormat_Undefined;

    SizeKind sizeKind = SizeKind::Absolute;
    float scaleX = 1.0f;
    float scaleY = 1.0f;
    ResourceHandle relativeToHandle {};
    WGPUExtent3D absolute = WGPU_EXTENT_3D_INIT;
    uint32_t mipLevelCount = 1; // > 1 = mip chain, created once with per-mip views built at bind/attach time
    uint32_t sampleCount = 1; // > 1 = MSAA (multisampled attachment)

    // buffer fields
    uint64_t bufferSize {};

    // realized GPU handles
    WGPUTexture texture {}; // created: the texture object backing `view`
    WGPUTextureView view {}; // imported: the registered swapchain view
    // subresource-view cache of the pooled physical texture backing this node (persistent/history/plain
    // transient). Set in realize() via find_subcache, null for imported (caller owns the view). ctx.view()
    // routes non-full views through it so they persist across frames instead of per-pass churn.
    SubviewCache* subviews {};
    WGPUBuffer buffer {}; // imported: the registered buffer
    WGPUExtent3D resolved = WGPU_EXTENT_3D_INIT; // imported: registered size, the base for Relative resolution
    WGPUTextureUsage texUsage {}; // accumulated in compile() from the access list
    WGPUBufferUsage bufUsage {}; // accumulated in compile() from the access list

    // first/last surviving pass (execution-order index) to touch this, filled in compile() phase 3.
    // kNoPass = no live pass touched it (dead transient) or imported. Imported is left out: the graph
    // doesn't own its memory, so an aliasing lifetime would be meaningless.
    static constexpr uint32_t kNoPass = ~0u;
    uint32_t firstUse = kNoPass;
    uint32_t lastUse = kNoPass;

    // aliasing (compile() phase 4, only when enableAlias): the physical slot this transient shares with
    // other disjoint-lifetime resources. kNoSlot means its own object, ineligible or aliasing off.
    // hasWriter and firstDefines are the eligibility inputs captured during the phase-3 access walk: a
    // transient may take over another's storage only if some pass writes it, else its bytes are
    // host-owned, and its first touch fully overwrites, else it would read the previous occupant's leftovers.
    static constexpr uint32_t kNoSlot = ~0u;
    uint32_t aliasSlot = kNoSlot;

    // live-access bookkeeping, filled during the phase-3 access walk so detect_transient_attachments is
    // O(R) instead of re-scanning all passes per resource. liveAccessCount is the number of surviving
    // accesses touching this node, and soleAccess is the last one recorded (the only one when count == 1).
    uint32_t liveAccessCount {};
    const ResourceAccess* soleAccess {};
    // NOTE: transientAttachment (realize() adds WGPUTextureUsage_TransientAttachment when supported) is in
    // the clustered flags near the top.

    ResourceNode* next {};
};

struct NodeAdjacency;

// Internal pass node: an intrusive linked-list node with a growable access array, so the
// builder records accesses with few allocations.
struct PassNode {
    ResourceId id {};
    PassKind kind {};

    // dense declaration-order index 0..P-1 assigned early in compile() (over all passes, incl. skipInit).
    // Backbone for O(1) hazard-edge dedup: sweep_resource_versions stamps linkedBy[dep->index].
    uint32_t index {};

    // type-erased pass body, invoked by execute()
    void* exec_obj {};
    void (*exec_fn)(void*, PassContext&) {};

    // growable access list
    ResourceAccess* accesses {};
    uint32_t accessCount {};
    uint32_t accessCap {};

    // inline adjacency list
    NodeAdjacency* adjacency {};

    bool placed {}; // topo sort: already emitted into execution order
    bool onStack {}; // topo sort: on the current DFS recursion stack -> a re-entry is a back-edge (cycle)
    bool sink {}; // cull root: writes an imported or history resource, so it survives with no reader
    bool forceKeep {}; // force_keep(): extra cull root. Survives even with no reader and no imported/persistent write

    // initialize(): if non-empty, this pass (re)bakes the persistent/history targets it lists, one
    // initialize() call each. compile() sets skipInit, dropping the pass this frame, only when every
    // target's pool entry is populated and was baked with its matching hash. A fresh or evicted target,
    // or a changed hash on any one, re-arms the whole pass, since a body bakes all its targets or none.
    // An empty list is an ordinary pass.
    static constexpr uint32_t kMaxInitTargets = 8;
    struct InitTarget {
        ResourceHandle target;
        uint64_t hash;
    };
    // Out-of-line: almost every pass has initCount==0, so the sidecar is arena-allocated lazily on the
    // first initialize() call, like accesses[]. Null until then, and all readers are initCount-bounded.
    InitTarget* initTargets {};
    uint32_t initCount {};
    bool skipInit {};

    PassNode* next {};
};

// One dependency edge: `pass` must execute before the owner of this list.
struct NodeAdjacency {
    PassNode* pass {};
    NodeAdjacency* next {};
};

// one physical GPU object shared by one or more transients with disjoint lifetimes and an identical
// signature (aliasing, compile() phase 4). Carries the texture arm or the buffer arm per `kind`. A
// slot never mixes the two, since phase 4 buckets by kind. `freeFrom` and the alias bookkeeping are
// shared across both arms.
struct PhysicalResource {
    ResourceKind kind {}; // picks which arm below is live
    uint32_t freeFrom {}; // occupant lastUse, reusable iff the next member's firstUse > this
    // Texture
    WGPUTextureDimension dimension {}; // signature: members must match exactly to share this slot
    WGPUTextureFormat format {};
    WGPUExtent3D size {};
    WGPUTextureUsage texUsage {}; // union over members (WebGPU needs every member's bits at create)
    WGPUTexture texture {}; // filled by realize() via transient.acquire()
    WGPUTextureView view {};
    // Buffer
    uint64_t bufferSize {}; // signature
    WGPUBufferUsage bufUsage {}; // union over members
    WGPUBuffer buffer {}; // filled by realize() via transient.acquire() (buffer overload)
};

// Per-frame lifecycle of a graph.
enum struct RenderGraphState {
    Recording = 0, // declaring passes/resources (mutable)
    Compiled, // compile() succeeded: read-only, ready to realize/execute
    Failed, // compile() hit an authoring error (see getErrors()): realize/execute no-op
    Finished // execute() ran, graph is spent for this frame
};

// Concrete graph state behind the opaque RenderGraph handle, co-allocated in the same block.
struct RenderGraphStorage {
    GraphAllocator* m_allocator {};
    ResourceNode* m_resouces {}; // linked list of resources declared for the graph
    ResourceNode* m_resoucesTail {};
    PassNode* m_passes {}; // linked list of passes declared for the graph
    PassNode* m_passesTail {};
    RenderGraphState m_state {}; // Recording until compile(), read-only after
    uint32_t next_resource_id = 1; // 0 = invalid handle, so next_resource_id - 1 is the resource count
    uint32_t passCount {}; // total passes (incl. skipInit), assigned in compile() W1, sizes the dedup stamp
    uint64_t generation {}; // graph-instance id and generation
    ResourceNode** byId {}; // fast lookup of resources by id
    PhysicalResource* m_slots {};
    uint32_t m_slotCount {};
    // transient-attachment inference: count of graph-owned attachments flagged memoryless
    uint32_t transientCount {};
    // execute() scratch: subresource views built for the current pass (attachments + the body's ctx.view()),
    // released after the body. Reset per pass, one view per access, so the per-pass access ceiling bounds it.
    WGPUTextureView* viewScratch {};
    uint32_t viewScratchN {};
    uint32_t viewScratchCap {}; // grows on demand in resolve_view, monotonic across passes
    ErrorMessage* m_errors {}; // linked list of error messages, non-null => m_state is Failed
    ErrorMessage* m_errorsTail {};

    // CPU timing infos
    float timing_compile_us {};
    float timing_realize_us {};
    float timing_execute_us {};
};

// Returns the internal storage of the render graph.
RenderGraphStorage* storage(RenderGraph* rg);

// Resolve a handle to its node.
ResourceNode* find_node(RenderGraph* rg, ResourceHandle h);

// True for the write half of AccessType (hazard classification).
bool access_is_write(AccessType t);

// Name of the pass at execution-order index idx, empty past the end. Used by the debug lifetime widget.
WGPUStringView pass_name_at(PassNode* head, uint32_t idx);

// rough bytes-per-texel for the formats this project creates. Used only by the ImGui memory widgets
// for peak-allocation accounting, never for allocation. No block-compressed formats here, so a linear
// width*height*layers*texel estimate is exact enough, and unknown formats fall back to 4.
uint32_t texel_bytes(WGPUTextureFormat f);

// bytes one texture of this size+format would occupy, base mip only since aliasing members are single-mip.
uint64_t texture_bytes(WGPUExtent3D size, WGPUTextureFormat format);

} // namespace webgpu::rg::Internal
