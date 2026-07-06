#include "RenderGraph.h"
#include <QtAssert>
#include <QDebug>
#include <cassert>
#include <chrono>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>
#include <webgpu/webgpu.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
// On WebGPU, TRANSIENT_ATTACHMENT is exposed as a usage flag, so detect support by checking whether the constant exists.
EM_JS(int, rg_web_has_transient_attachment, (), { return (typeof GPUTextureUsage != = 'undefined' && ('TRANSIENT_ATTACHMENT' in GPUTextureUsage)) ? 1 : 0; });
#endif

#include "RenderGraph_internal.h"

namespace RG {

// The internal node/pool/helper layout now lives in RG::Internal (RenderGraph_internal.h). Pull it in
// unqualified so the file-static helpers and RenderGraph:: methods below keep resolving it as before.
using namespace Internal;


// Out-of-line definitions for the declarations that RenderGraph_internal.h no longer inlines. Kept in
// one block so the header stays a pure layout/declaration surface shared with the debug tooling.
namespace Internal {

size_t sv_length(WGPUStringView s) { return (s.length == WGPU_STRLEN) ? (s.data ? std::strlen(s.data) : 0) : s.length; }

bool sv_eq(WGPUStringView a, WGPUStringView b)
{
    size_t na = sv_length(a), nb = sv_length(b);
    return na == nb && (na == 0 || std::memcmp(a.data, b.data, na) == 0);
}

WGPUStringView group_prefix(WGPUStringView name)
{
    size_t n = sv_length(name);
    for (size_t i = 0; i < n; ++i)
        if (name.data[i] == '.')
            return WGPUStringView { .data = name.data, .length = i };
    return WGPUStringView {};
}

bool SubviewCache::same(const WGPUTextureViewDescriptor& a, const WGPUTextureViewDescriptor& b)
{
    return a.format == b.format && a.dimension == b.dimension && a.aspect == b.aspect && a.baseMipLevel == b.baseMipLevel
        && a.mipLevelCount == b.mipLevelCount && a.baseArrayLayer == b.baseArrayLayer && a.arrayLayerCount == b.arrayLayerCount;
}

WGPUTextureView SubviewCache::get_or_create(WGPUTexture tex, const WGPUTextureViewDescriptor& d)
{
    for (const Slot& s : e)
        if (same(s.d, d))
            return s.view;
    WGPUTextureView v = wgpuTextureCreateView(tex, &d);
    e.push_back({ d, v });
    return v;
}

void SubviewCache::release_all()
{
    for (Slot& s : e)
        if (s.view)
            wgpuTextureViewRelease(s.view);
    e.clear();
}

PersistentResourcePool::Entry* PersistentResourcePool::find(ResourceId id)
{
    for (Entry& e : entries)
        if (e.idValue == id.value && sv_eq(WGPUStringView { e.name.c_str(), e.name.size() }, id.name))
            return &e;
    return nullptr;
}

PersistentResourcePool::Entry* PersistentResourcePool::touch(ResourceId id, uint32_t layers, uint64_t hash, bool isBuffer)
{
    if (Entry* e = find(id)) {
        if (e->lastTouched == evictClock)
            return e; // sibling layer node already touched this frame -> no re-rotate
        if (e->isBuffer != isBuffer || e->layers != layers)
            destroy(e); // a carried-over entry whose kind or layer count no longer matches the request would leave
        e->prevTouched = e->lastTouched; // capture when it was last used, before overwriting -> honest historyValid
        ++e->frame;
        e->lastTouched = evictClock;
        e->layers = layers;
        e->isBuffer = isBuffer;
        if (hash && e->historyHash != hash) {
            destroy(e);
            e->historyHash = hash;
        }
        return e;
    }
    entries.emplace_back();
    Entry& e = entries.back();
    e.name.assign(id.name.data ? id.name.data : "", sv_length(id.name));
    e.idValue = id.value;
    e.lastTouched = evictClock;
    e.prevTouched = evictClock;
    e.layers = layers;
    e.isBuffer = isBuffer;
    e.historyHash = hash;
    return &e;
}

uint32_t PersistentResourcePool::slot(const Entry& e, uint32_t layerIndex) const { return (uint32_t)((e.frame + layerIndex) % e.layers); }

bool PersistentResourcePool::tex_descriptor_differs(
    const Entry& e, WGPUExtent3D size, WGPUTextureFormat format, WGPUTextureDimension dim, uint32_t mipLevelCount, uint32_t sampleCount)
{
    return e.size.width != size.width || e.size.height != size.height || e.size.depthOrArrayLayers != size.depthOrArrayLayers || e.format != format
        || e.dim != dim || e.mipLevelCount != mipLevelCount || e.sampleCount != sampleCount;
}

void PersistentResourcePool::realize_entry(
    Entry* e, WGPUDevice device, WGPUExtent3D size, WGPUTextureFormat format, WGPUTextureDimension dim, uint32_t mipLevelCount, uint32_t sampleCount)
{
    bool same = e->created && !tex_descriptor_differs(*e, size, format, dim, mipLevelCount, sampleCount) && e->usageAtCreate == e->usage;
    if (same)
        return;
    // a recreate after another layer node already realized this frame would dangle that node's
    // cached tex/view. Layer nodes of one entry must agree on the descriptor; they come from one declaration.
    Q_ASSERT(e->createdClock != evictClock && "history layer nodes disagree on texture descriptor");
    destroy(e);
    e->size = size;
    e->format = format;
    e->dim = dim;
    e->mipLevelCount = mipLevelCount;
    e->sampleCount = sampleCount;
    e->usageAtCreate = e->usage;
    for (uint32_t i = 0; i < e->layers; ++i) {
        WGPUTextureDescriptor d {
            .label = WGPUStringView { e->name.c_str(), e->name.size() },
            .usage = e->usage,
            .dimension = dim,
            .size = size,
            .format = format,
            .mipLevelCount = mipLevelCount,
            .sampleCount = sampleCount,
        };
        e->tex[i] = wgpuDeviceCreateTexture(device, &d);
        e->view[i] = wgpuTextureCreateView(e->tex[i], nullptr);
    }
    e->created = true;
    e->createdClock = evictClock;
}

void PersistentResourcePool::realize_buffer_entry(Entry* e, WGPUDevice device, uint64_t size)
{
    bool same = e->created && e->bufferSize == size && e->bufUsageAtCreate == e->bufUsage;
    if (same)
        return;
    Q_ASSERT(e->createdClock != evictClock && "history layer nodes disagree on buffer descriptor");
    destroy(e);
    e->bufferSize = size;
    e->bufUsageAtCreate = e->bufUsage;
    for (uint32_t i = 0; i < e->layers; ++i) {
        WGPUBufferDescriptor d {
            .label = WGPUStringView { e->name.c_str(), e->name.size() },
            .usage = e->bufUsage,
            .size = size,
        };
        e->buf[i] = wgpuDeviceCreateBuffer(device, &d);
    }
    e->created = true;
    e->createdClock = evictClock;
}

void PersistentResourcePool::destroy(Entry* e)
{
    for (uint32_t i = 0; i < kLayers; ++i) {
        e->sub[i].release_all();
        if (e->view[i]) {
            wgpuTextureViewRelease(e->view[i]);
            e->view[i] = nullptr;
        }
        if (e->tex[i]) {
            wgpuTextureRelease(e->tex[i]);
            e->tex[i] = nullptr;
        }
        if (e->buf[i]) {
            wgpuBufferRelease(e->buf[i]);
            e->buf[i] = nullptr;
        }
    }
    e->created = false;
    e->baked = false;
}

void PersistentResourcePool::end_frame()
{
    for (size_t i = entries.size(); i-- > 0;)
        if (evictClock - entries[i].lastTouched >= kRetain) { // lastTouched <= evictClock -> no underflow
            destroy(&entries[i]);
            if (i + 1 != entries.size())
                entries[i] = std::move(entries.back());
            entries.pop_back();
        }
    ++evictClock;
}

PersistentResourcePool::~PersistentResourcePool()
{
    for (Entry& e : entries)
        destroy(&e);
}

void TransientResourcePool::log_event(Event kind, const Entry& e) { eventLog[eventCount++ % kLog] = { frame, kind, e.size, e.format }; }

void TransientResourcePool::log_reset() { eventCount = 0; }

void TransientResourcePool::acquire(WGPUDevice device,
    WGPUExtent3D size,
    WGPUTextureFormat format,
    WGPUTextureDimension dim,
    uint32_t mipLevelCount,
    uint32_t sampleCount,
    WGPUTextureUsage usage,
    WGPUTexture& outTex,
    WGPUTextureView& outView)
{
    for (Entry& e : entries) {
        if (!e.isBuffer && !e.inUse && e.format == format && e.dim == dim && e.usage == usage && e.mipLevelCount == mipLevelCount
            && e.sampleCount == sampleCount && e.size.width == size.width && e.size.height == size.height
            && e.size.depthOrArrayLayers == size.depthOrArrayLayers) {
            e.inUse = true;
            e.lastUsedFrame = frame;
            outTex = e.tex;
            outView = e.view;
            return;
        }
    }
    entries.emplace_back();
    Entry& e = entries.back();
    e.isBuffer = false;
    e.size = size;
    e.format = format;
    e.dim = dim;
    e.mipLevelCount = mipLevelCount;
    e.sampleCount = sampleCount;
    e.usage = usage;
    WGPUTextureDescriptor d {
        .usage = usage,
        .dimension = dim,
        .size = size,
        .format = format,
        .mipLevelCount = mipLevelCount,
        .sampleCount = sampleCount,
    };
    e.tex = wgpuDeviceCreateTexture(device, &d);
    e.view = wgpuTextureCreateView(e.tex, nullptr);
    e.inUse = true;
    e.lastUsedFrame = frame;
    ++createdThisFrame;
    log_event(Event::Create, e);
    outTex = e.tex;
    outView = e.view;
}

void TransientResourcePool::acquire(WGPUDevice device, uint64_t size, WGPUBufferUsage usage, WGPUBuffer& outBuf)
{
    for (Entry& e : entries) {
        if (e.isBuffer && !e.inUse && e.bufferSize == size && e.bufUsage == usage) {
            e.inUse = true;
            e.lastUsedFrame = frame;
            outBuf = e.buf;
            return;
        }
    }
    entries.emplace_back();
    Entry& e = entries.back();
    e.isBuffer = true;
    e.bufferSize = size;
    e.bufUsage = usage;
    WGPUBufferDescriptor d { .usage = usage, .size = size };
    e.buf = wgpuDeviceCreateBuffer(device, &d);
    e.inUse = true;
    e.lastUsedFrame = frame;
    ++createdThisFrame;
    outBuf = e.buf;
}

void TransientResourcePool::release_claims()
{
    for (Entry& e : entries)
        e.inUse = false;
}

void TransientResourcePool::end_frame()
{
    for (size_t i = entries.size(); i-- > 0;) {
        Entry& e = entries[i];
        if (frame - e.lastUsedFrame >= kRetain) { // lastUsedFrame <= frame always -> no underflow
            if (!e.isBuffer)
                log_event(Event::Evict, e);
            destroy(&e);
            entries[i] = entries.back();
            entries.pop_back();
        }
    }
    createdThisFrame = 0;
    ++frame;
}

void TransientResourcePool::destroy(Entry* e)
{
    e->sub.release_all();
    if (e->view) {
        wgpuTextureViewRelease(e->view);
        e->view = nullptr;
    }
    if (e->tex) {
        wgpuTextureRelease(e->tex);
        e->tex = nullptr;
    }
    if (e->buf) {
        wgpuBufferRelease(e->buf);
        e->buf = nullptr;
    }
}

TransientResourcePool::~TransientResourcePool()
{
    for (Entry& e : entries)
        destroy(&e);
}

void GpuProfiler::copy_name(char* dst, WGPUStringView s)
{
    size_t n = sv_length(s);
    if (n >= kMaxName)
        n = kMaxName - 1;
    std::memcpy(dst, s.data, n);
    dst[n] = '\0';
}

void GpuProfiler::clear_history()
{
    seriesCount = 0;
    historyHead = 0;
    historyLen = 0;
    lastSampledId = 0;
}

void GpuProfiler::sample_history()
{
    if (!recording || resultId == lastSampledId)
        return;
    lastSampledId = resultId;
    for (uint32_t s = 0; s < seriesCount; ++s)
        series[s].v[historyHead] = 0.0f;
    for (uint32_t i = 0; i < resultCount; ++i) {
        uint32_t occ = 0; // which occurrence of this name result i is, among this read-back's results
        for (uint32_t j = 0; j < i; ++j)
            if (std::strcmp(resultNames[j], resultNames[i]) == 0)
                ++occ;
        uint32_t s = 0;
        for (uint32_t seen = 0; s < seriesCount; ++s)
            if (std::strcmp(series[s].name, resultNames[i]) == 0 && seen++ == occ)
                break;
        if (s == seriesCount) {
            if (seriesCount >= kMaxPasses)
                continue;
            series[s] = {};
            std::memcpy(series[s].name, resultNames[i], kMaxName);
            ++seriesCount;
        }
        series[s].v[historyHead] = resultUs[i];
    }
    historyHead = (historyHead + 1) % kHistory;
    if (historyLen < kHistory)
        ++historyLen;
}

void GpuProfiler::init(WGPUDevice device)
{
    if (initialized)
        return;
    const uint64_t bytes = uint64_t(2) * kMaxPasses * sizeof(uint64_t);
    WGPUQuerySetDescriptor qd { .type = WGPUQueryType_Timestamp, .count = 2 * kMaxPasses };
    querySet = wgpuDeviceCreateQuerySet(device, &qd);
    WGPUBufferDescriptor rd { .usage = WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopySrc, .size = bytes };
    resolveBuf = wgpuDeviceCreateBuffer(device, &rd);
    for (Slot& s : ring) {
        WGPUBufferDescriptor bd { .usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst, .size = bytes };
        s.buf = wgpuDeviceCreateBuffer(device, &bd);
    }
    initialized = true;
}

int GpuProfiler::free_slot() const
{
    for (uint32_t i = 0; i < kRing; ++i)
        if (!ring[i].pending)
            return (int)i;
    return -1;
}

Arena::ArenaBlock* Arena::new_block(size_t minPayload)
{
    const size_t maxAlign = alignof(std::max_align_t);
    const size_t payloadBytes = minPayload < blockSize ? blockSize : minPayload;
    const size_t raw = sizeof(ArenaBlock) + maxAlign + payloadBytes; // +maxAlign covers the round-up
    ArenaBlock* b = static_cast<ArenaBlock*>(std::malloc(raw));
    if (!b)
        return nullptr;
    uint8_t* start = reinterpret_cast<uint8_t*>(b) + sizeof(ArenaBlock);
    b->payload = reinterpret_cast<uint8_t*>(align_up(reinterpret_cast<size_t>(start), maxAlign));
    b->capacity = payloadBytes;
    b->used = 0;
    b->next = nullptr;
    totalCapacity += payloadBytes;
    ++blockCount;
    return b;
}

void Arena::reserve()
{
    if (!current)
        head = current = new_block(blockSize);
}

size_t Arena::live_used() const
{
    size_t total = 0;
    for (ArenaBlock* b = head; b; b = b->next) {
        total += b->used;
        if (b == current)
            break;
    }
    return total;
}

void* Arena::alloc_raw(size_t size, size_t align)
{
    if (!current) // lazy first block
    {
        head = current = new_block(size + align);
        if (!current)
            return oom(size);
    }

    size_t off = align_up(current->used, align);
    if (off + size > current->capacity)
    {
        if (current->next && size + align <= current->next->capacity) {
            current = current->next;
            current->used = 0;
        } else {
            ArenaBlock* b = new_block(size + align);
            if (!b)
                return oom(size);
            b->next = current->next;
            current->next = b;
            current = b;
        }
        off = align_up(current->used, align);
    }

    void* p = current->payload + off;
    current->used = off + size;
    const size_t live = live_used();
    if (live > peakUsage)
        peakUsage = live;
    return p;
}

void* Arena::extend_tail(void* p, size_t oldSize, size_t addSize)
{
    if (!current || !p)
        return nullptr;
    if (reinterpret_cast<uint8_t*>(p) + oldSize != current->payload + current->used)
        return nullptr;
    if (current->used + addSize > current->capacity)
        return nullptr;
    current->used += addSize;
    const size_t live = live_used();
    if (live > peakUsage)
        peakUsage = live;
    return p;
}

void* Arena::oom(size_t size)
{
    qFatal("[RenderGraph] error: arena alloc failed; malloc of %zu bytes failed, heap exhausted.\n", size);
    Q_ASSERT(false && "Arena malloc failed");
    return nullptr;
}

void Arena::reset()
{
    for (ArenaBlock* b = head; b; b = b->next)
        b->used = 0;
    current = head;
    peakUsage = 0; // per-frame peak
}

void Arena::free_all()
{
    for (ArenaBlock* b = head; b;) {
        ArenaBlock* n = b->next;
        std::free(b);
        b = n;
    }
    head = current = nullptr;
    totalCapacity = peakUsage = blockCount = 0;
}

Arena::Mark Arena::mark() const { return Mark { current, current ? current->used : 0 }; }

void Arena::rewind(Mark m)
{
    for (ArenaBlock* b = (m.block ? m.block->next : head); b; b = b->next)
        b->used = 0;
    if (m.block) {
        m.block->used = m.used;
        current = m.block;
    } else {
        current = head;
    }
}

WGPUStringView Arena::copy_string(WGPUStringView s)
{
    const size_t len = (s.length == WGPU_STRLEN) ? (s.data ? std::strlen(s.data) : 0) : s.length;
    char* buf = alloc<char>(len + 1);
    if (!buf)
        return {};
    if (len)
        std::memcpy(buf, s.data, len);
    buf[len] = '\0';
    return WGPUStringView { buf, len };
}

ScopedScratch::ScopedScratch(Arena& a)
    : arena(&a)
    , mark(a.mark())
{
}

ScopedScratch::~ScopedScratch() { arena->rewind(mark); }

void ScopedScratch::reset() { arena->rewind(mark); }

bool ResourceNode::is_external() const { return imported || persistent; }

RenderGraphStorage* storage(RenderGraph* rg)
{
    return reinterpret_cast<RenderGraphStorage*>(reinterpret_cast<uint8_t*>(rg) + GraphAllocator::align_up(sizeof(RenderGraph), alignof(RenderGraphStorage)));
}

ResourceNode* find_node(RenderGraph* rg, ResourceHandle h)
{
    RenderGraphStorage& s = *storage(rg);

    Q_ASSERT(!(h.id && h.generation && h.generation != s.generation) && "stale/foreign ResourceHandle passed to a ctx resolver");

    if (s.byId && h.id && h.id < s.next_resource_id)
        return s.byId[h.id]; // fast path: after compiled graph
    for (ResourceNode* r = s.m_resouces; r; r = r->next) // pre-compile or id 0: walk
        if (r->handle.id == h.id)
            return r;
    return nullptr;
}

bool access_is_write(AccessType t)
{
    return t == AccessType::ColorAttachment || t == AccessType::DepthStencilAttachment || t == AccessType::ResolveAttachment || t == AccessType::StorageWrite
        || t == AccessType::CopyDst;
}

WGPUStringView pass_name_at(PassNode* head, uint32_t idx)
{
    for (PassNode* p = head; p; p = p->next) {
        if (idx == 0)
            return p->id.name;
        --idx;
    }
    return WGPUStringView {};
}

uint32_t texel_bytes(WGPUTextureFormat f)
{
    switch (f) {
    case WGPUTextureFormat_R8Unorm:
    case WGPUTextureFormat_R8Uint:
    case WGPUTextureFormat_Stencil8:
        return 1;
    case WGPUTextureFormat_R16Float:
    case WGPUTextureFormat_RG8Unorm:
    case WGPUTextureFormat_Depth16Unorm:
        return 2;
    case WGPUTextureFormat_RGBA8Unorm:
    case WGPUTextureFormat_BGRA8Unorm:
    case WGPUTextureFormat_RG16Float:
    case WGPUTextureFormat_R32Float:
    case WGPUTextureFormat_Depth32Float:
    case WGPUTextureFormat_Depth24Plus:
        return 4;
    case WGPUTextureFormat_RGBA16Float:
    case WGPUTextureFormat_RG32Float:
        return 8;
    case WGPUTextureFormat_RGBA32Float:
        return 16;
    default:
        return 4;
    }
}

uint64_t texture_bytes(WGPUExtent3D size, WGPUTextureFormat format)
{
    uint32_t layers = size.depthOrArrayLayers ? size.depthOrArrayLayers : 1;
    return (uint64_t)size.width * size.height * layers * texel_bytes(format);
}

} // namespace Internal

// GraphAllocator lives in RG (public opaque handle); its member bodies moved out of the header too.
void* GraphAllocator::alloc_raw(size_t size, size_t align) { return front.alloc_raw(size, align); }

WGPUStringView GraphAllocator::copy_string(WGPUStringView s) { return front.copy_string(s); }

void GraphAllocator::reset()
{
    front.reset();
    scratch.reset();
}


template <typename T>
static void list_append(T** head, T** tail, T* newNode)
{
    if (*tail)
        (*tail)->next = newNode;
    else
        *head = newNode;
    *tail = newNode;
}

// append an error message and switch the graph to its failed state
static void push_error(RenderGraphStorage& s, const char* fmt, ...)
{
    ScopedScratch scratch(s.m_allocator->scratch);

    va_list ap;
    va_start(ap, fmt);
    va_list probe;
    va_copy(probe, ap);
    const int len = std::vsnprintf(nullptr, 0, fmt, probe);
    va_end(probe);

    char* buf = scratch.alloc<char>(size_t(len) + 1);
    std::vsnprintf(buf, size_t(len) + 1, fmt, ap);
    va_end(ap);

    if (ErrorMessage* e = s.m_allocator->make<ErrorMessage>()) {
        e->message = s.m_allocator->copy_string(WGPUStringView { buf, size_t(len) });
        list_append(&s.m_errors, &s.m_errorsTail, e);
    }
    s.m_state = RenderGraphState::Failed;
}


static ResourceHandle create_handle(RenderGraphStorage& s, ResourceKind kind)
{
    ResourceHandle h {
        .id = s.next_resource_id++,
        .kind = kind,
        .generation = s.generation,
    };
    return h;
}

// duplicate-declaration check shared by every create_*/import_*..
static void assert_unique_id(RenderGraphStorage& s, ResourceId id)
{
#ifndef NDEBUG
    for (ResourceNode* r = s.m_resouces; r; r = r->next)
        Q_ASSERT(!(r->id == id) && "ResourceIds must to be unique");
#else
    (void)s;
    (void)id;
#endif
}

GraphAllocator* create_allocator()
{
    GraphAllocator* allocator = new GraphAllocator;
    allocator->front.blockSize = ARENA_DEFAULT_BLOCK_SIZE;
    allocator->scratch.blockSize = ARENA_SCRATCH_DEFAULT_BLOCK_SIZE;
    allocator->front.reserve();
    allocator->scratch.reserve();
    return allocator;
}

void destroy_allocator(GraphAllocator* allocator)
{
    allocator->front.free_all();
    allocator->scratch.free_all();
    delete allocator;
}

void begin_frame(GraphAllocator* allocator) { allocator->reset(); }

void end_frame(GraphAllocator* allocator)
{
    allocator->pool.end_frame();
    allocator->transient.end_frame();
}

RenderGraph* start_recording(GraphAllocator* allocator)
{
    struct RGPair {
        RenderGraph rg;
        RenderGraphStorage st;
    };
    RGPair* pair = allocator->make<RGPair>();
    RenderGraph* rg = &pair->rg;
    RenderGraphStorage* st = &pair->st;
    Q_ASSERT(st == storage(rg) && "storage must sit immediately after the RenderGraph");
    st->m_allocator = allocator;

    static uint64_t g_next_graph_generation = 1;
    st->generation = g_next_graph_generation++;
    return rg;
}


static bool access_defines(const ResourceAccess& a)
{
    if (a.type == AccessType::ColorAttachment || a.type == AccessType::DepthStencilAttachment)
        return a.loadOp == WGPULoadOp_Clear;
    return a.type == AccessType::StorageWrite || a.type == AccessType::CopyDst;
}

static WGPUTextureUsage tex_usage_of(AccessType t)
{
    switch (t) {
    case AccessType::ColorAttachment:
    case AccessType::DepthStencilAttachment:
    case AccessType::DepthStencilReadOnly:
    case AccessType::ResolveAttachment:
        return WGPUTextureUsage_RenderAttachment;
    case AccessType::Sampled:
        return WGPUTextureUsage_TextureBinding;
    case AccessType::StorageRead:
    case AccessType::StorageWrite:
        return WGPUTextureUsage_StorageBinding;
    case AccessType::CopySrc:
        return WGPUTextureUsage_CopySrc;
    case AccessType::CopyDst:
        return WGPUTextureUsage_CopyDst;
    default:
        return WGPUTextureUsage_None; // buffer-only access
    }
}
static WGPUBufferUsage buf_usage_of(AccessType t)
{
    switch (t) {
    case AccessType::StorageRead:
    case AccessType::StorageWrite:
        return WGPUBufferUsage_Storage;
    case AccessType::Uniform:
        return WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    case AccessType::CopySrc:
        return WGPUBufferUsage_CopySrc;
    case AccessType::CopyDst:
        return WGPUBufferUsage_CopyDst;
    case AccessType::Vertex:
        return WGPUBufferUsage_Vertex;
    case AccessType::Index:
        return WGPUBufferUsage_Index;
    case AccessType::Indirect:
        return WGPUBufferUsage_Indirect;
    default:
        return WGPUBufferUsage_None; // texture-only access
    }
}

// half-open subresource ranges [base, base+count); count 0 means "all remaining" -> open-ended
static constexpr bool ranges_overlap(uint32_t aBase, uint32_t aCount, uint32_t bBase, uint32_t bCount)
{
    uint32_t aEnd = aCount ? aBase + aCount : UINT32_MAX;
    uint32_t bEnd = bCount ? bBase + bCount : UINT32_MAX;
    return aBase < bEnd && bBase < aEnd;
}

// All covers depth+stencil, so it overlaps either aspect, otherwise two accesses clash only on the same one.
static constexpr bool aspects_overlap(WGPUTextureAspect a, WGPUTextureAspect b) { return a == WGPUTextureAspect_All || b == WGPUTextureAspect_All || a == b; }

// whether two accesses to the same resource in one pass (one usage scope) conflict. read+read never does.
// disjoint subresources never do either: WebGPU usage scopes are per-(mip,layer,aspect), so sampling mip i
// while rendering into mip j!=i is legal (a mip-chain downsample/upsample pass). this compares the full
// (mip range x layer range x aspect) each access declares, so a wide sampled range overlapping a written mip
// in the same pass is caught here, not left to Dawn. the lone overlapping read+write exception is
// StorageRead+StorageWrite: that is how the graph spells a read-modify-write storage binding
// (var<storage, read_write>). One writable-storage usage, not an alias (the "multi-writer chain" test +
// the sweep's WAR self-guard depend on it). Any other overlapping pairing involving a write is illegal: a
// read-only binding aliasing a write (e.g. Sampled+StorageWrite, the named case), or two writes the graph
// can't order within an atomic pass ("multiple unsynchronized writes").
static constexpr bool in_pass_accesses_conflict(AccessType a,
    uint32_t aBaseMip,
    uint32_t aMipCount,
    uint32_t aBaseLayer,
    uint32_t aLayerCount,
    WGPUTextureAspect aAspect,
    AccessType b,
    uint32_t bBaseMip,
    uint32_t bMipCount,
    uint32_t bBaseLayer,
    uint32_t bLayerCount,
    WGPUTextureAspect bAspect)
{
    if (!ranges_overlap(aBaseMip, aMipCount, bBaseMip, bMipCount))
        return false; // disjoint mips
    if (!ranges_overlap(aBaseLayer, aLayerCount, bBaseLayer, bLayerCount))
        return false; // disjoint layers
    if (!aspects_overlap(aAspect, bAspect))
        return false; // disjoint aspects
    if (!access_is_write(a) && !access_is_write(b))
        return false;
    if ((a == AccessType::StorageRead && b == AccessType::StorageWrite) || (a == AccessType::StorageWrite && b == AccessType::StorageRead))
        return false;
    return true;
}

static void validate_texture_desc(const TextureDesc& desc)
{
    Q_ASSERT(desc.relativeTo.id == 0 || (desc.scaleX > 0.0f && desc.scaleY > 0.0f) && "When relative to another texture it cannot be a scale of 0");

    // MSAA constraints
    Q_ASSERT(desc.sampleCount == 1 || desc.sampleCount == 4 && "WebGPU(1.0) only supports MSAA samples of 1 or 4.");
    Q_ASSERT(desc.dimension != WGPUTextureDimension_3D || desc.sampleCount == 1 && "Texture3D does not support MSAA");

    Q_ASSERT(desc.dimension != WGPUTextureDimension_3D || desc.mipLevelCount == 1 && "Texture3D only supports mipLevelCount of 1");
}

ResourceHandle RenderGraph::create_transient_texture(ResourceId id, const TextureDesc& desc)
{
    validate_texture_desc(desc);
    RenderGraphStorage& s = *storage(this);
    Q_ASSERT(s.m_state == RenderGraphState::Recording && "Render graph is in read only mode after compile()");

    assert_unique_id(s, id);
    Q_ASSERT((desc.relativeTo.id == 0 || desc.relativeTo.generation == s.generation) && "TextureDesc.relativeTo is a stale/foreign handle");

    ResourceNode* resouce = s.m_allocator->make<ResourceNode>();

    resouce->handle = create_handle(s, ResourceKind::Texture);
    resouce->id = { id.value, s.m_allocator->copy_string(id.name) };
    resouce->kind = ResourceKind::Texture;
    resouce->dimension = desc.dimension;
    resouce->format = desc.format;
    resouce->sizeKind = desc.sizeKind;
    resouce->scaleX = desc.scaleX;
    resouce->scaleY = desc.scaleY;
    resouce->relativeToHandle = desc.relativeTo;
    resouce->absolute = desc.absolute;
    resouce->mipLevelCount = desc.mipLevelCount ? desc.mipLevelCount : 1;
    resouce->sampleCount = desc.sampleCount ? desc.sampleCount : 1;

    list_append(&s.m_resouces, &s.m_resoucesTail, resouce);

    return resouce->handle;
}

ResourceHandle RenderGraph::create_transient_buffer(ResourceId id, const BufferDesc& desc)
{
    RenderGraphStorage& s = *storage(this);
    Q_ASSERT(s.m_state == RenderGraphState::Recording && "Render graph is in read only mode after compile()");

    assert_unique_id(s, id);

    ResourceNode* resouce = s.m_allocator->make<ResourceNode>();

    resouce->handle = create_handle(s, ResourceKind::Buffer);
    resouce->id = { id.value, s.m_allocator->copy_string(id.name) };
    resouce->kind = ResourceKind::Buffer;
    resouce->bufferSize = desc.size;

    list_append(&s.m_resouces, &s.m_resoucesTail, resouce);

    return resouce->handle;
}

ResourceHandle RenderGraph::importe_texture(ResourceId id, WGPUTextureView view, WGPUExtent3D size, WGPUTextureFormat format)
{
    RenderGraphStorage& s = *storage(this);
    Q_ASSERT(s.m_state == RenderGraphState::Recording && "Render graph is in read only mode after compile()");

    assert_unique_id(s, id);

    ResourceNode* resouce = s.m_allocator->make<ResourceNode>();

    resouce->handle = create_handle(s, ResourceKind::Texture);
    resouce->id = { id.value, s.m_allocator->copy_string(id.name) };
    resouce->kind = ResourceKind::Texture;
    Q_ASSERT(format != WGPUTextureFormat_Undefined && "importe_texture: pass the imported view's format");
    resouce->imported = true;
    resouce->view = view;
    resouce->resolved = size;
    resouce->format = format;

    list_append(&s.m_resouces, &s.m_resoucesTail, resouce);

    return resouce->handle;
}

ResourceHandle RenderGraph::import_buffer(ResourceId id, WGPUBuffer buffer)
{
    RenderGraphStorage& s = *storage(this);
    Q_ASSERT(s.m_state == RenderGraphState::Recording && "Render graph is in read only mode after compile()");

    assert_unique_id(s, id);

    ResourceNode* resouce = s.m_allocator->make<ResourceNode>();

    resouce->handle = create_handle(s, ResourceKind::Buffer);
    resouce->id = { id.value, s.m_allocator->copy_string(id.name) };
    resouce->kind = ResourceKind::Buffer;
    resouce->imported = true;
    resouce->buffer = buffer;
    resouce->bufferSize = wgpuBufferGetSize(buffer); // so ctx.buffer_size() works for imported too

    list_append(&s.m_resouces, &s.m_resoucesTail, resouce);

    return resouce->handle;
}

// history resource: two rotating physical textures owned by the PersistentResourcePool. allocates
// two ResourceNodes (curr = layer 0, prev = layer 1); the pool backs them and swaps which physical texture
// each maps to every frame (see realize()), so this frame's curr is next frame's prev.
HistoryResource RenderGraph::create_history_texture(ResourceId id, const TextureDesc& desc, uint64_t hash)
{
    RenderGraphStorage& s = *storage(this);
    Q_ASSERT(s.m_state == RenderGraphState::Recording && "Render graph is in read only mode after compile()");

    assert_unique_id(s, id);
    Q_ASSERT((desc.relativeTo.id == 0 || desc.relativeTo.generation == s.generation) && "TextureDesc.relativeTo is a stale/foreign handle");

    HistoryResource out {};
    for (uint32_t i = 0; i < PersistentResourcePool::kLayers; ++i) {
        ResourceNode* resouce = s.m_allocator->make<ResourceNode>();
        resouce->handle = create_handle(s, ResourceKind::Texture);
        resouce->id = { id.value, s.m_allocator->copy_string(id.name) };
        resouce->kind = ResourceKind::Texture;
        resouce->persistent = true;
        resouce->history = true;
        resouce->historyIndex = i;
        resouce->historyHash = hash;
        resouce->dimension = desc.dimension;
        resouce->format = desc.format;
        resouce->sizeKind = desc.sizeKind;
        resouce->scaleX = desc.scaleX;
        resouce->scaleY = desc.scaleY;
        resouce->relativeToHandle = desc.relativeTo;
        resouce->absolute = desc.absolute;
        resouce->mipLevelCount = desc.mipLevelCount ? desc.mipLevelCount : 1;
        resouce->sampleCount = desc.sampleCount ? desc.sampleCount : 1;
        list_append(&s.m_resouces, &s.m_resoucesTail, resouce);
        if (i == 0)
            out.curr = resouce->handle;
        else
            out.prev = resouce->handle;
    }
    return out;
}

// history buffer: two rotating physical buffers
// owned by the PersistentResourcePool; allocates two ResourceNodes (curr = layer 0, prev = layer 1) and
// the pool swaps which physical buffer each maps to every frame, so this frame's curr is next frame's prev.
HistoryResource RenderGraph::create_history_buffer(ResourceId id, const BufferDesc& desc, uint64_t hash)
{
    RenderGraphStorage& s = *storage(this);
    Q_ASSERT(s.m_state == RenderGraphState::Recording && "Render graph is in read only mode after compile()");

    assert_unique_id(s, id);

    HistoryResource out {};
    for (uint32_t i = 0; i < PersistentResourcePool::kLayers; ++i) {
        ResourceNode* resouce = s.m_allocator->make<ResourceNode>();
        resouce->handle = create_handle(s, ResourceKind::Buffer);
        resouce->id = { id.value, s.m_allocator->copy_string(id.name) };
        resouce->kind = ResourceKind::Buffer;
        resouce->persistent = true;
        resouce->history = true;
        resouce->historyIndex = i;
        resouce->historyHash = hash;
        resouce->bufferSize = desc.size;
        list_append(&s.m_resouces, &s.m_resoucesTail, resouce);
        if (i == 0)
            out.curr = resouce->handle;
        else
            out.prev = resouce->handle;
    }
    return out;
}


ResourceHandle RenderGraph::create_persistent_buffer(ResourceId id, const BufferDesc& desc)
{
    RenderGraphStorage& s = *storage(this);
    Q_ASSERT(s.m_state == RenderGraphState::Recording && "Render graph is in read only mode after compile()");

    assert_unique_id(s, id);

    ResourceNode* resouce = s.m_allocator->make<ResourceNode>();
    resouce->handle = create_handle(s, ResourceKind::Buffer);
    resouce->id = { id.value, s.m_allocator->copy_string(id.name) };
    resouce->kind = ResourceKind::Buffer;
    resouce->persistent = true;
    resouce->historyIndex = 0; // the only layer
    resouce->bufferSize = desc.size;
    list_append(&s.m_resouces, &s.m_resoucesTail, resouce);
    return resouce->handle;
}


ResourceHandle RenderGraph::create_persistent_texture(ResourceId id, const TextureDesc& desc)
{
    RenderGraphStorage& s = *storage(this);
    Q_ASSERT(s.m_state == RenderGraphState::Recording && "Render graph is in read only mode after compile()");

    assert_unique_id(s, id);
    Q_ASSERT((desc.relativeTo.id == 0 || desc.relativeTo.generation == s.generation) && "TextureDesc.relativeTo is a stale/foreign handle");

    ResourceNode* resouce = s.m_allocator->make<ResourceNode>();
    resouce->handle = create_handle(s, ResourceKind::Texture);
    resouce->id = { id.value, s.m_allocator->copy_string(id.name) };
    resouce->kind = ResourceKind::Texture;
    resouce->persistent = true;
    resouce->historyIndex = 0; // the only layer
    resouce->dimension = desc.dimension;
    resouce->format = desc.format;
    resouce->sizeKind = desc.sizeKind;
    resouce->scaleX = desc.scaleX;
    resouce->scaleY = desc.scaleY;
    resouce->relativeToHandle = desc.relativeTo;
    resouce->absolute = desc.absolute;
    resouce->mipLevelCount = desc.mipLevelCount ? desc.mipLevelCount : 1;
    resouce->sampleCount = desc.sampleCount ? desc.sampleCount : 1;
    list_append(&s.m_resouces, &s.m_resoucesTail, resouce);
    return resouce->handle;
}

static bool is_depth_format(WGPUTextureFormat f)
{
    switch (f) {
    case WGPUTextureFormat_Depth16Unorm:
    case WGPUTextureFormat_Depth24Plus:
    case WGPUTextureFormat_Depth24PlusStencil8:
    case WGPUTextureFormat_Depth32Float:
    case WGPUTextureFormat_Depth32FloatStencil8:
        return true;
    default:
        return false;
    }
}

ResourceHandle RenderGraph::create_initialized_texture(ResourceId id, const TextureDesc& desc, WGPUColor fill)
{
    ResourceHandle h = create_persistent_texture(id, desc);
    uint32_t layers = desc.absolute.depthOrArrayLayers ? desc.absolute.depthOrArrayLayers : 1;
    bool depth = is_depth_format(desc.format);
    for (uint32_t layer = 0; layer < layers; ++layer)
        add_pass(
            id,
            PassKind::Graphics,
            [h, depth, fill, layer](PassBuilder& b) {
                if (depth)
                    b.depth_stencil(h, WGPULoadOp_Clear, WGPUStoreOp_Store, (float)fill.r, 0, layer);
                else
                    b.color(h, WGPULoadOp_Clear, WGPUStoreOp_Store, fill, 0, layer);
                b.initialize(h);
            },
            [](PassContext&) {});
    return h;
}

ResourceHandle RenderGraph::create_initialized_buffer(ResourceId id, const BufferDesc& desc, const void* data)
{
    ResourceHandle h = create_persistent_buffer(id, desc);
    RenderGraphStorage& s = *storage(this);
    uint8_t* bytes = s.m_allocator->alloc<uint8_t>(desc.size); // zeroed; null data leaves it zero-filled
    if (data)
        std::memcpy(bytes, data, desc.size);
    uint64_t size = desc.size;
    add_pass(
        id,
        PassKind::Transfer,
        [h](PassBuilder& b) {
            b.host_write(h);
            b.initialize(h);
        },
        [h, bytes, size](PassContext& ctx) { wgpuQueueWriteBuffer(ctx.queue, ctx.buffer(h), 0, bytes, size); });
    return h;
}

WGPUTextureFormat RenderGraph::texture_format(ResourceHandle h) const
{
    ResourceNode* r = find_node(const_cast<RenderGraph*>(this), h);
    Q_ASSERT(r && "texture_format: unknown handle");
    Q_ASSERT(r->kind == ResourceKind::Texture && "texture_format: handle is a buffer, not a texture");
    return r->format;
}

PassBuilder RenderGraph::begin_pass(ResourceId id, PassKind kind)
{
    RenderGraphStorage& s = *storage(this);
    Q_ASSERT(s.m_state == RenderGraphState::Recording && "Render graph is in read only mode after compile()");
    PassNode* pass = s.m_allocator->make<PassNode>();
    pass->id = { id.value, s.m_allocator->copy_string(id.name) };
    pass->kind = kind;

    PassBuilder builder;
    builder.m_pass = pass;
    builder.m_graph = this;
    return builder;
}

void RenderGraph::end_pass(PassBuilder& builder)
{
    PassNode* pass = builder.m_pass;
    for (uint32_t t = 0; t < pass->initCount; ++t) {
        bool wrote = false;
        for (uint32_t i = 0; i < pass->accessCount; ++i)
            if (pass->accesses[i].handle.id == pass->initTargets[t].target.id && access_is_write(pass->accesses[i].type)) {
                wrote = true;
                break;
            }
        Q_ASSERT(wrote
            && "initialize() target must also be written by this pass (e.g. .color()/.storage_write()). "
               "initialize() only gates skip/run, it is not itself a write");
    }
    RenderGraphStorage& s = *storage(this);
    list_append(&s.m_passes, &s.m_passesTail, pass);
}

void* RenderGraph::alloc_exec(size_t size, size_t align) { return storage(this)->m_allocator->alloc_raw(size, align); }

void RenderGraph::set_exec(PassBuilder& builder, void* obj, void (*fn)(void*, PassContext&))
{
    builder.m_pass->exec_obj = obj;
    builder.m_pass->exec_fn = fn;
}

// records that pass `p` depends on pass `dep` (dedup; p->adjacency = predecessors)
static void add_dependency(GraphAllocator* alloc, PassNode* p, PassNode* dep)
{
    for (NodeAdjacency* a = p->adjacency; a; a = a->next)
        if (a->pass == dep)
            return; // already linked
    NodeAdjacency* link = alloc->make<NodeAdjacency>();
    link->pass = dep;
    link->next = p->adjacency;
    p->adjacency = link; // prepend
}

enum struct HazardKind : uint8_t { RAW, WAW, WAR };

// One declaration-order sweep with implicit SSA resource versioning, driving compile() phase 1;
// each discovered edge becomes an add_dependency. Each write to a resource starts a new version, with
// the writing pass as the version identity; each read binds to the current one. Calls
// onEdge(dependent, dep, id, kind) per discovered hazard: RAW (read -> producer), WAW (write -> prev
// writer), WAR (write -> readers of the version being clobbered); `dependent` is always later in the
// walk than `dep`. A read seen before any writer of its resource binds to no producer, emitting no
// edge. Detecting that authoring error is left to compile()'s post-cull pass, which sees the final
// schedule; the sweep stays edge-only.
// NOTE(Huerbe): versioning is per resource id; a write to any subresource supersedes the whole resource,
// so disjoint mip/layer touches get a false but harmless ordering edge. Subresource-precise is declined.
template <typename OnEdge>
static void sweep_resource_versions(Arena& scratch, PassNode* head, uint32_t next_resource_id, OnEdge&& onEdge)
{
    // per resource id (1..next_resource_id-1): the pass holding the current version, and the readers of
    // that version not yet retired by a newer write. Scratch only; the permanent DAG edges are built by
    // the onEdge callback, which compile() phase 1 routes through add_dependency onto the front arena.
    ScopedScratch ss(scratch);
    PassNode** currentProducer = ss.alloc<PassNode*>(next_resource_id);
    NodeAdjacency** pendingReaders = ss.alloc<NodeAdjacency*>(next_resource_id);

    for (PassNode* p = head; p; p = p->next) {
        if (p->skipInit)
            continue; // initialize() pass already satisfied -> treat as absent (no versions/edges)
        for (uint32_t i = 0; i < p->accessCount; ++i) {
            uint32_t id = p->accesses[i].handle.id;
            if (!id)
                continue; // invalid/default handle: nothing to version (post-cull check skips id 0 too)
            if (access_is_write(p->accesses[i].type)) {
                // WAW: order this write after the previous writer. without it two writers of one
                // resource have no edge -> undefined order -> corruption.
                if (currentProducer[id] && currentProducer[id] != p)
                    onEdge(p, currentProducer[id], id, HazardKind::WAW);
                // WAR: order this write after every reader still using the version it clobbers.
                for (NodeAdjacency* r = pendingReaders[id]; r; r = r->next)
                    if (r->pass != p)
                        onEdge(p, r->pass, id, HazardKind::WAR);
                currentProducer[id] = p; // new version born
                pendingReaders[id] = nullptr; // its readers retired (old nodes are arena garbage)
            } else {
                // RAW: this read depends on the producer of the version it sees. a read before any
                // writer binds to "no producer" (no edge); compile()'s post-cull pass flags it.
                if (currentProducer[id] && currentProducer[id] != p)
                    onEdge(p, currentProducer[id], id, HazardKind::RAW);
                // register as a pending reader of the current version (for a future write's WAR).
                NodeAdjacency* link = ss.make<NodeAdjacency>();
                link->pass = p;
                link->next = pendingReaders[id];
                pendingReaders[id] = link;
            }
        }
    }

    // both self-guards above are load-bearing: read-then-write of one handle in a single pass would
    // WAR-self-edge; write-then-write would WAW-self-edge. every edge points from a later- to an
    // earlier-visited pass, so adjacency is acyclic by construction (no compile()-made cycles).
}

// Topological sort via recursive DFS post-order over predecessor (depends-on) edges: a pass is appended
// only after everything it depends on, so `order` comes out deps-first. `placed` doubles as the visited
// marker so shared deps are emitted once. Seeded from sinks (see compile() phase 2), so recursion reaches
// only passes a sink transitively depends on and dead-node removal falls out for free. `onStack` catches a
// back-edge (a pass re-entered while on the recursion stack) and sets `hadCycle`; the graph is acyclic by
// construction (phase 1 emits only backward edges), so that is a backstop.
static void topo_visit(PassNode* p, PassNode** order, uint32_t& count, bool& hadCycle)
{
    if (p->onStack) {
        hadCycle = true;
        return;
    } // back-edge: pass depends (transitively) on itself
    if (p->placed)
        return;
    p->placed = p->onStack = true;
    for (NodeAdjacency* a = p->adjacency; a; a = a->next)
        topo_visit(a->pass, order, count, hadCycle);
    p->onStack = false;
    order[count++] = p; // all deps already placed
}

// a sink/output pass writes a resource whose value must survive without a same-frame reader: an imported
// resource (swapchain) or a history .curr (history, read NEXT frame). Those, plus force_keep() (checked at
// the seed), are the only cull roots. A plain persistent bake is deliberately NOT a root: it is pool-cached
// and dependency-driven, kept only if a surviving pass reads it, so an unread fallback bake is culled for
// free. `sinkRoot` is imported||history per id (see phase 2), NOT is_external() (which also covers bakes).
static bool is_sink(PassNode* p, const bool* sinkRoot)
{
    for (uint32_t i = 0; i < p->accessCount; ++i)
        if (access_is_write(p->accesses[i].type) && sinkRoot[p->accesses[i].handle.id])
            return true;
    return false;
}

// resolve one texture's concrete size by walking its relativeTo chain. imported (registered size)
// and Absolute are the base cases; a Relative node multiplies its base's resolved size by its own
// scale, so the scale accumulates down a chain (depth -> colorAttachment -> swapchain). memoized via
// resolved.width (non-zero => already done), which also makes declaration order irrelevant.
static WGPUExtent3D resolve_size(ResourceNode* r, ResourceNode** byId)
{
    Q_ASSERT(byId);
    if (r->resolved.width)
        return r->resolved; // imported, or already walked
    if (r->sizeKind == SizeKind::Absolute)
        return r->resolved = r->absolute;
    // memoization happens only after the recursion below, so a relativeTo cycle would recurse forever;
    // guard it here (author-acyclic assumption, same as topo_visit). break the cycle with a zero base.
    if (r->resolving) {
        Q_ASSERT(false && "cyclic relativeTo chain");
        return WGPUExtent3D {};
    }
    r->resolving = true;
    ResourceNode* base = byId[r->relativeToHandle.id];
    WGPUExtent3D b = base ? resolve_size(base, byId) : WGPUExtent3D {};
    r->resolving = false;
    // scale only width/height; layer count (cube/array) is the node's own, not the base's.
    uint32_t layers = r->absolute.depthOrArrayLayers ? r->absolute.depthOrArrayLayers : 1;
    // round, don't truncate: at scale 0.5 an odd base dim (1281 -> 640.5) must land on 641, not 640.
    // Truncation would drop a pixel. +0.5 then cast is round-half-up, valid since sizes are >= 0.
    return r->resolved = { (uint32_t)(b.width * r->scaleX + 0.5f), (uint32_t)(b.height * r->scaleY + 0.5f), layers };
}

// Transient-attachment inference. TRANSIENT_ATTACHMENT (gpuweb transient-attachments proposal) marks a
// texture whose contents never leave the GPU, so a TBDR GPU keeps it in tile memory and skips the VRAM
// allocation ("memoryless"). Inferred from the usage gather: a graph-owned texture used as a single
// cleared+discarded color/depth attachment, never sampled/copied/stored. realize() adds the usage bit
// when the device supports the feature, a no-op hint elsewhere (e.g. D3D12). The preconditions below are
// exactly what the proposal requires, so the descriptor is spec-valid by construction. Runs after phase 3.
static void detect_transient_attachments(RenderGraphStorage& s)
{
    for (ResourceNode* r = s.m_resouces; r; r = r->next) {
        if (r->is_external() || r->kind != ResourceKind::Texture)
            continue;
        if (r->texUsage != WGPUTextureUsage_RenderAttachment)
            continue; // sampled/storage/copy -> must keep bytes
        if (r->dimension != WGPUTextureDimension_2D || r->mipLevelCount != 1 || r->resolved.depthOrArrayLayers > 1)
            continue; // proposal: 2d, one mip, one layer

        // exactly one recorded access, or its contents span passes and aren't memoryless.
        uint32_t nAcc = 0;
        const ResourceAccess* a = nullptr;
        for (PassNode* p = s.m_passes; p; p = p->next)
            for (uint32_t i = 0; i < p->accessCount; ++i)
                if (p->accesses[i].handle.id == r->handle.id) {
                    ++nAcc;
                    a = &p->accesses[i];
                }
        if (nAcc != 1)
            continue;

        // a writable color/depth attachment only: a ResolveAttachment is the kept resolve output, and a
        // read-only depth loads prior contents; neither can be memoryless.
        if (a->type != AccessType::ColorAttachment && a->type != AccessType::DepthStencilAttachment)
            continue;
        if (a->loadOp != WGPULoadOp_Clear || a->storeOp != WGPUStoreOp_Discard)
            continue;
        // combined depth+stencil: a declared stencil aspect must clear+discard too. No format table
        // here; Undefined means none declared, as for a depth-only format.
        if (a->type == AccessType::DepthStencilAttachment && a->stencilLoadOp != WGPULoadOp_Undefined
            && (a->stencilLoadOp != WGPULoadOp_Clear || a->stencilStoreOp != WGPUStoreOp_Discard))
            continue;

        r->transientAttachment = true;
        ++s.transientCount;
    }
}

// A history resource is two layer nodes sharing one pool entry: .curr (layer 0, write target) and .prev
// (layer 1, read this frame). realize() rotates the entry per node a surviving pass uses, so the two must
// be co-used (read .prev + write .curr in one pass). If cull keeps .prev but drops .curr, the ping-pong
// rotates on the reader alone and .prev eventually samples a slot .curr never wrote while historyValid
// still reports true: an unrealizable graph, so error. Runs after phase 3 (needs the accumulated usage).
static void validate_history_layers(RenderGraphStorage& s)
{
    auto pool_used = [](const ResourceNode* r) { return r->kind == ResourceKind::Texture ? r->texUsage != 0 : r->bufUsage != 0; };
    for (ResourceNode* r = s.m_resouces; r; r = r->next) {
        if (!r->history || r->historyIndex != 0)
            continue; // anchor on .curr; skip .prev and single-layer persistent
        ResourceNode* prev = r->next; // .prev (layer 1) was appended right after .curr
        Q_ASSERT(prev && prev->history && prev->id.value == r->id.value && prev->historyIndex == 1 && "history layer nodes must be adjacent, curr then prev");
        if (pool_used(r) != pool_used(prev))
            push_error(s,
                "history resource \"%.*s\": .curr (layer 0) is %s but .prev (layer 1) is %s "
                "after culling; read .prev and write .curr in one pass, or use neither.",
                (int)r->id.name.length,
                r->id.name.data ? r->id.name.data : "",
                pool_used(r) ? "used" : "unused",
                pool_used(prev) ? "used" : "unused");
    }
}

void RenderGraph::compile(bool enableAlias)
{
    auto t0 = std::chrono::steady_clock::now();
    RenderGraphStorage& s = *storage(this);
    Q_ASSERT(s.m_state == RenderGraphState::Recording);

    // handle backstop (release-safe): every access and initialize() target must be a handle this frame's
    // graph created. Reject id 0, an id past next_resource_id (would index the phase 1/2 scratch tables out
    // of bounds), or a generation mismatch (a handle from another graph instance or an older frame; the
    // process-global counter makes each graph unique). Poison rather than assert so the frame is skipped in
    // release too; the matching debug assert also fires at the b.*() call in use().
    for (PassNode* p = s.m_passes; p; p = p->next) {
        for (uint32_t i = 0; i < p->accessCount; ++i) {
            ResourceHandle h = p->accesses[i].handle;
            if (h.id == 0 || h.id >= s.next_resource_id || h.generation != s.generation) {
                push_error(s,
                    "pass \"%.*s\" uses an invalid resource handle (id %u): zero, out of range, "
                    "or from another graph / a previous frame.",
                    (int)p->id.name.length,
                    p->id.name.data ? p->id.name.data : "",
                    h.id);
                return;
            }
        }
        for (uint32_t i = 0; i < p->initCount; ++i) {
            ResourceHandle t = p->initTargets[i].target;
            if (t.id >= s.next_resource_id || t.generation != s.generation) {
                push_error(s,
                    "pass \"%.*s\" initialize() target handle (id %u) is from another graph or a "
                    "previous frame.",
                    (int)p->id.name.length,
                    p->id.name.data ? p->id.name.data : "",
                    t.id);
                return;
            }
        }
    }
    // same backstop for relativeTo: resolve_size() indexes byId[relativeToHandle.id] unguarded, so a
    // stale/foreign handle would read out of bounds or size against the wrong resource. id 0 = absolute-
    // sized, no base. Invalidate rather than assert; the matching debug assert also fired at create_*_texture.
    for (ResourceNode* r = s.m_resouces; r; r = r->next) {
        ResourceHandle rel = r->relativeToHandle;
        if (rel.id != 0 && (rel.id >= s.next_resource_id || rel.generation != s.generation)) {
            push_error(s,
                "resource \"%.*s\" relativeTo (id %u) is from another graph or a previous frame.",
                (int)r->id.name.length,
                r->id.name.data ? r->id.name.data : "",
                rel.id);
            return;
        }
    }

    // Build fast index based lookup
    s.byId = s.m_allocator->alloc<ResourceNode*>(s.next_resource_id);
    for (ResourceNode* r = s.m_resouces; r; r = r->next)
        s.byId[r->handle.id] = r;

    WGPUTextureUsage* predTexUsage = s.m_allocator->alloc<WGPUTextureUsage>(s.next_resource_id);
    WGPUBufferUsage* predBufUsage = s.m_allocator->alloc<WGPUBufferUsage>(s.next_resource_id);
    bool* hasWriter = s.m_allocator->alloc<bool>(s.next_resource_id);
    for (PassNode* p = s.m_passes; p; p = p->next)
        for (uint32_t i = 0; i < p->accessCount; ++i) {
            const ResourceAccess& a = p->accesses[i];
            predTexUsage[a.handle.id] |= tex_usage_of(a.type);
            predBufUsage[a.handle.id] |= buf_usage_of(a.type);
            hasWriter[a.handle.id] = access_is_write(a.type);
        }

    // phase 0: skip up-to-date bake passes. An initialize() pass rebakes a persistent target; skip it
    // this frame only when the pool entry is realized (`created`), still holds baked content (`baked`,
    // cleared on any recreate so a resized/blank target re-bakes), and was baked with this settings hash.
    // A skipped pass produces no version and is not a cull root, so phases 1-2 drop it; readers still bind
    // to the pooled result (persistent is external, no in-graph writer needed). execute() re-stamps the
    // hash when the pass runs.
    for (PassNode* p = s.m_passes; p; p = p->next) {
        p->skipInit = false;
        if (!p->initCount)
            continue;
        // skip the whole pass only when every target is realized and holds content baked with its exact
        // hash. One stale or non-persistent target re-arms the whole pass: a body bakes all its targets or none.
        bool allBaked = true;
        for (uint32_t i = 0; i < p->initCount; ++i) {
            ResourceNode* r = s.byId[p->initTargets[i].target.id]; // non-null: backstop range-checked the id
            if (!r->persistent) {
                push_error(s, "initialize() target must be persistent or history (create_persistent_texture/_buffer or create_history_texture/_buffer)");
                allBaked = false;
                continue;
            }
            PersistentResourcePool::Entry* e = s.m_allocator->pool.find(r->id);
            bool upToDate = e && e->created && e->baked && e->initHash == p->initTargets[i].hash;
            // recreate-aware: a matching bake is still stale if realize() would destroy+recreate the entry
            // (clearing `baked`) after skipInit was decided, leaving it un-baked for a frame. Predict the
            // exact recreate condition (usage growth or a size/format/dim/mip/sample change; resolve_size is
            // memoized and device-free, so calling it here is safe) and keep the bake so it re-fills the
            // recreated resource on the same frame.
            bool wouldRecreate = false;
            if (e && e->created) {
                if (r->kind == ResourceKind::Texture) {
                    resolve_size(r, s.byId);
                    wouldRecreate = (predTexUsage[r->handle.id] & ~e->usageAtCreate) != 0
                        || PersistentResourcePool::tex_descriptor_differs(*e, r->resolved, r->format, r->dimension, r->mipLevelCount, r->sampleCount);
                } else {
                    wouldRecreate = (predBufUsage[r->handle.id] & ~e->bufUsageAtCreate) != 0 || e->bufferSize != r->bufferSize;
                }
                // touch() also destroys+recreates on a history-hash mismatch (camera cut / settings
                // change); mirror it so a hash-only change re-arms the bake this frame instead of
                // leaving the recreated target blank.
                wouldRecreate = wouldRecreate || (r->historyHash && e->historyHash != r->historyHash);
            }
            if (!upToDate || wouldRecreate)
                allBaked = false;
        }
        p->skipInit = allBaked;
    }

    // phase 1: build the dependency DAG in the "depends-on" direction. The versioning sweep
    // (see sweep_resource_versions) discovers every RAW/WAW/WAR hazard in declaration order; here all
    // three collapse to add_dependency, whose dedup folds multiple hazards between one pass pair into
    // the single ordering edge phase 2 needs, so the resource id and hazard kind are ignored. Reads
    // before any writer get no edge here; the post-cull pass below turns them into errors.
    sweep_resource_versions(s.m_allocator->scratch, s.m_passes, s.next_resource_id, [&](PassNode* dependent, PassNode* dep, uint32_t id, HazardKind kind) {
        add_dependency(s.m_allocator, dependent, dep);
    });

    // phase 2: cull dead passes and topo sort, fused into one DFS seeded from sinks.
    {
        // sinkRoot = imported or history .curr: value must survive without a same-frame reader, so a pass
        // writing one is an output sink. Persistent BAKES are excluded on purpose (pool-cached, dependency-
        // driven). accesses store only handle.id, so flatten the flags into an id-indexed table first.
        ScopedScratch ss(s.m_allocator->scratch);
        bool* sinkRoot = ss.alloc<bool>(s.next_resource_id);
        for (ResourceNode* r = s.m_resouces; r; r = r->next)
            sinkRoot[r->handle.id] = r->imported || r->history;

        // topo into a transient array, then relink the intrusive list into execution order. The
        // result lives in m_passes itself; the array is just DFS scratch, reclaimed when ss goes out of scope.
        uint32_t N = 0;
        for (PassNode* p = s.m_passes; p; p = p->next)
            ++N;

        PassNode** order = ss.alloc<PassNode*>(N);
        uint32_t count = 0;
        bool hadCycle = false;
        for (PassNode* p = s.m_passes; p; p = p->next)
            if (!p->skipInit
                && (is_sink(p, sinkRoot) || p->forceKeep)) // satisfied initialize() pass is not a root; force_keep() keeps a reader-less side-effect pass alive
            {
                p->sink = true;
                topo_visit(p, order, count, hadCycle); // only reaches passes that feed a sink
            }

        // acyclic by construction (phase 1 emits only backward edges), so a cycle means the graph
        // was corrupted, not merely misauthored. poison and bail before realize()/execute() ever run.
        if (hadCycle) {
            push_error(s,
                "compile() found a cycle in the pass dependency graph. The graph is "
                "acyclic by construction, so this is an internal error, not an authoring one.");
            return;
        }

        // relink next-pointers to follow topo order; m_passes is now == execution order, and any
        // pass not reachable from a sink was never emitted -> dead, dropped here for free.
        for (uint32_t i = 0; i + 1 < count; ++i)
            order[i]->next = order[i + 1];
        if (count)
            order[count - 1]->next = nullptr;
        s.m_passes = count ? order[0] : nullptr;
        s.m_passesTail = count ? order[count - 1] : nullptr;

        // NOTE(Huerbe): transient array as DFS scratch; can't sort a one-field intrusive list in
        // place (a dep emitted before the driver reaches it clobbers its `next`, dropping
        // disconnected nodes).
    }

    // post-cull validation. Over the FINAL schedule (m_passes is now culled + in execution order), a read
    // of a TRANSIENT resource that no earlier pass has produced is an authoring error: the reader would
    // sample uninitialized contents (its writer was declared after it, or culled). Walking the surviving
    // passes makes this culling-correct and catches every surviving reader, not just the first.
    //   imported + history/persistent resources  -> exempt (their value comes from outside this frame).
    //   resources with no writer at all (e.g. a host-uploaded uniform) -> exempt (hasWriter stays false).
    // bail before phase 3 so the caller never realize()/execute()s a misordered graph.
    {
        // hasWriter (from the pre-cull walk above): some DECLARED pass writes id, culled or not.
        ScopedScratch ss(s.m_allocator->scratch);
        bool* produced = ss.alloc<bool>(s.next_resource_id); // a surviving pass has written it so far, in order
        bool* external = ss.alloc<bool>(s.next_resource_id); // imported OR history: value from outside the frame
        bool* prevLayer = ss.alloc<bool>(s.next_resource_id); // history layer k>0: read-only this frame
        for (ResourceNode* r = s.m_resouces; r; r = r->next) {
            external[r->handle.id] = r->is_external();
            prevLayer[r->handle.id] = r->persistent && r->historyIndex != 0;
        }

        bool hadError = false;
        for (PassNode* p = s.m_passes; p; p = p->next)
            for (uint32_t i = 0; i < p->accessCount; ++i) {
                uint32_t id = p->accesses[i].handle.id;
                if (access_is_write(p->accesses[i].type)) {
                    produced[id] = true;
                    // older layers of a history resource are read-only this frame: writing layer k>0 clobbers
                    // a slot that becomes a future "current". layer 0 (the `.curr` handle from
                    // create_history_texture/_buffer) is the only legal write target.
                    if (prevLayer[id]) {
                        ResourceNode* w = find_node(this, p->accesses[i].handle);
                        WGPUStringView wn = w ? w->id.name : WGPUStringView {};
                        push_error(s,
                            "pass \"%.*s\" writes history resource \"%.*s\" "
                            "layer %u; only layer 0, the .curr handle, is writable.",
                            (int)p->id.name.length,
                            p->id.name.data ? p->id.name.data : "",
                            (int)wn.length,
                            wn.data ? wn.data : "",
                            w ? w->historyIndex : 0u);
                        hadError = true;
                    }
                    continue;
                }
                if (id == 0 || external[id] || produced[id] || !hasWriter[id])
                    continue;
                ResourceNode* r = find_node(this, p->accesses[i].handle);
                WGPUStringView rn = r ? r->id.name : WGPUStringView {};
                push_error(s,
                    "pass \"%.*s\" reads resource \"%.*s\" before any pass "
                    "writes it; declare a writer of \"%.*s\" first.",
                    (int)p->id.name.length,
                    p->id.name.data ? p->id.name.data : "",
                    (int)rn.length,
                    rn.data ? rn.data : "",
                    (int)rn.length,
                    rn.data ? rn.data : "");
                hadError = true;
            }
        if (hadError)
            return;
    }

    // phase 3: accumulate WGPU usage and resolve concrete sizes, frame-independent CPU analysis.
    // WebGPU requires the usage bit at create time; realize() then only does the device create calls.
    {
        uint32_t passIdx = 0;
        for (PassNode* p = s.m_passes; p; p = p->next, ++passIdx) // m_passes == surviving (post-cull) passes
            for (uint32_t i = 0; i < p->accessCount; ++i) {
                ResourceNode* r = s.byId[p->accesses[i].handle.id];
                if (!r)
                    continue;
                // lifetime: the walk is already in execution order, so the first touch is firstUse and
                // each later touch overwrites lastUse. imported resources are skipped, excluded from
                // aliasing, so a span would be meaningless.
                if (!r->is_external()) { // persistent is pool-owned -> excluded from aliasing, like imported
                    if (r->firstUse == ResourceNode::kNoPass) {
                        r->firstUse = passIdx;
                        r->firstDefines = access_defines(p->accesses[i]); // aliasing: does the first touch overwrite?
                    }
                    r->lastUse = passIdx;
                    if (access_is_write(p->accesses[i].type))
                        r->hasWriter = true;
                }
                // texture vs buffer accumulate into distinct usage fields; tex_usage_of/buf_usage_of are
                // kind-agnostic, so dispatch on the node kind (StorageRead/Write and Copy map to either).
                if (r->kind == ResourceKind::Texture)
                    r->texUsage |= tex_usage_of(p->accesses[i].type);
                else
                    r->bufUsage |= buf_usage_of(p->accesses[i].type);
            }

        // resolve concrete sizes here (CPU-only -> belongs in compile, not realize) by walking each
        // texture's relativeTo chain. memoized + recursive, so chains and any declaration order work.
        for (ResourceNode* r = s.m_resouces; r; r = r->next)
            if (r->kind == ResourceKind::Texture)
                resolve_size(r, s.byId);

        // NOTE(Huerbe): usage==0 here == untouched by a live pass -> future realize() skips it = free
        // dead-resource culling. no separate resource liveness list needed.
    }

    // reject a history whose .curr/.prev layers ended up asymmetrically culled (needs phase-3 usage);
    // bail before realize()/execute() ever sees the misauthored graph, like the read-before-write check.
    validate_history_layers(s);
    if (s.m_state == RenderGraphState::Failed)
        return;

    // infer transient (memoryless) attachments from the usage gather. after phase 3 (needs texUsage +
    // resolved size), before phase 4 so the aliasing packer skips them.
    detect_transient_attachments(s);

    // phase 4: transient memory aliasing (opt-in). Pack disjoint-lifetime, same-signature transients onto
    // a shared physical object so peak VRAM tracks max simultaneous overlap, not the sum of all transients.
    // The schedule is a single linear queue, so each transient's liveness is a closed [firstUse,lastUse]
    // interval and greedy left-edge first-fit is optimal (the linear-scan register-allocation trick).
    // Off (enableAlias==false) -> no slots, every aliasSlot stays kNoSlot, realize() takes the plain path.
    if (enableAlias) {
        // eligible transients (see ResourceNode::aliasSlot): graph-owned and touched by a live pass; the
        // first touch fully defines, so it never reads a previous occupant's bytes, and some pass writes
        // it, so host-uploaded contents are not stomped. textures must be single mip and sample so a
        // slot's default view fits every member; buffers carry no such constraint. imported/persistent
        // and dead (kNoPass) excluded. collect into scratch, then sort by firstUse for the left-edge sweep.
        // NOTE(Huerbe): firstDefines treats a StorageWrite as a full define. For a buffer a storage_write
        // may write only part, so the shader owns writing every byte it later reads.
        ScopedScratch ss(s.m_allocator->scratch);
        ResourceNode** elig = ss.alloc<ResourceNode*>(s.next_resource_id);
        uint32_t nElig = 0;
        for (ResourceNode* r = s.m_resouces; r; r = r->next) {
            // transientAttachment: memoryless, so there's no persistent storage to alias, and packing it
            // would widen the slot usage past the exact TRANSIENT|RENDER_ATTACHMENT the bit requires.
            if (r->is_external() || r->firstUse == ResourceNode::kNoPass || !r->hasWriter || !r->firstDefines || r->transientAttachment)
                continue;
            if (r->kind == ResourceKind::Texture && (r->mipLevelCount != 1 || r->sampleCount != 1 || r->resolved.depthOrArrayLayers > 1))
                continue; // mip chain / MSAA: a slot's default view wouldn't fit. array: a first-touch
                          // clear defines only one layer, so a successor reading other layers would see
                          // the previous occupant's bytes (firstDefines checks the first access only).
            elig[nElig++] = r;
        }

        // insertion sort by firstUse asc
        for (uint32_t i = 1; i < nElig; ++i) {
            ResourceNode* key = elig[i];
            uint32_t j = i;
            while (j && elig[j - 1]->firstUse > key->firstUse) {
                elig[j] = elig[j - 1];
                --j;
            }
            elig[j] = key;
        }

        // first-fit: reuse the first signature-matching slot whose occupant is already dead (STRICT
        // freeFrom < firstUse: equality means they share a pass -> binding one object as two resources in
        // one usage scope, illegal), else open a new one. slots live in the front arena (read this frame by
        // realize/release). m_slots == null / m_slotCount == 0 when nothing is eligible.
        if (nElig)
            s.m_slots = s.m_allocator->alloc<PhysicalResource>(nElig);
        for (uint32_t i = 0; i < nElig; ++i) {
            ResourceNode* r = elig[i];
            const bool isBuf = r->kind == ResourceKind::Buffer;
            uint32_t slot = ResourceNode::kNoSlot;
            for (uint32_t k = 0; k < s.m_slotCount; ++k) {
                PhysicalResource& ph = s.m_slots[k];
                if (ph.kind != r->kind)
                    continue; // never mix textures + buffers in one slot
                const bool sig = isBuf ? (ph.bufferSize == r->bufferSize)
                                       : (ph.dimension == r->dimension && ph.format == r->format && ph.size.width == r->resolved.width
                                             && ph.size.height == r->resolved.height && ph.size.depthOrArrayLayers == r->resolved.depthOrArrayLayers);
                if (sig && ph.freeFrom < r->firstUse) {
                    slot = k;
                    break;
                } // STRICT: no shared pass
            }
            if (slot == ResourceNode::kNoSlot) {
                slot = s.m_slotCount++;
                PhysicalResource& ph = s.m_slots[slot];
                ph.kind = r->kind;
                if (isBuf)
                    ph.bufferSize = r->bufferSize;
                else {
                    ph.dimension = r->dimension;
                    ph.format = r->format;
                    ph.size = r->resolved;
                }
            }
            PhysicalResource& ph = s.m_slots[slot];
            if (isBuf)
                ph.bufUsage |= r->bufUsage; // widen to the union: every member shares this one object
            else
                ph.texUsage |= r->texUsage;
            ph.freeFrom = r->lastUse;
            r->aliasSlot = slot;

            const bool sigOk = ph.kind == r->kind
                && (isBuf ? ph.bufferSize == r->bufferSize
                          : (ph.dimension == r->dimension && ph.format == r->format && ph.size.width == r->resolved.width
                                && ph.size.height == r->resolved.height));
            if (!sigOk) {
                WGPUStringView rn = r->id.name;
                push_error(s,
                    "alias slot %u signature mismatch for resource \"%.*s\": "
                    "left-edge packing put mismatched signatures on one physical slot.",
                    slot,
                    (int)rn.length,
                    rn.data ? rn.data : "");
                Q_ASSERT(false && "alias slot signature mismatch");
            }
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    s.timing_compile_us = std::chrono::duration<float, std::micro>(t1 - t0).count();

    // Compiled unless the graph was poisoned. Ordering/cycle errors already set Failed and returned early;
    // a phase-4 alias mismatch is the one error that falls through to here (m_errors set), so re-derive the
    // state from m_errors rather than assuming success.
    s.m_state = (s.m_errors == nullptr) ? RenderGraphState::Compiled : RenderGraphState::Failed;
}

// pass_name_at / texel_bytes / texture_bytes are declared in RenderGraph_internal.h and defined above in
// the RG::Internal block: pure helpers shared with the debug tooling (the ImGui graph panel consumes them).

// layer count of a texture node: 2D array depth, or 1 for 1D/3D. A 3D volume's depthOrArrayLayers is
// depth, addressed by the view as a single 3D layer, not an array.
static uint32_t node_layers(const ResourceNode* r)
{
    return (r->dimension == WGPUTextureDimension_2D) ? (r->resolved.depthOrArrayLayers ? r->resolved.depthOrArrayLayers : 1) : 1;
}

// the subresource-view cache owning a pooled physical texture (a persistent ping-pong slot, or a transient
// / aliased object). looked up by handle so the pools' Entry-vector reallocs never dangle a stored pointer;
// null for a texture the pools don't own (imported)
static SubviewCache* find_subcache(GraphAllocator* a, WGPUTexture tex)
{
    if (!tex)
        return nullptr;
    for (TransientResourcePool::Entry& e : a->transient.entries)
        if (!e.isBuffer && e.tex == tex)
            return &e.sub;
    for (PersistentResourcePool::Entry& e : a->pool.entries)
        for (uint32_t i = 0; i < PersistentResourcePool::kLayers; ++i)
            if (e.tex[i] == tex)
                return &e.sub[i];
    return nullptr;
}

// the view descriptor one access wants on node r: the declared ViewRange (dimension/counts/aspect/format)
// with a 0 count meaning "all remaining from base" and an Undefined dimension inferred (3D volume -> 3D,
// multi-layer -> 2DArray, else 2D). Cube/CubeArray must be declared explicitly (a texture is created as a
// 2D array; cube-ness is a view property the graph can't infer).
static WGPUTextureViewDescriptor view_desc_for(const ResourceNode* r, const ResourceAccess& a)
{
    uint32_t layers = node_layers(r);
    uint32_t mipCount = a.mipCount ? a.mipCount : (r->mipLevelCount - a.baseMip);
    uint32_t layerCount = a.layerCount ? a.layerCount : (layers - a.baseLayer);
    WGPUTextureViewDimension dim = a.viewDim;
    if (dim == WGPUTextureViewDimension_Undefined)
        dim = (r->dimension == WGPUTextureDimension_3D) ? WGPUTextureViewDimension_3D
            : (layerCount > 1)                          ? WGPUTextureViewDimension_2DArray
                                                        : WGPUTextureViewDimension_2D;
    return WGPUTextureViewDescriptor {
        .format = a.viewFormat ? a.viewFormat : r->format,
        .dimension = dim,
        .baseMipLevel = a.baseMip,
        .mipLevelCount = mipCount,
        .baseArrayLayer = a.baseLayer,
        .arrayLayerCount = layerCount,
        .aspect = a.viewAspect,
    };
}

// true when `d` equals the texture's default full view, what wgpuTextureCreateView(tex, nullptr) yields;
// then the node's cached r->view is it, with no build and no pooled subview. covers whole-2D, whole-3D, and
// full-array reads, e.g. a texture sampled as its full 2D-array is just the default view.
static bool is_full_view(const ResourceNode* r, const WGPUTextureViewDescriptor& d)
{
    uint32_t layers = node_layers(r);
    WGPUTextureViewDimension full = (r->dimension == WGPUTextureDimension_1D) ? WGPUTextureViewDimension_1D
        : (r->dimension == WGPUTextureDimension_3D)                           ? WGPUTextureViewDimension_3D
        : (layers > 1)                                                        ? WGPUTextureViewDimension_2DArray
                                                                              : WGPUTextureViewDimension_2D;
    return d.baseMipLevel == 0 && d.mipLevelCount == r->mipLevelCount && d.baseArrayLayer == 0 && d.arrayLayerCount == layers
        && d.aspect == WGPUTextureAspect_All && d.format == r->format && d.dimension == full;
}

// resolve the view for `desc` on node r. full view -> cached r->view. otherwise the pooled subview cache,
// which persists across frames with its physical texture (built once, not per pass per frame). imported /
// uncached fall back to a per-pass scratch view (freed after the pass). shared by ctx.view() + attach_view.
static WGPUTextureView resolve_view(RenderGraphStorage& s, ResourceNode* r, const WGPUTextureViewDescriptor& desc)
{
    if (!r->texture) { // imported: the caller-owned view spans the whole resource; a subresource pick is ignored
        Q_ASSERT(desc.baseMipLevel == 0 && desc.baseArrayLayer == 0 && "subresource selection on an imported texture is ignored (caller owns the view)");
        Q_ASSERT(desc.aspect == WGPUTextureAspect_All && desc.format == r->format
            && "aspect/format reinterpretation on an imported texture is ignored (caller owns the view)");
        return r->view;
    }
    if (is_full_view(r, desc))
        return r->view;
    if (r->subviews)
        return r->subviews->get_or_create(r->texture, desc);
    if (s.viewScratchN == s.viewScratchCap) { // grow: a body may build more scratch views than it has accesses
        uint32_t newCap = s.viewScratchCap ? s.viewScratchCap * 2 : 8;
        WGPUTextureView* grown = s.m_allocator->scratch.alloc<WGPUTextureView>(newCap);
        if (s.viewScratchN)
            std::memcpy(grown, s.viewScratch, (size_t)s.viewScratchN * sizeof(WGPUTextureView));
        s.viewScratch = grown;
        s.viewScratchCap = newCap;
    }
    return s.viewScratch[s.viewScratchN++] = wgpuTextureCreateView(r->texture, &desc);
}

static const ResourceAccess* find_pass_access(const PassNode* pass, ResourceHandle h)
{
    for (uint32_t i = 0; i < pass->accessCount; ++i)
        if (pass->accesses[i].handle.id == h.id)
            return &pass->accesses[i];
    return nullptr;
}

// find the one CopySrc or CopyDst access on h in `pass`; assert not-found / ambiguous. wantDst
// selects the direction so a same-texture copy() (h declared as both src at one mip/layer and dst
// at another) resolves unambiguously per call.
static const ResourceAccess* find_copy_access(PassNode* pass, ResourceHandle h, bool wantDst, const char* who)
{
    const ResourceAccess* found = nullptr;
    AccessType want = wantDst ? AccessType::CopyDst : AccessType::CopySrc;
    for (uint32_t i = 0; i < pass->accessCount; ++i) {
        const ResourceAccess& a = pass->accesses[i];
        if (a.handle.id != h.id || a.type != want)
            continue;
        Q_ASSERT(!found && "ctx: two copy() declarations pair the same handle+direction in this pass; ambiguous subresource");
        found = &a;
    }
    Q_ASSERT(found && who);
    return found;
}

static uint32_t mip_dim(uint32_t v, uint32_t mip)
{
    v >>= mip;
    return v ? v : 1;
}

static WGPUExtent3D copy_extent_for(RenderGraph* graph, PassNode* pass, ResourceHandle h, bool wantDst)
{
    Q_ASSERT(h.kind == ResourceKind::Texture);
    ResourceNode* r = find_node(graph, h);
    Q_ASSERT(r != nullptr && "failed to find node! Handle is 0 or from another graph");
    const ResourceAccess* a = find_copy_access(pass, h, wantDst, "copy_extent: no matching copy() declared with this handle in this pass");
    return WGPUExtent3D {
        mip_dim(r->resolved.width, a->baseMip), mip_dim(r->resolved.height, a->baseMip), r->resolved.depthOrArrayLayers ? r->resolved.depthOrArrayLayers : 1
    };
}

bool PassContext::history_valid(ResourceHandle h) const
{
    ResourceNode* r = find_node(graph, h);
    Q_ASSERT(r != nullptr && "failed to find node! Handle is 0 or from another graph");
    if (!r)
        return false;
    Q_ASSERT(r->persistent && "ctx: can only be called on persistent resources");
    return r->historyValid;
}

WGPUTextureView PassContext::view(ResourceHandle h) const
{
    Q_ASSERT(h.id != 0);
    Q_ASSERT(h.kind == ResourceKind::Texture && "ctx.view: only textures can create a view");
    Q_ASSERT(pass); // cannot be nullptr
    ResourceNode* r = find_node(graph, h);
    Q_ASSERT(r != nullptr && "failed to find node! Handle is 0 or from another graph");
    if (!r) {
        return {};
    } // unknown / default handle: bind nothing
    Q_ASSERT(find_pass_access(pass, h) && "ctx.view: resource not declared as an access in this pass's setup");
    if (!r->texture)
        return r->view; // imported: the caller-registered view (e.g. swapchain)

    // the view shape is inferred from this pass's access on the handle: baseMip/layer plus the declared
    // ViewRange. Prefer the read access, since ctx.view() binds a texture to sample or read, and a pass
    // may read one subresource while writing another. Fall back to a write for a write-only binding, a
    // storage_write target bound via ctx.view. Two differing reads are ambiguous.
    const ResourceAccess* rd = nullptr; // read access (drives the sampled/storage-read view)
    const ResourceAccess* wr = nullptr; // write fallback (storage_write target, no read in this pass)
    for (uint32_t i = 0; i < pass->accessCount; ++i) {
        const ResourceAccess& a = pass->accesses[i];
        if (a.handle.id != h.id)
            continue;
        if (!access_is_write(a.type)) {
            Q_ASSERT(!rd && "ctx.view: two reads of one handle in a pass; ambiguous subresource; use ctx.texture");
            rd = &a;
        } else {
            wr = &a; // last write wins; the StorageRead+StorageWrite RMW pair's write matches its read anyway
        }
    }
    const ResourceAccess* acc = rd ? rd : wr;
    if (!acc)
        return r->view; // undeclared (asserted above); release falls back to the full view
    RenderGraphStorage& s = *storage(graph);
    Q_ASSERT(acc->baseMip < r->mipLevelCount && "ctx.view: baseMip past the texture's mip count");
    Q_ASSERT(acc->baseLayer < node_layers(r) && "ctx.view: baseLayer past the texture's layer count");
    Q_ASSERT((!acc->mipCount || acc->baseMip + acc->mipCount <= r->mipLevelCount) && "ctx.view: declared mip range ends past the texture's mip count");
    Q_ASSERT((!acc->layerCount || acc->baseLayer + acc->layerCount <= node_layers(r)) && "ctx.view: declared layer range ends past the texture's layer count");
    return resolve_view(s, r, view_desc_for(r, *acc));
}

// true when (mip, layer) is inside a range some access of this pass declared on h. assert-only.
static bool subres_declared(const PassNode* pass, const ResourceNode* r, ResourceHandle h, uint32_t mip, uint32_t layer)
{
    for (uint32_t i = 0; i < pass->accessCount; ++i) {
        const ResourceAccess& a = pass->accesses[i];
        if (a.handle.id != h.id)
            continue;
        uint32_t mipEnd = a.mipCount ? a.baseMip + a.mipCount : r->mipLevelCount;
        uint32_t layerEnd = a.layerCount ? a.baseLayer + a.layerCount : node_layers(r);
        if (mip >= a.baseMip && mip < mipEnd && layer >= a.baseLayer && layer < layerEnd)
            return true;
    }
    return false;
}

WGPUTextureView PassContext::view(ResourceHandle h, uint32_t mip, uint32_t layer) const
{
    Q_ASSERT(h.id != 0);
    Q_ASSERT(h.kind == ResourceKind::Texture && "ctx.view: only textures can create a view");
    Q_ASSERT(pass); // cannot be nullptr
    ResourceNode* r = find_node(graph, h);
    Q_ASSERT(r != nullptr && "failed to find node! Handle is 0 or from another graph");
    if (!r) {
        return {};
    }
    Q_ASSERT(find_pass_access(pass, h) && "ctx.view: resource not declared as an access in this pass's setup");
    if (!r->texture) { // imported: the caller-owned view spans the whole resource
        Q_ASSERT(mip == 0 && layer == 0 && "subresource selection on an imported texture is ignored (caller owns the view)");
        return r->view;
    }
    Q_ASSERT(mip < r->mipLevelCount && "ctx.view: mip past the texture's mip count");
    Q_ASSERT(layer < node_layers(r) && "ctx.view: layer past the texture's layer count");
    Q_ASSERT(subres_declared(pass, r, h, mip, layer) && "ctx.view: mip/layer outside every range this pass declared on the resource");
    WGPUTextureViewDescriptor desc {
        .format = r->format,
        .dimension = (r->dimension == WGPUTextureDimension_3D) ? WGPUTextureViewDimension_3D : WGPUTextureViewDimension_2D,
        .baseMipLevel = mip,
        .mipLevelCount = 1,
        .baseArrayLayer = layer,
        .arrayLayerCount = 1,
        .aspect = WGPUTextureAspect_All,
    };
    return resolve_view(*storage(graph), r, desc);
}

WGPUTexture PassContext::texture(ResourceHandle h) const
{
    Q_ASSERT(h.id != 0);
    Q_ASSERT(h.kind == ResourceKind::Texture);
    ResourceNode* r = find_node(graph, h);
    Q_ASSERT(r != nullptr && "failed to find node! Handle is 0 or from another graph");
    if (!r) {
        return {};
    }
    Q_ASSERT(find_pass_access(pass, h) && "ctx.texture: resource not declared as an access in this pass's setup");
    return r->texture; // null for an imported texture, sampled via ctx.view
}

WGPUBuffer PassContext::buffer(ResourceHandle h) const
{
    Q_ASSERT(h.id != 0);
    Q_ASSERT(h.kind == ResourceKind::Buffer);
    ResourceNode* r = find_node(graph, h);
    Q_ASSERT(r != nullptr && "failed to find node! Handle is 0 or from another graph");
    if (!r) {
        return {};
    }
    Q_ASSERT(find_pass_access(pass, h) && "ctx.buffer: resource not declared as an access in this pass's setup");
    Q_ASSERT(r->buffer && "ctx.buffer(): resource declared but never realized (no live writer)");
    return r->buffer;
}

WGPUExtent3D PassContext::texture_size(ResourceHandle h) const
{
    Q_ASSERT(h.id != 0);
    Q_ASSERT(h.kind == ResourceKind::Texture);
    ResourceNode* r = find_node(graph, h);
    Q_ASSERT(r != nullptr && "failed to find node! Handle is 0 or from another graph");
    if (!r) {
        return {};
    }
    return r->resolved; // compile() phase 3 resolved every live resource
}

WGPUExtent3D PassContext::texture_size(ResourceHandle h, uint32_t mip) const
{
    Q_ASSERT(h.id != 0);
    Q_ASSERT(h.kind == ResourceKind::Texture);
    ResourceNode* r = find_node(graph, h);
    Q_ASSERT(r != nullptr && "failed to find node! Handle is 0 or from another graph");
    if (!r) {
        return {};
    }
    Q_ASSERT(mip < (r->mipLevelCount ? r->mipLevelCount : 1) && "ctx.texture_size: mip past the texture's mip count");
    uint32_t d = r->resolved.depthOrArrayLayers ? r->resolved.depthOrArrayLayers : 1;
    if (r->dimension == WGPUTextureDimension_3D)
        d = mip_dim(d, mip); // depth shifts, array layers don't
    return WGPUExtent3D { mip_dim(r->resolved.width, mip), mip_dim(r->resolved.height, mip), d };
}

WGPUTextureFormat PassContext::format(ResourceHandle h) const
{
    Q_ASSERT(h.id != 0);
    Q_ASSERT(h.kind == ResourceKind::Texture);
    ResourceNode* r = find_node(graph, h);
    Q_ASSERT(r != nullptr && "failed to find node! Handle is 0 or from another graph");
    if (!r) {
        return WGPUTextureFormat_Undefined;
    }
    Q_ASSERT(r->format != WGPUTextureFormat_Undefined && "ctx.format: imported texture without a format; pass it to importe_texture");
    return r->format;
}

uint32_t PassContext::mip_count(ResourceHandle h) const
{
    Q_ASSERT(h.id != 0);
    Q_ASSERT(h.kind == ResourceKind::Texture);
    ResourceNode* r = find_node(graph, h);
    Q_ASSERT(r != nullptr && "failed to find node! Handle is 0 or from another graph");
    if (!r) {
        return 0;
    }
    Q_ASSERT(r->mipLevelCount && "ctx.mip_count: unknown for an imported texture");
    return r->mipLevelCount;
}

uint32_t PassContext::sample_count(ResourceHandle h) const
{
    Q_ASSERT(h.id != 0);
    Q_ASSERT(h.kind == ResourceKind::Texture);
    ResourceNode* r = find_node(graph, h);
    Q_ASSERT(r != nullptr && "failed to find node! Handle is 0 or from another graph");
    if (!r) {
        return 0;
    }
    Q_ASSERT(r->sampleCount && "ctx.sample_count: unknown for an imported texture");
    return r->sampleCount;
}

uint64_t PassContext::buffer_size(ResourceHandle h) const
{
    Q_ASSERT(h.id != 0);
    Q_ASSERT(h.kind == ResourceKind::Buffer);
    ResourceNode* r = find_node(graph, h);
    Q_ASSERT(r != nullptr && "failed to find node! Handle is 0 or from another graph");
    if (!r) {
        return {};
    }
    return r->bufferSize;
}

static WGPUTexelCopyTextureInfo copy_texture_info(
    RenderGraph* graph, PassNode* pass, ResourceHandle h, bool wantDst, WGPUOrigin3D origin, WGPUTextureAspect aspect)
{
    Q_ASSERT(h.id != 0);
    Q_ASSERT(h.kind == ResourceKind::Texture);
    ResourceNode* r = find_node(graph, h);
    Q_ASSERT(r != nullptr && "failed to find node! Handle is 0 or from another graph");
    Q_ASSERT(r->texture
        && "copy_*_info: imported texture has no backing WGPUTexture; importe_texture "
           "only registers a view, so copy via a render-pass blit instead");
    const ResourceAccess* a = find_copy_access(pass,
        h,
        wantDst,
        wantDst ? "copy_dst_info: no copy() with this handle as dst declared in this pass"
                : "copy_src_info: no copy() with this handle as src declared in this pass");
    Q_ASSERT(a->baseMip < r->mipLevelCount && "copy_*_info: baseMip past the texture's mip count");
    Q_ASSERT(a->baseLayer < node_layers(r) && "copy_*_info: baseLayer past the texture's layer count");
    origin.z += a->baseLayer;
    return WGPUTexelCopyTextureInfo { .texture = r->texture, .mipLevel = a->baseMip, .origin = origin, .aspect = aspect };
}

WGPUTexelCopyTextureInfo PassContext::copy_src_info(ResourceHandle h, WGPUOrigin3D origin, WGPUTextureAspect aspect) const
{
    return copy_texture_info(graph, pass, h, /*wantDst=*/false, origin, aspect);
}

WGPUTexelCopyTextureInfo PassContext::copy_dst_info(ResourceHandle h, WGPUOrigin3D origin, WGPUTextureAspect aspect) const
{
    return copy_texture_info(graph, pass, h, /*wantDst=*/true, origin, aspect);
}

WGPUExtent3D PassContext::copy_extent_src(ResourceHandle h) const { return copy_extent_for(graph, pass, h, /*wantDst=*/false); }

WGPUExtent3D PassContext::copy_extent_dst(ResourceHandle h) const { return copy_extent_for(graph, pass, h, /*wantDst=*/true); }

WGPUTexelCopyBufferInfo PassContext::copy_src_buffer(ResourceHandle h, WGPUTexelCopyBufferLayout layout) const
{
    Q_ASSERT(h.id != 0);
    Q_ASSERT(h.kind == ResourceKind::Buffer);
    find_copy_access(pass, h, /*wantDst=*/false, "copy_src_buffer: no copy() with this handle as src declared in this pass");
    return WGPUTexelCopyBufferInfo { .layout = layout, .buffer = buffer(h) };
}

WGPUTexelCopyBufferInfo PassContext::copy_dst_buffer(ResourceHandle h, WGPUTexelCopyBufferLayout layout) const
{
    Q_ASSERT(h.id != 0);
    Q_ASSERT(h.kind == ResourceKind::Buffer);
    find_copy_access(pass, h, /*wantDst=*/true, "copy_dst_buffer: no copy() with this handle as dst declared in this pass");
    return WGPUTexelCopyBufferInfo { .layout = layout, .buffer = buffer(h) };
}

static void release_resources(RenderGraph* rg); // internal helper; definition below is static too

// create the GPU resources compile() worked out (size in `resolved`, usage in tex/bufUsage).
// imported resources are caller-owned and skipped; a resource with no accumulated usage was
// untouched by a live pass -> skipped too (the free dead-resource cull compile() phase 3 set up).
static void realize_graph(RenderGraph* rg, WGPUDevice device)
{
    auto t0 = std::chrono::steady_clock::now();
    RenderGraphStorage& s = *storage(rg);
    if (s.m_state == RenderGraphState::Failed)
        return; // poisoned graph: no-op (execute() also guards, this is defensive)
    Q_ASSERT(s.m_state == RenderGraphState::Compiled);

    // inferred transient attachments get the memoryless usage bit only where the platform supports it
#ifdef __EMSCRIPTEN__
    const bool transientOK = rg_web_has_transient_attachment() != 0;
#else
    const bool transientOK = wgpuDeviceHasFeature(device, kTransientAttachmentsFeature);
#endif
    s.m_allocator->transientFeatureOn = transientOK;

    PersistentResourcePool& pool = s.m_allocator->pool;
    TransientResourcePool& transient = s.m_allocator->transient;

    auto pool_used = [](const ResourceNode* r) { return r->kind == ResourceKind::Texture ? r->texUsage != 0 : r->bufUsage != 0; };

    // touch + union usage into the entry (touch rotates once per used frame; sibling layer nodes no-op).
    // curr and prev must be co-used; culling the writer but keeping the reader would rotate on the reader
    // alone and surface a never-written slot. compile() rejects that asymmetry (see validate_history_layers).
    for (ResourceNode* r = s.m_resouces; r; r = r->next) {
        if (!r->persistent || !pool_used(r))
            continue;
        PersistentResourcePool::Entry* e = pool.touch(r->id, r->history ? PersistentResourcePool::kLayers : 1, r->historyHash, r->kind == ResourceKind::Buffer);
        if (r->kind == ResourceKind::Texture)
            e->usage |= r->texUsage;
        else
            e->bufUsage |= r->bufUsage;
    }

    // realize + point each layer node at its rotated slot
    for (ResourceNode* r = s.m_resouces; r; r = r->next) {
        if (!r->persistent || !pool_used(r))
            continue;
        PersistentResourcePool::Entry* e = pool.find(r->id); // touched above -> non-null
        Q_ASSERT(e && "used persistent node must have a pool entry from the touch pass above");
        uint32_t sl = pool.slot(*e, r->historyIndex);
        if (r->kind == ResourceKind::Texture) {
            pool.realize_entry(e, device, r->resolved, r->format, r->dimension, r->mipLevelCount, r->sampleCount);
            r->texture = e->tex[sl];
            r->view = e->view[sl];
        } else {
            pool.realize_buffer_entry(e, device, r->bufferSize);
            r->buffer = e->buf[sl];
        }
        // .prev holds valid history only if the entry both survived from a prior frame and not recreated this
        // fram and was actually used last frame.
        r->historyValid = r->history && (e->createdClock != pool.evictClock) && (e->prevTouched == pool.evictClock - 1);
    }

    // aliasing (compile() phase 4): acquire one pooled object per physical slot with the union usage, then
    // point every member resource at it. m_slotCount == 0 when aliasing is off -> this is skipped and the
    // per-resource loops below realize each transient on its own. one acquire per slot (not per member)
    // keeps the claim count right: distinct slots get distinct objects, members share. a slot is a texture
    // or a buffer per ph.kind (phase 4 never mixes them).
    for (uint32_t i = 0; i < s.m_slotCount; ++i) {
        PhysicalResource& ph = s.m_slots[i];
        if (ph.kind == ResourceKind::Texture)
            transient.acquire(device, ph.size, ph.format, ph.dimension, 1, 1, ph.texUsage, ph.texture, ph.view);
        else
            transient.acquire(device, ph.bufferSize, ph.bufUsage, ph.buffer);
    }
    for (ResourceNode* r = s.m_resouces; r; r = r->next)
        if (r->aliasSlot != ResourceNode::kNoSlot) {
            PhysicalResource& ph = s.m_slots[r->aliasSlot];
            r->texture = ph.texture;
            r->view = ph.view;
            r->buffer = ph.buffer;
        }

    for (ResourceNode* r = s.m_resouces; r; r = r->next) {
        // graph-owned per-frame TEXTURES not on an alias slot: reuse a pooled texture matching this
        // descriptor; release_resources() leaves it alone and the pool recycles/evicts at end_frame().
        if (r->is_external() || r->kind != ResourceKind::Texture || !r->texUsage || r->aliasSlot != ResourceNode::kNoSlot)
            continue;
        // add the memoryless hint for an inferred transient attachment (usage stays exactly
        // TRANSIENT|RENDER_ATTACHMENT, as the proposal requires); the pool keys on usage, so these don't
        // collide with ordinary render targets.
        WGPUTextureUsage usage = r->texUsage;
        if (transientOK && r->transientAttachment)
            usage |= WGPUTextureUsage_TransientAttachment;
        transient.acquire(device, r->resolved, r->format, r->dimension, r->mipLevelCount, r->sampleCount, usage, r->texture, r->view);
    }
    for (ResourceNode* r = s.m_resouces; r; r = r->next) {
        // graph-owned per-frame BUFFERS not on an alias slot.
        if (r->is_external() || r->kind != ResourceKind::Buffer || !r->bufUsage || r->aliasSlot != ResourceNode::kNoSlot)
            continue;
        transient.acquire(device, r->bufferSize, r->bufUsage, r->buffer);
    }

    // point each texture node at the subview cache of its now-realized physical texture, so ctx.view() reuses
    // subresource views across frames. done after every acquire (the pools' Entry vectors are stable from
    // here to execute()); by handle, so a pool vector-realloc during the acquires above never dangles it.
    for (ResourceNode* r = s.m_resouces; r; r = r->next)
        if (r->kind == ResourceKind::Texture && r->texture)
            r->subviews = find_subcache(s.m_allocator, r->texture);

    auto t1 = std::chrono::steady_clock::now();
    s.timing_realize_us = std::chrono::duration<float, std::micro>(t1 - t0).count();
}

// record the compiled passes, already in execution order, into a caller-owned encoder: open the
// right pass kind, wire the attachments declared in setup, invoke the stored body against a live
// PassContext. caller owns submit and present.
void RenderGraph::execute(WGPUDevice device, WGPUCommandEncoder encoder, WGPUQueue queue, bool enableProfiling)
{
    RenderGraphStorage& s = *storage(this);
    if (s.m_state != RenderGraphState::Compiled) {
        // Failed (poisoned) is a legit no-op: the caller drains getErrors(). Recording/Finished means
        // compile() was skipped or execute() ran twice, so assert loudly and no-op in release rather
        // than record a frame against unresolved sizes or spent state.
        Q_ASSERT(s.m_state == RenderGraphState::Failed && "execute(): graph not compiled; compile() missing, or execute() called twice");
        return;
    }

    ScopedScratch scratch(s.m_allocator->scratch);
    s.viewScratchCap = 8; // initial; resolve_view doubles it on demand
    s.viewScratch = scratch.alloc<WGPUTextureView>(s.viewScratchCap);

    realize_graph(this, device);

    auto t0 = std::chrono::steady_clock::now();

    // opt-in GPU timing: lazily build the query set/buffers, then grab a free read-back ring slot. If every
    // slot is still awaiting its map callback, skip profiling this frame rather than stall. `pi` is a dense
    // begin/end pair index advanced only for passes that actually run -> the resolve range stays contiguous.
    GpuProfiler& prof = s.m_allocator->profiler;
    bool profiling = enableProfiling;
    int slotIdx = -1;
    // one pendingSlot per allocator -> one profiled graph per frame. An earlier graph already
    // filled it; run this one unprofiled instead of clobbering its results.
    if (profiling && prof.pendingSlot != -1) {
        qDebug("[RenderGraph] profiling skipped for this graph: an earlier graph this frame is already profiled\n");
        profiling = false;
    }
    if (profiling) {
        prof.init(device);
        // disable profiling when no free slot is found
        slotIdx = prof.free_slot();
        if (slotIdx < 0)
            profiling = false;
    }
    GpuProfiler::Slot* slot = profiling ? &prof.ring[slotIdx] : nullptr;
    uint32_t pi = 0;

    // bracket each contiguous run of same-prefix passes in an encoder debug group, so a RenderDoc/PIX
    // capture shows collapsible regions over the per-pass labels. push/pop happen between passes, with
    // no pass open at that point, so they are balanced and nested by construction.
    WGPUStringView openGroup {};
    for (PassNode* p = s.m_passes; p; p = p->next) {
        Q_ASSERT(p->kind != PassKind::None && "a pass of kind None is not allowed");
        WGPUStringView grp = group_prefix(p->id.name);
        if (!sv_eq(grp, openGroup)) {
            if (sv_length(openGroup))
                wgpuCommandEncoderPopDebugGroup(encoder);
            if (sv_length(grp))
                wgpuCommandEncoderPushDebugGroup(encoder, grp);
            openGroup = grp;
        }

        // initialize() pass that survived the cull == it (re)bakes this frame: mark each target's pool entry
        // baked + stamp its settings hash, so next frame's compile() skips the bake until the hash changes
        // (or the entry is recreated, which clears `baked`). recorded here, not in compile(), so a frame that
        // fails compile never claims a bake it didn't run; gated on exec_fn so a body-less pass can't claim
        // one either. skipped (already-baked) init passes were culled -> not in this loop.
        if (p->initCount && p->exec_fn)
            for (uint32_t i = 0; i < p->initCount; ++i)
                if (ResourceNode* t = find_node(this, p->initTargets[i].target))
                    if (PersistentResourcePool::Entry* e = s.m_allocator->pool.find(t->id)) {
                        e->initHash = p->initTargets[i].hash;
                        e->baked = true;
                    }

        PassContext ctx {};
        ctx.encoder = encoder;
        ctx.graph = this;
        ctx.queue = queue;
        ctx.pass = p;
        s.viewScratchN = 0; // per-pass: attachment + body ctx.view() views accumulate here, freed after the body

        // time this pass iff profiling is live, the pass records a body, and there is a free pair in the
        // set. begin/end ride the pass descriptor (render/compute); transfer passes only when
        // RG_TIME_TRANSFER_PASSES is defined, using the off-spec encoder timestamps in the transfer branch below.
#ifdef RG_TIME_TRANSFER_PASSES
        constexpr bool kTimeTransfer = true;
#else
        constexpr bool kTimeTransfer = false;
#endif
        const bool timeThis = profiling && p->exec_fn && pi < GpuProfiler::kMaxPasses && (kTimeTransfer || p->kind != PassKind::Transfer);
        WGPUPassTimestampWrites tw { .querySet = prof.querySet, .beginningOfPassWriteIndex = 2 * pi, .endOfPassWriteIndex = 2 * pi + 1 };
        if (timeThis)
            GpuProfiler::copy_name(slot->names[pi], p->id.name);

        if (p->kind == PassKind::Compute && p->exec_fn) {
            WGPUComputePassDescriptor cd { .label = p->id.name, .timestampWrites = timeThis ? &tw : nullptr };
            ctx.compute = wgpuCommandEncoderBeginComputePass(encoder, &cd);
            p->exec_fn(p->exec_obj, ctx);
            wgpuComputePassEncoderEnd(ctx.compute);
            wgpuComputePassEncoderRelease(ctx.compute);
        } else if (p->kind == PassKind::Graphics && p->exec_fn) {
            // gather declared attachments from the access list -> WebGPU render pass descriptor
            WGPURenderPassColorAttachment color[kMaxColorAttachments] {};
            uint32_t nc = 0;
            uint32_t lastColorSlot = ~0u; // slot of the most recent color() -> a following resolve() patches its resolveTarget
            WGPURenderPassDepthStencilAttachment depth {};
            bool hasDepth = false;

            // an attachment view is exactly one subresource (one mip, one layer): a render target can't span
            // a mip chain or array. an attachment access carries no ViewRange, so view_desc_for yields the
            // single (baseMip, baseLayer) 2D slice; resolve_view returns the node's full view when that IS the
            // whole texture, else the pooled subview cache (reused across frames), else a per-pass scratch
            // view. imported textures (r->texture null, e.g. swapchain) keep their caller-owned registered view.
            auto attach_view = [&](ResourceNode* r, const ResourceAccess& a) -> WGPUTextureView {
                Q_ASSERT(a.baseMip < r->mipLevelCount && "attachment baseMip past the texture's mip count");
                Q_ASSERT((!r->texture || a.baseLayer < node_layers(r)) && "attachment baseLayer past the texture's layer count");
                return resolve_view(s, r, view_desc_for(r, a));
            };

            for (uint32_t i = 0; i < p->accessCount; ++i) {
                const ResourceAccess& a = p->accesses[i];
                ResourceNode* r = find_node(this, a.handle);
                if (!r)
                    continue;
                if (a.type == AccessType::ColorAttachment && nc < kMaxColorAttachments) { // color() asserted; release backstop
                    color[nc] = WGPURenderPassColorAttachment {
                        .view = attach_view(r, a),
                        .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
                        .resolveTarget = nullptr, // patched below if a resolve() in this pass targets this slot
                        .loadOp = a.loadOp,
                        .storeOp = a.storeOp,
                        .clearValue = a.clearColor,
                    };
                    lastColorSlot = nc++;
                } else if (a.type == AccessType::ResolveAttachment) {
                    if (lastColorSlot != ~0u)
                        color[lastColorSlot].resolveTarget = attach_view(r, a);
                } else if (a.type == AccessType::DepthStencilAttachment || a.type == AccessType::DepthStencilReadOnly) {
                    depth = WGPURenderPassDepthStencilAttachment {
                        .view = attach_view(r, a),
                        .depthLoadOp = a.loadOp,
                        .depthStoreOp = a.storeOp,
                        .depthClearValue = a.clearDepth,
                        .depthReadOnly = a.type == AccessType::DepthStencilReadOnly,
                        .stencilLoadOp = a.stencilLoadOp,
                        .stencilStoreOp = a.stencilStoreOp,
                        .stencilClearValue = a.stencilClear,
                        .stencilReadOnly = a.type == AccessType::DepthStencilReadOnly,
                    };
                    hasDepth = true;
                }
            }

            WGPURenderPassDescriptor rd {
                .label = p->id.name,
                .colorAttachmentCount = nc,
                .colorAttachments = color,
                .depthStencilAttachment = hasDepth ? &depth : nullptr,
                .timestampWrites = timeThis ? &tw : nullptr,
            };
            ctx.render = wgpuCommandEncoderBeginRenderPass(encoder, &rd);
            p->exec_fn(p->exec_obj, ctx);
            wgpuRenderPassEncoderEnd(ctx.render);
            wgpuRenderPassEncoderRelease(ctx.render);
        } else { // Transfer: body records straight onto the encoder. no pass object -> bracket with encoder timestamps.
            if (p->exec_fn) {
#ifdef RG_TIME_TRANSFER_PASSES // off-spec: the device must be created with the allow_unsafe_apis toggle
                if (timeThis)
                    wgpuCommandEncoderWriteTimestamp(encoder, prof.querySet, 2 * pi);
#endif
                p->exec_fn(p->exec_obj, ctx);
#ifdef RG_TIME_TRANSFER_PASSES
                if (timeThis)
                    wgpuCommandEncoderWriteTimestamp(encoder, prof.querySet, 2 * pi + 1);
#endif
            }
        }

        if (timeThis)
            ++pi;

        // pass-scoped subresource views, the attachments built above plus any the body built via ctx.view().
        // Free them now the pass has ended. Runs for every kind, so ctx.view() auto-releases in compute/transfer too.
        for (uint32_t i = 0; i < s.viewScratchN; ++i)
            wgpuTextureViewRelease(s.viewScratch[i]);
    }
    if (sv_length(openGroup))
        wgpuCommandEncoderPopDebugGroup(encoder); // close the last open group

    // resolve the queries we wrote into the staging buffer, then stage them into the chosen ring slot. The
    // copy rides this frame's submit; collect_gpu_timings() kicks the async map once that submit is queued.
    if (profiling && pi > 0) {
        wgpuCommandEncoderResolveQuerySet(encoder, prof.querySet, 0, 2 * pi, prof.resolveBuf, 0);
        wgpuCommandEncoderCopyBufferToBuffer(encoder, prof.resolveBuf, 0, slot->buf, 0, uint64_t(pi) * 2 * sizeof(uint64_t));
        slot->count = pi;
        prof.pendingSlot = slotIdx;
    }

    s.m_state = RenderGraphState::Finished;
    release_resources(this);
    auto t1 = std::chrono::steady_clock::now();
    s.timing_execute_us = std::chrono::duration<float, std::micro>(t1 - t0).count();
}

// kick the async read-back of the slot execute() just filled. Call after the frame's queue submit, since
// the copy must be queued first. The spontaneous callback fires during a later instance.ProcessEvents()
// pump, a couple frames on when the GPU is done, and unpacks ns pairs into the profiler's results.
void RenderGraph::collect_gpu_timings()
{
    GpuProfiler& prof = storage(this)->m_allocator->profiler;
    if (prof.pendingSlot < 0)
        return;
    GpuProfiler::Slot* slot = &prof.ring[prof.pendingSlot];
    prof.pendingSlot = -1;
    slot->pending = true;

    WGPUBufferMapCallbackInfo cb {};
    cb.mode = WGPUCallbackMode_AllowSpontaneous;
    cb.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* u1, void* u2) {
        auto* slot = static_cast<GpuProfiler::Slot*>(u1);
        auto* prof = static_cast<GpuProfiler*>(u2);
        if (status == WGPUMapAsyncStatus_Success) {
            const auto* t = static_cast<const uint64_t*>(wgpuBufferGetConstMappedRange(slot->buf, 0, slot->count * 2 * sizeof(uint64_t)));
            if (t) {
                prof->resultCount = slot->count;
                for (uint32_t i = 0; i < slot->count; ++i) {
                    const uint64_t b = t[2 * i], e = t[2 * i + 1];
                    prof->resultUs[i] = (e > b) ? float((e - b) / 1000.0) : 0.0f; // ns -> us; clamp reorders
                    std::memcpy(prof->resultNames[i], slot->names[i], GpuProfiler::kMaxName);
                }
                ++prof->resultId; // signal the history sampler that a fresh read-back landed
            }
            wgpuBufferUnmap(slot->buf);
        }
        slot->pending = false;
    };
    cb.userdata1 = slot;
    cb.userdata2 = &prof;
    wgpuBufferMapAsync(slot->buf, WGPUMapMode_Read, 0, slot->count * 2 * sizeof(uint64_t), cb);
}

ErrorMessage* RenderGraph::get_errors() { return storage(this)->m_errors; }


static void release_resources(RenderGraph* rg)
{
    RenderGraphStorage& s = *storage(rg);
    Q_ASSERT(s.m_state == RenderGraphState::Finished);
    for (ResourceNode* r = s.m_resouces; r; r = r->next) {
        if (r->is_external())
            continue;
        r->texture = nullptr;
        r->view = nullptr;
        r->buffer = nullptr;
    }

    for (uint32_t i = 0; i < s.m_slotCount; ++i) {
        s.m_slots[i].texture = nullptr;
        s.m_slots[i].view = nullptr;
        s.m_slots[i].buffer = nullptr;
    }

    s.m_allocator->transient.release_claims();
}

static void use(PassBuilder& b,
    ResourceHandle handle,
    AccessType type,
    WGPULoadOp load = WGPULoadOp_Undefined,
    WGPUStoreOp store = WGPUStoreOp_Undefined,
    WGPUColor clear = {},
    float clearDepth = {},
    uint32_t baseMip = 0,
    uint32_t baseLayer = 0,
    ViewRange range = {},
    WGPULoadOp stencilLoad = WGPULoadOp_Undefined,
    WGPUStoreOp stencilStore = WGPUStoreOp_Undefined,
    uint32_t stencilClear = 0)
{
    PassNode* pass = b.m_pass;
    Q_ASSERT(handle.id != 0);
    Q_ASSERT(handle.id < storage(b.m_graph)->next_resource_id && "stale or foreign ResourceHandle; not created by this frame's graph");
    Q_ASSERT(handle.generation == storage(b.m_graph)->generation && "stale/foreign ResourceHandle: not created by this graph");

    // WebGPU permits multiple writable-storage uses in one scope; the graph is stricter since
    // it can't synchronize two writes inside a pass; relax if a shader needs it.
    // Transfer passes are exempt: per spec each copy command is its own usage scope, so several copies may
    // touch one resource. The per-command src == dst overlap is still asserted in copy().
    const bool w = access_is_write(type);
    if (pass->kind != PassKind::Transfer)
        for (uint32_t i = 0; i < pass->accessCount; ++i) {
            if (pass->accesses[i].handle.id != handle.id)
                continue;
            if (in_pass_accesses_conflict(type,
                    baseMip,
                    range.mipCount,
                    baseLayer,
                    range.layerCount,
                    range.aspect,
                    pass->accesses[i].type,
                    pass->accesses[i].baseMip,
                    pass->accesses[i].mipCount,
                    pass->accesses[i].baseLayer,
                    pass->accesses[i].layerCount,
                    pass->accesses[i].viewAspect)) {
                qDebug("[RenderGraph] error: pass \"%.*s\" uses resource id %u %s in one pass; a "
                            "written resource must be its only use in the pass.\n",
                    (int)pass->id.name.length,
                    pass->id.name.data ? pass->id.name.data : "",
                    handle.id,
                    (w && access_is_write(pass->accesses[i].type)) ? "as more than one write (unsynchronized)" : "as both written and read");
                Q_ASSERT(false && "RenderGraph: illegal in-pass resource usage (read+write or double write in one pass)");
            }
        }

    if (pass->accessCount == pass->accessCap) {
        GraphAllocator* A = storage(b.m_graph)->m_allocator;
        uint32_t oldCap = pass->accessCap;
        uint32_t newCap = oldCap ? oldCap * 2 : 4;
        ResourceAccess* buf = nullptr;
        if (oldCap)
            buf = static_cast<ResourceAccess*>(
                A->front.extend_tail(pass->accesses, (size_t)oldCap * sizeof(ResourceAccess), (size_t)(newCap - oldCap) * sizeof(ResourceAccess)));
        if (!buf) {
            buf = A->alloc<ResourceAccess>(newCap);
            if (!buf) {
                qDebug("[RenderGraph] error: pass \"%.*s\" out of arena memory growing accesses; "
                            "access on resource id %u dropped.\n",
                    (int)pass->id.name.length,
                    pass->id.name.data ? pass->id.name.data : "",
                    handle.id);
                Q_ASSERT(false && "RenderGraph: arena OOM growing pass accesses");
                return;
            }
            if (pass->accessCount)
                std::memcpy(buf, pass->accesses, (size_t)pass->accessCount * sizeof(ResourceAccess));
        }
        pass->accesses = buf;
        pass->accessCap = newCap;
    }
    pass->accesses[pass->accessCount++] = { handle,
        type,
        load,
        store,
        clear,
        clearDepth,
        stencilLoad,
        stencilStore,
        stencilClear,
        baseMip,
        baseLayer,
        range.mipCount,
        range.layerCount,
        range.dim,
        range.aspect,
        range.format };
}

void PassBuilder::color(ResourceHandle handle, WGPULoadOp load, WGPUStoreOp store, WGPUColor clear, uint32_t baseMip, uint32_t baseLayer)
{
    Q_ASSERT(handle.id != 0);
    Q_ASSERT(handle.kind == ResourceKind::Texture);
    Q_ASSERT(m_pass->kind == PassKind::Graphics && "color attachment is only legal for graphics passes.");

    uint32_t nc = 0;
    for (uint32_t i = 0; i < m_pass->accessCount; ++i)
        if (m_pass->accesses[i].type == AccessType::ColorAttachment)
            ++nc;
    if (nc >= kMaxColorAttachments) {
        qDebug("[RenderGraph] error: pass \"%.*s\" declares more than %u color attachments "
                    "(WebGPU maxColorAttachments); attachment on resource id %u dropped at execute.\n",
            (int)m_pass->id.name.length,
            m_pass->id.name.data ? m_pass->id.name.data : "",
            kMaxColorAttachments,
            handle.id);
        Q_ASSERT(false && "RenderGraph: more than kMaxColorAttachments color() calls in one pass");
    }
    use(*this, handle, AccessType::ColorAttachment, load, store, clear, {}, baseMip, baseLayer);
}

void PassBuilder::resolve(ResourceHandle handle, uint32_t baseMip, uint32_t baseLayer)
{
    Q_ASSERT(handle.id != 0);
    Q_ASSERT(handle.kind == ResourceKind::Texture);
    Q_ASSERT(m_pass->kind == PassKind::Graphics && "resolve target is only legal for graphics passes.");

    bool pairedColor = false, doubleResolve = false;
    for (uint32_t i = m_pass->accessCount; i-- > 0;) {
        AccessType t = m_pass->accesses[i].type;
        if (t == AccessType::ColorAttachment) {
            pairedColor = true;
            break;
        }
        if (t == AccessType::ResolveAttachment) {
            doubleResolve = true;
            break;
        }
    }
    if (doubleResolve) {
        qDebug("[RenderGraph] error: pass \"%.*s\" calls resolve() twice for one color(); one "
                    "resolve target per color attachment.\n",
            (int)m_pass->id.name.length,
            m_pass->id.name.data ? m_pass->id.name.data : "");
        Q_ASSERT(false && "RenderGraph: resolve() called twice for one color() in a pass");
    } else if (!pairedColor) {
        qDebug("[RenderGraph] error: pass \"%.*s\" calls resolve() with no preceding color(); a "
                    "resolve target pairs with the color() declared just before it.\n",
            (int)m_pass->id.name.length,
            m_pass->id.name.data ? m_pass->id.name.data : "");
        Q_ASSERT(false && "RenderGraph: resolve() with no preceding color() in a pass");
    }

    use(*this, handle, AccessType::ResolveAttachment, WGPULoadOp_Undefined, WGPUStoreOp_Undefined, {}, {}, baseMip, baseLayer);
}

static void assert_single_depth(PassNode* pass)
{
    for (uint32_t i = 0; i < pass->accessCount; ++i) {
        AccessType t = pass->accesses[i].type;
        if (t == AccessType::DepthStencilAttachment || t == AccessType::DepthStencilReadOnly) {
            qDebug("[RenderGraph] error: pass \"%.*s\" declares a second depth-stencil attachment; "
                        "a render pass has one depth-stencil slot, so only the last would take effect.\n",
                (int)pass->id.name.length,
                pass->id.name.data ? pass->id.name.data : "");
            Q_ASSERT(false && "RenderGraph: two depth-stencil attachments in one pass");
        }
    }
}

void PassBuilder::depth_stencil(ResourceHandle handle,
    WGPULoadOp load,
    WGPUStoreOp store,
    float clearDepth,
    uint32_t baseMip,
    uint32_t baseLayer,
    WGPULoadOp stencilLoad,
    WGPUStoreOp stencilStore,
    uint32_t stencilClear)
{
    Q_ASSERT(handle.id != 0);
    Q_ASSERT(handle.kind == ResourceKind::Texture);
    Q_ASSERT(m_pass->kind == PassKind::Graphics && "depth-stencil attachment is only legal for graphics passes.");
    assert_single_depth(m_pass);
    use(*this, handle, AccessType::DepthStencilAttachment, load, store, {}, clearDepth, baseMip, baseLayer, {}, stencilLoad, stencilStore, stencilClear);
}

void PassBuilder::depth_stencil_read_only(ResourceHandle handle, uint32_t baseMip, uint32_t baseLayer)
{
    Q_ASSERT(handle.id != 0);
    Q_ASSERT(handle.kind == ResourceKind::Texture);
    Q_ASSERT(m_pass->kind == PassKind::Graphics && "depth-stencil attachment is only legal for graphics passes.");
    assert_single_depth(m_pass);
    use(*this, handle, AccessType::DepthStencilReadOnly, WGPULoadOp_Undefined, WGPUStoreOp_Undefined, {}, {}, baseMip, baseLayer);
}

void PassBuilder::sampled(ResourceHandle handle, uint32_t baseMip, uint32_t baseLayer, ViewRange range)
{
    Q_ASSERT(handle.id != 0);
    Q_ASSERT(handle.kind == ResourceKind::Texture);
    use(*this, handle, AccessType::Sampled, WGPULoadOp_Undefined, WGPUStoreOp_Undefined, {}, {}, baseMip, baseLayer, range);
}

void PassBuilder::storage_read(ResourceHandle handle, uint32_t baseMip, uint32_t baseLayer, ViewRange range)
{
    Q_ASSERT(handle.id != 0);
    Q_ASSERT((handle.kind == ResourceKind::Texture || (baseMip == 0 && baseLayer == 0))
        && "storage_read: baseMip/baseLayer are texture-only; a buffer handle has no subresource");
    use(*this, handle, AccessType::StorageRead, WGPULoadOp_Undefined, WGPUStoreOp_Undefined, {}, {}, baseMip, baseLayer, range);
}

void PassBuilder::storage_write(ResourceHandle handle, uint32_t baseMip, uint32_t baseLayer, ViewRange range)
{
    Q_ASSERT(handle.id != 0);
    Q_ASSERT((handle.kind == ResourceKind::Texture || (baseMip == 0 && baseLayer == 0))
        && "storage_write: baseMip/baseLayer are texture-only; a buffer handle has no subresource");
    use(*this, handle, AccessType::StorageWrite, WGPULoadOp_Undefined, WGPUStoreOp_Undefined, {}, {}, baseMip, baseLayer, range);
}

// in-place read-modify-write
void PassBuilder::storage_read_write(ResourceHandle handle, uint32_t baseMip, uint32_t baseLayer, ViewRange range)
{
    Q_ASSERT(handle.id != 0);
    storage_read(handle, baseMip, baseLayer, range);
    storage_write(handle, baseMip, baseLayer, range);
}

void PassBuilder::uniform(ResourceHandle handle)
{
    Q_ASSERT(handle.id != 0);
    Q_ASSERT(handle.kind == ResourceKind::Buffer);
    use(*this, handle, AccessType::Uniform);
}

void PassBuilder::host_write(ResourceHandle handle)
{
    Q_ASSERT(handle.id != 0);
    Q_ASSERT(handle.kind == ResourceKind::Buffer && "host_write is buffer-only");
    use(*this, handle, AccessType::CopyDst);
}

void PassBuilder::copy(ResourceHandle src, ResourceHandle dst, uint32_t srcMip, uint32_t srcLayer, uint32_t dstMip, uint32_t dstLayer)
{
    Q_ASSERT(src.id != 0 && dst.id != 0);
    Q_ASSERT(m_pass->kind == PassKind::Transfer && "copy() is only legal in Transfer passes");
    Q_ASSERT(
        (src.kind == ResourceKind::Texture || (srcMip == 0 && srcLayer == 0)) && "copy: srcMip/srcLayer are texture-only; a buffer handle has no subresource");
    Q_ASSERT(
        (dst.kind == ResourceKind::Texture || (dstMip == 0 && dstLayer == 0)) && "copy: dstMip/dstLayer are texture-only; a buffer handle has no subresource");
    Q_ASSERT(!(src.id == dst.id && src.kind == dst.kind && srcMip == dstMip && srcLayer == dstLayer) && "copy: src and dst are the same subresource");
    use(*this, src, AccessType::CopySrc, WGPULoadOp_Undefined, WGPUStoreOp_Undefined, {}, {}, srcMip, srcLayer);
    use(*this, dst, AccessType::CopyDst, WGPULoadOp_Undefined, WGPUStoreOp_Undefined, {}, {}, dstMip, dstLayer);
}

void PassBuilder::vertex_buffer(ResourceHandle handle)
{
    Q_ASSERT(handle.id != 0);
    Q_ASSERT(handle.kind == ResourceKind::Buffer);
    use(*this, handle, AccessType::Vertex);
}

void PassBuilder::index_buffer(ResourceHandle handle)
{
    Q_ASSERT(handle.id != 0);
    Q_ASSERT(handle.kind == ResourceKind::Buffer);
    use(*this, handle, AccessType::Index);
}

void PassBuilder::indirect_buffer(ResourceHandle handle)
{
    Q_ASSERT(handle.id != 0);
    Q_ASSERT(handle.kind == ResourceKind::Buffer);
    use(*this, handle, AccessType::Indirect);
}


void PassBuilder::initialize(ResourceHandle target, uint64_t hash)
{
    Q_ASSERT(target.id != 0);
    Q_ASSERT(target.id < storage(m_graph)->next_resource_id && "stale or foreign ResourceHandle; not created by this frame's graph");
    if (target.id == 0)
        return; // release-safe: a zero target would deref byId[0] in phase 0
    // one entry per target; the same target twice in one pass is an authoring slip (which hash wins?).
    for (uint32_t i = 0; i < m_pass->initCount; ++i)
        Q_ASSERT(m_pass->initTargets[i].target.id != target.id && "initialize() called twice for the same target in one pass");
    // execute() drops any target past the cap before it can bake; be loud at the offending call instead.
    if (m_pass->initCount >= PassNode::kMaxInitTargets) {
        qDebug("[RenderGraph] error: pass \"%.*s\" declares more than %u initialize() targets; "
                    "target on resource id %u dropped.\n",
            (int)m_pass->id.name.length,
            m_pass->id.name.data ? m_pass->id.name.data : "",
            PassNode::kMaxInitTargets,
            target.id);
        Q_ASSERT(false && "RenderGraph: more than kMaxInitTargets initialize() targets in one pass");
        return;
    }
    m_pass->initTargets[m_pass->initCount++] = { target, hash };
}

// keep this pass even with no in-graph reader and no imported/persistent write: mark it an extra cull root
// so compile() phase 2 never drops it (and keeps everything it depends on). not an access (records no
// hazard/usage), just a marker. for side-effect-only passes: readback, timestamp/profiling resolve,
// indirect-arg gen consumed outside the graph.
void PassBuilder::force_keep() { m_pass->forceKeep = true; }

} // namespace RG
