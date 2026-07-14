/*****************************************************************************
 * weBIGeo
 * Copyright (C) 2026 Matthias Huerbe
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#include "RenderGraphPanel.h"

#include "ImGuiManager.h"
#include <IconsFontAwesome5.h>
#include <imgui.h>

#include <cmath>
#include <cstdarg>
#include <cstdio>

#include "webgpu/base/RenderGraph_internal.h"


using namespace webgpu;
using namespace webgpu::rg;
using namespace webgpu::rg::Internal;


#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4456 4457 4100 4189 4505 4996)
#endif

// short labels/tints for the node boxes.
static const char* rg_kind_name(PassKind k)
{
    switch (k) {
    case PassKind::Graphics: return "gfx";
    case PassKind::Compute:  return "compute";
    case PassKind::Transfer: return "transfer";
    default:                 return "none";
    }
}
static ImU32 rg_kind_color(PassKind k)
{
    switch (k) {
    case PassKind::Graphics: return IM_COL32(54, 96, 156, 255);
    case PassKind::Compute:  return IM_COL32(156, 96, 40, 255);
    case PassKind::Transfer: return IM_COL32(56, 120, 70, 255);
    default:                 return IM_COL32(90, 90, 90, 255);
    }
}
static const char* rg_access_name(AccessType t)
{
    switch (t) {
    case AccessType::ColorAttachment:        return "color";
    case AccessType::DepthStencilAttachment: return "depth";
    case AccessType::DepthStencilReadOnly:   return "depth(ro)";
    case AccessType::Sampled:                return "sampled";
    case AccessType::StorageRead:            return "storage(r)";
    case AccessType::StorageWrite:           return "storage(w)";
    case AccessType::Uniform:                return "uniform";
    case AccessType::CopySrc:                return "copy(src)";
    case AccessType::CopyDst:                return "copy(dst)";
    case AccessType::Vertex:                 return "vertex";
    case AccessType::Index:                  return "index";
    case AccessType::Indirect:               return "indirect";
    }
    return "?";
}
// colour a resource lifetime bar by kind: textures cool, buffers warm.
static ImU32 rg_resource_color(ResourceKind k)
{
    return k == ResourceKind::Texture ? IM_COL32(70, 120, 170, 255)
                                            : IM_COL32(170, 120, 60, 255);
}

// short format label for the transient-pool view. Covers the formats this sample creates; anything
// else falls back to the raw enum so an unexpected recreate still shows something legible.
static const char* rg_format_name(WGPUTextureFormat f)
{
    switch (f) {
    case WGPUTextureFormat_BGRA8Unorm:   return "BGRA8";
    case WGPUTextureFormat_RGBA8Unorm:   return "RGBA8";
    case WGPUTextureFormat_RGBA16Float:  return "RGBA16F";
    case WGPUTextureFormat_RG16Float:    return "RG16F";
    case WGPUTextureFormat_R16Float:     return "R16F";
    case WGPUTextureFormat_R8Unorm:      return "R8";
    case WGPUTextureFormat_R32Float:     return "R32F";
    case WGPUTextureFormat_Depth32Float: return "D32F";
    default: break;
    }
    static char buf[16];
    std::snprintf(buf, sizeof buf, "fmt#%d", (int)f);
    return buf;
}

// usage bits the pool keys on
static void rg_usage_str(WGPUTextureUsage u, char* out, size_t n)
{
    std::snprintf(out, n, "%s%s%s%s%s",
        (u & WGPUTextureUsage_RenderAttachment) ? "A" : "",
        (u & WGPUTextureUsage_TextureBinding)   ? "T" : "",
        (u & WGPUTextureUsage_StorageBinding)   ? "S" : "",
        (u & WGPUTextureUsage_CopySrc)          ? "r" : "",
        (u & WGPUTextureUsage_CopyDst)          ? "w" : "");
}

// usage bits the buffer usage keys on
static void rg_buf_usage_str(WGPUBufferUsage u, char* out, size_t n)
{
    std::snprintf(out, n, "%s%s%s%s%s%s%s",
        (u & WGPUBufferUsage_Uniform)  ? "U" : "",
        (u & WGPUBufferUsage_Storage)  ? "S" : "",
        (u & WGPUBufferUsage_Vertex)   ? "V" : "",
        (u & WGPUBufferUsage_Index)    ? "I" : "",
        (u & WGPUBufferUsage_Indirect) ? "X" : "",
        (u & WGPUBufferUsage_CopySrc)  ? "r" : "",
        (u & WGPUBufferUsage_CopyDst)  ? "w" : "");
}

// bytes per texel for the formats this sample makes; unknown -> 0 so the UI shows "?" instead of
// inventing a number. tracks the set rg_format_name covers.
static uint32_t rg_format_bytes(WGPUTextureFormat f)
{
    switch (f) {
    case WGPUTextureFormat_R8Unorm:      return 1;
    case WGPUTextureFormat_R16Float:     return 2;
    case WGPUTextureFormat_RG16Float:    return 4;
    case WGPUTextureFormat_R32Float:     return 4;
    case WGPUTextureFormat_RGBA8Unorm:   return 4;
    case WGPUTextureFormat_BGRA8Unorm:   return 4;
    case WGPUTextureFormat_Depth32Float: return 4;
    case WGPUTextureFormat_RGBA16Float:  return 8;
    default: return 0;
    }
}

// one texture's footprint: every mip level, times array layers. NOTE(Huerbe): layers stays constant
// across mips, right for 2D / 2D-array (all this sample makes), over-counts a true 3D texture whose
// depth also halves each level. shared by the transient pool and the history (history) pool.
static uint64_t rg_texture_bytes(WGPUExtent3D size, WGPUTextureFormat format, uint32_t mipLevelCount)
{
    const uint64_t bpp = rg_format_bytes(format);
    if (!bpp) return 0;
    const uint32_t layers = size.depthOrArrayLayers ? size.depthOrArrayLayers : 1;
    uint64_t total = 0;
    for (uint32_t m = 0; m < mipLevelCount; ++m) {
        const uint32_t w = (size.width  >> m) ? (size.width  >> m) : 1u;   // max(1, ..) without <algorithm>
        const uint32_t h = (size.height >> m) ? (size.height >> m) : 1u;
        total += (uint64_t)w * h * layers * bpp;
    }
    return total;
}

static uint64_t rg_entry_bytes(const TransientResourcePool::Entry& e)
{
    return rg_texture_bytes(e.size, e.format, e.mipLevelCount);
}

// byte count -> short human string. NOTE(Huerbe): no GB tier, a transient pool is a few MB.
static void rg_bytes_str(uint64_t bytes, char* out, size_t n)
{
    if      (bytes >= (1u << 20)) std::snprintf(out, n, "%.1f MB", bytes / (1024.0 * 1024.0));
    else if (bytes >= (1u << 10)) std::snprintf(out, n, "%.1f KB", bytes / 1024.0);
    else                          std::snprintf(out, n, "%llu B",  (unsigned long long)bytes);
}

// argb-alpha tweak + warm/cool access tints, shared by the DAG pins below and the lifetime bars.
static ImU32 rg_with_alpha(ImU32 c, ImU32 a) { return (c & ~IM_COL32_A_MASK) | (a << IM_COL32_A_SHIFT); }
static constexpr ImU32 kRGWrite = IM_COL32(232, 145, 64, 255);   // write / output pin
static constexpr ImU32 kRGRead = IM_COL32(74, 158, 206, 255);   // read / input pin
static constexpr ImU32 kRGExt = IM_COL32(118, 196, 132, 255);   // external-input source node (imported read)
static constexpr ImU32 kRGPresent = IM_COL32(196, 122, 214, 255);   // present / display sink node (imported output)

// stable distinct colour per group, hashed from the prefix (FNV-1a -> small palette).
static ImU32 group_color(WGPUStringView prefix)
{
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < sv_length(prefix); ++i) { h ^= (uint8_t)prefix.data[i]; h *= 16777619u; }
    static const ImU32 pal[] = {
        IM_COL32(120, 180, 230, 230), IM_COL32(230, 170, 90, 230), IM_COL32(150, 210, 140, 230),
        IM_COL32(210, 140, 200, 230), IM_COL32(220, 205, 110, 230), IM_COL32(140, 205, 210, 230),
    };
    return pal[h % (sizeof pal / sizeof pal[0])];
}

// stable, ID-stack-independent key for a group's collapse state. ImGui::GetID is seeded by the current
// window, so the pre-layout read (outer window) and the in-canvas click write (rg_canvas child) would hash
// the same string to DIFFERENT ids; a hash of the prefix (salted) is the same wherever it's computed.
static ImGuiID rg_grp_key(WGPUStringView prefix)
{
    ImU32 h = 2166136261u;
    for (const char* s = "rg.grp."; *s; ++s) { h ^= (uint8_t)*s; h *= 16777619u; }
    for (size_t i = 0; i < sv_length(prefix); ++i) { h ^= (uint8_t)prefix.data[i]; h *= 16777619u; }
    return (ImGuiID)h;
}

// DAG view -----------------------------------------------------------------------------------------
// node-graph layout: one box per pass in dependency columns, one pin per resource access (reads left,
// writes right), edges run producer-output -> consumer-input (true RAW data flow). Hovering a pin lights
// the upstream producer cone, every pass that must run to make that resource. Reads the .cpp-internal
// node structs directly; assumes a compiled, realized graph like the other dumps.

static constexpr int kRgDagMax = 128;
static constexpr int kRgGPinMax = 32;   // max interface pins drawn on a collapsed group node (silently capped)

// one laid-out pass box. its index in box[] == execution-order index (boxes built by walking m_passes),
// so it doubles as the adjacency / cone index.
struct RgDagBox { PassNode* p; int layer; ImVec2 tl; float w, h; int nIn, nOut; };

// does pass p write resource id? one access == one pin, writes land on the right.
static bool rg_pass_writes(PassNode* p, uint32_t id)
{
    for (uint32_t i = 0; i < p->accessCount; ++i)
        if (p->accesses[i].handle.id == id && access_is_write(p->accesses[i].type)) return true;
    return false;
}

// does this access consume the resource's prior contents -> an input pin? plain reads do; so does a
// color/depth attachment with LoadOp_Load (reads what's there, then stores == read-modify-write, e.g. the
// imgui overlay loading the swapchain). a Load attachment therefore gets BOTH an input and an output pin.
static bool rg_access_reads(const ResourceAccess& a)
{
    if (!access_is_write(a.type)) return true;
    return (a.type == AccessType::ColorAttachment || a.type == AccessType::DepthStencilAttachment)
        && a.loadOp == WGPULoadOp_Load;
}

// do a and b both write a common resource? 0 = no, 1 = yes at DIFFERENT subresources (parallel layers/
// mips, e.g. CSM cascades, not a true conflict), 2 = yes at the SAME subresource (a real WAW).
static int rg_shared_write(PassNode* a, PassNode* b)
{
    int r = 0;
    for (uint32_t i = 0; i < a->accessCount; ++i) {
        if (!access_is_write(a->accesses[i].type)) continue;
        for (uint32_t j = 0; j < b->accessCount; ++j) {
            if (!access_is_write(b->accesses[j].type) || a->accesses[i].handle.id != b->accesses[j].handle.id) continue;
            if (a->accesses[i].baseLayer == b->accesses[j].baseLayer && a->accesses[i].baseMip == b->accesses[j].baseMip) return 2;
            r = 1;
        }
    }
    return r;
}

// index of pass in box[] (== execution order), or -1.
static int rg_box_index(const RgDagBox* box, int n, PassNode* p)
{
    for (int i = 0; i < n; ++i) if (box[i].p == p) return i;
    return -1;
}

// producer of resource `id` as read by pass `p`: the highest-execution-index writer of `id` among p's
// predecessors. compile() always links the current (RAW) producer into adjacency, and a later writer
// would have started a version p can't see, so the latest writer-predecessor IS that producer. -1 =
// no in-graph producer (imported value, or read-before-write -> an external input pin).
static int rg_producer_of(const RgDagBox* box, int n, PassNode* p, uint32_t id)
{
    int best = -1;
    for (NodeAdjacency* a = p->adjacency; a; a = a->next) {
        if (!rg_pass_writes(a->pass, id)) continue;
        int idx = rg_box_index(box, n, a->pass);
        if (idx > best) best = idx;
    }
    return best;
}

// output-pin slot index of resource id on pass p (writes only), or -1.
static int rg_out_slot(PassNode* p, uint32_t id)
{
    int slot = 0;
    for (uint32_t k = 0; k < p->accessCount; ++k)
        if (access_is_write(p->accesses[k].type)) { if (p->accesses[k].handle.id == id) return slot; ++slot; }
    return -1;
}

// input-pin slot index of resource id on pass p (reads only), mirroring rg_out_slot. matches the
// encounter-order slot the pin loop assigns, so it lands on the right pin centre.
static int rg_in_slot(PassNode* p, uint32_t id)
{
    int slot = 0;
    for (uint32_t k = 0; k < p->accessCount; ++k)
        if (rg_access_reads(p->accesses[k])) { if (p->accesses[k].handle.id == id) return slot; ++slot; }
    return -1;
}

// external interface of the pass run [gi, gj): reads whose producer sits OUTSIDE the range become in-pins;
// writes that are imported/persistent or consumed outside the range become out-pins. interior resources
// (produced and consumed within the run, e.g. bloom's mip chain) get no pin. both lists dedup and cap at
// kRgGPinMax (silently). one walk for every group node: collapsed slot, expanded border, and the draw.
static void rg_group_interface(RenderGraph* rg, const RgDagBox* box, int n, int gi, int gj,
    uint32_t* inId, int& nIn, uint32_t* outId, int& nOut)
{
    nIn = 0; nOut = 0;
    for (int k = gi; k < gj; ++k) {
        PassNode* p = box[k].p;
        for (uint32_t ai = 0; ai < p->accessCount; ++ai) {
            if (!rg_access_reads(p->accesses[ai])) continue;
            uint32_t id = p->accesses[ai].handle.id;
            int prod = rg_producer_of(box, n, p, id);
            if (prod >= gi && prod < gj) continue;
            bool seen = false; for (int s = 0; s < nIn; ++s) seen |= inId[s] == id;
            if (!seen && nIn < kRgGPinMax) inId[nIn++] = id;
        }
    }
    for (int k = gi; k < gj; ++k) {
        PassNode* p = box[k].p;
        for (uint32_t ai = 0; ai < p->accessCount; ++ai) {
            if (!access_is_write(p->accesses[ai].type)) continue;
            uint32_t id = p->accesses[ai].handle.id;
            ResourceNode* r = find_node(rg, { id });
            bool external = r && (r->imported || r->persistent);   // history write leaves to next frame -> group output
            for (int j = 0; j < n && !external; ++j) {
                if (j >= gi && j < gj) continue;
                if (rg_in_slot(box[j].p, id) < 0) continue;
                int pr = rg_producer_of(box, n, box[j].p, id);
                if (pr >= gi && pr < gj) external = true;
            }
            if (!external) continue;
            bool seen = false; for (int s = 0; s < nOut; ++s) seen |= outId[s] == id;
            if (!seen && nOut < kRgGPinMax) outId[nOut++] = id;
        }
    }
}

// mark seed + everything it transitively depends on (the upstream producer cone): DFS over predecessor
// edges. iterative, since a long pass chain would overflow a recursive stack.
static void rg_mark_cone(const RgDagBox* box, int n, int seed, bool* inCone)
{
    if (seed < 0) return;
    int stack[kRgDagMax], sp = 0;
    stack[sp++] = seed; inCone[seed] = true;
    while (sp) {
        PassNode* p = box[stack[--sp]].p;
        for (NodeAdjacency* a = p->adjacency; a; a = a->next) {
            int idx = rg_box_index(box, n, a->pass);
            if (idx >= 0 && !inCone[idx]) { inCone[idx] = true; stack[sp++] = idx; }
        }
    }
}

// case-insensitive: does pass name contain needle? (name filter)
static bool rg_name_has(WGPUStringView name, const char* needle)
{
    if (!needle || !needle[0]) return true;
    if (!name.data) return false;
    size_t q = 0; while (needle[q]) ++q;
    auto lc = [](char c) { return (c >= 'A' && c <= 'Z') ? char(c + 32) : c; };
    for (size_t i = 0; i + q <= name.length; ++i) {
        size_t j = 0; while (j < q && lc(name.data[i + j]) == lc(needle[j])) ++j;
        if (j == q) return true;
    }
    return false;
}

// squared distance from point p to segment ab (edge hit-testing).
static float rg_seg_d2(ImVec2 p, ImVec2 a, ImVec2 b)
{
    float vx = b.x - a.x, vy = b.y - a.y, wx = p.x - a.x, wy = p.y - a.y;
    float L = vx * vx + vy * vy, t = L > 0 ? (wx * vx + wy * vy) / L : 0;
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    float dx = a.x + t * vx - p.x, dy = a.y + t * vy - p.y;
    return dx * dx + dy * dy;
}

// true graph output for the halo: writes an *imported* resource (swapchain) whose value leaves the
// frame to be presented. compile()'s p->sink also fires for passes that only write a history
// layer (persistent, read next frame), which keep the pass alive but aren't graph outputs, so the
// halo skips them.
static bool rg_pass_is_sink(RenderGraph* rg, PassNode* p)
{
    for (uint32_t i = 0; i < p->accessCount; ++i) {
        if (!access_is_write(p->accesses[i].type)) continue;
        ResourceNode* r = find_node(rg, p->accesses[i].handle);
        if (r && r->imported) return true;
    }
    return false;
}

// one pin glyph: round for textures, square for buffers. filled = normal, hollow = external input (no
// in-graph producer). square half-extent == circle radius, so the hover/lock ring (kPinR + 3) still
// frames either shape.
static void rg_draw_pin(ImDrawList* dl, ImVec2 c, float r, ImU32 col, bool filled, bool buffer)
{
    if (buffer) {
        ImVec2 a(c.x - r, c.y - r), b(c.x + r, c.y + r);
        if (filled) dl->AddRectFilled(a, b, col, 1.5f);
        else        dl->AddRect(a, b, col, 1.5f, 0, 2.0f);
    }
    else {
        if (filled) dl->AddCircleFilled(c, r, col, 12);
        else        dl->AddCircle(c, r, col, 12, 2.0f);
    }
}

// dashed cubic bezier: ImDrawList has no dashed stroke, so sample the curve and draw every other span.
// used by the history feedback links so they read as cross-frame, distinct from the solid data edges.
static void rg_dashed_cubic(ImDrawList* dl, ImVec2 p0, ImVec2 p1, ImVec2 p2, ImVec2 p3, ImU32 col, float th)
{
    constexpr int kSeg = 24;
    ImVec2 prev = p0;
    for (int i = 1; i <= kSeg; ++i) {
        float t = (float)i / kSeg, u = 1 - t, w0 = u*u*u, w1 = 3*u*u*t, w2 = 3*u*t*t, w3 = t*t*t;
        ImVec2 q(w0*p0.x + w1*p1.x + w2*p2.x + w3*p3.x, w0*p0.y + w1*p1.y + w2*p2.y + w3*p3.y);
        if (i & 1) dl->AddLine(prev, q, col, th);   // every other span = a dash
        prev = q;
    }
}

// little filled arrowhead at `tip`, pointing along (tip - from). marks the read end of a feedback link.
static void rg_arrowhead(ImDrawList* dl, ImVec2 from, ImVec2 tip, ImU32 col, float sz)
{
    float dx = tip.x - from.x, dy = tip.y - from.y, L = std::sqrt(dx*dx + dy*dy);
    if (L < 1e-3f) return;
    dx /= L; dy /= L;
    ImVec2 a(tip.x - dx*sz - dy*sz*0.5f, tip.y - dy*sz + dx*sz*0.5f);
    ImVec2 b(tip.x - dx*sz + dy*sz*0.5f, tip.y - dy*sz - dx*sz*0.5f);
    dl->AddTriangleFilled(tip, a, b, col);
}

// ---- DAG intermediate representation (see docs/rendergraph-nested-layout.md). built once per frame from
// the passes + frame-boundary endpoints, then drawn. a node is a pass or a virtual endpoint (history/
// imported/present), linked by RgEdges. the pass-graph (box[]) still drives layout; the draw consumes this
// IR for the nodes + virtual links. groups live in the GView side-table, not here.
struct RgNode {
    enum class Kind : uint8_t { Pass, Group, Virtual };
    Kind kind{};
    PassNode*      pass{};      // Kind::Pass
    ResourceNode*  res{};       // Kind::Virtual (the endpoint's resource)
    WGPUStringView label{};     // Kind::Virtual caption
    ImVec2 pos{}; float w = 0, h = 0; int col = 0;              // layout result -> consumed by draw
    ImU32 tint = 0;             // Kind::Virtual base colour (read/write/imported/present)
};
struct RgEdge {
    int srcNode = -1, dstNode = -1; uint16_t srcPin = 0, dstPin = 0; uint32_t resId = 0;
    enum class Kind : uint8_t { Raw, History, Fanout } kind = Kind::Raw;
};

// barycenter crossing-reduction + y-relax over a layered node set (Sugiyama). lcol[c] = node indices in
// column c (reordered in place); lleft/lright[i] = a node's neighbours; h[i] = node height. writes cy[i].
// shared by the top-level layout and each expanded group's interior so internal edges route like the top.
// clus/clusPar (optional): per-node cluster id + per-cluster parent (-1 = root), a gtree over the nodes.
// when given, the column ordering keeps every cluster's members, along with any nested sub-clusters, in one
// contiguous run, so an expanded group's members never interleave with outsiders and its border stays a clean hull.
static void rg_barycenter_relax(std::vector<int>* lcol, int maxCol,
    const std::vector<std::vector<int>>& lleft, const std::vector<std::vector<int>>& lright,
    const std::vector<float>& h, std::vector<float>& cy, float rowGap,
    const int* clus = nullptr, const int* clusPar = nullptr, int nClus = 0)
{
    int LN = (int)cy.size();
    static std::vector<int> lrank; lrank.assign(LN, 0);
    auto reix = [&](int c) { for (int r = 0; r < (int)lcol[c].size(); ++r) lrank[lcol[c][r]] = r; };
    for (int c = 0; c <= maxCol; ++c) reix(c);
    auto bnb = [&](int li, bool right) { const std::vector<int>& nb = right ? lright[li] : lleft[li]; float s = 0; int c = 0; for (int x : nb) { s += lrank[x]; ++c; } return c ? s / c : -1.0f; };
    auto lsweep = [&](bool backward) {
        int from = backward ? maxCol - 1 : 1, to = backward ? -1 : maxCol + 1, step = backward ? -1 : 1;
        for (int c = from; c != to; c += step) {
            std::vector<int>& m = lcol[c];
            std::vector<float> key(m.size());
            for (int j = 0; j < (int)m.size(); ++j) { float b = bnb(m[j], backward); key[j] = b < 0 ? (float)lrank[m[j]] : b; }
            for (int a = 1; a < (int)m.size(); ++a) { int mv = m[a]; float kv = key[a]; int b = a - 1; while (b >= 0 && key[b] > kv) { m[b + 1] = m[b]; key[b + 1] = key[b]; --b; } m[b + 1] = mv; key[b + 1] = kv; }
            reix(c);
        }
        };
    lsweep(true); lsweep(false); lsweep(true);
    // cluster-contiguity pass: after the barycenter order is settled, re-sort each column so a cluster's
    // members (and any nested sub-cluster) stay together. clusters are weighted by their members' mean rank,
    // so a group still floats to roughly its barycenter; it just moves as one block instead of scattering.
    if (clus) {
        // root..innermost chain of cluster ids for a node; depth returned. (shallow, path length <= ~4.)
        auto chain = [&](int li, int* out) -> int {
            int tmp[16], d = 0; for (int c = clus[li]; c >= 0 && d < 16; c = clusPar[c]) tmp[d++] = c;
            for (int i = 0; i < d; ++i) out[i] = tmp[d - 1 - i]; return d;
        };
        static std::vector<float> cavg; static std::vector<int> ccnt;
        for (int c = 0; c <= maxCol; ++c) {
            std::vector<int>& m = lcol[c];
            cavg.assign(nClus, 0.0f); ccnt.assign(nClus, 0);
            for (int li : m) { int ch[16]; int d = chain(li, ch); for (int i = 0; i < d; ++i) { cavg[ch[i]] += lrank[li]; ++ccnt[ch[i]]; } }
            for (int x = 0; x < nClus; ++x) if (ccnt[x]) cavg[x] /= ccnt[x];
            // a precedes b: compare down their cluster paths, weighting a shared-prefix cluster by its mean
            // rank and a node's own level by its rank; first differing weight decides (ties keep stable order).
            auto before = [&](int a, int b) -> bool {
                int ca[16], cb[16]; int da = chain(a, ca), db = chain(b, cb);
                for (int d = 0;; ++d) {
                    if (d < da && d < db && ca[d] == cb[d]) continue;   // same cluster here -> go deeper
                    float wa = d < da ? cavg[ca[d]] : (float)lrank[a];
                    float wb = d < db ? cavg[cb[d]] : (float)lrank[b];
                    if (wa != wb) return wa < wb;
                    return lrank[a] < lrank[b];
                }
            };
            for (int i = 1; i < (int)m.size(); ++i) { int v = m[i], j = i - 1; while (j >= 0 && before(v, m[j])) { m[j + 1] = m[j]; --j; } m[j + 1] = v; }
            reix(c);
        }
    }
    // gap only BETWEEN two real (non-zero-height) nodes; a folded collapsed member is zero-height and must
    // not reserve a row, else a collapsed group leaves a tall empty column (N-1 phantom rowGaps).
    auto gap = [&](int a, int b) { return (h[a] > 0.5f && h[b] > 0.5f) ? rowGap : 0.0f; };
    for (int c = 0; c <= maxCol; ++c) { float y = 0; int prev = -1; for (int li : lcol[c]) { if (prev >= 0) y += gap(prev, li); cy[li] = y + h[li] * 0.5f; y += h[li]; prev = li; } }
    for (int it = 0; it < 8; ++it)
        for (int c = 0; c <= maxCol; ++c) {
            std::vector<int>& m = lcol[c];
            for (int r = 0; r < (int)m.size(); ++r) {
                int li = m[r]; float s = 0; int cnt = 0;
                for (int x : lleft[li])  { s += cy[x]; ++cnt; }
                for (int x : lright[li]) { s += cy[x]; ++cnt; }
                if (!cnt) continue;
                float d = s / cnt;
                float lo = r > 0                ? cy[m[r - 1]] + (h[m[r - 1]] + h[li]) * 0.5f + gap(m[r - 1], li) : -1e30f;
                float hi = r + 1 < (int)m.size() ? cy[m[r + 1]] - (h[li] + h[m[r + 1]]) * 0.5f - gap(li, m[r + 1]) :  1e30f;
                cy[li] = d < lo ? lo : d > hi ? hi : d;
            }
        }
}

// span of `name` covering its first `segs` dotted segments ("bloom.down.0" with segs=2 -> "bloom.down").
// empty when the name has fewer than `segs` segments. group_prefix is the segs==1 case; this generalizes
// it so a group tree can partition a deeper level (bloom.down vs bloom.up) at each recursion step.
static WGPUStringView group_prefix_n(WGPUStringView name, int segs)
{
    size_t n = sv_length(name); int seen = 0;
    for (size_t i = 0; i < n; ++i)
        if (name.data[i] == '.' && ++seen == segs) return WGPUStringView{ name.data, i };
    return (seen + 1 == segs) ? name : WGPUStringView{};   // exactly `segs` segments -> whole name is the prefix
}

// one node of the group tree: a contiguous run [gi, gj) of box[] (passes are declared grouped, so a
// dotted-name group is always contiguous). gtree[0] is the root (whole graph, empty prefix). a node's
// kids are deeper sub-runs; passes in [gi,gj) not covered by any kid are this node's leaf members. layout
// fields are filled post-order by layout_gnode (w/h + interface pins, relative to the node's content origin).
struct GNode {
    WGPUStringView prefix;     // full dotted prefix at this depth ("bloom", "bloom.down"); empty at root
    int gi, gj, depth;
    bool collapsed;
    std::vector<int> kids;     // child GNode indices in the gtree arena
    float w, h;                // box (collapsed) or region (expanded) size, content-local
    int nIn, nOut;
    uint32_t inId[kRgGPinMax], outId[kRgGPinMax];
    float inY[kRgGPinMax], outY[kRgGPinMax];   // interface pin y, relative to the node top
};

// build the group tree over box[gi, gj) at `depth`, appending nodes to gtree; returns the new node's index.
// partitions the range into maximal sub-runs sharing their first (depth+1) name segments; a run of >=2 (that
// isn't the whole range, which would just re-wrap the same passes) becomes a child group and recurses.
// collapse state is the same tri-state ImGuiStorage read the draw uses (0 follow-default / 1 open / 2 closed).
static int build_gtree(std::vector<GNode>& gtree, const RgDagBox* box, int n,
    ImGuiStorage* grpStore, bool collapseDefault, int gi, int gj, int depth, WGPUStringView prefix)
{
    int self = (int)gtree.size();
    gtree.push_back(GNode{});
    GNode g{};
    g.prefix = prefix; g.gi = gi; g.gj = gj; g.depth = depth;
    if (sv_length(prefix)) { int st = grpStore->GetInt(rg_grp_key(prefix), 0); g.collapsed = st == 2 ? true : st == 1 ? false : collapseDefault; }
    for (int a = gi; a < gj;) {
        WGPUStringView sub = group_prefix_n(box[a].p->id.name, depth + 1);
        int b = a + 1;
        while (b < gj && sv_length(sub) && sv_eq(group_prefix_n(box[b].p->id.name, depth + 1), sub)) ++b;
        if (sv_length(sub) && b - a >= 2 && !(a == gi && b == gj))
            g.kids.push_back(build_gtree(gtree, box, n, grpStore, collapseDefault, a, b, depth + 1, sub));
        a = b;
    }
    gtree[self] = std::move(g);
    return self;
}

// collOwner[k] = the OUTERMOST collapsed group ancestor of pass k (-1 = visible at every level). a pass with
// an owner is hidden; the owner's first member (gtree[owner].gi) is the rep that stands in for the whole
// collapsed subtree. one notion for collapse at any depth: a top-level collapsed group and a collapsed
// nested subgroup differ only in where their rep cell is laid out, not in how membership is decided.
static void rg_mark_collapsed(const std::vector<GNode>& gt, int* collOwner, int node, int owner)
{
    const GNode& g = gt[node];
    int myOwner = owner;
    if (owner < 0 && sv_length(g.prefix) && g.collapsed) myOwner = node;
    if (myOwner >= 0) for (int k = g.gi; k < g.gj; ++k) if (collOwner[k] < 0) collOwner[k] = myOwner;
    for (int kid : g.kids) rg_mark_collapsed(gt, collOwner, kid, myOwner);
}

// clusterOf[k] = the innermost EXPANDED gtree node containing pass k (-1 = ungrouped); clusterParent[o] =
// its parent gtree node. drives the cluster-contiguity ordering so an expanded group, along with its nested
// expanded subgroups, lays out as one block. a collapsed group folds (its members get its parent cluster),
// so its subtree is not descended; members keep their nearest expanded ancestor.
static void rg_assign_clusters(const std::vector<GNode>& gt, int* clusterOf, int* clusterParent, int node)
{
    const GNode& g = gt[node];
    for (int kid : g.kids) clusterParent[kid] = node;
    if (sv_length(g.prefix)) {
        if (g.collapsed) return;   // folded subtree -> members keep the parent expanded cluster
        for (int k = g.gi; k < g.gj; ++k) clusterOf[k] = node;
    }
    for (int kid : g.kids) rg_assign_clusters(gt, clusterOf, clusterParent, kid);
}

// smooth "bump" link: leaves a box horizontally, S-curves up to a flat top lane, runs across, then S-curves
// back down into the consumer. horizontal tangents at the pins AND at the plateau, so no corner is ever sharp.
static void rg_over_arc(ImDrawList* dl, ImVec2 a, ImVec2 b, float topY, ImU32 col, float th, float maxSh)
{
    float dx = b.x - a.x, adx = dx < 0 ? -dx : dx, dir = dx < 0 ? -1.0f : 1.0f;
    float sh = adx * 0.5f; if (sh > maxSh) sh = maxSh;   // shoulder run; shrinks for short hops so the two S's don't overlap
    ImVec2 aTop(a.x + dir * sh, topY), bTop(b.x - dir * sh, topY);
    float m1 = a.x + dir * sh * 0.5f, m2 = b.x - dir * sh * 0.5f;
    dl->AddBezierCubic(a, ImVec2(m1, a.y), ImVec2(m1, topY), aTop, col, th);
    if ((bTop.x - aTop.x) * dir > 0.5f) dl->AddLine(aTop, bTop, col, th);
    dl->AddBezierCubic(bTop, ImVec2(m2, topY), ImVec2(m2, b.y), b, col, th);
}

static void rg_draw_dag(RenderGraph* rg, RenderGraphStorage& s)
{
    float kBoxW = 190.0f, kColGap = 65.0f, kRowGap = 50.0f;
    float kHeaderH = 22.0f, kFooterH = 14.0f, kPinRowH = 18.0f, kMinBodyH = 12.0f;
    float kPinR = 5.0f, kPinHit = 8.0f;
    float kRegionPad = 24.0f;   // padding from a group's member bbox out to its hull border

    // pan + zoom state, shared between the wheel handler here and the canvas below. kept at function scope so
    // the wheel is resolved BEFORE layout, so positions, scrolling and font then all use one zoom this frame
    // (resolving it after layout drew the graph at the old scale for a frame -> flicker).
    static ImVec2 scrolling(0, 0);
    static bool userMoved = false;          // true once the user pans/zooms; until then keep the graph centred
    static bool s_canvasHovered = false;    // last frame's hover (input is read before the canvas item exists)
    static ImVec2 s_winPos(0, 0);           // last frame's canvas top-left, for the cursor-anchored zoom

    // zoom: 1.0 = closest (the authored sizes), the wheel only zooms OUT toward the cursor. every layout
    // constant scales by it so layout + screen draw stay one coordinate system; text scales separately via
    // the child font scale below (SetWindowFontScale).
    static float zoom = 1.0f;
    {
        ImGuiIO& io = ImGui::GetIO();
        if (s_canvasHovered && io.MouseWheel != 0.0f) {
            float z0 = zoom;
            zoom *= io.MouseWheel > 0 ? 1.1f : 1.0f / 1.1f;
            if (zoom > 1.0f) zoom = 1.0f;
            if (zoom < 0.2f) zoom = 0.2f;
            float r = zoom / z0;   // positions rescale by r this frame; anchor the canvas point under the cursor
            scrolling.x = io.MousePos.x - s_winPos.x - (io.MousePos.x - s_winPos.x - scrolling.x) * r;
            scrolling.y = io.MousePos.y - s_winPos.y - (io.MousePos.y - s_winPos.y - scrolling.y) * r;
            userMoved = true;
        }
    }
    const float z = zoom;
    kBoxW *= z; kColGap *= z; kRowGap *= z;
    kHeaderH *= z; kFooterH *= z; kPinRowH *= z; kMinBodyH *= z;
    kPinR *= z; kPinHit *= z;
    kRegionPad *= z;


    // virtual nodes (frame-boundary endpoints: history read/write so far), toggled from the toolbar.
    // read before the layout pass so disabling them also drops their layout influence (no reserved columns).
    static bool showVirtual = true;
    // imported buffers (uniforms read by many passes): on = one source node fanning faint edges to every
    // reader; off = a node at each use site, like an imported texture. read here so it shapes the layout too.
    static bool fanBuffers = true;
    // collapse groups by default (a group-header click overrides per group). read here, before the layout,
    // so a collapsed group can reserve one compact slot the same frame the checkbox flips.
    static bool collapseDefault = false;
    // one storage object for all group collapse state, captured here in the OUTER window so the pre-layout
    // reads and the in-canvas-child click writes hit the same store (see rg_grp_key for the matching key).
    ImGuiStorage* grpStore = ImGui::GetStateStorage();

    // ---- layout pass (sink-anchored, Sugiyama-style). columns place each pass left of what depends on
    // it: column = longest distance TO a sink over the FULL dependency graph, so sinks land rightmost and
    // a WAW-ordered pass (a shadow cascade, no pin to its successor) still sits in order instead of
    // floating off as a fake sink. crossing reduction, in contrast, counts only the DRAWN edges (the
    // RAW producer->consumer pins) because WAW has no pin to cross. one barycenter sweep from the sinks
    // back, then a settle. no dummy nodes for multi-column edges; this is a debug view.
    RgDagBox box[kRgDagMax];
    int n = 0;
    for (PassNode* p = s.m_passes; p && n < kRgDagMax; p = p->next, ++n) {
        int nIn = 0, nOut = 0;   // a Load attachment counts as both (read-modify-write)
        for (uint32_t k = 0; k < p->accessCount; ++k) {
            if (rg_access_reads(p->accesses[k]))      ++nIn;
            if (access_is_write(p->accesses[k].type)) ++nOut;
        }
        int rows = nIn > nOut ? nIn : nOut;
        float h = kHeaderH + (rows ? rows * kPinRowH : kMinBodyH) + kFooterH;
        box[n] = { p, 0, ImVec2(0, 0), kBoxW, h, nIn, nOut };
    }

    // dist = longest path to a sink over all deps; reverse exec order visits sinks first and relaxes
    // each producer. column = maxDist - dist.
    int dist[kRgDagMax] = {}, maxDist = 0;
    for (int i = n - 1; i >= 0; --i) {
        for (NodeAdjacency* a = box[i].p->adjacency; a; a = a->next) {
            int q = rg_box_index(box, n, a->pass);
            if (q >= 0 && dist[i] + 1 > dist[q]) dist[q] = dist[i] + 1;
        }
        if (dist[i] > maxDist) maxDist = dist[i];
    }

    // parallel-writer groups: passes writing one resource at DIFFERENT subresources (CSM cascades: same
    // csm handle, layer 0/1/2) with no data dependency between them. the whole-resource WAW would string
    // them across columns; union them onto one DAG level instead. anc[v][u] = u is a transitive data-
    // ancestor of v (built in exec order, producers precede consumers), so a real chain like the bloom
    // mip pyramid, where each level samples the previous, is NOT mistaken for parallel.
    static bool anc[kRgDagMax][kRgDagMax];
    for (int v = 0; v < n; ++v) for (int u = 0; u < n; ++u) anc[v][u] = false;
    for (int v = 0; v < n; ++v) {
        PassNode* p = box[v].p;
        for (uint32_t k = 0; k < p->accessCount; ++k) {
            if (!rg_access_reads(p->accesses[k])) continue;
            int u = rg_producer_of(box, n, p, p->accesses[k].handle.id);
            if (u < 0) continue;
            anc[v][u] = true;
            for (int w = 0; w < n; ++w) if (anc[u][w]) anc[v][w] = true;
        }
    }
    int groupRep[kRgDagMax];
    for (int i = 0; i < n; ++i) groupRep[i] = i;
    auto find = [&](int x) { while (groupRep[x] != x) { groupRep[x] = groupRep[groupRep[x]]; x = groupRep[x]; } return x; };
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (rg_shared_write(box[i].p, box[j].p) == 1 && !anc[i][j] && !anc[j][i]) groupRep[find(i)] = find(j);
    for (int i = 0; i < n; ++i) groupRep[i] = find(i);   // flatten to roots

    // column = maxDist - dist, then snap each group to its rightmost member (nearest its consumer) so the
    // siblings share one level, and compact away the columns that leaves empty.
    int colOf[kRgDagMax], groupCol[kRgDagMax] = {};
    for (int i = 0; i < n; ++i) colOf[i] = maxDist - dist[i];
    for (int i = 0; i < n; ++i) if (colOf[i] > groupCol[groupRep[i]]) groupCol[groupRep[i]] = colOf[i];
    for (int i = 0; i < n; ++i) colOf[i] = groupCol[groupRep[i]];
    int remap[kRgDagMax]; for (int c = 0; c <= maxDist; ++c) remap[c] = -1;
    for (int i = 0; i < n; ++i) remap[colOf[i]] = 1;
    int nextCol = 0; for (int c = 0; c <= maxDist; ++c) if (remap[c] == 1) remap[c] = nextCol++;
    for (int i = 0; i < n; ++i) colOf[i] = remap[colOf[i]];
    maxDist = nextCol ? nextCol - 1 : 0;

    // group tree: the recursive replacement for the per-level group/subgroup detection scattered below. built
    // here (columns are final) and consumed by the layout + draw. not yet wired into layout: the reservation
    // blocks below still run; this just stands up the structure and asserts it covers every pass.
    static std::vector<GNode> gtree; gtree.clear();
    build_gtree(gtree, box, n, grpStore, collapseDefault, 0, n, 0, WGPUStringView{});
    IM_ASSERT(gtree[0].gi == 0 && gtree[0].gj == n);
    int collOwner[kRgDagMax]; for (int i = 0; i < n; ++i) collOwner[i] = -1;
    rg_mark_collapsed(gtree, collOwner, 0, -1);
    // a pass inside ANY collapsed group (top-level or subgroup) is not drawn as its own box; the collapsed
    // group draws one compact node in its place.
    bool drawHidden[kRgDagMax]; for (int i = 0; i < n; ++i) drawHidden[i] = collOwner[i] >= 0;

    // ---- collapse folding (pre-layout). every OUTERMOST collapsed group, top-level or nested, folds to
    // ONE compact node at its rep cell (gtree[o].gi, carrying the group's interface pins); the other members
    // get zero height and snap onto the rep's (leftmost) column so they reserve no layout space. EXPANDED
    // groups do NOT reserve a slot; their members stay first-class nodes in their own dependency
    // columns and the cluster-contiguity pass keeps them together; a hull border is drawn around them later.
    float effH[kRgDagMax];
    for (int i = 0; i < n; ++i) effH[i] = box[i].h;
    for (int o = 0; o < (int)gtree.size(); ++o) {
        GNode& g = gtree[o];
        if (!sv_length(g.prefix) || !g.collapsed) continue;
        if (collOwner[g.gi] != o) continue;   // shadowed by a collapsed ancestor -> that outer group is the rep
        uint32_t inId[kRgGPinMax], outId[kRgGPinMax]; int ni = 0, no = 0;
        rg_group_interface(rg, box, n, g.gi, g.gj, inId, ni, outId, no);
        int rows = ni > no ? ni : no;
        float compactH = kHeaderH + (rows ? rows * kPinRowH : kMinBodyH) + kFooterH;
        int colA = colOf[g.gi]; for (int k = g.gi; k < g.gj; ++k) if (colOf[k] < colA) colA = colOf[k];   // leftmost member column
        for (int k = g.gi; k < g.gj; ++k) { effH[k] = (k == g.gi ? compactH : 0.0f); colOf[k] = colA; }
    }

    // cluster ids + parent links for the layout ordering (innermost expanded group per pass).
    int clusterOf[kRgDagMax]; for (int i = 0; i < n; ++i) clusterOf[i] = -1;
    static std::vector<int> clusterParent; clusterParent.assign(gtree.size(), -1);
    rg_assign_clusters(gtree, clusterOf, clusterParent.data(), 0);

    // recompute columns with each collapsed group CONTRACTED to one column (its rep). longest-path-to-sink
    // over the rep graph: an edge inside a collapsed group costs 0, a crossing edge costs 1. so a consumer of
    // a collapsed sub-chain (bloom.composite after a collapsed bloom.up) lands right after the rep instead of
    // where the un-folded chain stranded it, closing the empty span. expanded groups (rep[i]==i) reproduce
    // the original columns exactly, so nothing else moves.
    {
        int rep[kRgDagMax]; for (int i = 0; i < n; ++i) rep[i] = collOwner[i] >= 0 ? gtree[collOwner[i]].gi : i;
        int dist2[kRgDagMax]; for (int i = 0; i < n; ++i) dist2[i] = 0; int maxD2 = 0;
        for (int i = n - 1; i >= 0; --i)
            for (NodeAdjacency* a = box[i].p->adjacency; a; a = a->next) {
                int q = rg_box_index(box, n, a->pass); if (q < 0) continue;   // a->pass = producer of i
                int w = rep[i] == rep[q] ? 0 : 1;
                if (dist2[rep[i]] + w > dist2[rep[q]]) dist2[rep[q]] = dist2[rep[i]] + w;
            }
        for (int i = 0; i < n; ++i) if (dist2[rep[i]] > maxD2) maxD2 = dist2[rep[i]];
        for (int i = 0; i < n; ++i) colOf[i] = maxD2 - dist2[rep[i]];   // members of a group share their rep's column
        // re-snap parallel-writer groups (cascades) to one column, then compact any empty columns.
        int gc[kRgDagMax]; for (int i = 0; i < n; ++i) gc[i] = 0;
        for (int i = 0; i < n; ++i) if (colOf[i] > gc[groupRep[i]]) gc[groupRep[i]] = colOf[i];
        for (int i = 0; i < n; ++i) colOf[i] = gc[groupRep[i]];
        maxDist = maxD2;
        for (int c = 0; c <= maxDist; ++c) remap[c] = -1;
        for (int i = 0; i < n; ++i) remap[colOf[i]] = 1;
        nextCol = 0; for (int c = 0; c <= maxDist; ++c) if (remap[c] == 1) remap[c] = nextCol++;
        for (int i = 0; i < n; ++i) colOf[i] = remap[colOf[i]];
        maxDist = nextCol ? nextCol - 1 : 0;
    }

    // ---- virtual nodes: frame-boundary endpoints drawn as real DAG nodes so the column/barycenter/y code
    // below places them like any pass (no bespoke overlay). each attaches to ONE pass pin: a read node one
    // column BEFORE its reader, a write node one column AFTER its writer, so a widely-read texture gets a
    // small node at each use site (like repeated power symbols) instead of one node fanning edges everywhere.
    // three kinds, all gated by the toolbar toggle:
    //   * history: create_history_image/_buffer make two resource nodes sharing a name: curr (historyIndex 0,
    //     written THIS frame for next) and prev (historyIndex 1, read this frame = LAST frame's curr; the
    //     pool rotates two physical textures/buffers). cross-frame, so no in-frame edge joins the pair.
    //   * external input: an IMPORTED resource read with no in-graph writer (importe_image'd / import_buffer'd,
    //     value from outside the frame). a texture gets a node per reader pin; a buffer (a uniform read almost
    //     everywhere) is ONE source node fanning faint edges to every reader, so it doesn't swamp the view.
    //   * present: an imported resource that IS written: the swapchain, whose final value leaves to display.
    // a virtual endpoint for a pass inside an expanded group parks in the column JUST OUTSIDE that group (left
    // for a read, right for a write), so the relax reserves it space outside the hull instead of placing it at
    // member_col +/-1 where it can land inside the border. uses the OUTERMOST expanded group (shallowest depth).
    static std::vector<int> vReadCol, vWriteCol; vReadCol.assign(n, 0); vWriteCol.assign(n, 0);
    for (int i = 0; i < n; ++i) { vReadCol[i] = colOf[i] - 1; vWriteCol[i] = colOf[i] + 1; }
    {
        static std::vector<int> sMin, sMax; sMin.assign(gtree.size(), 1 << 30); sMax.assign(gtree.size(), -1);
        for (int o = 1; o < (int)gtree.size(); ++o) {
            GNode& g = gtree[o]; if (!sv_length(g.prefix) || g.collapsed) continue;
            if (collOwner[g.gi] >= 0 && collOwner[g.gi] != o) continue;
            for (int k = g.gi; k < g.gj; ++k) { if (colOf[k] < sMin[o]) sMin[o] = colOf[k]; if (colOf[k] > sMax[o]) sMax[o] = colOf[k]; }
        }
        for (int i = 0; i < n; ++i) {
            if (collOwner[i] >= 0) continue;   // folded -> routes to a compact pin, not a hull pin
            int best = -1, bd = 1 << 30;
            for (int o = 1; o < (int)gtree.size(); ++o) {
                GNode& g = gtree[o]; if (!sv_length(g.prefix) || g.collapsed || sMax[o] < 0) continue;
                if (g.gi <= i && i < g.gj && g.depth < bd) { bd = g.depth; best = o; }
            }
            if (best >= 0) { vReadCol[i] = sMin[best] - 1; vWriteCol[i] = sMax[best] + 1; }
        }
    }
    struct TNode { bool isRead; int passBox, pin; ResourceNode* res; int col; float w, h; const char* cap; ImU32 tint; int li; };
    static std::vector<TNode> tnodes; tnodes.clear();
    auto push_tnode = [&](bool isRead, int passBox, int pin, ResourceNode* res, const char* cap, ImU32 tint) {
        char b[48]; std::snprintf(b, sizeof b, "%.*s", (int)res->id.name.length, res->id.name.data ? res->id.name.data : "?");
        ImVec2 ns = ImGui::CalcTextSize(b), cs = ImGui::CalcTextSize(cap);
        float w = ((ns.x > cs.x ? ns.x : cs.x) + 16) * zoom, h = (ns.y + cs.y + 10) * zoom;
        tnodes.push_back({ isRead, passBox, pin, res, isRead ? vReadCol[passBox] : vWriteCol[passBox], w, h, cap, tint, -1 });
    };
    if (showVirtual)
        for (ResourceNode* r = s.m_resouces; r; r = r->next) {
            if (r->persistent) {   // history: write node at the curr writer, read node at each prev reader
                bool curr = r->historyIndex == 0;
                for (int i = 0; i < n; ++i) {
                    int sl = curr ? rg_out_slot(box[i].p, r->handle.id) : rg_in_slot(box[i].p, r->handle.id);
                    if (sl >= 0) push_tnode(!curr, i, sl, r, curr ? "next frame" : "last frame", curr ? kRGWrite : kRGRead);
                }
                continue;
            }
            int lwb = -1, lws = -1;   // last writer, if any
            for (int i = 0; i < n; ++i) { int sl = rg_out_slot(box[i].p, r->handle.id); if (sl >= 0) { lwb = i; lws = sl; } }
            if (lwb < 0) {             // no in-graph writer: external input read from outside the frame
                if (r->imported && r->kind == ResourceKind::Buffer && fanBuffers) {
                    // collapsed: a uniform is read almost everywhere, so emit ONE source node anchored at its
                    // earliest reader (far left); the draw loop fans faint edges to every reader, so all
                    // consumers show without a node per use site swamping the view.
                    int best = -1, bestSl = -1;
                    for (int i = 0; i < n; ++i) { int sl = rg_in_slot(box[i].p, r->handle.id); if (sl >= 0 && (best < 0 || colOf[i] < colOf[best])) { best = i; bestSl = sl; } }
                    if (best >= 0) push_tnode(true, best, bestSl, r, "imported", kRGExt);
                }
                else if (r->imported)  // imported texture, or a buffer with fan-out off -> a node at each reader pin
                    for (int i = 0; i < n; ++i) { int sl = rg_in_slot(box[i].p, r->handle.id); if (sl >= 0) push_tnode(true, i, sl, r, "imported", kRGExt); }
            }
            else if (r->imported)      // present: imported + written -> sink node at the last writer
                push_tnode(false, lwb, lws, r, "present", kRGPresent);
        }
    // keep columns in [0, maxDist]: shift everything right if a read node landed left of 0, extend for writes.
    int tmin = 0; for (TNode& t : tnodes) if (t.col < tmin) tmin = t.col;
    if (tmin < 0) { for (int i = 0; i < n; ++i) colOf[i] -= tmin; maxDist -= tmin; for (TNode& t : tnodes) t.col -= tmin; }
    for (TNode& t : tnodes) if (t.col > maxDist) maxDist = t.col;

    // ===== layered routing: real boxes + dummy waypoints. an edge that skips columns gets a dummy in each
    // crossed column, reserving a lane there so it never hides behind a box (Sugiyama virtual nodes).
    struct LNode { int col, box, tn; float x, y; };       // box == -1 => dummy/history; tn >= 0 => history node
    struct REdge { int src, dst, sOut, dIn; uint32_t id; int chainN, chain[kRgDagMax]; };
    static std::vector<LNode> lnode; lnode.clear();
    static std::vector<REdge> edge;  edge.clear();
    for (int i = 0; i < n; ++i) lnode.push_back({ colOf[i], i, -1, 0, 0 });   // reals: lnode index == box index

    // drawn edges (a read pin fed by its producer + parallel-writer siblings), each with a dummy chain
    // through the columns it skips.
    for (int i = 0; i < n; ++i) {
        PassNode* p = box[i].p; int inS = 0;
        for (uint32_t k = 0; k < p->accessCount; ++k) {
            if (!rg_access_reads(p->accesses[k])) continue;
            uint32_t id = p->accesses[k].handle.id; int dIn = inS++;
            int prod = rg_producer_of(box, n, p, id);
            if (prod < 0) continue;
            for (int w = 0; w < n; ++w) {
                if (groupRep[w] != groupRep[prod] || colOf[w] >= colOf[i]) continue;
                int sOut = rg_out_slot(box[w].p, id);
                if (sOut < 0) continue;
                REdge e{ w, i, sOut, dIn, id, 0, {} };
                for (int c = colOf[w] + 1; c < colOf[i]; ++c) { e.chain[e.chainN++] = (int)lnode.size(); lnode.push_back({ c, -1, -1, 0, 0 }); }
                edge.push_back(e);
            }
        }
    }

    // history nodes join the layered list as their own (box == -1, tn >= 0) layered nodes.
    for (int ti = 0; ti < (int)tnodes.size(); ++ti) { tnodes[ti].li = (int)lnode.size(); lnode.push_back({ tnodes[ti].col, -1, ti, 0, 0 }); }

    // left/right neighbour lists for the barycenter (per-column members built after the per-component remap below).
    const int LN = (int)lnode.size();
    static std::vector<std::vector<int>> lleft, lright; lleft.assign(LN, {}); lright.assign(LN, {});
    for (REdge& e : edge) {
        int prev = e.src;
        for (int t = 0; t < e.chainN; ++t) { lright[prev].push_back(e.chain[t]); lleft[e.chain[t]].push_back(prev); prev = e.chain[t]; }
        lright[prev].push_back(e.dst); lleft[e.dst].push_back(prev);
    }
    // history adjacency: read node sits left of its reader, write node right of its writer. one link each,
    // enough for the barycenter to order them and y-relax to align them to the pass's row.
    for (TNode& t : tnodes)
        if (t.isRead) { lright[t.li].push_back(t.passBox); lleft[t.passBox].push_back(t.li); }
        else          { lright[t.passBox].push_back(t.li); lleft[t.li].push_back(t.passBox); }

    // per-lnode cluster (innermost expanded group) for the cluster-contiguity ordering: a real box carries
    // its pass's cluster; a dummy carries the deepest expanded group containing BOTH ends of its edge (so an
    // interior-edge lane stays inside the group's band and a boundary-edge lane stays outside); history = -1.
    static std::vector<int> lclus; lclus.assign(LN, -1);
    for (int i = 0; i < n; ++i) lclus[i] = clusterOf[i];
    {
        static std::vector<char> anc; anc.assign(gtree.size(), 0);
        auto lca = [&](int a, int b) -> int {
            if (a < 0 || b < 0) return -1;
            for (int x = a; x >= 0; x = clusterParent[x]) anc[x] = 1;
            int r = -1; for (int x = b; x >= 0; x = clusterParent[x]) if (anc[x]) { r = x; break; }
            for (int x = a; x >= 0; x = clusterParent[x]) anc[x] = 0;
            return r;
        };
        for (REdge& e : edge) { int ec = lca(clusterOf[e.src], clusterOf[e.dst]); for (int t = 0; t < e.chainN; ++t) lclus[e.chain[t]] = ec; }
    }

    // ---- independent work graphs: lay each out on its own. disconnected components (separate frame pipelines
    // sharing no resource) were right-anchored to the global sink column, so they shared column buckets; the
    // barycenter + column-width passes then cross-influenced them. union nodes over the adjacency, then give
    // each component its OWN disjoint, dense column range starting at its base. a component is gap-free within
    // its span (every skipped column carries an edge dummy), so base + (col - min) densely remaps it and the
    // per-component ranges sum to <= n columns. now each column bucket holds one component => the relax + colW
    // see only that component; the x write left-aligns each to its base, and a vertical band separates them.
    static std::vector<int> comp; comp.assign(LN, 0);
    static std::vector<int> compMinCol; compMinCol.assign(LN, 1 << 30);
    {
        for (int li = 0; li < LN; ++li) comp[li] = li;
        auto cfind = [&](int x) { while (comp[x] != x) { comp[x] = comp[comp[x]]; x = comp[x]; } return x; };
        for (int li = 0; li < LN; ++li) { for (int x : lleft[li]) comp[cfind(li)] = cfind(x); for (int x : lright[li]) comp[cfind(li)] = cfind(x); }
        // a group run lays out as one bordered region -> force it into one component. clear-only / parallel-
        // writer members (cube's 6 face clears) have no in-graph read edge, so each would otherwise split into
        // its own component and the banding would stack them, shoving the group's consumers far down/over.
        for (int ga = 0; ga < n;) {
            WGPUStringView pre = group_prefix(box[ga].p->id.name);
            int gb = ga + 1;
            while (gb < n && sv_length(pre) && sv_eq(group_prefix(box[gb].p->id.name), pre)) ++gb;
            if (sv_length(pre) && gb - ga >= 2) for (int k = ga + 1; k < gb; ++k) comp[cfind(ga)] = cfind(k);
            ga = gb;
        }
        for (int li = 0; li < LN; ++li) comp[li] = cfind(li);
        // compact each component root to a first-seen id (reals come first, so this is exec order).
        static std::vector<int> cid; cid.assign(LN, -1); int numComp = 0;
        for (int li = 0; li < LN; ++li) if (cid[comp[li]] < 0) cid[comp[li]] = numComp++;
        static std::vector<int> cMin, cMax; cMin.assign(numComp, 1 << 30); cMax.assign(numComp, -1);
        for (int li = 0; li < LN; ++li) { int c = cid[comp[li]], oc = lnode[li].col; if (oc < cMin[c]) cMin[c] = oc; if (oc > cMax[c]) cMax[c] = oc; }
        static std::vector<int> cBase; cBase.assign(numComp, 0); int acc = 0;
        for (int c = 0; c < numComp; ++c) { cBase[c] = acc; acc += cMax[c] - cMin[c] + 1; }
        for (int li = 0; li < LN; ++li) { int c = cid[comp[li]]; lnode[li].col = cBase[c] + (lnode[li].col - cMin[c]); }
        for (int i = 0; i < n; ++i) colOf[i] = lnode[i].col;   // reals: lnode i == box i; keep colW indexing in sync
        maxDist = acc > 0 ? acc - 1 : 0;
        for (int li = 0; li < LN; ++li) if (lnode[li].col < compMinCol[comp[li]]) compMinCol[comp[li]] = lnode[li].col;
    }

    // per-column members (reals + dummies + history), now that columns are final per-component.
    static std::vector<int> lcol[kRgDagMax];
    for (int c = 0; c <= maxDist; ++c) lcol[c].clear();
    for (int li = 0; li < LN; ++li) lcol[lnode[li].col].push_back(li);

    // barycenter crossing-reduction + y-relax, factored out so each expanded group's interior reuses it.
    // materialize node heights first: a box -> its reserved height, a history node -> its own, dummy -> a lane.
    const float kLane = 16.0f * zoom;
    static std::vector<float> lh; lh.assign(LN, 0.0f);
    for (int li = 0; li < LN; ++li) { int b = lnode[li].box; lh[li] = b >= 0 ? effH[b] : (lnode[li].tn >= 0 ? tnodes[lnode[li].tn].h : kLane); }
    static std::vector<float> cy; cy.assign(LN, 0.0f);
    rg_barycenter_relax(lcol, maxDist, lleft, lright, lh, cy, kRowGap, lclus.data(), clusterParent.data(), (int)gtree.size());

    // stack the components into disjoint vertical bands (the relax floats every column from y=0). running cursor,
    // first-seen order, each component shifted below the previous one's extent.
    {
        static std::vector<float> cTop, cBot, cOff; static std::vector<char> seen;
        cTop.assign(LN, 1e30f); cBot.assign(LN, -1e30f); cOff.assign(LN, 0.0f); seen.assign(LN, 0);
        for (int li = 0; li < LN; ++li) { float t = cy[li] - lh[li] * 0.5f, b = cy[li] + lh[li] * 0.5f; if (t < cTop[comp[li]]) cTop[comp[li]] = t; if (b > cBot[comp[li]]) cBot[comp[li]] = b; }
        float cursor = 0.0f;
        for (int li = 0; li < LN; ++li) { int c = comp[li]; if (seen[c]) continue; seen[c] = 1; cOff[c] = cursor - cTop[c]; cursor += (cBot[c] - cTop[c]) + kRowGap * 2.0f; }
        for (int li = 0; li < LN; ++li) cy[li] += cOff[comp[li]];
    }

    // exclusivity: shove non-members out of each expanded group's vertical band so the hull (drawn at the
    // member bbox +/- padding) encloses ONLY its own members, e.g. a history node (taa.history) that the
    // relax floated into a column the group spans gets pushed clear of the border. deepest-first, so a node
    // ends up outside the OUTERMOST band it intruded on. O(clusters * LN), trivial here and only a debug view.
    {
        auto inSub = [&](int li, int anc) { for (int c = lclus[li]; c >= 0; c = clusterParent[c]) if (c == anc) return true; return false; };
        for (int o = (int)gtree.size() - 1; o >= 1; --o) {
            GNode& g = gtree[o];
            if (!sv_length(g.prefix) || g.collapsed) continue;          // only expanded groups draw a hull
            if (collOwner[g.gi] >= 0 && collOwner[g.gi] != o) continue;  // shadowed by a collapsed ancestor -> not drawn
            float bandTop = 1e30f, bandBot = -1e30f; int colMin = 1 << 30, colMax = -1;
            for (int li = 0; li < LN; ++li) if (inSub(li, o)) {
                if (cy[li] - lh[li] * 0.5f < bandTop) bandTop = cy[li] - lh[li] * 0.5f;
                if (cy[li] + lh[li] * 0.5f > bandBot) bandBot = cy[li] + lh[li] * 0.5f;
                int c = lnode[li].col; if (c < colMin) colMin = c; if (c > colMax) colMax = c;
            }
            if (colMax < colMin) continue;
            float hullTop = bandTop - kRegionPad - kHeaderH, hullBot = bandBot + kRegionPad, mid = (bandTop + bandBot) * 0.5f;
            const float kClear = kRowGap * 0.5f;
            for (int li = 0; li < LN; ++li) {
                if (inSub(li, o)) continue;                                      // a member / interior dummy belongs inside
                if (lnode[li].col < colMin || lnode[li].col > colMax) continue;  // not under the hull
                // a node wired to a group member (a history/present endpoint or a boundary-edge dummy) belongs
                // AT the border next to its member, so don't evict it.
                bool linked = false;
                for (int x : lleft[li])  if (inSub(x, o)) { linked = true; break; }
                if (!linked) for (int x : lright[li]) if (inSub(x, o)) { linked = true; break; }
                if (linked) continue;
                float t = cy[li] - lh[li] * 0.5f, b = cy[li] + lh[li] * 0.5f;
                if (b <= hullTop || t >= hullBot) continue;                      // already clear
                cy[li] = cy[li] < mid ? hullTop - lh[li] * 0.5f - kClear : hullBot + lh[li] * 0.5f + kClear;
            }
        }
    }

    // write positions + the graph's bounding box (canvas-local), used to centre the view.
    float gxMin = 1e30f, gyMin = 1e30f, gxMax = -1e30f, gyMax = -1e30f;
    // every node is one box wide (expanded groups don't reserve a wide region; their members are
    // real boxes in their own columns), so column x is the uniform c*(kBoxW+kColGap) cadence.
    float colW[kRgDagMax], colX[kRgDagMax];
    for (int c = 0; c <= maxDist; ++c) colW[c] = kBoxW;
    colX[0] = 0; for (int c = 1; c <= maxDist; ++c) colX[c] = colX[c - 1] + colW[c - 1] + kColGap;
    for (int c = 0; c <= maxDist; ++c) {
        for (int li : lcol[c]) {
            float cx = colX[c] - colX[compMinCol[comp[li]]];   // left-align this component to x=0
            if (lnode[li].box >= 0) {
                int b = lnode[li].box; box[b].tl = ImVec2(cx, cy[li] - effH[b] * 0.5f); box[b].layer = c;
                if (cx < gxMin) gxMin = cx; if (cx + box[b].w > gxMax) gxMax = cx + box[b].w;
                if (box[b].tl.y < gyMin) gyMin = box[b].tl.y; if (box[b].tl.y + effH[b] > gyMax) gyMax = box[b].tl.y + effH[b];
            }
            else {
                lnode[li].x = cx; lnode[li].y = cy[li];   // dummy: column left edge + lane centre
                int t = lnode[li].tn;
                if (t >= 0) {   // history node: centre in the column slot, include in the view bbox
                    float ctr = cx + kBoxW * 0.5f, hw = tnodes[t].w * 0.5f, hh = tnodes[t].h * 0.5f;
                    lnode[li].x = ctr;
                    if (ctr - hw < gxMin) gxMin = ctr - hw; if (ctr + hw > gxMax) gxMax = ctr + hw;
                    if (cy[li] - hh < gyMin) gyMin = cy[li] - hh; if (cy[li] + hh > gyMax) gyMax = cy[li] + hh;
                }
            }
        }
    }
    if (gxMax < gxMin) { gxMin = gyMin = gxMax = gyMax = 0; }   // empty graph

    // ---- IR mirror: build RgNodes (pass nodes index-aligned to box[], then Virtual endpoints) + RgEdges.
    // the draw consumes this for the nodes + virtual links; the pass-graph (box[]/edge[]) still drives layout
    // and the solid pass-edge routing. layout() producing the IR directly (retiring box[]) is future work.
    static std::vector<RgNode> rgn; rgn.clear();
    static std::vector<RgEdge> rge; rge.clear();
    for (int i = 0; i < n; ++i) {
        RgNode nd{}; nd.kind = RgNode::Kind::Pass; nd.pass = box[i].p;
        nd.label = box[i].p->id.name; nd.pos = box[i].tl; nd.w = box[i].w; nd.h = box[i].h; nd.col = box[i].layer;
        rgn.push_back(nd);
    }
    for (REdge& e : edge) {
        RgEdge ge{}; ge.srcNode = e.src; ge.dstNode = e.dst;
        ge.srcPin = (uint16_t)e.sOut; ge.dstPin = (uint16_t)e.dIn; ge.resId = e.id; ge.kind = RgEdge::Kind::Raw;
        rge.push_back(ge);
    }
    // virtual endpoints mirror into the IR: one Virtual RgNode at its laid-out spot + an RgEdge per link to
    // the anchor pass pin (a fan source emits one edge per reader). lets the draw + cones treat them like any
    // node, retiring the parallel tnodes draw path. additive here: the draw still reads tnodes.
    for (TNode& t : tnodes) {
        int vi = (int)rgn.size();
        RgNode nd{}; nd.kind = RgNode::Kind::Virtual; nd.res = t.res;
        nd.label = WGPUStringView{ t.cap, t.cap ? strlen(t.cap) : 0 };
        nd.pos = ImVec2(lnode[t.li].x - t.w * 0.5f, lnode[t.li].y - t.h * 0.5f);
        nd.w = t.w; nd.h = t.h; nd.col = t.col; nd.tint = t.tint;
        rgn.push_back(nd);
        RgEdge::Kind ek = t.res->persistent ? RgEdge::Kind::History : RgEdge::Kind::Raw;
        bool fan = fanBuffers && t.isRead && t.res->imported && t.res->kind == ResourceKind::Buffer;
        if (fan)
            for (int i = 0; i < n; ++i) { int sl = rg_in_slot(box[i].p, t.res->handle.id); if (sl < 0) continue;
                rge.push_back(RgEdge{ vi, i, 0, (uint16_t)sl, t.res->handle.id, RgEdge::Kind::Fanout }); }
        else if (t.isRead) rge.push_back(RgEdge{ vi, t.passBox, 0, (uint16_t)t.pin, t.res->handle.id, ek });
        else               rge.push_back(RgEdge{ t.passBox, vi, (uint16_t)t.pin, 0, t.res->handle.id, ek });
    }

    // toolbar: reset/centre the view + a name filter that dims non-matching passes.
    static char filter[64] = "";
    bool doReset = ImGui::Button("Reset view");
    ImGui::SameLine(); ImGui::SetNextItemWidth(180);
    ImGui::InputTextWithHint("##rgfilter", "filter passes...", filter, sizeof filter);
    ImGui::SameLine(); ImGui::Checkbox("virtual nodes", &showVirtual);
    ImGui::SameLine(); ImGui::Checkbox("fan-out buffers", &fanBuffers);   // off -> one virtual node per buffer use site
    ImGui::SameLine(); ImGui::Checkbox("collapse groups", &collapseDefault);   // default state; a group header click overrides
    const bool filterActive = filter[0] != 0;

    // pannable canvas: drag left/middle to pan, wheel to zoom (handled at the top, before layout) + a grid,
    // after the imgui node-graph example. no scrollbar: navigation is panning, so a big graph isn't boxed in.
    ImGui::BeginChild("rg_canvas", ImVec2(0, 0), true,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoMove);
    ImGui::SetWindowFontScale(zoom);   // text scales with the canvas; restored to 1.0 before the overlays
    ImGuiIO& io = ImGui::GetIO();
    const ImVec2 winPos = ImGui::GetCursorScreenPos();
    const ImVec2 winSize = ImGui::GetContentRegionAvail();
    ImGui::InvisibleButton("canvas", ImVec2(winSize.x > 0 ? winSize.x : 1, winSize.y > 0 ? winSize.y : 1));
    const bool canvasHovered = ImGui::IsItemHovered();
    const bool canvasActive = ImGui::IsItemActive();
    s_canvasHovered = canvasHovered; s_winPos = winPos;   // for next frame's wheel zoom (resolved pre-layout)
    bool panned = false;
    if (canvasActive && (ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f) || ImGui::IsMouseDragging(ImGuiMouseButton_Middle, 0.0f))) {
        scrolling.x += io.MouseDelta.x; scrolling.y += io.MouseDelta.y; userMoved = true; panned = true;
    }
    // snap to the graph's top-left corner (small margin) until the user pans. this rides through the first
    // frames where the child's content region is still settling. the Reset button + a double-click re-arm it.
    if (doReset || (canvasHovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))) userMoved = false;
    if (doReset) zoom = 1.0f;   // Reset view also returns to the closest zoom
    if (!userMoved && !panned && winSize.x > 1 && winSize.y > 1) {
        const float kCornerPad = 40.0f;
        scrolling.x = kCornerPad - gxMin;
        scrolling.y = kCornerPad - gyMin;
    }
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 origin(winPos.x + scrolling.x, winPos.y + scrolling.y);   // panned top-left; node coords add this

    // faint grid, scrolling with the canvas.
    const float kGrid = 48.0f;
    for (float gx = std::fmod(scrolling.x, kGrid); gx < winSize.x; gx += kGrid)
        dl->AddLine(ImVec2(winPos.x + gx, winPos.y), ImVec2(winPos.x + gx, winPos.y + winSize.y), IM_COL32(255, 255, 255, 14));
    for (float gy = std::fmod(scrolling.y, kGrid); gy < winSize.y; gy += kGrid)
        dl->AddLine(ImVec2(winPos.x, winPos.y + gy), ImVec2(winPos.x + winSize.x, winPos.y + gy), IM_COL32(255, 255, 255, 14));

    // ---- group runs: a contiguous run of passes sharing a dotted-name prefix (shadow.cascade x3,
    // bloom.extract/down/up/composite) is one logical group. expanded -> a labelled frame behind the
    // member boxes (keyed on the prefix, so bloom.* reads as one); collapsed -> a single synthetic node
    // (drawn later, on top of the edges) carrying the group's external read/write pins. the collapse flag
    // is a tri-state in ImGuiStorage (0 follow-default, 1 force-open, 2 force-closed) so it survives the
    // per-frame rebuild and the toolbar default can flip every group at once.
    struct GView {
        int gi, gj, depth; WGPUStringView prefix; bool collapsed; ImGuiID key;   // depth 1 = top-level group, >=2 = subgroup
        ImVec2 bb0, bb1, h0, h1;                                    // box rect + header (click) rect, screen
        int nIn, nOut; uint32_t inId[kRgGPinMax], outId[kRgGPinMax];
        ImVec2 inC[kRgGPinMax], outC[kRgGPinMax]; bool inDrawn[kRgGPinMax];
    };
    static std::vector<GView> groups; groups.clear();
    static std::vector<int> nodeGView; nodeGView.assign(gtree.size(), -1);   // gtree node index -> its GView in groups (subgroups), else -1
    int groupOf[kRgDagMax]; for (int i = 0; i < n; ++i) groupOf[i] = -1;
    // y of an expanded group's interface pin: the mean y-centre of the members touching that resource, so the
    // pin sits beside them instead of top-aligned (top-aligned pins forced every edge to arc up to the header).
    auto groupPinY = [&](int gi, int gj, uint32_t id, bool write, float fb) -> float {
        float sy = 0; int c = 0;
        for (int k = gi; k < gj; ++k) if ((write ? rg_out_slot(rgn[k].pass, id) : rg_in_slot(rgn[k].pass, id)) >= 0) { sy += origin.y + rgn[k].pos.y + rgn[k].h * 0.5f; ++c; }
        return c ? sy / c : fb;
    };
    // de-overlap a column of pins: members in one horizontal row all want the SAME aligned y, which would stack
    // every pin on one point. sort by desired y, enforce a min gap, then recentre the run on the original mean.
    auto spreadPins = [&](ImVec2* pins, int cnt, float gap) {
        if (cnt < 2) return;
        int idx[kRgGPinMax]; for (int i = 0; i < cnt; ++i) idx[i] = i;
        for (int a = 1; a < cnt; ++a) { int v = idx[a], b = a - 1; while (b >= 0 && pins[idx[b]].y > pins[v].y) { idx[b + 1] = idx[b]; --b; } idx[b + 1] = v; }
        float before = 0, after = 0, prev = -1e30f;
        for (int i = 0; i < cnt; ++i) before += pins[i].y;
        for (int i = 0; i < cnt; ++i) { int p = idx[i]; if (pins[p].y < prev + gap) pins[p].y = prev + gap; prev = pins[p].y; after += pins[p].y; }
        float shift = (before - after) / cnt;
        for (int i = 0; i < cnt; ++i) pins[i].y += shift;
    };
    for (int gi = 0; gi < n;) {
        WGPUStringView pre = group_prefix(box[gi].p->id.name);
        int gj = gi + 1;
        while (gj < n && sv_length(pre) && sv_eq(group_prefix(box[gj].p->id.name), pre)) ++gj;
        if (!(sv_length(pre) && gj - gi >= 2)) { gi = gj; continue; }

        float x0 = 1e30f, y0 = 1e30f, x1 = -1e30f, y1 = -1e30f;
        for (int k = gi; k < gj; ++k) {
            float ax = origin.x + box[k].tl.x, ay = origin.y + box[k].tl.y;
            if (ax < x0) x0 = ax; if (ay < y0) y0 = ay;
            if (ax + box[k].w > x1) x1 = ax + box[k].w; if (ay + box[k].h > y1) y1 = ay + box[k].h;
        }

        GView g{}; g.gi = gi; g.gj = gj; g.depth = 1; g.prefix = pre; g.key = rg_grp_key(pre);
        int st = grpStore->GetInt(g.key, 0);
        g.collapsed = st == 2 ? true : st == 1 ? false : collapseDefault;

        // external interface pins, computed for collapsed AND expanded groups: a member read whose producer
        // is outside the group -> in-pin; an external/imported/persistent member write -> out-pin. interior
        // resources (produced + consumed within the group, e.g. bloom's mips) get no pin.
        rg_group_interface(rg, box, n, gi, gj, g.inId, g.nIn, g.outId, g.nOut);

        if (!g.collapsed) {
            // expanded -> a hull border around the member boxes. members are first-class nodes placed by the one
            // Sugiyama pass; the cluster-contiguity ordering keeps them a tight block, so the bbox (x0..y1, taken
            // above) is theirs alone. pad past any nested subgroup border (kRegionPad > kSubPad) to enclose it.
            ImU32 gcol = group_color(pre);
            ImVec2 a(x0 - kRegionPad, y0 - kRegionPad - kHeaderH), b(x1 + kRegionPad, y1 + kRegionPad);
            dl->AddRectFilled(a, b, rg_with_alpha(gcol, 22), 6.0f);
            dl->AddRect(a, b, gcol, 6.0f, 0, 2.0f);
            char lbl[64]; std::snprintf(lbl, sizeof lbl, "[-] %.*s  x%d", (int)sv_length(pre), pre.data, gj - gi);
            dl->AddText(ImVec2(a.x + 6, a.y + 3), gcol, lbl);
            g.bb0 = a; g.bb1 = b; g.h0 = a; g.h1 = ImVec2(b.x, a.y + kHeaderH);
            // interface pins on the hull edges (reads left, writes right), each aligned to the members it serves
            // so boundary edges run roughly straight across instead of arcing up to a top-aligned row.
            for (int s = 0; s < g.nIn; ++s)  g.inC[s]  = ImVec2(a.x, groupPinY(gi, gj, g.inId[s],  false, a.y + kHeaderH + s * kPinRowH + kPinRowH * 0.5f));
            for (int s = 0; s < g.nOut; ++s) g.outC[s] = ImVec2(b.x, groupPinY(gi, gj, g.outId[s], true,  a.y + kHeaderH + s * kPinRowH + kPinRowH * 0.5f));
            spreadPins(g.inC, g.nIn, kPinRowH); spreadPins(g.outC, g.nOut, kPinRowH);
        }
        else {
            int rows = g.nIn > g.nOut ? g.nIn : g.nOut;
            float needH = kHeaderH + (rows ? rows * kPinRowH : kMinBodyH) + kFooterH;
            // collapse to a single node over the FIRST member's slot (already an overlap-free reserved
            // position), not the member bbox; a multi-column group's bbox would span and cover the nodes
            // laid out between its members. the other members' slots simply vacate. NOTE(Huerbe): no relayout,
            // so a group with more pins than the anchor pass's rows grows down and can still touch a
            // neighbour; promote collapsed groups to single layout nodes if that ever bites.
            ImVec2 a(origin.x + rgn[gi].pos.x, origin.y + rgn[gi].pos.y);
            ImVec2 b(a.x + kBoxW, a.y + needH);   // == effH[gi], the slot reserved before layout
            g.bb0 = a; g.bb1 = b; g.h0 = a; g.h1 = ImVec2(b.x, a.y + kHeaderH);
            for (int s = 0; s < g.nIn; ++s)  g.inC[s]  = ImVec2(a.x, a.y + kHeaderH + s * kPinRowH + kPinRowH * 0.5f);
            for (int s = 0; s < g.nOut; ++s) g.outC[s] = ImVec2(b.x, a.y + kHeaderH + s * kPinRowH + kPinRowH * 0.5f);
        }

        int idxG = (int)groups.size(); groups.push_back(g);
        for (int k = gi; k < gj; ++k) groupOf[k] = idxG;
        // map this top-level group to its gtree node too, so collOwner-keyed lookups (memberOut/memberIn) find
        // a collapsed top-level group's compact pin, not just subgroups. (without this, collapsing a top-level
        // group sends its members' edges to their own undrawn pins -> broken connections.)
        for (int gn = 1; gn < (int)gtree.size(); ++gn) if (gtree[gn].depth == 1 && gtree[gn].gi == gi && gtree[gn].gj == gj) { nodeGView[gn] = idxG; break; }
        gi = gj;
    }

    // subgroups as first-class groups: every gtree node below the top level becomes a GView too, so the same
    // draw / click / pin passes treat it exactly like a top-level group. collapsed -> a compact node (members
    // hidden); expanded -> a bordered region with border pins. tree-driven, nests to any depth.
    // pass 1: rects, deepest-first (reverse gtree order = children before parents) so a parent's border encloses
    // its child borders with a gap and members never sit flush. pass 2 builds the GViews parent-first.
    static std::vector<ImVec2> subA, subB; subA.assign(gtree.size(), ImVec2(0, 0)); subB.assign(gtree.size(), ImVec2(0, 0));
    const float kSubPad = 11.0f * zoom;   // gap from a subgroup border to its members / nested child borders
    for (int gni = (int)gtree.size() - 1; gni >= 1; --gni) {
        GNode& g2 = gtree[gni];
        if (g2.depth < 2 || !sv_length(g2.prefix)) continue;
        if (collOwner[g2.gi] >= 0 && collOwner[g2.gi] != gni) continue;   // inside a collapsed ancestor -> not drawn
        if (g2.collapsed) {   // compact node over the rep cell (sized compact at layout time via repH)
            uint32_t ii[kRgGPinMax], oo[kRgGPinMax]; int ni = 0, no = 0; rg_group_interface(rg, box, n, g2.gi, g2.gj, ii, ni, oo, no);
            int rows = ni > no ? ni : no;
            ImVec2 a(origin.x + rgn[g2.gi].pos.x, origin.y + rgn[g2.gi].pos.y);
            subA[gni] = a; subB[gni] = ImVec2(a.x + kBoxW, a.y + kHeaderH + (rows ? rows * kPinRowH : kMinBodyH) + kFooterH);
            continue;
        }
        float x0 = 1e30f, y0 = 1e30f, x1 = -1e30f, y1 = -1e30f;
        for (int k = g2.gi; k < g2.gj; ++k) {
            if (collOwner[k] >= 0 && k != gtree[collOwner[k]].gi) continue;   // folded member -> not framed
            float kx = origin.x + rgn[k].pos.x, ky = origin.y + rgn[k].pos.y;
            if (kx < x0) x0 = kx; if (ky < y0) y0 = ky;
            if (kx + rgn[k].w > x1) x1 = kx + rgn[k].w; if (ky + rgn[k].h > y1) y1 = ky + rgn[k].h;
        }
        for (int kid : g2.kids) {   // enclose child subgroup borders (computed already, this pass runs deepest-first)
            if (subB[kid].x < subA[kid].x) continue;
            if (subA[kid].x < x0) x0 = subA[kid].x; if (subA[kid].y < y0) y0 = subA[kid].y;
            if (subB[kid].x > x1) x1 = subB[kid].x; if (subB[kid].y > y1) y1 = subB[kid].y;
        }
        if (x1 < x0) continue;
        subA[gni] = ImVec2(x0 - kSubPad, y0 - kSubPad - kHeaderH); subB[gni] = ImVec2(x1 + kSubPad, y1 + kSubPad);
    }
    for (int gni = 1; gni < (int)gtree.size(); ++gni) {
        GNode& g2 = gtree[gni];
        if (g2.depth < 2 || !sv_length(g2.prefix)) continue;
        if (collOwner[g2.gi] >= 0 && collOwner[g2.gi] != gni) continue;
        if (subB[gni].x < subA[gni].x) continue;   // nothing visible
        GView g{}; g.gi = g2.gi; g.gj = g2.gj; g.depth = g2.depth; g.prefix = g2.prefix;
        g.key = rg_grp_key(g2.prefix); g.collapsed = g2.collapsed;
        rg_group_interface(rg, box, n, g2.gi, g2.gj, g.inId, g.nIn, g.outId, g.nOut);
        ImVec2 a = subA[gni], b = subB[gni];
        g.bb0 = a; g.bb1 = b; g.h0 = a; g.h1 = ImVec2(b.x, a.y + kHeaderH);
        if (!g.collapsed) {   // expanded: bordered region + label (the compact-node pass draws the collapsed ones)
            ImU32 sc = group_color(g2.prefix);
            dl->AddRect(a, b, sc, 5.0f, 0, 1.5f);
            WGPUStringView sn = g2.prefix; size_t off = 0; for (size_t i = 0; i < sv_length(sn); ++i) if (sn.data[i] == '.') off = i + 1;
            char slbl[48]; std::snprintf(slbl, sizeof slbl, "[-] %.*s x%d", (int)(sv_length(sn) - off), sn.data + off, g2.gj - g2.gi);
            dl->AddText(ImVec2(a.x + 4, a.y + 1), sc, slbl);
        }
        // expanded: pins aligned to the members they serve (so edges run across, not up to the header); collapsed:
        // top-aligned in the compact box.
        for (int s = 0; s < g.nIn; ++s)  g.inC[s]  = ImVec2(a.x, g.collapsed ? a.y + kHeaderH + s * kPinRowH + kPinRowH * 0.5f : groupPinY(g2.gi, g2.gj, g.inId[s],  false, a.y + kHeaderH + s * kPinRowH + kPinRowH * 0.5f));
        for (int s = 0; s < g.nOut; ++s) g.outC[s] = ImVec2(b.x, g.collapsed ? a.y + kHeaderH + s * kPinRowH + kPinRowH * 0.5f : groupPinY(g2.gi, g2.gj, g.outId[s], true,  a.y + kHeaderH + s * kPinRowH + kPinRowH * 0.5f));
        if (!g.collapsed) { spreadPins(g.inC, g.nIn, kPinRowH); spreadPins(g.outC, g.nOut, kPinRowH); }
        nodeGView[gni] = (int)groups.size();
        groups.push_back(g);
    }

    // pin centres, screen space. reads fill input slots (left) in encounter order; writes fill output
    // slots (right) the same way.
    auto inPin = [&](int b, int slot) { return ImVec2(origin.x + rgn[b].pos.x,
        origin.y + rgn[b].pos.y + kHeaderH + slot * kPinRowH + kPinRowH * 0.5f); };
    auto outPin = [&](int b, int slot) { return ImVec2(origin.x + rgn[b].pos.x + rgn[b].w,
        origin.y + rgn[b].pos.y + kHeaderH + slot * kPinRowH + kPinRowH * 0.5f); };
    // screen-space polyline of an edge (src out-pin, two points per dummy, dst in-pin); shared by hit-test + draw.
    auto edgePoints = [&](const REdge& e, ImVec2* pts) -> int {
        int np = 0;
        pts[np++] = outPin(e.src, e.sOut);
        for (int t = 0; t < e.chainN; ++t) { float lx = origin.x + lnode[e.chain[t]].x, ly = origin.y + lnode[e.chain[t]].y; pts[np++] = ImVec2(lx, ly); pts[np++] = ImVec2(lx + kBoxW, ly); }
        pts[np++] = inPin(e.dst, e.dIn);
        return np;
    };
    bool matchBox[kRgDagMax];
    for (int i = 0; i < n; ++i) matchBox[i] = rg_name_has(rgn[i].pass->id.name, filter);

    // ---- find the single hovered pin (manual rect test: pins are small + overlap the box button).
    int hovB = -1, hovSlot = -1; bool hovWrite = false; uint32_t hovId = 0, hovMip = 0, hovLayer = 0; AccessType hovType{};
    if (canvasHovered && !canvasActive) {
        for (int i = 0; i < n && hovB < 0; ++i) {
            if (drawHidden[i]) continue;   // hidden inside a collapsed group
            int inS = 0, outS = 0; PassNode* p = rgn[i].pass;
            for (uint32_t k = 0; k < p->accessCount && hovB < 0; ++k) {
                const ResourceAccess& acc = p->accesses[k];
                if (rg_access_reads(acc)) {   // input pin (left)
                    ImVec2 c = inPin(i, inS);
                    if (ImGui::IsMouseHoveringRect(ImVec2(c.x - kPinHit, c.y - kPinHit), ImVec2(c.x + kPinHit, c.y + kPinHit)))
                        { hovB = i; hovSlot = inS; hovWrite = false; hovId = acc.handle.id; hovType = acc.type; hovMip = acc.baseMip; hovLayer = acc.baseLayer; }
                    ++inS;
                }
                if (hovB < 0 && access_is_write(acc.type)) {   // output pin (right)
                    ImVec2 c = outPin(i, outS);
                    if (ImGui::IsMouseHoveringRect(ImVec2(c.x - kPinHit, c.y - kPinHit), ImVec2(c.x + kPinHit, c.y + kPinHit)))
                        { hovB = i; hovSlot = outS; hovWrite = true; hovId = acc.handle.id; hovType = acc.type; hovMip = acc.baseMip; hovLayer = acc.baseLayer; }
                    ++outS;
                }
            }
        }
    }
    // ---- group border pins (collapsed compact node AND expanded region edge): hover/select them like a
    // member's pin so cone + tooltip + lock work. map the border pin to a representative member that reads/
    // writes that resource, then reuse the normal hovered-pin path below. (expanded member pins are caught
    // directly by the per-pass test above; this adds the border pins, which sit on the region edge.)
    if (hovB < 0 && canvasHovered && !canvasActive)
        for (GView& g : groups) {
            if (hovB >= 0) continue;
            // both collapsed (compact node) and expanded (hull) groups carry selectable interface pins now.
            for (int s = 0; s < g.nIn && hovB < 0; ++s) {
                ImVec2 c = g.inC[s];
                if (!ImGui::IsMouseHoveringRect(ImVec2(c.x - kPinHit, c.y - kPinHit), ImVec2(c.x + kPinHit, c.y + kPinHit))) continue;
                for (int k = g.gi; k < g.gj && hovB < 0; ++k) { PassNode* mp = rgn[k].pass;
                    for (uint32_t ai = 0; ai < mp->accessCount; ++ai)
                        if (mp->accesses[ai].handle.id == g.inId[s] && rg_access_reads(mp->accesses[ai]))
                            { hovB = k; hovWrite = false; hovId = g.inId[s]; hovSlot = rg_in_slot(mp, g.inId[s]); hovType = mp->accesses[ai].type; hovMip = mp->accesses[ai].baseMip; hovLayer = mp->accesses[ai].baseLayer; break; }
                }
            }
            for (int s = 0; s < g.nOut && hovB < 0; ++s) {
                ImVec2 c = g.outC[s];
                if (!ImGui::IsMouseHoveringRect(ImVec2(c.x - kPinHit, c.y - kPinHit), ImVec2(c.x + kPinHit, c.y + kPinHit))) continue;
                for (int k = g.gi; k < g.gj && hovB < 0; ++k) { PassNode* mp = rgn[k].pass;
                    for (uint32_t ai = 0; ai < mp->accessCount; ++ai)
                        if (mp->accesses[ai].handle.id == g.outId[s] && access_is_write(mp->accesses[ai].type))
                            { hovB = k; hovWrite = true; hovId = g.outId[s]; hovSlot = rg_out_slot(mp, g.outId[s]); hovType = mp->accesses[ai].type; hovMip = mp->accesses[ai].baseMip; hovLayer = mp->accesses[ai].baseLayer; break; }
                }
            }
        }

    // hovered box: only used for the fallback reads/writes tooltip when no pin caught the mouse.
    int hovBox = -1;
    if (hovB < 0 && canvasHovered && !canvasActive)
        for (int i = 0; i < n; ++i) {
            if (drawHidden[i]) continue;   // hidden inside a collapsed group
            ImVec2 tl(origin.x + rgn[i].pos.x, origin.y + rgn[i].pos.y);
            if (ImGui::IsMouseHoveringRect(tl, ImVec2(tl.x + rgn[i].w, tl.y + rgn[i].h))) { hovBox = i; break; }
        }

    // ---- click a pin to LOCK its cone (stays without holding the mouse); click empty canvas to clear.
    static int lockB = -1, lockSlot = -1; static bool lockWrite = false; static uint32_t lockId = 0;
    static bool pressed = false; static ImVec2 pressAt;
    if (canvasHovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) { pressed = true; pressAt = io.MousePos; }
    if (pressed && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
        pressed = false;
        float mdx = io.MousePos.x - pressAt.x, mdy = io.MousePos.y - pressAt.y;
        if (mdx * mdx + mdy * mdy < 16) {   // a click, not a pan
            int hitG = -1;   // a group header click toggles that group's collapse state, overriding the default
            for (int gI = 0; gI < (int)groups.size(); ++gI)
                if (io.MousePos.x >= groups[gI].h0.x && io.MousePos.x <= groups[gI].h1.x &&
                    io.MousePos.y >= groups[gI].h0.y && io.MousePos.y <= groups[gI].h1.y) hitG = gI;
            if (hitG >= 0) grpStore->SetInt(groups[hitG].key, groups[hitG].collapsed ? 1 : 2);   // flip effective state
            else if (hovB >= 0) {
                if (lockB == hovB && lockWrite == hovWrite && lockSlot == hovSlot) lockB = -1;   // toggle off
                else { lockB = hovB; lockWrite = hovWrite; lockSlot = hovSlot; lockId = hovId; }
            }
            else if (hovBox >= 0) {   // a pass-body click selects the whole pass (lockSlot -1 / lockId 0 = no pin)
                if (lockB == hovBox && lockSlot == -1) lockB = -1;   // toggle off
                else { lockB = hovBox; lockWrite = false; lockSlot = -1; lockId = 0; }
            }
            else lockB = -1;
        }
    }
    if (lockB >= n) lockB = -1;   // stale lock after the graph shrank

    // slot of resource `id` among a group's interface pins (-1 = none), for edge rerouting.
    auto gpin_slot = [](const uint32_t* ids, int cnt, uint32_t id) { for (int i = 0; i < cnt; ++i) if (ids[i] == id) return i; return -1; };
    // pin a member's edge attaches to: if the member sits in a collapsed group (its own box isn't drawn), the
    // group's compact-node pin for that resource; otherwise the member's own pass pin.
    auto memberOut = [&](int k, uint32_t id) -> ImVec2 {
        if (collOwner[k] >= 0) { int gv = nodeGView[collOwner[k]]; if (gv >= 0) { int sl = gpin_slot(groups[gv].outId, groups[gv].nOut, id); if (sl >= 0) return groups[gv].outC[sl]; } }
        int ss = rg_out_slot(rgn[k].pass, id); return outPin(k, ss >= 0 ? ss : 0);
    };
    auto memberIn = [&](int k, uint32_t id) -> ImVec2 {
        if (collOwner[k] >= 0) { int gv = nodeGView[collOwner[k]]; if (gv >= 0) { int sl = gpin_slot(groups[gv].inId, groups[gv].nIn, id); if (sl >= 0) return groups[gv].inC[sl]; } }
        int ds = rg_in_slot(rgn[k].pass, id); return inPin(k, ds >= 0 ? ds : 0);
    };
    // the drawn polyline for an edge: terminals + the dummy lane chain + any crossed group interface pins,
    // ordered left-to-right by x (which also nests correctly). shared by the hover hit-test and the draw so a
    // rerouted edge is hoverable exactly where it's drawn. caller's wp must hold >= 3*kRgDagMax points.
    auto buildEdgePath = [&](const REdge& e, ImVec2* wp) -> int {
        ImVec2 sp = collOwner[e.src] >= 0 ? memberOut(e.src, e.id) : outPin(e.src, e.sOut);
        ImVec2 dp = collOwner[e.dst] >= 0 ? memberIn(e.dst, e.id)  : inPin(e.dst, e.dIn);
        int exitG[kRgGPinMax], entryG[kRgGPinMax], nex = 0, nen = 0;
        for (int gx = 0; gx < (int)groups.size(); ++gx) {
            GView& g = groups[gx]; if (g.collapsed) continue;
            bool hS = g.gi <= e.src && e.src < g.gj, hD = g.gi <= e.dst && e.dst < g.gj;
            if (hS && !hD && gpin_slot(g.outId, g.nOut, e.id) >= 0 && nex < kRgGPinMax) exitG[nex++] = gx;
            if (hD && !hS && gpin_slot(g.inId,  g.nIn,  e.id) >= 0 && nen < kRgGPinMax) entryG[nen++] = gx;
        }
        if (nex == 0 && nen == 0 && collOwner[e.src] < 0 && collOwner[e.dst] < 0) return edgePoints(e, wp);   // plain edge
        int nw = 0; wp[nw++] = sp;
        for (int t = 0; t < e.chainN; ++t) { float lx = origin.x + lnode[e.chain[t]].x, ly = origin.y + lnode[e.chain[t]].y; wp[nw++] = ImVec2(lx, ly); wp[nw++] = ImVec2(lx + kBoxW, ly); }
        for (int i2 = 0; i2 < nex; ++i2) { GView& g = groups[exitG[i2]];  wp[nw++] = g.outC[gpin_slot(g.outId, g.nOut, e.id)]; }
        for (int i2 = 0; i2 < nen; ++i2) { GView& g = groups[entryG[i2]]; wp[nw++] = g.inC [gpin_slot(g.inId,  g.nIn,  e.id)]; }
        wp[nw++] = dp;
        for (int a2 = 1; a2 < nw; ++a2) { ImVec2 v = wp[a2]; int b2 = a2 - 1; while (b2 >= 0 && wp[b2].x > v.x) { wp[b2 + 1] = wp[b2]; --b2; } wp[b2 + 1] = v; }
        return nw;
    };

    // ---- hovered edge (only when not over a pin/box): nearest edge polyline within a few px.
    int hovEdge = -1;
    if (hovB < 0 && hovBox < 0 && canvasHovered && !canvasActive) {
        float best = 6.0f * 6.0f;
        for (int ei = 0; ei < (int)edge.size(); ++ei) {
            if (collOwner[edge[ei].src] >= 0 && collOwner[edge[ei].src] == collOwner[edge[ei].dst]) continue;   // vanished
            ImVec2 pts[3 * kRgDagMax]; int np = buildEdgePath(edge[ei], pts);
            for (int t = 0; t + 1 < np; ++t) {
                ImVec2 a2 = pts[t], b2 = pts[t + 1]; float dx = (b2.x - a2.x) * 0.5f;
                ImVec2 c1(a2.x + dx, a2.y), c2(b2.x - dx, b2.y), prev = a2;
                for (int s2 = 1; s2 <= 10; ++s2) {   // sample the cubic
                    float tt = s2 / 10.0f, u = 1 - tt, w0 = u * u * u, w1 = 3 * u * u * tt, w2 = 3 * u * tt * tt, w3 = tt * tt * tt;
                    ImVec2 q(w0 * a2.x + w1 * c1.x + w2 * c2.x + w3 * b2.x, w0 * a2.y + w1 * c1.y + w2 * c2.y + w3 * b2.y);
                    float d2 = rg_seg_d2(io.MousePos, prev, q); if (d2 < best) { best = d2; hovEdge = ei; }
                    prev = q;
                }
            }
        }
    }

    // ---- cones. seed from the hovered pin, else the locked pin. upstream = producers needed to make the
    // resource; for an output pin also mark downstream = who consumes it.
    int coneB = hovB >= 0 ? hovB : lockB;
    bool coneWrite = hovB >= 0 ? hovWrite : lockWrite;
    uint32_t coneId = hovB >= 0 ? hovId : lockId;
    const bool conePass = hovB < 0 && lockB >= 0 && lockSlot == -1;   // whole-pass selection: both cones from the pass
    bool inCone[kRgDagMax] = {}, downCone[kRgDagMax] = {}; bool coneActive = false;
    if (coneB >= 0) {
        int seed = (coneWrite || conePass) ? coneB : rg_producer_of(box, n, box[coneB].p, coneId);
        if (seed >= 0) { rg_mark_cone(box, n, seed, inCone); inCone[coneB] = true; coneActive = true; }   // include the hovered node so its immediate edge lights
        if (coneActive && (coneWrite || conePass)) {   // downstream descendants over the data edges
            int st[kRgDagMax], sp = 0; st[sp++] = coneB;
            while (sp) { int u = st[--sp]; for (REdge& e : edge) if (e.src == u && !downCone[e.dst]) { downCone[e.dst] = true; st[sp++] = e.dst; } }
        }
    }
    auto fout  = [&](int i) { return filterActive && !matchBox[i]; };
    auto isDim = [&](int i) { return fout(i) || (coneActive && !inCone[i] && !downCone[i] && !(coneWrite && i == coneB)); };
    auto inUp  = [&](int i) { return coneActive && inCone[i] && !fout(i); };
    auto inDn  = [&](int i) { return coneActive && (downCone[i] || (coneWrite && i == coneB)) && !fout(i); };
    // edge colour + thickness for an edge between two pass boxes, by cone state (gold upstream, teal down,
    // grey normal, faint dim). shared by the top-level edge loop AND the interior group-edge loops so an
    // edge inside a group reads the same as one outside: same hue, same forward/back cone highlight.
    auto edgeCol = [&](int s, int d, float* th) {
        bool eup = inUp(s) && inUp(d), edn = inDn(s) && inDn(d);
        if (eup)                          { if (th) *th = 2.5f; return IM_COL32(245, 222, 120, 235); }
        if (edn)                          { if (th) *th = 2.5f; return IM_COL32(120, 222, 180, 235); }
        if (isDim(s) || isDim(d))         { if (th) *th = 2.0f; return IM_COL32(150, 150, 150, 34);  }
                                            if (th) *th = 2.0f; return IM_COL32(170, 170, 170, 200);
    };

    // ---- data edges. an edge interior to one collapsed group vanishes; an edge crossing a group border (any
    // nesting, collapsed or expanded) threads through that group's interface pin; everything else routes through
    // its dummy waypoints so it never hides behind a box. gold = upstream cone, teal = downstream, white = hovered.
    for (int ei = 0; ei < (int)edge.size(); ++ei) {
        REdge& e = edge[ei];
        if (collOwner[e.src] >= 0 && collOwner[e.src] == collOwner[e.dst]) continue;   // interior to one collapsed group (any level) -> vanishes
        ImU32 col; float th;
        if (ei == hovEdge) { col = IM_COL32(255, 255, 255, 255); th = 3.0f; }
        else col = edgeCol(e.src, e.dst, &th);
        ImVec2 wp[3 * kRgDagMax]; int nw = buildEdgePath(e, wp);
        for (int t = 0; t + 1 < nw; ++t) { ImVec2 a2 = wp[t], b2 = wp[t + 1]; float dx = (b2.x - a2.x) * 0.5f; dl->AddBezierCubic(a2, ImVec2(a2.x + dx, a2.y), ImVec2(b2.x - dx, b2.y), b2, col, th); }
    }

    // (interior group edges + subgroup stubs are gone: an expanded group's members are first-class boxes, so
    // their intra-group flow is ordinary edge[] entries drawn by the loop above, same routing, colour, cone.)

    // ---- boxes.
    for (int i = 0; i < n; ++i) {
        if (drawHidden[i]) continue;   // hidden inside a collapsed group node (drawn separately below)
        ImVec2 tl(origin.x + rgn[i].pos.x, origin.y + rgn[i].pos.y), br(tl.x + rgn[i].w, tl.y + rgn[i].h);
        bool dim = isDim(i), up = inUp(i), dn = inDn(i);

        ImU32 fill = rg_kind_color(rgn[i].pass->kind);
        dl->AddRectFilled(tl, br, dim ? rg_with_alpha(fill, 55) : fill, 5.0f);
        ImU32 edgec = up ? IM_COL32(255, 255, 255, 255) : dn ? IM_COL32(120, 222, 180, 255) : dim ? IM_COL32(40, 40, 40, 120) : IM_COL32(20, 20, 20, 255);
        dl->AddRect(tl, br, edgec, 5.0f, 0, (up || dn) ? 2.5f : 1.0f);
        if (lockB == i && lockSlot == -1)   // whole-pass selection (body click): blue selection ring
            dl->AddRect(ImVec2(tl.x - 2, tl.y - 2), ImVec2(br.x + 2, br.y + 2), IM_COL32(90, 200, 230, 255), 6.0f, 0, 2.0f);
        dl->AddLine(ImVec2(tl.x, tl.y + kHeaderH), ImVec2(br.x, tl.y + kHeaderH), IM_COL32(255, 255, 255, dim ? 18 : 40));

        // sinks: red halo. imported writers only; history writers are sinks to the culler
        // (compile sets p->sink) but aren't graph outputs, so they don't get the halo.
        if (rg_pass_is_sink(rg, rgn[i].pass))
            for (int e = 3; e >= 1; --e)
                dl->AddRect(ImVec2(tl.x - e, tl.y - e), ImVec2(br.x + e, br.y + e), IM_COL32(255, 100, 0, dim ? 80 : 200), 6.0f, 0, 1.0f);

        WGPUStringView nm = rgn[i].pass->id.name;
        char head[96];
        std::snprintf(head, sizeof head, "P%d  %.*s", i, (int)nm.length, nm.data ? nm.data : "");
        dl->PushClipRect(tl, ImVec2(br.x - 4, br.y), true);
        dl->AddText(ImVec2(tl.x + 7, tl.y + 4), dim ? IM_COL32(190, 190, 190, 120) : IM_COL32(255, 255, 255, 255), head);
        dl->PopClipRect();

        const char* kn = rg_kind_name(rgn[i].pass->kind);
        ImVec2 ks = ImGui::CalcTextSize(kn);
        dl->AddText(ImVec2(tl.x + (rgn[i].w - ks.x) * 0.5f, br.y - kFooterH + 1),
            dim ? IM_COL32(200, 200, 200, 110) : IM_COL32(225, 225, 225, 220), kn);
    }

    // ---- pins + resource labels.
    for (int i = 0; i < n; ++i) {
        if (drawHidden[i]) continue;   // hidden inside a collapsed group node (drawn separately below)
        PassNode* p = rgn[i].pass; bool dim = isDim(i);
        const float mid = origin.x + rgn[i].pos.x + rgn[i].w * 0.5f;
        int inS = 0, outS = 0;
        for (uint32_t k = 0; k < p->accessCount; ++k) {
            const ResourceAccess& acc = p->accesses[k];
            ResourceNode* r = find_node(rg, acc.handle);
            WGPUStringView rn = r ? r->id.name : WGPUStringView{};
            char lbl[48]; std::snprintf(lbl, sizeof lbl, "%.*s", (int)rn.length, rn.data ? rn.data : "?");
            ImVec2 ls = ImGui::CalcTextSize(lbl);
            const ImU32 lc = dim ? IM_COL32(190, 190, 190, 110) : IM_COL32(230, 230, 230, 255);
            const bool isBuf = r && r->kind == ResourceKind::Buffer;   // square pin vs round (texture)

            if (rg_access_reads(acc)) {   // input pin (left); hollow if no in-graph producer (external input)
                int slot = inS++; ImVec2 c = inPin(i, slot);
                ImU32 base = dim ? rg_with_alpha(kRGRead, 70) : kRGRead;
                rg_draw_pin(dl, c, kPinR, base, rg_producer_of(box, n, p, acc.handle.id) >= 0, isBuf);
                if (lockB == i && !lockWrite && slot == lockSlot) dl->AddCircle(c, kPinR + 3.0f, IM_COL32(90, 200, 230, 255), 16, 2.0f);
                if (i == hovB && !hovWrite && slot == hovSlot) dl->AddCircle(c, kPinR + 3.0f, IM_COL32(255, 255, 255, 255), 16, 2.0f);
                dl->PushClipRect(ImVec2(c.x + kPinR + 3, c.y - kPinRowH * 0.5f), ImVec2(mid, c.y + kPinRowH * 0.5f), true);
                dl->AddText(ImVec2(c.x + kPinR + 3, c.y - ls.y * 0.5f), lc, lbl);
                dl->PopClipRect();
            }
            if (access_is_write(acc.type)) {   // output pin (right)
                int slot = outS++; ImVec2 c = outPin(i, slot);
                ImU32 base = dim ? rg_with_alpha(kRGWrite, 70) : kRGWrite;
                rg_draw_pin(dl, c, kPinR, base, true, isBuf);
                if (lockB == i && lockWrite && slot == lockSlot) dl->AddCircle(c, kPinR + 3.0f, IM_COL32(90, 200, 230, 255), 16, 2.0f);
                if (i == hovB && hovWrite && slot == hovSlot) dl->AddCircle(c, kPinR + 3.0f, IM_COL32(255, 255, 255, 255), 16, 2.0f);
                dl->PushClipRect(ImVec2(mid, c.y - kPinRowH * 0.5f), ImVec2(c.x - kPinR - 3, c.y + kPinRowH * 0.5f), true);
                dl->AddText(ImVec2(c.x - kPinR - 3 - ls.x, c.y - ls.y * 0.5f), lc, lbl);
                dl->PopClipRect();
            }
        }
    }

    // ---- collapsed group nodes: one synthetic box per collapsed group, drawn on top of the edges like a
    // real pass, carrying the group's external read pins (left) and write pins (right).
    for (GView& g : groups) {
        if (!g.collapsed) continue;
        ImVec2 a = g.bb0, b = g.bb1;
        const float mid = (a.x + b.x) * 0.5f;
        dl->AddRectFilled(a, b, IM_COL32(70, 70, 96, 255), 5.0f);
        dl->AddRect(a, b, IM_COL32(20, 20, 20, 255), 5.0f, 0, 1.0f);
        dl->AddLine(ImVec2(a.x, a.y + kHeaderH), ImVec2(b.x, a.y + kHeaderH), IM_COL32(255, 255, 255, 40));
        char head[64]; std::snprintf(head, sizeof head, "[+] %.*s  x%d", (int)sv_length(g.prefix), g.prefix.data, g.gj - g.gi);
        dl->PushClipRect(a, ImVec2(b.x - 4, b.y), true);
        dl->AddText(ImVec2(a.x + 7, a.y + 4), IM_COL32(255, 255, 255, 255), head);
        dl->PopClipRect();
        for (int s = 0; s < g.nIn; ++s) {
            ResourceNode* r = find_node(rg, { g.inId[s] });
            WGPUStringView rn = r ? r->id.name : WGPUStringView{};
            bool buf = r && r->kind == ResourceKind::Buffer;
            bool prod = false; for (int j = 0; j < n; ++j) if (rg_pass_writes(box[j].p, g.inId[s])) prod = true;
            rg_draw_pin(dl, g.inC[s], kPinR, kRGRead, prod, buf);   // hollow = external input (no in-graph producer)
            if (hovB >= g.gi && hovB < g.gj && !hovWrite && hovId == g.inId[s]) dl->AddCircle(g.inC[s], kPinR + 3.0f, IM_COL32(255, 255, 255, 255), 16, 2.0f);
            if (lockB >= g.gi && lockB < g.gj && !lockWrite && lockId == g.inId[s]) dl->AddCircle(g.inC[s], kPinR + 3.0f, IM_COL32(90, 200, 230, 255), 16, 2.0f);
            char lbl[48]; std::snprintf(lbl, sizeof lbl, "%.*s", (int)rn.length, rn.data ? rn.data : "?");
            ImVec2 ls = ImGui::CalcTextSize(lbl);
            dl->PushClipRect(ImVec2(g.inC[s].x + kPinR + 3, g.inC[s].y - kPinRowH * 0.5f), ImVec2(mid, g.inC[s].y + kPinRowH * 0.5f), true);
            dl->AddText(ImVec2(g.inC[s].x + kPinR + 3, g.inC[s].y - ls.y * 0.5f), IM_COL32(230, 230, 230, 255), lbl);
            dl->PopClipRect();
        }
        for (int s = 0; s < g.nOut; ++s) {
            ResourceNode* r = find_node(rg, { g.outId[s] });
            WGPUStringView rn = r ? r->id.name : WGPUStringView{};
            bool buf = r && r->kind == ResourceKind::Buffer;
            rg_draw_pin(dl, g.outC[s], kPinR, kRGWrite, true, buf);
            if (hovB >= g.gi && hovB < g.gj && hovWrite && hovId == g.outId[s]) dl->AddCircle(g.outC[s], kPinR + 3.0f, IM_COL32(255, 255, 255, 255), 16, 2.0f);
            if (lockB >= g.gi && lockB < g.gj && lockWrite && lockId == g.outId[s]) dl->AddCircle(g.outC[s], kPinR + 3.0f, IM_COL32(90, 200, 230, 255), 16, 2.0f);
            char lbl[48]; std::snprintf(lbl, sizeof lbl, "%.*s", (int)rn.length, rn.data ? rn.data : "?");
            ImVec2 ls = ImGui::CalcTextSize(lbl);
            dl->PushClipRect(ImVec2(mid, g.outC[s].y - kPinRowH * 0.5f), ImVec2(g.outC[s].x - kPinR - 3, g.outC[s].y + kPinRowH * 0.5f), true);
            dl->AddText(ImVec2(g.outC[s].x - kPinR - 3 - ls.x, g.outC[s].y - ls.y * 0.5f), IM_COL32(230, 230, 230, 255), lbl);
            dl->PopClipRect();
        }
    }

    // expanded group interface pins: the dots on the hull edges (reads left, writes right) that every boundary
    // edge threads through. a ring lights when a member pin for the same resource is hovered/locked.
    for (GView& g : groups) {
        if (g.collapsed) continue;
        for (int s = 0; s < g.nIn; ++s) {
            ResourceNode* r = find_node(rg, { g.inId[s] }); bool buf = r && r->kind == ResourceKind::Buffer;
            bool prod = false; for (int j = 0; j < n; ++j) if (rg_pass_writes(box[j].p, g.inId[s])) { prod = true; break; }
            rg_draw_pin(dl, g.inC[s], kPinR, kRGRead, prod, buf);
            if (hovB  >= g.gi && hovB  < g.gj && !hovWrite  && hovId  == g.inId[s]) dl->AddCircle(g.inC[s], kPinR + 3.0f, IM_COL32(255, 255, 255, 255), 16, 2.0f);
            if (lockB >= g.gi && lockB < g.gj && !lockWrite && lockId == g.inId[s]) dl->AddCircle(g.inC[s], kPinR + 3.0f, IM_COL32(90, 200, 230, 255), 16, 2.0f);
        }
        for (int s = 0; s < g.nOut; ++s) {
            ResourceNode* r = find_node(rg, { g.outId[s] }); bool buf = r && r->kind == ResourceKind::Buffer;
            rg_draw_pin(dl, g.outC[s], kPinR, kRGWrite, true, buf);
            if (hovB  >= g.gi && hovB  < g.gj && hovWrite  && hovId  == g.outId[s]) dl->AddCircle(g.outC[s], kPinR + 3.0f, IM_COL32(255, 255, 255, 255), 16, 2.0f);
            if (lockB >= g.gi && lockB < g.gj && lockWrite && lockId == g.outId[s]) dl->AddCircle(g.outC[s], kPinR + 3.0f, IM_COL32(90, 200, 230, 255), 16, 2.0f);
        }
    }

    // ---- virtual endpoint nodes + dashed links, drawn from the IR mirror (rgn Virtual nodes + rge links).
    // a read/source node feeds a pass input pin from the left; a write/sink node is fed by an output pin; an
    // imported buffer fans one source to all readers. collapse-reroute + cone dimming reuse the pass helpers.
    // interface-pin position of resource `id` on a collapsed group, or the group's box edge at height `y` when
    // the resource crosses the boundary without a pin (e.g. a history write consumed only next frame).
    auto grpPin = [&](GView& g, uint32_t id, bool write, float y) -> ImVec2 {
        int sl = gpin_slot(write ? g.outId : g.inId, write ? g.nOut : g.nIn, id);
        if (sl >= 0) return write ? g.outC[sl] : g.inC[sl];
        return ImVec2(write ? g.bb1.x : g.bb0.x, y);
    };
    // expanded group interface pins between member pe and the outside for resource id (write=out-pin), so a
    // virtual link threads through the same border pins the data edges do. fills outermost-first; returns count.
    auto memberPins = [&](int pe, uint32_t id, bool write, ImVec2* out) -> int {
        int idx[kRgGPinMax], ni = 0;
        for (int gx = 0; gx < (int)groups.size(); ++gx) {
            GView& g = groups[gx]; if (g.collapsed) continue;
            if (!(g.gi <= pe && pe < g.gj)) continue;
            if (gpin_slot(write ? g.outId : g.inId, write ? g.nOut : g.nIn, id) < 0) continue;
            if (ni < kRgGPinMax) idx[ni++] = gx;
        }
        for (int a = 1; a < ni; ++a) { int v = idx[a], b = a - 1; while (b >= 0 && groups[idx[b]].depth > groups[v].depth) { idx[b + 1] = idx[b]; --b; } idx[b + 1] = v; }
        for (int i = 0; i < ni; ++i) { GView& g = groups[idx[i]]; out[i] = write ? g.outC[gpin_slot(g.outId, g.nOut, id)] : g.inC[gpin_slot(g.inId, g.nIn, id)]; }
        return ni;
    };
    static std::vector<char> vLive; vLive.assign(rgn.size(), 0);
    for (RgEdge& ge : rge) {
        bool srcV = ge.srcNode >= n, dstV = ge.dstNode >= n;
        if (!srcV && !dstV) continue;                 // pass<->pass edge: drawn earlier
        int vi = srcV ? ge.srcNode : ge.dstNode, pe = srcV ? ge.dstNode : ge.srcNode;   // virtual node, anchor pass
        bool faint = ge.kind == RgEdge::Kind::Fanout;   // imported-buffer fan: one faint edge per reader
        if (fout(pe) || (faint && drawHidden[pe])) continue;
        RgNode& vn = rgn[vi];
        ImU32 tint = isDim(pe) ? rg_with_alpha(vn.tint, 40) : (faint ? rg_with_alpha(vn.tint, 90) : vn.tint);   // cone-dim with anchor
        float vcy = origin.y + vn.pos.y + vn.h * 0.5f;
        // thread the dashed link through the same expanded-group border pins as a data edge: tnode -> [group
        // in-pins] -> reader, or writer -> [group out-pins] -> tnode. collapsed groups terminate at the compact pin.
        ImVec2 wp[2 + kRgGPinMax]; int nw = 0; ImVec2 gp[kRgGPinMax];
        if (srcV) {   // read: node right edge -> reader input pin
            wp[nw++] = ImVec2(origin.x + vn.pos.x + vn.w, vcy);
            if (groupOf[pe] >= 0 && groups[groupOf[pe]].collapsed) {   // collapsed top-level group: its compact in-pin
                GView& g = groups[groupOf[pe]];
                if (faint) { int s = gpin_slot(g.inId, g.nIn, ge.resId); if (s < 0 || g.inDrawn[s]) continue; g.inDrawn[s] = true; wp[nw++] = g.inC[s]; }
                else wp[nw++] = grpPin(g, ge.resId, false, wp[0].y);
            }
            else if (collOwner[pe] >= 0) wp[nw++] = memberIn(pe, ge.resId);   // folded into a collapsed subgroup
            else {                                                            // expanded/ungrouped: thread group pins -> member
                int ng = memberPins(pe, ge.resId, false, gp);
                for (int i = 0; i < ng; ++i) wp[nw++] = gp[i];
                wp[nw++] = inPin(pe, ge.dstPin);
            }
        }
        else {        // write: writer output pin -> node left edge
            ImVec2 qq(origin.x + vn.pos.x, vcy);
            if (groupOf[pe] >= 0 && groups[groupOf[pe]].collapsed) wp[nw++] = grpPin(groups[groupOf[pe]], ge.resId, true, qq.y);
            else if (collOwner[pe] >= 0) wp[nw++] = memberOut(pe, ge.resId);
            else { wp[nw++] = outPin(pe, ge.srcPin); int ng = memberPins(pe, ge.resId, true, gp); for (int i = ng - 1; i >= 0; --i) wp[nw++] = gp[i]; }
            wp[nw++] = qq;
        }
        vLive[vi] = 1;
        for (int t = 0; t + 1 < nw; ++t) { ImVec2 a2 = wp[t], b2 = wp[t + 1]; float dx = (b2.x - a2.x) * 0.5f; rg_dashed_cubic(dl, a2, ImVec2(a2.x + dx, a2.y), ImVec2(b2.x - dx, b2.y), b2, tint, faint ? 1.5f : 2.0f); }
        ImVec2 tip = wp[nw - 1];
        rg_arrowhead(dl, ImVec2(tip.x - 8, tip.y), tip, tint, faint ? 6.0f : 7.0f);
    }
    for (int vi = n; vi < (int)rgn.size(); ++vi) {
        if (!vLive[vi]) continue;   // node with no surviving link (all readers filtered/deduped) stays hidden
        RgNode& vn = rgn[vi];
        ImVec2 a(origin.x + vn.pos.x, origin.y + vn.pos.y), b(a.x + vn.w, a.y + vn.h);
        char nm[48]; std::snprintf(nm, sizeof nm, "%.*s", (int)vn.res->id.name.length, vn.res->id.name.data ? vn.res->id.name.data : "?");
        const char* cap = vn.label.data ? vn.label.data : "";
        dl->AddRectFilled(a, b, IM_COL32(32, 30, 40, 240), 4.0f);
        dl->AddRect(a, b, vn.tint, 4.0f, 0, 1.5f);
        float cx = (a.x + b.x) * 0.5f;
        ImVec2 ns = ImGui::CalcTextSize(nm), cs = ImGui::CalcTextSize(cap);
        dl->AddText(ImVec2(cx - ns.x * 0.5f, a.y + 3), IM_COL32(238, 236, 242, 255), nm);
        dl->AddText(ImVec2(cx - cs.x * 0.5f, b.y - cs.y - 3), rg_with_alpha(vn.tint, 220), cap);
    }

    // ---- tooltip: hovered pin wins; else fall back to the per-pass reads/writes list.
    if (hovB >= 0) {
        PassNode* p = rgn[hovB].pass; ResourceNode* r = find_node(rg, { hovId });
        WGPUStringView rn = r ? r->id.name : WGPUStringView{};
        ImGui::BeginTooltip();
        ImGui::Text("%.*s", (int)rn.length, rn.data ? rn.data : "?");
        if (r) {
            if (r->kind == ResourceKind::Texture) ImGui::Text("texture  %u x %u", r->resolved.width, r->resolved.height);
            else                                         ImGui::Text("buffer  %llu bytes", (unsigned long long)r->bufferSize);
        }
        // subresource this pin touches: layer for an array (CSM cascades), mip for a chain (bloom).
        if (r && r->kind == ResourceKind::Texture) {
            uint32_t layers = r->resolved.depthOrArrayLayers ? r->resolved.depthOrArrayLayers : 1;
            uint32_t mips   = r->mipLevelCount ? r->mipLevelCount : 1;
            if (layers > 1 || mips > 1 || hovLayer > 0 || hovMip > 0) {
                char sub[64]; int o = 0;
                if (layers > 1 || hovLayer > 0) o += std::snprintf(sub + o, sizeof sub - o, "layer %u / %u", hovLayer, layers);
                if (mips > 1 || hovMip > 0)     o += std::snprintf(sub + o, sizeof sub - o, "%smip %u / %u", o ? "   " : "", hovMip, mips);
                ImGui::TextDisabled("%s", sub);
            }
        }
        const char* verb = hovWrite ? "write"
            : (hovType == AccessType::ColorAttachment || hovType == AccessType::DepthStencilAttachment) ? "load"
            : "read";
        ImGui::TextDisabled("%s on P%d  (%s)", verb, hovB, rg_access_name(hovType));
        ImGui::Separator();
        if (hovWrite) ImGui::Text("produced here");
        else {
            int prod = rg_producer_of(box, n, p, hovId);
            if (prod >= 0) {
                WGPUStringView pn = rgn[prod].pass->id.name;
                ImGui::Text("produced by P%d %.*s", prod, (int)pn.length, pn.data ? pn.data : "");
            }
            else ImGui::TextDisabled(r && r->imported ? "imported (external input)" : "external input (no producer)");
        }
        ImGui::EndTooltip();
    }
    else if (hovBox >= 0) {
        PassNode* p = rgn[hovBox].pass; WGPUStringView nm = p->id.name;
        const char* kn = rg_kind_name(p->kind);
        ImGui::BeginTooltip();
        ImGui::Text("%.*s  [%s]", (int)nm.length, nm.data ? nm.data : "", kn);
        ImGui::Separator();
        for (uint32_t k = 0; k < p->accessCount; ++k) {
            const ResourceAccess& acc = p->accesses[k];
            ResourceNode* r = find_node(rg, acc.handle);
            WGPUStringView rn = r ? r->id.name : WGPUStringView{};
            ImGui::Text("[%s] %.*s  (%s)%s", access_is_write(acc.type) ? "W" : "R",
                (int)rn.length, rn.data ? rn.data : "", rg_access_name(acc.type),
                r ? (r->imported ? "  [imported]" : r->persistent ? "  [history]" : "") : "");
        }
        ImGui::EndTooltip();
    }
    else if (hovEdge >= 0) {
        REdge& e = edge[hovEdge]; ResourceNode* r = find_node(rg, { e.id });
        WGPUStringView rn = r ? r->id.name : WGPUStringView{};
        WGPUStringView sn = rgn[e.src].pass->id.name, dn = rgn[e.dst].pass->id.name;
        ImGui::BeginTooltip();
        ImGui::Text("%.*s", (int)rn.length, rn.data ? rn.data : "?");
        ImGui::TextDisabled("P%d %.*s  ->  P%d %.*s", e.src, (int)sn.length, sn.data ? sn.data : "",
            e.dst, (int)dn.length, dn.data ? dn.data : "");
        ImGui::EndTooltip();
    }

    ImGui::SetWindowFontScale(1.0f);   // overlays (details panel, zoom indicator) stay readable at any zoom

    // ---- details panel: persistent info for the locked (selected) pin + its pass. top-right overlay,
    // drawn with the canvas draw list (no nested child). click a pin to fill it; click empty canvas clears.
    if (lockB >= 0 && lockB < n) {
        PassNode* p = rgn[lockB].pass;
        ResourceNode* r = find_node(rg, { lockId });
        // the access this pin stands for (match id + read/write side) -> type + touched subresource.
        AccessType selType{}; uint32_t selMip = 0, selLayer = 0;
        for (uint32_t k = 0; k < p->accessCount; ++k) {
            const ResourceAccess& a = p->accesses[k];
            if (a.handle.id == lockId && access_is_write(a.type) == lockWrite)
                { selType = a.type; selMip = a.baseMip; selLayer = a.baseLayer; break; }
        }

        char ln[48][128]; ImU32 lc[48]; int nl = 0;
        const ImU32 cTitle = IM_COL32(238, 236, 242, 255), cDim = IM_COL32(150, 150, 162, 255),
                    cHead = IM_COL32(120, 200, 190, 255);
        auto add = [&](ImU32 col, const char* fmt, ...) {
            if (nl >= 48) return;
            va_list ap; va_start(ap, fmt); std::vsnprintf(ln[nl], sizeof ln[nl], fmt, ap); va_end(ap);
            lc[nl++] = col;
        };

        const bool pinSel = lockSlot >= 0;   // a specific pin vs. a whole-pass selection (body click)
        WGPUStringView pn = p->id.name;
        add(cTitle, "P%d  %.*s  [%s]", lockB, (int)pn.length, pn.data ? pn.data : "", rg_kind_name(p->kind));
        add(cDim, "");
        if (pinSel) {
            add(cHead, "SELECTED PIN");
            WGPUStringView rn = r ? r->id.name : WGPUStringView{};
            add(r ? rg_resource_color(r->kind) : cDim, "%.*s", (int)rn.length, rn.data ? rn.data : "?");
            if (r) {
                if (r->kind == ResourceKind::Texture) add(cDim, "texture  %u x %u", r->resolved.width, r->resolved.height);
                else                                         add(cDim, "buffer  %llu bytes", (unsigned long long)r->bufferSize);
            }
            if (r && r->kind == ResourceKind::Texture) {
                uint32_t layers = r->resolved.depthOrArrayLayers ? r->resolved.depthOrArrayLayers : 1;
                uint32_t mips   = r->mipLevelCount ? r->mipLevelCount : 1;
                if (layers > 1 || mips > 1 || selLayer > 0 || selMip > 0)
                    add(cDim, "layer %u / %u    mip %u / %u", selLayer, layers, selMip, mips);
            }
            const char* verb = lockWrite ? "write"
                : (selType == AccessType::ColorAttachment || selType == AccessType::DepthStencilAttachment) ? "load" : "read";
            add(cDim, "%s  (%s)", verb, rg_access_name(selType));
            if (lockWrite) add(cDim, "produced here");
            else {
                int prod = rg_producer_of(box, n, p, lockId);
                if (prod >= 0) { WGPUStringView qn = rgn[prod].pass->id.name; add(cDim, "produced by P%d %.*s", prod, (int)qn.length, qn.data ? qn.data : ""); }
                else add(cDim, r && r->imported ? "imported (external input)" : "external input");
            }
            add(cDim, "");
        }
        add(cHead, "PASS ACCESSES");
        for (uint32_t k = 0; k < p->accessCount; ++k) {
            const ResourceAccess& acc = p->accesses[k];
            ResourceNode* ar = find_node(rg, acc.handle);
            WGPUStringView arn = ar ? ar->id.name : WGPUStringView{};
            bool sel = acc.handle.id == lockId && access_is_write(acc.type) == lockWrite;
            add(sel ? cTitle : cDim, "%s [%s] %.*s (%s)%s", sel ? ">" : " ",
                access_is_write(acc.type) ? "W" : "R", (int)arn.length, arn.data ? arn.data : "",
                rg_access_name(acc.type), ar ? (ar->imported ? " [imp]" : ar->persistent ? " [tmp]" : "") : "");
        }

        // size to content, place top-right (clamped into the canvas), draw bg + border + lines.
        float lineH = ImGui::GetTextLineHeightWithSpacing(), maxW = 0;
        for (int i = 0; i < nl; ++i) { float w = ImGui::CalcTextSize(ln[i]).x; if (w > maxW) maxW = w; }
        const float padX = 10, padY = 8;
        float panelW = maxW + padX * 2, panelH = nl * lineH + padY * 2;
        ImVec2 tl(winPos.x + winSize.x - panelW - 8, winPos.y + 8);
        if (tl.x < winPos.x + 8) tl.x = winPos.x + 8;
        dl->AddRectFilled(tl, ImVec2(tl.x + panelW, tl.y + panelH), IM_COL32(24, 26, 32, 235), 5.0f);
        dl->AddRect(tl, ImVec2(tl.x + panelW, tl.y + panelH), IM_COL32(80, 86, 100, 255), 5.0f);
        float y = tl.y + padY;
        for (int i = 0; i < nl; ++i) { dl->AddText(ImVec2(tl.x + padX, y), lc[i], ln[i]); y += lineH; }
    }

    // ---- zoom level, bottom-right corner.
    char zb[16]; std::snprintf(zb, sizeof zb, "%d%%", (int)(zoom * 100.0f + 0.5f));
    ImVec2 zs = ImGui::CalcTextSize(zb);
    ImVec2 zp(winPos.x + winSize.x - zs.x - 10, winPos.y + winSize.y - zs.y - 8);
    dl->AddRectFilled(ImVec2(zp.x - 6, zp.y - 4), ImVec2(zp.x + zs.x + 6, zp.y + zs.y + 4), IM_COL32(24, 26, 32, 200), 4.0f);
    dl->AddText(zp, IM_COL32(220, 220, 230, 255), zb);

    ImGui::EndChild();
}

// what `p` does to resource `id`: bit 1 = reads, bit 2 = writes (0 = doesn't touch it this pass).
static int rg_pass_access(PassNode* p, uint32_t id)
{
    int a = 0;
    for (uint32_t i = 0; i < p->accessCount; ++i)
        if (p->accesses[i].handle.id == id)
            a |= access_is_write(p->accesses[i].type) ? 2 : 1;
    return a;
}

// distinct-ish colour per physical alias slot, so transients packed onto the same slot read as one hue
// (swatch in the gutter + bar outline). cycles a small palette; adjacent slot indices stay distinguishable.
static ImU32 rg_slot_color(uint32_t slot)
{
    static const ImU32 pal[] = {
        IM_COL32( 90, 170, 250, 255), IM_COL32(250, 170,  90, 255), IM_COL32(130, 220, 130, 255),
        IM_COL32(220, 130, 220, 255), IM_COL32(230, 210,  90, 255), IM_COL32(110, 210, 220, 255),
        IM_COL32(240, 140, 140, 255), IM_COL32(170, 160, 250, 255),
    };
    return pal[slot % (sizeof pal / sizeof pal[0])];
}

// short format tag for the physical-slot row labels in the lifetime view (display only).
static const char* rg_format_short(WGPUTextureFormat f)
{
    switch (f) {
      case WGPUTextureFormat_RGBA8Unorm:   return "RGBA8";
      case WGPUTextureFormat_BGRA8Unorm:   return "BGRA8";
      case WGPUTextureFormat_RGBA16Float:  return "RGBA16F";
      case WGPUTextureFormat_R32Float:     return "R32F";
      case WGPUTextureFormat_Depth32Float: return "D32F";
      case WGPUTextureFormat_Depth24Plus:  return "D24+";
      case WGPUTextureFormat_R8Unorm:      return "R8";
      default:                             return "tex";
    }
}

// Lifetime grid over the graph's own per-frame GPU textures. Top axis = passes in execution order. Each
// row is ONE dedicated allocation, never a bare logical handle:
//   * aliasing ON  -> a physical slot ("image N (fmt)"), its bar showing every occupant's writes (warm)
//     and reads (cool) with empty gaps where the texture is free between occupants;
//   * aliasing OFF -> each realized transient is its own texture (one row, named for the resource).
// Toggling 'alias transients' (Demos) collapses N transient rows into M shared textures; that row-count
// drop, plus the reuse gaps, is the aliasing made visible. Imported (caller-owned), history/persistent
// (pool-backed) and dead/zero-usage resources have no dedicated per-frame allocation and are NOT shown.
// firstUse/lastUse come from compile() phase 3; the slot table from phase 4.
static void rg_draw_lifetimes(RenderGraph* rg, RenderGraphStorage& s)
{
    constexpr int   kMax = 128;
    constexpr float kLabelW = 150.0f, kColW = 88.0f, kRowH = 24.0f, kHeaderH = 24.0f;

    ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(kRGWrite), "write");
    ImGui::SameLine(0, 4);
    ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(kRGRead), "read");
    ImGui::SameLine();
    ImGui::TextDisabled("(rows = the graph's dedicated per-frame GPU objects: textures + buffers)");

    // header: rows below are physical slots (aliasing on) or per-transient objects (off). either way only
    // dedicated, graph-owned allocations, imported/history/dead excluded. the row-count + saved MB are
    // the win; toggle 'alias transients' in Demos and watch the rows collapse.
    auto phBytes = [](const PhysicalResource& ph) -> uint64_t {
        return ph.kind == ResourceKind::Buffer ? ph.bufferSize : texture_bytes(ph.size, ph.format);
    };
    if (s.m_slotCount) {
        uint32_t logical = 0; uint64_t logicalBytes = 0, physicalBytes = 0;
        for (uint32_t i = 0; i < s.m_slotCount; ++i) physicalBytes += phBytes(s.m_slots[i]);
        for (ResourceNode* r = s.m_resouces; r; r = r->next)
            if (r->aliasSlot != ResourceNode::kNoSlot) { ++logical; logicalBytes += phBytes(s.m_slots[r->aliasSlot]); }
        ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(IM_COL32(150, 230, 150, 255)),
            "aliasing ON: %u transients packed onto %u GPU objects, saved %.2f MB",
            logical, s.m_slotCount, (double)(logicalBytes - physicalBytes) / (1024.0 * 1024.0));
    } else {
        uint32_t live = 0;
        for (ResourceNode* r = s.m_resouces; r; r = r->next)
            if (!r->is_external() && r->firstUse != ResourceNode::kNoPass) ++live;
        ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(IM_COL32(205, 205, 120, 255)),
            "aliasing OFF: %u dedicated transient objects (tick 'alias transients' in Demos to pack)", live);
    }

    PassNode* passAt[kMax];
    int nPass = 0;
    for (PassNode* p = s.m_passes; p && nPass < kMax; p = p->next) passAt[nPass++] = p;

    // rows = the graph's dedicated per-frame GPU textures, unified across aliasing on/off: every row is one
    // physical texture. a slot (aliasing on) hosts >=1 disjoint-lifetime transient; a non-aliased transient
    // is its own single-occupant texture. with aliasing off there are no slots, so each transient is its own
    // row, same view, no sharing. imported/history/persistent (not graph-allocated this frame) and
    // dead/zero-usage transients (never realized) are excluded: only real GPU textures appear.
    struct Row { ResourceNode* r; int slot; uint32_t first, last; };
    Row row[kMax];
    int nRow = 0;
    for (uint32_t si = 0; si < s.m_slotCount && nRow < kMax; ++si) {
        uint32_t f = ResourceNode::kNoPass, l = 0;
        for (ResourceNode* o = s.m_resouces; o; o = o->next)
            if (o->aliasSlot == si) { if (o->firstUse < f) f = o->firstUse; if (o->lastUse > l) l = o->lastUse; }
        if (f != ResourceNode::kNoPass) row[nRow++] = { nullptr, (int)si, f, l };
    }
    for (ResourceNode* r = s.m_resouces; r && nRow < kMax; r = r->next)
        if (!r->is_external() && r->aliasSlot == ResourceNode::kNoSlot && r->firstUse != ResourceNode::kNoPass)
            row[nRow++] = { r, -1, r->firstUse, r->lastUse };

    // uniform per-row queries over the two kinds (a slot reads its m_slots entry + ORs its occupants; a solo
    // row is its one resource). keeps the draw + details code identical for aliased and non-aliased.
    auto rowFmt   = [&](const Row& rw) { return rw.slot >= 0 ? s.m_slots[rw.slot].format   : rw.r->format;   };
    auto rowSize  = [&](const Row& rw) { return rw.slot >= 0 ? s.m_slots[rw.slot].size     : rw.r->resolved; };
    auto rowUsage = [&](const Row& rw) { return rw.slot >= 0 ? s.m_slots[rw.slot].texUsage : rw.r->texUsage; };
    auto rowKind  = [&](const Row& rw) { return rw.slot >= 0 ? s.m_slots[rw.slot].kind     : rw.r->kind;     };
    auto rowAccess = [&](const Row& rw, uint32_t c) -> int {
        if (rw.slot < 0) return rg_pass_access(passAt[c], rw.r->handle.id);
        int a = 0;
        for (ResourceNode* o = s.m_resouces; o; o = o->next)
            if (o->aliasSlot == (uint32_t)rw.slot) a |= rg_pass_access(passAt[c], o->handle.id);
        return a;
    };
    // live this pass? a solo row spans [first,last]; a slot row is live only while some occupant is, so the
    // gaps between successive occupants render empty: the texture being freed and reused, on the timeline.
    auto rowLive = [&](const Row& rw, uint32_t c) -> bool {
        if (rw.slot < 0) return rw.first <= c && c <= rw.last;
        for (ResourceNode* o = s.m_resouces; o; o = o->next)
            if (o->aliasSlot == (uint32_t)rw.slot && o->firstUse <= c && c <= o->lastUse) return true;
        return false;
    };
    // occupant count, and (when exactly one) that sole resource -> drives the label: a texture backing one
    // resource is shown by its NAME (plain); a shared texture (>1 occupant) by the synthetic "image N".
    auto rowOccupants = [&](const Row& rw, ResourceNode*& sole) -> int {
        if (rw.slot < 0) { sole = rw.r; return 1; }
        int n = 0; sole = nullptr;
        for (ResourceNode* o = s.m_resouces; o; o = o->next)
            if (o->aliasSlot == (uint32_t)rw.slot) { if (!n) sole = o; ++n; }
        return n;
    };

    // selection persists across frames. the graph is rebuilt each frame (handle ids not stable), so key a
    // slot row by its slot index and a solo row by its resource NAME (stable for a given graph).
    static int  selSlot = -2;        // -2 = nothing; -1 = a solo row (selName); >= 0 = slot index
    static char selName[96] = {};
    auto rowSelected = [&](const Row& rw) -> bool {
        if (rw.slot >= 0) return rw.slot == selSlot;
        return selSlot == -1 && rw.r->id.name.data && std::strcmp(selName, rw.r->id.name.data) == 0;
    };

    // lifetime grid on top, details panel for the selected texture below (only when the tab is tall enough).
    const float availY = ImGui::GetContentRegionAvail().y;
    const float gridH  = availY > 240.0f ? availY * 0.62f : availY;

    ImGui::BeginChild("rg_life_grid", ImVec2(0, gridH), true, ImGuiWindowFlags_HorizontalScrollbar);
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(kLabelW + nPass * kColW, kHeaderH + nRow * kRowH));   // reserve scroll region
    ImDrawList* dl = ImGui::GetWindowDrawList();

    const float gridR = origin.x + kLabelW + nPass * kColW;
    const float gridT = origin.y + kHeaderH;
    const float gridB = gridT + nRow * kRowH;

    // top axis: one clipped pass name per column + a faint column line; hover a header for the full name.
    for (int c = 0; c < nPass; ++c) {
        float x = origin.x + kLabelW + c * kColW;
        dl->AddLine(ImVec2(x, origin.y), ImVec2(x, gridB), IM_COL32(255, 255, 255, 22));
        WGPUStringView nm = passAt[c]->id.name;
        if (nm.data) {
            dl->PushClipRect(ImVec2(x + 4, origin.y), ImVec2(x + kColW, gridT), true);
            dl->AddText(ImVec2(x + 6, origin.y + 5), IM_COL32(225, 225, 225, 255), nm.data, nm.data + nm.length);
            dl->PopClipRect();
        }
        ImGui::SetCursorScreenPos(ImVec2(x, origin.y));
        ImGui::PushID(c);
        ImGui::InvisibleButton("h", ImVec2(kColW, kHeaderH));
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::Text("P%d  %.*s  [%s]", c, (int)nm.length, nm.data ? nm.data : "", rg_kind_name(passAt[c]->kind));
            ImGui::EndTooltip();
        }
        ImGui::PopID();
    }
    dl->AddLine(ImVec2(origin.x, gridT), ImVec2(gridR, gridT), IM_COL32(255, 255, 255, 40));   // axis underline

    // one row per physical texture: "image i (fmt WxH)" + swatch in the gutter, occupant write/read cells
    // across the timeline with empty gaps where the texture is free. click a row to inspect it below.
    for (int i = 0; i < nRow; ++i) {
        const float y    = gridT + i * kRowH;
        ResourceNode* sole = nullptr;
        const int   nocc   = rowOccupants(row[i], sole);
        const bool  shared = nocc > 1;
        // shared textures get the synthetic "image N" + a slot colour so they stand out; a single-occupant
        // texture keeps the resource name and reads as a plain row. format/size are in the details panel.
        const ImU32 col  = shared ? rg_slot_color((uint32_t)i) : IM_COL32(210, 210, 210, 255);
        const bool  seld = rowSelected(row[i]);
        if (seld)       dl->AddRectFilled(ImVec2(origin.x, y), ImVec2(gridR, y + kRowH), IM_COL32(255, 255, 255, 30));
        else if (i & 1) dl->AddRectFilled(ImVec2(origin.x, y), ImVec2(gridR, y + kRowH), IM_COL32(255, 255, 255, 10));

        const bool isBuf = rowKind(row[i]) == ResourceKind::Buffer;
        char label[96];
        if (shared) std::snprintf(label, sizeof label, "%s %d (x%d)", isBuf ? "buffer" : "image", i, nocc);
        else        std::snprintf(label, sizeof label, "%.*s", (int)sole->id.name.length, sole->id.name.data ? sole->id.name.data : "");
        if (shared)   // gutter swatch only for shared (aliased) textures
            dl->AddRectFilled(ImVec2(origin.x + 1, y + 4), ImVec2(origin.x + 4, y + kRowH - 4), col);
        dl->PushClipRect(ImVec2(origin.x + 6, y), ImVec2(origin.x + kLabelW - 4, y + kRowH), true);
        dl->AddText(ImVec2(origin.x + 8, y + 4), col, label);
        dl->PopClipRect();

        // whole-row click target (gutter + grid) selects this texture for the details panel below.
        ImGui::SetCursorScreenPos(ImVec2(origin.x, y));
        ImGui::PushID(kMax + i);
        if (ImGui::InvisibleButton("row", ImVec2(kLabelW + nPass * kColW, kRowH))) {
            selSlot = row[i].slot;
            if (row[i].slot < 0) std::snprintf(selName, sizeof selName, "%.*s",
                (int)row[i].r->id.name.length, row[i].r->id.name.data ? row[i].r->id.name.data : "");
        }
        const bool hov = ImGui::IsItemHovered();
        ImGui::PopID();

        const float x0 = origin.x + kLabelW + row[i].first * kColW + 3.0f;
        const float x1 = origin.x + kLabelW + (row[i].last + 1) * kColW - 3.0f;
        const float ty = y + 3.0f, by = y + kRowH - 3.0f;

        // occupant write/read cells; gaps (texture free) left empty so reuse reads off the timeline.
        const ImU32 bandCol = rg_with_alpha(col, 45);
        for (uint32_t c = row[i].first; c <= row[i].last; ++c) {
            if (!rowLive(row[i], c)) continue;
            float sx0 = (c == row[i].first) ? x0 : origin.x + kLabelW + c * kColW;
            float sx1 = (c == row[i].last)  ? x1 : origin.x + kLabelW + (c + 1) * kColW;
            dl->AddRectFilled(ImVec2(sx0, ty), ImVec2(sx1, by), bandCol, 0.0f);
            int acc = rowAccess(row[i], c);
            if (!acc) continue;
            if (acc == 3) {   // read + write -> split top (write) / bottom (read)
                float mid = (ty + by) * 0.5f;
                dl->AddRectFilled(ImVec2(sx0, ty), ImVec2(sx1, mid), kRGWrite, 0.0f);
                dl->AddRectFilled(ImVec2(sx0, mid), ImVec2(sx1, by), kRGRead, 0.0f);
            } else
                dl->AddRectFilled(ImVec2(sx0, ty), ImVec2(sx1, by), acc == 2 ? kRGWrite : kRGRead, 0.0f);
        }

        // outline each contiguous occupancy run in the row colour (brighter when selected or hovered), so the
        // gaps stay open and each occupant reads as its own block.
        const ImU32 edge = (seld || hov) ? IM_COL32(255, 255, 255, 255) : col;
        const float wth  = (seld || hov) ? 2.0f : 1.0f;
        for (uint32_t c = row[i].first; c <= row[i].last; ) {
            if (!rowLive(row[i], c)) { ++c; continue; }
            uint32_t segStart = c;
            while (c <= row[i].last && rowLive(row[i], c)) ++c;
            float sx0 = origin.x + kLabelW + segStart * kColW + 3.0f;
            float sx1 = origin.x + kLabelW + c * kColW - 3.0f;
            dl->AddRect(ImVec2(sx0, ty), ImVec2(sx1, by), edge, 0.0f, 0, wth);
        }
    }
    ImGui::EndChild();

    // ---- details panel: the selected physical texture and the logical resource(s) it backs ----
    if (gridH < availY) {
        ImGui::BeginChild("rg_life_details", ImVec2(0, 0), true);
        int selIdx = -1;
        for (int i = 0; i < nRow; ++i) if (rowSelected(row[i])) { selIdx = i; break; }
        if (selIdx < 0) {
            ImGui::TextDisabled("click a texture above to inspect it");
        } else {
            const Row&    rw    = row[selIdx];
            const bool    isBuf = rowKind(rw) == ResourceKind::Buffer;
            ResourceNode* psole = nullptr;
            const int     pnocc = rowOccupants(rw, psole);
            const ImU32   hcol  = pnocc > 1 ? rg_slot_color((uint32_t)selIdx) : IM_COL32(225, 225, 225, 255);
            if (pnocc > 1) ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(hcol), "%s %d  (shared by %d)",
                isBuf ? "buffer" : "image", selIdx, pnocc);
            else           ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(hcol), "%.*s",
                (int)psole->id.name.length, psole->id.name.data ? psole->id.name.data : "");
            ImGui::SameLine();

            char ub[160]; ub[0] = '\0';
            if (isBuf) {
                uint64_t        bytes = rw.slot >= 0 ? s.m_slots[rw.slot].bufferSize : rw.r->bufferSize;
                WGPUBufferUsage u     = rw.slot >= 0 ? s.m_slots[rw.slot].bufUsage   : rw.r->bufUsage;
                ImGui::Text("- buffer  -  %.1f KB", bytes / 1024.0);
                auto addb = [&](WGPUBufferUsage bit, const char* nm) { if (u & bit) { if (ub[0]) std::strcat(ub, " | "); std::strcat(ub, nm); } };
                addb(WGPUBufferUsage_Storage,  "Storage");   addb(WGPUBufferUsage_Uniform,  "Uniform");
                addb(WGPUBufferUsage_Vertex,   "Vertex");    addb(WGPUBufferUsage_Index,    "Index");
                addb(WGPUBufferUsage_Indirect, "Indirect");  addb(WGPUBufferUsage_CopySrc,  "CopySrc");
                addb(WGPUBufferUsage_CopyDst,  "CopyDst");
            } else {
                WGPUExtent3D      sz  = rowSize(rw);
                WGPUTextureFormat fmt = rowFmt(rw);
                WGPUTextureUsage  u   = rowUsage(rw);
                ImGui::Text("- %s  %u x %u  -  %.1f KB", rg_format_short(fmt), sz.width, sz.height, texture_bytes(sz, fmt) / 1024.0);
                auto addu = [&](WGPUTextureUsage bit, const char* nm) { if (u & bit) { if (ub[0]) std::strcat(ub, " | "); std::strcat(ub, nm); } };
                addu(WGPUTextureUsage_RenderAttachment, "RenderAttachment");
                addu(WGPUTextureUsage_TextureBinding,   "TextureBinding");
                addu(WGPUTextureUsage_StorageBinding,   "StorageBinding");
                addu(WGPUTextureUsage_CopySrc,          "CopySrc");
                addu(WGPUTextureUsage_CopyDst,          "CopyDst");
            }
            ImGui::TextDisabled("usage: %s", ub[0] ? ub : "(none)");

            int occ = 0;
            if (rw.slot >= 0) { for (ResourceNode* o = s.m_resouces; o; o = o->next) if (o->aliasSlot == (uint32_t)rw.slot) ++occ; }
            else occ = 1;
            ImGui::Separator();
            if (occ > 1) ImGui::Text("%d logical resources share this %s (disjoint lifetimes):", occ, isBuf ? "buffer" : "texture");
            else         ImGui::Text("backs 1 logical resource:");

            auto detailOne = [&](ResourceNode* o) {
                WGPUStringView f = pass_name_at(s.m_passes, o->firstUse);
                WGPUStringView l = pass_name_at(s.m_passes, o->lastUse);
                ImGui::BulletText("%.*s   [%.*s .. %.*s]",
                    (int)o->id.name.length, o->id.name.data ? o->id.name.data : "",
                    (int)f.length, f.data ? f.data : "", (int)l.length, l.data ? l.data : "");
                ImGui::Indent();
                for (uint32_t c = o->firstUse; c <= o->lastUse; ++c) {
                    int a = rg_pass_access(passAt[c], o->handle.id);
                    if (!a) continue;
                    WGPUStringView pn = passAt[c]->id.name;
                    ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(a == 1 ? kRGRead : kRGWrite),
                        "%s  %.*s", a == 3 ? "rw" : a == 2 ? " w" : " r", (int)pn.length, pn.data ? pn.data : "");
                }
                ImGui::Unindent();
            };
            if (rw.slot >= 0) { for (ResourceNode* o = s.m_resouces; o; o = o->next) if (o->aliasSlot == (uint32_t)rw.slot) detailOne(o); }
            else detailOne(rw.r);
        }
        ImGui::EndChild();
    }
}

// GPU-memory view across every pool the graph owns: the transient pool (one descriptor-keyed cache of
// textures + buffers, tagged by kind) and history textures + buffers (PersistentResourcePool
// ping-pong). Grand total at the top answers "how much VRAM does the graph cost"; the transient pool also
// keeps its create/evict log so steady-state reuse is still verifiable (0 created after warmup). Drawn after
// realize() and before release_resources()/end_frame(), so every count is this frame's live allocation.
static void rg_draw_memory(RenderGraphStorage& s)
{
    TransientResourcePool&  tp   = s.m_allocator->transient;
    PersistentResourcePool& pool = s.m_allocator->pool;

    // transient pool: textures + buffers in one descriptor-keyed cache, tagged by Entry::kind. one
    // physical object per entry; idle ones retained kRetain frames. tally each kind (held includes idle).
    int held = 0, inUse = 0, texHeld = 0, bufHeld = 0;
    uint64_t texBytes = 0, texInUseBytes = 0, bufBytes = 0, bufInUseBytes = 0;
    for (const TransientResourcePool::Entry& e : tp.entries) {
        ++held; if (e.inUse) ++inUse;
        if (e.kind == ResourceKind::Buffer) { ++bufHeld; bufBytes += e.bufferSize; if (e.inUse) bufInUseBytes += e.bufferSize; }
        else {
            const uint64_t b = rg_entry_bytes(e);
            ++texHeld; texBytes += b;
            if (e.inUse) texInUseBytes += b;
        }
    }

    // persistent + history resources share one name-keyed pool: a history entry holds kLayers physical
    // objects (current + previous, ping-ponged), a persistent entry holds 1 (single in-place). the buffer arm
    // leaves size {} / format Undefined, so split on bufferSize to size each right; mem already scales by layers.
    int tmpTexCount = 0, tmpBufCount = 0;
    uint64_t tmpTexBytes = 0, tmpBufBytes = 0;
    for (const PersistentResourcePool::Entry& e : pool.entries) {
        if (!e.created) continue;
        if (e.bufferSize) { ++tmpBufCount; tmpBufBytes += e.bufferSize * e.layers; }
        else              { ++tmpTexCount; tmpTexBytes += rg_texture_bytes(e.size, e.format, e.mipLevelCount) * e.layers; }
    }

    const uint64_t grand = texBytes + bufBytes + tmpTexBytes + tmpBufBytes;

    ImGui::Text("frame %llu  |  transient pool: %d held, %d in use", (unsigned long long)tp.frame, held, inUse);
    ImGui::SameLine();
    if (tp.createdThisFrame == 0)
        ImGui::TextColored(ImVec4(0.45f, 0.85f, 0.45f, 1), "  |  0 created (reused from pool)");
    else
        ImGui::TextColored(ImVec4(0.95f, 0.70f, 0.30f, 1), "  |  %u created this frame", tp.createdThisFrame);

    char gb[24]; rg_bytes_str(grand, gb, sizeof gb);
    ImGui::Text("VRAM %s total", gb);
    ImGui::SameLine(); ImGui::TextDisabled("(graph-allocated; imported/caller-owned excluded, listed below)");
    {
        char a[24], ib[24], idb[24], bb[24], bib[24], bidb[24], cb[24], cbb[24];
        rg_bytes_str(texBytes, a, sizeof a);
        rg_bytes_str(texInUseBytes, ib, sizeof ib);
        rg_bytes_str(texBytes - texInUseBytes, idb, sizeof idb);   // in use is a subset sum -> no underflow
        rg_bytes_str(bufBytes, bb, sizeof bb);
        rg_bytes_str(bufInUseBytes, bib, sizeof bib);
        rg_bytes_str(bufBytes - bufInUseBytes, bidb, sizeof bidb);
        rg_bytes_str(tmpTexBytes, cb, sizeof cb);
        rg_bytes_str(tmpBufBytes, cbb, sizeof cbb);
        ImGui::BulletText("transient tex  %-9s  %d held (%s in use, %s idle)", a, texHeld, ib, idb);
        ImGui::BulletText("transient buf  %-9s  %d held (%s in use, %s idle)", bb, bufHeld, bib, bidb);
        ImGui::BulletText("pool tex       %-9s  %d entries", cb, tmpTexCount);
        ImGui::BulletText("pool buf       %-9s  %d entries", cbb, tmpBufCount);
    }

    // memory saved by aliasing: each packed transient would otherwise own a slot-sized object, so the win is
    // the slot bytes counted once per logical member minus once per physical slot. mirrors the lifetime view.
    if (s.m_slotCount) {
        auto phBytes = [](const PhysicalResource& ph) -> uint64_t {
            return ph.kind == ResourceKind::Buffer ? ph.bufferSize : texture_bytes(ph.size, ph.format);
        };
        uint32_t logical = 0; uint64_t logicalBytes = 0, physicalBytes = 0;
        for (uint32_t i = 0; i < s.m_slotCount; ++i) physicalBytes += phBytes(s.m_slots[i]);
        for (ResourceNode* r = s.m_resouces; r; r = r->next)
            if (r->aliasSlot != ResourceNode::kNoSlot) { ++logical; logicalBytes += phBytes(s.m_slots[r->aliasSlot]); }
        char sb[24]; rg_bytes_str(logicalBytes - physicalBytes, sb, sizeof sb);
        ImGui::TextColored(ImVec4(0.59f, 0.90f, 0.59f, 1),
            "aliasing: %u transients packed onto %u objects, saved %s", logical, s.m_slotCount, sb);
    }

    // transient (memoryless) attachments: inferred from the usage gather (detect_transient_attachments); an
    // attachment cleared+discarded in one pass and never read never needs to leave the GPU. the feature flag
    // decides whether the TRANSIENT_ATTACHMENT bit reaches the driver; the list shows exactly what qualified.
    {
        if (s.m_allocator->transientFeatureOn)
            ImGui::TextColored(ImVec4(0.59f, 0.90f, 0.59f, 1),
                "transient attachments: feature ON  |  %u memoryless (TRANSIENT_ATTACHMENT emitted; driver may skip VRAM)",
                s.transientCount);
        else
            ImGui::TextColored(ImVec4(0.85f, 0.80f, 0.45f, 1),
                "transient attachments: feature OFF  |  %u candidate(s) (bit dropped -> realized as plain RENDER_ATTACHMENT)",
                s.transientCount);
        if (ImGui::IsItemHovered() && ImGui::BeginTooltip()) {
            ImGui::TextUnformatted(
                "A graph-owned attachment cleared + discarded in one render pass, never sampled/copied/stored,\n"
                "never needs to leave the GPU. On a TBDR/tile-memory device it stays on-chip and skips its VRAM\n"
                "allocation. Inferred from usage; no API declares it. No-op on D3D12 (feature absent there).");
            ImGui::EndTooltip();
        }
        const ImGuiTableFlags ttf = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit;
        if (s.transientCount && ImGui::BeginTable("rg_transient", 4, ttf)) {
            ImGui::TableSetupColumn("resource");
            ImGui::TableSetupColumn("size");
            ImGui::TableSetupColumn("format");
            ImGui::TableSetupColumn("samples");
            ImGui::TableHeadersRow();
            for (ResourceNode* r = s.m_resouces; r; r = r->next) {
                if (!r->transientAttachment) continue;
                ImGui::TableNextRow();
                ImGui::TableNextColumn(); ImGui::Text("%.*s", (int)r->id.name.length, r->id.name.data ? r->id.name.data : "");
                ImGui::TableNextColumn(); ImGui::Text("%ux%u", r->resolved.width, r->resolved.height);
                ImGui::TableNextColumn(); ImGui::Text("%s", rg_format_name(r->format));
                ImGui::TableNextColumn(); ImGui::Text("%ux", r->sampleCount);
            }
            ImGui::EndTable();
        }
        ImGui::Spacing();
    }

    ImGui::TextDisabled("tex usage A=attach T=sampled S=storage r=copy-src w=copy-dst   |   buf usage U=uniform S=storage V=vertex I=index X=indirect   |   evict after %llu idle frames",
        (unsigned long long)TransientResourcePool::kRetain);
    ImGui::Separator();

    const ImGuiTableFlags tf = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit;

    // currently held physical textures.
    ImGui::TextDisabled("transient textures");
    if (ImGui::BeginTable("tp_live", 7, tf)) {
        ImGui::TableSetupColumn("size");
        ImGui::TableSetupColumn("mips");
        ImGui::TableSetupColumn("layers");
        ImGui::TableSetupColumn("format");
        ImGui::TableSetupColumn("usage");
        ImGui::TableSetupColumn("mem");
        ImGui::TableSetupColumn("state");
        ImGui::TableHeadersRow();
        int idx = 0;
        for (const TransientResourcePool::Entry& e : tp.entries) {
            if (e.kind == ResourceKind::Buffer) continue;   // buffers listed in their own table below
            char ub[8]; rg_usage_str(e.usage, ub, sizeof ub);
            const uint64_t eb = rg_entry_bytes(e);
            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::Text("%ux%u", e.size.width, e.size.height);
            ImGui::TableNextColumn(); ImGui::Text("%u", e.mipLevelCount);
            ImGui::TableNextColumn(); ImGui::Text("%u", e.size.depthOrArrayLayers);
            ImGui::TableNextColumn(); ImGui::Text("%s", rg_format_name(e.format));
            ImGui::TableNextColumn(); ImGui::Text("%s", ub);
            ImGui::TableNextColumn();
            if (eb) { char mb[24]; rg_bytes_str(eb, mb, sizeof mb); ImGui::Text("%s", mb); }
            else    ImGui::TextDisabled("?");
            ImGui::TableNextColumn();
            if (e.inUse) ImGui::TextColored(ImVec4(0.55f, 0.80f, 1.0f, 1), "in use");
            else         ImGui::TextDisabled("idle %lluf", (unsigned long long)(tp.frame - e.lastUsedFrame));
        }
        ImGui::EndTable();
    }

    // currently held physical buffers (the buffer arm of the same pool, tagged kind == Buffer).
    ImGui::Spacing();
    ImGui::TextDisabled("transient buffers");
    if (ImGui::BeginTable("tp_buf", 4, tf)) {
        ImGui::TableSetupColumn("size");
        ImGui::TableSetupColumn("usage");
        ImGui::TableSetupColumn("mem");
        ImGui::TableSetupColumn("state");
        ImGui::TableHeadersRow();
        bool any = false;
        for (const TransientResourcePool::Entry& e : tp.entries) {
            if (e.kind != ResourceKind::Buffer) continue;
            any = true;
            char ub[12]; rg_buf_usage_str(e.bufUsage, ub, sizeof ub);
            char mb[24]; rg_bytes_str(e.bufferSize, mb, sizeof mb);
            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::Text("%llu B", (unsigned long long)e.bufferSize);
            ImGui::TableNextColumn(); ImGui::Text("%s", ub);
            ImGui::TableNextColumn(); ImGui::Text("%s", mb);
            ImGui::TableNextColumn();
            if (e.inUse) ImGui::TextColored(ImVec4(0.55f, 0.80f, 1.0f, 1), "in use");
            else         ImGui::TextDisabled("idle %lluf", (unsigned long long)(tp.frame - e.lastUsedFrame));
        }
        ImGui::EndTable();
        if (!any) ImGui::TextDisabled("(none)");
    }

    // persistent + history textures (PersistentResourcePool): one row per name; the layers column reads 2 for
    // a ping-pong history entry, 1 for a persistent (single in-place) one.
    ImGui::Spacing();
    ImGui::TextDisabled("persistent + history textures  (layers: 2 = history ping-pong, 1 = persistent)");
    if (ImGui::BeginTable("tp_tmp", 7, tf)) {
        ImGui::TableSetupColumn("name");
        ImGui::TableSetupColumn("size");
        ImGui::TableSetupColumn("mips");
        ImGui::TableSetupColumn("layers");
        ImGui::TableSetupColumn("format");
        ImGui::TableSetupColumn("usage");
        ImGui::TableSetupColumn("mem");
        ImGui::TableHeadersRow();
        bool any = false;
        for (const PersistentResourcePool::Entry& e : pool.entries) {
            if (!e.created || e.bufferSize) continue;   // buffer-arm entries listed in their own table below
            any = true;
            char ub[8]; rg_usage_str(e.usage, ub, sizeof ub);
            const uint64_t eb = rg_texture_bytes(e.size, e.format, e.mipLevelCount) * e.layers;
            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::Text("%s", e.name.c_str());
            ImGui::TableNextColumn(); ImGui::Text("%ux%u", e.size.width, e.size.height);
            ImGui::TableNextColumn(); ImGui::Text("%u", e.mipLevelCount);
            ImGui::TableNextColumn(); ImGui::Text("%u x%u", e.size.depthOrArrayLayers, e.layers);
            ImGui::TableNextColumn(); ImGui::Text("%s", rg_format_name(e.format));
            ImGui::TableNextColumn(); ImGui::Text("%s", ub);
            ImGui::TableNextColumn();
            if (eb) { char mb[24]; rg_bytes_str(eb, mb, sizeof mb); ImGui::Text("%s", mb); }
            else    ImGui::TextDisabled("?");
        }
        ImGui::EndTable();
        if (!any) ImGui::TextDisabled("(none)");
    }

    // persistent + history buffers (PersistentResourcePool buffer arm): one row per name; mem already scales
    // by layers (2 = history ping-pong, 1 = persistent single in-place).
    ImGui::Spacing();
    ImGui::TextDisabled("persistent + history buffers");
    if (ImGui::BeginTable("tp_tmpbuf", 3, tf)) {
        ImGui::TableSetupColumn("name");
        ImGui::TableSetupColumn("usage");
        ImGui::TableSetupColumn("mem (x layers)");
        ImGui::TableHeadersRow();
        bool any = false;
        for (const PersistentResourcePool::Entry& e : pool.entries) {
            if (!e.created || !e.bufferSize) continue;
            any = true;
            char ub[12]; rg_buf_usage_str(e.bufUsage, ub, sizeof ub);
            char mb[24]; rg_bytes_str(e.bufferSize * e.layers, mb, sizeof mb);
            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::Text("%s", e.name.c_str());
            ImGui::TableNextColumn(); ImGui::Text("%s", ub);
            ImGui::TableNextColumn(); ImGui::Text("%s", mb);
        }
        ImGui::EndTable();
        if (!any) ImGui::TextDisabled("(none)");
    }

    // imported (caller-owned) resources: the swapchain target + any import_image/import_buffer. real GPU
    // memory the graph writes but does NOT allocate, so it sits outside the VRAM total above. importe_image
    // records a size but no format, import_buffer records neither, and the graph never owns the bytes, so the
    // mem column stays blank here; the point is only to make the caller-owned surface visible, not double-count.
    ImGui::Spacing();
    ImGui::TextDisabled("imported (caller-owned, not in VRAM total)");
    if (ImGui::BeginTable("tp_imported", 3, tf)) {
        ImGui::TableSetupColumn("name");
        ImGui::TableSetupColumn("kind");
        ImGui::TableSetupColumn("size");
        ImGui::TableHeadersRow();
        bool any = false;
        for (ResourceNode* r = s.m_resouces; r; r = r->next) {
            if (!r->imported) continue;
            any = true;
            const bool isBuf = r->kind == ResourceKind::Buffer;
            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::Text("%.*s", (int)r->id.name.length, r->id.name.data ? r->id.name.data : "");
            ImGui::TableNextColumn(); ImGui::Text("%s", isBuf ? "buffer" : "texture");
            ImGui::TableNextColumn();
            if (isBuf) ImGui::TextDisabled("-");
            else       ImGui::Text("%u x %u", r->resolved.width, r->resolved.height);
        }
        ImGui::EndTable();
        if (!any) ImGui::TextDisabled("(none)");
    }

    ImGui::Spacing();
    ImGui::Text("transient pool events (newest first)");
    if (ImGui::Button("reset log")) tp.log_reset();
    ImGui::BeginChild("tp_log", ImVec2(0, 0), true);
    const uint64_t total = tp.eventCount;
    const uint64_t shown = total < TransientResourcePool::kLog ? total : TransientResourcePool::kLog;
    for (uint64_t k = 0; k < shown; ++k) {
        const TransientResourcePool::LogRec& r = tp.eventLog[(total - 1 - k) % TransientResourcePool::kLog];
        const bool create = r.event == TransientResourcePool::Event::Create;
        const ImVec4 col = create ? ImVec4(0.95f, 0.70f, 0.30f, 1) : ImVec4(0.60f, 0.60f, 0.60f, 1);
        if (r.kind == ResourceKind::Buffer)
            ImGui::TextColored(col, "f%-6llu  %-6s  buf %llu B", (unsigned long long)r.frame, create ? "CREATE" : "evict",
                (unsigned long long)r.bufferSize);
        else
            ImGui::TextColored(col, "f%-6llu  %-6s  %ux%u  %s", (unsigned long long)r.frame, create ? "CREATE" : "evict",
                r.size.width, r.size.height, rg_format_name(r.format));
    }
    if (shown == 0) ImGui::TextDisabled("(no events yet)");
    ImGui::EndChild();
}

// One usage bar for a single growable, block-chained arena: `value` bytes (live for `front`, per-frame
// peak for `scratch`, which is rewound to empty by draw time) filled against the arena's CURRENT
// capacity, which itself grows as blocks are chained on. The block count is shown in the overlay: > 1 means
// the arena outgrew its first block this run. label doubles as the InvisibleButton id, so keep them unique.
static void rg_draw_arena_bar(const char* label, const char* valueLabel, size_t value, const Arena& arena, ImU32 color)
{
    const double kib  = 1024.0;
    const double cap  = arena.totalCapacity ? (double)arena.totalCapacity : 1.0;
    float        frac = (float)((double)value / cap);
    if (frac > 1.0f) frac = 1.0f;
    if (frac < 0.0f) frac = 0.0f;

    constexpr float kLabelW = 56.0f;   // fixed gutter so both bars' left edges line up under each other
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(label);
    ImGui::SameLine(kLabelW);

    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    float        w  = ImGui::GetContentRegionAvail().x;
    if (w < 1.0f) w = 1.0f;                                   // collapsed window -> keep InvisibleButton happy
    const float  h  = ImGui::GetFrameHeight();
    const ImVec2 p1(p0.x + w, p0.y + h);

    ImGui::InvisibleButton(label, ImVec2(w, h));
    const bool  hov = ImGui::IsItemHovered();
    ImDrawList* dl  = ImGui::GetWindowDrawList();

    dl->AddRectFilled(p0, p1, IM_COL32(28, 28, 28, 255), 3.0f);                       // track
    if (frac > 0.0f)
        dl->AddRectFilled(p0, ImVec2(p0.x + w * frac, p1.y), color, 0.0f);            // fill
    dl->AddRect(p0, p1, hov ? IM_COL32(255, 255, 255, 255) : IM_COL32(20, 20, 20, 180), 3.0f);

    char ov[80];
    std::snprintf(ov, sizeof ov, "%.1f / %.0f KB  x%llu blk", value / kib, cap / kib,
        (unsigned long long)arena.blockCount);
    const ImVec2 ts = ImGui::CalcTextSize(ov);
    dl->AddText(ImVec2(p0.x + (w - ts.x) * 0.5f, p0.y + (h - ts.y) * 0.5f), IM_COL32(235, 235, 235, 255), ov);

    if (hov) {
        ImGui::BeginTooltip();
        ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(color), "%s", label);
        ImGui::Text("%-9s%llu B  (%.2f KB)", valueLabel, (unsigned long long)value, value / kib);
        ImGui::Text("capacity %.2f KB  (grows)", cap / kib);
        ImGui::Text("blocks   %llu  x %.0f KB", (unsigned long long)arena.blockCount, arena.blockSize / kib);
        ImGui::EndTooltip();
    }
}

// The per-frame GraphAllocator as two stacked usage bars: it is two independent, growable, block-chained
// arenas (not one buffer): `front` holds the frame's permanent objects (RenderGraph + storage, every
// resource/pass node, the type-erased execute closures, copied strings); `scratch` holds compile()'s
// temporaries, rewound per scope, so we show its per-frame high-water rather than the (empty) live value.
static void rg_draw_arena(GraphAllocator& a)
{
    rg_draw_arena_bar("front",   "used", a.front.live_used(), a.front,   kRGRead);
    rg_draw_arena_bar("scratch", "peak", a.scratch.peakUsage, a.scratch, kRGWrite);
}

// distinct line colors for the timing series, cycled by series index. kept opaque/bright so thin polylines
// read against the dark canvas.
static ImU32 rg_series_color(uint32_t i)
{
    static const ImU32 pal[] = {
        IM_COL32(235, 110, 110, 255), IM_COL32(110, 200, 120, 255), IM_COL32(110, 170, 240, 255),
        IM_COL32(235, 200, 90, 255),  IM_COL32(200, 130, 230, 255), IM_COL32(90, 215, 215, 255),
        IM_COL32(240, 150, 90, 255),  IM_COL32(170, 220, 110, 255), IM_COL32(150, 150, 240, 255),
        IM_COL32(230, 130, 180, 255), IM_COL32(120, 220, 170, 255), IM_COL32(210, 210, 130, 255),
    };
    return pal[i % (sizeof(pal) / sizeof(pal[0]))];
}

// The "Timings" tab: record per-pass GPU us over time and draw it as a running multi-line graph, one line
// per pass (filtered by the per-pass checkboxes). Sampling itself happens in imgui_layer_draw_graph so it
// keeps running while another tab is foregrounded.
static void rg_draw_timings(RenderGraphStorage& s)
{
    GpuProfiler& gp = s.m_allocator->profiler;

    if (!gp.initialized) {
        ImGui::TextDisabled("Enable \"gpu timings\" in the Demos window to record per-pass GPU time.");
        return;
    }

    if (gp.recording) { if (ImGui::Button("Stop recording")) gp.recording = false; }
    else              { if (ImGui::Button("Start recording")) gp.recording = true; }
    ImGui::SameLine();
    if (ImGui::Button("Clear")) gp.clear_history();
    ImGui::SameLine();
    ImGui::Text("%u / %u frames%s", gp.historyLen, GpuProfiler::kHistory, gp.recording ? "  (rec)" : "");

    // peak us across the ENABLED series over the retained window -> the graph's Y scale.
    float maxV = 0.0f;
    for (uint32_t si = 0; si < gp.seriesCount; ++si)
        if (gp.series[si].enabled)
            for (uint32_t c = 0; c < gp.historyLen; ++c)
                if (gp.series[si].v[c] > maxV) maxV = gp.series[si].v[c];
    if (maxV <= 0.0f) maxV = 1.0f;   // avoid div-by-zero before any data

    // ---- canvas ----
    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    const float  w  = ImGui::GetContentRegionAvail().x;
    const float  h  = 200.0f;
    ImGui::InvisibleButton("rg_timings_canvas", ImVec2(w > 0 ? w : 1, h));
    const bool  canvasHovered = ImGui::IsItemHovered();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 p1(p0.x + w, p0.y + h);
    dl->AddRectFilled(p0, p1, IM_COL32(18, 18, 24, 255), 3.0f);
    dl->AddRect(p0, p1, IM_COL32(80, 86, 100, 255), 3.0f);
    dl->PushClipRect(p0, p1, true);

    // faint mid gridline + the peak label
    dl->AddLine(ImVec2(p0.x, (p0.y + p1.y) * 0.5f), ImVec2(p1.x, (p0.y + p1.y) * 0.5f), IM_COL32(255, 255, 255, 24));
    char lbl[32];
    std::snprintf(lbl, sizeof(lbl), "%.1f us", maxV);
    dl->AddText(ImVec2(p0.x + 4, p0.y + 3), IM_COL32(180, 180, 190, 220), lbl);

    // ring geometry, shared by the line draw + the scrub cursor. valid only with >=1 column.
    const uint32_t oldest = (gp.historyHead + GpuProfiler::kHistory - gp.historyLen) % GpuProfiler::kHistory;
    const float    stepX  = w / float(GpuProfiler::kHistory - 1);
    auto colY = [&](float v) { return p1.y - (v / maxV) * (h - 4.0f) - 2.0f; };

    // scrub: snap the mouse X to the nearest column while hovering -> the cursor line + the list readout.
    int scrubCol = -1;   // logical column in [0, historyLen)
    if (canvasHovered && gp.historyLen >= 1) {
        int c = (int)((ImGui::GetIO().MousePos.x - p0.x) / stepX + 0.5f);
        scrubCol = c < 0 ? 0 : (c > (int)gp.historyLen - 1 ? (int)gp.historyLen - 1 : c);
    }

    // oldest valid column in the ring -> draw left to right
    if (gp.historyLen >= 2) {
        static ImVec2 pts[GpuProfiler::kHistory];
        for (uint32_t si = 0; si < gp.seriesCount; ++si) {
            if (!gp.series[si].enabled) continue;
            for (uint32_t c = 0; c < gp.historyLen; ++c)
                pts[c] = ImVec2(p0.x + c * stepX, colY(gp.series[si].v[(oldest + c) % GpuProfiler::kHistory]));
            dl->AddPolyline(pts, (int)gp.historyLen, rg_series_color(si), 0, 1.5f);
        }
    }

    // scrub cursor: vertical line + a dot on each enabled series at the snapped column
    if (scrubCol >= 0) {
        const float cx = p0.x + scrubCol * stepX;
        dl->AddLine(ImVec2(cx, p0.y), ImVec2(cx, p1.y), IM_COL32(230, 230, 235, 160));
        const uint32_t col = (oldest + (uint32_t)scrubCol) % GpuProfiler::kHistory;
        for (uint32_t si = 0; si < gp.seriesCount; ++si)
            if (gp.series[si].enabled)
                dl->AddCircleFilled(ImVec2(cx, colY(gp.series[si].v[col])), 2.5f, rg_series_color(si));
    }
    dl->PopClipRect();

    // the list reads the scrubbed column while hovering, else the latest sample.
    const uint32_t readCol = gp.historyLen
        ? (oldest + (uint32_t)(scrubCol >= 0 ? scrubCol : (int)gp.historyLen - 1)) % GpuProfiler::kHistory
        : 0;
    ImGui::Spacing();
    if (scrubCol >= 0) ImGui::TextDisabled("scrub: %d frames ago", (int)gp.historyLen - 1 - scrubCol);
    else               ImGui::TextDisabled("latest sample (hover the graph to scrub)");

    // ---- per-pass list: color swatch + enable checkbox + value at readCol ----
    for (uint32_t si = 0; si < gp.seriesCount; ++si) {
        const ImVec2 cur = ImGui::GetCursorScreenPos();
        dl->AddRectFilled(ImVec2(cur.x, cur.y + 3), ImVec2(cur.x + 11, cur.y + 14), rg_series_color(si), 2.0f);
        ImGui::Dummy(ImVec2(15, 0));
        ImGui::SameLine();
        char id[80];
        std::snprintf(id, sizeof(id), "%s##series%u", gp.series[si].name, si);
        ImGui::Checkbox(id, &gp.series[si].enabled);
        ImGui::SameLine(ImGui::GetContentRegionAvail().x - 70.0f);
        ImGui::Text("%6.1f us", gp.historyLen ? gp.series[si].v[readCol] : 0.0f);
    }
}

namespace webgpu_app {

void RenderGraphPanel::draw()
{
    static uint32_t render_gaph_enabled = 0;
    if(ImGuiManager::FloatingToggleButton("ToggleRenderGraphButton", ICON_FA_PROJECT_DIAGRAM, "RenderGraph", &render_gaph_enabled))
    {
        render_gaph_enabled &= 1;
    }

    if(render_gaph_enabled == 0)
        return;

    ImGui::Begin("RenderGraph");

    // Keep the toggle reachable whether or not a graph ran this frame, so the user can switch back from the
    // legacy path (which feeds no graph, leaving m_graph null).
    ImGui::Checkbox("Drive frame from render graph", &g_use_render_graph);
    ImGui::Separator();

    if (!m_graph) {
        ImGui::TextUnformatted("Render graph inactive (legacy render path).");
        ImGui::End();
        return;
    }

    webgpu::rg::RenderGraph* rg = m_graph;
    RenderGraphStorage& s = *storage(rg);

    // append a history column each frame the window draws, independent of which tab is open.
    s.m_allocator->profiler.sample_history();

    ImGui::Text(" %1.1f FPS (%1.2f ms)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
    ImGui::Text(" compile %1.0f us  realize %1.0f us  execute %1.0f us", s.timing_compile_us, s.timing_realize_us, s.timing_execute_us);

    if (ErrorMessage* e = rg->get_errors()) {
        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(230, 90, 80, 255));
        ImGui::TextUnformatted("compile failed:");
        for (; e; e = e->next)
            ImGui::TextWrapped("  %.*s", (int)e->message.length, e->message.data ? e->message.data : "");
        ImGui::PopStyleColor();
        ImGui::Separator();
    }

    // GPU timing (opt-in): total of the last completed per-pass timestamp read-back, breakdown on hover.
    if (const GpuProfiler& gp = s.m_allocator->profiler; gp.resultCount) {
        float totalUs = 0.0f;
        for (uint32_t i = 0; i < gp.resultCount; ++i)
            totalUs += gp.resultUs[i];
        ImGui::SameLine();
        ImGui::Text("  gpu %.2f ms", totalUs / 1000.0f);
        if (ImGui::IsItemHovered() && ImGui::BeginTooltip()) {
            for (uint32_t i = 0; i < gp.resultCount; ++i)
                ImGui::Text("%6.1f us  %s", gp.resultUs[i], gp.resultNames[i]);
            ImGui::EndTooltip();
        }
    }

    rg_draw_arena(*s.m_allocator);
    ImGui::Separator();

    if (ImGui::BeginTabBar("rg_tabs")) {
        if (ImGui::BeginTabItem("Graph")) {
            rg_draw_dag(rg, s);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Lifetimes")) {
            rg_draw_lifetimes(rg, s);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Memory")) {
            rg_draw_memory(s);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Timings")) {
            rg_draw_timings(s);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::End();

    m_graph = nullptr; // set to nullptr because the render graph is only valid for 0 frame
}

} // namespace webgpu_app

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
