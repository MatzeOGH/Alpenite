/*****************************************************************************
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


#include "UnittestWebgpuContext.h"
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <string>
#include <vector>
#include <webgpu/base/RenderGraph.h>
#include <webgpu/base/RenderGraph_internal.h>
#include <webgpu/base/raii/BindGroup.h>
#include <webgpu/base/raii/RawBuffer.h>

using namespace webgpu;
using namespace webgpu::rg;
using webgpu::rg::Internal::AccessType;
using webgpu::rg::Internal::PassNode;
using webgpu::rg::Internal::ResourceAccess;
using webgpu::rg::Internal::storage;

struct TestGraph {
    GraphAllocator* allocator = create_allocator();
    RenderGraph* rg;

    TestGraph()
    {
        begin_frame(allocator);
        rg = start_recording(allocator);
    }
    ~TestGraph() { destroy_allocator(allocator); }

    TestGraph(const TestGraph&) = delete;
    TestGraph& operator=(const TestGraph&) = delete;

    ResourceHandle transient(std::string_view id, uint32_t mipLevelCount = 1)
    {
        TextureDesc desc {};
        desc.dimension = WGPUTextureDimension_2D;
        desc.format = WGPUTextureFormat_RGBA8Unorm;
        desc.absolute = { 16, 16, 1 };
        desc.mipLevelCount = mipLevelCount;
        return rg->create_transient_texture(id, desc);
    }

    ResourceHandle transient_buffer(std::string_view id, uint64_t size = 64)
    {
        BufferDesc desc {};
        desc.size = size;
        return rg->create_transient_buffer(id, desc);
    }
};

// declare a compute producer for buf so a later copy_buffer src is not a read-before-write error
static void add_buffer_producer(RenderGraph* rg, ResourceHandle buf)
{
    rg->add_pass(
        "produce", PassKind::Compute, [&](PassBuilder& b) { b.storage_write(buf); }, [](PassContext&) {});
}

// A 16x16 RGBA8Unorm 2D descriptor, matching TestGraph::transient, for the history/persistent/import
// helpers below.
static TextureDesc tex2d()
{
    TextureDesc d {};
    d.dimension = WGPUTextureDimension_2D;
    d.format = WGPUTextureFormat_RGBA8Unorm;
    d.absolute = { 16, 16, 1 };
    return d;
}

// Imported texture with a fake non-null view. compile() only reads the `imported` flag and never
// dereferences the view; realize()/execute() would, but the culling/aliasing tests below stop at
// compile(). (import_buffer, by contrast, calls wgpuBufferGetSize, so import textures here.)
static ResourceHandle import_tex(RenderGraph* rg, std::string_view id) { return rg->import_texture(id, (WGPUTextureView)0x1, { 16, 16, 1 }, WGPUTextureFormat_RGBA8Unorm); }

// Surviving passes in execution order after compile() (compile relinks m_passes into topo order).
static std::vector<std::string> pass_order(RenderGraph* rg)
{
    std::vector<std::string> v;
    for (PassNode* p = storage(rg)->m_passes; p; p = p->next)
        v.emplace_back(p->id.name.data, p->id.name.length);
    return v;
}
static int idx_of(const std::vector<std::string>& v, const char* n)
{
    for (int i = 0; i < static_cast<int>(v.size()); ++i)
        if (v[i] == n)
            return i;
    return -1;
}

// ---------------------------------------------------------------------------------------------------
// ViewRange presets and cube-view validation.

// True when any compile() error mentions `needle`, so a test pins the specific diagnostic rather than
// just "some error fired".
static bool error_mentions(RenderGraph* rg, std::string_view needle)
{
    for (ErrorMessage* e = rg->get_errors(); e; e = e->next)
        if (std::string_view(e->message.data, e->message.length).find(needle) != std::string_view::npos)
            return true;
    return false;
}


TEST_CASE("rg::cube preset builds a 6-layer cube view", "[RenderGraph]")
{
    constexpr ViewRange c = cube();
    STATIC_REQUIRE(c.dim == WGPUTextureViewDimension_Cube);
    STATIC_REQUIRE(c.layerCount == 6); // the whole point: the bare ViewRange default (1) is an invalid cube
    STATIC_REQUIRE(c.mipCount == 1); // default stays single-mip
    STATIC_REQUIRE(c.aspect == WGPUTextureAspect_All);

    // aspect passes through, for a depth shadow cube
    STATIC_REQUIRE(cube(WGPUTextureAspect_DepthOnly).aspect == WGPUTextureAspect_DepthOnly);
    // mipCount 0 == all remaining, for a prefiltered IBL env cube spanning its whole mip chain
    STATIC_REQUIRE(cube(WGPUTextureAspect_All, 0).mipCount == 0);
}

TEST_CASE("rg::cube_array preset builds a 6*N-layer cube-array view", "[RenderGraph]")
{
    STATIC_REQUIRE(cube_array(1).layerCount == 6);
    STATIC_REQUIRE(cube_array(3).layerCount == 18);
    STATIC_REQUIRE(cube_array(3).dim == WGPUTextureViewDimension_CubeArray);
    STATIC_REQUIRE(cube_array(3).mipCount == 1);
    STATIC_REQUIRE(cube_array(2, WGPUTextureAspect_All, 0).mipCount == 0);
}

TEST_CASE("rg::whole preset is all mips and all layers with an inferred dimension", "[RenderGraph]")
{
    constexpr ViewRange w = whole();
    STATIC_REQUIRE(w.dim == WGPUTextureViewDimension_Undefined); // inferred from the texture
    STATIC_REQUIRE(w.mipCount == 0); // 0 == all remaining
    STATIC_REQUIRE(w.layerCount == 0);
}

// A transient 2D array texture with `layers` layers, written by a producer pass so the sampled consumer
// below is not a read-before-write error, then sampled through `range`. Returns the compiled graph's
// errors via g. Only compile() runs, so no device is needed.
static ResourceHandle cube_source(TestGraph& g, uint32_t layers, WGPUTextureDimension dim = WGPUTextureDimension_2D)
{
    TextureDesc desc {};
    desc.dimension = dim;
    desc.format = WGPUTextureFormat_RGBA8Unorm;
    desc.absolute = { 16, 16, layers };
    ResourceHandle h = g.rg->create_transient_texture("cubesrc", desc);
    g.rg->add_pass(
        "produce", PassKind::Graphics,
        [h, layers, dim](PassBuilder& b) {
            // a 3D texture has no array layers to attach per-layer; one attachment defines it
            uint32_t n = (dim == WGPUTextureDimension_2D) ? layers : 1;
            for (uint32_t l = 0; l < n; ++l)
                b.color(h, l, WGPULoadOp_Clear, WGPUStoreOp_Store, { 0, 0, 0, 1 }, 0, l);
        },
        [](PassContext&) {});
    return h;
}

TEST_CASE("compile - a valid 6-layer cube view passes validation", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle h = cube_source(g, 6);
    g.rg->add_pass(
        "read", PassKind::Graphics,
        [h](PassBuilder& b) {
            b.sampled(h, 0, 0, cube());
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);
}

TEST_CASE("compile - a cube view without exactly 6 layers is rejected", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle h = cube_source(g, 4);
    g.rg->add_pass(
        "read", PassKind::Graphics,
        [h](PassBuilder& b) {
            b.sampled(h, 0, 0, ViewRange { WGPUTextureViewDimension_Cube, 1, 4, WGPUTextureAspect_All });
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(error_mentions(g.rg, "needs exactly 6")); // named error, not an opaque device error
}

TEST_CASE("compile - a cube-array view whose layer count is not a multiple of 6 is rejected", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle h = cube_source(g, 8);
    g.rg->add_pass(
        "read", PassKind::Graphics,
        [h](PassBuilder& b) {
            b.sampled(h, 0, 0, ViewRange { WGPUTextureViewDimension_CubeArray, 1, 8, WGPUTextureAspect_All });
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(error_mentions(g.rg, "positive multiple of 6"));
}

// The overrun check is an `else if` after the exactly-6 / multiple-of-6 arms, so reaching it needs a view
// those arms accept: a well-formed 6-layer cube that simply starts too late in an 8-layer texture.
// (cube_source declares one color() per layer, and a slot >= kMaxColorAttachments would be dropped, so 8
// layers is the widest source available here.)
TEST_CASE("compile - a well-formed cube view starting past the layer count is rejected", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle h = cube_source(g, 8);
    g.rg->add_pass(
        "read", PassKind::Graphics,
        [h](PassBuilder& b) {
            b.sampled(h, 0, /*baseLayer*/ 4, cube()); // 4 + 6 > 8
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(error_mentions(g.rg, "exceeds the texture's 8 layer(s)"));
    REQUIRE_FALSE(error_mentions(g.rg, "needs exactly 6")); // the layer count itself is fine
}

TEST_CASE("compile - a cube view on a storage access is rejected", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle h = cube_source(g, 6);
    g.rg->add_pass(
        "read", PassKind::Compute,
        [h](PassBuilder& b) {
            b.storage_read(h, 0, 0, cube());
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(error_mentions(g.rg, "cannot be cube"));
}

TEST_CASE("compile - a cube view on a non-2D texture is rejected", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle h = cube_source(g, 6, WGPUTextureDimension_3D);
    g.rg->add_pass(
        "read", PassKind::Graphics,
        [h](PassBuilder& b) {
            b.sampled(h, 0, 0, cube());
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(error_mentions(g.rg, "not a 2D texture"));
}

// An imported 6-layer cube source (the IBL environment map path: loaded outside the graph, imported,
// sampled as a cube) must pass cube validation. import_texture defaults `dimension` to 2D precisely so
// node_layers sees the imported extent's 6 layers instead of the Undefined default's 1.
TEST_CASE("compile - an imported 6-layer cube view passes validation", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle h = g.rg->import_texture("envcube", (WGPUTextureView)0x1, { 16, 16, 6 }, WGPUTextureFormat_RGBA8Unorm);
    g.rg->add_pass(
        "read", PassKind::Graphics,
        [h](PassBuilder& b) {
            b.sampled(h, 0, 0, cube());
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);
}

// The layer count of an imported texture is its extent's, not 1: an imported 4-layer texture asked for a
// cube must be rejected for having 4 layers, which only happens if the node knows it is 2D.
TEST_CASE("compile - an imported texture reports its real layer count to cube validation", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle h = g.rg->import_texture("arr", (WGPUTextureView)0x1, { 16, 16, 4 }, WGPUTextureFormat_RGBA8Unorm);
    g.rg->add_pass(
        "read", PassKind::Graphics,
        [h](PassBuilder& b) {
            b.sampled(h, 0, 0, ViewRange { WGPUTextureViewDimension_Cube, 1, 0, WGPUTextureAspect_All }); // 0 == all remaining
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(error_mentions(g.rg, "covers 4 layer(s)")); // not "1 layer(s)"
}

// An imported 3D texture is now expressible, and a cube view of it is rejected as non-2D.
TEST_CASE("compile - an imported 3D texture is rejected for a cube view", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle h = g.rg->import_texture("vol", (WGPUTextureView)0x1, { 16, 16, 6 }, WGPUTextureFormat_RGBA8Unorm,
        nullptr, 1, 1, WGPUTextureDimension_3D);
    g.rg->add_pass(
        "read", PassKind::Graphics,
        [h](PassBuilder& b) {
            b.sampled(h, 0, 0, cube());
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(error_mentions(g.rg, "not a 2D texture"));
}

// ---------------------------------------------------------------------------------------------------
// Access range validation. An access's subresource/byte range is fully declared in setup, so compile()
// rejects a range that does not fit its resource.

TEST_CASE("compile - an access past the texture's mip count is rejected", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle h = g.transient("tex"); // 1 mip
    g.rg->add_pass(
        "write", PassKind::Graphics,
        [h](PassBuilder& b) {
            b.color(h, 0, WGPULoadOp_Clear, WGPUStoreOp_Store, {}, 3);
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(error_mentions(g.rg, "accessed at mip 3 but has 1 mip(s)"));
}

TEST_CASE("compile - an access past the texture's layer count is rejected", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle h = cube_source(g, 6);
    g.rg->add_pass(
        "read", PassKind::Graphics,
        [h](PassBuilder& b) {
            b.sampled(h, 0, 9);
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(error_mentions(g.rg, "accessed at layer 9 but has 6 layer(s)"));
}

TEST_CASE("compile - a view whose mip range ends past the chain is rejected", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle h = g.transient("chain", 2);
    g.rg->add_pass(
        "write", PassKind::Compute,
        [h](PassBuilder& b) {
            b.storage_write(h, 0, 0, ViewRange { WGPUTextureViewDimension_Undefined, 3, 1 });
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(error_mentions(g.rg, "view of 3 mip(s) from mip 0 but has 2 mip(s)"));
}

// The layer half of the same check. A non-cube viewDim keeps this out of cube validation, so the range
// check is what has to catch it.
TEST_CASE("compile - a view whose layer range ends past the array is rejected", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle h = cube_source(g, 2);
    g.rg->add_pass(
        "read", PassKind::Graphics,
        [h](PassBuilder& b) {
            b.sampled(h, 0, 0, ViewRange { WGPUTextureViewDimension_Undefined, 1, 3 }); // 0 + 3 > 2
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(error_mentions(g.rg, "view of 3 layer(s) from layer 0 but has 2 layer(s)"));
}

//  0 means "all remaining", so rg::whole() must stay clean on a chain no matter how many mips it has.
TEST_CASE("compile - a whole() view over a mip chain passes range validation", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle h = g.transient("chain", 4);
    g.rg->add_pass(
        "write", PassKind::Compute,
        [h](PassBuilder& b) {
            b.storage_write(h, 0, 0, whole());
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);
}

TEST_CASE("compile - a copy_texture past the dst's layer count is rejected", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle src = g.transient("src");
    ResourceHandle dst = g.transient("dst"); // 1 layer
    g.rg->add_pass(
        "produce", PassKind::Graphics, [src](PassBuilder& b) { b.color(src, 0); }, [](PassContext&) {});
    g.rg->add_pass(
        "copy", PassKind::Transfer,
        [src, dst](PassBuilder& b) {
            b.copy_texture(src, dst, 0, 0, 0, /*dstLayer*/ 2);
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(error_mentions(g.rg, "accessed at layer 2 but has 1 layer(s)"));
}

// The dst range is checked against the dst buffer even though the size was declared on the copy as a
// whole: copy_buffer resolves the size at declare and records it on both sides.
TEST_CASE("compile - a copy_buffer range that overruns the dst is rejected", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle bufA = g.transient_buffer("bufA", 64);
    ResourceHandle bufB = g.transient_buffer("bufB", 64);

    add_buffer_producer(g.rg, bufA);
    g.rg->add_pass(
        "copy", PassKind::Transfer,
        [bufA, bufB](PassBuilder& b) {
            b.copy_buffer(bufA, bufB, 0, /*dstOffset*/ 32, /*size*/ 64); // 32 + 64 > 64
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(error_mentions(g.rg, "covers bytes [32, 96) but the buffer is 64 byte(s)"));
}

// A size of 0 expands against the src at declare time, so an offset src copy lands a resolved (smaller)
// size on both accesses and stays inside a same-sized dst.
TEST_CASE("compile - a whole copy_buffer from an offset src fits a same-sized dst", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle bufA = g.transient_buffer("bufA", 64);
    ResourceHandle bufB = g.transient_buffer("bufB", 64);

    add_buffer_producer(g.rg, bufA);
    g.rg->add_pass(
        "copy", PassKind::Transfer,
        [bufA, bufB](PassBuilder& b) {
            b.copy_buffer(bufA, bufB, /*srcOffset*/ 32, 0, /*size*/ 0); // resolves to 32 bytes
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);
}

// ---------------------------------------------------------------------------------------------------
// compile() reports through the error chain rather than asserting, so these two cases are observable in
// any build.

TEST_CASE("compile - a cyclic relativeTo chain is rejected", "[RenderGraph]")
{
    TestGraph g;
    // The public API cannot author a cycle: a desc names an existing handle, so relativeTo only ever
    // points backwards. Build the loop on the nodes to reach the recursion guard resolve_size carries.
    ResourceHandle a = g.transient("a");
    TextureDesc rel {};
    rel.dimension = WGPUTextureDimension_2D;
    rel.format = WGPUTextureFormat_RGBA8Unorm;
    rel.sizeKind = SizeKind::Relative;
    rel.relativeTo = a;
    ResourceHandle b = g.rg->create_transient_texture("b", rel);

    Internal::ResourceNode* an = Internal::find_node(g.rg, a);
    an->sizeKind = SizeKind::Relative; // close the loop: a sizes against b, b sizes against a
    an->relativeToHandle = b;

    g.rg->compile();
    REQUIRE(error_mentions(g.rg, "cyclic relativeTo chain"));
}

// ---------------------------------------------------------------------------------------------------
// Foreign/stale handle backstops. These are not authoring diagnostics: an out-of-range id would index
// compile()'s phase-1/2 scratch tables (sized by next_resource_id) out of bounds, so the prologue checks
// every handle before any table is built. The builder's own Q_ASSERTs stop such a handle from ever being
// declared, which is why each test stamps the bad handle onto the node afterwards, the same white-box
// route the cyclic relativeTo test above takes. Each also checks byId, which compile() allocates only
// after the prologue clears every handle: a null byId is the witness that compile() bailed out before it
// built the tables the bad id would have indexed.

TEST_CASE("compile - an out-of-range access handle is rejected before the scratch tables are built", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle tex = g.transient("tex");
    g.rg->add_pass(
        "write", PassKind::Graphics,
        [tex](PassBuilder& b) {
            b.color(tex, 0);
            b.force_keep();
        },
        [](PassContext&) {});

    // an id past next_resource_id is the case that would read byId[] out of bounds
    storage(g.rg)->m_passes->accesses[0].handle.id = 9999;

    g.rg->compile();
    REQUIRE(error_mentions(g.rg, "uses an invalid resource handle (id 9999)"));
    REQUIRE(storage(g.rg)->byId == nullptr); // bailed before byId[9999] could be written
}

TEST_CASE("compile - an access handle from another graph is rejected", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle tex = g.transient("tex");
    g.rg->add_pass(
        "write", PassKind::Graphics,
        [tex](PassBuilder& b) {
            b.color(tex, 0);
            b.force_keep();
        },
        [](PassContext&) {});

    // in range, but stamped with another graph's generation: the id would resolve to the wrong node
    storage(g.rg)->m_passes->accesses[0].handle.generation = storage(g.rg)->generation + 1;

    g.rg->compile();
    REQUIRE(error_mentions(g.rg, "uses an invalid resource handle"));
    REQUIRE(storage(g.rg)->byId == nullptr);
}

TEST_CASE("compile - a foreign initialize() target handle is rejected", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle pers = g.rg->create_persistent_texture("pers", tex2d());
    g.rg->add_pass(
        "bake", PassKind::Graphics,
        [pers](PassBuilder& b) {
            b.color(pers, 0);
            b.initialize(pers);
            b.force_keep();
        },
        [](PassContext&) {});

    // only the init target goes stale; the access still carries the good handle, so the access loop passes
    // and the init loop is what has to catch this
    storage(g.rg)->m_passes->initTargets[0].target.generation = storage(g.rg)->generation + 1;

    g.rg->compile();
    REQUIRE(error_mentions(g.rg, "initialize() target handle"));
    REQUIRE(storage(g.rg)->byId == nullptr);
}

TEST_CASE("compile - a foreign relativeTo handle is rejected", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle base = g.transient("base");
    TextureDesc rel {};
    rel.dimension = WGPUTextureDimension_2D;
    rel.format = WGPUTextureFormat_RGBA8Unorm;
    rel.sizeKind = SizeKind::Relative;
    rel.relativeTo = base;
    ResourceHandle scaled = g.rg->create_transient_texture("scaled", rel);

    // resolve_size() indexes relativeTo unguarded, so the prologue is the only thing between a stale
    // relativeTo and an out-of-bounds read
    Internal::find_node(g.rg, scaled)->relativeToHandle.generation = storage(g.rg)->generation + 1;

    g.rg->compile();
    REQUIRE(error_mentions(g.rg, "relativeTo"));
    REQUIRE(error_mentions(g.rg, "from another graph or a previous frame"));
    REQUIRE(storage(g.rg)->byId == nullptr);
}

TEST_CASE("compile - a second compile() on the same graph is rejected", "[RenderGraph]")
{
    TestGraph g;
    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);

    g.rg->compile();
    REQUIRE(error_mentions(g.rg, "already compiled"));
}

// ---------------------------------------------------------------------------------------------------
// Explicit color-attachment slots

// the ColorAttachment access for `h` in `p`, or null when the pass declares none
static const ResourceAccess* color_access(PassNode* p, ResourceHandle h)
{
    for (uint32_t i = 0; i < p->accessCount; ++i)
        if (p->accesses[i].type == AccessType::ColorAttachment && p->accesses[i].handle.id == h.id)
            return &p->accesses[i];
    return nullptr;
}

static uint32_t count_access(PassNode* p, AccessType t)
{
    uint32_t n = 0;
    for (uint32_t i = 0; i < p->accessCount; ++i)
        if (p->accesses[i].type == t)
            ++n;
    return n;
}

TEST_CASE("PassBuilder::color - the declared slot is carried on the access", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle a = g.transient("a");
    ResourceHandle b = g.transient("b");

    g.rg->add_pass(
        "mrt", PassKind::Graphics,
        [a, b](PassBuilder& pb) {
            pb.color(a, 3);
            pb.color(b, 1);
            pb.force_keep();
        },
        [](PassContext&) {});

    PassNode* p = storage(g.rg)->m_passes;
    REQUIRE(color_access(p, a)->colorIndex == 3);
    REQUIRE(color_access(p, b)->colorIndex == 1); // declared second, but slot 1
}

// Release-only: the four tests below feed color()/resolve() input that is an author error, which the
// builder guards with Q_ASSERT. In a debug build the assert fires and aborts (by design, and the abort
// IS the contract there), so these can only observe the release backstop
#ifdef QT_NO_DEBUG
TEST_CASE("PassBuilder::color - a slot declared twice in one pass is rejected", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle a = g.transient("a");
    ResourceHandle b = g.transient("b");

    g.rg->add_pass(
        "dup", PassKind::Graphics,
        [a, b](PassBuilder& pb) {
            pb.color(a, 0);
            pb.color(b, 0); // same slot -> rejected, access not recorded
            pb.force_keep();
        },
        [](PassContext&) {});

    PassNode* p = storage(g.rg)->m_passes;
    REQUIRE(count_access(p, AccessType::ColorAttachment) == 1);
    REQUIRE(color_access(p, a) != nullptr);
    REQUIRE(color_access(p, b) == nullptr); // the duplicate never bound
}
#endif

#ifdef QT_NO_DEBUG
TEST_CASE("PassBuilder::color - a slot at or past kMaxColorAttachments is rejected", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle a = g.transient("a");

    g.rg->add_pass(
        "oob", PassKind::Graphics,
        [a](PassBuilder& pb) {
            pb.color(a, webgpu::rg::Internal::kMaxColorAttachments);
            pb.force_keep();
        },
        [](PassContext&) {});

    REQUIRE(count_access(storage(g.rg)->m_passes, AccessType::ColorAttachment) == 0);
}
#endif

TEST_CASE("PassBuilder::resolve - the target carries its src color slot", "[RenderGraph]")
{
    TestGraph g;
    TextureDesc msaaDesc = tex2d();
    msaaDesc.sampleCount = 4;
    ResourceHandle msaa = g.rg->create_transient_texture("msaa", msaaDesc);
    ResourceHandle target = g.transient("target");

    g.rg->add_pass(
        "fwd", PassKind::Graphics,
        [msaa, target](PassBuilder& b) {
            b.color(msaa, 2);
            b.resolve(msaa, target);
            b.force_keep();
        },
        [](PassContext&) {});

    PassNode* p = storage(g.rg)->m_passes;
    bool found = false;
    for (uint32_t i = 0; i < p->accessCount; ++i)
        if (p->accesses[i].type == AccessType::ResolveAttachment && p->accesses[i].handle.id == target.id) {
            REQUIRE(p->accesses[i].colorIndex == 2); // the slot of the color() it resolves, not 0
            found = true;
        }
    REQUIRE(found);
}

// The pairing no longer depends on declaration adjacency: an unrelated color() may sit between the
// resolved color() and its resolve().
TEST_CASE("PassBuilder::resolve - src pairing does not depend on declaration adjacency", "[RenderGraph]")
{
    TestGraph g;
    TextureDesc msaaDesc = tex2d();
    msaaDesc.sampleCount = 4;
    ResourceHandle msaa = g.rg->create_transient_texture("msaa", msaaDesc);
    ResourceHandle other = g.transient("other");
    ResourceHandle target = g.transient("target");

    g.rg->add_pass(
        "fwd", PassKind::Graphics,
        [msaa, other, target](PassBuilder& b) {
            b.color(msaa, 0);
            b.color(other, 1); // declared between the color() and its resolve()
            b.resolve(msaa, target);
            b.force_keep();
        },
        [](PassContext&) {});

    PassNode* p = storage(g.rg)->m_passes;
    bool found = false;
    for (uint32_t i = 0; i < p->accessCount; ++i)
        if (p->accesses[i].type == AccessType::ResolveAttachment && p->accesses[i].handle.id == target.id) {
            REQUIRE(p->accesses[i].colorIndex == 0); // msaa's slot, not the adjacent other's
            found = true;
        }
    REQUIRE(found);
}

#ifdef QT_NO_DEBUG
TEST_CASE("PassBuilder::resolve - a src that is not a color() in this pass is rejected", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle notAnAttachment = g.transient("nope");
    ResourceHandle target = g.transient("target");

    g.rg->add_pass(
        "fwd", PassKind::Graphics,
        [notAnAttachment, target](PassBuilder& b) {
            b.resolve(notAnAttachment, target); // src was never declared via color()
            b.force_keep();
        },
        [](PassContext&) {});

    REQUIRE(count_access(storage(g.rg)->m_passes, AccessType::ResolveAttachment) == 0);
}
#endif

#ifdef QT_NO_DEBUG
TEST_CASE("PassBuilder::resolve - two resolve targets for one color slot are rejected", "[RenderGraph]")
{
    TestGraph g;
    TextureDesc msaaDesc = tex2d();
    msaaDesc.sampleCount = 4;
    ResourceHandle msaa = g.rg->create_transient_texture("msaa", msaaDesc);
    ResourceHandle t1 = g.transient("t1");
    ResourceHandle t2 = g.transient("t2");

    g.rg->add_pass(
        "fwd", PassKind::Graphics,
        [msaa, t1, t2](PassBuilder& b) {
            b.color(msaa, 0);
            b.resolve(msaa, t1);
            b.resolve(msaa, t2); // same slot resolved twice -> rejected
            b.force_keep();
        },
        [](PassContext&) {});

    REQUIRE(count_access(storage(g.rg)->m_passes, AccessType::ResolveAttachment) == 1);
}
#endif

// The remaining builder guards, same release-only reasoning as above. Unlike color()/resolve(), these two
// do NOT drop the offending declaration, so the release behavior worth pinning is that the access lands
// anyway; the assert is the whole contract in debug.
#ifdef QT_NO_DEBUG
TEST_CASE("PassBuilder - reading and writing one resource in a pass is rejected", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle tex = g.transient("tex");

    g.rg->add_pass(
        "rw", PassKind::Graphics,
        [tex](PassBuilder& b) {
            b.sampled(tex); // same subresource, read and written in one pass: the graph cannot order these
            b.color(tex, 0);
            b.force_keep();
        },
        [](PassContext&) {});

    // both accesses are recorded: the guard traps, it does not filter
    PassNode* p = storage(g.rg)->m_passes;
    REQUIRE(count_access(p, AccessType::Sampled) == 1);
    REQUIRE(count_access(p, AccessType::ColorAttachment) == 1);
}
#endif

// The Transfer exemption is not release-only: each copy is its own usage scope, so this must stay silent in
// a debug build too, which is exactly what the existing "copy_texture between subresources of one texture"
// test proves by not aborting. This pins the access count that exemption produces.
TEST_CASE("PassBuilder - a Transfer pass may read and write one resource", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle tex = g.transient("tex", /*mipLevelCount*/ 2);

    g.rg->add_pass(
        "produce", PassKind::Graphics, [tex](PassBuilder& b) { b.color(tex, 0); }, [](PassContext&) {});
    g.rg->add_pass(
        "copy", PassKind::Transfer,
        [tex](PassBuilder& b) {
            b.copy_texture(tex, tex, 0, 0, 1, 0); // src and dst are the same handle, no conflict trap
            b.force_keep();
        },
        [](PassContext&) {});

    PassNode* copyPass = storage(g.rg)->m_passes->next;
    REQUIRE(count_access(copyPass, AccessType::CopySrc) == 1);
    REQUIRE(count_access(copyPass, AccessType::CopyDst) == 1);
}

#ifdef QT_NO_DEBUG
TEST_CASE("PassBuilder::depth_stencil - a second depth-stencil attachment is rejected", "[RenderGraph]")
{
    TestGraph g;
    TextureDesc dsDesc = tex2d();
    dsDesc.format = WGPUTextureFormat_Depth32Float;
    ResourceHandle d1 = g.rg->create_transient_texture("d1", dsDesc);
    ResourceHandle d2 = g.rg->create_transient_texture("d2", dsDesc);

    g.rg->add_pass(
        "shadow", PassKind::Graphics,
        [d1, d2](PassBuilder& b) {
            b.depth_stencil(d1);
            b.depth_stencil(d2); // a render pass has one depth-stencil slot
            b.force_keep();
        },
        [](PassContext&) {});

    // no early return, so both land and execute() would silently let the last one win
    REQUIRE(count_access(storage(g.rg)->m_passes, AccessType::DepthStencilAttachment) == 2);
}
#endif

#ifdef QT_NO_DEBUG
TEST_CASE("PassBuilder::initialize - the same target twice in one pass is rejected", "[RenderGraph]")
{
    TestGraph g;
    BufferDesc bd {};
    bd.size = 64;
    ResourceHandle p = g.rg->create_persistent_buffer("pers", bd);

    g.rg->add_pass(
        "bake", PassKind::Compute,
        [p](PassBuilder& b) {
            b.storage_write(p);
            b.initialize(p);
            b.initialize(p); // duplicate: asserted, but not dropped
            b.force_keep();
        },
        [](PassContext&) {});

    REQUIRE(storage(g.rg)->m_passes->initCount == 2);
}
#endif

#ifdef QT_NO_DEBUG
TEST_CASE("PassBuilder::initialize - a target past kMaxInitTargets is dropped", "[RenderGraph]")
{
    TestGraph g;
    BufferDesc bd {};
    bd.size = 64;
    // buffers, not textures: kMaxColorAttachments would cap a multi-target Graphics pass at 8 first
    constexpr uint32_t kOver = Internal::PassNode::kMaxInitTargets + 1;
    std::vector<ResourceHandle> targets;
    std::vector<std::string> names(kOver);
    for (uint32_t i = 0; i < kOver; ++i) {
        names[i] = "pers" + std::to_string(i);
        targets.push_back(g.rg->create_persistent_buffer(names[i], bd));
    }

    g.rg->add_pass(
        "bake", PassKind::Compute,
        [&targets](PassBuilder& b) {
            for (ResourceHandle t : targets) {
                b.storage_write(t);
                b.initialize(t);
            }
            b.force_keep();
        },
        [](PassContext&) {});

    // the 9th target is dropped rather than overrunning the fixed array
    REQUIRE(storage(g.rg)->m_passes->initCount == Internal::PassNode::kMaxInitTargets);
}
#endif

TEST_CASE("RenderGraph - valid graph compiles clean", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle tex = g.transient("tex");

    g.rg->add_pass(
        "write", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.color(tex, 0);
            b.force_keep(); // nothing reads tex; keep the pass past culling
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);
    // keeps the foreign-handle backstop tests honest: they assert byId stays null when compile() bails, so
    // a clean compile has to be the case that populates it
    REQUIRE(storage(g.rg)->byId != nullptr);
}

TEST_CASE("RenderGraph - read before write is reported as an error", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle tex = g.transient("tex");

    // reads a transient no pass ever writes -> compile() must poison the graph
    g.rg->add_pass(
        "read", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.sampled(tex);
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    // nobody declared a write anywhere, which is the first of the three read-before-write diagnoses
    REQUIRE(error_mentions(g.rg, "that no pass ever writes"));
}

// The third read-before-write diagnosis: a writer exists, so it is not the "no pass ever writes" case, but
// the reader is declared first, so no RAW edge forms and the writer is culled instead of being pulled in
// ahead of the reader.
TEST_CASE("RenderGraph - reading before a later-declared writer names the ordering diagnosis", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle tex = g.transient("tex");

    g.rg->add_pass( // declared first, so at sweep time tex has no producer to depend on
        "read", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.sampled(tex);
            b.force_keep();
        },
        [](PassContext&) {});
    g.rg->add_pass(
        "write", PassKind::Graphics, [&](PassBuilder& b) { b.color(tex, 0, WGPULoadOp_Clear, WGPUStoreOp_Store); }, [](PassContext&) {});

    g.rg->compile();
    REQUIRE(error_mentions(g.rg, "before any pass writes it"));
    REQUIRE_FALSE(error_mentions(g.rg, "that no pass ever writes")); // a writer does exist
}


TEST_CASE("RenderGraph - single clear+discard attachment is inferred transient", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle tex = g.transient("tex");

    g.rg->add_pass(
        "write", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.color(tex, 0, WGPULoadOp_Clear, WGPUStoreOp_Discard);
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);
    REQUIRE(storage(g.rg)->transientCount == 1);
}


TEST_CASE("RenderGraph - sampled attachment is not transient", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle tex = g.transient("tex");
    ResourceHandle out = g.transient("out");

    g.rg->add_pass(
        "write", PassKind::Graphics,
        [&](PassBuilder& b) { b.color(tex, 0, WGPULoadOp_Clear, WGPUStoreOp_Store); },
        [](PassContext&) {});
    g.rg->add_pass(
        "read", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.sampled(tex);
            b.color(out, 0);
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);
    REQUIRE(storage(g.rg)->transientCount == 0);
}

TEST_CASE("RenderGraph - consumer runs after producer across a deduped edge", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle a = g.transient("a");
    ResourceHandle b = g.transient("b");
    ResourceHandle out = g.transient("out");

    g.rg->add_pass(
        "producer", PassKind::Graphics,
        [&](PassBuilder& pb) {
            pb.color(a, 0, WGPULoadOp_Clear, WGPUStoreOp_Store);
            pb.color(b, 1, WGPULoadOp_Clear, WGPUStoreOp_Store);
        },
        [](PassContext&) {});
    g.rg->add_pass(
        "consumer", PassKind::Graphics,
        [&](PassBuilder& pb) {
            pb.sampled(a);
            pb.sampled(b);
            pb.color(out, 0);
            pb.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);

    // m_passes is the post-cull execution order after compile(); the producer must precede the consumer.
    int producerPos = -1, consumerPos = -1, pos = 0;
    for (PassNode* p = storage(g.rg)->m_passes; p; p = p->next, ++pos) {
        if (p->id == make_resource_id("producer"))
            producerPos = pos;
        else if (p->id == make_resource_id("consumer"))
            consumerPos = pos;
    }
    REQUIRE(producerPos >= 0);
    REQUIRE(consumerPos >= 0);
    REQUIRE(producerPos < consumerPos);
}

// Performance: record + compile an N-pass linear chain. Pass i samples tex[i-1] and writes tex[i],
// so every edge is a real RAW hazard the compiler must order -> stresses hazard detection,
// topological sort and (with alias on) lifetime packing. Measures the per-frame CPU cost, which is
// what actually runs every frame; execute() needs a GPU device and is out of scope here.
TEST_CASE("RenderGraph - compile perf (linear chain)", "[RenderGraph][!benchmark]")
{
    constexpr int N = 256;

    std::vector<std::string> passNames(N), texNames(N);
    for (int i = 0; i < N; ++i) {
        passNames[i] = "pass" + std::to_string(i);
        texNames[i] = "tex" + std::to_string(i);
    }

    GraphAllocator* allocator = create_allocator();

    auto build_chain = [&](RenderGraph* rg) {
        TextureDesc desc {};
        desc.dimension = WGPUTextureDimension_2D;
        desc.format = WGPUTextureFormat_RGBA8Unorm;
        desc.absolute = { 16, 16, 1 };

        ResourceHandle prev {};
        for (int i = 0; i < N; ++i) {
            ResourceHandle tex = rg->create_transient_texture(texNames[i], desc);
            const bool last = (i == N - 1);
            rg->add_pass(
                passNames[i], PassKind::Graphics,
                [&, i, tex, prev, last](PassBuilder& b) {
                    if (i > 0)
                        b.sampled(prev);
                    b.color(tex, 0, WGPULoadOp_Clear, WGPUStoreOp_Store);
                    if (last)
                        b.force_keep();
                },
                [](PassContext&) {});
            prev = tex;
        }
    };


    {
        begin_frame(allocator);
        RenderGraph* rg = start_recording(allocator);
        build_chain(rg);
        rg->compile();
        REQUIRE(rg->get_errors() == nullptr);
    }

    BENCHMARK("record + compile " + std::to_string(N) + "-pass chain")
    {
        begin_frame(allocator); // per-frame arena reset
        RenderGraph* rg = start_recording(allocator);
        build_chain(rg);
        rg->compile();
        return rg->get_errors();
    };

    destroy_allocator(allocator);
}

TEST_CASE("RenderGraph - copy_buffer compiles clean and records the byte range", "[RenderGraph]")
{
    using webgpu::rg::Internal::AccessType;
    using webgpu::rg::Internal::ResourceAccess;

    TestGraph g;
    ResourceHandle bufA = g.transient_buffer("bufA");
    ResourceHandle bufB = g.transient_buffer("bufB");

    add_buffer_producer(g.rg, bufA);
    g.rg->add_pass(
        "copy", PassKind::Transfer,
        [&](PassBuilder& b) {
            b.copy_buffer(bufA, bufB, 0, 4, 4);
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);

    // second declared pass is the transfer pass; check both recorded accesses
    PassNode* copyPass = storage(g.rg)->m_passes->next;
    REQUIRE(copyPass != nullptr);
    const ResourceAccess* src = nullptr;
    const ResourceAccess* dst = nullptr;
    for (uint32_t i = 0; i < copyPass->accessCount; ++i) {
        const ResourceAccess& a = copyPass->accesses[i];
        if (a.type == AccessType::CopySrc)
            src = &a;
        if (a.type == AccessType::CopyDst)
            dst = &a;
    }
    REQUIRE(src != nullptr);
    REQUIRE(dst != nullptr);
    REQUIRE(src->handle.id == bufA.id);
    REQUIRE(src->bufOffset == 0);
    REQUIRE(src->bufSize == 4);
    REQUIRE(dst->handle.id == bufB.id);
    REQUIRE(dst->bufOffset == 4);
    REQUIRE(dst->bufSize == 4);
}

TEST_CASE("RenderGraph - copy_buffer from an unwritten transient is an error", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle bufA = g.transient_buffer("bufA");
    ResourceHandle bufB = g.transient_buffer("bufB");

    // no producer for bufA -> read before write must poison the graph
    g.rg->add_pass(
        "copy", PassKind::Transfer,
        [&](PassBuilder& b) {
            b.copy_buffer(bufA, bufB);
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(error_mentions(g.rg, "that no pass ever writes"));
}

TEST_CASE("RenderGraph - partial copy_buffer dst is not a first-define", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle bufA = g.transient_buffer("bufA");
    ResourceHandle bufB = g.transient_buffer("bufB");

    add_buffer_producer(g.rg, bufA);
    g.rg->add_pass(
        "copy", PassKind::Transfer,
        [&](PassBuilder& b) {
            b.copy_buffer(bufA, bufB, 0, /*dstOffset*/ 4, /*size*/ 4);
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);
    // bytes outside [4,8) are untouched, so bufB must not be eligible as a full first-define for aliasing
    REQUIRE(webgpu::rg::Internal::find_node(g.rg, bufB)->firstDefines == false);
}

TEST_CASE("RenderGraph - copy_texture compiles clean and records the subresources", "[RenderGraph]")
{
    using webgpu::rg::Internal::AccessType;
    using webgpu::rg::Internal::ResourceAccess;

    TestGraph g;
    ResourceHandle texA = g.transient("texA");
    ResourceHandle texB = g.transient("texB", /*mipLevelCount*/ 2);

    g.rg->add_pass(
        "produce", PassKind::Graphics, [&](PassBuilder& b) { b.color(texA, 0); }, [](PassContext&) {});
    g.rg->add_pass(
        "copy", PassKind::Transfer,
        [&](PassBuilder& b) {
            b.copy_texture(texA, texB, /*srcMip*/ 0, /*srcLayer*/ 0, /*dstMip*/ 1, /*dstLayer*/ 0);
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);

    PassNode* copyPass = storage(g.rg)->m_passes->next;
    REQUIRE(copyPass != nullptr);
    const ResourceAccess* src = nullptr;
    const ResourceAccess* dst = nullptr;
    for (uint32_t i = 0; i < copyPass->accessCount; ++i) {
        const ResourceAccess& a = copyPass->accesses[i];
        if (a.type == AccessType::CopySrc)
            src = &a;
        if (a.type == AccessType::CopyDst)
            dst = &a;
    }
    REQUIRE(src != nullptr);
    REQUIRE(dst != nullptr);
    REQUIRE(src->handle.id == texA.id);
    REQUIRE(src->baseMip == 0);
    REQUIRE(dst->handle.id == texB.id);
    REQUIRE(dst->baseMip == 1);
}

TEST_CASE("RenderGraph - copy_texture between subresources of one texture is legal", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle tex = g.transient("tex", /*mipLevelCount*/ 2);

    g.rg->add_pass(
        "produce", PassKind::Graphics, [&](PassBuilder& b) { b.color(tex, 0); }, [](PassContext&) {});
    g.rg->add_pass(
        "copy", PassKind::Transfer,
        [&](PassBuilder& b) {
            b.copy_texture(tex, tex, /*srcMip*/ 0, /*srcLayer*/ 0, /*dstMip*/ 1, /*dstLayer*/ 0);
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);
}

TEST_CASE("RenderGraph - copy_texture_to_buffer dst is never a first-define", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle tex = g.transient("tex");
    ResourceHandle buf = g.transient_buffer("buf");

    g.rg->add_pass(
        "produce", PassKind::Graphics, [&](PassBuilder& b) { b.color(tex, 0); }, [](PassContext&) {});
    g.rg->add_pass(
        "readback", PassKind::Transfer,
        [&](PassBuilder& b) {
            b.copy_texture_to_buffer(tex, buf);
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);
    // coverage depends on the body's WGPUTexelCopyBufferLayout, so the aliaser must never treat the
    // dst as fully defined by this copy
    REQUIRE(webgpu::rg::Internal::find_node(g.rg, buf)->firstDefines == false);
}

TEST_CASE("RenderGraph - copy_buffer_to_texture write satisfies a later read", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle buf = g.transient_buffer("buf");
    ResourceHandle tex = g.transient("tex");
    ResourceHandle out = g.transient("out");

    g.rg->add_pass(
        "upload", PassKind::Transfer, [&](PassBuilder& b) { b.host_write(buf); }, [](PassContext&) {});
    g.rg->add_pass(
        "stage", PassKind::Transfer, [&](PassBuilder& b) { b.copy_buffer_to_texture(buf, tex); }, [](PassContext&) {});
    // the copy's CopyDst is tex's writer; sampling it afterwards must not be a read-before-write error
    g.rg->add_pass(
        "consume", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.sampled(tex);
            b.color(out, 0);
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);
}

TEST_CASE("RenderGraph - explicit full-size copy_buffer dst is a first-define", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle bufA = g.transient_buffer("bufA", 64);
    ResourceHandle bufB = g.transient_buffer("bufB", 64);

    add_buffer_producer(g.rg, bufA);
    g.rg->add_pass(
        "copy", PassKind::Transfer,
        [&](PassBuilder& b) {
            b.copy_buffer(bufA, bufB, 0, 0, /*size*/ 64); // explicit size spanning the whole dst
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);
    REQUIRE(webgpu::rg::Internal::find_node(g.rg, bufB)->firstDefines == true);
}

TEST_CASE("RenderGraph - whole copy_buffer dst is a first-define", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle bufA = g.transient_buffer("bufA");
    ResourceHandle bufB = g.transient_buffer("bufB");

    add_buffer_producer(g.rg, bufA);
    g.rg->add_pass(
        "copy", PassKind::Transfer,
        [&](PassBuilder& b) {
            b.copy_buffer(bufA, bufB); // offsets 0, size 0 = whole buffer
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);
    REQUIRE(webgpu::rg::Internal::find_node(g.rg, bufB)->firstDefines == true);
}

// A pass writing a transient nobody reads (not imported/persistent, no force_keep) is dead -> culled.
TEST_CASE("RenderGraph - dead pass with no reader is culled", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle sink = import_tex(g.rg, "sink");
    ResourceHandle mid = g.transient("mid");
    ResourceHandle orphan = g.transient("orphan");

    g.rg->add_pass(
        "writer", PassKind::Graphics, [&](PassBuilder& b) { b.color(mid, 0, WGPULoadOp_Clear, WGPUStoreOp_Store); }, [](PassContext&) {});
    g.rg->add_pass( // writes orphan, read by nobody -> dead
        "dead", PassKind::Graphics, [&](PassBuilder& b) { b.color(orphan, 0, WGPULoadOp_Clear, WGPUStoreOp_Store); }, [](PassContext&) {});
    g.rg->add_pass(
        "reader", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.sampled(mid);
            b.color(sink, 0, WGPULoadOp_Load, WGPUStoreOp_Store);
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);
    auto order = pass_order(g.rg);
    REQUIRE(order.size() == 2);
    REQUIRE(idx_of(order, "dead") < 0); // dropped
    REQUIRE(idx_of(order, "writer") >= 0);
    REQUIRE(idx_of(order, "reader") >= 0);
}

// force_keep() rescues a reader-less, sink-less side-effect pass that would otherwise be culled.
TEST_CASE("RenderGraph - force_keep rescues an otherwise-dead pass", "[RenderGraph]")
{
    { // baseline: no force_keep -> the pass is dead and culled to nothing
        TestGraph g;
        ResourceHandle buf = g.transient_buffer("scratch", 256);
        g.rg->add_pass(
            "effect", PassKind::Compute, [&](PassBuilder& b) { b.storage_write(buf); }, [](PassContext&) {});
        g.rg->compile();
        REQUIRE(g.rg->get_errors() == nullptr);
        REQUIRE(pass_order(g.rg).empty());
    }
    { // same pass + force_keep() -> survives
        TestGraph g;
        ResourceHandle buf = g.transient_buffer("scratch", 256);
        g.rg->add_pass(
            "effect", PassKind::Compute,
            [&](PassBuilder& b) {
                b.storage_write(buf);
                b.force_keep();
            },
            [](PassContext&) {});
        g.rg->compile();
        REQUIRE(g.rg->get_errors() == nullptr);
        REQUIRE(pass_order(g.rg).size() == 1);
    }
}

// Two passes write the same resource before it is read. The write-after-write hazard is an ordering
// edge, so the earlier writer survives cull and is scheduled first: w1 -> w2 -> reader.
TEST_CASE("RenderGraph - write-after-write orders both writers before the reader", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle sink = import_tex(g.rg, "sink");
    ResourceHandle x = g.transient("x");

    g.rg->add_pass(
        "w1", PassKind::Graphics, [&](PassBuilder& b) { b.color(x, 0, WGPULoadOp_Clear, WGPUStoreOp_Store); }, [](PassContext&) {});
    g.rg->add_pass(
        "w2", PassKind::Graphics, [&](PassBuilder& b) { b.color(x, 0, WGPULoadOp_Clear, WGPUStoreOp_Store); }, [](PassContext&) {});
    g.rg->add_pass(
        "reader", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.sampled(x);
            b.color(sink, 0, WGPULoadOp_Load, WGPUStoreOp_Store);
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);
    auto order = pass_order(g.rg);
    REQUIRE(order.size() == 3);
    REQUIRE(idx_of(order, "w1") < idx_of(order, "w2"));
    REQUIRE(idx_of(order, "w2") < idx_of(order, "reader"));
}

// Reading an IMPORTED resource with no in-graph writer is legal (its value comes from outside the
// frame), unlike a transient read-before-write, which poisons the graph.
TEST_CASE("RenderGraph - reading an imported resource with no writer is legal", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle src = import_tex(g.rg, "src"); // imported, never written by any pass
    ResourceHandle sink = import_tex(g.rg, "sink");

    g.rg->add_pass(
        "blit", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.sampled(src);
            b.color(sink, 0, WGPULoadOp_Load, WGPUStoreOp_Store);
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);
    REQUIRE(pass_order(g.rg).size() == 1);
}

// A history .curr producer is a cull root: the pass survives with no same-frame reader of .curr and
// no imported sink. Reads .prev + writes .curr in one pass (the ping-pong pattern).
TEST_CASE("RenderGraph - history .curr write is a cull sink", "[RenderGraph]")
{
    TestGraph g;
    HistoryResource hist = g.rg->create_history_texture("hist", tex2d());

    g.rg->add_pass(
        "temporal", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.sampled(hist.prev);
            b.color(hist.curr, 0, WGPULoadOp_Clear, WGPUStoreOp_Store);
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);
    auto order = pass_order(g.rg);
    REQUIRE(order.size() == 1);
    REQUIRE(idx_of(order, "temporal") >= 0);
}

// A plain persistent write is NOT a sink (pool-cached, dependency-driven), so a bake with no reader
// is culled. The contrast to imported/history, which are sinks.
TEST_CASE("RenderGraph - persistent write with no reader is culled", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle pers = g.rg->create_persistent_texture("pers", tex2d());

    g.rg->add_pass(
        "bake", PassKind::Graphics, [&](PassBuilder& b) { b.color(pers, 0, WGPULoadOp_Clear, WGPUStoreOp_Store); }, [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);
    REQUIRE(pass_order(g.rg).empty());
}

// Two transients with disjoint lifetimes ([a,b] then [c,d]) and identical signatures pack onto one
// physical slot when aliasing is on (compile() default).
TEST_CASE("RenderGraph - disjoint transients share one physical slot", "[RenderGraph]")
{
    using webgpu::rg::Internal::find_node;
    using webgpu::rg::Internal::ResourceNode;

    TestGraph g;
    ResourceHandle sink1 = import_tex(g.rg, "sink1");
    ResourceHandle sink2 = import_tex(g.rg, "sink2");
    ResourceHandle t1 = g.transient("t1");
    ResourceHandle t2 = g.transient("t2");

    g.rg->add_pass(
        "a", PassKind::Graphics, [&](PassBuilder& b) { b.color(t1, 0, WGPULoadOp_Clear, WGPUStoreOp_Store); }, [](PassContext&) {});
    g.rg->add_pass(
        "b", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.sampled(t1);
            b.color(sink1, 0, WGPULoadOp_Load, WGPUStoreOp_Store);
        },
        [](PassContext&) {});
    g.rg->add_pass( // t1 is dead by here, so t2 can reuse its storage
        "c", PassKind::Graphics, [&](PassBuilder& b) { b.color(t2, 0, WGPULoadOp_Clear, WGPUStoreOp_Store); }, [](PassContext&) {});
    g.rg->add_pass(
        "d", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.sampled(t2);
            b.color(sink2, 0, WGPULoadOp_Load, WGPUStoreOp_Store);
        },
        [](PassContext&) {});

    g.rg->compile(true); // enableAlias
    REQUIRE(g.rg->get_errors() == nullptr);
    REQUIRE(pass_order(g.rg).size() == 4);
    REQUIRE(storage(g.rg)->m_slotCount == 1);
    ResourceNode* r1 = find_node(g.rg, t1);
    ResourceNode* r2 = find_node(g.rg, t2);
    REQUIRE(r1->aliasSlot != ResourceNode::kNoSlot);
    REQUIRE(r1->aliasSlot == r2->aliasSlot);
}

// Aliasing off (compile(false)): no slots are built and every transient keeps its own object.
TEST_CASE("RenderGraph - aliasing disabled builds no slots", "[RenderGraph]")
{
    using webgpu::rg::Internal::find_node;
    using webgpu::rg::Internal::ResourceNode;

    TestGraph g;
    ResourceHandle sink1 = import_tex(g.rg, "sink1");
    ResourceHandle sink2 = import_tex(g.rg, "sink2");
    ResourceHandle t1 = g.transient("t1");
    ResourceHandle t2 = g.transient("t2");

    g.rg->add_pass(
        "a", PassKind::Graphics, [&](PassBuilder& b) { b.color(t1, 0, WGPULoadOp_Clear, WGPUStoreOp_Store); }, [](PassContext&) {});
    g.rg->add_pass(
        "b", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.sampled(t1);
            b.color(sink1, 0, WGPULoadOp_Load, WGPUStoreOp_Store);
        },
        [](PassContext&) {});
    g.rg->add_pass(
        "c", PassKind::Graphics, [&](PassBuilder& b) { b.color(t2, 0, WGPULoadOp_Clear, WGPUStoreOp_Store); }, [](PassContext&) {});
    g.rg->add_pass(
        "d", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.sampled(t2);
            b.color(sink2, 0, WGPULoadOp_Load, WGPUStoreOp_Store);
        },
        [](PassContext&) {});

    g.rg->compile(false); // aliasing OFF
    REQUIRE(g.rg->get_errors() == nullptr);
    REQUIRE(storage(g.rg)->m_slotCount == 0);
    REQUIRE(find_node(g.rg, t1)->aliasSlot == ResourceNode::kNoSlot);
    REQUIRE(find_node(g.rg, t2)->aliasSlot == ResourceNode::kNoSlot);
}

// Overlapping lifetimes must NOT share: both t1 and t2 are alive at the consumer, so they get
// distinct slots. Proves the packer respects lifetime, not just signature.
TEST_CASE("RenderGraph - overlapping-lifetime transients do not share a slot", "[RenderGraph]")
{
    using webgpu::rg::Internal::find_node;
    using webgpu::rg::Internal::ResourceNode;

    TestGraph g;
    ResourceHandle sink = import_tex(g.rg, "sink");
    ResourceHandle t1 = g.transient("t1");
    ResourceHandle t2 = g.transient("t2");

    g.rg->add_pass(
        "a", PassKind::Graphics, [&](PassBuilder& b) { b.color(t1, 0, WGPULoadOp_Clear, WGPUStoreOp_Store); }, [](PassContext&) {});
    g.rg->add_pass(
        "b", PassKind::Graphics, [&](PassBuilder& b) { b.color(t2, 0, WGPULoadOp_Clear, WGPUStoreOp_Store); }, [](PassContext&) {});
    g.rg->add_pass( // reads both -> t1 and t2 alive together here
        "c", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.sampled(t1);
            b.sampled(t2);
            b.color(sink, 0, WGPULoadOp_Load, WGPUStoreOp_Store);
        },
        [](PassContext&) {});

    g.rg->compile(true);
    REQUIRE(g.rg->get_errors() == nullptr);
    REQUIRE(storage(g.rg)->m_slotCount == 2);
    ResourceNode* r1 = find_node(g.rg, t1);
    ResourceNode* r2 = find_node(g.rg, t2);
    REQUIRE(r1->aliasSlot != ResourceNode::kNoSlot);
    REQUIRE(r2->aliasSlot != ResourceNode::kNoSlot);
    REQUIRE(r1->aliasSlot != r2->aliasSlot);
}

// Greedy reuse across more than two: three transients with back-to-back disjoint lifetimes all
// collapse onto one slot (t2 reuses t1's, t3 reuses that). force_keep gives each a one-pass life.
TEST_CASE("RenderGraph - three disjoint transients collapse onto one slot", "[RenderGraph]")
{
    using webgpu::rg::Internal::find_node;
    using webgpu::rg::Internal::ResourceNode;

    TestGraph g;
    ResourceHandle t1 = g.transient("t1");
    ResourceHandle t2 = g.transient("t2");
    ResourceHandle t3 = g.transient("t3");

    g.rg->add_pass(
        "k1", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.color(t1, 0, WGPULoadOp_Clear, WGPUStoreOp_Store);
            b.force_keep();
        },
        [](PassContext&) {});
    g.rg->add_pass(
        "k2", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.color(t2, 0, WGPULoadOp_Clear, WGPUStoreOp_Store);
            b.force_keep();
        },
        [](PassContext&) {});
    g.rg->add_pass(
        "k3", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.color(t3, 0, WGPULoadOp_Clear, WGPUStoreOp_Store);
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile(true);
    REQUIRE(g.rg->get_errors() == nullptr);
    REQUIRE(pass_order(g.rg).size() == 3);
    REQUIRE(storage(g.rg)->m_slotCount == 1); // one physical slot for three transients
    ResourceNode* r1 = find_node(g.rg, t1);
    ResourceNode* r2 = find_node(g.rg, t2);
    ResourceNode* r3 = find_node(g.rg, t3);
    REQUIRE(r1->aliasSlot != ResourceNode::kNoSlot);
    REQUIRE(r1->aliasSlot == r2->aliasSlot);
    REQUIRE(r2->aliasSlot == r3->aliasSlot);
}

// ---------------------------------------------------------------------------------------------------

// "MSAA & resolve": sampleCount 4 color + single-sample resolve target; resolve()
// is a write, so a later sampled(resolved) is ordered after and is not read-before-write.
TEST_CASE("RenderGraph - msaa resolve is a write that satisfies a later read", "[RenderGraph]")
{
    using webgpu::rg::Internal::AccessType;
    using webgpu::rg::Internal::find_node;

    TestGraph g;
    TextureDesc msaaDesc {};
    msaaDesc.dimension = WGPUTextureDimension_2D;
    msaaDesc.format = WGPUTextureFormat_RGBA8Unorm;
    msaaDesc.absolute = { 16, 16, 1 };
    msaaDesc.sampleCount = 4;
    ResourceHandle msaaColor = g.rg->create_transient_texture("msaa.color", msaaDesc);
    ResourceHandle resolved = g.transient("resolved"); // single-sample, same format+size
    ResourceHandle out = g.transient("out");

    g.rg->add_pass(
        "forward.msaa", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.color(msaaColor, 0);
            b.resolve(msaaColor, resolved);
        },
        [](PassContext&) {});
    g.rg->add_pass(
        "present", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.sampled(resolved);
            b.color(out, 0);
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);
    REQUIRE(find_node(g.rg, msaaColor)->sampleCount == 4);

    // the resolve access is recorded as a ResolveAttachment write on `resolved`
    PassNode* msaaPass = storage(g.rg)->m_passes;
    bool foundResolve = false;
    for (uint32_t i = 0; i < msaaPass->accessCount; ++i)
        if (msaaPass->accesses[i].type == AccessType::ResolveAttachment && msaaPass->accesses[i].handle.id == resolved.id)
            foundResolve = true;
    REQUIRE(foundResolve);

    // present ordered after the resolve
    int msaaPos = -1, presentPos = -1, pos = 0;
    for (PassNode* p = storage(g.rg)->m_passes; p; p = p->next, ++pos) {
        if (p->id == make_resource_id("forward.msaa"))
            msaaPos = pos;
        else if (p->id == make_resource_id("present"))
            presentPos = pos;
    }
    REQUIRE(msaaPos >= 0);
    REQUIRE(presentPos >= 0);
    REQUIRE(msaaPos < presentPos);
}

// "Stencil": the stencil load/store/clear passed to depth_stencil() is recorded on
// the access, and a depth_stencil_read_only() reader orders after the mask writer.
TEST_CASE("RenderGraph - stencil mask write then read-only test pass", "[RenderGraph]")
{
    using webgpu::rg::Internal::AccessType;

    TestGraph g;
    TextureDesc dsDesc {};
    dsDesc.dimension = WGPUTextureDimension_2D;
    dsDesc.format = WGPUTextureFormat_Depth24PlusStencil8;
    dsDesc.absolute = { 16, 16, 1 };
    ResourceHandle ds = g.rg->create_transient_texture("mask.ds", dsDesc);
    ResourceHandle out = g.transient("out");

    g.rg->add_pass(
        "stencil.mask", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.depth_stencil(ds, WGPULoadOp_Clear, WGPUStoreOp_Store, 1.0f, 0, 0, WGPULoadOp_Clear, WGPUStoreOp_Store, /*stencilClear*/ 7);
        },
        [](PassContext&) {});
    g.rg->add_pass(
        "stencil.effect", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.depth_stencil_read_only(ds);
            b.color(out, 0);
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);

    PassNode* maskPass = storage(g.rg)->m_passes;
    const webgpu::rg::Internal::ResourceAccess* dsWrite = nullptr;
    for (uint32_t i = 0; i < maskPass->accessCount; ++i)
        if (maskPass->accesses[i].type == AccessType::DepthStencilAttachment)
            dsWrite = &maskPass->accesses[i];
    REQUIRE(dsWrite != nullptr);
    REQUIRE(dsWrite->stencilLoadOp == WGPULoadOp_Clear);
    REQUIRE(dsWrite->stencilStoreOp == WGPUStoreOp_Store);
    REQUIRE(dsWrite->stencilClear == 7);

    // effect pass reads via DepthStencilReadOnly and runs after the mask write
    int maskPos = -1, effectPos = -1, pos = 0;
    bool foundReadOnly = false;
    for (PassNode* p = storage(g.rg)->m_passes; p; p = p->next, ++pos) {
        if (p->id == make_resource_id("stencil.mask"))
            maskPos = pos;
        if (p->id == make_resource_id("stencil.effect")) {
            effectPos = pos;
            for (uint32_t i = 0; i < p->accessCount; ++i)
                if (p->accesses[i].type == AccessType::DepthStencilReadOnly)
                    foundReadOnly = true;
        }
    }
    REQUIRE(foundReadOnly);
    REQUIRE(maskPos >= 0);
    REQUIRE(effectPos >= 0);
    REQUIRE(maskPos < effectPos);
}

// "ViewRange": dim/mipCount/layerCount/aspect and baseMip/baseLayer declared on a
// sampled() read are recorded on the access, so ctx.view()/ctx.bind() hand back that exact shape.
TEST_CASE("RenderGraph - sampled ViewRange is recorded on the access", "[RenderGraph]")
{
    using webgpu::rg::Internal::AccessType;

    TestGraph g;
    TextureDesc desc {};
    desc.dimension = WGPUTextureDimension_2D;
    desc.format = WGPUTextureFormat_RGBA8Unorm;
    desc.absolute = { 16, 16, 6 };
    desc.mipLevelCount = 4;
    ResourceHandle env = g.rg->create_transient_texture("env", desc);
    ResourceHandle out = g.transient("out");

    g.rg->add_pass(
        "produce", PassKind::Compute, [&](PassBuilder& b) { b.storage_write(env); }, [](PassContext&) {});
    g.rg->add_pass(
        "consume", PassKind::Graphics,
        [&](PassBuilder& b) {
            // cube view over all 6 layers, mips 1..2, from rendergraph.md Views
            b.sampled(env, /*baseMip*/ 1, /*baseLayer*/ 0, { .dim = WGPUTextureViewDimension_Cube, .mipCount = 2, .layerCount = 6 });
            b.color(out, 0);
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);

    PassNode* consume = storage(g.rg)->m_passes->next;
    REQUIRE(consume != nullptr);
    const webgpu::rg::Internal::ResourceAccess* read = nullptr;
    for (uint32_t i = 0; i < consume->accessCount; ++i)
        if (consume->accesses[i].type == AccessType::Sampled)
            read = &consume->accesses[i];
    REQUIRE(read != nullptr);
    REQUIRE(read->viewDim == WGPUTextureViewDimension_Cube);
    REQUIRE(read->baseMip == 1);
    REQUIRE(read->mipCount == 2);
    REQUIRE(read->baseLayer == 0);
    REQUIRE(read->layerCount == 6);
}

// "Recipe: generate a mip chain": per level, sample mip i-1 and render into mip i of the
// SAME handle. Whole-resource hazards serialize the chain in level order.
TEST_CASE("RenderGraph - mip chain blit on one handle compiles and serializes", "[RenderGraph]")
{
    constexpr uint32_t kMips = 4;
    TestGraph g;
    ResourceHandle tex = g.transient("chain", kMips);

    g.rg->add_pass(
        "mip0", PassKind::Graphics, [&](PassBuilder& b) { b.color(tex, 0, WGPULoadOp_Clear, WGPUStoreOp_Store, {}, /*baseMip*/ 0); },
        [](PassContext&) {});

    std::string names[kMips];
    for (uint32_t mip = 1; mip < kMips; ++mip) {
        names[mip] = "mip" + std::to_string(mip);
        const bool last = (mip == kMips - 1);
        g.rg->add_pass(
            names[mip], PassKind::Graphics,
            [&](PassBuilder& b) {
                b.sampled(tex, mip - 1);
                b.color(tex, 0, WGPULoadOp_Clear, WGPUStoreOp_Store, {}, mip);
                if (last)
                    b.force_keep();
            },
            [](PassContext&) {});
    }

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);

    // execution order preserves the level order: mip0, mip1, mip2, mip3
    uint32_t pos = 0;
    for (PassNode* p = storage(g.rg)->m_passes; p; p = p->next, ++pos) {
        REQUIRE(pos < kMips);
        std::string expected = "mip" + std::to_string(pos);
        REQUIRE(std::string_view(p->id.name.data, p->id.name.length) == expected);
    }
    REQUIRE(pos == kMips); // all passes survived culling
}

// ---------------------------------------------------------------------------------------------------
// TransientResourcePool supersede-eviction (device-free). destroy() null-guards every GPU release, so
// entries hand-populated with null tex/view/buf drive end_frame() directly, and superseded_by is pure.

using webgpu::rg::Internal::Arena;
using webgpu::rg::Internal::StringInterner;
using webgpu::rg::Internal::SubviewPool;
using webgpu::rg::Internal::TexSignature;
using webgpu::rg::Internal::TransientResourcePool;

// ---------------------------------------------------------------------------------------------------
// StringInterner. A stack arena stands in for the allocator's `persist`; nothing here needs a device.

struct InternerFixture {
    Arena arena {};
    StringInterner interner {};

    InternerFixture() { interner.arena = &arena; }
    ~InternerFixture() { arena.free_all(); }
};

TEST_CASE("StringInterner - equal strings share one canonical copy", "[RenderGraph]")
{
    InternerFixture f;

    // distinct std::string objects, so a pointer match cannot come from the caller's own storage
    std::string a = "gbuffer.albedo";
    std::string b = "gbuffer.albedo";
    REQUIRE(a.data() != b.data());

    ResourceId ia = f.interner.intern(a);
    ResourceId ib = f.interner.intern(b);

    REQUIRE(ia.name.data == ib.name.data); // one canonical copy
    REQUIRE(ia.value == ib.value);
    REQUIRE(ia.value == webgpu::rg::fnv1a("gbuffer.albedo")); // the id hash is still fnv1a of the name
    REQUIRE(std::string_view(ia.name.data, ia.name.length) == "gbuffer.albedo");
}

TEST_CASE("StringInterner - distinct strings get distinct canonical copies", "[RenderGraph]")
{
    InternerFixture f;
    ResourceId a = f.interner.intern("gbuffer.albedo");
    ResourceId b = f.interner.intern("gbuffer.normal");

    REQUIRE(a.name.data != b.name.data);
    REQUIRE(a.value != b.value);
}

// The pre-fix pool compared names under a 63-char truncation, so two names agreeing on their first 63
// chars were equal to it. The interner stores each in full and hands back distinct canonical pointers.
TEST_CASE("StringInterner - names differing only past 63 chars stay distinct", "[RenderGraph]")
{
    InternerFixture f;
    const std::string prefix(63, 'x');
    ResourceId a = f.interner.intern(prefix + "alpha");
    ResourceId b = f.interner.intern(prefix + "beta");

    REQUIRE(a.name.data != b.name.data);
    REQUIRE(a.value != b.value);
    REQUIRE(a.name.length == 68);
    REQUIRE(std::string_view(a.name.data, a.name.length) == prefix + "alpha"); // stored in full, not truncated
    REQUIRE(std::string_view(b.name.data, b.name.length) == prefix + "beta");
}

TEST_CASE("StringInterner - a name longer than the old inline caps is stored in full", "[RenderGraph]")
{
    InternerFixture f;
    const std::string longName(200, 'q'); // past both the old 64-char pool cap and 48-char profiler cap
    ResourceId id = f.interner.intern(longName);

    REQUIRE(id.name.length == 200);
    REQUIRE(std::string_view(id.name.data, id.name.length) == longName);
    REQUIRE(id.name.data[200] == '\0'); // NUL-terminated, so it is usable as a WebGPU label
}

// The interner never rehashes, which is what makes a handed-out canonical pointer safe to keep. Fill
// well past kBuckets so every bucket chains, then re-check the first pointer still resolves.
TEST_CASE("StringInterner - canonical pointers stay stable across many interns", "[RenderGraph]")
{
    InternerFixture f;
    ResourceId first = f.interner.intern("first.name");
    const char* firstData = first.name.data;

    std::vector<ResourceId> ids;
    for (int i = 0; i < StringInterner::kBuckets * 8; ++i)
        ids.push_back(f.interner.intern("filler." + std::to_string(i)));

    REQUIRE(f.interner.intern("first.name").name.data == firstData); // same node, no relocation
    REQUIRE(std::string_view(firstData, first.name.length) == "first.name"); // bytes still intact

    // every filler is still its own canonical copy and re-interns to the same pointer
    for (size_t i = 0; i < ids.size(); ++i) {
        ResourceId again = f.interner.intern("filler." + std::to_string(i));
        REQUIRE(again.name.data == ids[i].name.data);
    }
}

// Live-cell count for either pool's intrusive entry list (both link cells through Entry::next).
template <class E> static uint32_t list_count(const E* head)
{
    uint32_t n = 0;
    for (const E* e = head; e; e = e->next)
        ++n;
    return n;
}

// A pool wired to a stack arena, the minimum end_frame()/superseded_by need without a device: entry cells
// come from `arena`, and destroy() dereferences `subviewPool` to recycle the (always null here) subview list.
struct PoolFixture {
    Arena arena {};
    SubviewPool subviewPool {};
    TransientResourcePool pool {};

    PoolFixture()
    {
        subviewPool.arena = &arena;
        pool.arena = &arena;
        pool.subviewPool = &subviewPool;
    }
    // Drain the pool BEFORE freeing the arena its Entry cells live in. A destructor body runs before its
    // members are destroyed, so a bare arena.free_all() here would leave ~TransientResourcePool() walking
    // `entries` through freed memory. destroy_all() nulls both heads, so the member destructor's own call
    // is then a no-op. (Arena has no destructor; free_all is manual.)
    ~PoolFixture()
    {
        pool.destroy_all();
        arena.free_all();
    }

    // Append `proto` to the live list. Appends rather than prepends so list order matches declaration
    // order, which the "keeps only the newest generation" test reads back positionally.
    TransientResourcePool::Entry* push(const TransientResourcePool::Entry& proto)
    {
        TransientResourcePool::Entry* e = pool.alloc_entry();
        TransientResourcePool::Entry* next = e->next;
        *e = proto;
        e->next = next;
        TransientResourcePool::Entry** pp = &pool.entries;
        while (*pp)
            pp = &(*pp)->next;
        *pp = e;
        e->next = nullptr;
        return e;
    }

    uint32_t count() const { return list_count(pool.entries); }

    // nth live entry, for the positional checks the old vector-backed tests did with entries[i]
    const TransientResourcePool::Entry* at(uint32_t i) const
    {
        const TransientResourcePool::Entry* e = pool.entries;
        for (uint32_t k = 0; e && k < i; ++k)
            e = e->next;
        return e;
    }
};

// Stand-ins for the StringInterner canonical name pointers the real acquire path passes as an entry's
// supersede identity. Only the address matters: superseded_by compares pointers, never bytes.
static const char kIdA[] = "resourceA";
static const char kIdB[] = "resourceB";

// a texture Entry with the given size, created in `createdFrame`, last used in `lastUsedFrame`. All GPU
// handles left null so destroy() is a no-op. `identity` defaults to kIdA, so a test that does not care
// about identity gets two entries of the SAME logical resource, i.e. the resize case.
static TransientResourcePool::Entry tex_entry(
    WGPUExtent3D size, uint64_t createdFrame, uint64_t lastUsedFrame, const void* identity = kIdA)
{
    TransientResourcePool::Entry e {};
    e.kind = ResourceKind::Texture;
    e.sig.size = size;
    e.sig.format = WGPUTextureFormat_RGBA8Unorm;
    e.sig.dim = WGPUTextureDimension_2D;
    e.sig.mipLevelCount = 1;
    e.sig.sampleCount = 1;
    e.usage = (WGPUTextureUsage)(WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding);
    e.createdFrame = createdFrame;
    e.lastUsedFrame = lastUsedFrame;
    e.identity = identity;
    return e;
}

TEST_CASE("TransientResourcePool - old-size idle texture is superseded by a new-size sibling", "[RenderGraph]")
{
    PoolFixture f;
    f.pool.frame = 10;
    f.pool.createdThisFrame = 1;
    f.push(tex_entry({ 800, 600, 1 }, /*created*/ 3, /*lastUsed*/ 9)); // old size, idle this frame
    f.push(tex_entry({ 1024, 768, 1 }, /*created*/ 10, /*lastUsed*/ 10)); // new size, created this frame

    f.pool.end_frame();

    // the old-size entry is evicted at end_frame, before kRetain (frame - 9 == 1 < 4) would fire.
    REQUIRE(f.count() == 1);
    REQUIRE(f.at(0)->sig.size.width == 1024);
}

TEST_CASE("TransientResourcePool - texture claimed this frame is not superseded", "[RenderGraph]")
{
    PoolFixture f;
    f.pool.frame = 10;
    f.pool.createdThisFrame = 1;
    f.push(tex_entry({ 800, 600, 1 }, /*created*/ 3, /*lastUsed*/ 10)); // used THIS frame -> live
    f.push(tex_entry({ 1024, 768, 1 }, /*created*/ 10, /*lastUsed*/ 10));

    f.pool.end_frame();

    REQUIRE(f.count() == 2); // both kept
}

TEST_CASE("TransientResourcePool - descriptor mismatch does not supersede", "[RenderGraph]")
{
    auto survives = [](auto mutate) {
        PoolFixture f;
        f.pool.frame = 10;
        f.pool.createdThisFrame = 1;
        f.push(tex_entry({ 800, 600, 1 }, /*created*/ 3, /*lastUsed*/ 9)); // old, idle
        TransientResourcePool::Entry sibling = tex_entry({ 1024, 768, 1 }, /*created*/ 10, /*lastUsed*/ 10);
        mutate(sibling);
        f.push(sibling);
        f.pool.end_frame();
        // old entry not superseded -> both survive (frame - 9 == 1 < kRetain, so kRetain keeps it too)
        return f.count() == 2;
    };

    REQUIRE(survives([](TransientResourcePool::Entry& s) { s.sig.format = WGPUTextureFormat_BGRA8Unorm; }));
    REQUIRE(survives([](TransientResourcePool::Entry& s) { s.sig.dim = WGPUTextureDimension_3D; }));
    REQUIRE(survives([](TransientResourcePool::Entry& s) { s.sig.mipLevelCount = 4; }));
    REQUIRE(survives([](TransientResourcePool::Entry& s) { s.sig.sampleCount = 4; }));
    // a differently-purposed sibling must not sweep the old entry: STORAGE does not cover SAMPLED+ATTACHMENT
    REQUIRE(survives([](TransientResourcePool::Entry& s) { s.usage = WGPUTextureUsage_StorageBinding; }));
}

TEST_CASE("TransientResourcePool - sibling created in an earlier frame does not supersede", "[RenderGraph]")
{
    PoolFixture f;
    f.pool.frame = 10;
    f.pool.createdThisFrame = 1;
    f.push(tex_entry({ 800, 600, 1 }, /*created*/ 3, /*lastUsed*/ 9)); // old, idle
    f.push(tex_entry({ 1024, 768, 1 }, /*created*/ 8, /*lastUsed*/ 9)); // different size but NOT this frame

    f.pool.end_frame();

    REQUIRE(f.count() == 2); // no supersede; kRetain keeps both (idle only 1 frame)
}

// A resize that also flips a usage bit is still a resize. Usage matches by the same superset rule acquire
// reuses by, so the added STORAGE bit no longer hides the supersede (an exact-usage match missed this).
TEST_CASE("TransientResourcePool - a wider-usage new-size sibling supersedes the old extent", "[RenderGraph]")
{
    PoolFixture f;
    f.pool.frame = 10;
    f.pool.createdThisFrame = 1;
    f.push(tex_entry({ 800, 600, 1 }, /*created*/ 3, /*lastUsed*/ 9)); // old size, idle
    TransientResourcePool::Entry sibling = tex_entry({ 1024, 768, 1 }, /*created*/ 10, /*lastUsed*/ 10);
    sibling.usage = (WGPUTextureUsage)(sibling.usage | WGPUTextureUsage_StorageBinding); // superset of the old entry's
    f.push(sibling);

    f.pool.end_frame();

    REQUIRE(f.count() == 1);
    REQUIRE(f.at(0)->sig.size.width == 1024);
}

// a buffer Entry with the given size and usage. All GPU handles null so destroy() is a no-op. `identity`
// defaults to kIdA, see tex_entry.
static TransientResourcePool::Entry buf_entry(
    uint64_t size, WGPUBufferUsage usage, uint64_t createdFrame, uint64_t lastUsedFrame, const void* identity = kIdA)
{
    TransientResourcePool::Entry e {};
    e.kind = ResourceKind::Buffer;
    e.bufferSize = size;
    e.bufUsage = usage;
    e.createdFrame = createdFrame;
    e.lastUsedFrame = lastUsedFrame;
    e.identity = identity;
    return e;
}

// The buffer arm of superseded_by. Pre-patch this short-circuited on non-textures, so a continuous resize
// drag stacked up to kRetain generations of screen-derived buffers.
TEST_CASE("TransientResourcePool - a resized idle buffer is superseded by its new-size sibling", "[RenderGraph]")
{
    PoolFixture f;
    f.pool.frame = 10;
    f.pool.createdThisFrame = 1;
    f.push(buf_entry(256, WGPUBufferUsage_Storage, /*created*/ 3, /*lastUsed*/ 9)); // old size, idle
    f.push(buf_entry(512, (WGPUBufferUsage)(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst), /*created*/ 10, /*lastUsed*/ 10));

    f.pool.end_frame();

    REQUIRE(f.count() == 1); // collapsed same-frame, not after kRetain
    REQUIRE(f.at(0)->bufferSize == 512);
}

TEST_CASE("TransientResourcePool - a buffer whose usage the sibling does not cover is not superseded", "[RenderGraph]")
{
    PoolFixture f;
    f.pool.frame = 10;
    f.pool.createdThisFrame = 1;
    f.push(buf_entry(256, (WGPUBufferUsage)(WGPUBufferUsage_Storage | WGPUBufferUsage_Indirect), /*created*/ 3, /*lastUsed*/ 9));
    f.push(buf_entry(512, WGPUBufferUsage_Storage, /*created*/ 10, /*lastUsed*/ 10)); // misses Indirect

    f.pool.end_frame();

    REQUIRE(f.count() == 2); // differently-purposed, so not swept
}

TEST_CASE("TransientResourcePool - a buffer claimed this frame is not superseded", "[RenderGraph]")
{
    PoolFixture f;
    f.pool.frame = 10;
    f.pool.createdThisFrame = 1;
    f.push(buf_entry(256, WGPUBufferUsage_Storage, /*created*/ 3, /*lastUsed*/ 10)); // used THIS frame -> live
    f.push(buf_entry(512, WGPUBufferUsage_Storage, /*created*/ 10, /*lastUsed*/ 10));

    f.pool.end_frame();

    REQUIRE(f.count() == 2); // the lastUsedFrame >= frame guard holds for buffers too
}

TEST_CASE("TransientResourcePool - a buffer sibling created in an earlier frame does not supersede", "[RenderGraph]")
{
    PoolFixture f;
    f.pool.frame = 10;
    f.pool.createdThisFrame = 1;
    f.push(buf_entry(256, WGPUBufferUsage_Storage, /*created*/ 3, /*lastUsed*/ 9));
    f.push(buf_entry(512, WGPUBufferUsage_Storage, /*created*/ 8, /*lastUsed*/ 9)); // not this frame's replacement

    f.pool.end_frame();

    REQUIRE(f.count() == 2);
}

// a texture and a buffer never supersede each other, whatever their sizes
TEST_CASE("TransientResourcePool - kinds do not supersede across each other", "[RenderGraph]")
{
    PoolFixture f;
    f.pool.frame = 10;
    f.pool.createdThisFrame = 1;
    f.push(tex_entry({ 800, 600, 1 }, /*created*/ 3, /*lastUsed*/ 9));
    f.push(buf_entry(512, WGPUBufferUsage_Storage, /*created*/ 10, /*lastUsed*/ 10));

    f.pool.end_frame();

    REQUIRE(f.count() == 2);
}

TEST_CASE("TransientResourcePool - same-size buffers are not superseded", "[RenderGraph]")
{
    PoolFixture f;
    f.pool.frame = 10;
    f.pool.createdThisFrame = 1;

    f.push(buf_entry(256, WGPUBufferUsage_Storage, /*created*/ 3, /*lastUsed*/ 9));
    // same size -> not a resize, so not a supersede
    f.push(buf_entry(256, (WGPUBufferUsage)(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst), /*created*/ 10, /*lastUsed*/ 10));

    f.pool.end_frame();

    REQUIRE(f.count() == 2);
}

TEST_CASE("TransientResourcePool - drag resize keeps only the newest generation each frame", "[RenderGraph]")
{
    PoolFixture f;

    // simulate three consecutive frames, each acquiring a fresh, larger size while the previous
    // generation goes idle. After each end_frame() only the size just created must remain.
    const WGPUExtent3D sizes[3] = { { 800, 600, 1 }, { 900, 675, 1 }, { 1000, 750, 1 } };
    for (int i = 0; i < 3; ++i) {
        // this frame's new-size entry, claimed this frame
        f.push(tex_entry(sizes[i], /*created*/ f.pool.frame, /*lastUsed*/ f.pool.frame));
        f.pool.createdThisFrame = 1;

        f.pool.end_frame(); // supersede-evicts any older-size idle generation, then advances frame

        REQUIRE(f.count() == 1);
        REQUIRE(f.at(0)->sig.size.width == sizes[i].width);

        // going into the next frame the surviving entry is now idle (not touched next frame until acquired)
    }
}

TEST_CASE("TransientResourcePool - two idle generations are both swept by one new-size sibling", "[RenderGraph]")
{
    PoolFixture f;
    f.pool.frame = 10;
    f.pool.createdThisFrame = 1;
    // Two older generations of one resource idle at once, which the drag-resize test never produces (it
    // creates and sweeps one per frame). Both are well inside kRetain (10-8 and 10-9 < 4), so only
    // supersede can take them: this pins that end_frame's scan clears a whole backlog in one pass rather
    // than one generation per frame.
    f.push(tex_entry({ 800, 600, 1 }, /*created*/ 3, /*lastUsed*/ 8));
    f.push(tex_entry({ 900, 675, 1 }, /*created*/ 8, /*lastUsed*/ 9));
    f.push(tex_entry({ 1024, 768, 1 }, /*created*/ 10, /*lastUsed*/ 10));

    f.pool.end_frame();

    REQUIRE(f.count() == 1);
    REQUIRE(f.at(0)->sig.size.width == 1024);
}

TEST_CASE("TransientResourcePool - the event log records the supersede evict", "[RenderGraph]")
{
    PoolFixture f;
    f.pool.frame = 10;
    f.pool.createdThisFrame = 1;
    f.push(tex_entry({ 800, 600, 1 }, /*created*/ 3, /*lastUsed*/ 9));
    f.push(tex_entry({ 1024, 768, 1 }, /*created*/ 10, /*lastUsed*/ 10));
    f.pool.log_reset();

    f.pool.end_frame();

    // The surviving-count assertions cannot say WHEN an eviction fired; the log can. Only the Evict is
    // visible here: Create is logged by acquire, which needs a device (see the [gpu] resize tests).
    REQUIRE(f.pool.eventCount == 1);
    REQUIRE(f.pool.eventLog[0].event == TransientResourcePool::Event::Evict);
    REQUIRE(f.pool.eventLog[0].kind == ResourceKind::Texture);
    REQUIRE(f.pool.eventLog[0].frame == 10); // logged before end_frame advances the clock
    REQUIRE(f.pool.eventLog[0].size.width == 800); // the old generation, not the survivor
    REQUIRE(f.pool.eventLog[0].size.height == 600);
}

// ---------------------------------------------------------------------------------------------------
// Supersede must not fire across two DIFFERENT logical resources that merely differ in size. The pool
// matches by descriptor, so without an identity key an idle resource looks exactly like a stale
// generation of any same-shaped sibling created this frame, evicting it churns.

// Claim the pooled buffer matching `size`+`usage` if there is one, else create the entry acquire's miss
// path would. Mirrors acquire: match by descriptor, then stamp the claimant's identity onto the entry
// (an entry created for one resource may legitimately be handed to another later, so identity always
// tracks the LAST claimant). Returns true if this claim created an object.
static bool claim_or_create_buf(PoolFixture& f, uint64_t size, WGPUBufferUsage usage, const void* identity)
{
    for (TransientResourcePool::Entry* e = f.pool.entries; e; e = e->next)
        if (e->kind == ResourceKind::Buffer && !e->inUse && e->bufferSize == size && e->bufUsage == usage) {
            e->lastUsedFrame = f.pool.frame;
            e->identity = identity;
            return false;
        }
    f.push(buf_entry(size, usage, /*created*/ f.pool.frame, /*lastUsed*/ f.pool.frame, identity));
    ++f.pool.createdThisFrame;
    return true;
}

// tex_entry's counterpart to claim_or_create_buf.
static bool claim_or_create_tex(PoolFixture& f, WGPUExtent3D size, const void* identity)
{
    TransientResourcePool::Entry want = tex_entry(size, 0, 0, identity);
    for (TransientResourcePool::Entry* e = f.pool.entries; e; e = e->next)
        if (e->kind == ResourceKind::Texture && !e->inUse && e->sig == want.sig && e->usage == want.usage) {
            e->lastUsedFrame = f.pool.frame;
            e->identity = identity;
            return false;
        }
    f.push(tex_entry(size, /*created*/ f.pool.frame, /*lastUsed*/ f.pool.frame, identity));
    ++f.pool.createdThisFrame;
    return true;
}

TEST_CASE("TransientResourcePool - alternately claimed buffers of different sizes do not churn", "[RenderGraph]")
{
    PoolFixture f;

    // Two unrelated transient buffers, same usage, different sizes, each claimed every OTHER frame, a
    // conditionally recorded pass, or an every-other-frame half-res pass. Neither is a resize of the
    // other. Each is idle for exactly 1 frame at a time, well inside kRetain (4), so both must survive
    // untouched: two objects created in total, and nothing ever destroyed.
    uint32_t creates = 0;
    for (uint64_t i = 0; i < 6; ++i) {
        if (i % 2 == 0)
            creates += claim_or_create_buf(f, 256, WGPUBufferUsage_Storage, kIdA);
        else
            creates += claim_or_create_buf(f, 512, WGPUBufferUsage_Storage, kIdB);
        f.pool.release_claims();
        f.pool.end_frame();
    }

    REQUIRE(creates == 2); // one per resource, on its first claim, never recreated
    REQUIRE(f.count() == 2);
    REQUIRE(f.pool.eventCount == 0); // no evict ever logged
}

TEST_CASE("TransientResourcePool - alternately claimed textures of different extents do not churn", "[RenderGraph]")
{
    PoolFixture f;

    // The texture arm of the same hole: identical format/dim/mips/samples/usage, differing only in extent.
    uint32_t creates = 0;
    for (uint64_t i = 0; i < 6; ++i) {
        if (i % 2 == 0)
            creates += claim_or_create_tex(f, { 800, 600, 1 }, kIdA);
        else
            creates += claim_or_create_tex(f, { 400, 300, 1 }, kIdB);
        f.pool.release_claims();
        f.pool.end_frame();
    }

    REQUIRE(creates == 2);
    REQUIRE(f.count() == 2);
    REQUIRE(f.pool.eventCount == 0);
}

TEST_CASE("TransientResourcePool - a size-only difference across identities does not supersede", "[RenderGraph]")
{
    // The identity bail on its own, isolated from the alternating-frame timing above: one idle entry, one
    // sibling created this frame, same descriptor but for the size, DIFFERENT resources. Same shape as
    // "a resized idle buffer is superseded by its new-size sibling", only the identity differs.
    {
        PoolFixture f;
        f.pool.frame = 10;
        f.pool.createdThisFrame = 1;
        f.push(buf_entry(256, WGPUBufferUsage_Storage, /*created*/ 3, /*lastUsed*/ 9, kIdA));
        f.push(buf_entry(512, WGPUBufferUsage_Storage, /*created*/ 10, /*lastUsed*/ 10, kIdB));

        f.pool.end_frame();

        REQUIRE(f.count() == 2);
    }
    {
        PoolFixture f;
        f.pool.frame = 10;
        f.pool.createdThisFrame = 1;
        f.push(tex_entry({ 800, 600, 1 }, /*created*/ 3, /*lastUsed*/ 9, kIdA));
        f.push(tex_entry({ 1024, 768, 1 }, /*created*/ 10, /*lastUsed*/ 10, kIdB));

        f.pool.end_frame();

        REQUIRE(f.count() == 2);
    }
}

TEST_CASE("TransientResourcePool - different-size buffers claimed every frame are never superseded", "[RenderGraph]")
{
    PoolFixture f;

    // The common case, and the guard that the identity key does not regress it: two differently-sized
    // resources both live every frame. The lastUsedFrame >= frame bail already covers this, so it holds
    // with or without identity, it must keep holding.
    uint32_t creates = 0;
    for (uint64_t i = 0; i < 6; ++i) {
        creates += claim_or_create_buf(f, 256, WGPUBufferUsage_Storage, kIdA);
        creates += claim_or_create_buf(f, 512, WGPUBufferUsage_Storage, kIdB);
        f.pool.release_claims();
        f.pool.end_frame();
    }

    REQUIRE(creates == 2);
    REQUIRE(f.count() == 2);
}

// ---------------------------------------------------------------------------------------------------
// Transient buffer acquire matching. A real device is used so the miss branch actually creates a buffer
// rather than crashing, which is what makes "reused" vs "created" observable as an entry count.

TEST_CASE("TransientResourcePool - a usage-superset idle buffer is reused", "[RenderGraph][gpu]")
{
    UnittestWebgpuContext gpu;
    PoolFixture f;

    // seed the pool the way a previous frame would have: one idle Storage|CopySrc buffer
    WGPUBuffer first = nullptr;
    f.pool.acquire(gpu.device, 256, (WGPUBufferUsage)(WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc), kIdA, first);
    REQUIRE(first != nullptr);
    REQUIRE(f.count() == 1);
    f.pool.release_claims();

    // a narrower request is covered by that buffer's usage -> reuse the same object, no second entry
    WGPUBuffer second = nullptr;
    f.pool.acquire(gpu.device, 256, WGPUBufferUsage_Storage, kIdA, second);
    REQUIRE(second == first);
    REQUIRE(f.count() == 1);

    f.pool.destroy_all();
}

TEST_CASE("TransientResourcePool - a buffer whose usage misses a requested bit is not reused", "[RenderGraph][gpu]")
{
    UnittestWebgpuContext gpu;
    PoolFixture f;

    WGPUBuffer first = nullptr;
    f.pool.acquire(gpu.device, 256, WGPUBufferUsage_Storage, kIdA, first);
    f.pool.release_claims();

    // Storage does not cover Storage|Indirect -> a fresh object, since usage matches by superset
    WGPUBuffer second = nullptr;
    f.pool.acquire(gpu.device, 256, (WGPUBufferUsage)(WGPUBufferUsage_Storage | WGPUBufferUsage_Indirect), kIdA, second);
    REQUIRE(second != first);
    REQUIRE(f.count() == 2);

    f.pool.destroy_all();
}

// acquire matches by descriptor, so an object created for one resource can be handed to another later. Its
// identity must follow the claimant, or the original creator's resize sweeps an object that now belongs to
// someone else -- the same churn the identity key exists to prevent, one level less obvious. A real acquire
// is required: the device-free fixtures stamp identity themselves and so cannot see this.

TEST_CASE("TransientResourcePool - a rehomed buffer is not swept by its creator's resize", "[RenderGraph][gpu]")
{
    UnittestWebgpuContext gpu;
    PoolFixture f;

    // frame 0: A creates a 256B object
    WGPUBuffer a0 = nullptr;
    f.pool.acquire(gpu.device, 256, WGPUBufferUsage_Storage, kIdA, a0);
    f.pool.release_claims();
    f.pool.end_frame();

    // frame 1: A does not record; B asks for the same descriptor and is handed A's object
    WGPUBuffer b1 = nullptr;
    f.pool.acquire(gpu.device, 256, WGPUBufferUsage_Storage, kIdB, b1);
    REQUIRE(b1 == a0); // descriptor match -> reused, the object now serves B
    f.pool.release_claims();
    f.pool.end_frame();

    // frame 2: B is idle, A returns resized. A's new object must not sweep the one B is still using.
    WGPUBuffer a2 = nullptr;
    f.pool.acquire(gpu.device, 512, WGPUBufferUsage_Storage, kIdA, a2);
    f.pool.release_claims();
    f.pool.end_frame();

    REQUIRE(f.count() == 2); // B's object survived A's resize
}

TEST_CASE("TransientResourcePool - a rehomed texture is not swept by its creator's resize", "[RenderGraph][gpu]")
{
    UnittestWebgpuContext gpu;
    PoolFixture f;

    const TransientResourcePool::Entry small = tex_entry({ 16, 16, 1 }, 0, 0);
    const TransientResourcePool::Entry large = tex_entry({ 32, 32, 1 }, 0, 0);

    WGPUTexture a0 = nullptr, a2 = nullptr, b1 = nullptr;
    WGPUTextureView v = nullptr;

    f.pool.acquire(gpu.device, small.sig, small.usage, kIdA, a0, v);
    f.pool.release_claims();
    f.pool.end_frame();

    f.pool.acquire(gpu.device, small.sig, small.usage, kIdB, b1, v);
    REQUIRE(b1 == a0);
    f.pool.release_claims();
    f.pool.end_frame();

    f.pool.acquire(gpu.device, large.sig, large.usage, kIdA, a2, v);
    f.pool.release_claims();
    f.pool.end_frame();

    REQUIRE(f.count() == 2);
}

// The removed MapRead|MapWrite exact-match reservation. Nothing in the graph ever gives a transient buffer
// those bits (readback goes through an imported caller-owned buffer), so the reservation only ever blocked
// reuse of a state the graph cannot produce, hand-seeded here to pin that the restriction is gone.
TEST_CASE("TransientResourcePool - a map-bit-carrying idle buffer no longer blocks superset reuse", "[RenderGraph][gpu]")
{
    UnittestWebgpuContext gpu;
    PoolFixture f;

    WGPUBuffer seeded = nullptr;
    f.pool.acquire(gpu.device, 256, (WGPUBufferUsage)(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst), kIdA, seeded);
    f.pool.release_claims();
    // Stamp the map bit onto the entry rather than asking WebGPU for a Storage|MapRead buffer, which is an
    // invalid usage combination. Only the flags matter here: acquire compares bufUsage, never the object.
    f.pool.entries->bufUsage
        = (WGPUBufferUsage)(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);

    // pre-patch the MapRead bit made this an exact-match miss and forced a second object
    WGPUBuffer plain = nullptr;
    f.pool.acquire(gpu.device, 256, WGPUBufferUsage_Storage, kIdA, plain);
    REQUIRE(plain == seeded);
    REQUIRE(f.count() == 1);

    f.pool.destroy_all();
}

// ===================================================================================================
// Execute-level tests. Everything above stops at compile(); these run realize_graph() + execute()
// against a real Dawn device (UnittestWebgpuContext) and drive the cross-frame pools by running whole
// frames on one long-lived allocator. See docs/rendergraph.md and todo.md "Test coverage".
//
// NOTE: release_resources() nulls a transient node's realized handles at the end of execute(), so
// find_node(rg, h)->texture is null once frame() returns. White-box handle checks (rotation, reuse,
// history_valid) must therefore capture inside a pass body via ctx.*; the pool `entries` vectors on
// the allocator persist across frames and are read after frame() for counts/eviction.

using webgpu::rg::Internal::PersistentResourcePool;
using webgpu::rg::Internal::ResourceNode;
using webgpu::rg::Internal::find_node;

struct ExecGraph {
    UnittestWebgpuContext gpu;
    GraphAllocator* allocator = create_allocator();

    ~ExecGraph() { destroy_allocator(allocator); }
    ExecGraph(const ExecGraph&) = delete;
    ExecGraph& operator=(const ExecGraph&) = delete;
    ExecGraph() = default;

    // Block until the queue drains, so a readback or pool inspection sees finished work.
    void wait_idle()
    {
        bool done = false;
        WGPUQueueWorkDoneCallbackInfo cb {
            .nextInChain = nullptr,
            .mode = WGPUCallbackMode_AllowProcessEvents,
            .callback = [](WGPUQueueWorkDoneStatus, WGPUStringView, void* d, void*) { *reinterpret_cast<bool*>(d) = true; },
            .userdata1 = &done,
            .userdata2 = nullptr,
        };
        WGPUFuture f = wgpuQueueOnSubmittedWorkDone(gpu.queue, cb);
        WGPUFutureWaitInfo wi { .future = f, .completed = false };
        REQUIRE(wgpuInstanceWaitAny(gpu.instance, 1, &wi, 1000ull * 1000 * 1000) == WGPUWaitStatus_Success);
    }

    // compile + execute + submit one already-recorded graph on its own command buffer
    void submit_graph(RenderGraph* rg)
    {
        rg->compile();
        REQUIRE(rg->get_errors() == nullptr);

        WGPUCommandEncoderDescriptor ed {};
        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(gpu.device, &ed);
        rg->execute(gpu.device, gpu.queue, enc);
        WGPUCommandBufferDescriptor cd {};
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, &cd);
        wgpuQueueSubmit(gpu.queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(enc);
    }

    // Run one full frame: record -> compile -> execute -> submit -> wait -> inspect -> end_frame
    template <class Record, class Inspect>
    void frame(Record&& record, Inspect&& inspect)
    {
        begin_frame(allocator);
        RenderGraph* rg = start_recording(allocator);
        record(rg);
        submit_graph(rg);
        wait_idle();
        inspect(rg);
        end_frame(allocator);
    }

    template <class Record>
    void frame(Record&& record)
    {
        frame(std::forward<Record>(record), [](RenderGraph*) {});
    }
};

// Encode a texture->buffer readback into `dst` (an imported buffer) for a WxH RGBA8 `src`, using the
// documented copy recipe (rendergraph.md). Rows are 256-aligned, so buf size must be alignedRow*H.
static void encode_readback(PassContext& ctx, ResourceHandle src, ResourceHandle dst, uint32_t w, uint32_t h)
{
    WGPUTexelCopyBufferLayout layout { .offset = 0, .bytesPerRow = webgpu::rg::aligned_bytes_per_row(w, 4), .rowsPerImage = h };
    auto s = ctx.copy_src_info(src);
    auto d = ctx.copy_dst_buffer(dst, layout);
    auto sz = ctx.copy_extent_src(src);
    wgpuCommandEncoderCopyTextureToBuffer(ctx.encoder, &s, &d, &sz);
}

// Smoke: a Graphics pass clears a transient 16x16 texture to a known color; a Transfer pass copies it
// into an imported mappable buffer; host readback confirms the cleared pixel. Exercises realize's
// transient acquire, attachment build, pass ordering and the copy family end to end.
TEST_CASE("RenderGraph exec - clear then readback returns the clear color", "[RenderGraph][gpu]")
{
    constexpr uint32_t W = 16, H = 16;
    const uint32_t alignedRow = webgpu::rg::aligned_bytes_per_row(W, 4);

    ExecGraph g;
    // imported readback target; MapRead so the host can read it directly after submit.
    webgpu::raii::RawBuffer<uint8_t> readback(g.gpu.device, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead, alignedRow * H, "rg readback");

    g.frame([&](RenderGraph* rg) {
        ResourceHandle tex = rg->create_transient_texture("smoke.tex", [] {
            TextureDesc d {};
            d.dimension = WGPUTextureDimension_2D;
            d.format = WGPUTextureFormat_RGBA8Unorm;
            d.absolute = { W, H, 1 };
            return d;
        }());
        ResourceHandle buf = rg->import_buffer("smoke.readback", readback.handle());

        rg->add_pass(
            "clear", PassKind::Graphics,
            [&](PassBuilder& b) { b.color(tex, 0, WGPULoadOp_Clear, WGPUStoreOp_Store, { 1.0, 0.0, 0.0, 1.0 }); }, [](PassContext&) {});
        rg->add_pass(
            "readback", PassKind::Transfer,
            [&](PassBuilder& b) {
                b.copy_texture_to_buffer(tex, buf);
                b.force_keep();
            },
            [=](PassContext& ctx) { encode_readback(ctx, tex, buf, W, H); });
    });

    std::vector<uint8_t> pixels;
    REQUIRE(readback.read_back_sync(g.gpu.instance, g.gpu.device, pixels) == WGPUMapAsyncStatus_Success);
    REQUIRE(pixels.size() == alignedRow * H);
    // pixel (0,0) = first RGBA8 texel = the clear color (1,0,0,1) -> 255,0,0,255
    REQUIRE(pixels[0] == 255);
    REQUIRE(pixels[1] == 0);
    REQUIRE(pixels[2] == 0);
    REQUIRE(pixels[3] == 255);
}

// The ping-pong pass shared by the history tests: sample .prev, render .curr. .curr is a cull sink, so
// the pass survives without force_keep. `body` rides the arena exec slot, so it must be trivially
// destructible (capture handles/pointers by value)
template <class Body>
static void add_temporal_pass(RenderGraph* rg, HistoryResource h, Body body)
{
    rg->add_pass(
        "temporal", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.sampled(h.prev);
            b.color(h.curr, 0, WGPULoadOp_Clear, WGPUStoreOp_Store);
        },
        body);
}

// PersistentResourcePool rotation + history_valid: over two frames a history pair ping-pongs its two
// physical textures (this frame's .curr is next frame's .prev), and ctx.history_valid(.prev) is false
// on the first frame (nothing written yet) but true afterwards.
TEST_CASE("RenderGraph exec - history ping-pongs and history_valid trips only after the first frame", "[RenderGraph][gpu]")
{
    ExecGraph g;
    WGPUTexture curr[2] = {}, prev[2] = {};
    bool validPrev[2] = {};

    for (int f = 0; f < 2; ++f) {
        WGPUTexture* co = &curr[f];
        WGPUTexture* po = &prev[f];
        bool* vo = &validPrev[f];
        g.frame([&](RenderGraph* rg) {
            HistoryResource h = rg->create_history_texture("hist", tex2d());
            add_temporal_pass(rg, h, [h, co, po, vo](PassContext& ctx) {
                *co = ctx.texture(h.curr);
                *po = ctx.texture(h.prev);
                *vo = ctx.history_valid(h.prev);
            });
        });
    }

    REQUIRE(curr[0] != nullptr);
    REQUIRE(prev[0] != nullptr);
    REQUIRE(curr[0] != prev[0]); // two distinct physical textures
    REQUIRE(curr[1] == prev[0]); // ping-pong: last frame's curr is this frame's prev
    REQUIRE(prev[1] == curr[0]);
    REQUIRE(validPrev[0] == false); // frame 0: .prev never written
    REQUIRE(validPrev[1] == true); // frame 1: .prev holds frame 0's .curr
}

// A changed history hash destroys + recreates the pool entry, zeroing .prev, so history_valid drops
// back to false the frame the hash changes.
TEST_CASE("RenderGraph exec - a changed history hash invalidates .prev", "[RenderGraph][gpu]")
{
    ExecGraph g;
    bool validPrev[3] = {};
    const uint64_t hashes[3] = { 111, 111, 222 }; // frame 2 changes the settings hash

    for (int f = 0; f < 3; ++f) {
        bool* vo = &validPrev[f];
        const uint64_t hash = hashes[f];
        g.frame([&](RenderGraph* rg) {
            HistoryResource h = rg->create_history_texture("hist", tex2d(), hash);
            add_temporal_pass(rg, h, [h, vo](PassContext& ctx) { *vo = ctx.history_valid(h.prev); });
        });
    }

    REQUIRE(validPrev[0] == false); // first use
    REQUIRE(validPrev[1] == true); // steady state
    REQUIRE(validPrev[2] == false); // hash change -> recreate -> prev invalid
}

// A history entry no pass touches for kRetain frames is freed by PersistentResourcePool::end_frame,
// so its physical textures don't leak for the process lifetime after a feature goes inactive.
TEST_CASE("RenderGraph exec - an untouched history entry is evicted after kRetain frames", "[RenderGraph][gpu]")
{
    ExecGraph g;
    g.frame([&](RenderGraph* rg) {
        HistoryResource h = rg->create_history_texture("hist", tex2d());
        add_temporal_pass(rg, h, [](PassContext&) {});
    });
    REQUIRE(list_count(g.allocator->pool.entries) == 1);

    for (uint32_t i = 0; i < PersistentResourcePool::kRetain; ++i)
        g.frame([](RenderGraph*) {}); // empty graph: entry goes untouched
    REQUIRE((g.allocator->pool.entries == nullptr));
}

// The pool entry now holds the interned view instead of an inline char[64], so a name past the old cap
// survives whole. Pre-patch this entry's name was truncated to 63 chars.
TEST_CASE("RenderGraph exec - a pool entry keeps a long name in full", "[RenderGraph][gpu]")
{
    ExecGraph g;
    const std::string longName(120, 'z');

    g.frame([&](RenderGraph* rg) {
        HistoryResource h = rg->create_history_texture(longName, tex2d());
        add_temporal_pass(rg, h, [](PassContext&) {});
    });

    PersistentResourcePool::Entry* e = g.allocator->pool.entries;
    REQUIRE(e != nullptr);
    REQUIRE(e->name_view().length == 120);
    REQUIRE(std::string_view(e->name_view().data, e->name_view().length) == longName);
}

// The entry stores the interner's canonical pointer, so a re-declared name across frames resolves to the
// same entry (this is what pool find() now keys on) rather than allocating a second one.
TEST_CASE("RenderGraph exec - a re-declared name reuses its pool entry across frames", "[RenderGraph][gpu]")
{
    ExecGraph g;
    auto record = [](RenderGraph* rg) {
        HistoryResource h = rg->create_history_texture("hist.reused", tex2d());
        add_temporal_pass(rg, h, [](PassContext&) {});
    };

    g.frame(record);
    REQUIRE(list_count(g.allocator->pool.entries) == 1);
    const char* firstName = g.allocator->pool.entries->name_view().data;

    g.frame(record); // same name next frame -> the same canonical pointer -> the same entry
    REQUIRE(list_count(g.allocator->pool.entries) == 1);
    REQUIRE(g.allocator->pool.entries->name_view().data == firstName);
}

// ---------------------------------------------------------------------------------------------------
// TransientResourcePool, device-backed. Complements the device-free superseded_by/end_frame unit tests
// above by driving acquire()/release_claims across real frames.

using webgpu::rg::Internal::TransientResourcePool;

// A transient 16x16 render target used and stored (StoreOp_Store keeps it out of the memoryless
// transient-attachment path, so its usage bits are stable frame to frame).
TEST_CASE("RenderGraph exec - a same-descriptor transient is reused across frames", "[RenderGraph][gpu]")
{
    ExecGraph g;
    WGPUTexture seen[2] = {};

    for (int f = 0; f < 2; ++f) {
        WGPUTexture* out = &seen[f];
        g.frame([&](RenderGraph* rg) {
            ResourceHandle tex = rg->create_transient_texture("t", tex2d());
            rg->add_pass(
                "w", PassKind::Graphics,
                [&](PassBuilder& b) {
                    b.color(tex, 0, WGPULoadOp_Clear, WGPUStoreOp_Store);
                    b.force_keep();
                },
                [tex, out](PassContext& ctx) { *out = ctx.texture(tex); });
        });
    }

    REQUIRE(seen[0] != nullptr);
    REQUIRE(seen[0] == seen[1]); // same pooled object handed back
    REQUIRE(list_count(g.allocator->transient.entries) == 1); // not reallocated
}

// A pooled texture whose usage is a superset of a later request still satisfies it (superset-reuse):
// frame 0 gives the texture RenderAttachment|CopySrc, frame 1 asks for RenderAttachment only.
TEST_CASE("RenderGraph exec - a wider-usage pooled transient satisfies a narrower request", "[RenderGraph][gpu]")
{
    constexpr uint32_t W = 16, H = 16;
    const uint32_t alignedRow = webgpu::rg::aligned_bytes_per_row(W, 4);

    ExecGraph g;
    webgpu::raii::RawBuffer<uint8_t> readback(g.gpu.device, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead, alignedRow * H, "superset readback");
    WGPUTexture seen[2] = {};

    // frame 0: color() + copy_texture_to_buffer -> usage RenderAttachment|CopySrc
    g.frame([&](RenderGraph* rg) {
        ResourceHandle tex = rg->create_transient_texture("t", tex2d());
        ResourceHandle buf = rg->import_buffer("rb", readback.handle());
        rg->add_pass(
            "w", PassKind::Graphics, [&](PassBuilder& b) { b.color(tex, 0, WGPULoadOp_Clear, WGPUStoreOp_Store); },
            [tex, out = &seen[0]](PassContext& ctx) { *out = ctx.texture(tex); });
        rg->add_pass(
            "rb", PassKind::Transfer,
            [&](PassBuilder& b) {
                b.copy_texture_to_buffer(tex, buf);
                b.force_keep();
            },
            [tex, buf](PassContext& ctx) { encode_readback(ctx, tex, buf, W, H); });
    });

    // frame 1: color() only -> usage RenderAttachment (a subset) -> must reuse the wider pooled object
    g.frame([&](RenderGraph* rg) {
        ResourceHandle tex = rg->create_transient_texture("t", tex2d());
        rg->add_pass(
            "w", PassKind::Graphics,
            [&](PassBuilder& b) {
                b.color(tex, 0, WGPULoadOp_Clear, WGPUStoreOp_Store);
                b.force_keep();
            },
            [tex, out = &seen[1]](PassContext& ctx) { *out = ctx.texture(tex); });
    });

    REQUIRE(seen[0] != nullptr);
    REQUIRE(seen[0] == seen[1]); // narrower request reused the wider object
    REQUIRE(list_count(g.allocator->transient.entries) == 1);
}

// A transient object left unclaimed for kRetain frames is destroyed by TransientResourcePool::end_frame.
TEST_CASE("RenderGraph exec - an idle transient is evicted after kRetain frames", "[RenderGraph][gpu]")
{
    ExecGraph g;
    g.frame([&](RenderGraph* rg) {
        ResourceHandle tex = rg->create_transient_texture("t", tex2d());
        rg->add_pass(
            "w", PassKind::Graphics,
            [&](PassBuilder& b) {
                b.color(tex, 0, WGPULoadOp_Clear, WGPUStoreOp_Store);
                b.force_keep();
            },
            [](PassContext&) {});
    });
    REQUIRE(list_count(g.allocator->transient.entries) == 1);

    for (uint32_t i = 0; i < TransientResourcePool::kRetain; ++i)
        g.frame([](RenderGraph*) {}); // empty graph: object goes unclaimed
    REQUIRE((g.allocator->transient.entries == nullptr));
}

// ---------------------------------------------------------------------------------------------------
// initialize() gating. A force_keep()'d init pass is a cull root only while armed; once its target's
// pool entry is baked with the matching hash, compile() sets skipInit and drops it, so the body stops
// running until the hash changes (or the entry is recreated). The counter proves the exec body runs
// exactly on the frames the pass is armed.
TEST_CASE("RenderGraph exec - initialize() bakes once, skips while the hash holds, re-arms on change", "[RenderGraph][gpu]")
{
    ExecGraph g;
    int bakes = 0;
    int snap[3] = {};
    const uint64_t hashes[3] = { 7, 7, 9 }; // stable for two frames, then changes

    for (int f = 0; f < 3; ++f) {
        const uint64_t hash = hashes[f];
        int* counter = &bakes;
        g.frame(
            [&](RenderGraph* rg) {
                ResourceHandle p = rg->create_persistent_texture("baked", tex2d());
                rg->add_pass(
                    "bake", PassKind::Graphics,
                    [&](PassBuilder& b) {
                        b.color(p, 0, WGPULoadOp_Clear, WGPUStoreOp_Store); // declare the write
                        b.initialize(p, hash); // gate: run only when stale
                        b.force_keep(); // keep the armed bake (no same-frame reader)
                    },
                    [counter](PassContext&) { (*counter)++; });
            },
            [&, f](RenderGraph*) { snap[f] = bakes; });
    }

    REQUIRE(snap[0] == 1); // frame 0: first bake
    REQUIRE(snap[1] == 1); // frame 1: same hash -> skipInit, body did not run
    REQUIRE(snap[2] == 2); // frame 2: hash changed -> re-armed
}

// Multi-graph-per-frame: two graphs recorded and executed within one begin_frame/end_frame on the same
// allocator. release_resources() at the end of graph A's execute() frees its transient claim, so graph
// B's realize hands back the very same physical texture. This is the "graphs submitted in execute
// order on one queue" contract in execute()'s header comment.
TEST_CASE("RenderGraph exec - a second graph in one frame reuses the first graph's released transient", "[RenderGraph][gpu]")
{
    ExecGraph g;
    WGPUTexture a = nullptr, b = nullptr;

    auto record_one = [](RenderGraph* rg, WGPUTexture* out) {
        ResourceHandle tex = rg->create_transient_texture("t", tex2d());
        rg->add_pass(
            "w", PassKind::Graphics,
            [&](PassBuilder& bb) {
                bb.color(tex, 0, WGPULoadOp_Clear, WGPUStoreOp_Store);
                bb.force_keep();
            },
            [tex, out](PassContext& ctx) { *out = ctx.texture(tex); });
    };

    begin_frame(g.allocator);
    {
        RenderGraph* rg = start_recording(g.allocator);
        record_one(rg, &a);
        g.submit_graph(rg); // execute() releases the transient claim on finish
    }
    {
        RenderGraph* rg = start_recording(g.allocator); // same frame, same allocator
        record_one(rg, &b);
        g.submit_graph(rg);
    }
    g.wait_idle();
    end_frame(g.allocator);

    REQUIRE(a != nullptr);
    REQUIRE(a == b); // graph B reused graph A's now-idle transient
    REQUIRE(list_count(g.allocator->transient.entries) == 1); // one physical object served both graphs
}

// Multi-graph-per-frame eviction: the transient pool's aging clock ticks once per begin_frame/end_frame,
// not once per execute(). Two graphs run every aging frame here, yet a transient last used in frame 0
// survives exactly kRetain idle frames, graph count does not accelerate eviction. release_claims() at
// each execute() only frees claims for same-frame reuse; only end_frame() advances `frame`.
TEST_CASE("RenderGraph exec - transient eviction clock is per-frame, not per-graph", "[RenderGraph][gpu]")
{
    using webgpu::rg::Internal::TransientResourcePool;
    ExecGraph g;

    // Count the tracked transient: 16x16 RGBA8. The filler below uses a different format so it neither
    // reuses this object nor trips the same-format resize-supersede path (superseded_by), leaving only the
    // kRetain idle clock to evict it.
    auto count_tracked = [&] {
        size_t n = 0;
        for (const TransientResourcePool::Entry* e = g.allocator->transient.entries; e; e = e->next)
            if (e->kind == ResourceKind::Texture && e->sig.size.width == 16 && e->sig.format == WGPUTextureFormat_RGBA8Unorm)
                ++n;
        return n;
    };

    auto clear_transient = [](RenderGraph* rg, const char* id, uint32_t wh, WGPUTextureFormat fmt) {
        TextureDesc d = tex2d();
        d.absolute = { wh, wh, 1 };
        d.format = fmt;
        ResourceHandle t = rg->create_transient_texture(id, d);
        rg->add_pass(
            "w", PassKind::Graphics,
            [&](PassBuilder& b) {
                b.color(t, 0, WGPULoadOp_Clear, WGPUStoreOp_Store);
                b.force_keep(); // nothing reads it; keep the pass so the transient is realized
            },
            [](PassContext&) {});
    };

    // Frame 0: one graph creates the 16x16 RGBA8 transient we track (lastUsedFrame = frame 0).
    g.frame([&](RenderGraph* rg) { clear_transient(rg, "tracked", 16, WGPUTextureFormat_RGBA8Unorm); });
    REQUIRE(count_tracked() == 1);

    // Aging frames: TWO graphs each, both using an R8 filler (distinct format) so nothing reuses the idle
    // tracked object or refreshes its recency. If eviction were per-execute, 2 graphs/frame would age it
    // twice as fast; because it is per-frame, it survives kRetain-1 idle frames then dies on kRetain.
    auto aging_frame = [&] {
        begin_frame(g.allocator);
        { RenderGraph* rg = start_recording(g.allocator); clear_transient(rg, "filler", 8, WGPUTextureFormat_R8Unorm); g.submit_graph(rg); }
        { RenderGraph* rg = start_recording(g.allocator); clear_transient(rg, "filler", 8, WGPUTextureFormat_R8Unorm); g.submit_graph(rg); }
        g.wait_idle();
        end_frame(g.allocator);
    };

    for (uint64_t i = 1; i < TransientResourcePool::kRetain; ++i) {
        aging_frame();
        REQUIRE(count_tracked() == 1); // idle < kRetain frames -> retained despite 2 graphs/frame
    }
    aging_frame();
    REQUIRE(count_tracked() == 0); // idle == kRetain frames -> evicted
}

// Two graphs in one frame requesting the same transient (same id/format) at different sizes. acquire() is
// exact-size create-on-miss and destruction is deferred to end_frame(), so: graph B never gets handed
// graph A's wrong-sized object, both physical objects live simultaneously through the frame, and because
// both were used this frame supersede-eviction does not fire on either, both survive end_frame().
TEST_CASE("RenderGraph exec - two sizes used in one frame coexist and neither is evicted", "[RenderGraph][gpu]")
{
    using webgpu::rg::Internal::TransientResourcePool;
    ExecGraph g;
    WGPUTexture big = nullptr, small = nullptr;

    auto clear_sized = [](RenderGraph* rg, uint32_t wh, WGPUTexture* out) {
        TextureDesc d = tex2d(); // same format; only the size differs between the two graphs
        d.absolute = { wh, wh, 1 };
        ResourceHandle t = rg->create_transient_texture("t", d);
        rg->add_pass(
            "w", PassKind::Graphics,
            [&](PassBuilder& b) {
                b.color(t, 0, WGPULoadOp_Clear, WGPUStoreOp_Store);
                b.force_keep();
            },
            [t, out](PassContext& ctx) { *out = ctx.texture(t); });
    };

    auto count = [&](uint32_t w) {
        size_t n = 0;
        for (const TransientResourcePool::Entry* e = g.allocator->transient.entries; e; e = e->next)
            if (e->kind == ResourceKind::Texture && e->sig.size.width == w)
                ++n;
        return n;
    };

    begin_frame(g.allocator);
    { RenderGraph* rg = start_recording(g.allocator); clear_sized(rg, 16, &big); g.submit_graph(rg); }
    { RenderGraph* rg = start_recording(g.allocator); clear_sized(rg, 8, &small); g.submit_graph(rg); }
    g.wait_idle();

    // Mid-frame (before end_frame): both objects live at once; graph B got a fresh 8x8, not graph A's 16x16.
    REQUIRE(big != nullptr);
    REQUIRE(small != nullptr);
    REQUIRE(big != small); // exact-size match -> distinct objects, no wrong-size reuse
    REQUIRE(list_count(g.allocator->transient.entries) == 2); // no mid-frame destruction

    end_frame(g.allocator);

    // Both were touched this frame -> superseded_by bails on lastUsedFrame >= frame -> neither is swept.
    REQUIRE(count(16) == 1);
    REQUIRE(count(8) == 1);
}

// ---------------------------------------------------------------------------------------------------
// Resize end to end. The device-free pool tests drive end_frame() on hand-built entries; these are the
// only proof that the real realize -> acquire -> end_frame path resizes as designed, identity and all.

// An RGBA8 2D desc at a caller-chosen absolute size, for the resize and relative-sizing tests.
static TextureDesc tex2d_sized(uint32_t w, uint32_t h)
{
    TextureDesc d = tex2d();
    d.absolute = { w, h, 1 };
    return d;
}

TEST_CASE("RenderGraph exec - a resize destroys the old-size transient", "[RenderGraph][gpu]")
{
    ExecGraph g;
    WGPUTexture before = nullptr, after = nullptr;

    // One transient, same name every frame: a screen-sized target across a window resize.
    auto frame_at = [](RenderGraph* rg, uint32_t w, WGPUTexture* out) {
        ResourceHandle t = rg->create_transient_texture("screen", tex2d_sized(w, w));
        rg->add_pass(
            "w", PassKind::Graphics,
            [&](PassBuilder& b) {
                b.color(t, 0, WGPULoadOp_Clear, WGPUStoreOp_Store);
                b.force_keep();
            },
            [t, out](PassContext& ctx) { *out = ctx.texture(t); });
    };

    uint32_t slotCount = 99;
    g.frame([&](RenderGraph* rg) { frame_at(rg, 16, &before); },
        [&](RenderGraph* rg) { slotCount = storage(rg)->m_slotCount; });
    // Records which acquire arm this test covers: a lone clear+store attachment is aliasing-eligible, so
    // it rides an alias slot and the identity under test is PhysicalResource::identity. The non-aliased
    // per-resource arm is covered by the alias-excluded resize below.
    REQUIRE(slotCount == 1);

    g.frame([&](RenderGraph* rg) { frame_at(rg, 16, &before); }); // steady state: pooled, not recreated
    REQUIRE(list_count(g.allocator->transient.entries) == 1);

    g.frame([&](RenderGraph* rg) { frame_at(rg, 32, &after); });

    // The 16x16 is gone the frame the 32x32 appeared, one frame after its last use -- well inside
    // kRetain (4), so only supersede can have taken it.
    REQUIRE(after != nullptr);
    REQUIRE(after != before);
    REQUIRE(list_count(g.allocator->transient.entries) == 1);
    REQUIRE(g.allocator->transient.entries->sig.size.width == 32);
}

TEST_CASE("RenderGraph exec - a non-aliased transient resizes without leaking a generation", "[RenderGraph][gpu]")
{
    ExecGraph g;
    WGPUTexture before = nullptr, after = nullptr;

    // A mip>1 transient is excluded from aliasing, so this drives the per-resource acquire arm, whose
    // identity comes straight off the resource name rather than an alias slot. Without that identity the
    // supersede silently stops firing and the old generation lingers to kRetain: exactly the regression
    // the aliased test above cannot see.
    auto frame_at = [](RenderGraph* rg, uint32_t w, WGPUTexture* out) {
        TextureDesc d = tex2d_sized(w, w);
        d.mipLevelCount = 2;
        ResourceHandle t = rg->create_transient_texture("chain", d);
        rg->add_pass(
            "w", PassKind::Graphics,
            [&](PassBuilder& b) {
                b.color(t, 0, WGPULoadOp_Clear, WGPUStoreOp_Store, {}, /*baseMip*/ 0);
                b.force_keep();
            },
            [t, out](PassContext& ctx) { *out = ctx.texture(t); });
    };

    uint32_t slotCount = 99;
    g.frame([&](RenderGraph* rg) { frame_at(rg, 16, &before); },
        [&](RenderGraph* rg) { slotCount = storage(rg)->m_slotCount; });
    REQUIRE(slotCount == 0); // not aliased -> the per-resource arm

    g.frame([&](RenderGraph* rg) { frame_at(rg, 32, &after); });

    REQUIRE(after != before);
    REQUIRE(list_count(g.allocator->transient.entries) == 1);
    REQUIRE(g.allocator->transient.entries->sig.size.width == 32);
}

TEST_CASE("RenderGraph exec - an aliased slot resizes without leaking a generation", "[RenderGraph][gpu]")
{
    ExecGraph g;
    WGPUTexture a = nullptr, b = nullptr;

    // Two disjoint transients collapsed onto one alias slot, resized. The slot's pooled object is
    // superseded like any other: aliasing must not multiply generations across a resize.
    auto frame_at = [](RenderGraph* rg, uint32_t w, WGPUTexture* o1, WGPUTexture* o2) {
        ResourceHandle t1 = rg->create_transient_texture("t1", tex2d_sized(w, w));
        ResourceHandle t2 = rg->create_transient_texture("t2", tex2d_sized(w, w));
        rg->add_pass(
            "k1", PassKind::Graphics,
            [&](PassBuilder& bb) {
                bb.color(t1, 0, WGPULoadOp_Clear, WGPUStoreOp_Store);
                bb.force_keep();
            },
            [t1, o1](PassContext& ctx) { *o1 = ctx.texture(t1); });
        rg->add_pass(
            "k2", PassKind::Graphics,
            [&](PassBuilder& bb) {
                bb.color(t2, 0, WGPULoadOp_Clear, WGPUStoreOp_Store);
                bb.force_keep();
            },
            [t2, o2](PassContext& ctx) { *o2 = ctx.texture(t2); });
    };

    uint32_t slotsBefore = 0;
    g.frame([&](RenderGraph* rg) { frame_at(rg, 16, &a, &b); },
        [&](RenderGraph* rg) { slotsBefore = storage(rg)->m_slotCount; });
    REQUIRE(slotsBefore == 1); // both transients share one slot
    REQUIRE(a == b);
    REQUIRE(list_count(g.allocator->transient.entries) == 1);

    uint32_t slotsAfter = 0;
    g.frame([&](RenderGraph* rg) { frame_at(rg, 32, &a, &b); },
        [&](RenderGraph* rg) { slotsAfter = storage(rg)->m_slotCount; });

    REQUIRE(slotsAfter == 1);
    REQUIRE(a == b);
    REQUIRE(list_count(g.allocator->transient.entries) == 1); // the 16x16 slot object did not survive
    REQUIRE(g.allocator->transient.entries->sig.size.width == 32);
}

// ===================================================================================================
// Gap-analysis additions (docs plan do-a-gap-analysis). Compile-only unless tagged [gpu]. These close
// coverage holes in relative sizing, history-layer validation, storage RMW, buffer/excluded aliasing,
// the buffer read access types, and the initialized/history-buffer constructors.

// Sparse color slots end-to-end. Declaring only slots 0 and 2 must make execute() emit a 3-entry color
// array whose slot 1 is a WebGPU null attachment. A bare render pass would not prove that: with no
// pipeline bound Dawn validates almost nothing about the color array, so a wrong colorAttachmentCount or
// an attachment packed at the wrong index still passes. Binding a pipeline whose fragment writes
// @location(0) and @location(2) forces Dawn to match the pipeline's 3 targets against the pass, which
// fails loudly on either mistake.
TEST_CASE("RenderGraph exec - a sparse color slot is emitted as a valid null attachment", "[RenderGraph][gpu]")
{
    ExecGraph g;

    static const char* kWgsl = R"(
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    var p = array<vec2f, 3>(vec2f(-1.0, -3.0), vec2f(-1.0, 1.0), vec2f(3.0, 1.0));
    return vec4f(p[i], 0.0, 1.0);
}
struct Out {
    @location(0) a: vec4f,
    @location(2) c: vec4f,
}
@fragment fn fs() -> Out {
    return Out(vec4f(1.0, 0.0, 0.0, 1.0), vec4f(0.0, 0.0, 1.0, 1.0));
}
)";
    WGPUShaderSourceWGSL wgsl {};
    wgsl.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgsl.code = WGPUStringView { .data = kWgsl, .length = WGPU_STRLEN };
    WGPUShaderModuleDescriptor smd {};
    smd.nextInChain = &wgsl.chain;
    WGPUShaderModule sm = wgpuDeviceCreateShaderModule(g.gpu.device, &smd);
    REQUIRE(sm != nullptr);

    // slot 1 is deliberately an unused target (format Undefined) -> the pipeline expects a null attachment
    // exactly where the graph must emit one.
    WGPUColorTargetState targets[3] {};
    targets[0].format = WGPUTextureFormat_RGBA8Unorm;
    targets[0].writeMask = WGPUColorWriteMask_All;
    targets[1].format = WGPUTextureFormat_Undefined;
    targets[2].format = WGPUTextureFormat_RGBA8Unorm;
    targets[2].writeMask = WGPUColorWriteMask_All;

    WGPUFragmentState fs {};
    fs.module = sm;
    fs.entryPoint = WGPUStringView { .data = "fs", .length = WGPU_STRLEN };
    fs.targetCount = 3;
    fs.targets = targets;

    WGPURenderPipelineDescriptor rpd {};
    rpd.vertex.module = sm;
    rpd.vertex.entryPoint = WGPUStringView { .data = "vs", .length = WGPU_STRLEN };
    rpd.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    rpd.multisample.count = 1;
    rpd.multisample.mask = 0xFFFFFFFF;
    rpd.fragment = &fs;
    WGPURenderPipeline pipe = wgpuDeviceCreateRenderPipeline(g.gpu.device, &rpd);
    REQUIRE(pipe != nullptr);

    wgpuDevicePushErrorScope(g.gpu.device, WGPUErrorFilter_Validation);

    g.frame([pipe](RenderGraph* rg) {
        ResourceHandle s0 = rg->create_transient_texture("slot0", tex2d_sized(16, 16));
        ResourceHandle s2 = rg->create_transient_texture("slot2", tex2d_sized(16, 16));
        rg->add_pass(
            "sparse", PassKind::Graphics,
            [s0, s2](PassBuilder& b) {
                b.color(s0, 0);
                b.color(s2, 2); // slot 1 left unused -> null attachment, colorAttachmentCount == 3
                b.force_keep();
            },
            [pipe](PassContext& c) {
                wgpuRenderPassEncoderSetPipeline(c.render_pass, pipe);
                wgpuRenderPassEncoderDraw(c.render_pass, 3, 1, 0, 0);
            });
    });

    struct ScopeResult {
        bool done = false;
        bool failed = false;
        std::string message;
    } res;
    WGPUPopErrorScopeCallbackInfo ci {
        .nextInChain = nullptr,
        .mode = WGPUCallbackMode_AllowProcessEvents,
        .callback = [](WGPUPopErrorScopeStatus, WGPUErrorType type, WGPUStringView msg, void* d, void*) {
            ScopeResult* r = reinterpret_cast<ScopeResult*>(d);
            r->done = true;
            r->failed = (type != WGPUErrorType_NoError);
            if (msg.data)
                r->message.assign(msg.data, msg.length);
        },
        .userdata1 = &res,
        .userdata2 = nullptr,
    };
    WGPUFuture f = wgpuDevicePopErrorScope(g.gpu.device, ci);
    WGPUFutureWaitInfo wi { .future = f, .completed = false };
    REQUIRE(wgpuInstanceWaitAny(g.gpu.instance, 1, &wi, 1000ull * 1000 * 1000) == WGPUWaitStatus_Success);
    REQUIRE(res.done);
    INFO(res.message);
    REQUIRE_FALSE(res.failed);

    wgpuRenderPipelineRelease(pipe);
    wgpuShaderModuleRelease(sm);
}


// A relativeTo child is width*scale (rounded, not truncated). base 10 x 0.25 = 2.5 -> 3.
TEST_CASE("RenderGraph - relative-sized transient rounds against its base", "[RenderGraph]")
{
    using webgpu::rg::Internal::find_node;

    TestGraph g;
    ResourceHandle base = g.rg->create_transient_texture("base", tex2d_sized(10, 10));

    TextureDesc rd {};
    rd.dimension = WGPUTextureDimension_2D;
    rd.format = WGPUTextureFormat_RGBA8Unorm;
    rd.sizeKind = SizeKind::Relative;
    rd.scaleX = 0.25f;
    rd.scaleY = 0.25f;
    rd.relativeTo = base;
    rd.absolute = { 0, 0, 1 }; // width/height come from the base; depthOrArrayLayers stays 1
    ResourceHandle child = g.rg->create_transient_texture("child", rd);

    g.rg->add_pass(
        "w", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.color(child, 0, WGPULoadOp_Clear, WGPUStoreOp_Store);
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);
    // 10 * 0.25 = 2.5 -> round to 3 (truncation would give 2)
    REQUIRE(find_node(g.rg, child)->resolved.width == 3);
    REQUIRE(find_node(g.rg, child)->resolved.height == 3);
}

// A two-deep relativeTo chain resolves recursively: base 40 -> mid 0.5 -> leaf 0.5.
TEST_CASE("RenderGraph - relative-size chain resolves each level", "[RenderGraph]")
{
    using webgpu::rg::Internal::find_node;

    TestGraph g;
    ResourceHandle base = g.rg->create_transient_texture("base", tex2d_sized(40, 40));

    auto half_of = [&](std::string_view id, ResourceHandle parent) {
        TextureDesc d {};
        d.dimension = WGPUTextureDimension_2D;
        d.format = WGPUTextureFormat_RGBA8Unorm;
        d.sizeKind = SizeKind::Relative;
        d.scaleX = 0.5f;
        d.scaleY = 0.5f;
        d.relativeTo = parent;
        d.absolute = { 0, 0, 1 };
        return g.rg->create_transient_texture(id, d);
    };
    ResourceHandle mid = half_of("mid", base);
    ResourceHandle leaf = half_of("leaf", mid);

    g.rg->add_pass(
        "w", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.color(leaf, 0, WGPULoadOp_Clear, WGPUStoreOp_Store);
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);
    REQUIRE(find_node(g.rg, mid)->resolved.width == 20);
    REQUIRE(find_node(g.rg, leaf)->resolved.width == 10);
}

// Writing .curr while nothing reads .prev leaves the two layers
// asymmetrically culled (.curr used, .prev unused), which is unrealizable -> error.
TEST_CASE("RenderGraph - history with a curr writer but no prev reader is an error", "[RenderGraph]")
{
    TestGraph g;
    HistoryResource hist = g.rg->create_history_texture("hist", tex2d());

    g.rg->add_pass(
        "writeCurr", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.color(hist.curr, 0, WGPULoadOp_Clear, WGPUStoreOp_Store); // .curr used, .prev never read
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(error_mentions(g.rg, "read .prev and write .curr in one pass"));
}

// Only layer 0 (.curr) is writable; writing the .prev handle poisons the graph.
TEST_CASE("RenderGraph - writing a history .prev layer is an error", "[RenderGraph]")
{
    TestGraph g;
    HistoryResource hist = g.rg->create_history_texture("hist", tex2d());

    g.rg->add_pass(
        "writePrev", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.color(hist.prev, 0, WGPULoadOp_Clear, WGPUStoreOp_Store); // layer 1 is read-only this frame
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(error_mentions(g.rg, "only layer 0, the .curr handle, is writable"));
}

// storage_read_write on a transient no earlier pass produced is a read-before-write: the same-pass
// write can't seed the read (dispatch invocations are unordered), so compile() must poison the graph.
TEST_CASE("RenderGraph - storage_read_write on an unproduced transient is an error", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle buf = g.transient_buffer("rmw");

    g.rg->add_pass(
        "rmw", PassKind::Compute,
        [&](PassBuilder& b) {
            b.storage_read_write(buf);
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    // the read-modify-write diagnosis, not the plain "no pass ever writes" one: the same-pass write means
    // a writer exists, it just cannot seed this read
    REQUIRE(error_mentions(g.rg, "can't seed the read"));
}

// storage_read_write is legal once an earlier pass has produced the buffer: producer -> RMW compiles clean.
TEST_CASE("RenderGraph - storage_read_write after a producer compiles clean", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle buf = g.transient_buffer("rmw");

    add_buffer_producer(g.rg, buf); // storage_write producer, satisfies the RMW read
    g.rg->add_pass(
        "rmw", PassKind::Compute,
        [&](PassBuilder& b) {
            b.storage_read_write(buf);
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);
    REQUIRE(idx_of(pass_order(g.rg), "produce") < idx_of(pass_order(g.rg), "rmw"));
}

// Two disjoint-lifetime transient buffers of one size pack onto a single physical
// slot, exercising the isBuf path the texture alias tests never reach. storage_write fully defines.
TEST_CASE("RenderGraph - disjoint transient buffers share one physical slot", "[RenderGraph]")
{
    using webgpu::rg::Internal::find_node;
    using webgpu::rg::Internal::ResourceNode;

    TestGraph g;
    ResourceHandle b1 = g.transient_buffer("b1", 64);
    ResourceHandle b2 = g.transient_buffer("b2", 64);

    g.rg->add_pass(
        "k1", PassKind::Compute,
        [&](PassBuilder& b) {
            b.storage_write(b1);
            b.force_keep();
        },
        [](PassContext&) {});
    g.rg->add_pass(
        "k2", PassKind::Compute,
        [&](PassBuilder& b) {
            b.storage_write(b2);
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile(true);
    REQUIRE(g.rg->get_errors() == nullptr);
    REQUIRE(storage(g.rg)->m_slotCount == 1);
    ResourceNode* r1 = find_node(g.rg, b1);
    ResourceNode* r2 = find_node(g.rg, b2);
    REQUIRE(r1->aliasSlot != ResourceNode::kNoSlot);
    REQUIRE(r1->aliasSlot == r2->aliasSlot);
}

// Aliasing excludes mip-chain transients (a slot's default view wouldn't fit): two disjoint mip-chain
// transients each keep their own object, so no slots are built and both stay kNoSlot.
TEST_CASE("RenderGraph - mip-chain transients are excluded from aliasing", "[RenderGraph]")
{
    using webgpu::rg::Internal::find_node;
    using webgpu::rg::Internal::ResourceNode;

    TestGraph g;
    ResourceHandle t1 = g.transient("t1", /*mipLevelCount*/ 2);
    ResourceHandle t2 = g.transient("t2", /*mipLevelCount*/ 2);

    g.rg->add_pass(
        "k1", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.color(t1, 0, WGPULoadOp_Clear, WGPUStoreOp_Store, {}, /*baseMip*/ 0);
            b.force_keep();
        },
        [](PassContext&) {});
    g.rg->add_pass(
        "k2", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.color(t2, 0, WGPULoadOp_Clear, WGPUStoreOp_Store, {}, /*baseMip*/ 0);
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile(true);
    REQUIRE(g.rg->get_errors() == nullptr);
    REQUIRE(storage(g.rg)->m_slotCount == 0);
    REQUIRE(find_node(g.rg, t1)->aliasSlot == ResourceNode::kNoSlot);
    REQUIRE(find_node(g.rg, t2)->aliasSlot == ResourceNode::kNoSlot);
}

// A transient whose first touch is a non-Clear load does not fully define its storage (access_defines
// keys on loadOp==Clear), so it is ineligible for aliasing even though a pass writes it -> no slots.
TEST_CASE("RenderGraph - a load-first transient is excluded from aliasing", "[RenderGraph]")
{
    using webgpu::rg::Internal::find_node;
    using webgpu::rg::Internal::ResourceNode;

    TestGraph g;
    ResourceHandle t1 = g.transient("t1");
    ResourceHandle t2 = g.transient("t2");

    g.rg->add_pass(
        "k1", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.color(t1, 0, WGPULoadOp_Load, WGPUStoreOp_Store); // Load-first: not a full define
            b.force_keep();
        },
        [](PassContext&) {});
    g.rg->add_pass(
        "k2", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.color(t2, 0, WGPULoadOp_Load, WGPUStoreOp_Store);
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile(true);
    REQUIRE(g.rg->get_errors() == nullptr);
    REQUIRE(storage(g.rg)->m_slotCount == 0);
    REQUIRE(find_node(g.rg, t1)->aliasSlot == ResourceNode::kNoSlot);
    REQUIRE(find_node(g.rg, t2)->aliasSlot == ResourceNode::kNoSlot);
}

// detect_transient_attachments also covers depth: a depth attachment cleared then discarded and touched
// once is inferred memoryless, the depth analogue of the color clear+discard test.
TEST_CASE("RenderGraph - single clear+discard depth attachment is inferred transient", "[RenderGraph]")
{
    TestGraph g;
    TextureDesc dd {};
    dd.dimension = WGPUTextureDimension_2D;
    dd.format = WGPUTextureFormat_Depth32Float;
    dd.absolute = { 16, 16, 1 };
    ResourceHandle ds = g.rg->create_transient_texture("ds", dd);

    g.rg->add_pass(
        "depth", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.depth_stencil(ds, WGPULoadOp_Clear, WGPUStoreOp_Discard);
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);
    REQUIRE(storage(g.rg)->transientCount == 1);
}

// Buffer read access types (uniform/vertex/index/indirect) accumulate their WGPUBufferUsage onto the
// node. Read against persistent buffers (external -> no read-before-write) kept alive with force_keep.
TEST_CASE("RenderGraph - buffer read accesses accumulate their usage bits", "[RenderGraph]")
{
    using webgpu::rg::Internal::find_node;

    TestGraph g;
    BufferDesc bd {};
    bd.size = 64;
    ResourceHandle ubo = g.rg->create_persistent_buffer("ubo", bd);
    ResourceHandle vbo = g.rg->create_persistent_buffer("vbo", bd);
    ResourceHandle ibo = g.rg->create_persistent_buffer("ibo", bd);
    ResourceHandle indirect = g.rg->create_persistent_buffer("indirect", bd);

    g.rg->add_pass(
        "read", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.uniform(ubo);
            b.vertex_buffer(vbo);
            b.index_buffer(ibo);
            b.indirect_buffer(indirect);
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);
    REQUIRE((find_node(g.rg, ubo)->bufUsage & WGPUBufferUsage_Uniform) != 0);
    REQUIRE((find_node(g.rg, vbo)->bufUsage & WGPUBufferUsage_Vertex) != 0);
    REQUIRE((find_node(g.rg, ibo)->bufUsage & WGPUBufferUsage_Index) != 0);
    REQUIRE((find_node(g.rg, indirect)->bufUsage & WGPUBufferUsage_Indirect) != 0);
}

// A write-after-read hazard orders the later writer after the reader of the version it clobbers:
// produce x -> read x -> rewrite x. The rewrite shares no edge with the reader except the WAR one, so
// reader-before-rewrite proves the WAR edge fired.
TEST_CASE("RenderGraph - write-after-read orders the rewriter after the reader", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle sink = import_tex(g.rg, "sink");
    ResourceHandle x = g.transient("x");

    g.rg->add_pass(
        "produce", PassKind::Graphics, [&](PassBuilder& b) { b.color(x, 0, WGPULoadOp_Clear, WGPUStoreOp_Store); }, [](PassContext&) {});
    g.rg->add_pass(
        "read", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.sampled(x);
            b.color(sink, 0, WGPULoadOp_Load, WGPUStoreOp_Store);
        },
        [](PassContext&) {});
    g.rg->add_pass( // clobbers x's version the reader still uses -> WAR edge onto "read"
        "rewrite", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.color(x, 0, WGPULoadOp_Clear, WGPUStoreOp_Store);
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);
    auto order = pass_order(g.rg);
    REQUIRE(idx_of(order, "read") < idx_of(order, "rewrite"));
}

// initialize() targets must be pool-backed (persistent/history). Arming it on a transient is an error.
TEST_CASE("RenderGraph - initialize() on a transient target is an error", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle tex = g.transient("tex");

    g.rg->add_pass(
        "bake", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.color(tex, 0, WGPULoadOp_Clear, WGPUStoreOp_Store); // the required write
            b.initialize(tex); // ... but tex is transient, not persistent
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(error_mentions(g.rg, "must be persistent or history"));
}

// create_history_buffer smoke: a compute pass reads .prev and writes .curr (the ping-pong pattern for
// a GPU-authored buffer). .curr is a cull sink, .prev is external, so it compiles clean and survives.
TEST_CASE("RenderGraph - history buffer read-prev/write-curr compiles clean", "[RenderGraph]")
{
    TestGraph g;
    BufferDesc bd {};
    bd.size = 64;
    HistoryResource h = g.rg->create_history_buffer("hbuf", bd);

    g.rg->add_pass(
        "temporal", PassKind::Compute,
        [&](PassBuilder& b) {
            b.storage_read(h.prev);
            b.storage_write(h.curr);
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);
    REQUIRE(pass_order(g.rg).size() == 1);
}

// create_initialized_texture synthesizes an initialize()'d bake pass. A reader pulls that bake in as a
// dependency, so both survive and the graph compiles clean.
TEST_CASE("RenderGraph - create_initialized_texture bakes a pass a reader can consume", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle fallback = g.rg->create_initialized_texture("fallback", tex2d(), { 0, 0, 0, 1 });
    ResourceHandle out = g.transient("out");

    g.rg->add_pass(
        "use", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.sampled(fallback);
            b.color(out, 0);
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);
    REQUIRE(pass_order(g.rg).size() == 2); // the synthesized bake + the reader
    REQUIRE(idx_of(pass_order(g.rg), "fallback") != -1); // single layer keeps the bare id, no suffix
}

// A multi-layer init bakes one pass per layer, each named "<id>.layer<n>" so the encoder debug groups and
// profiler series stay distinct. assert_unique_id also rejects duplicate pass ids, so a missing suffix
// would not merely blur the labels.
TEST_CASE("RenderGraph - create_initialized_texture names one bake pass per layer", "[RenderGraph]")
{
    TestGraph g;
    TextureDesc d = tex2d();
    d.absolute = { 16, 16, 3 };
    ResourceHandle fallback = g.rg->create_initialized_texture("fallback", d, { 0, 0, 0, 1 });
    ResourceHandle out = g.transient("out");

    g.rg->add_pass(
        "use", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.sampled(fallback, 0, 0, ViewRange { WGPUTextureViewDimension_2DArray, 1, 3, WGPUTextureAspect_All });
            b.color(out, 0);
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);

    std::vector<std::string> order = pass_order(g.rg);
    REQUIRE(order.size() == 4); // three bakes + the reader
    REQUIRE(idx_of(order, "fallback.layer0") != -1);
    REQUIRE(idx_of(order, "fallback.layer1") != -1);
    REQUIRE(idx_of(order, "fallback.layer2") != -1);
    REQUIRE(idx_of(order, "fallback") == -1); // suffixed, not bare
}

// The suffixed name is built in the scratch arena, sized from the id rather than a fixed buffer, so a
// long id must come through whole rather than truncating into a duplicate/blurred pass name.
TEST_CASE("RenderGraph - a long multi-layer init id is not truncated", "[RenderGraph]")
{
    TestGraph g;
    const std::string longId(200, 'y');
    TextureDesc d = tex2d();
    d.absolute = { 16, 16, 2 };
    ResourceHandle fallback = g.rg->create_initialized_texture(longId, d, { 0, 0, 0, 1 });
    ResourceHandle out = g.transient("out");

    g.rg->add_pass(
        "use", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.sampled(fallback, 0, 0, ViewRange { WGPUTextureViewDimension_2DArray, 1, 2, WGPUTextureAspect_All });
            b.color(out, 0);
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);

    std::vector<std::string> order = pass_order(g.rg);
    REQUIRE(idx_of(order, (longId + ".layer0").c_str()) != -1);
    REQUIRE(idx_of(order, (longId + ".layer1").c_str()) != -1);
}

// create_initialized_buffer synthesizes a host_write + initialize() bake pass; a uniform reader pulls
// it in as a dependency, so both survive and the graph compiles clean.
TEST_CASE("RenderGraph - create_initialized_buffer bakes a pass a reader can consume", "[RenderGraph]")
{
    TestGraph g;
    BufferDesc bd {};
    bd.size = 64;
    ResourceHandle fallback = g.rg->create_initialized_buffer("fallback", bd, nullptr);

    g.rg->add_pass(
        "use", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.uniform(fallback);
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);
    REQUIRE(pass_order(g.rg).size() == 2); // the synthesized upload + the reader
}

// A clear+discard color transient is inferred memoryless (transientAttachment) and, because there is no
// persistent storage to pack, is excluded from aliasing: it keeps its own object even with aliasing on.
// Closes the third exclusion branch alongside the mip-chain and load-first tests.
TEST_CASE("RenderGraph - a memoryless attachment is excluded from aliasing", "[RenderGraph]")
{
    using webgpu::rg::Internal::find_node;
    using webgpu::rg::Internal::ResourceNode;

    TestGraph g;
    ResourceHandle t = g.transient("t");

    g.rg->add_pass(
        "w", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.color(t, 0, WGPULoadOp_Clear, WGPUStoreOp_Discard); // clear+discard -> memoryless
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile(true); // aliasing on: a plain clear+store transient here would be eligible
    REQUIRE(g.rg->get_errors() == nullptr);
    REQUIRE(storage(g.rg)->transientCount == 1); // inferred memoryless
    REQUIRE(storage(g.rg)->m_slotCount == 0); // ... so no alias slot was opened for it
    REQUIRE(find_node(g.rg, t)->aliasSlot == ResourceNode::kNoSlot);
}

// storage_read on a texture (the read side of a storage-texture binding; the suite only ever exercised
// storage_write on a texture). Producer storage_write -> reader storage_read accumulates StorageBinding.
TEST_CASE("RenderGraph - storage_read on a texture accumulates StorageBinding", "[RenderGraph]")
{
    using webgpu::rg::Internal::find_node;

    TestGraph g;
    ResourceHandle tex = g.transient("tex");

    g.rg->add_pass(
        "produce", PassKind::Compute, [&](PassBuilder& b) { b.storage_write(tex); }, [](PassContext&) {});
    g.rg->add_pass(
        "read", PassKind::Compute,
        [&](PassBuilder& b) {
            b.storage_read(tex);
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);
    REQUIRE((find_node(g.rg, tex)->texUsage & WGPUTextureUsage_StorageBinding) != 0);
}

// initialize() with several targets in one pass: end_pass records one InitTarget per call and phase 0
// arms the whole pass while any target is stale. First compile (nothing baked) keeps the 2-target bake.
TEST_CASE("RenderGraph - initialize() records multiple targets in one pass", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle p1 = g.rg->create_persistent_texture("p1", tex2d());
    ResourceHandle p2 = g.rg->create_persistent_texture("p2", tex2d());

    g.rg->add_pass(
        "bake", PassKind::Graphics,
        [&](PassBuilder& b) {
            b.color(p1, 0, WGPULoadOp_Clear, WGPUStoreOp_Store, {}, 0, 0); // MRT: both targets written
            b.color(p2, 1, WGPULoadOp_Clear, WGPUStoreOp_Store, {}, 0, 0);
            b.initialize(p1);
            b.initialize(p2);
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);
    REQUIRE(pass_order(g.rg).size() == 1); // armed on the first frame, not skipped
    REQUIRE(storage(g.rg)->m_passes->initCount == 2); // both targets recorded
}

// Execute-level aliasing payoff: with aliasing on (compile default), two disjoint-lifetime transients
// pack onto one physical slot, so realize() hands both pass bodies the SAME WGPUTexture. The compile
// tests above only check the slot bookkeeping; this proves the shared object reaches execute().
TEST_CASE("RenderGraph exec - disjoint transients realize onto the same physical texture", "[RenderGraph][gpu]")
{
    ExecGraph g;
    WGPUTexture a = nullptr, b = nullptr;

    g.frame([&](RenderGraph* rg) {
        ResourceHandle t1 = rg->create_transient_texture("t1", tex2d());
        ResourceHandle t2 = rg->create_transient_texture("t2", tex2d());
        // t1 lives only in k1, t2 only in k2 -> disjoint intervals -> alias onto one slot
        rg->add_pass(
            "k1", PassKind::Graphics,
            [&](PassBuilder& bb) {
                bb.color(t1, 0, WGPULoadOp_Clear, WGPUStoreOp_Store);
                bb.force_keep();
            },
            [t1, out = &a](PassContext& ctx) { *out = ctx.texture(t1); });
        rg->add_pass(
            "k2", PassKind::Graphics,
            [&](PassBuilder& bb) {
                bb.color(t2, 0, WGPULoadOp_Clear, WGPUStoreOp_Store);
                bb.force_keep();
            },
            [t2, out = &b](PassContext& ctx) { *out = ctx.texture(t2); });
    });

    REQUIRE(a != nullptr);
    REQUIRE(a == b); // both aliased transients got the one pooled physical texture
    REQUIRE(list_count(g.allocator->transient.entries) == 1); // a single object served both
}

// ---------------------------------------------------------------------------------------------------
// PassContext resolvers. The suite above declares passes with empty bodies and asserts on the compiled
// graph. These run bodies and assert on what ctx hands back, which is the half a real renderer touches.

// The declared ViewRange picks the view: two passes read the same texture through different shapes, so
// the cube reader and the plain reader must not be handed the same view. Also pins the resolve_view
// cache: asking twice inside one body returns one object.
TEST_CASE("RenderGraph exec - ctx.view hands back the shape the access declared", "[RenderGraph][gpu]")
{
    ExecGraph g;
    WGPUTextureView cubeView = nullptr, cubeAgain = nullptr, plainView = nullptr;

    g.frame([&](RenderGraph* rg) {
        TextureDesc d = tex2d();
        d.absolute = { 16, 16, 6 }; // 6 layers: samplable as a cube, or as a plain 2D layer
        ResourceHandle env = rg->create_transient_texture("env", d);
        // Each reader renders into its own target: a render pass needs an attachment, and unlike the
        // compile-only cube tests this graph actually executes.
        ResourceHandle outCube = rg->create_transient_texture("outCube", tex2d());
        ResourceHandle outPlain = rg->create_transient_texture("outPlain", tex2d());

        // One layer per pass: six RGBA8 color attachments in a single pass would exceed
        // maxColorAttachmentBytesPerSample.
        static const char* kFaceNames[6] = { "face0", "face1", "face2", "face3", "face4", "face5" };
        for (uint32_t l = 0; l < 6; ++l)
            rg->add_pass(
                kFaceNames[l], PassKind::Graphics,
                [env, l](PassBuilder& b) { b.color(env, 0, WGPULoadOp_Clear, WGPUStoreOp_Store, { 0, 0, 0, 1 }, 0, l); },
                [](PassContext&) {});
        rg->add_pass(
            "readCube", PassKind::Graphics,
            [env, outCube](PassBuilder& b) {
                b.sampled(env, 0, 0, cube());
                b.color(outCube, 0);
                b.force_keep();
            },
            [env, cv = &cubeView, ca = &cubeAgain](PassContext& ctx) {
                *cv = ctx.view(env);
                *ca = ctx.view(env);
            });
        rg->add_pass(
            "readPlain", PassKind::Graphics,
            [env, outPlain](PassBuilder& b) {
                b.sampled(env); // default ViewRange: one layer, 2D
                b.color(outPlain, 0);
                b.force_keep();
            },
            [env, pv = &plainView](PassContext& ctx) { *pv = ctx.view(env); });
    });

    REQUIRE(cubeView != nullptr);
    REQUIRE(plainView != nullptr);
    REQUIRE(cubeView == cubeAgain); // one cached view per shape, not one per call
    REQUIRE(cubeView != plainView); // the declared range, not the texture, decides the view
}

// ctx.view(h, mip, layer) resolves a single subresource, so a blit pass sampling mip N-1 while
// rendering mip N gets two different views of the one texture it declared twice.
TEST_CASE("RenderGraph exec - ctx.view(mip, layer) resolves per subresource", "[RenderGraph][gpu]")
{
    ExecGraph g;
    WGPUTextureView srcMip = nullptr, dstMip = nullptr;

    g.frame([&](RenderGraph* rg) {
        TextureDesc d = tex2d();
        d.mipLevelCount = 4;
        ResourceHandle chain = rg->create_transient_texture("chain", d);

        rg->add_pass(
            "mip0", PassKind::Graphics,
            [chain](PassBuilder& b) { b.color(chain, 0, WGPULoadOp_Clear, WGPUStoreOp_Store, {}, /*baseMip*/ 0); },
            [](PassContext&) {});
        rg->add_pass(
            "mip1", PassKind::Graphics,
            [chain](PassBuilder& b) {
                b.sampled(chain, 0); // read mip 0
                b.color(chain, 0, WGPULoadOp_Clear, WGPUStoreOp_Store, {}, /*baseMip*/ 1); // write mip 1
                b.force_keep();
            },
            [chain, s = &srcMip, dm = &dstMip](PassContext& ctx) {
                *s = ctx.view(chain, 0, 0);
                *dm = ctx.view(chain, 1, 0);
            });
    });

    REQUIRE(srcMip != nullptr);
    REQUIRE(dstMip != nullptr);
    REQUIRE(srcMip != dstMip); // one view per subresource, not one per texture
}

// copy_buffer(size = 0) means "the rest of the src". That expansion happens at declare time, so the
// body sees a resolved, non-zero size with both offsets carried through.
TEST_CASE("RenderGraph exec - ctx.buffer_copy_info reports the resolved range", "[RenderGraph][gpu]")
{
    ExecGraph g;
    PassContext::BufferCopyInfo info {};

    g.frame([&](RenderGraph* rg) {
        BufferDesc bd {};
        bd.size = 64;
        ResourceHandle src = rg->create_transient_buffer("src", bd);
        ResourceHandle dst = rg->create_transient_buffer("dst", bd);

        add_buffer_producer(rg, src);
        rg->add_pass(
            "copy", PassKind::Transfer,
            [src, dst](PassBuilder& b) {
                b.copy_buffer(src, dst, /*srcOffset*/ 16, /*dstOffset*/ 8, /*size*/ 0); // -> 64 - 16 = 48
                b.force_keep();
            },
            [src, dst, out = &info](PassContext& ctx) { *out = ctx.buffer_copy_info(src, dst); });
    });

    REQUIRE(info.size == 48); // expanded against the src, not left at 0
    REQUIRE(info.srcOffset == 16);
    REQUIRE(info.dstOffset == 8);
    REQUIRE(info.src != nullptr);
    REQUIRE(info.dst != nullptr);
    REQUIRE(info.src != info.dst);
}

// A mip level halves a 3D texture's depth but never an array's layer count. Both handles are imported
// with a null WGPUTexture and never accessed by a pass: texture_size answers from the declaration
// alone, so nothing here needs the resources realized (an untouched resource is skipped by realize()).
TEST_CASE("RenderGraph exec - ctx.texture_size shifts 3D depth but not array layers", "[RenderGraph][gpu]")
{
    ExecGraph g;
    WGPUExtent3D vol0 {}, vol1 {}, arr0 {}, arr1 {};

    g.frame([&](RenderGraph* rg) {
        ResourceHandle vol = rg->import_texture("vol", (WGPUTextureView)0x1, { 16, 16, 8 },
            WGPUTextureFormat_RGBA8Unorm, nullptr, /*mipCount*/ 4, 1, WGPUTextureDimension_3D);
        ResourceHandle arr = rg->import_texture("arr", (WGPUTextureView)0x1, { 16, 16, 8 },
            WGPUTextureFormat_RGBA8Unorm, nullptr, /*mipCount*/ 4, 1, WGPUTextureDimension_2D);

        ResourceHandle t = rg->create_transient_texture("probe", tex2d());
        rg->add_pass(
            "probe", PassKind::Graphics,
            [t](PassBuilder& b) {
                b.color(t, 0);
                b.force_keep();
            },
            [vol, arr, v0 = &vol0, v1 = &vol1, a0 = &arr0, a1 = &arr1](PassContext& ctx) {
                *v0 = ctx.texture_size(vol, 0);
                *v1 = ctx.texture_size(vol, 1);
                *a0 = ctx.texture_size(arr, 0);
                *a1 = ctx.texture_size(arr, 1);
            });
    });

    REQUIRE(vol0.depthOrArrayLayers == 8);
    REQUIRE(vol1.width == 8);
    REQUIRE(vol1.height == 8);
    REQUIRE(vol1.depthOrArrayLayers == 4); // 3D: depth is a spatial axis, it halves

    REQUIRE(arr0.depthOrArrayLayers == 8);
    REQUIRE(arr1.width == 8);
    REQUIRE(arr1.height == 8);
    REQUIRE(arr1.depthOrArrayLayers == 8); // 2D array: every layer keeps its own mip chain
}

// host_write + ctx.queue: a pass body uploads through the forwarded queue, a later copy moves it into an
// imported mappable buffer, and the host reads the bytes back. Nothing else in the suite runs a
// host_write body or touches ctx.queue.
TEST_CASE("RenderGraph exec - a host_write body uploads through ctx.queue", "[RenderGraph][gpu]")
{
    constexpr uint32_t kMagic = 0xABCD1234u;

    ExecGraph g;
    webgpu::raii::RawBuffer<uint8_t> readback(g.gpu.device, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead, 4, "hostwrite readback");

    g.frame([&](RenderGraph* rg) {
        BufferDesc bd {};
        bd.size = 4;
        ResourceHandle staged = rg->create_transient_buffer("staged", bd);
        ResourceHandle out = rg->import_buffer("hostwrite.readback", readback.handle());

        rg->add_pass(
            "upload", PassKind::Transfer, [staged](PassBuilder& b) { b.host_write(staged); },
            [staged, magic = kMagic](PassContext& ctx) {
                wgpuQueueWriteBuffer(ctx.queue, ctx.buffer(staged), 0, &magic, sizeof(magic));
            });
        rg->add_pass(
            "copy", PassKind::Transfer,
            [staged, out](PassBuilder& b) { b.copy_buffer(staged, out, 0, 0, 4); },
            [staged, out](PassContext& ctx) {
                auto info = ctx.buffer_copy_info(staged, out);
                wgpuCommandEncoderCopyBufferToBuffer(ctx.encoder, info.src, info.srcOffset, info.dst, info.dstOffset, info.size);
            });
    });

    std::vector<uint8_t> bytes;
    REQUIRE(readback.read_back_sync(g.gpu.instance, g.gpu.device, bytes) == WGPUMapAsyncStatus_Success);
    REQUIRE(bytes.size() == 4);
    uint32_t got = 0;
    std::memcpy(&got, bytes.data(), sizeof(got));
    REQUIRE(got == kMagic); // the queue write landed and the copy carried it
}
