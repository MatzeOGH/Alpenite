/*****************************************************************************
 * Alpine Renderer
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


#include <catch2/catch_test_macros.hpp>
#include <webgpu/base/RenderGraph.h>

using namespace webgpu;

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

    ResourceHandle transient(ResourceId id)
    {
        TextureDesc desc {};
        desc.dimension = WGPUTextureDimension_2D;
        desc.format = WGPUTextureFormat_RGBA8Unorm;
        desc.absolute = { 16, 16, 1 };
        return rg->create_transient_texture(id, desc);
    }
};

TEST_CASE("RenderGraph - valid graph compiles clean", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle tex = g.transient("tex"_rid);

    g.rg->add_pass(
        "write"_rid, PassKind::Graphics,
        [&](PassBuilder& b) {
            b.color(tex);
            b.force_keep(); // nothing reads tex; keep the pass past culling
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() == nullptr);
}

TEST_CASE("RenderGraph - read before write is reported as an error", "[RenderGraph]")
{
    TestGraph g;
    ResourceHandle tex = g.transient("tex"_rid);

    // reads a transient no pass ever writes -> compile() must poison the graph
    g.rg->add_pass(
        "read"_rid, PassKind::Graphics,
        [&](PassBuilder& b) {
            b.sampled(tex);
            b.force_keep();
        },
        [](PassContext&) {});

    g.rg->compile();
    REQUIRE(g.rg->get_errors() != nullptr);
}
