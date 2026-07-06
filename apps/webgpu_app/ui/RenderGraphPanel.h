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

#pragma once

#include "ImGuiPanel.h"

namespace RG {
struct RenderGraph;
}

namespace webgpu_app {

// Debug view of a compiled RenderGraph: dependency DAG, resource lifetimes, transient pool and per-pass
// GPU timings. Reads the graph's internal node structs via RenderGraph_internal.h, so it lives in the app
// where ImGui is available rather than in the imgui-free base library.
class RenderGraphPanel : public ImGuiPanel {
public:
    // The graph is rebuilt every frame and its nodes live in a per-frame arena. Hand the panel the live
    // graph after execute() and before end_frame() (which resets the arena and invalidates the nodes).
    // Passing nullptr hides the window.
    void set_graph(RG::RenderGraph* graph) { m_graph = graph; }

    // Standalone window, drawn outside the sidebar block.
    void draw() override;

private:
    RG::RenderGraph* m_graph = nullptr;
};

} // namespace webgpu_app
