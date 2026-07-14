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

namespace webgpu::rg {
struct RenderGraph;
}

namespace webgpu_app {

// Debug view of a compiled RenderGraph: dependency DAG, resource lifetimes, transient pool and per-pass
// GPU timings. Also hosts the render-graph path toggle (see docs/rendergraph_status.md).
class RenderGraphPanel : public ImGuiPanel {
public:

    void set_graph(webgpu::rg::RenderGraph* graph) { m_graph = graph; }

    // Standalone window, drawn outside the sidebar block.
    void draw() override;

private:
    webgpu::rg::RenderGraph* m_graph = nullptr;
};

} // namespace webgpu_app
