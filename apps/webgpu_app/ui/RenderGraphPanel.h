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

namespace webgpu {
struct RenderGraph;
}

namespace webgpu_app {

// Debug view of a compiled RenderGraph: dependency DAG, resource lifetimes, transient pool and per-pass
// GPU timings
class RenderGraphPanel : public ImGuiPanel {
public:

    void set_graph(webgpu::RenderGraph* graph) { m_graph = graph; }

    // Standalone window, drawn outside the sidebar block.
    void draw() override;

private:
    webgpu::RenderGraph* m_graph = nullptr;
};

} // namespace webgpu_app
