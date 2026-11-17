/*****************************************************************************
 * Alpenite
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

#include "AlpenitePanel.h"

#include <IconsFontAwesome5.h>
#include <imgui.h>
#include <string>

#include "App.h"
#include "webgpu/engine/AlpeniteRenderer.h"

namespace webgpu_app {

AlpenitePanel::AlpenitePanel(App* app)
    : m_app(app)
{
}

void AlpenitePanel::draw_panel()
{
    if (!m_app->alpenite_mode()) {
        return;
    }

    auto* renderer = m_app->get_alpenite_renderer();
    if (renderer == nullptr) {
        return;
    }

    if (ImGui::CollapsingHeader(ICON_FA_CUBES " Clusters", ImGuiTreeNodeFlags_DefaultOpen)) {
        const int level = renderer->cluster_level();
        const std::string label = "Level " + std::to_string(level);
        if (ImGui::BeginCombo("DAG level", label.c_str())) {
            for (int l = 0; l <= int(webgpu_engine::AlpeniteRenderer::max_cluster_level()); ++l) {
                const bool selected = (l == level);
                const std::string item = "Level " + std::to_string(l) + (l == 0 ? "  (coarsest)" : l == int(webgpu_engine::AlpeniteRenderer::max_cluster_level()) ? "  (finest)" : "");
                if (ImGui::Selectable(item.c_str(), selected)) {
                    renderer->set_cluster_level(uint8_t(l));
                }
                if (selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        bool wireframe = renderer->wireframe_enabled();
        if (ImGui::Checkbox("Wireframe", &wireframe)) {
            renderer->set_wireframe_enabled(wireframe);
        }

        ImGui::TextDisabled("%u triangles", renderer->cluster_triangle_count());
    }
}

} // namespace webgpu_app
