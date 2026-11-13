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
#pragma once


#include "Context.h"
#include "UniformBufferObjects.h"
#include "nucleus/camera/Definition.h"
#include <QObject>
#include <memory>
#include <webgpu/base/Buffer.h>
#include <webgpu/base/raii/BindGroup.h>
#include <webgpu/base/raii/BindGroupLayout.h>
#include <webgpu/base/raii/PipelineLayout.h>
#include <webgpu/base/raii/base_types.h>
#include <webgpu/webgpu.h>

namespace webgpu::rg {
struct RenderGraph;
struct ResourceHandle;
struct TextureHandle;
}

namespace webgpu_engine {

class AlpeniteRenderer : public QObject {
    Q_OBJECT
public:
    AlpeniteRenderer();
    ~AlpeniteRenderer() override;

    void set_context(Context* context);

    void initialise_gpu();
    void paint(webgpu::rg::RenderGraph* rg, webgpu::rg::TextureHandle target, glm::uvec2 size);

public slots:
    void update_camera(const nucleus::camera::Definition& new_definition);

private:
    Context* m_context = nullptr;

    nucleus::camera::Definition m_camera;

    std::unique_ptr<webgpu::Buffer<uboCameraConfig>> m_camera_config_ubo;
    std::unique_ptr<webgpu::raii::BindGroupLayout> m_camera_bind_group_layout;
    std::unique_ptr<webgpu::raii::BindGroup> m_camera_bind_group;

    std::unique_ptr<webgpu::raii::ShaderModule> m_shader_module;
    std::unique_ptr<webgpu::raii::PipelineLayout> m_pipeline_layout;
    std::unique_ptr<webgpu::raii::RenderPipeline> m_pipeline;

    static constexpr WGPUTextureFormat m_depth_format = WGPUTextureFormat_Depth24Plus;
};

} // namespace webgpu_engine
