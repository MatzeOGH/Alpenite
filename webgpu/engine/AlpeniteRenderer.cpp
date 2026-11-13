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

#include "AlpeniteRenderer.h"

#include "webgpu/base/RenderGraph.h"
#include <QtAssert>
#include <array>
#include <vector>

namespace webgpu_engine {

namespace {
    constexpr const char* kGridShader = R"(
struct CameraConfig {
    position                : vec4<f32>,
    view_matrix             : mat4x4<f32>,
    proj_matrix             : mat4x4<f32>,
    view_proj_matrix        : mat4x4<f32>,
    inv_view_proj_matrix    : mat4x4<f32>,
    inv_view_matrix         : mat4x4<f32>,
    inv_proj_matrix         : mat4x4<f32>,
    viewport_size           : vec2<f32>,
    distance_scaling_factor : f32,
    buffer2                 : f32,
}

@group(0) @binding(0) var<uniform> camera : CameraConfig;

struct VsOut {
    @builtin(position) clip     : vec4<f32>,
    @location(0)       world_xy : vec2<f32>,
}

const CELL   : f32 = 1000.0;   // grid spacing in metres
const EXTENT : f32 = 20000.0;  // half-size of the ground quad in metres


@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> VsOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0,  1.0), vec2<f32>(-1.0, 1.0),
    );
    let c        = corners[vi];
    let center   = floor(camera.position.xy / CELL) * CELL;
    let world_xy = center + c * EXTENT;
    let cam_rel  = vec3<f32>(world_xy - camera.position.xy, 0.0 - camera.position.z);

    var out : VsOut;
    out.clip     = camera.view_proj_matrix * vec4<f32>(cam_rel, 1.0);
    out.world_xy = world_xy;
    return out;
}

@fragment
fn fs_main(in : VsOut) -> @location(0) vec4<f32> {
    let coord = in.world_xy / CELL;
    let g     = abs(fract(coord - 0.5) - 0.5) / fwidth(coord);
    let line  = min(g.x, g.y);
    let grid  = 1.0 - min(line, 1.0);

    let base_col = vec3<f32>(0.10, 0.12, 0.16);
    let line_col = vec3<f32>(0.85, 0.90, 1.00);
    return vec4<f32>(mix(base_col, line_col, grid), 1.0);
}
    )";

    WGPUStringView str(const char* s) { return WGPUStringView { .data = s, .length = WGPU_STRLEN }; }
} // namespace

AlpeniteRenderer::AlpeniteRenderer() = default;
AlpeniteRenderer::~AlpeniteRenderer() = default;

void AlpeniteRenderer::set_context(Context* context) { m_context = context; }

void AlpeniteRenderer::initialise_gpu()
{
    Q_ASSERT(m_context != nullptr);
    WGPUDevice device = m_context->webgpu_ctx().device();

    m_camera_config_ubo = std::make_unique<webgpu::Buffer<uboCameraConfig>>(device, WGPUBufferUsage(WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform));

    WGPUBindGroupLayoutEntry camera_entry {};
    camera_entry.binding = 0;
    camera_entry.visibility = WGPUShaderStage_Vertex;
    camera_entry.buffer.type = WGPUBufferBindingType_Uniform;
    camera_entry.buffer.minBindingSize = sizeof(uboCameraConfig);
    m_camera_bind_group_layout = std::make_unique<webgpu::raii::BindGroupLayout>(
        device, std::vector<WGPUBindGroupLayoutEntry> { camera_entry }, "alpenite camera layout");

    m_camera_bind_group = std::make_unique<webgpu::raii::BindGroup>(device, *m_camera_bind_group_layout,
        std::initializer_list<WGPUBindGroupEntry> { m_camera_config_ubo->raw_buffer().create_bind_group_entry(0) }, "alpenite camera bind group");

    m_shader_module = m_context->webgpu_ctx().resource_registry().compile_shader_from_code(device, kGridShader, "AlpeniteRenderer::grid");

    m_pipeline_layout = std::make_unique<webgpu::raii::PipelineLayout>(
        device, std::vector<WGPUBindGroupLayout> { m_camera_bind_group_layout->handle() }, "alpenite pipeline layout");

    WGPUColorTargetState color_target {};
    color_target.format = m_context->webgpu_ctx().surface_texture_format();
    color_target.writeMask = WGPUColorWriteMask_All;

    WGPUFragmentState fragment {};
    fragment.module = m_shader_module->handle();
    fragment.entryPoint = str("fs_main");
    fragment.targetCount = 1;
    fragment.targets = &color_target;

    WGPUStencilFaceState stencil_face {};
    stencil_face.compare = WGPUCompareFunction_Always;
    stencil_face.failOp = WGPUStencilOperation_Keep;
    stencil_face.depthFailOp = WGPUStencilOperation_Keep;
    stencil_face.passOp = WGPUStencilOperation_Keep;

    WGPUDepthStencilState depth_stencil {};
    depth_stencil.format = m_depth_format;
    depth_stencil.depthWriteEnabled = WGPUOptionalBool_True;
    depth_stencil.depthCompare = WGPUCompareFunction_GreaterEqual;
    depth_stencil.stencilReadMask = 0;
    depth_stencil.stencilWriteMask = 0;
    depth_stencil.stencilFront = stencil_face;
    depth_stencil.stencilBack = stencil_face;

    WGPURenderPipelineDescriptor desc {};
    desc.label = str("AlpeniteRenderer::grid");
    desc.layout = m_pipeline_layout->handle();
    desc.vertex.module = m_shader_module->handle();
    desc.vertex.entryPoint = str("vs_main");
    desc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    desc.primitive.frontFace = WGPUFrontFace_CCW;
    desc.primitive.cullMode = WGPUCullMode_None;
    desc.depthStencil = &depth_stencil;
    desc.multisample.count = 1;
    desc.multisample.mask = 0xFFFFFFFFu;
    desc.fragment = &fragment;

    m_pipeline = std::make_unique<webgpu::raii::RenderPipeline>(device, desc);
}

void AlpeniteRenderer::update_camera(const nucleus::camera::Definition& new_definition)
{
    m_camera = new_definition;
    if (!m_camera_config_ubo)
        return;

    uboCameraConfig* cc = &m_camera_config_ubo->data;
    cc->position = glm::vec4(new_definition.position(), 1.0);
    cc->view_matrix = new_definition.local_view_matrix();
    cc->proj_matrix = new_definition.projection_matrix();
    cc->view_proj_matrix = cc->proj_matrix * cc->view_matrix;
    cc->inv_view_proj_matrix = glm::inverse(cc->view_proj_matrix);
    cc->inv_view_matrix = glm::inverse(cc->view_matrix);
    cc->inv_proj_matrix = glm::inverse(cc->proj_matrix);
    cc->viewport_size = new_definition.viewport_size();
    cc->distance_scaling_factor = new_definition.distance_scale_factor();
    m_camera_config_ubo->update_gpu_data(m_context->webgpu_ctx().queue());
}

void AlpeniteRenderer::paint(webgpu::rg::RenderGraph* rg, webgpu::rg::TextureHandle target, [[maybe_unused]] glm::uvec2 size)
{
    auto depth = rg->create_transient_texture("alpenite.depth", webgpu::rg::texture_relative(m_depth_format, target, 1.0f));

    rg->add_pass("Alpenite", webgpu::rg::PassKind::Graphics,
        [target, depth](webgpu::rg::PassBuilder& b) {
            b.color(target, 0, { .clear = { 0.02, 0.02, 0.05, 1.0 } });
            b.depth_stencil(depth, { .clearDepth = 0.0f }); // reverse-Z
        },
        [this](webgpu::rg::PassContext& c) {
            wgpuRenderPassEncoderSetPipeline(c.render_pass, m_pipeline->handle());
            wgpuRenderPassEncoderSetBindGroup(c.render_pass, 0, m_camera_bind_group->handle(), 0, nullptr);
            wgpuRenderPassEncoderDraw(c.render_pass, 6, 1, 0, 0);
        });
}

} // namespace webgpu_engine
