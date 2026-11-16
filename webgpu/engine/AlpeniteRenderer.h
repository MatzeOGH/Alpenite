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
#include "nucleus/event_parameter.h"
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

// Orbit camera in ECEF
struct AlpeniteCamera
{
    double     yaw, pitch; // local ENU frame at the pivot, yaw is computed 
    double     distance;   // distance from the pivot to the camera
    glm::dvec3 centre;     // pivot target in ECEF meters
    double     speed;
};

void ecef_to_geodetic(glm::dvec3 point, double& lat, double& lon, double& height);
void camera_eye(AlpeniteCamera camera, glm::dvec3& out_eye, glm::dvec3& out_fwd, glm::dvec3& out_right, glm::dvec3& out_up);

// keys held for camera controll
struct CameraKeys
{
    bool forward = false, back = false, left = false, right = false, down = false, up = false, fast = false;
};

struct AlpeniteUniform 
{
    glm::vec4  eye;
    glm::vec4  right;
    glm::vec4  up;
    glm::vec4  fwd;
    glm::vec4  proj;     // f / aspect, f, near
    glm::ivec4 cam_cm;   // camera pos in cm
    glm::vec4  cam_frac; // fraction in m from `cam_frac`

    glm::vec4  globe_eye;
};


class AlpeniteRenderer : public QObject {
    Q_OBJECT
public:
    AlpeniteRenderer();
    ~AlpeniteRenderer() override;

    void set_context(Context* context);

    void initialise_gpu();
    void paint(webgpu::rg::RenderGraph* rg, webgpu::rg::TextureHandle target, glm::uvec2 size);
    void advance(double dt);

public slots:
    void mouse_move(nucleus::event_parameter::Mouse m);
    void mouse_press(nucleus::event_parameter::Mouse m);
    void wheel_turn(nucleus::event_parameter::Wheel w);
    void key_press(QKeyCombination key);
    void key_release(QKeyCombination key);

public:
    Context* m_context = nullptr;

    AlpeniteCamera   m_camera;
    CameraKeys       m_keys;
    Qt::MouseButtons m_buttons;
    glm::uvec2       m_viewport{1,1};

    // webgpu camera
    std::unique_ptr<webgpu::Buffer<AlpeniteUniform>> m_camera_config_ubo;
    std::unique_ptr<webgpu::raii::BindGroupLayout> m_camera_bind_group_layout;
    std::unique_ptr<webgpu::raii::BindGroup> m_camera_bind_group;

    std::unique_ptr<webgpu::raii::ShaderModule> m_shader_module;
    std::unique_ptr<webgpu::raii::PipelineLayout> m_pipeline_layout;
    std::unique_ptr<webgpu::raii::RenderPipeline> m_pipeline;

    static constexpr WGPUTextureFormat m_depth_format = WGPUTextureFormat_Depth32Float;
};

} // namespace webgpu_engine
