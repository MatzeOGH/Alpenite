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
#include <QDebug>
#include <QtAssert>
#include <array>
#include <glm/vec3.hpp>
#include <vector>

namespace webgpu_engine {

namespace {
    constexpr const char* kGlobeShader = R"(
struct Cam {
    eye       : vec4<f32>,
    right     : vec4<f32>,
    up        : vec4<f32>,
    fwd       : vec4<f32>,
    proj      : vec4<f32>,   // f / aspect, f, near, unused
    cam_cm    : vec4<i32>,
    cam_frac  : vec4<f32>,
    globe_eye : vec4<f32>,   // xyz: eye / (A, A, B), w: |eye / (A, A, B)|^2 - 1, from f64
}
@group(0) @binding(0) var<uniform> u : Cam;

struct VsOut {
    @builtin(position) clip : vec4<f32>,
    @location(0)       ndc  : vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> VsOut {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var out : VsOut;
    out.clip = vec4<f32>(p[vi], 0.0, 1.0);
    out.ndc  = p[vi];
    return out;
}

const RADII      : vec3<f32> = vec3<f32>(6378137.0, 6378137.0, 6356752.314245);
const SUN        : vec3<f32> = vec3<f32>(0.577, 0.577, 0.577);
const ATMOSPHERE : f32 = 0.012;

struct FsOut {
    @location(0)         color : vec4<f32>,
    @builtin(frag_depth) depth : f32,
}

fn level_fade(cells_per_px : f32) -> f32 {
    return 1.0 - smoothstep(0.03, 0.15, cells_per_px);
}

@fragment
fn fs_main(in : VsOut) -> FsOut {
    let dir = u.fwd.xyz + u.right.xyz * (in.ndc.x / u.proj.x) + u.up.xyz * (in.ndc.y / u.proj.y);

    let o = u.globe_eye.xyz;
    let d = dir / RADII;
    let a = dot(d, d);
    let b = dot(o, d);
    let c = u.globe_eye.w;
    let disc = b * b - a * c;
    let hit = c > 0.0 && b < 0.0 && disc >= 0.0;

    let t = select(1.0, c / (-b + sqrt(max(disc, 0.0))), hit);
    let n = normalize((o + d * t) / RADII);   // geodetic normal at the hit

    let lat_deg  = degrees(asin(clamp(n.z, -1.0, 1.0)));
    let lon_turn = atan2(n.y, n.x) / 6.2831853;   // (-0.5, 0.5]
    let lon_deg  = lon_turn * 360.0;

    let dlon = min(fwidth(lon_turn), fwidth(fract(lon_turn))) * 360.0;
    let dlat = fwidth(lat_deg);

    var grid = 0.0;
    var size = 10.0;
    for (var k = 0; k < 6; k++) {
        let g  = vec2<f32>(lon_deg, lat_deg) / size;
        let fw = max(vec2<f32>(dlon, dlat) / size, vec2<f32>(1e-6));
        let e  = abs(fract(g + 0.5) - 0.5) / fw;

        let lx = (1.0 - min(e.x, 1.0)) * level_fade(fw.x);
        let ly = (1.0 - min(e.y, 1.0)) * level_fade(fw.y);

        let major = mix(0.45, 1.0, level_fade(fw.y * 10.0));
        grid = max(grid, max(lx, ly) * major);
        size = size * 0.1;
    }

    let shade  = 0.35 + 0.65 * max(dot(n, -normalize(dir)), 0.0);
    let ground = mix(vec3<f32>(0.10, 0.12, 0.16), vec3<f32>(0.85, 0.90, 1.00), grid) * shade;

    let r2   = select(c + 1.0, c + 1.0 - b * b / a, b < 0.0);
    let glow = exp(-max(sqrt(max(r2, 0.0)) - 1.0, 0.0) / ATMOSPHERE);
    let sky  = vec3<f32>(0.25, 0.5, 1.0) * glow + vec3<f32>(0.01, 0.01, 0.03);

    var out : FsOut;
    out.color = vec4<f32>(select(sky, ground, hit), 1.0);
    out.depth = select(0.0, min(u.proj.z / t, 1.0), hit);
    return out;
}
)";

    constexpr const char* kClusterShader = R"(
struct Cam {
    eye       : vec4<f32>,
    right     : vec4<f32>,
    up        : vec4<f32>,
    fwd       : vec4<f32>,
    proj      : vec4<f32>,
    cam_cm    : vec4<i32>,
    cam_frac  : vec4<f32>,
    globe_eye : vec4<f32>,
    misc      : vec4<f32>,   // x: wireframe on/off
}
@group(0) @binding(0) var<uniform> u : Cam;

struct Vtx {
    pos_cm     : vec3<i32>,
    cluster_id : u32,
}
@group(1) @binding(0) var<storage, read> verts : array<Vtx>;

fn to_rel(pos_cm : vec3<i32>) -> vec3<f32> {
    return vec3<f32>(pos_cm - u.cam_cm.xyz) * 0.01 - u.cam_frac.xyz;
}

fn to_clip(rel : vec3<f32>) -> vec4<f32> {
    let vx = dot(rel, u.right.xyz);
    let vy = dot(rel, u.up.xyz);
    let vz = dot(rel, u.fwd.xyz);
    return vec4<f32>(vx * u.proj.x, vy * u.proj.y, u.proj.z, vz);
}

// cluster id to color
fn cluster_color(i : u32) -> vec3<f32> {
    let h = fract(f32(i) * 0.6180339887);
    let r = abs(h * 6.0 - 3.0) - 1.0;
    let g = 2.0 - abs(h * 6.0 - 2.0);
    let b = 2.0 - abs(h * 6.0 - 4.0);
    let rgb = clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
    return mix(vec3<f32>(1.0), rgb, 0.8);
}

struct FillOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) rel  : vec3<f32>,
    @location(1) @interpolate(flat) cid : u32,
    @location(2) bary : vec3<f32>,
}

@vertex
fn vs_fill(@builtin(vertex_index) vi : u32) -> FillOut {
    let v = verts[vi];
    let rel = to_rel(v.pos_cm);
    var out : FillOut;
    out.clip = to_clip(rel);
    out.rel  = rel;
    out.cid  = v.cluster_id;
    let k = vi % 3u;
    out.bary = vec3<f32>(f32(k == 0u), f32(k == 1u), f32(k == 2u));
    return out;
}

@fragment
fn fs_fill(in : FillOut) -> @location(0) vec4<f32> {

    let n = normalize(cross(dpdx(in.rel), dpdy(in.rel)));
    let sun = normalize(vec3<f32>(0.4, 0.3, 0.85));
    let l = 0.45 + 0.55 * abs(dot(n, sun));
    var col = cluster_color(in.cid) * l;

    // barycentric wireframe
    if (u.misc.x > 0.5) {
        let d = fwidth(in.bary);
        let a = smoothstep(vec3<f32>(0.0), d * 1.2, in.bary);
        col = mix(vec3<f32>(0.0), col, min(a.x, min(a.y, a.z)) * 0.7 + 0.3);
    }
    return vec4<f32>(col, 1.0);
}
)";

    WGPUStringView str(const char* s) { return WGPUStringView { .data = s, .length = WGPU_STRLEN }; }
} // namespace


constexpr double WGS84_A = 6378137.0;
constexpr double WGS84_B = 6356752.314245;
constexpr double WGS84_A_SQ = WGS84_A * WGS84_A;
constexpr double WGS84_B_SQ = WGS84_B * WGS84_B;

constexpr double FOV      = 75.0 * 3.14159265358979323846 / 180.0; 
constexpr double MAX_DIST = 8e6;  // max camera distance in meters


// ECEF to geodetic latitude, longitude and height
// Bowring (1976) seed refined by Heiskanen–Moritz (1967) iteration.
void ecef_to_geodetic(glm::dvec3 point, double& lat, double& lon, double& height)
{
    constexpr double e2 = 1.0 - WGS84_B_SQ / WGS84_A_SQ;  // first eccentricity²
    constexpr double ep2 = WGS84_A_SQ / WGS84_B_SQ - 1.0; // second eccentricity²

    double r = glm::sqrt(point.x * point.x + point.y * point.y);
    lon = glm::atan(point.y, point.x);
    double th = glm::atan(point.z * WGS84_A, r * WGS84_B);
    double th_sin = glm::sin(th);
    double th_cos = glm::cos(th);
    lat = glm::atan(point.z + ep2 * WGS84_B * th_sin * th_sin * th_sin,
                    r - e2 * WGS84_A * th_cos * th_cos * th_cos);

    for (int i = 0; i < 3; i++)
    {
        double lat_sin = glm::sin(lat);
        double n = WGS84_A /glm::sqrt(1.0 - e2 * lat_sin * lat_sin);
        height = r / glm::cos(lat) - n;
        lat = glm::atan(point.z, r * (1 - e2 * n / (n + height)));
    }

    double lat_sin = glm::sin(lat);
    double n = WGS84_A / glm::sqrt(1.0 - e2 * lat_sin * lat_sin);
    height = glm::abs(lat) < 1.5 ? r / glm::cos(lat) - n 
                                 : point.z / glm::sin(lat) - n * ( 1 - e2);
}

// geodetic to ECEF
// https://arxiv.org/pdf/2405.05352
static glm::dvec3 geodetic_to_ecef(double lat, double lon, double height)
{
    constexpr double e2 = 1.0 - WGS84_B_SQ / WGS84_A_SQ;  // first eccentricity²
    double s = glm::sin(lat);
    double n = WGS84_A / glm::sqrt(1.0 - e2 * s * s);
    return { (n + height) * glm::cos(lat) * glm::cos(lon),
             (n + height) * glm::cos(lat) * glm::sin(lon),
             (n * (1.0 - e2) + height) * s};
}


// geodetic normal of an ECEF point
static glm::dvec3 ecef_up(glm::dvec3 point) 
{
    glm::dvec3 n {
        point.x / (WGS84_A_SQ),
        point.y / (WGS84_A_SQ),
        point.z / (WGS84_B_SQ)
    };

    return glm::normalize(n);
}

// local east north up at an ECEF point
// input: ecef point
// output: east, north & up 
static void local_enu( glm::dvec3 point, glm::dvec3& east, glm::dvec3& north, glm::dvec3& up)
{
    up = ecef_up(point);
    east = glm::cross(glm::dvec3(0.0, 0.0, 1.0), up);
    if (glm::length(east) < 1e-9) // at the pole
    {
        east = { 0, 1, 0};
    }
    east = glm::normalize(east);
    north = glm::cross(up, east);
}

// Cameras north and up follow the globe
// input: `AlpeniteCamera`
// output: `out_eye`   vector from the camera to the pivot time `distance`
//         `out_fwd`   points from the camera to the pivot
//         `out_right` points along the horizon 
//         `out_up`    camera's screen up vector 
void camera_eye(AlpeniteCamera camera, glm::dvec3& out_eye, glm::dvec3& out_fwd, glm::dvec3& out_right, glm::dvec3& out_up) {
    glm::dvec3 east, north, up;
    local_enu(camera.centre, east, north, up);

    double cp = glm::cos(camera.pitch);
    double sp = glm::sin(camera.pitch);
    double cy = glm::cos(camera.yaw);
    double sy = glm::sin(camera.yaw);

    out_fwd = -(east * cy + north * sy) * cp - up * sp;
    out_right = -east * sy + north * cy;
    out_up = glm::cross(out_right, out_fwd);

    out_eye = camera.centre - out_fwd * camera.distance;
}

// orbit camera
// input: `dx`
//        `dy`
static void camera_orbit(AlpeniteCamera& camera, double dx, double dy) 
{
    constexpr double scale = 0.004;
    camera.yaw -= dx * scale;
    camera.pitch = glm::clamp(camera.pitch + dy * scale, glm::min(-0.2, camera.pitch), 1.55);
}

static void camera_aim(AlpeniteCamera& camera, glm::dvec3 eye, glm::dvec3 fwd)
{
    double lat, lon, height;
    ecef_to_geodetic(eye, lat, lon, height);

    double r = glm::length(geodetic_to_ecef(lat, lon, 0.0));
    double d = glm::clamp(glm::max(glm::abs(height), 20.0) * 10.0, 1.0, MAX_DIST);

    double b = glm::dot(eye, fwd);
    double disc = b * b - ( glm::dot(eye,eye) - r * r);
    if (disc > 0.0)
    {
        double t = -b - glm::sqrt(disc);
        if(t > 1.0) d = glm::min(d, t);
    }
    camera.distance = d;
    camera.centre = eye + fwd * d;

    // update new enu frame
    glm::dvec3 east, north, up;
    local_enu(camera.centre, east, north, up);
    camera.yaw = glm::atan(-glm::dot(fwd, north), -glm::dot(fwd, east));
    camera.pitch = glm::asin(glm::clamp(-glm::dot(fwd, up), -1.0, 1.0));
}

static void camera_look(AlpeniteCamera& camera, double dx, double dy)
{
    glm::dvec3 eye, fwd, right, up;
    camera_eye(camera, eye, fwd, right, up);
    glm::dvec3 e, n, u;
    local_enu(eye, e, n, u);

    double heading = glm::atan(glm::dot(fwd, e), glm::dot(fwd, n)) + dx * 0.003;   // from north
    double elevation = glm::clamp(glm::asin(glm::clamp(glm::dot(fwd, u), -1.0, 1.0)) - dy * 0.003, -1.55, 1.55);
    
    camera_aim(camera, eye, glm::cos(elevation) * (glm::sin(heading) * e + glm::cos(heading) * n) + glm::sin(elevation) * u);
}

static void camera_pan(AlpeniteCamera& camera, double dx, double dy, double viewport_height)
{
    glm::dvec3 eye, fwd, right, up;
    camera_eye(camera, eye, fwd, right, up);

    double lat, lon, height;
    ecef_to_geodetic(camera.centre, lat, lon, height);

    glm::dvec3 ahead = glm::cross(ecef_up(camera.centre), right);   // horizontal, so a steep view does not push the pivot into the ground
    double new_lat, new_lon, new_height;
    double metres_per_px = camera.distance * 2.0 * glm::tan(FOV * 0.5) / viewport_height;
    ecef_to_geodetic(camera.centre + (right * -dx + ahead * dy) * metres_per_px, new_lat, new_lon, new_height);
    camera.centre = geodetic_to_ecef(new_lat, new_lon, height);   // back to the pivot's height, or it leaves the surface
}


static bool* fly_key(CameraKeys& keys, Qt::Key key)
{
    switch (key) {
    case Qt::Key_W:     return &keys.forward;
    case Qt::Key_S:     return &keys.back;
    case Qt::Key_A:     return &keys.left;
    case Qt::Key_D:     return &keys.right;
    case Qt::Key_Q:     return &keys.down;
    case Qt::Key_E:     return &keys.up;
    case Qt::Key_Shift: return &keys.fast;
    default:            return nullptr;
    }
}

AlpeniteRenderer::AlpeniteRenderer()
{
    m_camera.centre   = geodetic_to_ecef(glm::radians(48.2085), glm::radians(16.3731), 170.0);
    m_camera.distance = 1500.0;
    m_camera.yaw      = -1.9;
    m_camera.pitch    = 0.6;
    m_camera.speed    = 1.0;
}

AlpeniteRenderer::~AlpeniteRenderer() = default;

void AlpeniteRenderer::set_context(Context* context) { m_context = context; }

void AlpeniteRenderer::initialise_gpu()
{
    Q_ASSERT(m_context != nullptr);
    WGPUDevice device = m_context->webgpu_ctx().device();

    m_camera_config_ubo = std::make_unique<webgpu::Buffer<AlpeniteUniform>>(device, WGPUBufferUsage(WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform));

    WGPUBindGroupLayoutEntry camera_entry {
        .binding = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer = {
            .type = WGPUBufferBindingType_Uniform,
            .minBindingSize = sizeof(AlpeniteUniform),
        },
    };
    m_camera_bind_group_layout = std::make_unique<webgpu::raii::BindGroupLayout>(
        device, std::vector<WGPUBindGroupLayoutEntry> { camera_entry }, "alpenite camera layout");

    m_camera_bind_group = std::make_unique<webgpu::raii::BindGroup>(device, *m_camera_bind_group_layout,
        std::initializer_list<WGPUBindGroupEntry> { m_camera_config_ubo->raw_buffer().create_bind_group_entry(0) }, "alpenite camera bind group");

    m_shader_module = m_context->webgpu_ctx().resource_registry().compile_shader_from_code(device, kGlobeShader, "AlpeniteRenderer::grid");

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

    m_mesh_shader_module = m_context->webgpu_ctx().resource_registry().compile_shader_from_code(device, kClusterShader, "AlpeniteRenderer::cluster");

    WGPUBindGroupLayoutEntry vertex_storage_entry {};
    vertex_storage_entry.binding = 0;
    vertex_storage_entry.visibility = WGPUShaderStage_Vertex;
    vertex_storage_entry.buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    m_mesh_vertex_bind_group_layout = std::make_unique<webgpu::raii::BindGroupLayout>(
        device, std::vector<WGPUBindGroupLayoutEntry> { vertex_storage_entry }, "alpenite mesh vertex storage layout");

    m_mesh_pipeline_layout = std::make_unique<webgpu::raii::PipelineLayout>(
        device, std::vector<WGPUBindGroupLayout> { m_camera_bind_group_layout->handle(), m_mesh_vertex_bind_group_layout->handle() },
        "alpenite mesh pipeline layout");

    WGPUColorTargetState mesh_color {};
    mesh_color.format = m_context->webgpu_ctx().surface_texture_format();
    mesh_color.writeMask = WGPUColorWriteMask_All;

    WGPUFragmentState mesh_fragment {};
    mesh_fragment.module = m_mesh_shader_module->handle();
    mesh_fragment.entryPoint = str("fs_fill");
    mesh_fragment.targetCount = 1;
    mesh_fragment.targets = &mesh_color;

    WGPURenderPipelineDescriptor mesh_desc {};
    mesh_desc.label = str("AlpeniteRenderer::cluster_fill");
    mesh_desc.layout = m_mesh_pipeline_layout->handle();
    mesh_desc.vertex.module = m_mesh_shader_module->handle();
    mesh_desc.vertex.entryPoint = str("vs_fill");
    mesh_desc.vertex.bufferCount = 0;
    mesh_desc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    mesh_desc.primitive.frontFace = WGPUFrontFace_CCW;
    mesh_desc.primitive.cullMode = WGPUCullMode_None;
    mesh_desc.depthStencil = &depth_stencil;
    mesh_desc.multisample.count = 1;
    mesh_desc.multisample.mask = 0xFFFFFFFFu;
    mesh_desc.fragment = &mesh_fragment;

    m_mesh_pipeline = std::make_unique<webgpu::raii::RenderPipeline>(device, mesh_desc);
    
    m_dag_dir = R"(C:\Users\matth\Downloads\stephansdom-2026-11-10\dag)";
    try {
        load_cluster_level(m_dag_dir, m_level);
    } catch (const std::exception& e) {
        qWarning() << "alpenite: failed to load clusters:" << e.what();
    }
}

void AlpeniteRenderer::load_cluster_level(const std::filesystem::path& dag_dir, uint8_t level)
{
    const alp_load::Mesh mesh = alp_load::load_level(dag_dir, level);

    m_mesh_vbo.reset();
    m_mesh_vertex_bind_group.reset();
    m_mesh_vertex_count = 0;
    if (mesh.vertices.empty()) {
        qWarning() << "alpenite: dag level" << level << "produced no geometry";
        return;
    }

    WGPUDevice device = m_context->webgpu_ctx().device();
    WGPUQueue  queue  = m_context->webgpu_ctx().queue();

    m_mesh_vbo = std::make_unique<webgpu::raii::RawBuffer<alp_load::Vertex>>(
        device, WGPUBufferUsage(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst), mesh.vertices.size(), "alpenite cluster vbo");
    m_mesh_vbo->write(queue, mesh.vertices.data(), mesh.vertices.size());
    m_mesh_vertex_count = uint32_t(mesh.vertices.size());

    m_mesh_vertex_bind_group = std::make_unique<webgpu::raii::BindGroup>(device, *m_mesh_vertex_bind_group_layout,
        std::initializer_list<WGPUBindGroupEntry> { m_mesh_vbo->create_bind_group_entry(0) }, "alpenite mesh vertex bind group");

    qInfo() << "alpenite: loaded dag level" << level << "-" << m_mesh_vertex_count << "verts," << (m_mesh_vertex_count / 3) << "tris";
}

void AlpeniteRenderer::paint(webgpu::rg::RenderGraph* rg, webgpu::rg::TextureHandle target, [[maybe_unused]] glm::uvec2 size)
{
    m_viewport = size;

    if (m_reload_pending) {
        m_reload_pending = false;
        try {
            load_cluster_level(m_dag_dir, m_level);
        } catch (const std::exception& e) {
            qWarning() << "alpenite: reload failed:" << e.what();
        }
    }

    // update alpenite uniform
    auto& uniform = m_camera_config_ubo->data;
    glm::dvec3 eye, fwd, right, up;
    camera_eye(m_camera, eye, fwd, right, up);

    double near = glm::max(m_camera.distance * 0.002, 0.05);
    double f = 1.0 / glm::tan(glm::radians(75.0) * 0.5); // field of view

    for ( int i = 0; i < 3; ++i) 
    {
        // split into cm and fract part
        double cm = glm::floor(eye[i] * 100.0);
        uniform.cam_cm[i] = int32_t(cm);
        uniform.cam_frac[i] = float(eye[i] - cm*0.01);
    }

    uniform.eye = glm::vec4(glm::vec3(eye), 1.0);
    uniform.right = glm::vec4(glm::vec3(right), 1.0);
    uniform.up = glm::vec4(glm::vec3(up), 1.0);
    uniform.fwd = glm::vec4(glm::vec3(fwd), 1.0);
    uniform.proj = glm::vec4(float ( f * size.y / size.x), float(f), float(near), 0.0f);

    glm::dvec3 eye_unit = eye / glm::dvec3(WGS84_A, WGS84_A, WGS84_B);
    uniform.globe_eye = glm::vec4(glm::vec3(eye_unit), float(glm::dot(eye_unit, eye_unit) - 1.0));

    uniform.misc = glm::vec4(m_wireframe_enabled ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f);

    m_camera_config_ubo->update_gpu_data(m_context->webgpu_ctx().queue());



    auto depth = rg->create_transient_texture("alpenite.depth", webgpu::rg::texture_relative(m_depth_format, target, 1.0f));

    rg->add_pass("Alpenite", webgpu::rg::PassKind::Graphics,
        [target, depth](webgpu::rg::PassBuilder& b) {
            b.color(target, 0, { .clear = { 0.02, 0.02, 0.05, 1.0 } });
            b.depth_stencil(depth, { .clearDepth = 0.0f }); // reverse-Z
        },
        [this](webgpu::rg::PassContext& c) {
            wgpuRenderPassEncoderSetPipeline(c.render_pass, m_pipeline->handle());
            wgpuRenderPassEncoderSetBindGroup(c.render_pass, 0, m_camera_bind_group->handle(), 0, nullptr);
            wgpuRenderPassEncoderDraw(c.render_pass, 3, 1, 0, 0);

            // clusters: vertex pulling, so no vertex buffer, just the storage bind group
            if (m_mesh_pipeline && m_mesh_vertex_bind_group && m_mesh_vertex_count > 0) {
                wgpuRenderPassEncoderSetPipeline(c.render_pass, m_mesh_pipeline->handle());
                wgpuRenderPassEncoderSetBindGroup(c.render_pass, 0, m_camera_bind_group->handle(), 0, nullptr);
                wgpuRenderPassEncoderSetBindGroup(c.render_pass, 1, m_mesh_vertex_bind_group->handle(), 0, nullptr);
                wgpuRenderPassEncoderDraw(c.render_pass, m_mesh_vertex_count, 1, 0, 0);
            }
        });
}

void AlpeniteRenderer::advance(double dt) 
{
    double move_fwd   = (double(m_keys.forward) - double(m_keys.back));
    double move_right = (double(m_keys.right) - double(m_keys.left));
    double move_up    = (double(m_keys.up)- double(m_keys.down));

    if( glm::dvec3(move_fwd, move_right, move_up) == glm::dvec3(0.0)) return;


    glm::dvec3 eye, fwd, right, up;
    camera_eye(m_camera, eye, fwd, right, up);

    double lat, lon, alt;
    ecef_to_geodetic(eye, lat, lon, alt);

    double a = glm::max(glm::abs(alt), 20.0);
    if (a > 1000.0)
        a = 1000.0 * glm::pow(a / 1000.0, 0.75);
    
    double step = a * 0.5 * m_camera.speed * ( m_keys.fast ? 10.0 : 1.0) * glm::min(dt, 0.1);
    glm::dvec3 new_eye = eye + (fwd * move_fwd + ecef_up(eye) * move_up) * step;

    if (move_right != 0)
    {
        glm::dvec3 axis = glm::normalize(glm::cross(new_eye, right));
        glm::dquat turn = glm::angleAxis(move_right * step / glm::length(new_eye), axis);

        double height, unused, turned_lat, turned_lon;
        ecef_to_geodetic(new_eye, unused, unused, height);
        ecef_to_geodetic(turn * new_eye, turned_lat, turned_lon, unused);
        new_eye = geodetic_to_ecef(turned_lat, turned_lon, height);
        fwd = turn * fwd;
    }

    if (glm::length(new_eye) > 1.4e7) // |vertex - eye| bounds
    {
        return;
    }
    
    camera_aim(m_camera, new_eye, fwd);
}

void AlpeniteRenderer::mouse_move(nucleus::event_parameter::Mouse m)
{
    m_buttons = m.buttons;
    glm::dvec2 d = glm::dvec2(m.point.position - m.point.last_position);
    if (m.buttons & Qt::MiddleButton)
    {
        camera_orbit(m_camera, d.x, d.y);
    }
    else if (m.buttons & Qt::LeftButton)
    {
        camera_pan(m_camera, d.x, d.y, double(m_viewport.y));
    }
    else if (m.buttons & Qt::RightButton)
    {
        camera_look(m_camera, d.x, d.y);
    }
}

void AlpeniteRenderer::mouse_press(nucleus::event_parameter::Mouse m) 
{ 
    m_buttons = m.buttons;
}

void AlpeniteRenderer::wheel_turn(nucleus::event_parameter::Wheel w) 
{ 
    double notch = w.angle_delta.y() / 200.0;

    if(m_buttons & Qt::RightButton)
    {
        m_camera.speed = glm::clamp(m_camera.speed * glm::pow(1.25, notch), 1e-3, 1e3);
    }
    else 
    {
        m_camera.distance = glm::clamp(m_camera.distance * glm::pow(0.85, notch), 1.0, 8e6);
    }
}

void AlpeniteRenderer::key_press(QKeyCombination key)
{
    if(bool* k = fly_key(m_keys, key.key()))
    {
        *k = true;
    }
}

void AlpeniteRenderer::key_release(QKeyCombination key) 
{
    if (bool* k = fly_key(m_keys, key.key()))
    {
        *k = false;
    }
}

} // namespace webgpu_engine
