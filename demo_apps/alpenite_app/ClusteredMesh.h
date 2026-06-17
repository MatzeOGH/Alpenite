#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <webgpu/webgpu.h>

// uncompresed verison of a mesh vertex
struct MeshVertex
{
    float x, y, z;
    float nx, ny, nz;
    float tu, tv;
};

struct ClusterN
{
    int32_t  refined;               // index of more-refined group, or -1 for base geometry
    float    groupCenter[3];
    float    groupRadius;
    float    groupError;
    float    refinedCenter[3];
    float    refinedRadius;
    float    refinedError;
    uint32_t meshletVertexOffset;
    uint32_t meshletTriangleOffset;
    uint32_t vertexCount;
    uint32_t triangleCount;
};

struct Surface
{
    std::vector<MeshVertex> vertices;
    std::vector<uint32_t>   indices;
};

struct IndexedMesh {};

struct ClusteredMeshScene {};

// GPU-resident buffers produced by loadGltfMeshToGPU.
// All primitives across all glTF meshes are flattened into a single set of buffers.
struct ClusteredMeshGPU
{
    WGPUBuffer vertexBuffer;          // array of MeshVertex
    WGPUBuffer clusterBuffer;         // array of ClusterN
    WGPUBuffer meshletVertexBuffer;   // array of uint32_t
    WGPUBuffer meshletTriangleBuffer; // array of uint8_t (padded to 4-byte alignment)

    uint32_t vertexCount;
    uint32_t clusterCount;
    uint32_t meshletVertexCount;
    uint32_t meshletTriangleByteCount; // raw byte count before alignment padding
};

bool loadMesh(std::string path);

// Loads every triangle primitive from a glTF file, builds nanite clusters for each,
// concatenates the results into a single flat buffer set, and uploads it to the GPU.
// Returns false on any parse/validation error; out is not modified in that case.
bool loadGltfMeshToGPU(const std::string& path, WGPUDevice device, WGPUQueue queue, ClusteredMeshGPU& out);

void buildNanite(
    const std::vector<MeshVertex>& vertices,
    const std::vector<uint32_t>&   indices,
    std::vector<ClusterN>&         outClusters,
    std::vector<uint32_t>&         outMeshletVertices,
    std::vector<uint8_t>&          outMeshletTriangles);