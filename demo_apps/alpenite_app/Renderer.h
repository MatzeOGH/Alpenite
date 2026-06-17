#pragma once

#include "FrameworkSubsystem.h"
#include "ClusteredMesh.h"
#include <QObject>


// Call after initWebGPUBackend to bind a mesh's cluster buffer to the cull pass.
// Safe to call again when the mesh changes.
void initClusterCullPass(const ClusteredMeshGPU& mesh);

// Creates the render pipeline that decodes (cluster_id<<8|local_idx) packed indices
// produced by the expand pass. Safe to call again when the mesh changes.
void initClusteredRenderPipeline(const ClusteredMeshGPU& mesh);

// Convenience: loads a glTF, builds nanite clusters, uploads to GPU, and calls
// initClusterCullPass + initClusterExpandPass + initClusteredRenderPipeline.
void loadMeshForRendering(const std::string& path);

// Call after initClusterCullPass. Creates gOutputIndexBuffer and gDrawArgsBuffer
// sized for the worst-case (all clusters visible) and binds them together with
// the visible-clusters and meshlet-triangle buffers from the cull pass.
// After each frame's expand dispatch, gOutputIndexBuffer holds packed u32 indices
// and gDrawArgsBuffer holds a ready DrawIndexedIndirectParams.
// Safe to call again when the mesh changes.
void initClusterExpandPass(const ClusteredMeshGPU& mesh);

class IRenderer : public QObject, public IFrameworkSubsystem
{
    Q_OBJECT
public:
    IRenderer(QObject* parent) : QObject(parent) {}

    bool register_subsystem(Framework* framework) override;

    void unregister_subsystem(Framework* framework) override {}

    void render();

public slots:
    void mouse_moved(int dx, int dy);
    void mouse_button(int button, bool pressed);
    void key_event(int key, bool pressed);

private slots:
    void window_created(WindowHandel window);
    void window_resized(int width, int height);

private:
    int32_t m_pendingDx{0};
    int32_t m_pendingDy{0};

    bool m_keyW{false};
    bool m_keyS{false};
    bool m_keyA{false};
    bool m_keyD{false};
    bool m_rmb{false};
};
