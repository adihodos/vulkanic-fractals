#version 460 core

#include "core/bindless.core.glsl"
#include "defs.glsl"

out gl_PerVertex {
    vec4 gl_Position;
};

out VS_OUT_FS_IN {
    layout (location = 0) vec2 uv;
    layout (location = 1) flat uint mtl;
} vs_out;

void main() {
    const uint frame_idx = (g_GlobalPushConst.data) & 0xFF;
    const FrameGlobalData_t fgd = g_FrameGlobal[frame_idx].data[0];

    const InstanceRenderInfo_t inst = g_InstanceGlobal[frame_idx].data[gl_InstanceIndex];
    const uint vertex_index = g_IndexBufferGlobal[inst.idx_buff].data[gl_VertexIndex];
    const VertexPTC vtx = g_VertexBufferGlobal[inst.vtx_buff].data[vertex_index];

    gl_Position = fgd.world_view_proj * inst.model * vec4(vtx.pos, 0.0, 1.0);
    vs_out.uv = vtx.uv;
    vs_out.mtl = inst.mtl_id;
}
