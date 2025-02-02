#ifndef XR_VK_BINDLESS_CORE_GLSL_INCLUDED
#define XR_VK_BINDLESS_CORE_GLSL_INCLUDED

#extension GL_EXT_nonuniform_qualifier : require

struct FrameGlobalData_t {
    mat4 world_view_proj;
    mat4 projection;
    mat4 inv_projection;
    mat4 view;
    mat4 ortho_proj;
    vec3 eye_pos;
    uint frame_id;
};

layout (std140, set = 0, binding  = 0) uniform FrameGlobal {
    FrameGlobalData_t data[];
} g_FrameGlobal[];

struct InstanceRenderInfo_t {
    mat4 model;
    uint vtx_buff;
    uint idx_buff;
    uint mtl_coll_offset;
    uint mtl_id;
};

layout (set = 1, binding = 0) readonly buffer InstancesGlobal {
    InstanceRenderInfo_t data[];
} g_InstanceGlobal[];

struct VertexPTC {
    vec2 pos;
    vec2 uv;
    vec4 color;
};

layout (set = 1, binding = 0) readonly buffer VerticesGlobal {
    VertexPTC data[];
} g_VertexBufferGlobal[];

layout (set = 1, binding = 0) readonly buffer IndicesGlobal {
    uint data[];
} g_IndexBufferGlobal[];

layout (set = 2, binding = 0) uniform sampler1DArray g_Textures1DArrayGlobal[];
layout (set = 2, binding = 0) uniform sampler2D g_Textures2DGlobal[];
layout (set = 2, binding = 0) uniform sampler2DArray g_Textures2DArrayGlobal[];
layout (set = 2, binding = 0) uniform samplerCube g_TexturesCubeGlobal[];

layout (push_constant) uniform PushConstantsGlobal {
    uint data;
} g_GlobalPushConst;

//
// [0 .. 3] bits is frame index
// [4 .. 19] bits is buffer id
// [20 .. 31] bits is instance id
// [0..3] frame id (x)
// [4..14] resource index (y)
uvec3 unpack_global_pushconst() {
//     const uint frame_idx = (g_GlobalPushConst.data) & 0xFF;
//     const uint inst_buffer_idx = (g_GlobalPushConst.data & 0xFFFF0000) >> 16;
//     const uint instance_index = (g_GlobalPushConst.data & 0x0000FF00) >> 8;
//     return uvec2(g_GlobalPushConst.data >> 20, (g_GlobalPushConst.data & 0xFFFFF));

    const uint frame_id = g_GlobalPushConst.data & 0x10;
    const uint resource_id = (g_GlobalPushConst.data >> 4) & 0xFF;
    const uint inst_id = g_GlobalPushConst.data >> 20;

    return uvec3(frame_id, resource_id, inst_id);
}

#endif /* !defined XR_VK_BINDLESS_CORE_GLSL_INCLUDED */
