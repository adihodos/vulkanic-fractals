#version 460 core

// layout (location = 0) in vec2 vs_in_pos;
// layout (location = 1) in vec4 vs_in_color;

// layout (set = 0, binding = 0) uniform Transforms {
//     mat4 projection;
// } transforms;

out gl_PerVertex {
    vec4 gl_Position;
};

out VS_OUT_FS_IN {
    layout (location = 0) vec4 color;
} vs_out_fs_in;

const vec2 VERTEX_POS[] = vec2[](
    vec2(-1, 1),
    vec2(1, -1),
    vec2(-1, -1),
    vec2(-1, 1),
    vec2(1, 1),
    vec2(1, -1)
);

void main() {
    gl_Position = vec4(VERTEX_POS[gl_VertexIndex], 0.0, 1.0);
    // gl_Position = transforms.projection * mat4(vs_in_pos, 0.0, 1.0);
    vs_out_fs_in.color = vec4(1.0, 0.0, 0.0, 1.0);
}
