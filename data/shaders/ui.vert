#version 460 core

layout (location = 0) in vec2 vs_in_pos;
layout (location = 1) in vec2 vs_in_uv;
layout (location = 2) in vec4 vs_in_color;

layout (set = 0, binding = 0) uniform Transforms {
	mat4 world_view_proj;
} transforms;

layout (location = 0) out VS_OUT_FS_IN {
	vec2 uv;
	vec4 color;
} vs_out;

void main() {
	gl_Position = transforms.world_view_proj * vec4(vs_in_pos, 0.0, 1.0);
	vs_out.uv = vs_in_uv;
	vs_out.color = vs_in_color;
}
