#version 460 core

layout (location = 0) in VS_OUT_FS_IN {
	vec2 uv;
	vec4 color;
} fs_in;

layout (set = 0, binding = 1) uniform sampler2D s;

layout (location = 0) out vec4 FinalFragColor;

void main() {
	FinalFragColor = vec4(texture(s, fs_in.uv).r) * fs_in.color;
}

