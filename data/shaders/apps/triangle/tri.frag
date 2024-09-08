#version 460 core

#include "core/bindless.core.glsl"

in VS_OUT_FS_IN {
    layout (location = 0) vec2 uv;
    layout (location = 1) flat uint mtl;
} fs_in;

layout (location = 0) out vec4 FinalFragColor;

void main() {
    FinalFragColor = texture(g_Textures2DGlobal[fs_in.mtl], fs_in.uv);
}
