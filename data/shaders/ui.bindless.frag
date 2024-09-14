#version 460 core

#include "core/bindless.core.glsl"

layout (location = 0) in VS_OUT_FS_IN {
  vec2 uv;
  vec4 color;
  flat uint atlas_id;
} fs_in;

layout (location = 0) out vec4 FinalFragColor;

void main() {
  const uvec2 global_data = unpack_global_pushconst();
  const uint frame_id = global_data.x;
  const uint buffer_idx = global_data.y;

  FinalFragColor = vec4(texture(g_Textures2DGlobal[fs_in.atlas_id], fs_in.uv).r) * fs_in.color;
}
