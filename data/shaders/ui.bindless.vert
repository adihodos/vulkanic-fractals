#version 460 core

#include "core/bindless.core.glsl"

struct UiBackendParams {
  mat4 worldViewProjection;
  uint fontAtlasId;
};

layout (set = 1, binding = 0) readonly buffer UiBackendParamsGlobal {
  UiBackendParams p[];
} g_UiBackendParams[];

layout (location = 0) in vec2 vs_in_pos;
layout (location = 1) in vec2 vs_in_uv;
layout (location = 2) in vec4 vs_in_color;

layout (location = 0) out VS_OUT_FS_IN {
  vec2 uv;
  vec4 color;
  flat uint atlas_id;
} vs_out;

void main() {
  const uvec3 global_data = unpack_global_pushconst();
  // const uint frame_id = global_data.x;
  const uint buffer_idx = global_data.y;

  const UiBackendParams beParams = g_UiBackendParams[buffer_idx].p[0];

  gl_Position = beParams.worldViewProjection * vec4(vs_in_pos, 0.0, 1.0);
  vs_out.uv = vs_in_uv;
  vs_out.color = vs_in_color;
  vs_out.atlas_id = beParams.fontAtlasId;
}
