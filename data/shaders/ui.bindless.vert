#version 460 core

#include "bindless.common.glsl"

layout (location = 0) in vec2 vs_in_pos;
layout (location = 1) in vec2 vs_in_uv;
layout (location = 2) in vec4 vs_in_color;

layout (location = 0) out VS_OUT_FS_IN {
  vec2 uv;
  vec4 color;
  flat uint atlas_id;
} vs_out;

void main() {
  const uvec2 ids = GetBufferAndOffset(g_UniqueResourceId.id);

  const UiBackendParams beParams = g_UiBackendParams[nonuniformEXT(ids.x)].p[nonuniformEXT(ids.y)];

  gl_Position = beParams.worldViewProjection * vec4(vs_in_pos, 0.0, 1.0);
  vs_out.uv = vs_in_uv;
  vs_out.color = vs_in_color;
  vs_out.atlas_id = beParams.fontAtlasId;
}
