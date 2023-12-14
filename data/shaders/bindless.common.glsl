#extension GL_EXT_nonuniform_qualifier : require

layout (std140) buffer;

struct JuliaFractalParams {
  uint screen_width;
  uint screen_height;
  uint iterations;
  float zoom;
  float ox;
  float oy;
  uint coloring;
  float fxmin;
  float fxmax;
  float fymin;
  float fymax;
  uint escape_radius;
  float cx;
  float cy;
  uint iteration_type;
};

struct MandelbrotFractalParams {
  uint screen_width;
  uint screen_height;
  uint iterations;
  float zoom;
  float ox;
  float oy;
  uint coloring;
  float fxmin;
  float fxmax;
  float fymin;
  float fymax;
  uint escape_radius;
  uint ftype;
};

struct UiBackendParams {
  mat4 worldViewProjection;
  uint fontAtlasId;
};

layout (set = 0, binding = 0) readonly buffer GlobalJuliaParamsBuffer  {
  JuliaFractalParams p[];
} g_JuliaFractalParams[];

layout (set = 0, binding = 0) readonly buffer GlobalMandelbrotParamsBuffer {
  MandelbrotFractalParams p[];
} g_MandelbrotFractalParams[];

layout (set = 0, binding = 0) readonly buffer GlobalUiParamsBuffer {
  UiBackendParams p[];
} g_UiBackendParams[];

layout (set = 1, binding = 0) uniform sampler2D g_GlobalTexture2DPool[];

layout (push_constant) uniform GlobalResourceId {
  uint id;
} g_UniqueResourceId;

uvec2 GetBufferAndOffset(uint resid) {
  return uvec2(resid & 0x0000FFFF, (resid >> 16) & 0x0000FFFF);
}
