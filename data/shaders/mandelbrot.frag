#version 460 core

layout (location = 0) out vec4 FragColor;

const uint COLORING_BASIC = 0;
const uint COLORING_SMOOTH = 1;
const uint COLORING_LOG = 2;
const uint COLORING_HSV = 3;
const uint COLORING_RAINBOW = 4;
const uint COLORING_PALETTE = 5;

const uint M_TYPE_STANDARD = 0;
const uint M_TYPE_BURNING_SHIP = 1;

layout (std140, set = 0, binding = 0) uniform FractalParams {
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
} params;

struct Complex {
    float re;
    float im;
};

float complex_mag_squared(in Complex c) {
    return c.re * c.re + c.im * c.im;
}

float complex_mag(in Complex c) {
    return sqrt(complex_mag_squared(c));
}

Complex complex_add(in Complex a, in Complex b) {
    return Complex(
        a.re + b.re,
        a.im + b.im);
}

Complex complex_sub(in Complex a, in Complex b) {
    return Complex(
        a.re - b.re,
        a.im - b.im);
}

Complex complex_mul(in Complex a, in Complex b) {
    return Complex(
        a.re * b.re - a.im * b.im,
        a.re * b.im + a.im * b.re
    );
}

Complex complex_mul_scalar(in Complex a, in float b) {
    return Complex(
        a.re * b,
        a.im * b
    );
}

Complex complex_mul_scalar(in float a, in Complex b) {
    return Complex(
        b.re * a,
        b.im * a
    );
}

Complex screen_coords_to_complex_coords(
    in float px,
    in float py,
    in float dxmin,
    in float dxmax,
    in float dymin,
    in float dymax
) {
    const float x = (px / params.screen_width) * (dxmax - dxmin) + dxmin;
    const float y = (py / params.screen_height) * (dymax - dymin) + dymin;

    return Complex(x, y);
}

vec3 color_simple(in float n, in uint max_iterations) {
    return vec3(n);
}

vec3 color_smooth(in float n, in uint max_iterations) {
    const float t = n; // clamp(n / max_iterations, 0.0, 1.0);
    const float u = 1.0 - t;

    return vec3(
        9.0 * u * t * t * t,
        15.0 * u * u * t * t,
        8.5 * u * u * u * t
    );
}

vec3 color_log(in float n, in uint max_iterations) {
    const float k = n;

    return vec3(
        (1.0 -cos(k * 0.25)) * .5,
        (1.0 - cos(k * 0.8)) * .5,
        (1.0 - cos(k * 0.120)) * .5
    );
}

//
// from: https://gist.github.com/983/e170a24ae8eba2cd174f
vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

vec3 color_hsv(in float iterations, in float max_iterations) {
    const vec3 hsv = vec3(
        iterations,
        1.0,
        iterations < max_iterations ? 1.0 : 0.0
    );

    return hsv2rgb(hsv);
}

// Copyright 2019 Google LLC.
// SPDX-License-Identifier: Apache-2.0

// Polynomial approximation in GLSL for the Turbo colormap
// Original LUT: https://gist.github.com/mikhailov-work/ee72ba4191942acecc03fe6da94fc73f

// Authors:
//   Colormap Design: Anton Mikhailov (mikhailov@google.com)
//   GLSL Approximation: Ruofei Du (ruofei@google.com)

vec3 TurboColormap(in float x) {
  const vec4 kRedVec4 = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
  const vec4 kGreenVec4 = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
  const vec4 kBlueVec4 = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
  const vec2 kRedVec2 = vec2(-152.94239396, 59.28637943);
  const vec2 kGreenVec2 = vec2(4.27729857, 2.82956604);
  const vec2 kBlueVec2 = vec2(-89.90310912, 27.34824973);

  x = clamp(x, 0.0, 1.0);
  vec4 v4 = vec4( 1.0, x, x * x, x * x * x);
  vec2 v2 = v4.zw * v4.z;
  return vec3(
    dot(v4, kRedVec4)   + dot(v2, kRedVec2),
    dot(v4, kGreenVec4) + dot(v2, kGreenVec2),
    dot(v4, kBlueVec4)  + dot(v2, kBlueVec2)
  );
}

vec3 color_rainbow(in float n, in float max_iterations) {
    return TurboColormap(n);
}

vec3 palette(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
    return a + b*cos( 6.28318*(c*t+d) );
}

vec3 color_palette(in float n, in float max_iterations) {

#define PALETTE1

#ifdef PALETTE1
   const vec3 a = vec3(0.5, 0.5, 0.5);
   const vec3 b = vec3(0.5, 0.5, 0.5);
   const vec3 c = vec3(1.0, 1.0, 1.0);
   const vec3 d = vec3(0.00, 0.33, 0.67);
#elif defined PALETTE2
   const vec3 a = vec3(0.5, 0.5, 0.5);
   const vec3 b = vec3(0.5, 0.5, 0.5);
   const vec3 c = vec3(1.0, 1.0, 1.0);
   const vec3 d = vec3(0.00, 0.10, 0.20);
#elif defined PALETTE3
   const vec3 a = vec3(0.5, 0.5, 0.5);
   const vec3 b = vec3(0.5, 0.5, 0.5);
   const vec3 c = vec3(1.0, 1.0, 1.0);
   const vec3 d = vec3(0.30, 0.20, 0.20);
#elif defined PALETTE4
   const vec3 a = vec3(0.5, 0.5, 0.5);
   const vec3 b = vec3(0.5, 0.5, 0.5);
   const vec3 c = vec3(1.0, 1.0, 0.5);
   const vec3 d = vec3(0.80, 0.90, 0.30);
#elif defined PALETTE5
   const vec3 a = vec3(0.5, 0.5, 0.5);
   const vec3 b = vec3(0.5, 0.5, 0.5);
   const vec3 c = vec3(1.0, 0.7, 0.4);
   const vec3 d = vec3(0.00, 0.15, 0.20);
#else
   const vec3 a = vec3(0.5, 0.5, 0.5);
   const vec3 b = vec3(0.5, 0.5, 0.5);
   const vec3 c = vec3(2.0, 1.0, 0.0);
   const vec3 d = vec3(0.50, 0.20, 0.25);
#endif

   return palette(n, a, b, c, d);
}

float mandelbrot_standard(in Complex z, in Complex c) {
    uint iterations = 0;

    while ((complex_mag_squared(z) <= (params.escape_radius * params.escape_radius)) && (iterations < params.iterations)) {
        z = complex_add(complex_mul(z, z), c);
        iterations += 1;
    }

    z = complex_add(complex_mul(z, z), c);
    iterations += 1;    // a couple of extra iterations helps
    z = complex_add(complex_mul(z, z), c);
    iterations += 1;    // a couple of extra iterations helps
    float modulus = complex_mag(z);
    float mu = iterations - (log2 (log2(modulus)))/ log2(2.0);

    return mu;
}

float mandelbrot_burning_ship(in Complex z, in Complex c) {
    uint iterations = 0;

    while (complex_mag_squared(z) <= 10.0 && iterations < params.iterations) {
        const float re = z.re * z.re - z.im * z.im - c.re;
        const float im = 2.0 * abs(z.re * z.im) - c.im;
        z = Complex(re, im);
        iterations += 1;
    }

    float modulus = complex_mag(z);
    float mu = iterations - (log2 (log2(modulus)))/ log2(2.0);

    return mu;
}

void main() {
    const Complex c = screen_coords_to_complex_coords(
        gl_FragCoord.x, gl_FragCoord.y, params.fxmin, params.fxmax, params.fymin, params.fymax
    );

    Complex z = Complex(0.0, 0.0);

    float mu = 0.0;

    switch (params.ftype) {
        case M_TYPE_STANDARD:
            mu = mandelbrot_standard(z, c);
            break;

        case M_TYPE_BURNING_SHIP:
            mu = mandelbrot_burning_ship(z, c);
            break;

        default:
            break;
    }

    float mu2 = mu;
    mu = sqrt(mu / params.iterations);

    vec3 color = vec3(0.0);

    switch (params.coloring) {
        case COLORING_SMOOTH:
            color = color_smooth(mu, params.iterations);
            break;

        case COLORING_LOG:
            color = color_log(mu2, params.iterations);
            break;

        case COLORING_HSV:
            color = color_hsv(mu, params.iterations);
            break;

        case COLORING_RAINBOW:
            color = color_rainbow(mu, params.iterations);
            break;

        case COLORING_BASIC: default:
            color = color_simple(mu, params.iterations);
            break;

        case COLORING_PALETTE:
            color = color_palette(mu, params.iterations);
            break;
    }

    FragColor = vec4(color, 1.0);
}
