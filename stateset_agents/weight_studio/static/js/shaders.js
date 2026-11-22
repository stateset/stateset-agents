/**
 * Custom GLSL ES 3.00 Shaders for StateSet 3D Weight Studio v2.
 *
 * Implements:
 * - Real-time 1024x1024 PCF soft shadow mapping
 * - Real-time planar optical-bench floor reflections with anisotropic brushed blur
 * - Cook-Torrance GGX PBR instanced crystal & chassis shading + shadow self-occlusion
 * - Procedural HDR studio softbox IBL environment reflections
 * - Volumetric curved optical conduits with traveling forward/backprop photon waves
 * - Multi-pass HDR Bright-Pass, dual-tier Gaussian Bloom, Depth SSAO crevice occlusion,
 *   Optical Bokeh Depth-of-Field, and ACES Filmic Tonemapping
 */

export const STUDIO_RIGS = {
  obsidian: {
    name: 'Obsidian Lab',
    bgTop: [0.007, 0.011, 0.022],
    bgHorizon: [0.020, 0.032, 0.056],
    floorColor: [0.014, 0.019, 0.030],
    gridColor: [0.09, 0.24, 0.44],
    keyDir: [0.58, 0.72, 0.36],
    keyColor: [1.38, 1.26, 1.10],
    rimDir: [-0.70, 0.38, -0.60],
    rimColor: [0.48, 0.98, 1.48],
    fillDir: [0.0, 0.95, -0.32],
    fillColor: [0.24, 0.34, 0.54],
    ambient: [0.032, 0.044, 0.068],
    exposure: 0.98,
    bloomStrength: 0.32,
  },
  copper: {
    name: 'Copper Forge',
    bgTop: [0.016, 0.010, 0.008],
    bgHorizon: [0.046, 0.028, 0.018],
    floorColor: [0.024, 0.017, 0.014],
    gridColor: [0.42, 0.24, 0.10],
    keyDir: [0.62, 0.68, 0.38],
    keyColor: [1.55, 1.12, 0.74],
    rimDir: [-0.66, 0.44, -0.58],
    rimColor: [0.42, 0.85, 1.38],
    fillDir: [0.0, 0.92, -0.38],
    fillColor: [0.32, 0.24, 0.20],
    ambient: [0.040, 0.028, 0.022],
    exposure: 0.98,
    bloomStrength: 0.36,
  },
  photonics: {
    name: 'Deep Photonics',
    bgTop: [0.005, 0.010, 0.024],
    bgHorizon: [0.014, 0.042, 0.072],
    floorColor: [0.010, 0.019, 0.032],
    gridColor: [0.08, 0.46, 0.56],
    keyDir: [0.48, 0.78, 0.39],
    keyColor: [0.85, 1.35, 1.55],
    rimDir: [-0.72, 0.38, -0.58],
    rimColor: [1.05, 0.48, 1.55],
    fillDir: [0.0, 0.95, -0.30],
    fillColor: [0.22, 0.52, 0.48],
    ambient: [0.024, 0.042, 0.065],
    exposure: 1.02,
    bloomStrength: 0.40,
  },
  cleanroom: {
    name: 'Cleanroom White',
    bgTop: [0.64, 0.70, 0.78],
    bgHorizon: [0.82, 0.86, 0.91],
    floorColor: [0.68, 0.73, 0.80],
    gridColor: [0.28, 0.38, 0.52],
    keyDir: [0.54, 0.78, 0.32],
    keyColor: [1.45, 1.42, 1.38],
    rimDir: [-0.62, 0.50, -0.60],
    rimColor: [0.65, 0.82, 1.05],
    fillDir: [0.0, 0.96, -0.28],
    fillColor: [0.48, 0.54, 0.62],
    ambient: [0.18, 0.21, 0.26],
    exposure: 0.90,
    bloomStrength: 0.14,
  },
};

export const SHADOW_VS = `#version 300 es
precision highp float;
layout(location = 0) in vec3 aPos;
layout(location = 2) in vec4 aInstPosScale;
layout(location = 3) in vec4 aInstDimMat;
uniform mat4 uLightViewProj;
void main() {
  vec3 scale = vec3(aInstPosScale.w, aInstDimMat.x, aInstDimMat.y);
  vec3 worldPos = aInstPosScale.xyz + aPos * scale;
  gl_Position = uLightViewProj * vec4(worldPos, 1.0);
}
`;

export const SHADOW_FS = `#version 300 es
precision highp float;
out vec4 fragColor;
void main() {
  fragColor = vec4(1.0);
}
`;

export const SKYBOX_VS = `#version 300 es
precision highp float;
const vec2 POS[3] = vec2[](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
out vec2 vUv;
void main() {
  vUv = POS[gl_VertexID] * 0.5 + 0.5;
  gl_Position = vec4(POS[gl_VertexID], 0.9999, 1.0);
}
`;

export const SKYBOX_FS = `#version 300 es
precision highp float;
in vec2 vUv;
uniform mat4 uInvViewProj;
uniform vec3 uCameraPos;
uniform vec3 uBgTop;
uniform vec3 uBgHorizon;
uniform vec3 uKeyDir;
uniform vec3 uRimDir;
uniform vec3 uKeyColor;
uniform vec3 uRimColor;
out vec4 fragColor;

void main() {
  vec2 ndc = vUv * 2.0 - 1.0;
  vec4 farP = uInvViewProj * vec4(ndc, 1.0, 1.0);
  vec3 dir = normalize(farP.xyz / farP.w - uCameraPos);

  float h = clamp(dir.y * 0.85 + 0.22, 0.0, 1.0);
  vec3 col = mix(uBgHorizon, uBgTop, pow(h, 0.65));

  float keyGlow = pow(max(dot(dir, normalize(uKeyDir)), 0.0), 18.0);
  float rimGlow = pow(max(dot(dir, normalize(uRimDir)), 0.0), 14.0);
  col += uKeyColor * keyGlow * 0.045;
  col += uRimColor * rimGlow * 0.038;

  float band = exp(-abs(dir.y) * 9.0);
  col += mix(uBgHorizon, uRimColor * 0.14, 0.22) * band * 0.35;

  fragColor = vec4(col, 1.0);
}
`;

export const FLOOR_VS = `#version 300 es
precision highp float;
layout(location = 0) in vec3 aPos;
uniform mat4 uViewProj;
uniform mat4 uLightViewProj;
uniform float uFloorY;
out vec3 vWorldPos;
out vec4 vClipPos;
out vec4 vLightPos;
void main() {
  vec3 wp = vec3(aPos.x, uFloorY, aPos.z);
  vWorldPos = wp;
  vClipPos = uViewProj * vec4(wp, 1.0);
  vLightPos = uLightViewProj * vec4(wp, 1.0);
  gl_Position = vClipPos;
}
`;

export const FLOOR_FS = `#version 300 es
precision highp float;
in vec3 vWorldPos;
in vec4 vClipPos;
in vec4 vLightPos;

uniform vec3 uCameraPos;
uniform vec3 uFloorColor;
uniform vec3 uGridColor;
uniform vec3 uBgHorizon;
uniform vec3 uKeyDir;
uniform vec3 uKeyColor;
uniform vec3 uRimDir;
uniform vec3 uRimColor;
uniform vec2 uFootprintSize;
uniform vec3 uGlowCenter;
uniform vec3 uGlowColor;
uniform float uTime;
uniform sampler2D uReflectTex;
uniform float uReflectStrength;
uniform sampler2D uShadowMap;
uniform float uShadowEnabled;
out vec4 fragColor;

float gridLine(vec2 p, float stepSize, float width) {
  vec2 coord = p / stepSize;
  vec2 grid = abs(fract(coord - 0.5) - 0.5) / max(fwidth(coord), 0.0005);
  float line = min(grid.x, grid.y);
  return 1.0 - smoothstep(0.0, width, line);
}

float sampleShadow(vec4 lp) {
  if (uShadowEnabled < 0.5) return 1.0;
  vec3 proj = lp.xyz / max(lp.w, 0.0001) * 0.5 + 0.5;
  if (proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0 || proj.z > 1.0) return 1.0;
  vec2 texel = vec2(1.0 / 1024.0);
  float shadow = 0.0;
  float bias = 0.003;
  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      float d = texture(uShadowMap, proj.xy + vec2(float(x), float(y)) * texel * 1.6).r;
      shadow += (proj.z - bias > d) ? 0.35 : 1.0;
    }
  }
  return shadow / 9.0;
}

void main() {
  vec3 N = vec3(0.0, 1.0, 0.0);
  vec3 V = normalize(uCameraPos - vWorldPos);
  float dist = length(vWorldPos.xz);

  // Micro-brushed anodized metallic grain
  float brush = sin(vWorldPos.x * 110.0) * 0.5 + sin(vWorldPos.z * 14.0) * 0.5;
  vec3 baseCol = uFloorColor * (1.0 + brush * 0.025);

  // Multi-scale precision optical bench coordinate grid
  float fine = gridLine(vWorldPos.xz, 1.0, 1.15) * 0.14;
  float major = gridLine(vWorldPos.xz, 5.0, 1.45) * 0.36;
  float axisX = 1.0 - smoothstep(0.0, 0.045, abs(vWorldPos.z));
  float axisZ = 1.0 - smoothstep(0.0, 0.045, abs(vWorldPos.x));
  float ring = (1.0 - smoothstep(0.0, 0.09, abs(fract(dist * 0.1) - 0.5) * 10.0)) * 0.22;

  float radialFade = exp(-dist * dist * 0.00055);
  vec3 gridEmit = uGridColor * ((fine + major + ring) + (axisX + axisZ) * 0.45) * radialFade;

  // Analytic soft contact shadow + real PCF shadow map
  vec2 dBox = abs(vWorldPos.xz) - uFootprintSize;
  float sd = length(max(dBox, 0.0)) + min(max(dBox.x, dBox.y), 0.0);
  float contactAO = mix(0.45, 1.0, smoothstep(-0.8, 3.6, sd));
  float castShadow = sampleShadow(vLightPos);
  float ao = contactAO * mix(0.65, 1.0, castShadow);

  // Specular studio softbox reflections
  vec3 L1 = normalize(uKeyDir);
  vec3 H1 = normalize(L1 + V);
  float spec1 = pow(max(dot(N, H1), 0.0), 42.0) * 0.22 * castShadow;

  vec3 L2 = normalize(uRimDir);
  vec3 H2 = normalize(L2 + V);
  float spec2 = pow(max(dot(N, H2), 0.0), 28.0) * 0.16;

  float fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.6);
  vec3 refl = mix(baseCol, uBgHorizon * 1.15, fresnel * 0.45);

  // Real-time planar reflection of the 3D crystal towers with brushed anodized blur!
  // Because the reflection pass mirrors worldPos.y -> -worldPos.y under the same uViewProj,
  // each reflected point lands directly at screenUv beneath its optical-bench footprint.
  vec3 planarRefl = vec3(0.0);
  if (uReflectStrength > 0.01) {
    vec2 rUv = (vClipPos.xy / max(vClipPos.w, 0.0001)) * 0.5 + 0.5;
    float blurR = 0.0028;
    vec3 rSum = texture(uReflectTex, clamp(rUv, 0.005, 0.995)).rgb * 0.40;
    rSum += texture(uReflectTex, clamp(rUv + vec2(blurR, 0.0), 0.005, 0.995)).rgb * 0.15;
    rSum += texture(uReflectTex, clamp(rUv - vec2(blurR, 0.0), 0.005, 0.995)).rgb * 0.15;
    rSum += texture(uReflectTex, clamp(rUv + vec2(0.0, blurR * 2.0), 0.005, 0.995)).rgb * 0.15;
    rSum += texture(uReflectTex, clamp(rUv - vec2(0.0, blurR * 2.0), 0.005, 0.995)).rgb * 0.15;
    float reflMask = clamp(0.18 + fresnel * 0.65, 0.0, 0.85) * radialFade * uReflectStrength;
    planarRefl = rSum * reflMask;
  }

  float glowDist = length(vWorldPos.xz - uGlowCenter.xz);
  vec3 bounce = uGlowColor * exp(-glowDist * glowDist * 0.014) * 0.24;

  vec3 col = refl * ao + planarRefl + (uKeyColor * spec1 + uRimColor * spec2) * ao + gridEmit + bounce;

  float fog = 1.0 - exp(-dist * dist * 0.00028);
  col = mix(col, uBgHorizon, clamp(fog, 0.0, 1.0));

  fragColor = vec4(col, 1.0);
}
`;

export const INSTANCED_PBR_VS = `#version 300 es
precision highp float;

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

layout(location = 2) in vec4 aInstPosScale;
layout(location = 3) in vec4 aInstDimMat;
layout(location = 4) in vec4 aInstColor;
layout(location = 5) in vec4 aInstFlags;

uniform mat4 uViewProj;
uniform mat4 uLightViewProj;
uniform float uMirrorY; // 1.0 normal, -1.0 planar reflection pass
uniform float uTime;

out vec3 vWorldPos;
out vec3 vNormal;
out vec3 vLocalPos;
out vec4 vColor;
out vec4 vMat;
out vec4 vFlags;
out vec4 vLightPos;

void main() {
  vec3 scale = vec3(aInstPosScale.w, aInstDimMat.x, aInstDimMat.y);
  vec3 worldPos = aInstPosScale.xyz + aPos * scale;
  vec3 scaledNormal = normalize(aNormal / max(scale, vec3(0.0001)));

  vLightPos = uLightViewProj * vec4(worldPos, 1.0);

  if (uMirrorY < 0.0) {
    worldPos.y = -worldPos.y;
    scaledNormal.y = -scaledNormal.y;
  }

  vWorldPos = worldPos;
  vNormal = scaledNormal;
  vLocalPos = aPos;
  vColor = aInstColor;
  vMat = vec4(aInstDimMat.z, aInstDimMat.w, scale.y, clamp(aPos.y + 0.5, 0.0, 1.0));
  vFlags = aInstFlags;

  gl_Position = uViewProj * vec4(worldPos, 1.0);
}
`;

export const INSTANCED_PBR_FS = `#version 300 es
precision highp float;

in vec3 vWorldPos;
in vec3 vNormal;
in vec3 vLocalPos;
in vec4 vColor;
in vec4 vMat;
in vec4 vFlags;
in vec4 vLightPos;

uniform vec3 uCameraPos;
uniform vec3 uBgTop;
uniform vec3 uBgHorizon;
uniform vec3 uKeyDir;
uniform vec3 uKeyColor;
uniform vec3 uRimDir;
uniform vec3 uRimColor;
uniform vec3 uFillDir;
uniform vec3 uFillColor;
uniform vec3 uAmbient;
uniform float uTime;
uniform float uPulseWave;
uniform float uPulseMode;
uniform sampler2D uShadowMap;
uniform float uShadowEnabled;
uniform float uSliceAxis; // 0=off, 1=X, 2=Z, 3=Y
uniform float uSlicePos;

out vec4 fragColor;

const float PI = 3.14159265359;

float D_GGX(float NdotH, float roughness) {
  float a = roughness * roughness;
  float a2 = a * a;
  float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
  return a2 / max(PI * d * d, 0.0001);
}

float G_SchlickGGX(float NdotV, float roughness) {
  float r = roughness + 1.0;
  float k = (r * r) * 0.125;
  return NdotV / max(NdotV * (1.0 - k) + k, 0.0001);
}

float G_Smith(float NdotV, float NdotL, float roughness) {
  return G_SchlickGGX(NdotV, roughness) * G_SchlickGGX(NdotL, roughness);
}

vec3 F_Schlick(float cosTheta, vec3 F0) {
  return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float evalShadow(vec4 lp, vec3 N, vec3 L) {
  if (uShadowEnabled < 0.5) return 1.0;
  vec3 proj = lp.xyz / max(lp.w, 0.0001) * 0.5 + 0.5;
  if (proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0 || proj.z > 1.0) return 1.0;
  float bias = max(0.0025 * (1.0 - max(dot(N, L), 0.0)), 0.0012);
  vec2 texel = vec2(1.0 / 1024.0);
  float occ = 0.0;
  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      float d = texture(uShadowMap, proj.xy + vec2(float(x), float(y)) * texel).r;
      occ += (proj.z - bias > d) ? 0.42 : 1.0;
    }
  }
  return occ / 9.0;
}

vec3 sampleStudioEnv(vec3 R, float roughness) {
  float up = clamp(R.y * 0.65 + 0.35, 0.0, 1.0);
  vec3 env = mix(uBgHorizon * 0.9, uBgTop * 1.4, up);

  float strip = smoothstep(0.68 - roughness * 0.45, 0.96, R.y) *
                smoothstep(0.25, 0.85, 1.0 - abs(R.z));
  env += vec3(1.15, 1.22, 1.35) * strip * (1.2 - roughness * 0.7);

  float kBox = pow(max(dot(R, normalize(uKeyDir)), 0.0), mix(26.0, 3.0, roughness));
  float rBox = pow(max(dot(R, normalize(uRimDir)), 0.0), mix(22.0, 3.0, roughness));
  env += uKeyColor * kBox * 0.45 + uRimColor * rBox * 0.38;
  return env;
}

vec3 evalLight(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 albedo, float roughness, float metallic, vec3 F0) {
  vec3 H = normalize(V + L);
  float NdotL = max(dot(N, L), 0.0);
  float NdotV = max(dot(N, V), 0.001);
  float NdotH = max(dot(N, H), 0.0);
  float HdotV = max(dot(H, V), 0.0);

  float D = D_GGX(NdotH, roughness);
  float G = G_Smith(NdotV, NdotL, roughness);
  vec3 F = F_Schlick(HdotV, F0);

  vec3 spec = (D * G * F) / max(4.0 * NdotV * NdotL, 0.001);
  vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
  return (kD * albedo / PI + spec) * radiance * NdotL;
}

void main() {
  // Optional 3D Tomographic MRI Cutaway Plane with luminous laser cross-section edge
  float sliceGlow = 0.0;
  if (uSliceAxis > 0.5) {
    float coord = (uSliceAxis < 1.5) ? vWorldPos.x : ((uSliceAxis < 2.5) ? vWorldPos.z : vWorldPos.y);
    float dSlice = coord - uSlicePos;
    if (dSlice > 0.0) discard;
    sliceGlow = smoothstep(-0.18, 0.0, dSlice);
  }

  vec3 N = normalize(vNormal);
  vec3 V = normalize(uCameraPos - vWorldPos);
  float NdotV = max(dot(N, V), 0.001);

  vec3 albedo = vColor.rgb;
  float emitBase = vColor.a;
  float roughness = clamp(vMat.x, 0.06, 0.95);
  float metallic = clamp(vMat.y, 0.0, 1.0);
  float heightNorm = vMat.w;

  float rootAO = mix(0.42, 1.0, smoothstep(0.0, 0.52, heightNorm));
  albedo *= mix(0.72, 1.08, heightNorm);

  vec3 F0 = mix(vec3(0.045), albedo, metallic);
  vec3 Lkey = normalize(uKeyDir);
  float keyShadow = evalShadow(vLightPos, N, Lkey);

  vec3 Lo = vec3(0.0);
  Lo += evalLight(N, V, Lkey, uKeyColor * keyShadow, albedo, roughness, metallic, F0);
  Lo += evalLight(N, V, normalize(uRimDir), uRimColor, albedo, roughness, metallic, F0);
  Lo += evalLight(N, V, normalize(uFillDir), uFillColor, albedo, roughness, metallic, F0);

  vec3 R = reflect(-V, N);
  vec3 envRefl = sampleStudioEnv(R, roughness);
  vec3 F_env = F_Schlick(NdotV, F0);
  vec3 kD_env = (vec3(1.0) - F_env) * (1.0 - metallic);
  vec3 ambient = (kD_env * albedo * (uAmbient * 1.9 + uBgHorizon * 0.25) + F_env * envRefl * (0.72 - roughness * 0.42)) * rootAO;

  float fresnel = pow(1.0 - NdotV, 3.2);
  vec3 emission = albedo * emitBase * (0.24 + 0.42 * heightNorm + fresnel * 0.48);

  vec3 edgeDist = abs(vLocalPos);
  float bevelMask = smoothstep(0.41, 0.495, max(edgeDist.x, max(edgeDist.y, edgeDist.z)));
  emission += albedo * bevelMask * (0.04 + emitBase * 0.10);

  // Highlight states: 1=hover, 2=selected, 3=brush, 4=edited
  float flag = vFlags.x;
  if (flag > 0.5 && flag < 1.5) {
    emission += vec3(0.35, 0.85, 1.25) * (0.35 + fresnel * 0.9);
  } else if (flag >= 1.5 && flag < 2.5) {
    float beat = 0.78 + 0.22 * sin(uTime * 4.5);
    emission += vec3(0.25, 0.95, 1.35) * (0.55 + fresnel * 1.4) * beat;
  } else if (flag >= 2.5 && flag < 3.5) {
    float scan = 0.75 + 0.25 * sin(uTime * 7.0 + vWorldPos.x * 2.0);
    emission += vec3(1.45, 0.35, 0.85) * (0.75 + fresnel * 1.3) * scan;
  } else if (flag >= 3.5) {
    emission += vec3(0.10, 0.72, 0.46) * (0.05 + fresnel * 0.34);
  }

  float phase = vFlags.z;
  float actHeat = vFlags.w;
  if (uPulseWave >= 0.0) {
    float waveDist = abs(phase - uPulseWave);
    float wave = exp(-waveDist * waveDist * 48.0);
    vec3 waveCol = mix(vec3(0.18, 1.25, 1.45), vec3(1.65, 0.55, 0.18), uPulseMode);
    emission += waveCol * wave * (0.45 + actHeat * 1.1);
  } else if (actHeat > 0.02) {
    vec3 heatCol = mix(vec3(0.15, 0.95, 1.2), vec3(1.35, 0.65, 0.15), uPulseMode);
    emission += heatCol * actHeat * (0.28 + fresnel * 0.5);
  }

  vec3 col = ambient + Lo * rootAO + emission + vec3(0.20, 1.55, 1.95) * sliceGlow * 1.35;

  float camDist = length(uCameraPos - vWorldPos);
  float fog = 1.0 - exp(-camDist * camDist * 0.000085);
  col = mix(col, uBgHorizon, clamp(fog * 0.65, 0.0, 0.65));

  float alpha = vFlags.y;
  if (alpha < 0.99) {
    alpha = clamp(alpha + fresnel * 0.38, 0.08, 0.92);
  }
  fragColor = vec4(col, alpha);
}
`;

export const CONDUIT_VS = `#version 300 es
precision highp float;
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aMeta;
uniform mat4 uViewProj;
uniform float uMirrorY;
out vec3 vWorldPos;
out vec3 vNormal;
out vec3 vMeta;
void main() {
  vec3 wp = aPos;
  vec3 wn = aNormal;
  if (uMirrorY < 0.0) {
    wp.y = -wp.y;
    wn.y = -wn.y;
  }
  vWorldPos = wp;
  vNormal = wn;
  vMeta = aMeta;
  gl_Position = uViewProj * vec4(wp, 1.0);
}
`;

export const CONDUIT_FS = `#version 300 es
precision highp float;
in vec3 vWorldPos;
in vec3 vNormal;
in vec3 vMeta;
uniform vec3 uCameraPos;
uniform float uTime;
uniform float uPulseSpeed;
uniform float uPulseMode;
out vec4 fragColor;

void main() {
  vec3 N = normalize(vNormal);
  vec3 V = normalize(uCameraPos - vWorldPos);
  float fresnel = pow(1.0 - max(dot(N, V), 0.0), 2.4);

  float t = vMeta.x;
  float lane = vMeta.y;
  float dir = mix(1.0, -1.0, uPulseMode);
  float travel = fract(t * 2.5 - uTime * uPulseSpeed * dir);
  float packet = pow(smoothstep(0.62, 1.0, travel), 2.8) + pow(smoothstep(0.18, 0.0, travel), 2.0) * 0.45;

  vec3 baseCol = mix(vec3(0.08, 0.24, 0.36), vec3(0.32, 0.16, 0.08), uPulseMode);
  vec3 pulseCol = mix(vec3(0.25, 1.35, 1.65), vec3(1.85, 0.62, 0.20), uPulseMode);

  if (lane > 1.5 && lane < 2.5) {
    // Token -> Head attention flow ribbon (electric cyan -> violet)
    baseCol = mix(vec3(0.08, 0.36, 0.56), vec3(0.38, 0.16, 0.62), t);
    pulseCol = mix(vec3(0.28, 1.45, 1.85), vec3(1.15, 0.55, 1.95), t);
  } else if (lane >= 2.5 && lane < 3.5) {
    // Causal Circuit superhighway arc (molten amber -> rose gold)
    baseCol = vec3(0.42, 0.20, 0.07);
    pulseCol = mix(vec3(1.95, 1.18, 0.24), vec3(1.95, 0.36, 0.68), t);
  } else if (lane >= 3.5) {
    // Output unembedding logit beam (luminous emerald)
    baseCol = vec3(0.08, 0.36, 0.24);
    pulseCol = vec3(0.24, 1.75, 1.12);
  }

  vec3 col = baseCol * (0.55 + fresnel * 0.85) + pulseCol * packet * (0.45 + vMeta.z * 1.15);
  float alpha = clamp(0.34 + fresnel * 0.35 + packet * 0.55, 0.18, 0.96);
  fragColor = vec4(col, alpha);
}
`;

export const POST_VS = `#version 300 es
precision highp float;
const vec2 POS[3] = vec2[](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
out vec2 vUv;
void main() {
  vUv = POS[gl_VertexID] * 0.5 + 0.5;
  gl_Position = vec4(POS[gl_VertexID], 0.0, 1.0);
}
`;

export const BRIGHT_PASS_FS = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uScene;
uniform float uThreshold;
out vec4 fragColor;

void main() {
  vec3 col = texture(uScene, vUv).rgb;
  float luma = dot(col, vec3(0.2126, 0.7152, 0.0722));
  float knee = 0.45;
  float soft = clamp(luma - uThreshold + knee, 0.0, 2.0 * knee);
  soft = (soft * soft) / max(4.0 * knee, 0.0001);
  float contrib = max(soft, luma - uThreshold) / max(luma, 0.0001);
  fragColor = vec4(col * max(contrib, 0.0), 1.0);
}
`;

export const BLUR_FS = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uTex;
uniform vec2 uTexelStep;
out vec4 fragColor;

void main() {
  vec3 sum = texture(uTex, vUv).rgb * 0.227027;
  sum += texture(uTex, vUv + uTexelStep * 1.384615).rgb * 0.316216;
  sum += texture(uTex, vUv - uTexelStep * 1.384615).rgb * 0.316216;
  sum += texture(uTex, vUv + uTexelStep * 3.230769).rgb * 0.070270;
  sum += texture(uTex, vUv - uTexelStep * 3.230769).rgb * 0.070270;
  fragColor = vec4(sum, 1.0);
}
`;

export const COMPOSITE_FS = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uScene;
uniform sampler2D uDepth;
uniform sampler2D uBloom1;
uniform sampler2D uBloom2;
uniform vec2 uTexelSize;
uniform float uExposure;
uniform float uBloomStrength;
uniform float uSsaoStrength;
uniform float uDofStrength;
uniform float uFocusDist;
out vec4 fragColor;

const vec2 POISSON[8] = vec2[](
  vec2(-0.707, -0.707), vec2(0.707, -0.707),
  vec2(-0.707,  0.707), vec2(0.707,  0.707),
  vec2(-1.0,    0.0  ), vec2(1.0,    0.0  ),
  vec2( 0.0,   -1.0  ), vec2(0.0,    1.0  )
);

float linearizeDepth(float d) {
  float near = 0.2;
  float far = 260.0;
  float z = d * 2.0 - 1.0;
  return (2.0 * near * far) / max(far + near - z * (far - near), 0.0001);
}

vec3 acesFilm(vec3 x) {
  float a = 2.51;
  float b = 0.03;
  float c = 2.43;
  float d = 0.59;
  float e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
  float rawD = texture(uDepth, vUv).r;
  float linD = linearizeDepth(rawD);

  // 1. Optional Optical Bokeh Depth-of-Field gather around camera focal plane
  vec3 scene = texture(uScene, vUv).rgb;
  if (uDofStrength > 0.01 && rawD < 0.9995) {
    float coc = clamp(abs(linD - uFocusDist) / max(uFocusDist * 0.55, 2.0), 0.0, 1.0) * uDofStrength;
    if (coc > 0.03) {
      vec3 acc = scene;
      float wSum = 1.0;
      float radius = coc * 5.2;
      for (int i = 0; i < 8; i++) {
        vec2 off = POISSON[i] * uTexelSize * radius;
        acc += texture(uScene, clamp(vUv + off, 0.001, 0.999)).rgb;
        wSum += 1.0;
      }
      scene = acc / wSum;
    }
  }

  // 2. Fast Screen-Space Depth-Crevice Ambient Occlusion (SSAO)
  float ao = 1.0;
  if (uSsaoStrength > 0.01 && rawD < 0.998) {
    float occ = 0.0;
    float radiusPx = clamp(24.0 / max(linD, 3.0), 1.5, 5.5);
    for (int i = 0; i < 8; i++) {
      vec2 suv = clamp(vUv + POISSON[i] * uTexelSize * radiusPx, 0.001, 0.999);
      float sd = linearizeDepth(texture(uDepth, suv).r);
      float diff = linD - sd;
      if (diff > 0.04 && diff < 1.8) {
        occ += smoothstep(0.04, 0.85, diff) * (1.0 - smoothstep(0.85, 1.8, diff));
      }
    }
    ao = 1.0 - clamp((occ / 8.0) * uSsaoStrength * 0.85, 0.0, 0.52);
  }

  vec3 bloom = texture(uBloom1, vUv).rgb * 0.58 + texture(uBloom2, vUv).rgb * 0.42;
  vec3 hdr = scene * ao + bloom * uBloomStrength;

  // 3. ACES Filmic Tonemapping + Optical Vignette + Micro-dither
  vec3 mapped = acesFilm(hdr * uExposure);
  vec2 q = vUv - 0.5;
  float vig = 1.0 - dot(q, q) * 0.34;
  mapped *= clamp(vig, 0.72, 1.0);
  mapped = pow(max(mapped, 0.0), vec3(1.0 / 2.2));
  float dither = (fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453) - 0.5) / 255.0;
  fragColor = vec4(clamp(mapped + dither, 0.0, 1.0), 1.0);
}
`;

export const PARTICLE_VS = `#version 300 es
precision highp float;
layout(location = 0) in vec4 aSeed; // x=laneX, y=height, z=phase, w=speed
uniform mat4 uViewProj;
uniform float uTime;
uniform float uSpanX;
uniform float uSpanZ;
uniform float uPulseMode;
uniform float uBurst;
out float vAlpha;
out vec3 vCol;

void main() {
  float dir = mix(1.0, -1.0, uPulseMode);
  float t = fract(aSeed.z + uTime * aSeed.w * (0.14 + uBurst * 0.42) * dir);
  float x = aSeed.x * uSpanX + sin(uTime * 0.7 + aSeed.z * 18.0) * 0.35;
  float y = 0.55 + aSeed.y * (3.6 + uBurst * 1.4) + sin(t * 3.14159) * 1.1;
  float z = (t * 2.0 - 1.0) * uSpanZ;

  vec4 clip = uViewProj * vec4(x, y, z, 1.0);
  gl_Position = clip;
  gl_PointSize = clamp((38.0 + uBurst * 26.0) / max(clip.w, 2.0), 2.0, 9.0);

  float edgeFade = smoothstep(0.0, 0.14, t) * smoothstep(1.0, 0.86, t);
  vAlpha = edgeFade * (0.28 + uBurst * 0.48);
  vec3 fwdCol = mix(vec3(0.24, 0.86, 1.0), vec3(0.75, 0.48, 1.0), fract(aSeed.y * 3.7));
  vec3 bwdCol = mix(vec3(1.0, 0.58, 0.18), vec3(1.0, 0.28, 0.58), fract(aSeed.y * 3.7));
  vCol = mix(fwdCol, bwdCol, uPulseMode);
}
`;

export const PARTICLE_FS = `#version 300 es
precision highp float;
in float vAlpha;
in vec3 vCol;
out vec4 fragColor;
void main() {
  vec2 p = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(p, p);
  if (r2 > 1.0) discard;
  float core = exp(-r2 * 4.5);
  fragColor = vec4(vCol * (0.9 + core * 1.4), core * vAlpha);
}
`;
