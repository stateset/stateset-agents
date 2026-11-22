/**
 * Custom GLSL ES 3.00 Shaders for the StateSet 3D Weight Studio.
 *
 * Implements:
 * - Cook-Torrance GGX physically-based instanced crystal & machined chassis shading
 * - Procedural HDR studio softbox IBL environment reflections
 * - Anodized brushed-metal optical bench floor with analytic soft contact shadows
 * - Volumetric curved optical conduits with traveling forward/backprop photon waves
 * - Multi-pass HDR Bright-Pass, 9-tap Separable Gaussian Bloom, and ACES Filmic Tonemapping
 */

export const STUDIO_RIGS = {
  obsidian: {
    name: 'Obsidian Lab',
    bgTop: [0.007, 0.011, 0.022],
    bgHorizon: [0.020, 0.032, 0.056],
    floorColor: [0.014, 0.019, 0.030],
    gridColor: [0.09, 0.24, 0.44],
    keyDir: [0.58, 0.72, 0.36],
    keyColor: [1.35, 1.24, 1.08],
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

  // Subtle studio dome softbox atmospheric glow
  float keyGlow = pow(max(dot(dir, normalize(uKeyDir)), 0.0), 18.0);
  float rimGlow = pow(max(dot(dir, normalize(uRimDir)), 0.0), 14.0);
  col += uKeyColor * keyGlow * 0.045;
  col += uRimColor * rimGlow * 0.038;

  // Horizon studio ring haze
  float band = exp(-abs(dir.y) * 9.0);
  col += mix(uBgHorizon, uRimColor * 0.14, 0.22) * band * 0.35;

  fragColor = vec4(col, 1.0);
}
`;

export const FLOOR_VS = `#version 300 es
precision highp float;
layout(location = 0) in vec3 aPos;
uniform mat4 uViewProj;
uniform float uFloorY;
out vec3 vWorldPos;
void main() {
  vec3 wp = vec3(aPos.x, uFloorY, aPos.z);
  vWorldPos = wp;
  gl_Position = uViewProj * vec4(wp, 1.0);
}
`;

export const FLOOR_FS = `#version 300 es
precision highp float;
in vec3 vWorldPos;
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
out vec4 fragColor;

float gridLine(vec2 p, float stepSize, float width) {
  vec2 coord = p / stepSize;
  vec2 grid = abs(fract(coord - 0.5) - 0.5) / max(fwidth(coord), 0.0005);
  float line = min(grid.x, grid.y);
  return 1.0 - smoothstep(0.0, width, line);
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

  // Analytic soft contact shadow under the 3D neural bench footprint
  vec2 dBox = abs(vWorldPos.xz) - uFootprintSize;
  float sd = length(max(dBox, 0.0)) + min(max(dBox.x, dBox.y), 0.0);
  float shadow = smoothstep(-0.8, 3.8, sd);
  float ao = mix(0.42, 1.0, shadow);

  // Specular studio softbox reflections on the optical bench
  vec3 L1 = normalize(uKeyDir);
  vec3 H1 = normalize(L1 + V);
  float spec1 = pow(max(dot(N, H1), 0.0), 42.0) * 0.22;

  vec3 L2 = normalize(uRimDir);
  vec3 H2 = normalize(L2 + V);
  float spec2 = pow(max(dot(N, H2), 0.0), 28.0) * 0.16;

  float fresnel = pow(1.0 - max(dot(N, V), 0.0), 4.0);
  vec3 refl = mix(baseCol, uBgHorizon * 1.15, fresnel * 0.55);

  // Subtle colored bounce glow from active 3D crystals above
  float glowDist = length(vWorldPos.xz - uGlowCenter.xz);
  vec3 bounce = uGlowColor * exp(-glowDist * glowDist * 0.014) * 0.28;

  vec3 col = refl * ao + (uKeyColor * spec1 + uRimColor * spec2) * ao + gridEmit + bounce;

  // Seamless atmospheric horizon blend
  float fog = 1.0 - exp(-dist * dist * 0.00028);
  col = mix(col, uBgHorizon, clamp(fog, 0.0, 1.0));

  fragColor = vec4(col, 1.0);
}
`;

/**
 * Instanced PBR Beveled Crystal & Chassis Shader.
 * Per-instance attributes (stride = 12 floats / 3 vec4s):
 *   aInstPosScale: xyz = center position, w = scaleX
 *   aInstDimMat:   x = scaleY, y = scaleZ, z = roughness (0..1), w = metallic (0..1)
 *   aInstColor:    rgb = base PBR albedo, a = emissive/heat intensity (0..3)
 *   aInstFlags:    x = highlight (0=normal, 1=hover, 2=selected, 3=brush, 4=edited),
 *                  y = alpha (1.0 opaque, <1.0 translucent glass/ghost),
 *                  z = pulse phase (layer/rank normalized 0..1),
 *                  w = extra intensity (activation / gradient heat)
 */
export const INSTANCED_PBR_VS = `#version 300 es
precision highp float;

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

layout(location = 2) in vec4 aInstPosScale;
layout(location = 3) in vec4 aInstDimMat;
layout(location = 4) in vec4 aInstColor;
layout(location = 5) in vec4 aInstFlags;

uniform mat4 uViewProj;
uniform float uTime;

out vec3 vWorldPos;
out vec3 vNormal;
out vec3 vLocalPos;
out vec4 vColor;
out vec4 vMat;     // x=roughness, y=metallic, z=scaleY, w=heightNorm
out vec4 vFlags;

void main() {
  vec3 scale = vec3(aInstPosScale.w, aInstDimMat.x, aInstDimMat.y);
  vec3 worldPos = aInstPosScale.xyz + aPos * scale;
  // Non-uniform normal scaling
  vec3 scaledNormal = normalize(aNormal / max(scale, vec3(0.0001)));

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
uniform float uPulseWave;     // -1 inactive, >= 0 traveling forward/backprop wave position
uniform float uPulseMode;     // 0=forward cyan-emerald, 1=backprop amber-crimson

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

// Procedural HDR studio IBL softbox reflection
vec3 sampleStudioEnv(vec3 R, float roughness) {
  float up = clamp(R.y * 0.65 + 0.35, 0.0, 1.0);
  vec3 env = mix(uBgHorizon * 0.9, uBgTop * 1.4, up);

  // Overhead rectangular softbox strip
  float strip = smoothstep(0.68 - roughness * 0.45, 0.96, R.y) *
                smoothstep(0.25, 0.85, 1.0 - abs(R.z));
  env += vec3(1.15, 1.22, 1.35) * strip * (1.2 - roughness * 0.7);

  // Key & Rim softboxes
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
  vec3 N = normalize(vNormal);
  vec3 V = normalize(uCameraPos - vWorldPos);
  float NdotV = max(dot(N, V), 0.001);

  vec3 albedo = vColor.rgb;
  float emitBase = vColor.a;
  float roughness = clamp(vMat.x, 0.06, 0.95);
  float metallic = clamp(vMat.y, 0.0, 1.0);
  float heightNorm = vMat.w;

  // Height-based cavity AO at the crystal root + rich vertical crystal depth
  float rootAO = mix(0.42, 1.0, smoothstep(0.0, 0.52, heightNorm));
  albedo *= mix(0.72, 1.08, heightNorm);

  vec3 F0 = mix(vec3(0.045), albedo, metallic);

  vec3 Lo = vec3(0.0);
  Lo += evalLight(N, V, normalize(uKeyDir), uKeyColor, albedo, roughness, metallic, F0);
  Lo += evalLight(N, V, normalize(uRimDir), uRimColor, albedo, roughness, metallic, F0);
  Lo += evalLight(N, V, normalize(uFillDir), uFillColor, albedo, roughness, metallic, F0);

  // Procedural HDR studio IBL
  vec3 R = reflect(-V, N);
  vec3 envRefl = sampleStudioEnv(R, roughness);
  vec3 F_env = F_Schlick(NdotV, F0);
  vec3 kD_env = (vec3(1.0) - F_env) * (1.0 - metallic);
  vec3 ambient = (kD_env * albedo * (uAmbient * 1.9 + uBgHorizon * 0.25) + F_env * envRefl * (0.72 - roughness * 0.42)) * rootAO;

  // Subsurface / optical crystal emission concentrated toward upper facets & Fresnel edges
  float fresnel = pow(1.0 - NdotV, 3.2);
  vec3 emission = albedo * emitBase * (0.24 + 0.42 * heightNorm + fresnel * 0.48);

  // Crisp chamfer micro-edge specular accent
  vec3 edgeDist = abs(vLocalPos);
  float bevelMask = smoothstep(0.41, 0.495, max(edgeDist.x, max(edgeDist.y, edgeDist.z)));
  emission += albedo * bevelMask * (0.04 + emitBase * 0.10);

  // Highlight states: 1=hover, 2=selected, 3=brush, 4=edited
  float flag = vFlags.x;
  if (flag > 0.5 && flag < 1.5) {
    // Hover laser sheen
    emission += vec3(0.35, 0.85, 1.25) * (0.35 + fresnel * 0.9);
  } else if (flag >= 1.5 && flag < 2.5) {
    // Selected precision lock-on
    float beat = 0.78 + 0.22 * sin(uTime * 4.5);
    emission += vec3(0.25, 0.95, 1.35) * (0.55 + fresnel * 1.4) * beat;
  } else if (flag >= 2.5 && flag < 3.5) {
    // 3D Brush surgical target selection
    float scan = 0.75 + 0.25 * sin(uTime * 7.0 + vWorldPos.x * 2.0);
    emission += vec3(1.45, 0.35, 0.85) * (0.75 + fresnel * 1.3) * scan;
  } else if (flag >= 3.5) {
    // Surgically edited tensor emerald/gold contour
    emission += vec3(0.18, 1.15, 0.72) * (0.22 + fresnel * 0.75);
  }

  // Traveling activation / backpropagation wave along the neural stack!
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

  vec3 col = ambient + Lo * rootAO + emission;

  // Gentle distance atmospheric integration
  float camDist = length(uCameraPos - vWorldPos);
  float fog = 1.0 - exp(-camDist * camDist * 0.000085);
  col = mix(col, uBgHorizon, clamp(fog * 0.65, 0.0, 0.65));

  float alpha = vFlags.y;
  if (alpha < 0.99) {
    // Translucent optical glass / pre-edit ghost surface
    alpha = clamp(alpha + fresnel * 0.38, 0.08, 0.92);
  }
  fragColor = vec4(col, alpha);
}
`;

export const CONDUIT_VS = `#version 300 es
precision highp float;
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aMeta; // x = path t (0..1), y = lane type, z = intensity
uniform mat4 uViewProj;
out vec3 vWorldPos;
out vec3 vNormal;
out vec3 vMeta;
void main() {
  vWorldPos = aPos;
  vNormal = aNormal;
  vMeta = aMeta;
  gl_Position = uViewProj * vec4(aPos, 1.0);
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
uniform float uPulseMode; // 0 = forward cyan/emerald, 1 = backprop amber/crimson
out vec4 fragColor;

void main() {
  vec3 N = normalize(vNormal);
  vec3 V = normalize(uCameraPos - vWorldPos);
  float fresnel = pow(1.0 - max(dot(N, V), 0.0), 2.4);

  float t = vMeta.x;
  float dir = mix(1.0, -1.0, uPulseMode);
  float travel = fract(t * 2.5 - uTime * uPulseSpeed * dir);
  float packet = pow(smoothstep(0.65, 1.0, travel), 3.0) + pow(smoothstep(0.18, 0.0, travel), 2.0) * 0.5;

  vec3 baseCol = mix(vec3(0.08, 0.24, 0.36), vec3(0.32, 0.16, 0.08), uPulseMode);
  vec3 pulseCol = mix(vec3(0.25, 1.35, 1.65), vec3(1.85, 0.62, 0.20), uPulseMode);

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
uniform sampler2D uBloom1;
uniform sampler2D uBloom2;
uniform float uExposure;
uniform float uBloomStrength;
out vec4 fragColor;

vec3 acesFilm(vec3 x) {
  float a = 2.51;
  float b = 0.03;
  float c = 2.43;
  float d = 0.59;
  float e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
  vec3 scene = texture(uScene, vUv).rgb;
  vec3 bloom = texture(uBloom1, vUv).rgb * 0.58 + texture(uBloom2, vUv).rgb * 0.42;
  vec3 hdr = scene + bloom * uBloomStrength;

  // ACES Filmic Tonemapping
  vec3 mapped = acesFilm(hdr * uExposure);

  // Subtle precision optical vignette
  vec2 q = vUv - 0.5;
  float vig = 1.0 - dot(q, q) * 0.36;
  mapped *= clamp(vig, 0.72, 1.0);

  // sRGB gamma encode + subtle micro-dither to prevent dark gradient banding
  mapped = pow(max(mapped, 0.0), vec3(1.0 / 2.2));
  float dither = (fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453) - 0.5) / 255.0;
  fragColor = vec4(clamp(mapped + dither, 0.0, 1.0), 1.0);
}
`;
