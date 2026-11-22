/**
 * High-performance 3D math, camera matrices, and ray-box picking utilities.
 */

export function vec3(x = 0, y = 0, z = 0) {
  return new Float32Array([x, y, z]);
}

export function sub3(a, b, out = new Float32Array(3)) {
  out[0] = a[0] - b[0];
  out[1] = a[1] - b[1];
  out[2] = a[2] - b[2];
  return out;
}

export function add3(a, b, out = new Float32Array(3)) {
  out[0] = a[0] + b[0];
  out[1] = a[1] + b[1];
  out[2] = a[2] + b[2];
  return out;
}

export function scale3(a, s, out = new Float32Array(3)) {
  out[0] = a[0] * s;
  out[1] = a[1] * s;
  out[2] = a[2] * s;
  return out;
}

export function dot3(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function cross3(a, b, out = new Float32Array(3)) {
  const ax = a[0], ay = a[1], az = a[2];
  const bx = b[0], by = b[1], bz = b[2];
  out[0] = ay * bz - az * by;
  out[1] = az * bx - ax * bz;
  out[2] = ax * by - ay * bx;
  return out;
}

export function len3(a) {
  return Math.hypot(a[0], a[1], a[2]);
}

export function normalize3(a, out = new Float32Array(3)) {
  const l = Math.hypot(a[0], a[1], a[2]);
  if (l > 1e-8) {
    out[0] = a[0] / l;
    out[1] = a[1] / l;
    out[2] = a[2] / l;
  } else {
    out[0] = 0; out[1] = 1; out[2] = 0;
  }
  return out;
}

export function mat4Identity(out = new Float32Array(16)) {
  out.fill(0);
  out[0] = 1; out[5] = 1; out[10] = 1; out[15] = 1;
  return out;
}

export function mat4Perspective(fovRad, aspect, near, far, out = new Float32Array(16)) {
  const f = 1.0 / Math.tan(fovRad * 0.5);
  const nf = 1.0 / (near - far);
  out.fill(0);
  out[0] = f / Math.max(aspect, 1e-4);
  out[5] = f;
  out[10] = (far + near) * nf;
  out[11] = -1.0;
  out[14] = 2.0 * far * near * nf;
  return out;
}

export function mat4LookAt(eye, target, up, out = new Float32Array(16)) {
  const z = normalize3(sub3(eye, target));
  const x = normalize3(cross3(up, z));
  const y = cross3(z, x);

  out[0] = x[0]; out[1] = y[0]; out[2] = z[0]; out[3] = 0;
  out[4] = x[1]; out[5] = y[1]; out[6] = z[1]; out[7] = 0;
  out[8] = x[2]; out[9] = y[2]; out[10] = z[2]; out[11] = 0;
  out[12] = -dot3(x, eye);
  out[13] = -dot3(y, eye);
  out[14] = -dot3(z, eye);
  out[15] = 1;
  return out;
}

export function mat4Multiply(a, b, out = new Float32Array(16)) {
  const res = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      res[c * 4 + r] =
        a[0 * 4 + r] * b[c * 4 + 0] +
        a[1 * 4 + r] * b[c * 4 + 1] +
        a[2 * 4 + r] * b[c * 4 + 2] +
        a[3 * 4 + r] * b[c * 4 + 3];
    }
  }
  out.set(res);
  return out;
}

export function mat4Invert(m, out = new Float32Array(16)) {
  const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
  const a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
  const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
  const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];

  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;

  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (Math.abs(det) < 1e-12) return null;
  det = 1.0 / det;

  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
  return out;
}

export function projectPoint(worldPos, viewProj, width, height) {
  const x = worldPos[0], y = worldPos[1], z = worldPos[2];
  const cx = viewProj[0] * x + viewProj[4] * y + viewProj[8] * z + viewProj[12];
  const cy = viewProj[1] * x + viewProj[5] * y + viewProj[9] * z + viewProj[13];
  const cz = viewProj[2] * x + viewProj[6] * y + viewProj[10] * z + viewProj[14];
  const cw = viewProj[3] * x + viewProj[7] * y + viewProj[11] * z + viewProj[15];
  if (cw <= 0.001) return null;
  const ndcX = cx / cw;
  const ndcY = cy / cw;
  const ndcZ = cz / cw;
  if (ndcZ < -1 || ndcZ > 1) return null;
  return {
    x: (ndcX * 0.5 + 0.5) * width,
    y: (0.5 - ndcY * 0.5) * height,
    depth: ndcZ,
  };
}

export function screenToRay(ndcX, ndcY, invViewProj) {
  const unproject = (x, y, z) => {
    const rx = invViewProj[0] * x + invViewProj[4] * y + invViewProj[8] * z + invViewProj[12];
    const ry = invViewProj[1] * x + invViewProj[5] * y + invViewProj[9] * z + invViewProj[13];
    const rz = invViewProj[2] * x + invViewProj[6] * y + invViewProj[10] * z + invViewProj[14];
    const rw = invViewProj[3] * x + invViewProj[7] * y + invViewProj[11] * z + invViewProj[15];
    return vec3(rx / rw, ry / rw, rz / rw);
  };
  const nearP = unproject(ndcX, ndcY, -1.0);
  const farP = unproject(ndcX, ndcY, 1.0);
  return {
    origin: nearP,
    dir: normalize3(sub3(farP, nearP)),
  };
}

export function rayIntersectsAABB(origin, dir, minB, maxB) {
  let tmin = -Infinity;
  let tmax = Infinity;
  for (let i = 0; i < 3; i++) {
    if (Math.abs(dir[i]) < 1e-8) {
      if (origin[i] < minB[i] || origin[i] > maxB[i]) return null;
    } else {
      const invD = 1.0 / dir[i];
      let t1 = (minB[i] - origin[i]) * invD;
      let t2 = (maxB[i] - origin[i]) * invD;
      if (t1 > t2) { const tmp = t1; t1 = t2; t2 = tmp; }
      tmin = Math.max(tmin, t1);
      tmax = Math.min(tmax, t2);
      if (tmin > tmax) return null;
    }
  }
  return tmax >= 0 ? (tmin >= 0 ? tmin : tmax) : null;
}

export function rayIntersectsPlaneY(origin, dir, planeY = 0) {
  if (Math.abs(dir[1]) < 1e-6) return null;
  const t = (planeY - origin[1]) / dir[1];
  if (t < 0) return null;
  return vec3(origin[0] + dir[0] * t, planeY, origin[2] + dir[2] * t);
}

export function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

export function lerp(a, b, t) {
  return a + (b - a) * t;
}
