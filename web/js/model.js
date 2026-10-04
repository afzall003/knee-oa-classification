/* KneeGrade inference in the browser with TensorFlow.js.
 *
 * Each CNN is rebuilt from exported weights: 4 blocks of 2 x (3x3 conv + folded batch norm + ReLU),
 * 2x2 max pooling, global average pooling, dense softmax.
 *
 * Grad-CAM uses an exact closed form instead of automatic differentiation. With max pooling,
 * global average pooling and one dense layer after the last conv, the mean gradient of
 * softmax output c with respect to feature map j is proportional to
 *     p_c * (W[j, c] - sum_k W[j, k] * p_k),
 * which is all Grad-CAM needs (the constant factor cancels when the map is normalized).
 */
(function (root) {
  "use strict";
  const SIZE = 224;

  async function loadMember(tf, baseUrl, spec) {
    const res = await fetch(baseUrl + spec.file);
    if (!res.ok) throw new Error(`Model file ${spec.file} couldn't be downloaded.`);
    const data = new Float32Array(await res.arrayBuffer());
    const take = (t) => data.subarray(t.offset, t.offset + t.shape.reduce((a, b) => a * b, 1));
    return {
      name: spec.name,
      convs: spec.convs.map((c) => ({ kernel: tf.tensor(take(c.kernel), c.kernel.shape), bias: tf.tensor(take(c.bias), c.bias.shape) })),
      denseKernel: tf.tensor(take(spec.dense.kernel), spec.dense.kernel.shape),
      denseBias: tf.tensor(take(spec.dense.bias), spec.dense.bias.shape),
      W: Float32Array.from(take(spec.dense.kernel)),          // [channels, 5], row-major
      channels: spec.dense.kernel.shape[0],
    };
  }

  // Returns { probs: Float32Array(5), feat: Float32Array(28*28*C) } for one preprocessed image.
  function forward(tf, m, pixels) {
    const [probsT, featT] = tf.tidy(() => {
      let h = tf.tensor4d(Float32Array.from(pixels, (v) => v / 255), [1, SIZE, SIZE, 1]);
      let feat = null;
      m.convs.forEach((c, i) => {
        h = tf.relu(tf.add(tf.conv2d(h, c.kernel, 1, "same"), c.bias));
        if (i === m.convs.length - 1) feat = h;
        if (i % 2 === 1) h = tf.maxPool(h, 2, 2, "valid");
      });
      const pooled = tf.mean(h, [1, 2]);
      return [tf.softmax(tf.add(tf.matMul(pooled, m.denseKernel), m.denseBias)), feat];
    });
    const out = { probs: probsT.dataSync().slice(), feat: featT.dataSync().slice(), featSize: featT.shape[1] };
    probsT.dispose(); featT.dispose();
    return out;
  }

  // Bilinear resize of a float map, with OpenCV's half-pixel convention (cv2.INTER_LINEAR).
  function resizeLinear(src, s, d) {
    const out = new Float32Array(d * d), scale = s / d;
    const idx = (o) => {
      let f = (o + 0.5) * scale - 0.5, i = Math.floor(f); f -= i;
      if (i < 0) { i = 0; f = 0; }
      if (i >= s - 1) { i = s - 1; f = 0; }
      return [i, Math.min(i + 1, s - 1), f];
    };
    const xs = Array.from({ length: d }, (_, o) => idx(o));
    for (let y = 0; y < d; y++) {
      const [y0, y1, fy] = idx(y);
      for (let x = 0; x < d; x++) {
        const [x0, x1, fx] = xs[x];
        const top = src[y0 * s + x0] * (1 - fx) + src[y0 * s + x1] * fx;
        const bot = src[y1 * s + x0] * (1 - fx) + src[y1 * s + x1] * fx;
        out[y * d + x] = top * (1 - fy) + bot * fy;
      }
    }
    return out;
  }

  function normalize(map) {
    let max = 0;
    for (const v of map) if (v > max) max = v;
    if (max > 0) for (let i = 0; i < map.length; i++) map[i] /= max;
    return map;
  }

  // Grad-CAM for one member, at full image resolution, values 0..1.
  function gradcam(m, out, cls) {
    const C = m.channels, K = 5, p = out.probs, s = out.featSize;
    const w = new Float64Array(C);
    for (let j = 0; j < C; j++) {
      let mix = 0;
      for (let k = 0; k < K; k++) mix += m.W[j * K + k] * p[k];
      w[j] = p[cls] * (m.W[j * K + cls] - mix);
    }
    const cam = new Float32Array(s * s);
    for (let q = 0; q < s * s; q++) {
      let v = 0;
      const base = q * C;
      for (let j = 0; j < C; j++) v += w[j] * out.feat[base + j];
      cam[q] = v > 0 ? v : 0;
    }
    return normalize(resizeLinear(cam, s, SIZE));
  }

  // Jet colour map with transparency that grows with the signal (matches the server version).
  function heatmapRGBA(cam) {
    const rgba = new Uint8ClampedArray(SIZE * SIZE * 4);
    const clamp = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
    for (let i = 0; i < cam.length; i++) {
      const t = cam[i];
      rgba[i * 4] = 255 * clamp(1.5 - Math.abs(4 * t - 3));
      rgba[i * 4 + 1] = 255 * clamp(1.5 - Math.abs(4 * t - 2));
      rgba[i * 4 + 2] = 255 * clamp(1.5 - Math.abs(4 * t - 1));
      rgba[i * 4 + 3] = 255 * clamp(t * 1.5);
    }
    return rgba;
  }

  function jointShare(cam, band) {
    const [r0, r1, c0, c1] = band;
    let inside = 0, total = 0;
    for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
      const v = cam[y * SIZE + x];
      total += v;
      if (y >= r0 && y < r1 && x >= c0 && x < c1) inside += v;
    }
    return total > 0 ? inside / total : null;
  }

  // Full pipeline for one preprocessed image: ensemble, rarity correction, optional Grad-CAM.
  function analyze(tf, members, cfg, pixels, withHeatmap) {
    const outs = members.map((m) => forward(tf, m, pixels));
    const probs = Array.from({ length: 5 }, (_, k) => outs.reduce((s, o) => s + o.probs[k], 0) / outs.length);
    const counts = cfg.train_counts, total = counts.reduce((a, b) => a + b, 0);
    const raw = probs.map((p, k) => p / (counts[k] / total));
    const rawSum = raw.reduce((a, b) => a + b, 0);
    const adjusted = raw.map((v) => v / rawSum);
    const argmax = (a) => a.indexOf(Math.max(...a));
    const grade = argmax(adjusted);
    const result = { grade, topGrade: argmax(probs), probabilities: probs, adjustedScores: adjusted, cam: null, jointShare: null };
    if (withHeatmap) {
      const cams = members.map((m, i) => gradcam(m, outs[i], grade));
      const mean = normalize(cams[0].map((v, i) => cams.reduce((s, c) => s + c[i], 0) / cams.length));
      result.cam = mean;
      result.jointShare = jointShare(mean, cfg.joint_band);
    }
    return result;
  }

  const api = { loadMember, forward, gradcam, analyze, heatmapRGBA, resizeLinear };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.KneeModel = api;
})(typeof self !== "undefined" ? self : this);
