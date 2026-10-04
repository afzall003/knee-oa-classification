/* KneeGrade preprocessing, ported from src/preprocessing.py to run in the browser.
 * Matches the Python pipeline pixel for pixel on 224x224 images:
 * polarity fix -> 1st-99th percentile stretch measured on the central joint region -> CLAHE (2.0, 8x8).
 */
(function (root) {
  "use strict";
  const SIZE = 224;

  /* ---------- Decoding and resizing ---------- */

  // RGB to gray with OpenCV's fixed-point coefficients (cv2.COLOR_RGB2GRAY).
  function rgbaToGray(rgba, n) {
    const out = new Uint8Array(n);
    for (let i = 0, p = 0; i < n; i++, p += 4) {
      const r = rgba[p], g = rgba[p + 1], b = rgba[p + 2];
      out[i] = r === g && g === b ? r : (r * 4899 + g * 9617 + b * 1868 + 8192) >> 14;
    }
    return out;
  }

  // Area-weighted downscale (like cv2.INTER_AREA); bilinear when enlarging.
  function resizeTo224(gray, w, h) {
    if (w === SIZE && h === SIZE) return gray;
    const out = new Uint8Array(SIZE * SIZE);
    const sx = w / SIZE, sy = h / SIZE;
    if (sx >= 1 && sy >= 1) {
      for (let y = 0; y < SIZE; y++) {
        const y0 = y * sy, y1 = y0 + sy;
        for (let x = 0; x < SIZE; x++) {
          const x0 = x * sx, x1 = x0 + sx;
          let sum = 0, area = 0;
          for (let yy = Math.floor(y0); yy < Math.min(Math.ceil(y1), h); yy++) {
            const wy = Math.min(yy + 1, y1) - Math.max(yy, y0);
            for (let xx = Math.floor(x0); xx < Math.min(Math.ceil(x1), w); xx++) {
              const wx = Math.min(xx + 1, x1) - Math.max(xx, x0);
              sum += gray[yy * w + xx] * wx * wy;
              area += wx * wy;
            }
          }
          out[y * SIZE + x] = Math.min(255, Math.round(sum / area));
        }
      }
    } else {
      for (let y = 0; y < SIZE; y++) {
        let fy = (y + 0.5) * sy - 0.5; let y0 = Math.floor(fy); fy -= y0;
        if (y0 < 0) { y0 = 0; fy = 0; } if (y0 >= h - 1) { y0 = h - 1; fy = 0; }
        const y1 = Math.min(y0 + 1, h - 1);
        for (let x = 0; x < SIZE; x++) {
          let fx = (x + 0.5) * sx - 0.5; let x0 = Math.floor(fx); fx -= x0;
          if (x0 < 0) { x0 = 0; fx = 0; } if (x0 >= w - 1) { x0 = w - 1; fx = 0; }
          const x1 = Math.min(x0 + 1, w - 1);
          const top = gray[y0 * w + x0] * (1 - fx) + gray[y0 * w + x1] * fx;
          const bot = gray[y1 * w + x0] * (1 - fx) + gray[y1 * w + x1] * fx;
          out[y * SIZE + x] = Math.round(top * (1 - fy) + bot * fy);
        }
      }
    }
    return out;
  }

  /* ---------- Pipeline steps ---------- */

  // Centre minus edge brightness. Negative means the X-ray is inverted.
  function polarityScore(img) {
    let cs = 0, cn = 0, bs = 0, bn = 0;
    for (let y = 40; y < 184; y++) for (let x = 62; x < 162; x++) { cs += img[y * SIZE + x]; cn++; }
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < 20; x++) { bs += img[y * SIZE + x]; bn++; }
      for (let x = SIZE - 20; x < SIZE; x++) { bs += img[y * SIZE + x]; bn++; }
    }
    return cs / cn - bs / bn;
  }

  function fixPolarity(img) {
    if (polarityScore(img) >= 0) return img;
    const out = new Uint8Array(img.length);
    for (let i = 0; i < img.length; i++) out[i] = 255 - img[i];
    return out;
  }

  // numpy.percentile with the default 'linear' method, reproduced step by step.
  function percentileLinear(sorted, q) {
    const n = sorted.length;
    const qq = q / 100;
    const vi = n * qq + (1 + qq * (1 - 1 - 1)) - 1;
    let prev = Math.floor(vi);
    const gamma = vi - prev;
    let next = prev + 1;
    if (vi >= n - 1) { prev = n - 1; next = n - 1; }
    if (vi < 0) { prev = 0; next = 0; }
    const a = sorted[prev], b = sorted[next], diff = b - a;
    return gamma >= 0.5 ? b - diff * (1 - gamma) : a + diff * gamma;
  }

  function stretch(img) {
    const counts = new Uint32Array(256);
    for (let y = 40; y < 184; y++) for (let x = 40; x < 184; x++) counts[img[y * SIZE + x]]++;
    const sorted = new Uint8Array(144 * 144);
    for (let v = 0, k = 0; v < 256; v++) for (let c = 0; c < counts[v]; c++) sorted[k++] = v;
    const pLo = percentileLinear(sorted, 1), pHi = percentileLinear(sorted, 99);
    if (pHi <= pLo) return img;
    const out = new Uint8Array(img.length);
    for (let i = 0; i < img.length; i++) {
      let v = (img[i] - pLo) * 255.0 / (pHi - pLo);
      v = v < 0 ? 0 : v > 255 ? 255 : v;
      out[i] = Math.trunc(v);
    }
    return out;
  }

  // Round half to even, like OpenCV's cvRound / saturate_cast<uchar>.
  function roundEven(v) {
    const r = Math.round(v);
    return Math.abs(v % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
  }

  // OpenCV CLAHE (clipLimit 2.0, 8x8 tiles) for a 224x224 image, single-precision as in OpenCV.
  function clahe(img) {
    const f = Math.fround;
    const tiles = 8, tile = SIZE / tiles, area = tile * tile;
    const clipLimit = Math.max(Math.trunc((2.0 * area) / 256), 1);
    const lutScale = f(255 / area);
    const luts = new Uint8Array(tiles * tiles * 256);
    const hist = new Int32Array(256);

    for (let ty = 0; ty < tiles; ty++) {
      for (let tx = 0; tx < tiles; tx++) {
        hist.fill(0);
        for (let y = ty * tile; y < (ty + 1) * tile; y++) for (let x = tx * tile; x < (tx + 1) * tile; x++) hist[img[y * SIZE + x]]++;
        let clipped = 0;
        for (let i = 0; i < 256; i++) if (hist[i] > clipLimit) { clipped += hist[i] - clipLimit; hist[i] = clipLimit; }
        const batch = Math.trunc(clipped / 256);
        let residual = clipped - batch * 256;
        for (let i = 0; i < 256; i++) hist[i] += batch;
        if (residual !== 0) {
          const step = Math.max(Math.trunc(256 / residual), 1);
          for (let i = 0; i < 256 && residual > 0; i += step, residual--) hist[i]++;
        }
        const base = (ty * tiles + tx) * 256;
        let sum = 0;
        for (let i = 0; i < 256; i++) {
          sum += hist[i];
          luts[base + i] = Math.min(255, Math.max(0, roundEven(f(f(sum) * lutScale))));
        }
      }
    }

    const out = new Uint8Array(img.length);
    const inv = f(1 / tile);
    const xi1 = new Int32Array(SIZE), xi2 = new Int32Array(SIZE), xa = new Float32Array(SIZE), xa1 = new Float32Array(SIZE);
    for (let x = 0; x < SIZE; x++) {
      const txf = f(f(x * inv) - 0.5);
      const t1 = Math.floor(txf);
      xa[x] = f(txf - t1); xa1[x] = f(1 - xa[x]);
      xi1[x] = Math.max(t1, 0); xi2[x] = Math.min(t1 + 1, tiles - 1);
    }
    for (let y = 0; y < SIZE; y++) {
      const tyf = f(f(y * inv) - 0.5);
      const t1 = Math.floor(tyf);
      const ya = f(tyf - t1), ya1 = f(1 - ya);
      const r1 = Math.max(t1, 0) * tiles, r2 = Math.min(t1 + 1, tiles - 1) * tiles;
      for (let x = 0; x < SIZE; x++) {
        const v = img[y * SIZE + x];
        const a = luts[(r1 + xi1[x]) * 256 + v], b = luts[(r1 + xi2[x]) * 256 + v];
        const c = luts[(r2 + xi1[x]) * 256 + v], d = luts[(r2 + xi2[x]) * 256 + v];
        const top = f(f(a * xa1[x]) + f(b * xa[x]));
        const bot = f(f(c * xa1[x]) + f(d * xa[x]));
        const res = f(f(top * ya1) + f(bot * ya));
        out[y * SIZE + x] = Math.min(255, Math.max(0, roundEven(res)));
      }
    }
    return out;
  }

  // Browser only: decode a PNG/JPG file or blob to a 224x224 grayscale array, without colour management.
  async function decodeToGray224(blob) {
    let bitmap;
    try {
      bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
    } catch {
      throw new Error("This file isn't a readable image. Upload a PNG or JPG.");
    }
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(bitmap, 0, 0);
    const { data } = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
    return resizeTo224(rgbaToGray(data, bitmap.width * bitmap.height), bitmap.width, bitmap.height);
  }

  function preprocess(raw224) {
    return clahe(stretch(fixPolarity(raw224)));
  }

  const api = { SIZE, rgbaToGray, resizeTo224, decodeToGray224, polarityScore, fixPolarity, percentileLinear, stretch, clahe, preprocess };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.KneePreprocess = api;
})(typeof self !== "undefined" ? self : this);
