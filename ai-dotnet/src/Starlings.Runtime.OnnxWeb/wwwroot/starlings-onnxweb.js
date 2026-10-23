// Starlings.Runtime.OnnxWeb: the JavaScript side of OnnxWebRuntime, OnnxWebAssets and OnnxWebSession, imported by C#
// through JSHost.ImportAsync and called with [JSImport]. onnxruntime-web (1.30.0, MIT) is served next to this file
// under ort/ (or from the WebGPU adapter's ort/). Model bytes stay in JavaScript: they are fetched, verified with
// SubtleCrypto SHA-256, optionally kept in Cache Storage, and handed to onnxruntime-web without entering the .NET heap.

let ort = null;
let loadedBundle = null;
let info = null;
const controllers = new Map();

const resolve = (url) => new URL(url, document.baseURI).href;

// ONNX Runtime Web's own default when numThreads is not set (backend-wasm.ts): half the logical cores, at most 4,
// and 1 without cross-origin isolation (no SharedArrayBuffer).
function defaultThreads() {
  if (!self.crossOriginIsolated) {
    return 1;
  }
  return Math.min(4, Math.ceil((navigator.hardwareConcurrency || 1) / 2));
}

export async function initialize(bundleUrl, wasmDirUrl, bundleName, numThreads, proxy) {
  const moduleUrl = resolve(bundleUrl);
  if (ort) {
    if (moduleUrl !== loadedBundle) {
      throw new Error(`onnxruntime-web is already initialized from ${loadedBundle}; a page loads one bundle (requested ${moduleUrl}).`);
    }
    return JSON.stringify(info);
  }
  const isolated = self.crossOriginIsolated === true;
  if (numThreads > 1 && !isolated) {
    throw new Error(
      `NumThreads = ${numThreads} needs cross-origin isolation (Cross-Origin-Opener-Policy: same-origin and ` +
        'Cross-Origin-Embedder-Policy: require-corp or credentialless); this page is not isolated, so only 1 thread is possible.');
  }
  const module = await import(moduleUrl);
  module.env.wasm.wasmPaths = resolve(wasmDirUrl);
  module.env.wasm.numThreads = numThreads > 0 ? numThreads : defaultThreads();
  module.env.wasm.proxy = proxy === true;
  ort = module;
  loadedBundle = moduleUrl;
  info = {
    ort_version: module.env.versions?.web ?? null,
    bundle: bundleName,
    bundle_url: moduleUrl,
    cross_origin_isolated: isolated,
    hardware_concurrency: navigator.hardwareConcurrency ?? 0,
    num_threads: module.env.wasm.numThreads,
    proxy: module.env.wasm.proxy,
    webgpu_exposed: typeof navigator.gpu !== 'undefined',
    secure_context: self.isSecureContext === true,
  };
  return JSON.stringify(info);
}

async function sha256Hex(bytes) {
  if (!self.crypto?.subtle) {
    throw new Error('SHA-256 verification needs SubtleCrypto, which browsers expose only in a secure context (https or localhost).');
  }
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  return Array.from(digest, (b) => b.toString(16).padStart(2, '0')).join('');
}

async function readWithProgress(response, onProgress) {
  const encoded = response.headers.get('content-encoding');
  const length = Number(response.headers.get('content-length'));
  // With a content encoding, Content-Length counts compressed bytes; the body stream yields decoded ones.
  const total = !encoded && Number.isFinite(length) && length > 0 ? length : -1;
  if (!response.body) {
    const bytes = new Uint8Array(await response.arrayBuffer());
    onProgress(bytes.byteLength, bytes.byteLength);
    return bytes;
  }
  const reader = response.body.getReader();
  let buffer = total > 0 ? new Uint8Array(total) : null;
  const chunks = [];
  let loaded = 0;
  let reported = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    if (buffer && loaded + value.byteLength <= buffer.byteLength) {
      buffer.set(value, loaded);
    } else {
      if (buffer) {
        chunks.push(buffer.subarray(0, loaded));
        buffer = null;
      }
      chunks.push(value);
    }
    loaded += value.byteLength;
    if (loaded - reported >= 1 << 22) {
      reported = loaded;
      onProgress(loaded, total);
    }
  }
  let bytes;
  if (buffer) {
    bytes = loaded === buffer.byteLength ? buffer : buffer.slice(0, loaded);
  } else {
    bytes = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.byteLength;
    }
  }
  onProgress(loaded, loaded);
  return bytes;
}

async function openCache(cacheName) {
  if (!cacheName || typeof caches === 'undefined') {
    return null;
  }
  try {
    return await caches.open(cacheName);
  } catch {
    return null; // Opaque origins and some private modes refuse Cache Storage; fetching still works.
  }
}

// Returns { bytes, url, sha256, byteLength, fromCache, cached, cacheError }. Cache Storage only ever holds bytes that
// matched expectedSha256; a cached copy that no longer matches is deleted and fetched again.
export async function fetchAsset(url, expectedSha256, cacheName, requestId, onProgress) {
  const absolute = resolve(url);
  const expected = expectedSha256 ? expectedSha256.toLowerCase() : null;
  const controller = new AbortController();
  controllers.set(requestId, controller);
  try {
    const cache = expected ? await openCache(cacheName) : null;
    const result = { url: absolute, fromCache: false, cached: false, cacheError: null };
    let bytes = null;
    if (cache) {
      const hit = await cache.match(absolute);
      if (hit) {
        const candidate = new Uint8Array(await hit.arrayBuffer());
        if ((await sha256Hex(candidate)) === expected) {
          bytes = candidate;
          result.fromCache = true;
          result.cached = true;
          onProgress(bytes.byteLength, bytes.byteLength);
        } else {
          await cache.delete(absolute);
        }
      }
    }
    if (!bytes) {
      const response = await fetch(absolute, { signal: controller.signal, mode: 'cors', credentials: 'same-origin' });
      if (!response.ok) {
        throw new Error(`GET ${absolute}: HTTP ${response.status} ${response.statusText}`);
      }
      bytes = await readWithProgress(response, onProgress);
    }
    const actual = expected || self.crypto?.subtle ? await sha256Hex(bytes) : null;
    if (expected && actual !== expected) {
      throw new Error(`${absolute}: SHA-256 ${actual} does not match the expected ${expected}.`);
    }
    if (cache && !result.fromCache) {
      try {
        await cache.put(absolute, new Response(bytes, { headers: { 'content-type': 'application/octet-stream' } }));
        result.cached = true;
      } catch (e) {
        result.cacheError = String(e?.message ?? e); // Quota exceeded, for example; the bytes are still usable.
      }
    }
    result.sha256 = actual;
    result.byteLength = bytes.byteLength;
    result.bytes = bytes;
    return result;
  } finally {
    controllers.delete(requestId);
  }
}

export function abortFetch(requestId) {
  controllers.get(requestId)?.abort();
}

export function assetLength(asset) {
  return asset.bytes ? asset.bytes.byteLength : -1;
}

export function copyAsset(asset, view) {
  if (!asset.bytes) {
    throw new Error('The asset bytes were released or handed to a session.');
  }
  view.set(asset.bytes);
}

export function releaseAsset(asset) {
  asset.bytes = null;
}

function requireOrt() {
  if (!ort) {
    throw new Error('OnnxWebRuntime.InitializeAsync has not completed.');
  }
  return ort;
}

export async function createSession(asset, executionProvider, graphOptimizationLevel) {
  const o = requireOrt();
  if (!asset.bytes) {
    throw new Error('The asset bytes were released or already handed to a session.');
  }
  const bytes = asset.bytes;
  // With proxy on, onnxruntime-web transfers the buffer to its worker, which detaches it; the asset is consumed.
  asset.bytes = null;
  const session = await o.InferenceSession.create(bytes, {
    executionProviders: [executionProvider],
    graphOptimizationLevel,
  });
  return session;
}

export function sessionMetadata(session) {
  const describe = (names, metadata) =>
    names.map((name, i) => {
      const m = metadata?.[i];
      return {
        name,
        type: m && m.isTensor ? m.type : null,
        shape: m && m.isTensor ? m.shape.map((d) => (typeof d === 'number' ? d : -1)) : null,
      };
    });
  return JSON.stringify({
    inputs: describe(session.inputNames, session.inputMetadata),
    outputs: describe(session.outputNames, session.outputMetadata),
  });
}

export function createTensor(type, view, dims) {
  const o = requireOrt();
  const bytes = view.slice();
  let data;
  switch (type) {
    case 'int64':
      data = new BigInt64Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 8);
      break;
    case 'float32':
      data = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
      break;
    default:
      throw new Error(`Unsupported input tensor type ${type}.`);
  }
  // JSImport marshals int[] as a typed array; onnxruntime-web requires a plain Array of numbers.
  return new o.Tensor(type, data, Array.from(dims, Number));
}

export async function run(session, names, tensors, outputName) {
  const feeds = {};
  names.forEach((name, i) => {
    feeds[name] = tensors[i];
  });
  const outputs = await session.run(feeds, [outputName]);
  const output = outputs[outputName];
  if (output.location !== 'cpu') {
    // GPU-resident output (WebGPU): download it and free the GPU buffer.
    const data = await output.getData(true);
    return new (requireOrt().Tensor)(output.type, data, output.dims);
  }
  return output;
}

export function tensorType(tensor) {
  return tensor.type;
}

export function tensorDims(tensor) {
  return tensor.dims.map(Number);
}

export function copyTensor(tensor, view) {
  const data = tensor.data;
  view.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
}

export function disposeTensor(tensor) {
  tensor.dispose?.();
}

export async function releaseSession(session) {
  await session.release();
}
