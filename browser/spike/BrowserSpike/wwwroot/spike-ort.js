// onnxruntime-web side of the browser spike (docs/m0/browser-wasm-spike.md). Called from C# through IJSRuntime.
// onnxruntime-web 1.30.0 (pinned in ../../package.json) is served by the harness under /ort/, outside the app.
import * as ort from '/ort/ort.webgpu.min.mjs';

let session = null;
let backend = null;

const ids64 = (bytes) => new BigInt64Array(bytes.slice().buffer);

function feeds(idBytes, maskBytes, batch, seq) {
  const ids = ids64(idBytes);
  const mask = ids64(maskBytes);
  const types = new BigInt64Array(batch * seq);
  return {
    input_ids: new ort.Tensor('int64', ids, [batch, seq]),
    attention_mask: new ort.Tensor('int64', mask, [batch, seq]),
    token_type_ids: new ort.Tensor('int64', types, [batch, seq]),
  };
}

export async function create(modelUrl, requestedBackend, threads) {
  const info = {
    ort_version: ort.env.versions?.web ?? null,
    cross_origin_isolated: self.crossOriginIsolated === true,
    hardware_concurrency: navigator.hardwareConcurrency,
    user_agent: navigator.userAgent,
  };
  try {
    ort.env.wasm.wasmPaths = '/ort/';
    ort.env.wasm.numThreads = threads;
    ort.env.wasm.simd = true;
    info.num_threads = ort.env.wasm.numThreads;
    if (requestedBackend === 'webgpu') {
      if (!navigator.gpu) {
        return JSON.stringify({ ...info, error: 'navigator.gpu is undefined (WebGPU not exposed)' });
      }
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) {
        return JSON.stringify({ ...info, error: 'requestAdapter returned null' });
      }
      const a = adapter.info ?? {};
      info.adapter = { vendor: a.vendor, architecture: a.architecture, device: a.device, description: a.description };
      info.adapter_features = [...adapter.features].sort();
      info.adapter_limits = {
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      };
      ort.env.webgpu.adapter = adapter;
    }
    let t0 = performance.now();
    const response = await fetch(modelUrl);
    if (!response.ok) {
      return JSON.stringify({ ...info, error: `fetch ${modelUrl}: HTTP ${response.status}` });
    }
    const bytes = new Uint8Array(await response.arrayBuffer());
    info.fetch_ms = performance.now() - t0;
    info.model_bytes = bytes.byteLength;
    t0 = performance.now();
    session = await ort.InferenceSession.create(bytes, {
      executionProviders: [requestedBackend],
      graphOptimizationLevel: 'all',
    });
    info.create_ms = performance.now() - t0;
    backend = requestedBackend;
    info.inputs = session.inputNames;
    info.outputs = session.outputNames;
    return JSON.stringify(info);
  } catch (e) {
    return JSON.stringify({ ...info, error: String(e?.message ?? e), stack: String(e?.stack ?? '') });
  }
}

export async function run(idBytes, maskBytes, batch, seq) {
  const out = await session.run(feeds(idBytes, maskBytes, batch, seq));
  const hidden = out.last_hidden_state;
  const data = hidden.location === 'cpu' ? hidden.data : await hidden.getData();
  return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
}

export async function bench(idBytes, maskBytes, batch, seq, warmup, n) {
  const f = feeds(idBytes, maskBytes, batch, seq);
  for (let i = 0; i < warmup; i++) {
    await session.run(f);
  }
  const samples = [];
  for (let i = 0; i < n; i++) {
    const t0 = performance.now();
    await session.run(f);
    samples.push(performance.now() - t0);
  }
  return JSON.stringify({ batch, seq, warmup, samples_ms: samples });
}

export async function memory() {
  const result = { backend };
  if (performance.memory) {
    result.js_heap_used = performance.memory.usedJSHeapSize;
    result.js_heap_total = performance.memory.totalJSHeapSize;
  }
  if (typeof performance.measureUserAgentSpecificMemory === 'function') {
    try {
      const m = await performance.measureUserAgentSpecificMemory();
      result.ua_specific_bytes = m.bytes;
      result.ua_breakdown = m.breakdown
        .filter((b) => b.bytes > 1 << 20)
        .map((b) => ({ bytes: b.bytes, types: b.types, scope: b.attribution?.[0]?.scope ?? null }));
    } catch (e) {
      result.ua_error = String(e);
    }
  }
  return JSON.stringify(result);
}

// Load-only probe for the catalog models (no inference): can onnxruntime-web create a session from this graph and
// its external data file at all, how long does it take, and how much memory does the page hold afterwards.
export async function probe(graphUrl, dataUrl, dataPath, requestedBackend, threads) {
  const info = { graph_url: graphUrl, data_url: dataUrl, backend: requestedBackend };
  let s = null;
  try {
    ort.env.wasm.wasmPaths = '/ort/';
    ort.env.wasm.numThreads = threads;
    let t0 = performance.now();
    const graph = new Uint8Array(await (await fetch(graphUrl)).arrayBuffer());
    let data = null;
    if (dataUrl) {
      data = new Uint8Array(await (await fetch(dataUrl)).arrayBuffer());
    }
    info.fetch_ms = performance.now() - t0;
    info.graph_bytes = graph.byteLength;
    info.data_bytes = data ? data.byteLength : 0;
    t0 = performance.now();
    const options = { executionProviders: [requestedBackend], graphOptimizationLevel: 'all' };
    if (data) {
      options.externalData = [{ path: dataPath, data }];
    }
    s = await ort.InferenceSession.create(graph, options);
    info.create_ms = performance.now() - t0;
    info.inputs = s.inputNames;
    info.outputs = s.outputNames;
    info.ok = true;
  } catch (e) {
    info.ok = false;
    info.error = String(e?.message ?? e).slice(0, 2000);
  }
  try {
    info.memory = JSON.parse(await memory());
  } catch (e) {
    info.memory_error = String(e);
  }
  if (s) {
    await s.release();
  }
  return JSON.stringify(info);
}

export async function release() {
  if (session) {
    await session.release();
    session = null;
  }
}
