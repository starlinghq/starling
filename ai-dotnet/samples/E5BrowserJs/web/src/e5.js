// The JavaScript API of the e5 bundle: a promise-based client over the Web Worker in public/e5/e5-worker.js.
// Nothing here knows about .NET; any JavaScript app (React, Vue, plain) can use it.
//
//   const e5 = await createE5({ variant: "int8embed", modelUrl: "/models/model-int8-embed.onnx", onProgress });
//   const q = await e5.embed("how do I keep basil fresh?", "query");          // Float32Array(384), L2-normalized
//   const m = await e5.embedBatch(["...", "..."], "passage");                  // Float32Array(n * 384), row-major
//   e5.info  // { profile, dimension, ortVersion, bundle, threads, cores, crossOriginIsolated }

export async function createE5({ variant = "fp32", modelUrl = "", provider = "wasm", onProgress, base = "/e5/" } = {}) {
  const worker = new Worker(new URL("e5-worker.js", new URL(base, location.href)), { type: "module" });
  let nextId = 0;
  const pending = new Map();
  worker.onmessage = ({ data }) => {
    const entry = pending.get(data.id);
    if (!entry) return;
    if (data.progress) {
      entry.onProgress?.(data.progress);
    } else {
      pending.delete(data.id);
      if (data.error) entry.reject(new Error(data.error));
      else entry.resolve(data.result);
    }
  };
  worker.onerror = (e) => {
    for (const entry of pending.values()) entry.reject(new Error(e.message || "worker error"));
    pending.clear();
  };
  const call = (op, args, progress) =>
    new Promise((resolve, reject) => {
      const id = nextId++;
      pending.set(id, { resolve, reject, onProgress: progress });
      worker.postMessage({ id, op, args });
    });

  const info = await call("load", { variant, modelUrl, provider }, onProgress);
  return {
    info,
    dimension: info.dimension,
    embed: (text, inputType = "query") => call("embed", { text, inputType }),
    async embedBatch(texts, inputType = "passage") {
      const flat = await call("embedBatch", { texts, inputType });
      return Array.from({ length: texts.length }, (_, i) => flat.subarray(i * info.dimension, (i + 1) * info.dimension));
    },
    terminate: () => worker.terminate(),
  };
}

/** Dot product; e5 vectors are L2-normalized, so this is the cosine. */
export function cosine(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}
