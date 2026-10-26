// The JavaScript API of the e5 bundle: a promise-based client over the Web Worker in public/e5/e5-worker.js.
// Nothing here knows about .NET; any JavaScript app (React, Vue, plain) can use it.
//
//   const e5 = await createE5({ variant: "int8embed", modelUrl: "models/model-int8-embed.onnx", onProgress });
//   const q = await e5.embed("how do I keep basil fresh?", "query");          // Float32Array(384), L2-normalized
//   const m = await e5.embedBatch(["...", "..."], "passage");                  // Float32Array(384)[]
//   e5.info  // { profile, dimension, ortVersion, bundle, threads, cores, crossOriginIsolated }
//
// variant: "fp32" | "int8embed"; provider: "wasm" | "webgpu"; inputType: "query" | "passage" | "none". Other values
// are rejected. modelUrl is resolved against the page, like any URL the page uses. The worker script must be on the
// page's origin (browsers require it); `base` is where the bundle is served, by default the app's base path + "e5/".

const appBase = () => new URL(import.meta.env?.BASE_URL ?? "/", location.href);

export async function createE5({ variant = "fp32", modelUrl = "", provider = "wasm", onProgress, base } = {}) {
  const bundleBase = new URL(base ?? "e5/", appBase());
  const worker = new Worker(new URL("e5-worker.js", bundleBase), { type: "module" });
  let nextId = 0;
  const pending = new Map();
  const failAll = (error) => {
    for (const entry of pending.values()) entry.reject(error);
    pending.clear();
  };
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
  worker.onerror = (e) => failAll(new Error(e.message || "e5 worker error"));
  const call = (op, args, progress) =>
    new Promise((resolve, reject) => {
      const id = nextId++;
      pending.set(id, { resolve, reject, onProgress: progress });
      worker.postMessage({ id, op, args });
    });
  const terminate = () => {
    worker.terminate();
    failAll(new Error("e5 was terminated"));
  };

  let info;
  try {
    // The worker resolves URLs against its own location, so hand it an absolute one.
    const absoluteModelUrl = modelUrl ? new URL(modelUrl, location.href).href : "";
    info = await call("load", { variant, modelUrl: absoluteModelUrl, provider }, onProgress);
  } catch (e) {
    terminate(); // A failed load must not leave a worker holding the runtime and whatever it fetched.
    throw e;
  }
  return {
    info,
    dimension: info.dimension,
    embed: (text, inputType = "query") => call("embed", { text, inputType }),
    async embedBatch(texts, inputType = "passage") {
      const flat = await call("embedBatch", { texts, inputType });
      return Array.from({ length: texts.length }, (_, i) => flat.subarray(i * info.dimension, (i + 1) * info.dimension));
    },
    terminate,
  };
}

/** Dot product; e5 vectors are L2-normalized, so this is the cosine. */
export function cosine(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}
