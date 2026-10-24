// Hosts the .NET bundle (../../../E5Wasm, published into this directory: _framework/, _content/) in a Web Worker, so
// tokenization, inference and pooling never block the page. The page talks to it through ../../src/e5.js.
//
// Messages in:  { id, op: "load" | "embed" | "embedBatch" | "unload", args }
// Messages out: { id, progress } while loading, then { id, result } or { id, error }.
// Vectors come back as Float32Arrays whose buffers are transferred, not copied.
import { dotnet } from "./_framework/dotnet.js";

// Called by the .NET side ([JSImport("globalThis.e5interop.makeFloat32")]): the memory view is only valid during the
// call, so copy it into a JavaScript-owned Float32Array.
globalThis.e5interop = {
  makeFloat32(view) {
    const bytes = view.slice();
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  },
};

const ready = (async () => {
  const runtime = await dotnet.create();
  const exports = await runtime.getAssemblyExports(runtime.getConfig().mainAssemblyName);
  return exports.E5Wasm.E5Exports;
})();

self.onmessage = async ({ data }) => {
  const { id, op, args = {} } = data;
  try {
    const e5 = await ready;
    switch (op) {
      case "load": {
        const json = await e5.Load(args.variant ?? "fp32", args.modelUrl ?? "", args.provider ?? "wasm", (progress) =>
          self.postMessage({ id, progress: JSON.parse(progress) }),
        );
        self.postMessage({ id, result: JSON.parse(json) });
        break;
      }
      case "embed": {
        const vector = await e5.Embed(args.text, args.inputType ?? "none");
        self.postMessage({ id, result: vector }, [vector.buffer]);
        break;
      }
      case "embedBatch": {
        const flat = await e5.EmbedBatch(args.texts, args.inputType ?? "none");
        self.postMessage({ id, result: flat }, [flat.buffer]);
        break;
      }
      case "unload":
        e5.Unload();
        self.postMessage({ id, result: true });
        break;
      default:
        throw new Error(`unknown op ${op}`);
    }
  } catch (e) {
    self.postMessage({ id, error: String(e?.message ?? e) });
  }
};
