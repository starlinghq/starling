import { useEffect, useRef, useState } from "react";
import { createE5, cosine } from "./e5.js";
import { exampleQueries, passages } from "./corpus.js";

const MB = (bytes) => `${Math.round(bytes / 1e6)} MB`;

function stageText(p) {
  if (!p) return "Starting the .NET runtime in a Web Worker…";
  const size = p.total ? `${MB(p.received)} of ${MB(p.total)}` : MB(p.received);
  switch (p.stage) {
    case "Runtime":
      return "Starting onnxruntime-web…";
    case "Tokenizer":
      return `Tokenizer: ${size}`;
    case "Model":
      return `Model: ${size}`;
    case "Session":
      return "Creating the inference session…";
    default:
      return "Embedding the corpus…";
  }
}

/** Counts animation frames: it keeps ticking while the model loads and searches, because they run in the worker. */
function FrameTicker() {
  const [frames, setFrames] = useState(0);
  useEffect(() => {
    let handle;
    const tick = () => {
      setFrames((f) => f + 1);
      handle = requestAnimationFrame(tick);
    };
    handle = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(handle);
  }, []);
  return (
    <span className="ticker" title="Rendered frames: the page stays responsive while e5 works in the worker">
      <span className="track">
        <span className="dot" style={{ transform: `translateX(${(frames % 60) * 1.4}px)` }} />
      </span>
      {frames} frames
    </span>
  );
}

export default function App() {
  const [variant, setVariant] = useState("int8embed");
  const [modelUrl, setModelUrl] = useState("/models/model-int8-embed.onnx");
  const [provider, setProvider] = useState("wasm");
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(null);
  const [error, setError] = useState(null);
  const [engine, setEngine] = useState(null);
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState(null);
  const [timing, setTiming] = useState(null);
  const e5 = useRef(null);

  async function load() {
    setLoading(true);
    setError(null);
    const started = performance.now();
    try {
      const client = await createE5({
        variant,
        modelUrl: variant === "int8embed" ? modelUrl : "",
        provider,
        onProgress: setProgress,
      });
      setProgress({ stage: "Corpus" });
      const corpusStarted = performance.now();
      const vectors = await client.embedBatch(passages.map((p) => p.text), "passage");
      e5.current = client;
      setEngine({
        info: client.info,
        corpus: passages.map((p, i) => ({ ...p, vector: vectors[i] })),
        loadMs: performance.now() - started,
        corpusMs: performance.now() - corpusStarted,
      });
    } catch (e) {
      setError(String(e.message ?? e));
    } finally {
      setLoading(false);
    }
  }

  async function search(text) {
    if (!text.trim() || !e5.current) return;
    setQuery(text);
    const started = performance.now();
    const q = await e5.current.embed(text, "query");
    const ms = performance.now() - started;
    const ranked = engine.corpus
      .map((p) => ({ ...p, score: cosine(q, p.vector) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 5);
    setHits({ query: text, ranked });
    setTiming(ms);
  }

  return (
    <main>
      <header>
        <h1>
          Multilingual search from JavaScript <FrameTicker />
        </h1>
        <p className="lede">
          A React page calling <code>intfloat/multilingual-e5-small</code> through a JavaScript API (<code>src/e5.js</code>).
          Behind it, a .NET WebAssembly bundle with no Blazor and no UI (<code>Starlings.Models.E5.Browser</code> on
          onnxruntime-web) runs in a Web Worker. Nothing you type leaves the page.
        </p>
      </header>

      {!engine ? (
        <section className="card">
          <h2>1. Load the model</h2>
          <div className="options">
            <label>
              Graph
              <select value={variant} onChange={(e) => setVariant(e.target.value)} disabled={loading}>
                <option value="int8embed">int8-embed, self-hosted (182 MB)</option>
                <option value="fp32">fp32 reference, from the Hugging Face Hub (470 MB)</option>
              </select>
            </label>
            {variant === "int8embed" && (
              <label>
                Graph URL
                <input value={modelUrl} onChange={(e) => setModelUrl(e.target.value)} disabled={loading} />
              </label>
            )}
            <label>
              Backend
              <select value={provider} onChange={(e) => setProvider(e.target.value)} disabled={loading}>
                <option value="wasm">WebAssembly (CPU)</option>
                <option value="webgpu">WebGPU (experimental)</option>
              </select>
            </label>
          </div>
          <button className="primary" onClick={load} disabled={loading}>
            {loading ? "Loading…" : "Load"}
          </button>
          {loading && (
            <div className="progress">
              <div className="stage">{stageText(progress)}</div>
              <progress max="1" value={progress?.total ? progress.received / progress.total : 0} />
            </div>
          )}
        </section>
      ) : (
        <section className="card">
          <h2>2. Search {engine.corpus.length} passages in {new Set(engine.corpus.map((p) => p.language)).size} languages</h2>
          <form
            className="query"
            onSubmit={(e) => {
              e.preventDefault();
              search(query);
            }}
          >
            <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Ask in any language…" aria-label="Query" />
            <button className="primary" type="submit" disabled={!query.trim()}>
              Search
            </button>
          </form>
          <div className="examples">
            Try:
            {exampleQueries.map((q) => (
              <button key={q} className="chip" onClick={() => search(q)}>
                {q}
              </button>
            ))}
          </div>
          {hits && (
            <>
              <p className="meta">Query embedded in {Math.round(timing)} ms (in the worker).</p>
              <ol className="hits" data-query={hits.query}>
                {hits.ranked.map((h) => (
                  <li key={h.text}>
                    <span className="score">{h.score.toFixed(3)}</span>
                    <span className="lang">{h.language}</span>
                    <span>{h.text}</span>
                  </li>
                ))}
              </ol>
            </>
          )}
        </section>
      )}

      {error && (
        <section className="card error">
          <h2>Something went wrong</h2>
          <pre>{error}</pre>
        </section>
      )}

      {engine && (
        <section className="card diagnostics">
          <h2>Runtime</h2>
          <dl>
            <dt>Profile</dt>
            <dd>{engine.info.profile}</dd>
            <dt>onnxruntime-web</dt>
            <dd>
              {engine.info.ortVersion} ({engine.info.bundle} bundle), in a Web Worker
            </dd>
            <dt>Threads</dt>
            <dd>
              {engine.info.threads} of {engine.info.cores} logical cores
              {!engine.info.crossOriginIsolated && <span className="hint">: not cross-origin isolated, so one thread.</span>}
            </dd>
            <dt>Load</dt>
            <dd>
              {(engine.loadMs / 1000).toFixed(1)} s, corpus embedded in {Math.round(engine.corpusMs)} ms
            </dd>
          </dl>
        </section>
      )}
    </main>
  );
}
