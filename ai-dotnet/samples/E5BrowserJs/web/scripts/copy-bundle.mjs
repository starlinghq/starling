// Copies the published .NET bundle (dotnet publish ../E5Wasm -> <out>/wwwroot) into public/e5/, next to e5-worker.js.
// The bundle is served as static files, not bundled by Vite: dotnet.js loads its fingerprinted parts at run time.
import { cpSync, existsSync, rmSync } from "node:fs";
import { join } from "node:path";

const [from, to] = process.argv.slice(2);
for (const dir of ["_framework", "_content"]) {
  if (!existsSync(join(from, dir))) throw new Error(`${join(from, dir)} is missing: publish ../E5Wasm first`);
  rmSync(join(to, dir), { recursive: true, force: true });
  cpSync(join(from, dir), join(to, dir), { recursive: true });
}
console.log(`copied _framework and _content from ${from} to ${to}`);
