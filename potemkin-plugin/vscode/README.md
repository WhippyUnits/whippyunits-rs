# whippyunits for Potemkin (VS Code)

A **companion extension** that adds whippyunits type pretty-printing to your
editor via the [Potemkin](https://github.com/potemkin-lsp/potemkin) composable
LSP proxy. It turns verbose `Quantity<Unit<Scale<…>>, …>` into readable
`Quantity<m, f64>` in hovers and inlay hints.

This is the reference example of Potemkin's plugin-distribution model: a library
ships a tiny extension that **bundles its plugin binary** and **registers it with
Potemkin** — no manual manifests, no `cargo build`.

## How it works

- Depends on the Potemkin extension (`potemkin-lsp.potemkin`) via
  `extensionDependencies`, so installing this pulls Potemkin in automatically.
- Bundles the native `whippyunits-potemkin-plugin` binary under
  `bin/<platform>-<arch>/` (per-platform VSIX).
- On activation, calls Potemkin's exported API:

  ```ts
  const potemkin = vscode.extensions.getExtension("potemkin-lsp.potemkin");
  const api = await potemkin.activate();
  await api.registerPlugin({
    name: "whippyunits",
    command: bundledBinaryPath,
    markers: ["Quantity", "Unit<", "Scale", "Dimension"],
    languages: ["rust"],                // Rust only
    languageServers: ["rust-analyzer"], // formatter is tuned to rust-analyzer's output
  });
  ```

  Potemkin writes the manifest to its managed plugins dir and restarts wrapped
  servers so the plugin loads. Because of the `languages`/`languageServers`
  scoping, the plugin is only ever spawned under rust-analyzer — never under a
  C++/Go/other server you may also have Potemkin wrapping.

## Commands

- **whippyunits: Re-register Potemkin plugin** — force re-registration (e.g.
  after moving the binary).

## Building locally

```bash
npm install
./scripts/stage-binary.sh   # builds + copies the plugin binary into bin/<platform>
npm test                    # compile + headless activation test
npm run package             # produces whippyunits-potemkin-<version>.vsix
```
