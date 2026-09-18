// Headless test of the companion extension: mocks `vscode`, stubs the Potemkin
// extension's exported API, and verifies that activating the companion registers
// the whippyunits plugin with the bundled binary path and correct markers.
"use strict";

const fs = require("fs");
const os = require("os");
const path = require("path");
const assert = require("assert");
const Module = require("module");

const extRoot = path.join(__dirname, "..");
const bundle = path.join(extRoot, "dist", "extension.js");

// Stage a fake binary so existsSync passes without a real cargo build.
const platformDir = `${process.platform}-${process.arch}`;
const binName =
  process.platform === "win32"
    ? "whippyunits-potemkin-plugin.exe"
    : "whippyunits-potemkin-plugin";
const binPath = path.join(extRoot, "bin", platformDir, binName);
fs.mkdirSync(path.dirname(binPath), { recursive: true });
if (!fs.existsSync(binPath)) fs.writeFileSync(binPath, "#!/bin/sh\n");

const registrations = [];
const potemkinApi = {
  apiVersion: 1,
  pluginsDir: fs.mkdtempSync(path.join(os.tmpdir(), "potemkin-mgd-")),
  async registerPlugin(p) {
    registrations.push(p);
  },
  async unregisterPlugin() {},
};

const vscodeMock = {
  extensions: {
    getExtension: (id) =>
      id === "potemkin-lsp.potemkin" ? { id, activate: async () => potemkinApi } : undefined,
  },
  window: {
    showInformationMessage: () => Promise.resolve(undefined),
    showWarningMessage: () => Promise.resolve(undefined),
    showErrorMessage: (m) => {
      throw new Error("unexpected error message: " + m);
    },
  },
  commands: {
    registerCommand: (id, fn) => ({ id, fn, dispose() {} }),
    executeCommand: () => Promise.resolve(),
  },
};

const origLoad = Module._load;
Module._load = function (request) {
  if (request === "vscode") return vscodeMock;
  return origLoad.apply(this, arguments);
};

(async () => {
  const ext = require(bundle);
  const context = { extensionPath: extRoot, subscriptions: [] };

  await ext.activate(context);

  assert.strictEqual(registrations.length, 1, "should register exactly once");
  const reg = registrations[0];
  assert.strictEqual(reg.name, "whippyunits");
  assert.strictEqual(reg.command, binPath, "should register the bundled binary path");
  assert.ok(Array.isArray(reg.markers) && reg.markers.includes("Quantity"), "markers set");
  assert.ok(
    Array.isArray(reg.languages) && reg.languages.includes("rust"),
    "should register languages: [rust]",
  );
  assert.ok(
    Array.isArray(reg.languageServers) && reg.languageServers.includes("rust-analyzer"),
    "should register languageServers: [rust-analyzer]",
  );

  fs.rmSync(potemkinApi.pluginsDir, { recursive: true, force: true });
  console.log("companion activation test OK: registers whippyunits plugin with Potemkin");
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
