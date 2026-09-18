import * as fs from "fs";
import * as path from "path";
import * as vscode from "vscode";

/**
 * The slice of Potemkin's public API this companion depends on. Kept minimal and
 * local so we don't need to import Potemkin's types at build time.
 */
interface PotemkinApi {
  apiVersion: number;
  pluginsDir: string;
  registerPlugin(plugin: {
    name: string;
    command: string;
    markers?: string[];
    languages?: string[];
    languageServers?: string[];
    args?: string[];
    owner?: string;
  }): Promise<void>;
  unregisterPlugin(name: string): Promise<void>;
}

const PLUGIN_NAME = "whippyunits";
const POTEMKIN_EXTENSION_ID = "potemkin-lsp.potemkin";
// Fast-path substrings: Potemkin only invokes the plugin on payloads containing
// one of these, avoiding JSON parsing / IPC for unrelated messages.
const MARKERS = ["Quantity", "Unit<", "Scale", "Dimension"];
// whippyunits is a Rust library: only load under a Rust language server.
const LANGUAGES = ["rust"];
// The formatter is written against rust-analyzer's exact hover/inlay output, so
// scope it to that server specifically rather than any Rust server.
const LANGUAGE_SERVERS = ["rust-analyzer"];

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  context.subscriptions.push(
    vscode.commands.registerCommand("whippyunits-potemkin.register", () =>
      register(context, { interactive: true }),
    ),
  );
  await register(context, { interactive: false });
}

export function deactivate(): void {
  // Potemkin keeps the manifest across sessions; re-registration on next launch
  // refreshes it. We intentionally don't unregister here so pretty-printing
  // survives a reload of just this companion.
}

async function register(
  context: vscode.ExtensionContext,
  opts: { interactive: boolean },
): Promise<void> {
  const potemkin = vscode.extensions.getExtension(POTEMKIN_EXTENSION_ID);
  if (!potemkin) {
    // With extensionDependencies this shouldn't happen, but guard anyway.
    const choice = await vscode.window.showWarningMessage(
      "whippyunits pretty-printing needs the Potemkin extension.",
      "Install Potemkin",
    );
    if (choice === "Install Potemkin") {
      await vscode.commands.executeCommand(
        "workbench.extensions.installExtension",
        POTEMKIN_EXTENSION_ID,
      );
    }
    return;
  }

  const bin = bundledBinaryPath(context);
  if (!fs.existsSync(bin)) {
    vscode.window.showErrorMessage(
      `whippyunits: no bundled Potemkin plugin binary for this platform (${platformDir()}). ` +
        `Expected at ${bin}.`,
    );
    return;
  }
  ensureExecutable(bin);

  try {
    const api: PotemkinApi = await potemkin.activate();
    await api.registerPlugin({
      name: PLUGIN_NAME,
      command: bin,
      markers: MARKERS,
      languages: LANGUAGES,
      languageServers: LANGUAGE_SERVERS,
      // Let Potemkin auto-remove this plugin if this companion is disabled/uninstalled.
      owner: context.extension?.id,
    });
    if (opts.interactive) {
      vscode.window.showInformationMessage(
        "whippyunits: registered with Potemkin. Language servers will restart to pick it up.",
      );
    }
  } catch (err) {
    vscode.window.showErrorMessage(`whippyunits: failed to register with Potemkin: ${err}`);
  }
}

function platformDir(): string {
  return `${process.platform}-${process.arch}`;
}

function bundledBinaryPath(context: vscode.ExtensionContext): string {
  const name =
    process.platform === "win32"
      ? "whippyunits-potemkin-plugin.exe"
      : "whippyunits-potemkin-plugin";
  return path.join(context.extensionPath, "bin", platformDir(), name);
}

function ensureExecutable(binary: string): void {
  if (process.platform === "win32") return;
  try {
    fs.chmodSync(binary, 0o755);
  } catch {
    /* non-fatal */
  }
}
