//! Potemkin plugin for **whippyunits**.
//!
//! This is a straight port of the standalone `whippyunits-lsp-proxy` into the
//! Potemkin plugin model: instead of proxying the whole LSP stream itself, it
//! receives already-extracted type text from the Potemkin proxy and returns the
//! pretty-printed form. It reuses the exact formatting logic from
//! `whippyunits-lsp-proxy` (hover trait-signature simplification, `Quantity<…>`
//! and bare `Unit<Scale…>` rewriting, inlay-hint exponent pruning), so output
//! matches the original proxy.
//!
//! Transport: newline-delimited JSON on stdio (the Potemkin plugin protocol).

use std::io::{self, BufRead, Write};

use potemkin_plugin_api::protocol::{
    InitializeParams, InitializeResult, Request, Response, TransformItem, TransformParams,
    TransformResult, TransformResultItem,
};
use potemkin_plugin_api::TextKind;

use whippyunits_lsp_proxy::hover_processor::HoverProcessor;
use whippyunits_lsp_proxy::inlay_hint_processor::InlayHintProcessor;
use whippyunits_lsp_proxy::lsp_structures::{HoverContent, HoverContentItem, HoverContents};
use whippyunits_lsp_proxy::unit_formatter::{DisplayConfig, UnitFormatter};

/// Cheap substrings that gate the Potemkin fast path. Deliberately a superset of
/// the precise detector in `whippyunits_lsp_proxy::quantity_detection`: an
/// over-match just costs one no-op transform. These cover both the full hover
/// form (`Quantity<…>`, `Unit<Scale…>`) and the split inlay-hint / deconstructed
/// form (separate `Scale` and `Dimension` tokens).
const MARKERS: &[&str] = &["Quantity", "Unit<", "Scale", "Dimension"];

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }

        let response = match serde_json::from_str::<Request>(&line) {
            Ok(req) => {
                let shutdown = req.method == "shutdown";
                let resp = handle(req);
                if let Some(resp) = resp {
                    write_response(&mut out, &resp);
                }
                if shutdown {
                    break;
                }
                continue;
            }
            Err(e) => Response {
                id: 0,
                result: None,
                error: Some(format!("invalid request JSON: {e}")),
            },
        };
        write_response(&mut out, &response);
    }
}

fn write_response(out: &mut impl Write, resp: &Response) {
    if let Ok(s) = serde_json::to_string(resp) {
        let _ = writeln!(out, "{s}");
        let _ = out.flush();
    }
}

fn handle(req: Request) -> Option<Response> {
    match req.method.as_str() {
        "initialize" => {
            // Params are accepted for forward-compat; per-request config on
            // `transform` is what actually drives formatting.
            let _params: Option<InitializeParams> =
                req.params.and_then(|p| serde_json::from_value(p).ok());
            let result = InitializeResult {
                name: "whippyunits".to_string(),
                markers: MARKERS.iter().map(|s| s.to_string()).collect(),
            };
            Some(ok(req.id, &result))
        }
        "transform" => {
            let params: TransformParams = match req
                .params
                .ok_or_else(|| "missing params".to_string())
                .and_then(|p| serde_json::from_value(p).map_err(|e| e.to_string()))
            {
                Ok(p) => p,
                Err(e) => return Some(err(req.id, &e)),
            };

            let config = DisplayConfig {
                verbose: params.verbosity >= 1,
                unicode: params.unicode,
                // Raw ("Raw:" pre-transform types) is no longer the plugin's job:
                // the Potemkin proxy appends it, rendered per the user's
                // per-language-server `potemkin.raw` config. Keeping it off here
                // avoids a double raw section.
                include_raw: false,
            };

            let kind = params.kind;
            let processor = InlayHintProcessor::with_config(config.clone());
            let items = params
                .items
                .into_iter()
                .map(|item| transform_item(&item, kind, &config, &processor))
                .collect();
            Some(ok(req.id, &TransformResult { items }))
        }
        "shutdown" => Some(ok_empty(req.id)),
        other => Some(err(req.id, &format!("unknown method: {other}"))),
    }
}

/// Apply the whippyunits formatting appropriate to the text kind, plus — for
/// plain `Quantity<…>` inlay hints — a seeded `qty!(…)` `textEdit`.
fn transform_item(
    item: &TransformItem,
    kind: TextKind,
    config: &DisplayConfig,
    processor: &InlayHintProcessor,
) -> TransformResultItem {
    match kind {
        TextKind::InlayHint => {
            let pretty = transform_inlay(&item.text, config);
            let text_edits = seeded_edits(processor, item, &pretty);
            TransformResultItem::Rich { text: pretty, text_edits }
        }
        // Hover / diagnostics / other: full hover treatment (trait-signature
        // simplification + type rewriting). No structured edits.
        _ => TransformResultItem::Text(transform_hover(&item.text, config)),
    }
}

/// For a plain `Quantity<…>` inlay hint, produce the seeded `qty!(…)` textEdit.
/// Mirrors the original whippyunits proxy: only *plain* quantities get the macro
/// (composites like `MixedUnitMatrix<…>` keep the server's own edits).
fn seeded_edits(
    processor: &InlayHintProcessor,
    item: &TransformItem,
    pretty: &str,
) -> Option<Vec<serde_json::Value>> {
    let raw_type = item.text.trim_start_matches(": ");
    if !raw_type.starts_with("Quantity<") {
        return None;
    }
    let pretty_type = pretty.trim_start_matches(": ");
    processor.seeded_text_edits(pretty_type, item.position.as_ref(), item.text_edits.as_ref())
}

/// Reuse `HoverProcessor` by wrapping the text in a single-item hover.
fn transform_hover(text: &str, config: &DisplayConfig) -> String {
    let hover = HoverContent {
        contents: HoverContents::Single(HoverContentItem {
            language: Some("rust".to_string()),
            value: text.to_string(),
            kind: Some("markdown".to_string()),
        }),
        range: None,
    };
    let processor = HoverProcessor::new(config.clone());
    match processor.improve_hover_content(hover).contents {
        HoverContents::Single(item) => item.value,
        HoverContents::Multiple(mut items) => {
            items.pop().map(|i| i.value).unwrap_or_else(|| text.to_string())
        }
    }
}

/// Reuse the inlay-hint formatter: rewrite types, then prune `¹` exponents. The
/// Potemkin proxy has already concatenated the label's parts into one string, so
/// this mirrors `InlayHintProcessor::convert_whippyunits_hint`'s string logic.
fn transform_inlay(text: &str, config: &DisplayConfig) -> String {
    let formatter = UnitFormatter::new();
    let pretty = formatter.format_types_inlay_hint(text, config);
    let processor = InlayHintProcessor::with_config(config.clone());
    processor.prune_inlay_hint_exponents(&pretty)
}

fn ok<T: serde::Serialize>(id: u64, result: &T) -> Response {
    Response {
        id,
        result: Some(serde_json::to_value(result).unwrap_or(serde_json::Value::Null)),
        error: None,
    }
}

fn ok_empty(id: u64) -> Response {
    Response {
        id,
        result: Some(serde_json::json!({})),
        error: None,
    }
}

fn err(id: u64, message: &str) -> Response {
    Response {
        id,
        result: None,
        error: Some(message.to_string()),
    }
}
