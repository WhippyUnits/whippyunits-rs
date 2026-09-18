//! Spawn the built plugin and drive the Potemkin plugin protocol (JSONL over
//! stdio), verifying it pretty-prints whippyunits types for both hover and inlay
//! kinds and leaves unrelated text untouched.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

use serde_json::{json, Value};

const QUANTITY: &str = "Quantity<Unit<Scale<_2<0>, _3<0>, _5<0>, _Pi<0>>, Dimension<_M<0>, _L<1>, _T<0>, _I<0>, _Θ<0>, _N<0>, _J<0>, _A<0>>>, f64>";

/// Send a batch of request lines, return the parsed response lines.
fn roundtrip(requests: &[Value]) -> Vec<Value> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_whippyunits-potemkin-plugin"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn plugin");

    {
        let mut stdin = child.stdin.take().unwrap();
        for req in requests {
            writeln!(stdin, "{}", serde_json::to_string(req).unwrap()).unwrap();
        }
        // Drop stdin to signal EOF (a shutdown request is also included below).
    }

    let mut out = Vec::new();
    let reader = BufReader::new(child.stdout.take().unwrap());
    for line in reader.lines() {
        let line = line.unwrap();
        if line.trim().is_empty() {
            continue;
        }
        out.push(serde_json::from_str(&line).unwrap());
    }
    let _ = child.wait();
    out
}

#[test]
fn initialize_reports_name_and_markers() {
    let resp = roundtrip(&[
        json!({"id": 1, "method": "initialize", "params": {"protocol_version": 1, "verbosity": 0, "unicode": true}}),
        json!({"id": 2, "method": "shutdown"}),
    ]);
    let init = &resp[0];
    assert_eq!(init["result"]["name"], "whippyunits");
    assert!(init["result"]["markers"]
        .as_array()
        .unwrap()
        .iter()
        .any(|m| m == "Quantity"));
}

#[test]
fn transforms_hover_and_inlay() {
    let hover_text = format!("```rust\nlet distance: {QUANTITY}\n```");
    let inlay_text = format!(": {QUANTITY}");

    let resp = roundtrip(&[
        json!({"id": 1, "method": "transform", "params": {
            "kind": "hover", "verbosity": 0, "unicode": true,
            "items": [{"text": hover_text, "original": hover_text}]
        }}),
        json!({"id": 2, "method": "transform", "params": {
            "kind": "inlay_hint", "verbosity": 0, "unicode": true,
            "items": [{"text": inlay_text, "original": inlay_text}]
        }}),
        json!({"id": 3, "method": "shutdown"}),
    ]);

    // Hover items are bare strings (the simple result form).
    let hover_out = resp[0]["result"]["items"][0].as_str().unwrap();
    assert!(!hover_out.contains("Unit<Scale"), "hover not simplified: {hover_out}");
    assert!(hover_out.contains("Quantity<m"), "unexpected hover: {hover_out}");

    // Inlay items are rich objects { text, text_edits? }.
    let inlay_out = resp[1]["result"]["items"][0]["text"].as_str().unwrap();
    assert!(!inlay_out.contains("Unit<Scale"), "inlay not simplified: {inlay_out}");
    assert!(inlay_out.contains('m'), "unexpected inlay: {inlay_out}");
}

#[test]
fn inlay_plain_quantity_seeds_qty_macro_text_edit() {
    // A plain Quantity annotation with a position but no existing textEdits: the
    // plugin should synthesize a `: qty!(m)` edit with a zero-width range at the
    // hint position.
    let inlay_text = format!(": {QUANTITY}");
    let resp = roundtrip(&[
        json!({"id": 1, "method": "transform", "params": {
            "kind": "inlay_hint", "verbosity": 0, "unicode": true,
            "items": [{
                "text": inlay_text, "original": inlay_text,
                "position": {"line": 1, "character": 10}
            }]
        }}),
        json!({"id": 2, "method": "shutdown"}),
    ]);

    let item = &resp[0]["result"]["items"][0];
    let edits = item["text_edits"].as_array().expect("text_edits present");
    assert_eq!(edits.len(), 1, "expected one seeded edit: {item}");
    let new_text = edits[0]["newText"].as_str().unwrap();
    assert!(new_text.contains("qty!("), "expected qty! macro, got {new_text}");
    // Range falls back to the hint position when no existing edit is supplied.
    assert_eq!(edits[0]["range"]["start"]["line"], 1);
    assert_eq!(edits[0]["range"]["end"]["character"], 10);
}

#[test]
fn inlay_non_quantity_gets_no_seeded_edit() {
    // A bare `Unit<…>` (composite) is pretty-printed but must not get a qty! seed.
    let unit = "Unit<Scale<_2<0>, _3<0>, _5<0>, _Pi<0>>, Dimension<_M<0>, _L<1>, _T<0>, _I<0>, _Θ<0>, _N<0>, _J<0>, _A<0>>>";
    let inlay_text = format!(": {unit}");
    let resp = roundtrip(&[
        json!({"id": 1, "method": "transform", "params": {
            "kind": "inlay_hint", "verbosity": 0, "unicode": true,
            "items": [{"text": inlay_text, "original": inlay_text, "position": {"line": 0, "character": 0}}]
        }}),
        json!({"id": 2, "method": "shutdown"}),
    ]);
    let item = &resp[0]["result"]["items"][0];
    assert!(item.get("text_edits").is_none() || item["text_edits"].is_null(),
        "unexpected seeded edit for non-plain-quantity: {item}");
}

#[test]
fn leaves_unrelated_text_untouched() {
    let text = "let s: String = String::new();";
    let resp = roundtrip(&[
        json!({"id": 1, "method": "transform", "params": {
            "kind": "hover", "verbosity": 0, "unicode": true,
            "items": [{"text": text, "original": text}]
        }}),
        json!({"id": 2, "method": "shutdown"}),
    ]);
    assert_eq!(resp[0]["result"]["items"][0], text);
}
