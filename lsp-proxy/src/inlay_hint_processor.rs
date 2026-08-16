use crate::{
    quantity_detection,
    unit_formatter::{DisplayConfig, UnitFormatter},
};
use anyhow::Result;
use serde_json::{json, Value};

/// Process inlay hint responses to pretty-print whippyunits types
#[derive(Clone)]
pub struct InlayHintProcessor {
    formatter: UnitFormatter,
    display_config: DisplayConfig,
}

impl Default for InlayHintProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl InlayHintProcessor {
    pub fn new() -> Self {
        Self {
            formatter: UnitFormatter::new(),
            display_config: DisplayConfig::default(),
        }
    }

    pub fn with_config(display_config: DisplayConfig) -> Self {
        Self {
            formatter: UnitFormatter::new(),
            display_config,
        }
    }

    /// Process an inlay hint response, converting whippyunits types to pretty format
    pub fn process_inlay_hint_response(&self, message: &str) -> Result<String> {
        // Fast string search to detect if this message contains Quantity types
        if !self.contains_quantity_types_fast(message) {
            // No Quantity types detected, return original message unchanged
            return Ok(message.to_string());
        }

        // Parse the JSON message only if we detected Quantity types
        let mut json_value: Value = serde_json::from_str(message)?;

        // Check if this is an inlay hint response with results
        if let Some(result) = json_value.get_mut("result") {
            // Handle both array (inlay hint requests) and object (resolve responses)
            if let Some(results_array) = result.as_array_mut() {
                // Process each inlay hint in the results array
                for hint in results_array.iter_mut() {
                    self.process_single_hint(hint)?;
                }
            } else if let Some(single_hint) = result.as_object_mut() {
                // Process a single inlay hint object (resolve response)
                self.process_single_hint_object(single_hint)?;
            }
        } 

        // Convert back to string
        Ok(serde_json::to_string(&json_value)?)
    }

    /// Process a single inlay hint, converting whippyunits types if present
    fn process_single_hint(&self, hint: &mut Value) -> Result<()> {
        // Get the label array
        if let Some(label) = hint.get_mut("label") {
            if let Some(label_array) = label.as_array_mut() {
                if self.contains_whippyunits_type(label_array) {
                    self.convert_whippyunits_hint(label_array)?;
                }
            }
        }
        Ok(())
    }

    /// Process a single inlay hint object (for resolve responses)
    fn process_single_hint_object(
        &self,
        hint_obj: &mut serde_json::Map<String, Value>,
    ) -> Result<()> {
        // Check if this hint contains a whippyunits type first
        let has_whippyunits = if let Some(label) = hint_obj.get("label") {
            if let Some(label_array) = label.as_array() {
                self.contains_whippyunits_type(label_array)
            } else {
                false
            }
        } else {
            false
        };

        if has_whippyunits {
            // Determine whether this hint is a plain `Quantity` annotation (as
            // opposed to a composite like `MixedUnitMatrix<…>` or a bare `Unit`).
            // The `qty!` macro completion only makes sense for plain quantities.
            let is_plain_quantity = hint_obj
                .get("label")
                .and_then(|label| label.as_array())
                .map(|label_array| {
                    let full = Self::concat_label_values(label_array);
                    full.trim_start_matches(": ").starts_with("Quantity<")
                })
                .unwrap_or(false);

            // Get the label array from the object and process it
            if let Some(label) = hint_obj.get_mut("label") {
                if let Some(label_array) = label.as_array_mut() {
                    self.convert_whippyunits_hint(label_array)?;
                }
            }

            // Add the qty! macro completion only for plain quantities.
            if is_plain_quantity {
                self.add_seeded_unit_macro_text_edit(hint_obj)?;
            }
        }
        Ok(())
    }

    /// Fast string search to detect Quantity types without deserialization
    /// This performs a performant string search for "Quantity<" patterns
    fn contains_quantity_types_fast(&self, json_payload: &str) -> bool {
        quantity_detection::contains_quantity_types_fast(json_payload)
    }

    /// Check if a label array contains a whippyunits type.
    ///
    /// rust-analyzer splits an inlay-hint label into many parts (named types
    /// carry go-to-definition locations, punctuation is separate). Rather than
    /// keying on a single `Quantity` part, we concatenate the whole label and
    /// reuse the same fast detection used for hovers, so bare `Unit<Scale…>`
    /// (e.g. from `whippyalgebra`'s `MixedUnitMatrix`) is recognized too.
    pub fn contains_whippyunits_type(&self, label_array: &[Value]) -> bool {
        let full = Self::concat_label_values(label_array);
        quantity_detection::contains_quantity_types_fast(&full)
    }

    /// Concatenate the `value` fields of every label part into one string.
    fn concat_label_values(label_array: &[Value]) -> String {
        let mut out = String::new();
        for part in label_array {
            if let Some(text) = part.get("value").and_then(|v| v.as_str()) {
                out.push_str(text);
            }
        }
        out
    }

    /// Convert a whippyunits inlay hint to pretty format.
    ///
    /// Works uniformly for a bare `Quantity`, a bare `Unit<Scale…>`, or a
    /// composite like `MixedUnitMatrix<DCons<Unit<…>>, …>`: the entire label is
    /// concatenated, run through the inlay formatter (which rewrites every
    /// `Quantity<Unit<…>>` and bare `Unit<Scale…>` in place), and re-emitted as
    /// a leading `": "` (when present) plus a single type part that keeps the
    /// first located part's `location` for go-to-definition.
    pub fn convert_whippyunits_hint(&self, label_array: &mut Vec<Value>) -> Result<()> {
        let full = Self::concat_label_values(label_array);

        // Preserve a leading ": " type-annotation prefix as its own part.
        let (prefix, type_text) = match full.strip_prefix(": ") {
            Some(rest) => (": ", rest),
            None => ("", full.as_str()),
        };

        let pretty_type = self
            .formatter
            .format_types_inlay_hint(type_text, &self.display_config);
        let pretty_type = self.prune_inlay_hint_exponents(&pretty_type);

        // Preserve the first available location so go-to-definition still works.
        let location = label_array
            .iter()
            .find_map(|part| part.get("location").cloned());

        let mut new_label = Vec::new();
        if !prefix.is_empty() {
            new_label.push(json!({ "value": prefix }));
        }
        let mut type_part = json!({ "value": pretty_type });
        if let Some(location) = location {
            type_part["location"] = location;
        }
        new_label.push(type_part);

        *label_array = new_label;
        Ok(())
    }

    /// Add a qty! macro text edit to the hint
    fn add_seeded_unit_macro_text_edit(
        &self,
        hint_obj: &mut serde_json::Map<String, Value>,
    ) -> Result<()> {
        // Generate the qty! macro text
        let seeded_text = self.generate_seeded_unit_macro(hint_obj)?;

        // Extract position before any mutable borrows
        let position = if let Some(pos) = hint_obj.get("position") {
            pos.clone()
        } else {
            return Ok(());
        };

        // Replace existing textEdits with our seeded unit macro
        if let Some(existing_text_edits) = hint_obj.get_mut("textEdits") {
            if let Some(text_edits_array) = existing_text_edits.as_array_mut() {
                // Clear existing text edits and add our qty! macro one
                text_edits_array.clear();

                // Get the range from the first existing text edit (if any)
                let range = if let Some(first_edit) = text_edits_array.first() {
                    if let Some(range) = first_edit.get("range") {
                        range.clone()
                    } else {
                        // Fallback: use position as range
                        json!({
                            "start": position,
                            "end": position
                        })
                    }
                } else {
                    // No existing text edits, use position as range
                    json!({
                        "start": position,
                        "end": position
                    })
                };

                // Add our qty! macro text edit
                let text_edit = json!({
                    "range": range,
                    "newText": seeded_text
                });
                text_edits_array.push(text_edit);
            }
        } else {
            // No existing textEdits, create new array
            let range = json!({
                "start": position,
                "end": position
            });

            let text_edit = json!({
                "range": range,
                "newText": seeded_text
            });
            hint_obj.insert("textEdits".to_string(), json!([text_edit]));
        }

        Ok(())
    }

    /// Generate a seeded unit macro based on the unresolved type
    fn generate_seeded_unit_macro(
        &self,
        hint_obj: &serde_json::Map<String, Value>,
    ) -> Result<String> {
        // We need to reconstruct the original unresolved type from the pretty-printed version
        // Look for the pretty-printed type in the label
        let mut pretty_type = String::new();

        if let Some(label) = hint_obj.get("label") {
            if let Some(label_array) = label.as_array() {
                for part in label_array {
                    if let Some(value) = part.get("value") {
                        if let Some(text) = value.as_str() {
                            // Skip the ": " part
                            if text != ": " {
                                pretty_type.push_str(text);
                            }
                        }
                    }
                }
            }
        }

        // Extract the datatype suffix if present
        let (unit_part, datatype) = self.extract_unit_and_datatype(&pretty_type);

        // Generate the appropriate qty! macro format based on datatype
        let unit_macro = if datatype == "f64" {
            // For f64, use the simple format: qty!(mm)
            format!("qty!({})", unit_part)
        } else {
            // For other datatypes, use the type-specified format: qty!(mm, i32)
            format!("qty!({}, {})", unit_part, datatype)
        };

        // Return just the type annotation with the qty! macro
        Ok(format!(": {}", unit_macro))
    }

    /// Extract unit part and datatype from a pretty-printed type with suffix
    fn extract_unit_and_datatype(&self, pretty_type: &str) -> (String, String) {
        // Remove the "Unresolved type - " prefix if present
        let clean_type = pretty_type
            .strip_prefix("Unresolved type - ")
            .unwrap_or(pretty_type);

        // Check for new Quantity<unit, datatype> format
        if clean_type.starts_with("Quantity<") && clean_type.ends_with('>') {
            let inner = &clean_type[9..clean_type.len() - 1]; // Remove "Quantity<" and ">"

            if let Some(comma_pos) = inner.rfind(',') {
                let unit_part = self.convert_pretty_type_to_unit_macro(inner[..comma_pos].trim());
                let datatype = inner[comma_pos + 1..].trim().to_string();
                return (unit_part, datatype);
            } else {
                // No comma found, this might be a malformed Quantity type
                // Try to extract just the unit part without datatype
                let unit_part = self.convert_pretty_type_to_unit_macro(inner.trim());
                return (unit_part, "f64".to_string());
            }
        }

        // Check for old backing datatype suffix format (fallback for compatibility)
        if let Some(underscore_pos) = clean_type.rfind('_') {
            let suffix = &clean_type[underscore_pos + 1..];
            if suffix == "f64"
                || suffix == "f32"
                || suffix == "i64"
                || suffix == "i32"
                || suffix == "i16"
                || suffix == "i8"
                || suffix == "u64"
                || suffix == "u32"
                || suffix == "u16"
                || suffix == "u8"
                || suffix == "isize"
                || suffix == "usize"
            {
                let unit_part =
                    self.convert_pretty_type_to_unit_macro(&clean_type[..underscore_pos]);
                return (unit_part, suffix.to_string());
            }
        }

        // No datatype suffix found, treat as f64 (default)
        let unit_part = self.convert_pretty_type_to_unit_macro(clean_type);
        (unit_part, "f64".to_string())
    }

    /// Convert pretty-printed type to qty! macro format
    fn convert_pretty_type_to_unit_macro(&self, pretty_type: &str) -> String {
        // Convert pretty-printed types like "mmˀ" to unit macro format like "mm^"
        // Remove the ? and place cursor after the caret
        pretty_type
            .replace("ˀ", "^") // Replace superscript question mark with just ^
            .replace("⁻", "^-") // Replace superscript minus with ^-
            .replace("¹", "^1") // Replace superscript 1 with ^1
            .replace("²", "^2") // Replace superscript 2 with ^2
            .replace("³", "^3") // Replace superscript 3 with ^3
            .replace("⁴", "^4") // Replace superscript 4 with ^4
            .replace("⁵", "^5") // Replace superscript 5 with ^5
            .replace("⁶", "^6") // Replace superscript 6 with ^6
            .replace("⁷", "^7") // Replace superscript 7 with ^7
            .replace("⁸", "^8") // Replace superscript 8 with ^8
            .replace("⁹", "^9") // Replace superscript 9 with ^9
            .replace("⁰", "^0") // Replace superscript 0 with ^0
    }

    /// Prune ^1 exponents from inlay hint display while keeping meaningful exponents
    pub fn prune_inlay_hint_exponents(&self, pretty_type: &str) -> String {
        // In our pretty-printed output, we use Unicode superscripts like ¹, ², ³, etc.
        // We want to remove ¹ (superscript 1) but keep all other superscripts
        // For negative exponents, we want to preserve the full -1 to make it clear
        let mut result = pretty_type.to_string();

        // Remove standalone ¹ (but preserve ⁻¹ as it represents a meaningful negative exponent)
        // Since ⁻¹ is two separate Unicode characters (⁻ + ¹), we need to handle this carefully
        // First, temporarily replace ⁻¹ with a placeholder to protect it
        result = result.replace("⁻¹", "PLACEHOLDER_MINUS_ONE");

        // Then remove standalone ¹
        result = result.replace("¹", "");

        // Finally, restore the ⁻¹
        result = result.replace("PLACEHOLDER_MINUS_ONE", "⁻¹");

        result
    }
}
