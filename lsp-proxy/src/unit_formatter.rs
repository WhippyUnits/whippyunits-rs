use syn::{parse_str, TypePath};
use whippyunits_core::{
    dimension_exponents::DynDimensionExponents, scale_exponents::ScaleExponents,
};

/// Display configuration for whippyunits type formatting
#[derive(Debug, Clone)]
pub struct DisplayConfig {
    pub verbose: bool,
    pub unicode: bool,
    pub include_raw: bool,
}

impl Default for DisplayConfig {
    fn default() -> Self {
        Self {
            verbose: false,
            unicode: true,
            include_raw: false,
        }
    }
}

/// Formatter for whippyunits types using the new prettyprint API
#[derive(Clone)]
pub struct UnitFormatter;

impl Default for UnitFormatter {
    fn default() -> Self {
        Self::new()
    }
}

impl UnitFormatter {
    pub fn new() -> Self {
        Self
    }

    /// Format whippyunits types in text with the specified configuration
    pub fn format_types(&self, text: &str, config: &DisplayConfig) -> String {
        let mut result = self.format_quantity_types(text, config.verbose, config.unicode, false);

        // Add raw type if requested and we actually made changes
        if config.include_raw && result != text {
            result.push_str(&format!("\n\nRaw:\n\n{}", text));
        }

        result
    }

    /// Format whippyunits types in text with original text for Raw section.
    ///
    /// For a concrete `MixedUnitMatrix<…>` hover, also appends a values-free diagram
    /// of the matrix: the row/column unit labels in the margins and each entry's
    /// unit literal (`RowDims[i] / ColDims[j]`) in the cells — the type-level
    /// analog of the runtime `Display` impl.
    pub fn format_types_with_original(
        &self,
        text: &str,
        config: &DisplayConfig,
        original_text: &str,
    ) -> String {
        let base = self.format_types_with_original_impl(text, config, original_text);
        match self.unit_matrix_trace(original_text) {
            Some(trace) => Self::insert_after_first_code_block(&base, &trace),
            None => base,
        }
    }

    /// Insert a fenced `text` block containing `trace` immediately after the
    /// first fenced code block of `base` (i.e. right after the prettified type,
    /// and before any `Raw:` section). Falls back to appending if no code block
    /// is found.
    fn insert_after_first_code_block(base: &str, trace: &str) -> String {
        let block = format!("\n\n```text\n{trace}\n```");
        if let Some(start) = base.find("```rust") {
            let after_open = start + "```rust".len();
            if let Some(rel_close) = base[after_open..].find("```") {
                let close_end = after_open + rel_close + "```".len();
                return format!("{}{}{}", &base[..close_end], block, &base[close_end..]);
            }
        }
        format!("{base}{block}")
    }

    fn format_types_with_original_impl(
        &self,
        text: &str,
        config: &DisplayConfig,
        original_text: &str,
    ) -> String {
        // Check if this contains a generic type definition that we passed through unchanged
        let contains_generic_definition =
            text.contains("T = f64") || original_text.contains("T = f64");

        // If it's a generic definition, just format normally without raw
        if contains_generic_definition {
            return self.format_quantity_types(text, config.verbose, config.unicode, false);
        }

        // Work within the existing markdown structure
        // Find the first code block and replace its content
        // Use `text` (which may have been pre-transformed, e.g. trait simplification)
        // rather than `original_text` so earlier transformations aren't discarded.
        if let Some(code_start) = text.find("```rust") {
            let after_code_start = &text[code_start + 7..]; // Skip "```rust"
            if let Some(code_end) = after_code_start.find("```") {
                let code_content = &after_code_start[..code_end];

                // Format the code content
                let formatted_content =
                    self.format_quantity_types(code_content, config.verbose, config.unicode, false);

                // Check if we actually transformed anything (compare against original)
                let original_code_content = original_text
                    .find("```rust")
                    .and_then(|s| {
                        let after = &original_text[s + 7..];
                        after.find("```").map(|e| &after[..e])
                    });
                let was_transformed = original_code_content
                    .is_none_or(|orig| formatted_content != orig);

                // Extract raw type from the original code content if we transformed it
                let raw_type = if was_transformed && config.include_raw {
                    original_code_content
                        .map(|c| self.extract_raw_type_from_hover(c))
                        .unwrap_or_default()
                } else {
                    String::new()
                };

                // Replace the content in the existing code block
                let before_code = &text[..code_start + 7]; // Include "```rust"
                let after_code_block = &after_code_start[code_end + 3..]; // Skip closing ```

                let result = if !raw_type.is_empty() {
                    // Insert raw section after the first --- separator
                    if let Some(separator_pos) = after_code_block.find("---") {
                        let after_separator = &after_code_block[separator_pos..];
                        format!(
                            "{}\n{}\n```\n\n---\nRaw:\n\n```rust\n{}\n```\n{}",
                            before_code,
                            formatted_content.trim(),
                            raw_type,
                            after_separator
                        )
                    } else {
                        format!(
                            "{}\n{}\n```\n\n---\nRaw:\n\n```rust\n{}\n```\n",
                            before_code,
                            formatted_content.trim(),
                            raw_type
                        )
                    }
                } else {
                    format!(
                        "{}\n{}\n```{}",
                        before_code,
                        formatted_content.trim(),
                        after_code_block
                    )
                };

                return result;
            }
        }

        // Fallback to normal formatting if we can't parse the markdown structure
        self.format_quantity_types(text, config.verbose, config.unicode, false)
    }

    /// Format whippyunits types for inlay hints (compact format)
    pub fn format_types_inlay_hint(&self, text: &str, config: &DisplayConfig) -> String {
        self.format_quantity_types(text, config.verbose, config.unicode, true)
    }

    /// Core method to format whippyunits types with configurable parameters.
    ///
    /// Runs two passes. The first rewrites `Quantity<Unit<Scale<…>, Dimension<…>>, T, Brand>`
    /// wrappers, preserving the historical `Quantity<…>` presentation exactly. The
    /// second rewrites any remaining *bare* `Unit<Scale<…>, Dimension<…>>` types
    /// (as produced by, e.g., `whippyalgebra`) into their unit label.
    fn format_quantity_types(
        &self,
        text: &str,
        verbose: bool,
        unicode: bool,
        is_inlay_hint: bool,
    ) -> String {
        let mut result = text.to_string();
        if result.contains("Quantity<Unit<") && result.contains("Scale") {
            result = self.format_quantity_pass(&result, verbose, unicode, is_inlay_hint);
        }
        if result.contains("Unit<Scale") {
            result = self.format_unit_pass(&result, verbose, unicode, is_inlay_hint);
        }
        // Final pass: collapse whippyalgebra's `DCons<…, DNil>` cons-lists into
        // bracketed array notation. Runs after unit formatting so the entries are
        // already prettified (e.g. `DCons<(m·s⁻¹), DCons<m, DNil>>` -> `[(m·s⁻¹), m]`).
        result = self.collapse_dim_lists(&result);
        // Drop a trailing default `, ()` brand from `MixedUnitMatrix<…>` so unbranded
        // matrices read as `MixedUnitMatrix<[…], […], Matrix<…>>`.
        result = self.strip_default_brand(&result);
        result
    }

    /// Remove a trailing default brand (`, ()`) from every `MixedUnitMatrix<…>` in
    /// `text`. `Brand` is the last, defaulted (`()`) type parameter, so an
    /// unbranded matrix carries a redundant `()` we elide from the display.
    fn strip_default_brand(&self, text: &str) -> String {
        if !text.contains("MixedUnitMatrix<") {
            return text.to_string();
        }

        let bytes = text.as_bytes();
        let mut result = String::new();
        let mut i = 0;

        while i < text.len() {
            if text[i..].starts_with("MixedUnitMatrix<") && Self::is_ident_boundary(text, i) {
                let lt = i + 15; // byte index of the '<' after "MixedUnitMatrix"
                if let Some(gt) = Self::find_matching_angle(bytes, lt) {
                    if let Some(new_inner) = self.drop_trailing_unit_brand(&text[lt + 1..gt]) {
                        result.push_str("MixedUnitMatrix<");
                        result.push_str(&new_inner);
                        result.push('>');
                        i = gt + 1;
                        continue;
                    }
                }
            }

            let ch = text[i..].chars().next().unwrap();
            let len = ch.len_utf8();
            result.push_str(&text[i..i + len]);
            i += len;
        }

        result
    }

    /// Build a values-free diagram of the first concrete unit matrix in `text`:
    /// a [`MixedUnitMatrix<…>`](Self::mixed_matrix_trace) (row/column margins and
    /// per-cell quotient units) or, failing that, a
    /// [`UniformUnitMatrix<…>`](Self::uniform_matrix_trace) (one unit per cell, no
    /// margins — a uniform matrix has no gauge). `None` when neither is present or
    /// its shape/units are generic / not fully resolvable.
    fn unit_matrix_trace(&self, text: &str) -> Option<String> {
        self.mixed_matrix_trace(text)
            .or_else(|| self.uniform_matrix_trace(text))
    }

    /// The trace for the first concrete `MixedUnitMatrix<…>`: row/column unit
    /// labels in the margins, and each entry's unit literal
    /// (`RowDims[i] / ColDims[j]`) where its value would be. `None` when there is
    /// no `MixedUnitMatrix`, or its dimension lists are generic / not fully
    /// resolvable.
    fn mixed_matrix_trace(&self, text: &str) -> Option<String> {
        let bytes = text.as_bytes();
        let mut search = 0;
        loop {
            let rel = text[search..].find("MixedUnitMatrix<")?;
            let start = search + rel;
            if Self::is_ident_boundary(text, start) {
                let lt = start + 15; // '<' after "MixedUnitMatrix"
                let gt = Self::find_matching_angle(bytes, lt)?;
                let args = Self::split_top_level(&text[lt + 1..gt]);
                if args.len() >= 2 {
                    if let (Some(rows), Some(cols)) =
                        (self.parse_dim_list(args[0]), self.parse_dim_list(args[1]))
                    {
                        if !rows.is_empty() && !cols.is_empty() {
                            return Some(self.render_trace(&rows, &cols));
                        }
                    }
                }
                return None;
            }
            search = start + 15;
        }
    }

    /// The trace for the first concrete `UniformUnitMatrix<U, Matrix<…>, …>`: an
    /// `R × C` grid with the single entry unit `U` in every cell and no
    /// margins — unlike the mixed diagram, a uniform matrix carries one shared
    /// unit and no row/column gauge to label. `None` when there is no
    /// `UniformUnitMatrix`, or its unit / shape (`Const<_>` dims) is generic / not
    /// fully resolvable.
    fn uniform_matrix_trace(&self, text: &str) -> Option<String> {
        let bytes = text.as_bytes();
        let mut search = 0;
        loop {
            let rel = text[search..].find("UniformUnitMatrix<")?;
            let start = search + rel;
            if Self::is_ident_boundary(text, start) {
                let lt = start + 17; // '<' after "UniformUnitMatrix"
                let gt = Self::find_matching_angle(bytes, lt)?;
                let args = Self::split_top_level(&text[lt + 1..gt]);
                if args.len() >= 2 {
                    if let (Some(unit), Some((r, c))) = (
                        self.parse_new_quantity_params(args[0].trim()),
                        Self::parse_matrix_dims(args[1]),
                    ) {
                        if r > 0 && c > 0 {
                            return Some(self.render_uniform_trace(&unit, r, c));
                        }
                    }
                }
                return None;
            }
            search = start + 17;
        }
    }

    /// Extract the `(rows, cols)` shape from a nalgebra storage type argument —
    /// `Matrix<T, R, C, S>` / `SMatrix<T, R, C>` / `OMatrix<T, R, C>` — reading
    /// `R`/`C` as its second/third generic arguments. `None` unless both resolve
    /// to a concrete size (a `Const<N>` or a bare integer); a `Dyn` (dynamic) or
    /// generic dimension yields no trace.
    fn parse_matrix_dims(arg: &str) -> Option<(usize, usize)> {
        let arg = arg.trim();
        let lt = arg.find('<')?;
        if !arg[..lt].ends_with("Matrix") {
            return None;
        }
        let bytes = arg.as_bytes();
        let gt = Self::find_matching_angle(bytes, lt)?;
        if gt + 1 != arg.len() {
            return None;
        }
        let parts = Self::split_top_level(&arg[lt + 1..gt]);
        if parts.len() < 3 {
            return None;
        }
        Some((Self::parse_dim_size(parts[1])?, Self::parse_dim_size(parts[2])?))
    }

    /// Parse one nalgebra dimension argument to a concrete size: a bare integer
    /// (`SMatrix<f64, 2, 1>`) or a `Const<N>` (possibly path-qualified). `None`
    /// for `Dyn` or a generic dimension.
    fn parse_dim_size(s: &str) -> Option<usize> {
        let s = s.trim();
        if let Ok(n) = s.parse::<usize>() {
            return Some(n);
        }
        let idx = s.find("Const<")?;
        let inner = &s[idx + 6..];
        let end = inner.find('>')?;
        inner[..end].trim().parse::<usize>().ok()
    }

    /// Render the marginless `R × C` diagram for a uniform matrix: the single
    /// entry unit `U` repeated in every cell. Every cell is identical, so the
    /// columns need no per-column width negotiation.
    fn render_uniform_trace(&self, unit: &QuantityParams, r: usize, c: usize) -> String {
        let label = self.unit_label(unit.dimensions, unit.scale);
        let mut out = String::new();
        for i in 0..r {
            out.push('[');
            for j in 0..c {
                if j > 0 {
                    out.push_str("  ");
                }
                out.push_str(&label);
            }
            out.push(']');
            if i + 1 < r {
                out.push('\n');
            }
        }
        out
    }

    /// Parse a dimension-list argument (`DCons<Unit<…>, …, DNil>` or `DNil`) into
    /// one `QuantityParams` per entry. Returns `None` if it isn't a resolvable
    /// cons-list of `Unit`s.
    fn parse_dim_list(&self, arg: &str) -> Option<Vec<QuantityParams>> {
        let arg = arg.trim();
        if arg == "DNil" {
            return Some(Vec::new());
        }
        if !arg.starts_with("DCons<") {
            return None;
        }
        let bytes = arg.as_bytes();
        let gt = Self::find_matching_angle(bytes, 5)?; // '<' after "DCons"
        if gt + 1 != arg.len() {
            return None;
        }
        let elems = self.parse_cons_list(&arg[6..gt])?;
        let mut out = Vec::with_capacity(elems.len());
        for e in elems {
            out.push(self.parse_new_quantity_params(&e)?);
        }
        Some(out)
    }

    /// The quotient unit `row / col` (dimension exponents and scale subtracted),
    /// or `None` if either operand carries an unresolved component.
    // The loop index addresses three parallel fixed-width arrays (`dims`, plus
    // the two operands) and bails out mid-iteration on an unresolved component,
    // so an index walk reads more clearly here than a zipped iterator.
    #[allow(clippy::needless_range_loop)]
    fn quotient(
        row: &QuantityParams,
        col: &QuantityParams,
    ) -> Option<(DynDimensionExponents, ScaleExponents)> {
        let mut dims = [0i16; 8];
        for k in 0..8 {
            let (a, b) = (row.dimensions.0[k], col.dimensions.0[k]);
            if a == i16::MIN || b == i16::MIN {
                return None;
            }
            dims[k] = a - b;
        }
        let mut scale = [0i16; 4];
        for k in 0..4 {
            let (a, b) = (row.scale.0[k], col.scale.0[k]);
            if a == i16::MIN || b == i16::MIN {
                return None;
            }
            scale[k] = a - b;
        }
        Some((DynDimensionExponents(dims), ScaleExponents(scale)))
    }

    /// The unit label for a `(dims, scale)` pair; a dimensionless, unscaled unit
    /// renders as the scalar `1` (matching the `Display` impl's convention).
    fn unit_label(&self, dims: DynDimensionExponents, scale: ScaleExponents) -> String {
        use whippyunits::print::prettyprint::pretty_print_unit_label;
        if dims == DynDimensionExponents::ZERO && scale == ScaleExponents::IDENTITY {
            return "1".to_string();
        }
        let label = pretty_print_unit_label(dims, scale);
        if label.is_empty() || label == "?" {
            "1".to_string()
        } else {
            label
        }
    }

    /// Render the aligned matrix diagram from parsed row/column dimension lists.
    fn render_trace(&self, rows: &[QuantityParams], cols: &[QuantityParams]) -> String {
        let row_labels: Vec<String> = rows.iter().map(|r| self.unit_label(r.dimensions, r.scale)).collect();
        let col_labels: Vec<String> = cols.iter().map(|c| self.unit_label(c.dimensions, c.scale)).collect();

        let cells: Vec<Vec<String>> = rows
            .iter()
            .map(|r| {
                cols.iter()
                    .map(|c| match Self::quotient(r, c) {
                        Some((d, s)) => self.unit_label(d, s),
                        None => "?".to_string(),
                    })
                    .collect()
            })
            .collect();

        let width = |s: &str| s.chars().count();
        let rjust = |s: &str, w: usize| format!("{}{}", " ".repeat(w.saturating_sub(width(s))), s);
        let center = |s: &str, w: usize| {
            let pad = w.saturating_sub(width(s));
            let left = pad / 2;
            format!("{}{}{}", " ".repeat(left), s, " ".repeat(pad - left))
        };

        let ncols = cols.len();
        let col_w: Vec<usize> = (0..ncols)
            .map(|j| {
                let mut w = width(&col_labels[j]);
                for row in &cells {
                    w = w.max(width(&row[j]));
                }
                w
            })
            .collect();
        let lpad = row_labels.iter().map(|l| width(l)).max().unwrap_or(0);

        let mut out = String::new();

        // Top margin: column unit labels, centered over their columns. The two
        // leading spaces line the first column up with the `[`-prefixed body.
        out.push_str(&" ".repeat(lpad));
        out.push_str("  ");
        for j in 0..ncols {
            if j > 0 {
                out.push_str("  ");
            }
            out.push_str(&center(&col_labels[j], col_w[j]));
        }
        out.push('\n');

        // Body: each row prefixed by its right-justified row unit label.
        for (i, row) in cells.iter().enumerate() {
            out.push_str(&rjust(&row_labels[i], lpad));
            out.push_str(" [");
            for (j, cell) in row.iter().enumerate() {
                if j > 0 {
                    out.push_str("  ");
                }
                out.push_str(&rjust(cell, col_w[j]));
            }
            out.push(']');
            if i + 1 < cells.len() {
                out.push('\n');
            }
        }

        out
    }

    /// Split a generic-argument list on its top-level commas.
    fn split_top_level(inner: &str) -> Vec<&str> {
        let bytes = inner.as_bytes();
        let mut depth = 0i32;
        let mut parts = Vec::new();
        let mut start = 0;
        let mut i = 0;
        while i < bytes.len() {
            match bytes[i] {
                b'<' | b'(' | b'[' => depth += 1,
                b'>' | b')' | b']' => depth -= 1,
                b',' if depth == 0 => {
                    parts.push(&inner[start..i]);
                    start = i + 1;
                }
                _ => {}
            }
            i += 1;
        }
        parts.push(&inner[start..]);
        parts
    }

    /// If the last top-level argument of a `MixedUnitMatrix<…>` inner is `()`, return
    /// the inner with that trailing brand removed; otherwise `None`.
    fn drop_trailing_unit_brand(&self, inner: &str) -> Option<String> {
        let bytes = inner.as_bytes();
        let mut depth = 0i32;
        let mut last_comma = None;
        let mut i = 0;
        while i < bytes.len() {
            match bytes[i] {
                b'<' | b'(' | b'[' => depth += 1,
                b'>' | b')' | b']' => depth -= 1,
                b',' if depth == 0 => last_comma = Some(i),
                _ => {}
            }
            i += 1;
        }
        let lc = last_comma?;
        if inner[lc + 1..].trim() == "()" {
            Some(inner[..lc].trim_end().to_string())
        } else {
            None
        }
    }

    /// Collapse type-level cons-lists (`DCons<Head, …, DNil>`, as used by
    /// whippyalgebra's `MixedUnitMatrix`) into bracketed array notation (`[Head, …]`),
    /// recursively. Only well-formed, `DNil`-terminated lists are collapsed;
    /// anything else is left untouched.
    fn collapse_dim_lists(&self, text: &str) -> String {
        if !text.contains("DCons<") {
            return text.to_string();
        }

        let bytes = text.as_bytes();
        let mut result = String::new();
        let mut i = 0;

        while i < text.len() {
            if text[i..].starts_with("DCons<") && Self::is_ident_boundary(text, i) {
                let lt = i + 5; // byte index of the '<' after "DCons"
                if let Some(gt) = Self::find_matching_angle(bytes, lt) {
                    if let Some(elems) = self.parse_cons_list(&text[lt + 1..gt]) {
                        result.push('[');
                        result.push_str(&elems.join(", "));
                        result.push(']');
                        i = gt + 1;
                        continue;
                    }
                }
            }

            // Copy the next full UTF-8 char (unit labels contain multibyte chars).
            let ch = text[i..].chars().next().unwrap();
            let len = ch.len_utf8();
            result.push_str(&text[i..i + len]);
            i += len;
        }

        result
    }

    /// Whether the byte at `i` begins a fresh identifier (not preceded by an
    /// identifier char), so `DCons` inside a longer name isn't matched.
    fn is_ident_boundary(text: &str, i: usize) -> bool {
        if i == 0 {
            return true;
        }
        match text[..i].chars().next_back() {
            Some(prev) => !(prev.is_alphanumeric() || prev == '_'),
            None => true,
        }
    }

    /// Byte index of the '>' matching the '<' at byte index `lt`.
    fn find_matching_angle(bytes: &[u8], lt: usize) -> Option<usize> {
        let mut depth = 0i32;
        let mut i = lt;
        while i < bytes.len() {
            match bytes[i] {
                b'<' => depth += 1,
                b'>' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i);
                    }
                }
                _ => {}
            }
            i += 1;
        }
        None
    }

    /// Parse the inside of a `DCons<…>` (`Head, Tail`) into collapsed elements.
    /// Returns `None` unless the list is cleanly `DNil`-terminated.
    fn parse_cons_list(&self, inner: &str) -> Option<Vec<String>> {
        let comma = self.top_level_comma(inner)?;
        let head = inner[..comma].trim();
        let tail = inner[comma + 1..].trim();

        let mut elems = vec![self.collapse_dim_lists(head)];

        if tail == "DNil" {
            return Some(elems);
        }
        if tail.starts_with("DCons<") {
            let tail_bytes = tail.as_bytes();
            let lt = 5; // '<' after "DCons"
            let gt = Self::find_matching_angle(tail_bytes, lt)?;
            // The tail must be exactly one `DCons<…>` with nothing trailing.
            if gt + 1 != tail.len() {
                return None;
            }
            let mut rest = self.parse_cons_list(&tail[lt + 1..gt])?;
            elems.append(&mut rest);
            return Some(elems);
        }
        None
    }

    /// Byte index of the first top-level comma in `inner`, tracking nesting of
    /// `<>`, `()`, and `[]`.
    fn top_level_comma(&self, inner: &str) -> Option<usize> {
        let bytes = inner.as_bytes();
        let mut depth = 0i32;
        let mut i = 0;
        while i < bytes.len() {
            match bytes[i] {
                b'<' | b'(' | b'[' => depth += 1,
                b'>' | b')' | b']' => depth -= 1,
                b',' if depth == 0 => return Some(i),
                _ => {}
            }
            i += 1;
        }
        None
    }

    /// First pass: rewrite `Quantity<Unit<…>, …>` wrappers.
    fn format_quantity_pass(
        &self,
        text: &str,
        verbose: bool,
        unicode: bool,
        is_inlay_hint: bool,
    ) -> String {
        if text.contains("Quantity<Unit<") && text.contains("Scale") {
            // First pass: find all Quantity<Unit<Scale types and their positions
            struct QuantityMatch {
                start: usize,
                end: usize,
                formatted: String,
            }

            let mut matches = Vec::new();
            let mut i = 0;

            while i < text.len() {
                // Search for "Quantity<Unit<" starting from position i
                if let Some(relative_start) = text[i..].find("Quantity<Unit<") {
                    let start_pos = i + relative_start;

                    // Count brackets to find the matching end of this Quantity type
                    // start_pos points to "Q" in "Quantity<Unit<"
                    // start_pos + 8 points to the '<' after "Quantity"
                    let quantity_start = start_pos + 8; // Position of the '<'
                                                        // Start bracket_count at 1 because we're already inside Quantity<
                                                        // Start j after the '<' so we don't count it again
                    let mut bracket_count = 1;
                    let mut found_end = false;
                    // Use byte indices, not char indices, to match the string slicing
                    let mut byte_pos = quantity_start + 1; // Start after the opening '<'
                    let text_bytes = text.as_bytes();

                    while byte_pos < text_bytes.len() {
                        match text_bytes[byte_pos] {
                            b'<' => bracket_count += 1,
                            b'>' => {
                                bracket_count -= 1;
                                if bracket_count == 0 {
                                    found_end = true;
                                    break;
                                }
                            }
                            _ => {}
                        }
                        byte_pos += 1;
                    }

                    if found_end {
                        let actual_end = byte_pos + 1; // +1 to include the '>'
                                                       // Ensure we don't go past the end of the string
                        let actual_end = actual_end.min(text.len());
                        let quantity_type = &text[start_pos..actual_end];

                        let formatted = self.format_new_quantity_type(
                            quantity_type,
                            verbose,
                            unicode,
                            is_inlay_hint,
                        );

                        matches.push(QuantityMatch {
                            start: start_pos,
                            end: actual_end,
                            formatted,
                        });

                        // Continue searching from after this Quantity type
                        i = actual_end;
                    } else {
                        // Bracket counting failed, stop searching
                        break;
                    }
                } else {
                    // No more Quantity<Scale found
                    break;
                }
            }

            // Second pass: replace all matches from end to start to preserve positions
            if matches.is_empty() {
                return text.to_string();
            }

            let mut result = text.to_string();
            // Replace from end to start to preserve positions
            for m in matches.iter().rev() {
                result.replace_range(m.start..m.end, &m.formatted);
            }

            return result;
        }

        // If we reach here, no new format was found, return original text
        text.to_string()
    }

    /// Second pass: rewrite bare `Unit<Scale<…>, Dimension<…>>` types (as
    /// produced by `whippyalgebra`) into their unit label, in place.
    fn format_unit_pass(
        &self,
        text: &str,
        verbose: bool,
        unicode: bool,
        is_inlay_hint: bool,
    ) -> String {
        struct UnitMatch {
            start: usize,
            end: usize,
            formatted: String,
        }

        let mut matches = Vec::new();
        let mut i = 0;

        while i < text.len() {
            // Search for "Unit<Scale" starting from position i
            if let Some(relative_start) = text[i..].find("Unit<Scale") {
                let start_pos = i + relative_start;
                // start_pos points to "U" in "Unit<Scale"; +4 is the '<' after "Unit"
                let unit_open = start_pos + 4; // Position of the '<'
                let mut bracket_count = 1;
                let mut found_end = false;
                let mut byte_pos = unit_open + 1;
                let text_bytes = text.as_bytes();

                while byte_pos < text_bytes.len() {
                    match text_bytes[byte_pos] {
                        b'<' => bracket_count += 1,
                        b'>' => {
                            bracket_count -= 1;
                            if bracket_count == 0 {
                                found_end = true;
                                break;
                            }
                        }
                        _ => {}
                    }
                    byte_pos += 1;
                }

                if found_end {
                    let actual_end = (byte_pos + 1).min(text.len());
                    let unit_type = &text[start_pos..actual_end];
                    let formatted =
                        self.format_unit_type(unit_type, verbose, unicode, is_inlay_hint);
                    matches.push(UnitMatch {
                        start: start_pos,
                        end: actual_end,
                        formatted,
                    });
                    i = actual_end;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        if matches.is_empty() {
            return text.to_string();
        }

        let mut result = text.to_string();
        for m in matches.iter().rev() {
            result.replace_range(m.start..m.end, &m.formatted);
        }
        result
    }

    /// Format a bare `Unit<Scale<…>, Dimension<…>>` type into its unit label.
    ///
    /// A dimensionless, unscaled unit renders as `1`; a wholly-unresolved unit
    /// renders as `?`; otherwise the SI-preferred unit literal (e.g. `m/s`).
    fn format_unit_type(
        &self,
        full_match: &str,
        _verbose: bool,
        _unicode: bool,
        _is_inlay_hint: bool,
    ) -> String {
        use whippyunits::print::prettyprint::pretty_print_unit_label;

        // Leave generic type definitions (e.g. `Unit<Scale, Dimension>` inside a
        // `T = f64` signature) untouched.
        if self.is_generic_type_definition(full_match) {
            return full_match.to_string();
        }

        if let Some(params) = self.parse_new_quantity_params(full_match) {
            let all_dimensions_unresolved = params.dimensions.0.iter().all(|&exp| exp == i16::MIN);
            let all_scales_unresolved = params.scale.0.iter().all(|&exp| exp == i16::MIN);
            if all_dimensions_unresolved && all_scales_unresolved {
                return "?".to_string();
            }

            if params.dimensions == DynDimensionExponents::ZERO
                && params.scale == ScaleExponents::IDENTITY
            {
                return "1".to_string();
            }

            let label = pretty_print_unit_label(params.dimensions, params.scale);
            if label.is_empty() || label == "?" {
                "1".to_string()
            } else {
                label
            }
        } else {
            full_match.to_string()
        }
    }

    /// Format the new Quantity type with Scale<...> and Dimension<...> structs
    fn format_new_quantity_type(
        &self,
        full_match: &str,
        verbose: bool,
        _unicode: bool,
        is_inlay_hint: bool,
    ) -> String {
        use whippyunits::print::prettyprint::pretty_print_quantity_type;

        // Check if this is a generic type definition (contains parameter names like Scale, Dimension, T)
        // rather than a concrete instantiation with actual values
        if self.is_generic_type_definition(full_match) {
            // Pass through generic definitions unchanged
            return full_match.to_string();
        }

        // Parse the new format: Quantity<Scale<_2<P2>, _3<P3>, _5<P5>, _Pi<PI>>, Dimension<_M<MASS>, _L<LENGTH>, _T<TIME>, _I<CURRENT>, _Θ<TEMP>, _N<AMOUNT>, _J<LUMINOSITY>, _A<ANGLE>>, T>
        if let Some(params) = self.parse_new_quantity_params(full_match) {
            // Check if this is a wholly unresolved type (all parameters are sentinel values)
            let all_dimensions_unresolved = params.dimensions.0.iter().all(|&exp| exp == i16::MIN);
            let all_scales_unresolved = params.scale.0.iter().all(|&exp| exp == i16::MIN);

            // Get Brand name for passing to prettyprint function
            let brand_name = params.brand.as_deref();

            if all_dimensions_unresolved && all_scales_unresolved {
                // Format as wholly unresolved type
                let mut result = format!("Quantity<?, {}>", params.generic_type);
                if let Some(brand) = brand_name {
                    if brand != "()" {
                        result = format!("{}, {}>", &result[..result.len() - 1], brand);
                    }
                }
                return result;
            }

            // Check if this is a dimensionless quantity (all dimensions are zero)
            if params.dimensions == DynDimensionExponents::ZERO
                && params.scale == ScaleExponents::IDENTITY
            {
                // Format as dimensionless quantity
                let mut result = format!("Quantity<1, {}>", params.generic_type);
                if let Some(brand) = brand_name {
                    if brand != "()" {
                        result = format!("{}, {}>", &result[..result.len() - 1], brand);
                    }
                }
                return result;
            }
            if is_inlay_hint {
                // Use the main pretty print function with verbose=false to get the unit literal
                let full_output = pretty_print_quantity_type(
                    params.dimensions,
                    params.scale,
                    &params.generic_type,
                    false, // Non-verbose mode for inlay hints
                    false, // Don't show type in brackets
                    brand_name,
                );

                // Check if the pretty print function returned just "?" for wholly unresolved types
                if full_output == "?" {
                    let mut result = format!("Quantity<?, {}>", params.generic_type);
                    if let Some(brand) = brand_name {
                        if brand != "()" {
                            result = format!("{}, {}>", &result[..result.len() - 1], brand);
                        }
                    }
                    return result;
                }

                full_output
            } else {
                // Use the prettyprint API with configurable parameters
                let result = pretty_print_quantity_type(
                    params.dimensions,
                    params.scale,
                    &params.generic_type,
                    verbose,
                    false, // show_type_in_brackets = false for pretty printer
                    brand_name,
                );

                // Check if the pretty print function returned just "?" for wholly unresolved types
                if result == "?" {
                    let mut formatted = format!("Quantity<?, {}>", params.generic_type);
                    if let Some(brand) = brand_name {
                        if brand != "()" {
                            formatted =
                                format!("{}, {}>", &formatted[..formatted.len() - 1], brand);
                        }
                    }
                    return formatted;
                }

                result
            }
        } else {
            // If parsing fails, return the original
            full_match.to_string()
        }
    }

    /// Parse the new Quantity type format with Scale<...> and Dimension<...> structs
    fn parse_new_quantity_params(&self, quantity_type: &str) -> Option<QuantityParams> {
        // Parse Scale parameters - handle all possible combinations of defaulted parameters
        let scale = if quantity_type.contains("Scale<") {
            // Any Scale format with parameters - handle all combinations of defaulted values
            let (p2, p3, p5, pi) = self.parse_scale_general_format(quantity_type)?;
            ScaleExponents([p2, p3, p5, pi])
        } else if quantity_type.contains("Scale,") {
            // Truncated format: Scale, or Scale> or Scale, Dimension (all parameters default to 0)
            ScaleExponents::IDENTITY
        } else {
            // Unknown format
            return None;
        };

        // Parse Dimension parameters - handle both full format and truncated format
        let dimensions = if quantity_type.contains("Dimension<_M<") && quantity_type.contains("_A<")
        {
            // Full format: Dimension<_M<MASS>, _L<LENGTH>, _T<TIME>, _I<CURRENT>, _Θ<TEMP>, _N<AMOUNT>, _J<LUMINOSITY>, _A<ANGLE>>
            let (mass, length, time, current, temp, amount, lum, angle) =
                self.parse_dimension_full_format(quantity_type)?;
            DynDimensionExponents([mass, length, time, current, temp, amount, lum, angle])
        } else if quantity_type.contains("Dimension,") || quantity_type.contains("Dimension>") {
            // Fully defaulted Dimension (dimensionless): Dimension, T or Dimension> T
            DynDimensionExponents::ZERO
        } else {
            // Truncated format: parse only the non-zero parameters
            // Look for patterns like Dimension<_M<0>, _L<1>> (only non-zero parameters are shown)
            let (mass, length, time, current, temp, amount, lum, angle) =
                self.parse_dimension_truncated_format(quantity_type);
            DynDimensionExponents([mass, length, time, current, temp, amount, lum, angle])
        };

        // Don't apply base scale offset here - let the prettyprint functions handle it
        // The prettyprint functions already have the correct base scale offset logic
        let adjusted_scale = scale;

        // Extract the actual generic type parameter and brand from the type string
        let (generic_type, brand) = self.extract_generic_type_and_brand(quantity_type);

        Some(QuantityParams {
            dimensions,
            scale: adjusted_scale,
            generic_type,
            brand,
        })
    }

    /// Parse general Scale format: Scale<_2[<P2>], _3[<P3>], _5[<P5>], _Pi[<PI>]>
    /// Handles all possible combinations of defaulted parameters
    /// Parameters can be either _2<value> (explicit) or _2 (defaulted to 0)
    fn parse_scale_general_format(&self, quantity_type: &str) -> Option<(i16, i16, i16, i16)> {
        let scale_start = quantity_type.find("Scale<")?;
        let scale_content = &quantity_type[scale_start + 6..]; // Skip "Scale<"

        // Find the end of the Scale struct
        let scale_end = self.find_matching_bracket(scale_content, 0)?;
        let scale_params = &scale_content[..scale_end];

        // Parse each parameter, handling both explicit values and defaults
        let p2 = self.parse_scale_param_with_default(scale_params, "_2");
        let p3 = self.parse_scale_param_with_default(scale_params, "_3");
        let p5 = self.parse_scale_param_with_default(scale_params, "_5");
        let pi = self.parse_scale_param_with_default(scale_params, "_Pi");

        Some((p2, p3, p5, pi))
    }

    /// Parse a scale parameter that may be either explicit (_2<value>) or defaulted (_2)
    // The "defaulted form" and "not found" cases both yield 0 but are kept as
    // separate branches to document the two distinct parse outcomes.
    #[allow(clippy::if_same_then_else)]
    fn parse_scale_param_with_default(&self, content: &str, prefix: &str) -> i16 {
        // First try to find explicit value: _2<value>
        if let Some(value) = self.parse_scale_param(content, &format!("{}<", prefix)) {
            value
        } else {
            // Check if parameter exists in defaulted form: _2 (without <value>)
            if content.contains(&format!("{},", prefix)) || content.ends_with(prefix) {
                0 // Default value
            } else {
                0 // Parameter not found, default to 0
            }
        }
    }

    /// Parse full Dimension format: Dimension<_M<MASS>, _L<LENGTH>, _T<TIME>, _I<CURRENT>, _Θ<TEMP>, _N<AMOUNT>, _J<LUMINOSITY>, _A<ANGLE>>
    fn parse_dimension_full_format(
        &self,
        quantity_type: &str,
    ) -> Option<(i16, i16, i16, i16, i16, i16, i16, i16)> {
        let dimension_start = quantity_type.find("Dimension<_M<")?;
        let dimension_content = &quantity_type[dimension_start + 9..]; // Skip "Dimension<"

        // Parse each parameter directly from the dimension content. A zero
        // exponent is the const-generic default (`_Θ<const EXP: i16 = 0>`), so
        // rust-analyzer prints it *without* its `<0>` (a bare `_Θ`). A missing
        // marker therefore means exponent 0 — not a parse failure. (This branch
        // is entered whenever the angle marker is explicit, `_A<…>`, which is
        // common for angular units like `rot`/`rad` even when the interior
        // temperature/amount/luminosity dims are defaulted to zero.)
        let mass = self.parse_dimension_param(dimension_content, "_M<").unwrap_or(0);
        let length = self.parse_dimension_param(dimension_content, "_L<").unwrap_or(0);
        let time = self.parse_dimension_param(dimension_content, "_T<").unwrap_or(0);
        let current = self.parse_dimension_param(dimension_content, "_I<").unwrap_or(0);
        let temp = self.parse_dimension_param(dimension_content, "_Θ<").unwrap_or(0);
        let amount = self.parse_dimension_param(dimension_content, "_N<").unwrap_or(0);
        let lum = self.parse_dimension_param(dimension_content, "_J<").unwrap_or(0);
        let angle = self.parse_dimension_param(dimension_content, "_A<").unwrap_or(0);

        Some((mass, length, time, current, temp, amount, lum, angle))
    }

    /// Parse truncated Dimension format: Dimension<_M<0>, _L<1>> (only non-zero parameters are shown)
    fn parse_dimension_truncated_format(
        &self,
        quantity_type: &str,
    ) -> (i16, i16, i16, i16, i16, i16, i16, i16) {
        let mut mass_exp = 0;
        let mut length_exp = 0;
        let mut time_exp = 0;
        let mut electric_current_exp = 0;
        let mut temperature_exp = 0;
        let mut amount_of_substance_exp = 0;
        let mut luminous_intensity_exp = 0;
        let mut angle_exp = 0;

        // Parse individual dimension parameters that are present
        if let Some(value) = self.parse_dimension_param(quantity_type, "_M<") {
            mass_exp = value;
        }
        if let Some(value) = self.parse_dimension_param(quantity_type, "_L<") {
            length_exp = value;
        }
        if let Some(value) = self.parse_dimension_param(quantity_type, "_T<") {
            time_exp = value;
        }
        if let Some(value) = self.parse_dimension_param(quantity_type, "_I<") {
            electric_current_exp = value;
        }
        if let Some(value) = self.parse_dimension_param(quantity_type, "_Θ<") {
            temperature_exp = value;
        }
        if let Some(value) = self.parse_dimension_param(quantity_type, "_N<") {
            amount_of_substance_exp = value;
        }
        if let Some(value) = self.parse_dimension_param(quantity_type, "_J<") {
            luminous_intensity_exp = value;
        }
        if let Some(value) = self.parse_dimension_param(quantity_type, "_A<") {
            angle_exp = value;
        }

        (
            mass_exp,
            length_exp,
            time_exp,
            electric_current_exp,
            temperature_exp,
            amount_of_substance_exp,
            luminous_intensity_exp,
            angle_exp,
        )
    }

    /// Parse a scale parameter like "_2<5>" and return the value
    fn parse_scale_param(&self, content: &str, prefix: &str) -> Option<i16> {
        let start = content.find(prefix)?;
        let param_start = start + prefix.len();
        let param_end = content[param_start..].find('>')?;
        let param_value = &content[param_start..param_start + param_end];
        Some(self.parse_parameter(param_value))
    }

    /// Parse a dimension parameter like "_M<1>" and return the value
    fn parse_dimension_param(&self, content: &str, prefix: &str) -> Option<i16> {
        let start = content.find(prefix)?;
        let param_start = start + prefix.len();
        let param_end = content[param_start..].find('>')?;
        let param_value = &content[param_start..param_start + param_end];
        let result = self.parse_parameter(param_value);
        Some(result)
    }

    /// Find the matching closing bracket for a given opening bracket
    fn find_matching_bracket(&self, content: &str, start_pos: usize) -> Option<usize> {
        let mut depth = 1;
        let mut i = start_pos;

        while i < content.len() {
            match content.chars().nth(i) {
                Some('<') => depth += 1,
                Some('>') => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i);
                    }
                }
                _ => {}
            }
            i += 1;
        }
        None
    }

    /// Extract both the generic type parameter and brand from a Quantity type string
    /// Returns (generic_type, brand) where brand is Some(...) if present, None otherwise
    fn extract_generic_type_and_brand(&self, quantity_type: &str) -> (String, Option<String>) {
        // Parse as a Rust type using syn - it handles const generics reliably
        if let Some(quantity_start) = quantity_type.find("Quantity<") {
            let quantity_content = &quantity_type[quantity_start..];

            match parse_str::<TypePath>(quantity_content) {
                Ok(type_path) => {
                    // Extract generic arguments
                    if let Some(segment) = type_path.path.segments.last() {
                        if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                            let args_vec: Vec<_> = args.args.iter().collect();

                            // We expect: [Unit<Scale, Dimension>, T, Brand?]
                            // Unit is at index 0, type T at 1, Brand at 2 (if present)

                            if args_vec.len() >= 2 {
                                // Extract the type parameter (index 1)
                                let type_arg = &args_vec[1];
                                let generic_type = match type_arg {
                                    syn::GenericArgument::Type(ty) => {
                                        quote::quote!(#ty).to_string()
                                    }
                                    _ => quote::quote!(#type_arg).to_string(),
                                };

                                // Extract Brand if present (index 2)
                                let brand = if args_vec.len() >= 3 {
                                    let brand_arg = &args_vec[2];
                                    match brand_arg {
                                        syn::GenericArgument::Type(ty) => {
                                            Some(quote::quote!(#ty).to_string())
                                        }
                                        _ => Some(quote::quote!(#brand_arg).to_string()),
                                    }
                                } else {
                                    None
                                };

                                return (
                                    generic_type.trim().to_string(),
                                    brand.map(|s| s.trim().to_string()),
                                );
                            } else if args_vec.len() == 1 {
                                // Only Unit<Scale, Dimension>, type defaults to f64
                                return ("f64".to_string(), None);
                            }
                        }
                    }
                }
                Err(_) => {
                    // syn parsing failed - this shouldn't happen for valid Rust types
                    // but default to f64 if it does
                }
            }
        }

        // Default fallback if parsing fails
        ("f64".to_string(), None)
    }

    /// Parse a parameter that could be a number or underscore placeholder.
    /// Non-numeric identifiers (e.g. const generic names like `SCALE_P2`) are
    /// treated as unresolved, same as `_`.
    fn parse_parameter(&self, param: &str) -> i16 {
        param.parse().unwrap_or(i16::MIN)
    }

    /// Extract just the raw type information from hover content
    /// Looks for any type declaration pattern: `let [mut] [var]: TypeName<...>`
    pub fn extract_raw_type_from_hover(&self, hover_text: &str) -> String {
        // Look for any type declaration pattern: ": TypeName<"
        // Find the first occurrence of ": " followed by something that looks like a type with generics
        if let Some(start) = hover_text.find(": ") {
            let after_colon = &hover_text[start + 2..];

            // Find where the type name ends (either at '<' for generics, or at whitespace/newline)
            let type_end = after_colon
                .char_indices()
                .find(|(_, ch)| *ch == '<' || *ch == '\n' || *ch == '\r')
                .map(|(i, _)| i);

            if let Some(type_end) = type_end {
                if after_colon.chars().nth(type_end) == Some('<') {
                    // We found a generic type, extract the full declaration
                    // Find the start of the variable declaration by looking backwards for 'let'
                    let mut var_start = start;
                    let mut found_let = false;

                    // Look backwards to find the start of the declaration
                    while var_start > 0 {
                        let char_before =
                            hover_text[var_start - 1..var_start].chars().next().unwrap();
                        if char_before.is_whitespace() {
                            // Check if we've found "let" or "let mut"
                            let potential_start = var_start;
                            let before_whitespace = &hover_text[..potential_start];

                            // Look for "let" at the end of the string
                            if before_whitespace.ends_with("let") {
                                // Check if there's whitespace before "let" or if it's at the start
                                let let_start = before_whitespace.len() - 3;
                                if let_start == 0
                                    || hover_text[let_start - 1..let_start]
                                        .chars()
                                        .next()
                                        .unwrap()
                                        .is_whitespace()
                                {
                                    var_start = let_start;
                                    found_let = true;
                                    break;
                                }
                            }
                        }
                        var_start -= 1;
                    }

                    if !found_let {
                        // Fallback: find the start of the variable name
                        while var_start > 0
                            && !hover_text[var_start..var_start + 1]
                                .chars()
                                .next()
                                .unwrap()
                                .is_whitespace()
                        {
                            var_start -= 1;
                        }
                        if var_start > 0
                            && hover_text[var_start..var_start + 1]
                                .chars()
                                .next()
                                .unwrap()
                                .is_whitespace()
                        {
                            var_start += 1; // Skip the whitespace
                        }
                    }

                    // Find the end of the type declaration by matching brackets
                    let type_start = start + ": ".len();
                    let after_type = &hover_text[type_start..];

                    // Find the end of the generic type by looking for the closing >
                    let mut bracket_count = 0;
                    let mut end_pos = 0;

                    for (i, ch) in after_type.char_indices() {
                        match ch {
                            '<' => bracket_count += 1,
                            '>' => {
                                bracket_count -= 1;
                                if bracket_count == 0 {
                                    end_pos = i + 1;
                                    break;
                                }
                            }
                            _ => {}
                        }
                    }

                    if end_pos > 0 {
                        let full_declaration = &hover_text[var_start..start + ": ".len() + end_pos];
                        return full_declaration.to_string();
                    }
                }
            }
        }

        String::new()
    }

    /// Check if this is a generic type definition rather than a concrete instantiation
    /// Specifically looks for the pattern "T = f64" which indicates a generic type definition
    fn is_generic_type_definition(&self, text: &str) -> bool {
        // Only detect the specific case where we have "T = f64" which reliably indicates
        // a generic type definition like "Quantity<Scale, Dimension, T = f64>"
        text.contains("T = f64")
    }
}

#[derive(Debug)]
struct QuantityParams {
    dimensions: DynDimensionExponents,
    scale: ScaleExponents,
    generic_type: String,
    brand: Option<String>,
}
