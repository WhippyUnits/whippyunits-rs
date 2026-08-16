#[allow(unused_imports)] // re-exported siblings, for intra-doc links
use crate::nalgebra::*;

use core::fmt;

use whippyunits::api::{UnitDisplayExt, UnitLabel};
use whippyunits::quantity::Quantity;
use whippyunits::{DivUnit, UnitDiv};

use super::MixedUnitMatrix;
use crate::dims::{DCons, DNil};
use crate::entry::FromRaw;

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------
//
// Every entry `(i, j)` has a *different* static unit `RowDims[i] / ColDims[j]`,
// so there is no single `Quantity` type spanning the matrix. `Display` is a
// runtime scan over all entries, which is exactly the "recurse the type-level
// lists, reify each element at its statically-known unit, dispatch into that
// element's `Display`" pattern: `RowCells` walks `ColDims` for a fixed row
// element, and `MatrixCells` walks `RowDims`, tracking the runtime row index.
//
// To align columns we first render every cell to a `String` (so the recursion
// happens once, up front), then measure the per-column widths and emit padded
// rows in a second pass.

/// Reads the scalar at `(row, col)` of the underlying matrix. Boxed as a trait
/// object so the recursive collectors need not be generic over the closure.
type ScalarAt<'a, T> = dyn Fn(usize, usize) -> T + 'a;

/// The element type at a `(RowUnit, ColUnit)` pair: `RowUnit / ColUnit` as a
/// `Quantity` with storage `T` and `Brand`.
type EntryQuotient<RowUnit, ColUnit, T, Brand> = Quantity<DivUnit<RowUnit, ColUnit>, T, Brand>;

/// Collects the rendered cells of a single row whose row-dimension entry is
/// `RowUnit`.
///
/// Implemented for the column dimension list: each `DCons` step reifies the
/// entry unit `RowUnit / ColUnit` (with storage `T` and `Brand`), renders it via
/// that `Quantity`'s unit `Display`, then recurses. Column index is `out.len()`.
#[doc(hidden)]
pub trait RowCells<RowUnit, T, Brand> {
    fn row_cells(row: usize, at: &ScalarAt<'_, T>, out: &mut Vec<String>);
}

impl<RowUnit, T, Brand> RowCells<RowUnit, T, Brand> for DNil {
    fn row_cells(_row: usize, _at: &ScalarAt<'_, T>, _out: &mut Vec<String>) {}
}

impl<RowUnit, ColUnit, Tail, T, Brand> RowCells<RowUnit, T, Brand> for DCons<ColUnit, Tail>
where
    RowUnit: UnitDiv<ColUnit>,
    EntryQuotient<RowUnit, ColUnit, T, Brand>: FromRaw<T> + UnitDisplayExt,
    Tail: RowCells<RowUnit, T, Brand>,
{
    fn row_cells(row: usize, at: &ScalarAt<'_, T>, out: &mut Vec<String>) {
        let col = out.len();
        let entry =
            <EntryQuotient<RowUnit, ColUnit, T, Brand> as FromRaw<T>>::from_raw(at(row, col));
        out.push(format!("{}", entry.unit_display()));
        <Tail as RowCells<RowUnit, T, Brand>>::row_cells(row, at, out);
    }
}

/// Collects the rendered cells of every row into `out`.
///
/// Implemented for the row dimension list: each `DCons` step renders one row
/// (delegating to `ColDims::row_cells` for the row entry `RowUnit`), then
/// recurses into the tail with the next runtime row index.
#[doc(hidden)]
pub trait MatrixCells<ColDims, T, Brand> {
    fn matrix_cells(row: usize, at: &ScalarAt<'_, T>, out: &mut Vec<Vec<String>>);
}

impl<ColDims, T, Brand> MatrixCells<ColDims, T, Brand> for DNil {
    fn matrix_cells(_row: usize, _at: &ScalarAt<'_, T>, _out: &mut Vec<Vec<String>>) {}
}

impl<RowUnit, Tail, ColDims, T, Brand> MatrixCells<ColDims, T, Brand> for DCons<RowUnit, Tail>
where
    ColDims: RowCells<RowUnit, T, Brand>,
    Tail: MatrixCells<ColDims, T, Brand>,
{
    fn matrix_cells(row: usize, at: &ScalarAt<'_, T>, out: &mut Vec<Vec<String>>) {
        let mut cells = Vec::new();
        <ColDims as RowCells<RowUnit, T, Brand>>::row_cells(row, at, &mut cells);
        out.push(cells);
        <Tail as MatrixCells<ColDims, T, Brand>>::matrix_cells(row + 1, at, out);
    }
}

/// Collects the unit label of every entry of a dimension list, in order.
///
/// Each entry is reified into a `Quantity` (with the matrix's storage type and
/// brand) purely so its type-level unit can be named — no value is involved.
/// Used to render the row/column dimension "margins" around the matrix.
#[doc(hidden)]
pub trait UnitLabels<T, Brand> {
    fn unit_labels(out: &mut Vec<String>);
}

impl<T, Brand> UnitLabels<T, Brand> for DNil {
    fn unit_labels(_out: &mut Vec<String>) {}
}

impl<U, Tail, T, Brand> UnitLabels<T, Brand> for DCons<U, Tail>
where
    Quantity<U, T, Brand>: UnitLabel,
    Tail: UnitLabels<T, Brand>,
{
    fn unit_labels(out: &mut Vec<String>) {
        let label = <Quantity<U, T, Brand> as UnitLabel>::unit_label();
        // A dimensionless entry has no unit symbol; render it as the scalar `1`.
        out.push(if label.is_empty() {
            "1".to_string()
        } else {
            label
        });
        <Tail as UnitLabels<T, Brand>>::unit_labels(out);
    }
}

impl<RowDims, ColDims, Brand, T, R, C, St> fmt::Display
    for MixedUnitMatrix<RowDims, ColDims, nalgebra::Matrix<T, R, C, St>, Brand>
where
    T: Copy,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    St: nalgebra::RawStorage<T, R, C>,
    RowDims: MatrixCells<ColDims, T, Brand> + UnitLabels<T, Brand>,
    ColDims: UnitLabels<T, Brand>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let at = |row: usize, col: usize| self.inner[(row, col)];

        let mut rows: Vec<Vec<String>> = Vec::new();
        <RowDims as MatrixCells<ColDims, T, Brand>>::matrix_cells(0, &at, &mut rows);

        // The dimension "margins": `RowDims[i]` labels row `i` (down the left
        // edge) and `ColDims[j]` labels column `j` (across the top), so entry
        // `(i, j)` reads as `row_label[i] / col_label[j]`.
        let mut row_labels: Vec<String> = Vec::new();
        <RowDims as UnitLabels<T, Brand>>::unit_labels(&mut row_labels);
        let mut col_labels: Vec<String> = Vec::new();
        <ColDims as UnitLabels<T, Brand>>::unit_labels(&mut col_labels);

        // Each cell is rendered as `value unit` (or just `value` when
        // dimensionless). Split on the first space so columns can align on the
        // numeric value's *right* edge: the value is right-aligned and the unit
        // left-aligned after a single space. Widths are measured in characters,
        // since unit glyphs like `s⁻²` are multi-byte and `fmt` pads by chars.
        fn split(cell: &str) -> (&str, &str) {
            match cell.split_once(' ') {
                Some((value, unit)) => (value, unit),
                None => (cell, ""),
            }
        }
        fn width(s: &str) -> usize {
            s.chars().count()
        }
        fn center(s: &str, w: usize) -> String {
            let len = s.chars().count();
            if len >= w {
                return s.to_string();
            }
            let total = w - len;
            let left = total / 2;
            let right = total - left;
            format!("{:left$}{s}{:right$}", "", "")
        }

        let ncols = col_labels.len();
        let mut value_widths = vec![0usize; ncols];
        let mut unit_widths = vec![0usize; ncols];
        for row in &rows {
            for (col, cell) in row.iter().enumerate() {
                let (value, unit) = split(cell);
                value_widths[col] = value_widths[col].max(width(value));
                unit_widths[col] = unit_widths[col].max(width(unit));
            }
        }

        // Base cell width (numeric field + optional ` unit`) and the final
        // column field width, widened if needed to fit the column's top label.
        let base_widths: Vec<usize> = (0..ncols)
            .map(|c| {
                value_widths[c]
                    + if unit_widths[c] > 0 {
                        1 + unit_widths[c]
                    } else {
                        0
                    }
            })
            .collect();
        let field_widths: Vec<usize> = (0..ncols)
            .map(|c| base_widths[c].max(width(&col_labels[c])))
            .collect();

        let render_cell = |col: usize, cell: &str| -> String {
            let (value, unit) = split(cell);
            let vw = value_widths[col];
            let base = if unit_widths[col] == 0 {
                format!("{value:>vw$}")
            } else {
                let uw = unit_widths[col];
                format!("{value:>vw$} {unit:<uw$}")
            };
            // Right-justify the whole cell within the (possibly wider) field.
            let field = field_widths[col];
            format!("{base:>field$}")
        };

        // Left margin width: the widest row label. The matrix body therefore
        // starts at column `label_pad + 2` (label, a space, and `[`).
        let label_pad = row_labels.iter().map(|l| width(l)).max().unwrap_or(0);

        // Top margin: the column labels, centered over their fields. The
        // two-space gap matches the data rows' column separator so columns stay
        // aligned.
        write!(f, "{:pad$}  ", "", pad = label_pad)?;
        for (col, label) in col_labels.iter().enumerate() {
            if col > 0 {
                write!(f, "  ")?;
            }
            f.write_str(&center(label, field_widths[col]))?;
        }
        writeln!(f)?;

        // Body: each row prefixed by its right-justified row label.
        for (i, row) in rows.iter().enumerate() {
            let label = &row_labels[i];
            write!(f, "{label:>label_pad$} [")?;
            for (col, cell) in row.iter().enumerate() {
                if col > 0 {
                    write!(f, "  ")?;
                }
                f.write_str(&render_cell(col, cell))?;
            }
            write!(f, "]")?;
            if i + 1 < rows.len() {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}
