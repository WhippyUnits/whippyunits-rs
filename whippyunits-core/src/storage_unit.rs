use crate::{
    Dimension, SiPrefix, dimension_exponents::DynDimensionExponents,
    scale_exponents::ScaleExponents,
};

#[cfg(not(test))]
extern crate alloc;

#[cfg(not(test))]
use alloc::format;
#[cfg(not(test))]
use alloc::string::String;
#[cfg(not(test))]
use alloc::string::ToString;
#[cfg(not(test))]
use alloc::vec::Vec;

/// Configuration for unit literal generation
#[derive(Debug, Clone, Copy)]
pub struct UnitLiteralConfig {
    pub verbose: bool,
    pub prefer_si_units: bool,
}

impl Default for UnitLiteralConfig {
    fn default() -> Self {
        Self {
            verbose: false,
            prefer_si_units: true,
        }
    }
}

/// Generate the best unit literal for a given set of dimensions and scales
/// This is the exact logic from the main crate's unit_literal_generator
pub fn generate_unit_literal(
    exponents: DynDimensionExponents,
    scale_factors: ScaleExponents,
    config: UnitLiteralConfig,
) -> String {
    // Convert DynDimensionExponents to Vec<i16> for compatibility with existing functions
    let exponents_vec = exponents.0.to_vec();

    // Generate systematic unit literal (base unit without prefix)
    let base_systematic_literal = generate_systematic_unit_name_with_scale_factors(
        exponents_vec.clone(),
        scale_factors,
        config.verbose,
    );

    // Check if we found a unit literal match - if so, use it directly without conversion factor
    let pure_systematic = generate_systematic_unit_name(exponents_vec.clone(), config.verbose);
    let found_unit_literal = base_systematic_literal != pure_systematic;
    let systematic_literal = if found_unit_literal {
        // We found a unit literal match, use it directly
        base_systematic_literal
    } else {
        // No unit literal match, apply SI prefix to the systematic unit literal
        generate_prefixed_systematic_unit(
            exponents,
            scale_factors,
            &base_systematic_literal,
            config.verbose,
        )
    };

    // If we don't prefer SI units, return the systematic literal
    if !config.prefer_si_units {
        return systematic_literal;
    }

    // Recognize pure integer powers of a named unit (e.g. V², V⁻¹, rot⁻¹) so that
    // they stay compact instead of expanding into a base-dimension decomposition
    // or being recast to the identity-scale unit with a numeric coefficient. This
    // takes precedence over the per-dimension systematic literal below, but never
    // over an exact (k = 1) match.
    if let Some(power_literal) =
        try_unit_power_literal(exponents, scale_factors, config.verbose)
    {
        return power_literal;
    }

    // Check if we have a recognized dimension with a specific SI unit
    if let Some(info) = lookup_dimension_name(exponents_vec) {
        if let Some(si_shortname) = if config.verbose {
            info.unit_si_shortname
        } else {
            info.unit_si_shortname_symbol
        } {
            // Apply SI prefix to the specific SI unit name
            let prefixed_si_unit =
                generate_prefixed_si_unit(scale_factors, si_shortname, config.verbose);

            // Return the prefixed SI unit if it's different from the systematic literal
            // But only if we didn't find a unit literal match
            if prefixed_si_unit != systematic_literal && !found_unit_literal {
                prefixed_si_unit
            } else {
                systematic_literal
            }
        } else {
            // No specific SI unit defined, use the systematic literal
            systematic_literal
        }
    } else {
        // Unknown dimension, use the systematic literal
        systematic_literal
    }
}

/// Recognize an exponent vector that is a pure integer power of a named unit
/// and render it compactly as `symbol` + superscript, rather than expanding it
/// into a base-dimension decomposition. This covers both:
///
/// - composite SI units at identity scale — `V²`, `V⁻¹`, `N⁻²` instead of
///   `kg⁻²·m⁻⁴·s⁶·A²`; and
/// - scale-distinguished units — the alternate angular units `rot`, `deg`,
///   `grad`, … — so `rot⁻¹` stays `rot⁻¹` instead of being recast to
///   `(0.1590)rad⁻¹`.
///
/// It matches a candidate power `k` against both the dimension and the scale:
/// the base `exponents / k` and `scale / k` must together be a registered unit.
/// Because the scale is reduced too, a unit carrying an extra decimal prefix
/// (e.g. `mrad²`, whose scale is not a clean multiple of any registered base
/// scale) simply fails to match and falls through to the SI-prefix path — no
/// prefix is ever silently dropped.
///
/// The `k = 1` case is skipped (an exact match, already handled by the
/// dimension-name lookup with full prefix support), and pure single-dimension
/// powers at identity scale (`m²`, `s⁻¹`) are left alone so that named
/// reciprocals like hertz keep priority.
fn try_unit_power_literal(
    exponents: DynDimensionExponents,
    scale_factors: ScaleExponents,
    long_name: bool,
) -> Option<String> {
    // Bail out on any unresolved component (`i16::MIN` is the "unknown" sentinel);
    // powers of an unknown are meaningless and `abs()` would overflow.
    if exponents.0.contains(&i16::MIN)
        || scale_factors.0.contains(&i16::MIN)
    {
        return None;
    }

    let nonzero_count = exponents.0.iter().filter(|&&e| e != 0).count();
    if nonzero_count == 0 {
        return None;
    }
    // Only fire for genuine composites (>= 2 base dimensions) or scale-carrying
    // units; a bare single dimension at identity scale is the systematic
    // generator's job (and must not shadow named reciprocals such as hertz).
    if nonzero_count < 2 && scale_factors == ScaleExponents::IDENTITY {
        return None;
    }

    // The primitive divisor is the gcd of the exponent magnitudes.
    let g = exponents
        .0
        .iter()
        .fold(0i16, |acc, &e| gcd_i16(acc, e.abs()));
    if g == 0 {
        return None;
    }

    // Candidate powers. For a primitive vector (g == 1) only the negated base is
    // interesting — the positive `k = 1` is the exact match handled elsewhere.
    // For g >= 2 the vector is genuinely a power, so check both signs.
    let candidates: &[i16] = if g == 1 { &[-1] } else { &[g, -g] };

    for &k in candidates {
        // Reduce the dimension by k (exact by construction).
        let mut base_dim = [0i16; 8];
        for (b, &e) in base_dim.iter_mut().zip(exponents.0.iter()) {
            *b = e / k;
        }

        // Reduce the scale by k; every component must divide exactly, otherwise
        // this power does not correspond to a clean unit^k.
        let mut base_scale = [0i16; 4];
        let mut scale_ok = true;
        for (b, &s) in base_scale.iter_mut().zip(scale_factors.0.iter()) {
            if s % k != 0 {
                scale_ok = false;
                break;
            }
            *b = s / k;
        }
        if !scale_ok {
            continue;
        }
        let base_scale = ScaleExponents(base_scale);

        // Never let a negated power shadow a unit that is itself exactly named
        // at this dimension and scale: `V/A` is `Ω`, not `S⁻¹` (and a bare
        // siemens is `S`, not `Ω⁻¹`). The exact (k = 1) match is produced by the
        // dimension-name lookup in the caller, which runs *after* this pass, so
        // we must decline the reciprocal here and let that path win.
        if k < 0
            && Dimension::find_dimension_by_exponents(exponents).is_some_and(|dim| {
                dim.units
                    .iter()
                    .any(|u| u.scale == scale_factors && u.conversion_factor == 1.0)
            })
        {
            continue;
        }

        if let Some(dim) = Dimension::find_dimension_by_exponents(DynDimensionExponents(base_dim))
            && let Some(unit) = dim
                .units
                .iter()
                .find(|u| u.scale == base_scale && u.conversion_factor == 1.0)
            {
                let symbol = if long_name {
                    unit.name
                } else {
                    unit.symbols.first().copied().unwrap_or("?")
                };
                return Some(format!(
                    "{}{}",
                    symbol,
                    crate::to_unicode_superscript(k, false)
                ));
            }
    }

    None
}

fn gcd_i16(a: i16, b: i16) -> i16 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Calculate the storage unit name from scale factors and dimension exponents
/// This is the canonical implementation used by both proc macros and LSP proxy
pub fn get_storage_unit_name(
    scale_factors: ScaleExponents,
    dimension_exponents: DynDimensionExponents,
    long_name: bool,
) -> String {
    // Use the exact same logic as the prettyprint
    let unit_literal = generate_unit_literal(
        dimension_exponents,
        scale_factors,
        UnitLiteralConfig {
            verbose: long_name,
            prefer_si_units: true,
        },
    );

    // If we got a unit literal, use it; otherwise fall back to systematic generation
    if !unit_literal.is_empty() {
        unit_literal
    } else {
        // Fallback to systematic generation
        let exponents_vec = dimension_exponents.0.to_vec();
        generate_systematic_unit_name(exponents_vec, long_name)
    }
}

// Supporting functions moved from the main crate to ensure identical logic

/// Generate systematic unit name with scale factors
pub fn generate_systematic_unit_name_with_scale_factors(
    exponents: Vec<i16>,
    scale_factors: ScaleExponents,
    long_name: bool,
) -> String {
    // Check if all exponents are unknown
    if exponents.iter().all(|&exp| exp == i16::MIN) {
        return "?".to_string();
    }

    // check if the unit is "pure" (e.g. if only one exponent is nonzero)
    let is_pure = exponents.iter().filter(|&exp| *exp != 0).count() == 1;

    // For pure units, first try to find a unit literal that matches the scale factors
    if is_pure
        && let Some(unit_name) =
            lookup_unit_literal_by_scale_factors(&exponents, scale_factors, long_name)
        {
            return unit_name.to_string();
        }

    // For compound units, pass scale factors to help match the correct units
    // Fall back to the original logic but with scale factors
    generate_systematic_unit_name_with_format_and_scale(
        exponents,
        Some(scale_factors),
        long_name,
        UnitFormat::Unicode,
    )
}

/// Generate systematic unit name
pub fn generate_systematic_unit_name(exponents: Vec<i16>, long_name: bool) -> String {
    generate_systematic_unit_name_with_format(exponents, long_name, UnitFormat::Unicode)
}

/// Unit format enum
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnitFormat {
    Unicode,
    Ucum,
}

/// Generate systematic unit name with format
pub fn generate_systematic_unit_name_with_format(
    exponents: Vec<i16>,
    long_name: bool,
    format: UnitFormat,
) -> String {
    generate_systematic_unit_name_with_format_and_scale(exponents, None, long_name, format)
}

/// Generate systematic unit name with format and optional scale factors
pub fn generate_systematic_unit_name_with_format_and_scale(
    exponents: Vec<i16>,
    scale_factors: Option<ScaleExponents>,
    long_name: bool,
    format: UnitFormat,
) -> String {
    // Convert Vec<i16> to DynDimensionExponents for the core function
    if exponents.len() != 8 {
        return "?".to_string();
    }

    let dimension_exponents = DynDimensionExponents([
        exponents[0],
        exponents[1],
        exponents[2],
        exponents[3],
        exponents[4],
        exponents[5],
        exponents[6],
        exponents[7],
    ]);

    // Use the centralized logic from whippyunits-core
    let base_result = generate_systematic_composite_unit_name_with_scale(
        dimension_exponents,
        scale_factors,
        long_name,
    );

    // Apply format-specific transformations
    match format {
        UnitFormat::Unicode => base_result,
        UnitFormat::Ucum => {
            // Convert Unicode format to UCUM format
            convert_unicode_to_ucum_format(&base_result)
        }
    }
}

/// Convert Unicode format unit string to UCUM format
fn convert_unicode_to_ucum_format(unicode_unit: &str) -> String {
    // This is a simplified conversion - in practice, you might need more sophisticated logic
    // For now, just return the unicode format as-is since the core logic already handles
    // the basic formatting correctly
    unicode_unit.to_string()
}

/// Look up a unit literal by its dimension exponents and scale factors
pub fn lookup_unit_literal_by_scale_factors(
    exponents: &[i16],
    scale_factors: ScaleExponents,
    long_name: bool,
) -> Option<&'static str> {
    // Convert Vec<i16> to DynDimensionExponents for comparison
    if exponents.len() != 8 {
        return None;
    }

    let dyn_exponents = DynDimensionExponents([
        exponents[0],
        exponents[1],
        exponents[2],
        exponents[3],
        exponents[4],
        exponents[5],
        exponents[6],
        exponents[7],
    ]);

    if let Some(dimension) = Dimension::find_dimension_by_exponents(dyn_exponents) {
        // First, try to find a unit that matches the exact scale factors
        // This is the preferred approach - use exact matches when possible
        if let Some(exact_unit) = dimension.units.iter().find(|unit| {
            unit.scale == scale_factors && unit.conversion_factor == 1.0 // Only consider pure SI units, not imperial units
        }) {
            return Some(if long_name {
                exact_unit.name
            } else {
                exact_unit.symbols[0]
            });
        }

        // If no exact match found, fall back to base unit (identity scale)
        dimension
            .units
            .iter()
            .find(|unit| unit.scale == ScaleExponents::IDENTITY && unit.conversion_factor == 1.0)
            .map(|unit| {
                if long_name {
                    unit.name
                } else {
                    unit.symbols[0]
                }
            })
    } else {
        None
    }
}

/// Dimension names struct
pub struct DimensionNames {
    pub dimension_name: &'static str,
    pub unit_si_shortname_symbol: Option<&'static str>,
    pub unit_si_shortname: Option<&'static str>,
}

/// Look up dimension name
pub fn lookup_dimension_name(exponents: Vec<i16>) -> Option<DimensionNames> {
    // Convert Vec<i16> to DynDimensionExponents for lookup
    if exponents.len() != 8 {
        return None;
    }

    let dyn_exponents = DynDimensionExponents([
        exponents[0],
        exponents[1],
        exponents[2],
        exponents[3],
        exponents[4],
        exponents[5],
        exponents[6],
        exponents[7],
    ]);

    // Use Dimension::find_dimension_by_exponents directly
    Dimension::find_dimension_by_exponents(dyn_exponents).and_then(|dim_info| {
        // For pure atomic dimensions (like area = length²), prefer systematic generation
        // over predefined units when we have exact matches of atomic unit exponents
        let is_pure_atomic = exponents.iter().filter(|&exp| *exp != 0).count() == 1;

        if is_pure_atomic {
            // For pure atomic dimensions, check if we have an exact match with identity scale factors
            let has_exact_match = dim_info.units.iter().any(|unit| {
                unit.scale == ScaleExponents::IDENTITY && unit.conversion_factor == 1.0
            });

            if !has_exact_match {
                // No exact match found, return None to force systematic generation
                return None;
            }
        }

        // Prioritize exact matches of atomic unit exponents (scale factors of [0, 0, 0, 0])
        // over the first unit in the lexical list
        let preferred_unit = dim_info
            .units
            .iter()
            .find(|unit| unit.scale == ScaleExponents::IDENTITY && unit.conversion_factor == 1.0)
            .or_else(|| dim_info.units.first()); // Fall back to first unit if no exact match

        let unit_symbol = preferred_unit.and_then(|unit| unit.symbols.first().copied());
        let unit_long_name = preferred_unit.map(|unit| unit.name);

        Some(DimensionNames {
            dimension_name: dim_info.name,
            unit_si_shortname_symbol: unit_symbol, // Use actual unit symbol (e.g., "J") instead of dimension symbol (e.g., "ML²T⁻²")
            unit_si_shortname: unit_long_name, // Use unit long name (e.g., "joule") instead of dimension name (e.g., "Energy")
        })
    })
}

/// Generate prefixed systematic unit
pub fn generate_prefixed_systematic_unit(
    _exponents: DynDimensionExponents,
    scale_factors: ScaleExponents,
    base_unit: &str,
    long_name: bool,
) -> String {
    let total_scale_p10 = calculate_total_scale_p10(
        scale_factors.0[0],
        scale_factors.0[1],
        scale_factors.0[2],
        scale_factors.0[3],
    );

    // Check if this is a pure unit (not compound)
    let is_pure_unit = !base_unit.contains("·");

    // For pure units, check if we need to apply base scale offset
    let effective_scale_p10 = if is_pure_unit {
        // Find the base scale offset by looking up the unit's scale from whippyunits-core
        // The base_unit comes from systematic unit name generation, so it should always be valid
        let base_scale_offset = Dimension::find_unit_by_symbol(base_unit)
            .or_else(|| Dimension::find_unit_by_name(base_unit))
            .map(|(_unit, _dimension)| _unit.scale.log10().unwrap_or(0))
            .unwrap_or(0);

        // Apply the base scale offset to the scale calculation
        // The base scale offset represents the offset of the base unit (e.g., gram = -3)
        // We need to subtract it from the total scale to get the effective scale
        total_scale_p10 - base_scale_offset
    } else {
        // For compound units, don't apply base scale offset to the aggregate prefix
        // The individual parts already have their base scale offsets applied
        total_scale_p10
    };

    if let Some(prefix) = get_si_prefix(effective_scale_p10, long_name) {
        // For pure powers of base units, add disambiguating parentheses
        if !base_unit.contains("·")
            && (base_unit.contains("^") || base_unit.contains("²") || base_unit.contains("³"))
        {
            format!("{}({})", prefix, base_unit)
        } else {
            format!("{}{}", prefix, base_unit)
        }
    } else {
        // Check if this is a pure power of 10 using whippyunits-core
        let is_pure_power_of_10 = scale_factors.log10().is_some();

        if is_pure_power_of_10 {
            // Fall back to SI unit with 10^n notation when SI prefix lookup fails
            generate_si_unit_with_scale(effective_scale_p10, base_unit, long_name)
        } else {
            // Not a pure power of 10, show the scale factors explicitly
            let scale_factors_str = format_scale_factors(
                scale_factors.0[0],
                scale_factors.0[1],
                scale_factors.0[2],
                scale_factors.0[3],
            );
            if scale_factors_str.is_empty() {
                base_unit.to_string()
            } else {
                format!("{}{}", scale_factors_str, base_unit)
            }
        }
    }
}

/// Generate prefixed SI unit
pub fn generate_prefixed_si_unit(
    scale_factors: ScaleExponents,
    base_si_unit: &str,
    long_name: bool,
) -> String {
    let total_scale_p10 = calculate_total_scale_p10(
        scale_factors.0[0],
        scale_factors.0[1],
        scale_factors.0[2],
        scale_factors.0[3],
    );

    // Apply base scale offset for mass units (same logic as generate_prefixed_systematic_unit)
    let effective_scale_p10 =
        if let Some((_unit, _dimension)) = Dimension::find_unit_by_symbol(base_si_unit) {
            // Get the base scale offset from the unit's scale (systematic approach)
            let base_scale_offset = _unit.scale.log10().unwrap_or(0);
            total_scale_p10 - base_scale_offset
        } else {
            // Fallback: try to find by name if symbol lookup fails
            if let Some((_unit, _dimension)) = Dimension::find_unit_by_name(base_si_unit) {
                let base_scale_offset = _unit.scale.log10().unwrap_or(0);
                total_scale_p10 - base_scale_offset
            } else {
                // No base scale offset found, use total scale as-is
                total_scale_p10
            }
        };

    if let Some(prefix) = get_si_prefix(effective_scale_p10, long_name) {
        // For pure powers of base units, add disambiguating parentheses
        if !base_si_unit.contains("·")
            && (base_si_unit.contains("^")
                || base_si_unit.contains("²")
                || base_si_unit.contains("³"))
        {
            format!("{}({})", prefix, base_si_unit)
        } else {
            format!("{}{}", prefix, base_si_unit)
        }
    } else {
        // Check if this is a pure power of 10 using whippyunits-core
        let is_pure_power_of_10 = scale_factors.log10().is_some();

        if is_pure_power_of_10 {
            // Fall back to SI unit with 10^n notation when SI prefix lookup fails
            generate_si_unit_with_scale(effective_scale_p10, base_si_unit, long_name)
        } else {
            // Not a pure power of 10, show the scale factors explicitly
            let scale_factors_str = format_scale_factors(
                scale_factors.0[0],
                scale_factors.0[1],
                scale_factors.0[2],
                scale_factors.0[3],
            );
            if scale_factors_str.is_empty() {
                base_si_unit.to_string()
            } else {
                format!("{}{}", scale_factors_str, base_si_unit)
            }
        }
    }
}

/// Calculate total power of 10 using whippyunits-core ScaleExponents
fn calculate_total_scale_p10(scale_p2: i16, scale_p3: i16, scale_p5: i16, scale_pi: i16) -> i16 {
    let scale_exponents = ScaleExponents([scale_p2, scale_p3, scale_p5, scale_pi]);
    scale_exponents.log10().unwrap_or(0)
}

/// Generate SI unit with 10^n notation when no standard prefix is available
fn generate_si_unit_with_scale(
    total_scale_p10: i16,
    base_si_unit: &str,
    _long_name: bool,
) -> String {
    if total_scale_p10 == 0 {
        base_si_unit.to_string()
    } else {
        format!(
            "10{} {}",
            crate::to_unicode_superscript(total_scale_p10, false),
            base_si_unit
        )
    }
}

/// Format scale factors by calculating the actual numeric value
/// Returns a prefix string like "(0.318)" for non-power-of-10 scales
fn format_scale_factors(scale_p2: i16, scale_p3: i16, scale_p5: i16, scale_pi: i16) -> String {
    let scale_exponents = ScaleExponents([scale_p2, scale_p3, scale_p5, scale_pi]);

    // If it's a pure power of 10, we don't need to show scale factors
    if scale_exponents.log10().is_some() {
        return String::new();
    }

    // Calculate the actual numeric value: 2^p2 * 3^p3 * 5^p5 * π^pi
    let mut value = 1.0;

    if scale_p2 != 0 {
        value *= 2.0_f64.powi(scale_p2 as i32);
    }
    if scale_p3 != 0 {
        value *= 3.0_f64.powi(scale_p3 as i32);
    }
    if scale_p5 != 0 {
        value *= 5.0_f64.powi(scale_p5 as i32);
    }
    if scale_pi != 0 {
        value *= core::f64::consts::PI.powi(scale_pi as i32);
    }

    // If the value is 1.0, no scaling needed
    if value == 1.0 {
        String::new()
    } else {
        // Format with 2-3 significant figures for inlay hints (compact display)
        format!("({})", format_float_with_sig_figs(value, 3))
    }
}

/// Format a float with specified number of significant figures
/// Optimized for inlay hints (compact display)
fn format_float_with_sig_figs(value: f64, sig_figs: usize) -> String {
    if value == 0.0 {
        return "0".to_string();
    }

    let abs_value = value.abs();
    let magnitude = abs_value.log10().floor() as i32;
    let scale_factor = 10_f64.powi(sig_figs as i32 - 1 - magnitude);

    let rounded = (value * scale_factor).round() / scale_factor;

    // Format with appropriate precision
    

    if magnitude >= 0 {
        // For values >= 1, show up to sig_figs digits total
        let precision = (sig_figs as i32 - magnitude - 1).max(0) as usize;
        format!("{:.precision$}", rounded, precision = precision)
    } else {
        // For values < 1, show sig_figs significant digits after decimal
        format!(
            "{:.precision$}",
            rounded,
            precision = (sig_figs as i32 + magnitude.abs()) as usize
        )
    }
}

/// Get SI prefix for a given power of 10
fn get_si_prefix(power_of_10: i16, long_name: bool) -> Option<&'static str> {
    SiPrefix::ALL
        .iter()
        .find(|prefix| prefix.factor_log10() == power_of_10)
        .map(|prefix| {
            if long_name {
                prefix.name()
            } else {
                prefix.symbol()
            }
        })
}

/// Index of the angle dimension within the 8-element base-dimension vector.
const ANGLE_DIMENSION_INDEX: usize = 7;

/// Raise a scale vector to the integer power `k` (multiply every prime exponent).
fn scale_pow(scale: ScaleExponents, k: i16) -> ScaleExponents {
    ScaleExponents([
        scale.0[0] * k,
        scale.0[1] * k,
        scale.0[2] * k,
        scale.0[3] * k,
    ])
}

/// Generate systematic unit name for composite dimensions
/// This is a public function that can be used by the main crate's prettyprint module
pub fn generate_systematic_composite_unit_name(
    dimension_exponents: DynDimensionExponents,
    long_name: bool,
) -> String {
    generate_systematic_composite_unit_name_with_scale(dimension_exponents, None, long_name)
}

/// Render the base-dimension factors of `exponents` as individual `unit^exp`
/// parts (one per non-zero base dimension), applying scale-matched unit
/// selection, power-aware angular attribution, and the compound-context g→kg
/// fix-up. Factored out of `generate_systematic_composite_unit_name_with_scale`
/// so the composite pre-pass can render its residual through the same logic.
///
/// `is_compound` is passed by the caller rather than inferred locally: when a
/// composite has been peeled off (`V·s`) the residual may itself be a single
/// base dimension yet still lives in a compound context, which is what decides
/// the `g → kg` mass fix-up.
fn base_dimension_parts(
    exponents: [i16; 8],
    scale_factors: Option<ScaleExponents>,
    long_name: bool,
    is_compound: bool,
) -> Vec<String> {
    let mut parts: Vec<String> = Vec::new();

    for (index, &exp) in exponents.iter().enumerate() {
        if exp == 0 {
            continue;
        }

        // Get unit configuration from Dimension::BASIS
        // Try to match scale factors if provided, otherwise use first unit
        let (unit_name, unit_symbol, base_scale_offset) =
            if let Some(dimension) = Dimension::BASIS.get(index) {
                // If scale factors are provided, try to find a matching unit
                let matched_unit = if let Some(scale) = scale_factors {
                    if index == ANGLE_DIMENSION_INDEX && exp != i16::MIN {
                        // Power-aware angular attribution. An alternate angular
                        // unit is distinguished by a scale factor (rot = 2π,
                        // deg = π/180, …), so a power `expᵗʰ` scales that factor
                        // by `exp` and no longer matches the exp-1 registration —
                        // which is why `s/rot` used to print as `(1/2π)(s·rad⁻¹)`.
                        // Pick the angular unit `u` whose scaled contribution
                        // `exp · scale(u)` accounts for the *entire* aggregate
                        // scale (i.e. every non-angular slot is at identity), so
                        // the result is `s·rot⁻¹` with no leftover coefficient.
                        // If a non-angular prefix is also present nothing matches
                        // and we fall back to radian, leaving the existing prefix
                        // path to render the coefficient as before.
                        dimension
                            .units
                            .iter()
                            .find(|unit| {
                                unit.conversion_factor == 1.0
                                    && scale_pow(unit.scale, exp) == scale
                            })
                            .or_else(|| {
                                dimension.units.iter().find(|unit| {
                                    unit.scale == ScaleExponents::IDENTITY
                                        && unit.conversion_factor == 1.0
                                })
                            })
                    } else {
                        // For compound units, try to match the aggregate scale factors
                        // This works when the scale factors come from a single dimension (e.g., deg/m where deg has the scale)
                        dimension
                            .units
                            .iter()
                            .find(|unit| unit.scale == scale && unit.conversion_factor == 1.0)
                            // If no exact match, try identity scale (for cases like deg/m where meter has identity scale)
                            .or_else(|| {
                                if scale == ScaleExponents::IDENTITY {
                                    None // Already tried identity above
                                } else {
                                    dimension.units.iter().find(|unit| {
                                        unit.scale == ScaleExponents::IDENTITY
                                            && unit.conversion_factor == 1.0
                                    })
                                }
                            })
                    }
                } else {
                    None
                };

                if let Some(unit) = matched_unit.or_else(|| dimension.units.first()) {
                    let base_scale_offset = unit.scale.log10().unwrap_or(0);
                    (unit.name, unit.symbols[0], base_scale_offset)
                } else {
                    ("?", "?", 0)
                }
            } else {
                ("?", "?", 0)
            };

        // For compound units, convert g to kg for mass terms
        let (adjusted_unit_name, adjusted_unit_symbol) =
            if is_compound && base_scale_offset != 0 && index == 0 {
                // This is a compound unit with mass dimension (index 0) that has a scale offset
                if long_name {
                    match unit_name {
                        "gram" => ("kilogram", unit_symbol),
                        _ => (unit_name, unit_symbol),
                    }
                } else {
                    match unit_symbol {
                        "g" => (unit_name, "kg"),
                        _ => (unit_name, unit_symbol),
                    }
                }
            } else {
                (unit_name, unit_symbol)
            };

        // Generate the unit part with exponent
        let base_name = if long_name {
            adjusted_unit_name
        } else {
            adjusted_unit_symbol
        };
        let unit_part = if exp == 1 {
            base_name.to_string()
        } else {
            format!("{}{}", base_name, crate::to_unicode_superscript(exp, false))
        };

        parts.push(unit_part);
    }

    parts
}

/// Cost of a dimension vector as `(factors, magnitude)`: the number of distinct
/// non-zero base dimensions and the sum of their absolute exponents. Compared
/// lexicographically, so factor count dominates (a shorter product always wins)
/// but exponent magnitude breaks ties — which is what lets `s²·rot⁻¹` (mag 3)
/// lose to `s·rot⁻¹` (mag 2) at equal factor count.
fn dimension_cost(exponents: &[i16; 8]) -> (usize, i32) {
    let factors = exponents.iter().filter(|&&e| e != 0).count();
    let magnitude = exponents.iter().map(|&e| (e as i32).abs()).sum();
    (factors, magnitude)
}

/// Try to rewrite a compound dimension vector as a single identity-scale
/// composite unit (`V`, `J`, `N`, …) times a base-dimension residual, when
/// doing so produces a strictly shorter product. Returns the rendered
/// composite part (`C` or `Cᵏ`) together with the residual dimension vector,
/// which the caller renders through [`base_dimension_parts`].
///
/// This resolves the "partial factorization" wart where e.g. `V·s·rot⁻¹` would
/// otherwise expand all the way to `kg·m²·s⁻²·A⁻¹·rot⁻¹`. The dictionary is the
/// set of identity-scale composites carrying at least two base dimensions, and
/// only the powers `k ∈ {+1, −1}` are considered. Exact composites and pure
/// powers (empty residual) are deliberately skipped — those go through the
/// dimension-name lookup and `try_unit_power_literal`, which keep SI prefixes.
///
/// The policy has two levels, both using [`dimension_cost`] =
/// `(factors, magnitude)` with the composite contributing one factor at
/// magnitude `|k| = 1`:
///
/// 1. **Acceptance is factor-count-strict.** A composite is only peeled when it
///    yields fewer factors than the baseline. Magnitude alone must not justify
///    it, or acceleration (`m·s⁻²`, 2 factors) would flip to `N·kg⁻¹` (also 2
///    factors) merely because the residual exponent is smaller.
/// 2. **Ties break lexicographically on the full cost**, so exponent magnitude
///    decides between accepted candidates. This keeps neighbouring quantities
///    consistent: `V·s/rot` → `Wb·rot⁻¹` and `V·s²/rot` → `Wb·s·rot⁻¹` both
///    settle on the weber basis (`Wb`'s residual `s·rot⁻¹` beats `V`'s
///    `s²·rot⁻¹`), rather than the latter tying at `V·s²·rot⁻¹`.
///
/// Ambiguity is still inherent (an overcomplete dictionary: `V·s` is also `Wb`,
/// `J/A`, …), so any remaining ties fall back to fixed policy: prefer the
/// composite that absorbs the most base dimensions (highest arity), then the
/// fixed `Dimension::ALL` order for determinism. The extracted composite is
/// scale-free, so the entire aggregate scale flows to the residual (which is
/// why a prefix on a compound may land on an unexpected residual factor).
fn try_extract_composite(
    exponents: [i16; 8],
    long_name: bool,
) -> Option<(String, [i16; 8])> {
    // Powers of an unknown are meaningless (and would overflow below).
    if exponents.contains(&i16::MIN) {
        return None;
    }

    let baseline = dimension_cost(&exponents);
    if baseline.0 < 2 {
        return None; // nothing to factor
    }

    // best = (cost, arity, rendered composite part, residual)
    let mut best: Option<((usize, i32), usize, String, [i16; 8])> = None;

    for dim in Dimension::ALL {
        let c = dim.exponents.0;
        let arity = c.iter().filter(|&&e| e != 0).count();
        if arity < 2 {
            // Base dimensions and single-dimension composites (area, volume,
            // frequency) are the base decomposition's / pure-power's job.
            continue;
        }
        // Canonical identity-scale SI unit for this composite (skip composites
        // with no named identity-scale unit, e.g. mass densities).
        let unit = match dim
            .units
            .iter()
            .find(|u| u.scale == ScaleExponents::IDENTITY && u.conversion_factor == 1.0)
        {
            Some(u) => u,
            None => continue,
        };

        for &k in &[1i16, -1] {
            let mut residual = [0i16; 8];
            for i in 0..8 {
                residual[i] = exponents[i] - k * c[i];
            }
            // Empty residual is an exact composite / pure power; leave it to the
            // prefix-aware paths.
            if residual.iter().all(|&e| e == 0) {
                continue;
            }
            // The composite contributes one factor at magnitude |k| = 1.
            let (res_factors, res_magnitude) = dimension_cost(&residual);
            let cost = (1 + res_factors, 1 + res_magnitude);
            // Acceptance is factor-count-strict: we only peel a composite when
            // it produces a genuinely *shorter* product. Magnitude alone must
            // not justify extraction, or acceleration (`m·s⁻²`, 2 factors) would
            // flip to `N·kg⁻¹` (also 2 factors, but lower residual magnitude).
            // Magnitude only breaks ties *between* accepted candidates below.
            if cost.0 >= baseline.0 {
                continue;
            }

            let symbol = if long_name { unit.name } else { unit.symbols[0] };
            let part = if k == 1 {
                symbol.to_string()
            } else {
                format!("{}{}", symbol, crate::to_unicode_superscript(k, false))
            };

            let better = match &best {
                None => true,
                Some((best_cost, best_arity, _, _)) => {
                    cost < *best_cost || (cost == *best_cost && arity > *best_arity)
                }
            };
            if better {
                best = Some((cost, arity, part, residual));
            }
        }
    }

    best.map(|(_, _, part, residual)| (part, residual))
}

/// Generate systematic unit name for composite dimensions with optional scale factors
/// When scale factors are provided, tries to match them to the correct unit for each dimension
pub fn generate_systematic_composite_unit_name_with_scale(
    dimension_exponents: DynDimensionExponents,
    scale_factors: Option<ScaleExponents>,
    long_name: bool,
) -> String {
    let exponents = dimension_exponents.0;

    // Check if all exponents are unknown
    if exponents.iter().all(|&exp| exp == i16::MIN) {
        return "?".to_string();
    }

    // Partial-factorization pre-pass: peel off a single named composite unit
    // (V, J, N, …) when it strictly reduces the factor count, then render the
    // residual base dimensions through the shared per-dimension logic.
    let mut parts: Vec<String> = Vec::new();
    let (residual, composite_extracted) =
        if let Some((composite_part, residual)) = try_extract_composite(exponents, long_name) {
            parts.push(composite_part);
            (residual, true)
        } else {
            (exponents, false)
        };

    // A peeled composite always leaves us in a compound context (composite +
    // non-empty residual); otherwise fall back to the plain non-zero count.
    let is_compound =
        composite_extracted || residual.iter().filter(|&&e| e != 0).count() > 1;

    parts.extend(base_dimension_parts(
        residual,
        scale_factors,
        long_name,
        is_compound,
    ));

    if parts.len() == 1 {
        parts.pop().unwrap()
    } else {
        format!("({})", parts.join("·"))
    }
}

/// Calculate the storage unit name from scale factors and dimension name
/// This is a convenience function for proc macros that have dimension name as string
pub fn get_storage_unit_name_by_dimension_name(
    scale_factors: ScaleExponents,
    dimension_name: &str,
    long_name: bool,
) -> String {
    // Map dimension name to exponents
    let dimension_exponents = match dimension_name {
        "Mass" => DynDimensionExponents([1, 0, 0, 0, 0, 0, 0, 0]),
        "Length" => DynDimensionExponents([0, 1, 0, 0, 0, 0, 0, 0]),
        "Time" => DynDimensionExponents([0, 0, 1, 0, 0, 0, 0, 0]),
        "Current" => DynDimensionExponents([0, 0, 0, 1, 0, 0, 0, 0]),
        "Temperature" => DynDimensionExponents([0, 0, 0, 0, 1, 0, 0, 0]),
        "Amount" => DynDimensionExponents([0, 0, 0, 0, 0, 1, 0, 0]),
        "Luminous Intensity" => DynDimensionExponents([0, 0, 0, 0, 0, 0, 1, 0]),
        "Angle" => DynDimensionExponents([0, 0, 0, 0, 0, 0, 0, 1]),
        _ => return "unknown".to_string(),
    };

    get_storage_unit_name(scale_factors, dimension_exponents, long_name)
}
