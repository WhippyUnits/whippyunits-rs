#[cfg(not(test))]
extern crate alloc;

use syn::parse::{Parse, ParseStream, Result};
use syn::token::{Caret, Dot, Slash, Star};
use syn::{Ident, LitInt};

#[cfg(not(test))]
use alloc::boxed::Box;
#[cfg(not(test))]
use alloc::string::ToString;
#[cfg(not(test))]
use alloc::vec::Vec;

use crate::{
    Dimension, SiPrefix, Unit,
    dimension_exponents::{DimensionExponents, DynDimensionExponents},
    scale_exponents::ScaleExponents,
};

/// Represents a unit with optional exponent
#[derive(Debug, Clone)]
pub struct UnitExprUnit {
    pub name: Ident,
    pub exponent: i16,
}

/// Placeholder unit names a numeric factor in a unit expression parses to:
/// `dimensionless` for `1`, and `power_of_N` (with `exponent` as the power)
/// for its factors of 2, 3 and 5.  They carry scale but no dimension.
pub const NUMERIC_FACTOR_UNITS: [&str; 5] =
    ["dimensionless", "power_of_2", "power_of_3", "power_of_5", "power_of_10"];

impl UnitExprUnit {
    /// Whether this is a numeric scale factor rather than a named unit.
    pub fn is_numeric_factor(&self) -> bool {
        let name = self.name.to_string();
        NUMERIC_FACTOR_UNITS.contains(&name.as_str())
    }

    /// The scale of a numeric factor (`None` for named units).
    fn numeric_scale(&self) -> Option<ScaleExponents> {
        let power = self.exponent;
        match self.name.to_string().as_str() {
            "dimensionless" => Some(ScaleExponents::IDENTITY),
            "power_of_2" => Some(ScaleExponents::_2(power)),
            "power_of_3" => Some(ScaleExponents::_3(power)),
            "power_of_5" => Some(ScaleExponents::_5(power)),
            "power_of_10" => Some(ScaleExponents::_10(power)),
            _ => None,
        }
    }
}

/// Parse a numeric literal in a unit expression (`100`, `60`, `0.6`, `1e-3`)
/// into its exact factorization over 2, 3 and 5, or a compile error spanned
/// on the literal when it has no such factorization.
fn parse_numeric_factor(input: ParseStream) -> Result<UnitExpr> {
    let (digits, suffix, span) = if input.peek(syn::LitFloat) {
        let lit: syn::LitFloat = input.parse()?;
        (lit.base10_digits().to_string(), lit.suffix().to_string(), lit.span())
    } else {
        let lit: LitInt = input.parse()?;
        (lit.base10_digits().to_string(), lit.suffix().to_string(), lit.span())
    };
    let text = alloc_format(format_args!("{digits}{suffix}"));
    if !suffix.is_empty() {
        const NUMERIC_TYPES: [&str; 14] = [
            "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64", "i128",
            "isize", "f32", "f64",
        ];
        let fix = if NUMERIC_TYPES.contains(&suffix.as_str()) {
            alloc_format(format_args!("drop the type suffix and write `{digits}`"))
        } else {
            alloc_format(format_args!("separate it from the unit: `{digits} * {suffix}`"))
        };
        return Err(syn::Error::new(
            span,
            alloc_format(format_args!("numeric scale factor `{text}` has a suffix; {fix}")),
        ));
    }

    // digits = mantissa [e exponent], mantissa = int [. frac]; value = m · 10^e.
    let (mantissa, exp10) = match digits.split_once(['e', 'E']) {
        Some((m, e)) => (m, e.parse::<i32>().map_err(|_| too_large(span, &text))?),
        None => (digits.as_str(), 0),
    };
    let (int, frac) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let mut m: u128 = alloc_format(format_args!("{int}{frac}"))
        .parse()
        .map_err(|_| too_large(span, &text))?;
    let exp10 = exp10 - frac.len() as i32;

    if m == 0 {
        return Err(syn::Error::new(
            span,
            alloc_format(format_args!(
                "`{text}` is not a valid scale factor: a unit's scale must be positive"
            )),
        ));
    }
    let mut powers = [0i32; 3];
    for (power, prime) in powers.iter_mut().zip([2u128, 3, 5]) {
        while m.is_multiple_of(prime) {
            m /= prime;
            *power += 1;
        }
    }
    if m != 1 {
        let value = if exp10 < 0 {
            let numerator = mantissa.replace('.', "");
            let numerator = numerator.trim_start_matches('0');
            if -exp10 <= 6 {
                alloc_format(format_args!(" (= {numerator}/1{})", "0".repeat(-exp10 as usize)))
            } else {
                alloc_format(format_args!(" (= {numerator}/10^{})", -exp10))
            }
        } else {
            alloc_format(format_args!(""))
        };
        return Err(syn::Error::new(
            span,
            alloc_format(format_args!(
                "`{text}`{value} is not on the whippyunits scale grid: a numeric scale factor \
                 must be a product of integer powers of 2, 3 and 5 (e.g. `100` = 2²·5², \
                 `60` = 2²·3·5, `0.6` = 3·5⁻¹), but this one leaves a factor of {m}. \
                 Powers of π come from angle units such as `rot` or `deg`."
            )),
        ));
    }
    powers[0] += exp10;
    powers[2] += exp10;

    let mut expr: Option<UnitExpr> = None;
    for (power, name) in powers.into_iter().zip(["power_of_2", "power_of_3", "power_of_5"]) {
        if power == 0 {
            continue;
        }
        let exponent = i16::try_from(power).map_err(|_| too_large(span, &text))?;
        let factor = UnitExpr::Unit(UnitExprUnit { name: Ident::new(name, span), exponent });
        expr = Some(match expr {
            Some(left) => UnitExpr::Mul(Box::new(left), Box::new(factor)),
            None => factor,
        });
    }
    Ok(expr.unwrap_or_else(|| {
        UnitExpr::Unit(UnitExprUnit { name: Ident::new("dimensionless", span), exponent: 1 })
    }))
}

fn too_large(span: proc_macro2::Span, text: &str) -> syn::Error {
    syn::Error::new(
        span,
        alloc_format(format_args!("numeric scale factor `{text}` is too large to represent")),
    )
}

#[cfg(not(test))]
fn alloc_format(args: core::fmt::Arguments) -> alloc::string::String {
    alloc::fmt::format(args)
}

#[cfg(test)]
fn alloc_format(args: core::fmt::Arguments) -> std::string::String {
    std::fmt::format(args)
}

/// Evaluation mode for unit expressions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvaluationMode {
    /// Strict mode: only allows storage units (for `unit!` macro)
    /// Nonstorage units will be treated as unknown/dimensionless
    Strict,
    /// Tolerant mode: allows both storage and nonstorage units (for `quantity!`, `value!`, serialization)
    Tolerant,
}

/// Result of evaluating a unit expression
#[derive(Debug, Clone, Copy)]
pub struct UnitEvaluationResult {
    pub dimension_exponents: DynDimensionExponents,
    pub scale_exponents: ScaleExponents,
}

/// Represents a unit expression that can be parsed
#[derive(Clone)]
pub enum UnitExpr {
    Unit(UnitExprUnit),
    Mul(Box<UnitExpr>, Box<UnitExpr>),
    Div(Box<UnitExpr>, Box<UnitExpr>),
    Pow(Box<UnitExpr>, LitInt),
}

impl Parse for UnitExpr {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut left = Self::parse_factor(input)?;

        while input.peek(Slash) {
            let _slash: Slash = input.parse()?;
            let right = Self::parse_factor(input)?;
            left = UnitExpr::Div(Box::new(left), Box::new(right));
        }

        Ok(left)
    }
}

impl UnitExpr {
    fn parse_factor(input: ParseStream) -> Result<Self> {
        let mut left = Self::parse_power(input)?;

        // Handle both * and . as multiplication operators (UCUM format uses .)
        while input.peek(Star) || input.peek(Dot) {
            if input.peek(Star) {
                let _star: Star = input.parse()?;
            } else if input.peek(Dot) {
                let _dot: Dot = input.parse()?;
            }
            let right = Self::parse_power(input)?;
            left = UnitExpr::Mul(Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    /// Collect all unit identifiers used in this expression
    pub fn collect_unit_identifiers(&self) -> Vec<Ident> {
        let mut identifiers = Vec::new();
        self.collect_identifiers_recursive(&mut identifiers);
        identifiers
    }

    fn collect_identifiers_recursive(&self, identifiers: &mut Vec<Ident>) {
        match self {
            UnitExpr::Unit(unit) => {
                identifiers.push(unit.name.clone());
            }
            UnitExpr::Mul(a, b) => {
                a.collect_identifiers_recursive(identifiers);
                b.collect_identifiers_recursive(identifiers);
            }
            UnitExpr::Div(a, b) => {
                a.collect_identifiers_recursive(identifiers);
                b.collect_identifiers_recursive(identifiers);
            }
            UnitExpr::Pow(base, _) => {
                base.collect_identifiers_recursive(identifiers);
            }
        }
    }

    fn parse_power(input: ParseStream) -> Result<Self> {
        let base = Self::parse_atom(input)?;

        if input.peek(Caret) {
            let _caret: Caret = input.parse()?;
            let exponent: LitInt = input.parse()?;
            Ok(UnitExpr::Pow(Box::new(base), exponent))
        } else {
            Ok(base)
        }
    }

    fn parse_atom(input: ParseStream) -> Result<Self> {
        if input.peek(syn::token::Paren) {
            let content;
            syn::parenthesized!(content in input);
            content.parse()
        } else if input.peek(syn::LitInt) || input.peek(syn::LitFloat) {
            // Numeric scale factors: "1" in "1 / m", "100" in "100 * rot / min"
            // (`^` is handled by `parse_power`).
            parse_numeric_factor(input)
        } else {
            let ident: Ident = input.parse()?;

            // Check for implicit exponent notation (UCUM format like "s2" instead of "s^2")
            let ident_str = ident.to_string();
            if let Some(pos) = ident_str.chars().position(|c| c.is_ascii_digit()) {
                let base_name = &ident_str[..pos];
                let exp_str = &ident_str[pos..];
                if let Ok(exp) = exp_str.parse::<i16>() {
                    // This is implicit exponent notation
                    let base_ident = syn::Ident::new(base_name, ident.span());
                    Ok(UnitExpr::Unit(UnitExprUnit {
                        name: base_ident,
                        exponent: exp,
                    }))
                } else {
                    // Not a valid exponent, treat as regular unit
                    Ok(UnitExpr::Unit(UnitExprUnit {
                        name: ident,
                        exponent: 1,
                    }))
                }
            } else {
                // Regular unit identifier
                Ok(UnitExpr::Unit(UnitExprUnit {
                    name: ident,
                    exponent: 1,
                }))
            }
        }
    }

    /// Validate that the unit expression doesn't contain nonstorage units (for strict mode)
    /// Returns an error message if any nonstorage units are found, None otherwise
    #[cfg(not(test))]
    pub fn validate_strict(&self) -> Option<alloc::string::String> {
        self.validate_strict_recursive()
    }

    #[cfg(test)]
    pub fn validate_strict(&self) -> Option<std::string::String> {
        self.validate_strict_recursive()
    }

    #[cfg(not(test))]
    fn validate_strict_recursive(&self) -> Option<alloc::string::String> {
        match self {
            UnitExpr::Unit(unit) => {
                if unit.is_numeric_factor() {
                    return None;
                }

                if let Some(unit_info) = get_unit_info(&unit.name.to_string()) {
                    // Check if this is a nonstorage unit
                    if unit_info.conversion_factor != 1.0 {
                        let unit_name = unit.name.to_string();
                        return Some(alloc::format!(
                            "Nonstorage unit '{}' cannot be used in `unit!` macro. Use `quantity!` macro instead, or use a storage unit.",
                            unit_name
                        ));
                    }
                }
                None
            }
            UnitExpr::Mul(a, b) => a
                .validate_strict_recursive()
                .or_else(|| b.validate_strict_recursive()),
            UnitExpr::Div(a, b) => a
                .validate_strict_recursive()
                .or_else(|| b.validate_strict_recursive()),
            UnitExpr::Pow(base, _) => base.validate_strict_recursive(),
        }
    }

    #[cfg(test)]
    fn validate_strict_recursive(&self) -> Option<std::string::String> {
        match self {
            UnitExpr::Unit(unit) => {
                if unit.is_numeric_factor() {
                    return None;
                }

                if let Some(unit_info) = get_unit_info(&unit.name.to_string()) {
                    // Check if this is a nonstorage unit
                    if unit_info.conversion_factor != 1.0 {
                        let unit_name = unit.name.to_string();
                        return Some(std::format!(
                            "Nonstorage unit '{}' cannot be used in `unit!` macro. Use `quantity!` macro instead, or use a storage unit.",
                            unit_name
                        ));
                    }
                }
                None
            }
            UnitExpr::Mul(a, b) => a
                .validate_strict_recursive()
                .or_else(|| b.validate_strict_recursive()),
            UnitExpr::Div(a, b) => a
                .validate_strict_recursive()
                .or_else(|| b.validate_strict_recursive()),
            UnitExpr::Pow(base, _) => base.validate_strict_recursive(),
        }
    }

    /// Evaluate the unit expression to get dimension exponents and scale factors
    ///
    /// In strict mode, nonstorage units are treated as unknown/dimensionless.
    /// In tolerant mode, all units (including nonstorage) are evaluated normally.
    pub fn evaluate(&self) -> UnitEvaluationResult {
        self.evaluate_with_mode(EvaluationMode::Strict)
    }

    /// Evaluate the unit expression with a specific evaluation mode
    pub fn evaluate_with_mode(&self, mode: EvaluationMode) -> UnitEvaluationResult {
        match self {
            UnitExpr::Unit(unit) => {
                if let Some(scale_exponents) = unit.numeric_scale() {
                    return UnitEvaluationResult {
                        dimension_exponents: DynDimensionExponents::ZERO,
                        scale_exponents,
                    };
                }

                if let Some(unit_info) = get_unit_info(&unit.name.to_string()) {
                    // In strict mode, nonstorage units should have been caught by validate_strict()
                    // But we still need to handle them here for safety - treat as unknown/dimensionless
                    if mode == EvaluationMode::Strict && unit_info.conversion_factor != 1.0 {
                        // This shouldn't happen if validate_strict() was called, but handle gracefully
                        return UnitEvaluationResult {
                            dimension_exponents: DynDimensionExponents::ZERO,
                            scale_exponents: ScaleExponents::IDENTITY,
                        };
                    }
                    // Get the dimension exponents and scale exponents from the unit
                    let mut dimension_exponents = unit_info.exponents.value();
                    let mut scale_exponents = unit_info.scale;

                    // Check if this is a prefixed unit and adjust scale factors accordingly
                    // BUT ONLY if the unit name is NOT a valid unit symbol by itself
                    let is_valid_unit_symbol =
                        Dimension::find_unit_by_symbol(&unit.name.to_string()).is_some();
                    if !is_valid_unit_symbol {
                        // Try all prefixes until we find one with a valid base unit
                        for prefix in SiPrefix::ALL {
                            // Try prefix symbol first (e.g., "kW" -> "W")
                            if let Some(base) = prefix.strip_prefix_symbol(&unit.name.to_string())
                                && !base.is_empty() {
                                    // Check if the base unit exists
                                    if Dimension::find_unit_by_symbol(base).is_some() {
                                        let prefix_factor = prefix.factor_log10();
                                        // Apply the prefix factor to the scale factors (powers of 2 and 5 for log10)
                                        scale_exponents =
                                            scale_exponents.mul(ScaleExponents::_10(prefix_factor));
                                        break;
                                    }
                                }
                            // Try prefix name (e.g., "kilowatt" -> "watt")
                            if let Some(base) = prefix.strip_prefix_name(&unit.name.to_string())
                                && !base.is_empty() {
                                    // Check if the base unit exists by name
                                    if Dimension::find_unit_by_name(base).is_some() {
                                        let prefix_factor = prefix.factor_log10();
                                        // Apply the prefix factor to the scale factors (powers of 2 and 5 for log10)
                                        scale_exponents =
                                            scale_exponents.mul(ScaleExponents::_10(prefix_factor));
                                        break;
                                    }
                                }
                        }
                    }

                    // Apply the unit exponent to both dimension and scale exponents
                    if unit.exponent != 1 {
                        dimension_exponents = dimension_exponents * unit.exponent;
                        scale_exponents = scale_exponents.scalar_exp(unit.exponent);
                    }

                    UnitEvaluationResult {
                        dimension_exponents,
                        scale_exponents,
                    }
                } else {
                    // Handle dimensionless or unknown units
                    UnitEvaluationResult {
                        dimension_exponents: DynDimensionExponents::ZERO,
                        scale_exponents: ScaleExponents::IDENTITY,
                    }
                }
            }
            UnitExpr::Mul(a, b) => {
                let result_a = a.evaluate_with_mode(mode);
                let result_b = b.evaluate_with_mode(mode);
                UnitEvaluationResult {
                    dimension_exponents: result_a.dimension_exponents
                        + result_b.dimension_exponents,
                    scale_exponents: result_a.scale_exponents.mul(result_b.scale_exponents),
                }
            }
            UnitExpr::Div(a, b) => {
                let result_a = a.evaluate_with_mode(mode);
                let result_b = b.evaluate_with_mode(mode);
                UnitEvaluationResult {
                    dimension_exponents: result_a.dimension_exponents
                        + (-result_b.dimension_exponents),
                    scale_exponents: result_a.scale_exponents.mul(result_b.scale_exponents.neg()),
                }
            }
            UnitExpr::Pow(base, exp) => {
                let result = base.evaluate_with_mode(mode);
                let exp_val: i16 = exp.base10_parse().unwrap();
                UnitEvaluationResult {
                    dimension_exponents: result.dimension_exponents * exp_val,
                    scale_exponents: result.scale_exponents.scalar_exp(exp_val),
                }
            }
        }
    }
}

/// Calculate conversion factor and affine offset from a parsed unit expression
/// Returns (conversion_factor, affine_offset) for nonstorage units
/// For storage units, returns (1.0, 0.0)
///
/// This handles compound units by recursively calculating conversion factors:
/// - Multiplication: conversion factors multiply, affine offsets propagate
/// - Division: conversion factors divide, affine offsets are divided
/// - Exponentiation: conversion factors are raised to power, affine offsets are multiplied
pub fn calculate_unit_conversion_factors(expr: &UnitExpr) -> (f64, f64) {
    calculate_conversion_factors_recursive(expr)
}

/// Recursively calculate conversion factors and affine offsets from a UnitExpr
fn calculate_conversion_factors_recursive(expr: &UnitExpr) -> (f64, f64) {
    match expr {
        UnitExpr::Unit(unit) => {
            if unit.is_numeric_factor() {
                return (1.0, 0.0);
            }

            if let Some(unit_info) = get_unit_info(&unit.name.to_string()) {
                let mut conversion_factor = unit_info.conversion_factor;
                let mut affine_offset = unit_info.affine_offset;

                // Apply exponent if present
                if unit.exponent != 1 {
                    // For exponents, conversion factor is raised to the power
                    conversion_factor = conversion_factor.powi(unit.exponent as i32);
                    // Affine offset is multiplied by the exponent (for temperature scales, etc.)
                    affine_offset *= unit.exponent as f64;
                }

                (conversion_factor, affine_offset)
            } else {
                // Unknown unit - assume storage unit
                (1.0, 0.0)
            }
        }
        UnitExpr::Mul(a, b) => {
            let (cf_a, af_a) = calculate_conversion_factors_recursive(a);
            let (cf_b, af_b) = calculate_conversion_factors_recursive(b);
            // For multiplication: conversion factors multiply
            // Affine offsets: only one unit should have an affine offset (typically temperature)
            // If both have affine offsets, we combine them (though this is unusual)
            // The affine offset applies to the entire product
            let total_affine = if af_a != 0.0 && af_b != 0.0 {
                // Both have affine offsets - this is unusual but we'll combine them
                // This typically doesn't make physical sense, but we handle it
                af_a * cf_b + af_b * cf_a
            } else if af_a != 0.0 {
                // If a has affine offset, it applies to the whole product
                af_a * cf_b
            } else {
                // If b has affine offset, it applies to the whole product
                af_b * cf_a
            };
            (cf_a * cf_b, total_affine)
        }
        UnitExpr::Div(a, b) => {
            let (cf_a, af_a) = calculate_conversion_factors_recursive(a);
            let (cf_b, af_b) = calculate_conversion_factors_recursive(b);
            // For division: conversion factors divide
            // Affine offsets: if denominator has affine offset, it's complex
            // If numerator has affine offset, it's divided by the denominator's conversion factor
            if af_b != 0.0 {
                // Denominator has affine offset - this is complex and typically invalid
                // For now, we'll treat it as an error case by returning a large value
                // In practice, this should be caught earlier, but we handle it gracefully
                (cf_a / cf_b, af_a / cf_b)
            } else {
                let total_affine = if af_a != 0.0 {
                    // Numerator affine offset is divided by denominator conversion factor
                    af_a / cf_b
                } else {
                    0.0
                };
                (cf_a / cf_b, total_affine)
            }
        }
        UnitExpr::Pow(base, exp) => {
            let (cf, af) = calculate_conversion_factors_recursive(base);
            let exp_val: i16 = exp.base10_parse().unwrap_or(1);
            // Conversion factor raised to power
            let new_cf = cf.powi(exp_val as i32);
            // Affine offset multiplied by exponent
            let new_af = af * exp_val as f64;
            (new_cf, new_af)
        }
    }
}

/// Get unit information for a unit name, handling prefixes and conversions
/// Returns the complete Unit struct with dimensions and scale factors
pub fn get_unit_info(unit_name: &str) -> Option<&'static Unit> {
    // Handle dimensionless units (like "1" in "1 / km")
    if unit_name == "dimensionless" {
        return None; // Dimensionless units don't have a corresponding Unit
    }

    // Check for exact unit match (symbol first, then name) before prefix checking
    if let Some((unit, _dimension)) = Dimension::find_unit(unit_name) {
        return Some(unit);
    }

    // Then check if this is a prefixed unit (like kg, kW, mm, etc.)
    // Only allow prefixing of base units (first unit in each dimension) and only for metric units
    for prefix in SiPrefix::ALL {
        if let Some(base) = prefix.strip_prefix_symbol(unit_name)
            && !base.is_empty() {
                // Check if the base unit exists and is a base unit (first unit in its dimension)
                if let Some((unit, dimension)) = Dimension::find_unit_by_symbol(base) {
                    // Check if this is the first unit in its dimension (base unit)
                    if dimension
                        .units
                        .first()
                        .map(|first_unit| first_unit.name == unit.name)
                        .unwrap_or(false)
                    {
                        // Only allow prefixing if the base unit is a metric unit (not imperial)
                        if unit.system == crate::System::Metric {
                            return Some(unit);
                        }
                    }
                }
            }
        if let Some(base) = prefix.strip_prefix_name(unit_name)
            && !base.is_empty() {
                // Check if the base unit exists by name and is a base unit
                if let Some((unit, dimension)) = Dimension::find_unit_by_name(base) {
                    // Check if this is the first unit in its dimension (base unit)
                    if dimension
                        .units
                        .first()
                        .map(|first_unit| first_unit.name == unit.name)
                        .unwrap_or(false)
                    {
                        // Only allow prefixing if the base unit is a metric unit (not imperial)
                        if unit.system == crate::System::Metric {
                            return Some(unit);
                        }
                    }
                }
            }
    }

    // If not found, return None
    None
}
