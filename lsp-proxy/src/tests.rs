use crate::{inlay_hint_processor, quantity_detection, unit_formatter::UnitFormatter, LspProxy};
use serde_json::json;

#[test]
fn test_extract_raw_type_with_full_declaration() {
    let formatter = UnitFormatter::new();

    // Test with let declaration
    let let_declaration = "let result: Quantity<Unit<Scale, Dimension<_M, _L<1>>>, f64>";
    let raw_type = formatter.extract_raw_type_from_hover(let_declaration);
    assert_eq!(
        raw_type,
        "let result: Quantity<Unit<Scale, Dimension<_M, _L<1>>>, f64>"
    );

    // Test with let mut declaration
    let let_mut_declaration = "let mut result: Quantity<Unit<Scale, Dimension<_M, _L<1>>>, f64>";
    let raw_type = formatter.extract_raw_type_from_hover(let_mut_declaration);
    assert_eq!(
        raw_type,
        "let mut result: Quantity<Unit<Scale, Dimension<_M, _L<1>>>, f64>"
    );

    // Test with const declaration
    let const_declaration = "const result: Quantity<Unit<Scale, Dimension<_M, _L<1>>>, f64>";
    let raw_type = formatter.extract_raw_type_from_hover(const_declaration);
    assert_eq!(
        raw_type,
        "const result: Quantity<Unit<Scale, Dimension<_M, _L<1>>>, f64>"
    );

    // Test with static declaration
    let static_declaration = "static result: Quantity<Unit<Scale, Dimension<_M, _L<1>>>, f64>";
    let raw_type = formatter.extract_raw_type_from_hover(static_declaration);
    assert_eq!(
        raw_type,
        "static result: Quantity<Unit<Scale, Dimension<_M, _L<1>>>, f64>"
    );

    // Test with complex variable name
    let complex_var = "let my_complex_variable_name: Quantity<Unit<Scale, Dimension<_M, _L<1>>>, f64>";
    let raw_type = formatter.extract_raw_type_from_hover(complex_var);
    assert_eq!(
        raw_type,
        "let my_complex_variable_name: Quantity<Unit<Scale, Dimension<_M, _L<1>>>, f64>"
    );

    // Test with no declaration (should return empty)
    let no_declaration = "Quantity<Unit<Scale, Dimension<_M, _L<1>>>, f64>";
    let raw_type = formatter.extract_raw_type_from_hover(no_declaration);
    assert_eq!(raw_type, "");
}

#[test]
fn test_fast_quantity_detection() {
    // Test with message containing new Quantity types with Scale<...> and Dimension<...> structs
    let message_with_quantity = r#"{"jsonrpc":"2.0","id":1,"result":{"contents":{"kind":"markdown","value":"```rust\nlet x: Quantity<Unit<Scale<_2<0>, _3<0>, _5<0>, _Pi<0>>, Dimension<_M<0>, _L<1>, _T<0>, _I<0>, _Θ<0>, _N<0>, _J<0>, _A<0>>>, f64> = 5.0.meters();\n```"}}}"#;
    assert!(quantity_detection::contains_quantity_types_fast(
        message_with_quantity
    ));

    // Test with message containing truncated Quantity types (Scale and Dimension with defaulted parameters)
    let message_with_truncated_quantity = r#"{"jsonrpc":"2.0","id":1,"result":{"contents":{"kind":"markdown","value":"```rust\nlet x: Quantity<Unit<Scale, Dimension<_M<0>, _L<1>>>, f64> = 5.0.meters();\n```"}}}"#;
    assert!(quantity_detection::contains_quantity_types_fast(
        message_with_truncated_quantity
    ));

    // Test with message containing fully defaulted Quantity types (Scale and Dimension with no parameters)
    let message_with_fully_defaulted_quantity = r#"{"jsonrpc":"2.0","id":1,"result":{"contents":{"kind":"markdown","value":"```rust\nlet x: Quantity<Unit<Scale, Dimension>, f64> = 5.0;\n```"}}}"#;
    assert!(quantity_detection::contains_quantity_types_fast(
        message_with_fully_defaulted_quantity
    ));

    // Test with message not containing Quantity types
    let message_without_quantity = r#"{"jsonrpc":"2.0","id":1,"result":{"contents":{"kind":"markdown","value":"```rust\nlet x: String = \"hello\";\n```"}}}"#;
    assert!(!quantity_detection::contains_quantity_types_fast(
        message_without_quantity
    ));

    // Test with message containing "Quantity" but not in proper format
    let message_with_quantity_text = r#"{"jsonrpc":"2.0","id":1,"result":{"contents":{"kind":"markdown","value":"```rust\nlet x: String = \"Quantity\";\n```"}}}"#;
    assert!(!quantity_detection::contains_quantity_types_fast(
        message_with_quantity_text
    ));

    // Test early opt-out: message with "Quantity<" but no Scale/Dimension (should be fast rejection)
    let message_with_quantity_but_no_whippyunits = r#"{"jsonrpc":"2.0","id":1,"result":{"contents":{"kind":"markdown","value":"```rust\nlet x: Quantity<SomeOtherType> = something;\n```"}}}"#;
    assert!(!quantity_detection::contains_quantity_types_fast(
        message_with_quantity_but_no_whippyunits
    ));
}

#[test]
fn test_validate_quantity_format() {
    // Test valid new Quantity format with Scale<...> and Dimension<...> structs
    let valid_quantity = "Quantity<Unit<Scale<_2<0>, _3<0>, _5<0>, _Pi<0>>, Dimension<_M<0>, _L<1>, _T<0>, _I<0>, _Θ<0>, _N<0>, _J<0>, _A<0>>>, f64>";
    assert!(quantity_detection::validate_quantity_format(valid_quantity));

    // Test valid truncated Quantity format (Scale and Dimension with defaulted parameters)
    let valid_truncated_quantity = "Quantity<Unit<Scale, Dimension<_M<0>, _L<1>>>, f64>";
    assert!(quantity_detection::validate_quantity_format(
        valid_truncated_quantity
    ));

    // Test valid fully defaulted Quantity format (Scale and Dimension with no parameters)
    let valid_fully_defaulted_quantity = "Quantity<Unit<Scale, Dimension>, f64>";
    assert!(quantity_detection::validate_quantity_format(
        valid_fully_defaulted_quantity
    ));

    // Test invalid format without Scale<...> and Dimension<...> structs
    let invalid_quantity = "Quantity<1, 2, 3>";
    assert!(!quantity_detection::validate_quantity_format(
        invalid_quantity
    ));

    // Test with nested angle brackets
    let nested_quantity = "Quantity<Unit<Scale<_2<0>, _3<0>, _5<0>, _Pi<0>>, Dimension<_M<0>, _L<1>, _T<0>, _I<0>, _Θ<0>, _N<0>, _J<0>, _A<0>>>, Some<f64>>";
    assert!(quantity_detection::validate_quantity_format(
        nested_quantity
    ));
}

#[test]
fn test_find_matching_angle_bracket() {
    // Test simple case
    assert_eq!(
        quantity_detection::find_matching_angle_bracket("1, 2, 3>"),
        Some(7)
    );

    // Test with nested brackets
    assert_eq!(
        quantity_detection::find_matching_angle_bracket("1, 2, Some<f64>, 3>"),
        Some(18)
    );

    // Test with no closing bracket
    assert_eq!(
        quantity_detection::find_matching_angle_bracket("1, 2, 3"),
        None
    );

    // Test with multiple closing brackets
    assert_eq!(
        quantity_detection::find_matching_angle_bracket("1, 2, 3>, 4>"),
        Some(7)
    );
}

#[test]
fn test_hover_tooltip_processing() {
    let proxy = LspProxy::new();

    // Test hover response with new Quantity types (energy example)
    let hover_response = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "contents": {
                "kind": "markdown",
                "value": "```rust\nlet energy_j: Quantity<Unit<Scale<_2<0>, _3<0>, _5<0>, _Pi<0>>, Dimension<_M<1>, _L<2>, _T<-2>, _I<0>, _Θ<0>, _N<0>, _J<0>, _A<0>>>, f64> = 5.0.joules();\n```"
            }
        }
    });

    let response_str = serde_json::to_string(&hover_response).unwrap();
    let processed = proxy.process_incoming(&response_str).unwrap();

    // Should contain pretty-printed type (hover format) - should be Joules (J)
    println!("Hover processed: {}", processed);
    assert!(processed.contains("Quantity<J, f64>"));
    // Should not contain the raw const generic parameters
    assert!(!processed.contains("const MASS_EXPONENT: i16"));
    assert!(!processed.contains("const LENGTH_EXPONENT: i16"));
    assert!(!processed.contains("const TIME_EXPONENT: i16"));
    // Should not contain the incorrect _A<0> generic type
    assert!(!processed.contains("_A<0>"));
}

#[test]
fn test_hover_tooltip_processing_i32() {
    let proxy = LspProxy::new();

    // Test hover response with i32 type (like in the user's case)
    let hover_response = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "contents": {
                "kind": "markdown",
                "value": "```rust\nlet length_i32: Quantity<Unit<Scale, Dimension<_M, _L<1>>>, i32> = 10.meters();\n```"
            }
        }
    });

    let response_str = serde_json::to_string(&hover_response).unwrap();
    let processed = proxy.process_incoming(&response_str).unwrap();

    println!("Input hover: {}", response_str);
    println!("Processed hover: {}", processed);

    // Should contain pretty-printed type with i32, not f64
    assert!(processed.contains("Quantity<") && processed.contains("i32"));
    assert!(!processed.contains("f64"));
}

#[test]
fn test_bare_unit_type_formatting() {
    let converter = UnitFormatter::new();

    // A bare `Unit<...>` (as produced by whippyalgebra) prettifies to its label,
    // both standalone and when nested inside another generic type.
    let input =
        "MixedUnitMatrix<DCons<Unit<Scale, Dimension<_M<0>, _L<1>>>, DNil>, ColDims, Frame, M>";
    let result = converter.format_types(input, &crate::DisplayConfig::default());
    assert!(
        result.contains("MixedUnitMatrix<[m]"),
        "bare Unit should render as its label inside a collapsed list, got: {}",
        result
    );
    assert!(
        !result.contains("Unit<Scale") && !result.contains("DCons"),
        "raw Unit and cons-list noise should be gone, got: {}",
        result
    );

    // A dimensionless, unscaled bare Unit (all params defaulted) renders as `1`.
    let dimensionless = "Unit<Scale, Dimension>";
    let dimensionless_result =
        converter.format_types(dimensionless, &crate::DisplayConfig::default());
    assert_eq!(dimensionless_result, "1");
}

#[test]
fn test_type_conversion() {
    let converter = UnitFormatter::new();

    // Test basic type conversion with new format
    let input = "Quantity<Unit<Scale<_2<0>, _3<0>, _5<0>, _Pi<0>>, Dimension<_M<0>, _L<1>, _T<0>, _I<0>, _Θ<0>, _N<0>, _J<0>, _A<0>>>, f64>";
    let result = converter.format_types(input, &crate::DisplayConfig::default());
    assert!(result.contains("Quantity<m, f64>"));
    assert!(!result.contains("const"));
    assert!(!result.contains("MASS_EXPONENT"));

    // Test type conversion with truncated format
    let truncated_input = "Quantity<Unit<Scale, Dimension<_M<0>, _L<1>>>, f64>";
    let truncated_result =
        converter.format_types(truncated_input, &crate::DisplayConfig::default());
    println!("Truncated input: {}", truncated_input);
    println!("Truncated result: {}", truncated_result);
    assert!(truncated_result.contains("Quantity<m, f64>"));
    assert!(!truncated_result.contains("const"));
    assert!(!truncated_result.contains("MASS_EXPONENT"));

    // Test type conversion with fully defaulted format (dimensionless)
    let fully_defaulted_input = "Quantity<Unit<Scale, Dimension>, f64>";
    let fully_defaulted_result =
        converter.format_types(fully_defaulted_input, &crate::DisplayConfig::default());
    assert!(
        fully_defaulted_result.contains("Quantity<1, f64>")
            || fully_defaulted_result.contains("Quantity<dimensionless, f64>")
    );
    assert!(!fully_defaulted_result.contains("const"));
    assert!(!fully_defaulted_result.contains("MASS_EXPONENT"));
}

#[test]
fn test_angular_unit_with_defaulted_interior_dims() {
    let converter = UnitFormatter::new();

    // Regression: an angular unit like `V/rot` prints an explicit angle marker
    // (`_A<-1>`) while the interior temperature/amount/luminosity dims are at
    // their zero default and therefore rendered *without* brackets (`_Θ, _N,
    // _J`). The dimension parser must treat those bare markers as exponent 0,
    // not as a parse failure — otherwise the whole type is left unformatted.
    let input = "Quantity<Unit<Scale<_2<-1>, _3, _5, _Pi<-1>>, Dimension<_M<1>, _L<2>, _T<-3>, _I<-1>, _Θ, _N, _J, _A<-1>>>, f64>";
    let result = converter.format_types(input, &crate::DisplayConfig::default());
    assert!(
        !result.contains("Unit<Scale") && !result.contains("Dimension<"),
        "angular unit should be prettified, not left raw, got: {}",
        result
    );
    // The `_A<-1>` (per-rotation) dimension must survive into the label. The
    // electrical dims are now recognized as the composite `V` (partial
    // factorization) instead of expanding to SI base units, and the angular
    // scale is attributed to its unit, so the whole thing reads `V·rot⁻¹`
    // rather than `kg·m²·s⁻³·A⁻¹·rot⁻¹` or a `1/(2π)` radian recast.
    assert!(
        result.contains("V·rot⁻¹"),
        "expected the electrical dims to factor as V and the angle as rot⁻¹, got: {}",
        result
    );
    assert!(
        !result.contains("rad"),
        "angular scale should attribute to rot, not leak a radian recast, got: {}",
        result
    );
}

#[test]
fn test_pure_powers_of_composite_units() {
    let converter = UnitFormatter::new();

    // A pure integer power of a named composite unit (volt = kg·m²·s⁻³·A⁻¹) should
    // stay compact (`V²`, `V⁻¹`, `V⁻²`) rather than expanding into its
    // base-dimension decomposition (`kg⁻²·m⁻⁴·s⁶·A²`).
    let cases = [
        // (M, L, T, I) exponents, expected label
        ("_M<2>, _L<4>, _T<-6>, _I<-2>", "V²"), // V²
        ("_M<-1>, _L<-2>, _T<3>, _I<1>", "V⁻¹"), // 1/V
        ("_M<-2>, _L<-4>, _T<6>, _I<2>", "V⁻²"), // 1/V²
    ];

    for (dims, expected) in cases {
        let input = format!(
            "Quantity<Unit<Scale<_2<0>, _3<0>, _5<0>, _Pi<0>>, Dimension<{}, _Θ<0>, _N<0>, _J<0>, _A<0>>>, f64>",
            dims
        );
        let result = converter.format_types(&input, &crate::DisplayConfig::default());
        assert!(
            result.contains(expected),
            "expected `{}` in prettified output, got: {}",
            expected,
            result
        );
        assert!(
            !result.contains("kg"),
            "composite power should not expand to base dimensions, got: {}",
            result
        );
    }
}

#[test]
fn test_reciprocal_named_units_prefer_their_own_name() {
    let converter = UnitFormatter::new();

    // When a dimension is *itself* an exactly-named unit whose reciprocal is
    // *also* named (ohm ↔ siemens), the exact name must win over the reciprocal
    // spelling. `V/A` is `Ω`, not `S⁻¹`; a bare siemens is `S`, not `Ω⁻¹`.
    // Regression: `try_unit_power_literal`'s negated-power branch used to fire
    // before the exact dimension-name lookup, swapping the two.
    let cases = [
        // (M, L, T, I) exponents, expected label, forbidden reciprocal
        ("_M<1>, _L<2>, _T<-3>, _I<-2>", "Ω", "S⁻¹"), // ohm = V/A
        ("_M<-1>, _L<-2>, _T<3>, _I<2>", "S", "Ω⁻¹"), // siemens = A/V
    ];

    for (dims, expected, forbidden) in cases {
        let input = format!(
            "Quantity<Unit<Scale<_2<0>, _3<0>, _5<0>, _Pi<0>>, Dimension<{}, _Θ<0>, _N<0>, _J<0>, _A<0>>>, f64>",
            dims
        );
        let result = converter.format_types(&input, &crate::DisplayConfig::default());
        assert!(
            result.contains(expected),
            "expected `{}` in prettified output, got: {}",
            expected,
            result
        );
        assert!(
            !result.contains(forbidden),
            "reciprocal spelling `{}` should not shadow the exact named unit, got: {}",
            forbidden,
            result
        );
        assert!(
            !result.contains("kg"),
            "named unit should not expand to base dimensions, got: {}",
            result
        );
    }
}

#[test]
fn test_pure_powers_of_angular_units() {
    let converter = UnitFormatter::new();

    // Alternate angular units (rot = 2π rad = `_2<1>·_Pi<1>`, deg = `_2<-2>·_3<-2>·
    // _5<-1>·_Pi<1>`) are distinguished by their *scale*. A pure power scales that
    // by `k`, which stops matching the exp-1 registration — so `rot⁻¹` used to be
    // recast to `(0.1590)rad⁻¹`. It should stay `rot⁻¹` / `rot⁻²` / `deg²`.
    let cases = [
        // (scale, angle exponent), expected label
        ("_2<-1>, _3<0>, _5<0>, _Pi<-1>", "_A<-1>", "rot⁻¹"),
        ("_2<-2>, _3<0>, _5<0>, _Pi<-2>", "_A<-2>", "rot⁻²"),
        ("_2<-4>, _3<-4>, _5<-2>, _Pi<2>", "_A<2>", "deg²"),
    ];

    for (scale, angle, expected) in cases {
        let input = format!(
            "Quantity<Unit<Scale<{}>, Dimension<_M<0>, _L<0>, _T<0>, _I<0>, _Θ<0>, _N<0>, _J<0>, {}>>, f64>",
            scale, angle
        );
        let result = converter.format_types(&input, &crate::DisplayConfig::default());
        assert!(
            result.contains(expected),
            "expected `{}` in prettified output, got: {}",
            expected,
            result
        );
        assert!(
            !result.contains("rad") && !result.contains("0.1590") && !result.contains("0.02530"),
            "angular power should not be recast to radians with a coefficient, got: {}",
            result
        );
    }
}

#[test]
fn test_compound_angular_units_no_coefficient() {
    let converter = UnitFormatter::new();

    // Partial factorization: a compound that mixes a scale-bearing angular unit
    // with an ordinary dimension (e.g. `s/rot`) used to leak the angular scale
    // out as a numeric coefficient — `(0.1590)(s·rad⁻¹)`. The angular scale must
    // instead be attributed to the angle slot: `s·rot⁻¹`, `s·rot⁻²`, `s²·rot⁻²`.
    let cases = [
        // (scale, T exponent, A exponent), expected substring
        ("_2<-1>, _3<0>, _5<0>, _Pi<-1>", "_T<1>", "_A<-1>", "s·rot⁻¹"),
        ("_2<-2>, _3<0>, _5<0>, _Pi<-2>", "_T<1>", "_A<-2>", "s·rot⁻²"),
        ("_2<-2>, _3<0>, _5<0>, _Pi<-2>", "_T<2>", "_A<-2>", "s²·rot⁻²"),
        // rot/s (positive angular power in a compound) already worked; keep it honest.
        ("_2<1>, _3<0>, _5<0>, _Pi<1>", "_T<-1>", "_A<1>", "s⁻¹·rot"),
    ];

    for (scale, t_exp, a_exp, expected) in cases {
        let input = format!(
            "Quantity<Unit<Scale<{}>, Dimension<_M<0>, _L<0>, {}, _I<0>, _Θ<0>, _N<0>, _J<0>, {}>>, f64>",
            scale, t_exp, a_exp
        );
        let result = converter.format_types(&input, &crate::DisplayConfig::default());
        assert!(
            result.contains(expected),
            "expected `{}` in prettified output, got: {}",
            expected,
            result
        );
        assert!(
            !result.contains("rad") && !result.contains("0.1590") && !result.contains("0.02530"),
            "compound angular unit should not leak a radian coefficient, got: {}",
            result
        );
    }
}

#[test]
fn test_partial_factorization_of_compound_composites() {
    let converter = UnitFormatter::new();

    // A compound whose electrical/mechanical dims form a named composite (V, J,
    // N, Wb, …) but that is *not itself* a registered dimension should peel off
    // the composite instead of fully decomposing into base SI units. The policy
    // is min-factor / biggest-composite, so `V·s·rot⁻¹` (3 factors) collapses to
    // `Wb·rot⁻¹` (weber = V·s, 2 factors), not the base `kg·m²·s⁻²·A⁻¹·rot⁻¹`.
    let cases = [
        // (scale, dims, expected substring, forbidden substring)
        (
            "_2<-1>, _3<0>, _5<0>, _Pi<-1>",
            "_M<1>, _L<2>, _T<-2>, _I<-1>, _Θ<0>, _N<0>, _J<0>, _A<-1>",
            "Wb·rot⁻¹",
            "kg",
        ),
        // `V·s²/rot`: the exponent-aware cost keeps this consistent with the
        // `V·s/rot` case above — both settle on the weber (V·s) basis, so this
        // is `Wb·s·rot⁻¹` rather than tying at `V·s²·rot⁻¹`.
        (
            "_2<-1>, _3<0>, _5<0>, _Pi<-1>",
            "_M<1>, _L<2>, _T<-1>, _I<-1>, _Θ<0>, _N<0>, _J<0>, _A<-1>",
            "Wb·s·rot⁻¹",
            "kg",
        ),
        (
            "_2<-1>, _3<0>, _5<0>, _Pi<-1>",
            "_M<1>, _L<2>, _T<-2>, _I<0>, _Θ<0>, _N<0>, _J<0>, _A<-1>",
            "J·rot⁻¹",
            "kg",
        ),
        (
            "_2<0>, _3<0>, _5<0>, _Pi<0>",
            "_M<1>, _L<1>, _T<-1>, _I<0>, _Θ<0>, _N<0>, _J<0>, _A<0>",
            "N·s",
            "kg",
        ),
    ];

    for (scale, dims, expected, forbidden) in cases {
        let input = format!(
            "Quantity<Unit<Scale<{}>, Dimension<{}>>, f64>",
            scale, dims
        );
        let result = converter.format_types(&input, &crate::DisplayConfig::default());
        assert!(
            result.contains(expected),
            "expected `{}` in prettified output, got: {}",
            expected,
            result
        );
        assert!(
            !result.contains(forbidden),
            "composite should be peeled, not fully decomposed, got: {}",
            result
        );
    }

    // Negative control: a compound with no factor-reducing composite (velocity)
    // must stay as its plain base decomposition rather than over-extracting.
    let velocity = "Quantity<Unit<Scale<_2<0>, _3<0>, _5<0>, _Pi<0>>, Dimension<_M<0>, _L<1>, _T<-1>, _I<0>, _Θ<0>, _N<0>, _J<0>, _A<0>>>, f64>";
    let result = converter.format_types(velocity, &crate::DisplayConfig::default());
    assert!(
        result.contains("m·s⁻¹"),
        "velocity should stay a plain base decomposition, got: {}",
        result
    );
}

#[test]
fn test_composite_unresolved_type_conversion() {
    let converter = UnitFormatter::new();

    // Test composite unresolved type conversion with new format
    let input = "Quantity<Unit<Scale<_2<0>, _3<0>, _5<0>, _Pi<0>>, Dimension<_M<0>, _L<1>, _T<0>, _I<0>, _Θ<0>, _N<0>, _J<0>, _A<0>>>, f64> + Quantity<Unit<Scale<_2<0>, _3<0>, _5<0>, _Pi<0>>, Dimension<_M<0>, _L<1>, _T<0>, _I<0>, _Θ<0>, _N<0>, _J<0>, _A<0>>>, f64>";
    let result = converter.format_types(input, &crate::DisplayConfig::default());
    assert!(result.contains("m"));
    assert!(!result.contains("const"));
}

#[test]
fn test_verbose_partially_resolved_type() {
    let converter = UnitFormatter::new();

    // Test verbose partially resolved type conversion with new format
    let input = "Quantity<Unit<Scale<_2<0>, _3<0>, _5<0>, _Pi<0>>, Dimension<_M<0>, _L<1>, _T<0>, _I<0>, _Θ<0>, _N<0>, _J<0>, _A<0>>>, f64>";
    let result = converter.format_types(
        input,
        &crate::DisplayConfig {
            verbose: true,
            unicode: true,
            include_raw: false,
        },
    );
    assert!(result.contains("Quantity<meter"));
    assert!(result.contains("f64"));
    assert!(!result.contains("const"));
}

#[test]
fn test_add_sub_trait_signature_transformation() {
    let proxy = LspProxy::new();

    // Test Add trait
    let add_hover_response = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "contents": {
                "kind": "markdown",
                "value": "```rust\nimpl<const MASS_EXPONENT: i16, const LENGTH_EXPONENT: i16, const TIME_EXPONENT: i16, const CURRENT_EXPONENT: i16, const TEMPERATURE_EXPONENT: i16, const AMOUNT_EXPONENT: i16, const LUMINOSITY_EXPONENT: i16, const ANGLE_EXPONENT: i16, const SCALE_P2: i16, const SCALE_P3: i16, const SCALE_P5: i16, const SCALE_PI: i16, T> Add for Quantity<Unit<Scale<_2<SCALE_P2>, _3<SCALE_P3>, _5<SCALE_P5>, _Pi<SCALE_PI>>, Dimension<_M<MASS_EXPONENT>, _L<LENGTH_EXPONENT>, _T<TIME_EXPONENT>, _I<CURRENT_EXPONENT>, _Θ<TEMPERATURE_EXPONENT>, _N<AMOUNT_EXPONENT>, _J<LUMINOSITY_EXPONENT>, _A<ANGLE_EXPONENT>>>, T>\n```"
            }
        }
    });

    let response_str = serde_json::to_string(&add_hover_response).unwrap();
    let processed = proxy.process_incoming(&response_str).unwrap();

    // Extract JSON payload from LSP message format
    let json_payload = if processed.starts_with("Content-Length:") {
        // Extract JSON from LSP message format
        let json_start = processed.find("\r\n\r\n").unwrap() + 4;
        &processed[json_start..]
    } else {
        &processed
    };

    // Parse the processed result to verify the transformation
    let processed_json: serde_json::Value = serde_json::from_str(json_payload).unwrap();
    let contents = &processed_json["result"]["contents"];
    let contents_str = contents["value"].as_str().unwrap();

    println!("Processed Add trait: {}", contents_str);
    // Should show a simplified Add trait signature
    assert!(contents_str.contains("impl Add for"));
    assert!(contents_str.contains("Quantity<"));
    // Should not contain the const generic parameters
    assert!(!contents_str.contains("const MASS_EXPONENT: i16"));
    assert!(!contents_str.contains("const LENGTH_EXPONENT: i16"));
    assert!(!contents_str.contains("const TIME_EXPONENT: i16"));
}

// Inlay hint processor tests (moved from inlay_hint_processor.rs)

#[test]
fn test_inlay_hint_contains_whippyunits_type() {
    let processor = inlay_hint_processor::InlayHintProcessor::new();

    // Test with whippyunits type
    let label_with_quantity = vec![
        json!({"value": ": "}),
        json!({"value": "Quantity", "location": {"uri": "file://test.rs", "range": {"start": {"line": 1, "character": 0}, "end": {"line": 1, "character": 8}}}}),
        json!({"value": "<Scale<_2<0>, _3<0>, _5<0>, _Pi<0>>, Dimension<_M<1>, _L<0>, _T<0>, _I<0>, _Θ<0>, _N<0>, _J<0>, _A<0>>, f64>"}),
    ];
    assert!(processor.contains_whippyunits_type(&label_with_quantity));

    // Test without whippyunits type
    let label_without_quantity = vec![
        json!({"value": ": "}),
        json!({"value": "String"}),
        json!({"value": "()"}),
    ];
    assert!(!processor.contains_whippyunits_type(&label_without_quantity));
}

#[test]
fn test_inlay_hint_convert_whippyunits_hint() {
    let processor = inlay_hint_processor::InlayHintProcessor::new();

    let mut label_array = vec![
        json!({"value": ": "}),
        json!({
            "value": "Quantity",
            "location": {
                "uri": "file://test.rs",
                "range": {
                    "start": {"line": 1, "character": 0},
                    "end": {"line": 1, "character": 8}
                }
            }
        }),
        json!({"value": "<Unit<Scale<_2<0>, _3<0>, _5<0>, _Pi<0>>, Dimension<_M<0>, _L<1>, _T<0>, _I<0>, _Θ<0>, _N<0>, _J<0>, _A<0>>>, f64>"}),
    ];

    processor
        .convert_whippyunits_hint(&mut label_array)
        .unwrap();

    // Should have 2 parts now (removed generic params)
    assert_eq!(label_array.len(), 2);

    // First part should still be ": "
    assert_eq!(label_array[0]["value"], ": ");

    // Second part should be pretty-printed and have location preserved
    let second_part = &label_array[1];
    assert!(second_part["value"]
        .as_str()
        .unwrap()
        .contains("Quantity<m, f64>"));
    assert!(second_part.get("location").is_some());
}

#[test]
fn test_dcons_collapse_hover() {
    let converter = UnitFormatter::new();

    // A MixedUnitMatrix hover: DCons cons-lists collapse to bracketed arrays, and the
    // inner bare Units prettify to their labels.
    let input = "MixedUnitMatrix<DCons<Unit<Scale, Dimension<_M, _L<1>, _T<-1>>>, DCons<Unit<Scale, Dimension<_M, _L<1>, _T<-2>>>, DNil>>, DCons<Unit<Scale, Dimension<_M, _L<1>>>, DNil>, Matrix<f64, Const<2>, Const<1>, ArrayStorage<f64, 2, 1>>, ()>";
    let result = converter.format_types(input, &crate::DisplayConfig::default());

    assert!(
        !result.contains("DCons") && !result.contains("DNil"),
        "cons-list noise should be gone, got: {}",
        result
    );
    assert!(
        result.contains("MixedUnitMatrix<[") && result.contains("], [") ,
        "dim lists should render as bracketed arrays, got: {}",
        result
    );
    // The M parameter (nalgebra Matrix<…>) must be preserved verbatim.
    assert!(
        result.contains("Matrix<f64, Const<2>, Const<1>, ArrayStorage<f64, 2, 1>>"),
        "backing matrix type should be preserved, got: {}",
        result
    );
    // The trailing default `()` brand should be elided.
    assert!(
        !result.contains("()"),
        "default brand should be stripped, got: {}",
        result
    );
}

#[test]
fn test_unit_matrix_trace_diagram() {
    let converter = UnitFormatter::new();
    // A 2x2 matrix: rows [m/s, m/s²], cols [s, 1]. Entry (i,j) = row_i / col_j.
    let hover = "```rust\nlet m: MixedUnitMatrix<DCons<Unit<Scale, Dimension<_M, _L<1>, _T<-1>>>, DCons<Unit<Scale, Dimension<_M, _L<1>, _T<-2>>>, DNil>>, DCons<Unit<Scale, Dimension<_T<1>>>, DCons<Unit<Scale, Dimension>, DNil>>, Matrix<f64, Const<2>, Const<2>, ArrayStorage<f64, 2, 2>>, ()>\n```";
    let out = converter.format_types_with_original(hover, &crate::DisplayConfig::default(), hover);

    // The prettified type is unchanged, followed by a fenced trace diagram.
    assert!(
        out.contains("MixedUnitMatrix<[(m·s⁻¹), (m·s⁻²)], [s, 1], Matrix<"),
        "type line missing/incorrect: {out}"
    );
    assert!(out.contains("```text"), "trace should be a fenced block: {out}");

    let trace = out.split("```text").nth(1).unwrap();
    // Column labels in the top margin, row labels in the left margin.
    assert!(trace.contains("[(m·s⁻²)  (m·s⁻¹)]"), "row 0 cells wrong: {out}");
    assert!(trace.contains("[(m·s⁻³)  (m·s⁻²)]"), "row 1 cells wrong: {out}");
    // Row-label margin present.
    assert!(trace.contains("(m·s⁻¹) ["), "row-0 margin label missing: {out}");

    // With Raw enabled, the trace must come *before* the Raw section.
    let cfg = crate::DisplayConfig {
        include_raw: true,
        ..crate::DisplayConfig::default()
    };
    let with_raw = converter.format_types_with_original(hover, &cfg, hover);
    let trace_pos = with_raw.find("```text").expect("trace block present");
    let raw_pos = with_raw.find("Raw:").expect("raw section present");
    assert!(
        trace_pos < raw_pos,
        "trace should render before Raw section: {with_raw}"
    );
}

#[test]
fn test_unit_matrix_trace_absent_for_generic() {
    let converter = UnitFormatter::new();
    // A generic MixedUnitMatrix (non-concrete dim lists) yields no diagram.
    let hover = "```rust\nlet m: MixedUnitMatrix<RowDims, ColDims, M>\n```";
    let out = converter.format_types_with_original(hover, &crate::DisplayConfig::default(), hover);
    assert!(!out.contains("```text"), "no trace expected for generic: {out}");
}

#[test]
fn test_uniform_unit_matrix_trace_diagram() {
    let converter = UnitFormatter::new();
    // A 2×3 uniform matrix, every entry in meters. Shape comes from the storage
    // `Const<_>` dims (there are no dimension lists); the single unit `m` fills
    // every cell, with no row/column margins (a uniform matrix has no gauge).
    let hover = "```rust\nlet m: UniformUnitMatrix<Unit<Scale, Dimension<_M, _L<1>>>, Matrix<f64, Const<2>, Const<3>, ArrayStorage<f64, 2, 3>>, ()>\n```";
    let out = converter.format_types_with_original(hover, &crate::DisplayConfig::default(), hover);

    assert!(out.contains("```text"), "trace should be a fenced block: {out}");
    let trace = out.split("```text").nth(1).unwrap();
    // Two rows of three `m` cells each…
    assert!(trace.contains("[m  m  m]"), "uniform cells wrong: {out}");
    assert_eq!(
        trace.matches("[m  m  m]").count(),
        2,
        "expected 2 rows: {out}"
    );
    // …and NO margins: the mixed diagram would prefix a row with `m [`, the
    // uniform one must start each row flush at `[`.
    assert!(
        !trace.contains("m ["),
        "uniform trace must have no margins: {out}"
    );
}

#[test]
fn test_uniform_unit_matrix_trace_absent_for_generic() {
    let converter = UnitFormatter::new();
    // Generic storage (no concrete `Const<_>` dims) → no diagram.
    let hover = "```rust\nlet m: UniformUnitMatrix<Unit<Scale, Dimension<_L<1>>>, M, ()>\n```";
    let out = converter.format_types_with_original(hover, &crate::DisplayConfig::default(), hover);
    assert!(!out.contains("```text"), "no trace expected for generic: {out}");
}

#[test]
fn test_dcons_collapse_nested() {
    let converter = UnitFormatter::new();
    // Bare, pre-formatted list (no units to resolve) should still collapse.
    let input = "DCons<a, DCons<b, DCons<c, DNil>>>";
    let result = converter.format_types(input, &crate::DisplayConfig::default());
    assert_eq!(result, "[a, b, c]");
}

#[test]
fn test_inlay_hint_unit_matrix() {
    let processor = inlay_hint_processor::InlayHintProcessor::new();

    // rust-analyzer splits a composite type into many located/punctuation parts.
    // A `MixedUnitMatrix` carries no `Quantity` part, only `MixedUnitMatrix`/`DCons`/`Unit`.
    let mut label_array = vec![
        json!({"value": ": "}),
        json!({"value": "MixedUnitMatrix", "location": {"uri": "file://test.rs", "range": {"start": {"line": 1, "character": 0}, "end": {"line": 1, "character": 10}}}}),
        json!({"value": "<"}),
        json!({"value": "DCons"}),
        json!({"value": "<"}),
        json!({"value": "Unit"}),
        json!({"value": "<Scale, Dimension<_M, _L<1>, _T<-1>>>, DNil>, Matrix<f64, Const<6>, Const<6>, ArrayStorage<f64, 6, 6>>, ()>"}),
    ];

    assert!(processor.contains_whippyunits_type(&label_array));
    processor
        .convert_whippyunits_hint(&mut label_array)
        .unwrap();

    // Leading ": " preserved; type collapsed into a single located part.
    assert_eq!(label_array.len(), 2);
    assert_eq!(label_array[0]["value"], ": ");
    let type_value = label_array[1]["value"].as_str().unwrap();
    assert!(
        type_value.contains("MixedUnitMatrix<[(m") && type_value.contains("s⁻¹)"),
        "inner Unit should be prettified into a bracketed list, got: {}",
        type_value
    );
    assert!(
        !type_value.contains("Unit<Scale") && !type_value.contains("DCons"),
        "raw Unit and cons-list noise should be gone, got: {}",
        type_value
    );
    assert!(
        !type_value.contains("()"),
        "default brand should be stripped, got: {}",
        type_value
    );
    // Location preserved for go-to-definition.
    assert!(label_array[1].get("location").is_some());
}

#[test]
fn test_inlay_hint_exponent_pruning() {
    let processor = inlay_hint_processor::InlayHintProcessor::new();

    // Test that ^1 exponents are pruned but meaningful exponents are preserved
    let test_cases = vec![
        ("mm¹", "mm"),           // ^1 should be removed
        ("mm²", "mm²"),          // ^2 should be preserved
        ("mm³", "mm³"),          // ^3 should be preserved
        ("mm⁻¹", "mm⁻¹"),        // ^-1 should be preserved (meaningful negative exponent)
        ("m¹s²", "ms²"),         // ^1 should be removed, ^2 preserved
        ("kg¹m²s⁻²", "kgm²s⁻²"), // ^1 should be removed, others preserved
    ];

    for (input, expected) in test_cases {
        let result = processor.prune_inlay_hint_exponents(input);
        assert_eq!(result, expected, "Failed for input: {}", input);
    }
}

#[test]
fn test_scale_parsing_with_missing_pi_parameter() {
    let converter = UnitFormatter::new();

    // Test the user's specific case: Scale<_2<-3>, _3<0>, _5<-3>> (missing _Pi parameter)
    let input = "Quantity<Unit<Scale<_2<-3>, _3<0>, _5<-3>>, Dimension<_M<1>>>, f64>";
    let result = converter.format_types(input, &crate::DisplayConfig::default());

    println!("Input: {}", input);
    println!("Result: {}", result);

    // Should successfully parse and format the type
    assert!(result.contains("Quantity<"));
    assert!(!result.contains("Scale<_2<-3>"));
    assert!(!result.contains("const"));
}

#[test]
fn test_wholly_unresolved_type_formatting() {
    let converter = UnitFormatter::new();

    // Test the user's specific case: wholly unresolved type with all parameters as _
    // This matches the exact format from the IDE hover
    let input = "Quantity<Unit<Scale<_2<_>, _3<_>, _5<_>, _Pi<_>>, Dimension<_M<_>, _L<_>, _T<_>, _I<_>, _Θ<_>, _N<_>, _J<_>, _A<_>>>>";
    let result = converter.format_types(input, &crate::DisplayConfig::default());

    // Should format as wholly unresolved type
    assert!(result.contains("Quantity<?, f64>"));
    assert!(!result.contains("Scale<_2<_>"));
    assert!(!result.contains("Dimension<_M<_>"));
    assert!(!result.contains("const"));

    // Test with inlay hint formatting
    let inlay_result = converter.format_types_inlay_hint(input, &crate::DisplayConfig::default());
    println!("Inlay hint result: {}", inlay_result);
    assert!(inlay_result.contains("Quantity<?, f64>"));
}

#[test]
fn test_partially_resolved_type_formatting() {
    let converter = UnitFormatter::new();

    // Test partially resolved type: some dimensions known, some scales unknown
    let input = "Quantity<Unit<Scale<_2<_>, _3<0>, _5<_>, _Pi<0>>, Dimension<_M<0>, _L<1>, _T<0>, _I<_>, _Θ<0>, _N<0>, _J<0>, _A<0>>>, f64>";
    let result = converter.format_types(input, &crate::DisplayConfig::default());

    println!("Input: {}", input);
    println!("Result: {}", result);

    // Should format with best-effort guesses and unicode question marks for unresolved parts
    assert!(result.contains("Quantity<"));
    assert!(result.contains("f64"));
    assert!(!result.contains("Scale<_2<_>"));
    assert!(!result.contains("Dimension<_M<_>"));
    assert!(!result.contains("const"));

    // The result should contain some resolved parts (like length dimension) and question marks for unresolved parts
    // This tests that the partial resolution logic works correctly
}

#[test]
fn test_pid_controller_nested_quantity_types() {
    let converter = UnitFormatter::new();

    // Test complex nested type with multiple Quantity types in PIDController
    // This tests that the algorithm finds and transforms ALL Quantity types, not just the first one
    let input = "let mut controller: PIDController<Quantity<Unit<Scale, Dimension<_M, _L<1>>>>, Quantity<Unit<Scale<_2<_>, _3<_>, _5<_>, _Pi<_>>, Dimension<_M<_>, _L<_>, _T<_>, _I<_>, _Θ<_>, _N<_>, _J<_>, _A<_>>>>, Quantity<Unit<Scale<_2<-3>, _3, _5<-3>>, Dimension<_M, _L, _T<1>>>>, Quantity<Unit<Scale, Dimension<_M<1>, _L<1>, _T<-3>, _I<-1>>>>, Quantity<Unit<Scale<_2<3>, _3, _5<3>>, Dimension<_M<1>, _L<1>, _T<-4>, _I<-1>>>>, Quantity<Unit<Scale<_2<-3>, _3, _5<-3>>, Dimension<_M<1>, _L<1>, _T<-2>, _I<-1>>>>, Quantity<Unit<Scale<_2<_>, _3<_>, _5<_>, _Pi<_>>, Dimension<_M<_>, _L<_>, _T<_>, _I<_>, _Θ<_>, _N<_>, _J<_>, _A<_>>>>, Quantity<Unit<Scale<_2<_>, _3<_>, _5<_>, _Pi<_>>, Dimension<_M<_>, _L<_>, _T<_>, _I<_>, _Θ<_>, _N<_>, _J<_>, _A<_>>>>>";

    let result = converter.format_types(input, &crate::DisplayConfig::default());

    println!("Input: {}", input);
    println!("Result: {}", result);

    // Verify that all Quantity types were transformed (not left as raw Unit/Scale/Dimension format)
    // Count how many "Unit<Scale" appear in the result - should be 0 (all transformed)
    let raw_quantity_count = result.matches("Unit<Scale").count();
    if raw_quantity_count > 0 {
        // Find where the untransformed types are
        let mut positions = Vec::new();
        let mut search_pos = 0;
        while let Some(pos) = result[search_pos..].find("Unit<Scale") {
            positions.push(search_pos + pos);
            search_pos += pos + 1;
        }
        panic!("Found {} untransformed Unit<Scale types at positions {:?}. All should be transformed.\nResult: {}", raw_quantity_count, positions, result);
    }

    // Verify that we have formatted Quantity types (should contain "Quantity<" with formatted units)
    assert!(
        result.contains("Quantity<"),
        "Result should contain formatted Quantity types"
    );

    // Verify that the PIDController structure is preserved
    assert!(
        result.contains("PIDController<"),
        "Result should preserve PIDController structure"
    );

    // Count how many formatted Quantity types appear - should be 8 (one for each parameter)
    let formatted_quantity_count = result.matches("Quantity<").count();
    assert_eq!(
        formatted_quantity_count, 8,
        "Expected 8 formatted Quantity types, found {}",
        formatted_quantity_count
    );

    // Verify that unresolved types are formatted as "Quantity<?"
    let unresolved_count = result.matches("Quantity<?").count();
    assert_eq!(
        unresolved_count, 3,
        "Expected 3 unresolved Quantity types (the ones with _ placeholders), found {}",
        unresolved_count
    );

    // Verify that resolved types are formatted with actual units (not raw Scale/Dimension)
    assert!(
        !result.contains("Scale<_2<"),
        "Result should not contain raw Scale parameters"
    );
    assert!(
        !result.contains("Dimension<_M<"),
        "Result should not contain raw Dimension parameters"
    );
}

#[test]
fn test_dimensionless_ratio_with_scale_factor() {
    let converter = UnitFormatter::new();

    // Test dimensionless ratio with scale factor (m/mm) - this should format as k()
    // The type should be: Quantity<Unit<Scale<_2<3>, _3<0>, _5<3>, _Pi<0>>, Dimension>>, f64>
    // Note: Dimension has no brackets when all dimensions are zero

    // Test hover format (in markdown code block)
    let hover_input = r#"```rust
let ratio: Quantity<Unit<Scale<_2<3>, _3<0>, _5<3>, _Pi<0>>, Dimension>>, f64> = quantity!(1.0, m/mm);
```"#;

    let hover_result = converter.format_types(hover_input, &crate::DisplayConfig::default());
    println!("Hover input: {}", hover_input);
    println!("Hover result: {}", hover_result);

    // Should detect and transform the type
    assert!(
        !hover_result.contains("Scale<_2<3>"),
        "Hover should transform the type"
    );
    assert!(
        hover_result.contains("Quantity<"),
        "Hover should contain formatted Quantity"
    );

    // Test resolve format (label array)
    let resolve_label = vec![
        json!({"value": ": "}),
        json!({"value": "Quantity"}),
        json!({"value": "<Scale<_2<3>, _3<0>, _5<3>, _Pi<0>>, Dimension>, f64>"}),
    ];

    let processor = inlay_hint_processor::InlayHintProcessor::new();
    let mut label_array = resolve_label.clone();

    println!("Resolve label array: {:?}", label_array);

    if processor.contains_whippyunits_type(&label_array) {
        processor
            .convert_whippyunits_hint(&mut label_array)
            .unwrap();
        println!("Resolve result: {:?}", label_array);

        // Should have transformed the Quantity part
        let quantity_value = label_array.iter().find(|part| {
            part.get("value")
                .and_then(|v| v.as_str())
                .map(|s| s.starts_with("Quantity<"))
                .unwrap_or(false)
        });
        assert!(
            quantity_value.is_some(),
            "Resolve should transform the type"
        );
    } else {
        panic!("Resolve label array should be detected as containing whippyunits type");
    }

    // Test direct type string format
    let direct_type = "Quantity<Unit<Scale<_2<3>, _3<0>, _5<3>, _Pi<0>>, Dimension>>, f64>";
    let direct_result =
        converter.format_types_inlay_hint(direct_type, &crate::DisplayConfig::default());
    println!("Direct type: {}", direct_type);
    println!("Direct result: {}", direct_result);

    // Should format as k() for dimensionless ratio with scale factor
    assert!(
        direct_result.contains("k()"),
        "Should format as k() for m/mm ratio"
    );
    assert!(
        !direct_result.contains("Scale<_2<3>"),
        "Should not contain raw Scale"
    );

    // Test actual LSP hover response format
    let hover_response = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "contents": {
                "kind": "markdown",
                "value": "```rust\nlet ratio: Quantity<Unit<Scale<_2<3>, _3<0>, _5<3>, _Pi<0>>, Dimension>>, f64> = quantity!(1.0, m/mm);\n```"
            }
        }
    });

    let proxy = LspProxy::new();
    let message = format!(
        "Content-Length: {}\r\n\r\n{}",
        serde_json::to_string(&hover_response).unwrap().len(),
        serde_json::to_string(&hover_response).unwrap()
    );

    let processed = proxy.process_incoming(&message).unwrap();
    println!("Hover response message: {}", message);
    println!("Processed hover response: {}", processed);

    // Check if detection is working first
    let hover_json_str = serde_json::to_string(&hover_response).unwrap();
    let detected = quantity_detection::contains_quantity_types_fast(&hover_json_str);
    println!("Detection result for hover JSON: {}", detected);

    // Should detect and transform
    if detected {
        assert!(
            processed.contains("k()"),
            "Hover response should be transformed"
        );
        assert!(
            !processed.contains("Scale<_2<3>"),
            "Hover response should not contain raw Scale"
        );
    } else {
        panic!("Hover JSON should be detected by contains_quantity_types_fast");
    }

    // Test actual LSP resolve response format
    let resolve_response = json!({
        "jsonrpc": "2.0",
        "id": 2,
        "result": {
            "position": {"line": 30, "character": 8},
            "label": [
                {"value": ": "},
                {"value": "Quantity"},
                {"value": "<Unit<Scale<_2<3>, _3<0>, _5<3>, _Pi<0>>, Dimension>>, f64>"}
            ]
        }
    });

    let resolve_message = format!(
        "Content-Length: {}\r\n\r\n{}",
        serde_json::to_string(&resolve_response).unwrap().len(),
        serde_json::to_string(&resolve_response).unwrap()
    );

    let processed_resolve = proxy.process_incoming(&resolve_message).unwrap();
    println!("Resolve response message: {}", resolve_message);
    println!("Processed resolve response: {}", processed_resolve);

    // Should detect and transform
    assert!(
        processed_resolve.contains("k()"),
        "Resolve response should be transformed"
    );
    assert!(
        !processed_resolve.contains("Scale<_2<3>"),
        "Resolve response should not contain raw Scale"
    );

    // Test detection directly
    let hover_json = r#"{"id":1,"jsonrpc":"2.0","result":{"contents":{"kind":"markdown","value":"```rust\nlet ratio: Quantity<Unit<Scale<_2<3>, _3<0>, _5<3>, _Pi<0>>, Dimension>>, f64> = quantity!(1.0, m/mm);\n```"}}}"#;
    let detected = quantity_detection::contains_quantity_types_fast(hover_json);
    println!("Detection result for hover JSON: {}", detected);
    assert!(detected, "Hover JSON should be detected");
}
