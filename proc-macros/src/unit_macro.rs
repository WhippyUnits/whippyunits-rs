use proc_macro2::TokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream, Result};
use syn::token::Comma;
use syn::Type;
use whippyunits_core::UnitExpr;

use crate::utils::shared_utils::{
    const_exp, generate_unit_documentation_for_expr, validate_known_units, whippyunits_path,
};

/// Input for the unit macro
pub struct UnitMacroInput {
    pub unit_expr: UnitExpr,
    pub storage_type: Option<Type>,
    pub brand_type: Option<Type>,
}

impl Parse for UnitMacroInput {
    fn parse(input: ParseStream) -> Result<Self> {
        let unit_expr = input.parse()?;

        // Check if there's a comma followed by a type parameter
        let storage_type = if input.peek(Comma) {
            let _comma: Comma = input.parse()?;
            Some(input.parse()?)
        } else {
            None
        };

        // Check if there's another comma followed by a brand type parameter
        let brand_type = if input.peek(Comma) {
            let _comma: Comma = input.parse()?;
            Some(input.parse()?)
        } else {
            None
        };

        Ok(UnitMacroInput {
            unit_expr,
            storage_type,
            brand_type,
        })
    }
}

impl UnitMacroInput {
    /// Shared front-end: validates and evaluates the unit expression, returning
    /// the bare `Unit<Scale, Dimension>` type tokens and the doc-struct block, or
    /// a `compile_error!` token stream on failure.
    fn evaluate_unit(&self) -> std::result::Result<(TokenStream, TokenStream), TokenStream> {
        if let Some(error) = validate_known_units(&self.unit_expr) {
            return Err(error);
        }

        // Validate that no nonstorage units are used (strict mode requirement)
        if let Some(error_msg) = self.unit_expr.validate_strict() {
            return Err(quote! {
                compile_error!(#error_msg);
            });
        }

        let result = self.unit_expr.evaluate();
        // Emit each exponent via `const_exp` so negative values round-trip
        // through rust-analyzer's proc-macro bridge (see `const_exp` docs).
        let (mass_exp, length_exp, time_exp, current_exp, temp_exp, amount_exp, lum_exp, angle_exp) = (
            const_exp(result.dimension_exponents.0[0]),
            const_exp(result.dimension_exponents.0[1]),
            const_exp(result.dimension_exponents.0[2]),
            const_exp(result.dimension_exponents.0[3]),
            const_exp(result.dimension_exponents.0[4]),
            const_exp(result.dimension_exponents.0[5]),
            const_exp(result.dimension_exponents.0[6]),
            const_exp(result.dimension_exponents.0[7]),
        );
        let (p2, p3, p5, pi) = (
            const_exp(result.scale_exponents.0[0]),
            const_exp(result.scale_exponents.0[1]),
            const_exp(result.scale_exponents.0[2]),
            const_exp(result.scale_exponents.0[3]),
        );

        // Generate documentation structs for unit identifiers in const expression.
        // Affine units aren't allowed in strict mode, so storage type is unused here.
        let doc_structs = generate_unit_documentation_for_expr(&self.unit_expr, false);

        let wu = whippyunits_path();
        let unit_type = quote! {
            #wu::quantity::Unit<
                #wu::quantity::Scale<#wu::quantity::_2<#p2>, #wu::quantity::_3<#p3>, #wu::quantity::_5<#p5>, #wu::quantity::_Pi<#pi>>,
                #wu::quantity::Dimension<#wu::quantity::_M<#mass_exp>, #wu::quantity::_L<#length_exp>, #wu::quantity::_T<#time_exp>, #wu::quantity::_I<#current_exp>, #wu::quantity::_Θ<#temp_exp>, #wu::quantity::_N<#amount_exp>, #wu::quantity::_J<#lum_exp>, #wu::quantity::_A<#angle_exp>>
            >
        };

        Ok((unit_type, doc_structs))
    }

    /// Expands `unit!(...)` to a bare [`Unit`] type. Any storage/brand arguments
    /// are ignored: a `Unit` carries only scale and dimension.
    pub fn expand_unit(self) -> TokenStream {
        let (unit_type, doc_structs) = match self.evaluate_unit() {
            Ok(pair) => pair,
            Err(err) => return err,
        };

        let wu = whippyunits_path();
        quote! {
            <#wu::Helper<{
                #doc_structs
                0
            }, #unit_type> as #wu::GetSecondGeneric>::Type
        }
    }

    /// Expands `qty!(...)` to a concrete [`Quantity`] type wrapping the unit,
    /// using the supplied storage type (default `f64`) and brand (default `()`).
    pub fn expand_quantity(self) -> TokenStream {
        // Use the specified storage type or default to f64
        let storage_type = self
            .storage_type
            .clone()
            .unwrap_or_else(|| syn::parse_str::<Type>("f64").unwrap());

        // Use the specified brand type or default to ()
        let brand_type = self
            .brand_type
            .clone()
            .unwrap_or_else(|| syn::parse_str::<Type>("()").unwrap());

        let (unit_type, doc_structs) = match self.evaluate_unit() {
            Ok(pair) => pair,
            Err(err) => return err,
        };

        let wu = whippyunits_path();
        let quantity_type = quote! {
            #wu::quantity::Quantity<
                #unit_type,
                #storage_type,
                #brand_type
            >
        };

        quote! {
            <#wu::Helper<{
                #doc_structs
                0
            }, #quantity_type> as #wu::GetSecondGeneric>::Type
        }
    }
}
