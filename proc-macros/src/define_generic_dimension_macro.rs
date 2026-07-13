use proc_macro2::TokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream, Result};
use syn::punctuated::Punctuated;
use syn::token::{Caret, Comma, Dot, Slash, Star};
use syn::{Ident, LitInt, Token};
use whippyunits_core::Dimension;

use crate::utils::dimension_suggestions::find_similar_dimensions;

// Parse dimension expressions like "Length / Time", "L / T", or "Mass * Length^2 / Time^2", "M * L^2 / T^2"
pub enum DimensionExpr {
    Dimension(Ident),
    Dimensionless, // Represents "1" - dimensionless quantity
    Mul(Box<DimensionExpr>, Box<DimensionExpr>),
    Div(Box<DimensionExpr>, Box<DimensionExpr>),
    Pow(Box<DimensionExpr>, LitInt),
}

impl Parse for DimensionExpr {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut left = Self::parse_factor(input)?;

        while input.peek(Slash) {
            let _slash: Slash = input.parse()?;
            let right = Self::parse_factor(input)?;
            left = DimensionExpr::Div(Box::new(left), Box::new(right));
        }

        Ok(left)
    }
}

impl DimensionExpr {
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
            left = DimensionExpr::Mul(Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    fn parse_power(input: ParseStream) -> Result<Self> {
        let base = Self::parse_atom(input)?;

        if input.peek(Caret) {
            let _caret: Caret = input.parse()?;
            let exponent: LitInt = input.parse()?;
            Ok(DimensionExpr::Pow(Box::new(base), exponent))
        } else {
            Ok(base)
        }
    }

    fn parse_atom(input: ParseStream) -> Result<Self> {
        if input.peek(syn::token::Paren) {
            let content;
            syn::parenthesized!(content in input);
            content.parse()
        } else if input.peek(syn::LitInt) {
            // Check for literal integer "1" representing dimensionless
            let lit: syn::LitInt = input.parse()?;
            let value: i64 = lit.base10_parse()?;
            if value == 1 {
                Ok(DimensionExpr::Dimensionless)
            } else {
                return Err(syn::Error::new(
                    lit.span(),
                    "Only the literal '1' is supported to represent dimensionless quantities",
                ));
            }
        } else {
            let ident: Ident = input.parse()?;

            // Check for implicit exponent notation (UCUM format like "L2" instead of "L^2")
            let ident_str = ident.to_string();
            if let Some(pos) = ident_str.chars().position(|c| c.is_ascii_digit()) {
                let base_name = &ident_str[..pos];
                let exp_str = &ident_str[pos..];
                if let Ok(exp) = exp_str.parse::<i16>() {
                    // This is implicit exponent notation
                    let base_ident = syn::Ident::new(base_name, ident.span());
                    Ok(DimensionExpr::Pow(
                        Box::new(DimensionExpr::Dimension(base_ident)),
                        syn::LitInt::new(&exp.to_string(), ident.span()),
                    ))
                } else {
                    // Not a valid exponent, treat as regular dimension
                    Ok(DimensionExpr::Dimension(ident))
                }
            } else {
                // Regular dimension identifier
                Ok(DimensionExpr::Dimension(ident))
            }
        }
    }

    // Evaluate the expression to get dimension exponents (safe version that doesn't panic)
    fn evaluate_safe(&self) -> (i16, i16, i16, i16, i16, i16, i16, i16) {
        match self {
            DimensionExpr::Dimensionless => {
                // Dimensionless quantity - all exponents are zero
                (0, 0, 0, 0, 0, 0, 0, 0)
            }
            DimensionExpr::Dimension(ident) => {
                let name_or_symbol = ident.to_string();

                // Look up dimension by name or symbol using direct API
                if let Some(dim_info) = Dimension::find_dimension(&name_or_symbol) {
                    return (
                        dim_info.exponents.0[0], // mass
                        dim_info.exponents.0[1], // length
                        dim_info.exponents.0[2], // time
                        dim_info.exponents.0[3], // current
                        dim_info.exponents.0[4], // temperature
                        dim_info.exponents.0[5], // amount
                        dim_info.exponents.0[6], // luminous_intensity
                        dim_info.exponents.0[7], // angle
                    );
                }

                // If not found, return zero exponents (error will be caught in documentation generation)
                (0, 0, 0, 0, 0, 0, 0, 0)
            }
            DimensionExpr::Mul(a, b) => {
                let (ma, la, ta, ca, tempa, aa, luma, anga) = a.evaluate_safe();
                let (mb, lb, tb, cb, tempb, ab, lumb, angb) = b.evaluate_safe();
                (
                    ma + mb,
                    la + lb,
                    ta + tb,
                    ca + cb,
                    tempa + tempb,
                    aa + ab,
                    luma + lumb,
                    anga + angb,
                )
            }
            DimensionExpr::Div(a, b) => {
                let (ma, la, ta, ca, tempa, aa, luma, anga) = a.evaluate_safe();
                let (mb, lb, tb, cb, tempb, ab, lumb, angb) = b.evaluate_safe();
                (
                    ma - mb,
                    la - lb,
                    ta - tb,
                    ca - cb,
                    tempa - tempb,
                    aa - ab,
                    luma - lumb,
                    anga - angb,
                )
            }
            DimensionExpr::Pow(base, exp) => {
                let (m, l, t, c, temp, a, lum, ang) = base.evaluate_safe();
                let exp_val: i16 = exp.base10_parse().unwrap();
                (
                    m * exp_val,
                    l * exp_val,
                    t * exp_val,
                    c * exp_val,
                    temp * exp_val,
                    a * exp_val,
                    lum * exp_val,
                    ang * exp_val,
                )
            }
        }
    }
}

pub struct DefineGenericDimensionInput {
    pub trait_name: Ident,
    pub _comma: Token![,],
    pub dimension_exprs: Punctuated<DimensionExpr, Comma>,
}

impl Parse for DefineGenericDimensionInput {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(DefineGenericDimensionInput {
            trait_name: input.parse()?,
            _comma: input.parse()?,
            dimension_exprs: input.parse_terminated(DimensionExpr::parse, Token![,])?,
        })
    }
}

impl DefineGenericDimensionInput {
    pub fn expand(self) -> TokenStream {
        let trait_name = &self.trait_name;

        // Generate documentation structs for dimension identifiers used in expressions
        let doc_structs = Self::generate_dimension_documentation(&self.dimension_exprs);

        // Generate the trait definition
        let trait_def = quote! {
            pub trait #trait_name {
                type Unit;
            }
        };

        // Generate implementations for each dimension expression
        let impl_blocks: Vec<TokenStream> = self
            .dimension_exprs
            .iter()
            .map(|expr| {
                // Try to evaluate the expression, but handle errors gracefully
                let (
                    mass_exp,
                    length_exp,
                    time_exp,
                    current_exp,
                    temp_exp,
                    amount_exp,
                    lum_exp,
                    angle_exp,
                ) = expr.evaluate_safe();
                self.generate_impl(
                    mass_exp,
                    length_exp,
                    time_exp,
                    current_exp,
                    temp_exp,
                    amount_exp,
                    lum_exp,
                    angle_exp,
                )
            })
            .collect();

        quote! {
            #doc_structs

            #trait_def

            #(#impl_blocks)*
        }
    }

    fn generate_impl(
        &self,
        mass_exp: i16,
        length_exp: i16,
        time_exp: i16,
        current_exp: i16,
        temp_exp: i16,
        amount_exp: i16,
        lum_exp: i16,
        angle_exp: i16,
    ) -> TokenStream {
        let trait_name = &self.trait_name;

        // For simplicity, we'll use 0 for all scale parameters
        // In a more sophisticated implementation, we could determine scale parameters based on the dimensions
        quote! {
            impl <
                const SCALE_P2: i16,
                const SCALE_P3: i16,
                const SCALE_P5: i16,
                const SCALE_PI: i16,
                T
            > #trait_name for whippyunits::quantity::Quantity<
                whippyunits::quantity::Scale<whippyunits::quantity::_2<SCALE_P2>, whippyunits::quantity::_3<SCALE_P3>, whippyunits::quantity::_5<SCALE_P5>, whippyunits::quantity::_Pi<SCALE_PI>>,
                whippyunits::quantity::Dimension<whippyunits::quantity::_M<#mass_exp>, whippyunits::quantity::_L<#length_exp>, whippyunits::quantity::_T<#time_exp>, whippyunits::quantity::_I<#current_exp>, whippyunits::quantity::_Θ<#temp_exp>, whippyunits::quantity::_N<#amount_exp>, whippyunits::quantity::_J<#lum_exp>, whippyunits::quantity::_A<#angle_exp>>,
                T
            > {
                type Unit = Self;
            }
        }
    }

    /// Generate documentation structs for dimension identifiers used in expressions
    fn generate_dimension_documentation(
        dimension_exprs: &Punctuated<DimensionExpr, Comma>,
    ) -> TokenStream {
        let mut doc_structs = Vec::new();

        // Generate documentation for each identifier occurrence (no filtering)
        for expr in dimension_exprs {
            Self::collect_and_generate_dimension_docs(expr, &mut doc_structs);
        }

        quote! {
            #(#doc_structs)*
        }
    }

    /// Recursively collect dimension identifiers and generate documentation for each occurrence
    fn collect_and_generate_dimension_docs(
        expr: &DimensionExpr,
        doc_structs: &mut Vec<TokenStream>,
    ) {
        match expr {
            DimensionExpr::Dimensionless => {
                // No documentation needed for dimensionless
            }
            DimensionExpr::Dimension(ident) => {
                // Generate documentation for this specific occurrence
                if let Some(doc_struct) = Self::generate_single_dimension_doc(ident) {
                    doc_structs.push(doc_struct);
                }
            }
            DimensionExpr::Mul(a, b) => {
                Self::collect_and_generate_dimension_docs(a, doc_structs);
                Self::collect_and_generate_dimension_docs(b, doc_structs);
            }
            DimensionExpr::Div(a, b) => {
                Self::collect_and_generate_dimension_docs(a, doc_structs);
                Self::collect_and_generate_dimension_docs(b, doc_structs);
            }
            DimensionExpr::Pow(base, _) => {
                Self::collect_and_generate_dimension_docs(base, doc_structs);
            }
        }
    }

    /// Generate documentation for a single dimension identifier.
    ///
    /// Looks up the dimension in `Dimension::ALL` via `find_dimension` and derives
    /// both the doc text and the trait path from the source-of-truth name.
    fn generate_single_dimension_doc(identifier: &Ident) -> Option<TokenStream> {
        let dimension_name = identifier.to_string();

        let Some(dim) = Dimension::find_dimension(&dimension_name) else {
            let error_message = Self::generate_dimension_error_message(&dimension_name);
            return Some(quote! {
                const _: () = {
                    compile_error!(#error_message);
                };
            });
        };

        // Trait name = dimension name with spaces stripped (e.g. "Electric Potential" -> "ElectricPotential")
        let trait_name = dim.name.replace(' ', "");
        let trait_ident = syn::Ident::new(&trait_name, proc_macro2::Span::call_site());

        // Build doc text from the source of truth
        let is_basis = dim.symbol.is_some();
        let kind = if is_basis { "Atomic" } else { "Derived" };
        let symbol_part = match dim.symbol {
            Some(s) => format!(" ({s})"),
            None => {
                let s = dim.exponents.to_symbol_string();
                if s == "1" { String::new() } else { format!(" ({s})") }
            }
        };
        let doc_text = format!("{kind} dimension: {}{symbol_part}", dim.name);

        let doc_ident = syn::Ident::new(&dimension_name, identifier.span());

        Some(quote! {
            const _: () = {
                #[doc = #doc_text]
                #[allow(dead_code)]
                trait #doc_ident: whippyunits::dimension_traits::#trait_ident {}

                impl<U: whippyunits::dimension_traits::#trait_ident> #doc_ident for U {}
            };
        })
    }

    /// Generate error message with suggestions for an unknown dimension
    fn generate_dimension_error_message(dimension_name: &str) -> String {
        let suggestions = find_similar_dimensions(dimension_name, 0.7);
        if suggestions.is_empty() {
            let supported_names: Vec<&str> = Dimension::ALL.iter().map(|dim| dim.name).collect();
            let supported_symbols: Vec<&str> = Dimension::ALL
                .iter()
                .filter_map(|dim| dim.symbol)
                .collect();

            format!(
                "Unknown dimension '{}'. Supported dimension names: {}. Supported dimension symbols: {}",
                dimension_name,
                supported_names.join(", "),
                supported_symbols.join(", ")
            )
        } else {
            let suggestion_list = suggestions
                .iter()
                .map(|(suggestion, _)| format!("'{}'", suggestion))
                .collect::<Vec<_>>()
                .join(", ");

            format!(
                "Unknown dimension '{}'. Did you mean: {}?",
                dimension_name, suggestion_list
            )
        }
    }
}
