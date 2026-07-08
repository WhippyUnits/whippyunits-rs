use whippyunits::quantity;
use whippyunits::value;

fn main() {
    let distance = quantity!(5.0, m);
    let _x: f64 = value!(distance, xyz);
}
