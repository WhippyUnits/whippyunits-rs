use whippyunits::quantity;

fn main() {
    // 0.33 = 33/100 leaves a factor of 11.
    let _x = quantity!(1.0, 0.33 * V);
}
