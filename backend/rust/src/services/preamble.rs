/// Rust code prepended to every user submission.
/// Provides helper functions for metrics and tensor output
/// that match the sentinel patterns the frontend expects.
pub const SANDBOX_PREAMBLE: &str = r#"
#![allow(unused_imports, dead_code, unused_variables, unused_mut)]

fn kata_metric(name: &str, value: f64) {
    println!("__KATA_METRIC__:{}:{}:__END_KATA_METRIC__", name, value);
}

fn kata_tensor(name: &str, shape: &[usize], values: &[f64]) {
    let rows = if shape.len() >= 2 { shape[0] } else { 1 };
    let cols = if shape.len() >= 2 { shape[1] } else { shape.get(0).copied().unwrap_or(values.len()) };

    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let mut grid = String::from("[");
    for r in 0..rows {
        if r > 0 { grid.push(','); }
        grid.push('[');
        for c in 0..cols {
            if c > 0 { grid.push(','); }
            let idx = r * cols + c;
            if idx < values.len() {
                grid.push_str(&format!("{}", values[idx]));
            } else {
                grid.push_str("0");
            }
        }
        grid.push(']');
    }
    grid.push(']');

    let shape_str: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
    println!(
        "__KATA_TENSOR__:{{\"name\":\"{}\",\"shape\":[{}],\"values\":{},\"min\":{},\"max\":{}}}:__END_KATA_TENSOR__",
        name,
        shape_str.join(","),
        grid,
        min,
        max
    );
}

"#;
