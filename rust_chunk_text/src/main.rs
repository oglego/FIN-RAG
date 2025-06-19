// src/main.rs

/// Splits a text file into chunks of a specified size and prints each chunk,
/// separated by "===CHUNK===".
///
/// # Usage
/// ```sh
/// cargo run -- <file_path> <chunk_size>
/// ```
///
/// # Arguments
/// * `<file_path>` - Path to the input text file.
/// * `<chunk_size>` - (Optional) Size of each chunk in bytes (default: 500).
///
/// # Example
/// ```sh
/// cargo run -- sample.txt 1000
/// ```
use std::env;
use std::fs;
use rayon::prelude::*;

fn main() {
    let args: Vec<String> = env::args().collect();
    let file_path = &args[1];
    let chunk_size = args[2].parse::<usize>().unwrap_or(500);

    let content = fs::read_to_string(file_path).expect("Could not read file");
    let chunks: Vec<&str> = content
        .as_bytes()
        .par_chunks(chunk_size)
        .map(|chunk| std::str::from_utf8(chunk).unwrap_or(""))
        .collect();

    for chunk in chunks {
        println!("===CHUNK===\n{}", chunk);
    }
}