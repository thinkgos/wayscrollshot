use std::path::PathBuf;

use clap::{Parser, ValueEnum};

use crate::constants::PREVIEW_MAX_WIDTH;

#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub enum Algorithm {
    /// Column sampling (fast, good for most cases)
    #[default]
    ColSample,
    /// Template matching (slower, more accurate)
    Template,
    /// Edge detection (for transparent backgrounds)
    Edge,
    /// FAST corner + HNSW index (high accuracy, from snow-shot)
    Fast,
}

#[derive(Parser, Debug, Clone)]
#[command(name = "wayscrollshot")]
#[command(about = "A scrolling screenshot tool for Wayland", long_about = None)]
pub struct Args {
    /// Output file path (default: ~/Pictures/wayscrollshot-<timestamp>.png)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Preview width in pixels
    #[arg(short = 'w', long, default_value_t = PREVIEW_MAX_WIDTH)]
    pub preview_width: u32,

    /// Copy to clipboard instead of saving to file
    #[arg(short, long)]
    pub clipboard: bool,

    /// Disable preview window
    #[arg(long)]
    pub no_preview: bool,

    /// Disable region border overlay
    #[arg(long)]
    pub no_border: bool,

    /// Stitching algorithm to use
    #[arg(short, long, value_enum, default_value_t = Algorithm::ColSample)]
    pub algorithm: Algorithm,
}

impl Args {
    pub fn parse_args() -> Self {
        Args::parse()
    }
}
