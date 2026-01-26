use std::path::PathBuf;

use clap::Parser;

use crate::constants::{CAPTURE_INTERVAL_MS, PREVIEW_MAX_WIDTH};

#[derive(Parser, Debug, Clone)]
#[command(name = "long-shot")]
#[command(about = "A scrolling screenshot tool for Wayland", long_about = None)]
pub struct Args {
    /// Capture interval in milliseconds
    #[arg(short = 'i', long, default_value_t = CAPTURE_INTERVAL_MS)]
    pub interval: u64,

    /// Output file path (default: ~/Pictures/long-shot-<timestamp>.png)
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
}

impl Args {
    pub fn parse_args() -> Self {
        Args::parse()
    }
}
