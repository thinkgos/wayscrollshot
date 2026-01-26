use std::path::PathBuf;

use clap::Parser;

use crate::constants::PREVIEW_MAX_WIDTH;

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
}

impl Args {
    pub fn parse_args() -> Self {
        Args::parse()
    }
}
