use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use chrono::Local;
use image::codecs::png::PngEncoder;
use image::{ColorType, ImageEncoder, RgbaImage};

pub fn save_image(image: Arc<RgbaImage>, output_path: Option<PathBuf>) -> Result<PathBuf> {
    let path = match output_path {
        Some(p) => {
            if let Some(parent) = p.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)?;
                }
            }
            p
        }
        None => {
            let output_dir = default_output_dir();
            std::fs::create_dir_all(&output_dir)?;
            let filename = format!("wayscrollshot-{}.png", Local::now().format("%Y%m%d-%H%M%S"));
            output_dir.join(filename)
        }
    };
    image.save(&path)?;
    Ok(path)
}

pub fn copy_to_clipboard(image: Arc<RgbaImage>) -> Result<()> {
    let png_bytes = encode_png(&image)?;
    if command_exists("wl-copy") {
        let mut child = Command::new("wl-copy")
            .arg("--type")
            .arg("image/png")
            .stdin(Stdio::piped())
            .spawn()
            .context("failed to spawn wl-copy")?;
        if let Some(stdin) = child.stdin.as_mut() {
            stdin.write_all(&png_bytes)?;
        }
        let status = child.wait()?;
        if !status.success() {
            bail!("wl-copy failed");
        }
        return Ok(());
    }

    if command_exists("xclip") {
        let mut child = Command::new("xclip")
            .args(["-selection", "clipboard", "-t", "image/png", "-i"])
            .stdin(Stdio::piped())
            .spawn()
            .context("failed to spawn xclip")?;
        if let Some(stdin) = child.stdin.as_mut() {
            stdin.write_all(&png_bytes)?;
        }
        let status = child.wait()?;
        if !status.success() {
            bail!("xclip failed");
        }
        return Ok(());
    }

    bail!("no clipboard tool found (wl-copy/xclip)")
}

fn encode_png(image: &RgbaImage) -> Result<Vec<u8>> {
    let mut data = Vec::new();
    let encoder = PngEncoder::new(&mut data);
    encoder.write_image(image.as_raw(), image.width(), image.height(), ColorType::Rgba8)?;
    Ok(data)
}

fn command_exists(cmd: &str) -> bool {
    if let Some(paths) = std::env::var_os("PATH") {
        for path in std::env::split_paths(&paths) {
            let full = path.join(cmd);
            if full.is_file() {
                return true;
            }
        }
    }
    false
}

fn default_output_dir() -> PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        let pictures = PathBuf::from(&home).join("Pictures");
        if pictures.is_dir() {
            return pictures;
        }
        return PathBuf::from(home);
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}
