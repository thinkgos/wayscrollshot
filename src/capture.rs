use std::process::Command;

use anyhow::{anyhow, bail, Context, Result};
use image::RgbaImage;

use crate::types::Region;

pub fn select_region() -> Result<Region> {
    let output = Command::new("slurp")
        .arg("-f")
        .arg("%x,%y %wx%h")
        .output()
        .context("failed to run slurp")?;
    if !output.status.success() {
        bail!("slurp exited with non-zero status");
    }
    let raw = String::from_utf8(output.stdout)?.trim().to_string();
    if raw.is_empty() {
        bail!("slurp returned empty selection");
    }
    log::debug!("slurp output: {}", raw);
    parse_region(&raw)
}

fn parse_region(raw: &str) -> Result<Region> {
    let mut parts = raw.split_whitespace();
    let coords = parts.next().ok_or_else(|| anyhow!("missing coords"))?;
    let size = parts.next().ok_or_else(|| anyhow!("missing size"))?;
    let (x_str, y_str) = coords
        .split_once(',')
        .ok_or_else(|| anyhow!("invalid coords"))?;
    let (w_str, h_str) = size
        .split_once('x')
        .ok_or_else(|| anyhow!("invalid size"))?;
    let x: i32 = x_str.parse()?;
    let y: i32 = y_str.parse()?;
    let w: u32 = w_str.parse()?;
    let h: u32 = h_str.parse()?;
    Ok(Region {
        raw: raw.to_string(),
        x,
        y,
        w,
        h,
    })
}

pub fn capture_frame(region: &Region) -> Result<RgbaImage> {
    log::debug!("grim capture region: {}", region.raw);
    let output = Command::new("grim")
        .arg("-g")
        .arg(&region.raw)
        .arg("-t")
        .arg("png")
        .arg("-l")
        .arg("0")
        .arg("-s")
        .arg("1")
        .arg("-")
        .output()
        .context("failed to run grim")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        log::error!("grim stderr: {}", stderr);
        bail!("grim exited with non-zero status");
    }
    let image = image::load_from_memory(&output.stdout)?;
    Ok(image.to_rgba8())
}
