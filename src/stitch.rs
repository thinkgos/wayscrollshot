use std::sync::Arc;

use image::imageops::{self, FilterType};
use image::{DynamicImage, GenericImage, GrayImage, RgbaImage};

use crate::types::{PreviewImage, StitchStats};

pub struct MatchConfig {
    pub match_width: u32,
    pub min_overlap_full: u32,
    pub accept_diff: f32,
    pub min_append_full: u32,
}

pub struct Stitcher {
    full_image: Option<Arc<RgbaImage>>,
    last_frame: Option<RgbaImage>,
    stats: StitchStats,
    config: MatchConfig,
}

pub enum StitchOutcome {
    FirstFrame,
    Appended { added: u32 },
    NoProgress,
    NoMatch,
}

impl Stitcher {
    pub fn new(config: MatchConfig) -> Self {
        Self {
            full_image: None,
            last_frame: None,
            stats: StitchStats {
                frame_count: 0,
                total_height: 0,
                last_append: 0,
            },
            config,
        }
    }

    pub fn push_frame(&mut self, frame: RgbaImage) -> StitchOutcome {
        if self.full_image.is_none() {
            let height = frame.height();
            self.stats.frame_count = 1;
            self.stats.total_height = height;
            self.stats.last_append = height;
            self.full_image = Some(Arc::new(frame.clone()));
            self.last_frame = Some(frame);
            return StitchOutcome::FirstFrame;
        }

        let last_frame = match &self.last_frame {
            Some(prev) => prev,
            None => {
                self.last_frame = Some(frame);
                return StitchOutcome::NoMatch;
            }
        };

        let overlap = match find_overlap(last_frame, &frame, &self.config) {
            Some(overlap) => overlap,
            None => return StitchOutcome::NoMatch,
        };

        let new_height = frame.height().saturating_sub(overlap);
        if new_height < self.config.min_append_full {
            return StitchOutcome::NoProgress;
        }

        let full = self.full_image.as_ref().expect("full image set");
        let mut combined = RgbaImage::new(full.width(), full.height() + new_height);
        combined
            .copy_from(full.as_ref(), 0, 0)
            .expect("copy full image");
        let slice = imageops::crop_imm(&frame, 0, overlap, frame.width(), new_height).to_image();
        combined
            .copy_from(&slice, 0, full.height())
            .expect("copy slice");

        self.full_image = Some(Arc::new(combined));
        self.last_frame = Some(frame);
        self.stats.frame_count += 1;
        self.stats.total_height = self.full_image.as_ref().unwrap().height();
        self.stats.last_append = new_height;
        StitchOutcome::Appended { added: new_height }
    }

    pub fn full_image(&self) -> Option<Arc<RgbaImage>> {
        self.full_image.clone()
    }

    pub fn stats(&self) -> StitchStats {
        self.stats.clone()
    }
}

pub fn build_preview(image: &RgbaImage, fixed_width: u32) -> PreviewImage {
    let width = image.width();
    let height = image.height();
    let scale = (fixed_width as f32) / (width as f32).max(1.0);
    let target_width = fixed_width.max(1);
    let target_height = ((height as f32) * scale).round().max(1.0) as u32;
    // Use Nearest for speed, Triangle is too slow for real-time preview
    let resized = imageops::resize(image, target_width, target_height, FilterType::Nearest);
    PreviewImage {
        width: resized.width(),
        height: resized.height(),
        pixels: resized.into_raw(),
    }
}

fn find_overlap(prev: &RgbaImage, new: &RgbaImage, config: &MatchConfig) -> Option<u32> {
    if prev.width() == 0 || new.width() == 0 || prev.height() < 50 || new.height() < 50 {
        return None;
    }

    let target_width = config.match_width.min(prev.width()).max(1);
    let scale = target_width as f32 / prev.width() as f32;

    let prev_small = imageops::resize(
        prev,
        target_width,
        (prev.height() as f32 * scale).round().max(1.0) as u32,
        FilterType::Triangle,
    );
    let new_small = imageops::resize(
        new,
        target_width,
        (new.height() as f32 * scale).round().max(1.0) as u32,
        FilterType::Triangle,
    );
    let prev_gray = DynamicImage::ImageRgba8(prev_small).into_luma8();
    let new_gray = DynamicImage::ImageRgba8(new_small).into_luma8();

    // Use fixed template from top of new frame (like the SSD algorithm)
    // Template height: ~20% of frame height, min 30px, max 100px
    let template_height = (new_gray.height() / 5).clamp(30, 100);

    // Search range: where in prev_gray could this template be?
    // It should be near the bottom of prev_gray
    let search_start = (prev_gray.height() / 2).max(config.min_overlap_full); // Start from middle
    let search_end = prev_gray.height().saturating_sub(template_height);

    if search_start >= search_end {
        return None;
    }

    let mut best_y = None;
    let mut best_ssd = f64::MAX;

    // Search for the template position in prev_gray
    for y in search_start..=search_end {
        let ssd = compute_ssd(&prev_gray, &new_gray, y, template_height);
        if ssd < best_ssd {
            best_ssd = ssd;
            best_y = Some(y);
        }
    }

    let best_y = best_y?;

    // Convert SSD to average per-pixel difference for threshold check
    let pixel_count = (template_height * prev_gray.width()) as f64;
    let avg_diff = (best_ssd / pixel_count).sqrt();

    if avg_diff > config.accept_diff as f64 {
        return None;
    }

    // The overlap is from best_y to the end of prev
    // overlap_small = prev_gray.height() - best_y
    let overlap_small = prev_gray.height() - best_y;

    // Convert back to full resolution
    let overlap_full = ((overlap_small as f32) / scale).round() as u32;

    // Sanity check: overlap should leave at least min_append_full new pixels
    let new_height = new.height().saturating_sub(overlap_full);
    if new_height < config.min_append_full {
        return Some(new.height()); // No progress, return full overlap
    }

    Some(overlap_full)
}

/// Compute Sum of Squared Differences between prev[y..y+h] and new[0..h]
fn compute_ssd(prev: &GrayImage, new: &GrayImage, y: u32, h: u32) -> f64 {
    let width = prev.width().min(new.width());
    let mut sum = 0u64;

    for row in 0..h {
        let py = y + row;
        let ny = row;
        if py >= prev.height() || ny >= new.height() {
            break;
        }
        for x in 0..width {
            let p = prev.get_pixel(x, py)[0] as i32;
            let n = new.get_pixel(x, ny)[0] as i32;
            let diff = p - n;
            sum += (diff * diff) as u64;
        }
    }

    sum as f64
}
