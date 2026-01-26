use std::sync::Arc;

use image::imageops::{self, FilterType};
use image::{GenericImage, RgbaImage};

use crate::cli::Algorithm;
use crate::types::{PreviewImage, StitchStats};

pub struct MatchConfig {
    pub min_overlap: u32,
    pub accept_diff: f32,
    pub min_append: u32,
    pub approx_diff: f32,
    pub algorithm: Algorithm,
}

pub struct Stitcher {
    full_image: Option<Arc<RgbaImage>>,
    last_frame: Option<RgbaImage>,
    last_cols: Option<ColSamples>,
    last_offset: i32,
    stats: StitchStats,
    config: MatchConfig,
}

pub enum StitchOutcome {
    FirstFrame,
    Appended { added: u32 },
    NoProgress,
    NoMatch,
}

/// Column samples: (height, num_groups) matrix
type ColSamples = Vec<Vec<f32>>;

impl Stitcher {
    pub fn new(config: MatchConfig) -> Self {
        Self {
            full_image: None,
            last_frame: None,
            last_cols: None,
            last_offset: 0,
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
            self.last_frame = Some(frame.clone());
            self.last_cols = Some(self.compute_cols(&frame));
            return StitchOutcome::FirstFrame;
        }

        let (offset, confidence) = match self.config.algorithm {
            Algorithm::Template => self.find_offset_template(&frame),
            Algorithm::ColSample | Algorithm::Edge => self.find_offset_colsample(&frame),
        };

        if confidence > self.config.accept_diff {
            self.last_frame = Some(frame.clone());
            self.last_cols = Some(self.compute_cols(&frame));
            return StitchOutcome::NoMatch;
        }

        let new_height = if offset > 0 { offset as u32 } else { 0 };

        if new_height < self.config.min_append {
            self.last_frame = Some(frame.clone());
            self.last_cols = Some(self.compute_cols(&frame));
            self.last_offset = offset;
            return StitchOutcome::NoProgress;
        }

        // Append new content
        let full = self.full_image.as_ref().expect("full image set");
        let mut combined = RgbaImage::new(full.width(), full.height() + new_height);
        combined
            .copy_from(full.as_ref(), 0, 0)
            .expect("copy full image");

        let overlap = frame.height().saturating_sub(new_height);
        let slice = imageops::crop_imm(&frame, 0, overlap, frame.width(), new_height).to_image();
        combined
            .copy_from(&slice, 0, full.height())
            .expect("copy slice");

        self.full_image = Some(Arc::new(combined));
        self.last_frame = Some(frame.clone());
        self.last_cols = Some(self.compute_cols(&frame));
        self.last_offset = offset;
        self.stats.frame_count += 1;
        self.stats.total_height = self.full_image.as_ref().unwrap().height();
        self.stats.last_append = new_height;
        StitchOutcome::Appended { added: new_height }
    }

    fn compute_cols(&self, frame: &RgbaImage) -> ColSamples {
        match self.config.algorithm {
            Algorithm::Edge => col_sampling_edge(frame),
            _ => col_sampling(frame),
        }
    }

    fn find_offset_colsample(&self, frame: &RgbaImage) -> (i32, f32) {
        let cols = self.compute_cols(frame);
        let last_cols = match &self.last_cols {
            Some(c) => c,
            None => return (0, f32::MAX),
        };

        diff_overlap(
            last_cols,
            &cols,
            self.last_offset,
            self.config.approx_diff,
            self.config.min_overlap,
        )
    }

    /// Template matching algorithm (similar to OpenCV TM_CCOEFF_NORMED)
    fn find_offset_template(&self, frame: &RgbaImage) -> (i32, f32) {
        let prev = match &self.last_frame {
            Some(f) => f,
            None => return (0, f32::MAX),
        };

        let h = prev.height() as i32;
        let w = prev.width() as i32;

        if h < 100 || w < 50 {
            return (0, f32::MAX);
        }

        // Template: top 20% of current frame (skip top 5% for sticky headers)
        let skip_top = (h as f32 * 0.05) as u32;
        let template_height = (h as f32 * 0.20) as u32;
        let template = imageops::crop_imm(frame, 0, skip_top, w as u32, template_height).to_image();
        let template_gray = to_grayscale(&template);

        // Search in previous frame
        let prev_gray = to_grayscale(prev);

        // Search range: from skip_top to bottom
        let search_start = skip_top as i32;
        let search_end = h - template_height as i32;

        if search_end <= search_start {
            return (0, f32::MAX);
        }

        let mut best_offset = 0i32;
        let mut best_score = f32::MIN;

        // Start from predicted offset and expand outward
        let predict = self.last_offset.clamp(0, search_end - search_start);
        let offsets = predict_offset_iter(search_end - search_start, predict);

        for offset in offsets {
            let search_y = search_start + offset;
            if search_y < 0 || search_y + template_height as i32 > h {
                continue;
            }

            let score = ncc_score(&prev_gray, &template_gray, search_y as u32);

            if score > best_score {
                best_score = score;
                best_offset = offset;
            }

            // Early termination if we found a very good match
            if best_score > 0.95 {
                break;
            }
        }

        // Convert NCC score to "diff" (lower is better)
        // NCC ranges from -1 to 1, we want 0 to be perfect match
        let diff = 1.0 - best_score.max(0.0);

        (best_offset, diff * 10.0) // Scale to match accept_diff threshold
    }

    pub fn full_image(&self) -> Option<Arc<RgbaImage>> {
        self.full_image.clone()
    }

    pub fn stats(&self) -> StitchStats {
        self.stats.clone()
    }
}

/// Convert image to grayscale (single channel f32)
fn to_grayscale(img: &RgbaImage) -> Vec<f32> {
    img.pixels()
        .map(|p| 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32)
        .collect()
}

/// Normalized Cross-Correlation score between template and a region of the image
fn ncc_score(image_gray: &[f32], template_gray: &[f32], y_offset: u32) -> f32 {
    let img_h = (image_gray.len() as f32).sqrt() as u32; // Approximate, assuming square-ish
    let tmpl_len = template_gray.len();

    if tmpl_len == 0 {
        return f32::MIN;
    }

    // Calculate template mean
    let tmpl_mean: f32 = template_gray.iter().sum::<f32>() / tmpl_len as f32;

    // Calculate template std dev
    let tmpl_var: f32 = template_gray
        .iter()
        .map(|&v| (v - tmpl_mean).powi(2))
        .sum::<f32>()
        / tmpl_len as f32;
    let tmpl_std = tmpl_var.sqrt();

    if tmpl_std < 1.0 {
        return f32::MIN; // Template is too uniform
    }

    // For simplicity, we'll use a row-based comparison
    // This is a simplified NCC that works well for scrolling content
    let mut sum_img_sq = 0.0f32;
    let mut img_sum = 0.0f32;
    let mut count = 0usize;

    let start_idx = (y_offset as usize) * (img_h as usize);
    let end_idx = start_idx + tmpl_len;

    if end_idx > image_gray.len() {
        return f32::MIN;
    }

    for (i, &_tmpl_val) in template_gray.iter().enumerate() {
        let img_val = image_gray[start_idx + i];
        img_sum += img_val;
        sum_img_sq += img_val * img_val;
        count += 1;
    }

    if count == 0 {
        return f32::MIN;
    }

    let img_mean = img_sum / count as f32;
    let img_var = sum_img_sq / count as f32 - img_mean * img_mean;
    let img_std = img_var.max(0.0).sqrt();

    if img_std < 1.0 {
        return f32::MIN;
    }

    // Recalculate with proper means
    let mut ncc = 0.0f32;
    for (i, &tmpl_val) in template_gray.iter().enumerate() {
        let img_val = image_gray[start_idx + i];
        ncc += (tmpl_val - tmpl_mean) * (img_val - img_mean);
    }

    ncc / (count as f32 * tmpl_std * img_std)
}

pub fn build_preview(image: &RgbaImage, fixed_width: u32) -> PreviewImage {
    let width = image.width();
    let height = image.height();
    let scale = (fixed_width as f32) / (width as f32).max(1.0);
    let target_width = fixed_width.max(1);
    let target_height = ((height as f32) * scale).round().max(1.0) as u32;
    let resized = imageops::resize(image, target_width, target_height, FilterType::Triangle);
    PreviewImage {
        width: resized.width(),
        height: resized.height(),
        pixels: resized.into_raw(),
    }
}

/// Column sampling: extract a few columns from the image and average them
fn col_sampling(img: &RgbaImage) -> ColSamples {
    let w = img.width() as usize;
    let h = img.height() as usize;

    if w == 0 || h == 0 {
        return vec![];
    }

    let groups: Vec<Vec<usize>> = vec![
        linspace(20.min(w - 1), w / 4, 3),
        linspace(w / 2, 5 * w / 8, 3),
        linspace(6 * w / 8, 7 * w / 8, 3),
    ];

    let mut result: Vec<Vec<f32>> = vec![vec![0.0; groups.len()]; h];

    for (group_idx, cols) in groups.iter().enumerate() {
        for y in 0..h {
            let mut sum = 0.0f32;
            let mut count = 0;
            for &x in cols {
                if x < w {
                    let pixel = img.get_pixel(x as u32, y as u32);
                    let gray = 0.299 * pixel[0] as f32
                        + 0.587 * pixel[1] as f32
                        + 0.114 * pixel[2] as f32;
                    sum += gray;
                    count += 1;
                }
            }
            result[y][group_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
        }
    }

    result
}

fn linspace(start: usize, end: usize, n: usize) -> Vec<usize> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![start];
    }
    let step = (end.saturating_sub(start)) as f32 / (n - 1) as f32;
    (0..n)
        .map(|i| (start as f32 + i as f32 * step).round() as usize)
        .collect()
}

fn predict_offset_iter(max: i32, predict: i32) -> Vec<i32> {
    let p = predict.clamp(0, max);
    let mut result = vec![p];

    for delta in 1..=max {
        if p + delta <= max {
            result.push(p + delta);
        }
        if p - delta >= 0 {
            result.push(p - delta);
        }
    }

    result
}

fn diff_overlap(
    cols1: &ColSamples,
    cols2: &ColSamples,
    predict: i32,
    approx_diff: f32,
    min_overlap: u32,
) -> (i32, f32) {
    let h1 = cols1.len() as i32;
    let h2 = cols2.len() as i32;

    if h1 == 0 || h2 == 0 {
        return (0, f32::MAX);
    }

    let max_offset = (h1 - min_overlap as i32).max(0);
    let mut best = (0i32, f32::MAX);
    let mut approach_count = 0;

    for offset in predict_offset_iter(max_offset, predict) {
        let diff = compute_col_diff(cols1, cols2, offset);

        if diff < best.1 {
            best = (offset, diff);
        }

        if best.1 < approx_diff {
            approach_count += 1;
            if approach_count > 10 {
                return best;
            }
            if diff < approx_diff / 4.0 {
                return best;
            }
        }
    }

    best
}

fn compute_col_diff(cols1: &ColSamples, cols2: &ColSamples, offset: i32) -> f32 {
    let h1 = cols1.len();
    let h2 = cols2.len();

    if h1 == 0 || h2 == 0 {
        return f32::MAX;
    }

    let num_groups = cols1.get(0).map(|v| v.len()).unwrap_or(0);
    if num_groups == 0 {
        return f32::MAX;
    }

    let mut sum = 0.0f32;
    let mut count = 0usize;

    if offset == 0 {
        let len = h1.min(h2);
        for y in 0..len {
            for g in 0..num_groups {
                let diff = (cols1[y][g] - cols2[y][g]).abs();
                sum += diff;
                count += 1;
            }
        }
    } else if offset > 0 {
        let offset_u = offset as usize;
        let len = (h1 - offset_u).min(h2 - offset_u);
        for i in 0..len {
            let y1 = offset_u + i;
            let y2 = i;
            if y1 < h1 && y2 < h2 {
                for g in 0..num_groups {
                    let diff = (cols1[y1][g] - cols2[y2][g]).abs();
                    sum += diff;
                    count += 1;
                }
            }
        }
    } else {
        let offset_u = (-offset) as usize;
        let len = (h1 - offset_u).min(h2 - offset_u);
        for i in 0..len {
            let y1 = i;
            let y2 = offset_u + i;
            if y1 < h1 && y2 < h2 {
                for g in 0..num_groups {
                    let diff = (cols1[y1][g] - cols2[y2][g]).abs();
                    sum += diff;
                    count += 1;
                }
            }
        }
    }

    if count == 0 {
        return f32::MAX;
    }

    sum / count as f32
}

/// Column sampling with edge detection for transparent backgrounds
fn col_sampling_edge(img: &RgbaImage) -> ColSamples {
    let w = img.width() as usize;
    let h = img.height() as usize;

    if w == 0 || h < 2 {
        return vec![];
    }

    let groups: Vec<Vec<usize>> = vec![
        linspace(20.min(w - 1), w / 4, 3),
        linspace(w / 2, 5 * w / 8, 3),
        linspace(6 * w / 8, 7 * w / 8, 3),
    ];

    let mut result: Vec<Vec<f32>> = vec![vec![0.0; groups.len()]; h];

    for (group_idx, cols) in groups.iter().enumerate() {
        for y in 1..h {
            let mut sum = 0.0f32;
            let mut count = 0;
            for &x in cols {
                if x < w {
                    let curr = img.get_pixel(x as u32, y as u32);
                    let prev = img.get_pixel(x as u32, (y - 1) as u32);

                    let gray_curr = 0.299 * curr[0] as f32
                        + 0.587 * curr[1] as f32
                        + 0.114 * curr[2] as f32;
                    let gray_prev = 0.299 * prev[0] as f32
                        + 0.587 * prev[1] as f32
                        + 0.114 * prev[2] as f32;

                    let edge = (gray_curr - gray_prev).abs();
                    sum += edge;
                    count += 1;
                }
            }
            result[y][group_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
        }
        if h > 1 {
            result[0][group_idx] = result[1][group_idx];
        }
    }

    result
}
