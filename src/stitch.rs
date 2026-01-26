use std::sync::Arc;

use image::imageops::{self, FilterType};
use image::{GenericImage, RgbaImage};

use crate::types::{PreviewImage, StitchStats};

pub struct MatchConfig {
    pub min_overlap: u32,
    pub accept_diff: f32,
    pub min_append: u32,
    pub approx_diff: f32,
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
        let cols = col_sampling(&frame);

        if self.full_image.is_none() {
            let height = frame.height();
            self.stats.frame_count = 1;
            self.stats.total_height = height;
            self.stats.last_append = height;
            self.full_image = Some(Arc::new(frame.clone()));
            self.last_frame = Some(frame);
            self.last_cols = Some(cols);
            return StitchOutcome::FirstFrame;
        }

        let last_cols = match &self.last_cols {
            Some(c) => c,
            None => {
                self.last_frame = Some(frame);
                self.last_cols = Some(cols);
                return StitchOutcome::NoMatch;
            }
        };

        let (offset, diff) = diff_overlap(
            last_cols,
            &cols,
            self.last_offset,
            self.config.approx_diff,
            self.config.min_overlap,
        );

        if diff > self.config.accept_diff {
            self.last_frame = Some(frame);
            self.last_cols = Some(cols);
            return StitchOutcome::NoMatch;
        }

        // offset > 0 means new frame scrolled down (normal case)
        // offset < 0 means new frame scrolled up (reverse scroll)
        // offset == 0 means no scroll
        let new_height = if offset > 0 { offset as u32 } else { 0 };

        if new_height < self.config.min_append {
            self.last_frame = Some(frame);
            self.last_cols = Some(cols);
            self.last_offset = offset;
            return StitchOutcome::NoProgress;
        }

        // Append new content
        let full = self.full_image.as_ref().expect("full image set");
        let mut combined = RgbaImage::new(full.width(), full.height() + new_height);
        combined
            .copy_from(full.as_ref(), 0, 0)
            .expect("copy full image");

        // Copy the new portion from the bottom of the new frame
        let overlap = frame.height().saturating_sub(new_height);
        let slice = imageops::crop_imm(&frame, 0, overlap, frame.width(), new_height).to_image();
        combined
            .copy_from(&slice, 0, full.height())
            .expect("copy slice");

        self.full_image = Some(Arc::new(combined));
        self.last_frame = Some(frame);
        self.last_cols = Some(cols);
        self.last_offset = offset;
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
    let resized = imageops::resize(image, target_width, target_height, FilterType::Triangle);
    PreviewImage {
        width: resized.width(),
        height: resized.height(),
        pixels: resized.into_raw(),
    }
}

/// Column sampling: extract a few columns from the image and average them
/// Returns a (height, num_groups) matrix where each row is the averaged grayscale values
fn col_sampling(img: &RgbaImage) -> ColSamples {
    let w = img.width() as usize;
    let h = img.height() as usize;

    if w == 0 || h == 0 {
        return vec![];
    }

    // Sample 3 groups of columns (like the Python implementation)
    // Group 1: left region (20 to w/4)
    // Group 2: middle region (w/2 to 5w/8)
    // Group 3: right region (6w/8 to 7w/8)
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
                    // Convert to grayscale: 0.299*R + 0.587*G + 0.114*B
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

/// Generate evenly spaced values
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

/// Generate search offsets starting from prediction, expanding outward
/// e.g., predict=50, max=100 -> [50, 51, 49, 52, 48, ...]
fn predict_offset_iter(max: i32, predict: i32) -> Vec<i32> {
    let p = predict.clamp(-max, max);
    let mut result = vec![0i32];

    for delta in 1..=max {
        if p + delta <= max {
            result.push(p + delta);
        }
        if p - delta >= -max {
            result.push(p - delta);
        }
    }

    // Add remaining values
    for i in -max..=max {
        if !result.contains(&i) {
            result.push(i);
        }
    }

    result
}

/// Find the overlap between two column samples
/// Returns (offset, diff) where offset is the scroll distance
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

        // Early termination like the Python implementation
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

/// Compute mean absolute difference between cols1[offset:] and cols2[:-offset] (or vice versa)
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
        // Compare entire columns
        let len = h1.min(h2);
        for y in 0..len {
            for g in 0..num_groups {
                let diff = (cols1[y][g] - cols2[y][g]).abs();
                sum += diff;
                count += 1;
            }
        }
    } else if offset > 0 {
        // cols1[offset:] vs cols2[:-offset]
        // This means: new frame scrolled down by `offset` pixels
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
        // offset < 0: cols1[:offset] vs cols2[-offset:]
        // This means: new frame scrolled up (reverse scroll)
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
