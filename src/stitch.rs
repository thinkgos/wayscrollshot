use std::collections::HashMap;
use std::sync::Arc;

use hora::core::ann_index::ANNIndex;
use hora::index::hnsw_idx::HNSWIndex;
use hora::index::hnsw_params::HNSWParams;
use image::imageops::{self, FilterType};
use image::{GenericImage, GrayImage, RgbaImage};
use imageproc::corners::{corners_fast12, corners_fast9, Corner};
use opencv::calib3d;
use opencv::core::{self, Point2f, Rect, Scalar, Vector, CV_8UC1, NORM_HAMMING};
use opencv::features2d;
use opencv::imgproc;
use opencv::prelude::*;
use rayon::prelude::*;

use crate::cli::Algorithm;
use crate::types::{PreviewImage, StitchStats};

const DESCRIPTOR_PATCH_SIZE: usize = 9;
const DESCRIPTOR_DIM: usize = DESCRIPTOR_PATCH_SIZE & !1;
const CORNER_THRESHOLD: u8 = 64;
const DISTANCE_THRESHOLD: f32 = 0.1;
const MAX_FAST_CORNERS: usize = 1200;
const MIN_FAST_CORNERS: usize = 30;
const STATIC_DIFF_THRESHOLD: u8 = 6;
const DX_TOLERANCE: i32 = 2;
const MIN_OFFSET_FILTER: i32 = 2;
const ORB_MAX_FEATURES: i32 = 1500;
const ORB_MIN_KEYPOINTS: usize = 80;
const ORB_MIN_MATCHES: usize = 24;
const ORB_MIN_INLIERS: usize = 18;
const ORB_MAX_DX: f64 = 12.0;
const ORB_MAX_GEOMETRY_DRIFT: f64 = 0.12;
const ORB_TOP_IGNORE_RATIO: f32 = 0.12;
const ORB_BOTTOM_IGNORE_RATIO: f32 = 0.08;
const ORB_SIDE_IGNORE_RATIO: f32 = 0.04;
const ORB_MIN_IGNORE_PX: u32 = 24;
const TEMPLATE_MIN_HEIGHT: u32 = 48;
const TEMPLATE_FALLBACK_MIN_SCORE: f32 = 0.72;
const TEMPLATE_FALLBACK_MIN_MARGIN: f32 = 0.015;
const TEMPLATE_VERIFY_MAX_DIFF: f32 = 18.0;
const RELAXED_MIN_OVERLAP_FLOOR: u32 = 72;

pub struct MatchConfig {
    pub min_overlap: u32,
    pub accept_diff: f32,
    pub min_append: u32,
    pub approx_diff: f32,
    pub algorithm: Algorithm,
    pub match_width: u32,
}

/// FAST corner index with HNSW
struct FastIndex {
    corners: Vec<(u32, u32)>,
    descriptors: Vec<Vec<f32>>,
    hnsw: HNSWIndex<f32, usize>,
}

impl FastIndex {
    fn new() -> Self {
        let mut params = HNSWParams::<f32>::default();
        params.ef_search = 32;
        params.ef_build = 16;
        let hnsw = HNSWIndex::new(DESCRIPTOR_DIM, &params);
        Self {
            corners: Vec::new(),
            descriptors: Vec::new(),
            hnsw,
        }
    }

    fn build(gray: &GrayImage) -> Self {
        let features = FastFeatures::build(gray, None);
        let mut index = Self::new();

        index.corners = features.corners;
        index.descriptors = features.descriptors;

        for (i, desc) in index.descriptors.iter().enumerate() {
            let _ = index.hnsw.add(desc, i);
        }
        let _ = index.hnsw.build(hora::core::metrics::Metric::Euclidean);

        index
    }
}

struct FastFeatures {
    corners: Vec<(u32, u32)>,
    descriptors: Vec<Vec<f32>>,
}

impl FastFeatures {
    fn build(gray: &GrayImage, prev_gray: Option<&GrayImage>) -> Self {
        let mut index = Self {
            corners: Vec::new(),
            descriptors: Vec::new(),
        };

        // Detect corners using FAST
        let corners_fast12 = corners_fast12(gray, CORNER_THRESHOLD);
        let corners_fast9 = corners_fast9(gray, CORNER_THRESHOLD);
        let mut corners = if corners_fast12.len() > 200 {
            corners_fast12.clone()
        } else {
            corners_fast9.clone()
        };
        let original_corners = corners.clone();

        if let Some(prev) = prev_gray {
            if prev.width() == gray.width() && prev.height() == gray.height() {
                corners = filter_corners_by_diff(&corners, gray, prev);
                if corners.len() < MIN_FAST_CORNERS {
                    corners = original_corners;
                }
            }
        }

        corners = downsample_corners(corners, MAX_FAST_CORNERS);

        // Compute descriptors and build index
        for corner in &corners {
            let desc = compute_descriptor(gray, corner.x, corner.y);
            index.corners.push((corner.x, corner.y));
            index.descriptors.push(desc);
        }

        index
    }
}

/// Compute descriptor for a corner point (row + column features)
fn compute_descriptor(gray: &GrayImage, x: u32, y: u32) -> Vec<f32> {
    let w = gray.width() as i32;
    let h = gray.height() as i32;
    let descriptor_size = DESCRIPTOR_PATCH_SIZE;
    let half_size = descriptor_size as i32 / 2;
    let mut desc = Vec::with_capacity(DESCRIPTOR_DIM);

    // Row features
    for row in 0..(descriptor_size / 2) {
        let yy = y as i32 + (-half_size + row as i32 * 2);
        let mut sum = 0.0;
        let mut count = 0;
        for col in 0..(descriptor_size / 2) {
            let xx = x as i32 + (-half_size + col as i32 * 2);
            if xx >= 0 && xx < w && yy >= 0 && yy < h {
                let pixel = gray.get_pixel(xx as u32, yy as u32)[0] as f32 / 255.0;
                sum += pixel;
                count += 1;
            }
        }
        desc.push(if count > 0 { sum / count as f32 } else { 0.0 });
    }

    // Column features
    for col in 0..(descriptor_size / 2) {
        let xx = x as i32 + (-half_size + col as i32 * 2);
        let mut sum = 0.0;
        let mut count = 0;
        for row in 0..(descriptor_size / 2) {
            let yy = y as i32 + (-half_size + row as i32 * 2);
            if xx >= 0 && xx < w && yy >= 0 && yy < h {
                let pixel = gray.get_pixel(xx as u32, yy as u32)[0] as f32 / 255.0;
                sum += pixel;
                count += 1;
            }
        }
        desc.push(if count > 0 { sum / count as f32 } else { 0.0 });
    }

    desc
}

/// Euclidean distance between two descriptors
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

pub struct Stitcher {
    full_image: Option<Arc<RgbaImage>>,
    last_frame: Option<RgbaImage>,
    last_cols: Option<ColSamples>,
    last_fast_index: Option<FastIndex>,
    last_fast_gray: Option<GrayImage>,
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

type ColSamples = Vec<Vec<f32>>;

struct OrbEstimate {
    dy: f64,
    confidence: f32,
}

impl Stitcher {
    pub fn new(config: MatchConfig) -> Self {
        Self {
            full_image: None,
            last_frame: None,
            last_cols: None,
            last_fast_index: None,
            last_fast_gray: None,
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
        log::debug!("push_frame: {}x{}", frame.width(), frame.height());

        if self.full_image.is_none() {
            let height = frame.height();
            self.stats.frame_count = 1;
            self.stats.total_height = height;
            self.stats.last_append = height;
            self.full_image = Some(Arc::new(frame.clone()));
            self.last_frame = Some(frame.clone());

            match self.config.algorithm {
                Algorithm::Fast => {
                    let gray = prepare_fast_gray(&frame, self.config.match_width);
                    self.last_fast_index = Some(FastIndex::build(&gray));
                    self.last_fast_gray = Some(gray);
                }
                Algorithm::OpenCvOrb => {}
                _ => {
                    self.last_cols = Some(self.compute_cols(&frame));
                }
            }
            return StitchOutcome::FirstFrame;
        }

        let (offset, confidence) = match self.config.algorithm {
            Algorithm::Fast => self.find_offset_fast(&frame),
            Algorithm::Template => self.find_offset_template(&frame),
            Algorithm::OpenCvOrb => self.find_offset_opencv_orb(&frame),
            Algorithm::ColSample | Algorithm::Edge => self.find_offset_colsample(&frame),
        };

        log::debug!("Offset: {}, confidence: {}", offset, confidence);
        let preserve_anchor = matches!(self.config.algorithm, Algorithm::OpenCvOrb);

        if confidence > self.config.accept_diff {
            if !preserve_anchor {
                self.update_last_frame(frame);
            }
            return StitchOutcome::NoMatch;
        }

        let new_height = if offset > 0 { offset as u32 } else { 0 };

        if new_height < self.config.min_append {
            if !preserve_anchor {
                self.update_last_frame(frame);
                self.last_offset = offset;
            }
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
        self.update_last_frame(frame);
        self.last_offset = offset;
        self.stats.frame_count += 1;
        self.stats.total_height = self.full_image.as_ref().unwrap().height();
        self.stats.last_append = new_height;
        StitchOutcome::Appended { added: new_height }
    }

    fn update_last_frame(&mut self, frame: RgbaImage) {
        match self.config.algorithm {
            Algorithm::Fast => {
                let gray = prepare_fast_gray(&frame, self.config.match_width);
                self.last_fast_index = Some(FastIndex::build(&gray));
                self.last_fast_gray = Some(gray);
            }
            Algorithm::OpenCvOrb => {}
            _ => {
                self.last_cols = Some(self.compute_cols(&frame));
            }
        }
        self.last_frame = Some(frame);
    }

    fn compute_cols(&self, frame: &RgbaImage) -> ColSamples {
        match self.config.algorithm {
            Algorithm::Edge => col_sampling_edge(frame),
            _ => col_sampling(frame),
        }
    }

    /// FAST corner + HNSW matching (from snow-shot)
    fn find_offset_fast(&self, frame: &RgbaImage) -> (i32, f32) {
        let prev_index = match &self.last_fast_index {
            Some(idx) => idx,
            None => return (0, f32::MAX),
        };

        if prev_index.corners.is_empty() {
            return (0, f32::MAX);
        }

        let gray = prepare_fast_gray(frame, self.config.match_width);
        let curr_features = FastFeatures::build(&gray, self.last_fast_gray.as_ref());

        if curr_features.corners.is_empty() {
            return (0, f32::MAX);
        }

        // Match features using HNSW
        let offsets: Vec<i32> = curr_features
            .descriptors
            .par_iter()
            .enumerate()
            .filter_map(|(i, desc)| {
                let search_result = prev_index.hnsw.search(desc, 1);
                if search_result.is_empty() {
                    return None;
                }
                let idx = search_result[0];
                let dist = euclidean_distance(&prev_index.descriptors[idx], desc);

                if dist > DISTANCE_THRESHOLD {
                    return None;
                }

                // Calculate Y offset (vertical scroll)
                let (prev_x, prev_y) = prev_index.corners[idx];
                let (curr_x, curr_y) = curr_features.corners[i];
                if (curr_x as i32 - prev_x as i32).abs() > DX_TOLERANCE {
                    return None;
                }
                let dy = curr_y as i32 - prev_y as i32;

                // For vertical scrolling, we expect negative dy (content moves up)
                let offset = -dy;
                if offset < MIN_OFFSET_FILTER {
                    return None;
                }
                Some(offset)
            })
            .collect();

        if offsets.is_empty() {
            log::debug!("FAST: no valid offsets found");
            return (0, f32::MAX);
        }

        // Frequency voting: find most common offset
        let mut counts: HashMap<i32, i32> = HashMap::new();
        for &offset in &offsets {
            *counts.entry(offset).or_insert(0) += 1;
        }

        let mut sorted: Vec<_> = counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));

        let (best_offset, best_count) = sorted[0];
        let second_count = sorted.get(1).map(|(_, c)| *c).unwrap_or(0);

        log::debug!(
            "FAST: corners={}, offsets={}, best_offset={}, best_count={}, second_count={}",
            curr_features.corners.len(),
            offsets.len(),
            best_offset,
            best_count,
            second_count
        );

        // Confidence checks
        let min_matches = (curr_features.corners.len() as i32 / 10).max(3);
        if best_count < min_matches {
            log::debug!(
                "FAST: best_count {} < min_matches {}",
                best_count,
                min_matches
            );
            return (0, f32::MAX);
        }

        // Avoid ambiguity
        if best_count < second_count * 2 {
            log::debug!("FAST: ambiguous result");
            return (0, f32::MAX);
        }

        // Convert count to confidence (lower is better for our interface)
        let confidence = 1.0 - (best_count as f32 / offsets.len() as f32);

        (best_offset, confidence * 10.0)
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

        let skip_top = (h as f32 * 0.05) as u32;
        let template_height = (h as f32 * 0.20) as u32;
        let template = imageops::crop_imm(frame, 0, skip_top, w as u32, template_height).to_image();
        let template_gray = to_grayscale_vec(&template);

        let prev_gray = to_grayscale_vec(prev);

        let search_start = skip_top as i32;
        let search_end = h - template_height as i32;

        if search_end <= search_start {
            return (0, f32::MAX);
        }

        let mut best_offset = 0i32;
        let mut best_score = f32::MIN;

        let predict = self.last_offset.clamp(0, search_end - search_start);
        let offsets = predict_offset_iter(search_end - search_start, predict);

        for offset in offsets {
            let search_y = search_start + offset;
            if search_y < 0 || search_y + template_height as i32 > h {
                continue;
            }

            let score = ncc_score(&prev_gray, &template_gray, search_y as u32, w as u32);

            if score > best_score {
                best_score = score;
                best_offset = offset;
            }

            if best_score > 0.95 {
                break;
            }
        }

        let diff = 1.0 - best_score.max(0.0);
        (best_offset, diff * 10.0)
    }

    fn find_offset_opencv_orb(&self, frame: &RgbaImage) -> (i32, f32) {
        let prev = match &self.last_frame {
            Some(f) => f,
            None => return (0, f32::MAX),
        };

        match estimate_orb_offset(prev, frame, self.config.min_overlap) {
            Ok(Some(estimate)) => (estimate.dy.round() as i32, estimate.confidence),
            Ok(None) => {
                if let Some((offset, confidence)) = self.find_offset_opencv_relaxed(prev, frame) {
                    return (offset, confidence);
                }
                self.find_offset_template_fallback(prev, frame)
            }
            Err(err) => {
                log::debug!("OpenCV ORB match failed: {err}");
                if let Some((offset, confidence)) = self.find_offset_opencv_relaxed(prev, frame) {
                    return (offset, confidence);
                }
                self.find_offset_template_fallback(prev, frame)
            }
        }
    }

    fn find_offset_opencv_relaxed(&self, prev: &RgbaImage, frame: &RgbaImage) -> Option<(i32, f32)> {
        if self.config.min_overlap <= RELAXED_MIN_OVERLAP_FLOOR {
            return None;
        }

        let relaxed_overlap = self
            .config
            .min_overlap
            .saturating_sub(40)
            .max(RELAXED_MIN_OVERLAP_FLOOR);

        match estimate_orb_offset(prev, frame, relaxed_overlap) {
            Ok(Some(estimate)) => {
                let confidence = estimate.confidence + 0.45;
                Some((estimate.dy.round() as i32, confidence))
            }
            Ok(None) => None,
            Err(err) => {
                log::debug!("Relaxed OpenCV ORB match failed: {err}");
                None
            }
        }
    }

    fn find_offset_template_fallback(&self, prev: &RgbaImage, frame: &RgbaImage) -> (i32, f32) {
        let Some((offset, confidence)) =
            find_offset_template_content(prev, frame, self.last_offset, self.config.min_overlap)
        else {
            return self.find_offset_template(frame);
        };
        (offset, confidence)
    }

    pub fn full_image(&self) -> Option<Arc<RgbaImage>> {
        self.full_image.clone()
    }

    pub fn stats(&self) -> StitchStats {
        self.stats.clone()
    }
}

fn prepare_fast_gray(img: &RgbaImage, target_width: u32) -> GrayImage {
    let gray = rgba_to_gray(img);
    let width = gray.width();
    if target_width == 0 || width <= target_width {
        return gray;
    }
    let target_width = target_width.max(1).min(width);
    let height = gray.height();
    imageops::resize(&gray, target_width, height, FilterType::Nearest)
}

fn downsample_corners(corners: Vec<Corner>, max_corners: usize) -> Vec<Corner> {
    if corners.len() <= max_corners {
        return corners;
    }
    let step = corners.len() / max_corners + 1;
    corners.into_iter().step_by(step).collect()
}

fn filter_corners_by_diff(
    corners: &[Corner],
    gray: &GrayImage,
    prev_gray: &GrayImage,
) -> Vec<Corner> {
    corners
        .iter()
        .filter_map(|corner| {
            if corner.x >= gray.width() || corner.y >= gray.height() {
                return None;
            }
            let curr = gray.get_pixel(corner.x, corner.y)[0];
            let prev = prev_gray.get_pixel(corner.x, corner.y)[0];
            if curr.abs_diff(prev) >= STATIC_DIFF_THRESHOLD {
                Some(*corner)
            } else {
                None
            }
        })
        .collect()
}

fn rgba_to_gray(img: &RgbaImage) -> GrayImage {
    GrayImage::from_fn(img.width(), img.height(), |x, y| {
        let p = img.get_pixel(x, y);
        let gray = (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32) as u8;
        image::Luma([gray])
    })
}

fn estimate_orb_offset(
    prev: &RgbaImage,
    frame: &RgbaImage,
    min_overlap: u32,
) -> opencv::Result<Option<OrbEstimate>> {
    if prev.width() != frame.width() || prev.height() != frame.height() {
        return Ok(None);
    }
    if prev.width() < 80 || prev.height() < 120 {
        return Ok(None);
    }

    let prev_gray = rgba_to_gray(prev);
    let frame_gray = rgba_to_gray(frame);
    let prev_mat = gray_to_mat(&prev_gray)?;
    let frame_mat = gray_to_mat(&frame_gray)?;
    let mask = build_feature_mask(prev.width(), prev.height())?;

    let mut orb = features2d::ORB::create_def()?;
    orb.set_max_features(ORB_MAX_FEATURES)?;

    let mut prev_keypoints = Vector::<core::KeyPoint>::new();
    let mut prev_descriptors = core::Mat::default();
    orb.detect_and_compute_def(&prev_mat, &mask, &mut prev_keypoints, &mut prev_descriptors)?;

    let mut curr_keypoints = Vector::<core::KeyPoint>::new();
    let mut curr_descriptors = core::Mat::default();
    orb.detect_and_compute_def(
        &frame_mat,
        &mask,
        &mut curr_keypoints,
        &mut curr_descriptors,
    )?;

    if prev_keypoints.len() < ORB_MIN_KEYPOINTS
        || curr_keypoints.len() < ORB_MIN_KEYPOINTS
        || prev_descriptors.empty()
        || curr_descriptors.empty()
    {
        return Ok(None);
    }

    let matcher = features2d::BFMatcher::create(NORM_HAMMING, false)?;
    let mut matches = Vector::<Vector<core::DMatch>>::new();
    matcher.knn_train_match_def(&curr_descriptors, &prev_descriptors, &mut matches, 2)?;

    let mut curr_points = Vector::<Point2f>::new();
    let mut prev_points = Vector::<Point2f>::new();
    let mut raw_matches = 0usize;

    for pair in matches.iter() {
        if pair.len() < 2 {
            continue;
        }

        let best = pair.get(0)?;
        let second = pair.get(1)?;

        if best.distance >= second.distance * 0.78 {
            continue;
        }

        let curr_pt = curr_keypoints.get(best.query_idx as usize)?.pt();
        let prev_pt = prev_keypoints.get(best.train_idx as usize)?.pt();
        let dx = (prev_pt.x - curr_pt.x) as f64;
        let dy = (prev_pt.y - curr_pt.y) as f64;

        if dy <= 1.0 || dx.abs() > ORB_MAX_DX * 2.0 {
            continue;
        }

        curr_points.push(curr_pt);
        prev_points.push(prev_pt);
        raw_matches += 1;
    }

    if raw_matches < ORB_MIN_MATCHES {
        return Ok(None);
    }

    let mut inliers = core::Mat::default();
    let affine = calib3d::estimate_affine_partial_2d(
        &curr_points,
        &prev_points,
        &mut inliers,
        calib3d::RANSAC,
        3.0,
        2000,
        0.99,
        10,
    )?;

    if affine.empty() {
        return Ok(None);
    }

    let a = *affine.at_2d::<f64>(0, 0)?;
    let b = *affine.at_2d::<f64>(0, 1)?;
    let c = *affine.at_2d::<f64>(1, 0)?;
    let d = *affine.at_2d::<f64>(1, 1)?;
    let tx = *affine.at_2d::<f64>(0, 2)?;
    let ty = *affine.at_2d::<f64>(1, 2)?;

    let scale = ((a * a + c * c).sqrt() + (b * b + d * d).sqrt()) * 0.5;
    let geom_drift = (a - 1.0).abs() + (d - 1.0).abs() + b.abs() + c.abs();

    if tx.abs() > ORB_MAX_DX
        || ty <= 1.0
        || ty >= (prev.height() - min_overlap) as f64
        || (scale - 1.0).abs() > ORB_MAX_GEOMETRY_DRIFT
        || geom_drift > ORB_MAX_GEOMETRY_DRIFT
    {
        return Ok(None);
    }

    let mut inlier_count = 0usize;
    for row in 0..inliers.rows() {
        if *inliers.at_2d::<u8>(row, 0)? != 0 {
            inlier_count += 1;
        }
    }

    if inlier_count < ORB_MIN_INLIERS {
        return Ok(None);
    }

    let inlier_ratio = inlier_count as f32 / raw_matches as f32;
    let confidence = (1.0 - inlier_ratio) * 3.5
        + (tx.abs() as f32 / ORB_MAX_DX as f32)
        + (geom_drift as f32 * 6.0);

    Ok(Some(OrbEstimate { dy: ty, confidence }))
}

fn gray_to_mat(gray: &GrayImage) -> opencv::Result<core::Mat> {
    let rows = gray.height() as i32;
    let cols = gray.width() as i32;
    let mut mat = core::Mat::new_rows_cols_with_default(rows, cols, CV_8UC1, Scalar::all(0.0))?;

    for y in 0..rows {
        for x in 0..cols {
            *mat.at_2d_mut::<u8>(y, x)? = gray.get_pixel(x as u32, y as u32)[0];
        }
    }

    Ok(mat)
}

fn build_feature_mask(width: u32, height: u32) -> opencv::Result<core::Mat> {
    let mut mask = core::Mat::new_rows_cols_with_default(
        height as i32,
        width as i32,
        CV_8UC1,
        Scalar::all(0.0),
    )?;

    let side = ((width as f32 * ORB_SIDE_IGNORE_RATIO) as u32).max(ORB_MIN_IGNORE_PX);
    let top = ((height as f32 * ORB_TOP_IGNORE_RATIO) as u32).max(ORB_MIN_IGNORE_PX);
    let bottom = ((height as f32 * ORB_BOTTOM_IGNORE_RATIO) as u32).max(ORB_MIN_IGNORE_PX);
    let roi_x = side.min(width.saturating_sub(1));
    let roi_y = top.min(height.saturating_sub(1));
    let roi_w = width.saturating_sub(roi_x.saturating_mul(2)).max(1);
    let roi_h = height.saturating_sub(roi_y).saturating_sub(bottom).max(1);
    let rect = Rect::new(roi_x as i32, roi_y as i32, roi_w as i32, roi_h as i32);

    imgproc::rectangle(&mut mask, rect, Scalar::all(255.0), -1, imgproc::LINE_8, 0)?;

    Ok(mask)
}

fn content_roi(width: u32, height: u32) -> (u32, u32, u32, u32) {
    let side = ((width as f32 * ORB_SIDE_IGNORE_RATIO) as u32).max(ORB_MIN_IGNORE_PX);
    let top = ((height as f32 * ORB_TOP_IGNORE_RATIO) as u32).max(ORB_MIN_IGNORE_PX);
    let bottom = ((height as f32 * ORB_BOTTOM_IGNORE_RATIO) as u32).max(ORB_MIN_IGNORE_PX);
    let x = side.min(width.saturating_sub(1));
    let y = top.min(height.saturating_sub(1));
    let roi_w = width.saturating_sub(x.saturating_mul(2)).max(1);
    let roi_h = height.saturating_sub(y).saturating_sub(bottom).max(1);
    (x, y, roi_w, roi_h)
}

fn to_grayscale_vec(img: &RgbaImage) -> Vec<f32> {
    img.pixels()
        .map(|p| 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32)
        .collect()
}

fn find_offset_template_content(
    prev: &RgbaImage,
    frame: &RgbaImage,
    predict: i32,
    min_overlap: u32,
) -> Option<(i32, f32)> {
    if prev.width() != frame.width() || prev.height() != frame.height() {
        return None;
    }

    let width = prev.width();
    let height = prev.height();
    let (roi_x, roi_y, roi_w, roi_h) = content_roi(width, height);
    if roi_h < TEMPLATE_MIN_HEIGHT * 2 || roi_w < 40 {
        return None;
    }

    let template_h = (roi_h / 3).max(TEMPLATE_MIN_HEIGHT).min(roi_h - 1);
    let search_start = roi_y as i32;
    let search_end = (roi_y + roi_h - template_h) as i32;
    if search_end <= search_start {
        return None;
    }

    let prev_gray = to_grayscale_vec(prev);
    let frame_gray = to_grayscale_vec(frame);
    let frame_template_y = roi_y;

    let mut best_offset = 0i32;
    let mut best_score = f32::MIN;
    let mut second_score = f32::MIN;

    let max_offset = (height as i32 - min_overlap as i32).max(0);
    let predict = predict.clamp(0, max_offset.min(search_end - search_start));
    for offset in predict_offset_iter(search_end - search_start, predict) {
        let search_y = search_start + offset;
        if search_y < 0 || search_y + template_h as i32 > height as i32 {
            continue;
        }

        let score = ncc_score_region(
            &prev_gray,
            &frame_gray,
            width,
            roi_x,
            roi_w,
            search_y as u32,
            frame_template_y,
            template_h,
        );

        if score > best_score {
            second_score = best_score;
            best_score = score;
            best_offset = offset;
        } else if score > second_score {
            second_score = score;
        }
    }

    if best_score < TEMPLATE_FALLBACK_MIN_SCORE {
        return None;
    }

    if second_score.is_finite() && best_score - second_score < TEMPLATE_FALLBACK_MIN_MARGIN {
        return None;
    }

    let verification = overlap_mean_abs_diff(
        &prev_gray,
        &frame_gray,
        width,
        roi_x,
        roi_w,
        best_offset as u32,
        height.saturating_sub(best_offset as u32),
    );

    if !verification.is_finite() || verification > TEMPLATE_VERIFY_MAX_DIFF {
        return None;
    }

    let confidence = (1.0 - best_score.max(0.0)) * 8.0 + verification / 10.0;
    Some((best_offset, confidence))
}

fn ncc_score(image_gray: &[f32], template_gray: &[f32], y_offset: u32, width: u32) -> f32 {
    let tmpl_len = template_gray.len();
    if tmpl_len == 0 {
        return f32::MIN;
    }

    let tmpl_mean: f32 = template_gray.iter().sum::<f32>() / tmpl_len as f32;
    let tmpl_var: f32 = template_gray
        .iter()
        .map(|&v| (v - tmpl_mean).powi(2))
        .sum::<f32>()
        / tmpl_len as f32;
    let tmpl_std = tmpl_var.sqrt();

    if tmpl_std < 1.0 {
        return f32::MIN;
    }

    let start_idx = (y_offset as usize) * (width as usize);
    let end_idx = start_idx + tmpl_len;

    if end_idx > image_gray.len() {
        return f32::MIN;
    }

    let mut img_sum = 0.0f32;
    let mut sum_img_sq = 0.0f32;

    for i in 0..tmpl_len {
        let img_val = image_gray[start_idx + i];
        img_sum += img_val;
        sum_img_sq += img_val * img_val;
    }

    let img_mean = img_sum / tmpl_len as f32;
    let img_var = sum_img_sq / tmpl_len as f32 - img_mean * img_mean;
    let img_std = img_var.max(0.0).sqrt();

    if img_std < 1.0 {
        return f32::MIN;
    }

    let mut ncc = 0.0f32;
    for (i, &tmpl_val) in template_gray.iter().enumerate() {
        let img_val = image_gray[start_idx + i];
        ncc += (tmpl_val - tmpl_mean) * (img_val - img_mean);
    }

    ncc / (tmpl_len as f32 * tmpl_std * img_std)
}

fn ncc_score_region(
    image_gray: &[f32],
    template_gray: &[f32],
    width: u32,
    roi_x: u32,
    roi_w: u32,
    image_y: u32,
    template_y: u32,
    template_h: u32,
) -> f32 {
    if roi_w == 0 || template_h == 0 || width == 0 {
        return f32::MIN;
    }

    let mut tmpl_sum = 0.0f32;
    let mut img_sum = 0.0f32;
    let mut count = 0usize;

    for row in 0..template_h {
        let tmpl_base = ((template_y + row) * width + roi_x) as usize;
        let img_base = ((image_y + row) * width + roi_x) as usize;
        for col in 0..roi_w as usize {
            tmpl_sum += template_gray[tmpl_base + col];
            img_sum += image_gray[img_base + col];
            count += 1;
        }
    }

    if count == 0 {
        return f32::MIN;
    }

    let tmpl_mean = tmpl_sum / count as f32;
    let img_mean = img_sum / count as f32;
    let mut numerator = 0.0f32;
    let mut tmpl_var = 0.0f32;
    let mut img_var = 0.0f32;

    for row in 0..template_h {
        let tmpl_base = ((template_y + row) * width + roi_x) as usize;
        let img_base = ((image_y + row) * width + roi_x) as usize;
        for col in 0..roi_w as usize {
            let tmpl = template_gray[tmpl_base + col] - tmpl_mean;
            let img = image_gray[img_base + col] - img_mean;
            numerator += tmpl * img;
            tmpl_var += tmpl * tmpl;
            img_var += img * img;
        }
    }

    if tmpl_var <= 1.0 || img_var <= 1.0 {
        return f32::MIN;
    }

    numerator / (tmpl_var.sqrt() * img_var.sqrt())
}

fn overlap_mean_abs_diff(
    prev_gray: &[f32],
    frame_gray: &[f32],
    width: u32,
    roi_x: u32,
    roi_w: u32,
    offset: u32,
    overlap_h: u32,
) -> f32 {
    if roi_w == 0 || overlap_h == 0 {
        return f32::MAX;
    }

    let sample_h = overlap_h.min(160);
    let start_prev_y = offset + overlap_h.saturating_sub(sample_h);
    let start_frame_y = overlap_h.saturating_sub(sample_h);
    let mut sum = 0.0f32;
    let mut count = 0usize;

    for row in 0..sample_h {
        let prev_base = ((start_prev_y + row) * width + roi_x) as usize;
        let frame_base = ((start_frame_y + row) * width + roi_x) as usize;
        for col in 0..roi_w as usize {
            sum += (prev_gray[prev_base + col] - frame_gray[frame_base + col]).abs();
            count += 1;
        }
    }

    if count == 0 {
        return f32::MAX;
    }

    sum / count as f32
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
                    let gray =
                        0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32;
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

                    let gray_curr =
                        0.299 * curr[0] as f32 + 0.587 * curr[1] as f32 + 0.114 * curr[2] as f32;
                    let gray_prev =
                        0.299 * prev[0] as f32 + 0.587 * prev[1] as f32 + 0.114 * prev[2] as f32;

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

#[cfg(test)]
mod tests {
    use super::*;
    use image::{imageops, Rgba};

    fn make_scroll_canvas(width: u32, height: u32) -> RgbaImage {
        let mut img = RgbaImage::from_pixel(width, height, Rgba([245, 245, 245, 255]));

        for y in (0..height).step_by(36) {
            let accent = ((y / 3) % 180) as u8;
            for x in 24..width.saturating_sub(24) {
                let stripe = if (x / 7 + y / 11) % 2 == 0 { 220 } else { 180 };
                img.put_pixel(x, y, Rgba([accent, stripe, 80, 255]));
                if y + 1 < height {
                    img.put_pixel(x, y + 1, Rgba([30, 30, 30, 255]));
                }
            }
        }

        for block in 0..10 {
            let y0 = 30 + block * 80;
            let block_h = 34 + (block % 3) * 8;
            let color = [
                ((40u16 + block as u16 * 17) % 200) as u8,
                ((90u16 + block as u16 * 11) % 200) as u8,
                ((140u16 + block as u16 * 19) % 200) as u8,
                255,
            ];
            for y in y0..(y0 + block_h).min(height) {
                for x in 30..width.saturating_sub(30) {
                    if x % (9 + block as u32 % 5) == 0 || y % (7 + block as u32 % 4) == 0 {
                        img.put_pixel(x, y, Rgba(color));
                    }
                }
            }
        }

        for col in [42, 96, 154, 211, 268] {
            if col >= width {
                continue;
            }
            for y in 20..height.saturating_sub(20) {
                if (y / 13) % 3 != 0 {
                    img.put_pixel(col, y, Rgba([20, 20, 20, 255]));
                }
            }
        }

        img
    }

    fn crop_frame(canvas: &RgbaImage, y: u32, height: u32) -> RgbaImage {
        imageops::crop_imm(canvas, 0, y, canvas.width(), height).to_image()
    }

    fn make_line_canvas(width: u32, height: u32) -> RgbaImage {
        let mut img = RgbaImage::from_pixel(width, height, Rgba([250, 250, 250, 255]));
        let mut y = 16u32;
        let mut band = 0u32;

        while y < height.saturating_sub(16) {
            let band_h = 6 + (band % 5) * 4;
            let gray = (40 + ((band * 29) % 160)) as u8;
            for yy in y..(y + band_h).min(height) {
                for x in 28..width.saturating_sub(28) {
                    img.put_pixel(x, yy, Rgba([gray, gray, gray, 255]));
                }
            }
            y += band_h + 9 + (band % 4) * 3;
            band += 1;
        }

        img
    }

    #[test]
    fn opencv_orb_estimates_vertical_offset() {
        let canvas = make_scroll_canvas(320, 1000);
        let first = crop_frame(&canvas, 0, 320);
        let second = crop_frame(&canvas, 84, 320);

        let estimate = estimate_orb_offset(&first, &second, 120)
            .expect("opencv estimate")
            .expect("orb match");

        assert!((estimate.dy - 84.0).abs() <= 4.0, "dy={}", estimate.dy);
        assert!(
            estimate.confidence < 3.5,
            "confidence={}",
            estimate.confidence
        );
    }

    #[test]
    fn opencv_orb_keeps_anchor_after_bad_frame() {
        let canvas = make_scroll_canvas(320, 1000);
        let first = crop_frame(&canvas, 0, 320);
        let shifted = crop_frame(&canvas, 96, 320);
        let bad = RgbaImage::from_pixel(320, 320, Rgba([255, 255, 255, 255]));

        let mut stitcher = Stitcher::new(MatchConfig {
            min_overlap: 120,
            accept_diff: 3.5,
            min_append: 10,
            approx_diff: 1.0,
            algorithm: Algorithm::OpenCvOrb,
            match_width: 320,
        });

        assert!(matches!(
            stitcher.push_frame(first),
            StitchOutcome::FirstFrame
        ));
        assert!(matches!(stitcher.push_frame(bad), StitchOutcome::NoMatch));

        match stitcher.push_frame(shifted) {
            StitchOutcome::Appended { added } => assert!(added >= 92 && added <= 100, "{added}"),
            _ => panic!("expected appended after bad frame"),
        }
    }

    #[test]
    fn opencv_orb_falls_back_to_template_on_low_feature_frames() {
        let canvas = make_line_canvas(320, 1000);
        let first = crop_frame(&canvas, 0, 320);
        let second = crop_frame(&canvas, 72, 320);

        let mut stitcher = Stitcher::new(MatchConfig {
            min_overlap: 120,
            accept_diff: 3.5,
            min_append: 10,
            approx_diff: 1.0,
            algorithm: Algorithm::OpenCvOrb,
            match_width: 320,
        });

        assert!(matches!(
            stitcher.push_frame(first),
            StitchOutcome::FirstFrame
        ));

        match stitcher.push_frame(second) {
            StitchOutcome::Appended { added } => assert!(added >= 68 && added <= 76, "{added}"),
            _ => panic!("expected appended via template fallback"),
        }
    }

    #[test]
    fn opencv_orb_relaxed_overlap_handles_large_jump() {
        let canvas = make_scroll_canvas(320, 1200);
        let first = crop_frame(&canvas, 0, 320);
        let second = crop_frame(&canvas, 208, 320);

        let mut stitcher = Stitcher::new(MatchConfig {
            min_overlap: 120,
            accept_diff: 3.5,
            min_append: 10,
            approx_diff: 1.0,
            algorithm: Algorithm::OpenCvOrb,
            match_width: 320,
        });

        assert!(matches!(
            stitcher.push_frame(first),
            StitchOutcome::FirstFrame
        ));

        match stitcher.push_frame(second) {
            StitchOutcome::Appended { added } => assert!(added >= 202 && added <= 214, "{added}"),
            _ => panic!("expected appended via relaxed overlap"),
        }
    }
}
