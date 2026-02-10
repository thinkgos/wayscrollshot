use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use image::RgbaImage;

/// User-selected capture region in global compositor coordinates.
#[derive(Clone, Debug)]
pub struct Region {
    pub raw: String,
    pub x: i32,
    pub y: i32,
    pub w: u32,
    pub h: u32,
}

/// Shared run-state flags for capture worker and command loop.
#[derive(Default)]
pub struct Control {
    running: AtomicBool,
    paused: AtomicBool,
}

impl Control {
    /// Creates a running, unpaused control state.
    pub fn new() -> Self {
        Self {
            running: AtomicBool::new(true),
            paused: AtomicBool::new(false),
        }
    }

    /// Requests worker shutdown.
    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
    }

    /// Toggles pause state.
    pub fn toggle_pause(&self) {
        let current = self.paused.load(Ordering::Relaxed);
        self.paused.store(!current, Ordering::Relaxed);
    }

    /// Returns whether the worker should continue running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// Returns whether capture is currently paused.
    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Relaxed)
    }
}

#[derive(Clone, Debug)]
pub struct PreviewImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

#[derive(Clone, Debug, Default)]
pub struct StitchStats {
    pub frame_count: u32,
    pub total_height: u32,
    pub last_append: u32,
}

#[derive(Default)]
pub struct StitchState {
    pub full_image: Option<Arc<RgbaImage>>,
    pub preview: Option<PreviewImage>,
    pub stats: StitchStats,
    pub revision: u64,
    pub last_message: String,
    pub last_error: Option<String>,
}

#[derive(Clone, Debug)]
pub enum UserCommand {
    Save,
    Copy,
    Cancel,
    TogglePause,
}

#[derive(Clone, Debug)]
pub enum LayerMessage {
    Preview(PreviewImage),
    Paused(bool),
}
