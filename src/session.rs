use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use image::RgbaImage;

use crate::capture::{capture_frame, select_region};
use crate::cli::Args;
use crate::overlay::LayerShellOverlay;
use crate::output::{copy_to_clipboard, save_image};
use crate::region_overlay::RegionOverlay;
use crate::stitch::{build_preview, MatchConfig, StitchOutcome, Stitcher};
use crate::types::{Control, LayerMessage, Region, StitchState, UserCommand};

pub fn run(args: Args) -> Result<()> {
    if !is_wayland_session() {
        bail!("Wayland session required (X11 not supported)");
    }

    let region = select_region().context("slurp selection failed")?;
    log::info!(
        "Capture region: {},{} {}x{}",
        region.x,
        region.y,
        region.w,
        region.h
    );

    let mut region_overlay = if args.no_border {
        None
    } else {
        Some(RegionOverlay::new(region.clone())?)
    };

    let control = Arc::new(Control::new());
    let state = Arc::new(Mutex::new(StitchState::default()));

    let (command_tx, command_rx) = mpsc::channel();

    let mut layer_overlay = if args.no_preview {
        None
    } else {
        Some(LayerShellOverlay::new(command_tx.clone(), region.clone())?)
    };

    let preview_tx = layer_overlay.as_ref().and_then(|o| o.sender());
    let worker = spawn_capture_worker(
        region,
        control.clone(),
        state.clone(),
        preview_tx,
        args.interval,
        args.preview_width,
    );

    let result = run_session(
        control,
        state,
        worker,
        &mut layer_overlay,
        command_rx,
        &args,
    );

    if let Some(ref mut overlay) = region_overlay {
        overlay.stop();
    }
    result
}

fn run_session(
    control: Arc<Control>,
    state: Arc<Mutex<StitchState>>,
    worker: thread::JoinHandle<()>,
    layer_overlay: &mut Option<LayerShellOverlay>,
    command_rx: mpsc::Receiver<UserCommand>,
    args: &Args,
) -> Result<()> {
    let mut paused = false;

    while let Ok(command) = command_rx.recv() {
        match command {
            UserCommand::TogglePause => {
                control.toggle_pause();
                paused = !paused;
                if let Some(ref overlay) = layer_overlay {
                    overlay.send(LayerMessage::Paused(paused));
                }
            }
            UserCommand::Save => {
                match take_snapshot(&state)
                    .and_then(|img| save_image(img, args.output.clone()))
                {
                    Ok(path) => {
                        log::info!("Saved to {}", path.display());
                        control.stop();
                        break;
                    }
                    Err(err) => {
                        log::error!("Save failed: {err}");
                    }
                }
            }
            UserCommand::Copy => match take_snapshot(&state).and_then(copy_to_clipboard) {
                Ok(()) => {
                    log::info!("Copied to clipboard");
                    control.stop();
                    break;
                }
                Err(err) => {
                    log::error!("Copy failed: {err}");
                }
            },
            UserCommand::Cancel => {
                control.stop();
                break;
            }
        }
    }

    control.stop();
    if let Some(ref mut overlay) = layer_overlay {
        overlay.stop();
    }
    let _ = worker.join();
    Ok(())
}

fn take_snapshot(state: &Arc<Mutex<StitchState>>) -> Result<Arc<RgbaImage>> {
    let state = state.lock().expect("state lock");
    state
        .full_image
        .clone()
        .ok_or_else(|| anyhow!("no frames captured yet"))
}

fn is_wayland_session() -> bool {
    if std::env::var_os("WAYLAND_DISPLAY").is_some() {
        return true;
    }
    matches!(
        std::env::var("XDG_SESSION_TYPE").ok().as_deref(),
        Some("wayland")
    )
}

fn spawn_capture_worker(
    region: Region,
    control: Arc<Control>,
    state: Arc<Mutex<StitchState>>,
    preview_tx: Option<mpsc::Sender<LayerMessage>>,
    interval_ms: u64,
    preview_width: u32,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let config = MatchConfig {
            match_width: 240,
            min_overlap_full: 30,
            accept_diff: 12.0,  // sqrt(SSD / pixel_count) threshold
            min_append_full: 20,
        };
        let mut stitcher = Stitcher::new(config);

        while control.is_running() {
            if control.is_paused() {
                update_status(&state, "Paused".to_string(), None, None, None);
                thread::sleep(Duration::from_millis(120));
                continue;
            }

            let start = Instant::now();
            match capture_frame(&region) {
                Ok(frame) => {
                    let outcome = stitcher.push_frame(frame);
                    match outcome {
                        StitchOutcome::FirstFrame => {
                            apply_state_update(
                                &state,
                                &stitcher,
                                "First frame captured".to_string(),
                                preview_tx.as_ref(),
                                preview_width,
                            );
                        }
                        StitchOutcome::Appended { added } => {
                            apply_state_update(
                                &state,
                                &stitcher,
                                format!("Appended {added} px"),
                                preview_tx.as_ref(),
                                preview_width,
                            );
                        }
                        StitchOutcome::NoProgress => {
                            update_status(
                                &state,
                                "No scroll detected".to_string(),
                                Some(&stitcher),
                                None,
                                None,
                            );
                        }
                        StitchOutcome::NoMatch => {
                            update_status(
                                &state,
                                "No overlap match".to_string(),
                                Some(&stitcher),
                                None,
                                None,
                            );
                        }
                    }
                }
                Err(err) => {
                    update_status(
                        &state,
                        "Capture error".to_string(),
                        Some(&stitcher),
                        None,
                        Some(err.to_string()),
                    );
                }
            }

            let elapsed = start.elapsed();
            if elapsed < Duration::from_millis(interval_ms) {
                thread::sleep(Duration::from_millis(interval_ms) - elapsed);
            }
        }
    })
}

fn apply_state_update(
    state: &Arc<Mutex<StitchState>>,
    stitcher: &Stitcher,
    message: String,
    preview_tx: Option<&mpsc::Sender<LayerMessage>>,
    preview_width: u32,
) {
    let preview = stitcher
        .full_image()
        .map(|img| build_preview(img.as_ref(), preview_width));
    if let (Some(tx), Some(preview)) = (preview_tx, preview.as_ref()) {
        let _ = tx.send(LayerMessage::Preview(preview.clone()));
    }
    let mut st = state.lock().expect("state lock");
    st.full_image = stitcher.full_image();
    st.preview = preview;
    st.stats = stitcher.stats();
    st.last_message = message;
    st.last_error = None;
    st.revision = st.revision.wrapping_add(1);
}

fn update_status(
    state: &Arc<Mutex<StitchState>>,
    message: String,
    stitcher: Option<&Stitcher>,
    preview: Option<crate::types::PreviewImage>,
    error: Option<String>,
) {
    let mut st = state.lock().expect("state lock");
    if let Some(stitcher) = stitcher {
        st.full_image = stitcher.full_image();
        st.stats = stitcher.stats();
    }
    if preview.is_some() {
        st.preview = preview;
    }
    st.last_message = message;
    st.last_error = error;
    st.revision = st.revision.wrapping_add(1);
}
