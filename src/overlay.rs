use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState},
    delegate_compositor, delegate_keyboard, delegate_layer, delegate_output, delegate_pointer,
    delegate_registry, delegate_seat, delegate_shm,
    output::{OutputHandler, OutputState},
    registry::{ProvidesRegistryState, RegistryState},
    registry_handlers,
    seat::{
        keyboard::{KeyEvent, KeyboardHandler, Keysym, Modifiers},
        pointer::{PointerEvent, PointerEventKind, PointerHandler},
        Capability, SeatHandler, SeatState,
    },
    shell::{
        wlr_layer::{
            Anchor, KeyboardInteractivity, Layer, LayerShell, LayerShellHandler, LayerSurface,
            LayerSurfaceConfigure,
        },
        WaylandSurface,
    },
    shm::{slot::SlotPool, Shm, ShmHandler},
};
use wayland_client::{
    globals::registry_queue_init,
    protocol::{wl_keyboard, wl_output, wl_pointer, wl_region, wl_seat, wl_shm, wl_surface},
    Connection, QueueHandle,
};

use crate::constants::CONTROL_BAR_HEIGHT;
use crate::types::{LayerMessage, PreviewImage, Region, UserCommand};

mod drawing;
mod placement;

use drawing::{blit_preview_bottom, draw_control_bar};
use placement::{
    compute_layer_margins, find_output_by_id, output_rects_from_state, probe_output_rects,
    select_output_for_region, OutputRect,
};

const CONTROL_BUTTON_COUNT: u32 = 4;
const INITIAL_HEIGHT: u32 = CONTROL_BAR_HEIGHT;
const PREVIEW_GAP: i32 = 8;

pub struct LayerShellOverlay {
    tx: Option<mpsc::Sender<LayerMessage>>,
    handle: Option<thread::JoinHandle<()>>,
}

impl LayerShellOverlay {
    /// Spawns the preview overlay thread and waits for initialization.
    pub fn new(
        command_tx: mpsc::Sender<UserCommand>,
        region: Region,
        preview_width: u32,
    ) -> Result<Self> {
        let (tx, rx) = mpsc::channel();
        let ready = Arc::new(AtomicBool::new(false));
        let ready_clone = ready.clone();
        let handle = thread::spawn(move || {
            if let Err(err) =
                run_layer_shell_overlay(rx, command_tx, ready_clone, region, preview_width)
            {
                log::warn!("layer-shell overlay failed: {err}");
            }
        });
        // Wait briefly for layer-shell to initialize
        thread::sleep(Duration::from_millis(200));
        if !ready.load(Ordering::Relaxed) {
            bail!("layer-shell overlay did not initialize in time");
        }
        Ok(Self {
            tx: Some(tx),
            handle: Some(handle),
        })
    }

    pub fn sender(&self) -> Option<mpsc::Sender<LayerMessage>> {
        self.tx.as_ref().map(|tx| tx.clone())
    }

    /// Sends a message to the overlay thread, ignoring send failures.
    pub fn send(&self, message: LayerMessage) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(message);
        }
    }

    /// Stops the overlay thread and joins it.
    pub fn stop(&mut self) {
        self.tx.take();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

struct LayerPreview {
    registry_state: RegistryState,
    seat_state: SeatState,
    output_state: OutputState,
    shm: Shm,
    pool: SlotPool,
    layer: LayerSurface,
    input_region: wl_region::WlRegion,
    input_size: Option<(u32, u32, u32)>, // (width, height, y)
    width: u32,
    height: u32,
    max_height: u32,
    region: Region,
    configured: bool,
    exit: bool,
    preview: Option<PreviewImage>,
    command_tx: mpsc::Sender<UserCommand>,
    paused: bool,
    keyboard: Option<wl_keyboard::WlKeyboard>,
    keyboard_focus: bool,
    pointer: Option<wl_pointer::WlPointer>,
    hover_button: Option<u32>,
}

impl LayerPreview {
    fn output_rects(&self) -> Vec<OutputRect> {
        output_rects_from_state(&self.output_state)
    }

    fn update_position(&mut self) {
        let output_rects = self.output_rects();
        let (margin_top, margin_left) =
            compute_layer_margins(&self.region, self.width, &output_rects);
        self.layer.set_margin(margin_top, 0, 0, margin_left);
    }

    fn desired_size_from_preview(&self, preview: &PreviewImage) -> (u32, u32) {
        let target_width = preview.width.max(1);
        let max_preview_height = self.max_height.saturating_sub(CONTROL_BAR_HEIGHT);
        let display_preview_height = preview.height.min(max_preview_height);
        let target_height = display_preview_height
            .saturating_add(CONTROL_BAR_HEIGHT)
            .max(1);
        (target_width, target_height)
    }

    fn update_preview(&mut self, qh: &QueueHandle<Self>, preview: PreviewImage) {
        let (target_width, target_height) = self.desired_size_from_preview(&preview);
        let size_changed = self.width != target_width || self.height != target_height;

        self.preview = Some(preview);

        if size_changed {
            self.width = target_width;
            self.height = target_height;
            self.layer.set_size(self.width, self.height);
            self.update_position();
            self.update_input_region();
        }

        self.request_redraw(qh);
    }

    fn set_paused(&mut self, qh: &QueueHandle<Self>, paused: bool) {
        if self.paused == paused {
            return;
        }
        self.paused = paused;
        self.request_redraw(qh);
    }

    fn update_input_region(&mut self) {
        let bar_height = CONTROL_BAR_HEIGHT.min(self.height);
        let width = self.width.max(1);
        if let Some((old_w, old_h, old_y)) = self.input_size.take() {
            self.input_region
                .subtract(0, old_y as i32, old_w as i32, old_h as i32);
        }
        // Control bar is at the bottom
        let bar_y = self.height.saturating_sub(bar_height);
        self.input_region
            .add(0, bar_y as i32, width as i32, bar_height as i32);
        self.input_size = Some((width, bar_height, bar_y));
        self.layer
            .wl_surface()
            .set_input_region(Some(&self.input_region));
    }

    fn request_redraw(&mut self, qh: &QueueHandle<Self>) {
        if self.configured {
            self.draw(qh);
        } else {
            self.layer.commit();
        }
    }

    fn draw(&mut self, _qh: &QueueHandle<Self>) {
        if self.width == 0 || self.height == 0 {
            return;
        }

        // Resize pool if needed
        let needed = (self.width * self.height * 4) as usize;
        if self.pool.len() < needed {
            if let Err(err) = self.pool.resize(needed) {
                log::warn!("layer-shell pool resize failed: {err}");
                return;
            }
        }

        let stride = self.width as i32 * 4;
        let (buffer, canvas) = match self.pool.create_buffer(
            self.width as i32,
            self.height as i32,
            stride,
            wl_shm::Format::Argb8888,
        ) {
            Ok(result) => result,
            Err(err) => {
                log::warn!("layer-shell buffer create failed: {err}");
                return;
            }
        };

        canvas.fill(0);

        // Draw preview - show bottom part if preview is taller than available space
        if let Some(preview) = &self.preview {
            if preview.width != self.width {
                log::warn!(
                    "preview width mismatch: {} vs {}",
                    preview.width,
                    self.width
                );
            } else {
                let available_height = self.height.saturating_sub(CONTROL_BAR_HEIGHT);
                blit_preview_bottom(canvas, self.width, available_height, preview);
            }
        }

        // Draw control bar at the bottom
        let bar_y = self.height.saturating_sub(CONTROL_BAR_HEIGHT);
        draw_control_bar(
            canvas,
            self.width,
            self.height,
            bar_y,
            self.paused,
            self.hover_button,
        );

        self.layer
            .wl_surface()
            .damage_buffer(0, 0, self.width as i32, self.height as i32);
        if let Err(err) = buffer.attach_to(self.layer.wl_surface()) {
            log::warn!("layer-shell buffer attach failed: {err}");
            return;
        }
        self.layer.commit();
    }

    fn handle_command(&mut self, qh: &QueueHandle<Self>, command: UserCommand) {
        if matches!(command, UserCommand::TogglePause) {
            self.paused = !self.paused;
            self.request_redraw(qh);
        }
        let _ = self.command_tx.send(command);
    }

    fn handle_bar_press(&mut self, qh: &QueueHandle<Self>, position: (f64, f64)) {
        if self.width == 0 {
            return;
        }
        if position.1 < 0.0 || position.0 < 0.0 {
            return;
        }

        // Control bar is at the bottom
        let bar_y = self.height.saturating_sub(CONTROL_BAR_HEIGHT) as f64;
        if position.1 < bar_y || position.1 >= self.height as f64 {
            return;
        }

        let x = position.0 as u32;
        let segment = self.width / CONTROL_BUTTON_COUNT;
        let index = if segment == 0 {
            0
        } else {
            (x / segment).min(CONTROL_BUTTON_COUNT - 1)
        };
        let command = match index {
            0 => UserCommand::Save,
            1 => UserCommand::Copy,
            2 => UserCommand::TogglePause,
            _ => UserCommand::Cancel,
        };
        self.handle_command(qh, command);
    }

    fn update_hover(&mut self, qh: &QueueHandle<Self>, position: (f64, f64)) {
        if self.width == 0 {
            return;
        }

        let bar_y = self.height.saturating_sub(CONTROL_BAR_HEIGHT) as f64;
        let new_hover = if position.0 >= 0.0
            && position.1 >= bar_y
            && position.1 < self.height as f64
            && position.0 < self.width as f64
        {
            let x = position.0 as u32;
            let segment = self.width / CONTROL_BUTTON_COUNT;
            let index = if segment == 0 {
                0
            } else {
                (x / segment).min(CONTROL_BUTTON_COUNT - 1)
            };
            Some(index)
        } else {
            None
        };

        if new_hover != self.hover_button {
            self.hover_button = new_hover;
            self.request_redraw(qh);
        }
    }

    fn clear_hover(&mut self, qh: &QueueHandle<Self>) {
        if self.hover_button.is_some() {
            self.hover_button = None;
            self.request_redraw(qh);
        }
    }
}

impl CompositorHandler for LayerPreview {
    fn scale_factor_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _new_factor: i32,
    ) {
    }

    fn transform_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _new_transform: wl_output::Transform,
    ) {
    }

    fn frame(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _time: u32,
    ) {
    }

    fn surface_enter(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _output: &wl_output::WlOutput,
    ) {
    }

    fn surface_leave(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _output: &wl_output::WlOutput,
    ) {
    }
}

impl OutputHandler for LayerPreview {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }

    fn new_output(
        &mut self,
        _conn: &Connection,
        qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
        self.update_position();
        self.request_redraw(qh);
    }

    fn update_output(
        &mut self,
        _conn: &Connection,
        qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
        self.update_position();
        self.request_redraw(qh);
    }

    fn output_destroyed(
        &mut self,
        _conn: &Connection,
        qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
        self.update_position();
        self.request_redraw(qh);
    }
}

impl LayerShellHandler for LayerPreview {
    fn closed(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, _layer: &LayerSurface) {
        self.exit = true;
    }

    fn configure(
        &mut self,
        _conn: &Connection,
        qh: &QueueHandle<Self>,
        _layer: &LayerSurface,
        configure: LayerSurfaceConfigure,
        _serial: u32,
    ) {
        let (desired_w, desired_h) = self
            .preview
            .as_ref()
            .map(|preview| self.desired_size_from_preview(preview))
            .unwrap_or((self.width.max(1), self.height.max(1)));
        if configure.new_size.0 == 0 || configure.new_size.1 == 0 {
            self.width = desired_w;
            self.height = desired_h;
        } else {
            self.width = configure.new_size.0;
            self.height = configure.new_size.1;
        }
        self.configured = true;
        self.layer.set_size(self.width, self.height);
        self.update_position();
        self.update_input_region();
        self.draw(qh);
    }
}

impl SeatHandler for LayerPreview {
    fn seat_state(&mut self) -> &mut SeatState {
        &mut self.seat_state
    }

    fn new_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_seat::WlSeat) {}

    fn new_capability(
        &mut self,
        _conn: &Connection,
        qh: &QueueHandle<Self>,
        seat: wl_seat::WlSeat,
        capability: Capability,
    ) {
        if capability == Capability::Keyboard && self.keyboard.is_none() {
            let keyboard = self
                .seat_state
                .get_keyboard(qh, &seat, None)
                .expect("failed to create keyboard");
            self.keyboard = Some(keyboard);
        }

        if capability == Capability::Pointer && self.pointer.is_none() {
            let pointer = self
                .seat_state
                .get_pointer(qh, &seat)
                .expect("failed to create pointer");
            self.pointer = Some(pointer);
        }
    }

    fn remove_capability(
        &mut self,
        _conn: &Connection,
        _: &QueueHandle<Self>,
        _: wl_seat::WlSeat,
        capability: Capability,
    ) {
        if capability == Capability::Keyboard {
            if let Some(keyboard) = self.keyboard.take() {
                keyboard.release();
            }
        }

        if capability == Capability::Pointer {
            if let Some(pointer) = self.pointer.take() {
                pointer.release();
            }
        }
    }

    fn remove_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_seat::WlSeat) {}
}

impl KeyboardHandler for LayerPreview {
    fn enter(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        surface: &wl_surface::WlSurface,
        _: u32,
        _: &[u32],
        _: &[Keysym],
    ) {
        if self.layer.wl_surface() == surface {
            self.keyboard_focus = true;
        }
    }

    fn leave(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        surface: &wl_surface::WlSurface,
        _: u32,
    ) {
        if self.layer.wl_surface() == surface {
            self.keyboard_focus = false;
        }
    }

    fn press_key(
        &mut self,
        _: &Connection,
        qh: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: u32,
        event: KeyEvent,
    ) {
        if !self.keyboard_focus {
            return;
        }
        if event.keysym == Keysym::Escape {
            self.handle_command(qh, UserCommand::Cancel);
            return;
        }

        if let Some(text) = event.utf8.as_deref() {
            match text {
                "s" | "S" => self.handle_command(qh, UserCommand::Save),
                "c" | "C" => self.handle_command(qh, UserCommand::Copy),
                "q" | "Q" => self.handle_command(qh, UserCommand::Cancel),
                " " => self.handle_command(qh, UserCommand::TogglePause),
                _ => {}
            }
        }
    }

    fn release_key(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: u32,
        _: KeyEvent,
    ) {
    }

    fn update_modifiers(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: u32,
        _: Modifiers,
        _: u32,
    ) {
    }
}

impl PointerHandler for LayerPreview {
    fn pointer_frame(
        &mut self,
        _conn: &Connection,
        qh: &QueueHandle<Self>,
        _pointer: &wl_pointer::WlPointer,
        events: &[PointerEvent],
    ) {
        for event in events {
            if &event.surface != self.layer.wl_surface() {
                continue;
            }
            match event.kind {
                PointerEventKind::Press { .. } => {
                    self.handle_bar_press(qh, event.position);
                }
                PointerEventKind::Motion { .. } => {
                    self.update_hover(qh, event.position);
                }
                PointerEventKind::Leave { .. } => {
                    self.clear_hover(qh);
                }
                _ => {}
            }
        }
    }
}

impl ShmHandler for LayerPreview {
    fn shm_state(&mut self) -> &mut Shm {
        &mut self.shm
    }
}

delegate_compositor!(LayerPreview);
delegate_output!(LayerPreview);
delegate_shm!(LayerPreview);
delegate_seat!(LayerPreview);
delegate_keyboard!(LayerPreview);
delegate_pointer!(LayerPreview);
delegate_layer!(LayerPreview);
delegate_registry!(LayerPreview);

impl ProvidesRegistryState for LayerPreview {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }

    registry_handlers!(OutputState, SeatState);
}

impl wayland_client::Dispatch<wl_region::WlRegion, ()> for LayerPreview {
    fn event(
        _state: &mut Self,
        _proxy: &wl_region::WlRegion,
        _event: wl_region::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

fn run_layer_shell_overlay(
    rx: mpsc::Receiver<LayerMessage>,
    command_tx: mpsc::Sender<UserCommand>,
    ready: Arc<AtomicBool>,
    region: Region,
    preview_width: u32,
) -> Result<()> {
    log::info!("Starting layer-shell overlay thread");
    let output_rects = match probe_output_rects() {
        Ok(rects) => rects,
        Err(err) => {
            log::warn!("output probe failed, using fallback placement: {err}");
            Vec::new()
        }
    };

    let conn = Connection::connect_to_env().context("failed to connect to Wayland")?;
    log::info!("Connected to Wayland");
    let (globals, mut event_queue) =
        registry_queue_init(&conn).context("failed to init Wayland registry")?;
    let qh = event_queue.handle();
    let compositor = CompositorState::bind(&globals, &qh).context("wl_compositor not available")?;
    log::info!("Compositor bound");

    let layer_shell = LayerShell::bind(&globals, &qh).context("layer-shell not available")?;
    log::info!("Layer-shell bound successfully");
    let shm = Shm::bind(&globals, &qh).context("wl_shm not available")?;
    let output_state = OutputState::new(&globals, &qh);

    let initial_preview_width = preview_width.max(1);
    let selected_output_rect = select_output_for_region(&region, &output_rects);
    let selected_output =
        selected_output_rect.and_then(|rect| find_output_by_id(&output_state, rect.id));

    let surface = compositor.create_surface(&qh);
    let layer = layer_shell.create_layer_surface(
        &qh,
        surface,
        Layer::Overlay,
        Some("wayscrollshot-overlay"),
        selected_output.as_ref(),
    );

    let (margin_top, margin_left) =
        compute_layer_margins(&region, initial_preview_width, &output_rects);

    layer.set_anchor(Anchor::TOP | Anchor::LEFT);
    layer.set_keyboard_interactivity(KeyboardInteractivity::OnDemand);
    layer.set_exclusive_zone(0);
    layer.set_margin(margin_top, 0, 0, margin_left);
    layer.set_size(initial_preview_width, INITIAL_HEIGHT);

    let input_region = compositor.wl_compositor().create_region(&qh, ());
    layer.wl_surface().set_input_region(Some(&input_region));
    layer.commit();

    let pool = SlotPool::new((initial_preview_width * INITIAL_HEIGHT * 4) as usize, &shm)
        .context("failed to create shm pool")?;

    let mut preview = LayerPreview {
        registry_state: RegistryState::new(&globals),
        seat_state: SeatState::new(&globals, &qh),
        output_state,
        shm,
        pool,
        layer,
        input_region,
        input_size: None,
        width: initial_preview_width,
        height: INITIAL_HEIGHT,
        max_height: region.h,
        region,
        configured: false,
        exit: false,
        preview: None,
        command_tx,
        paused: false,
        keyboard: None,
        keyboard_focus: false,
        pointer: None,
        hover_button: None,
    };

    // Perform initial roundtrip to ensure layer surface is configured
    event_queue
        .roundtrip(&mut preview)
        .context("initial roundtrip failed")?;

    // Signal that layer-shell is ready
    ready.store(true, Ordering::Relaxed);

    loop {
        // Process all pending messages without blocking
        loop {
            match rx.try_recv() {
                Ok(LayerMessage::Preview(preview_img)) => {
                    preview.update_preview(&qh, preview_img);
                }
                Ok(LayerMessage::Paused(paused)) => {
                    preview.set_paused(&qh, paused);
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    preview.exit = true;
                    break;
                }
            }
        }

        if preview.exit {
            break;
        }

        // Flush outgoing requests
        conn.flush().ok();

        // Process Wayland events with a short timeout
        if let Some(guard) = event_queue.prepare_read() {
            let _ = guard.read();
        }
        event_queue
            .dispatch_pending(&mut preview)
            .context("failed to process Wayland events")?;

        // Small sleep to avoid busy loop
        thread::sleep(Duration::from_millis(4));
    }

    Ok(())
}
