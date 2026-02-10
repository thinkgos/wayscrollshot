use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState},
    delegate_compositor, delegate_keyboard, delegate_layer, delegate_output, delegate_pointer,
    delegate_registry, delegate_seat, delegate_shm,
    output::{OutputData, OutputHandler, OutputState},
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
use tiny_skia::{Color, FillRule, Paint, PathBuilder, Pixmap, Transform};
use wayland_client::{
    globals::registry_queue_init,
    protocol::{wl_keyboard, wl_output, wl_pointer, wl_region, wl_seat, wl_shm, wl_surface},
    Connection, Proxy, QueueHandle,
};

use crate::constants::CONTROL_BAR_HEIGHT;
use crate::types::{LayerMessage, PreviewImage, Region, UserCommand};

const CONTROL_BUTTON_COUNT: u32 = 4;
const INITIAL_HEIGHT: u32 = CONTROL_BAR_HEIGHT;
const PREVIEW_GAP: i32 = 8;
const COLOR_SAVE: [u8; 4] = [39, 174, 96, 230];
const COLOR_COPY: [u8; 4] = [52, 152, 219, 230];
const COLOR_PAUSE: [u8; 4] = [241, 196, 15, 230];
const COLOR_RESUME: [u8; 4] = [26, 188, 156, 230];
const COLOR_CANCEL: [u8; 4] = [231, 76, 60, 230];
const COLOR_LABEL: [u8; 4] = [255, 255, 255, 255];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct OutputRect {
    id: u32,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
}

impl OutputRect {
    fn right(self) -> i32 {
        self.x.saturating_add(self.width)
    }

    fn bottom(self) -> i32 {
        self.y.saturating_add(self.height)
    }

    fn contains_point(self, x: i64, y: i64) -> bool {
        let left = i64::from(self.x);
        let top = i64::from(self.y);
        let right = i64::from(self.right());
        let bottom = i64::from(self.bottom());
        x >= left && x < right && y >= top && y < bottom
    }

    fn distance_squared_to_point(self, x: i64, y: i64) -> i128 {
        let left = i64::from(self.x);
        let top = i64::from(self.y);
        let right = i64::from(self.right());
        let bottom = i64::from(self.bottom());

        let dx = if x < left {
            left - x
        } else if x >= right {
            x - right + 1
        } else {
            0
        };

        let dy = if y < top {
            top - y
        } else if y >= bottom {
            y - bottom + 1
        } else {
            0
        };

        let dx = i128::from(dx);
        let dy = i128::from(dy);
        dx * dx + dy * dy
    }
}

fn i64_to_i32_saturating(value: i64) -> i32 {
    value.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

fn output_rect_from_info(info: &smithay_client_toolkit::output::OutputInfo) -> Option<OutputRect> {
    let (fallback_width, fallback_height) = info
        .modes
        .iter()
        .find(|mode| mode.current)
        .or_else(|| info.modes.first())
        .map(|mode| mode.dimensions)
        .unwrap_or((0, 0));

    let (width, height) = info
        .logical_size
        .unwrap_or((fallback_width, fallback_height));
    if width <= 0 || height <= 0 {
        return None;
    }

    let (x, y) = info.logical_position.unwrap_or(info.location);

    Some(OutputRect {
        id: info.id,
        x,
        y,
        width,
        height,
    })
}

fn output_id(output: &wl_output::WlOutput) -> Option<u32> {
    output
        .data::<OutputData>()
        .map(|data| data.with_output_info(|info| info.id))
}

fn find_output_by_id(output_state: &OutputState, id: u32) -> Option<wl_output::WlOutput> {
    output_state
        .outputs()
        .find(|output| output_id(output) == Some(id))
}

fn select_output_for_region(region: &Region, outputs: &[OutputRect]) -> Option<OutputRect> {
    if outputs.is_empty() {
        return None;
    }

    let center_x = i64::from(region.x).saturating_add(i64::from(region.w) / 2);
    let center_y = i64::from(region.y).saturating_add(i64::from(region.h) / 2);

    if let Some(output) = outputs
        .iter()
        .copied()
        .find(|output| output.contains_point(center_x, center_y))
    {
        return Some(output);
    }

    outputs
        .iter()
        .copied()
        .min_by_key(|output| output.distance_squared_to_point(center_x, center_y))
}

fn compute_preview_margin_left(region: &Region, preview_width: u32, outputs: &[OutputRect]) -> i32 {
    let preview_width = preview_width.max(1);
    let preview_width_i64 = i64::from(preview_width);
    let right_candidate = i64::from(region.x)
        .saturating_add(i64::from(region.w))
        .saturating_add(i64::from(PREVIEW_GAP));

    let Some(output) = select_output_for_region(region, outputs) else {
        return i64_to_i32_saturating(right_candidate);
    };

    let output_left = i64::from(output.x);
    let output_right = i64::from(output.right());

    if right_candidate.saturating_add(preview_width_i64) <= output_right {
        return i64_to_i32_saturating(right_candidate);
    }

    let left_candidate = i64::from(region.x)
        .saturating_sub(i64::from(PREVIEW_GAP))
        .saturating_sub(preview_width_i64);

    if left_candidate >= output_left {
        return i64_to_i32_saturating(left_candidate);
    }

    let max_left = output_right.saturating_sub(preview_width_i64);
    let clamped = if max_left < output_left {
        output_left
    } else {
        right_candidate.clamp(output_left, max_left)
    };
    i64_to_i32_saturating(clamped)
}

fn compute_layer_margins(region: &Region, preview_width: u32, outputs: &[OutputRect]) -> (i32, i32) {
    let left_global = compute_preview_margin_left(region, preview_width, outputs);
    if let Some(output) = select_output_for_region(region, outputs) {
        (
            region.y.saturating_sub(output.y),
            left_global.saturating_sub(output.x),
        )
    } else {
        (region.y, left_global)
    }
}

fn output_rects_from_state(output_state: &OutputState) -> Vec<OutputRect> {
    output_state
        .outputs()
        .filter_map(|output| output_state.info(&output))
        .filter_map(|info| output_rect_from_info(&info))
        .collect()
}

struct OutputProbe {
    registry_state: RegistryState,
    output_state: OutputState,
}

impl OutputHandler for OutputProbe {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }

    fn new_output(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
    }

    fn update_output(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
    }

    fn output_destroyed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
    }
}

delegate_output!(OutputProbe);
delegate_registry!(OutputProbe);

impl ProvidesRegistryState for OutputProbe {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }

    registry_handlers!(OutputState);
}

fn probe_output_rects() -> Result<Vec<OutputRect>> {
    let conn = Connection::connect_to_env().context("failed to connect to Wayland for output probe")?;
    let (globals, mut event_queue) =
        registry_queue_init(&conn).context("failed to init Wayland registry for output probe")?;
    let qh = event_queue.handle();

    let mut output_probe = OutputProbe {
        registry_state: RegistryState::new(&globals),
        output_state: OutputState::new(&globals, &qh),
    };

    event_queue
        .roundtrip(&mut output_probe)
        .context("failed to gather output geometry")?;

    Ok(output_rects_from_state(&output_probe.output_state))
}

pub struct LayerShellOverlay {
    tx: Option<mpsc::Sender<LayerMessage>>,
    handle: Option<thread::JoinHandle<()>>,
}

impl LayerShellOverlay {
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

    pub fn send(&self, message: LayerMessage) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(message);
        }
    }

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
    let selected_output = selected_output_rect
        .and_then(|rect| find_output_by_id(&output_state, rect.id));

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

fn draw_control_bar(
    canvas: &mut [u8],
    width: u32,
    height: u32,
    bar_y: u32,
    paused: bool,
    hover_button: Option<u32>,
) {
    let bar_height = CONTROL_BAR_HEIGHT.min(height.saturating_sub(bar_y));
    if width == 0 || bar_height == 0 {
        return;
    }

    // Create a pixmap for the control bar
    let mut pixmap = match Pixmap::new(width, bar_height) {
        Some(p) => p,
        None => return,
    };

    let segment = (width / CONTROL_BUTTON_COUNT).max(1);
    let padding = 2.0;
    let radius = 6.0;

    for index in 0..CONTROL_BUTTON_COUNT {
        let start_x = index * segment;
        let end_x = if index == CONTROL_BUTTON_COUNT - 1 {
            width
        } else {
            (index + 1) * segment
        };
        let button_width = end_x.saturating_sub(start_x);
        if button_width == 0 {
            continue;
        }

        let is_hovered = hover_button == Some(index);

        let base_color = match index {
            0 => COLOR_SAVE,
            1 => COLOR_COPY,
            2 => {
                if paused {
                    COLOR_RESUME
                } else {
                    COLOR_PAUSE
                }
            }
            _ => COLOR_CANCEL,
        };

        // Lighten color on hover
        let color = if is_hovered {
            lighten_color(base_color, 40)
        } else {
            base_color
        };

        // Draw rounded rectangle button
        let x = start_x as f32 + padding;
        let y = padding;
        let w = button_width as f32 - padding * 2.0;
        let h = bar_height as f32 - padding * 2.0;

        if let Some(path) = rounded_rect_path(x, y, w, h, radius) {
            let mut paint = Paint::default();
            paint.set_color(Color::from_rgba8(color[0], color[1], color[2], color[3]));
            paint.anti_alias = true;
            pixmap.fill_path(
                &path,
                &paint,
                FillRule::Winding,
                Transform::identity(),
                None,
            );
        }

        // Draw label
        let label = match index {
            0 => 'S',
            1 => 'C',
            2 => 'P',
            _ => 'X',
        };
        let scale = ((bar_height as f32 - 8.0) / 7.0).max(1.0) as u32;
        let glyph_width = 5 * scale;
        let glyph_height = 7 * scale;
        let label_x = start_x + (button_width.saturating_sub(glyph_width)) / 2;
        let label_y = (bar_height.saturating_sub(glyph_height)) / 2;
        draw_glyph_to_pixmap(&mut pixmap, label_x, label_y, scale, label, COLOR_LABEL);
    }

    // Copy pixmap to canvas at bar_y position
    let pixmap_data = pixmap.data();
    for y in 0..bar_height {
        let dst_y = bar_y + y;
        if dst_y >= height {
            break;
        }
        let src_row = (y * width * 4) as usize;
        let dst_row = (dst_y * width * 4) as usize;
        for x in 0..width {
            let src_idx = src_row + (x * 4) as usize;
            let dst_idx = dst_row + (x * 4) as usize;
            if src_idx + 3 < pixmap_data.len() && dst_idx + 3 < canvas.len() {
                // tiny-skia uses RGBA, Wayland uses BGRA (ARGB8888)
                let a = pixmap_data[src_idx + 3];
                if a > 0 {
                    canvas[dst_idx] = pixmap_data[src_idx + 2]; // B
                    canvas[dst_idx + 1] = pixmap_data[src_idx + 1]; // G
                    canvas[dst_idx + 2] = pixmap_data[src_idx]; // R
                    canvas[dst_idx + 3] = a; // A
                }
            }
        }
    }
}

fn lighten_color(color: [u8; 4], amount: u8) -> [u8; 4] {
    [
        color[0].saturating_add(amount),
        color[1].saturating_add(amount),
        color[2].saturating_add(amount),
        color[3],
    ]
}

fn rounded_rect_path(x: f32, y: f32, w: f32, h: f32, r: f32) -> Option<tiny_skia::Path> {
    let r = r.min(w / 2.0).min(h / 2.0);
    let mut pb = PathBuilder::new();
    pb.move_to(x + r, y);
    pb.line_to(x + w - r, y);
    pb.quad_to(x + w, y, x + w, y + r);
    pb.line_to(x + w, y + h - r);
    pb.quad_to(x + w, y + h, x + w - r, y + h);
    pb.line_to(x + r, y + h);
    pb.quad_to(x, y + h, x, y + h - r);
    pb.line_to(x, y + r);
    pb.quad_to(x, y, x + r, y);
    pb.close();
    pb.finish()
}

fn draw_glyph_to_pixmap(pixmap: &mut Pixmap, x: u32, y: u32, scale: u32, ch: char, color: [u8; 4]) {
    let rows = match glyph_rows(ch) {
        Some(rows) => rows,
        None => return,
    };
    let mut paint = Paint::default();
    paint.set_color(Color::from_rgba8(color[0], color[1], color[2], color[3]));
    paint.anti_alias = false;

    for (row_index, row_bits) in rows.iter().enumerate() {
        for col in 0..5u32 {
            let mask = 1 << (4 - col);
            if row_bits & mask == 0 {
                continue;
            }
            let px = x + col * scale;
            let py = y + row_index as u32 * scale;
            if let Some(rect) =
                tiny_skia::Rect::from_xywh(px as f32, py as f32, scale as f32, scale as f32)
            {
                pixmap.fill_rect(rect, &paint, Transform::identity(), None);
            }
        }
    }
}

fn blit_preview_bottom(
    canvas: &mut [u8],
    canvas_width: u32,
    available_height: u32,
    preview: &PreviewImage,
) {
    let bytes_per_row = canvas_width.saturating_mul(4) as usize;
    if bytes_per_row == 0 || available_height == 0 {
        return;
    }

    let max_cols = preview.width.min(canvas_width);

    // If preview fits, show from top; otherwise show bottom part
    let (src_start_y, display_height) = if preview.height <= available_height {
        (0, preview.height)
    } else {
        // Show the bottom part of the preview
        (preview.height - available_height, available_height)
    };

    for y in 0..display_height {
        let src_y = src_start_y + y;
        let src_row = (src_y * preview.width * 4) as usize;
        let dst_row = (y * canvas_width * 4) as usize;
        for x in 0..max_cols {
            let src_idx = src_row + (x * 4) as usize;
            let dst_idx = dst_row + (x * 4) as usize;
            if src_idx + 3 < preview.pixels.len() && dst_idx + 3 < canvas.len() {
                canvas[dst_idx] = preview.pixels[src_idx + 2];
                canvas[dst_idx + 1] = preview.pixels[src_idx + 1];
                canvas[dst_idx + 2] = preview.pixels[src_idx];
                canvas[dst_idx + 3] = preview.pixels[src_idx + 3];
            }
        }
    }
}

#[allow(dead_code)]
fn blit_preview(canvas: &mut [u8], canvas_width: u32, preview: &PreviewImage, offset_y: u32) {
    let bytes_per_row = canvas_width.saturating_mul(4) as usize;
    if bytes_per_row == 0 {
        return;
    }
    let canvas_height = (canvas.len() / bytes_per_row) as u32;
    if offset_y >= canvas_height {
        return;
    }

    let max_rows = (canvas_height - offset_y).min(preview.height);
    let max_cols = preview.width.min(canvas_width);

    for y in 0..max_rows {
        let src_row = (y * preview.width * 4) as usize;
        let dst_row = ((offset_y + y) * canvas_width * 4) as usize;
        for x in 0..max_cols {
            let src_idx = src_row + (x * 4) as usize;
            let dst_idx = dst_row + (x * 4) as usize;
            canvas[dst_idx] = preview.pixels[src_idx + 2];
            canvas[dst_idx + 1] = preview.pixels[src_idx + 1];
            canvas[dst_idx + 2] = preview.pixels[src_idx];
            canvas[dst_idx + 3] = preview.pixels[src_idx + 3];
        }
    }
}

fn glyph_rows(ch: char) -> Option<[u8; 7]> {
    match ch {
        'C' => Some([
            0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110,
        ]),
        'P' => Some([
            0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000,
        ]),
        'S' => Some([
            0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110,
        ]),
        'X' => Some([
            0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001,
        ]),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{compute_preview_margin_left, OutputRect, PREVIEW_GAP};
    use crate::types::Region;

    fn region(x: i32, y: i32, w: u32, h: u32) -> Region {
        Region {
            raw: String::new(),
            x,
            y,
            w,
            h,
        }
    }

    fn output(x: i32, y: i32, width: i32, height: i32) -> OutputRect {
        OutputRect {
            id: 0,
            x,
            y,
            width,
            height,
        }
    }

    #[test]
    fn places_preview_to_right_when_space_available() {
        let outputs = [output(0, 0, 1920, 1080)];
        let region = region(200, 100, 400, 300);

        let left = compute_preview_margin_left(&region, 280, &outputs);

        assert_eq!(left, 200 + 400 + PREVIEW_GAP);
    }

    #[test]
    fn moves_preview_to_left_when_right_side_overflows() {
        let outputs = [output(0, 0, 1920, 1080)];
        let region = region(1700, 100, 200, 300);

        let left = compute_preview_margin_left(&region, 280, &outputs);

        assert_eq!(left, 1700 - PREVIEW_GAP - 280);
    }

    #[test]
    fn clamps_preview_when_neither_side_fits() {
        let outputs = [output(0, 0, 800, 600)];
        let region = region(300, 120, 200, 250);

        let left = compute_preview_margin_left(&region, 700, &outputs);

        assert_eq!(left, 100);
    }

    #[test]
    fn chooses_output_containing_region_center() {
        let outputs = [output(0, 0, 1920, 1080), output(1920, 0, 1920, 1080)];
        let region = region(3600, 100, 200, 300);

        let left = compute_preview_margin_left(&region, 400, &outputs);

        assert_eq!(left, 3600 - PREVIEW_GAP - 400);
    }

    #[test]
    fn falls_back_to_right_side_without_outputs() {
        let region = region(50, 60, 120, 220);

        let left = compute_preview_margin_left(&region, 300, &[]);

        assert_eq!(left, 50 + 120 + PREVIEW_GAP);
    }
}
