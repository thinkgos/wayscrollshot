use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState},
    delegate_compositor, delegate_layer, delegate_output, delegate_registry, delegate_shm,
    output::{OutputData, OutputHandler, OutputState},
    registry::{ProvidesRegistryState, RegistryState},
    registry_handlers,
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
    protocol::{wl_output, wl_region, wl_shm, wl_surface},
    Connection, Proxy, QueueHandle,
};

use crate::constants::{REGION_BORDER_COLOR, REGION_BORDER_WIDTH};
use crate::types::Region;

#[derive(Clone, Copy, Debug)]
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

fn output_rects_from_state(output_state: &OutputState) -> Vec<OutputRect> {
    output_state
        .outputs()
        .filter_map(|output| output_state.info(&output))
        .filter_map(|info| output_rect_from_info(&info))
        .collect()
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

pub struct RegionOverlay {
    stop_flag: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl RegionOverlay {
    /// Spawns the region border overlay thread and waits for initialization.
    pub fn new(region: Region) -> Result<Self> {
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_clone = stop_flag.clone();
        let ready = Arc::new(AtomicBool::new(false));
        let ready_clone = ready.clone();

        let handle = thread::spawn(move || {
            if let Err(err) = run_region_overlay(region, stop_clone, ready_clone) {
                log::warn!("region overlay failed: {err}");
            }
        });

        thread::sleep(Duration::from_millis(200));
        if !ready.load(Ordering::Relaxed) {
            bail!("region overlay did not initialize in time");
        }

        Ok(Self {
            stop_flag,
            handle: Some(handle),
        })
    }

    /// Stops the region overlay thread and joins it.
    pub fn stop(&mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for RegionOverlay {
    fn drop(&mut self) {
        self.stop();
    }
}

struct RegionBorder {
    registry_state: RegistryState,
    output_state: OutputState,
    shm: Shm,
    pool: SlotPool,
    layer: LayerSurface,
    width: u32,
    height: u32,
    configured: bool,
    exit: bool,
}

impl RegionBorder {
    fn draw(&mut self, _qh: &QueueHandle<Self>) {
        if self.width == 0 || self.height == 0 {
            return;
        }

        let needed = (self.width * self.height * 4) as usize;
        if self.pool.len() < needed {
            if let Err(err) = self.pool.resize(needed) {
                log::warn!("region overlay pool resize failed: {err}");
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
                log::warn!("region overlay buffer create failed: {err}");
                return;
            }
        };

        canvas.fill(0);
        draw_border(canvas, self.width, self.height);

        self.layer
            .wl_surface()
            .damage_buffer(0, 0, self.width as i32, self.height as i32);
        if let Err(err) = buffer.attach_to(self.layer.wl_surface()) {
            log::warn!("region overlay buffer attach failed: {err}");
            return;
        }
        self.layer.commit();
    }
}

impl CompositorHandler for RegionBorder {
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

impl OutputHandler for RegionBorder {
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

impl LayerShellHandler for RegionBorder {
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
        if configure.new_size.0 != 0 && configure.new_size.1 != 0 {
            self.width = configure.new_size.0;
            self.height = configure.new_size.1;
        }
        self.configured = true;
        self.draw(qh);
    }
}

impl ShmHandler for RegionBorder {
    fn shm_state(&mut self) -> &mut Shm {
        &mut self.shm
    }
}

delegate_compositor!(RegionBorder);
delegate_output!(RegionBorder);
delegate_shm!(RegionBorder);
delegate_layer!(RegionBorder);
delegate_registry!(RegionBorder);

impl ProvidesRegistryState for RegionBorder {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }

    registry_handlers!(OutputState);
}

impl wayland_client::Dispatch<wl_region::WlRegion, ()> for RegionBorder {
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

fn run_region_overlay(
    region: Region,
    stop_flag: Arc<AtomicBool>,
    ready: Arc<AtomicBool>,
) -> Result<()> {
    log::info!("Starting region overlay thread");
    let conn = Connection::connect_to_env().context("failed to connect to Wayland")?;
    let (globals, mut event_queue) =
        registry_queue_init(&conn).context("failed to init Wayland registry")?;
    let qh = event_queue.handle();
    let compositor = CompositorState::bind(&globals, &qh).context("wl_compositor not available")?;
    let layer_shell = LayerShell::bind(&globals, &qh).context("layer-shell not available")?;
    let shm = Shm::bind(&globals, &qh).context("wl_shm not available")?;
    let output_state = OutputState::new(&globals, &qh);
    let output_rects = output_rects_from_state(&output_state);
    let selected_output_rect = select_output_for_region(&region, &output_rects);
    let selected_output =
        selected_output_rect.and_then(|rect| find_output_by_id(&output_state, rect.id));

    let surface = compositor.create_surface(&qh);
    let layer = layer_shell.create_layer_surface(
        &qh,
        surface,
        Layer::Overlay,
        Some("wayscrollshot-region"),
        selected_output.as_ref(),
    );

    let border = REGION_BORDER_WIDTH;
    let width = region.w + border * 2;
    let height = region.h + border * 2;

    let output_origin = selected_output_rect
        .map(|rect| (rect.x, rect.y))
        .unwrap_or((0, 0));

    layer.set_anchor(Anchor::TOP | Anchor::LEFT);
    layer.set_keyboard_interactivity(KeyboardInteractivity::None);
    layer.set_exclusive_zone(-1);
    layer.set_margin(
        region.y - output_origin.1 - border as i32,
        0,
        0,
        region.x - output_origin.0 - border as i32,
    );
    layer.set_size(width, height);

    let input_region = compositor.wl_compositor().create_region(&qh, ());
    layer.wl_surface().set_input_region(Some(&input_region));
    layer.commit();

    let pool =
        SlotPool::new((width * height * 4) as usize, &shm).context("failed to create shm pool")?;

    let mut border_state = RegionBorder {
        registry_state: RegistryState::new(&globals),
        output_state,
        shm,
        pool,
        layer,
        width,
        height,
        configured: false,
        exit: false,
    };

    event_queue
        .roundtrip(&mut border_state)
        .context("initial roundtrip failed")?;

    ready.store(true, Ordering::Relaxed);

    loop {
        if stop_flag.load(Ordering::Relaxed) || border_state.exit {
            break;
        }

        event_queue
            .dispatch_pending(&mut border_state)
            .context("failed to process Wayland events")?;

        conn.flush().ok();
        if let Some(guard) = event_queue.prepare_read() {
            let _ = guard.read();
        }

        thread::sleep(Duration::from_millis(50));
    }

    Ok(())
}

fn draw_border(canvas: &mut [u8], width: u32, height: u32) {
    let border = REGION_BORDER_WIDTH;
    let color = [
        REGION_BORDER_COLOR[2],
        REGION_BORDER_COLOR[1],
        REGION_BORDER_COLOR[0],
        REGION_BORDER_COLOR[3],
    ];

    // Top border
    fill_rect(canvas, width, 0, 0, width, border, color);
    // Bottom border
    fill_rect(canvas, width, 0, height - border, width, border, color);
    // Left border
    fill_rect(canvas, width, 0, border, border, height - border * 2, color);
    // Right border
    fill_rect(
        canvas,
        width,
        width - border,
        border,
        border,
        height - border * 2,
        color,
    );
}

fn fill_rect(canvas: &mut [u8], canvas_width: u32, x: u32, y: u32, w: u32, h: u32, color: [u8; 4]) {
    let max_x = (x + w).min(canvas_width);
    let canvas_height = canvas.len() as u32 / (canvas_width * 4);
    let max_y = (y + h).min(canvas_height);

    for yy in y..max_y {
        let row = (yy * canvas_width * 4) as usize;
        for xx in x..max_x {
            let idx = row + (xx * 4) as usize;
            if idx + 3 < canvas.len() {
                canvas[idx] = color[0];
                canvas[idx + 1] = color[1];
                canvas[idx + 2] = color[2];
                canvas[idx + 3] = color[3];
            }
        }
    }
}
