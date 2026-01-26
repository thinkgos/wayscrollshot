use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState},
    delegate_compositor, delegate_layer, delegate_output, delegate_registry, delegate_shm,
    output::{OutputHandler, OutputState},
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
    Connection, QueueHandle,
};

use crate::constants::{REGION_BORDER_COLOR, REGION_BORDER_WIDTH};
use crate::types::Region;

pub struct RegionOverlay {
    stop_flag: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl RegionOverlay {
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

    let surface = compositor.create_surface(&qh);
    let layer = layer_shell.create_layer_surface(
        &qh,
        surface,
        Layer::Overlay,
        Some("wayscrollshot-region"),
        None,
    );

    let border = REGION_BORDER_WIDTH;
    let width = region.w + border * 2;
    let height = region.h + border * 2;

    layer.set_anchor(Anchor::TOP | Anchor::LEFT);
    layer.set_keyboard_interactivity(KeyboardInteractivity::None);
    layer.set_exclusive_zone(-1);
    layer.set_margin(
        region.y - border as i32,
        0,
        0,
        region.x - border as i32,
    );
    layer.set_size(width, height);

    let input_region = compositor.wl_compositor().create_region(&qh, ());
    layer.wl_surface().set_input_region(Some(&input_region));
    layer.commit();

    let pool = SlotPool::new((width * height * 4) as usize, &shm)
        .context("failed to create shm pool")?;

    let mut border_state = RegionBorder {
        registry_state: RegistryState::new(&globals),
        output_state: OutputState::new(&globals, &qh),
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
