use tiny_skia::{Color, FillRule, Paint, PathBuilder, Pixmap, Transform};

use crate::constants::CONTROL_BAR_HEIGHT;
use crate::types::PreviewImage;

const CONTROL_BUTTON_COUNT: u32 = 4;
const COLOR_SAVE: [u8; 4] = [39, 174, 96, 230];
const COLOR_COPY: [u8; 4] = [52, 152, 219, 230];
const COLOR_PAUSE: [u8; 4] = [241, 196, 15, 230];
const COLOR_RESUME: [u8; 4] = [26, 188, 156, 230];
const COLOR_CANCEL: [u8; 4] = [231, 76, 60, 230];
const COLOR_LABEL: [u8; 4] = [255, 255, 255, 255];

/// Draws the control bar (Save/Copy/Pause/Cancel) at the bottom.
pub(super) fn draw_control_bar(
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

        let color = if is_hovered {
            lighten_color(base_color, 40)
        } else {
            base_color
        };

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
                let a = pixmap_data[src_idx + 3];
                if a > 0 {
                    canvas[dst_idx] = pixmap_data[src_idx + 2];
                    canvas[dst_idx + 1] = pixmap_data[src_idx + 1];
                    canvas[dst_idx + 2] = pixmap_data[src_idx];
                    canvas[dst_idx + 3] = a;
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

pub(super) fn blit_preview_bottom(
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

    let (src_start_y, display_height) = if preview.height <= available_height {
        (0, preview.height)
    } else {
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
