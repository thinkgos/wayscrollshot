# Long Shot (Rust)

Minimal long screenshot tool for Linux that stitches while you scroll.

## Requirements

- `slurp` for region selection
- `grim` for capture
- `wl-copy` (Wayland) or `xclip` (X11) for clipboard copy

## Build

```bash
cargo build --release
```

## Run

```bash
cargo run --release
```

## Flow

1. Select a capture region with slurp.
2. A layer-shell overlay appears in the top-right with a preview + control bar.
3. Scroll the target app; the preview grows as new frames are stitched.
4. Click the control bar: `S` save, `C` copy, `P` pause/resume, `X` cancel.

## Hotkeys

- `S` save and exit (overlay focused)
- `C` copy and exit (overlay focused)
- `Esc` or `Q` cancel and exit (overlay focused)
- `Space` pause or resume capture (overlay focused)

## Notes

- Wayland only; X11 is not supported.
- The preview uses `wlr-layer-shell-unstable-v1` (wlroots-based compositors).
- Only the control bar accepts clicks; the preview stays click-through.
- Best results if each scroll step leaves some overlap with the previous view.
- Default output is `~/Pictures` if it exists, otherwise `$HOME`.
