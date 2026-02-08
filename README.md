# wayscrollshot

A scrolling screenshot tool for Wayland that captures and stitches images in real-time as you scroll.

[中文文档](README_CN.md)

## Features

- Real-time preview with automatic stitching
- Column sampling algorithm for fast and accurate overlap detection
- Rounded button UI with hover effects (powered by tiny-skia)
- Keyboard shortcuts and mouse control
- Save to file or copy to clipboard
- Supports reverse scrolling

## How It Works

### Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Capture   │────>│   Stitcher   │────>│   Preview   │
│   (grim)    │     │ (col-sample) │     │ (layer-shell)│
└─────────────┘     └──────────────┘     └─────────────┘
```

### Column Sampling Algorithm

Instead of comparing entire images pixel-by-pixel, wayscrollshot uses a column sampling approach inspired by [screenshot-splicing](https://github.com/aspect-ratio/screenshot-splicing):

1. **Sample 3 column groups** from each frame:
   - Left region (20 to width/4)
   - Middle region (width/2 to 5*width/8)
   - Right region (6*width/8 to 7*width/8)

2. **Convert to grayscale** and average each group

3. **Search for overlap** using Mean Absolute Difference (MAD):
   - Start from the predicted offset (based on previous scroll)
   - Expand search outward: `[p, p+1, p-1, p+2, p-2, ...]`
   - Early termination when MAD < threshold

4. **Append new content** to the stitched image

**Complexity**: O(9 * height) instead of O(width * height) - a significant speedup.

### Overlap Detection

```
Frame 1 (previous):          Frame 2 (current):
┌────────────────┐           ┌────────────────┐
│    Content A   │           │    Content B   │
│                │           │                │
│    Content B   │ <──────── │    Content B   │  (overlap)
│                │           │                │
│    Content C   │           │    Content C   │
└────────────────┘           │                │
                             │    Content D   │  (new)
                             └────────────────┘
```

The algorithm finds where Frame 2's top matches Frame 1's content, then appends only the new portion.

## Dependencies

### Runtime Dependencies

| Tool | Purpose | Required |
|------|---------|----------|
| `slurp` | Region selection | Yes |
| `grim` | Screen capture | Yes |
| `wl-copy` | Clipboard (Wayland) | For clipboard feature |
| `xclip` | Clipboard (X11 fallback) | Alternative |

### Build Dependencies

| Crate | Purpose |
|-------|---------|
| `smithay-client-toolkit` | Wayland client library |
| `wayland-client` | Wayland protocol bindings |
| `tiny-skia` | 2D graphics (rounded buttons) |
| `image` | Image processing and resizing |
| `clap` | Command-line argument parsing |
| `anyhow` | Error handling |
| `chrono` | Timestamp for filenames |
| `log` / `env_logger` | Logging |

## Installation

### From Source

```bash
# Install runtime dependencies (Arch Linux)
sudo pacman -S slurp grim wl-clipboard

# Build
cargo build --release

# Install (optional)
cp target/release/wayscrollshot ~/.local/bin/
```

## Usage

```bash
# Basic usage
wayscrollshot

# Save to specific file
wayscrollshot -o ~/screenshot.png

# Copy to clipboard instead of saving
wayscrollshot -c

# Custom preview width
wayscrollshot -w 320

# Disable preview window
wayscrollshot --no-preview

# Disable region border overlay
wayscrollshot --no-border

# Use different stitching algorithms
wayscrollshot -a col-sample  # Default: fast column sampling
wayscrollshot -a template    # Template matching (more accurate)
wayscrollshot -a edge        # Edge detection (for transparent backgrounds)
wayscrollshot -a fast        # FAST corner + HNSW index (experimental)
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output <PATH>` | Output file path | `$XDG_PICTURES_DIR/wayscrollshot-<timestamp>.png` |
| `-w, --preview-width <PX>` | Preview width in pixels | 280 |
| `-c, --clipboard` | Copy to clipboard instead of saving | false |
| `--no-preview` | Disable preview window | false |
| `--no-border` | Disable region border overlay | false |
| `-a, --algorithm <ALG>` | Stitching algorithm: `col-sample`, `template`, `edge`, `fast` | col-sample |

### Controls

**Mouse:**
- Click buttons in the control bar

**Keyboard (when overlay is focused):**
| Key | Action |
|-----|--------|
| `S` | Save and exit |
| `C` | Copy to clipboard and exit |
| `Space` | Pause/Resume capture |
| `Q` / `Esc` | Cancel and exit |

## Limitations

1. **Wayland only**: X11 is not supported. The tool uses `wlr-layer-shell-unstable-v1` protocol.

2. **wlroots-based compositors**: Works on Sway, Hyprland, river, etc. May not work on GNOME/KDE Wayland.

3. **Overlap requirement**: Each scroll step must leave some overlap with the previous view. Very fast scrolling may cause stitching failures.

4. **Static content assumption**: The algorithm assumes the scrolling content is static. Dynamic content (animations, videos) will cause artifacts.

5. **Vertical scrolling only**: Horizontal scrolling is not currently supported.

6. **Fixed header/footer**: If the page has fixed headers or footers, they will be captured repeatedly. Consider selecting a region that excludes them.

## Troubleshooting

### "slurp selection failed"
- Ensure `slurp` is installed and in PATH
- Check if you're running on Wayland

### "layer-shell not available"
- Your compositor doesn't support `wlr-layer-shell-unstable-v1`
- Try a wlroots-based compositor (Sway, Hyprland)

### "No overlap match"
- Scroll more slowly
- Ensure there's visible overlap between frames
- Avoid scrolling through completely different content

### Preview not updating
- Check if the capture region is correct
- Try running with `RUST_LOG=debug` for more info

## License

MIT

## Contributing

Contributions are welcome! Areas that could use improvement:

- **Algorithm optimization**: The `fast` algorithm (FAST corner + HNSW) needs tuning for better accuracy
- **Cross-platform support**: Currently Linux-only due to Wayland dependency
- **Performance**: Reduce memory usage for very long screenshots
- **UI improvements**: Better visual feedback during capture

Please open an issue to discuss major changes before submitting a PR.

## Acknowledgments

- [screenshot-splicing](https://github.com/aspect-ratio/screenshot-splicing) - Column sampling algorithm inspiration
- [snow-shot](https://github.com/mg-chao/snow-shot) - FAST corner + HNSW algorithm reference
- [smithay-client-toolkit](https://github.com/Smithay/client-toolkit) - Wayland client library
- [tiny-skia](https://github.com/RazrFalcon/tiny-skia) - 2D graphics library
