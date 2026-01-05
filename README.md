# Video Watermark Remover 🎬

A Python script to remove watermarks from videos using multiple techniques.

## Features

- **Interactive Selection**: Click and drag to select the watermark region
- **Multiple Methods**:
  - **Inpainting** (OpenCV) - Best quality, fills watermark area intelligently
  - **Delogo** (FFmpeg) - Fast professional filter
  - **Blur** - Quick and simple blurring

## Requirements

### Python Packages
```bash
pip install -r requirements.txt
```

### FFmpeg
FFmpeg must be installed and available in your system PATH.

**Windows:**
1. Download from [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extract and add `bin` folder to PATH

**Or using Chocolatey:**
```bash
choco install ffmpeg
```

## Usage

### Basic Usage (Interactive Region Selection)

```bash
# Best quality - uses inpainting
python watermark_remover.py -i input_video.mp4 -o output_video.mp4 -m inpaint

# Fastest - uses FFmpeg delogo filter
python watermark_remover.py -i input_video.mp4 -o output_video.mp4 -m delogo

# Simple blur
python watermark_remover.py -i input_video.mp4 -o output_video.mp4 -m blur
```

### With Predefined Coordinates

Skip interactive selection by providing coordinates directly:

```bash
python watermark_remover.py -i video.mp4 -o clean.mp4 -m inpaint --coords 100,50,200,80
```

Format: `x,y,width,height` (in pixels from top-left corner)

### Advanced Options

```bash
# Increase inpainting radius for better quality (slower)
python watermark_remover.py -i video.mp4 -o clean.mp4 -m inpaint --radius 10

# Adjust blur strength
python watermark_remover.py -i video.mp4 -o clean.mp4 -m blur --blur-strength 35
```

## Method Comparison

| Method | Quality | Speed | Best For |
|--------|---------|-------|----------|
| `inpaint` | ★★★★★ | ★★☆☆☆ | Small/medium watermarks, complex backgrounds |
| `delogo` | ★★★★☆ | ★★★★★ | Large watermarks, uniform backgrounds |
| `blur` | ★★☆☆☆ | ★★★★★ | Quick removal, privacy purposes |

## How It Works

1. **Inpainting**: Uses OpenCV's Telea or Navier-Stokes algorithms to reconstruct the region under the watermark based on surrounding pixels.

2. **Delogo**: Uses FFmpeg's professional delogo filter which interpolates the watermark region using edge detection.

3. **Blur**: Simply applies Gaussian blur to the watermark region - fastest but least effective.

## Tips for Best Results

1. **Select precisely**: Include only the watermark area, nothing more
2. **Higher radius for inpainting**: Increase `--radius` for better quality with complex backgrounds
3. **Consistent watermarks**: Works best when watermark stays in the same position throughout the video
4. **Test first**: Process a short clip first to check quality

## Limitations

- Works best with static watermarks (same position throughout video)
- Complex or large watermarks may leave visible artifacts
- Moving/animated watermarks are harder to remove
- Very thin/transparent watermarks work better with inpainting

## Example

```bash
# Remove TikTok watermark from video
python watermark_remover.py -i tiktok_video.mp4 -o clean_video.mp4 -m inpaint --radius 7
```

## License

MIT License - Feel free to use and modify!
