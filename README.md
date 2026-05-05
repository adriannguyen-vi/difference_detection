# Difference Detection (Site Change Detection)

This project provides a video processing pipeline to detect progress or changes over a site (e.g., a construction site) by comparing drone videos taken at different times (e.g., previous week vs. current week). 

The pipeline aligns Geographic/Telemetry data and uses feature matching to pinpoint exact geographical changes efficiently.

## Workflow

The workflow consists of two main steps:

### 1. Extract Reference Frames
First, use `src/extract_frames.py` to extract reference frames from the previous week's drone video. This script identifies key frames based on ORB feature matching and records them along with their spatial and telemetry metadata (parsed from the drone's `.srt` file).

```bash
python src/extract_frames.py \
    --video_path path/to/previous_week_video.mp4 \
    --output_dir ./data \
    --threshold 0.2
```
This process will create a subfolder in `--output_dir` containing the extracted reference frames (`.jpg`) along with a `metadata.json` file.

### 2. Process Current Video and Detect Changes
Then, use these extracted reference frames as baseline images in `src/main.py`. The `SiteChangeProcessor` (defined in `src/processor.py`) will match the current week's video against these reference frames, geometrically align them, and detect significant structural changes. It then creates an overlaid heatmap.

```bash
python src/main.py \
    --metadata ./data/previous_week_video/metadata.json \
    --new_video path/to/current_week_video.MP4 \
    --new_srt path/to/current_week_video.SRT \
    --output_dir comparison_results/ \
    --out_video_name difference_summary.mp4
```

The output will be a 3-panel video (`comparison_results/difference_summary.mp4`) that side-by-side displays:
1. **Previous**: The reference frame from the previous week.
2. **Current**: The matched frame from the current week.
3. **Change**: A heatmap overlay highlighting the detected differences.

## Project Structure

- `src/extract_frames.py`: Script to extract reference frames and `metadata.json` from baseline videos.
- `src/main.py`: Entry point for processing changes against a new video.
- `src/processor.py`: Core `SiteChangeProcessor` object tying together video alignment, matching, and difference mapping.
- `src/core/`: Contains discrete algorithmic core components for alignment, telemetry parsing, change detection, and geospatial processing.
- `src/utils.py`: Contains auxiliary functions (e.g., parsing drone `.srt` files).
