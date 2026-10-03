import cv2
import os
import glob
import re

def extract_frames_from_clips(clips_folder, output_folder, default_sec=5.0, custom_timestamps=None):
    """
    Extracts a single PNG frame from each clip in clips_folder. 
    Uses custom timestamps for specific clips and a standard default timestamp 
    for all others to avoid duplicate processing.
    
    :param clips_folder: Path to directory containing .mp4 clips
    :param output_folder: Path to directory where PNG frames will be saved
    :param default_sec: Standard timestamp (in seconds) for general clips
    :param custom_timestamps: Dict mapping clip ID or filename to custom timestamp
    """
    os.makedirs(output_folder, exist_ok=True)
    
    if custom_timestamps is None:
        custom_timestamps = {}

    # Standardize custom dictionary keys to strings for flexible matching
    formatted_custom = {str(k): float(v) for k, v in custom_timestamps.items()}

    clip_files = glob.glob(os.path.join(clips_folder, "*.mp4"))

    if not clip_files:
        print(f"No .mp4 files found in '{clips_folder}'. Check directory path.")
        return

    print(f"Processing {len(clip_files)} clip(s)...")

    for clip_path in sorted(clip_files):
        filename = os.path.basename(clip_path)
        base_name = os.path.splitext(filename)[0]

        # Extract numerical ID from filename (e.g., 'tray_94' -> '94')
        numbers_in_name = re.findall(r'\d+', base_name)
        clip_id = numbers_in_name[0] if numbers_in_name else None

        # Apply custom timestamp if matched, otherwise fall back to standard timestamp
        if base_name in formatted_custom:
            target_sec = formatted_custom[base_name]
            tag = "Custom"
        elif clip_id and clip_id in formatted_custom:
            target_sec = formatted_custom[clip_id]
            tag = "Custom"
        else:
            target_sec = default_sec
            tag = "Default"

        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            print(f"Error opening clip: {clip_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        target_frame = int(target_sec * fps)

        # Seek directly to the target timestamp
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()

        if ret:
            output_filename = os.path.join(output_folder, f"{base_name}.png")
            cv2.imwrite(output_filename, frame)
            print(f"Saved: {output_filename} [{tag} Timestamp: {target_sec}s]")
        else:
            print(f"Failed to read frame at {target_sec}s in {filename}")

        cap.release()

    print("\nFrame extraction complete!")


# ==========================================
# Execution Configuration
# ==========================================
if __name__ == "__main__":
    CLIPS_DIR = "extracted_clips"   # Source folder containing .mp4 clips
    FRAMES_DIR = "extracted_frames" # Output folder for PNG images

    # Custom timestamps for specific clips
    CUSTOM_TIMESTAMPS = {
        20: 5.9,
        21: 6.0,
        22: 6.0,
        38: 5.0,
        48: 5.5,
        53: 6.0,
        54: 6.0,
        55: 5.3,
        62: 5.0,
        63: 6.6,
        72: 6.1,
        85: 5.2,
        86: 6.0,
        87: 6.0,
        88: 5.0,
        89: 5.2,
        94: 6.9,
        97: 6.3,
        103: 6.95,

    }

    extract_frames_from_clips(
        clips_folder=CLIPS_DIR,
        output_folder=FRAMES_DIR,
        default_sec=5.69,               # Standard timestamp for all other clips
        custom_timestamps=CUSTOM_TIMESTAMPS # Custom overrides (prevents duplicates)
    )