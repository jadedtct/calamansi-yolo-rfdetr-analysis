import os
import re
import cv2

def batch_crop_range(input_folder, output_folder, start_num, end_num, prefix="clip_", crop_size=(1280, 736), display_size=(1280, 720)):
    """
    Selects a fixed 1280x736 crop region on the first image by left-clicking 
    the upper-left corner, then applies that crop across all files in the range.
    """
    os.makedirs(output_folder, exist_ok=True)
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    crop_w, crop_h = crop_size
    
    # Filter files in range
    target_files = []
    for file_name in os.listdir(input_folder):
        ext = os.path.splitext(file_name)[1].lower()
        if ext in valid_exts and file_name.startswith(prefix):
            match = re.search(r'(\d+)(?=\.[^.]+$)', file_name)
            if match:
                file_num = int(match.group(1))
                if start_num <= file_num <= end_num:
                    target_files.append((file_num, file_name))

    target_files.sort(key=lambda x: x[0])

    if not target_files:
        print(f"\n[Skipped] No files found for range '{prefix}[{start_num}-{end_num}]'.")
        return

    # Select crop area using the first file in this range
    first_file = target_files[0][1]
    sample_path = os.path.join(input_folder, first_file)
    sample_img = cv2.imread(sample_path)

    if sample_img is None:
        print(f"Error loading image: {first_file}")
        return

    h_img, w_img = sample_img.shape[:2]
    crop_coords = []
    preview = sample_img.copy()

    def mouse_callback(event, x, y, flags, param):
        nonlocal crop_coords, preview
        
        # Clamp upper-left point so the fixed box stays inside image boundaries
        x1 = max(0, min(x, w_img - crop_w))
        y1 = max(0, min(y, h_img - crop_h))
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        # Live preview box that follows the cursor before clicking
        if event == cv2.EVENT_MOUSEMOVE and not crop_coords:
            preview = sample_img.copy()
            cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Cyan box hover
            cv2.imshow(window_title, preview)
        
        # Left-click locks the top-left corner and draws fixed 1280x736 box
        elif event == cv2.EVENT_LBUTTONDOWN:
            crop_coords = [(x1, y1), (x2, y2)]
            preview = sample_img.copy()
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box locked
            cv2.imshow(window_title, preview)

    window_title = f"Select Top-Left Corner for {prefix}[{start_num}-{end_num}] (Press ENTER)"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, display_size[0], display_size[1])
    cv2.setMouseCallback(window_title, mouse_callback)

    print(f"\n--- Range {prefix}{start_num} to {prefix}{end_num} ---")
    print(f"Reference image: {first_file}")
    print(" -> Hover mouse, left-click top-left corner, then press 'ENTER' to confirm.")

    while True:
        if not crop_coords:
            cv2.imshow(window_title, preview)
        
        key = cv2.waitKey(20) & 0xFF
        if key == 13 and crop_coords:  # ENTER key
            break
        elif key == ord('r') or key == ord('R'):  # Reset click
            crop_coords = []
            preview = sample_img.copy()

    cv2.destroyAllWindows()

    # Apply native 1280x736 crop across range
    (x1, y1), (x2, y2) = crop_coords[0], crop_coords[1]
    for file_num, file_name in target_files:
        in_path = os.path.join(input_folder, file_name)
        name, ext = os.path.splitext(file_name)
        out_path = os.path.join(output_folder, f"{name}_cropped{ext}")
        
        img = cv2.imread(in_path)
        if img is None:
            continue
        
        cropped = img[y1:y2, x1:x2]
        cv2.imwrite(out_path, cropped)
        print(f"Saved: {file_name} ({cropped.shape[1]}x{cropped.shape[0]}px)")


# --- Execute 3 Separate Ranges ---
if __name__ == "__main__":
    INPUT_DIR = "extracted_frames"   # Source folder
    OUTPUT_DIR = "cropped_images"    # Destination folder
    CROP_SIZE = (1280, 736)          # Fixed crop dimensions (Width, Height)
    PREVIEW_SIZE = (1280, 720)       # Display window size

    # Call 1: Range 1 (clip_1 to clip_28)
    batch_crop_range(
        input_folder=INPUT_DIR,
        output_folder=OUTPUT_DIR,
        start_num=1,
        end_num=28,
        prefix="clip_",
        crop_size=CROP_SIZE,
        display_size=PREVIEW_SIZE
    )

    # Call 2: Range 2 (clip_29 to clip_79)
    batch_crop_range(
        input_folder=INPUT_DIR,
        output_folder=OUTPUT_DIR,
        start_num=29,
        end_num=79,
        prefix="clip_",
        crop_size=CROP_SIZE,
        display_size=PREVIEW_SIZE
    )

    # Call 3: Range 3 (clip_80 to clip_110)
    batch_crop_range(
        input_folder=INPUT_DIR,
        output_folder=OUTPUT_DIR,
        start_num=80,
        end_num=110,
        prefix="clip_",
        crop_size=CROP_SIZE,
        display_size=PREVIEW_SIZE
    )

    print("\nAll 3 ranges processed and saved directly at 1280x736 native resolution.")