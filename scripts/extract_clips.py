import cv2
import os

def extract_clips(video_path, output_folder, start_time, end_time, start_index=1, interval=15, clip_duration=5):
    """
    Extracts video clips from a video at a specific interval and saves them.
    
    :param video_path: Path to the video file
    :param output_folder: Where to save the video clips
    :param start_time: Start time in seconds
    :param end_time: End time in seconds
    :param start_index: The starting number for the filename (e.g., 1 for tray_1.mp4)
    :param interval: Seconds between each extracted clip start
    :param clip_duration: Duration of each clip in seconds
    :return: The next available index for chaining videos
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return start_index

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Calculate frame counts
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    frame_interval = int(interval * fps)
    clip_frames = int(clip_duration * fps)

    current_frame = start_frame
    current_index = start_index

    print(f"Processing '{video_path}'...")

    while current_frame <= end_frame:
        # Jump directly to the start frame of the clip
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        filename = os.path.join(output_folder, f"clip_{current_index}.mp4")
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

        frames_written = 0
        for _ in range(clip_frames):
            ret, frame = cap.read()
            if not ret:
                print(f"Reached end of video while writing {filename}.")
                break
            out.write(frame)
            frames_written += 1

        out.release()

        # Stop processing if no frames could be read for this clip
        if frames_written == 0:
            break

        print(f"Saved: {filename} (Video Time: {current_frame/fps:.1f}s)")

        # Advance our counters
        current_frame += frame_interval
        current_index += 1

    # Release the video object
    cap.release()
    print("Done!")
    
    # Return the next available index so you can use it for the next video
    return current_index

# ==========================================
# Example Usage: Processing multiple videos
# ==========================================
if __name__ == "__main__":
    output_dir = "extracted_clips"
    
    # Process First Video 
    # first tray has no transition, so we can start from 0 seconds to 5 seconds
    next_index = extract_clips(
        video_path="../videos/video_1.mkv",
        output_folder=output_dir,
        start_time=0,      # Start at 0 seconds
        end_time=7,      # End at 7 seconds
        start_index=1,     # Starts at clip_1.mp4
        interval=15,       # Starts a new clip every 15 seconds
        clip_duration=7    # Each clip is 7 seconds long
    )

    # rest of first video
    next_index = extract_clips(
        video_path="../videos/video_1.mkv",
        output_folder=output_dir,
        start_time=4,      # Start at 4 seconds
        end_time=403,      # End at 403 seconds
        start_index=next_index,     # Starts at clip_1.mp4
        interval=15,       # Starts a new clip every 15 seconds
        clip_duration=7    # Each clip is 7 seconds long
    )
    
    print(f"\nVideo 1 finished. The next available index is {next_index}.\n")
    
    # Process Second Video
    next_index = extract_clips(
        video_path="../videos/video_2.mkv",
        output_folder=output_dir,
        start_time=11,     # Start at 11 seconds
        end_time=562,      # End at 562 seconds
        start_index=next_index, # Automatically continues the sequence
        interval=15,
        clip_duration=7
    )
    
    # Process Third Video
    next_index = extract_clips(
        video_path="../videos/video_3.mkv",
        output_folder=output_dir,
        start_time=18,     # Start at 19 seconds
        end_time=223,      # End at 225 seconds
        start_index=next_index,
        interval=15,
        clip_duration=7
    )
    
    # Process Fourth Video
    # process first 5 seconds due to lack of transition
    next_index = extract_clips(
        video_path="../videos/video_4.mkv",
        output_folder=output_dir,
        start_time=5,      # Start at 5 seconds
        end_time=12,      # End at 12 seconds
        start_index=next_index,
        interval=15,
        clip_duration=7
    )
    # rest of fourth video until 3rd to the last tray since 2nd to the last is under 30 calamansi
    next_index = extract_clips(
        video_path="../videos/video_4.mkv",
        output_folder=output_dir,
        start_time=9,      # Start at 9 seconds
        end_time=440,      # End at 440 seconds
        start_index=next_index,
        interval=15,
        clip_duration=7
    )

    # last tray of fourth video is 30 calamansi so we will extract it separately
    next_index = extract_clips(
        video_path="../videos/video_4.mkv",
        output_folder=output_dir,
        start_time=463,      # Start at 464 seconds
        end_time=471,      # End at 469 seconds
        start_index=next_index,
        interval=15,
        clip_duration=7
    )