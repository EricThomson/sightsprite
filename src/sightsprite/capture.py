"""
sightsprite OpenCV utilities for interfacing with video devices
mainly, capturing training data, but also some for displaying data
"""
import cv2
from datetime import datetime
import os
from pathlib import Path
import time

ASSETS_DIR = "../../assets/"

def show_test_image():
    print("Showing built-in image...click anywhere in image to close")
    image_path = ASSETS_DIR + "test_image.png"
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Resource not found: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to load image")
        return None
    else:
        cv2.imshow("Sample Image", img)
        cv2.waitKey(0)
        cv2.destroyWindow("Sample Image")


def show_test_video(fps=30):
    """
    Display test video in assets, allow setting the fps to control frame rate.
    Cap the fps at 30 to avoid overloading the system.

    Will show in continuous loop until user closes window by pressing a key while window is 
    highlighted.
    """
    print("Showing test video... click any key to close the video window")
    video_path = ASSETS_DIR + "test_video.mp4"  
    fps = min(fps, 30) # cap fps at 30 for now

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Resource not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video")
        return None
    
    delay = int(1000 / fps)  # delay b/w frames in ms

    while True:
        ret, frame = cap.read()

        # loop back to beginning at end
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to the first frame
            continue

        # Show the video frame
        cv2.imshow("Test Video", frame)

        # Wait for key press or window close event until 'x' is pressed
        key = cv2.waitKey(delay) & 0xFF
        if key != 255:  # If any key is pressed (except no key)
            print("Closing video window (key pressed)")
            break

    cap.release()  # Release the video capture object
    cv2.destroyAllWindows()  # Close all OpenCV windows


def capture_video(filepath, fps=30, duration=5):
    print(f"Capturing {duration} seconds of video at up to {fps} FPS...")
    filepath = Path(filepath)
    video_path = filepath / "captured_test.avi"

    # Open webcam
    print("Setting up webcam and video writer... this may take a moment.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Get intrinsic webcam FPS (may return 0.0 if unknown)
    webcam_fps = cap.get(cv2.CAP_PROP_FPS)
    if webcam_fps == 0.0:
        webcam_fps = 30.0  # fallback default
    actual_fps = min(fps, webcam_fps)
    print(f"Using {actual_fps:.2f} FPS (webcam supports {webcam_fps:.2f})")

    # Frame size
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video resolution: {width}x{height}")

    # VideoWriter setup
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(video_path), fourcc, actual_fps, (width, height))

    # Number of frames to capture
    total_frames = int(actual_fps * duration)
    print(f"Capturing {total_frames} frames...")

    print("Setup complete -- starting capture now!")
    frame_count = 0
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        out.write(frame)
        cv2.imshow("Captured Video", frame)
        if cv2.waitKey(int(1000 / actual_fps)) & 0xFF != 255:
            break  # Exit if any key is pressed

        frame_count += 1

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved to {video_path}")


def get_snapshot(image_path, show=True):
    """
    Capture a single frame from the default video device and save it to disk.

    Parameters
    ----------
    image_path : str or Path
        Full file path (including name and extension) to save image.
    show : bool
        If True, display the image in a window using opencv

    Returns
    -------
    image_path : Path or None
        Path to saved image, or None if capture failed.
    """
    image_path = Path(image_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not capture image.")
        cap.release()
        return None

    cv2.imwrite(str(image_path), frame)

    if show:
        cv2.imshow("Captured Image", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cap.release()
    return image_path


def get_snapshots(directory, filename_stem="image", 
                  save_interval=60, duration=3600,  
                  display_interval=0.1, show=True):
    """
    Capture a sequence of snapshots at regular intervals 

    Parameters
    ----------
    directory : str or Path
        Directory to save images.
    filename_stem : str
        Prefix for each image file.
    save_interval : float
        Time between snapshots (seconds).
    duration : float
        Total time to run (seconds).
    display_interval : float
        How often to refresh the display (seconds).
    show : bool
        Whether to show a live window with captured images.

    Returns
    -------
    None
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print(f"Saving snapshots to {directory}")
    print(f"Saving every {save_interval:.1f}s for {duration:.1f}s")

    start_time = time.time()
    next_snapshot_time = start_time
    snapshot_count = 0
    display_refresh_delay = int(display_interval*1000) # convert interval to ms (opencv units)

    try:
        while True:
            current_time = time.time()
            if current_time - start_time > duration:
                break

            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to read frame.")
                continue

            save_this_frame = False
            if current_time >= next_snapshot_time:
                timestamp = datetime.now().strftime("%m_%d_%H_%M_%S_%f")[:-3]
                filename = f"{filename_stem}_{timestamp}.png"
                path = directory / filename
                cv2.imwrite(str(path), frame)
                print(f"\t[{snapshot_count}] Saved: {path.name}")
                snapshot_count += 1
                next_snapshot_time = current_time + save_interval
                save_this_frame = True  # use to draw red circle on frames that are being captured

            if show:
                display_frame = frame.copy()
                if save_this_frame:
                    cv2.circle(display_frame, (30, 30), 25, (0, 0, 255), -1)  # draw red circle in top left corner: center, radius, color
                cv2.imshow("Snapshot", display_frame)

                key = cv2.waitKey(display_refresh_delay) & 0xFF
                if key != 255:
                    print("Key press detected — exiting early.")
                    break
            else:
                # Even if not showing, sleep to avoid tight loop
                cv2.waitKey(display_refresh_delay)

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Done. Released camera.")


if __name__ == "__main__":
    # create dir for sightsprite if it doesn't exist
    app_home = Path.home() / ".sightsprite"
    app_home.mkdir(exist_ok = True)

    print("__TESTING CAPTURE SANDBOX__")
    print("Current working directory:", os.getcwd())
    print(f"App home: {app_home}")
    print("OpenCV version:", cv2.__version__)

    # intended options: image_show, video_show, video_capture, image_capture, video_capture, capture_snapshots
    test_option = "capture_snapshots"

    if test_option == "image_show":
        try:
            show_test_image()
        except FileNotFoundError as e:
            print("\tDidn't find the file. Error:\n\t", e)
    elif test_option == "video_show":
        try:
            show_test_video()
        except FileNotFoundError as e:
            print("\tDidn't find the file. Error:\n\t", e)
    elif test_option == "image_capture":
        im_path = app_home / "captured_test.png"
        print(f"Attempting to capture to {im_path}")
        get_snapshot(im_path)
    elif test_option == "capture_snapshots":
        test_shots = app_home / "nightwatch"
        print(f"Attempting to get snapshots to {test_shots}")
        get_snapshots(test_shots, save_interval=5, duration=60, 
                      filename_stem="night2", show=True)
    elif test_option == "video_capture":
        capture_video(app_home, fps=10, duration=3)
