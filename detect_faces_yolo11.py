#!/usr/bin/env python3
"""
YOLOv11 face detection using pre-trained model from HuggingFace
Draws bounding boxes on detected faces in video
"""
import cv2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import argparse
from pathlib import Path


def detect_faces_in_video(video_path, output_path=None, conf_threshold=0.25, show_display=False, play_after=False):
    """
    Process video and detect faces using YOLOv11
    
    Args:
        video_path: Path to input video
        output_path: Path to save output video (optional)
        conf_threshold: Confidence threshold for detections
        show_display: Whether to attempt showing GUI window (requires GTK support)
    """
    # Download and load pre-trained YOLOv11 face detection model
    print("Loading YOLOv11 face detection model from HuggingFace...")
    model_path = hf_hub_download(
        repo_id="AdamCodd/YOLOv11n-face-detection",
        filename="model.pt"
    )
    model = YOLO(model_path)
    
    # Auto-detect and use GPU if available
    import torch
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    model.to(device)
    print(f"Using device: {device}")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing {total_frames} frames at {fps} FPS ({width}x{height})")
    
    # Setup output writer if path provided
    writer = None
    if output_path:
        # Try different codecs for maximum compatibility
        codecs = [('mp4v', 'MPEG-4'), ('XVID', 'Xvid'), ('MJPG', 'Motion JPEG')]
        
        for codec, name in codecs:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            if writer.isOpened():
                print(f"Using {name} codec")
                break
            writer.release()
            writer = None
        
        if writer is None:
            raise RuntimeError("Failed to initialize video writer")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference on frame with explicit device
            results = model(frame, conf=conf_threshold, verbose=False, device=device)
            
            # Draw bounding boxes on frame
            annotated_frame = results[0].plot()
            
            if writer:
                writer.write(annotated_frame)
            
            # Display progress
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
            
            # Show frame and check for 'q' to quit (only if display enabled)
            if show_display:
                try:
                    cv2.imshow('Face Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Interrupted by user")
                        break
                except cv2.error:
                    pass  # No GUI support available
                
    finally:
        cap.release()
        if writer:
            writer.release()
            writer = None
        if show_display:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
    
    print(f"Completed processing {frame_count} frames")
    if output_path:
        from pathlib import Path
        import subprocess
        import shutil
        
        out_file = Path(output_path)
        if out_file.exists():
            size_mb = out_file.stat().st_size / (1024 * 1024)
            print(f"Output saved to: {output_path} ({size_mb:.2f} MB)")
            
            # Verify file can be opened
            test_cap = cv2.VideoCapture(str(output_path))
            if test_cap.isOpened():
                test_frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print(f"Verification: File contains {test_frames} frames")
                test_cap.release()
            else:
                print("WARNING: Output file cannot be opened by OpenCV")
            
            # Re-encode with ffmpeg for better compatibility
            if shutil.which('ffmpeg'):
                print("Re-encoding with ffmpeg for better compatibility...")
                temp_output = out_file.with_suffix('.tmp.mp4')
                result = subprocess.run([
                    'ffmpeg', '-y', '-i', str(output_path),
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                    '-pix_fmt', 'yuv420p', str(temp_output)
                ], capture_output=True, text=True)
                
                if result.returncode == 0 and temp_output.exists():
                    temp_output.replace(output_path)
                    new_size_mb = out_file.stat().st_size / (1024 * 1024)
                    print(f"Re-encoded successfully ({new_size_mb:.2f} MB)")
                else:
                    print(f"Re-encoding failed, keeping original file")
                    if temp_output.exists():
                        temp_output.unlink()
            
            # Play video with mpv if requested
            if play_after and shutil.which('mpv'):
                print(f"Playing video with mpv...")
                subprocess.run(['mpv', str(output_path)])
        else:
            print(f"ERROR: Output file was not created: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv11 face detection in video')
    parser.add_argument('--source', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, default=None, help='Path to output video')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--show', action='store_true', help='Show GUI window (requires GTK support)')
    parser.add_argument('--play', action='store_true', help='Play output video with mpv after processing')
    
    args = parser.parse_args()
    
    detect_faces_in_video(
        video_path=args.source,
        output_path=args.output,
        conf_threshold=args.conf,
        show_display=args.show,
        play_after=args.play
    )

