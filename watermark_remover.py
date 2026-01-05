"""
Video Watermark Remover
=======================
A Python script to remove watermarks from videos using various techniques:
1. Inpainting (OpenCV) - Fills watermark area with surrounding content
2. FFmpeg Delogo filter - Professional video filter for logo/watermark removal
3. Blur/Pixelate - Blur the watermark region

Requirements:
- opencv-python
- numpy
- ffmpeg (must be installed and in PATH)

Usage:
    python watermark_remover.py --input video.mp4 --output output.mp4 --method inpaint
    python watermark_remover.py --input video.mp4 --output output.mp4 --method delogo --coords 100,50,200,80
"""

import cv2
import numpy as np
import subprocess
import argparse
import os
import sys
from pathlib import Path


class WatermarkRemover:
    """Class to handle watermark removal from videos."""
    
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        self.watermark_region = None
        self.mask = None
        
    def select_watermark_region(self) -> tuple:
        """
        Opens a window to let user select the watermark region interactively.
        Supports frame navigation with arrow keys to find the watermark.
        Returns: (x, y, width, height) of the selected region
        """
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.input_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        current_frame = 0
        
        print("\n" + "="*60)
        print("WATERMARK SELECTION - FRAME NAVIGATOR")
        print("="*60)
        print("NAVIGATION CONTROLS:")
        print("  LEFT/RIGHT Arrow  : Previous/Next frame")
        print("  PAGE UP/DOWN      : Jump 30 frames back/forward")
        print("  HOME              : Go to first frame")
        print("  END               : Go to last frame")
        print("  +/-               : Jump 100 frames forward/back")
        print("")
        print("SELECTION:")
        print("  S                 : Select watermark region on current frame")
        print("  Q or ESC          : Quit without selection")
        print("="*60)
        print(f"Video: {total_frames} frames @ {fps} FPS")
        print("="*60 + "\n")
        
        # Get first frame
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError("Cannot read frame from video")
        
        # Setup display scaling
        max_display_width = 1280
        max_display_height = 720
        h, w = frame.shape[:2]
        scale = min(max_display_width / w, max_display_height / h, 1.0)
        
        window_name = "Frame Navigator - Press S to select watermark region"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        def get_display_frame(frame, frame_num, total):
            """Add frame info overlay to the frame."""
            if scale < 1.0:
                display = cv2.resize(frame, (int(w * scale), int(h * scale)))
            else:
                display = frame.copy()
            
            # Add frame counter overlay
            time_sec = frame_num / fps if fps > 0 else 0
            text = f"Frame: {frame_num}/{total-1} | Time: {time_sec:.2f}s"
            
            # Add black background for text
            cv2.rectangle(display, (5, 5), (400, 35), (0, 0, 0), -1)
            cv2.putText(display, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
            
            # Add instructions at bottom
            instructions = "LEFT/RIGHT: Navigate | S: Select | Q: Quit"
            text_size = cv2.getTextSize(instructions, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            y_pos = display.shape[0] - 10
            cv2.rectangle(display, (5, y_pos - 20), (text_size[0] + 15, y_pos + 5), (0, 0, 0), -1)
            cv2.putText(display, instructions, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (200, 200, 200), 1)
            
            return display
        
        display_frame = get_display_frame(frame, current_frame, total_frames)
        
        while True:
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(0) & 0xFF
            
            new_frame = current_frame
            
            # Key codes for arrow keys on Windows
            # Left arrow: 81, Right arrow: 83 (or 2424832, 2555904 with waitKeyEx)
            if key == ord('q') or key == 27:  # Q or ESC
                cv2.destroyAllWindows()
                cap.release()
                raise ValueError("Selection cancelled by user")
            
            elif key == ord('s') or key == ord('S'):  # S to select
                # Show ROI selection on current frame
                cv2.destroyAllWindows()
                
                if scale < 1.0:
                    selection_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                else:
                    selection_frame = frame.copy()
                
                selection_window = "Draw rectangle around watermark - ENTER to confirm"
                cv2.namedWindow(selection_window, cv2.WINDOW_NORMAL)
                
                print(f"\nSelecting on frame {current_frame}...")
                print("Click and drag to select the watermark region")
                print("Press ENTER or SPACE to confirm, C to cancel and go back")
                
                roi = cv2.selectROI(selection_window, selection_frame, 
                                   fromCenter=False, showCrosshair=True)
                cv2.destroyAllWindows()
                
                if roi[2] == 0 or roi[3] == 0:
                    # User cancelled, go back to navigation
                    print("Selection cancelled, returning to frame navigation...")
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    display_frame = get_display_frame(frame, current_frame, total_frames)
                    continue
                
                # Scale back to original coordinates
                x, y, w_roi, h_roi = roi
                if scale < 1.0:
                    x = int(x / scale)
                    y = int(y / scale)
                    w_roi = int(w_roi / scale)
                    h_roi = int(h_roi / scale)
                
                self.watermark_region = (x, y, w_roi, h_roi)
                cap.release()
                print(f"Selected region: x={x}, y={y}, width={w_roi}, height={h_roi}")
                return self.watermark_region
            
            # Navigation keys
            elif key == 81 or key == 2:  # Left arrow
                new_frame = max(0, current_frame - 1)
            elif key == 83 or key == 3:  # Right arrow
                new_frame = min(total_frames - 1, current_frame + 1)
            elif key == 85 or key == 0:  # Page Up (or Up arrow)
                new_frame = max(0, current_frame - 30)
            elif key == 86 or key == 1:  # Page Down (or Down arrow)
                new_frame = min(total_frames - 1, current_frame + 30)
            elif key == ord('=') or key == ord('+'):  # + key
                new_frame = min(total_frames - 1, current_frame + 100)
            elif key == ord('-'):  # - key
                new_frame = max(0, current_frame - 100)
            elif key == 36:  # Home key
                new_frame = 0
            elif key == 35:  # End key
                new_frame = total_frames - 1
            
            # Navigate to new frame if changed
            if new_frame != current_frame:
                current_frame = new_frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = cap.read()
                if ret:
                    display_frame = get_display_frame(frame, current_frame, total_frames)
                    print(f"Frame: {current_frame}/{total_frames-1}", end='\r')
        
        cap.release()
        cv2.destroyAllWindows()
    
    def create_mask(self, frame_shape: tuple, region: tuple, feather: int = 5) -> np.ndarray:
        """
        Creates a mask for the watermark region with optional feathering.
        """
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        x, y, w, h = region
        mask[y:y+h, x:x+w] = 255
        
        if feather > 0:
            mask = cv2.GaussianBlur(mask, (feather*2+1, feather*2+1), 0)
            
        return mask
    
    def remove_with_inpainting(self, radius: int = 5, method: str = "telea") -> None:
        """
        Remove watermark using OpenCV inpainting.
        
        Args:
            radius: Inpainting radius (higher = smoother but slower)
            method: 'telea' or 'ns' (Navier-Stokes)
        """
        if self.watermark_region is None:
            self.select_watermark_region()
        
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create mask
        mask = self.create_mask((height, width), self.watermark_region, feather=0)
        
        # Setup video writer
        temp_output = self.output_path.replace('.mp4', '_temp.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
        
        inpaint_method = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
        
        print(f"\nProcessing {total_frames} frames with inpainting...")
        print("This may take a while for long videos.\n")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply inpainting
            result = cv2.inpaint(frame, mask, radius, inpaint_method)
            out.write(result)
            
            frame_count += 1
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)", end='\r')
        
        cap.release()
        out.release()
        
        # Re-encode with FFmpeg to fix codec issues and copy audio
        print("\n\nRe-encoding with audio...")
        self._reencode_with_audio(temp_output)
        
        # Clean up temp file
        if os.path.exists(temp_output):
            os.remove(temp_output)
        
        print(f"\n✓ Watermark removed successfully!")
        print(f"  Output saved to: {self.output_path}")
    
    def remove_with_delogo(self) -> None:
        """
        Remove watermark using FFmpeg's delogo filter.
        This is faster but may leave visible artifacts.
        """
        if self.watermark_region is None:
            self.select_watermark_region()
        
        x, y, w, h = self.watermark_region
        
        # Build FFmpeg command with delogo filter
        cmd = [
            'ffmpeg', '-y',
            '-i', self.input_path,
            '-vf', f'delogo=x={x}:y={y}:w={w}:h={h}:show=0',
            '-c:a', 'copy',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '18',
            self.output_path
        ]
        
        print(f"\nRunning FFmpeg delogo filter...")
        print(f"Command: {' '.join(cmd)}\n")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"\n✓ Watermark removed successfully!")
            print(f"  Output saved to: {self.output_path}")
        except subprocess.CalledProcessError as e:
            print(f"\n✗ FFmpeg error: {e.stderr}")
            raise
        except FileNotFoundError:
            print("\n✗ FFmpeg not found! Please install FFmpeg and add it to PATH.")
            print("  Download from: https://ffmpeg.org/download.html")
            raise
    
    def remove_with_blur(self, blur_strength: int = 25) -> None:
        """
        Blur/pixelate the watermark region instead of removing it.
        Faster than inpainting but watermark may still be visible.
        """
        if self.watermark_region is None:
            self.select_watermark_region()
        
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.input_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        x, y, w, h = self.watermark_region
        
        temp_output = self.output_path.replace('.mp4', '_temp.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
        
        print(f"\nProcessing {total_frames} frames with blur...")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract and blur the watermark region
            roi = frame[y:y+h, x:x+w]
            blurred_roi = cv2.GaussianBlur(roi, (blur_strength, blur_strength), 0)
            frame[y:y+h, x:x+w] = blurred_roi
            
            out.write(frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)", end='\r')
        
        cap.release()
        out.release()
        
        print("\n\nRe-encoding with audio...")
        self._reencode_with_audio(temp_output)
        
        if os.path.exists(temp_output):
            os.remove(temp_output)
        
        print(f"\n✓ Watermark blurred successfully!")
        print(f"  Output saved to: {self.output_path}")
    
    def _reencode_with_audio(self, temp_video: str) -> None:
        """Re-encode video and copy audio from original. Falls back if FFmpeg unavailable."""
        import shutil
        
        # Check if FFmpeg is available
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            ffmpeg_available = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            ffmpeg_available = False
        
        if not ffmpeg_available:
            print("\n⚠ FFmpeg not found - saving video without audio")
            print("  To preserve audio, install FFmpeg: https://ffmpeg.org/download.html")
            # Just rename the temp file to output
            shutil.move(temp_video, self.output_path)
            return
        
        cmd = [
            'ffmpeg', '-y',
            '-i', temp_video,
            '-i', self.input_path,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '18',
            '-map', '0:v:0',
            '-map', '1:a:0?',
            '-c:a', 'aac',
            '-b:a', '192k',
            self.output_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            # If audio mapping fails, try without audio
            try:
                cmd_no_audio = [
                    'ffmpeg', '-y',
                    '-i', temp_video,
                    '-c:v', 'libx264',
                    '-preset', 'medium',
                    '-crf', '18',
                    self.output_path
                ]
                subprocess.run(cmd_no_audio, capture_output=True, check=True)
            except subprocess.CalledProcessError:
                # Last resort: just use the temp file
                print("\n⚠ FFmpeg encoding failed - saving raw video")
                shutil.move(temp_video, self.output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Remove watermarks from videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive selection with inpainting (best quality)
  python watermark_remover.py -i video.mp4 -o clean_video.mp4 -m inpaint
  
  # Using delogo filter (faster)
  python watermark_remover.py -i video.mp4 -o clean_video.mp4 -m delogo
  
  # Blur the watermark region
  python watermark_remover.py -i video.mp4 -o clean_video.mp4 -m blur
  
  # Specify coordinates directly (skip interactive selection)
  python watermark_remover.py -i video.mp4 -o clean_video.mp4 -m inpaint --coords 100,50,200,80
        """
    )
    
    parser.add_argument('-i', '--input', required=True, help='Input video file')
    parser.add_argument('-o', '--output', required=True, help='Output video file')
    parser.add_argument('-m', '--method', 
                        choices=['inpaint', 'delogo', 'blur'], 
                        default='inpaint',
                        help='Removal method (default: inpaint)')
    parser.add_argument('--coords', 
                        help='Watermark coordinates: x,y,width,height (skip interactive selection)')
    parser.add_argument('--radius', type=int, default=5, 
                        help='Inpainting radius (default: 5)')
    parser.add_argument('--blur-strength', type=int, default=25, 
                        help='Blur strength for blur method (default: 25)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize remover
    remover = WatermarkRemover(args.input, args.output)
    
    # Set coordinates if provided
    if args.coords:
        try:
            coords = tuple(map(int, args.coords.split(',')))
            if len(coords) != 4:
                raise ValueError
            remover.watermark_region = coords
            print(f"Using provided coordinates: x={coords[0]}, y={coords[1]}, w={coords[2]}, h={coords[3]}")
        except ValueError:
            print("Error: Invalid coordinates format. Use: x,y,width,height")
            sys.exit(1)
    
    # Process based on method
    print("\n" + "="*60)
    print(f"VIDEO WATERMARK REMOVER")
    print("="*60)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Method: {args.method}")
    print("="*60)
    
    try:
        if args.method == 'inpaint':
            remover.remove_with_inpainting(radius=args.radius)
        elif args.method == 'delogo':
            remover.remove_with_delogo()
        elif args.method == 'blur':
            remover.remove_with_blur(blur_strength=args.blur_strength)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
