import cv2
import numpy as np
from pathlib import Path

def create_physics_breakdown(bev_image_path: str, output_path: str):
    # Load the 3-channel BEV image
    img = cv2.imread(bev_image_path)
    if img is None:
        print(f"Could not load image at {bev_image_path}")
        return

    h, w, _ = img.shape
    
    # Split channels (OpenCV uses BGR)
    b, g, r = cv2.split(img)
    
    # Create zero arrays for empty channels
    zeros = np.zeros_like(b)
    
    # Reconstruct individual color channels for display
    img_r = cv2.merge([zeros, zeros, r]) # Channel 0: Occupancy (Red)
    img_g = cv2.merge([zeros, g, zeros]) # Channel 1: Amplitude (Green)
    img_b = cv2.merge([b, zeros, zeros]) # Channel 2: Doppler (Blue)
    
    # Create a dark canvas to hold the original + 3 splits
    pad = 40
    canvas_w = (w * 4) + (pad * 5)
    canvas_h = h + 120
    canvas = np.full((canvas_h, canvas_w, 3), 20, dtype=np.uint8) # Dark gray background
    
    # Define text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    text_color = (220, 220, 220)
    
    titles = [
        "Combined Radar BEV Tensor", 
        "Ch 0 (Red): Occupancy Density", 
        "Ch 1 (Green): R-Sq Amplitude", 
        "Ch 2 (Blue): Skewed Doppler Velocity"
    ]
    images = [img, img_r, img_g, img_b]
    
    for i, (title, patch) in enumerate(zip(titles, images)):
        x_offset = pad + i * (w + pad)
        y_offset = 80
        
        # Paste image
        canvas[y_offset:y_offset+h, x_offset:x_offset+w] = patch
        
        # Add Title
        (tw, th), _ = cv2.getTextSize(title, font, font_scale, thickness)
        cv2.putText(canvas, title, (x_offset + (w - tw)//2, 50), font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        # Draw sleek border
        cv2.rectangle(canvas, (x_offset-2, y_offset-2), (x_offset+w+1, y_offset+h+1), (100, 100, 100), 1)

    # Save
    cv2.imwrite(output_path, canvas)
    print(f"Physics breakdown saved to {output_path}")

if __name__ == "__main__":
    # TODO: Point this to a real BEV image in your measurements folder
    SAMPLE_BEV = "measurements/20260506-161743_mix/bev_images/18600.png" # Example path
    create_physics_breakdown(SAMPLE_BEV, "physics_breakdown.png")