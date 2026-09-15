import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

# --- Config matches your config.py ---
X_MIN, X_MAX = -9.6, 9.6
Y_MIN, Y_MAX = 0.0, 64.0
RES = 0.1
IMG_W, IMG_H = int((X_MAX - X_MIN) / RES), int((Y_MAX - Y_MIN) / RES)
PAD_R, PAD_B = 24, 4

def generate_analytical_scene():
    """Generates a highly specific scene to show off the 3 channels."""
    pts = []
    
    # 1. Parked Car (Static Clutter)
    for _ in range(80):
        px = -5.0 + np.random.uniform(-1.0, 1.0)
        py = 35.0 + np.random.uniform(-2.4, 2.4)
        amp = np.random.uniform(1e6, 3e6) / (px**2 + py**2)
        pts.append([px, py, amp, 32.0]) # 32.0 is zero doppler

    # 2. Speeding Car (Approaching) 
    for _ in range(80):
        px = 2.5 + np.random.uniform(-1.0, 1.0)
        py = 20.0 + np.random.uniform(-2.4, 2.4)
        amp = np.random.uniform(1e6, 3e6) / (px**2 + py**2)
        v_r = - (px * 0.0 + py * -18.0) / np.sqrt(px**2 + py**2)
        pts.append([px, py, amp, (v_r / 1.435986) + 32])
        
    # 3. Pedestrian Crossing
    for _ in range(25):
        px = -1.0 + np.random.uniform(-0.4, 0.4)
        py = 28.0 + np.random.uniform(-0.4, 0.4)
        amp = np.random.uniform(2e4, 5e4) / (px**2 + py**2)
        v_r = - (px * 1.5 + py * 0.0) / np.sqrt(px**2 + py**2)
        pts.append([px, py, amp, (v_r / 1.435986) + 32])

    # 4. Metal Pole / Traffic Sign 
    for _ in range(3):
        px = -8.0 + np.random.uniform(-0.1, 0.1)
        py = 25.0 + np.random.uniform(-0.1, 0.1)
        pts.append([px, py, 5e7 / (px**2 + py**2), 32.0])
        
    # 5. Foliage / Rain Clutter 
    for _ in range(120):
        px = 7.0 + np.random.uniform(-1.5, 1.5)
        py = 35.0 + np.random.uniform(-1.5, 1.5)
        amp = np.random.uniform(100, 500) / (px**2 + py**2)
        freq = 32.0 + np.random.uniform(-0.5, 0.5) # Random micro-Doppler!
        pts.append([px, py, amp, freq])
        
    return np.array(pts)

def create_tensor(pts):
    """Processes points into the exact padded tensor YOLO sees."""
    grid = np.zeros((IMG_H, IMG_W, 3), dtype=np.float32)
    
    u_idx = np.floor((pts[:, 0] - X_MIN) / RES).astype(int).clip(0, IMG_W - 1)
    v_idx = np.floor((Y_MAX - pts[:, 1]) / RES).astype(int).clip(0, IMG_H - 1)
    
    flat_idx = v_idx * IMG_W + u_idx
    counts = np.bincount(flat_idx, minlength=IMG_H*IMG_W)
    grid[:, :, 0] = counts.reshape((IMG_H, IMG_W))
    
    r_sq = pts[:, 0]**2 + pts[:, 1]**2
    log_amp = np.log1p(pts[:, 2] * r_sq)
    raw_vel = (pts[:, 3] - 32) * 1.435986
    skewed_vel = np.sign(raw_vel) * np.sqrt(np.abs(raw_vel))
    
    for i in range(len(u_idx)):
        u, v = u_idx[i], v_idx[i]
        grid[v, u, 1] = max(grid[v, u, 1], log_amp[i])
        grid[v, u, 2] = skewed_vel[i] if abs(skewed_vel[i]) > abs(grid[v, u, 2]) else grid[v, u, 2]

    ch0 = np.clip(grid[:, :, 0] / 5.0, 0, 1) * 255
    ch1 = np.clip(grid[:, :, 1] / 15.0, 0, 1) * 255
    ch2 = np.clip((grid[:, :, 2] / 5.5) * 127 + 127, 0, 255)
    ch2[grid[:, :, 0] == 0] = 0  
    
    img = np.stack([ch0, ch1, ch2], axis=-1).astype(np.uint8)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    ch0_dil = cv2.dilate(img[:, :, 0], kernel, iterations=1)
    ch1_dil = cv2.dilate(img[:, :, 1], kernel, iterations=1)
    ch2_dil = cv2.dilate(img[:, :, 2], kernel, iterations=1)
    
    img_spread = np.stack([
        ch0_dil, 
        np.where(ch0_dil > 0, ch1_dil, 0), 
        np.where(ch0_dil > 0, ch2_dil, 127)
    ], axis=-1).astype(np.uint8)
    
    return cv2.copyMakeBorder(img_spread, 0, PAD_B, 0, PAD_R, cv2.BORDER_CONSTANT, value=[0, 0, 127])

# --- Generate and Crop Data ---
pts = generate_analytical_scene()
tensor = create_tensor(pts)
row_start = int((Y_MAX - 45.0) / RES) # ~190
row_end = int((Y_MAX - 10.0) / RES)   # ~540
tensor_cropped = tensor[row_start:row_end, :]

# --- Custom Colormaps to match the final RGB image perfectly ---
cmap_b = LinearSegmentedColormap.from_list('BlackBlue', ['black', 'dodgerblue'])
cmap_g = LinearSegmentedColormap.from_list('BlackGreen', ['black', 'lime'])
cmap_r = LinearSegmentedColormap.from_list('BlackRed', ['black', 'red'])

# --- Plotting ---
bg_color = '#0d1117'
fig = plt.figure(figsize=(16, 8), facecolor=bg_color)
gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1.2], wspace=0.15)

titles = [
    "Ch 0: Point Density\n(Mapped to Blue)", 
    "Ch 1: Radar RCS\n(Mapped to Green)", 
    "Ch 2: Radial Velocity\n(Mapped to Red)", 
    "Final Merged Tensor\n(YOLOv11 Input)"
]
cmaps = [cmap_b, cmap_g, cmap_r, None]

# Remember: OpenCV natively stacks as B, G, R
channels = [
    tensor_cropped[:,:,0], # Density (Blue)
    tensor_cropped[:,:,1], # Amp (Green)
    tensor_cropped[:,:,2], # Doppler (Red)
    cv2.cvtColor(tensor_cropped, cv2.COLOR_BGR2RGB)
]

for i in range(4):
    ax = fig.add_subplot(gs[0, i])
    ax.set_title(titles[i], color='white', fontsize=13, pad=15, fontweight='bold')
    
    if i < 3:
        im = ax.imshow(channels[i], cmap=cmaps[i], vmin=0, vmax=255)
    else:
        ax.imshow(channels[i])
    
    ax.axis('off')

    # Add perfectly aligned boxes on the 4th panel using exact metric math
    if i == 3:
        def draw_box(x_m, y_m, w_m, h_m, color, label, text_offset):
            # Convert physical meters to cropped pixel coordinates
            px_x = int((x_m - X_MIN) / RES)
            px_y = int((Y_MAX - y_m) / RES) - row_start
            pw, ph = int(w_m / RES), int(h_m / RES)
            
            ax.add_patch(patches.Rectangle((px_x - pw//2, px_y - ph//2), pw, ph, 
                                           fill=False, edgecolor=color, lw=1.5, ls='--'))
            ax.text(px_x + text_offset[0], px_y + text_offset[1], label, 
                    color=color, fontsize=9, va='center', ha='left')

        # 1. Parked Car
        draw_box(-5.0, 35.0, 2.0, 4.8, 'cyan', "Parked Car\n(Neutral Doppler)", (13, 0))
        # 2. Speeding Car
        draw_box(2.5, 20.0, 2.0, 4.8, 'orange', "Approaching Car\n(High Amplitude,\nShifted Doppler)", (13, 0))
        # 3. Pedestrian
        draw_box(-1.0, 28.0, 0.8, 0.8, 'lime', "Crossing Pedestrian\n(Sparse/Low RCS)", (6, -6))
        # 4. Metal Pole
        draw_box(-8.0, 25.0, 0.8, 0.8, 'white', "Metal Pole\n(Low Density,\nMassive RCS)", (6, 0))
        # 5. Foliage
        draw_box(7.0, 35.0, 3.0, 3.0, '#aaaaaa', "Foliage/Clutter\n(High Density, Weak RCS,\nRandom Micro-Doppler)", (-40, 35))

plt.suptitle("Multi-Channel Physics Encoding", color='white', fontsize=20, fontweight='bold', y=0.98)
fig.text(0.5, 0.05, "Synthetic Feature Map Visualization cropped to active 35m ROI. Zero-Doppler padded regions encode to 127 (Neutral Red).", 
         ha='center', color='#8b949e', fontsize=11, fontstyle='italic')

plt.savefig('radar_feature_engineering.png', dpi=300, bbox_inches='tight', facecolor=bg_color)
print("Saved perfectly aligned, color-matched infographic to radar_feature_engineering.png!")