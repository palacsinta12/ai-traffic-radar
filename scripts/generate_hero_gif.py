import numpy as np
import cv2
import imageio
from tqdm import tqdm

# --- Config matches your config.py ---
X_MIN, X_MAX = -9.6, 9.6
Y_MIN, Y_MAX = 0.0, 64.0
RES = 0.1
IMG_W, IMG_H = int((X_MAX - X_MIN) / RES), int((Y_MAX - Y_MIN) / RES)
PAD_R, PAD_B = 24, 4

# YOLO Class Colors (BGR)
C_PED = (0, 200, 0)      # Green
C_CAR = (0, 165, 255)    # Orange
C_CYC = (0, 220, 100)    # Yellow-Green

class RigidTarget:
    def __init__(self, cls_id, cls_name, w, l, x, y, vx, vy, base_amp, density, color):
        self.cls_id = cls_id
        self.cls_name = cls_name
        self.w, self.l = w, l
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.base_amp = base_amp
        self.color = color
        self.conf = np.random.uniform(0.88, 0.96)
        
        # Generate rigid body points ONCE so they don't "wiggle"
        num_pts = int(density * (w * l))
        self.local_pts = np.random.uniform(
            low=[-w/2, -l/2], high=[w/2, l/2], size=(num_pts, 2)
        )

    def step(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt

    def get_points(self):
        world_pts = self.local_pts + np.array([self.x, self.y])
        pts = []
        for px, py in world_pts:
            if py <= 0.5: continue # Radar blind spot
            
            r_sq = px**2 + py**2
            amp = np.random.uniform(self.base_amp * 0.8, self.base_amp * 1.2) / r_sq
            
            # Physics: Radial Velocity = negative dot product of pos and vel unit vector
            v_r = - (px * self.vx + py * self.vy) / np.sqrt(r_sq)
            freq = (v_r / 1.435986) + 32
            pts.append([px, py, amp, freq])
        return pts

def generate_static_guardrails():
    """Generates non-wiggling static clutter along the road edges."""
    pts = []
    for y in np.arange(2.0, 62.0, 1.5):
        for x in [-8.5, 8.5]:
            amp = np.random.uniform(1e3, 5e3) / (x**2 + y**2)
            pts.append([x + np.random.normal(0, 0.05), y, amp, 32.0])
    return pts

STATIC_CLUTTER = generate_static_guardrails()

def render_panel(targets):
    """Renders a single BEV tensor panel."""
    all_pts = list(STATIC_CLUTTER)
    for t in targets:
        all_pts.extend(t.get_points())
        
    grid = np.zeros((IMG_H, IMG_W, 3), dtype=np.float32)
    
    if not all_pts:
        return cv2.copyMakeBorder(np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8), 
                                  0, PAD_B, 0, PAD_R, cv2.BORDER_CONSTANT, value=[0, 0, 127])
        
    pts = np.array(all_pts)
    valid_mask = (pts[:, 0] >= X_MIN) & (pts[:, 0] <= X_MAX) & (pts[:, 1] >= Y_MIN) & (pts[:, 1] <= Y_MAX)
    pts = pts[valid_mask]
    
    if len(pts) == 0:
        return cv2.copyMakeBorder(np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8), 
                                  0, PAD_B, 0, PAD_R, cv2.BORDER_CONSTANT, value=[0, 0, 127])

    u_idx = np.floor((pts[:, 0] - X_MIN) / RES).astype(int)
    v_idx = np.floor((Y_MAX - pts[:, 1]) / RES).astype(int)
    
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

    
    ch0 = np.clip(grid[:, :, 0] / 8.0, 0, 1) * 120
    ch1 = np.clip(grid[:, :, 1] / 18.0, 0, 1) * 150
    ch2 = np.clip((grid[:, :, 2] / 4.0) * 127 + 127, 0, 255)

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
    
    padded = cv2.copyMakeBorder(img_spread, 0, PAD_B, 0, PAD_R, cv2.BORDER_CONSTANT, value=[0, 0, 127])
    
    for t in targets:
        if t.y < -3.0 or t.y > Y_MAX + 3.0: continue
            
        px_cx = int((t.x - X_MIN) / RES)
        px_cy = int((Y_MAX - t.y) / RES)
        px_w, px_l = int(t.w / RES), int(t.l / RES)
        
        x1, y1 = px_cx - px_w//2, px_cy - px_l//2
        x2, y2 = px_cx + px_w//2, px_cy + px_l//2
        
        cv2.rectangle(padded, (x1, y1), (x2, y2), t.color, 1)
        speed = abs(t.vy) if abs(t.vy) > 0.1 else abs(t.vx)
        label = f"{t.cls_name} {t.conf:.2f} | {speed:.1f}m/s"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(padded, (x1, y1 - th - 4), (x1 + tw, y1), t.color, -1)
        cv2.putText(padded, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
        
        t.conf = np.clip(t.conf + np.random.uniform(-0.01, 0.01), 0.75, 0.98)

    return padded

# ==========================================
# 6.0 SECONDS = 150 FRAMES @ 25 FPS
# ==========================================
FRAMES = 150
FPS = 25
dt = 1.0 / FPS

# --- SCENARIO A: VRU Intersection ---
# Car will start fast, brake to a crawl to yield, then speed up again
'''scenario_A = [
    RigidTarget(1, "Car", 2.0, 4.8, x=-2.0, y=60.0, vx=0.0, vy=-15.0, base_amp=2e6, density=20, color=C_CAR),
    RigidTarget(2, "Cyc", 1.0, 2.0, x=3.5, y=5.0, vx=0.0, vy=9.0, base_amp=1e5, density=15, color=C_CYC),
    RigidTarget(0, "Ped", 0.8, 0.8, x=-8.0, y=25.0, vx=1.8, vy=0.0, base_amp=2e4, density=25, color=C_PED)
]'''
scenario_A = [
    RigidTarget(1, "Car", 2.0, 4.8, x=-2.0, y=60.0, vx=0.0, vy=-15.0, base_amp=8e4, density=8, color=C_CAR),
    RigidTarget(2, "Cyc", 1.0, 2.0, x=3.5, y=5.0, vx=0.0, vy=9.0, base_amp=1e5, density=15, color=C_CYC),
    RigidTarget(0, "Ped", 0.8, 0.8, x=-8.0, y=25.0, vx=1.8, vy=0.0, base_amp=2e4, density=25, color=C_PED)
]


# --- SCENARIO B: High-Speed Loop ---
scenario_B = [
    RigidTarget(1, "Car", 2.0, 4.8, x=2.5, y=77.0, vx=0.0, vy=-15.0, base_amp=2e6, density=20, color=C_CAR),
    RigidTarget(1, "Car", 2.0, 4.8, x=-2.5, y=-13.0, vx=0.0, vy=15.0, base_amp=2e6, density=20, color=C_CAR)
]

# --- SCENARIO C: Cluttered Environment ---
scenario_C = [
    RigidTarget(1, "Car", 2.0, 4.8, x=-5.0, y=15.0, vx=0.0, vy=0.0, base_amp=2e6, density=20, color=C_CAR),
    RigidTarget(1, "Car", 2.0, 4.8, x=-5.0, y=35.0, vx=0.0, vy=0.0, base_amp=2e6, density=20, color=C_CAR),
    RigidTarget(1, "Car", 2.0, 4.8, x=-5.0, y=55.0, vx=0.0, vy=0.0, base_amp=2e6, density=20, color=C_CAR),
    RigidTarget(1, "Car", 2.0, 4.8, x=2.0, y=2.0, vx=0.0, vy=8.0, base_amp=2e6, density=20, color=C_CAR),
    RigidTarget(0, "Ped", 0.8, 0.8, x=-4.0, y=26.0, vx=1.1, vy=0.4, base_amp=2e4, density=25, color=C_PED),
    RigidTarget(2, "Cyc", 1.0, 2.0, x=6.5, y=-8.0, vx=-0.3, vy=11.0, base_amp=1e5, density=15, color=C_CYC)
]

frames = []

print("Simulating 3 parallel traffic scenarios for 6.0 seconds...")
for i in tqdm(range(FRAMES), desc="Rendering frames"):
    
    # ---------------------------------------------------------
    # DYNAMIC VELOCITY CONTROLLER for Scenario A (The Yielding Car)
    # ---------------------------------------------------------
    t_sec = i * dt
    yielding_car = scenario_A[0]
    
    if t_sec < 1.2:
        yielding_car.vy = -15.0  # Fast approach
    elif t_sec < 2.1:
        progress = (t_sec - 1.2) / 0.9
        yielding_car.vy = -15.0 + (13.0 * progress)  # Decelerate smoothly
    elif t_sec < 3.8:
        yielding_car.vy = -2.0   # Crawl/Yield while ped passes safely
    else:
        progress = (t_sec - 3.8) / 1.5
        yielding_car.vy = max(-15.0, -2.0 - (13.0 * progress))  # Accelerate away
    # ---------------------------------------------------------

    # Render panels
    panel_A = render_panel(scenario_A)
    panel_B = render_panel(scenario_B)
    panel_C = render_panel(scenario_C)
    
    # Combine horizontally with 4px borders
    border = np.zeros((panel_A.shape[0], 4, 3), dtype=np.uint8)
    border[:] = (40, 40, 40)
    
    combined = np.hstack([panel_A, border, panel_B, border, panel_C])
    
    # Add Dashboard Header
    header = np.zeros((40, combined.shape[1], 3), dtype=np.uint8)
    header[:] = (20, 20, 20)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(header, "VRU Intersection", (40, 25), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(header, "High-Speed Passing", (panel_A.shape[1] + 30, 25), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(header, "Cluttered Environment", ((panel_A.shape[1]*2) + 20, 25), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    
    final_frame = np.vstack([header, combined])
    frames.append(cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB))
    
    # Step physics
    for t in scenario_A + scenario_B + scenario_C:
        t.step(dt)

imageio.mimsave('hero_dashboard_extended.gif', frames, fps=FPS, loop=0)
print("Saved dynamic dashboard animation to hero_dashboard_extended.gif!")