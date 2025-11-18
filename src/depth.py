import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
# Get the path to the 'src' folder
SRC_DIR = Path(__file__).resolve().parent
# Go up one level to get the Project Root (DTU-34759-Final-Project)
REPO_ROOT = SRC_DIR.parent

# FIX: The dataset folders are directly inside the repo root, not a subfolder
DATASET_ROOT = REPO_ROOT 
OUTPUT_ROOT = REPO_ROOT / "output"

# Default Calibration Files
DEFAULT_NPZ = SRC_DIR / "stereo_params.npz"
DEFAULT_TXT = DATASET_ROOT / "34759_final_project_rect" / "calib_cam_to_cam.txt"

# Input Directories
DIR_RAW = DATASET_ROOT / "34759_final_project_raw"
DIR_RECT = DATASET_ROOT / "34759_final_project_rect"

# Sequences to process
SEQUENCES = ["seq_01", "seq_02", "seq_03"]

class StereoDepthProcessor:
    def __init__(self, algorithm='sgbm'):
        self.map1x, self.map1y = None, None
        self.map2x, self.map2y = None, None
        self.Q = None
        
        # Tuned SGBM Parameters for outdoor driving scenes
        min_disp = 0
        num_disp = 16 * 10  # 160 disparity levels
        block_size = 11
        
        if algorithm == 'sgbm':
            self.stereo = cv2.StereoSGBM_create(
                minDisparity=min_disp,
                numDisparities=num_disp,
                blockSize=block_size,
                P1=8 * 3 * block_size**2,
                P2=32 * 3 * block_size**2,
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )
        else:
            self.stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_size)

    def init_rectification_map(self, calib_data, image_size):
        """
        Initialize maps for RAW images that need rectification.
        """
        M1, D1 = calib_data['M1'], calib_data['D1']
        M2, D2 = calib_data['M2'], calib_data['D2']
        
        # Check if we have pre-calculated rectification matrices (from custom NPZ)
        if all(k in calib_data for k in ['R1', 'P1', 'R2', 'P2', 'Q']):
            print("   -> Using pre-calculated rectification matrices (R1, P1, Q...)")
            R1, P1 = calib_data['R1'], calib_data['P1']
            R2, P2 = calib_data['R2'], calib_data['P2']
            self.Q = calib_data['Q']
        else:
            print("   -> Computing rectification matrices via stereoRectify...")
            R, T = calib_data['R'], calib_data['T']
            R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
                M1, D1, M2, D2, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
            )
            self.Q = Q
        
        # Initialize mapping for Remap
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(M1, D1, R1, P1, image_size, cv2.CV_32FC1)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(M2, D2, R2, P2, image_size, cv2.CV_32FC1)

    def init_for_rectified_images(self, calib_data):
        """
        Setup for images that are ALREADY rectified (from the '_rect' folders).
        We don't need remap maps. We just need Q for depth calculation.
        """
        print("   -> Initializing for ALREADY RECTIFIED images (No Remap).")
        self.map1x, self.map1y = None, None
        self.map2x, self.map2y = None, None
        
        if 'P1' in calib_data and 'P2' in calib_data:
            P1 = calib_data['P1'] # Left
            P2 = calib_data['P2'] # Right
            
            # Extract parameters to build Q matrix manually
            # P = [f  0 cx Tx*f]
            fx = P1[0, 0]
            fy = P1[1, 1] # Should be same as fx usually
            cx = P1[0, 2]
            cy = P1[1, 2]
            
            # Baseline Calculation
            # For KITTI/Cam_to_Cam: P_right_x = P_left_x + B*f
            # So B*f = P2[0,3] - P1[0,3]  => B = (P2[0,3] - P1[0,3]) / fx
            # Note: Usually Tx in P is (baseline * focal_length)
            T1x = P1[0, 3]
            T2x = P2[0, 3]
            baseline = (T2x - T1x) / fx # result in meters (if P is in meters) or units of T
            
            # Construct Q matrix
            # Q = [[1, 0, 0, -cx], [0, 1, 0, -cy], [0, 0, 0, f], [0, 0, -1/Tx, (cx - cx')/Tx]]
            # Simpler approximation for Depth Z = f * B / d
            # We can put this into Q format so reprojectImageTo3D works.
            # Q[3, 2] = -1.0 / baseline
            # Q[3, 3] = 0
            
            self.Q = np.float32([
                [1, 0, 0, -cx],
                [0, 1, 0, -cy],
                [0, 0, 0, fx],
                [0, 0, -1.0/baseline, 0] # 1/Tx term
            ])
        else:
             print("   [ERROR] Missing P1/P2 for rectified setup. Depth will be wrong.")

    def process_frame(self, imgL, imgR, is_already_rectified=False):
        """
        Returns:
        - rect_imgL, rect_imgR: The images (remapped if needed, else passed through)
        - disparity: The disparity map (in pixels)
        - depth_map: The calculated depth (Z) in meters
        """
        
        if is_already_rectified:
            rect_imgL = imgL
            rect_imgR = imgR
        else:
            if self.map1x is None: raise ValueError("Maps not initialized for Raw processing.")
            # Rectify Raw Images
            rect_imgL = cv2.remap(imgL, self.map1x, self.map1y, cv2.INTER_LINEAR)
            rect_imgR = cv2.remap(imgR, self.map2x, self.map2y, cv2.INTER_LINEAR)
        
        # 2. Compute Disparity
        grayL = cv2.cvtColor(rect_imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(rect_imgR, cv2.COLOR_BGR2GRAY)
        
        # Result is fixed-point (multiplied by 16), so divide by 16.0
        disparity = self.stereo.compute(grayL, grayR).astype(np.float32) / 16.0
        
        # 3. Compute 3D Depth (Z in meters)
        depth_map = None
        if self.Q is not None:
            # reprojectImageTo3D returns (X, Y, Z) for each pixel
            points_3d = cv2.reprojectImageTo3D(disparity, self.Q)
            depth_map = points_3d[:, :, 2]
            
            # Filter infinite/bad values
            depth_map[depth_map > 100] = 100 # Cap max depth
            depth_map[depth_map < 0] = 0     # Remove negative
        
        return rect_imgL, rect_imgR, disparity, depth_map

    def load_calibration_from_npz(self, filepath):
        print(f"   -> Loading NPZ: {filepath.name}")
        try:
            data = np.load(filepath)
            calib = {
                'M1': data['K1'], 'D1': data['D1'],
                'M2': data['K2'], 'D2': data['D2'],
                'R': data['R'],   'T': data['T']
            }
            if 'R1' in data and 'Q' in data:
                calib.update({
                    'R1': data['R1'], 'P1': data['P1'],
                    'R2': data['R2'], 'P2': data['P2'],
                    'Q': data['Q']
                })
            return calib
        except Exception as e:
            print(f"   [ERROR] Loading .npz: {e}")
            return None

    def load_calibration_from_txt_rectified(self, filepath, left_id='02', right_id='03'):
        """
        Loads specifically P_rect_xx matrices for calculating Depth from 
        already rectified images.
        """
        print(f"   -> Loading TXT (Rectified Mode): {filepath.name}")
        params = {}
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or ':' not in line: continue
                    key, val_str = line.split(':', 1)
                    params[key] = np.fromstring(val_str, sep=' ')

            def get_mat(prefix, cid, shape): return params[f"{prefix}_{cid}"].reshape(shape)

            # For rectified images, we need P_rect_02 and P_rect_03
            P1 = get_mat('P_rect', left_id, (3, 4))
            P2 = get_mat('P_rect', right_id, (3, 4))
            
            return {'P1': P1, 'P2': P2}
            
        except Exception as e:
            print(f"   [ERROR] Loading .txt: {e}")
            return None

def save_results(output_root, seq_name, method_label, file_stem, rect_L, rect_R, disparity, depth):
    base_dir = output_root / seq_name / method_label
    
    dirs = {
        "img_L": base_dir / "rectified_image_02", # Only useful if we rectified them ourselves
        "disp_vis": base_dir / "disparity_vis",
        "depth_npy": base_dir / "depth_maps",
    }
    
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Save Rectified Left (Reference)
    cv2.imwrite(str(dirs["img_L"] / f"{file_stem}.png"), rect_L)
    
    # Save Disparity Visualization
    disp_vis = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_PLASMA)
    cv2.imwrite(str(dirs["disp_vis"] / f"{file_stem}.png"), disp_color)
    
    # Save RAW Depth Map
    if depth is not None:
        np.save(str(dirs["depth_npy"] / f"{file_stem}.npy"), depth)

def process_sequence(processor, calibration_data, input_root_dir, seq_name, output_root, method_label, is_already_rectified, limit=None):
    if not calibration_data: return

    seq_dir = input_root_dir / seq_name
    left_dir = seq_dir / "image_02" / "data"
    right_dir = seq_dir / "image_03" / "data"

    if not left_dir.exists():
        print(f"[SKIP] Sequence not found at: {left_dir}")
        return

    print(f"\n=== Processing {seq_name} [{method_label}] ===")
    
    left_images = sorted(list(left_dir.glob("*.png")))
    if limit: left_images = left_images[:limit]

    if not left_images:
        print("No images found.")
        return

    # Initialize
    first_img = cv2.imread(str(left_images[0]))
    h, w = first_img.shape[:2]
    
    if is_already_rectified:
        processor.init_for_rectified_images(calibration_data)
    else:
        processor.init_rectification_map(calibration_data, (w, h))

    for i, l_path in enumerate(left_images):
        file_stem = l_path.stem
        r_path = right_dir / l_path.name
        
        if not r_path.exists(): continue

        imgL = cv2.imread(str(l_path))
        imgR = cv2.imread(str(r_path))

        # Process
        rect_L, _, disp, depth = processor.process_frame(imgL, imgR, is_already_rectified)
        
        # Save
        save_results(output_root, seq_name, method_label, file_stem, rect_L, None, disp, depth)
        
        if i % 20 == 0:
            print(f"   {i}/{len(left_images)}: {file_stem}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--txt", type=Path, default=DEFAULT_TXT)
    parser.add_argument("--limit", type=int, default=None, help="Limit frames per seq")
    args = parser.parse_args()

    print("--- Configuration Paths ---")
    print(f"Repo Root:    {REPO_ROOT}")
    print(f"Dataset Root: {DATASET_ROOT}")
    print(f"Raw Data Dir: {DIR_RAW}")
    print(f"Rect Data Dir:{DIR_RECT}")
    print("-------------------------")

    processor = StereoDepthProcessor(algorithm='sgbm')

    # --- PREPARE CALIBRATIONS ---
    
    # 1. Custom Calibration for RAW images
    calib_custom = None
    if args.npz.exists():
        calib_custom = processor.load_calibration_from_npz(args.npz)
    else:
        print(f"[WARN] Custom calibration not found at {args.npz}")
    
    # 2. Standard Calibration for RECTIFIED images
    calib_standard = None
    if args.txt.exists():
        # We only need P matrices for depth calc here
        calib_standard = processor.load_calibration_from_txt_rectified(args.txt)
    else:
         print(f"[WARN] Standard calibration not found at {args.txt}")

    # --- PROCESS SEQUENCES ---
    for seq in SEQUENCES:
        
        # METHOD 1: RAW Images + Custom Calibration -> Rectify -> Depth
        if calib_custom:
            process_sequence(
                processor, 
                calib_custom, 
                DIR_RAW,        # Input: Raw Folder
                seq, 
                OUTPUT_ROOT, 
                "custom_raw_calibration", 
                is_already_rectified=False, # Needs Rectification
                limit=args.limit
            )

        # METHOD 2: RECTIFIED Images + Standard Calib -> Depth
        if calib_standard:
            process_sequence(
                processor, 
                calib_standard, 
                DIR_RECT,       # Input: Rect Folder
                seq, 
                OUTPUT_ROOT, 
                "standard_rect_calibration", 
                is_already_rectified=True,  # Already Rectified
                limit=args.limit
            )

    print("\nProcessing Complete.")
    print(f"Results saved to: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()