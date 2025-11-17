import cv2
import numpy as np
import glob
import os

# ==========================================
#               CONFIGURATION
# ==========================================

# 1. PATTERNS
# List of (Rows, Cols) for all board types in your room.
# Note: These are INNER CORNERS (squares - 1).
PATTERNS = [
    (5, 15), # Large board
    (11, 7), 
    (7, 5), 
    (5, 7)
]
# Sort patterns by size (biggest first) to avoid finding small parts of big boards
PATTERNS.sort(key=lambda x: x[0]*x[1], reverse=True)

# 2. SQUARE SIZE
# If you want real-world units (meters), change this to the physical size of a square.
# If you leave it as 1.0, your output units will be arbitrary "scale units".
SQUARE_SIZE = 1.0  # e.g., 0.034 for 34mm

# 3. PATHS
# Setup paths relative to this script file
script_dir = os.path.dirname(os.path.abspath(__file__))
base_data_dir = os.path.join(script_dir, '..', '34759_final_project_raw', 'calib')

LEFT_SEARCH_PATH = os.path.join(base_data_dir, 'image_02', 'data', '*.png')
RIGHT_SEARCH_PATH = os.path.join(base_data_dir, 'image_03', 'data', '*.png')
OUTPUT_DIR = os.path.join(script_dir, 'rectified_output')

# ==========================================
#             HELPER FUNCTIONS
# ==========================================

def get_board_centroid(corners):
    """Calculates the center (x,y) of a set of corners for sorting."""
    return np.mean(corners, axis=0).flatten()

def find_all_boards(img, patterns):
    """
    Iteratively detects multiple chessboards in a single image using masking.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray.copy() 
    detected_boards = []
    
    # Use Sector Based (SB) detector for wide-angle/distortion robustness
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY

    while True:
        found_in_pass = False
        for (rows, cols) in patterns:
            # Look for board in the unmasked area
            ret, corners = cv2.findChessboardCornersSB(mask, (rows, cols), flags=flags)
            
            if ret:
                detected_boards.append((corners, (rows, cols)))
                
                # Mask out this board so we can find the next one
                corners_int = corners.astype(np.int32)
                hull = cv2.convexHull(corners_int)
                
                # Draw black polygon over found board (thickness ensures edges are covered)
                cv2.fillConvexPoly(mask, hull, 0)
                cv2.polylines(mask, [hull], True, 0, thickness=25) 
                
                found_in_pass = True
                break # Restart loop to prioritize big boards again
        
        if not found_in_pass:
            break # No more boards found
            
    return detected_boards

def visualize_rectification(imgL, imgR, title="Rectified"):
    """
    Stacks images side-by-side, resized to fit on a standard screen.
    Draws green epipolar lines.
    """
    MAX_SCREEN_WIDTH = 1600 
    target_width_per_img = int(MAX_SCREEN_WIDTH / 2)
    
    h, w = imgL.shape[:2]
    scale = target_width_per_img / w
    dim = (int(w * scale), int(h * scale))
    
    rL = cv2.resize(imgL, dim, interpolation=cv2.INTER_AREA)
    rR = cv2.resize(imgR, dim, interpolation=cv2.INTER_AREA)
    
    combined = np.hstack((rL, rR))
    
    # Draw Green Lines
    for y in range(0, combined.shape[0], 25):
        cv2.line(combined, (0, y), (combined.shape[1], y), (0, 255, 0), 1)
        
    cv2.imshow(title, combined)

# ==========================================
#               MAIN SCRIPT
# ==========================================

if __name__ == "__main__":
    # 1. Load Images
    #images_left = sorted(glob.glob(LEFT_SEARCH_PATH))
    #images_right = sorted(glob.glob(RIGHT_SEARCH_PATH))
    
    # 1. Load Images (Subsampled)
    # [::3] means "take every 3rd image" to reduce dataset size
    images_left = sorted(glob.glob(LEFT_SEARCH_PATH))[::3]
    images_right = sorted(glob.glob(RIGHT_SEARCH_PATH))[::3]

    if not images_left:
        print(f"Error: No images found at {LEFT_SEARCH_PATH}")
        exit()

    objpoints = [] # 3D points in real world space
    imgpoints_l = [] # 2D points in left image
    imgpoints_r = [] # 2D points in right image
    
    img_shape = None

    print(f"Found {len(images_left)} image pairs. Processing...")

    # 2. Detection Loop
    for f_left, f_right in zip(images_left, images_right):
        imgL = cv2.imread(f_left)
        imgR = cv2.imread(f_right)
        
        if img_shape is None:
            img_shape = imgL.shape[:2][::-1] # (width, height)

        # Detect boards
        boards_L = find_all_boards(imgL, PATTERNS)
        boards_R = find_all_boards(imgR, PATTERNS)
        
        # Sort by Vertical Centroid (Y-axis) to match Left boards to Right boards
        # (Assumes stereo rig is horizontal, so boards appear in similar Y-order)
        boards_L.sort(key=lambda x: get_board_centroid(x[0])[1])
        boards_R.sort(key=lambda x: get_board_centroid(x[0])[1])

        # Only use if we found same number of boards in both
        if len(boards_L) > 0 and len(boards_L) == len(boards_R):
            print(f"--> {os.path.basename(f_left)}: Matched {len(boards_L)} boards.")
            
            for i in range(len(boards_L)):
                cornersL, (rows, cols) = boards_L[i]
                cornersR, (r_rows, r_cols) = boards_R[i]
                
                # Verify shapes match
                if (rows, cols) != (r_rows, r_cols):
                    print("    Mismatch in board sizes, skipping specific board.")
                    continue

                # Create 3D Object Grid
                objp = np.zeros((rows * cols, 3), np.float32)
                objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2) * SQUARE_SIZE
                
                objpoints.append(objp)
                imgpoints_l.append(cornersL)
                imgpoints_r.append(cornersR)
        else:
            print(f"--> {os.path.basename(f_left)}: Skipped (Count mismatch L={len(boards_L)} R={len(boards_R)})")

    # 3. Calibration
    print("\nRunning Stereo Calibration... (This takes time)")
    
    # --- NEW: Define Termination Criteria ---
    # Stop after 100 iterations OR if the error drops below 1e-5
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

    ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r, 
        None, None, None, None, img_shape,
        flags=cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_SAME_FOCAL_LENGTH,
        criteria=criteria  # <--- PASS THE CRITERIA HERE
    )
    
    print(f"Calibration RMS Error: {ret:.4f}")

    # 4. Rectification Calculation
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, img_shape, R, T, alpha=0
    )

    # 5. Save Parameters
    print("\nSaving calibration data to 'stereo_params.npz'...")
    np.savez('stereo_params.npz', K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T, R1=R1, P1=P1, R2=R2, P2=P2, Q=Q)

    # 6. Save & Visualize Rectified Images
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_shape, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_shape, cv2.CV_32FC1)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Saving rectified images to '{OUTPUT_DIR}'...")
    print("Press any key to advance visualization (or 'q' to quit early).")

    for f_left, f_right in zip(images_left, images_right):
        imgL = cv2.imread(f_left)
        imgR = cv2.imread(f_right)
        
        rect_L = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
        rect_R = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)
        
        # Save
        base_name = os.path.basename(f_left).split('.')[0]
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_L_rect.png"), rect_L)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_R_rect.png"), rect_R)
        
        # Show
        visualize_rectification(rect_L, rect_R)
        key = cv2.waitKey(0)
        if key & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Done.")