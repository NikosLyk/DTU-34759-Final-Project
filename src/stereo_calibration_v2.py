import cv2
import numpy as np
import glob
import os
import math

# ==========================================
#               CONFIGURATION
# ==========================================

# 1. PATTERNS (INNER CORNERS)
PATTERNS = [
    (5, 15), 
    (11, 7), 
    (7, 5), 
    (5, 7)
]
PATTERNS.sort(key=lambda x: x[0]*x[1], reverse=True)

# 2. SQUARE SIZE
SQUARE_SIZE = 0.095  # 9.5 cm

# 3. SELECTION
MAX_BOARDS_PER_FRAME = 20 

# 4. PATHS
script_dir = os.path.dirname(os.path.abspath(__file__))
base_data_dir = os.path.join(script_dir, '..', '34759_final_project_raw', 'calib')

LEFT_SEARCH_PATH = os.path.join(base_data_dir, 'image_02', 'data', '*.png')
RIGHT_SEARCH_PATH = os.path.join(base_data_dir, 'image_03', 'data', '*.png')
OUTPUT_DIR = os.path.join(script_dir, 'rectified_output')

# ==========================================
#             HELPER FUNCTIONS
# ==========================================

def get_board_centroid(corners):
    return np.mean(corners, axis=0).flatten()

def get_dist_from_center(corners, img_w, img_h):
    c_x, c_y = get_board_centroid(corners)
    return math.sqrt((c_x - img_w/2.0)**2 + (c_y - img_h/2.0)**2)

def find_all_boards(img, patterns):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray.copy() 
    detected_boards = []
    
    # High accuracy flags
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY

    while True:
        found_in_pass = False
        for (rows, cols) in patterns:
            ret, corners = cv2.findChessboardCornersSB(mask, (rows, cols), flags=flags)
            
            if ret:
                detected_boards.append((corners, (rows, cols)))
                corners_int = corners.astype(np.int32)
                hull = cv2.convexHull(corners_int)
                cv2.fillConvexPoly(mask, hull, 0)
                cv2.polylines(mask, [hull], True, 0, thickness=25) 
                found_in_pass = True
                break 
        
        if not found_in_pass:
            break 
            
    return detected_boards

def visualize_rectification(imgL, imgR, title="Rectified"):
    MAX_SCREEN_WIDTH = 1600 
    target_width_per_img = int(MAX_SCREEN_WIDTH / 2)
    h, w = imgL.shape[:2]
    scale = target_width_per_img / w
    dim = (int(w * scale), int(h * scale))
    rL = cv2.resize(imgL, dim, interpolation=cv2.INTER_AREA)
    rR = cv2.resize(imgR, dim, interpolation=cv2.INTER_AREA)
    combined = np.hstack((rL, rR))
    for y in range(0, combined.shape[0], 25):
        cv2.line(combined, (0, y), (combined.shape[1], y), (0, 255, 0), 1)
    cv2.imshow(title, combined)

# ==========================================
#               MAIN SCRIPT
# ==========================================

if __name__ == "__main__":
    # 1. Load Images
    images_left = sorted(glob.glob(LEFT_SEARCH_PATH))[::3]
    images_right = sorted(glob.glob(RIGHT_SEARCH_PATH))[::3]

    if not images_left:
        print(f"Error: No images found at {LEFT_SEARCH_PATH}")
        exit()

    objpoints = [] 
    imgpoints_l = [] 
    imgpoints_r = [] 
    img_shape = None

    print(f"Found {len(images_left)} image pairs. Detecting boards...")

    # 2. Detection Loop
    for f_left, f_right in zip(images_left, images_right):
        imgL = cv2.imread(f_left)
        imgR = cv2.imread(f_right)
        
        if img_shape is None:
            img_shape = imgL.shape[:2][::-1]

        boards_L = find_all_boards(imgL, PATTERNS)
        boards_R = find_all_boards(imgR, PATTERNS)
        
        boards_L.sort(key=lambda x: get_board_centroid(x[0])[1])
        boards_R.sort(key=lambda x: get_board_centroid(x[0])[1])

        if len(boards_L) > 0 and len(boards_L) == len(boards_R):
            # Match and Filter
            candidates = []
            for i in range(len(boards_L)):
                cornersL, (rows, cols) = boards_L[i]
                cornersR, (r_rows, r_cols) = boards_R[i]
                
                if (rows, cols) != (r_rows, r_cols): continue
                
                dist = get_dist_from_center(cornersL, img_shape[0], img_shape[1])
                candidates.append({'dist': dist, 'cL': cornersL, 'cR': cornersR, 'r': rows, 'c': cols})

            candidates.sort(key=lambda x: x['dist'])
            final_selection = candidates[:MAX_BOARDS_PER_FRAME]
            
            for item in final_selection:
                rows, cols = item['r'], item['c']
                objp = np.zeros((rows * cols, 3), np.float32)
                objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2) * SQUARE_SIZE
                
                objpoints.append(objp)
                imgpoints_l.append(item['cL'])
                imgpoints_r.append(item['cR'])
        else:
            pass

    print(f"Collected {len(objpoints)} board views. Starting 2-Step Calibration...")

    # ==========================================
    #   STEP 1: INDIVIDUAL CAMERA CALIBRATION
    # ==========================================
    print("\n[Step 1] Calibrating Single Cameras...")
    
    # Standard Rational Model flags for single camera
    single_flags = (cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_PRINCIPAL_POINT)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

    # Calibrate Left
    retL, K1, D1, _, _ = cv2.calibrateCamera(objpoints, imgpoints_l, img_shape, None, None, flags=single_flags, criteria=criteria)
    print(f"  -> Left Camera RMS: {retL:.4f}")

    # Calibrate Right
    retR, K2, D2, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, img_shape, None, None, flags=single_flags, criteria=criteria)
    print(f"  -> Right Camera RMS: {retR:.4f}")

    # ==========================================
    #   STEP 2: STEREO CALIBRATION (FIXED INTRINSICS)
    # ==========================================
    print("\n[Step 2] Calibrating Stereo Extrinsics (using intrinsics from Step 1)...")

    # We USE the intrinsics (K, D) we just found, and only calculate R, T, E, F
    stereo_flags = (cv2.CALIB_FIX_INTRINSIC | 
                    cv2.CALIB_RATIONAL_MODEL | 
                    cv2.CALIB_FIX_PRINCIPAL_POINT)

    retStereo, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r, 
        K1, D1, K2, D2, img_shape,
        flags=stereo_flags,
        criteria=criteria
    )
    
    print(f"  -> Stereo RMS Error: {retStereo:.4f}")

    # --- VALIDATION ---
    calculated_baseline = np.linalg.norm(T)
    print("------------------------------------------------")
    print(f"Calculated Baseline: {calculated_baseline:.4f} meters")
    print(f"Real/Known Baseline: 0.5400 meters")
    print(f"Difference:          {abs(calculated_baseline - 0.54):.4f} meters")
    print("------------------------------------------------")

    # 4. Rectification
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, img_shape, R, T, alpha=1)

    # 5. Save
    print("\nSaving calibration data to 'stereo_params.npz'...")
    np.savez('stereo_params.npz', K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T, R1=R1, P1=P1, R2=R2, P2=P2, Q=Q)

    # 6. Visual Check
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_shape, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_shape, cv2.CV_32FC1)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    for f_left, f_right in zip(images_left, images_right):
        imgL = cv2.imread(f_left)
        imgR = cv2.imread(f_right)
        
        rect_L = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
        rect_R = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)
        
        visualize_rectification(rect_L, rect_R)
        key = cv2.waitKey(0)
        if key & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()