import cv2
import numpy as np
import sys

# Simple area-based bubble/artifact detection
# Reads 'img.png' (or path from argv[1]) and writes 'img_bubbles_area.png'

INPUT = sys.argv[1] if len(sys.argv) > 1 else "img.png"
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "img_bubbles_area.png"

# HSV thresholds - copied from project files (tweak if needed)
LOWER_BROWN = np.array([0, 15, 40])
UPPER_BROWN = np.array([35, 255, 255])
MIN_AREA = 50
MAX_AREA = 2500

img = cv2.imread(INPUT, cv2.IMREAD_COLOR)
if img is None:
    print(f"Failed to load {INPUT}")
    sys.exit(1)

h, w, _ = img.shape

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, LOWER_BROWN, UPPER_BROWN)

# find contours (OpenCV version-agnostic)
conts_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = conts_info[0] if len(conts_info) == 2 else conts_info[1]


def is_circle(contour, thresh=0.15, angle_thresh=1.7 * np.pi):
    if contour is None:
        return False, False

    pts = contour.reshape(-1, 2)
    if pts.shape[0] < 5:
        return False, False

    pts = pts.astype(np.float32)
    x = pts[:, 0]
    y = pts[:, 1]

    # x^2 + y^2 + Dx + Ey + F = 0
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x**2 + y**2)
    try:
        D, E, F = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        print("ERROR:linalg:", e, file=sys.stderr)
        return False, False
    center = np.array([-D / 2, -E / 2])

    r = np.linalg.norm(pts - center, axis=1)
    mean_r = r.mean()
    if mean_r == 0:
        return False
    is_partial_circle = r.std() / mean_r < thresh

    cx, cy = -D / 2, -E / 2
    angles = np.arctan2(y - cy, x - cx)

    # below depends on points being ordered along the boundary of the contour
    # cv2.findContours follows that order, but if you change it, you have to reorder it
    angles = np.unwrap(angles)
    is_full_circle = angles.max() - angles.min() > angle_thresh
    is_full_circle = is_full_circle and is_partial_circle

    return is_partial_circle, is_full_circle


def is_close_to_image_edge(contour, h, w, margin=5):
    x, y, bw, bh = cv2.boundingRect(contour)
    return x <= margin or y <= margin or x + bw >= w - margin or y + bh >= h - margin


# We'll draw black boundaries around contours that are considered 'bubbles/artifacts' -> i.e., outside area range
output = img.copy()
for c in contours:
    area = cv2.contourArea(c)
    if MIN_AREA < area < MAX_AREA:
        continue

    is_partial_circle, is_full_circle = is_circle(c, thresh=0.2)
    is_bubble = is_full_circle or (is_partial_circle and is_close_to_image_edge(c, h, w, margin=5))
    if is_bubble:
        cv2.drawContours(output, [c], -1, (0, 0, 255), thickness=2)
    else:
        cv2.drawContours(output, [c], -1, (0, 0, 0), thickness=2)

cv2.imwrite(OUTPUT, output)
print(f"Wrote {OUTPUT}")
