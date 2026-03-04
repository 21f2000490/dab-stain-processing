import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided

from autoparse import parse_args


class Parameters:
    eccentricity_threshold = 0.9
    hard_area_threshold = 50
    soft_area_threshold = 100
    bubble_circle_threshold = 0.2


@dataclass(eq=True, frozen=True)
class Config:
    input_file: Path = field(metadata={"help": "Path to the input image file."})
    out_dir: Path = field(metadata={"help": "Path to save the output files."})
    debug: bool = field(default=False, metadata={"help": "Enable debug mode to print intermediate information."})

    dab_strength_threshold: float = field(
        default=0.8, metadata={"help": "Threshold for DAB stain strength used in detection or filtering."}
    )

    # validation
    def __post_init__(self: "Config") -> None:
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file '{self.input_file}' does not exist")

        subdir_name = "_".join(
            f"{k}:{v}" for k, v in self.__dict__.items() if k not in ("input_file", "out_dir", "debug")
        )
        subdir_name = subdir_name.replace("/", "_").replace(" ", "")
        object.__setattr__(self, "out_dir", self.out_dir / subdir_name)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        if not (0 <= self.dab_strength_threshold <= 1):
            raise ValueError(f"--dab-strength-threshold: expected between 0 and 1, got {self.dab_strength_threshold}")


def read_image_rgb(path, normalize=False):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    if normalize:
        img = img.astype(np.float32) / 255.0
    return img


def calculate_stain_strengths(img, deconv):
    _, _, c = img.shape
    img_flat = img.reshape(-1, c).T

    X = np.linalg.pinv(deconv) @ -np.log(img_flat + 1e-8)  # 2x262144 i.e for each pixel strength of HEMA and DAB

    X /= np.percentile(X, 99)
    X = np.clip(X, 0, 1)

    return X


def apply_filter(X, k, f):
    h, w = X.shape
    X = np.pad(X, k // 2)
    s0, s1 = X.strides
    return f(as_strided(X, shape=(h, w, k, k), strides=(s0, s1, s0, s1)), axis=(2, 3))


def connected_components(mask):
    h, w = mask.shape
    # labels for each connected component
    labels = np.zeros((h, w), dtype=np.int32)
    # label id
    n = 0
    for y in range(h):
        for x in range(w):
            # True in mask that is not labelled yet ...
            if labels[y][x] or not mask[y][x]:
                continue
            n += 1

            stack = [(y, x)]
            while stack:
                y, x = stack.pop()
                labels[y][x] = n
                for ny, nx in [(y - 1, x), (y + 1, x), (y, x + 1), (y, x - 1)]:
                    # valid, True in mask, and not labelled already
                    if 0 <= ny < h and 0 <= nx < w and mask[ny][nx] and not labels[ny][nx]:
                        stack.append((ny, nx))

    return [labels == i for i in range(1, n + 1)]


def get_eccentricity(ys, xs):
    if len(ys) < 2:
        cov = np.zeros((2, 2))
    else:
        cov = np.cov(np.stack((ys, xs), axis=0))
    l_min, l_max = np.linalg.eigvalsh(cov)
    return np.sqrt(1 - l_min / l_max) if l_max > 0 else 0


def write_img(path, img, grayscale=False, invert=False):
    if img.dtype != np.uint8:
        img = (255 * img).astype(np.uint8)
    if not grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif invert:
        img = 255 - img
    cv2.imwrite(path, img)


def remove_bubbles(
    img,
    circle_thresh,
    lower_brown=np.array([0, 15, 40]),
    upper_brown=np.array([35, 255, 255]),
    angle_thresh=1.7 * np.pi,
    edge_margin=5,
):
    h, w, _ = img.shape
    output = img.copy()

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower_brown, upper_brown)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def is_circle(contour):
        if contour is None:
            return False, False

        pts = contour.reshape(-1, 2)
        if pts.shape[0] < 5:
            return False, False

        pts = pts.astype(np.float32)
        x = pts[:, 0]
        y = pts[:, 1]

        A = np.column_stack([x, y, np.ones_like(x)])
        b = -(x**2 + y**2)

        try:
            D, E, F = np.linalg.lstsq(A, b, rcond=None)[0]
        except np.linalg.LinAlgError:
            return False, False

        cx, cy = -D / 2, -E / 2
        center = np.array([cx, cy])

        r = np.linalg.norm(pts - center, axis=1)
        mean_r = r.mean()
        if mean_r == 0:
            return False, False

        is_partial_circle = r.std() / mean_r < circle_thresh

        angles = np.arctan2(y - cy, x - cx)
        angles = np.unwrap(angles)
        is_full_circle = (angles.max() - angles.min()) > angle_thresh
        is_full_circle = is_full_circle and is_partial_circle

        return is_partial_circle, is_full_circle

    def is_close_to_image_edge(contour):
        x, y, bw, bh = cv2.boundingRect(contour)
        return x <= edge_margin or y <= edge_margin or x + bw >= w - edge_margin or y + bh >= h - edge_margin

    for c in contours:
        is_partial_circle, is_full_circle = is_circle(c)

        is_bubble = is_full_circle or (is_partial_circle and is_close_to_image_edge(c))
        if is_bubble:
            cv2.drawContours(output, [c], -1, (255, 255, 255), thickness=cv2.FILLED)

    return output


def process(config: Config):
    config.out_dir.mkdir(parents=True, exist_ok=True)
    components_dir = config.out_dir / "components"
    annotation_dir = config.out_dir / "annotated"
    debug_dir = config.out_dir / "debug"
    components_dir.mkdir(exist_ok=True)
    annotation_dir.mkdir(exist_ok=True)
    debug_dir.mkdir(exist_ok=True)

    img = read_image_rgb(config.input_file)
    h, w, _ = img.shape

    img = remove_bubbles(img, circle_thresh=Parameters.bubble_circle_threshold)

    if config.debug:
        write_img(debug_dir / ("bubbles-removed_" + config.input_file.stem + ".png"), img)

    # color deconv matrix for HEMA (first column) /DAB (second column) stains
    K = np.array([[0.650, 0.704, 0.286], [0.268, 0.570, 0.776]]).T  # 3x2
    strengths = calculate_stain_strengths(img.astype(np.float32) / 255.0, K)
    dab_strength = strengths[1]
    mask = (dab_strength > config.dab_strength_threshold).astype(np.uint8).reshape(h, w)

    if config.debug:
        write_img(debug_dir / ("mask-dab-thresholded_" + config.input_file.stem + ".png"), mask, grayscale=True)

    mask = apply_filter(mask, 3, np.min)  # remove white specks in black blobs
    mask = apply_filter(mask, 3, np.max)  # restore white edges that got serrated
    mask = apply_filter(mask, 3, np.max)  # remove black specks inside white blobs
    mask = apply_filter(mask, 3, np.min)  # restore black edges that got serrated

    if config.debug:
        write_img(debug_dir / ("mask-box-filtered_" + config.input_file.stem + ".png"), mask, grayscale=True)

    component_masks = connected_components(mask)
    if config.debug:
        with open(debug_dir / ("components-pre-area-filter_" + config.input_file.stem + ".txt"), "w") as f:
            f.write("x y area\n")
            for cm in component_masks:
                ys, xs = np.nonzero(cm)
                area = len(ys)
                if area > 0:
                    cy = ys.mean()
                    cx = xs.mean()
                    f.write(f"{cx} {cy} {area}\n")

    for i in range(len(component_masks)):
        cm = component_masks[i]
        area = cm.sum()
        ys, xs = np.nonzero(cm)
        ecc = get_eccentricity(ys, xs)

        if area < Parameters.hard_area_threshold:
            mask[cm] = 0
        if area < Parameters.soft_area_threshold and ecc < Parameters.eccentricity_threshold:
            mask[cm] = 0
        component_masks[i] &= mask.astype(bool)

    component_masks = [cm for cm in component_masks if cm.sum() > 0]

    if config.debug:
        write_img(debug_dir / ("mask-area-filtered_" + config.input_file.stem + ".png"), mask, grayscale=True)

    components_of_img = img.copy()
    components_of_img[mask == 0] = (255, 255, 255)

    annotated_img = img.copy()
    for cm in component_masks:
        cm_uint8 = cm.astype(np.uint8) * 255
        contours, _ = cv2.findContours(cm_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(annotated_img, contours, -1, (0, 0, 0), thickness=1)

    write_img(components_dir / Path(config.input_file.stem + ".png"), components_of_img)
    write_img(annotation_dir / Path(config.input_file.stem + ".png"), annotated_img)
    print("Done")


def main():
    try:
        config = parse_args("DAB stain processing", Config)
    except Exception as e:
        print("Exception occured when parsing config:", e, file=sys.stderr)
        return 1

    try:
        process(config)
    except Exception as e:
        print(f"Exception occured when processing input {config.input_file}:", e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    exit(main())
