"""
Devanagari word -> character (akshara) segmentation, guided by ground truth.

General pipeline (works across fonts/styles, single or multi-word images)
--------------------------------------------------------------------------
1. Binarize (Otsu) with automatic polarity detection (works for both
   dark-text-on-light and light-text-on-dark scans/renders) + a small
   morphological close to patch texture/anti-aliasing noise.
2. Detect and remove shirorekha (headline) connected components -- wide,
   thin, near the top. Removes ALL matching components, so a headline
   broken into several pieces (common in stylized/brush fonts) is still
   fully stripped, not just the single largest piece.
3. Drop noise specks using a resolution-relative area threshold (a fixed
   pixel count doesn't generalize across image sizes; a fraction of
   image area does).
4. (Optional, multi-word) If the ground truth contains whitespace, first
   split the image into per-word column ranges using large gaps, then run
   steps 5-7 independently per word.
5. Compute the column-wise ink projection and find "hard" gaps: runs of
   columns with EXACTLY zero ink. These are unambiguous, evidence-based
   separators -- if two shapes don't share a single ink column, they are
   definitely visually distinct, regardless of font style.
6. Split the ground truth into akshara (visual character cluster) units
   via a Devanagari Unicode-range regex. This gives k, the expected
   number of character groups for this word (NOT len(text), since a
   consonant + vowel-sign + anusvara render as one glyph cluster).
7. Reconcile the G hard-gap-derived runs against k:
     - G == k: use the real, evidence-based boundaries directly. This is
       the common case for cleanly printed text -- no guesswork needed.
     - G > k: a genuine ink gap fell *inside* a single akshara (e.g. a
       vowel sign that doesn't quite touch its consonant). Merge adjacent
       runs, smallest empty-gap-width first, until the count matches.
     - G < k: two or more aksharas are visually fused (touching strokes,
       common in brush/calligraphic fonts). Repeatedly bisect the
       currently-widest run at an ink-weighted valley (equal cumulative-
       ink split, snapped to the nearest real local minimum) until the
       count matches. This only touches the specific fused run(s) --
       already-correct boundaries elsewhere are never disturbed.
8. Crop each final column range with a tight y-bbox (from real ink rows),
   map left-to-right to the akshara list, and save crops + a manifest +
   a debug visualization.

This is a heuristic CV pipeline, not a trained model -- it will get you
very good automatic segmentation for most printed/handwritten Devanagari
word images, but always spot-check the *_segmentation_debug.png on a
sample of your data and adjust the tunable parameters in SegmentConfig
if needed (see the docstring on that class).
"""

import cv2
import numpy as np
import re
import json
import glob
import os
from dataclasses import dataclass


# ---------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------
def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    return img
@dataclass
class SegmentConfig:
    # -- binarization / cleanup --
    morph_close_iters: int = 2          # patches small texture/brush holes
    morph_kernel_size: int = 3

    # -- headline (shirorekha) detection --
    headline_aspect_min: float = 4.0    # width/height ratio to count as "headline-like"
    headline_top_fraction: float = 0.25 # must sit within this fraction of image height from the top

    # -- noise removal --
    noise_area_ratio: float = 0.0015    # components smaller than this * (h*w) are dropped as noise

    # -- valley snapping (used only when G < k, i.e. splitting fused runs) --
    smooth_window: int = 5
    snap_window_ratio: float = 0.15     # search window for the nearest valley, as a fraction of the run's width
    min_snap_window: int = 5


# ---------------------------------------------------------------------
# Ground truth -> akshara (visual character cluster) splitting
# ---------------------------------------------------------------------

_AKSHARA_RE = re.compile(
    r'[\u0904-\u0939\u0958-\u0961]'             # independent vowel or consonant
    r'(?:\u094D[\u0904-\u0939\u0958-\u0961])*'   # halant + consonant, repeated (conjuncts)
    r'[\u093E-\u094C\u0962\u0963]?'              # optional dependent vowel sign (matra)
    r'[\u0900-\u0903]?',                         # optional candrabindu/anusvara/visarga
    re.UNICODE,
)


def split_aksharas(text: str) -> list[str]:

    clusters, i = [], 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        m = _AKSHARA_RE.match(text, i)
        if m and m.end() > i:
            clusters.append(m.group())
            i = m.end()
        else:
            clusters.append(text[i])
            i += 1
    return clusters


# ---------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------

def binarize(img_gray: np.ndarray, cfg: SegmentConfig) -> np.ndarray:

    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]])
    if np.mean(border == 255) > 0.5:
        binary = 255 - binary
    k = cfg.morph_kernel_size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=cfg.morph_close_iters)


def remove_headline(binary: np.ndarray, cfg: SegmentConfig) -> np.ndarray:

    h, _ = binary.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = binary.copy()
    for i in range(1, num_labels):
        x, y, cw, ch, area = stats[i]
        if cw / max(ch, 1) > cfg.headline_aspect_min and y < h * cfg.headline_top_fraction:
            out[labels == i] = 0
    return out


def remove_noise(binary: np.ndarray, cfg: SegmentConfig) -> np.ndarray:
    """Drop connected components smaller than a resolution-relative area threshold."""
    h, w = binary.shape
    min_area = cfg.noise_area_ratio * h * w
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = binary.copy()
    for i in range(1, num_labels):
        if stats[i][4] < min_area:
            out[labels == i] = 0
    return out


# ---------------------------------------------------------------------
# Ink-run detection + ground-truth-guided reconciliation
# ---------------------------------------------------------------------

def find_ink_runs(binary: np.ndarray, x0: int = 0, x1: int | None = None) -> list[list[int]]:
    """Find maximal runs of columns (within [x0, x1)) with nonzero ink --
    i.e. evidence-based, unambiguous visual groups separated by true gaps."""
    if x1 is None:
        x1 = binary.shape[1]
    v = np.sum(binary[:, x0:x1] == 255, axis=0)
    is_ink = v > 0
    runs, start = [], None
    for i, ink in enumerate(is_ink):
        x = x0 + i
        if ink and start is None:
            start = x
        if not ink and start is not None:
            runs.append([start, x])
            start = None
    if start is not None:
        runs.append([start, x1])
    return runs


def _bisect_run(x0: int, x1: int, v_full: np.ndarray, cfg: SegmentConfig) -> int:
    """Split [x0, x1) into 2 at an ink-weighted valley: target the column
    where cumulative ink mass reaches 50%, then snap to the nearest local
    minimum nearby (avoids being fooled by an off-center density skew)."""
    seg = v_full[x0:x1].astype(float)
    if cfg.smooth_window > 1 and len(seg) > cfg.smooth_window:
        smooth = np.convolve(seg, np.ones(cfg.smooth_window) / cfg.smooth_window, mode='same')
    else:
        smooth = seg
    cum = np.cumsum(seg)
    total = cum[-1]
    if total == 0:
        return (x0 + x1) // 2
    idx = int(np.searchsorted(cum, total / 2))
    window = max(cfg.min_snap_window, int(cfg.snap_window_ratio * (x1 - x0)))
    lo, hi = max(0, idx - window), min(len(seg), idx + window)
    cut_rel = lo + int(np.argmin(smooth[lo:hi]))
    cut_rel = max(1, min(len(seg) - 1, cut_rel))
    return x0 + cut_rel


def reconcile_runs(runs: list[list[int]], k: int, v_full: np.ndarray, cfg: SegmentConfig) -> list[list[int]]:
    """Adjust evidence-based ink runs to match the expected akshara count k."""
    runs = [r[:] for r in runs]
    if not runs:
        return runs

    # too many runs: merge the pair with the smallest gap between them, repeatedly
    while len(runs) > k:
        gaps = [runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)]
        gi = int(np.argmin(gaps))
        runs[gi] = [runs[gi][0], runs[gi + 1][1]]
        del runs[gi + 1]

    # too few runs: bisect the widest run at a valley, repeatedly
    while len(runs) < k:
        widths = [r[1] - r[0] for r in runs]
        wi = int(np.argmax(widths))
        x0, x1 = runs[wi]
        cut = _bisect_run(x0, x1, v_full, cfg)
        if cut <= x0 or cut >= x1:
            break  # can't split further (degenerate run); avoid infinite loop
        runs[wi:wi + 1] = [[x0, cut], [cut, x1]]

    runs.sort()
    return runs

def preprocess(image, cfg=None):
    if cfg is None:
        cfg = SegmentConfig()

    # Convert BGR to Grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply existing binarization pipeline
    binary = binarize(gray, cfg)
    binary = remove_headline(binary, cfg)
    binary = remove_noise(binary, cfg)

    return binary
# ---------------------------------------------------------------------
# Word-level splitting (only engaged for multi-word ground truth)
# ---------------------------------------------------------------------

def split_words(binary: np.ndarray, num_words: int) -> list[list[int]]:

    runs = find_ink_runs(binary)
    if not runs:
        return []
    if num_words <= 1 or len(runs) <= 1:
        return [[runs[0][0], runs[-1][1]]]

    gaps = [(runs[i + 1][0] - runs[i][1], i) for i in range(len(runs) - 1)]
    n_splits = min(num_words - 1, len(gaps))
    split_indices = sorted(i for _, i in sorted(gaps, key=lambda g: -g[0])[:n_splits])

    words, start = [], runs[0][0]
    for i in split_indices:
        words.append([start, runs[i][1]])
        start = runs[i + 1][0]
    words.append([start, runs[-1][1]])
    return words


# ---------------------------------------------------------------------
# Core per-word segmentation
# ---------------------------------------------------------------------

def segment_word_region(orig_bgr, binary, x0, x1, gt_word, cfg: SegmentConfig):
    """Segment a single word's column range [x0, x1) into its aksharas."""
    aksharas = split_aksharas(gt_word.strip())
    k = len(aksharas)
    if k == 0:
        return []

    region = binary[:, x0:x1]
    v_full = np.sum(region == 255, axis=0).astype(float)
    runs = find_ink_runs(region)
    runs = reconcile_runs(runs, k, v_full, cfg)

    # pad/truncate defensively in case reconciliation still fell short
    # (e.g. a totally blank region) so labels and runs always line up
    while len(runs) < len(aksharas):
        runs.append([region.shape[1] - 1, region.shape[1]])
    runs = runs[:len(aksharas)]

    h = binary.shape[0]
    segments = []
    for (rx0, rx1), label in zip(runs, aksharas):
        abs_x0, abs_x1 = x0 + rx0, x0 + rx1
        col_slice = binary[:, abs_x0:abs_x1]
        rows = np.where(col_slice.sum(axis=1) > 0)[0]
        if len(rows):
            y0, y1 = max(0, int(rows.min()) - 2), min(h, int(rows.max()) + 3)
        else:
            y0, y1 = 0, h
        crop = orig_bgr[y0:y1, abs_x0:abs_x1]
        segments.append({"label": label, "bbox": (int(abs_x0), y0, int(abs_x1), y1), "crop": crop})
    return segments
def segment_pair(image_path: str, gt_text: str, out_dir: str, stem: str, cfg: SegmentConfig | None = None):
    cfg = cfg or SegmentConfig()
    os.makedirs(out_dir, exist_ok=True)
    orig = cv2.imread(image_path)
    if orig is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    binary = binarize(gray, cfg)
    binary = remove_headline(binary, cfg)
    binary = remove_noise(binary, cfg)

    gt_words = gt_text.strip().split()
    if len(gt_words) <= 1:
        word_regions = [(0, binary.shape[1])]
        gt_words = [gt_text.strip()]
    else:
        word_runs = split_words(binary, num_words=len(gt_words))
        word_regions = [(r[0], r[1]) for r in word_runs] if word_runs else [(0, binary.shape[1])]
        if not word_runs:
            gt_words = [gt_text.strip()]

    vis = orig.copy()
    manifest = {"source_image": image_path, "ground_truth": gt_text, "words": []}
    char_counter = 0

    for word_idx, ((wx0, wx1), gt_word) in enumerate(zip(word_regions, gt_words)):
        segments = segment_word_region(orig, binary, wx0, wx1, gt_word, cfg)
        cv2.rectangle(vis, (wx0, 0), (max(wx0, wx1 - 1), binary.shape[0] - 1), (255, 0, 0), 1)
        word_entry = {"word_index": word_idx, "text": gt_word, "characters": []}

        for seg in segments:
            x0, y0, x1, y1 = seg["bbox"]
            crop_path = os.path.join(out_dir, f"{stem}_w{word_idx}_c{char_counter}_{seg['label']}.png")
            cv2.imwrite(crop_path, seg["crop"])
            cv2.rectangle(vis, (x0, y0), (max(x0, x1 - 1), max(y0, y1 - 1)), (0, 255, 0), 1)
            word_entry["characters"].append({
                "index": char_counter,
                "label": seg["label"],
                "bbox": seg["bbox"],
                "crop_file": os.path.basename(crop_path),
            })
            char_counter += 1
        manifest["words"].append(word_entry)

    vis_path = os.path.join(out_dir, f"{stem}_segmentation_debug.png")
    cv2.imwrite(vis_path, vis)
    manifest["debug_visualization"] = os.path.basename(vis_path)

    manifest_path = os.path.join(out_dir, f"{stem}_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest
# def segment_pair(image_path, gt_text, out_dir, stem):

#     # 1. Read image
#     image = load_image(image_path)

#     # 2. Normalize / preprocess
#     binary = preprocess(image)

#     # 3. Tokenize Sanskrit GT
#     tokens = tokenize_aksharas(gt_text)

#     # 4. Generate image boundary candidates
#     candidates = generate_vpp_candidates(binary)

#     # 5. Add additional evidence
#     #    - foreground density
#     #    - stroke continuity
#     #    - upper matra region
#     #    - lower matra region
#     #    - shirorekha
#     #    - inter-character whitespace

#     # 6. Align GT tokens with image
#     boundaries = align_tokens_to_image(
#         binary,
#         tokens,
#         candidates
#     )

#     # 7. Refine boundaries locally
#     boundaries = refine_boundaries(
#         binary,
#         boundaries
#     )

#     # 8. Crop complete aksharas
#     crops = crop_aksharas(
#         image,
#         boundaries
#     )

#     # 9. Save
#     manifest = save_results(
#         crops,
#         tokens,
#         boundaries,
#         out_dir,
#         stem
#     )

#     return manifest
# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

# def segment_pair(image_path: str, gt_text: str, out_dir: str, stem: str, cfg: SegmentConfig | None = None):
#     """Segment one image against its ground truth (single or multi-word).
#     Saves per-character crops, a debug visualization, and a JSON manifest
#     into out_dir. Returns the manifest dict."""
#     cfg = cfg or SegmentConfig()
#     os.makedirs(out_dir, exist_ok=True)

#     orig = cv2.imread(image_path)
#     gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

#     binary = binarize(gray, cfg)
#     binary = remove_headline(binary, cfg)
#     binary = remove_noise(binary, cfg)

#     gt_words = gt_text.strip().split()
#     if len(gt_words) <= 1:
#         word_regions = [(0, binary.shape[1])]
#         gt_words = [gt_text.strip()]
#     else:
#         word_runs = split_words(binary, num_words=len(gt_words))
#         word_regions = [(r[0], r[1]) for r in word_runs] if word_runs else [(0, binary.shape[1])]
#         if not word_runs:
#             gt_words = [gt_text.strip()]

#     vis = orig.copy()
#     manifest = {"source_image": image_path, "ground_truth": gt_text, "words": []}
#     char_counter = 0

#     for word_idx, ((wx0, wx1), gt_word) in enumerate(zip(word_regions, gt_words)):
#         segments = segment_word_region(orig, binary, wx0, wx1, gt_word, cfg)
#         cv2.rectangle(vis, (wx0, 0), (max(wx0, wx1 - 1), binary.shape[0] - 1), (255, 0, 0), 1)

#         word_entry = {"word_index": word_idx, "text": gt_word, "characters": []}
#         for seg in segments:
#             x0, y0, x1, y1 = seg["bbox"]
#             crop_path = os.path.join(out_dir, f"{stem}_w{word_idx}_c{char_counter}_{seg['label']}.png")
#             cv2.imwrite(crop_path, seg["crop"])
#             cv2.rectangle(vis, (x0, y0), (max(x0, x1 - 1), max(y0, y1 - 1)), (0, 255, 0), 1)
#             word_entry["characters"].append({
#                 "index": char_counter, "label": seg["label"],
#                 "bbox": seg["bbox"], "crop_file": os.path.basename(crop_path),
#             })
#             char_counter += 1
#         manifest["words"].append(word_entry)

#     vis_path = os.path.join(out_dir, f"{stem}_segmentation_debug.png")
#     cv2.imwrite(vis_path, vis)
#     manifest["debug_visualization"] = os.path.basename(vis_path)

#     manifest_path = os.path.join(out_dir, f"{stem}_manifest.json")
#     with open(manifest_path, "w", encoding="utf-8") as f:
#         json.dump(manifest, f, ensure_ascii=False, indent=2)

#     return manifest


def segment_batch(input_dir: str, out_dir: str, pattern: str = "*.png", cfg: SegmentConfig | None = None):
    """Segment every <name>.png / <name>.txt pair found in input_dir."""
    cfg = cfg or SegmentConfig()
    results = []
    for img_path in sorted(glob.glob(os.path.join(input_dir, pattern))):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(input_dir, stem + ".txt")
        if not os.path.exists(txt_path):
            print(f"skip {stem}: no matching .txt ground truth found")
            continue
        with open(txt_path, encoding="utf-8") as f:
            gt_text = f.read()
        manifest = segment_pair(img_path, gt_text, out_dir, stem, cfg)
        n_chars = sum(len(w["characters"]) for w in manifest["words"])
        print(f"{stem}: '{gt_text.strip()}' -> {len(manifest['words'])} word(s), {n_chars} character(s)")
        results.append(manifest)
    return results


if __name__ == "__main__":
    import sys
    in_dir = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "/mnt/user-data/outputs/segmented_chars"
    segment_batch(in_dir, out_dir)