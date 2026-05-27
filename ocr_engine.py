import cv2
import numpy as np
import traceback
import re
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
from corrections import apply_corrections

_ocr_engine = RapidOCR()


def load_image(path: str) -> np.ndarray:
    pil_img = Image.open(path)
    if pil_img.mode == 'RGBA':
        bg = Image.new('RGB', pil_img.size, (255, 255, 255))
        bg.paste(pil_img, mask=pil_img.split()[3])
        pil_img = bg
    elif pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _normalize_background(image: np.ndarray) -> np.ndarray:
    """Remove yellow/tinted paper background, push to white."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    bg_mask = (s > 25) & (v > 130)
    if np.sum(bg_mask) > image.shape[0] * image.shape[1] * 0.08:
        v = np.where(bg_mask, np.clip(v.astype(np.int32) + 70, 0, 255).astype(np.uint8), v)
        s = np.where(bg_mask, np.clip(s.astype(np.int32) - 50, 0, 255).astype(np.uint8), s)
        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image


def _enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Boost contrast for pen strokes on light paper."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    p_low, p_high = np.percentile(gray, (1, 99))
    if p_high - p_low < 60:
        alpha = 255.0 / max(p_high - p_low, 1)
        beta = -p_low * alpha
        gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    return gray


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    text = text.replace('\n', ' ')
    text = re.sub(r'\s{2,}', ' ', text)
    return text


# ============================================================
#  RapidOCR-based recognition
# ============================================================

def _box_center_y(box) -> float:
    return (box[0][1] + box[2][1]) / 2


def _box_center_x(box) -> float:
    return (box[0][0] + box[2][0]) / 2


def _box_height(box) -> float:
    return abs(box[2][1] - box[0][1])


def _group_into_rows(items: list) -> list:
    """Group OCR results into rows by Y coordinate."""
    if not items:
        return []
    items = sorted(items, key=lambda it: _box_center_y(it['box']))
    rows = []
    cur = [items[0]]
    for item in items[1:]:
        cur_h = max(_box_height(it['box']) for it in cur)
        if abs(_box_center_y(item['box']) - _box_center_y(cur[0]['box'])) < cur_h * 0.8:
            cur.append(item)
        else:
            rows.append(cur)
            cur = [item]
    rows.append(cur)
    return rows


def _cluster_x_positions(rows: list) -> list:
    """Cluster X centers across all rows into column bins."""
    all_x = []
    for row in rows:
        for it in row:
            all_x.append(_box_center_x(it['box']))
    if not all_x:
        return []

    all_x.sort()
    # merge X positions that are close together
    clusters = [[all_x[0]]]
    for x in all_x[1:]:
        # tolerance: use median item width across all items
        if x - clusters[-1][-1] < 40:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    # each cluster center = one column
    return [sum(c) / len(c) for c in clusters]


def _assign_to_columns(rows: list, col_centers: list) -> list:
    """Assign each item to its nearest column, return 2D grid."""
    n_cols = len(col_centers)
    result = []
    for row in rows:
        grid_row = [""] * n_cols
        for it in row:
            cx = _box_center_x(it['box'])
            # find nearest column
            best_col = 0
            best_dist = abs(cx - col_centers[0])
            for ci in range(1, n_cols):
                d = abs(cx - col_centers[ci])
                if d < best_dist:
                    best_dist = d
                    best_col = ci
            # merge text if column already has content
            existing = grid_row[best_col]
            new_text = clean_text(it['text'])
            if existing:
                grid_row[best_col] = existing + " " + new_text
            else:
                grid_row[best_col] = new_text
        result.append(grid_row)
    return result


def _run_rapidocr(image: np.ndarray) -> list:
    """Run RapidOCR, return list of {box, text, conf}."""
    result, _ = _ocr_engine(image)
    if not result:
        return []
    items = []
    for box, text, conf in result:
        if text and text.strip():
            items.append({'box': box, 'text': text.strip(), 'conf': conf})
    return items


def recognize_image(image_path: str) -> dict:
    try:
        image = load_image(image_path)
    except Exception:
        return {"success": False, "data": [["无法读取图片文件"]], "method": "error"}

    try:
        image = _normalize_background(image)

        # run OCR on original image
        items = _run_rapidocr(image)

        # if few results, try with enhanced contrast
        if len(items) < 3:
            enhanced = _enhance_contrast(image)
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR) if len(enhanced.shape) == 2 else enhanced
            items2 = _run_rapidocr(enhanced_bgr)
            if len(items2) > len(items):
                items = items2

        if not items:
            return {"success": False, "data": [["未识别到任何文字"]], "method": "none"}

        rows = _group_into_rows(items)
        col_centers = _cluster_x_positions(rows)
        if col_centers and len(col_centers) > 1:
            data = _assign_to_columns(rows, col_centers)
        else:
            # single column: each row is one item
            data = [[clean_text(it['text'])] for row in rows for it in sorted(row, key=lambda it: _box_center_x(it['box']))]
            # re-group: one row per Y group
            data = []
            for row in rows:
                sorted_row = sorted(row, key=lambda it: _box_center_x(it['box']))
                texts = [clean_text(it['text']) for it in sorted_row]
                data.append(texts)

        # filter out rows that are too short (noise)
        filtered = [r for r in data if any(len(t.strip()) > 0 for t in r)]
        if not filtered:
            filtered = data

        # apply learned corrections
        filtered = apply_corrections(filtered)

        return {"success": True, "data": filtered, "method": "rapidocr"}

    except Exception as e:
        traceback.print_exc()
        return {"success": False, "data": [[f"识别出错: {str(e)}"]], "method": "error"}
