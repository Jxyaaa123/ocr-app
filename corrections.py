import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corrections.json")


def _load() -> dict:
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save(db: dict):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def _norm(text: str) -> str:
    """Normalize for matching: strip, lowercase, remove spaces."""
    return text.strip().lower().replace(" ", "").replace("　", "")


def save_corrections(pairs: list):
    """Save a list of (ocr_text, corrected_text) pairs."""
    db = _load()
    for ocr_text, corrected in pairs:
        ocr_text = ocr_text.strip()
        corrected = corrected.strip()
        if not ocr_text or not corrected or ocr_text == corrected:
            continue
        key = _norm(ocr_text)
        if key in db:
            entry = db[key]
            entry["corrected"] = corrected
            entry["count"] += 1
            entry["last_used"] = datetime.now().isoformat()
        else:
            db[key] = {
                "ocr": ocr_text,
                "corrected": corrected,
                "count": 1,
                "created": datetime.now().isoformat(),
                "last_used": datetime.now().isoformat(),
            }
    _save(db)


def apply_corrections(data: list) -> list:
    """Apply known corrections to OCR result data (2D list)."""
    db = _load()
    if not db:
        return data
    result = []
    for row in data:
        new_row = []
        for cell in row:
            key = _norm(cell)
            if key in db and db[key]["count"] >= 1:
                new_row.append(db[key]["corrected"])
            else:
                new_row.append(cell)
        result.append(new_row)
    return result


def get_all_corrections() -> dict:
    return _load()


def delete_correction(ocr_text: str):
    db = _load()
    key = ocr_text.strip().lower()
    if key in db:
        del db[key]
        _save(db)
