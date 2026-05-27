import os
import uuid
import tempfile
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from ocr_engine import recognize_image
from excel_export import export_to_excel
from history import save_record, get_all_records, get_record, delete_record
from corrections import save_corrections, get_all_corrections, delete_correction

app = FastAPI(title="图片表格识别")

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(STATIC_DIR, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/ocr")
async def ocr_recognize(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1] or ".png"
    filename = f"{uuid.uuid4().hex}{suffix}"
    filepath = os.path.join(UPLOAD_DIR, filename)

    content = await file.read()
    with open(filepath, "wb") as f:
        f.write(content)

    try:
        result = recognize_image(filepath)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"识别失败: {str(e)}")

    data = result.get("data", [["识别失败"]])

    return {
        "success": result.get("success", True),
        "data": data,
        "method": result.get("method", "unknown"),
        "image_path": filepath,
        "original_name": file.filename,
        "rows": len(data),
        "cols": max((len(r) for r in data), default=0)
    }


@app.post("/api/export")
async def export_excel(data: dict):
    table_data = data.get("data")
    if not table_data:
        raise HTTPException(status_code=400, detail="缺少数据")

    filename = data.get("filename", "识别结果")
    if not filename.endswith(".xlsx"):
        filename += ".xlsx"

    excel_bytes = export_to_excel(table_data)

    return Response(
        content=excel_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@app.post("/api/history/save")
async def save_to_history(data: dict):
    image_path = data.get("image_path", "")
    table_data = data.get("data")
    if not table_data:
        raise HTTPException(status_code=400, detail="缺少数据")

    record_id = save_record(image_path, table_data)
    return {"success": True, "id": record_id}


@app.get("/api/history")
async def list_history():
    records = get_all_records()
    return {"records": records}


@app.get("/api/history/{record_id}")
async def get_history_record(record_id: int):
    record = get_record(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="记录不存在")
    return record


@app.delete("/api/history/{record_id}")
async def delete_history_record(record_id: int):
    deleted = delete_record(record_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="记录不存在")
    return {"success": True}


@app.get("/api/history/{record_id}/export")
async def export_history_record(record_id: int):
    record = get_record(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="记录不存在")

    excel_bytes = export_to_excel(record["data"])
    return Response(
        content=excel_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f'attachment; filename="记录_{record_id}.xlsx"'}
    )


@app.post("/api/corrections")
async def save_correction_pairs(data: dict):
    pairs = data.get("pairs", [])
    if not pairs:
        raise HTTPException(status_code=400, detail="缺少修正数据")
    save_corrections([(p["ocr"], p["corrected"]) for p in pairs])
    return {"success": True, "saved": len(pairs)}


@app.get("/api/corrections")
async def list_corrections():
    return {"corrections": get_all_corrections()}


@app.delete("/api/corrections/{ocr_text}")
async def delete_correction_item(ocr_text: str):
    delete_correction(ocr_text)
    return {"success": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8501)
