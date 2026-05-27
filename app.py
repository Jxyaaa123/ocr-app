import streamlit as st
from local_storage_component import local_storage
import streamlit.components.v1 as components
import os
import json
import tempfile
import base64
import pandas as pd
from PIL import Image
from datetime import datetime
from excel_export import export_to_excel
from history import save_record, get_all_records, get_record, delete_record
from corrections import save_corrections, get_all_corrections, delete_correction

# ============================================================
#  配置文件
# ============================================================
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"model": "qwen-vl-plus"}

def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

cfg = load_config()


# ============================================================
#  自动纠偏
# ============================================================
def auto_deskew(pil_img):
    import cv2
    import numpy as np
    img = np.array(pil_img.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    if lines is None:
        return pil_img
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 30:
            angles.append(angle)
    if not angles:
        return pil_img
    median_angle = np.median(angles)
    if abs(median_angle) < 0.5:
        return pil_img
    return pil_img.rotate(-median_angle, expand=True, fillcolor=(255, 255, 255))


# ============================================================
#  AI 视觉识别
# ============================================================
def ai_recognize_image(image_path, api_key, model):
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "bmp": "image/bmp", "webp": "image/webp"}.get(ext, "image/png")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": (
                "你是表格识别助手。识别图片中的所有文字和数字，按表格结构输出。"
                "每行用换行分隔，每列用 | 分隔（竖线两边加空格）。"
                "只输出表格内容，不要任何解释。"
                "重要：必须输出图片中看到的每一行，包括第一行，一行都不能少！不要跳过任何行。"
            )},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                {"type": "text", "text": "请识别这张图片中的表格内容，按行和列输出。"},
            ]},
        ],
        max_tokens=2000,
    )

    text = response.choices[0].message.content.strip()
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    data = []
    for line in lines:
        if all(c in "-| " for c in line):
            continue
        cells = [c.strip() for c in line.split("|")]
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        if cells:
            data.append(cells)
    if not data:
        return {"success": False, "data": [["未识别到内容"]], "method": "ai"}
    max_cols = max(len(r) for r in data)
    data = [r + [""] * (max_cols - len(r)) for r in data]
    return {"success": True, "data": data, "method": "ai"}


# ============================================================
#  内置计算函数
# ============================================================
def _to_nums(data_list):
    """把数据列表转成数字列表，过滤掉非数字"""
    nums = []
    for v in data_list:
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        s = s.replace(",", "").replace("，", "").replace(" ", "")
        s = s.replace("%", "").replace("％", "")
        s = s.replace("元", "").replace("万", "").replace("个", "")
        try:
            nums.append(float(s))
        except ValueError:
            continue
    return nums

def calc_average(data):
    nums = _to_nums(data)
    if not nums:
        return ""
    return round(sum(nums) / len(nums), 4)

def calc_sum(data):
    nums = _to_nums(data)
    if not nums:
        return ""
    return round(sum(nums), 4)

def calc_max(data):
    nums = _to_nums(data)
    return round(max(nums), 4) if nums else ""

def calc_min(data):
    nums = _to_nums(data)
    return round(min(nums), 4) if nums else ""

def calc_median(data):
    nums = sorted(_to_nums(data))
    if not nums:
        return ""
    n = len(nums)
    if n % 2 == 1:
        return round(nums[n // 2], 4)
    return round((nums[n // 2 - 1] + nums[n // 2]) / 2, 4)

def calc_std(data):
    import numpy as np
    nums = _to_nums(data)
    if len(nums) < 2:
        return ""
    return round(float(np.std(nums, ddof=1)), 6)

def calc_count(data):
    return sum(1 for v in data if str(v).strip())

def calc_variance(data):
    import numpy as np
    nums = _to_nums(data)
    if len(nums) < 2:
        return ""
    return round(float(np.var(nums, ddof=1)), 6)

def calc_least_squares(x_data, y_data):
    """最小二乘法线性拟合 y = ax + b，返回 (a, b, r²)"""
    import numpy as np
    xs = _to_nums(x_data)
    ys = _to_nums(y_data)
    pairs = list(zip(xs, ys))
    if len(pairs) < 2:
        return None, None, None
    x_arr = np.array([p[0] for p in pairs])
    y_arr = np.array([p[1] for p in pairs])
    n = len(x_arr)
    sum_x = np.sum(x_arr)
    sum_y = np.sum(y_arr)
    sum_xy = np.sum(x_arr * y_arr)
    sum_x2 = np.sum(x_arr ** 2)
    denom = n * sum_x2 - sum_x ** 2
    if denom == 0:
        return None, None, None
    a = (n * sum_xy - sum_x * sum_y) / denom
    b = (sum_y - a * sum_x) / n
    ss_res = np.sum((y_arr - (a * x_arr + b)) ** 2)
    ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
    return round(float(a), 8), round(float(b), 8), round(float(r2), 8)


# ============================================================
#  页面
# ============================================================
st.set_page_config(page_title="图片表格识别", page_icon="📊", layout="wide")
st.title("📊 图片表格识别工具")

# 侧边栏设置
with st.sidebar:
    st.header("设置")
    saved_key = local_storage("ocr_api_key")
    api_key = st.text_input("API 密钥", type="password", value=saved_key or "",
                            help="阿里云 DashScope 获取")
    if st.button("保存 API Key"):
        if api_key.strip():
            local_storage("ocr_api_key", api_key.strip())
            st.success("已保存，下次自动读取")
            st.rerun()
        else:
            st.error("请输入有效的 API Key")
    model_list = ["qwen-vl-plus", "qwen-vl-max"]
    m_idx = model_list.index(cfg.get("model", "qwen-vl-plus")) if cfg.get("model") in model_list else 0
    model = st.selectbox("模型选择", model_list, index=m_idx)
    if model != cfg.get("model", ""):
        cfg["model"] = model
        save_config(cfg)
    if not api_key:
        st.warning("请输入 API 密钥")


st.caption("上传图片 → 自动纠偏 / 手动旋转 → AI 识别 → 编辑修正 → 计算分析 → 导出")

tab1, tab2, tab3 = st.tabs(["识别导出", "历史记录", "学习记录"])

with tab1:
    uploaded_file = st.file_uploader("上传图片", type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"])

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file)
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')

        col_img, col_result = st.columns([1, 1])

        with col_img:
            st.subheader("图片预览")

            auto_fix = st.checkbox("自动纠偏（歪了自动拉正）", value=True)
            if auto_fix:
                pil_img = auto_deskew(pil_img)

            st.markdown("**手动旋转：**")

            buf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            pil_img.save(buf, format='PNG')
            buf.close()
            with open(buf.name, "rb") as f:
                img_b64_for_js = base64.b64encode(f.read()).decode()
            os.unlink(buf.name)

            rotation_html = f"""
            <div style="text-align:center; padding:5px;">
                <div id="canvasWrap" style="display:inline-block; overflow:hidden; max-width:100%; border:1px solid #ddd; border-radius:8px;">
                    <canvas id="canvas"></canvas>
                </div>
                <br>
                <input type="range" id="rotSlider" min="-180" max="180" value="0" step="0.5"
                    style="width:80%; margin:8px 0;"
                    oninput="rotateImage(this.value)">
                <span id="angleLabel" style="font-weight:bold; margin-left:8px;">0°</span>
                <br>
                <button onclick="setRot(-90)" style="margin:2px; padding:4px 10px; cursor:pointer;">左转90°</button>
                <button onclick="setRot(-5)" style="margin:2px; padding:4px 10px; cursor:pointer;">左转5°</button>
                <button onclick="setRot(-1)" style="margin:2px; padding:4px 10px; cursor:pointer;">左转1°</button>
                <button onclick="setRot(0)" style="margin:2px; padding:4px 10px; cursor:pointer;">归零</button>
                <button onclick="setRot(1)" style="margin:2px; padding:4px 10px; cursor:pointer;">右转1°</button>
                <button onclick="setRot(5)" style="margin:2px; padding:4px 10px; cursor:pointer;">右转5°</button>
                <button onclick="setRot(90)" style="margin:2px; padding:4px 10px; cursor:pointer;">右转90°</button>
            </div>
            <script>
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            const wrap = document.getElementById('canvasWrap');
            const slider = document.getElementById('rotSlider');
            const label = document.getElementById('angleLabel');
            const img = new Image();
            img.src = 'data:image/png;base64,{img_b64_for_js}';
            let dispW = 1, dispH = 1;

            img.onload = function() {{
                const maxW = 480;
                const scale = Math.min(maxW / img.width, 1);
                dispW = img.width * scale;
                dispH = img.height * scale;
                wrap.style.width = dispW + 'px';
                wrap.style.height = dispH + 'px';
                canvas.width = dispW;
                canvas.height = dispH;
                drawRotated(0);
            }};

            function drawRotated(angle) {{
                const rad = angle * Math.PI / 180;
                const iw = img.width, ih = img.height;
                const cos = Math.abs(Math.cos(rad)), sin = Math.abs(Math.sin(rad));
                const needW = iw * cos + ih * sin;
                const needH = iw * sin + ih * cos;
                const s = Math.min(dispW / needW, dispH / needH);
                const dw = needW * s, dh = needH * s;
                canvas.width = dw;
                canvas.height = dh;
                ctx.clearRect(0, 0, dw, dh);
                ctx.save();
                ctx.translate(dw / 2, dh / 2);
                ctx.rotate(rad);
                ctx.drawImage(img, -iw * s / 2, -ih * s / 2, iw * s, ih * s);
                ctx.restore();
                label.textContent = parseFloat(angle).toFixed(1) + '°';
                slider.value = angle;
            }}

            function rotateImage(angle) {{
                drawRotated(parseFloat(angle));
            }}

            function setRot(a) {{
                slider.value = a;
                drawRotated(a);
            }}
            </script>
            """
            components.html(rotation_html, height=550)

            st.session_state['rotation'] = st.number_input(
                "输入旋转角度（正=右转，负=左转）", value=0.0, step=0.5, key="rot_input",
                help="在上方拖动滑块看到效果后，在此输入相同角度"
            )

            rot = st.session_state['rotation']
            if rot != 0:
                preview = pil_img.rotate(-rot, expand=True, fillcolor=(255, 255, 255))
            else:
                preview = pil_img
            st.image(preview, use_container_width=True, caption=f"旋转角度: {rot}°" if rot else "原图")

        with col_result:
            st.subheader("识别结果")

            if st.button("开始识别", type="primary", use_container_width=True):
                if not api_key:
                    st.error("请先在左侧设置中输入 API 密钥")
                else:
                    with st.spinner("AI 正在识别中..."):
                        rot = st.session_state.get('rotation', 0)
                        final_img = pil_img
                        if rot != 0:
                            final_img = pil_img.rotate(-rot, expand=True, fillcolor=(255, 255, 255))
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                            final_img.save(tmp, format='PNG')
                            tmp_path = tmp.name
                        try:
                            result = ai_recognize_image(tmp_path, api_key, model)
                            st.session_state['ocr_data'] = result
                            st.session_state['ocr_image_path'] = tmp_path
                            st.session_state['ocr_filename'] = uploaded_file.name
                            if 'original_data' in st.session_state:
                                del st.session_state['original_data']
                        except Exception as e:
                            st.error(f"识别失败: {e}")

            if 'ocr_data' in st.session_state and st.session_state['ocr_data']:
                result = st.session_state['ocr_data']
                data = result.get("data", [["识别失败"]]) if isinstance(result, dict) else result

                if 'original_data' not in st.session_state:
                    st.session_state['original_data'] = [row[:] for row in data]

                if st.button("在第一行插入空行", use_container_width=True):
                    n_cols = max(len(r) for r in data)
                    data = [r + [""] * (n_cols - len(r)) for r in data]
                    data = [[""] * n_cols] + data
                    st.session_state['ocr_data'] = {"success": True, "data": data, "method": "edited"}
                    st.session_state['original_data'] = [row[:] for row in data]
                    st.rerun()

                if len(data) > 1:
                    cols = []
                    seen = {}
                    for c in data[0]:
                        c = str(c) if c else ""
                        if c in seen:
                            seen[c] += 1; c = f"{c}_{seen[c]}"
                        else:
                            seen[c] = 0
                        cols.append(c)
                    df = pd.DataFrame(data[1:], columns=cols)
                else:
                    df = pd.DataFrame(data)

                cur_rows, cur_cols = len(df), len(df.columns)
                a1, a2 = st.columns(2)
                with a1:
                    new_rows = st.number_input("行数", 1, 100, cur_rows)
                with a2:
                    new_cols = st.number_input("列数", 1, 50, cur_cols)
                if new_rows > cur_rows:
                    for _ in range(new_rows - cur_rows):
                        df.loc[len(df)] = [""] * cur_cols
                elif new_rows < cur_rows:
                    df = df.iloc[:new_rows].reset_index(drop=True)
                if new_cols > cur_cols:
                    for i in range(new_cols - cur_cols):
                        df[f"列{cur_cols + i + 1}"] = ""
                elif new_cols < cur_cols:
                    df = df.iloc[:, :new_cols]

                st.markdown("### 识别结果（可直接编辑）")
                edited_df = st.data_editor(df, use_container_width=True, height=300, num_rows="dynamic", key="ed")

                # 操作按钮
                c1, c2, c3 = st.columns(3)
                with c1:
                    if st.button("记住修正", use_container_width=True, type="primary"):
                        original = st.session_state.get('original_data', [])
                        edited = edited_df.values.tolist()
                        if len(original) > 1:
                            edited = [list(original[0])] + edited
                        pairs = []
                        for r in range(min(len(original), len(edited))):
                            for c in range(min(len(original[r]), len(edited[r]))):
                                o = str(original[r][c] or "").strip()
                                e = str(edited[r][c] or "").strip()
                                if o and e and o.replace(" ", "").lower() != e.replace(" ", "").lower():
                                    pairs.append((o, e))
                        if pairs:
                            save_corrections(pairs)
                            st.success(f"已记住 {len(pairs)} 条修正")
                        else:
                            st.info("没有检测到修正")
                with c2:
                    if st.button("保存记录", use_container_width=True):
                        save_data = edited_df.values.tolist()
                        if len(data) > 1:
                            save_data = [list(data[0])] + save_data
                        rid = save_record(st.session_state.get('ocr_image_path', ''), save_data)
                        st.success(f"已保存 (ID: {rid})")
                with c3:
                    export_data = edited_df.values.tolist()
                    if len(data) > 1:
                        export_data = [list(data[0])] + export_data
                    st.download_button("下载 Excel", data=export_to_excel(export_data),
                                       file_name=os.path.splitext(st.session_state.get('ocr_filename', '结果'))[0] + ".xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                       use_container_width=True)

                # ============================================================
                #  数据计算
                # ============================================================
                st.markdown("---")
                st.markdown("### 数据计算")

                col_list = list(edited_df.columns)
                row_list = list(range(len(edited_df)))
                func_list = ["平均数", "求和", "最大值", "最小值", "中位数", "标准差", "方差", "计数"]
                func_map = {
                    "平均数": calc_average, "求和": calc_sum,
                    "最大值": calc_max, "最小值": calc_min,
                    "中位数": calc_median, "标准差": calc_std,
                    "方差": calc_variance, "计数": calc_count,
                }

                calc_tab1, calc_tab2, calc_tab3 = st.tabs(["列计算", "行计算", "最小二乘法"])

                # 列计算
                with calc_tab1:
                    if not col_list:
                        st.info("暂无数据列")
                    else:
                        c_func = st.selectbox("选择函数", func_list, key="col_func")
                        selected_cols = st.multiselect("选择要计算的列", col_list,
                                                       default=[col_list[0]] if col_list else [], key="sel_cols")
                        if st.button("执行列计算", use_container_width=True, key="run_col"):
                            if not selected_cols:
                                st.warning("请先选择要计算的列")
                            else:
                                fn = func_map[c_func]
                                result_row = []
                                for cn in col_list:
                                    if cn in selected_cols:
                                        val = fn(edited_df[cn].tolist())
                                        result_row.append(str(val) if val != "" else "")
                                    else:
                                        result_row.append("")
                                st.markdown(f"**{c_func}结果：**")
                                res_df = pd.DataFrame([result_row], columns=col_list)
                                st.dataframe(res_df, use_container_width=True)

                        if st.button("将列计算结果追加到表格末行", use_container_width=True, key="add_col"):
                            fn = func_map[c_func]
                            result_row = []
                            for cn in col_list:
                                if cn in selected_cols:
                                    val = fn(edited_df[cn].tolist())
                                    result_row.append(str(val) if val != "" else "")
                                else:
                                    result_row.append("")
                            new_data = edited_df.values.tolist()
                            new_data.append(result_row)
                            st.session_state['ocr_data'] = {"success": True, "data": [col_list] + new_data, "method": "calc"}
                            st.rerun()

                # 行计算
                with calc_tab2:
                    if not col_list:
                        st.info("暂无数据列")
                    else:
                        r_func = st.selectbox("选择函数", func_list, key="row_func")
                        selected_rows = st.multiselect("选择要计算的行（行号从0开始）", row_list,
                                                       default=row_list[:1] if row_list else [], key="sel_rows",
                                                       format_func=lambda x: f"第 {x} 行")
                        selected_cols_row = st.multiselect("对哪些列做计算", col_list,
                                                           default=col_list, key="sel_cols_row")
                        if st.button("执行行计算", use_container_width=True, key="run_row"):
                            if not selected_rows or not selected_cols_row:
                                st.warning("请先选择行和列")
                            else:
                                fn = func_map[r_func]
                                st.markdown(f"**{r_func}结果（每行在选定列上的计算）：**")
                                res_rows = []
                                for ri in selected_rows:
                                    row_vals = [edited_df.iloc[ri][cn] for cn in selected_cols_row]
                                    val = fn(row_vals)
                                    display = [f"第{ri}行", str(val) if val != "" else ""]
                                    res_rows.append(display)
                                res_df = pd.DataFrame(res_rows, columns=["行号", f"{r_func}"])
                                st.dataframe(res_df, use_container_width=True)

                        # 行计算结果追加为新列
                        if st.button("将行计算结果追加为新列", use_container_width=True, key="add_row"):
                            fn = func_map[r_func]
                            new_col_name = f"{r_func}_结果"
                            new_col_vals = []
                            for ri in range(len(edited_df)):
                                if ri in selected_rows:
                                    row_vals = [edited_df.iloc[ri][cn] for cn in selected_cols_row]
                                    val = fn(row_vals)
                                    new_col_vals.append(str(val) if val != "" else "")
                                else:
                                    new_col_vals.append("")
                            new_data = edited_df.values.tolist()
                            new_cols = col_list + [new_col_name]
                            for i, row in enumerate(new_data):
                                row.append(new_col_vals[i])
                            st.session_state['ocr_data'] = {"success": True, "data": [new_cols] + new_data, "method": "calc"}
                            st.rerun()

                # 最小二乘法
                with calc_tab3:
                    ls_mode = st.radio("拟合方式",
                                       ["列对列（每行一组数据）", "行对行（每列一组数据）", "列对行（一列做X，一行做Y）"],
                                       horizontal=True, key="ls_mode")

                    if ls_mode == "列对列（每行一组数据）":
                        if len(col_list) < 2:
                            st.info("至少需要两列数据")
                        else:
                            ls1, ls2 = st.columns(2)
                            with ls1:
                                x_col = st.selectbox("X 列", col_list, index=0, key="ls_x")
                            with ls2:
                                y_col = st.selectbox("Y 列", col_list, index=min(1, len(col_list)-1), key="ls_y")
                            if st.button("执行拟合", use_container_width=True, key="ls_run"):
                                a, b, r2 = calc_least_squares(
                                    edited_df[x_col].tolist(),
                                    edited_df[y_col].tolist()
                                )
                                if a is not None:
                                    st.success(f"拟合结果：y = {a}x + {b}")
                                    st.info(f"决定系数 R² = {r2}")
                                else:
                                    st.error("数据不足或格式错误，无法拟合")

                    elif ls_mode == "行对行（每列一组数据）":
                        if len(row_list) < 2:
                            st.info("至少需要两行数据")
                        else:
                            ls3, ls4 = st.columns(2)
                            with ls3:
                                x_row = st.selectbox("X 行（行号）", row_list, index=0, key="ls_xrow",
                                                     format_func=lambda x: f"第 {x} 行")
                            with ls4:
                                y_row = st.selectbox("Y 行（行号）", row_list, index=min(1, len(row_list)-1), key="ls_yrow",
                                                     format_func=lambda x: f"第 {x} 行")
                            if st.button("执行拟合", use_container_width=True, key="ls_run2"):
                                x_data = edited_df.iloc[x_row].tolist()
                                y_data = edited_df.iloc[y_row].tolist()
                                a, b, r2 = calc_least_squares(x_data, y_data)
                                if a is not None:
                                    st.success(f"拟合结果：y = {a}x + {b}")
                                    st.info(f"决定系数 R² = {r2}")
                                else:
                                    st.error("数据不足或格式错误，无法拟合")

                    else:
                        if not col_list or not row_list:
                            st.info("需要至少一列和一行数据")
                        else:
                            lc1, lc2 = st.columns(2)
                            with lc1:
                                ls_col_x = st.selectbox("X 数据来源", col_list, index=0, key="ls_colrow_x",
                                                        help="选一列，该列所有数值作为 X")
                            with lc2:
                                ls_row_y = st.selectbox("Y 数据来源", row_list, index=0, key="ls_colrow_y",
                                                        format_func=lambda x: f"第 {x} 行",
                                                        help="选一行，该行所有数值作为 Y")
                            if st.button("执行拟合", use_container_width=True, key="ls_run3"):
                                x_data = edited_df[ls_col_x].tolist()
                                y_data = edited_df.iloc[ls_row_y].tolist()
                                # 取较短的长度配对
                                n = min(len(x_data), len(y_data))
                                a, b, r2 = calc_least_squares(x_data[:n], y_data[:n])
                                if a is not None:
                                    st.success(f"拟合结果：y = {a}x + {b}")
                                    st.info(f"决定系数 R² = {r2}（使用了 {n} 组数据点）")
                                else:
                                    st.error("数据不足或格式错误，无法拟合")

with tab2:
    st.subheader("历史记录")
    records = get_all_records()
    if not records:
        st.info("暂无记录")
    else:
        for record in records:
            with st.expander(f"#{record['id']} | {record['row_count']}行×{record['col_count']}列 | {record['created_at']}"):
                detail = get_record(record['id'])
                if detail and detail.get('data'):
                    hd = detail['data']
                    mc = max(len(r) for r in hd)
                    hd = [r + [""] * (mc - len(r)) for r in hd]
                    if len(hd) > 1:
                        sc = []; sn = {}
                        for c in hd[0]:
                            c = str(c) if c else ""
                            if c in sn: sn[c] += 1; c = f"{c}_{sn[c]}"
                            else: sn[c] = 0
                            sc.append(c)
                        df = pd.DataFrame(hd[1:], columns=sc)
                    else:
                        df = pd.DataFrame(hd)
                    st.dataframe(df, use_container_width=True)
                    dc1, dc2 = st.columns(2)
                    with dc1:
                        st.download_button("下载", data=export_to_excel(detail['data']),
                                           file_name=f"记录_{record['id']}.xlsx",
                                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                           key=f"dl{record['id']}")
                    with dc2:
                        if st.button("删除", key=f"del{record['id']}"):
                            delete_record(record['id']); st.rerun()

with tab3:
    st.subheader("学习记录")
    corrections = get_all_corrections()
    if not corrections:
        st.info("暂无记录。识别后编辑修正，点击「记住修正」即可积累。")
    else:
        st.caption(f"共 {len(corrections)} 条")
        for key, entry in corrections.items():
            c1, c2, c3 = st.columns([2, 2, 1])
            with c1: st.write(f"**识别为:** {entry['ocr']}")
            with c2: st.write(f"**修正为:** {entry['corrected']}")
            with c3:
                if st.button("删除", key=f"dc_{key}"):
                    delete_correction(entry['ocr']); st.rerun()

st.markdown("---")
st.caption("通义千问 AI 视觉识别 | 自动纠偏 | 实时旋转 | 手写识别 | 自动学习修正 | 内置计算函数")
