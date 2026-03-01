import os
import re
import math
import uuid
import subprocess
import uvicorn
import logging
import json
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image  # 追加
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ts_solver")

app = FastAPI()
# T-Astro Web Studioからのアクセスを許可
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

WORK_DIR = "/tmp/sol"
DB_FILE = "astro_db.json"
os.makedirs(WORK_DIR, exist_ok=True)

def load_astro_db():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f:
                return json.load(f)
        except: pass
    return []

def wcs_to_pixel_perfect(ra, dec, wcs, img_w, img_h):
    try:
        rad = math.pi / 180.0
        alpha0, delta0 = wcs['crval1'] * rad, wcs['crval2'] * rad
        alpha, delta = ra * rad, dec * rad
        d_alpha = alpha - alpha0
        denom = math.sin(delta) * math.sin(delta0) + math.cos(delta) * math.cos(delta0) * math.cos(d_alpha)
        if denom <= 0: return None
        xi = (math.cos(delta) * math.sin(d_alpha)) / denom
        eta = (math.sin(delta) * math.cos(delta0) - math.cos(delta) * math.sin(delta0) * math.cos(d_alpha)) / denom
        xi_deg, eta_deg = xi / rad, eta / rad
        det = wcs['cd1_1'] * wcs['cd2_2'] - wcs['cd1_2'] * wcs['cd2_1']
        dx = (wcs['cd2_2'] * xi_deg - wcs['cd1_2'] * eta_deg) / det
        dy = (-wcs['cd2_1'] * xi_deg + wcs['cd1_1'] * eta_deg) / det
        return {"x": wcs['crpix1'] + dx - 1.0, "y": wcs['crpix2'] + dy - 1.0}
    except: return None

def parse_wcs_and_annotate(wcs_path, img_w, img_h):
    if not os.path.exists(wcs_path): return None
    db = load_astro_db()
    try:
        with open(wcs_path, "r", errors="ignore") as f: content = f.read()
        def get_v(k):
            m = re.search(rf"{k}\s*=\s*([+-]?[\d\.E\+\-]+)", content)
            return float(m.group(1)) if m else None
        
        wcs = { 
            'crval1': get_v('CRVAL1'), 'crval2': get_v('CRVAL2'), 
            'crpix1': get_v('CRPIX1'), 'crpix2': get_v('CRPIX2'),
            'cd1_1': get_v('CD1_1'), 'cd1_2': get_v('CD1_2'),
            'cd2_1': get_v('CD2_1'), 'cd2_2': get_v('CD2_2') 
        }
        
        if wcs['crval1'] is None: return None

        # --- T-Astro Web Studio の座標同期(Sync)に必要な計算 ---
        # 行列式からピクセルスケール(arcsec/pixel)を算出
        det = wcs['cd1_1'] * wcs['cd2_2'] - wcs['cd1_2'] * wcs['cd2_1']
        scale = math.sqrt(abs(det)) * 3600.0
        # パリティ（反転状態）
        parity = 1 if det > 0 else -1
        # 回転角
        rotation = math.degrees(math.atan2(wcs['cd1_2'], wcs['cd1_1']))
        # 視野半径（概算）
        actual_w = get_v('IMAGEW') or img_w
        actual_h = get_v('IMAGEH') or img_h
        radius = (scale * max(actual_w, actual_h) / 3600.0) / 2.0

        # アノテーション（既存データベース照合）
        ans = []
        for obj in db:
            p = wcs_to_pixel_perfect(obj['ra'], obj['dec'], wcs, actual_w, actual_h)
            if p and 0 <= p['x'] <= actual_w and 0 <= p['y'] <= actual_h:
                ans.append({"x": p['x'], "y": p['y'], "names": [obj['name']], "radius": 15})

        # T-Astro Web Studio の PlateSolvingService.ts が同期命令を出すために必要な形式
        return {
            "calibration": {
                "ra": wcs['crval1'],
                "dec": wcs['crval2'],
                "rotation": rotation,
                "scale": scale,
                "parity": parity,
                "radius": radius
            },
            "annotations": ans,
            "width": actual_w,
            "height": actual_h
        }
    except Exception as e:
        logger.error(f"WCS Parse Error: {e}")
        return None

@app.get("/", response_class=HTMLResponse)
async def index():
    db_json = json.dumps(load_astro_db())
    
    # 完全に復元したHTMLコンソール（元のUIを維持）
    html_template = r"""
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <title>TS-Solver Console</title>
        <style>
            :root {
                --bg-dark: #0f172a; --panel-bg: #1e293b; --accent-red: #e11d48;
                --text-main: #f1f5f9; --text-dim: #94a3b8; --border: #334155; --input-bg: #0f172a;
            }
            body { 
                font-family: sans-serif; background: var(--bg-dark); color: var(--text-main); 
                margin: 0; padding: 20px; display: flex; justify-content: center;
            }
            .container { 
                width: 100%; max-width: 800px; background: var(--panel-bg); padding: 25px; 
                border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); border: 1px solid var(--border);
            }
            header {
                display: flex; align-items: center; margin-bottom: 20px;
                border-bottom: 2px solid var(--accent-red); padding-bottom: 10px;
            }
            header h2 { margin: 0; font-size: 1.4rem; letter-spacing: 1px; color: #fff; }
            .section-title { font-size: 0.8rem; color: var(--accent-red); font-weight: bold; margin: 15px 0 8px 5px; text-transform: uppercase; }
            .section { 
                background: rgba(15, 23, 42, 0.4); border: 1px solid var(--border); 
                padding: 15px; border-radius: 8px; margin-bottom: 15px; 
            }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
            label { display: block; font-size: 0.7rem; color: var(--text-dim); margin-bottom: 4px; }
            input { 
                width: 100%; padding: 10px; background: var(--input-bg); border: 1px solid var(--border); 
                color: white; border-radius: 4px; box-sizing: border-box; font-size: 0.9rem;
            }
            .search-box { display: flex; gap: 8px; }
            button { 
                padding: 10px 20px; background: var(--accent-red); color: white; border: none; 
                border-radius: 4px; cursor: pointer; font-weight: bold; transition: opacity 0.2s;
            }
            button:hover { opacity: 0.8; }
            .search-btn { background: #475569; }
            .solve-btn { width: 100%; font-size: 1rem; margin-top: 10px; background: #e11d48; }
            pre { 
                background: #000; color: #10b981; padding: 15px; border-radius: 6px; 
                overflow: auto; max-height: 250px; font-size: 0.8rem; border: 1px solid #1e293b;
            }
        </style>
    </head>
    <body onload="loadSettings()">
        <div class="container">
            <header><h2>🔭 TSPS CONSOLE</h2></header>
            <div class="section-title">Object Search</div>
            <div class="section">
                <div class="search-box">
                    <input type="text" id="objName" placeholder="M31, M42, NGC...">
                    <button type="button" class="search-btn" onclick="searchObject()">SEARCH</button>
                </div>
            </div>
            <form id="solveForm">
                <div class="section-title">Image & Position</div>
                <div class="section">
                    <input type="file" name="file" required>
                    <div class="grid" style="margin-top:12px;">
                        <div><label>RA Hint (deg)</label><input type="number" id="ra" name="ra" step="any"></div>
                        <div><label>Dec Hint (deg)</label><input type="number" id="dec" name="dec" step="any"></div>
                        <div><label>Radius (deg)</label><input type="number" id="radius" name="radius" value="15"></div>
                    </div>
                </div>
                <div class="section-title">Settings</div>
                <div class="section">
                    <div class="grid">
                        <div><label>Downsample</label><input type="number" id="downsample" name="downsample" value="2"></div>
                        <div><label>SNR (Sigma)</label><input type="number" id="snr" name="snr" value="5"></div>
                        <div><label>Limit (sec)</label><input type="number" id="cpulimit" name="cpulimit" value="60"></div>
                    </div>
                    <div style="margin-top:12px;">
                        <label>Custom Options</label>
                        <input type="text" id="custom_args" name="custom_args" value="--scale-units degwidth --scale-low 1 --scale-high 10 --guess-scale --no-plots --no-verify --no-remove-lines --uniformize 0">
                    </div>
                </div>
                <button type="button" class="solve-btn" onclick="runSolve()">PLATE SOLVE</button>
            </form>
            <div class="section-title">Log</div>
            <pre id="out">// System Ready.</pre>
        </div>
        <script>
            const astroDB = JSON.parse('{{DB_JSON}}');
            function searchObject() {
                const val = document.getElementById('objName').value.toUpperCase().replace(/\s/g, '');
                const obj = astroDB.find(o => o.name.toUpperCase() === val);
                if (obj) {
                    document.getElementById('ra').value = obj.ra;
                    document.getElementById('dec').value = obj.dec;
                } else { alert("Target not found"); }
            }
            function saveSettings() {
                const cfg = {
                    radius: document.getElementById('radius').value,
                    downsample: document.getElementById('downsample').value,
                    snr: document.getElementById('snr').value,
                    cpulimit: document.getElementById('cpulimit').value,
                    custom_args: document.getElementById('custom_args').value
                };
                localStorage.setItem('ts_solver_v3', JSON.stringify(cfg));
            }
            function loadSettings() {
                const saved = localStorage.getItem('ts_solver_v3');
                if (saved) {
                    const s = JSON.parse(saved);
                    document.getElementById('radius').value = s.radius || 15;
                    document.getElementById('downsample').value = s.downsample || 2;
                    document.getElementById('snr').value = s.snr || 5;
                    document.getElementById('cpulimit').value = s.cpulimit || 60;
                    document.getElementById('custom_args').value = s.custom_args || "--scale-units degwidth --scale-low 1 --scale-high 10 --guess-scale --no-plots --no-verify --no-remove-lines --uniformize 0";
                }
            }
            async function runSolve(){
                saveSettings();
                const out = document.getElementById('out');
                out.innerText = "Analyzing...";
                try {
                    const formData = new FormData(document.getElementById('solveForm'));
                    const resp = await fetch('/solve', { method: 'POST', body: formData });
                    const result = await resp.json();
                    out.innerText = JSON.stringify(result, null, 2);
                } catch (e) { out.innerText = "Error: " + e; }
            }
        </script>
    </body>
    </html>
    """.replace("{{DB_JSON}}", db_json)
    
    return HTMLResponse(content=html_template)

@app.post("/solve")
async def solve_api(
    file: UploadFile = File(...), 
    ra: Optional[float] = Form(None), 
    dec: Optional[float] = Form(None), 
    radius: Optional[float] = Form(15.0), 
    snr: int = Form(5),
    downsample: int = Form(2),
    cpulimit: int = Form(60),
    custom_args: str = Form(None)
):
    sid = str(uuid.uuid4())
    img_path = os.path.join(WORK_DIR, f"{sid}.jpg")
    
    # 画像の保存
    img_data = await file.read()
    with open(img_path, "wb") as f:
        f.write(img_data)
    
    # --- 追加: 画像の実際のサイズを取得 ---
    try:
        with Image.open(img_path) as img_file:
            actual_w, actual_h = img_file.size
    except Exception as e:
        logger.error(f"Image open error: {e}")
        actual_w, actual_h = 1000.0, 1000.0  # 失敗時のフォールバック
    
    # solve-fieldコマンドの構築
    cmd = [
        "solve-field", img_path, "--overwrite", "--no-plots", 
        "--cpulimit", str(cpulimit), 
        "--downsample", str(downsample),
        "--sigma", str(snr) 
    ]
    if ra is not None and dec is not None:
        cmd.extend(["--ra", str(ra), "--dec", str(dec), "--radius", str(radius)])
    if custom_args:
        # 重複指定を避けるための処理
        cmd.extend(custom_args.replace("--snr", "--sigma").split())
    
    logger.info(f"Executing: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=WORK_DIR, capture_output=True, text=True)
    
    # WCSファイルのパース
    # T-Astro側でリサイズされて送られてくるケースを考慮し、画像サイズでフォールバック
    res = parse_wcs_and_annotate(img_path.replace(".jpg", ".wcs"), float(actual_w), float(actual_h))
    
    # 不要なファイルの削除
    for ext in [".jpg", ".wcs", ".solved", ".rdls", ".axy", ".match", ".xyls", ".new"]:
        p = img_path.replace(".jpg", ext)
        if os.path.exists(p): os.remove(p)
    
    if res:
        # T-Astro Web Studio の Auto Center (AstroService.syncToCoordinates) が起動するレスポンス
        return {
            "status": "success",
            "calibration": res["calibration"],
            "annotations": res["annotations"],
            "imageWidth": res["width"],
            "imageHeight": res["height"]
        }
    else:
        return {"status": "failed", "log": proc.stderr[-500:] if proc.stderr else "Solve failed."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6001)