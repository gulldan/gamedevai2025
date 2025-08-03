#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import json
import os

# --- импорт вашего генератора ---
import sys
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from flask import (
    Flask,
    abort,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import (
    clear_memory,
    generate_textured_mesh_from_image,
    load_image_gen_pipeline,
)

# ---------------------- Flask & FS ----------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "devkey")

BASE_DIR   = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"
RESULTS_DIR = STATIC_DIR / "results"
for p in [STATIC_DIR, UPLOADS_DIR, RESULTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

HISTORY_INDEX = RESULTS_DIR / "history_index.json"

# ---------------------- Настройки ----------------------
# фиксированные сиды для текстового ввода (3 варианта)
TEXT_SEEDS = [52, 1337, 3407]

# Для Playground v2.5: строго фронтально/ортографично, один объект; разнообразие только стилем
DIVERSITY_STYLES = [
    "low-poly, faceted, hard edges, matte surface, simplified silhouette, game-ready",
    "low-poly, hand-painted look, subtle edge wear, limited color palette, game asset",
    "low-poly, rounded, soft bevels, clean topology, minimal details, stylized",
]

# Жёсткий негатив (фон/сцена/тени/лишние объекты запрещены)
NEGATIVE_PROMPT = (
    "background, black background, dark background, scenery, environment, sky, room, floor, ground, "
    "shadow, ground shadow, multiple objects, extra parts, reflections, text, watermark, logo, caption, clutter, people"
)

# Параметры очистки краёв маски (после rembg)
EDGE_ERODE_PX = 2         # сжать маску на N пикселей (минимум артефактов по краям)
EDGE_FEATHER_PX = 1.5     # мягкое перо/размытие контура
EDGE_AUTOCONTRAST = True  # чуть растянуть уровни альфы

# ---------------------- Утилиты ----------------------
def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _t() -> float:
    return time.perf_counter()

def _ms_since(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0

def _fmt_ms(ms: float) -> str:
    return f"{ms/1000.0:.2f}s"

def _safe_tri_count(value: str, default: int = 2000) -> int:
    try:
        n = int(value)
    except Exception:
        n = default
    return max(1000, min(2000, n))  # требование 1000–2000

def _load_index() -> List[Dict[str, Any]]:
    if HISTORY_INDEX.exists():
        try:
            return json.loads(HISTORY_INDEX.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def _save_index(index: List[Dict[str, Any]]) -> None:
    HISTORY_INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

def _add_to_index(entry: Dict[str, Any]) -> None:
    idx = _load_index()
    idx.append(entry)
    idx.sort(key=lambda e: e.get("created_at", ""), reverse=True)
    _save_index(idx)

def _save_batch_meta(batch_id: str, meta: Dict[str, Any]) -> None:
    (RESULTS_DIR / batch_id).mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / batch_id / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def _load_batch_meta(batch_id: str) -> Dict[str, Any]:
    meta_path = RESULTS_DIR / batch_id / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("meta.json not found")
    return json.loads(meta_path.read_text(encoding="utf-8"))

# --------- Удаление фона и очистка краёв ----------
def _try_import_rembg_session():
    try:
        from rembg import new_session
        return new_session("u2net")
    except Exception:
        return None

def _remove_bg_bytes(data: bytes, session) -> bytes:
    try:
        from rembg import remove
        cut = remove(
            data,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=max(1, EDGE_ERODE_PX * 2),
        )
        return cut
    except Exception:
        try:
            from rembg import remove
            return remove(data)
        except Exception:
            return data

def _clean_alpha_edges_rgba(im_rgba) -> "Image.Image":
    """Эрозия/перо/контраст по альфе для уменьшения ореолов."""
    from PIL import Image, ImageFilter, ImageOps
    if im_rgba.mode != "RGBA":
        im_rgba = im_rgba.convert("RGBA")
    r, g, b, a = im_rgba.split()

    # Эрозия
    if EDGE_ERODE_PX > 0:
        k = EDGE_ERODE_PX * 2 + 1
        a = a.filter(ImageFilter.MinFilter(size=k))

    # Перо/размытие
    if EDGE_FEATHER_PX > 0:
        a = a.filter(ImageFilter.GaussianBlur(radius=EDGE_FEATHER_PX))

    # Чуть подтянуть контраст маски, чтобы край был чище
    if EDGE_AUTOCONTRAST:
        a = ImageOps.autocontrast(a, cutoff=1)

    return Image.merge("RGBA", (r, g, b, a))

def _crop_to_content_rgba(im) -> Tuple[int,int,int,int]:
    """BBox по альфе. Возвращает (l,t,r,b)."""
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    alpha = im.split()[-1]
    bbox = alpha.getbbox()
    if bbox:
        return bbox
    return (0, 0, im.width, im.height)

def _prep_ref_images(raw_in: Path, out_cut: Path, out_ref512: Path) -> Tuple[Path, Path, Dict[str, Any]]:
    """
    1) Удаляем фон (rembg) -> RGBA (out_cut, с чисткой краёв).
    2) Кадрируем по объекту + ресайз до 512 + центрирование.
    3) Для 3D пайплайна готовим out_ref512 на белом фоне (RGB).
    Возвращает (путь_png_UI, путь_png_для_3D, timings_dict).
    """
    from PIL import Image

    timings = {"bg_remove_ms": 0.0, "prep_ms": 0.0}

    # RAW -> RGBA
    t0 = _t()
    im = Image.open(raw_in).convert("RGBA")

    # rembg
    session = _try_import_rembg_session()
    if session is not None:
        cut_bytes = _remove_bg_bytes(raw_in.read_bytes(), session)
        try:
            im = Image.open(io.BytesIO(cut_bytes)).convert("RGBA")
        except Exception:
            pass
    timings["bg_remove_ms"] = _ms_since(t0)

    # очистка краёв
    t1 = _t()
    im = _clean_alpha_edges_rgba(im)

    # обрезка по альфе с полем
    l, t, r, b = _crop_to_content_rgba(im)
    margin = 10
    l = max(0, l - margin); t = max(0, t - margin)
    r = min(im.width,  r + margin); b = min(im.height, b + margin)
    im = im.crop((l, t, r, b))

    # ресайз до 512 и центрирование
    im.thumbnail((512, 512), Image.Resampling.LANCZOS)
    out_cut.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_cut)  # RGBA для UI

    # белый фон для 3D пайплайна
    bg = Image.new("RGB", (512, 512), (255, 255, 255))
    bg.paste(im, ((512 - im.width)//2, (512 - im.height)//2), im)
    bg.save(out_ref512)
    timings["prep_ms"] = _ms_since(t1)

    return out_cut, out_ref512, timings

# ---- Diffusers helpers (Playground v2.5, фронтально/без фона) ----
def _gen_image_playground(pipe, base_prompt: str, style_suffix: str, seed: int, out_png: Path) -> Dict[str, Any]:
    """
    Строго фронтально, один объект, без фона (идеально — белый; затем фон всё равно вырежется).
    Возвращает dict с полями: final_prompt, image_gen_ms.
    """
    # Сформировать жёсткий промпт
    prompt = (
        f"{base_prompt}, "
        f"single object, centered composition, front view, orthographic, "
        f"{style_suffix}, "
        f"low poly, 3d model, clean silhouette, "
        f"pure white background, isolated on white, product studio, no props"
    )

    gen = None
    try:
        import torch
        gen = torch.Generator("cpu").manual_seed(seed)
    except Exception:
        pass

    guidance = 3.0
    steps = 22

    t0 = _t()
    try:
        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            guidance_scale=guidance,
            num_inference_steps=steps,
            width=1024,
            height=1024,
            generator=gen,
        ).images[0]
    except TypeError:
        image = pipe(prompt=prompt, generator=gen).images[0]
    image_gen_ms = _ms_since(t0)

    image.convert("RGB").save(out_png)

    return {
        "final_prompt": prompt,
        "guidance": guidance,
        "steps": steps,
        "image_gen_ms": image_gen_ms,
    }

# ---------------------- Маршруты ----------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        face_count = _safe_tri_count(request.form.get("target_face_count", "2000"))
        use_remesh = request.form.get("use_remesh", "off") == "on"

        prompt = (request.form.get("prompt") or "").strip()
        file = request.files.get("image")

        if not (file and file.filename) and not prompt:
            flash("Нужно указать либо картинку, либо текстовый промпт.")
            return redirect(url_for("index"))

        batch_id = f"batch_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        created_at = _now_iso()
        out_items: List[Dict[str, Any]] = []
        input_type = "image" if (file and file.filename) else "text"

        # ====== КАРТИНКА -> 1 модель ======
        if input_type == "image":
            src_ext = Path(file.filename).suffix.lower() or ".png"
            src_name = f"{batch_id}_source{src_ext}"
            src_abs = UPLOADS_DIR / src_name
            file.save(src_abs)

            item_id = f"{batch_id}_1"
            item_dir = RESULTS_DIR / item_id
            item_dir.mkdir(parents=True, exist_ok=True)

            raw_ref = item_dir / f"{item_id}_raw.png"
            cut_ref = item_dir / f"{item_id}_cut.png"     # PNG с альфой для UI
            ref_512 = item_dir / f"{item_id}_ref.png"     # RGB 512x512 для 3D пайплайна
            glb_path = item_dir / f"{item_id}.glb"
            obj_path = item_dir / f"{item_id}.obj"

            # Сохраним RAW (PNG)
            from PIL import Image
            Image.open(src_abs).convert("RGB").save(raw_ref)

            # Подготовка референса: удаление фона/очистка краёв/сборка 512
            t_prep0 = _t()
            _, _, prep_times = _prep_ref_images(raw_ref, cut_ref, ref_512)
            prep_ms = prep_times["bg_remove_ms"] + prep_times["prep_ms"]

            # Генерация 3D
            t_mesh0 = _t()
            ok = False
            stage_log = []
            try:
                ok = generate_textured_mesh_from_image(
                    image_path=str(ref_512),
                    output_path=str(glb_path),
                    use_remesh=use_remesh,
                    target_face_count=face_count,
                )
            except Exception as e:
                stage_log.append(f"Error: {e}")
                ok = False
            mesh_ms = _ms_since(t_mesh0)

            model_rel = None
            if glb_path.exists():
                model_rel = f"results/{item_id}/{glb_path.name}"
            elif obj_path.exists():
                model_rel = f"results/{item_id}/{obj_path.name}"

            ref_img_rel = f"results/{item_id}/{cut_ref.name}" if cut_ref.exists() else f"results/{item_id}/{raw_ref.name}"

            timings = {
                "image_gen_ms": 0.0,
                "prep_ms": prep_ms,
                "bg_remove_ms": prep_times["bg_remove_ms"],
                "mesh_ms": mesh_ms,
                "total_ms": prep_ms + mesh_ms,
                "image_gen_text": "—",
                "prep_text": _fmt_ms(prep_ms),
                "bg_remove_text": _fmt_ms(prep_times["bg_remove_ms"]),
                "mesh_text": _fmt_ms(mesh_ms),
                "total_text": _fmt_ms(prep_ms + mesh_ms),
            }

            out_items.append({
                "ok": ok and (model_rel is not None),
                "prompt": prompt,
                "seed": None,
                "ref_img_rel": ref_img_rel,
                "model_rel": model_rel,
                "log": stage_log,
                "title": "Результат по изображению (bg removed + edge clean)",
                "timings": timings,
            })

            total_ms_sum = timings["total_ms"]
            meta = {
                "batch_id": batch_id,
                "created_at": created_at,
                "input_type": input_type,
                "prompt": prompt,
                "target_face_count": face_count,
                "use_remesh": use_remesh,
                "cover_rel": ref_img_rel,
                "items": out_items,
                "total_ms": total_ms_sum,
                "total_text": _fmt_ms(total_ms_sum),
            }
            _save_batch_meta(batch_id, meta)
            _add_to_index({
                "batch_id": batch_id,
                "created_at": created_at,
                "input_type": input_type,
                "prompt": prompt,
                "cover_rel": ref_img_rel,
                "items_count": len(out_items),
                "total_text": _fmt_ms(total_ms_sum),
            })

        # ====== ТЕКСТ -> 3 варианта (строго фронтально, удаление фона, очистка краёв) ======
        else:
            ref_cover_rel = None
            total_ms_sum = 0.0
            for i, (seed, style_suffix) in enumerate(zip(TEXT_SEEDS, DIVERSITY_STYLES), start=1):
                item_id = f"{batch_id}_{i}"
                item_dir = RESULTS_DIR / item_id
                item_dir.mkdir(parents=True, exist_ok=True)

                raw_ref = item_dir / f"{item_id}_raw.png"
                cut_ref = item_dir / f"{item_id}_cut.png"
                ref_512 = item_dir / f"{item_id}_ref.png"
                glb_path = item_dir / f"{item_id}.glb"
                obj_path = item_dir / f"{item_id}.obj"

                stage_log = []
                ok = False
                final_prompt = prompt
                used = {}

                # 1) Генерация изображения (Playground v2.5)
                img_ms = 0.0
                try:
                    pipe = load_image_gen_pipeline()
                    if pipe is None:
                        stage_log.append("Image pipeline not available")
                        ok = False
                    else:
                        res = _gen_image_playground(
                            pipe, prompt, style_suffix, seed, raw_ref
                        )
                        final_prompt = res["final_prompt"]
                        img_ms = res["image_gen_ms"]
                        used = {"guidance": res["guidance"], "steps": res["steps"]}
                        # выгрузить пайплайн
                        try: del pipe
                        except Exception: pass
                        clear_memory()
                except Exception as e:
                    stage_log.append(f"Error@image_gen: {e}")
                    ok = False

                if i == 1:
                    ref_cover_rel = f"results/{item_id}/{raw_ref.name}"

                # 2) Удаление фона / очистка краёв / сборка 512x512
                prep_times = {"bg_remove_ms": 0.0, "prep_ms": 0.0}
                try:
                    _, _, prep_times = _prep_ref_images(raw_ref, cut_ref, ref_512)
                except Exception as e:
                    stage_log.append(f"Error@prep: {e}")

                # 3) Генерация 3D
                t_mesh0 = _t()
                try:
                    ok = generate_textured_mesh_from_image(
                        image_path=str(ref_512),
                        output_path=str(glb_path),
                        use_remesh=use_remesh,
                        target_face_count=face_count,
                    )
                except Exception as e:
                    stage_log.append(f"Error@mesh: {e}")
                    ok = False
                mesh_ms = _ms_since(t_mesh0)

                model_rel = None
                if glb_path.exists():
                    model_rel = f"results/{item_id}/{glb_path.name}"
                elif obj_path.exists():
                    model_rel = f"results/{item_id}/{obj_path.name}"

                ref_img_rel = f"results/{item_id}/{cut_ref.name}" if cut_ref.exists() else f"results/{item_id}/{raw_ref.name}"

                total_ms = img_ms + prep_times["bg_remove_ms"] + prep_times["prep_ms"] + mesh_ms
                total_ms_sum += total_ms

                timings = {
                    "image_gen_ms": img_ms,
                    "prep_ms": prep_times["bg_remove_ms"] + prep_times["prep_ms"],
                    "bg_remove_ms": prep_times["bg_remove_ms"],
                    "mesh_ms": mesh_ms,
                    "total_ms": total_ms,
                    "image_gen_text": _fmt_ms(img_ms),
                    "prep_text": _fmt_ms(prep_times["bg_remove_ms"] + prep_times["prep_ms"]),
                    "bg_remove_text": _fmt_ms(prep_times["bg_remove_ms"]),
                    "mesh_text": _fmt_ms(mesh_ms),
                    "total_text": _fmt_ms(total_ms),
                }

                stage_log.append(
                    f"Seed={seed}; style='{style_suffix}'; "
                    f"guidance={used.get('guidance')}; steps={used.get('steps')}"
                )

                out_items.append({
                    "ok": ok and (model_rel is not None),
                    "prompt": final_prompt,
                    "seed": seed,
                    "ref_img_rel": ref_img_rel,
                    "model_rel": model_rel,
                    "log": stage_log,
                    "title": f"Вариант #{i} (front/ortho, bg removed + edge clean)",
                    "timings": timings,
                })

            meta = {
                "batch_id": batch_id,
                "created_at": created_at,
                "input_type": "text",
                "prompt": prompt,
                "target_face_count": face_count,
                "use_remesh": use_remesh,
                "cover_rel": ref_cover_rel,
                "items": out_items,
                "total_ms": total_ms_sum,
                "total_text": _fmt_ms(total_ms_sum),
            }
            _save_batch_meta(batch_id, meta)
            _add_to_index({
                "batch_id": batch_id,
                "created_at": created_at,
                "input_type": "text",
                "prompt": prompt,
                "cover_rel": ref_cover_rel,
                "items_count": len(out_items),
                "total_text": _fmt_ms(total_ms_sum),
            })

        return render_template("result.html", items=out_items, batch_id=batch_id)

    return render_template("index.html")

# ---------------------- История и загрузки ----------------------
@app.route("/history")
def history():
    items = _load_index()
    return render_template("history.html", batches=items)

@app.route("/view/<batch_id>")
def view_batch(batch_id: str):
    try:
        meta = _load_batch_meta(batch_id)
    except Exception:
        abort(404)
    return render_template("result.html", items=meta.get("items", []), batch_id=batch_id)

@app.route("/download/<batch_id>/<int:idx>")
def download_item(batch_id: str, idx: int):
    try:
        meta = _load_batch_meta(batch_id)
    except Exception:
        abort(404)
    items = meta.get("items", [])
    if not (0 <= idx < len(items)):
        abort(404)
    item_id = f"{batch_id}_{idx+1}"
    item_dir = RESULTS_DIR / item_id
    if not item_dir.exists():
        abort(404)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(item_dir):
            for fname in files:
                fp = Path(root) / fname
                arcname = fp.relative_to(item_dir.parent)
                zf.write(fp, arcname.as_posix())
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=f"{item_id}.zip", mimetype="application/zip")

@app.route("/download_batch/<batch_id>")
def download_batch(batch_id: str):
    batch_dir = RESULTS_DIR / batch_id
    if not batch_dir.exists():
        abort(404)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        meta_path = batch_dir / "meta.json"
        if meta_path.exists():
            zf.write(meta_path, arcname=f"{batch_id}/meta.json")
        for item_dir in sorted(RESULTS_DIR.glob(f"{batch_id}_*")):
            for root, _, files in os.walk(item_dir):
                for fname in files:
                    fp = Path(root) / fname
                    arcname = f"{batch_id}/{fp.relative_to(item_dir.parent)}"
                    zf.write(fp, arcname)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=f"{batch_id}.zip", mimetype="application/zip")

# ---------------------- Main ----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
