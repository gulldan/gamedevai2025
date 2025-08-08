#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import os

# --- импорт вашего генератора ---
import sys
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

from flask import (  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]
    Flask,
    Response,
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

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "devkey")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"
RESULTS_DIR = STATIC_DIR / "results"
for p in [STATIC_DIR, UPLOADS_DIR, RESULTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

HISTORY_INDEX = RESULTS_DIR / "history_index.json"

# ---------- значения по умолчанию для формы ----------
DEFAULT_TEXT_SEEDS = [52, 1337, 3407]

# Стили: #1 — «прямо по ТЗ», #2 — бумажная/оригами стилизация, #3 — воксель/кирпичики
DEFAULT_DIVERSITY_STYLES = [
    # Style #1 — максимально близко к запросу (база)
    "production-ready low poly, faceted, clean topology, neutral PBR, matte finish, balanced proportions",
    # Style #2 — сильная стилизация: бумага/оригами (другая форма/материал, но один объект)
    "origami papercraft, folded paper, visible creases, paper fibers texture, hand-painted soft gradients, pastel palette, minimal wear",
    # Style #3 — сильная стилизация: воксели/кирпичики (ступенчатый силуэт)
    "voxel / lego-like microblocks, chunky cubes, stepped silhouette, 8-bit 3D aesthetic, flat albedo, primary colors",
]

# Более жёсткий негатив — режем сцену, подставки и лишние атрибуты
DEFAULT_NEGATIVE = (
    "background, black background, dark background, scenery, environment, room, floor, ground, "
    "ground shadow, base, stand, pedestal, platform, diorama, stage, multiple objects, extra parts, "
    "reflections, glossy highlight, glass, text, watermark, logo, caption, people, hands"
)
DEFAULT_GUIDANCE = 3.0
DEFAULT_STEPS = 22
DEFAULT_ERODE = 2
DEFAULT_FEATHER = 1.5
DEFAULT_AUTO_CONTRAST = True


# ---------------------- утилиты ----------------------
def _now_iso() -> str:
    """Возвращает текущее время в формате ISO (UTC).

    Returns:
        str: Время в формате YYYY-MM-DDTHH:MM:SSZ.
    """
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _t() -> float:
    """Текущая высокоточная отметка времени.

    Returns:
        float: Секунды с произвольного начала отсчёта.
    """
    return time.perf_counter()


def _ms_since(t0: float) -> float:
    """Считает миллисекунды, прошедшие с момента t0.

    Args:
        t0 (float): Начальная отметка времени (секунды).

    Returns:
        float: Прошедшее время в миллисекундах.
    """
    return (time.perf_counter() - t0) * 1000.0


def _fmt_ms(ms: float) -> str:
    """Форматирует миллисекунды в секунды с двумя знаками.

    Args:
        ms (float): Время в миллисекундах.

    Returns:
        str: Строка вида "1.23s".
    """
    return f"{ms / 1000.0:.2f}s"


def _safe_tri_count(value: str, default: int = 2000) -> int:
    """Безопасно парсит желаемое число треугольников.

    Ограничивает результат диапазоном [1000, 2000].

    Args:
        value (str): Исходное строковое значение.
        default (int): Значение по умолчанию при ошибке парсинга.

    Returns:
        int: Нормализованное число треугольников.
    """
    try:
        n = int(value)
    except Exception:
        n = default
    return max(1000, min(2000, n))


def _load_index() -> list[dict[str, Any]]:
    """Читает индекс истории из JSON.

    Returns:
        list[dict[str, Any]]: Список записей истории (может быть пустым).
    """
    if HISTORY_INDEX.exists():
        try:
            return json.loads(HISTORY_INDEX.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_index(index: list[dict[str, Any]]) -> None:
    """Сохраняет индекс истории в JSON.

    Args:
        index (list[dict[str, Any]]): Список записей истории.
    """
    HISTORY_INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


def _add_to_index(entry: dict[str, Any]) -> None:
    """Добавляет запись в индекс истории и пересортировывает по дате.

    Args:
        entry (dict[str, Any]): Запись для добавления.
    """
    idx = _load_index()
    idx.append(entry)
    idx.sort(key=lambda e: e.get("created_at", ""), reverse=True)
    _save_index(idx)


def _save_batch_meta(batch_id: str, meta: dict[str, Any]) -> None:
    """Сохраняет метаданные батча в файл meta.json.

    Args:
        batch_id (str): Идентификатор батча.
        meta (dict[str, Any]): Объект метаданных.
    """
    (RESULTS_DIR / batch_id).mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / batch_id / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_batch_meta(batch_id: str) -> dict[str, Any]:
    """Загружает метаданные батча из meta.json.

    Args:
        batch_id (str): Идентификатор батча.

    Returns:
        dict[str, Any]: Метаданные батча.

    Raises:
        FileNotFoundError: Если meta.json отсутствует.
    """
    meta_path = RESULTS_DIR / batch_id / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("meta.json not found")
    return json.loads(meta_path.read_text(encoding="utf-8"))


# --------- удаление фона / чистка краёв ----------
def _try_import_rembg_session() -> Any | None:
    """Создаёт rembg-сессию, если библиотека доступна.

    Returns:
        Any | None: Объект сессии rembg или None при недоступности.
    """
    try:
        from rembg import new_session  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]

        return new_session("u2net")
    except Exception:
        return None


def _remove_bg_bytes(data: bytes, session: Any, erode_px: int) -> bytes:
    """Удаляет фон у изображения в байтах с помощью rembg.

    При недоступности rembg возвращает исходные данные.

    Args:
        data (bytes): Входные данные изображения.
        session (Any): rembg-сессия.
        erode_px (int): Размер эрозии для маттинга.

    Returns:
        bytes: Результат обработки с альфа-каналом.
    """
    try:
        from rembg import remove  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]

        cut = remove(
            data,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=max(1, erode_px * 2),
        )
        return cut
    except Exception:
        try:
            from rembg import remove  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]

            return remove(data)
        except Exception:
            return data


def _clean_alpha_edges_rgba(im_rgba: Any, erode_px: int, feather_px: float, autocontrast: bool) -> Any:
    """Чистит края альфа‑канала RGBA (эрозия/перо/автоконтраст).

    Args:
        im_rgba (Any): Изображение RGBA.
        erode_px (int): Пиксели эрозии.
        feather_px (float): Радиус размытия.
        autocontrast (bool): Применять ли автоконтраст к альфа‑каналу.

    Returns:
        Any: Обновлённое RGBA‑изображение.
    """
    from PIL import Image, ImageFilter, ImageOps

    if im_rgba.mode != "RGBA":
        im_rgba = im_rgba.convert("RGBA")
    r, g, b, a = im_rgba.split()
    if erode_px > 0:
        k = erode_px * 2 + 1
        a = a.filter(ImageFilter.MinFilter(size=k))
    if feather_px > 0:
        a = a.filter(ImageFilter.GaussianBlur(radius=feather_px))
    if autocontrast:
        a = ImageOps.autocontrast(a, cutoff=1)
    return Image.merge("RGBA", (r, g, b, a))


def _crop_to_content_rgba(im: Any) -> tuple[int, int, int, int]:
    """Вычисляет ограничивающую рамку по альфа‑каналу.

    Args:
        im (Any): Изображение (будет приведено к RGBA).

    Returns:
        tuple[int, int, int, int]: Координаты рамки (l, t, r, b).
    """
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    alpha = im.split()[-1]
    bbox = alpha.getbbox()
    return bbox if bbox else (0, 0, im.width, im.height)


def _prep_ref_images(
    raw_in: Path,
    out_cut: Path,
    out_ref512: Path,
    *,
    remove_bg: bool,
    erode_px: int,
    feather_px: float,
    auto_contrast: bool,
) -> tuple[Path, Path, dict[str, Any]]:
    """Готовит референс‑картинку (удаление фона, чистка краёв, центрирование 512x512).

    Args:
        raw_in (Path): Путь к исходному изображению.
        out_cut (Path): Путь для сохраняемого выреза RGBA.
        out_ref512 (Path): Путь к итоговому 512x512 (RGB, белый фон).
        remove_bg (bool): Включить удаление фона rembg.
        erode_px (int): Эррозия при маттинге/чистке краёв.
        feather_px (float): Радиус пера (размытия) краёв.
        auto_contrast (bool): Включать ли автоконтраст для альфа‑канала.

    Returns:
        tuple[Path, Path, dict[str, Any]]: Пути и тайминги операций.
    """
    from PIL import Image

    timings = {"bg_remove_ms": 0.0, "prep_ms": 0.0}

    # RAW -> RGBA
    im = Image.open(raw_in).convert("RGBA")

    # rembg
    t0 = _t()
    if remove_bg:
        session = _try_import_rembg_session()
        if session is not None:
            cut_bytes = _remove_bg_bytes(raw_in.read_bytes(), session, erode_px)
            try:
                im = Image.open(io.BytesIO(cut_bytes)).convert("RGBA")
            except Exception:
                pass
    timings["bg_remove_ms"] = _ms_since(t0)

    # очистка краёв, кадрирование, Resize 512 и центрирование
    t1 = _t()
    im = _clean_alpha_edges_rgba(im, erode_px, feather_px, auto_contrast)
    l, t, r, b = _crop_to_content_rgba(im)
    margin = 10
    l = max(0, l - margin)
    t = max(0, t - margin)
    r = min(im.width, r + margin)
    b = min(im.height, b + margin)
    im = im.crop((l, t, r, b))
    im.thumbnail((512, 512), Image.Resampling.LANCZOS)

    out_cut.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_cut)  # RGBA для UI

    bg = Image.new("RGB", (512, 512), (255, 255, 255))
    bg.paste(im, ((512 - im.width) // 2, (512 - im.height) // 2), im)
    bg.save(out_ref512)
    timings["prep_ms"] = _ms_since(t1)
    return out_cut, out_ref512, timings


# ---- генерация изображения (Playground v2.5) ----
def _gen_image_playground(
    pipe,
    base_prompt: str,
    style_suffix: str,
    seed: int,
    out_png: Path,
    *,
    negative_prompt: str,
    guidance: float,
    steps: int,
) -> dict[str, Any]:
    """Генерирует картинку по тексту через Playground v2.5.

    Args:
        pipe (Any): Инициализированный диффузионный пайплайн.
        base_prompt (str): Базовый промпт.
        style_suffix (str): Суффикс (стилизация) для диверсификации.
        seed (int): Сид генерации.
        out_png (Path): Куда сохранять финальное PNG (RGB).
        negative_prompt (str): Негативные токены.
        guidance (float): Guidance scale.
        steps (int): Кол-во шагов диффузии.

    Returns:
        dict[str, Any]: Параметры и тайминги генерации.
    """
    # строгая фронтальная композиция
    prompt = (
        "(low poly, game asset, best quality), "
        f"{base_prompt}, "
        f"{style_suffix}, "
        f"triangle count under 2000, front view, orthographic, 3d model, 3d render, flat shading, UV unwrapped, ((white background)), ((isolated on white)), realtime optimized"
    )

    gen = None
    try:
        import torch

        gen = torch.Generator("cpu").manual_seed(seed)
    except Exception:
        pass

    t0 = _t()
    try:
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=float(guidance),
            num_inference_steps=int(steps),
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
        "guidance": float(guidance),
        "steps": int(steps),
        "negative": negative_prompt,
        "image_gen_ms": image_gen_ms,
    }


# ---------------------- ROUTES ----------------------
@app.route("/", methods=["GET", "POST"])
def index() -> Response | str:
    """Главная страница: форма и запуск генерации.

    Returns:
        Response | str: HTML страницы или редирект/ответ скачивания.
    """
    if request.method == "POST":
        # базовые настройки
        face_count = _safe_tri_count(request.form.get("target_face_count", "2000"))
        use_remesh = request.form.get("use_remesh", "off") == "on"

        # расширенные: стили/сид/негатив/гайды/шаги/фон
        style1 = (request.form.get("style1") or DEFAULT_DIVERSITY_STYLES[0]).strip()
        style2 = (request.form.get("style2") or DEFAULT_DIVERSITY_STYLES[1]).strip()
        style3 = (request.form.get("style3") or DEFAULT_DIVERSITY_STYLES[2]).strip()
        styles = [style1, style2, style3]

        try:
            seed1 = int(request.form.get("seed1", DEFAULT_TEXT_SEEDS[0]))
        except:
            seed1 = DEFAULT_TEXT_SEEDS[0]
        try:
            seed2 = int(request.form.get("seed2", DEFAULT_TEXT_SEEDS[1]))
        except:
            seed2 = DEFAULT_TEXT_SEEDS[1]
        try:
            seed3 = int(request.form.get("seed3", DEFAULT_TEXT_SEEDS[2]))
        except:
            seed3 = DEFAULT_TEXT_SEEDS[2]
        seeds = [seed1, seed2, seed3]

        negative = (request.form.get("negative_prompt") or DEFAULT_NEGATIVE).strip()
        try:
            guidance = float(request.form.get("guidance_scale", str(DEFAULT_GUIDANCE)))
        except:
            guidance = DEFAULT_GUIDANCE
        try:
            steps = int(request.form.get("steps", str(DEFAULT_STEPS)))
        except:
            steps = DEFAULT_STEPS

        remove_bg = request.form.get("remove_bg", "on") == "on"
        clean_auto = request.form.get("auto_contrast", "on") == "on"
        try:
            erode_px = int(request.form.get("erode_px", str(DEFAULT_ERODE)))
        except:
            erode_px = DEFAULT_ERODE
        try:
            feather_px = float(request.form.get("feather_px", str(DEFAULT_FEATHER)))
        except:
            feather_px = DEFAULT_FEATHER

        prompt = (request.form.get("prompt") or "").strip()
        file = request.files.get("image")

        if not (file and file.filename) and not prompt:
            flash("Нужно указать либо картинку, либо текстовый промпт.")
            return redirect(url_for("index"))

        # собираем «params» для истории
        params_used = {
            "styles": styles,
            "seeds": seeds,
            "negative_prompt": negative,
            "guidance": guidance,
            "steps": steps,
            "remove_bg": remove_bg,
            "edge_clean": {
                "erode_px": erode_px,
                "feather_px": feather_px,
                "auto_contrast": clean_auto,
            },
            "target_face_count": face_count,
            "use_remesh": use_remesh,
        }

        batch_id = f"batch_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        created_at = _now_iso()
        out_items: list[dict[str, Any]] = []
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
            cut_ref = item_dir / f"{item_id}_cut.png"
            ref_512 = item_dir / f"{item_id}_ref.png"
            glb_path = item_dir / f"{item_id}.glb"
            obj_path = item_dir / f"{item_id}.obj"

            from PIL import Image

            Image.open(src_abs).convert("RGB").save(raw_ref)

            # подготовка рефера (удаление фона+чистка краёв)
            _, _, prep_times = _prep_ref_images(
                raw_ref,
                cut_ref,
                ref_512,
                remove_bg=remove_bg,
                erode_px=erode_px,
                feather_px=feather_px,
                auto_contrast=clean_auto,
            )
            prep_ms = prep_times["bg_remove_ms"] + prep_times["prep_ms"]

            # генерация 3D
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

            model_rel = (
                f"results/{item_id}/{glb_path.name}"
                if glb_path.exists()
                else (f"results/{item_id}/{obj_path.name}" if obj_path.exists() else None)
            )

            ref_img_rel = f"results/{item_id}/{(cut_ref if cut_ref.exists() else raw_ref).name}"

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

            out_items.append(
                {
                    "ok": ok and (model_rel is not None),
                    "prompt": prompt,
                    "seed": None,
                    "ref_img_rel": ref_img_rel,
                    "model_rel": model_rel,
                    "log": stage_log,
                    "title": "Результат по изображению (bg removed + edge clean)",
                    "timings": timings,
                }
            )

            total_ms_sum = timings["total_ms"]
            meta = {
                "batch_id": batch_id,
                "created_at": created_at,
                "input_type": input_type,
                "prompt": prompt,
                "items": out_items,
                "params": params_used,
                "total_ms": total_ms_sum,
                "total_text": _fmt_ms(total_ms_sum),
                "cover_rel": ref_img_rel,
            }
            _save_batch_meta(batch_id, meta)
            _add_to_index(
                {
                    "batch_id": batch_id,
                    "created_at": created_at,
                    "input_type": input_type,
                    "prompt": prompt,
                    "cover_rel": ref_img_rel,
                    "items_count": len(out_items),
                    "total_text": _fmt_ms(total_ms_sum),
                }
            )

        # ====== ТЕКСТ -> 3 варианта ======
        else:
            total_ms_sum = 0.0
            ref_cover_rel = None
            for i, (seed, style_suffix) in enumerate(zip(seeds, styles, strict=False), start=1):
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

                # 1) Генерация изображения
                img_ms = 0.0
                try:
                    pipe = load_image_gen_pipeline()
                    if pipe is None:
                        stage_log.append("Image pipeline not available")
                        ok = False
                    else:
                        res = _gen_image_playground(
                            pipe,
                            prompt,
                            style_suffix,
                            seed,
                            raw_ref,
                            negative_prompt=negative,
                            guidance=guidance,
                            steps=steps,
                        )
                        final_prompt = res["final_prompt"]
                        img_ms = res["image_gen_ms"]
                        used = {
                            "guidance": res["guidance"],
                            "steps": res["steps"],
                            "negative": res["negative"],
                        }
                        try:
                            del pipe
                        except Exception:
                            pass
                        clear_memory()
                except Exception as e:
                    stage_log.append(f"Error@image_gen: {e}")
                    ok = False

                if i == 1:
                    ref_cover_rel = f"results/{item_id}/{raw_ref.name}"

                # 2) Удаление фона / чистка краёв / сборка 512x512
                _, _, prep_times = _prep_ref_images(
                    raw_ref,
                    cut_ref,
                    ref_512,
                    remove_bg=remove_bg,
                    erode_px=erode_px,
                    feather_px=feather_px,
                    auto_contrast=clean_auto,
                )

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

                model_rel = (
                    f"results/{item_id}/{glb_path.name}"
                    if glb_path.exists()
                    else (f"results/{item_id}/{obj_path.name}" if obj_path.exists() else None)
                )

                ref_img_rel = f"results/{item_id}/{(cut_ref if cut_ref.exists() else raw_ref).name}"

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
                    f"Seed={seed}; style='{style_suffix}'; guidance={used.get('guidance')}; steps={used.get('steps')}"
                )

                out_items.append(
                    {
                        "ok": ok and (model_rel is not None),
                        "prompt": final_prompt,
                        "seed": seed,
                        "ref_img_rel": ref_img_rel,
                        "model_rel": model_rel,
                        "log": stage_log,
                        "title": f"Вариант #{i} (front/ortho, bg removed + edge clean)",
                        "timings": timings,
                    }
                )

            meta = {
                "batch_id": batch_id,
                "created_at": created_at,
                "input_type": "text",
                "prompt": prompt,
                "items": out_items,
                "params": params_used,
                "total_ms": total_ms_sum,
                "total_text": _fmt_ms(total_ms_sum),
                "cover_rel": ref_cover_rel,
            }
            _save_batch_meta(batch_id, meta)
            _add_to_index(
                {
                    "batch_id": batch_id,
                    "created_at": created_at,
                    "input_type": "text",
                    "prompt": prompt,
                    "cover_rel": ref_cover_rel,
                    "items_count": len(out_items),
                    "total_text": _fmt_ms(total_ms_sum),
                }
            )

        return render_template("result.html", items=out_items, batch_id=batch_id)

    # GET — страница с формой и примерами
    return render_template(
        "index.html",
        defaults={
            "styles": DEFAULT_DIVERSITY_STYLES,
            "seeds": DEFAULT_TEXT_SEEDS,
            "negative": DEFAULT_NEGATIVE,
            "guidance": DEFAULT_GUIDANCE,
            "steps": DEFAULT_STEPS,
            "erode": DEFAULT_ERODE,
            "feather": DEFAULT_FEATHER,
            "auto_contrast": DEFAULT_AUTO_CONTRAST,
        },
    )


# ---------------------- История и загрузки ----------------------
@app.route("/history")
def history() -> str:
    """Страница истории с батчами результатов.

    Returns:
        str: HTML страницы истории.
    """
    items = _load_index()
    return render_template("history.html", batches=items)


@app.route("/view/<batch_id>")
def view_batch(batch_id: str) -> str | Response:
    """Просмотр результатов конкретного батча.

    Args:
        batch_id (str): Идентификатор батча.

    Returns:
        str | Response: HTML страницы результатов.
    """
    try:
        meta = _load_batch_meta(batch_id)
    except Exception:
        abort(404)
    return render_template("result.html", items=meta.get("items", []), batch_id=batch_id)


@app.route("/download/<batch_id>/<int:idx>")
def download_item(batch_id: str, idx: int) -> Response:
    """Скачивание одного результата из батча в ZIP.

    Args:
        batch_id (str): Идентификатор батча.
        idx (int): Индекс результата (1..N).

    Returns:
        Response: Отдаёт ZIP с файлами результата.
    """
    try:
        meta = _load_batch_meta(batch_id)
    except Exception:
        abort(404)
    items = meta.get("items", [])
    if not (0 <= idx < len(items)):
        abort(404)
    item_id = f"{batch_id}_{idx + 1}"
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
    return send_file(
        buf,
        as_attachment=True,
        download_name=f"{item_id}.zip",
        mimetype="application/zip",
    )


@app.route("/download_batch/<batch_id>")
def download_batch(batch_id: str) -> Response:
    """Скачивание всего батча результатов (включая meta.json) в ZIP.

    Args:
        batch_id (str): Идентификатор батча.

    Returns:
        Response: Отдаёт ZIP архив батча.
    """
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
    return send_file(
        buf,
        as_attachment=True,
        download_name=f"{batch_id}.zip",
        mimetype="application/zip",
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
