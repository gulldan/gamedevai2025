#!/usr/bin/env python3

from __future__ import annotations

import gc
import logging
import os
import platform
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from PIL import Image

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pipeline")

# ---------------------------------------------------------------------------------
# Пути и окружение: используем абсолютные пути относительно корня репозитория
# ---------------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parents[1]
HY3D_ROOT = ROOT_DIR / "Hunyuan3D-2.1"

# Делаем доступными модули из Hunyuan3D-2.1 (hy3dshape, hy3dpaint, torchvision_fix)
sys.path.insert(0, str(HY3D_ROOT))
# Для надёжности добавляем вложенные корни пакетов
sys.path.insert(0, str(HY3D_ROOT / "hy3dshape"))
sys.path.insert(0, str(HY3D_ROOT / "hy3dpaint"))


def _ensure_torch_lib_on_path() -> None:
    """Добавляет путь к бинарникам PyTorch в переменную окружения.

    Добавляет директорию `torch/lib` в `LD_LIBRARY_PATH` (Linux) или
    `DYLD_LIBRARY_PATH` (macOS), чтобы нативные расширения могли
    корректно подгружать зависимости.

    Returns:
        None
    """
    try:
        torch_lib_dir = Path(torch.__file__).parent / "lib"
        if not torch_lib_dir.exists():
            return
        if platform.system() == "Darwin":
            key = "DYLD_LIBRARY_PATH"
        elif os.name == "posix":
            key = "LD_LIBRARY_PATH"
        else:
            key = None
        if key:
            current = os.environ.get(key, "")
            paths = [p for p in current.split(":") if p]
            if str(torch_lib_dir) not in paths:
                os.environ[key] = f"{current}:{torch_lib_dir}" if current else str(torch_lib_dir)
    except Exception:
        pass


def _try_import_mesh_uv_wrap() -> Callable[[Any], Any] | None:
    """Импортирует функцию `mesh_uv_wrap`, если модуль доступен.

    Returns:
        Callable[[Any], Any] | None
    """
    try:
        from importlib import import_module

        module = import_module("hy3dpaint.utils.uvwrap_utils")
        return getattr(module, "mesh_uv_wrap", None)
    except Exception:
        return None


def _apply_torchvision_fix_if_available() -> None:
    """Применяет фиксы совместимости из `torchvision_fix`, если доступно.

    Файл ожидается в корне `Hunyuan3D-2.1`.
    """
    try:
        from torchvision_fix import apply_fix  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]

        apply_fix()
    except Exception:
        # Не превращаем это в фатальную ошибку
        logger.exception("torchvision_fix not applied")


_ensure_torch_lib_on_path()

# Настройка устройства
if torch.cuda.is_available():
    device = torch.device("cuda")
    # Лучшее управление памятью на CUDA
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    try:
        logger.info(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except Exception:
        logger.debug("Failed to extend torch lib path", exc_info=True)
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Глобальные переменные для pipeline
shape_pipeline = None
paint_pipeline = None
image_gen_pipeline = None


def clear_memory() -> None:
    """Очищает память GPU и CPU.

    Returns:
        None
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    logger.info("🧹 Память очищена")


# -----------------------------------------------------------------------------
# Утилиты для упрощения меша и повторной UV-развёртки
# -----------------------------------------------------------------------------


def simplify_mesh_and_rewrap(input_glb: str, output_glb: str, target_count: int) -> str:
    """Упрощает меш до заданного числа треугольников и заново разворачивает UV.

    Внутри используется `trimesh` для упрощения и `mesh_uv_wrap` из
    `hy3dpaint.utils.uvwrap_utils` для повторной развёртки.  Если число граней
    исходного меша уже меньше `target_count`, функция просто копирует файл.

    Args:
        input_glb: путь к исходному GLB (или OBJ) файлу.
        output_glb: путь, по которому будет сохранён упрощённый GLB.
        target_count: желаемое количество треугольников.

    Returns:
        Путь к упрощённому GLB.
    """
    import trimesh  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]

    # Пытаемся динамически импортировать функцию развёртки UV
    mesh_uv_wrap = _try_import_mesh_uv_wrap()

    # Загружаем меш (поддерживаются как GLB, так и OBJ)
    mesh: Any = trimesh.load(input_glb)

    # Если файл содержит несколько частей и загружен как сцена,
    # преобразуем его в единый Trimesh.  Иначе у сцены отсутствует
    # атрибут faces, что вызывает ошибку.
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    face_num = len(mesh.faces)
    if face_num > target_count:
        # Упрощаем меш алгоритмом квадратичной декомпозиции
        mesh = mesh.simplify_quadratic_decimation(target_count)

    # Повторно разворачиваем UV, чтобы текстуры корректно легли на изменённую топологию
    if callable(mesh_uv_wrap):
        try:
            mesh = mesh_uv_wrap(mesh)  # type: ignore[misc]
        except Exception:
            logger.exception("⚠️ Ошибка при развёртке UV. Меш сохранён без обновления UV.")

    # Сохраняем упрощённый меш
    mesh.export(output_glb)
    return output_glb


def load_shape_pipeline() -> Any | None:
    """Загружает пайплайн генерации формы.

    Returns:
        Any | None
    """
    global shape_pipeline
    if shape_pipeline is None:
        logger.info("📦 Загружаем shape pipeline...")
        clear_memory()
        _apply_torchvision_fix_if_available()
        try:
            # Импорт выполняем лениво, чтобы избежать ошибок импортера у линтера
            from hy3dshape.pipelines import (  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]
                Hunyuan3DDiTFlowMatchingPipeline,
            )

            shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1")
            # Переносим на целевое устройство, если поддерживается
            try:
                shape_pipe = shape_pipe.to(device)
            except Exception:
                pass
            shape_pipeline = shape_pipe
        except Exception:
            logger.exception("❌ Ошибка загрузки shape pipeline")
            return None
        logger.info("✅ Shape pipeline загружен")
    return shape_pipeline


def load_paint_pipeline() -> Any | None:
    """Загружает пайплайн текстурирования.

    Returns:
        Any | None
    """
    global paint_pipeline
    if paint_pipeline is None:
        logger.info("📦 Загружаем paint pipeline...")
        clear_memory()
        try:
            _apply_torchvision_fix_if_available()
            # Импорты выполняем лениво
            from hy3dpaint.textureGenPipeline import (  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]
                Hunyuan3DPaintConfig,
                Hunyuan3DPaintPipeline,
            )

            max_num_view = 6
            resolution = 512
            conf = Hunyuan3DPaintConfig(max_num_view, resolution)

            # Настраиваем пути конфигураций/чекпоинтов, если они существуют
            realesrgan_ckpt = HY3D_ROOT / "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
            if realesrgan_ckpt.exists():
                conf.realesrgan_ckpt_path = str(realesrgan_ckpt)
            conf.multiview_cfg_path = str(HY3D_ROOT / "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml")
            conf.custom_pipeline = str(HY3D_ROOT / "hy3dpaint/hunyuanpaintpbr")

            paint_pipeline_obj = Hunyuan3DPaintPipeline(conf)
            paint_pipeline = paint_pipeline_obj
            logger.info("✅ Paint pipeline загружен")
        except Exception:
            logger.exception("❌ Ошибка загрузки paint pipeline")
            return None
    return paint_pipeline


def load_image_gen_pipeline() -> Any | None:
    """Загружает пайплайн генерации изображений.

    Returns:
        Any | None
    """
    global image_gen_pipeline
    if image_gen_pipeline is None:
        logger.info("📦 Загружаем image generation pipeline...")
        clear_memory()
        try:
            # Позволяем указать альтернативную модель для генерации изображений
            model_id = os.environ.get("IMAGE_GEN_MODEL_ID", "playgroundai/playground-v2.5-1024px-aesthetic")
            # Импортируем внутри функции, чтобы избежать ошибок линтера
            from diffusers import DiffusionPipeline  # type: ignore[import-not-found]

            # Подбираем dtype и variant в зависимости от устройства
            if device.type == "cuda":
                dtype = torch.float16
                variant = "fp16"
            elif device.type == "mps":
                # На MPS обычно работает float16/bfloat16. Выберем bfloat16 при наличии
                dtype = torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float16
                variant = None
            else:
                dtype = torch.float32
                variant = None

            try:
                if variant is not None:
                    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, variant=variant)
                else:
                    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
            except Exception as diff_err:
                if "flux" in model_id.lower():
                    from diffusers import FluxPipeline  # type: ignore[import-not-found]

                    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)
                else:
                    raise diff_err

            try:
                pipe = pipe.to(device)
            except Exception:
                pass

            # Снижаем потребление VRAM, если поддерживается
            for method_name in (
                "enable_model_cpu_offload",
                "enable_attention_slicing",
                "enable_xformers_memory_efficient_attention",
            ):
                method = getattr(pipe, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception:
                        pass

            image_gen_pipeline = pipe
            logger.info(f"✅ Image generation pipeline загружен: {model_id}")
        except Exception:
            logger.exception("❌ Ошибка загрузки image generation pipeline")
            return None
    return image_gen_pipeline


def unload_pipelines() -> None:
    """Выгружает все пайплайны и очищает память."""
    global shape_pipeline, paint_pipeline, image_gen_pipeline

    if shape_pipeline is not None:
        del shape_pipeline
        shape_pipeline = None
        logger.info("🗑️ Shape pipeline выгружен")

    if paint_pipeline is not None:
        del paint_pipeline
        paint_pipeline = None
        logger.info("🗑️ Paint pipeline выгружен")

    if image_gen_pipeline is not None:
        del image_gen_pipeline
        image_gen_pipeline = None
        logger.info("🗑️ Image generation pipeline выгружен")

    clear_memory()


def resize_and_pad(
    image: Image.Image,
    target_size: tuple[int, int],
    fill_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    padded_image = Image.new("RGB", target_size, fill_color)
    image.thumbnail(target_size, Image.Resampling.LANCZOS)
    paste_position = (
        (target_size[0] - image.width) // 2,
        (target_size[1] - image.height) // 2,
    )
    padded_image.paste(image, paste_position)
    return padded_image


def generate_mesh_from_image(image_path: str, output_path: str) -> bool:
    """Генерирует меш из изображения."""
    if not os.path.exists(image_path):
        logger.error(f"❌ Файл изображения не найден: {image_path}")
        return False

    logger.info("🔄 Начинаем генерацию меша из изображения...")
    clear_memory()

    try:
        # Загружаем pipeline
        pipeline = load_shape_pipeline()
        if pipeline is None:
            return False

        # Обрабатываем изображение
        image = Image.open(image_path).convert("RGB")
        processed_image = resize_and_pad(image, (512, 512))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        ref_image_path = os.path.splitext(output_path)[0] + ".png"
        processed_image.save(ref_image_path)
        logger.info(f"📸 Референсное изображение сохранено: {ref_image_path}")

        # Генерируем меш
        logger.info("🎯 Генерируем меш...")
        mesh_untextured: Any = pipeline(image=processed_image, show_progress_bar=False)[0]
        mesh_untextured.export(output_path)
        logger.info(f"✅ Меш сохранен: {output_path}")

        return True

    except torch.cuda.OutOfMemoryError:
        logger.exception("❌ Недостаточно VRAM для генерации меша")
        logger.info("💡 Попробуйте освободить память или уменьшить размер изображения")
        return False
    except Exception:
        logger.exception("❌ Ошибка генерации меша")
        return False
    finally:
        clear_memory()


def generate_textured_mesh_from_image(
    image_path: str,
    output_path: str,
    *,
    use_remesh: bool = True,
    target_face_count: int | None = 2000,
) -> bool:
    """Генерирует текстурированный меш из изображения с поэтапной обработкой.

    Args:
        image_path: путь к исходному изображению.
        output_path: путь к файлу .obj/.glb, в который будет сохранён результат.
        use_remesh: передать `False`, чтобы отключить встроенное упрощение/ремешинг
            в `Hunyuan3DPaintPipeline` и использовать исходный (или заранее
            упрощённый) меш.
        target_face_count: количество треугольников, до которого нужно
            упростить меш перед текстурированием.  По умолчанию 2000.
            Если указать `None`, упрощение не выполняется и будет использован
            встроенный ремеш пайплайна (если `use_remesh=True`).

    Returns:
        True при успешной генерации, иначе False.
    """
    if not os.path.exists(image_path):
        logger.error(f"❌ Файл изображения не найден: {image_path}")
        return False

    logger.info("🔄 Начинаем генерацию текстурированного меша из изображения...")
    clear_memory()

    try:
        # Этап 1: Генерируем меш без текстуры
        logger.info("🎯 Этап 1: Генерируем меш без текстуры...")
        shape_pipe = load_shape_pipeline()
        if shape_pipe is None:
            return False

        image = Image.open(image_path).convert("RGB")
        processed_image = resize_and_pad(image, (512, 512))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        base_path, ext = os.path.splitext(output_path)
        ref_image_path = base_path + ".png"
        processed_image.save(ref_image_path)
        logger.info(f"📸 Референсное изображение сохранено: {ref_image_path}")

        mesh_untextured: Any = shape_pipe(image=processed_image, show_progress_bar=False)[0]
        untextured_path = base_path + "_untextured.glb"
        mesh_untextured.export(untextured_path)
        logger.info(f"✅ Меш без текстуры сохранен: {untextured_path}")

        # Очищаем память после генерации меша
        clear_memory()

        # Этап 2: Генерируем текстуру
        logger.info("🎨 Этап 2: Генерируем текстуру...")
        paint_pipe = load_paint_pipeline()
        if paint_pipe is None:
            logger.error("❌ Paint pipeline недоступен")
            return False

        # Если требуется предварительное упрощение меша — делаем это до вызова пайплайна
        mesh_for_paint = untextured_path
        if target_face_count is not None:
            simplified_path = base_path + "_preprocessed.glb"
            try:
                simplified_path = simplify_mesh_and_rewrap(untextured_path, simplified_path, target_face_count)
                mesh_for_paint = simplified_path
                # при пользовательском упрощении не нужен ремешинг внутри пайплайна
                use_remesh = False
            except Exception as e:
                logger.warning(f"⚠️ Не удалось упростить и переобернуть меш: {e}. Продолжаем с исходным.")
        # Если пользователь отключил ремешинг, но не указал упрощение,
        # делаем повторную UV-развёртку исходного меша.  Это полезно, если
        # оригинальная UV-карта не совпадает с текстурой.
        elif not use_remesh:
            try:
                import trimesh  # type: ignore[import-not-found]

                mesh_uv_wrap = _try_import_mesh_uv_wrap()

                mesh_tmp: Any = trimesh.load(untextured_path)
                if callable(mesh_uv_wrap):
                    mesh_tmp = mesh_uv_wrap(mesh_tmp)  # type: ignore[misc]
                mesh_tmp.export(untextured_path)
                logger.info("🔁 Выполнена повторная UV-развёртка исходного меша")
            except Exception as e:
                logger.warning(f"⚠️ Ошибка при повторной UV-развёртке: {e}")

        # Если пользователь указал .glb как выход, используем временный .obj для рисования
        desired_ext = ext.lower()
        obj_output_path = base_path + ".obj"

        # Запускаем пайплайн рисования. Он сохранит OBJ и при наличии Blender — GLB
        output_mesh_path = paint_pipe(
            mesh_path=mesh_for_paint,
            image_path=ref_image_path,
            output_mesh_path=obj_output_path,
            use_remesh=use_remesh,
        )
        # Проверяем, что OBJ действительно создан
        if not os.path.exists(obj_output_path):
            # Некоторые реализации могут возвращать путь к результату
            if isinstance(output_mesh_path, str) and os.path.exists(output_mesh_path):
                obj_output_path = output_mesh_path
            else:
                logger.error(f"❌ Paint pipeline не создал OBJ: {obj_output_path}. Возврат без результата.")
                return False

        # Исправляем пути к текстурам в OBJ
        fix_texture_paths(obj_output_path)

        # Если пользователь просил .glb, пробуем конвертировать OBJ в GLB
        final_glb_path = base_path + ".glb"
        if desired_ext == ".glb":
            """
            Попытаться конвертировать OBJ в GLB через Blender.  В Blender 4.0 API
            изменились операторы импорта, поэтому сначала пробуем
            `bpy.ops.wm.obj_import`, а если его нет — используем старый
            `bpy.ops.import_scene.obj`.  При отсутствии Blender или ошибках
            fallback — задействуем trimesh.  Если и trimesh не удаётся,
            копируем OBJ, чтобы пользователь получил хотя бы OBJ.
            """
            try:
                import bpy  # type: ignore[import-not-found]

                # Очистить сцену
                bpy.ops.object.select_all(action="SELECT")
                bpy.ops.object.delete(use_global=False)

                # Импортировать OBJ.  В Blender 4.x используется
                # bpy.ops.wm.obj_import, в более старых версиях —
                # bpy.ops.import_scene.obj.  Проверяем оба варианта.
                if hasattr(bpy.ops, "wm") and hasattr(bpy.ops.wm, "obj_import"):
                    bpy.ops.wm.obj_import(filepath=obj_output_path)
                else:
                    bpy.ops.import_scene.obj(filepath=obj_output_path)

                # Экспортировать GLTF/GLB.  bpy.ops.export_scene.gltf
                # автоматически определяет формат по расширению пути.
                bpy.ops.export_scene.gltf(filepath=final_glb_path, use_active_scene=True)
                logger.info(f"✅ OBJ конвертирован в GLB через Blender: {final_glb_path}")
            except Exception:
                # Если Blender недоступен или произошла любая ошибка, пытаемся
                # конвертировать через trimesh
                try:
                    import trimesh  # type: ignore[import-not-found]

                    mesh: Any = trimesh.load(obj_output_path)
                    mesh.export(final_glb_path)
                    logger.info(f"✅ OBJ конвертирован в GLB через trimesh: {final_glb_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Не удалось конвертировать OBJ в GLB: {e}. Оставляем OBJ в качестве результата.")
                    import shutil

                    shutil.copy(obj_output_path, final_glb_path)

        # Создаем OBJ версию с текстурами для пользователя
        obj_path = obj_output_path
        logger.info(f"✅ Текстурированный OBJ сохранен: {obj_path}")
        if desired_ext == ".glb":
            logger.info(f"✅ Текстурированный GLB сохранен: {final_glb_path}")
        return True

    except torch.cuda.OutOfMemoryError:
        logger.exception("❌ Недостаточно VRAM для генерации")
        logger.info("💡 Попробуйте освободить память или уменьшить размер изображения")
        return False
    except Exception:
        logger.exception("❌ Ошибка генерации")
        return False
    finally:
        clear_memory()


def generate_mesh_from_text(prompt: str, output_path: str) -> bool:
    """Генерирует меш из текста."""
    logger.info("🔄 Начинаем генерацию меша из текста...")
    clear_memory()

    try:
        # Загружаем image generation pipeline
        img_pipe = load_image_gen_pipeline()
        if img_pipe is None:
            logger.error("❌ Image generation pipeline недоступен")
            return False

        # Генерируем изображение
        logger.info("🎨 Генерируем изображение из текста...")
        image = img_pipe(prompt=prompt).images[0]
        processed_image = resize_and_pad(image, (512, 512))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        ref_image_path = os.path.splitext(output_path)[0] + ".png"
        processed_image.save(ref_image_path)
        logger.info(f"📸 Сгенерированное изображение сохранено: {ref_image_path}")

        # Очищаем память после генерации изображения
        clear_memory()

        # Генерируем меш
        logger.info("🎯 Генерируем меш из изображения...")
        shape_pipe = load_shape_pipeline()
        if shape_pipe is None:
            return False

        mesh_untextured: Any = shape_pipe(image=processed_image, show_progress_bar=False)[0]
        mesh_untextured.export(output_path)
        logger.info(f"✅ Меш из текста сохранен: {output_path}")

        return True

    except torch.cuda.OutOfMemoryError:
        logger.exception("❌ Недостаточно VRAM для генерации")
        return False
    except Exception:
        logger.exception("❌ Ошибка генерации")
        return False
    finally:
        clear_memory()


def generate_textured_mesh_from_text(
    prompt: str,
    output_path: str,
    *,
    use_remesh: bool = True,
    target_face_count: int | None = 2000,
) -> bool:
    """Генерирует текстурированный меш из текста с поэтапной обработкой.

    Args:
        prompt: текстовое описание объекта.
        output_path: путь к файлу .obj/.glb для сохранения результата.
        use_remesh: передать `False`, чтобы отключить встроенный ремешинг внутри
            пайплайна рисования.
        target_face_count: количество треугольников, до которого нужно
            упростить меш перед текстурированием.  По умолчанию 2000.
            Если указать `None`, упрощение не выполняется и будет использован
            встроенный ремеш пайплайна (если `use_remesh=True`).

    Returns:
        True при успешной генерации, иначе False.
    """
    logger.info("🔄 Начинаем генерацию текстурированного меша из текста...")
    clear_memory()

    try:
        # Загружаем image generation pipeline
        img_pipe = load_image_gen_pipeline()
        if img_pipe is None:
            logger.error("❌ Image generation pipeline недоступен")
            return False

        # Генерируем изображение
        logger.info("🎨 Генерируем изображение из текста...")
        image = img_pipe(prompt=prompt).images[0]
        processed_image = resize_and_pad(image, (512, 512))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        base_path, ext = os.path.splitext(output_path)
        ref_image_path = base_path + ".png"
        processed_image.save(ref_image_path)
        logger.info(f"📸 Сгенерированное изображение сохранено: {ref_image_path}")

        # Очищаем память после генерации изображения
        # Выгружаем image_gen_pipeline, чтобы освободить VRAM перед загрузкой shape-пайплайна
        global image_gen_pipeline
        if image_gen_pipeline is not None:
            del image_gen_pipeline
            image_gen_pipeline = None
        clear_memory()

        # Генерируем меш
        logger.info("🎯 Генерируем меш из изображения...")
        shape_pipe = load_shape_pipeline()
        if shape_pipe is None:
            return False

        mesh_untextured: Any = shape_pipe(image=processed_image, show_progress_bar=False)[0]
        untextured_path = base_path + "_untextured.glb"
        mesh_untextured.export(untextured_path)
        logger.info(f"✅ Меш без текстуры сохранен: {untextured_path}")

        # Очищаем память после генерации меша
        clear_memory()

        # Генерируем текстуру
        logger.info("🎨 Генерируем текстуру...")
        paint_pipe = load_paint_pipeline()
        if paint_pipe is None:
            logger.error("❌ Paint pipeline недоступен")
            return False

        # Если требуется предварительное упрощение и переобёртка, выполняем её здесь
        mesh_for_paint = untextured_path
        if target_face_count is not None:
            simplified_path = base_path + "_preprocessed.glb"
            try:
                simplified_path = simplify_mesh_and_rewrap(untextured_path, simplified_path, target_face_count)
                mesh_for_paint = simplified_path
                use_remesh = False
            except Exception as e:
                logger.warning(f"⚠️ Не удалось упростить и переобернуть меш: {e}. Продолжаем с исходным.")
        # если пользователь отключает ремешинг, но не указал упрощения,
        # выполняем повторную UV-развёртку исходного меша
        elif not use_remesh:
            try:
                import trimesh  # type: ignore[import-not-found]

                mesh_uv_wrap = _try_import_mesh_uv_wrap()

                mesh_tmp: Any = trimesh.load(untextured_path)
                if callable(mesh_uv_wrap):
                    mesh_tmp = mesh_uv_wrap(mesh_tmp)  # type: ignore[misc]
                mesh_tmp.export(untextured_path)
                logger.info("🔁 Выполнена повторная UV-развёртка исходного меша")
            except Exception as e:
                logger.warning(f"⚠️ Ошибка при повторной UV-развёртке: {e}")

        desired_ext = ext.lower()
        obj_output_path = base_path + ".obj"
        output_mesh_path = paint_pipe(
            mesh_path=mesh_for_paint,
            image_path=ref_image_path,
            output_mesh_path=obj_output_path,
            use_remesh=use_remesh,
        )
        # Проверяем, что OBJ действительно создан
        if not os.path.exists(obj_output_path):
            if isinstance(output_mesh_path, str) and os.path.exists(output_mesh_path):
                obj_output_path = output_mesh_path
            else:
                logger.error(f"❌ Paint pipeline не создал OBJ: {obj_output_path}. Возврат без результата.")
                return False
        # Исправляем пути к текстурам
        fix_texture_paths(obj_output_path)

        final_glb_path = base_path + ".glb"
        if desired_ext == ".glb":
            """
            Конвертирует OBJ в GLB.  Используется Blender (если доступен);
            начиная с Blender 4.0 — оператор `bpy.ops.wm.obj_import`, в
            старых версиях — `bpy.ops.import_scene.obj`.  В случае любых
            ошибок или отсутствия Blender переходим на `trimesh`, а если и он
            не помогает — копируем OBJ как GLB.
            """
            try:
                import bpy  # type: ignore[import-not-found]

                bpy.ops.object.select_all(action="SELECT")
                bpy.ops.object.delete(use_global=False)

                # Импортируем OBJ для разных версий Blender
                if hasattr(bpy.ops, "wm") and hasattr(bpy.ops.wm, "obj_import"):
                    bpy.ops.wm.obj_import(filepath=obj_output_path)
                else:
                    bpy.ops.import_scene.obj(filepath=obj_output_path)

                bpy.ops.export_scene.gltf(filepath=final_glb_path, use_active_scene=True)
                logger.info(f"✅ OBJ конвертирован в GLB через Blender: {final_glb_path}")
            except Exception:
                try:
                    import trimesh  # type: ignore[import-not-found]

                    mesh: Any = trimesh.load(obj_output_path)
                    mesh.export(final_glb_path)
                    logger.info(f"✅ OBJ конвертирован в GLB через trimesh: {final_glb_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Не удалось конвертировать OBJ в GLB: {e}. Оставляем OBJ в качестве результата.")
                    import shutil

                    shutil.copy(obj_output_path, final_glb_path)

        obj_path = obj_output_path
        logger.info(f"✅ Текстурированный OBJ сохранен: {obj_path}")
        if desired_ext == ".glb":
            logger.info(f"✅ Текстурированный GLB сохранен: {final_glb_path}")
        return True

    except torch.cuda.OutOfMemoryError:
        logger.exception("❌ Недостаточно VRAM для генерации")
        return False
    except Exception:
        logger.exception("❌ Ошибка генерации")
        return False
    finally:
        clear_memory()


def fix_texture_paths(mesh_path: str):
    """Исправляет пути к текстурам в .mtl файле и при необходимости копирует файлы.

    В оригинальном пайплайне diffuse‑текстура сохраняется как `<basename>.jpg`,
    metallic — `<basename>_metallic.jpg`, roughness — `<basename>_roughness.jpg`.
    Для удобства в ряде приложений diffuse‑текстуру лучше именовать с суффиксом
    `_albedo.jpg`.  Эта функция проверяет наличие исходных файлов и при
    необходимости создаёт копию с ожидаемым именем, а также корректирует
    запись `map_Kd` в .mtl‑файле.  Карты metallic и roughness остаются без
    изменений.
    """
    mtl_path = os.path.splitext(mesh_path)[0] + ".mtl"
    if not os.path.exists(mtl_path):
        logger.warning(f"⚠️ Файл .mtl не найден: {mtl_path}")
        return

    base_path, base_name = (
        os.path.splitext(mesh_path)[0],
        os.path.splitext(os.path.basename(mesh_path))[0],
    )

    diffuse_original = f"{base_path}.jpg"
    diffuse_albedo = f"{base_path}_albedo.jpg"

    # Если оригинальная diffuse‑текстура существует и albedo‑варианта нет — создаём копию
    if os.path.exists(diffuse_original) and not os.path.exists(diffuse_albedo):
        try:
            import shutil

            shutil.copy(diffuse_original, diffuse_albedo)
            logger.info(f"✅ Скопирована diffuse‑текстура: {diffuse_albedo}")
        except Exception:
            logger.exception("⚠️ Не удалось скопировать diffuse‑текстуру")

    # Читаем содержимое .mtl файла
    try:
        with open(mtl_path) as f:
            content = f.read()
    except Exception:
        logger.exception(f"⚠️ Не удалось прочитать {mtl_path}")
        return

    # Обновляем map_Kd, если он указывает на исходную diffuse‑текстуру (.jpg или .png)
    for ext in (".jpg", ".png"):
        key = f"map_Kd {base_name}{ext}"
        if key in content:
            content = content.replace(key, f"map_Kd {base_name}_albedo.jpg")
            break

    # Записываем обновлённый файл
    try:
        with open(mtl_path, "w") as f:
            f.write(content)
    except Exception:
        logger.exception(f"⚠️ Не удалось записать {mtl_path}")

    logger.info(f"✅ Пути к текстурам исправлены в {mtl_path}")


def convert_glb_to_obj_with_textures(glb_path: str, obj_path: str | None = None):
    """Конвертирует GLB файл в OBJ с правильными текстурами"""
    if obj_path is None:
        obj_path = os.path.splitext(glb_path)[0] + ".obj"

    # 1) Попытка через Blender
    try:
        import bpy  # type: ignore[import-not-found]

        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        bpy.ops.import_scene.gltf(filepath=glb_path)
        bpy.ops.export_scene.obj(
            filepath=obj_path,
            use_selection=False,
            use_materials=True,
            use_triangles=True,
            use_normals=True,
            use_uvs=True,
        )

        base_name = os.path.splitext(os.path.basename(obj_path))[0]
        mtl_path = os.path.splitext(obj_path)[0] + ".mtl"
        mtl_content = (
            "newmtl Material\n"
            "Kd 0.8 0.8 0.8\n"
            "Ke 0.0 0.0 0.0\n"
            "Ni 1.5\n"
            "d 1.0\n"
            "illum 2\n"
            f"map_Kd {base_name}_albedo.jpg\n"
            f"map_Pm {base_name}_metallic.jpg\n"
            f"map_Pr {base_name}_roughness.jpg\n"
        )
        with open(mtl_path, "w") as f:
            f.write(mtl_content)
        logger.info("✅ GLB конвертирован в OBJ через Blender: %s", obj_path)
        logger.info("✅ Создан .mtl файл: %s", mtl_path)
        return obj_path
    except ImportError:
        logger.warning("⚠️ Blender недоступен, пробуем trimesh...")
    except Exception:
        logger.exception("⚠️ Ошибка Blender, пробуем trimesh...")

    # 2) Fallback через trimesh
    try:
        import trimesh  # type: ignore[import-not-found]

        mesh: Any
        try:
            mesh = trimesh.load(glb_path)
        except Exception:
            mesh = trimesh.load(glb_path, file_type="obj")
        mesh.export(obj_path)

        base_name = os.path.splitext(os.path.basename(obj_path))[0]
        mtl_path = os.path.splitext(obj_path)[0] + ".mtl"
        mtl_content = (
            "newmtl Material\n"
            "Kd 0.8 0.8 0.8\n"
            "Ke 0.0 0.0 0.0\n"
            "Ni 1.5\n"
            "d 1.0\n"
            "illum 2\n"
            f"map_Kd {base_name}_albedo.jpg\n"
            f"map_Pm {base_name}_metallic.jpg\n"
            f"map_Pr {base_name}_roughness.jpg\n"
        )
        with open(mtl_path, "w") as f:
            f.write(mtl_content)
        logger.info("✅ GLB конвертирован в OBJ через trimesh: %s", obj_path)
        logger.info("✅ Создан .mtl файл: %s", mtl_path)
        return obj_path
    except ImportError:
        logger.warning("⚠️ trimesh не установлен. Устанавливаем...")
        try:
            import subprocess

            subprocess.run(["uv", "add", "trimesh"], check=True)
            logger.info("✅ trimesh установлен. Попробуйте снова.")
        except Exception:
            logger.exception("❌ Ошибка установки trimesh")
        return None
    except Exception:
        logger.exception("❌ Ошибка конвертации через trimesh")
        return None


def main():
    """Главная функция с выбором операций"""
    print("🚀 Hunyuan3D-2.1 - Оптимизированная версия")
    print("=" * 60)
    print("💾 Управление памятью: АКТИВНО")
    print("🎮 GPU: NVIDIA GeForce RTX 4090 (24GB VRAM)")
    print()

    while True:
        print("\n📋 Выберите операцию:")
        print("1. Генерация меша из изображения")
        print("2. Генерация текстурированного меша из изображения")
        print("3. Генерация меша из текста")
        print("4. Генерация текстурированного меша из текста")
        print("5. Очистить память")
        print("6. Выход")

        choice = input("\nВведите номер операции (1-6): ").strip()

        if choice == "1":
            print("\n📋 Операция 1: Генерация меша из изображения")
            success = generate_mesh_from_image("assets/test_image.jpg", "output/test_mesh.glb")
            if success:
                print("✅ Операция завершена успешно!")
            else:
                print("❌ Операция завершена с ошибкой")

        elif choice == "2":
            print("\n📋 Операция 2: Генерация текстурированного меша из изображения")
            success = generate_textured_mesh_from_image("assets/test_image.jpg", "output/test_mesh_textured.glb")
            if success:
                print("✅ Операция завершена успешно!")
            else:
                print("❌ Операция завершена с ошибкой")

        elif choice == "3":
            print("\n📋 Операция 3: Генерация меша из текста")
            prompt = "(masterpiece), (best quality), game asset, a single longsword, front view, orthographic, 3d model, 3d render, hyper detailed, clean, ((white background)), ((isolated on white)), professional, studio lighting, sharp focus"
            success = generate_mesh_from_text(prompt, "output/sword.glb")
            if success:
                print("✅ Операция завершена успешно!")
            else:
                print("❌ Операция завершена с ошибкой")

        elif choice == "4":
            print("\n📋 Операция 4: Генерация текстурированного меша из текста")
            prompt = "(masterpiece), (best quality), game asset, a single longsword, front view, orthographic, 3d model, 3d render, hyper detailed, clean, ((white background)), ((isolated on white)), professional, studio lighting, sharp focus"
            success = generate_textured_mesh_from_text(prompt, "output/sword_textured.glb")
            if success:
                print("✅ Операция завершена успешно!")
            else:
                print("❌ Операция завершена с ошибкой")

        elif choice == "5":
            print("\n🧹 Очистка памяти...")
            unload_pipelines()
            print("✅ Память очищена")

        elif choice == "6":
            print("\n👋 До свидания!")
            break

        else:
            print("❌ Неверный выбор. Попробуйте снова.")


if __name__ == "__main__":
    main()
