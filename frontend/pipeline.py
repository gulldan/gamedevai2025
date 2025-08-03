#!/usr/bin/env python3
"""
Оптимизированная версия основного скрипта Hunyuan3D с агрессивным
управлением памятью и корректной работой с текстурированными мешами.

Изменения по сравнению с исходной версией:
* Исправлена ошибка, когда в pipeline рисования передавался путь с
  расширением «.glb».  Авторский pipeline ожидает путь к OBJ‑файлу и
  автоматически конвертирует его в GLB через Blender.  Передача пути с
  расширением «.glb» приводила к тому, что текстура записывалась в
  OBJ‑формате под неправильным именем, а затем попытка конвертации через
  trimesh выдавала ошибку «incorrect header on GLB file».  Теперь
  функция `generate_textured_mesh_from_image` проверяет расширение
  выходного файла.  Если пользователь указал «.glb», то временный
  OBJ‑файл генерируется с тем же именем, но расширением «.obj».  После
  завершения работы pipeline файл OBJ конвертируется в GLB при помощи
  `trimesh` (или Blender, если доступен), и возвращается ожидаемый
  GLB‑файл.
* В функции `convert_glb_to_obj_with_textures` добавлен надёжный
  обработчик: если чтение GLB через `trimesh` завершается исключением
  (например, «incorrect header on GLB file»), предпринимается попытка
  загрузить файл как OBJ.  Это позволяет корректно обрабатывать случаи,
  когда файл имеет расширение «.glb», но фактически представляет
  собой OBJ‑файл.

Этот скрипт предполагает, что все зависимости (Hunyuan3D, diffusers,
torch, PIL, trimesh и пр.) установлены и находятся в ожидаемых
каталогах.  В противном случае могут потребоваться дополнительные
действия по настройке окружения.
"""

import gc
import os
import sys
import warnings

# Устанавливаем путь к библиотекам PyTorch для custom_rasterizer
torch_lib_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".venv/lib/python3.10/site-packages/torch/lib",
)
if os.path.exists(torch_lib_path):
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if torch_lib_path not in current_ld_path:
        os.environ["LD_LIBRARY_PATH"] = (
            f"{current_ld_path}:{torch_lib_path}" if current_ld_path else torch_lib_path
        )

# Добавляем пути к модулям Hunyuan3D согласно официальной документации
sys.path.insert(0, "./Hunyuan3D-2.1/hy3dshape")
sys.path.insert(0, "./Hunyuan3D-2.1/hy3dpaint")

import torch
from diffusers import DiffusionPipeline
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from PIL import Image
from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

warnings.filterwarnings("ignore")

# Применяем torchvision fix как в официальном примере
try:
    sys.path.insert(0, "./Hunyuan3D-2.1")
    from torchvision_fix import apply_fix

    apply_fix()
except ImportError:
    print(
        "Warning: torchvision_fix module not found, proceeding without compatibility fix"
    )
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")

# Настройка управления памятью CUDA
torch.cuda.empty_cache()
if torch.cuda.is_available():
    # Устанавливаем переменную окружения для лучшего управления памятью
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Глобальные переменные для pipeline
shape_pipeline = None
paint_pipeline = None
image_gen_pipeline = None


def clear_memory():
    """Очищает память GPU и CPU"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    print("🧹 Память очищена")


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
    import trimesh
    from hy3dpaint.utils.uvwrap_utils import mesh_uv_wrap

    # Загружаем меш (поддерживаются как GLB, так и OBJ)
    mesh = trimesh.load(input_glb)

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
    try:
        mesh = mesh_uv_wrap(mesh)
    except Exception as e:
        print(f"⚠️ Ошибка при развёртке UV: {e}. Меш сохранён без обновления UV.")

    # Сохраняем упрощённый меш
    mesh.export(output_glb)
    return output_glb


def load_shape_pipeline():
    """Загружает shape pipeline с очисткой памяти"""
    global shape_pipeline
    if shape_pipeline is None:
        print("📦 Загружаем shape pipeline...")
        clear_memory()
        shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2.1"
        )
        print("✅ Shape pipeline загружен")
    return shape_pipeline


def load_paint_pipeline():
    """Загружает paint pipeline с очисткой памяти"""
    global paint_pipeline
    if paint_pipeline is None:
        print("📦 Загружаем paint pipeline...")
        clear_memory()
        try:
            max_num_view = 6
            resolution = 512
            conf = Hunyuan3DPaintConfig(max_num_view, resolution)
            conf.realesrgan_ckpt_path = (
                "Hunyuan3D-2.1/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
            )
            conf.multiview_cfg_path = (
                "Hunyuan3D-2.1/hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
            )
            conf.custom_pipeline = "Hunyuan3D-2.1/hy3dpaint/hunyuanpaintpbr"
            paint_pipeline = Hunyuan3DPaintPipeline(conf)
            print("✅ Paint pipeline загружен")
        except Exception as e:
            print(f"❌ Ошибка загрузки paint pipeline: {e}")
            return None
    return paint_pipeline


def load_image_gen_pipeline():
    """Загружает image generation pipeline с очисткой памяти"""
    global image_gen_pipeline
    if image_gen_pipeline is None:
        print("📦 Загружаем image generation pipeline...")
        clear_memory()
        try:
            # Позволяем указать альтернативную модель для генерации изображений
            model_id = os.environ.get(
                "IMAGE_GEN_MODEL_ID", "playgroundai/playground-v2.5-1024px-aesthetic"
            )
            try:
                # Попробуем загрузить стандартный диффузионный пайплайн
                image_gen_pipeline = DiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    variant="fp16",
                ).to(device)
            except Exception as diff_err:
                # Если загрузить DiffusionPipeline не удалось и имя модели содержит
                # "flux", попробуем использовать FluxPipeline (модели FLUX
                # используют собственный класс пайплайна в diffusers).
                if "flux" in model_id.lower():
                    from diffusers import FluxPipeline  # type: ignore

                    image_gen_pipeline = FluxPipeline.from_pretrained(
                        model_id,
                        torch_dtype=torch.bfloat16,
                    ).to(device)
                else:
                    raise diff_err

            # Снижаем потребление VRAM: переносим части модели на CPU и
            # включаем slicing внимания, если эти методы доступны
            for method_name in [
                "enable_model_cpu_offload",
                "enable_attention_slicing",
                "enable_xformers_memory_efficient_attention",
            ]:
                method = getattr(image_gen_pipeline, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception:
                        pass
            print(f"✅ Image generation pipeline загружен: {model_id}")
        except Exception as e:
            print(f"❌ Ошибка загрузки image generation pipeline: {e}")
            return None
    return image_gen_pipeline


def unload_pipelines():
    """Выгружает все pipeline для освобождения памяти"""
    global shape_pipeline, paint_pipeline, image_gen_pipeline

    if shape_pipeline is not None:
        del shape_pipeline
        shape_pipeline = None
        print("🗑️ Shape pipeline выгружен")

    if paint_pipeline is not None:
        del paint_pipeline
        paint_pipeline = None
        print("🗑️ Paint pipeline выгружен")

    if image_gen_pipeline is not None:
        del image_gen_pipeline
        image_gen_pipeline = None
        print("🗑️ Image generation pipeline выгружен")

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


def generate_mesh_from_image(image_path: str, output_path: str):
    """Генерирует меш из изображения с оптимизацией памяти"""
    if not os.path.exists(image_path):
        print(f"❌ Файл изображения не найден: {image_path}")
        return False

    print("🔄 Начинаем генерацию меша из изображения...")
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
        print(f"📸 Референсное изображение сохранено: {ref_image_path}")

        # Генерируем меш
        print("🎯 Генерируем меш...")
        mesh_untextured = pipeline(image=processed_image, show_progress_bar=False)[0]
        mesh_untextured.export(output_path)
        print(f"✅ Меш сохранен: {output_path}")

        return True

    except torch.cuda.OutOfMemoryError:
        print("❌ Недостаточно VRAM для генерации меша")
        print("💡 Попробуйте освободить память или уменьшить размер изображения")
        return False
    except Exception as e:
        print(f"❌ Ошибка генерации меша: {e}")
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
        print(f"❌ Файл изображения не найден: {image_path}")
        return False

    print("🔄 Начинаем генерацию текстурированного меша из изображения...")
    clear_memory()

    try:
        # Этап 1: Генерируем меш без текстуры
        print("🎯 Этап 1: Генерируем меш без текстуры...")
        shape_pipe = load_shape_pipeline()
        if shape_pipe is None:
            return False

        image = Image.open(image_path).convert("RGB")
        processed_image = resize_and_pad(image, (512, 512))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        base_path, ext = os.path.splitext(output_path)
        ref_image_path = base_path + ".png"
        processed_image.save(ref_image_path)
        print(f"📸 Референсное изображение сохранено: {ref_image_path}")

        mesh_untextured = shape_pipe(image=processed_image, show_progress_bar=False)[0]
        untextured_path = base_path + "_untextured.glb"
        mesh_untextured.export(untextured_path)
        print(f"✅ Меш без текстуры сохранен: {untextured_path}")

        # Очищаем память после генерации меша
        clear_memory()

        # Этап 2: Генерируем текстуру
        print("🎨 Этап 2: Генерируем текстуру...")
        paint_pipe = load_paint_pipeline()
        if paint_pipe is None:
            print("❌ Paint pipeline недоступен")
            return False

        # Если требуется предварительное упрощение меша — делаем это до вызова пайплайна
        mesh_for_paint = untextured_path
        if target_face_count is not None:
            simplified_path = base_path + "_preprocessed.glb"
            try:
                simplified_path = simplify_mesh_and_rewrap(
                    untextured_path, simplified_path, target_face_count
                )
                mesh_for_paint = simplified_path
                # при пользовательском упрощении не нужен ремешинг внутри пайплайна
                use_remesh = False
            except Exception as e:
                print(
                    f"⚠️ Не удалось упростить и переобернуть меш: {e}. Продолжаем с исходным."
                )
        else:
            # Если пользователь отключил ремешинг, но не указал упрощение,
            # делаем повторную UV-развёртку исходного меша.  Это полезно, если
            # оригинальная UV-карта не совпадает с текстурой.
            if not use_remesh:
                try:
                    import trimesh  # type: ignore
                    from hy3dpaint.utils.uvwrap_utils import mesh_uv_wrap

                    mesh_tmp = trimesh.load(untextured_path)
                    mesh_tmp = mesh_uv_wrap(mesh_tmp)
                    mesh_tmp.export(untextured_path)
                    print("🔁 Выполнена повторная UV-развёртка исходного меша")
                except Exception as e:
                    print(f"⚠️ Ошибка при повторной UV-развёртке: {e}")

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
                import bpy  # type: ignore

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
                bpy.ops.export_scene.gltf(
                    filepath=final_glb_path, use_active_scene=True
                )
                print(f"✅ OBJ конвертирован в GLB через Blender: {final_glb_path}")
            except Exception:
                # Если Blender недоступен или произошла любая ошибка, пытаемся
                # конвертировать через trimesh
                try:
                    import trimesh  # type: ignore

                    mesh = trimesh.load(obj_output_path)
                    mesh.export(final_glb_path)
                    print(f"✅ OBJ конвертирован в GLB через trimesh: {final_glb_path}")
                except Exception as e:
                    print(
                        f"⚠️ Не удалось конвертировать OBJ в GLB: {e}. Оставляем OBJ в качестве результата."
                    )
                    import shutil

                    shutil.copy(obj_output_path, final_glb_path)

        # Создаем OBJ версию с текстурами для пользователя
        obj_path = obj_output_path
        print(f"✅ Текстурированный OBJ сохранен: {obj_path}")
        if desired_ext == ".glb":
            print(f"✅ Текстурированный GLB сохранен: {final_glb_path}")
        return True

    except torch.cuda.OutOfMemoryError:
        print("❌ Недостаточно VRAM для генерации")
        print("💡 Попробуйте освободить память или уменьшить размер изображения")
        return False
    except Exception as e:
        print(f"❌ Ошибка генерации: {e}")
        return False
    finally:
        clear_memory()


def generate_mesh_from_text(prompt: str, output_path: str):
    """Генерирует меш из текста с оптимизацией памяти"""
    print("🔄 Начинаем генерацию меша из текста...")
    clear_memory()

    try:
        # Загружаем image generation pipeline
        img_pipe = load_image_gen_pipeline()
        if img_pipe is None:
            print("❌ Image generation pipeline недоступен")
            return False

        # Генерируем изображение
        print("🎨 Генерируем изображение из текста...")
        image = img_pipe(prompt=prompt).images[0]
        processed_image = resize_and_pad(image, (256, 512))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        ref_image_path = os.path.splitext(output_path)[0] + ".png"
        processed_image.save(ref_image_path)
        print(f"📸 Сгенерированное изображение сохранено: {ref_image_path}")

        # Очищаем память после генерации изображения
        clear_memory()

        # Генерируем меш
        print("🎯 Генерируем меш из изображения...")
        shape_pipe = load_shape_pipeline()
        if shape_pipe is None:
            return False

        mesh_untextured = shape_pipe(image=processed_image, show_progress_bar=False)[0]
        mesh_untextured.export(output_path)
        print(f"✅ Меш из текста сохранен: {output_path}")

        return True

    except torch.cuda.OutOfMemoryError:
        print("❌ Недостаточно VRAM для генерации")
        return False
    except Exception as e:
        print(f"❌ Ошибка генерации: {e}")
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
    print("🔄 Начинаем генерацию текстурированного меша из текста...")
    clear_memory()

    try:
        # Загружаем image generation pipeline
        img_pipe = load_image_gen_pipeline()
        if img_pipe is None:
            print("❌ Image generation pipeline недоступен")
            return False

        # Генерируем изображение
        print("🎨 Генерируем изображение из текста...")
        image = img_pipe(prompt=prompt).images[0]
        processed_image = resize_and_pad(image, (512, 512))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        base_path, ext = os.path.splitext(output_path)
        ref_image_path = base_path + ".png"
        processed_image.save(ref_image_path)
        print(f"📸 Сгенерированное изображение сохранено: {ref_image_path}")

        # Очищаем память после генерации изображения
        # Выгружаем image_gen_pipeline, чтобы освободить VRAM перед загрузкой shape-пайплайна
        global image_gen_pipeline
        if image_gen_pipeline is not None:
            del image_gen_pipeline
            image_gen_pipeline = None
        clear_memory()

        # Генерируем меш
        print("🎯 Генерируем меш из изображения...")
        shape_pipe = load_shape_pipeline()
        if shape_pipe is None:
            return False

        mesh_untextured = shape_pipe(image=processed_image, show_progress_bar=False)[0]
        untextured_path = base_path + "_untextured.glb"
        mesh_untextured.export(untextured_path)
        print(f"✅ Меш без текстуры сохранен: {untextured_path}")

        # Очищаем память после генерации меша
        clear_memory()

        # Генерируем текстуру
        print("🎨 Генерируем текстуру...")
        paint_pipe = load_paint_pipeline()
        if paint_pipe is None:
            print("❌ Paint pipeline недоступен")
            return False

        # Если требуется предварительное упрощение и переобёртка, выполняем её здесь
        mesh_for_paint = untextured_path
        if target_face_count is not None:
            simplified_path = base_path + "_preprocessed.glb"
            try:
                simplified_path = simplify_mesh_and_rewrap(
                    untextured_path, simplified_path, target_face_count
                )
                mesh_for_paint = simplified_path
                use_remesh = False
            except Exception as e:
                print(
                    f"⚠️ Не удалось упростить и переобернуть меш: {e}. Продолжаем с исходным."
                )
        else:
            # если пользователь отключает ремешинг, но не указал упрощения,
            # выполняем повторную UV-развёртку исходного меша
            if not use_remesh:
                try:
                    import trimesh  # type: ignore
                    from hy3dpaint.utils.uvwrap_utils import mesh_uv_wrap

                    mesh_tmp = trimesh.load(untextured_path)
                    mesh_tmp = mesh_uv_wrap(mesh_tmp)
                    mesh_tmp.export(untextured_path)
                    print("🔁 Выполнена повторная UV-развёртка исходного меша")
                except Exception as e:
                    print(f"⚠️ Ошибка при повторной UV-развёртке: {e}")

        desired_ext = ext.lower()
        obj_output_path = base_path + ".obj"
        output_mesh_path = paint_pipe(
            mesh_path=mesh_for_paint,
            image_path=ref_image_path,
            output_mesh_path=obj_output_path,
            use_remesh=use_remesh,
        )
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
                import bpy  # type: ignore

                bpy.ops.object.select_all(action="SELECT")
                bpy.ops.object.delete(use_global=False)

                # Импортируем OBJ для разных версий Blender
                if hasattr(bpy.ops, "wm") and hasattr(bpy.ops.wm, "obj_import"):
                    bpy.ops.wm.obj_import(filepath=obj_output_path)
                else:
                    bpy.ops.import_scene.obj(filepath=obj_output_path)

                bpy.ops.export_scene.gltf(
                    filepath=final_glb_path, use_active_scene=True
                )
                print(f"✅ OBJ конвертирован в GLB через Blender: {final_glb_path}")
            except Exception:
                try:
                    import trimesh  # type: ignore

                    mesh = trimesh.load(obj_output_path)
                    mesh.export(final_glb_path)
                    print(f"✅ OBJ конвертирован в GLB через trimesh: {final_glb_path}")
                except Exception as e:
                    print(
                        f"⚠️ Не удалось конвертировать OBJ в GLB: {e}. Оставляем OBJ в качестве результата."
                    )
                    import shutil

                    shutil.copy(obj_output_path, final_glb_path)

        obj_path = obj_output_path
        print(f"✅ Текстурированный OBJ сохранен: {obj_path}")
        if desired_ext == ".glb":
            print(f"✅ Текстурированный GLB сохранен: {final_glb_path}")
        return True

    except torch.cuda.OutOfMemoryError:
        print("❌ Недостаточно VRAM для генерации")
        return False
    except Exception as e:
        print(f"❌ Ошибка генерации: {e}")
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
        print(f"⚠️ Файл .mtl не найден: {mtl_path}")
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
            print(f"✅ Скопирована diffuse‑текстура: {diffuse_albedo}")
        except Exception as e:
            print(f"⚠️ Не удалось скопировать diffuse‑текстуру: {e}")

    # Читаем содержимое .mtl файла
    with open(mtl_path, "r") as f:
        content = f.read()

    # Обновляем map_Kd, если он указывает на исходную diffuse‑текстуру
    if f"map_Kd {base_name}.jpg" in content:
        content = content.replace(
            f"map_Kd {base_name}.jpg", f"map_Kd {base_name}_albedo.jpg"
        )

    # Записываем обновлённый файл
    with open(mtl_path, "w") as f:
        f.write(content)

    print(f"✅ Пути к текстурам исправлены в {mtl_path}")


def convert_glb_to_obj_with_textures(glb_path: str, obj_path: str = None):
    """Конвертирует GLB файл в OBJ с правильными текстурами"""
    if obj_path is None:
        obj_path = os.path.splitext(glb_path)[0] + ".obj"

    try:
        # Пробуем использовать Blender для конвертации
        import bpy  # type: ignore

        # Очищаем сцену
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)

        # Импортируем GLB
        bpy.ops.import_scene.gltf(filepath=glb_path)

        # Экспортируем в OBJ
        bpy.ops.export_scene.obj(
            filepath=obj_path,
            use_selection=False,
            use_materials=True,
            use_triangles=True,
            use_normals=True,
            use_uvs=True,
        )

        # Создаем .mtl файл с правильными путями к текстурам
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

        print(f"✅ GLB конвертирован в OBJ через Blender: {obj_path}")
        print(f"✅ Создан .mtl файл: {mtl_path}")

        return obj_path

    except ImportError:
        print("⚠️ Blender Python модуль недоступен, используем trimesh...")
        try:
            import trimesh  # type: ignore

            # Загружаем GLB или OBJ файл. Если загружать как GLB не удаётся,
            # попробуем загрузить как OBJ. Это устраняет ошибку 'incorrect header'.
            try:
                mesh = trimesh.load(glb_path)
            except Exception:
                # Файл может иметь расширение .glb, но фактически быть OBJ
                mesh = trimesh.load(glb_path, file_type="obj")

            # Экспортируем в OBJ
            mesh.export(obj_path)

            # Создаем .mtl файл с правильными путями к текстурам
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

            print(f"✅ GLB конвертирован в OBJ через trimesh: {obj_path}")
            print(f"✅ Создан .mtl файл: {mtl_path}")

            return obj_path

        except ImportError:
            print("⚠️ trimesh не установлен. Устанавливаем...")
            try:
                import subprocess

                subprocess.run(["uv", "add", "trimesh"], check=True)
                print("✅ trimesh установлен. Попробуйте снова.")
            except Exception as e:
                print(f"❌ Ошибка установки trimesh: {e}")
            return None
        except Exception as e:
            print(f"❌ Ошибка конвертации через trimesh: {e}")
            return None
    except Exception as e:
        print(f"❌ Ошибка конвертации через Blender: {e}")
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
            success = generate_mesh_from_image(
                "assets/test_image.jpg", "output/test_mesh.glb"
            )
            if success:
                print("✅ Операция завершена успешно!")
            else:
                print("❌ Операция завершена с ошибкой")

        elif choice == "2":
            print("\n📋 Операция 2: Генерация текстурированного меша из изображения")
            success = generate_textured_mesh_from_image(
                "assets/test_image.jpg", "output/test_mesh_textured.glb"
            )
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
            success = generate_textured_mesh_from_text(
                prompt, "output/sword_textured.glb"
            )
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
