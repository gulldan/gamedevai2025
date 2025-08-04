# 🎮 Low‑Poly 3D Generator

**Команда Квадрицепс** — Хакатон GameDevAI 2025

Облачный сервис, работающий в **одном Docker‑контейнере** и за минуту превращающий текст или референс‑картинку в low‑poly‑ассет с PBR‑текстурами и предпросмотром прямо в браузере.

> **Аппаратные требования**: RTX 4090 24 GB · CUDA 12.4 · PyTorch 2.3

---

## 1 Идея и цель

* **Проблема:** создание 3D‑ассетов требует дней ручного труда (моделирование → ретопология → UV → текстуры). Сток проламывает арт‑дирекцию, а фриланс дорог.
* **Цель:** сократить цикл производства **с нескольких дней до 60‑100 с** при сохранении контроля над стилем.
* **Решение:** единый веб‑сервис, который из текста или картинки генерирует готовый `.glb/.obj` с PBR‑текстурами. Пользователь видит результат сразу в браузере, может скачать и отправить в игровой движок.

---

## 2 Особенности и конкурентные преимущества

| Возможность         | Реализация в контейнере                              | Польза                                               |
| ------------------- | ---------------------------------------------------- | ---------------------------------------------------- |
| Два типа входа      | Текстовый промпт **или** изображение‑референс        | Закрывает и концепт‑арт, и фото‑проекции             |
| End‑to‑end‑конвейер | Image → Mesh → Textures → GLB в одном Python‑скрипте | Нет ручных переходов между Blender/Substance Painter |
| Пресеты стиля       | `production‑ready`, `origami`, `voxel`               | Смена визуала одним кликом                           |
| Управление VRAM     | Offload, attention slicing, expandable CUDA segments | Хватает 24 GB для всех этапов                        |
| Веб‑доставка        | Flask + Granian, `<model‑viewer>` + WhiteNoise       | Ассет просматривается прямо на GitHub                |
| **Единый UI**       | Форма генерации + 3D‑viewer                          | «Нажал — получил — прокрутил»                        |

> **Исследование моделей:** для image‑синтеза мы протестировали SD 1.5, SDXL 1.0, Midjourney v5.2 и выбрали `playground‑v2.5` — лучшая детализация и цвет. Сравнение → `demo/0_models_comparison.png`. Для 3D — `Hunyuan3D 2.1`, даёт водонепроницаемый меш и PBR‑текстуры.

---

## 3 Архитектура и реализация

### 3.1 Общая схема

```mermaid
flowchart TD
    A[Ввод пользователя<br/>(текст или изображение)] --> B[Playground v2.5<br/>1024×1024 RGB]
    B --> C[Препроцессинг<br/>удаление фона · crop 512²]
    C --> D[Hunyuan3D DiT FM<br/>реконструкция меша]
    D --> E[Децимация ≤2k трис<br/>повторная UV‑развёртка]
    E --> F[Hunyuan Paint<br/>синтез PBR‑текстур]
    F --> G[GLB/OBJ + viewer<br/>ZIP‑выгрузка]
```

### 3.2 Контейнер (всё‑в‑одном)

* **База:** `nvidia/cuda:12.4.1‑devel‑ubuntu22.04`.
* **Среда:** Conda env `hunyuan3d21`, Python 3.10, PyTorch 2.5.1 cu124.
* **Модели:** `playgroundai/playground‑v2.5‑1024px‑aesthetic`, `Hunyuan3D‑2.1` (+ Real‑ESRGAN ×4).
* **Сборка C++/CUDA:** `custom_rasterizer`, `DifferentiableRenderer` (pybind11).
* **Сервер:** Granian WSGI ≈ 25 k RPS, WhiteNoise для статики.
* **Viewer:** `<model-viewer>` с WebXR / Quick Look AR.

Полный Dockerfile — в репозитории; сборка: `docker build -t kvadricepts-gamedevai2025 .`.

### 3.3 Метрики (RTX 4090)

| Этап                     | Время        | VRAM       |
| ------------------------ | ------------ | ---------- |
| Синтез изображения 1024² | 6‑10 с       | 9‑10 GB    |
| Mesh reconstruction      | 20‑35 с      | 12‑13 GB   |
| PBR‑текстуры             | 30‑50 с      | 14‑15 GB   |
| **Полный цикл**          | **60‑100 с** | **≤15 GB** |

---

## 4 Пользовательский сценарий

<details>
<summary>Пошагово</summary>

1. **Ввод данных** — текстовый промпт *или* изображение.
2. **Настройка** — выбор стиля, лимит треугольников, CFG, steps.
3. **Генерация** — сервис показывает статус (SSE) и миниатюру.
4. **Предпросмотр** — интерактивный 3D‑viewer; при желании AR‑режим.
5. **Экспорт** — скачивание ZIP (GLB/OBJ + текстуры) или POST в хранилище студии.

</details>

### Видео‑демо

<table><tr>
<td width="50%"><b>Text‑to‑3D</b><br/>
<video src="demo/1_text_promt.webm" controls muted loop style="width:100%;border:1px solid #ccc;border-radius:6px"></video></td>
<td width="50%"><b>Image‑to‑3D</b><br/>
<video src="demo/2_image_promt.webm" controls muted loop style="width:100%;border:1px solid #ccc;border-radius:6px"></video></td>
</tr></table>

> GitHub правильно рендерит `<video>`; Markdown‑картинки для WEBM не поддерживаются.

### UI‑скриншоты

| Главная форма                                 | История задач                                  | Сравнение стилей                                     |
| --------------------------------------------- | ---------------------------------------------- | ---------------------------------------------------- |
| <img src="demo/main_page.png"   width="330"/> | <img src="demo/history.png"      width="330"/> | <img src="demo/text_promt_results.png" width="330"/> |

Результат Image‑to‑3D:<br/><img src="demo/image_promt_result.png" width="360"/>

---

## 🚀 Быстрый запуск (Docker)

```bash
docker run --gpus all -p 8880:8000 kvadricepts-gamedevai2025 \
           python -m granian --interface wsgi --workers 1 \
           --host 0.0.0.0 --port 8000 app:app
```

Откройте `http://localhost:8880`.

---

## 📜 Лицензии

* Playground v2.5 — PlaygroundAI Community License
* Hunyuan3D 2.1 — Apache 2.0
* Исходный код — MIT

---

💌 **Контакты:** quadriceps‑[team@example.com](mailto:team@example.com)
