Вот финальная версия `README.md`, без повторений, с исправленным форматом и рекомендацией по вставке видео:

````markdown
# 🎮 Low‑Poly 3D Generator  
**Команда Квадрицепс · GameDevAI 2025**

Облачный Docker‑сервис, превращающий текст или изображение в стилизованный low‑poly 3D‑ассет с PBR‑текстурами и предпросмотром в браузере.

> Минимум: RTX 4090 · CUDA 12.4 · PyTorch 2.5.1

---

## 🎯 Проблема и цель

Создание 3D ассетов — долгий и дорогой процесс. Наш сервис позволяет за минуты получить готовую low‑poly‑модель из текстового описания или изображения, ускоряя разработку и снижая затраты.

---

## ✨ Возможности и преимущества

| Функция                     | Описание                                                                 |
|----------------------------|--------------------------------------------------------------------------|
| **Два типа ввода**         | Текстовый промпт или загруженное изображение                             |
| **3D генерация**           | Генерация mesh, UV и PBR текстур через Hunyuan3D                         |
| **Стилизация**             | Presets: `production-ready`, `origami`, `voxel`                          |
| **Оптимизация VRAM**       | Attention slicing, выгрузка пайплайнов, до 15 GB VRAM                    |
| **Единый сервис**          | Все этапы от текста/изображения до .GLB в одном REST API                |
| **Интерактивный просмотр** | `<model-viewer>` с камерой и AR прямо в браузере                         |

---

## 🧱 Архитектура

```mermaid
flowchart TD
    subgraph docker[Docker Container]
        direction TB
        A[Flask / Granian<br>REST + SSE] --> B[Pipelines<br>Playground v2.5<br>Hunyuan3D 2.1]
        B --> C[File Storage<br>GLB / OBJ / ZIP]
        A --> D[Static UI<br>(HTML + <model‑viewer>)]
    end
    E[Браузер пользователя] -->|HTTP/WS| A
    C -->|WhiteNoise| D
````

---

## ⚙️ Сборка и запуск

```bash
# Сборка
docker build -t lowpoly-generator .

# Запуск
docker run --gpus all -p 8880:8000 lowpoly-generator \
  python -m granian --interface wsgi --workers 1 \
  --host 0.0.0.0 --port 8000 app:app
```

Интерфейс: [http://localhost:8880](http://localhost:8880)

---

## 📦 Стек

* **Изображения**: [Playground v2.5](https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic)
* **3D‑модели**: [Hunyuan3D 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)
* **Viewer**: Google `<model-viewer>` с AR
* **Backend**: Python 3.10, PyTorch 2.5.1 (CUDA 12.4), Flask + Granian
* **Container**: Ubuntu 22.04 + Blender (bpy 4.0)

---

## 📊 Производительность

| Этап                    | Среднее время  | Пик VRAM    |
| ----------------------- | -------------- | ----------- |
| Изображение (1024x1024) | 6‑10 сек       | \~10 GB     |
| Реконструкция меша      | 20‑35 сек      | \~13 GB     |
| Синтез текстур          | 30‑50 сек      | \~15 GB     |
| **Полный цикл**         | **60‑100 сек** | **\~15 GB** |

---

## 🧪 Демонстрации

### Текстовый запрос

<video width="640" controls>
  <source src="https://github.com/user-attachments/assets/f7e43b4e-48c2-4a96-a300-aa7e5593470e" type="video/mp4">
  Your browser does not support the video tag.
</video>


### Генерация из изображения

<video width="640" controls>
  <source src="https://github.com/user-attachments/assets/74f5cb44-2fab-447a-ad33-ba22a0de1eb9" type="video/mp4">
  Your browser does not support the video tag.
</video>


> **Важно**: для GitHub используйте `.mp4`, загружая через интерфейс issues/PR или dragging в редакторе. `.webm` не поддерживается как встраиваемое видео.

---

## 🧠 Как работает

1. **Промпт или изображение**
2. **Генерация изображения (Playground v2.5)**
3. **Удаление фона, подготовка**
4. **3D‑реконструкция + UV (Hunyuan3D)**
5. **PBR‑текстуры**
6. **Просмотр в `<model-viewer>`**
7. **Скачивание `.glb` / `.zip`**

---

## 📄 Лицензии

* Playground v2.5 — PlaygroundAI Community License
* Hunyuan3D‑2.1 — Apache 2.0
* Код проекта — MIT

---

## 📁 Скриншоты

| Главная страница            | История генераций            |
| --------------------------- | ---------------------------- |
| ![Main](demo/main_page.png) | ![History](demo/history.png) |

| Примеры ассетов из текста            | Изображений                           |
| ------------------------------------ | ------------------------------------- |
| ![Text](demo/text_promt_results.png) | ![Image](demo/image_promt_result.png) |

| Сравнение моделей                        |
| ---------------------------------------- |
| ![Compare](demo/0_models_comparison.png) |

```


