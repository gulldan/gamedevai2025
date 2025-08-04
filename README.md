# 🎮 Low-Poly 3D Generator

Современный веб-сервис для генерации low-poly 3D моделей из текста или изображений. Оптимизирован для 24GB VRAM с эффективным управлением памятью.

Запуск 
- ./ 

или

- IMAGE_GEN_MODEL_ID=playgroundai/playground-v2.5-1024px-aesthetic PYTHONPATH=frontend uv run granian --interface wsgi --workers 1 --host 0.0.0.0 --port 8000 app:app