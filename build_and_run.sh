docker build -t kvadricepts-gamedevai2025 .
docker run -it --gpus all -p 8880:8000 kvadricepts-gamedevai2025 python -m granian --interface wsgi --workers 1 --host 0.0.0.0 --port 8000 app:app