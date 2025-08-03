#!/bin/bash
set -e  # Остановка при ошибке

echo "🚀 Начинаем установку..."

# Шаг 1: Установка PyTorch 2.5.1+cu124
echo "📦 Устанавливаем PyTorch 2.5.1+cu124..."
uv add --frozen torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# Шаг 2: Установка зависимостей из requirements.txt
echo "📦 Устанавливаем зависимости из requirements.txt..."
cd Hunyuan3D-2.1
uv add --frozen -r requirements.txt

# Шаг 3: Установка custom_rasterizer с отключенной изоляцией сборки
echo "🔧 Устанавливаем custom_rasterizer..."
cd hy3dpaint/custom_rasterizer
uv pip install -e . --no-build-isolation

# Шаг 4: Компиляция DifferentiableRenderer
echo "🔧 Компилируем DifferentiableRenderer..."
cd ../DifferentiableRenderer
if [ -f "compile_mesh_painter.sh" ]; then
    bash compile_mesh_painter.sh
else
    echo "⚠️  Файл compile_mesh_painter.sh не найден"
fi

# Шаг 5: Скачивание RealESRGAN модели
echo "📥 Скачиваем RealESRGAN модель..."
cd ../..
mkdir -p hy3dpaint/ckpt
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P hy3dpaint/ckpt

# Возвращаемся в корневую директорию
cd ..

echo "✅ Установка завершена!"