# get the development image from nvidia cuda 12.4 (using devel for full CUDA toolkit)
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

LABEL name="hunyuan3d21" maintainer="hunyuan3d21"

# create workspace folder and set it as working directory
RUN mkdir -p /workspace
WORKDIR /workspace

# update package lists and install git, wget, vim, libegl1-mesa-dev, and libglib2.0-0
RUN apt-get update && apt-get install -y build-essential git wget vim libegl1-mesa-dev libglib2.0-0 unzip git-lfs

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends pkg-config libglvnd0 libgl1 libglx0 libegl1 libgles2 libglvnd-dev libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev cmake curl mesa-utils-extra libxrender1

# Install additional dependencies for compilation
RUN apt-get install -y libeigen3-dev python3-dev python3-setuptools libcgal-dev

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
ENV PYOPENGL_PLATFORM=egl

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"

# install conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -b -p /workspace/miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh

# update PATH environment variable
ENV PATH="/workspace/miniconda3/bin:${PATH}"

# initialize conda
RUN conda init bash

# Accept conda terms of service and create environment
RUN conda config --set always_yes true && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -n hunyuan3d21 python=3.10 --yes && \
    echo "source activate hunyuan3d21" > ~/.bashrc
ENV PATH="/workspace/miniconda3/envs/hunyuan3d21/bin:${PATH}"

RUN conda install Ninja --yes
RUN conda install cuda -c nvidia/label/cuda-12.4.1 --yes

# Update libstdcxx-ng to fix compatibility issues (auto-confirm)
RUN conda install -c conda-forge libstdcxx-ng --yes

RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Clone Hunyuan3D-2.1 repository
COPY Hunyuan3D-2.1 Hunyuan3D-2.1
COPY frontend frontend

# Install Python dependencies from modified requirements.txt
RUN pip install -r Hunyuan3D-2.1/requirements.txt
RUN pip install bpy==4.0 --extra-index-url https://download.blender.org/pypi/
RUN pip install granian rembg pybind11

# Install custom_rasterizer
RUN cd /workspace/Hunyuan3D-2.1/hy3dpaint/custom_rasterizer && \
    # Set compilation environment variables
    export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0" && \
    export CUDA_NVCC_FLAGS="-allow-unsupported-compiler" && \
    # Install with editable mode
    pip install -e .

# Install DifferentiableRenderer (fixed compilation without uv)
RUN cd /workspace/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer && \
    PYTHON_EXEC=$(python -c "import sys; print(sys.executable)") && \
    PYBIND11_INCLUDES=$(python -m pybind11 --includes) && \
    EXT_SUFFIX=$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))") && \
    c++ -O3 -Wall -shared -std=c++11 -fPIC $PYBIND11_INCLUDES mesh_inpaint_processor.cpp -o mesh_inpaint_processor$EXT_SUFFIX

# Create ckpt folder in hy3dpaint and download RealESRGAN model
RUN cd /workspace/Hunyuan3D-2.1/hy3dpaint && \
    mkdir -p ckpt && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P ckpt

# Modify textureGenPipeline.py to update config path
RUN cd /workspace/Hunyuan3D-2.1/hy3dpaint && \
    sed -i 's/self\.multiview_cfg_path = "cfgs\/hunyuan-paint-pbr\.yaml"/self.multiview_cfg_path = "hy3dpaint\/cfgs\/hunyuan-paint-pbr.yaml"/' textureGenPipeline.py

# Modify multiview_utils.py to update custom_pipeline path
RUN cd /workspace/Hunyuan3D-2.1/hy3dpaint/utils && \
    sed -i 's/custom_pipeline = config\.custom_pipeline/custom_pipeline = os.path.join(os.path.dirname(__file__),"..","hunyuanpaintpbr")/' multiview_utils.py

# Set working directory to the cloned repository
WORKDIR /workspace/Hunyuan3D-2.1

# Set global library paths to ensure proper linking at runtime
ENV LD_LIBRARY_PATH="/workspace/miniconda3/envs/hunyuan3d21/lib:${LD_LIBRARY_PATH}"

RUN apt-get install -y libxi6 libgconf-2-4 libxkbcommon-x11-0 libsm6 libxext6 libxrender-dev

# Activate conda environment by default
RUN echo "conda activate hunyuan3d21" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

#exposing 7860 port
EXPOSE 7860

# Cleanup
RUN rm -f /workspace/*.zip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set default command
WORKDIR /workspace
RUN mkdir gradio_cache
ENV IMAGE_GEN_MODEL_ID=playgroundai/playground-v2.5-1024px-aesthetic
ENV PYTHONPATH=frontend

CMD ["python", "-m", "granian", "--interface", "wsgi", "--workers", "1", "--host", "0.0.0.0", "--port", "8000", "app:app"]
