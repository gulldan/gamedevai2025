#!/bin/bash
PYTHON_EXEC=$(uv run python -c "import sys; print(sys.executable)")
PYBIND11_INCLUDES=$(uv run python -m pybind11 --includes)
EXT_SUFFIX=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

c++ -O3 -Wall -shared -std=c++11 -fPIC $PYBIND11_INCLUDES mesh_inpaint_processor.cpp -o mesh_inpaint_processor$EXT_SUFFIX