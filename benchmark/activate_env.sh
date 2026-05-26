#!/usr/bin/env bash
# Activate the robopiano conda env on Mac (Apple Silicon, miniforge3)
# Usage: source benchmark/activate_env.sh

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate robopiano

# Add planner subprojects to PYTHONPATH (editable installs cover most of this,
# but PYTHONPATH provides a safety net for any subproject not pip-installed)
export PYTHONPATH="/Users/aangeles/robopiano/Bagatelle/src:/Users/aangeles/robopiano/Impromptu/src:/Users/aangeles/robopiano/Intermezzo/src:/Users/aangeles/robopiano/Variations:${PYTHONPATH:-}"

# MuJoCo rendering backend for Apple Silicon. glfw is the safest default
# for offscreen GL on macOS. Override before sourcing if you need a different one.
export MUJOCO_GL="${MUJOCO_GL:-glfw}"

# Make Conda runtime libs first on the dyld path so mujoco's bundled libstdc++
# is used. Mirrors the LD_LIBRARY_PATH dance in HowToRun.md.
export DYLD_LIBRARY_PATH="$CONDA_PREFIX/lib:${DYLD_LIBRARY_PATH:-}"

echo "[activate_env] conda env: $CONDA_DEFAULT_ENV  python: $(python --version 2>&1)"
echo "[activate_env] MUJOCO_GL=$MUJOCO_GL"
