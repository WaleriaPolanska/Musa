#!/usr/bin/env bash
# Запуск скриптов Музы.
#
#   ./run.sh                          -> musa/musa_chain.py (три этажа мозга)
#   ./run.sh musa/viewer.py
#   ./run.sh experiments/cpg/musa_cpg_walk_test.py
#
# Скрипты со своим окном (musa/window.py) идут под обычным python.
# Старые, на встроенном вьювере MuJoCo, — под mjpython: на macOS
# mujoco.viewer.launch_passive иначе не открывается.

set -e
cd "$(dirname "$0")"

script="${1:-musa/musa_chain.py}"

if grep -q "mujoco.viewer" "$script"; then
    exec ./flygym21/bin/mjpython "$script" "${@:2}"
fi

exec ./flygym21/bin/python "$script" "${@:2}"


