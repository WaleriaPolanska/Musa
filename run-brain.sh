#!/usr/bin/env bash
# Скрипты по коннектому maleCNS. Отдельное окружение brainvenv:
# navis + neuprint-python, к симуляции flygym отношения не имеет.
#
#   ./run-brain.sh brain/find.py LPLC2
#   ./run-brain.sh brain/view.py --type LPLC2 --limit 5

set -e
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/brain:$PYTHONPATH"
exec ./brainvenv/bin/python "$@"
