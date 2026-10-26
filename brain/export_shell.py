"""Оболочка мозга из коннектома — в виде меша для симуляции.

Берёт слой brain-shell из того же неуроглансера, что открывается по ссылке
male-cns v1.0, прорежает его и переводит в миллиметры и оси MuJoCo,
в ту же систему координат, что и нейроны из export_circuit.py.

    ./run-brain.sh brain/export_shell.py

Сам файл оболочки (53 МБ) качается один раз.
"""

from pathlib import Path

import fast_simplification
import numpy as np
import trimesh

import malecns
from export_circuit import OUT as CIRCUIT
from export_for_sim import TO_MUJOCO

SHELL_URL = f"{malecns.BUCKET.replace('/v1.0', '')}/rois/brain-shell-v2.2/mesh/brain-shell.ngmesh"

NM_MM = 1e-6  # нанометры -> миллиметры
VOXEL_NM = 6000.0  # шаг сетки, по которой пересобираем поверхность
KEEP_FACES = 0.4  # сколько треугольников оставить после прорежения

DATA = Path(__file__).parent / "data"
OUT = DATA / "brain_shell.obj"


def read_ngmesh(path):
    """Формат neuroglancer legacy mesh: число вершин, вершины, треугольники."""

    raw = path.read_bytes()

    count = int(np.frombuffer(raw, dtype="<u4", count=1)[0])

    vertices = np.frombuffer(raw, dtype="<f4", count=3 * count, offset=4)
    faces = np.frombuffer(raw, dtype="<u4", offset=4 + 12 * count)

    return vertices.reshape(-1, 3).astype(float), faces.reshape(-1, 3).astype(np.int64)


def main():

    if not CIRCUIT.exists():
        raise SystemExit("сначала ./run-brain.sh brain/export_circuit.py")

    center = np.load(CIRCUIT)["center"]

    path = malecns._download(SHELL_URL, DATA / "brain-shell.ngmesh")

    vertices, faces = read_ngmesh(path)

    print(f"оболочка: {len(vertices)} вершин, {len(faces)} треугольников")

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # Меш собран из кусков-чанков, вершины на швах не общие, поэтому обычное
    # прорежение почти ничего не даёт. Пересобираем поверхность по сетке.
    voxels = mesh.voxelized(pitch=VOXEL_NM).fill()

    mesh = voxels.marching_cubes
    # marching_cubes отдаёт меш в номерах вокселей, возвращаем в нанометры.
    mesh.apply_transform(voxels.transform)

    simple = fast_simplification.simplify(
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.int32),
        target_reduction=1.0 - KEEP_FACES,
    )

    mesh = trimesh.Trimesh(simple[0], simple[1], process=True)
    trimesh.smoothing.filter_taubin(mesh, iterations=8)

    print(f"после пересборки: {len(mesh.faces)} треугольников")

    # В миллиметры, в оси MuJoCo, в общий с нейронами центр.
    mesh.vertices = mesh.vertices * NM_MM @ TO_MUJOCO.T - center

    mesh.export(OUT)

    size = mesh.vertices.max(0) - mesh.vertices.min(0)
    print(f"габариты: {size.round(3)} мм")
    print("сохранил:", OUT)


if __name__ == "__main__":
    main()
