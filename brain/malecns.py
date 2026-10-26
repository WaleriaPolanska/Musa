"""Общие куски для скриптов по коннектому maleCNS v1.0.

Данные лежат в публичном бакете Janelia, токен для них не нужен.
"""

from pathlib import Path

import pandas as pd
import requests

VERSION = "v1.0"

BUCKET = f"https://storage.googleapis.com/flyem-male-cns/{VERSION}"

ANNOTATIONS_URL = (
    f"{BUCKET}/connectome-data/flat-connectome/"
    f"body-annotations-male-cns-{VERSION}-minconf-0.5.feather"
)

SWC_URL = f"{BUCKET}/segmentation/skeletons-malecns/skeletons-swc/{{body_id}}.swc"

DATA = Path(__file__).parent / "data"


def _download(url: str, path: Path) -> Path:
    """Качает файл один раз и кладёт рядом со скриптами, в brain/data."""

    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)

    print(f"качаю {url.rsplit('/', 1)[-1]} ...")

    response = requests.get(url, timeout=600)
    response.raise_for_status()

    path.write_bytes(response.content)

    return path


def annotations() -> pd.DataFrame:
    """Таблица на 211 577 нейронов: bodyId, type, superclass, сторона тела и т.д."""

    return pd.read_feather(_download(ANNOTATIONS_URL, DATA / "body-annotations.feather"))


def swc_path(body_id: int) -> Path:
    """Путь к скелету нейрона, скачивает при первом обращении."""

    return _download(SWC_URL.format(body_id=body_id), DATA / "swc" / f"{body_id}.swc")


def find(pattern: str, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Нейроны, у которых type или instance содержит pattern (без учёта регистра)."""

    df = annotations() if df is None else df

    mask = df["type"].str.contains(pattern, case=False, na=False) | df[
        "instance"
    ].str.contains(pattern, case=False, na=False)

    return df.loc[mask]
