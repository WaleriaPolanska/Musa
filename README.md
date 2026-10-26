# Musa — a fruit fly driven by a real connectome

Interactive simulation of a fruit fly (*Drosophila*) built on **NeuroMechFly** (flygym 2.1) and **MuJoCo**. The fly walks on six legs, and its turning is driven by **real neural circuits from the maleCNS v1.0 connectome** (Janelia FlyEM): it sees light through a visual pathway, smells food through an olfactory chain and learns in the mushroom body.

Side project in Python, alongside my main work in C# / .NET.

## Highlights

- **Real circuit, real weights.** For example the escape chain LPLC2 → descending neurons (DNp01, DNp103, DNp04, DNg40) uses synaptic weights computed from the connectome.
- **Data pipeline.** `brain/` reads the public maleCNS release (≈500 MB of connection weights in Feather format) with pandas, queries the neuPrint API and exports compact circuits (`brain/data/*.npz`) for the simulation.
- **Interactive viewer.** 3D room with the fly, a 3D brain map with neuron activity highlighting, and live sliders for synapse strength.

## Structure

| Folder | What is inside |
|---|---|
| `musa/` | the simulation and the viewer (`window.py`, `musa_chain.py`, `musa_synapses.py`…) |
| `brain/` | connectome scripts: download, query, export circuits |
| `brain/data/` | small exported circuits used by the simulation |
| `experiments/` | earlier prototypes (walking, light, sensors, CPG) |
| `docs/ARCHITECTURE.md` | how the code is organised (in Russian) |

## Run

```bash
# simulation environment (Python 3.12)
python3.12 -m venv flygym21
./flygym21/bin/pip install -r requirements.txt
./run.sh                       # opens the viewer
./run.sh musa/musa_synapses.py # escape circuit LPLC2 → descending neurons

# connectome tools (separate environment)
python3 -m venv brainvenv
./brainvenv/bin/pip install -r requirements-brain.txt
./run-brain.sh brain/find.py LPLC2
```

On macOS, scripts that use the built-in MuJoCo viewer must run under `mjpython`; `run.sh` picks the right interpreter.

Re-exporting circuits needs the full connectome files from the public bucket `gs://flyem-male-cns/v1.0`. Queries for connections need a neuPrint token in the `NEUPRINT_APPLICATION_CREDENTIALS` environment variable; no token is stored in this repository.

## Data and credits

- Connectome: **maleCNS v1.0, Janelia FlyEM** (public release). Please cite the maleCNS publication when reusing the derived data in `brain/data/`.
- Body model and locomotion: **NeuroMechFly / flygym**.
- Physics: **MuJoCo**.
