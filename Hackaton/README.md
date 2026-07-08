
# Cosmic Symphony: Interactive Galaxy Collision Simulator

A interactive **galaxy collision simulator** built with **Python** and **Taichi**, featuring Newtonian gravity, spiral galaxies, and cinematic scientific visualization.

## Features

- Real Newtonian Gravity (Plummer Softening)
- Two Colliding Spiral Galaxies
- Supermassive Black Hole Dynamics
- Tidal Tails & Galaxy Mergers
- Energy-Based Star Colours
- Glow Halos, Bloom & Particle Trails
- Density-Based Nebula
- Cinematic Camera
- Procedural Milky Way & Dust Field
- Optimized for Integrated GPUs

## Controls

| Key             | Action                         |
| --------------- | ------------------------------ |
| **LMB**   | Attract stars                  |
| **RMB**   | Repel stars                    |
| **SPACE** | Pause / Resume                 |
| **R**     | Reset simulation               |
| **T**     | Toggle trails                  |
| **F**     | Toggle field lines             |
| **G**     | Toggle glow halos              |
| **N**     | Toggle nebula                  |
| **M**     | Toggle cinematic camera        |
| **H**     | Toggle Milky Way               |
| **J**     | Toggle parallax stars          |
| **K**     | Toggle dust particles          |
| **Y**     | Toggle shockwaves              |
| **U**     | Toggle tidal streamers         |
| **I**     | Toggle lens flares             |
| **C**     | Toggle cinematic color grading |
| **E**     | Toggle orbital energy colors   |
| **B**     | Toggle particle bloom          |
| **P**     | Performance mode               |
| **1**     | Spawn dwarf galaxy             |
| **+ / -** | Change simulation speed        |
| **Q**     | Quit                           |

## Built With

- Python 3.12
- Taichi 1.7.4
- NumPy

## Run

```bash
pip install uv
```

Then

```bash
uv run --python 3.12 --with taichi sim.py
```

---

**Author:** Sushobhan Mishra
