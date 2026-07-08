# Three Body Interaction Simulation

A interactive **Three-Body Problem** simulation built with **Python** and **Taichi**, demonstrating chaotic orbital dynamics using real Newtonian gravity.

## Features

- Real Newtonian Gravity
- Leapfrog (Symplectic) Integration
- Four Famous Three-Body Configurations
- Figure-8 Orbit
- Lagrange Triangle
- Broucke–Henon Orbit
- Chaotic Orbit
- Yin–Yang Orbit
- 12,000 Massless Test Particles
- Gravity-Based Particle Coloring
- HDR Glow & Bloom Rendering
- Gravitational Potential Field Visualization
- Hill Sphere Visualization
- Interactive Mouse Gravity
- Particle Burst Generator
- Cinematic Trails & Tone Mapping
- GPU Accelerated with Taichi

## Controls

| Key                   | Action                       |
| --------------------- | ---------------------------- |
| **Left Mouse**  | Apply gravitational pull     |
| **Right Mouse** | Spawn particle burst         |
| **SPACE**       | Pause / Resume               |
| **T**           | Toggle trails                |
| **R**           | Reset simulation             |
| **F**           | Cycle through presets        |
| **+ / =**       | Increase simulation speed    |
| **-**           | Decrease simulation speed    |
| **1 / 2 / 3**   | Highlight body & Hill sphere |
| **ESC**         | Quit                         |

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
uv run --python 3.12 --with taichi demo.py
```

---

**Author:** Sushobhan Mishra
