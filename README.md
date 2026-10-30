# Black hole imaging: ray tracing in Schwarzschild and Kerr spacetimes

> ⚠️ Work in progress: provisional README. This project is based on my bachelor's thesis in Physics (University of Trento, written in Italian).

A Python simulation of how a black hole looks to an observer at a finite distance, built with **backward ray tracing**. For each pixel of a virtual camera, the photon's null geodesic is integrated backwards in time to find out whether it falls into the horizon (the shadow) or which direction of the sky it comes from (gravitational lensing).

## Features

- **Schwarzschild**: integrates the photon orbit equation (Binet equation)

  $$\frac{d^2u}{d\phi^2} = -u + 3Mu^2, \qquad u = 1/r$$

  The deflection angle is tabulated as a function of the impact parameter $b$ and interpolated over all pixels. The change of variable $\chi = \ln(b/b_c - 1)$ resolves the region near the critical impact parameter $b_c = 3\sqrt{3}\,M$. The numerical value of $b_c$ is checked by bisection.
- **Kerr**: integrates null geodesics in Mino time, starting from the constants of motion $b$ and $Q$ (Carter constant), which are computed pixel by pixel.
- **Rendering**: maps the asymptotic photon directions onto an equirectangular sky image and compares the sky with and without the black hole.

## Repository structure

| File | Contents |
|---|---|
| `buchineri.py` | geodesic integrators, camera screen setup, image rendering functions |
| `immaginibuchineri.ipynb` | notebook that sets the parameters and generates the images |

## Requirements

Python 3 with `numpy`, `scipy`, `matplotlib` and `pillow`:

```bash
pip install numpy scipy matplotlib pillow
```

## Usage

1. Download an equirectangular sky image (for example a Milky Way panorama) and place it in the project folder. Background images are **not included** in the repository because of their size and licensing.
2. Open `immaginibuchineri.ipynb`, set the file name and the parameters (mass $M$, spin $a$, observer distance $r_0$, field of view, resolution $N$), then run the cells.

**Notes:**
- The Kerr case integrates one geodesic per pixel, so it is slow: $N = 1000$ takes about 2.5 hours.
- The notebook calls `caffeinate` (macOS only) to keep the machine awake during long runs.

## Author

Pietro Abeccara, bachelor's thesis in Physics, University of Trento.
