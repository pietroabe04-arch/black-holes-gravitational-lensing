# Black hole imaging: ray tracing in Schwarzschild and Kerr spacetimes

> Work based on my bachelor's thesis in Physics at the University of Trento, Italy (written in Italian).

A Python simulation of how a black hole looks to an observer at a finite distance $r_0$ looking form an angle $\theta_0$, built with **backward ray tracing**. For each pixel of a virtual camera, the photon's null geodesic is integrated backwards in time to find out whether it falls into the horizon (the shadow) or which direction of the sky it comes from (gravitational lensing).

## Output images

Schwarzschild black hole ($N=5000$, $FOV=70°$, $r_0=30M$):
<img width="977" height="506" alt="output" src="https://github.com/user-attachments/assets/69a9fc23-4c65-41c8-a4e3-5f172d99a594" />
<img width="977" height="506" alt="output2" src="https://github.com/user-attachments/assets/1d1fa362-8534-42fd-a3ea-f65e5b0d2e7f" />

Kerr black hole ($N=1000$, $FOV=70°$, $r_0=30M$, $\theta_0=\pi/2$, $a=0.99$):
<img width="977" height="506" alt="output4" src="https://github.com/user-attachments/assets/a64fe258-e077-45ae-8984-47b056e13d5a" />
<img width="977" height="506" alt="output3" src="https://github.com/user-attachments/assets/24e3ec8b-ca3c-4bc1-9420-ebbc759171d3" />


## Features

- **Schwarzschild**: integrates the photon orbit equation (Binet equation) $d^2u/d\phi^2 = -u + 3Mu^2$ with $u = 1/r$.
  The only relevant parameter is the impact parameter $b$ which determines into the initial condition of the photon associated to each pixel of the camera.
  The deflection angle is tabulated as a function of the impact parameter $b$ and interpolated over all pixels. The change of variable $\chi = \ln(b/b_c - 1)$ resolves the region near the critical impact parameter $b_c = 3\sqrt{3}\,M$. The numerical value of $b_c$ is checked by bisection. A conserved quantity is check on a bunch of geodesic.
- **Kerr**: integrates null geodesics in Mino time, starting from the constants of motion $b$ and $Q$ (Carter constant), which are computed pixel by pixel. Interpolation cannot be implement in this case, therefore it is much slower to execute than the Schwarzschild case. The impact parameters on the equatorial plane are checked to be equal to $b_c$ in the limit $a \to 0$, furthermore a conserved quantity is check to be conserved along the geodesic.
- **Rendering**: maps the asymptotic photon directions onto an equirectangular sky image and compares the sky with and without the black hole. The variables are the number of pixel per side $N$(⚠️ it must be an even number), the field of view $FOV$, the position of the camera $(r_0,\theta_0)$, the spin $a \in (-1,1)$ and the mass of the black hole $M$ (which is actually irrelevant because every physical quantity is calculated in unit of $M$).
  

## Repository structure

| File | Contents |
|---|---|
| `buchineri.py` | geodesic integrators, camera screen setup, image rendering functions |
| `immaginibuchineri.ipynb` | notebook that sets the parameters, generates the images and check the quality of the integration|

## Requirements

Python 3 with `numpy`, `scipy`, `matplotlib` and `pillow`

## Usage

1. Download an equirectangular sky image (for example a Milky Way panorama or a a checkered background) and place it in the project folder. Background images are **not included** in the repository because of their size and licensing.
2. Open `immaginibuchineri.ipynb`, set the file name and the parameters (mass $M$, spin $a$, observer distance $r_0$, field of view, resolution $N$), then run the cells.

**Notes:**
- The Kerr case integrates one geodesic per pixel, so it is slow: $N = 1000$ takes about 2.5 hours on my MacBook Air M4.
- The notebook calls `caffeinate` (macOS only) to keep the machine awake during long runs.

## Author

Pietro Abeccara, bachelor's thesis in Physics, University of Trento.
