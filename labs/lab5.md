# CS 108 — Week 6 Lab: Into the Third Dimension with Vispy

## Overview

Last week you built 2D worlds in Pygame — sprites, surfaces, pixel grids. This week you step through the screen. We're going from flatland into 3D space using **Vispy**, a Python library that talks directly to your GPU to render real-time 3D graphics.

You won't be writing a 3D engine. You'll be *observing* one, tuning it, and making it yours.

By the end of this lab you will have produced five 3D scenes, each demonstrating a different idea: parametric geometry, particle systems, simulation, generative art, and data visualization. All five are visually impressive with surprisingly little code.

---

## Setup

```bash
pip install vispy pyopengl pyqt6
```

Verify:
```python
import vispy
print(vispy.__version__)
```

Every model in this lab uses the same scaffold:

```python
from vispy import app, scene
import numpy as np

canvas = scene.SceneCanvas(title='My Scene', bgcolor='#050510', size=(900, 600), show=True)
view = canvas.central_widget.add_view()
view.camera = scene.TurntableCamera(fov=50, distance=5, elevation=20)

# --- your scene goes here ---

if __name__ == '__main__':
    app.run()
```

`TurntableCamera` gives you a free-orbit camera out of the box: **click-drag to rotate, scroll to zoom**. Use it on every model.

---

## Model 1 — Parametric Surface: The Torus

A **parametric surface** is defined by equations that map two angles to a point in 3D space. The torus (donut shape) is one of the most elegant: every point is determined by two angles, `u` (around the ring) and `v` (around the tube).

```python
import numpy as np
from vispy import app, scene

canvas = scene.SceneCanvas(title='3D Scene', bgcolor='#050510', size=(900, 600), show=True)
view = canvas.central_widget.add_view()
view.camera = scene.TurntableCamera(fov=50, distance=6, elevation=20)

# --- Starfield ---
n_stars = 2000
stars_xyz = (np.random.rand(n_stars, 3) - 0.5) * 30
star_sizes = np.random.uniform(1, 3, n_stars)
scene.visuals.Markers(pos=stars_xyz, size=star_sizes,
                      face_color=(1, 1, 1, 0.6), edge_width=0,
                      parent=view.scene)

# --- Torus of colored points ---
n = 8000
u = np.random.uniform(0, 2 * np.pi, n)  # around the tube
v = np.random.uniform(0, 2 * np.pi, n)  # around the ring
R, r = 1.5, 0.5                          # major / minor radius

x = (R + r * np.cos(v)) * np.cos(u)
y = (R + r * np.cos(v)) * np.sin(u)
z = r * np.sin(v)
torus_pts = np.column_stack([x, y, z])

# Color by angle around the ring
hue = (u / (2 * np.pi))
colors = np.zeros((n, 4))
colors[:, 0] = np.abs(np.sin(hue * np.pi))        # R
colors[:, 1] = np.abs(np.sin(hue * np.pi + 2.1))  # G
colors[:, 2] = np.abs(np.sin(hue * np.pi + 4.2))  # B
colors[:, 3] = 0.85

torus = scene.visuals.Markers(pos=torus_pts, size=2.5,
                               face_color=colors, edge_width=0,
                               parent=view.scene)

# --- Rotation ---
def on_timer(event):
    torus.transform = scene.transforms.MatrixTransform()
    angle = event.elapsed * 30  # degrees/sec
    torus.transform.rotate(angle, (0, 0, 1))
    canvas.update()

timer = app.Timer(interval=1/60, connect=on_timer, start=True)

if __name__ == '__main__':
    app.run()
```

### [OBSERVE]
1. Run the scene. Drag to orbit around the torus. What does the color pattern reveal about how `u` maps onto the surface?
2. Change `R` to `0.5` and `r` to `1.5` (swap them). Describe what happens to the shape. What does this tell you about the relationship between the two radii?
3. Replace `np.random.uniform` with `np.linspace` for both `u` and `v` (use `np.meshgrid` or just two `linspace` arrays of length ~100 each, flattened). How does the density pattern change?

### [REFLECT]
4. The color is computed from `u` but not `v`. What would happen if you added `v` into the color formula? Try it.
5. A sphere is also a parametric surface: `x = sin(v)cos(u)`, `y = sin(v)sin(u)`, `z = cos(v)`. Modify the code to render a sphere instead. What values of `u` and `v` do you need?

### [SYNTHESIS]
6. Create your own parametric surface. Two options: (a) look up the equations for a **Klein bottle** or **Möbius strip** and implement them, or (b) invent your own by modifying the torus equations in a creative way. Screenshot your result and include it in your portfolio post.

---

## Model 2 — Particle System: Nebula

A **particle system** represents a phenomenon as thousands of individual points, each with its own position, velocity, and color. Here we simulate a nebula: a cloud of particles drifting outward from a central origin, colored by speed.

```python
from vispy import app, scene
import numpy as np

canvas = scene.SceneCanvas(title='Nebula', bgcolor='#000008', size=(900, 600), show=True)
view = canvas.central_widget.add_view()
view.camera = scene.TurntableCamera(fov=60, distance=8, elevation=15)

N = 5000
pos = (np.random.randn(N, 3) * 0.1).astype(np.float32)
vel = (np.random.randn(N, 3) * 0.015).astype(np.float32)

# Color by initial speed
speed = np.linalg.norm(vel, axis=1)
speed_norm = (speed - speed.min()) / (speed.max() - speed.min())
colors = np.zeros((N, 4), dtype=np.float32)
colors[:, 0] = speed_norm               # fast = red
colors[:, 2] = 1 - speed_norm           # slow = blue
colors[:, 1] = 0.2
colors[:, 3] = 0.7

markers = scene.visuals.Markers(pos=pos, size=2, face_color=colors,
                                 edge_width=0, parent=view.scene)

def on_timer(event):
    global pos
    pos += vel
    # wrap particles that drift too far
    pos[np.abs(pos) > 5] *= -0.5
    markers.set_data(pos=pos, face_color=colors, size=2, edge_width=0)
    canvas.update()

app.Timer(interval=1/60, connect=on_timer, start=True)

if __name__ == '__main__':
    app.run()
```

### [OBSERVE]
1. Watch the particles for 10 seconds. Describe the motion. Where do the blue particles tend to be relative to the red ones, and why?
2. Change `vel` to `np.random.randn(N, 3) * 0.005`. How does this change the feel of the simulation?
3. Comment out the `pos[np.abs(pos) > 5] *= -0.5` line. What happens over time? What does this line do?

### [REFLECT]
4. Currently all particles have constant velocity. Add a gravity effect: subtract a small value from `vel[:, 1]` each frame (e.g. `0.0001`). What does the nebula look like after 5 seconds?
5. The colors are set once at startup based on initial speed. Recompute the colors each frame based on *current* speed. How does the visual change?

### [SYNTHESIS]
6. Design a particle system that tells a visual story — an explosion, a black hole pulling particles inward, a flock, or something of your own invention. The key requirement: the motion must be driven by a rule you can describe in one sentence. Write that sentence as a comment at the top of your code.

---

## Model 3 — Simulation: 3D Reaction-Diffusion Field

You've seen Gray-Scott in 2D. Now we'll visualize a 3D **scalar field** — a volume of values — as a point cloud, where each point's color and opacity encodes the value at that location. We'll use a simpler simulation here (a 3D Gaussian that pulses) to keep the focus on the visualization technique.

```python
from vispy import app, scene
import numpy as np

canvas = scene.SceneCanvas(title='Scalar Field', bgcolor='#020008', size=(900, 600), show=True)
view = canvas.central_widget.add_view()
view.camera = scene.TurntableCamera(fov=55, distance=7, elevation=25)

# Build a 3D grid of points
res = 30
lin = np.linspace(-2, 2, res)
gx, gy, gz = np.meshgrid(lin, lin, lin)
pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()]).astype(np.float32)

def field_value(t):
    # Pulsing Gaussian blob
    r2 = gx**2 + gy**2 + gz**2
    sigma = 0.5 + 0.3 * np.sin(t)
    return np.exp(-r2 / (2 * sigma**2)).ravel().astype(np.float32)

def make_colors(vals):
    colors = np.zeros((len(vals), 4), dtype=np.float32)
    colors[:, 0] = vals                    # R: high value = bright
    colors[:, 2] = 1 - vals               # B: low value = blue
    colors[:, 3] = vals * 0.9             # alpha = value (sparse edges)
    return colors

vals = field_value(0)
markers = scene.visuals.Markers(pos=pts, size=3,
                                 face_color=make_colors(vals),
                                 edge_width=0, parent=view.scene)

def on_timer(event):
    vals = field_value(event.elapsed)
    markers.set_data(pos=pts, face_color=make_colors(vals),
                     size=3, edge_width=0)
    canvas.update()

app.Timer(interval=1/60, connect=on_timer, start=True)

if __name__ == '__main__':
    app.run()
```

### [OBSERVE]
1. Watch the field pulse. What happens to the blue points as the Gaussian grows? What happens to the red points?
2. The `alpha` channel is set to `vals * 0.9`. This means points with low field values are nearly transparent. Temporarily set all alphas to `1.0`. How does the visualization change, and which version communicates the data better?
3. Change `res` from `30` to `15`, then to `50`. What are the tradeoffs?

### [REFLECT]
4. The Gaussian is symmetric (same width in x, y, z). Make it asymmetric: use different sigma values for each axis, e.g. `gx**2 / sx**2 + gy**2 / sy**2 + gz**2 / sz**2`. Make the sigmas pulse at different rates. Describe what you see.
5. This visualization maps a number (field value) to color and opacity. This technique is called **volume rendering**. Where else might this technique be useful? (Think: medical imaging, weather, physics simulations.)

### [SYNTHESIS]
6. Replace the Gaussian with two overlapping blobs whose centers orbit each other slowly. Color the regions of overlap differently (hint: compute each blob separately, then combine with `np.maximum` or addition). Screenshot the result.

---

## Model 4 — Generative Art: Lissajous Ribbon

**Lissajous curves** are formed by combining two or three sinusoids with different frequencies. They're a classic intersection of math and art. Here we extend them into 3D and give each point a color based on its phase — creating a glowing ribbon in space.

```python
from vispy import app, scene
import numpy as np

canvas = scene.SceneCanvas(title='Lissajous Ribbon', bgcolor='#000000', size=(900, 600), show=True)
view = canvas.central_widget.add_view()
view.camera = scene.TurntableCamera(fov=45, distance=5, elevation=30)

def make_lissajous(a=3, b=2, c=5, delta=np.pi/4, n=8000):
    t = np.linspace(0, 2 * np.pi, n)
    x = np.sin(a * t + delta)
    y = np.sin(b * t)
    z = np.sin(c * t)
    pts = np.column_stack([x, y, z]).astype(np.float32)

    colors = np.zeros((n, 4), dtype=np.float32)
    colors[:, 0] = (np.sin(t) + 1) / 2
    colors[:, 1] = (np.sin(t + 2.1) + 1) / 2
    colors[:, 2] = (np.sin(t + 4.2) + 1) / 2
    colors[:, 3] = 0.85
    return pts, colors

pts, colors = make_lissajous()
markers = scene.visuals.Markers(pos=pts, size=2, face_color=colors,
                                 edge_width=0, parent=view.scene)

phase = [0.0]

def on_timer(event):
    phase[0] += 0.015
    pts, colors = make_lissajous(delta=phase[0])
    markers.set_data(pos=pts, face_color=colors, size=2, edge_width=0)
    canvas.update()

app.Timer(interval=1/60, connect=on_timer, start=True)

if __name__ == '__main__':
    app.run()
```

### [OBSERVE]
1. Watch the ribbon morph over time. At what values of `delta` does it appear to "close" into a stable loop?
2. Change `a`, `b`, `c` to `3, 4, 5` then to `2, 3, 7`. How do the ratios between the three frequencies affect the shape?
3. The curve is drawn as individual points. Change `size` from `2` to `6`. What does the ribbon look like now?

### [REFLECT]
4. The color is based on parameter `t` (position along the curve). What would it look like to color by *speed* instead — i.e., how fast the curve is moving at each point? (Hint: compute `np.diff` on the positions and use the magnitude as color.)
5. Lissajous figures appear in oscilloscopes, signal processing, and music visualizers. Look up one real-world application and write 2–3 sentences explaining how the math connects.

### [SYNTHESIS]
6. Create a "Lissajous sculpture" — extend the curve into a ribbon by plotting multiple parallel offset copies (vary a small offset in x or y). Or animate the frequencies `a`, `b`, `c` themselves slowly over time. The goal is a piece you'd be proud to post. Screenshot and include it.

---

## Model 5 — 3D Data Visualization: Star Cluster

Data visualization in 3D means mapping real (or realistic) data onto spatial coordinates and visual properties. Here we generate a synthetic star cluster: each star has a 3D position, a temperature (which determines color), and a luminosity (which determines size).

```python
from vispy import app, scene
import numpy as np

canvas = scene.SceneCanvas(title='Star Cluster', bgcolor='#000005', size=(900, 600), show=True)
view = canvas.central_widget.add_view()
view.camera = scene.TurntableCamera(fov=60, distance=10, elevation=20)

np.random.seed(42)
N = 3000

# Positions: cluster core + sparse halo
core = np.random.randn(int(N * 0.7), 3) * 1.2
halo = np.random.randn(int(N * 0.3), 3) * 4.0
pos = np.vstack([core, halo]).astype(np.float32)

# Temperature: 3000K (red) to 30000K (blue-white)
temp = np.random.exponential(scale=0.3, size=len(pos))
temp = np.clip(temp, 0, 1)

# Color stars by temperature (blackbody approximation)
colors = np.zeros((len(pos), 4), dtype=np.float32)
colors[:, 0] = np.clip(1.0 - temp * 0.6, 0, 1)   # hot stars less red
colors[:, 1] = np.clip(0.5 - np.abs(temp - 0.5), 0, 1) * 1.5  # peak in mid-range
colors[:, 2] = np.clip(temp, 0, 1)                 # hot stars more blue
colors[:, 3] = np.clip(0.4 + temp * 0.6, 0, 1)    # hotter = brighter

# Size: luminosity (hotter stars are bigger)
sizes = (2 + temp * 5).astype(np.float32)

scene.visuals.Markers(pos=pos, size=sizes, face_color=colors,
                       edge_width=0, parent=view.scene)

# Mark the cluster center
scene.visuals.Markers(pos=np.array([[0, 0, 0]], dtype=np.float32),
                       size=12, face_color=(1, 1, 0.8, 1),
                       edge_width=0, parent=view.scene)

if __name__ == '__main__':
    app.run()
```

### [OBSERVE]
1. Orbit the cluster. Where are the blue-white stars concentrated relative to the red ones? Does this match what you'd expect given how `core` and `halo` are generated?
2. Three visual channels encode data: position, color, and size. List what each one represents in this simulation.
3. Change the `scale` in `np.random.exponential` from `0.3` to `0.7`. How does the population of the cluster change? What does this parameter control?

### [REFLECT]
4. Currently both core and halo stars share the same temperature distribution. Make core stars hotter on average (higher `temp` values) and halo stars cooler. What does this look like visually?
5. This is a *simulation* of data, not real data. What would be different — and harder — if you were plotting a real star catalog? (Think: what extra information would you need, and what decisions would you have to make?)

### [SYNTHESIS]
6. Add a second cluster at position `(6, 0, 0)` that is merging with the first — its stars are younger (bluer, bigger). Adjust the camera position to frame both clusters. Screenshot the result and write a two-sentence "artist's statement" about what your visualization shows.

---

## Submission

Your lab submission is a **portfolio post** containing:

1. **Five screenshots** — one per model, showing your final customized version (not the starter code unchanged).
2. **One paragraph per model** describing: what you changed, what you observed, and one thing that surprised you.
3. **Synthesis code** for at least two models (the ones you are most proud of), pasted or linked.

Your post should read like a creative journal entry, not a lab report. Write in first person. Include what broke, what confused you, and what delighted you.

---

## Going Further

These are not required, but if you want to push deeper:

- **Vispy `Line` visual** — connect your particle positions with lines for a network graph look
- **Custom GLSL shaders** — Vispy exposes raw shader code; try modifying a fragment shader to add glow
- **Real data** — load a CSV of 3D coordinates (e.g., from a protein structure database or NASA's star catalog) and plot it with Model 5's coloring approach
- **VR export** — export your point cloud to `.ply` format and load it in a WebXR viewer

---

*CS 108 — The Art and Practice of Computer Science | Walla Walla University*