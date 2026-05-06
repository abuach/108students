# CS 108 — Week 6 Lab: Into the Third Dimension with Vispy

## Overview

Last week you built 2D worlds in Pygame — sprites, surfaces, pixel grids. This week you step through the screen. We're going from flatland into 3D space using **Vispy**, a Python library that talks directly to your GPU to render real-time 3D graphics.

You won't be writing a 3D engine. You'll be *observing* one, tuning it, and making it yours.

By the end of this lab you will have produced five 3D scenes, each demonstrating a different idea: parametric geometry, particle systems, simulation, generative art, and data visualization. All five are visually impressive with surprisingly little code.

---

## Setup

```bash
uv add vispy pyopengl pyqt6
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

canvas = scene.SceneCanvas(title='My Scene', bgcolor='#FFFFFF', size=(900, 600), show=True)
view = canvas.central_widget.add_view()
view.camera = scene.TurntableCamera(fov=50, distance=5, elevation=20)

# --- your scene goes here ---

if __name__ == '__main__':
    app.run()
```

`TurntableCamera` gives you a free-orbit camera out of the box: **click-drag to rotate, scroll to zoom**. It is available in every model.

---

## Model 1 — Parametric Surface: The Torus

A **parametric surface** is defined by equations that map two angles to a point in 3D space. The torus (donut shape) is one of the most elegant: every point is determined by two angles, `u` (around the ring) and `v` (around the tube).

```python
import numpy as np
from vispy import app, scene

canvas = scene.SceneCanvas(title='3D Scene', bgcolor='#FFFFFF', size=(900, 600), show=True)
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
1. Change `R` to `0.5` and `r` to `1.5` (swap them). Describe what happens to the shape. What does this tell you about the relationship between the two radii?
2. Replace with the below version. How does the density pattern change?

```python
import numpy as np
from vispy import app, scene

canvas = scene.SceneCanvas(title='3D Scene', bgcolor='#FFFFFF', size=(900, 600), show=True)
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
# --- Torus of colored points (structured grid instead of random) ---
n_u = 120
n_v = 80

u_vals = np.linspace(0, 2 * np.pi, n_u)
v_vals = np.linspace(0, 2 * np.pi, n_v)

u, v = np.meshgrid(u_vals, v_vals)
u = u.ravel()
v = v.ravel()

R, r = 1.5, 0.5

x = (R + r * np.cos(v)) * np.cos(u)
y = (R + r * np.cos(v)) * np.sin(u)
z = r * np.sin(v)

torus_pts = np.column_stack([x, y, z])

# Color by angle around the ring
hue = (u / (2 * np.pi))

n = u.size  # IMPORTANT: matches meshgrid output

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

---

## Model 2 — Particle System: Nebula

A **particle system** represents a phenomenon as thousands of individual points, each with its own position, velocity, and color. Here we simulate a nebula: a cloud of particles drifting outward from a central origin, colored by speed.

```python
from vispy import app, scene
import numpy as np

canvas = scene.SceneCanvas(
    title='Nebula',
    bgcolor='#FFFFFF',
    size=(900, 600),
    show=True
)

view = canvas.central_widget.add_view()
view.camera = scene.TurntableCamera(fov=60, distance=8, elevation=15)

N = 4000

pos = (np.random.randn(N, 3) * 0.1).astype(np.float32)
vel = (np.random.randn(N, 3) * 0.03).astype(np.float32)  # faster

# Color by speed
speed = np.linalg.norm(vel, axis=1)
speed_norm = (speed - speed.min()) / (speed.max() - speed.min())

colors = np.zeros((N, 4), dtype=np.float32)
colors[:, 0] = speed_norm
colors[:, 2] = 1 - speed_norm
colors[:, 1] = 0.2
colors[:, 3] = 0.7

markers = scene.visuals.Markers(
    pos=pos,
    size=3,
    face_color=colors,
    edge_width=0,
    parent=view.scene
)

def on_timer(event):
    global pos

    # move particles
    pos += vel

    # gentle swirl (this makes it feel like a nebula)
    theta = 0.01
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x = pos[:, 0].copy()
    y = pos[:, 1].copy()
    pos[:, 0] = cos_t * x - sin_t * y
    pos[:, 1] = sin_t * x + cos_t * y

    # wrap
    mask = np.abs(pos) > 5
    pos[mask] *= -0.5

    # update GPU
    markers.set_data(pos=pos, face_color=colors, size=3)

timer = app.Timer(1/60, connect=on_timer, start=True)

if __name__ == '__main__':
    app.run()
```

### [OBSERVE]
3. Change `vel` to `np.random.randn(N, 3) * 0.005`. How does this change the feel of the simulation?
4. Comment out the `pos[mask] *= -0.5` line. What happens over time? What does this line do?

---

## Model 3 — Simulation: 3D Reaction-Diffusion Field

You've seen Gray-Scott in 2D. Now we'll visualize a 3D **scalar field** — a volume of values — as a point cloud, where each point's color and opacity encodes the value at that location. We'll use a simpler simulation here (a 3D Gaussian that pulses) to keep the focus on the visualization technique.

```python
from vispy import app, scene
import numpy as np

canvas = scene.SceneCanvas(
    title='Animated Scalar Field',
    bgcolor='#FFFFFF',
    size=(900, 600),
    show=True
)

view = canvas.central_widget.add_view()
view.camera = scene.TurntableCamera(fov=55, distance=7, elevation=25)

# --- Grid ---
res = 30
lin = np.linspace(-2, 2, res)
gx, gy, gz = np.meshgrid(lin, lin, lin)
pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()]).astype(np.float32)

# --- Field ---
def field_value(t):
    r2 = gx**2 + gy**2 + gz**2

    # BIGGER pulsation so it's visible
    sigma = 0.4 + 0.5 * np.sin(t * 1.5)

    blob = np.exp(-r2 / (2 * sigma**2))

    # add wave interference (this makes motion obvious)
    wave = 0.5 * np.sin(3*gx + t) * np.sin(3*gy - t)

    return (blob + wave).ravel().astype(np.float32)

def make_colors(vals):
    v = (vals - vals.min()) / (vals.max() - vals.min() + 1e-6)

    colors = np.zeros((len(v), 4), dtype=np.float32)
    colors[:, 0] = v
    colors[:, 1] = 0.2 * v
    colors[:, 2] = 1 - v
    colors[:, 3] = 0.15 + 0.85 * v  # ensure visibility

    return colors

vals = field_value(0)

markers = scene.visuals.Markers(
    pos=pts,
    size=4,
    face_color=make_colors(vals),
    edge_width=0,
    parent=view.scene
)

# --- Animation ---
t = 0.0

def on_timer(event):
    global t

    t += 0.03

    vals = field_value(t)

    markers.set_data(
        pos=pts,
        face_color=make_colors(vals),
        size=4
    )

    # camera motion = instant "this is alive"
    view.camera.azimuth += 0.2

    canvas.update()

timer = app.Timer(1/60, connect=on_timer, start=True)

if __name__ == '__main__':
    app.run()
```

### [OBSERVE]
5. Change `res` from `30` to `15`, then to `50`. What are the tradeoffs?


---

## Model 4 — Generative Art: Lissajous Ribbon

**Lissajous curves** are formed by combining two or three sinusoids with different frequencies. They're a classic intersection of math and art. Here we extend them into 3D and give each point a color based on its phase — creating a glowing ribbon in space.

```python
from vispy import app, scene
from vispy.visuals import transforms
import numpy as np

canvas = scene.SceneCanvas(
    title='Nebula Ribbon (Volumetric)',
    bgcolor='#FFFFFF',
    size=(1000, 700),
    show=True
)

view = canvas.central_widget.add_view()
view.camera = scene.TurntableCamera(fov=45, distance=6, elevation=25)

# -----------------------------
# Base parametric curve
# -----------------------------
n = 4000
t = np.linspace(0, 2 * np.pi, n)

# A, B, C
a = 3 
b = 2
c = 5

base = np.column_stack([
    np.sin(a * t + np.pi/4),
    np.sin(b * t),
    np.sin(c * t)
]).astype(np.float32)

# -----------------------------
# Ribbon object (line strip)
# -----------------------------
line = scene.visuals.Line(
    base,
    color=(0.4, 0.6, 1.0, 0.0),  # we will override via colors
    width=2,
    method='gl',
    parent=view.scene
)

# -----------------------------
# Color field (nebula gradient)
# -----------------------------
c = (np.sin(t) + 1) / 2
colors = np.zeros((n, 4), dtype=np.float32)
colors[:, 0] = c**2.5
colors[:, 1] = 0.3 + 0.4 * (1 - c)
colors[:, 2] = 0.8 + 0.2 * c
colors[:, 3] = 0.9

line.set_data(pos=base, color=colors)

# -----------------------------
# Secondary fog layer (points)
# -----------------------------
fog_n = 6000
fog = np.random.randn(fog_n, 3).astype(np.float32) * 1.5

fog_colors = np.zeros((fog_n, 4), dtype=np.float32)
fog_colors[:, 2] = 1.0
fog_colors[:, 3] = 0.03  # very faint fog

fog_markers = scene.visuals.Markers(
    pos=fog,
    size=2,
    face_color=fog_colors,
    edge_width=0,
    parent=view.scene
)

fog_markers.set_gl_state(
    'translucent',
    blend=True,
    depth_test=False
)

# -----------------------------
# Motion state
# -----------------------------
phase = 0.0

def deform(t, phase):
    warp = 0.25 * np.sin(6 * t + phase)

    pts = np.empty_like(base)
    pts[:, 0] = base[:, 0] + warp * np.cos(3 * t + phase)
    pts[:, 1] = base[:, 1] + warp * np.sin(2 * t - phase)
    pts[:, 2] = base[:, 2] + warp * np.sin(4 * t + phase * 0.5)

    return pts

# -----------------------------
# Animation loop
# -----------------------------
def on_timer(event):
    global phase

    phase += 0.02

    pts = deform(t, phase)

    # -------------------------
    # Ribbon glow dynamics
    # -------------------------
    pulse = 0.6 + 0.4 * np.sin(phase * 2.0)

    glow = colors.copy()
    glow[:, 3] = 0.3 + 0.7 * pulse  # alpha pulse

    line.set_data(pos=pts, color=glow, width=2 + 2 * pulse)

    # -------------------------
    # Fog drift (slow motion field)
    # -------------------------
    fog[:, 0] += 0.001 * np.sin(phase)
    fog[:, 1] += 0.001 * np.cos(phase * 0.7)
    fog[:, 2] += 0.001 * np.sin(phase * 0.3)

    fog_markers.set_data(pos=fog, face_color=fog_colors)

    # -------------------------
    # Camera motion (depth cue)
    # -------------------------
    view.camera.azimuth += 0.15
    view.camera.elevation = 25 + 8 * np.sin(phase * 0.3)

    canvas.update()

timer = app.Timer(1/60, connect=on_timer, start=True)

if __name__ == '__main__':
    # IMPORTANT: enable additive glow blending
    line.set_gl_state('translucent', blend=True, depth_test=False)

    app.run()
```

### [OBSERVE]
6. Change `a`, `b`, `c` to `3, 4, 5` then to `2, 3, 7`. How do the ratios between the three frequencies affect the shape?
7. The curve is drawn as individual points. Change `size` from `2` to `6`. What does the ribbon look like now?

---

## Model 5 — Quantum Star Bloom (3D Data Visualization)

Data visualization in 3D maps abstract values into spatial structure and visual properties. In this model, we extend a classic star cluster into a Quantum Star Bloom, where stars emerge from an evolving scalar field that behaves like a living medium.

Each point in space is influenced by:

- a quantum interference field (structure)
- a radial density potential (cluster formation)
- a temporal excitation wave (animation)

Stars are no longer static samples — they are responses of the field itself.

```python
from vispy import app, scene
import numpy as np

canvas = scene.SceneCanvas(
    title='Quantum Field Bloom',
    bgcolor='#FFFFFF',
    size=(1000, 700),
    show=True
)

view = canvas.central_widget.add_view()
view.camera = scene.TurntableCamera(fov=55, distance=6, elevation=30)

# -------------------------------------------------
# GRID POINT CLOUD (dense 3D field)
# -------------------------------------------------
res = 55
lin = np.linspace(-2, 2, res)

x, y, z = np.meshgrid(lin, lin, lin)
pos0 = np.column_stack([x.ravel(), y.ravel(), z.ravel()]).astype(np.float32)

N = len(pos0)

# -------------------------------------------------
# VISUALS
# -------------------------------------------------
colors = np.zeros((N, 4), dtype=np.float32)

markers = scene.visuals.Markers(
    pos=pos0,
    size=2,
    face_color=colors,
    edge_width=0,
    parent=view.scene
)

markers.set_gl_state('translucent', blend=True, depth_test=False)

# -------------------------------------------------
# TIME STATE
# -------------------------------------------------
t = 0.0

# -------------------------------------------------
# FIELD FUNCTION (this is the "physics")
# -------------------------------------------------
def field(p, t):
    x, y, z = p[:, 0], p[:, 1], p[:, 2]

    r = np.sqrt(x**2 + y**2 + z**2)

    # layered wave interference (key effect)
    f = (
        np.sin(3*x + t) +
        np.cos(3*y - t * 0.8) +
        np.sin(3*z + t * 1.2)
    )

    # radial collapse/expansion wave
    f += np.sin(r * 5 - t * 2)

    # vortex twist
    f += 0.5 * np.sin(x*y + t)

    return f

# -------------------------------------------------
# COLOR MAPPING (energy visualization)
# -------------------------------------------------
def colorize(f):
    f = (f - f.min()) / (f.max() - f.min() + 1e-6)

    c = np.zeros((N, 4), dtype=np.float32)

    c[:, 0] = f**2.2
    c[:, 1] = np.sin(f * 3.14)**2
    c[:, 2] = 1 - f

    c[:, 3] = 0.15 + f * 0.85

    return np.clip(c, 0, 1)

# -------------------------------------------------
# ANIMATION
# -------------------------------------------------
def on_timer(event):
    global t
    t += 0.03

    f = field(pos0, t)

    # deform space itself (this is what makes it “alive”)
    deformation = 0.3 * np.sin(f * 2 + t)

    pos = pos0.copy()
    pos[:, 0] += deformation * np.sin(t + pos[:, 1])
    pos[:, 1] += deformation * np.cos(t + pos[:, 2])
    pos[:, 2] += deformation * np.sin(t + pos[:, 0])

    markers.set_data(
        pos=pos,
        face_color=colorize(f),
        size=2
    )

    # camera slowly drifts through field
    view.camera.azimuth += 0.2
    view.camera.elevation = 30 + 10 * np.sin(t * 0.5)

    canvas.update()

timer = app.Timer(1/60, connect=on_timer, start=True)

if __name__ == '__main__':
    app.run()
```

### [OBSERVE]

8. Orbit and rotate the structure slowly. Where do the brightest regions form? Do they correspond to fixed “clusters,” or do they appear and disappear over time?
9. Watch how regions of high color intensity evolve. Do they remain stable, drift, or reorganize into new structures?


---


*CS 108 — The Art and Practice of Computer Science | Walla Walla University*