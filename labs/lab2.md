# CS 108 Lab — Generative Art & Emergent Systems

**Name:** _______________________________ **Date:** _____________

---

## Overview

In this lab you will run four generative art simulations, observe their behavior, and make
deliberate creative choices to produce portfolio-quality output. No new code needs to be
written from scratch — your job is to *observe, tweak, and reflect*.

**Install dependencies once:**
```bash
uv add numpy matplotlib
cd cs108 
mkdir art
cd art 
```

---

## Model 1: Reaction-Diffusion (Gray–Scott)

### The Code

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Grid size
N = 200

# Parameters (these control the pattern!)
Du, Dv = 0.16, 0.08
F, k = 0.060, 0.062

# Initialize chemicals
U = np.ones((N, N))
V = np.zeros((N, N))

# Seed in the center
r = 20
U[N//2 - r:N//2 + r, N//2 - r:N//2 + r] = 0.50
V[N//2 - r:N//2 + r, N//2 - r:N//2 + r] = 0.25

def laplacian(Z):
    return (
        -4 * Z
        + np.roll(Z, (1, 0), (0, 1))
        + np.roll(Z, (-1, 0), (0, 1))
        + np.roll(Z, (0, 1), (0, 1))
        + np.roll(Z, (0, -1), (0, 1))
    )

def update(frame):
    global U, V

    Lu = laplacian(U)
    Lv = laplacian(V)

    reaction = U * V * V

    U += (Du * Lu - reaction + F * (1 - U))
    V += (Dv * Lv + reaction - (F + k) * V)

    im.set_array(V)
    return [im]

fig, ax = plt.subplots()
im = ax.imshow(V, cmap='inferno', interpolation='bilinear')
ax.axis('off')

# --- Mouse interaction ---
brush = 5
def on_move(event):
    if event.inaxes and event.button == 1:
        x, y = int(event.xdata), int(event.ydata)
        V[y-brush:y+brush, x-brush:x+brush] = 0.5
        U[y-brush:y+brush, x-brush:x+brush] = 0.25

fig.canvas.mpl_connect('motion_notify_event', on_move)

ani = FuncAnimation(fig, update, frames=10000, interval=1)

plt.tight_layout()
plt.show()
```

### Reflection/Observations

Run the simulation and let it develop for at least 30 seconds before interacting.

**1.** Describe what you see during the first 10 seconds. How does the pattern change over time?

&nbsp;

&nbsp;

**2.** Click and drag on the canvas. What happens where you paint? Does the pattern respond differently in areas that are already patterned versus empty areas?

&nbsp;

&nbsp;

### Exploration

Try each parameter set below. Let each run for ~30 seconds. Write one word describing each pattern.

| F     | k     | Your description |
|-------|-------|-----------------|
| 0.035 | 0.065 |                 |
| 0.012 | 0.050 |                 |
| 0.025 | 0.055 |                 |

Now try at least two different colormaps (`magma`, `plasma`, `viridis`, `binary`).

**Which combination (F, k, colormap) do you find most visually interesting? Why?**

&nbsp;

&nbsp;

---

## Model 2: Particle Life

### The Code

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 400
NUM_COLORS = 4
COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
attraction = np.random.uniform(-1, 1, (NUM_COLORS, NUM_COLORS))

SPEED    = 0.0002
FRICTION = 0.85
FPS_MS   = 40

x = np.random.uniform(0, 1, N)
y = np.random.uniform(0, 1, N)
vx = np.zeros(N)
vy = np.zeros(N)
color_id = np.repeat(np.arange(NUM_COLORS), N // NUM_COLORS)
colors = [COLORS[c] for c in color_id]

def step():
    global x, y, vx, vy
    fx, fy = np.zeros(N), np.zeros(N)
    for i in range(NUM_COLORS):
        for j in range(NUM_COLORS):
            a = attraction[i, j]
            xi = x[color_id == i]; yi = y[color_id == i]
            xj = x[color_id == j]; yj = y[color_id == j]
            for idx, (px, py) in enumerate(zip(xi, yi)):
                dx = xj - px; dy = yj - py
                dist = np.sqrt(dx**2 + dy**2) + 1e-9
                mask = (dist < 0.2)
                f = a / dist[mask]
                global_idx = np.where(color_id == i)[0][idx]
                fx[global_idx] += np.sum(f * dx[mask] / dist[mask])
                fy[global_idx] += np.sum(f * dy[mask] / dist[mask])
    vx = (vx + fx * SPEED) * FRICTION
    vy = (vy + fy * SPEED) * FRICTION
    x = (x + vx) % 1.0
    y = (y + vy) % 1.0

fig, ax = plt.subplots(figsize=(6, 6))
sc = ax.scatter(x, y, c=colors, s=3, linewidths=0)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ax.set_facecolor('black'); fig.patch.set_facecolor('black')

def animate(frame):
    step()
    sc.set_offsets(np.c_[x, y])
    return [sc]

ani = FuncAnimation(fig, animate, interval=FPS_MS)
plt.tight_layout()
plt.show()
```

### Reflection/Observations

**1.** Without changing anything, describe the overall behavior after ~20 seconds. Do the colors cluster? Scatter? Chase each other?

&nbsp;

&nbsp;

**2.** Run the script again (a new random attraction matrix is generated each time). How different is this run from the last? What does that tell you about the role of the attraction matrix?

&nbsp;

&nbsp;

### Exploration

Try setting specific attraction values manually to create a predator-prey relationship:

```python
# Red chases blue, blue flees red
attraction[0, 1] =  0.9   # red attracted to blue
attraction[1, 0] = -0.9   # blue repelled by red
```

**What behavior do you observe? Does it look biological?**

&nbsp;

&nbsp;


---

## Model 3: Double Pendulum Chaos

### The Code

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

g, L1, L2, m1, m2 = 9.8, 1.0, 1.0, 1.0, 1.0
dt = 0.02
N = 30  # number of pendulums

# Nearly identical starting angles
base_angle = np.pi * 0.6
theta1 = np.full(N, base_angle) + np.linspace(-0.001, 0.001, N)
theta2 = np.full(N, base_angle) + np.linspace(-0.001, 0.001, N)
dth1 = np.zeros(N)
dth2 = np.zeros(N)

trail_x2 = [[] for _ in range(N)]
trail_y2 = [[] for _ in range(N)]
TRAIL_LEN = 120
cmap = plt.cm.plasma

def derivs(t1, t2, dt1, dt2):
    d = t1 - t2
    denom1 = (m1 + m2) * L1 - m2 * L1 * np.cos(d)**2
    denom2 = (L2 / L1) * denom1
    a1 = (m2*L1*dt1**2*np.sin(d)*np.cos(d)
          + m2*g*np.sin(t2)*np.cos(d)
          + m2*L2*dt2**2*np.sin(d)
          - (m1+m2)*g*np.sin(t1)) / denom1
    a2 = (-m2*L2*dt2**2*np.sin(d)*np.cos(d)
          + (m1+m2)*(g*np.sin(t1)*np.cos(d)
          - L1*dt1**2*np.sin(d)
          - g*np.sin(t2))) / denom2
    return a1, a2

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
ax.set_facecolor('black'); fig.patch.set_facecolor('black')
ax.axis('off')
lines = [ax.plot([], [], lw=0.8, alpha=0.7,
         color=cmap(i/N))[0] for i in range(N)]

def animate(frame):
    global theta1, theta2, dth1, dth2
    for _ in range(3):
        a1, a2 = derivs(theta1, theta2, dth1, dth2)
        dth1 += a1 * dt; dth2 += a2 * dt
        theta1 += dth1 * dt; theta2 += dth2 * dt
    x1 =  L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)
    for i in range(N):
        trail_x2[i].append(x2[i])
        trail_y2[i].append(y2[i])
        if len(trail_x2[i]) > TRAIL_LEN:
            trail_x2[i].pop(0); trail_y2[i].pop(0)
        lines[i].set_data(trail_x2[i], trail_y2[i])
    return lines

ani = FuncAnimation(fig, animate, interval=16)
plt.tight_layout()
plt.show()
```

### Reflection/Observations

**1.** For the first several seconds, the trails stay close together. Then they diverge. Approximately how long does it take before the trails look completely different from each other?

&nbsp;

&nbsp;

**2.** Try changing `base_angle` to `np.pi * 0.2` (a small, low-energy swing). Do the trails diverge as quickly? Why do you think that is?

&nbsp;

&nbsp;

### Exploration

Try these changes and note what you see:

- Change `N` to `5` — does it still look chaotic?
- Change `TRAIL_LEN` to `400` — what does this look like?
- Try `cmap = plt.cm.cool` or `plt.cm.spring`

**Which visual setting makes the chaos most readable as art?**

&nbsp;

&nbsp;

---

## Model 4: Generative Galaxy

### The Code

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(None)

# TRY CHANGING THESE!
NUM_ARMS = 4
STARS_PER_ARM = 2000
NOISE = 0.15
SPREAD = 0.12

fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
ax.set_facecolor('black')
ax.axis('off')

for arm in range(NUM_ARMS):
    offset = (2 * np.pi / NUM_ARMS) * arm

    theta = np.linspace(0.3, 4.5 * np.pi, STARS_PER_ARM)

    # Base spiral
    r = 0.12 * np.exp(0.25 * theta)

    # --- (4) Clumping ---
    clump_factor = np.random.normal(1, 0.25, STARS_PER_ARM)
    clump_factor = np.clip(clump_factor, 0.1, None)  # prevent negatives
    r *= clump_factor
    
    # --- (3) Arm distortion ---
    theta_distort = theta + 0.25 * np.sin(3 * theta + arm)

    # Spread + noise
    r += np.random.normal(0, SPREAD * r, STARS_PER_ARM)

    x = r * np.cos(theta_distort + offset)
    y = r * np.sin(theta_distort + offset)

    x += np.random.normal(0, NOISE * 0.1, STARS_PER_ARM)
    y += np.random.normal(0, NOISE * 0.1, STARS_PER_ARM)

    # --- (5) Dust lanes ---
    dust_mask = np.sin(theta * 2 + arm) > 0.85
    x = x[~dust_mask]
    y = y[~dust_mask]

    # Distance + brightness
    dist = np.sqrt(x**2 + y**2)
    brightness = np.clip(1 - dist / dist.max(), 0.05, 1)

    # --- (1) Color diversity ---
    star_types = np.random.choice(
        ['blue', 'white', 'yellow'],
        size=len(x),
        p=[0.2, 0.6, 0.2]
    )

    colors = []
    for i in range(len(x)):
        if star_types[i] == 'blue':
            colors.append((0.6, 0.8, 1.0, brightness[i]))
        elif star_types[i] == 'white':
            colors.append((1.0, 1.0, 1.0, brightness[i]))
        else:
            colors.append((1.0, 0.85, 0.6, brightness[i]))

    # --- (2) Size variation ---
    sizes = np.random.exponential(scale=0.6, size=len(x))
    sizes = np.clip(sizes, 0.1, 3.0)

    # --- (7) Glow effect ---
    ax.scatter(x, y, c=colors, s=sizes * 3, alpha=0.05, linewidths=0)
    ax.scatter(x, y, c=colors, s=sizes, alpha=0.85, linewidths=0)

# --- (8) Improved glowing core ---
for i in range(25):
    core = plt.Circle(
        (0, 0),
        0.02 * i,
        color=(1.0, 0.9, 0.7),
        alpha=0.025
    )
    ax.add_patch(core)

# --- (6) Bright hero stars ---
num_bright = 60
bx = np.random.uniform(-1.8, 1.8, num_bright)
by = np.random.uniform(-1.8, 1.8, num_bright)

ax.scatter(bx, by, c='white', s=12, alpha=0.9, linewidths=0)

# Background stars (slightly varied)
bg_x = np.random.uniform(-2, 2, 2000)
bg_y = np.random.uniform(-2, 2, 2000)
bg_sizes = np.random.uniform(0.05, 0.5, 2000)

ax.scatter(bg_x, bg_y, c='white', s=bg_sizes, alpha=0.25, linewidths=0)

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

plt.tight_layout()
plt.savefig('galaxy_enhanced.png', dpi=200, bbox_inches='tight', facecolor='black')
plt.show()
```

### Reflection/Observations

**1.** Run the script twice without changing anything. Describe how the two galaxies differ. What is the source of that variation?

&nbsp;

&nbsp;

**2.** Change `NUM_ARMS` to `2`, then `6`. How does the visual feel change?

&nbsp;

&nbsp;

### Exploration

Try these changes:

- Swap `cmap = plt.cm.Blues` for `plt.cm.Oranges`, `plt.cm.Reds`, or `plt.cm.cool`
- Change `NOISE` to `0.05` (tight) and `0.4` (loose) — what kind of galaxy do you get?
- Change `NUM_ARMS` to `1` for a one-armed spiral

**Describe the galaxy you find most beautiful. What parameters produced it?**

&nbsp;

&nbsp;

---

*CS 108 — The Art and Practice of Computer Science | Walla Walla University*