# CS 108 Lab — Pygame: From Pixels to Generative Art

**Name:** _______________________________ **Date:** _____________

---

## Submission 

> Submit only [OBSERVE] questions. Do not submit [REFLECT]s!!

## Overview

Pygame is a Python library for making interactive graphics and games. Under the hood it
gives you a **window**, a **drawing surface**, and an **event system**. Everything you see
on screen — a moving circle, a particle trail, a bouncing ball — is the result of the same
loop running hundreds of times per second.

By the end of this lab you will have built an interactive generative art toy driven by your
mouse and keyboard.

**Install once:**

```bash
uv add pygame
```

---

## Setup — The Game Loop

Every Pygame program has the same skeleton. Read it carefully before running.

```python
import pygame
import sys

pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("CS 108 Pygame")

clock = pygame.time.Clock()
FPS   = 60

while True:
    # 1. Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # 2. Update state
    # (nothing yet)

    # 3. Draw
    screen.fill((30, 30, 46))       # dark background
    pygame.display.flip()           # push frame to screen
    clock.tick(FPS)                 # cap at 60 FPS
```

Save this as `lab_pygame.py` and run it. You should see a dark window.

**[OBSERVE]** The loop has three numbered sections. In plain English, what does each section do? Why do you think they have to happen in that order?

&nbsp;

&nbsp;

&nbsp;

---

## Part 1: Drawing

### 1A — Basic Shapes

Add these lines inside the `# 3. Draw` section, after `screen.fill(...)`:

```python
# Circle: surface, color, center, radius
pygame.draw.circle(screen, (137, 180, 250), (400, 300), 60)

# Rectangle: surface, color, (x, y, width, height)
pygame.draw.rect(screen, (166, 227, 161), (100, 100, 120, 80))

# Line: surface, color, start, end, thickness
pygame.draw.line(screen, (243, 139, 168), (0, 0), (800, 600), 2)
```

**[REFLECT]** The coordinate `(0, 0)` is the **top-left** corner. The line goes from `(0,0)`
to `(800, 600)`. Where does it go? Does the y-axis point up or down in Pygame?

&nbsp;

&nbsp;

### 1B — Color Model

Colors in Pygame are `(R, G, B)` tuples, each value 0–255.

Try changing the circle color to each of these and observe:

```python
(255, 0,   0  )   # ?
(0,   255, 0  )   # ?
(0,   0,   255)   # ?
(255, 255, 0  )   # ?
(255, 255, 255)   # ?
(0,   0,   0  )   # ?
```

**[REFLECT]** Think what color each tuple produces. What does `(255, 255, 0)` look like?
Can you predict it from the R, G, B values before running it? Read this: https://en.wikipedia.org/wiki/RGB_color_model 

&nbsp;

&nbsp;

---

## Part 2: Movement

### 2A — A Moving Circle

Replace your `# Update state` section with this:

```python
# at the top of the file, outside the while loop:
x, y   = 400, 300
vx, vy = 3, 2

# inside the loop, in Update state:
x += vx
y += vy

# bounce off walls
if x <= 0 or x >= WIDTH:
    vx = -vx
if y <= 0 or y >= HEIGHT:
    vy = -vy

# in Draw:
screen.fill((30, 30, 46))
pygame.draw.circle(screen, (137, 180, 250), (int(x), int(y)), 30)
```

**[OBSERVE]** What happens when the circle reaches the edge? Trace through the bounce logic
in your head — why does negating `vx` cause a bounce?

&nbsp;

&nbsp;

**[REFLECT]** Change `vx, vy = 3, 2` to `vx, vy = 3, 3`. What changes about the path?
Try `vx, vy = 4, 2`.

&nbsp;

&nbsp;

### 2B — Keyboard Input

Replace the update section with this version that adds keyboard control:

```python
keys = pygame.key.get_pressed()
SPEED = 4

if keys[pygame.K_LEFT]:  x -= SPEED
if keys[pygame.K_RIGHT]: x += SPEED
if keys[pygame.K_UP]:    y -= SPEED
if keys[pygame.K_DOWN]:  y += SPEED

# clamp to window
x = max(0, min(WIDTH,  x))
y = max(0, min(HEIGHT, y))
```

**[REFLECT]** `pygame.key.get_pressed()` runs every frame. How is this different from
handling a `KEYDOWN` event? What would happen if movement only triggered on `KEYDOWN`?

&nbsp;

&nbsp;

### 2C — Mouse Input

Add this to the update section:

```python
mx, my = pygame.mouse.get_pos()
pressed = pygame.mouse.get_pressed()
```

And draw a second circle that follows the mouse:

```python
pygame.draw.circle(screen, (166, 227, 161), (mx, my), 15)
```

**[TASK]** Move the mouse quickly. Does the green circle keep up perfectly? Now hold the
left mouse button (`pressed[0]`). Add a print statement that fires only when the button is
held:

```python
if pressed[0]:
    print(f"clicking at {mx}, {my}")
```

**[OBSERVE]** How many times per second does the print fire? What controls that rate?

&nbsp;

&nbsp;

---

## Part 3: Classes and Particles

### Background

So far everything is a single object. Generative art usually involves **hundreds** of
objects behaving simultaneously. We need a way to manage them — that's where a class helps.

### 3A — The Particle Class

Add this class definition at the top of your file, before the game loop:

```python
import random, math

class Particle:
    def __init__(self, x, y):
        angle  = random.uniform(0, 2 * math.pi)
        speed  = random.uniform(1, 4)
        self.x  = x
        self.y  = y
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.life     = 1.0               # 1.0 = fully alive
        self.decay    = random.uniform(0.01, 0.03)
        self.radius   = random.randint(3, 8)
        self.color    = (
            random.randint(150, 255),
            random.randint(50,  150),
            random.randint(200, 255),
        )

    def update(self):
        self.x    += self.vx
        self.y    += self.vy
        self.vy   += 0.05          # gravity
        self.life -= self.decay

    def draw(self, surface):
        if self.life <= 0:
            return
        alpha  = int(self.life * 255)
        radius = int(self.radius * self.life)
        color  = (
            int(self.color[0] * self.life),
            int(self.color[1] * self.life),
            int(self.color[2] * self.life),
        )
        pygame.draw.circle(surface, color, (int(self.x), int(self.y)), max(1, radius))

    def is_dead(self):
        return self.life <= 0
```

Now update your game loop to spawn and manage particles:

```python
# outside loop:
particles = []

# in event handling, add:
if event.type == pygame.MOUSEBUTTONDOWN:
    for _ in range(20):
        particles.append(Particle(mx, my))

# in update:
for p in particles:
    p.update()
particles = [p for p in particles if not p.is_dead()]

# in draw (after screen.fill):
for p in particles:
    p.draw(screen)
```

Run it. Click on the window.

**[OBSERVE]** Describe what happens when you click. Where do the particles go and why?
Look through `__init__` — what determines the direction each particle travels?

&nbsp;

&nbsp;

**[REFLECT]** What does `self.life -= self.decay` do over time? What happens to a particle
when `life` reaches 0?

&nbsp;

&nbsp;

### 3B — Continuous Spawning

Change the `MOUSEBUTTONDOWN` event to spawn while the mouse is *held*, not just on click.
Move the spawn code into the update section:

```python
# in update, replace the event-based spawn:
if pygame.mouse.get_pressed()[0]:
    for _ in range(5):
        particles.append(Particle(mx, my))
```

**[OBSERVE]** How does the behavior change? What is the maximum number of particles alive
at once? What controls that?

&nbsp;

&nbsp;

### 3C — Trails: Persistence of Vision

Right now `screen.fill(...)` clears the screen every frame, erasing history. Replace it
with a semi-transparent fade:

```python
# replace screen.fill((30, 30, 46)) with:
fade = pygame.Surface((WIDTH, HEIGHT))
fade.set_alpha(20)                        # 0 = invisible, 255 = opaque
fade.fill((30, 30, 46))
screen.blit(fade, (0, 0))
```

**[REFLECT]** What do you see now when you move the mouse? Why does a low alpha value
produce trails instead of erasing?

&nbsp;

&nbsp;

**[OBSERVE]** Try `set_alpha(80)` and `set_alpha(5)`. Describe the difference. What does
alpha control about the trail?

&nbsp;

&nbsp;

---

## Part 4: Generative Art Mode

### 4A — Attractor: Particles Follow the Mouse

Add a gentle force that pulls each particle toward the mouse cursor:

```python
# add to Particle.update(), passing mx and my:
def update(self, mx=0, my=0, attract=False):
    if attract:
        dx = mx - self.x
        dy = my - self.y
        dist = math.sqrt(dx**2 + dy**2) + 1
        self.vx += (dx / dist) * 0.5
        self.vy += (dy / dist) * 0.5

    self.vx *= 0.98        # drag
    self.vy *= 0.98
    self.x  += self.vx
    self.y  += self.vy
    self.life -= self.decay
```

Pass the mouse position when updating:

```python
right_held = pygame.mouse.get_pressed()[2]
for p in particles:
    p.update(mx, my, attract=right_held)
```

**[OBSERVE]** Hold the right mouse button. What happens to the particles? Release it. What
happens then?

&nbsp;

&nbsp;

### 4B — Color Modes

Add a variable `color_mode` that cycles through presets when the spacebar is pressed:

```python
# outside loop:
color_mode = 0

# in event handling:
if event.type == pygame.KEYDOWN:
    if event.key == pygame.K_SPACE:
        color_mode = (color_mode + 1) % 3
```

Update the `Particle.__init__` to accept a color mode:

```python
def __init__(self, x, y, mode=0):
    ...
    if mode == 0:   # purple nebula
        self.color = (random.randint(150,255), random.randint(50,150), random.randint(200,255))
    elif mode == 1: # fire
        self.color = (random.randint(200,255), random.randint(50,150), 0)
    elif mode == 2: # ice
        self.color = (0, random.randint(150,255), random.randint(200,255))
```

Pass `color_mode` when spawning:

```python
particles.append(Particle(mx, my, mode=color_mode))
```

**[OBSERVE]** Press space to cycle modes while painting. Which mode looks most "alive"?
What is it about the color that creates that feeling?

&nbsp;

&nbsp;

### 4C — Gravity Toggle

Add a gravity toggle with the `G` key:

```python
# outside loop:
gravity = True

# in event handling:
if event.type == pygame.KEYDOWN:
    if event.key == pygame.K_g:
        gravity = not gravity
```

Update `Particle.update` to respect the toggle:

```python
if gravity:
    self.vy += 0.05
```

**[OBSERVE]** Turn gravity off and paint with the mouse. How does the behavior change?
What kind of natural system does zero-gravity mode remind you of?

&nbsp;

&nbsp;

---

## Reflection

**[REFLECT]** The game loop runs 60 times per second. If `update()` is slow (e.g. you have
10,000 particles), what do you think happens to the experience? What would you do about it?

&nbsp;

&nbsp;

---

*CS 108 — The Art and Practice of Computer Science | Walla Walla University*