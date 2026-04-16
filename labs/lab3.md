# CS 108 Lab — Stable Diffusion: Painting with Noise

**Name:** _______________________________ **Date:** _____________

---

## Overview

In this lab you will run Stable Diffusion locally and observe how it works from three angles:
**what you say** (prompt engineering), **how it works** (the diffusion process), and **how you
tune it** (parameters). No training. No GPU required. Just a model, a prompt, and your eyes.

**Install dependencies once:**

```bash
pip install diffusers transformers accelerate torch Pillow
```

**First run will download the model (~1.7 GB). Find somewhere with good wifi.**

---

## Setup — Run This First

Save this as `sd_base.py` and keep it open. All lab sections modify or extend it.

```python
import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
from PIL import Image

# ── Load model ────────────────────────────────────────────────────────────────
MODEL = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL,
    torch_dtype=torch.float32,   # CPU: float32
    # torch_dtype=torch.float16, # GPU: uncomment this, comment line above
)

pipe = pipe.to("cpu")
# pipe = pipe.to("cuda")         # GPU: uncomment, comment line above

# Optional: reduce RAM usage on CPU
pipe.enable_attention_slicing()

# ── Generate and show ─────────────────────────────────────────────────────────
def generate(prompt, negative_prompt="", steps=20, guidance=7.5, seed=42):
    generator = torch.manual_seed(seed)
    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    ).images[0]
    return image

def show(image, title=""):
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.axis('off')
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.show()

# ── Test it ───────────────────────────────────────────────────────────────────
img = generate("a red apple on a wooden table")
show(img, "test")
```

Run it. If you see an apple, you're ready.

> **Note on speed:** Each generation takes 2–5 minutes on CPU. Plan ahead — start a run,
> then read the next section while it processes.

---

## Part 1: Prompt Engineering

### Background

Stable Diffusion was trained on hundreds of millions of image-text pairs scraped from the
web. The model learned statistical associations between words and visual features. Your
prompt is not a command — it is a **query into that learned space**. Small word choices
move you to very different regions of that space.

### 1A — Adjective Pressure

Run all four prompts. Use the same seed (`seed=42`) for all.

```python
prompts = [
    "a house",
    "a cozy house",
    "a cozy Victorian house",
    "a cozy Victorian house at dusk, warm lighting, photorealistic",
]

for p in prompts:
    img = generate(p)
    show(img, p)
```

**[OBSERVE]** Describe how the image changes from prompt 1 to prompt 4. What does each
added word do?

&nbsp;

&nbsp;

&nbsp;

**[OBSERVE]** Does the model interpret "Victorian" as a visual style or as a historical fact?
How can you tell?

&nbsp;

&nbsp;

### 1B — Style Words

Run these three prompts. Same subject, different style tokens.

```python
subjects = [
    "a portrait of an old sailor",
    "a portrait of an old sailor, oil painting, Rembrandt",
    "a portrait of an old sailor, cyberpunk neon, digital art, trending on ArtStation",
]

for p in subjects:
    img = generate(p)
    show(img, p)
```

**[OBSERVE]** How much does the style token change the image versus the subject description?
Which has more visual weight?

&nbsp;

&nbsp;

### 1C — Negative Prompts

The `negative_prompt` parameter tells the model what to push *away* from.

```python
base = "a bowl of fruit"

img_without = generate(base)
img_with    = generate(base, negative_prompt="bananas, dark shadows, blurry")

show(img_without, "no negative prompt")
show(img_with,    "with negative prompt")
```

**[OBSERVE]** Describe what changed. Did the negative prompt remove exactly what you
specified, or did it change other things too?

&nbsp;

&nbsp;

**[REFLECT]** Why might a photographer or designer find negative prompts more useful than
just adding more words to the positive prompt?

&nbsp;

&nbsp;

---

## Part 2: The Diffusion Process

### Background

Stable Diffusion does not draw an image from scratch. It starts with **pure random noise**
and gradually removes noise over many steps, guided by your prompt. Each step asks:
*"given this noisy image and this prompt, what should the slightly less noisy version look like?"*

We can watch this process by capturing intermediate latent states.

### 2A — Watching Denoising Happen

This version captures the image at regular intervals during generation.

```python
import numpy as np
from diffusers import StableDiffusionPipeline
import torch, matplotlib.pyplot as plt

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
)
pipe = pipe.to("cpu")
pipe.enable_attention_slicing()

snapshots = []
snapshot_steps = [1, 3, 5, 10, 15, 20]

def capture_callback(pipe, step, timestep, callback_kwargs):
    if step in snapshot_steps:
        latents = callback_kwargs["latents"]
        # Decode latent to image
        with torch.no_grad():
            decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        decoded = decoded[0].permute(1, 2, 0).cpu().numpy()
        decoded = (decoded * 255).astype(np.uint8)
        snapshots.append((step, decoded.copy()))
    return callback_kwargs

generator = torch.manual_seed(42)
pipe(
    "a lighthouse on a rocky cliff, stormy sea, dramatic sky",
    num_inference_steps=20,
    guidance_scale=7.5,
    generator=generator,
    callback_on_step_end=capture_callback,
    callback_on_step_end_tensor_inputs=["latents"],
)

# Show progression
fig, axes = plt.subplots(1, len(snapshots), figsize=(16, 3))
for ax, (step, img) in zip(axes, snapshots):
    ax.imshow(img)
    ax.set_title(f"step {step}")
    ax.axis('off')
plt.suptitle("Denoising progression", fontsize=12)
plt.tight_layout()
plt.show()
```

**[OBSERVE]** At what step does the image first become recognizable as something specific?

&nbsp;

&nbsp;

**[OBSERVE]** What is present in the very early steps (1–3)? Describe it — is it pure noise,
or is there already structure?

&nbsp;

&nbsp;

### 2B — Same Noise, Different Prompt

The seed controls the starting noise. Run the same seed with two different prompts and
compare the early steps.

```python
prompt_a = "a dense forest"
prompt_b = "a crowded city street"

img_a = generate(prompt_a, seed=99)
img_b = generate(prompt_b, seed=99)

show(img_a, f"seed 99: '{prompt_a}'")
show(img_b, f"seed 99: '{prompt_b}'")
```

**[OBSERVE]** The two images started from identical noise. How similar or different are they?
What does this tell you about the role of the prompt versus the role of the seed?

&nbsp;

&nbsp;

**[REFLECT]** If the model starts from noise and denoises toward the prompt, what does it
mean for an image to be "in" the model? Where is the image before you generate it?

&nbsp;

&nbsp;

---

## Part 3: Parameters

### Background

Three parameters shape every generation. You will now isolate each one.

| Parameter | What it does | Typical range |
|---|---|---|
| `num_inference_steps` | How many denoising steps | 10–50 |
| `guidance_scale` | How strictly to follow the prompt | 1–20 |
| `seed` | Which starting noise pattern | any integer |

### 3A — Steps: Quality vs. Speed

```python
prompt = "a Japanese zen garden, raked sand, stone lantern, misty morning"

for steps in [5, 10, 20, 40]:
    img = generate(prompt, steps=steps, seed=42)
    show(img, f"steps={steps}")
```

**[OBSERVE]** At what step count does the image look "good enough"? At what point do more
steps stop making a visible difference?

&nbsp;

&nbsp;

### 3B — Guidance Scale: Literal vs. Creative

```python
prompt = "an astronaut riding a horse on Mars"

for cfg in [1.0, 4.0, 7.5, 15.0, 20.0]:
    img = generate(prompt, guidance=cfg, seed=42)
    show(img, f"guidance={cfg}")
```

**[OBSERVE]** Describe what happens at very low guidance (1.0). At very high guidance (20.0).
Where is the sweet spot for this prompt?

&nbsp;

&nbsp;

**[REFLECT]** Guidance scale is a tension between "follow the prompt" and "make a good
image." Why might these two goals conflict?

&nbsp;

&nbsp;

### 3C — Seeds: Identity of an Image

```python
prompt = "a fox in autumn leaves, soft light, detailed fur"

for seed in [0, 1, 2, 3]:
    img = generate(prompt, seed=seed)
    show(img, f"seed={seed}")
```

**[OBSERVE]** How much do the four images differ? What stays consistent (subject, style,
composition) and what changes?

&nbsp;

&nbsp;

**[OBSERVE]** Now change one word in the prompt (e.g. "fox" → "wolf") and run the same
four seeds. Do the same seeds produce similar compositions? Why or why not?

&nbsp;

&nbsp;

---

## Portfolio Submission

Pick any prompt and parameter combination from the lab. Tune it until you produce an image
you find genuinely interesting or beautiful. Save it:

```python
img = generate("your prompt here", negative_prompt="...", steps=20, guidance=7.5, seed=0)
img.save("my_sd_piece.png")
```

**My prompt:**

&nbsp;

**Parameters used** (steps, guidance, seed):

&nbsp;

**Artist statement** — what were you going for, and what surprised you?

&nbsp;

&nbsp;

&nbsp;

---

*CS 108 — The Art and Practice of Computer Science | Walla Walla University*