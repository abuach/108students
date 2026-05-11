# CS 108 Lab — Sound Programming II: Seeing and Composing Music

**Name:** _______________________________ **Date:** _____________

---

## Submission 

> Submit only [OBSERVE] questions. Do not submit [REFLECT]s!!

---

## Overview

In the first sound lab you built sound from math — sine waves, envelopes, effects.

This lab goes in new directions: **seeing** sound (visualizing frequency in real time)
and **composing** sound algorithmically (writing rules that generate music).

You will work with five models:

1. **FFT Audio Visualizer** — see the frequency content of any audio file
2. **Algorithmic MIDI** — compose by writing rules, play back through a synth
3. **Euclidean Rhythms** — distribute beats mathematically, the way West African and electronic music does
4. **Cellular Automaton Music** — let Conway's Game of Life compose a melody

You will **also** create music **generatively** using AI.

Large language models are trained on text — including vast amounts of text *about* music:
sheet music notation, MIDI descriptions, music theory textbooks, ABC files, and code that
generates sound. That means an LLM has latent knowledge of melody, rhythm, and harmony,
even though it has never heard a single note.

In this lab you will extract that musical knowledge three ways: as structured data, as a
formal notation, and as code. In each case, Ollama generates text and Python turns it into
sound.

**Install once:**

```bash
uv add numpy scipy matplotlib pygame mido ollama music21
```

---

## Part 1: FFT Audio Visualizer

### Background

Any sound — a voice, a chord, a drum hit — is a mixture of many frequencies playing
simultaneously. The **Fast Fourier Transform (FFT)** decomposes a chunk of audio into
its component frequencies. It answers: *right now, how much of each frequency is present?*

A visualizer runs the FFT on a sliding window of audio, hundreds of times per second,
and draws the result as bars. The bars bounce because the frequency content changes
as the music changes.

We will load an audio file, chunk it into frames, and animate the FFT of each frame.

### The Code

Save this as `visualizer.py`. You will need any `.wav` file — find one or grab one
from https://github.com/abuach/soundlab.

```python
import numpy as np
import pygame
import sys
from scipy.io.wavfile import read
from scipy.fft import rfft

# ── Load audio ────────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python visualizer.py path/to/file.wav")
    sys.exit(1)

sr, data = read(sys.argv[1])

# Flatten to mono if stereo
if data.ndim == 2:
    data = data.mean(axis=1)
data = data.astype(np.float32)
data /= np.max(np.abs(data))

# ── Settings ──────────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 900, 500
NUM_BARS      = 64          # number of frequency bars
FRAME_SIZE    = 2048        # samples per FFT frame
HOP           = 512         # samples to advance per frame
FPS           = 60
BAR_COLOR     = (137, 180, 250)
BG_COLOR      = (30, 30, 46)
PEAK_COLOR    = (243, 139, 168)

# ── Pygame setup ──────────────────────────────────────────────────────────────
pygame.init()
pygame.mixer.init(frequency=sr, size=-16, channels=1, buffer=512)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("FFT Visualizer")
clock  = pygame.time.Clock()
font   = pygame.font.SysFont("monospace", 13)

# Convert to 16-bit PCM for pygame mixer
pcm = (data * 32767).astype(np.int16)
sound = pygame.sndarray.make_sound(pcm)
sound.play()
start_tick = pygame.time.get_ticks()

# ── Peak hold ─────────────────────────────────────────────────────────────────
peaks      = np.zeros(NUM_BARS)
peak_decay = 0.97

def get_spectrum(sample_pos):
    chunk = data[sample_pos : sample_pos + FRAME_SIZE]
    if len(chunk) < FRAME_SIZE:
        chunk = np.pad(chunk, (0, FRAME_SIZE - len(chunk)))
    window  = chunk * np.hanning(FRAME_SIZE)
    fft_out = np.abs(rfft(window))
    # Group FFT bins into NUM_BARS using log spacing
    freqs     = np.logspace(np.log10(20), np.log10(sr / 2), NUM_BARS + 1)
    bin_freqs = np.linspace(0, sr / 2, len(fft_out))
    bars = np.zeros(NUM_BARS)
    for i in range(NUM_BARS):
        lo = np.searchsorted(bin_freqs, freqs[i])
        hi = np.searchsorted(bin_freqs, freqs[i + 1])
        if hi > lo:
            bars[i] = np.mean(fft_out[lo:hi])
    # Normalize
    bars = np.log1p(bars)
    if bars.max() > 0:
        bars /= bars.max()
    return bars

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit(); sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit(); sys.exit()

    elapsed_ms  = pygame.time.get_ticks() - start_tick
    sample_pos  = int((elapsed_ms / 1000) * sr)
    if sample_pos >= len(data):
        pygame.quit(); sys.exit()

    bars = get_spectrum(sample_pos)

    # Update peaks
    peaks = np.where(bars > peaks, bars, peaks * peak_decay)

    screen.fill(BG_COLOR)

    bar_w   = WIDTH // NUM_BARS
    padding = 2
    for i, (bar, peak) in enumerate(zip(bars, peaks)):
        x      = i * bar_w
        bh     = int(bar  * (HEIGHT - 60))
        ph     = int(peak * (HEIGHT - 60))

        # Bar fill — color shifts with height
        r = int(137 + (243 - 137) * bar)
        g = int(180 - 100 * bar)
        b = int(250 - 100 * bar)
        pygame.draw.rect(screen, (r, g, b),
                         (x + padding, HEIGHT - bh - 30,
                          bar_w - padding * 2, bh))
        # Peak dot
        pygame.draw.rect(screen, PEAK_COLOR,
                         (x + padding, HEIGHT - ph - 34,
                          bar_w - padding * 2, 4))

    # Time display
    secs = elapsed_ms // 1000
    txt  = font.render(f"{secs//60:02d}:{secs%60:02d}  |  {NUM_BARS} bands  "
                       f"|  ESC to quit", True, (108, 112, 134))
    screen.blit(txt, (10, 10))

    pygame.display.flip()
    clock.tick(FPS)
```

Run it:

```bash
python visualizer.py my_song.wav
```

**[OBSERVE1]** Watch the bars during a quiet section versus a loud section. Which bars
move the most? Are they on the left (low frequencies) or the right (high frequencies)?

&nbsp;

&nbsp;


**[OBSERVE2]** The pink dots above the bars are **peak hold** indicators — they stay at the
highest recent value and decay slowly. Change `peak_decay = 0.97` to `0.5`. What changes?

&nbsp;

&nbsp;

**[REFLECT]** The FFT doesn't know anything about music — it just counts frequencies.
Yet the visualizer clearly reacts to beats, bass drops, and melody lines. What does that
tell you about the relationship between music and math?

&nbsp;

&nbsp;

---

## Part 2: Algorithmic MIDI with `mido` and `pygame`

### Background

MIDI is not audio. It contains no sound — only **instructions**: play note 60 at velocity
80 for 0.5 seconds. A synthesizer reads those instructions and produces sound.

`mido` lets you build MIDI files in Python. `pygame` plays them using its built-in
synthesizer and a **soundfont** (a library of sampled instruments). Together they give
you a full composition pipeline: *rules → MIDI → sound*.

### The Code

```python
import mido
from mido import MidiFile, MidiTrack, Message
import pygame
import time
import random
import os

SR      = 44100
TEMPO   = 500000          # microseconds per beat = 120 BPM
TICKS   = 480             # ticks per beat

def play_midi(path):
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    # Wait for playback to finish
    clock = pygame.time.Clock()
    while pygame.mixer.music.get_busy():
        clock.tick(10)
    pygame.mixer.quit()

def note_on(track, note, velocity, time=0):
    track.append(Message('note_on',  note=note, velocity=velocity, time=time))

def note_off(track, note, time):
    track.append(Message('note_off', note=note, velocity=0,        time=time))

def add_note(track, note, duration_ticks, velocity=80, gap=10):
    """Add a note_on then note_off with a small gap."""
    note_on(track,  note, velocity, time=0)
    note_off(track, note, time=duration_ticks - gap)

# ── Compose ───────────────────────────────────────────────────────────────────
def compose_scale_melody(scale_notes, length=32, bpm=120):
    """Walk randomly through a scale."""
    tempo   = int(60_000_000 / bpm)
    quarter = TICKS

    mid   = MidiFile(ticks_per_beat=TICKS)
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))

    note = random.choice(scale_notes)
    for _ in range(length):
        duration = random.choice([quarter // 2, quarter, quarter * 2])
        velocity = random.randint(60, 100)
        add_note(track, note, duration, velocity)
        # Step up, down, or jump in the scale
        step = random.choice([-2, -1, -1, 0, 1, 1, 2])
        idx  = scale_notes.index(note)
        note = scale_notes[max(0, min(len(scale_notes)-1, idx + step))]

    mid.save("melody.mid")
    return "melody.mid"

# A minor pentatonic starting at A3
A_MINOR_PENTATONIC = [57, 60, 62, 64, 67, 69, 72, 74, 76]

path = compose_scale_melody(A_MINOR_PENTATONIC, length=48, bpm=130)

pygame.mixer.init()
pygame.mixer.music.load(path)
pygame.mixer.music.play()
while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)
```

**[OBSERVE3]** Run the script several times without changing anything. Does it sound the
same each time? What part of the code introduces variation?

&nbsp;

&nbsp;

**[OBSERVE4]** Change `bpm=130` to `bpm=60`. Then try `bpm=220`. How does the character
of the melody change beyond just speed?

&nbsp;

&nbsp;

### Exploration: Fractal Melody

Add this function below `compose_scale_melody` and call it instead:

```python
def compose_fractal(base_notes, depth=3, bpm=100):
    """Recursively repeat a motif at different scales."""
    tempo   = int(60_000_000 / bpm)
    quarter = TICKS

    mid   = MidiFile(ticks_per_beat=TICKS)
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))

    def motif(notes, dur, level):
        if level == 0:
            for n in notes:
                add_note(track, n, dur)
        else:
            for n in notes:
                motif([n, n+2, n+4], dur // 3, level - 1)

    motif(base_notes, quarter * 4, depth)
    mid.save("fractal.mid")
    return "fractal.mid"

path = compose_fractal([60, 64, 67, 72], depth=3, bpm=80)
```

**[OBSERVE5]** Try `depth=1`, `depth=2`, `depth=3`. How does the structure change?

&nbsp;

&nbsp;


---

## Part 3: Euclidean Rhythms

### Background

In 2005, computer scientist Godfried Toussaint discovered that many traditional rhythms
from West Africa, the Middle East, and Latin America can be described by a single
mathematical idea: **distributing k beats as evenly as possible across n slots**.

For example: 3 beats in 8 slots → `[1,0,0,1,0,0,1,0]` → the Cuban tresillo.
5 beats in 8 slots → `[1,0,1,1,0,1,1,0]` → the West African standard pattern.

The algorithm that does this is called **Bjorklund's algorithm**, and it produces
these rhythms automatically. The result sounds like world music — not because the
algorithm knows about culture, but because even distribution *sounds good*.

### The Code

```python
import numpy as np
import pygame
import time

SR = 44100

def bjorklund(beats, slots):
    """Return a Euclidean rhythm as a list of 0s and 1s."""
    if beats == 0:
        return [0] * slots
    if beats == slots:
        return [1] * slots

    pattern    = [[1]] * beats + [[0]] * (slots - beats)
    remainder  = slots - beats

    while remainder > 1:
        times = min(len(pattern) - remainder, remainder)
        for i in range(times):
            pattern[i] = pattern[i] + pattern[-(i+1)]
        pattern = pattern[:len(pattern)-times]
        remainder = len(pattern) - times
        if remainder <= 0:
            break

    return [x for group in pattern for x in group]

def make_click(freq, duration=0.05, sr=SR):
    """Simple sine burst for a drum sound."""
    t    = np.linspace(0, duration, int(sr * duration))
    wave = np.sin(2 * np.pi * freq * t) * np.exp(-t * 40)
    return (wave * 32767).astype(np.int16)

def play_euclidean(beats, slots, bpm=120, bars=4, freq=200):
    """Play a Euclidean rhythm."""
    pattern  = bjorklund(beats, slots)
    step_dur = 60 / (bpm * slots / 4)   # duration of one slot in seconds

    pygame.mixer.init(frequency=SR, size=-16, channels=1, buffer=512)
    click = make_click(freq)
    sound = pygame.sndarray.make_sound(click)

    print(f"E({beats},{slots}): {pattern}")

    for _ in range(bars):
        for hit in pattern:
            if hit:
                sound.play()
            time.sleep(step_dur)

    pygame.mixer.quit()

# ── Try these classic Euclidean rhythms ───────────────────────────────────────
print("Tresillo (3,8) — Cuban son")
play_euclidean(3, 8, bpm=120, freq=180)

print("Cinquillo (5,8) — Cuban/African")
play_euclidean(5, 8, bpm=120, freq=220)

print("Bossa nova (3,16) — Brazilian")
play_euclidean(3, 16, bpm=100, freq=160)
```

**[OBSERVE6]** Play E(4,4) — four beats in four slots. What do you hear? Play E(1,8).
What do you hear? What do these two extremes tell you about what the algorithm is doing?

&nbsp;

&nbsp;

### Layered Polyrhythm

Run multiple Euclidean rhythms simultaneously at different pitches:

```python
import threading

def play_layer(beats, slots, bpm, bars, freq):
    play_euclidean(beats, slots, bpm=bpm, bars=bars, freq=freq)

threads = [
    threading.Thread(target=play_layer, args=(3, 8,  120, 8, 80)),   # kick
    threading.Thread(target=play_layer, args=(5, 8,  120, 8, 300)),  # snare
    threading.Thread(target=play_layer, args=(7, 16, 120, 8, 800)),  # hihat
]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

**[OBSERVE7]** Each layer has a different Euclidean pattern. Does the combination sound
like a drumbeat you might hear in real music? What happens if you change the middle layer
to E(4,8)?

&nbsp;

&nbsp;

**[REFLECT]** The algorithm knows nothing about music, culture, or rhythm — it just
distributes dots evenly. Why do you think evenly distributed rhythms tend to sound
musical?

&nbsp;

&nbsp;

---

## Part 4: Cellular Automaton Music

### Background

Conway's Game of Life is a grid where cells are alive or dead, and each step updates
every cell based on its neighbors. From two simple rules, complex structure emerges.

We can **map the grid to music**: each column is a time step, each row is a pitch.
A living cell at row `r`, column `t` means: play note `r` at time `t`.

The music composes itself. You set the initial state and press play.

### The Code

```python
import numpy as np
import pygame
import mido
from mido import MidiFile, MidiTrack, Message
import time

def life_step(grid):
    """One step of Conway's Game of Life."""
    neighbors = sum(
        np.roll(np.roll(grid, i, 0), j, 1)
        for i in (-1, 0, 1) for j in (-1, 0, 1)
        if (i, j) != (0, 0)
    )
    return ((neighbors == 3) | (grid & (neighbors == 2))).astype(np.uint8)

def life_to_midi(rows=16, cols=64, bpm=120, seed=42):
    """
    Run Game of Life for `cols` steps.
    Map living cells to MIDI notes in a pentatonic scale.
    """
    np.random.seed(seed)
    grid = (np.random.random((rows, cols)) > 0.7).astype(np.uint8)

    # Run the automaton
    frames = [grid.copy()]
    for _ in range(cols - 1):
        grid = life_step(grid)
        frames.append(grid.copy())

    # Pentatonic scale — maps row index to MIDI note
    pentatonic = [60, 62, 64, 67, 69,
                  72, 74, 76, 79, 81,
                  84, 86, 88, 91, 93, 96]
    scale = pentatonic[:rows]

    tempo   = int(60_000_000 / bpm)
    quarter = 480
    step_t  = quarter // 2    # eighth note per column

    mid   = MidiFile(ticks_per_beat=quarter)
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))

    prev = np.zeros((rows, cols), dtype=np.uint8)

    for col, frame in enumerate(frames):
        for row in range(rows):
            note = scale[row]
            vel  = np.random.randint(50, 90)
            if frame[row, col % cols] and not prev[row, col % cols]:
                track.append(Message('note_on',  note=note,
                                     velocity=vel, time=0))
                track.append(Message('note_off', note=note,
                                     velocity=0,  time=step_t))
        prev = frame.copy()

    mid.save("life_music.mid")

    # Visualize the grid evolution
    import matplotlib.pyplot as plt
    arr = np.array(frames).T
    plt.figure(figsize=(14, 4))
    plt.imshow(arr[:, :, 0] if arr.ndim == 3 else
               np.array([f[:,0] for f in frames]).T,
               cmap='plasma', aspect='auto', origin='lower')
    plt.xlabel("Time step (column)")
    plt.ylabel("Pitch (row)")
    plt.title("Game of Life → Music  |  bright = alive = note plays")
    plt.colorbar(label="alive")
    plt.tight_layout()
    plt.show()

    return "life_music.mid"

path = life_to_midi(rows=16, cols=64, bpm=110, seed=42)

pygame.mixer.init()
pygame.mixer.music.load(path)
pygame.mixer.music.play()
while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)
```


**[OBSERVE8]** Change `seed=42` to `seed=0`, `seed=1`, `seed=7`. Each produces a
completely different piece. Run the same seed twice — do you get the same music?

&nbsp;

&nbsp;

**[OBSERVE9]** Change the density threshold from `> 0.7` to `> 0.5` (denser start) and
`> 0.9` (sparse start). How does starting density affect the music after 64 steps?

&nbsp;

&nbsp;

---

> **Shared synth helpers!!** — save this as `synth.py` and keep it in the same folder.
All three AI models import from it.

```python
import numpy as np
import pygame
from scipy.io.wavfile import write

SR = 44100

def play(wave):
    wave = wave.astype(np.float32)
    wave /= np.max(np.abs(wave)) + 1e-9
    pcm = (wave * 32767).astype(np.int16)
    pcm_stereo = np.column_stack([pcm, pcm])
    pygame.mixer.init(frequency=SR, size=-16, channels=2, buffer=2048)
    sound = pygame.sndarray.make_sound(pcm_stereo)
    sound.play()
    while pygame.mixer.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.quit()

def save(wave, filename):
    wave = wave.astype(np.float32)
    wave /= np.max(np.abs(wave)) + 1e-9
    write(filename, SR, (wave * 32767).astype(np.int16))
    print(f"Saved: {filename}")

def adsr(duration, attack=0.01, decay=0.1, sustain=0.7, release=0.15):
    n   = int(SR * duration)
    env = np.zeros(n)
    a, d, r = int(SR*attack), int(SR*decay), int(SR*release)
    s = max(0, n - a - d - r)
    env[:a]        = np.linspace(0, 1, a)
    env[a:a+d]     = np.linspace(1, sustain, d)
    env[a+d:a+d+s] = sustain
    env[a+d+s:]    = np.linspace(sustain, 0, max(1, n-a-d-s))
    return env

def note(freq, duration, amplitude=0.3, waveform='sine'):
    t = np.linspace(0, duration, int(SR * duration))
    if waveform == 'sine':
        wave = np.sin(2 * np.pi * freq * t)
    elif waveform == 'sawtooth':
        wave = 2 * (t * freq % 1) - 1
    elif waveform == 'square':
        wave = np.sign(np.sin(2 * np.pi * freq * t))
    else:
        wave = np.sin(2 * np.pi * freq * t)
    return amplitude * wave * adsr(duration)

def midi_to_freq(midi_note):
    return 440.0 * (2 ** ((midi_note - 69) / 12))
```

---

## How to Talk to Ollama from Python

All three models use the same pattern:

```python
import ollama

response = ollama.chat(
    model="llama3.2",
    messages=[
        {"role": "system", "content": "your system prompt here"},
        {"role": "user",   "content": "your request here"},
    ]
)
text = response["message"]["content"]
```

The model returns a string. Your job is to parse that string into something Python can
play. The three models differ only in *what format* you ask for and *how you parse it*.

---

## Model 5: Ollama → JSON → Sound

### Background

JSON is the most reliable format to request from an LLM. Every model can produce it
consistently, it maps directly to Python dictionaries and lists, and you control the
schema. Here we ask Ollama to compose a melody as a list of `{note, duration, velocity}`
objects, parse the JSON, and render it with our numpy synth.

The key insight: the LLM is acting as a **composer**, and Python is acting as a
**performer**. The model decides what to play; the code decides how it sounds.

### The Code

Save as `model1_json.py`:

```python
import ollama
import json
import numpy as np
import re
from synth import play, save, note, midi_to_freq

# Configure your server URL here
SERVER_HOST = 'http://ollama.cs.wallawalla.edu:11434'
client = ollama.Client(host=SERVER_HOST)

SYSTEM = """You are a music composer. When asked to compose,
respond ONLY with a valid JSON array. No explanation, no markdown,
no code blocks. Each element must have exactly these fields:
  "note"     : MIDI note number (integer, 48-84)
  "duration" : note length in seconds (float, 0.1-1.0)
  "velocity" : volume 0-127 (integer)

Example of valid output:
[{"note":60,"duration":0.4,"velocity":80},{"note":62,"duration":0.3,"velocity":70}]"""

def ask_ollama(prompt):
    print(f"Asking Ollama: {prompt}")
    response = client.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": prompt},
        ]
    )
    return response["message"]["content"]

def parse_json(text):
    text  = re.sub(r"```[a-z]*", "", text).strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array found in:\n{text}")
    return json.loads(match.group())

def render(notes, waveform='sine'):
    fragments = []
    for n in notes:
        freq = midi_to_freq(int(n["note"]))
        dur  = float(n["duration"])
        amp  = float(n["velocity"]) / 127 * 0.4
        frag = note(freq, dur, amplitude=amp, waveform=waveform)
        fragments.append(frag)
    # Concatenate safely — trim each fragment to its own length
    return np.concatenate([f.flatten() for f in fragments])

prompts = [
    "Compose a gentle 12-note lullaby melody in C major.",
    "Compose a 16-note tense, minor-key melody that builds in intensity.",
    "Compose a 14-note melody that sounds like walking through a forest.",
]

for prompt in prompts:
    raw = ask_ollama(prompt)
    print(f"Raw response:\n{raw}\n")
    try:
        notes = parse_json(raw)
        print(f"Parsed {len(notes)} notes.")
        wave  = render(notes, waveform='sine')
        play(wave)
        filename = prompt[:20].replace(" ", "_") + ".wav"
        save(wave, filename)
    except Exception as e:
        print(f"Error: {e}")
```

### Observations

**[OBSERVE10]** Run the same prompt twice. Do you get the same melody? What does that tell
you about how LLMs generate sequences?

&nbsp;

&nbsp;

**[OBSERVE11]**

Now try writing your own prompt. Ask for something specific — a genre, a mood, a cultural
style. Paste the prompt and describe what you got:

**Your prompt:**

&nbsp;

**What the model composed (describe the melody):**

&nbsp;

&nbsp;

---

## Model 6: Ollama → ABC Notation → MIDI → Sound

### Background

**ABC notation** is a plain-text music format invented in the 1980s for folk music. It
looks like this:

```
X:1
T:My Tune
M:4/4
L:1/8
K:Amin
A2BC DEFG | A4 z4 |
```

- Letters `A`–`G` are notes. Uppercase = middle octave. Lowercase = one octave up.
- Numbers after a letter = duration multiplier. `A2` = A held for 2 units.
- `z` = rest. `|` = bar line. `K:` = key signature.

LLMs know ABC notation well — there is a large corpus of ABC files on the internet.
`music21` can parse ABC and convert it to MIDI. pygame plays the MIDI.

This pipeline produces richer, more structured music than JSON because ABC encodes
meter, key, and phrasing — not just individual notes.

### The Code

Save as `model2_abc.py`:

```python
import ollama
import re
import pygame
from music21 import converter, midi as m21midi, stream, meter

# Configure your server URL here
SERVER_HOST = 'http://ollama.cs.wallawalla.edu:11434'
client = ollama.Client(host=SERVER_HOST)

SYSTEM = """You are a music composer who writes in ABC notation.
Respond ONLY with valid ABC notation. No explanation. No markdown. No extra text after the notes.

STRICT RULES:
1. Always include L:1/8 header — this is required.
2. M: must be exactly one of: 4/4  3/4  6/8  2/4
3. K: must be a standard key like: C  G  D  A  E  F  Amin  Dmin  Emin
4. Notes are letters A-G only. Use lowercase for one octave up (a,b,c...).
5. Accidentals: ^A = A sharp, _B = B flat. Never use # or b.
6. Duration multipliers only: A2 = double length, A/2 = half length.
7. No comments. No text after the last bar line.
8. End every line of music with a bar line |

VALID EXAMPLE:
X:1
T:Simple Tune
M:4/4
L:1/8
K:G
G2 AB c2 BA | G4 D4 | E2 FG A2 GF | G8 |"""

def ask_ollama(prompt):
    response = client.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": prompt},
        ]
    )
    return response["message"]["content"]

def clean_abc(text):
    text = re.sub(r"```[a-z]*", "", text)
    text = re.sub(r"```", "", text)
    lines    = []
    in_music = False
    has_L    = False
    has_X    = False
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r'^[A-Z]:', line):
            if line.startswith('X:'): has_X = True
            if line.startswith('L:'): has_L = True
            if line.startswith('K:'): in_music = True
            lines.append(line)
        elif in_music and re.search(r'[A-Ga-gz|]', line):
            line = re.sub(r'\|[^|A-Ga-gz\s]*$', '|', line)
            lines.append(line)

    if not has_X:
        lines.insert(0, 'X:1')
    if not has_L:
        for i, l in enumerate(lines):
            if l.startswith('K:'):
                lines.insert(i, 'L:1/8')
                break
    return '\n'.join(lines)

def validate_abc(text):
    errors = []
    for line in text.splitlines():
        if line.startswith('L:'):
            val = line[2:].strip()
            if val not in ('1/8', '1/4', '1/16', '1/2', '1'):
                errors.append(f"Bad L: value '{val}' — must be 1/8, 1/4, or 1/16")
        if line.startswith('K:'):
            val = line[2:].strip()
            if re.search(r'[^A-Ga-gmin#b ]', val):
                errors.append(f"Suspicious K: value '{val}'")
        if not line.startswith(('X:', 'T:', 'M:', 'L:', 'K:', 'Q:')):
            if re.search(r'\b[A-Ga-gN][1-9][0-9]+', line):
                errors.append(f"Possible MIDI-style pitch in: {line}")
            if re.search(r'\(Note:', line, re.IGNORECASE):
                errors.append(f"Trailing comment in: {line}")
    return errors

def fix_duplicate_timesig(score):
    """
    Workaround for music21 bug: TimeSignature already in Stream.
    Flatten the score and remove duplicate time signatures manually.
    """
    for part in score.parts:
        seen = set()
        for ts in part.recurse().getElementsByClass(meter.TimeSignature):
            key = (ts.numerator, ts.denominator)
            if key in seen:
                part.remove(ts, recurse=True)
            else:
                seen.add(key)
    return score

def abc_to_midi(abc_text, out_path="ollama_abc.mid"):
    score = converter.parse(abc_text, format='abc')
    score = fix_duplicate_timesig(score)
    mf    = m21midi.translate.music21ObjectToMidiFile(score)
    mf.open(out_path, 'wb')
    mf.write()
    mf.close()
    print(f"MIDI written: {out_path}")
    return out_path

def play_midi(path):
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.quit()

def generate_and_play(prompt, retries=4):
    print(f"\nPrompt: {prompt}")
    original_prompt = prompt
    for attempt in range(1, retries + 1):
        print(f"  Attempt {attempt}/{retries}...")
        raw = ask_ollama(prompt)
        abc = clean_abc(raw)
        print(f"  Cleaned ABC:\n{abc}\n")

        errors = validate_abc(abc)
        if errors:
            print(f"  Validation failed: {errors}")
            prompt = (f"{original_prompt}\n\nYour previous attempt had errors: "
                      f"{errors}. Fix them. Remember: always include L:1/8.")
            continue

        try:
            path = abc_to_midi(abc)
            play_midi(path)
            return
        except Exception as e:
            print(f"  Parse error: {e}")
            prompt = (f"{original_prompt}\n\nPrevious attempt failed: {e}. "
                      f"Write simpler ABC. Use L:1/8 and only basic notes.")

    print(f"  Failed after {retries} attempts.")

# ── Prompts ───────────────────────────────────────────────────────────────────
prompts = [
    "Compose a short Irish jig melody in D major, 6/8 time, 4 bars.",
    "Compose a simple Baroque-style melody in A minor, 4/4 time, 4 bars.",
    "Compose a pentatonic melody in C major using only C D E G A, 4/4 time, 4 bars.",
]

for prompt in prompts:
    generate_and_play(prompt)
```

### Observations

**[OBSERVE11]**

Ask Ollama for a melody in a specific cultural style. ABC notation has key signatures
and modes (Dorian, Mixolydian, etc.) that produce different emotional flavors.

Try:
```python
"Compose a melody in Dorian mode — dark but not quite minor."
"Compose a waltz melody in 3/4 time."
```

Does the model understand musical modes? Does the Dorian melody actually
sound different from a standard minor key melody?

&nbsp;

&nbsp;

**[REFLECT]** ABC notation encodes musical structure that JSON note lists don't — meter,
key, phrasing. Does the added structure produce better music? What is lost compared to
the JSON approach?

&nbsp;

&nbsp;

---

## Model 7: Ollama → Python Code → Sound

### Background

Instead of asking the LLM to *describe* music in a data format, this model asks it to
*write code* that generates music — using your own `synth.py` tools as context.

You give Ollama the API (the `note()`, `adsr()`, and `play()` functions), and ask it to
write a Python composition using them. Then you `exec()` the result.

This is the most open-ended approach: the model can use loops, randomness, conditionals,
and any numpy math it knows. It is also the least predictable — sometimes it writes
beautiful code, sometimes it hallucinates function names. Watching it fail is as
instructive as watching it succeed.

### The Code

Save as `model3_codegen.py`:

```python
import ollama
import re
import numpy as np
from synth import play, save, note, adsr, midi_to_freq, SR

# Configure your server URL here
SERVER_HOST = 'http://ollama.cs.wallawalla.edu:11434'
client = ollama.Client(host=SERVER_HOST)

SYSTEM = f"""You are a Python music programmer. You have access to exactly these names:

  SR = {SR}                          # int, sample rate

  note(freq, duration, amplitude=0.3, waveform='sine') -> np.ndarray
    # freq: Hz as a float
    # duration: seconds as a float
    # waveform: 'sine', 'sawtooth', or 'square'
    # returns a 1-D float32 numpy array

  midi_to_freq(midi_note) -> float
    # midi_note: int. Middle C = 60 = 261.63 Hz
    # A4 = 69 = 440 Hz

  np                                 # numpy, fully available

WORKING EXAMPLES — study these before writing code:

# Example 1: simple scale
wave = np.concatenate([
    note(midi_to_freq(m), 0.3)
    for m in [60, 62, 64, 65, 67]
])

# Example 2: rest = array of zeros
rest = np.zeros(int(SR * 0.25))     # 0.25 second silence
wave = np.concatenate([
    note(440.0, 0.2),
    rest,
    note(440.0, 0.2),
    rest,
])

# Example 3: chord = sum of arrays, padded to same length
a = note(midi_to_freq(60), 1.0)
b = note(midi_to_freq(64), 1.0)
c = note(midi_to_freq(67), 1.0)
wave = a + b + c

# Example 4: random melody
midi_pool = [60, 62, 64, 67, 69]
chosen    = np.random.choice(midi_pool, size=8)
wave      = np.concatenate([note(midi_to_freq(int(m)), 0.35) for m in chosen])

RULES:
- Never use note names like 'C4' or 'A#3'. Always use midi_to_freq(int).
- Never call functions that are not listed above.
- Never write import statements.
- The final result must be stored in a variable called `wave`.
- `wave` must be a 1-D numpy array. Never leave it as a list.
- For rests use np.zeros(int(SR * duration)).
- Do not call play() or save()."""

def ask_ollama(prompt):
    response = client.chat(
        model="qwen2.5-coder",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": prompt},
        ]
    )
    return response["message"]["content"]

def clean_code(text):
    text = re.sub(r"```[a-z]*", "", text)
    text = re.sub(r"```",       "", text)
    # Strip any import lines — model sometimes ignores the rule
    lines = [l for l in text.splitlines()
             if not l.strip().startswith("import")]
    return '\n'.join(lines).strip()

def run_code(code):
    namespace = {
        "np":           np,
        "note":         note,
        "adsr":         adsr,
        "midi_to_freq": midi_to_freq,
        "SR":           SR,
    }
    exec(code, namespace)
    if "wave" not in namespace:
        raise ValueError("No `wave` variable produced.")
    w = namespace["wave"]
    if not isinstance(w, np.ndarray):
        raise ValueError(f"`wave` is {type(w)}, expected np.ndarray.")
    if w.ndim != 1:
        raise ValueError(f"`wave` has shape {w.shape}, must be 1-D.")
    return w.astype(np.float32)

def generate_and_play(name, prompt, retries=3):
    print(f"\n--- {name} ---")
    original_prompt = prompt
    for attempt in range(1, retries + 1):
        print(f"Attempt {attempt}/{retries}...")
        raw  = ask_ollama(prompt)
        code = clean_code(raw)
        print(f"Code:\n{code}\n")
        try:
            wave = run_code(code)
            print(f"OK — shape {wave.shape}")
            play(wave)
            save(wave, f"codegen_{name}.wav")
            return
        except Exception as e:
            print(f"Error: {e}")
            prompt = (
                f"{original_prompt}\n\n"
                f"Your previous attempt failed with: {e}\n"
                f"Remember: use midi_to_freq(int) not note names. "
                f"Rests are np.zeros(int(SR * dur)). "
                f"Result must be a 1-D array named `wave`."
            )
    print(f"Failed after {retries} attempts.")

# ── Prompts ───────────────────────────────────────────────────────────────────
tasks = [
    ("arpeggio",
     "Write code that plays a C major arpeggio (MIDI 60 64 67 72) "
     "ascending then descending. Each note 0.25 seconds. Use sine waves."),

    ("rhythm",
     "Write code that alternates a 0.15-second note at 440 Hz "
     "with a 0.15-second rest, 16 times total. "
     "Use np.zeros for rests and np.concatenate to join everything."),

    ("generative",
     "Write code that picks 12 random MIDI notes from [60,62,64,65,67,69,71,72], "
     "gives each a random duration between 0.2 and 0.5 seconds, "
     "and concatenates them into `wave`."),

    ("chord_progression",
     "Write code that plays four chords in sequence: "
     "Am (57,60,64), C (60,64,67), G (55,59,62), E (52,56,59). "
     "Each chord lasts 1.0 second. A chord is three note() arrays summed together. "
     "Concatenate the four chords into `wave`."),
]

for name, prompt in tasks:
    generate_and_play(name, prompt)
```

### Observations

**[OBSERVE12]** Run the "generative" prompt several times. Does the model write the same
code each time? Does the code always work, or does it sometimes fail? Paste one error
message you get (if any):

&nbsp;

&nbsp;


**[OBSERVE13]**

Write your own prompt. Be specific about what you want — the more musical structure you
describe, the better the output tends to be.

Try:
```python
"Write code that plays a 12-bar blues progression using MIDI notes. "
"Each chord root lasts 1 second. Use midi_to_freq and sine waves."
```

**Your prompt:**

&nbsp;

**Did the code run? What did it produce?**

&nbsp;

&nbsp;

**[REFLECT]** Model 3 is the most flexible but the least reliable. Models 1 and 2
constrained the output format tightly (JSON, ABC). Model 3 gave the LLM almost complete
freedom. What is the tradeoff between **constraint** and **expressiveness** when using
LLMs as creative tools?

&nbsp;

&nbsp;

---

*CS 108 — The Art and Practice of Computer Science | Walla Walla University*