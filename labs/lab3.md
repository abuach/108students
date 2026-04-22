# CS 108 Lab — Sound Programming: Music from Math

**Name:** _______________________________ **Date:** _____________

---

## Overview

Sound is vibration. Vibration is a wave. A wave is a mathematical function. That means
every sound you have ever heard — a voice, a piano, an explosion — can be represented as
an array of numbers.

In this lab you will build sounds from scratch using math, shape them with envelopes,
layer them into chords and beats, and apply effects. No music theory required. No audio
files needed. Just Python, numpy, and your ears.

## Submission 

Submit all [OBSERVE] questions. Do not submit [REFLECT].

**Install dependencies once:**

```bash
uv add sounddevice numpy scipy matplotlib PyQt6
```

**Test your audio works:**

```python
import sounddevice as sd
import numpy as np

SR = 44100  # sample rate: 44100 samples per second
t = np.linspace(0, 1, SR)
wave = 0.3 * np.sin(2 * np.pi * 440 * t)
sd.play(wave, SR)
sd.wait()
```

If you heard a tone, you're ready.

---

## Setup — Constants and Helpers

Save this as `sound_base.py`. All sections build on it.

```python
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

SR = 44100  # samples per second — standard CD quality

# ── Playback ──────────────────────────────────────────────────────────────────
def play(wave):
    """Play a numpy array as audio."""
    sd.play(wave.astype(np.float32), SR)
    sd.wait()

# ── Save ─────────────────────────────────────────────────────────────────────
def save(wave, filename):
    """Save a numpy array as a 16-bit WAV file."""
    normalized = np.int16(wave / np.max(np.abs(wave)) * 32767)
    write(filename, SR, normalized)
    print(f"Saved: {filename}")

# ── Visualize ─────────────────────────────────────────────────────────────────
def show(wave, title="", duration=0.05):
    """Plot the first `duration` seconds of a wave."""
    samples = int(SR * duration)
    plt.figure(figsize=(10, 2))
    plt.plot(wave[:samples], linewidth=0.8)
    plt.title(title)
    plt.xlabel("samples")
    plt.ylabel("amplitude")
    plt.tight_layout()
    plt.show()

# ── Time axis ─────────────────────────────────────────────────────────────────
def timeline(duration):
    """Return a time array for a given duration in seconds."""
    return np.linspace(0, duration, int(SR * duration))

# ── Sine ──────────────────────────────────────────────────────────────────────
def sine(freq, duration, amplitude=0.3):
    t = timeline(duration)
    return amplitude * np.sin(2 * np.pi * freq * t)

```

---

## Part 1: Synthesis — Building Sound from Math

### Background

Sound is air pressure changing over time. A microphone measures that pressure thousands of
times per second and records it as numbers. A speaker does the reverse — it takes numbers
and moves a cone to push air.

The simplest sound is a **sine wave**: a single frequency oscillating smoothly. Every other
sound — a guitar, a voice, a drum — is just many sine waves added together at different
frequencies and volumes.

### 1A — The Sine Wave

```python
from sound_base import *

tone = sine(440, 2.0)   # 440 Hz = the note A4 (concert pitch)
show(tone, "440 Hz sine wave")
play(tone)
```

Now try these frequencies and note what you hear:

```python
for freq in [220, 440, 880, 1760]:
    print(f"Playing {freq} Hz")
    play(sine(freq, 1.0))
```

**[OBSERVE]** Each frequency is double the previous. What happens to the pitch? How would
you describe the relationship between the number (Hz) and what you hear?

&nbsp;

&nbsp;

### 1B — Waveform Shape Changes Timbre

**Timbre** is the "color" or "texture" of a sound — what makes a piano sound different from
a violin even on the same note. It comes from waveform shape.

```python
freq = 440
dur  = 1.5
t    = timeline(dur)

sine_wave     = 0.3 * np.sin(2 * np.pi * freq * t)
square_wave   = 0.3 * np.sign(np.sin(2 * np.pi * freq * t))
sawtooth_wave = 0.3 * (2 * (t * freq % 1) - 1)
triangle_wave = 0.3 * (2 * np.abs(2 * (t * freq % 1) - 1) - 1)

for wave, name in [
    (sine_wave,     "sine"),
    (square_wave,   "square"),
    (sawtooth_wave, "sawtooth"),
    (triangle_wave, "triangle"),
]:
    show(wave, name)
    play(wave)
```

**[OBSERVE]** All four waves have the same frequency (440 Hz). Do they sound the same pitch? Do they sound like the same instrument? Describe the texture of each.

&nbsp;

&nbsp;

&nbsp;

**[REFLECT]** The sine wave is the "purest" tone. The others sound harsher or buzzier.
Looking at their shapes, what visual property do you think causes that harshness?

&nbsp;

&nbsp;

### 1C — Harmonics: Why Instruments Sound Rich

A real instrument doesn't produce a single sine wave — it produces a **fundamental
frequency** plus **harmonics** (integer multiples) at decreasing volumes. This is what
makes a sound feel "warm" or "full."

```python
def harmonic_tone(freq, duration, num_harmonics=6):
    t = timeline(duration)
    wave = np.zeros(len(t))
    for n in range(1, num_harmonics + 1):
        amplitude = 1.0 / n          # each harmonic is quieter
        wave += amplitude * np.sin(2 * np.pi * freq * n * t)
    wave *= 0.3 / np.max(np.abs(wave))  # normalize
    return wave

plain    = sine(440, 2.0)
rich     = harmonic_tone(440, 2.0, num_harmonics=6)
very_rich = harmonic_tone(440, 2.0, num_harmonics=20)

play(plain)
play(rich)
play(very_rich)
```

**[OBSERVE]** How does the sound change as you add more harmonics? Does the pitch change?
What does change?

&nbsp;

&nbsp;

### 1D — Envelopes: Giving Sound a Shape Over Time

A raw sine wave starts and stops abruptly — it sounds like a computer, not an instrument.
Real sounds **attack** (ramp up), **sustain** (hold), and **release** (fade out).
This shape over time is called an **ADSR envelope**.

```python
def adsr(duration, attack=0.01, decay=0.1, sustain=0.7, release=0.1):
    """Return an ADSR envelope as a numpy array."""
    n = int(SR * duration)
    env = np.zeros(n)

    a = int(SR * attack)
    d = int(SR * decay)
    r = int(SR * release)
    s = n - a - d - r

    env[:a]         = np.linspace(0, 1, a)           # attack
    env[a:a+d]      = np.linspace(1, sustain, d)      # decay
    env[a+d:a+d+s]  = sustain                         # sustain
    env[a+d+s:]     = np.linspace(sustain, 0, r)      # release

    return env

t    = timeline(2.0)
tone = 0.5 * np.sin(2 * np.pi * 440 * t)

abrupt   = tone
shaped   = tone * adsr(2.0, attack=0.01, release=0.1)
piano_like = tone * adsr(2.0, attack=0.005, decay=0.3, sustain=0.3, release=0.4)
pad_like   = tone * adsr(2.0, attack=0.6,  decay=0.1, sustain=0.9, release=0.5)

for wave, name in [
    (abrupt,     "abrupt (no envelope)"),
    (shaped,     "basic envelope"),
    (piano_like, "piano-like"),
    (pad_like,   "slow pad"),
]:
    show(wave, name, duration=2.0)
    play(wave)
```

**[OBSERVE]** The `pad_like` ("slow pad") envelope has a long attack. What does that feel like to listen
to compared to `piano_like`? What kind of instrument or sound does each remind you of?

&nbsp;

&nbsp;

**[REFLECT]** An envelope doesn't change the frequency of a sound. What does it change,
and why does that matter so much to how we perceive it?

&nbsp;

&nbsp;

---

## Part 2: Composition — Sequences, Chords, and Beats

### Background

Music is sound organized in time. In code, that means concatenating and mixing arrays.
A **note** is a short tone with an envelope. A **chord** is multiple notes added together.
A **melody** is notes placed one after another.

No music theory required — we'll use frequency ratios to build everything from scratch.

### 2A — Notes from Frequencies

The Western scale divides an octave into 12 equal steps. Each step multiplies the frequency
by the 12th root of 2. We can compute any note from a base frequency:

```python
def note(freq, duration, amplitude=0.3, waveform='sine'):
    t = timeline(duration)
    if waveform == 'sine':
        wave = np.sin(2 * np.pi * freq * t)
    elif waveform == 'sawtooth':
        wave = 2 * (t * freq % 1) - 1
    elif waveform == 'square':
        wave = np.sign(np.sin(2 * np.pi * freq * t))

    env = adsr(duration, attack=0.02, decay=0.1, sustain=0.6, release=0.15)
    return amplitude * wave * env

def semitones(base_freq, steps):
    """Shift a frequency by `steps` semitones."""
    return base_freq * (2 ** (steps / 12))

# A minor scale starting at A3 (220 Hz)
# Steps: 0, 2, 3, 5, 7, 8, 10, 12
A3 = 220.0
scale_steps = [0, 2, 3, 5, 7, 8, 10, 12]

melody = np.concatenate([
    note(semitones(A3, s), 0.4) for s in scale_steps
])
play(melody)
save(melody, "scale.wav")
```

**[REFLECT]** Does the scale sound familiar? Try reversing it:

```python
play(np.concatenate([note(semitones(A3, s), 0.4) for s in reversed(scale_steps)]))
```

**[REFLECT]** How does the reversed scale feel emotionally compared to the ascending one?

&nbsp;

&nbsp;

### 2B — Chords

A chord is multiple notes played simultaneously — arrays added together.

```python
def chord(freqs, duration, amplitude=0.25):
    waves = [note(f, duration, amplitude) for f in freqs]
    mixed = sum(waves)
    return mixed / np.max(np.abs(mixed)) * amplitude

A3 = 220.0

# Build three chords using frequency ratios
# Major chord: root, major third (+4 semitones), perfect fifth (+7)
# Minor chord: root, minor third (+3 semitones), perfect fifth (+7)

am = chord([semitones(A3, 0), semitones(A3, 3), semitones(A3, 7)], 1.5)  # A minor
C  = chord([semitones(A3, 3), semitones(A3, 7), semitones(A3, 10)], 1.5) # C major
G  = chord([semitones(A3, 10), semitones(A3, 14), semitones(A3, 17)], 1.5) # G major

progression = np.concatenate([am, C, G, am])
play(progression)
save(progression, "chords.wav")
```

**[OBSERVE]** The Am chord uses intervals of 3 and 7 semitones (minor). The C chord uses 4 and 7 (major). Listen carefully — how would you describe the difference between a minor chord and a major chord?

&nbsp;

&nbsp;

### 2C — A Beat Sequencer

A drum machine works by placing short percussive sounds on a rhythmic grid.
We'll build simple drum sounds from noise and sine bursts.

```python
def kick(duration=0.4):
    """Low thump — a sine wave that drops in pitch quickly."""
    t = timeline(duration)
    freq_env = np.exp(-t * 20) * 150 + 50   # pitch drops from 200 to 50 Hz
    wave = np.sin(2 * np.pi * np.cumsum(freq_env) / SR)
    env  = np.exp(-t * 10)
    return 0.8 * wave * env

def snare(duration=0.2):
    """Snappy noise burst — filtered white noise."""
    t = timeline(duration)
    noise = np.random.uniform(-1, 1, len(t))
    env   = np.exp(-t * 20)
    return 0.5 * noise * env

def hihat(duration=0.05):
    """Short high noise click."""
    t = timeline(duration)
    noise = np.random.uniform(-1, 1, len(t))
    env   = np.exp(-t * 60)
    return 0.3 * noise * env

def place(sound, position_seconds, total_length_seconds):
    """Place a sound at a position in a longer buffer."""
    buf   = np.zeros(int(SR * total_length_seconds))
    start = int(SR * position_seconds)
    end   = start + len(sound)
    if end <= len(buf):
        buf[start:end] += sound
    return buf

# One measure = 2 seconds, 8 eighth-note slots
BPM       = 120
beat      = 60 / BPM          # one beat in seconds
measure   = beat * 4           # 4 beats per measure
slot      = beat / 2           # eighth note

# Patterns: 1 = hit, 0 = rest
kick_pat   = [1, 0, 0, 0, 1, 0, 0, 0]
snare_pat  = [0, 0, 1, 0, 0, 0, 1, 0]
hihat_pat  = [1, 1, 1, 1, 1, 1, 1, 1]

buf = np.zeros(int(SR * measure))
for i, hit in enumerate(kick_pat):
    if hit: buf += place(kick(),  i * slot, measure)
for i, hit in enumerate(snare_pat):
    if hit: buf += place(snare(), i * slot, measure)
for i, hit in enumerate(hihat_pat):
    if hit: buf += place(hihat(), i * slot, measure)

buf /= np.max(np.abs(buf))

# Loop it 4 times
beat_loop = np.tile(buf, 4)
play(beat_loop)
save(beat_loop, "beat.wav")
```

**[OBSERVE]** Listen to the beat. Try turning off the hihat by zeroing its pattern:

```python
hihat_pat = [0, 0, 0, 0, 0, 0, 0, 0]
```

How does removing the hihat change the feel of the beat?

&nbsp;

&nbsp;

**[OBSERVE]** Try moving the snare to a different slot, e.g. `snare_pat = [0,0,0,1,0,0,0,1]`.
What changes?

&nbsp;

&nbsp;

**[REFLECT]** The kick, snare, and hihat are built entirely from sine waves and random
noise. Nothing was recorded. What does that tell you about the nature of "real" drum sounds?

&nbsp;

&nbsp;

---

## Part 3: Effects — Shaping Sound in Post

### Background

An **effect** is a transformation applied to an existing audio signal. Reverb, distortion,
chorus, echo — all of these are mathematical operations on arrays of numbers. Understanding
them as math removes the mystery from audio engineering.

### 3A — Echo / Delay

Echo is simply adding a quieter, delayed copy of the signal to itself.

```python
def echo(wave, delay_seconds=0.3, decay=0.5, num_echoes=4):
    delay_samples = int(SR * delay_seconds)
    output = wave.copy()
    for i in range(1, num_echoes + 1):
        pad    = np.zeros(delay_samples * i)
        echo_i = np.concatenate([pad, wave * (decay ** i)])
        # match lengths
        if len(echo_i) > len(output):
            output = np.concatenate([output, np.zeros(len(echo_i) - len(output))])
        output[:len(echo_i)] += echo_i
    return output / np.max(np.abs(output)) * 0.8

dry = np.concatenate([note(440, 0.3), np.zeros(int(SR * 0.5)),
                      note(550, 0.3), np.zeros(int(SR * 0.5))])
wet = echo(dry, delay_seconds=0.2, decay=0.5)

play(dry)
play(wet)
save(wet, "echo.wav")
```

**[OBSERVE]** How does the echo change the perceived space of the sound? Does it feel like
a small room or a large one?

&nbsp;

&nbsp;

Try `delay_seconds=0.05` (very short) and `delay_seconds=0.8` (long). Describe how each feels.

&nbsp;

&nbsp;

### 3B — Reverb (Many Echoes)

True reverb is thousands of overlapping echoes. We can approximate it by adding many short,
randomized delays.

```python
def reverb(wave, room_size=0.3, num_reflections=80):
    output = wave.copy().astype(np.float64)
    max_delay = int(SR * room_size)
    for _ in range(num_reflections):
        delay   = np.random.randint(100, max_delay)
        decay   = np.random.uniform(0.1, 0.4)
        pad     = np.zeros(delay)
        reflect = np.concatenate([pad, wave * decay])
        if len(reflect) > len(output):
            output = np.concatenate([output, np.zeros(len(reflect) - len(output))])
        output[:len(reflect)] += reflect
    return output / np.max(np.abs(output)) * 0.8

dry  = note(440, 1.0, amplitude=0.5)
wet  = reverb(dry, room_size=0.3)
cave = reverb(dry, room_size=1.5, num_reflections=200)

play(dry)
play(wet)
play(cave)
save(cave, "reverb.wav")
```

**[REFLECT]** Compare `wet` (small room) and `cave` (large space). How does the perceived size of the space change?

&nbsp;

&nbsp;

### 3C — Distortion

Distortion **clips** the waveform — it cuts off peaks above a threshold, creating a harsh,
buzzy sound. This is exactly what happens when guitar amplifiers are overdriven.

```python
def distort(wave, gain=10.0, clip=0.3):
    """Amplify then hard-clip."""
    driven = wave * gain
    return np.clip(driven, -clip, clip) * (0.8 / clip)

dry        = note(220, 1.5, amplitude=0.3, waveform='sine')
mild_dist  = distort(dry, gain=3,  clip=0.5)
heavy_dist = distort(dry, gain=20, clip=0.1)

show(dry,        "clean",         duration=0.01)
show(mild_dist,  "mild distortion", duration=0.01)
show(heavy_dist, "heavy distortion", duration=0.01)

play(dry)
play(mild_dist)
play(heavy_dist)
```

**[REFLECT]** Look at the waveforms for clean vs. heavy distortion. What has physically happened to the shape of the wave?

&nbsp;

&nbsp;


### 3D — Low-Pass Filter (Making Sound Muffled)

A **low-pass filter** removes high frequencies, making a sound feel muffled or distant —
like hearing music through a wall.

```python
def lowpass(wave, cutoff_hz=800):
    """Apply a butterworth low-pass filter."""
    nyq = SR / 2
    b, a = butter(4, cutoff_hz / nyq, btype='low')
    return lfilter(b, a, wave)

dry      = note(440, 1.5, waveform='sawtooth')
muffled  = lowpass(dry, cutoff_hz=500)
brighter = lowpass(dry, cutoff_hz=3000)

play(dry)
play(muffled)
play(brighter)
save(muffled, "filtered.wav")
```

**[REFLECT]** The sawtooth wave is naturally bright and buzzy. How does lowpass filtering
at 500 Hz change its character? What frequencies did you lose?

&nbsp;

&nbsp;

**[REFLECT]** Filters, echo, and distortion are all just array math. Knowing that, what do
you think a "vintage analog warmth" plugin in a professional DAW is actually doing?

&nbsp;

&nbsp;

---

### 4 — Putting it all together 

Here's a complete beat-making script with chord progression. It builds a classic lo-fi feel: a swung boom-bap drum pattern, bassline, chord pads, and a simple melodic motif all layered.

What each layer does:

Kick — pitch-swept sine (120 Hz dropping), on beats 1 & 3 (the "boom")
Snare — noise + 180 Hz tone blend with ghost hits for texture
Hi-hats — swung 8th notes (SWING = 0.62) to give it that lazy pocket
Chords — Am → C → G → Am progression, soft sine pads
Bass — root notes an octave below with a small walk-up on beat 4
Melody — pentatonic hook (A C D E G) one octave up

```python
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

SR = 44100
BPM = 85
BEAT = 60 / BPM          # one quarter note in seconds
SWING = 0.62             # >0.5 pushes the "and" beats late (lo-fi swing feel)

# ── Utilities ────────────────────────────────────────────────────────────────

def semitones(freq, n):
    return freq * (2 ** (n / 12))

def tone(freq, dur, amp=0.3, wave="sine"):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    if wave == "sine":
        s = np.sin(2 * np.pi * freq * t)
    elif wave == "saw":
        s = 2 * (t * freq % 1) - 1
    elif wave == "square":
        s = np.sign(np.sin(2 * np.pi * freq * t))
    # Soft envelope (attack / release)
    env = np.ones_like(t)
    a = int(0.01 * SR); r = int(0.15 * SR)
    env[:a] = np.linspace(0, 1, a)
    env[-r:] = np.linspace(1, 0, r)
    return (s * env * amp).astype(np.float32)

def chord(freqs, dur, amp=0.18):
    return sum(tone(f, dur, amp) for f in freqs)

def silence(dur):
    return np.zeros(int(SR * dur), dtype=np.float32)

# ── Drums (synthesized) ───────────────────────────────────────────────────────

def kick(dur=0.35):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    freq = 120 * np.exp(-20 * t)          # pitch drop
    s = np.sin(2 * np.pi * freq * t)
    env = np.exp(-8 * t)
    return (s * env * 0.9).astype(np.float32)

def snare(dur=0.18):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    noise = np.random.randn(len(t))
    tone_s = np.sin(2 * np.pi * 180 * t)
    env = np.exp(-25 * t)
    return ((noise * 0.6 + tone_s * 0.4) * env * 0.55).astype(np.float32)

def hihat(dur=0.05, amp=0.12):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    noise = np.random.randn(len(t))
    env = np.exp(-60 * t)
    return (noise * env * amp).astype(np.float32)

# ── Pattern builder ───────────────────────────────────────────────────────────

def place(buf, clip, start_sec):
    i = int(start_sec * SR)
    end = min(i + len(clip), len(buf))
    buf[i:end] += clip[:end - i]

def make_bar(beats=4):
    """One bar of boom-bap with swing 8th notes."""
    bar_len = int(BEAT * beats * SR)
    buf = np.zeros(bar_len, dtype=np.float32)

    # Kick on 1 and 3
    for b in [0, 2]:
        place(buf, kick(), b * BEAT)

    # Snare on 2 and 4
    for b in [1, 3]:
        place(buf, snare(), b * BEAT)

    # Swung 8th-note hi-hats
    straight = BEAT / 2
    swung    = BEAT * SWING
    offbeat  = BEAT * (1 - SWING)
    for b in range(beats):
        place(buf, hihat(),       b * BEAT)                   # on the beat
        place(buf, hihat(amp=0.07), b * BEAT + swung)         # swung "and"

    # Ghost snare (quiet, add texture)
    place(buf, snare() * 0.2, 0.75 * BEAT)
    place(buf, snare() * 0.15, 2.5  * BEAT)

    return buf

# ── Chords & Bass ─────────────────────────────────────────────────────────────

A3 = 220.0

def make_chords(dur=BEAT * 4):
    am = chord([semitones(A3, 0), semitones(A3, 3), semitones(A3, 7)], dur)
    C  = chord([semitones(A3, 3), semitones(A3, 7), semitones(A3, 10)], dur)
    G  = chord([semitones(A3, 10), semitones(A3, 14), semitones(A3, 17)], dur)
    return np.concatenate([am, C, G, am])  # 4 bars

def make_bass():
    """Root notes an octave below, on beat 1 of each chord bar."""
    roots = [A3 / 2, semitones(A3, 3) / 2, semitones(A3, 10) / 2, A3 / 2]
    bars = []
    for root in roots:
        bar = silence(BEAT * 4)
        note = tone(root, BEAT * 0.9, amp=0.45, wave="sine")
        place(bar, note, 0)
        # Quick walk-up on beat 4
        walk = tone(semitones(root, 2), BEAT * 0.4, amp=0.3, wave="sine")
        place(bar, walk, BEAT * 3)
        bars.append(bar)
    return np.concatenate(bars)

def make_melody():
    """Simple pentatonic hook over the 4-bar loop."""
    # A minor pentatonic intervals (in semitones)
    penta = [0, 3, 5, 7, 10]

    # use indices into penta (NOT semitone values)
    pattern = [0, 2, 3, 2, 4, 3, 2, 0]

    notes = []
    for i, deg in enumerate(pattern):
        freq = semitones(A3 * 2, penta[deg])
        dur  = BEAT * 0.9 if i % 2 == 0 else BEAT * 0.45

        notes.append(tone(freq, dur, amp=0.12, wave="sine"))
        notes.append(silence(BEAT * 0.1))

    melody = np.concatenate(notes)

    full = silence(BEAT * 16)
    place(full, melody, 0)
    return full

# ── Assemble ──────────────────────────────────────────────────────────────────

BARS = 4
drums   = np.concatenate([make_bar() for _ in range(BARS)])
chords  = make_chords()
bass    = make_bass()
melody  = make_melody()

# Trim to shortest (floating-point length differences)
n = min(len(drums), len(chords), len(bass), len(melody))
mix = drums[:n] + chords[:n] + bass[:n] + melody[:n]

# Soft limiter
mix = np.tanh(mix * 1.2) * 0.85

print("Playing… (Ctrl+C to stop)")
sd.play(mix, SR)
sd.wait()
```

**[REFLECT]** Try tweaking:

BPM = 85 — drop to 80 for even more slumped energy
SWING = 0.62 — push toward 0.67 for more shuffle, back to 0.5 for straight
Change wave="sine" to "saw" on the chords for a grittier, less clean pad

&nbsp;

&nbsp;

*CS 108 — The Art and Practice of Computer Science | Walla Walla University*