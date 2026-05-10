# CS 108 Lab — AI-Generated Music with Ollama

**Name:** _______________________________ **Date:** _____________

---

## Overview

Large language models are trained on text — including vast amounts of text *about* music:
sheet music notation, MIDI descriptions, music theory textbooks, ABC files, and code that
generates sound. That means an LLM has latent knowledge of melody, rhythm, and harmony,
even though it has never heard a single note.

In this lab you will extract that musical knowledge three ways: as structured data, as a
formal notation, and as code. In each case, Ollama generates text and Python turns it into
sound.

**Install once:**

```bash
uv add ollama mido pygame numpy scipy music21
```

**Shared synth helpers** — save this as `synth.py` and keep it in the same folder.
All three models import from it.

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

## Model 1: Ollama → JSON → Sound

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
            {"role": "system",  "content": SYSTEM},
            {"role": "user",    "content": prompt},
        ]
    )
    return response["message"]["content"]

def parse_json(text):
    # Strip any accidental markdown fences
    text = re.sub(r"```[a-z]*", "", text).strip()
    # Find the first JSON array in the response
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array found in response:\n{text}")
    return json.loads(match.group())

def render(notes, waveform='sine'):
    fragments = []
    for n in notes:
        freq = midi_to_freq(int(n["note"]))
        dur  = float(n["duration"])
        amp  = float(n["velocity"]) / 127 * 0.4
        fragments.append(note(freq, dur, amplitude=amp, waveform=waveform))
    return np.concatenate(fragments)

# ── Prompts to try ────────────────────────────────────────────────────────────
prompts = [
    "Compose a gentle 12-note lullaby melody in C major.",
    "Compose an 16-note tense, minor-key melody that builds in intensity.",
    "Compose a 14-note melody that sounds like walking through a forest.",
]

for prompt in prompts:
    raw  = ask_ollama(prompt)
    print(f"Raw response:\n{raw}\n")
    try:
        notes = parse_json(raw)
        print(f"Parsed {len(notes)} notes.")
        wave  = render(notes, waveform='sine')
        play(wave)
        filename = prompt[:20].replace(" ", "_") + ".wav"
        save(wave, filename)
    except Exception as e:
        print(f"Parse error: {e}")
```

### Observations

**[OBSERVE]** Print the raw JSON before parsing it. Does the model always return clean
JSON, or does it sometimes wrap it in markdown or add explanation text? What does the
`re.sub` line do to handle that?

&nbsp;

&nbsp;

**[OBSERVE]** Run the same prompt twice. Do you get the same melody? What does that tell
you about how LLMs generate sequences?

&nbsp;

&nbsp;

### Exploration

Try changing the `waveform` argument in `render()` from `'sine'` to `'sawtooth'` or
`'square'`. The same notes, different timbre.

Now try writing your own prompt. Ask for something specific — a genre, a mood, a cultural
style. Paste the prompt and describe what you got:

**Your prompt:**

&nbsp;

**What the model composed (describe the melody):**

&nbsp;

&nbsp;

**[REFLECT]** The model has never heard music. It learned melody from reading text *about*
music. What does it mean to "know" how to compose a lullaby without having ears?

&nbsp;

&nbsp;

---

## Model 2: Ollama → ABC Notation → MIDI → Sound

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

**[OBSERVE]** Print the raw ABC before parsing. ABC has a very specific format — headers
must come first, note syntax is strict. Does the model get it right on the first try?
What kinds of errors does it make?

&nbsp;

&nbsp;

**[OBSERVE]** Compare a melody from Model 1 (JSON) to one from Model 2 (ABC). Which
sounds more "musical"? What does ABC encoding give the model that raw note numbers don't?

&nbsp;

&nbsp;

### Exploration

Ask Ollama for a melody in a specific cultural style. ABC notation has key signatures
and modes (Dorian, Mixolydian, etc.) that produce different emotional flavors.

Try:
```python
"Compose a melody in Dorian mode — dark but not quite minor."
"Compose a waltz melody in 3/4 time."
```

**[OBSERVE]** Does the model understand musical modes? Does the Dorian melody actually
sound different from a standard minor key melody?

&nbsp;

&nbsp;

**[REFLECT]** ABC notation encodes musical structure that JSON note lists don't — meter,
key, phrasing. Does the added structure produce better music? What is lost compared to
the JSON approach?

&nbsp;

&nbsp;

---

## Model 3: Ollama → Python Code → Sound

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

**[OBSERVE]** Read the generated code before it runs. Does it look like Python you would
write? Does it use the provided functions correctly, or does it invent new ones?

&nbsp;

&nbsp;

**[OBSERVE]** Run the "generative" prompt several times. Does the model write the same
code each time? Does the code always work, or does it sometimes fail? Paste one error
message you get (if any):

&nbsp;

&nbsp;

**[OBSERVE]** When the code fails, read the error message carefully. Is the mistake a
syntax error, a wrong function name, a shape mismatch? What does that tell you about
where the model's musical knowledge breaks down?

&nbsp;

&nbsp;

### Exploration

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

## Synthesis: Comparing the Three Approaches

Fill in the table based on your experience with all three models:

| | Model 1 (JSON) | Model 2 (ABC) | Model 3 (Code) |
|---|---|---|---|
| **Reliability** (did it always work?) | | | |
| **Musicality** (did it sound good?) | | | |
| **Controllability** (easy to steer?) | | | |
| **Surprise factor** (unexpected results?) | | | |

**[REFLECT]** All three models use the same underlying LLM. The only difference is the
output format you requested. What does that tell you about the role of *prompting* versus
the role of the *model* in AI-generated output?

&nbsp;

&nbsp;

---

## Portfolio Submission

Pick one model and one prompt. Tune the prompt until you produce a piece you find
genuinely interesting. Save the output wav.

**Which model?**

&nbsp;

**Your final prompt:**

&nbsp;

**Artist statement — what were you going for, what did the AI give you, and what
surprised you?**

&nbsp;

&nbsp;

&nbsp;

---

*CS 108 — The Art and Practice of Computer Science | Walla Walla University*