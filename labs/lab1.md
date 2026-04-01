# CS108
# Instructor: Chiké Abuah
# Lab 1: Getting Started with Ollama & GitHub

---

## Learning Objectives

By the end of this lab, you will be able to:
- Use GitHub correctly
- Connect to the Ollama API using the Python client library
- Send basic requests to a language model
- Begin to understand model parameters (temperature, top-k, top-p)

---

## Prerequisites

- Python 3.10+
- Network Access to the course Ollama server (Via CS Lab)

---


## Resources

- UV docs: https://docs.astral.sh/uv/
- Ollama Python library docs: https://github.com/ollama/ollama-python
- Ollama API documentation: https://github.com/ollama/ollama/blob/main/docs/api.md

---

## Part 0: Setup (10 minutes)

### Server Connection Information

- **Server URL**: `http://ollama.cs.wallawalla.edu:11434`
- **Available models**: gemma3

### Install Required Libraries

I highly recommend `uv`. It (according to their docs): 

```markdown
🚀 Is a single tool to replace pip, pip-tools, pipx, poetry, pyenv, twine, virtualenv, and more.
⚡️ Is 10-100x faster than pip.
🗂️ Provides comprehensive project management, with a universal lockfile.
❇️ Runs scripts, with support for inline dependency metadata.
🐍 Installs and manages Python versions.
```

# Create your workspace and virtual environment

Open your Terminal.

Run the following command:

```bash
unset LD_PRELOAD
```

And don't ask me why! 😅

**Then**

```bash
mkdir -p cs108/lab1
cd cs108/lab1
uv init
uv add ollama
```

---

# Part 0: Set up Github 

## Setting Up Git & GitHub on Linux

---

# What You’ll Learn

* Install Git on Linux
* Configure your identity
* Generate and add SSH keys
* Connect to GitHub
* Create and push your first repository

---

# Verify Installation

```bash
git --version
```

You should see something like:

```
git version 2.x.x
```

---

# Configure Git Identity

Set your name and email (used in commits):

```bash
git config --global user.name "Your Name"
git config --global user.email "your_email@example.com"
```

---

# Check Configuration

```bash
git config --list
```

Look for:

* `user.name`
* `user.email`

---

# Generate SSH Key

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Press Enter to accept defaults.

---

# Key Files Created

* Private key: `~/.ssh/id_ed25519`
* Public key: `~/.ssh/id_ed25519.pub`

⚠️ Never share your private key!

---

# Start SSH Agent

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

---

# Copy Public Key

```bash
cat ~/.ssh/id_ed25519.pub
```

Copy the entire output.

---

# Add Key to GitHub (After creating Github account at github.com)

1. Go to GitHub → Settings
2. Click **SSH and GPG keys**
3. Click **New SSH key**
4. Paste your key
5. Save

---

# Step 8 — Test Connection

```bash
ssh -T git@github.com
```

Expected output:

```
Hi username! You've successfully authenticated...
```

---

# Create a Repository (Local)

```bash
mkdir cs108
cd cs108
git init
```

---

# Add Your First File

```bash
echo "# My Project" > README.md
git add README.md
git commit -m "Initial commit"
```

---

# Create Repo on GitHub

1. Go to GitHub
2. Click **New Repository**
3. Name it (e.g., `cs108`)
4. Do NOT initialize with README

---

# Connect Local to GitHub

```bash
git remote add origin git@github.com:USERNAME/cs108.git
```

---

# Verify Remote

```bash
git remote -v
```

---

# Push to GitHub

```bash
git push -u origin main
```

---

# When you're done editing anything

```bash
git add .
git commit -m "message"
git push
```

---

# Common Commands

```bash
git status
git log
git diff
git pull
```

---

# VS Code 

Automates most git stuff

---

# You’re Ready! 🚀

You now know how to:

* Install Git
* Authenticate with GitHub
* Push code to a repository

---

# Read about Git + Github

https://docs.github.com/en

---


### Configure Server Connection

The Ollama Python library can connect to remote servers by setting the host URL.

Create a file called `test_connection.py`:

```python
import ollama

# Replace with your server URL
client = ollama.Client(host='http://ollama.cs.wallawalla.edu:11434')

def test_connection():
    try:
        # List available models
        models = client.list()
        print("🎉 Connected successfully!")
        print("\nAvailable models:")
        for model in models['models']:
            print(f"  - {model['model']}")
        return True
    except Exception as e:
        print(f"✗ Error connecting: {e}")
        return False

if __name__ == "__main__":
    test_connection()
```

Run it:
```bash
uv run test_connection.py
```

**Expected output:**
```markdown
🎉 Connected successfully!

Available models:
  - t1c/deepseek-math-7b-rl:latest
  - solobsd/llemma-7b:latest
  - qwen2-math:latest
  - cs396:latest
  - cs141:latest
  - qwen2.5-coder:latest
  - qwen3:latest
  - qwen2.5:latest
  - llama3.2:3b
```

---

## Part 1: Basic API Usage (10 minutes)

### 1.1 Create a Helper Module

Create `ollama_client.py`:

```python
import ollama

# Configure your server URL here
SERVER_HOST = 'http://ollama.cs.wallawalla.edu:11434'
client = ollama.Client(host=SERVER_HOST)

def call_ollama(prompt, model="gemma3", **options):
    """
    Send a prompt to the Ollama API.
    
    Args:
        prompt (str): The prompt to send
        model (str): Model name to use
        **options: Additional model parameters (temperature, top_k, etc.)
    
    Returns:
        str: The model's response
    """
    try:
        response = client.generate(
            model=model,
            prompt=prompt,
            options=options
        )
        return response['response']
    
    except Exception as e:
        return f"Error: {e}"

def chat_ollama(messages, model="gemma3", **options):
    """
    Send a chat conversation to the Ollama API.
    
    Args:
        messages (list): List of message dicts with 'role' and 'content'
        model (str): Model name to use
        **options: Additional model parameters
    
    Returns:
        str: The model's response
    """
    try:
        response = client.chat(
            model=model,
            messages=messages,
            options=options
        )
        return response['message']['content']
    
    except Exception as e:
        return f"Error: {e}"

def stream_ollama(prompt, model="gemma3", **options):
    """
    Stream a response from Ollama (for real-time output).
    
    Args:
        prompt (str): The prompt to send
        model (str): Model name to use
        **options: Additional model parameters
    
    Yields:
        str: Chunks of the response as they arrive
    """
    try:
        stream = client.generate(
            model=model,
            prompt=prompt,
            stream=True,
            options=options
        )
        for chunk in stream:
            yield chunk['response']
    
    except Exception as e:
        yield f"Error: {e}"

# Test the function
if __name__ == "__main__":
    test_prompt = "Say 'Hello, World!' and nothing else."
    print("Testing API call...")
    result = call_ollama(test_prompt, temperature=0.1)
    print(f"Response: {result}")
    
    print("\n" + "="*50)
    print("Testing streaming:")
    for chunk in stream_ollama("Count to 5 slowly.", temperature=0.1):
        print(chunk, end='', flush=True)
    print()
```

### 1.2 Run Your First Prompt

Test your helper function:

```bash
uv run ollama_client.py
```

**Checkpoint:** You should see "Hello, World!" and then a count to 5.

---

## Part 2: Understanding Model Parameters (15 minutes)

### 2.1 Temperature Experiments

Create `temperature_test.py`:

```python
from ollama_client import call_ollama

prompt = "Complete this sentence: The weather today is"

print("Testing different temperatures:\n")

for temp in [0.0, 0.5, 1.0, 1.5]:
    print(f"Temperature: {temp}")
    response = call_ollama(
        prompt, 
        temperature=temp, 
        num_predict=20
    )
    print(f"Response: {response}\n")
```

**Task 2.1 (Reflection):** Run this script 3 times. What do you notice?

> Your answer:

---

### 2.2 Top-K and Top-P

Create `sampling_test.py`:

```python
from ollama_client import call_ollama

prompt = "Write a creative opening line for a story:"

print("Testing sampling parameters:\n")

# Test 1: Low top-k (focused)
print("1. Low top-k (focused selection):")
response = call_ollama(
    prompt, 
    temperature=0.8, 
    top_k=10, 
    num_predict=30
)
print(f"{response}\n")

# Test 2: High top-k (diverse)
print("2. High top-k (diverse selection):")
response = call_ollama(
    prompt, 
    temperature=0.8, 
    top_k=50, 
    num_predict=30
)
print(f"{response}\n")

# Test 3: Low top-p (conservative)
print("3. Low top-p (conservative):")
response = call_ollama(
    prompt, 
    temperature=0.8, 
    top_p=0.5, 
    num_predict=30
)
print(f"{response}\n")

# Test 4: High top-p (exploratory)
print("4. High top-p (exploratory):")
response = call_ollama(
    prompt, 
    temperature=0.8, 
    top_p=0.95, 
    num_predict=30
)
print(f"{response}\n")
```

**Task 2.2:** Run this a few times and compare outputs.

---

## Part 3: Putting it all Together (10 minutes)

### 3.1 Final Experiments 


Create `story.py`:

```python
from ollama_client import call_ollama

prompt = "Tell me a cool story"

response = call_ollama(
    prompt, 
    temperature=0.1, 
    top_p=0.7,
    top_k=20,
    num_predict=40
)

print(f"Response: {response}\n")
```

**Run this program a few times and observe the results**

*then change `temperature=0.9,top_p=0.99,top_k=40`*,

**Run this new version of the program a few times and observe the results**

Try adjusting the parameters and experimenting!

---

### 3.3 Save your work and submit

Create a new **public** Github Repository called `cs108`, upload your local `cs108` folder there.

Email the GitHub repository web link to me at `chike.abuah@wallawalla.edu`

*If you're concerned about privacy* 

You can make a **private** Github Repo and add me as a collaborator, my username is `abuach`.

Congrats, you're done with the first lab! 🎉

Let me know when you're done.