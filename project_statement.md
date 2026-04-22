# CS 108 — Quarter Project Statement
### The Art and Practice of Computer Science | Spring 2026

---

## Overview

The quarter project is the centerpiece of CS 108. It replaces the final exam and accounts for a significant portion of your grade. The goal is simple: **build something real, in public, and be able to talk about it.**

By the end of the quarter you will have a project on GitHub that you created or meaningfully contributed to — something a future employer, graduate program, or collaborator can look at and understand. This is the beginning of your portfolio.

The project kicks off **today, April 8th**. You have approximately nine weeks of active work time before the final showcase on June 5th.

---

## What You Will Build

You have two options:

### Option A — Original Project
Build a new piece of software from scratch. It should solve a real problem, demonstrate something interesting, or express a creative idea. It does not have to be large — it has to be *good*: well-documented, intentionally designed, and something you would genuinely show someone.

### Option B — Open-Source Contribution
Make a meaningful contribution to an existing open-source project. This goes beyond fixing a typo — it means understanding the codebase, engaging with maintainers, and shipping something the project actually merges or seriously considers. A well-documented, thoughtful PR that is ultimately not merged still counts if the work is substantial.

### Option C — Hybrid
Build an original project *and* publish it as an open-source repository that welcomes contributors, or contribute a significant feature or plugin to an existing project that extends its capabilities in a new direction. This is the most ambitious option and is well-suited to groups.

---

## Groups

You may work individually or in groups of up to **3 students**. Groups are expected to produce proportionally more substantial work. Each group member must be able to speak to any part of the project during the final presentation — "that was my teammate's part" is not an acceptable answer.

Group work is coordinated through GitHub. Every meaningful contribution must appear in the commit history under the contributor's own account. Untracked work does not count.

---

## Tools

Your project can use any tools covered in the course:

| Tool | Possible uses |
|------|--------------|
| Pygame | Games, simulations, interactive visualizations |
| Discord.py | Bots, community tools, notification systems |
| Gradio | AI-powered demos, interactive ML interfaces |
| Streamlit | Data dashboards, research tools, portfolio sites |
| Ollama | Local LLM applications, AI writing tools, chatbots |
| Hugging Face | Fine-tuning, dataset creation, model deployment |
| FastAPI | Backend services, AI APIs, data pipelines |
| Blender Python API | Procedural art, 3D tools, animation scripts |
| A-Frame | WebXR experiences, 3D visualizations, interactive scenes |

You are not limited to these tools. If your project idea calls for something else, discuss it with the instructor.

---

## Project Ideas

If you are not sure what to build, here are starting points organized by theme. These are suggestions, not assignments — the best projects come from your own interests.

### Games & Creative Coding
- A Pygame platformer or puzzle game with original level design and a published itch.io page
- A procedurally generated dungeon game where maps are created by a local LLM (Pygame + Ollama)
- A music visualizer that reacts to microphone input in real time
- A generative art tool that produces new images from user-defined style parameters (Diffusers)
- A Blender script that generates an infinite variety of 3D creatures or environments from random seeds

### AI Tools & Applications
- A local study assistant that answers questions about your own notes using RAG + Ollama
- A Gradio app that translates between programming languages using a local model
- A Discord bot that summarizes long threads or answers server-specific questions (Discord.py + Ollama)
- A Streamlit dashboard that lets you explore and compare outputs from different open-source LLMs
- A tool that generates flashcards from a PDF or textbook chapter using Hugging Face pipelines

### Open-Source Contributions
- Add a missing feature or fix a confirmed bug in Pygame, Discord.py, Gradio, or another course tool
- Write a comprehensive tutorial or example for an underserved use case in an OSS project's docs
- Create a new Hugging Face dataset in a domain you care about, with a dataset card and demo Space
- Contribute a new component or template to the Gradio or Streamlit community gallery
- Port or update an existing Pygame example to use the modern Sprite and Group API

### Data & Science
- A Streamlit app that visualizes your college's public data (enrollment, course offerings, research output)
- A tool that tracks and visualizes your own habits or productivity data over time
- A dashboard that pulls live data from a public API (weather, transit, sports, finance) and tells a story
- A Hugging Face Space that demonstrates a fine-tuned sentiment or classification model on domain-specific text

### 3D & Immersive
- An A-Frame WebXR tour of a real or imagined place, with narration or interactive elements
- A Blender script that generates a 3D model from a text description via an Ollama prompt
- An interactive 3D data visualization in A-Frame connected to a live data source via FastAPI

---

## Timeline & Milestones

| Date | Milestone |
|------|-----------|
| **Tue Apr 8** | Project kickoff — start exploring ideas |
| **Mon Apr 13** | **Project proposal due** (see format below) |
| Every Friday | **Weekly journal report due** (see format below) |
| **Fri May 15** | Mid-quarter check-in — live demo to class, peer feedback |
| **Fri May 29** | Code freeze — final README and reflection doc due |
| **Fri Jun 5** | **Final showcase — presentations to class** |

---

## Project Proposal (due Monday, April 13)

Email a single file to me called `proposal`. It should answer the following questions in plain prose (not bullet points):

1. **What are you building?** Describe the project in 2–3 sentences as if explaining it to a friend who is not a CS student.
2. **Why does it matter?** Who would use this, and what problem does it solve or experience does it create?
3. **What tool(s) will you use?** Name the primary library or framework and explain why it is the right choice.
4. **Which option are you pursuing?** Original project, OSS contribution, or hybrid?
5. **Who is on your team?** List names and GitHub usernames. If working solo, say so.
6. **What is your biggest uncertainty?** What part of this are you least sure how to do? (This is not a weakness — it is a plan.)

Proposals are not binding contracts. You may change direction after getting feedback, but major pivots after Week 6 need instructor approval.

---

## Weekly Project Journal Reports (every Friday)

Each Friday, submit a short project journal entry to the course portal before the end of class. This is not a status report — it is a reflective log. Write it in the first person, in plain prose.

**Your entry should cover:**

- **What you did this week on the project.** Be specific: which files did you touch, what problem were you solving, what did you ship or attempt to ship?
- **What you learned.** Something technical, something about your tools, or something about how you work.
- **What blocked you or surprised you.** A bug you spent too long on, a design decision that turned out to be harder than expected, a tool that did not behave as documented.
- **What you plan to do next week.** One or two concrete goals, not a wish list.

Aim for 200–400 words. Quality matters more than length. Journals that say "I worked on my project and made progress" with no specifics will not receive full credit.

Journals are graded on completion and thoughtfulness, not on whether you had a good week. A journal entry about being stuck and not knowing why is more valuable than one that claims everything went smoothly.

> Starting with Journal #3 I will expect to see a Github commit(s) each week with your journal entries.

---

## Final Showcase (June 5)

The final showcase replaces the final exam. Every individual or group will present their project to the class in a **5–7 minute live demo**, followed by 2–3 minutes of questions.

**Your presentation should include:**

1. **The demo.** Show the project running. If it is a web app, open it live. If it is a game, play it. If it is an OSS contribution, show the PR and walk through the diff.
2. **The story.** What did you set out to build? What changed along the way? What would you do differently?
3. **The code.** Show one piece of code you are proud of and explain what it does and why you made the choices you did.
4. **The portfolio artifact.** Show your GitHub repository — the README, the commit history, and what a visitor would see landing on your page for the first time.

You do not need slides. A live demo is worth more than any slide deck.

**If your project is an OSS contribution:** show the PR, the discussion with maintainers, and the diff. If it was merged, celebrate that. If it was not, explain what you learned from the review process.

---

## Grading

| Component | Weight |
|-----------|--------|
| Project proposal | 5% |
| Weekly journal reports (9 entries) | 25% |
| Mid-quarter check-in | 10% |
| Final README and reflection doc | 15% |
| Final showcase presentation | 25% |
| Code quality and GitHub history | 20% |

**Code quality and GitHub history** is assessed on: meaningful commit messages, consistent commit cadence (not one giant commit the night before), a clear README, and code that is readable by someone other than you.

The project is not graded on ambition or complexity. A small, well-executed, well-documented project scores higher than a large, half-finished one.

---

## Expectations

**Time commitment:** The expectation is at least **5 hours of project work per week** — roughly one hour per weekday. This is in addition to class time. Students who treat the project as a Friday-only activity will not finish.

**AI use policy:** You may use AI tools to help you write code, debug, and explore ideas. You may not use AI to write your journal entries or your proposal — those must reflect your own thinking. In your final README, include a short section called "AI use" that describes how you used AI tools in your project and what you learned from doing so.

**Academic integrity:** All code you submit must be code you understand. If you used a code snippet from Stack Overflow, a tutorial, or an AI tool, you must be able to explain every line of it. During the final showcase, instructors may ask you to explain specific parts of your code.

**Minimum viable product:** Every project must have at least one thing that *runs* by the mid-quarter check-in on May 15. "I'm still setting up" is not acceptable at the halfway point.

---

## Getting Started

1. Create a new GitHub repository today. Name it something descriptive — not `cs108project`.
2. Add a `README.md` with your name, a one-line description, and a placeholder for setup instructions.
3. Make your first commit before Friday.
4. Write your proposal and submit it by **Monday, April 13**.

The first commit is often the hardest. Make it today!

---

*Questions? Ask me!*