# GEMMS: A Vision Language Framework For Automated Construction Safety Auditing From Photographs

[![Project Page](https://arifyakupogullari.github.io/GEMMS/)]  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Official repository for the paper:** *"A Vision Language Framework For Automated Construction Safety Auditing From Photographs: GEMMS"* > **Authors:** Ahmet Arif Yakupogullari, Mikael Johansson, Mattias Roupe, Max Bergstrom (Chalmers University of Technology)  
> **Conference:** Creative Construction Conference, 2026, Sopron, Hungary.


---

## 📖 About the Paper

Construction sites are inherently dynamic and mentally demanding, leading to reduced situational awareness and high injury rates (notably struck-by hazards). While general-purpose AI models (like Gemini 2.5) struggle with accurate, autonomous object identification in complex environments, hybrid models can achieve expert-level accuracy. 

**GEMMS** couples two state-of-the-art models to achieve this:
* **The "Retina" (Segment Anything 3 / SAM3):** Handles precise visual grounding and spatial bounding.
* **The "Brain" (Gemma 3):** Handles multi-step spatial reasoning and safety compliance validation.

In our benchmarks, GEMMS achieved an **86.6% positive accuracy rate** on intentional focal imagery. While Large Language Models exhibit an "alarmist" bias (frequently over-predicting hazards), we demonstrate that in the context of safety engineering, this asymmetrical risk profile is highly advantageous. 

---

## ⚙️ How It Works (The 4-Phase Reasoning Engine)

Instead of simply asking an AI "is this safe?", GEMMS executes a strict, structured protocol:
1. **The Assembly Scan:** SAM3 scans the image against a predefined vocabulary to extract a spatial inventory.
2. **The Ghost Scan:** Gemma 3 evaluates structural deficiencies (e.g., a guardrail missing a kickboard).
3. **The Context Filter:** The AI applies active-work logic (e.g., distinguishing between an abandoned cable hazard vs. a cable actively held by a worker).
4. **Relational Reporting:** Outputs a precise `JSON` verdict and draws laser-pointer annotations on the image to highlight the hazard or point to missing required components.

---

## 🚀 Getting Started

### Prerequisites
* **OS:** Ubuntu 20.04/22.04 or Windows 10/11
* **GPU:** NVIDIA GPU with at least 16GB+ VRAM recommended (for running Gemma-3-27b and SAM3 concurrently).
* **Python:** 3.9+
