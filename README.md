# SciSandboxAI
Agentic scientific discovery with your favorite AI-assisted coding CLIs (Claude Code or OpenCode) running on Ollama and LLMs at no-cost. 

<img width="1536" height="1024" alt="Image" src="https://github.com/user-attachments/assets/4f17dee2-3e67-49f2-8a18-1a681f3585e8" />

## Motivation

Companies are burning dollars investing on agentic AI tools to accelerate scientific discovery of new proteins and new materials. What if we can build an agentic scientific discovery on coding assistants at no-cost? What if coding assistants can create 1,000+ lines of codes of science research, the same amount as it creates to accelerate software development (so-called vibe coding)?

Software engineers don't need GPU to build powerful softwares, but researchers need GPU to build powerful research. The key of agentic scientific discovery is GPU + vibe coding + open source LLM, as sandbox for experimenters. 

This repository tries to prove this. 

Read more: 

<details>
  <summary><b>Agentic AI in scientific discovery</b></summary>

  Agentic AI for science is increasingly framed as an **autonomous discovery workflow**—hypothesis generation, experiment/simulation design, tool-mediated execution, analysis, and iterative refinement—implemented with LLMs/multimodal models plus research platforms across life sciences, chemistry, materials science, and physics. [1]  
  In **theoretical computer science and mathematics** (e.g., approximation algorithms and combinatorics), agentic systems contribute by reframing problems, proposing intermediate lemmas, generating and running code to validate conjectures or small cases, and tightening proofs via feedback loops (including “cross-pollination” from other mathematical domains). [3]  
  In **autonomous chemistry and automated wet-lab discovery**, agentic planners have been coupled to robotic synthesis and analytical instrumentation so the agent can design multi-step reactions, execute them via lab hardware/software APIs, and interpret results to close the “design–make–test” loop. [4]  
  In **materials informatics and materials property prediction**, agentic approaches such as retrieval-grounded reasoning agents can query high-fidelity databases, run atomistic simulations, and iteratively refine candidate compositions/properties to support materials discovery and optimization. [5]  
  In **nuclear science / chemical separations R&D**, agentic AI is used to develop and optimize machine-learning models with HPC-backed experimentation/simulation, accelerating the iterative design and tuning of separation processes within an enterprise research platform. [6]  
  In **cheminformatics and computational methods more broadly**, “agentic science” is positioned as (semi-)autonomous agents that reason and plan while interacting with digital and physical environments, with emphasis on responsible integration into scientific practice. [7]  
  Across **biomedicine, spatial biology, gene-expression analysis, and materials science**, the agentic AI landscape is being systematized through curated collections of “agent scientists” (e.g., spatial-biology agents, multi-agent genomics workflows, and LLM-enabled materials scientists), which helps researchers reuse architectures and compare capabilities across domains. [2]  

  **References**  
  [1] https://huggingface.co/papers/2508.14111  
  [2] https://github.com/AgenticScience/Awesome-Agent-Scientists  
  [3] https://arxiv.org/pdf/2602.03837  
  [4] https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1649155/full  
  [5] https://arxiv.org/pdf/2503.08979  
  [6] https://azure.microsoft.com/en-us/blog/transforming-rd-with-agentic-ai-introducing-microsoft-discovery/?msockid=15e293f7f224671800c385a1f30766c5  
  [7] https://www.nature.com/articles/s42256-025-01110-x

</details>

## How-To

### Setting Runpod

```
ssh-keygen -t ed25519

type %USERPROFILE%\.ssh\id_ed25519.pub
```

Copy paste the SSH key in user setting in Runpod

Deploy a pod, open VS code, CTRL+P Remote-SSH Connect to Host (or install `Remote - SSH` from extension if not installed yet), paste the SSH TCP from Runpod and connect. 

### Clone this repository

```
git clone https://github.com/yohanesnuwara/science-discovery-ai.git
```

### Install UV

UV is used to manage packages needed for the AI sandbox

```
curl -LsSf https://astral.sh/uv/install.sh | sh

source $HOME/.local/bin/env
```

Then activate the virtual environment and sync 

```
source .venv/bin/activate

uv sync
```

### Install Ollama

Guide: https://docs.runpod.io/tutorials/pods/run-ollama

```
apt update && apt install -y lshw zstd

(curl -fsSL https://ollama.com/install.sh | sh && ollama serve > ollama.log 2>&1) &
```

### Install Claude Code

Guide: https://code.claude.com/docs/en/setup

```
curl -fsSL https://claude.ai/install.cmd -o install.cmd && install.cmd && del install.cmd

echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc

ollama launch claude
```

### Install OpenCode

Guide: 

```
curl -fsSL https://opencode.ai/install | bash

source ~/.bashrc

ollama launch opencode --model glm-4.7-flash
```

### Spin up service in Runpod

Port number must be set in Runpod

```
opencode web --hostname 0.0.0.0 --port 4090
```

## Use cases

SciSandboxAI has been tested for several use cases:

<details>
  <summary><b>**Geophysics**: Earthquake probability prediction (Model: GLM-4.7 Flash)</b></summary>

  The next earthquake is (almost impossible) to predict. However, there may be possible approach. Here, 10,000 records of global eartquake for the last 10 years, and global fault coordinate are downloaded. Using this data, moment rate, stress drop, and stress accumulation are calculated. Using the calculated parameters, 
  statistical (Bayesian inference) is used to calculate the probability of occurrence in the next 2, 5, and 10 years
  for every plate; P(x,t) probability as function of space and time. 

  <img width="5345" height="2336" alt="Image" src="https://github.com/user-attachments/assets/5db58682-6a8e-4823-9000-3e8d84dcc879" />

  <img width="5345" height="2336" alt="Image" src="https://github.com/user-attachments/assets/e7edfa8b-04b5-467d-80ce-cca4bfd84b68" />

  <img width="5345" height="2336" alt="Image" src="https://github.com/user-attachments/assets/942caf85-9984-4e55-a4e0-a4f156bed9ca" />

</details>

<details>
  <summary><b>**Macroeconomics**: Global financial crisis prediction (Model: Kimi K-2.5)</b></summary>

  Financial crises are rare events (only 65 episodes in 150 years) that are notoriously difficult to predict. Here, three databases (JST Macrohistory, ESRB Financial Crises, GPR Geopolitical Risk) are integrated—though country coding incompatibility limited ESRB data usage. Using credit growth, yield spreads, and macro-financial indicators from 18 advanced economies (1870-2020), an ensemble model (MLP+LSTM+Random Forest) with heavy regularization calculates the probability of crisis occurrence within the next 5 years; AUCPR 0.420 vs baseline 0.137 (+206% improvement). However, insufficient training data (only 7 matched countries from ESRB) and severe class imbalance (13.7% positive) limited generalization, with the model overfitting to Japan's 1990s crisis patterns. This is a good start for such data-driven macroeconomic research.

  <img width="5970" height="3569" alt="Image" src="https://github.com/user-attachments/assets/ac1f4a26-bf3e-4fc6-969f-da52e7cfb166" />

</details>