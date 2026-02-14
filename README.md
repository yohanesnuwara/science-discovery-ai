# science-discovery-ai
Scientific discovery using agentic AI

<img width="1536" height="1024" alt="Image" src="https://github.com/user-attachments/assets/4f17dee2-3e67-49f2-8a18-1a681f3585e8" />

## Motivation

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
ollama launch opencode

ollama launch opencode --model glm-4.7-flash
```