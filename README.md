# science-discovery-ai
Scientific discovery using agentic AI

## Setting Runpod

```
ssh-keygen -t ed25519
type %USERPROFILE%\.ssh\id_ed25519.pub
```

Copy paste the SSH key in user setting in Runpod

Deploy a pod, open VS code, CTRL+P Remote-SSH Connect to Host (or install `Remote - SSH` from extension if not installed yet), paste the SSH TCP from Runpod and connect. 

## Install Ollama

Guide: https://docs.runpod.io/tutorials/pods/run-ollama

```
apt update && apt install -y lshw zstd
(curl -fsSL https://ollama.com/install.sh | sh && ollama serve > ollama.log 2>&1) &
```