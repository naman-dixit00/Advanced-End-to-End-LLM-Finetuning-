# Advanced PEFT-Based LLM & Multimodal Training Pipeline

**(Reasoning • RLHF • Multimodal • Flash-Attention • KV Cache)**

> **End-to-End Research-Grade System built with PEFT only**
> No full fine-tuning. No unnecessary compute waste.

---

## Overview

This repository implements a **modular, research-oriented training pipeline** for:

* **Reasoning-focused LLMs**
* **General chat LLMs**
* **Vision-Language Models (VLMs)**

using **Parameter-Efficient Fine-Tuning (PEFT)** techniques and **modern RLHF methods**, designed to be:

* **Google Colab friendly**
* **Single-GPU executable**
* **LLaMA-Factory–inspired**
* **Scalable to multi-node later**

---

## Key Goals

* ❌ No full parameter fine-tuning
* ✅ PEFT only (LoRA family + advanced variants)
* ✅ RLHF (DPO, ORPO, SimPO, short PPO)
* ✅ Reasoning + Multimodal alignment
* ✅ Flash-Attention v2 + KV Cache
* ✅ Clean separation of data, adapters, and models

---

## Supported Models (Fixed)

| Model                | Purpose          |
| -------------------- | ---------------- |
| **DeepSeek-R1 (8B)** | Reasoning & Math |
| **LLaMA-2 (7B)**     | General Chat     |
| **Qwen2-VL**         | Vision-Language  |

> No additional models are used.

---

##  Datasets

This project assumes **private datasets only** (no external mixing).

```
data/
├── sft_llama.json        # LLaMA-2 supervised fine-tuning
├── sft_deepseek.json    # DeepSeek-R1 supervised fine-tuning
├── dpo_deepseek.json    # Preference pairs for DPO
├── vl_qwen.json         # Image-text pairs
└── images/              # Image assets
```

### Dataset Types

* **Instruction SFT** → Behavior learning
* **Preference Pairs (DPO)** → Alignment learning
* **Image-Text Pairs** → Multimodal grounding

---

## Core Techniques Used

### PEFT (No Full Fine-Tuning)

* LoRA / QLoRA
* **AdaLoRA** (dynamic rank allocation)
* **DoRA / MoE-DoRA**
* **Fourier Fine-Tuning**
* Vision-LoRA (for VLMs)

### RLHF

* **DPO** (primary alignment method)
* ORPO / SimPO
* Short-run PPO (optional polish)
* KL-controlled updates

---

## Flash-Attention & KV Cache

| Phase         | Flash-Attention | KV Cache |
| ------------- | --------------- | -------- |
| Training      | ✅ ON            | ❌ OFF    |
| RLHF Sampling | ✅ ON            | ✅ ON     |
| Inference     | ✅ ON            | ✅ ON     |

---

## Training Workflow (High-Level)

```
Private Data
   ↓
Supervised Fine-Tuning (QLoRA + AdaLoRA)
   ↓
Advanced PEFT (MoE-DoRA / Fourier FT)
   ↓
RLHF (DPO / ORPO / SimPO)
   ↓
Optional PPO / KTO
   ↓
LoRA Merge (FP16 base)
   ↓
Optional Quantization
   ↓
Flash-Attention + KV-Cache Inference
   ↓
Evaluation
```

---

## Model-Wise Training Strategy

###  LLaMA-2 (7B)

* QLoRA + AdaLoRA
* MoE-DoRA (top layers only)
* ORPO / SimPO for alignment

### DeepSeek-R1 (8B)

* Light SFT (reasoning priming)
* AdaLoRA
* Fourier Fine-Tuning (last layers)
* **DPO as primary RLHF method**

### Qwen2-VL

* Vision-Tower LoRA only
* Language model frozen
* Multimodal SFT + Multimodal DPO

---

##  Project Structure

```
project/
├── data/
├── configs/
│   ├── sft.yaml
│   ├── dpo.yaml
│   ├── vl.yaml
├── scripts/
│   ├── preprocess.py
│   ├── train_sft.py
│   ├── train_peft.py
│   ├── train_dpo.py
│   ├── train_vl.py
│   ├── merge_lora.py
│   └── infer.py
└── outputs/
```

---

## Strict Constraints

* No full fine-tuning
* No TensorRT / Rapidus
* No LoRA merge on quantized models
* No unnecessary models or datasets

---

## Supported Hardware

* Google Colab Free / Pro
* Single GPU (T4 / L4 / A100)
* CPU fallback for preprocessing only

---

## Evaluation

* MMLU (subset)
* CMMLU
* C-Eval
* Reasoning quality analysis
* Multimodal accuracy checks

---

## Intended Audience

* Applied ML researchers
* LLM engineers
* RLHF practitioners
* Multimodal system builders
* Anyone studying **LLaMA-Factory-style pipelines**

---

## License

This project is released under the **MIT License**.

---

