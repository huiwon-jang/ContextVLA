<h1 align="center"> ContextVLA: Vision-Language-Action Model with Amortized Multi-Frame Context </h1>
<div align="center">
  <a href="https://huiwon-jang.github.io/" target="_blank">Huiwon&nbsp;Jang</a><sup>1,2</sup>
  &ensp; <b>&middot;</b> &ensp;
  <a href="https://sihyun.me/" target="_blank">Sihyun&nbsp;Yu</a><sup>1</sup>
  &ensp; <b>&middot;</b> &ensp;
  <a href="https://arunos728.github.io/" target="_blank">Heeseung&nbsp;Kwon</a><sup>1,2</sup>
  <br>
  Hojin Jeon<sup>1</sup>
  &ensp; <b>&middot;</b> &ensp;
  <a href="https://younggyo.me/" target="_blank">Younggyo&nbsp;Seo</a><sup>3*</sup>
  &ensp; <b>&middot;</b> &ensp;
  <a href="https://alinlab.kaist.ac.kr/shin.html" target="_blank">Jinwoo&nbsp;Shin</a><sup>1,2*</sup>
  <br>
  <sup>1</sup> KAIST &emsp; <sup>2</sup>RLWRLD &emsp; <sup>3</sup>UC Berkeley &emsp; *Equal Advising<br>
</div>
<h3 align="center">[<a href="https://huiwon-jang.github.io/contextvla/">project page</a>] [<a href="https://arxiv.org/abs/2510.04246">arxiv</a>]</h3>

### 1. Environment Setup
```bash
conda create -n contextvla python=3.11 -y
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
cd qwen_fast/packages/openpi-client && pip install -e . && cd ../../..
```

### 2. OXE Pre-trained Checkpoints
We provide checkpoints of ContextVLA pretrained on <a href="https://github.com/dusty-nv/openvla/blob/main/prismatic/vla/datasets/rlds/oxe/mixtures.py" target="_blank">OXE Magic Soup</a> in the below. If the link is blocked, please contact me.
- <a href="https://huggingface.co/huiwon/ContextVLA-3B-Qwen2.5VL-FAST" target="_blank">ContextVLA-3B-Qwen2.5VL-FAST</a> (180K iter pretrained with batch size of 128)

### 3. Fine-tuning
E.g., Fine-tuning ContextVLA on Libero dataset

(1) `qwen_fast/src/training/config.py`
 - Set `_CONFIGS`: TrainConfig.data.repo_id (e.g., "physical-intelligence/libero")
 - See <a href="https://github.com/Physical-Intelligence/openpi/tree/main?tab=readme-ov-file#fine-tuning-base-models-on-your-own-data" target="_blank">Physical-intelligence/openpi</a> for more details.

(2) run training script

```bash
cd qwen_fast
bash train_scripts/train_contextvla_libero.sh
```

### 4. Evaluation
TODO

### 5. Fine-tuning Pi0 / Pi0-FAST / GR00T N1.5 using ContextVLA
TODO

### Acknowledgement
This code is mainly built upon <a href="https://github.com/Physical-Intelligence/openpi/tree/main" target="_blank">Physical-intelligence/openpi</a>. We also appreciate <a href="https://github.com/declare-lab/nora" target="_blank">NORA</a> for providing pre-training code implementation using OXE Magic Soup dataset.
