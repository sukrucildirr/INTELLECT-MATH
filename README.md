<p align="center">
<h2 align="center">Notus-7B: State-of-the-Art Mathematical Reasoning through Better Initializations for Reinforcement Learning</h1>
</p>

<p align="center">
<a href="">🔗 Blog Post</a>
<span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
<a href="">🐦 X / Twitter</a>
<span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
<a href="">🤗 Hugging Face</a>
</p>


This repository contains code for reproducing Notus-7B, a state-of-the-art 7B parameter model for mathematical reasoning. 

By fine-tuning on synthetic reasoning data obtained from a strong teacher model (QwQ-32B), we can provide a better policy initialization for online reinforcement learning based on PRIME-RL. This way, we can match the performance of the existing SOTA model Eurus-2-7B-Prime with 10x less time spent on reinforcement learning, and outperform the model with further training.

## Reproducing Notus-7B

You can reproduce Notus-7B step by step using the following code:

1. `synthetic-data`: First, generate synthetic data from QwQ. We used this code to generate our SFT dataset [PrimeIntellect/Notus-7B-SFT-Data](https://huggingface.co/datasets/PrimeIntellect/Notus-7B-SFT-Data)
2. `sft`: Next, fine-tune your SFT model with [open-instruct](https://github.com/allenai/open-instruct) using your synthetic dataset. We trained [PrimeIntellect/Notus-7B-SFT](https://huggingface.co/PrimeIntellect/Notus-7B-SFT) using [Qwen/Qwen2.5-Math-7B](https://huggingface.co/Qwen/Qwen2.5-Math-7B) as a base model.
- `rl-and-evals`: Finally, train your SFT model with reinforcement learning using [PRIME-RL](https://github.com/PRIME-RL/PRIME). This last step was used to train [PrimeIntellect/Notus-7B](https://huggingface.co/PrimeIntellect/Notus-7B) from [PrimeIntellect/Notus-7B-SFT](https://huggingface.co/PrimeIntellect/Notus-7B-SFT).

