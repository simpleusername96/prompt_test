# Prompt Robustness Test for LLMs

## Overview
This repository hosts Python code and results for testing the robustness of Large Language Models (LLMs) against paraphrased prompts. It stems from insights in the paper "State of What Art? A Call for Multi-Prompt LLM Evaluation" (https://arxiv.org/pdf/2401.00595.pdf), which identifies the sensitivity of LLMs to minor changes in prompt structure.

## Motivation
Current LLM benchmarks like MMLU(https://github.com/hendrycks/test) cover a diverse array of domains and questions, but they typically utilize only a single, unaltered version of a prompt for each task. This approach could mask the tendency of LLMs to focus on specific phrases and formats, rather than understanding the true intent of prompts. The project addresses this by testing LLMs with multiple paraphrased prompts, better evaluating their understanding of the prompt's core meaning over its surface format.
