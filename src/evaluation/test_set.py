"""Hand-crafted evaluation set: 20 Q&A pairs over cs.LG papers (2023-2024).

Mix of factual, comparative and synthesis questions. ``ground_truth`` answers
are concise reference answers used by RAGAS context_recall.
"""
from __future__ import annotations

TEST_SET: list[dict[str, str]] = [
    {
        "question": "What methods have been proposed to reduce hallucination in large language models?",
        "ground_truth": (
            "Approaches include retrieval-augmented generation, RLHF, "
            "chain-of-thought prompting, self-consistency, constrained/grounded "
            "decoding, and factuality-focused fine-tuning."
        ),
    },
    {
        "question": "How does LoRA reduce the cost of fine-tuning large language models?",
        "ground_truth": (
            "LoRA freezes the pretrained weights and injects trainable low-rank "
            "decomposition matrices into each layer, drastically cutting the number "
            "of trainable parameters and optimizer memory."
        ),
    },
    {
        "question": "What is the difference between LoRA and QLoRA?",
        "ground_truth": (
            "QLoRA adds 4-bit NormalFloat quantization of the frozen base model, "
            "double quantization, and paged optimizers, enabling LoRA fine-tuning of "
            "much larger models on a single GPU."
        ),
    },
    {
        "question": "What are common parameter-efficient fine-tuning (PEFT) techniques?",
        "ground_truth": (
            "PEFT methods include LoRA, adapters, prefix tuning, prompt tuning, "
            "(IA)^3, and BitFit; they update a small subset of parameters."
        ),
    },
    {
        "question": "How does retrieval-augmented generation improve factual accuracy?",
        "ground_truth": (
            "RAG retrieves relevant documents and conditions generation on them, "
            "grounding outputs in external evidence and reducing reliance on "
            "parametric memory."
        ),
    },
    {
        "question": "What techniques improve the training efficiency of transformers?",
        "ground_truth": (
            "Techniques include FlashAttention, mixed-precision training, gradient "
            "checkpointing, ZeRO/optimizer sharding, and efficient attention variants."
        ),
    },
    {
        "question": "What is FlashAttention and why is it faster?",
        "ground_truth": (
            "FlashAttention is an IO-aware exact attention algorithm that tiles the "
            "computation and avoids materializing the full attention matrix in HBM, "
            "reducing memory reads/writes and improving speed."
        ),
    },
    {
        "question": "How do mixture-of-experts models scale parameters efficiently?",
        "ground_truth": (
            "MoE models route each token to a small subset of expert subnetworks, "
            "increasing total parameters while keeping per-token compute roughly "
            "constant through sparse activation."
        ),
    },
    {
        "question": "What approaches exist for aligning language models with human preferences?",
        "ground_truth": (
            "Alignment methods include RLHF with PPO, direct preference optimization "
            "(DPO), reward modeling, constitutional AI, and rejection sampling."
        ),
    },
    {
        "question": "How does direct preference optimization (DPO) differ from RLHF?",
        "ground_truth": (
            "DPO optimizes a classification-style loss directly on preference pairs "
            "without training a separate reward model or running RL, making it simpler "
            "and more stable than PPO-based RLHF."
        ),
    },
    {
        "question": "What methods are used for in-context learning and prompting?",
        "ground_truth": (
            "Methods include few-shot prompting, chain-of-thought, self-consistency, "
            "tree-of-thought, and instruction tuning."
        ),
    },
    {
        "question": "How is chain-of-thought prompting used to improve reasoning?",
        "ground_truth": (
            "Chain-of-thought prompts the model to produce intermediate reasoning "
            "steps before the final answer, improving performance on arithmetic, "
            "commonsense and symbolic reasoning tasks."
        ),
    },
    {
        "question": "What are common benchmarks for evaluating large language models?",
        "ground_truth": (
            "Benchmarks include MMLU, HellaSwag, GSM8K, BIG-bench, HumanEval, "
            "TruthfulQA and MT-Bench."
        ),
    },
    {
        "question": "What techniques reduce the memory footprint of LLM inference?",
        "ground_truth": (
            "Techniques include weight quantization (INT8/INT4), KV-cache "
            "quantization, paged attention, distillation, and pruning."
        ),
    },
    {
        "question": "How does quantization affect model accuracy and efficiency?",
        "ground_truth": (
            "Quantization lowers numerical precision of weights/activations to cut "
            "memory and speed up inference, with methods like GPTQ and AWQ minimizing "
            "the accuracy loss."
        ),
    },
    {
        "question": "What are recent advances in multimodal models that combine vision and language?",
        "ground_truth": (
            "Advances include contrastive vision-language pretraining (CLIP), "
            "vision-language adapters, and instruction-tuned multimodal LLMs that "
            "connect image encoders to language models."
        ),
    },
    {
        "question": "What safety techniques are used to prevent harmful outputs from LLMs?",
        "ground_truth": (
            "Safety techniques include RLHF, red-teaming, constitutional AI, content "
            "filtering/guardrails, and refusal training."
        ),
    },
    {
        "question": "How do diffusion models generate high-quality images?",
        "ground_truth": (
            "Diffusion models learn to reverse a gradual noising process, denoising "
            "from random noise to data; latent diffusion operates in a compressed "
            "latent space for efficiency."
        ),
    },
    {
        "question": "What methods improve the context length of transformer models?",
        "ground_truth": (
            "Methods include rotary position embeddings with interpolation, ALiBi, "
            "sparse/local attention, and recurrence or memory-augmented architectures."
        ),
    },
    {
        "question": "What are the main approaches to knowledge distillation for compressing models?",
        "ground_truth": (
            "Approaches include response-based (soft-label) distillation, "
            "feature-based distillation, and self-distillation, where a smaller "
            "student mimics a larger teacher."
        ),
    },
]

assert len(TEST_SET) == 20, "Test set must contain exactly 20 Q&A pairs"
