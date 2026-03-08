
Genesis Directive

Genesis Directive is a two-stage neural architecture that combines a diffusion-based latent planner with an autoregressive language model decoder through a learned projection bridge.

The core hypothesis is that generation quality and structural coherence improve when the decoder is conditioned on an explicit latent plan produced by a diffusion language model.

This repository explores whether a diffusion-generated latent trajectory can act as a soft structural directive for autoregressive text generation.

⸻

Overview

Most large language models generate text token by token, conditioning each new token only on previous context.

Humans typically write differently:
we first form a high-level outline, then fill in the details.

Genesis Directive introduces a system that explicitly separates:

What to say → latent planning (diffusion model)
How to say it → autoregressive generation

The architecture allows the decoder to condition on a structured latent prefix before generating text.

⸻

Key Idea

A frozen diffusion-style planner produces a latent sequence representation:

Z ∈ R^(L × d_p)

This representation is projected into the decoder embedding space:

Ẑ = gφ(LayerNorm(Z))

The projected plan is then prepended as a soft prefix to the decoder’s token embeddings.

The decoder generates text conditioned on:

[ Ẑ ; Embed(y) ]

The projection bridge is trained using a self-shuffle contrastive objective that teaches the bridge to encode token order and structure.

Importantly:

The decoder is never backpropagated through.

This avoids numerical instability in mixed-precision training.

⸻

Architecture

Prompt x
   │
   ▼
Diffusion Planner (frozen)
GPT-2 Large
   │
   ▼
Latent Plan Z
   │
   ▼
Projection Bridge gφ
   │
   ▼
Projected Plan Ẑ
   │
   ▼
Autoregressive Decoder
GPT-2 XL + LoRA
   │
   ▼
Generated Text

Training only updates the projection bridge parameters.

⸻

Core Contributions

1. Two-Stage Planning Architecture

Genesis Directive introduces a framework where a diffusion planner provides a latent structural plan that guides an autoregressive decoder.

⸻

2. Projection Bridge

A learned projection maps planner representations into decoder embedding space.

Four projection families were tested:
	•	Standard MLP
	•	Wide MLP
	•	Residual MLP
	•	Gated MLP

The gated projection showed the strongest structure sensitivity.

⸻

3. Order-Aware Self-Shuffle Contrastive Loss

We introduce a contrastive objective that forces the projected latent plan to encode token order.

L = max(0, m − ||Ẑ − Ẑπ||²_F)

Where Ẑπ is a randomly permuted version of the latent plan.

This ensures the decoder becomes sensitive to structural corruption.

⸻

4. Mixed-Precision Training Protocol (Apple Silicon)

Training frozen fp16 LLMs with soft prefixes on Apple MPS produces NaN gradients due to attention kernel instability.

Genesis Directive introduces a training protocol that eliminates this issue:
	•	All models run in eval() mode
	•	Only projection layers are in train()
	•	Projection trained entirely in fp32
	•	No backward pass through the decoder

This produced 0 NaN failures across all experiments.

⸻

Experiments

Experiments were performed using:
	•	Planner: GPT-2 Large
	•	Decoder: GPT-2 XL (LoRA)
	•	Dataset: WikiText-2
	•	Training set: 40 pairs
	•	Evaluation set: 15 pairs

The goal was architectural validation, not benchmark optimization.

⸻

Key Results

Attention Attribution

The decoder actively attends to the latent plan.

Region	Attention Mass
Latent Prefix	53.5%
Text Tokens	46.5%

This confirms the architectural coupling is active.

⸻

Structure Sensitivity

We measure a wrong-structure penalty:

S = loss_shuffle − loss_zero

Where:
	•	shuffle = permuted latent plan
	•	zero = no plan provided

Positive S means the decoder penalizes incorrect structure more than missing structure.

⸻

Best Result

After 300 contrastive steps:

S = +1.354 nats

This shows the decoder strongly rejects structurally corrupted plans.

⸻

Experimental Findings

Key observations:
	•	Decoder attention shifts strongly toward the latent prefix
	•	Contrastive training increases structural sensitivity
	•	Gated projections learn more effectively
	•	Mixed precision stability issues can be completely avoided

⸻

Limitations

Current experiments are intentionally small-scale:
	•	40 training pairs
	•	15 evaluation pairs
	•	short training schedules (40–300 steps)

The goal is proof-of-architecture, not generation performance.

Future work should explore:
	•	larger corpora
	•	longer training
	•	downstream tasks
	•	human evaluation of generated text

⸻

Related Ideas

Genesis Directive relates to several existing approaches:

Diffusion Language Models

Models like LLaDA perform masked diffusion over token sequences to generate latent representations.

⸻

Prefix Tuning

Prefix tuning prepends trainable embeddings to steer generation.

Genesis Directive replaces these static prefixes with dynamic latent plans.

⸻

Plan-Then-Generate Systems

Some reasoning frameworks generate intermediate plans before decoding.

Genesis Directive externalizes planning into a separate model entirely.

⸻

Conclusion

Genesis Directive demonstrates that coupling latent planning with autoregressive decoding can introduce structural awareness into language generation.

Our experiments show:
	•	The decoder attends heavily to latent plans
	•	Shuffled plans degrade generation
	•	Structural sensitivity emerges after contrastive training
	•	Mixed-precision training on Apple Silicon can be stabilized

The framework opens a path toward explicit planning in LLM generation systems, separating:

What to say  → latent planner
How to say it → autoregressive decoder

Potential applications include:
	•	long-form generation
	•	dialogue systems
	•	structured text generation
	•	reasoning pipelines

⸻

