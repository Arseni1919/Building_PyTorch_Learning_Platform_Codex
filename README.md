# PyTorch Mastery Course

Become fluent in PyTorch through a structured, hands-on journey that moves from foundational concepts to state-of-the-art deep learning techniques. The course follows a top-down, project-friendly approach so you always see the big picture before diving into details.

> **Note:** Network access is restricted in this environment, so all guidance reflects best practices and APIs current through October 2024. Refer to the linked official PyTorch docs for the latest updates.

## How to Use This Course
- Work through the notebooks in order; each topic builds on prior material and includes quick references to earlier lessons.
- Every notebook mixes conceptual overviews, annotated code labs, visual explanations, and mini coding tasks.
- Solutions live in hidden cells tagged as `solution`. In JupyterLab, enable *View → Cell Toolbar → Tags* or the *Hide Input* extension to reveal them after attempting each exercise.
- Each notebook ends with a comprehensive exercise tying together the lesson’s skills. Treat these as mini-projects to cement understanding.
- Use the “Further Reading” sections to dive deeper into official docs, papers, and community resources.

## Prerequisites
- Comfortable with core Python, NumPy, and basic linear algebra
- A working PyTorch installation (`>=2.2`)
- Recommended tooling: JupyterLab or VS Code’s notebook interface, GPU access for intermediate/advanced labs

## Course Layout

### Beginner Level – Fundamentals
1. **`notebooks/01_beginner/01_pytorch_overview.ipynb`** – Big-picture tour of tensors, automatic differentiation, and the PyTorch workflow.
2. **`notebooks/01_beginner/02_data_pipeline.ipynb`** – Datasets, DataLoaders, transforms, and batching strategies for different data modalities.
3. **`notebooks/01_beginner/03_building_models.ipynb`** – `nn.Module` patterns, linear/MLP models, initialization, and modular design.
4. **`notebooks/01_beginner/04_training_workflows.ipynb`** – Training loops, optimizers, checkpoints, experiment structure, and debugging basics.
5. **`notebooks/01_beginner/05_loss_functions.ipynb`** – Comprehensive guide to PyTorch loss functions, when to use each, and custom loss design.

### Intermediate Level – Core Deep Learning
1. **`notebooks/02_intermediate/01_computer_vision_cnns.ipynb`** – CNN architectures, transfer learning, and visual diagnostics.
2. **`notebooks/02_intermediate/02_sequence_modeling.ipynb`** – Embeddings, RNNs, LSTMs/GRUs, and sequence-to-sequence workflows.
3. **`notebooks/02_intermediate/03_attention_fundamentals.ipynb`** – Self-, cross-, and multi-head attention with step-by-step implementations.
4. **`notebooks/02_intermediate/04_transformer_architecture.ipynb`** – Encoder-decoder stacks, positional encoding, masking, and training tips.
5. **`notebooks/02_intermediate/05_training_best_practices.ipynb`** – Optimization schedules, mixed precision, gradient clipping, and troubleshooting.

### Advanced/Professional Level – Modern Architectures
1. **`notebooks/03_advanced/01_efficient_attention.ipynb`** – Flash Attention, memory-efficient attention patterns, and benchmarking.
2. **`notebooks/03_advanced/02_mixture_of_experts.ipynb`** – Sparse MoE layers, routing strategies, and load-balancing considerations.
3. **`notebooks/03_advanced/03_large_scale_training.ipynb`** – Distributed data/model/pipeline parallelism and fault-tolerant training.
4. **`notebooks/03_advanced/04_self_supervised_and_finetuning.ipynb`** – Contrastive/diffusion overviews, downstream fine-tuning, and representation learning links.
5. **`notebooks/03_advanced/05_production_and_monitoring.ipynb`** – Exporting models, serving, tracking drift, and responsible AI checks.

## Capstone Suggestions
- Re-implement a recent architecture (e.g., ViT, LLaMA-style decoder) with your own data pipeline and training loop.
- Build an end-to-end application (vision, NLP, multimodal) and deploy it with continuous evaluation hooks.
- Contribute enhancements or bug fixes to an open-source PyTorch project, documenting your debugging process.

## Staying Current
- Official PyTorch Docs: <https://pytorch.org/docs/stable/index.html>
- Tutorials & Recipes: <https://pytorch.org/tutorials/>
- PyTorch Forums: <https://discuss.pytorch.org/>
- Papers With Code: <https://paperswithcode.com/methods/category/pytorch>
- NVIDIA Developer Blog & Meta AI Research for updates on attention and distributed tooling

Experiment boldly, document your learnings, and revisit earlier notebooks to reinforce concepts as you progress. Happy building!
