import json
from pathlib import Path
import textwrap


def to_source(text: str) -> list[str]:
    cleaned = textwrap.dedent(text).rstrip("\n")
    if not cleaned:
        return []
    return [line + "\n" for line in cleaned.splitlines()]


def md_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": to_source(text)}


def code_cell(code: str, hidden: bool = False) -> dict:
    metadata = {}
    if hidden:
        metadata = {
            "tags": ["solution", "hide-input"],
            "jupyter": {"source_hidden": True},
        }
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": to_source(code),
    }


def mini_task_cells(task: dict) -> list[dict]:
    header = md_cell(
        f"### Mini Task – {task['title']}\n\n"
        f"{task['prompt']}\n\n"
        "Try the starter cell before revealing the hidden solution."
    )
    starter = code_cell(task["starter"])
    solution = code_cell(task["solution"], hidden=True)
    return [header, starter, solution]


def final_exercise_cells(spec: dict) -> list[dict]:
    header = md_cell(
        f"## Comprehensive Exercise – {spec['title']}\n\n"
        f"{spec['prompt']}\n\n"
        "Complete the starter template, then compare with the sample solution."
    )
    starter = code_cell(spec["starter"])
    solution = code_cell(spec["solution"], hidden=True)
    return [header, starter, solution]


def build_notebook(spec: dict) -> dict:
    cells: list[dict] = []

    cells.append(
        md_cell(
            f"# {spec['title']}\n\n"
            f"{spec['intro']}\n\n"
            "_Note: Network access is disabled in this environment. References reflect best practices "
            "current through October 2024._"
        )
    )

    objectives = "\n".join(f"- {item}" for item in spec["objectives"])
    cells.append(md_cell(f"## Learning Objectives\n\n{objectives}"))

    for section in spec["sections"]:
        if section["type"] == "md":
            cells.append(md_cell(section["content"]))
        elif section["type"] == "code":
            cells.append(code_cell(section["content"]))

    for task in spec.get("mini_tasks", []):
        cells.extend(mini_task_cells(task))

    if "final_exercise" in spec:
        cells.extend(final_exercise_cells(spec["final_exercise"]))

    further_reading = "\n".join(f"- {item}" for item in spec["further_reading"])
    cells.append(md_cell(f"## Further Reading\n\n{further_reading}"))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


NOTEBOOK_SPECS: list[dict] = [
    {
        "path": "notebooks/beginner/01_pytorch_overview.ipynb",
        "title": "PyTorch Overview & Workflow",
        "intro": (
            "This notebook orients you around tensors, autograd, and the big-picture training loop so every "
            "later implementation decision makes sense."
        ),
        "objectives": [
            "Understand how data, models, losses, and optimizers connect.",
            "Manipulate tensors and gradients in eager execution.",
            "Describe the deep learning lifecycle you'll refine throughout the course.",
            "Connect fundamentals to later notebooks on data pipelines, model composition, and optimization.",
        ],
        "sections": [
            {
                "type": "md",
                "content": (
                    "## The Big Picture\n\n"
                    "PyTorch projects revolve around four building blocks:\n\n"
                    "1. **Data pipeline** – Fetch, transform, and batch examples.\n"
                    "2. **Model definition** – Compose differentiable blocks with `nn.Module`.\n"
                    "3. **Optimization loop** – Compute losses, backpropagate gradients, update parameters.\n"
                    "4. **Evaluation & iteration** – Inspect metrics, visualize behavior, and refine assumptions."
                ),
            },
            {
                "type": "code",
                "content": (
                    "import torch\n\n"
                    "torch.manual_seed(0)\n\n"
                    "inputs = torch.tensor([[0.5, 1.0, -0.5]])\n"
                    "weights = torch.randn(3, 1, requires_grad=True)\n"
                    "bias = torch.zeros(1, requires_grad=True)\n\n"
                    "prediction = inputs @ weights + bias\n"
                    "target = torch.tensor([[1.0]])\n"
                    "loss = torch.nn.functional.mse_loss(prediction, target)\n"
                    "loss.backward()\n\n"
                    "print(f\"Prediction: {prediction.item():.3f}\")  # expected scalar near 0\n"
                    "print(weights.grad)\n"
                    "print(bias.grad)\n"
                ),
            },
            {
                "type": "md",
                "content": (
                    "### Autograd Checklist\n\n"
                    "1. Build the computation graph as you execute Python code.\n"
                    "2. Produce a scalar loss describing model quality.\n"
                    "3. Call `backward()` to compute gradients.\n"
                    "4. Inspect and reset gradients before the next iteration."
                ),
            },
            {
                "type": "code",
                "content": (
                    "import matplotlib.pyplot as plt\n"
                    "import matplotlib.patches as patches\n\n"
                    "fig, ax = plt.subplots(figsize=(8, 3))\n"
                    "ax.axis(\"off\")\n\n"
                    "boxes = [\n"
                    "    (0.05, 0.55, 0.25, 0.3, \"Data\\n(DataLoader)\"),\n"
                    "    (0.35, 0.55, 0.25, 0.3, \"Model\\n(nn.Module)\"),\n"
                    "    (0.65, 0.55, 0.25, 0.3, \"Loss\"),\n"
                    "    (0.35, 0.15, 0.25, 0.25, \"Optimizer\"),\n"
                    "]\n\n"
                    "for x, y, w, h, label in boxes:\n"
                    "    ax.add_patch(\n"
                    "        patches.FancyBboxPatch(\n"
                    "            (x, y), w, h, boxstyle=\"round,pad=0.03\", edgecolor=\"#1f77b4\", facecolor=\"#dce8ff\", linewidth=2\n"
                    "        )\n"
                    "    )\n"
                    "    ax.text(x + w / 2, y + h / 2, label, ha=\"center\", va=\"center\", fontsize=11)\n\n"
                    "arrows = [\n"
                    "    ((0.30, 0.70), (0.35, 0.70)),\n"
                    "    ((0.60, 0.70), (0.65, 0.70)),\n"
                    "    ((0.78, 0.55), (0.55, 0.35)),\n"
                    "    ((0.35, 0.40), (0.20, 0.55)),\n"
                    "]\n\n"
                    "for (x0, y0), (x1, y1) in arrows:\n"
                    "    ax.annotate(\"\", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle=\"->\", linewidth=2))\n\n"
                    "ax.text(0.52, 0.80, \"forward\", ha=\"center\")\n"
                    "ax.text(0.52, 0.62, \"loss\", ha=\"center\")\n"
                    "ax.text(0.47, 0.34, \"backward\", ha=\"center\")\n"
                    "ax.text(0.22, 0.47, \"step\", ha=\"center\")\n"
                    "plt.show()\n"
                ),
            },
        ],
        "mini_tasks": [
            {
                "title": "Tensor Warm-up",
                "prompt": (
                    "Construct a tensor with three samples and two features, apply a linear transformation, "
                    "compute the mean prediction, and verify gradients exist."
                ),
                "starter": (
                    "import torch\n\n"
                    "torch.manual_seed(12)\n\n"
                    "# TODO: create tensor `x` with shape (3, 2)\n"
                    "# TODO: initialize weights (2, 1) and bias (1,) with requires_grad=True\n"
                    "# TODO: compute predictions, mean value, and call backward()\n"
                ),
                "solution": (
                    "import torch\n\n"
                    "torch.manual_seed(12)\n\n"
                    "x = torch.randn(3, 2)\n"
                    "weights = torch.randn(2, 1, requires_grad=True)\n"
                    "bias = torch.zeros(1, requires_grad=True)\n\n"
                    "preds = x @ weights + bias\n"
                    "mean_pred = preds.mean()\n"
                    "mean_pred.backward()\n\n"
                    "print(preds)\n"
                    "print(f\"Mean prediction: {mean_pred.item():.3f}\")\n"
                    "print(weights.grad, bias.grad)\n"
                ),
            }
        ],
        "final_exercise": {
            "title": "Linear Regression From Scratch",
            "prompt": (
                "Train a simple linear regression model without `nn.Linear`. Generate synthetic data, run multiple "
                "epochs, track losses, and report the learned parameters."
            ),
            "starter": (
                "import torch\n\n"
                "torch.manual_seed(42)\n\n"
                "true_w, true_b = 2.5, -0.8\n"
                "x = torch.linspace(-2, 2, steps=64).unsqueeze(1)\n"
                "y = true_w * x + true_b + 0.3 * torch.randn_like(x)\n\n"
                "w = torch.randn(1, requires_grad=True)\n"
                "b = torch.zeros(1, requires_grad=True)\n"
                "optimizer = torch.optim.SGD([w, b], lr=0.1)\n\n"
                "num_epochs = 200\n"
                "history = []\n\n"
                "for epoch in range(num_epochs):\n"
                "    # TODO: forward pass, loss, backward, optimizer step, grad reset\n"
                "    pass\n\n"
                "print(f\"Learned parameters -> w: {w.item():.3f}, b: {b.item():.3f}\")\n"
            ),
            "solution": (
                "import torch\n\n"
                "torch.manual_seed(42)\n\n"
                "true_w, true_b = 2.5, -0.8\n"
                "x = torch.linspace(-2, 2, steps=64).unsqueeze(1)\n"
                "y = true_w * x + true_b + 0.3 * torch.randn_like(x)\n\n"
                "w = torch.randn(1, requires_grad=True)\n"
                "b = torch.zeros(1, requires_grad=True)\n"
                "optimizer = torch.optim.SGD([w, b], lr=0.1)\n\n"
                "num_epochs = 200\n"
                "history = []\n\n"
                "for epoch in range(num_epochs):\n"
                "    preds = x * w + b\n"
                "    loss = torch.nn.functional.mse_loss(preds, y)\n"
                "    loss.backward()\n"
                "    optimizer.step()\n"
                "    optimizer.zero_grad()\n"
                "    history.append(loss.item())\n\n"
                "print(f\"Learned parameters -> w: {w.item():.3f}, b: {b.item():.3f}\")\n"
            ),
        },
        "further_reading": [
            "PyTorch Documentation: https://pytorch.org/docs/stable/index.html",
            "PyTorch Tutorials: https://pytorch.org/tutorials/",
            "“Deep Learning with PyTorch: A 60 Minute Blitz”",
        ],
    },
    {
        "path": "notebooks/beginner/02_data_pipeline.ipynb",
        "title": "Data Pipelines with PyTorch",
        "intro": (
            "Feed your models efficiently by designing dependable dataset and dataloader abstractions."
        ),
        "objectives": [
            "Differentiate built-in datasets from custom `Dataset` implementations.",
            "Compose transforms and augmentations that integrate cleanly with the training loop.",
            "Tune `DataLoader` parameters to maximize throughput.",
            "Prepare for later notebooks on CNNs, sequence models, and transformers.",
        ],
        "sections": [
            {
                "type": "md",
                "content": (
                    "## Designing the Pipeline\n\n"
                    "A robust pipeline ingests raw data, applies deterministic transforms, introduces stochastic"
                    " augmentations, batches examples, and overlaps CPU work with accelerator execution."
                ),
            },
            {
                "type": "code",
                "content": (
                    "import torch\n"
                    "from torch.utils.data import Dataset, DataLoader\n"
                    "import numpy as np\n\n"
                    "class TabularDataset(Dataset):\n"
                    "    def __init__(self, features, targets):\n"
                    "        self.x = torch.from_numpy(features).float()\n"
                    "        self.y = torch.from_numpy(targets).float().unsqueeze(-1)\n\n"
                    "    def __len__(self):\n"
                    "        return len(self.x)\n\n"
                    "    def __getitem__(self, idx):\n"
                    "        return self.x[idx], self.y[idx]\n\n"
                    "rng = np.random.default_rng(7)\n"
                    "features = rng.normal(size=(128, 3))\n"
                    "targets = (features @ np.array([0.5, -1.2, 2.0]) + 0.3).astype(np.float32)\n\n"
                    "dataset = TabularDataset(features, targets)\n"
                    "loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)\n\n"
                    "xb, yb = next(iter(loader))\n"
                    "print(xb.shape, yb.shape)\n"
                ),
            },
            {
                "type": "code",
                "content": (
                    "import matplotlib.pyplot as plt\n\n"
                    "fig, axes = plt.subplots(1, 3, figsize=(9, 3))\n"
                    "for idx, ax in enumerate(axes):\n"
                    "    ax.hist(features[:, idx], bins=15, color=\"#4c72b0\", alpha=0.75)\n"
                    "    ax.set_title(f\"Feature {idx}\")\n"
                    "fig.suptitle(\"Feature Distributions\", fontsize=14)\n"
                    "plt.tight_layout()\n"
                    "plt.show()\n"
                ),
            },
        ],
        "mini_tasks": [
            {
                "title": "Padding Variable-Length Sequences",
                "prompt": (
                    "Implement a collate function that pads token sequences to the same length and returns"
                    " both the padded tensor and original lengths."
                ),
                "starter": (
                    "from torch.utils.data import Dataset, DataLoader\n"
                    "import torch\n\n"
                    "toy = [[1, 2, 3], [4, 5], [6]]\n\n"
                    "class ToyDataset(Dataset):\n"
                    "    def __len__(self):\n"
                    "        return len(toy)\n\n"
                    "    def __getitem__(self, idx):\n"
                    "        return torch.tensor(toy[idx], dtype=torch.long)\n\n"
                    "# TODO: create collate_fn(batch) -> (padded_tensor, lengths)\n"
                ),
                "solution": (
                    "from torch.nn.utils.rnn import pad_sequence\n"
                    "from torch.utils.data import Dataset, DataLoader\n"
                    "import torch\n\n"
                    "toy = [[1, 2, 3], [4, 5], [6]]\n\n"
                    "class ToyDataset(Dataset):\n"
                    "    def __len__(self):\n"
                    "        return len(toy)\n\n"
                    "    def __getitem__(self, idx):\n"
                    "        return torch.tensor(toy[idx], dtype=torch.long)\n\n"
                    "def collate_fn(batch):\n"
                    "    lengths = torch.tensor([item.size(0) for item in batch])\n"
                    "    padded = pad_sequence(batch, batch_first=True, padding_value=0)\n"
                    "    return padded, lengths\n\n"
                    "loader = DataLoader(ToyDataset(), batch_size=3, collate_fn=collate_fn)\n"
                    "padded, lengths = next(iter(loader))\n"
                    "print(padded)\n"
                    "print(lengths)\n"
                ),
            }
        ],
        "final_exercise": {
            "title": "Vision Pipeline Blueprint",
            "prompt": (
                "Generate synthetic RGB images, apply augmentations, normalize, and create train/validation"
                " loaders. Outline how you would swap in a real dataset such as CIFAR-10."
            ),
            "starter": (
                "import torch\n"
                "from torch.utils.data import Dataset, DataLoader, random_split\n"
                "import torchvision.transforms as T\n\n"
                "class SyntheticImages(Dataset):\n"
                "    def __init__(self, num_images=200):\n"
                "        self.data = torch.rand(num_images, 3, 32, 32)\n"
                "        self.targets = torch.randint(0, 10, (num_images,))\n"
                "        # TODO: define train/eval transforms\n"
                "\n"
                "    def __len__(self):\n"
                "        return len(self.data)\n\n"
                "    def __getitem__(self, idx):\n"
                "        # TODO: apply transforms and return (image, label)\n"
                "        raise NotImplementedError\n"
                "\n"
                "# TODO: split dataset and create loaders\n"
            ),
            "solution": (
                "import torch\n"
                "from torch.utils.data import Dataset, DataLoader, random_split\n"
                "import torchvision.transforms as T\n\n"
                "class SyntheticImages(Dataset):\n"
                "    def __init__(self, num_images=200):\n"
                "        self.data = torch.rand(num_images, 3, 32, 32)\n"
                "        self.targets = torch.randint(0, 10, (num_images,))\n"
                "        self.train_t = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4), T.Normalize([0.5]*3, [0.5]*3)])\n"
                "        self.eval_t = T.Compose([T.Normalize([0.5]*3, [0.5]*3)])\n"
                "        self.training = True\n\n"
                "    def train(self, mode=True):\n"
                "        self.training = mode\n"
                "        return self\n\n"
                "    def __len__(self):\n"
                "        return len(self.data)\n\n"
                "    def __getitem__(self, idx):\n"
                "        t = self.train_t if self.training else self.eval_t\n"
                "        return t(self.data[idx]), self.targets[idx]\n\n"
                "dataset = SyntheticImages(200)\n"
                "train_ds, val_ds = random_split(dataset, [160, 40], generator=torch.Generator().manual_seed(21))\n"
                "dataset.train(True)\n"
                "train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)\n"
                "dataset.train(False)\n"
                "val_loader = DataLoader(val_ds, batch_size=32)\n"
                "print(len(train_loader), len(val_loader))\n"
            ),
        },
        "further_reading": [
            "PyTorch Data Loading: https://pytorch.org/docs/stable/data.html",
            "TorchData datapipes for streaming scenarios",
            "NVIDIA DALI when GPU-driven preprocessing becomes necessary",
        ],
    },
    {
        "path": "notebooks/beginner/03_building_models.ipynb",
        "title": "Composing Models with nn.Module",
        "intro": "Structure reusable modules so you can experiment quickly and scale to complex architectures.",
        "objectives": [
            "Implement custom `nn.Module` classes and combine them with `nn.Sequential`.",
            "Apply normalization, activation, and dropout layers judiciously.",
            "Introduce residual connections to stabilize deeper networks.",
            "Prepare module patterns for CNNs, transformers, and Mixture-of-Experts models.",
        ],
        "sections": [
            {
                "type": "md",
                "content": (
                    "## Modular Design Principles\n\n"
                    "Build models as compositions of smaller, testable blocks: encoders generate representations,"
                    " heads map to outputs, and utility layers keep signals well-behaved."
                ),
            },
            {
                "type": "code",
                "content": (
                    "import torch\n"
                    "import torch.nn as nn\n\n"
                    "class SimpleMLP(nn.Module):\n"
                    "    def __init__(self, input_dim, hidden_dim, output_dim):\n"
                    "        super().__init__()\n"
                    "        self.net = nn.Sequential(\n"
                    "            nn.Linear(input_dim, hidden_dim),\n"
                    "            nn.ReLU(),\n"
                    "            nn.Linear(hidden_dim, hidden_dim),\n"
                    "            nn.ReLU(),\n"
                    "            nn.Linear(hidden_dim, output_dim),\n"
                    "        )\n\n"
                    "    def forward(self, x):\n"
                    "        return self.net(x)\n\n"
                    "model = SimpleMLP(3, 16, 1)\n"
                    "dummy = torch.randn(8, 3)\n"
                    "out = model(dummy)\n"
                    "print(out.shape)\n"
                ),
            },
            {
                "type": "code",
                "content": (
                    "class ResidualMLPBlock(nn.Module):\n"
                    "    def __init__(self, dim, hidden_dim, dropout=0.1):\n"
                    "        super().__init__()\n"
                    "        self.norm = nn.LayerNorm(dim)\n"
                    "        self.ff = nn.Sequential(\n"
                    "            nn.Linear(dim, hidden_dim),\n"
                    "            nn.GELU(),\n"
                    "            nn.Dropout(dropout),\n"
                    "            nn.Linear(hidden_dim, dim),\n"
                    "        )\n\n"
                    "    def forward(self, x):\n"
                    "        return x + self.ff(self.norm(x))\n\n"
                    "block = ResidualMLPBlock(32, 64)\n"
                    "print(block(torch.randn(4, 32)).shape)\n"
                ),
            },
        ],
        "mini_tasks": [
            {
                "title": "Linear-Norm-Activation Block",
                "prompt": "Build a module that applies Linear -> BatchNorm1d -> GELU and returns both outputs and pre-activations.",
                "starter": (
                    "import torch.nn as nn\n\n"
                    "class LinearNormActivation(nn.Module):\n"
                    "    def __init__(self, in_dim, out_dim):\n"
                    "        super().__init__()\n"
                    "        # TODO: define layers\n\n"
                    "    def forward(self, x):\n"
                    "        # TODO: return (post_activation, pre_activation)\n"
                    "        raise NotImplementedError\n"
                ),
                "solution": (
                    "import torch.nn as nn\n\n"
                    "class LinearNormActivation(nn.Module):\n"
                    "    def __init__(self, in_dim, out_dim):\n"
                    "        super().__init__()\n"
                    "        self.linear = nn.Linear(in_dim, out_dim)\n"
                    "        self.norm = nn.BatchNorm1d(out_dim)\n"
                    "        self.act = nn.GELU()\n\n"
                    "    def forward(self, x):\n"
                    "        pre = self.linear(x)\n"
                    "        normed = self.norm(pre)\n"
                    "        out = self.act(normed)\n"
                    "        return out, pre\n"
                ),
            }
        ],
        "final_exercise": {
            "title": "Configurable Feedforward Network",
            "prompt": (
                "Create an MLP that accepts a list of hidden dimensions, optional dropout, and residual connections"
                " when consecutive layer widths match. Return intermediate activations for debugging."
            ),
            "starter": (
                "class ConfigurableMLP(nn.Module):\n"
                "    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.0):\n"
                "        super().__init__()\n"
                "        # TODO: build layers and residual bookkeeping\n\n"
                "    def forward(self, x):\n"
                "        raise NotImplementedError\n\n"
                "    def forward_with_intermediates(self, x):\n"
                "        # TODO: return (output, activations)\n"
                "        raise NotImplementedError\n"
            ),
            "solution": (
                "import torch.nn as nn\n\n"
                "class ConfigurableMLP(nn.Module):\n"
                "    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.0):\n"
                "        super().__init__()\n"
                "        dims = [input_dim] + list(hidden_dims) + [output_dim]\n"
                "        self.layers = nn.ModuleList()\n"
                "        self.residual_flags = []\n"
                "        for idx in range(len(dims) - 1):\n"
                "            in_dim, out_dim = dims[idx], dims[idx + 1]\n"
                "            self.layers.append(nn.Linear(in_dim, out_dim))\n"
                "            if idx < len(dims) - 2:\n"
                "                self.layers.append(nn.LayerNorm(out_dim))\n"
                "                self.layers.append(nn.GELU())\n"
                "                if dropout > 0:\n"
                "                    self.layers.append(nn.Dropout(dropout))\n"
                "                self.residual_flags.append(in_dim == out_dim)\n"
                "        self.residual_flags.append(False)\n\n"
                "    def forward(self, x):\n"
                "        out, _ = self.forward_with_intermediates(x)\n"
                "        return out\n\n"
                "    def forward_with_intermediates(self, x):\n"
                "        activations = []\n"
                "        residual = x\n"
                "        flag_idx = 0\n"
                "        idx = 0\n"
                "        while idx < len(self.layers):\n"
                "            layer = self.layers[idx]\n"
                "            x = layer(x)\n"
                "            idx += 1\n"
                "            if idx < len(self.layers) and isinstance(self.layers[idx], nn.LayerNorm):\n"
                "                norm = self.layers[idx]\n"
                "                act = self.layers[idx + 1]\n"
                "                x = act(norm(x))\n"
                "                idx += 2\n"
                "                if idx < len(self.layers) and isinstance(self.layers[idx], nn.Dropout):\n"
                "                    x = self.layers[idx](x)\n"
                "                    idx += 1\n"
                "                if self.residual_flags[flag_idx]:\n"
                "                    x = x + residual\n"
                "                residual = x\n"
                "                flag_idx += 1\n"
                "            activations.append(x)\n"
                "        return x, activations\n"
            ),
        },
        "further_reading": [
            "PyTorch `nn` Module Reference: https://pytorch.org/docs/stable/nn.html",
            "He et al. – Deep Residual Learning for Image Recognition",
            "FastAI advice on building reusable module blocks",
        ],
    },
    # Additional notebook specs will be appended here...
]


def write_notebooks(base_path: Path) -> None:
    for spec in NOTEBOOK_SPECS:
        nb_path = base_path / spec["path"]
        nb_path.parent.mkdir(parents=True, exist_ok=True)
        nb = build_notebook(spec)
        with nb_path.open("w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=True, indent=2)


if __name__ == "__main__":
    write_notebooks(Path(__file__).parent)
