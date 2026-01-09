"""Model cards for physics-aware neural networks.

Model cards provide structured metadata about physics models,
including dimensional information, training details, and usage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import json

from ..core.dimensions import Dimension
from .registry import ModelInfo


@dataclass
class TrainingInfo:
    """Information about model training.

    Attributes:
        dataset: Name of training dataset.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Initial learning rate.
        optimizer: Optimizer used.
        physics_loss_weight: Weight for physics loss term.
        data_loss_weight: Weight for data loss term.
        conservation_laws: List of conservation laws enforced.
        notes: Additional training notes.
    """

    dataset: str = ""
    epochs: int = 0
    batch_size: int = 32
    learning_rate: float = 1e-3
    optimizer: str = "Adam"
    physics_loss_weight: float = 0.0
    data_loss_weight: float = 1.0
    conservation_laws: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dataset": self.dataset,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "physics_loss_weight": self.physics_loss_weight,
            "data_loss_weight": self.data_loss_weight,
            "conservation_laws": self.conservation_laws,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingInfo":
        """Create from dictionary."""
        return cls(
            dataset=data.get("dataset", ""),
            epochs=data.get("epochs", 0),
            batch_size=data.get("batch_size", 32),
            learning_rate=data.get("learning_rate", 1e-3),
            optimizer=data.get("optimizer", "Adam"),
            physics_loss_weight=data.get("physics_loss_weight", 0.0),
            data_loss_weight=data.get("data_loss_weight", 1.0),
            conservation_laws=data.get("conservation_laws", []),
            notes=data.get("notes", ""),
        )


@dataclass
class ModelCard:
    """Complete model card with all metadata.

    A model card contains all information needed to understand,
    use, and reproduce a physics model.

    Attributes:
        info: Basic model information.
        training: Training details.
        performance: Performance metrics.
        usage_examples: Code examples for using the model.
        limitations: Known limitations.
        citation: How to cite the model.
    """

    info: ModelInfo
    training: TrainingInfo = field(default_factory=TrainingInfo)
    performance: dict[str, float] = field(default_factory=dict)
    usage_examples: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    citation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "info": self.info.to_dict(),
            "training": self.training.to_dict(),
            "performance": self.performance,
            "usage_examples": self.usage_examples,
            "limitations": self.limitations,
            "citation": self.citation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelCard":
        """Create from dictionary."""
        return cls(
            info=ModelInfo.from_dict(data["info"]),
            training=TrainingInfo.from_dict(data.get("training", {})),
            performance=data.get("performance", {}),
            usage_examples=data.get("usage_examples", []),
            limitations=data.get("limitations", []),
            citation=data.get("citation", ""),
        )

    def to_markdown(self) -> str:
        """Generate markdown representation of the model card."""
        lines = [
            f"# {self.info.name}",
            "",
            f"**Version:** {self.info.version}",
            f"**Domain:** {self.info.domain}",
            f"**License:** {self.info.license}",
            "",
            "## Description",
            "",
            self.info.description or "No description provided.",
            "",
        ]

        # Input dimensions
        if self.info.input_dims:
            lines.extend([
                "## Input Dimensions",
                "",
                "| Name | Dimension |",
                "|------|-----------|",
            ])
            for name, dim in self.info.input_dims.items():
                lines.append(f"| {name} | {dim} |")
            lines.append("")

        # Output dimensions
        if self.info.output_dims:
            lines.extend([
                "## Output Dimensions",
                "",
                "| Name | Dimension |",
                "|------|-----------|",
            ])
            for name, dim in self.info.output_dims.items():
                lines.append(f"| {name} | {dim} |")
            lines.append("")

        # Characteristic scales
        if self.info.characteristic_scales:
            lines.extend([
                "## Characteristic Scales",
                "",
                "| Quantity | Value |",
                "|----------|-------|",
            ])
            for name, value in self.info.characteristic_scales.items():
                lines.append(f"| {name} | {value} |")
            lines.append("")

        # Training info
        if self.training.dataset:
            lines.extend([
                "## Training",
                "",
                f"- **Dataset:** {self.training.dataset}",
                f"- **Epochs:** {self.training.epochs}",
                f"- **Optimizer:** {self.training.optimizer}",
                f"- **Learning Rate:** {self.training.learning_rate}",
            ])
            if self.training.physics_loss_weight > 0:
                lines.append(
                    f"- **Physics Loss Weight:** {self.training.physics_loss_weight}"
                )
            if self.training.conservation_laws:
                laws = ", ".join(self.training.conservation_laws)
                lines.append(f"- **Conservation Laws:** {laws}")
            lines.append("")

        # Performance
        if self.performance:
            lines.extend([
                "## Performance",
                "",
                "| Metric | Value |",
                "|--------|-------|",
            ])
            for metric, value in self.performance.items():
                lines.append(f"| {metric} | {value:.4f} |")
            lines.append("")

        # Usage examples
        if self.usage_examples:
            lines.extend([
                "## Usage",
                "",
            ])
            for example in self.usage_examples:
                lines.extend([
                    "```python",
                    example,
                    "```",
                    "",
                ])

        # Limitations
        if self.limitations:
            lines.extend([
                "## Limitations",
                "",
            ])
            for limitation in self.limitations:
                lines.append(f"- {limitation}")
            lines.append("")

        # Citation
        if self.citation:
            lines.extend([
                "## Citation",
                "",
                "```",
                self.citation,
                "```",
                "",
            ])

        # Tags
        if self.info.tags:
            tags = ", ".join(f"`{t}`" for t in self.info.tags)
            lines.extend([
                "## Tags",
                "",
                tags,
                "",
            ])

        return "\n".join(lines)


def save_model_card(card: ModelCard, path: str | Path) -> None:
    """Save a model card to a JSON file.

    Args:
        card: ModelCard to save.
        path: Path to save to.
    """
    path = Path(path)

    if path.suffix == ".md":
        # Save as markdown
        with open(path, "w") as f:
            f.write(card.to_markdown())
    else:
        # Save as JSON
        with open(path, "w") as f:
            json.dump(card.to_dict(), f, indent=2)


def load_model_card(path: str | Path) -> ModelCard:
    """Load a model card from a JSON file.

    Args:
        path: Path to the model card file.

    Returns:
        ModelCard loaded from file.

    Raises:
        ValueError: If file format not supported.
    """
    path = Path(path)

    if path.suffix == ".md":
        raise ValueError("Cannot load model card from markdown (use JSON)")

    with open(path) as f:
        data = json.load(f)

    return ModelCard.from_dict(data)
