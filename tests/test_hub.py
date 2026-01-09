"""Tests for the model hub."""

import json
import tempfile
from pathlib import Path

import pytest

from dimtensor import Dimension
from dimtensor.hub import (
    ModelCard,
    ModelInfo,
    get_model_info,
    list_models,
    load_model,
    load_model_card,
    register_model,
    save_model_card,
)
from dimtensor.hub.registry import clear_registry
from dimtensor.hub.cards import TrainingInfo


@pytest.fixture(autouse=True)
def clean_registry():
    """Clean registry before and after each test."""
    clear_registry()
    yield
    clear_registry()


class TestModelInfo:
    """Tests for ModelInfo dataclass."""

    def test_basic_creation(self):
        """Test creating ModelInfo."""
        info = ModelInfo(
            name="test-model",
            version="1.0.0",
            description="A test model",
            domain="mechanics",
        )
        assert info.name == "test-model"
        assert info.version == "1.0.0"
        assert info.domain == "mechanics"

    def test_with_dimensions(self):
        """Test ModelInfo with dimensions."""
        info = ModelInfo(
            name="velocity-predictor",
            input_dims={
                "position": Dimension(length=1),
                "time": Dimension(time=1),
            },
            output_dims={
                "velocity": Dimension(length=1, time=-1),
            },
        )
        assert len(info.input_dims) == 2
        assert info.output_dims["velocity"] == Dimension(length=1, time=-1)

    def test_to_dict(self):
        """Test serialization to dict."""
        info = ModelInfo(
            name="test",
            input_dims={"x": Dimension(length=1)},
            tags=["physics", "test"],
        )
        data = info.to_dict()
        assert data["name"] == "test"
        assert data["input_dims"]["x"] == {"L": 1.0}
        assert data["tags"] == ["physics", "test"]

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "name": "test",
            "version": "2.0.0",
            "input_dims": {"x": {"L": 1, "T": -1}},
            "domain": "fluid_dynamics",
        }
        info = ModelInfo.from_dict(data)
        assert info.name == "test"
        assert info.version == "2.0.0"
        assert info.input_dims["x"] == Dimension(length=1, time=-1)

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = ModelInfo(
            name="roundtrip-test",
            version="1.2.3",
            description="Test model",
            input_dims={"force": Dimension(mass=1, length=1, time=-2)},
            output_dims={"accel": Dimension(length=1, time=-2)},
            domain="mechanics",
            characteristic_scales={"force": 100.0, "mass": 1.0},
            tags=["newton", "mechanics"],
        )
        data = original.to_dict()
        restored = ModelInfo.from_dict(data)

        assert restored.name == original.name
        assert restored.version == original.version
        assert restored.input_dims == original.input_dims
        assert restored.characteristic_scales == original.characteristic_scales


class TestRegistry:
    """Tests for model registry functions."""

    def test_register_direct(self):
        """Test direct registration."""
        info = ModelInfo(name="direct-model", domain="test")
        register_model("direct-model", info=info)

        result = get_model_info("direct-model")
        assert result.name == "direct-model"

    def test_register_decorator(self):
        """Test decorator registration."""

        @register_model("decorated-model", domain="mechanics")
        def create_model():
            return {"type": "test"}

        info = get_model_info("decorated-model")
        assert info.name == "decorated-model"
        assert info.domain == "mechanics"

        # Factory should work
        model = load_model("decorated-model")
        assert model == {"type": "test"}

    def test_register_with_factory(self):
        """Test registration with factory function."""

        def factory(hidden_size=64):
            return {"hidden_size": hidden_size}

        info = ModelInfo(name="factory-model")
        register_model("factory-model", info=info, factory=factory)

        model = load_model("factory-model", hidden_size=128)
        assert model["hidden_size"] == 128

    def test_list_all_models(self):
        """Test listing all models."""
        register_model("model-a", ModelInfo(name="model-a", domain="mechanics"))
        register_model("model-b", ModelInfo(name="model-b", domain="thermo"))
        register_model("model-c", ModelInfo(name="model-c", domain="mechanics"))

        models = list_models()
        assert len(models) == 3

    def test_list_by_domain(self):
        """Test filtering by domain."""
        register_model("mech-1", ModelInfo(name="mech-1", domain="mechanics"))
        register_model("mech-2", ModelInfo(name="mech-2", domain="mechanics"))
        register_model("thermo-1", ModelInfo(name="thermo-1", domain="thermo"))

        mechanics = list_models(domain="mechanics")
        assert len(mechanics) == 2
        assert all(m.domain == "mechanics" for m in mechanics)

    def test_list_by_tags(self):
        """Test filtering by tags."""
        register_model(
            "tagged-1",
            ModelInfo(name="tagged-1", tags=["fluid", "cfd", "3d"]),
        )
        register_model(
            "tagged-2",
            ModelInfo(name="tagged-2", tags=["fluid", "2d"]),
        )
        register_model(
            "tagged-3",
            ModelInfo(name="tagged-3", tags=["solid", "fem"]),
        )

        fluid_models = list_models(tags=["fluid"])
        assert len(fluid_models) == 2

        cfd_models = list_models(tags=["fluid", "cfd"])
        assert len(cfd_models) == 1

    def test_model_not_found(self):
        """Test error on missing model."""
        with pytest.raises(KeyError, match="not found"):
            get_model_info("nonexistent")

    def test_load_without_factory(self):
        """Test error when loading model without factory."""
        register_model("no-factory", ModelInfo(name="no-factory"))

        with pytest.raises(RuntimeError, match="no factory"):
            load_model("no-factory")


class TestModelCard:
    """Tests for model cards."""

    def test_basic_card(self):
        """Test creating a basic model card."""
        info = ModelInfo(name="test-model", domain="mechanics")
        card = ModelCard(info=info)

        assert card.info.name == "test-model"
        assert isinstance(card.training, TrainingInfo)

    def test_card_with_training(self):
        """Test card with training info."""
        info = ModelInfo(name="trained-model")
        training = TrainingInfo(
            dataset="physics-sim-v1",
            epochs=100,
            physics_loss_weight=0.1,
            conservation_laws=["energy", "momentum"],
        )
        card = ModelCard(info=info, training=training)

        assert card.training.dataset == "physics-sim-v1"
        assert card.training.epochs == 100
        assert "energy" in card.training.conservation_laws

    def test_card_to_dict(self):
        """Test serialization."""
        info = ModelInfo(
            name="test",
            input_dims={"x": Dimension(length=1)},
        )
        card = ModelCard(
            info=info,
            performance={"mse": 0.01, "physics_error": 0.001},
        )
        data = card.to_dict()

        assert data["info"]["name"] == "test"
        assert data["performance"]["mse"] == 0.01

    def test_card_roundtrip(self):
        """Test serialization roundtrip."""
        original = ModelCard(
            info=ModelInfo(
                name="roundtrip",
                input_dims={"v": Dimension(length=1, time=-1)},
            ),
            training=TrainingInfo(dataset="test-data", epochs=50),
            performance={"loss": 0.05},
            limitations=["Only works for laminar flow"],
        )
        data = original.to_dict()
        restored = ModelCard.from_dict(data)

        assert restored.info.name == original.info.name
        assert restored.training.epochs == 50
        assert restored.limitations == original.limitations

    def test_card_to_markdown(self):
        """Test markdown generation."""
        info = ModelInfo(
            name="Velocity Predictor",
            version="1.0.0",
            description="Predicts velocity from position",
            input_dims={"position": Dimension(length=1)},
            output_dims={"velocity": Dimension(length=1, time=-1)},
            domain="mechanics",
            tags=["physics", "ml"],
        )
        card = ModelCard(info=info)
        md = card.to_markdown()

        assert "# Velocity Predictor" in md
        assert "**Domain:** mechanics" in md
        assert "| position |" in md
        assert "| velocity |" in md

    def test_save_load_json(self):
        """Test saving and loading as JSON."""
        info = ModelInfo(name="save-test")
        card = ModelCard(info=info, performance={"accuracy": 0.95})

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            save_model_card(card, path)
            loaded = load_model_card(path)

            assert loaded.info.name == "save-test"
            assert loaded.performance["accuracy"] == 0.95
        finally:
            path.unlink()

    def test_save_markdown(self):
        """Test saving as markdown."""
        info = ModelInfo(name="md-test", description="Test description")
        card = ModelCard(info=info)

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            path = Path(f.name)

        try:
            save_model_card(card, path)
            content = path.read_text()

            assert "# md-test" in content
            assert "Test description" in content
        finally:
            path.unlink()


class TestTrainingInfo:
    """Tests for TrainingInfo."""

    def test_default_values(self):
        """Test default values."""
        info = TrainingInfo()
        assert info.epochs == 0
        assert info.batch_size == 32
        assert info.optimizer == "Adam"

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = TrainingInfo(
            dataset="my-data",
            epochs=100,
            conservation_laws=["energy"],
        )
        data = original.to_dict()
        restored = TrainingInfo.from_dict(data)

        assert restored.dataset == original.dataset
        assert restored.epochs == original.epochs
        assert restored.conservation_laws == original.conservation_laws
