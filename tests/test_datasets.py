"""Tests for the dataset registry."""

import pytest

from dimtensor import Dimension
from dimtensor.datasets import (
    DatasetInfo,
    get_dataset_info,
    list_datasets,
    load_dataset,
    register_dataset,
)
from dimtensor.datasets.registry import clear_datasets, _REGISTRY, _LOADERS


@pytest.fixture(autouse=True)
def preserve_registry():
    """Preserve and restore registry for tests that modify it."""
    original_registry = _REGISTRY.copy()
    original_loaders = _LOADERS.copy()
    yield
    _REGISTRY.clear()
    _REGISTRY.update(original_registry)
    _LOADERS.clear()
    _LOADERS.update(original_loaders)


class TestDatasetInfo:
    """Tests for DatasetInfo dataclass."""

    def test_basic_creation(self):
        """Test creating DatasetInfo."""
        info = DatasetInfo(
            name="test-dataset",
            description="A test dataset",
            domain="mechanics",
        )
        assert info.name == "test-dataset"
        assert info.description == "A test dataset"
        assert info.domain == "mechanics"

    def test_with_dimensions(self):
        """Test DatasetInfo with dimensions."""
        info = DatasetInfo(
            name="physics-data",
            features={
                "position": Dimension(length=1),
                "time": Dimension(time=1),
            },
            targets={
                "velocity": Dimension(length=1, time=-1),
            },
        )
        assert len(info.features) == 2
        assert info.targets["velocity"] == Dimension(length=1, time=-1)

    def test_to_dict(self):
        """Test serialization to dict."""
        info = DatasetInfo(
            name="test",
            features={"x": Dimension(length=1)},
            tags=["test", "example"],
        )
        data = info.to_dict()
        assert data["name"] == "test"
        assert data["tags"] == ["test", "example"]

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = DatasetInfo(
            name="roundtrip-test",
            description="Test dataset",
            features={"force": Dimension(mass=1, length=1, time=-2)},
            targets={"accel": Dimension(length=1, time=-2)},
            domain="mechanics",
            size=1000,
            tags=["test"],
        )
        data = original.to_dict()
        restored = DatasetInfo.from_dict(data)

        assert restored.name == original.name
        assert restored.description == original.description
        assert restored.size == original.size


class TestRegistry:
    """Tests for dataset registry functions."""

    def test_register_direct(self):
        """Test direct registration."""
        info = DatasetInfo(name="direct-dataset", domain="test")
        register_dataset("direct-dataset", info=info)

        result = get_dataset_info("direct-dataset")
        assert result.name == "direct-dataset"

    def test_register_decorator(self):
        """Test decorator registration."""

        @register_dataset("decorated-dataset", domain="mechanics")
        def load_data():
            return {"type": "test"}

        info = get_dataset_info("decorated-dataset")
        assert info.name == "decorated-dataset"
        assert info.domain == "mechanics"

        # Loader should work
        data = load_dataset("decorated-dataset")
        assert data == {"type": "test"}

    def test_register_with_loader(self):
        """Test registration with loader function."""

        def loader(split="train"):
            return {"split": split}

        info = DatasetInfo(name="loader-dataset")
        register_dataset("loader-dataset", info=info, loader=loader)

        data = load_dataset("loader-dataset", split="test")
        assert data["split"] == "test"

    def test_list_all_datasets(self):
        """Test listing all datasets."""
        # Built-in datasets should be present
        datasets = list_datasets()
        assert len(datasets) >= 5
        assert any(d.name == "pendulum" for d in datasets)

    def test_list_by_domain(self):
        """Test filtering by domain."""
        mechanics = list_datasets(domain="mechanics")
        assert len(mechanics) >= 3
        assert all(d.domain == "mechanics" for d in mechanics)

    def test_list_by_tags(self):
        """Test filtering by tags."""
        pde_datasets = list_datasets(tags=["pde"])
        assert len(pde_datasets) >= 2
        assert all("pde" in d.tags for d in pde_datasets)

    def test_dataset_not_found(self):
        """Test error on missing dataset."""
        with pytest.raises(KeyError, match="not found"):
            get_dataset_info("nonexistent")

    def test_load_without_loader(self):
        """Test error when loading dataset without loader."""
        info = DatasetInfo(name="no-loader-dataset")
        register_dataset("no-loader-dataset", info=info)

        with pytest.raises(RuntimeError, match="no loader"):
            load_dataset("no-loader-dataset")


class TestBuiltinDatasets:
    """Tests for built-in physics datasets."""

    def test_pendulum_dataset(self):
        """Test pendulum dataset metadata."""
        info = get_dataset_info("pendulum")
        assert info.domain == "mechanics"
        assert "time" in info.features
        assert "angle" in info.targets
        assert info.features["time"] == Dimension(time=1)

    def test_projectile_dataset(self):
        """Test projectile dataset metadata."""
        info = get_dataset_info("projectile")
        assert info.domain == "mechanics"
        assert "initial_velocity" in info.features
        assert info.features["initial_velocity"] == Dimension(length=1, time=-1)

    def test_heat_diffusion_dataset(self):
        """Test heat diffusion dataset."""
        info = get_dataset_info("heat_diffusion")
        assert info.domain == "thermodynamics"
        assert "temperature" in info.targets
        assert info.targets["temperature"] == Dimension(temperature=1)

    def test_burgers_dataset(self):
        """Test Burgers equation dataset."""
        info = get_dataset_info("burgers")
        assert info.domain == "fluid_dynamics"
        assert "pde" in info.tags

    def test_navier_stokes_dataset(self):
        """Test Navier-Stokes dataset."""
        info = get_dataset_info("navier_stokes_2d")
        assert info.domain == "fluid_dynamics"
        assert "pressure" in info.targets

    def test_lorenz_dataset(self):
        """Test Lorenz system dataset."""
        info = get_dataset_info("lorenz")
        assert "chaotic" in info.tags
        # Lorenz variables are dimensionless
        assert info.targets["x"] == Dimension()

    def test_wave_dataset(self):
        """Test wave equation dataset."""
        info = get_dataset_info("wave_1d")
        assert "pde" in info.tags
        assert info.targets["displacement"] == Dimension(length=1)


class TestDimensionalConsistency:
    """Tests for dimensional consistency of datasets."""

    def test_velocity_dimension(self):
        """Test velocity dimensions are consistent."""
        projectile = get_dataset_info("projectile")
        spring = get_dataset_info("spring_mass")

        # Both should have velocity = length/time
        vel_dim = Dimension(length=1, time=-1)
        assert projectile.features["initial_velocity"] == vel_dim
        assert spring.targets["velocity"] == vel_dim

    def test_pressure_dimension(self):
        """Test pressure dimensions."""
        ns = get_dataset_info("navier_stokes_2d")
        # Pressure = force/area = mass/(length*time²)
        assert ns.targets["pressure"] == Dimension(mass=1, length=-1, time=-2)

    def test_thermal_diffusivity_dimension(self):
        """Test thermal diffusivity dimension."""
        heat = get_dataset_info("heat_diffusion")
        # Thermal diffusivity = length²/time
        assert heat.features["thermal_diffusivity"] == Dimension(length=2, time=-1)
