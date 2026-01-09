# Marketing Posts for dimtensor

Ready-to-post content for various platforms.

---

## Reddit

### r/Python

**Title:** dimtensor: Unit-aware tensors for NumPy, PyTorch, and JAX - catch physics bugs before they cost you hours

**Body:**
```
I built dimtensor to solve a problem I kept running into: unit errors in physics simulations that only showed up after hours of computation.

**What it does:**
- Wraps NumPy/PyTorch/JAX arrays with physical units
- Catches dimensional errors immediately (can't add m/s to m/s²)
- Full autograd support in PyTorch
- JIT/vmap/grad compatible with JAX
- GPU acceleration (CUDA, MPS)
- Built-in uncertainty propagation

**Example:**
```python
from dimtensor import DimArray, units

velocity = DimArray([10, 20], units.m / units.s)
time = DimArray([1, 2], units.s)
distance = velocity * time  # [10, 40] m

acceleration = DimArray([9.8], units.m / units.s**2)
velocity + acceleration  # DimensionError!
```

**PyTorch example:**
```python
from dimtensor.torch import DimTensor
v = DimTensor(torch.tensor([1.0], requires_grad=True), units.m / units.s)
t = DimTensor(torch.tensor([2.0]), units.s)
d = v * t
d.backward()  # Gradients work!
```

Links:
- GitHub: https://github.com/marcoloco23/dimtensor
- Docs: https://marcoloco23.github.io/dimtensor
- PyPI: `pip install dimtensor`

Would love feedback from the community!
```

---

### r/MachineLearning

**Title:** [P] dimtensor: Physical units for PyTorch/JAX with full autograd and GPU support

**Body:**
```
Sharing a library I've been working on for physics-informed ML workflows.

**Problem:** When training physics-informed neural networks or running scientific ML, unit errors can silently corrupt your results. You might train for hours before realizing your loss function mixed meters with kilometers.

**Solution:** dimtensor adds dimensional analysis to PyTorch and JAX:

- Native autograd support (gradients flow through unit-aware ops)
- JAX pytree registration (jit, vmap, grad all work)
- CUDA/MPS GPU support
- ~2-5x overhead vs raw tensors (negligible on GPU)

```python
import jax
from dimtensor.jax import DimArray
from dimtensor import units

@jax.jit
def physics_loss(pred_velocity, true_velocity):
    return ((pred_velocity - true_velocity)**2).sum()

# Units checked at trace time, JIT runs at full speed
```

Comparison with alternatives: https://marcoloco23.github.io/dimtensor/comparisons/feature-matrix/

GitHub: https://github.com/marcoloco23/dimtensor
```

---

### r/Physics

**Title:** Python library for unit-safe physics calculations with NumPy/PyTorch/JAX

**Body:**
```
Built a library that might be useful for computational physics work.

dimtensor catches dimensional errors at operation time - so you find out immediately when you accidentally add velocity to acceleration, not after your simulation runs for hours.

Features:
- SI units + astronomy/chemistry/engineering domains
- CODATA 2022 physical constants with uncertainties
- Uncertainty propagation through all operations
- Save/load with units (HDF5, NetCDF, JSON, Parquet)

```python
from dimtensor import DimArray, units, constants

# E = mc²
mass = DimArray([1.0], units.kg)
E = mass * constants.c**2  # Returns Joules

# With uncertainty
length = DimArray([10.0], units.m, uncertainty=[0.1])
area = length**2  # Uncertainty propagates: 100 ± 2 m²
```

GitHub: https://github.com/marcoloco23/dimtensor
Docs: https://marcoloco23.github.io/dimtensor
```

---

## Hacker News

**Title:** Show HN: dimtensor – Unit-aware tensors for PyTorch and JAX

**Body:**
```
https://github.com/marcoloco23/dimtensor

dimtensor adds physical unit tracking to NumPy, PyTorch, and JAX arrays. It catches dimensional errors at operation time rather than after hours of computation.

Key differentiator from Pint/Astropy: native PyTorch autograd and JAX JIT support. Gradients flow through unit-aware operations, and JAX transformations (jit, vmap, grad) work out of the box.

Built this after wasting too many hours debugging physics simulations where the bug was a unit mismatch buried deep in the code.
```

---

## Dev.to

**Title:** Stop Wasting Hours on Unit Bugs: Introducing dimtensor for Scientific Python

**Body:**
```markdown
---
title: Stop Wasting Hours on Unit Bugs: Introducing dimtensor for Scientific Python
published: true
tags: python, machinelearning, physics, opensource
---

Every physicist and scientific ML researcher has been there: you run a simulation for hours, only to discover the results are garbage because somewhere you added meters to kilometers, or velocity to acceleration.

## The Problem

```python
# Spot the bug
velocity = np.array([10, 20, 30])  # m/s
acceleration = np.array([9.8])     # m/s²
result = velocity + acceleration   # No error... but meaningless
```

NumPy doesn't know or care about physical dimensions. Neither does PyTorch or JAX.

## The Solution: dimtensor

```python
from dimtensor import DimArray, units

velocity = DimArray([10, 20, 30], units.m / units.s)
acceleration = DimArray([9.8], units.m / units.s**2)
velocity + acceleration
# DimensionError: cannot add m/s to m/s^2
```

The error happens *immediately*, not after hours of computation.

## Why dimtensor?

There are other unit libraries (Pint, Astropy, unyt), but dimtensor is the only one with:

- **Native PyTorch autograd** - gradients flow through unit-aware operations
- **Native JAX support** - jit, vmap, grad all work
- **GPU acceleration** - CUDA and Apple Silicon
- **Built-in uncertainty propagation**

```python
# PyTorch with autograd
from dimtensor.torch import DimTensor
v = DimTensor(torch.tensor([1.0], requires_grad=True), units.m/units.s)
t = DimTensor(torch.tensor([2.0]), units.s)
d = v * t
d.backward()  # Works!

# JAX with JIT
@jax.jit
def kinetic_energy(m, v):
    return 0.5 * m * v**2
# Units checked at trace time, compiled code runs at full speed
```

## Get Started

```bash
pip install dimtensor
```

- [Documentation](https://marcoloco23.github.io/dimtensor)
- [GitHub](https://github.com/marcoloco23/dimtensor)
- [Comparison with alternatives](https://marcoloco23.github.io/dimtensor/comparisons/feature-matrix/)

Would love to hear your feedback!
```

---

## Twitter/X

**Thread:**

```
1/ Announcing dimtensor 🧪

Unit-aware tensors for physics and scientific ML.

Catch dimensional errors immediately - not after hours of computation.

Works with NumPy, PyTorch, and JAX.

🔗 https://github.com/marcoloco23/dimtensor

🧵 Thread below...

2/ The problem: silent unit bugs

velocity = np.array([10])  # m/s
acceleration = np.array([9.8])  # m/s²
result = velocity + acceleration  # No error!

This runs fine but is physically meaningless. Good luck debugging this in a 10,000 line simulation.

3/ The solution:

from dimtensor import DimArray, units

velocity = DimArray([10], units.m / units.s)
acceleration = DimArray([9.8], units.m / units.s**2)
velocity + acceleration
# DimensionError: cannot add m/s to m/s²

Bug caught instantly. ✅

4/ What makes dimtensor different from Pint/Astropy?

✅ Native PyTorch autograd
✅ Native JAX (jit, vmap, grad)
✅ GPU support (CUDA, MPS)
✅ Built-in uncertainty propagation
✅ 6+ I/O formats (HDF5, NetCDF, Parquet...)

5/ Perfect for:

- Physics-informed neural networks
- Scientific simulations
- Computational physics research
- Any ML project with physical quantities

pip install dimtensor

Docs: https://marcoloco23.github.io/dimtensor

Feedback welcome! 🙏
```

---

## Posting Checklist

- [ ] Reddit r/Python
- [ ] Reddit r/MachineLearning
- [ ] Reddit r/Physics
- [ ] Hacker News (Show HN)
- [ ] Dev.to article
- [ ] Twitter/X thread
- [ ] LinkedIn post

## Tips

1. **Best times to post:**
   - Reddit: Tuesday-Thursday, 9-11am EST
   - HN: Tuesday-Thursday, 8-10am EST
   - Dev.to: Tuesday-Wednesday

2. **Engage with comments** - respond to questions quickly

3. **Don't spam** - space posts across platforms by a day or two

4. **Cross-link** - mention if it's getting traction elsewhere ("Thanks for the feedback on HN!")
