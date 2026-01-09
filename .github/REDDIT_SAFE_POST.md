# Reddit-Safe Post (Spam Filter Friendly)

Tips to avoid Reddit filters:
1. No links in the body (add in comments after)
2. Shorter post
3. Don't mention "pip install" (triggers commercial filter)
4. Ask a question to encourage engagement

---

## r/Python

**Title:** I built a library that catches physics unit errors in NumPy/PyTorch/JAX - would love feedback

**Body:**
```
After wasting too many hours debugging simulations where the bug was a hidden unit mismatch, I built dimtensor - it adds physical unit tracking to arrays and catches dimensional errors immediately.

**What it does:**

- Wraps NumPy, PyTorch, and JAX arrays with physical units
- Catches errors like adding velocity to acceleration instantly
- Full autograd support in PyTorch (gradients flow through)
- Works with JAX jit/vmap/grad
- GPU support (CUDA, Apple Silicon)

**Quick example:**

    from dimtensor import DimArray, units

    velocity = DimArray([10, 20], units.m / units.s)
    acceleration = DimArray([9.8], units.m / units.s**2)

    velocity + acceleration  # DimensionError: cannot add m/s to m/s²

The error happens immediately, not after hours of computation.

**PyTorch example:**

    from dimtensor.torch import DimTensor
    import torch

    v = DimTensor(torch.tensor([1.0], requires_grad=True), units.m / units.s)
    t = DimTensor(torch.tensor([2.0]), units.s)
    d = v * t
    d.backward()  # Gradients work!

I'd love feedback from this community - what features would make this more useful for your workflows? The project is called dimtensor and it's on GitHub and PyPI.
```

**First comment to add after posting:**
```
Links for those interested:
- GitHub: github.com/marcoloco23/dimtensor
- Docs: marcoloco23.github.io/dimtensor
- Install: pip install dimtensor
```

---

## r/MachineLearning

**Title:** [P] Physical units for PyTorch/JAX with autograd - looking for feedback from ML practitioners

**Body:**
```
I've been working on a library for physics-informed ML that adds dimensional analysis to PyTorch and JAX tensors.

**The problem:** When training PINNs or running scientific ML, unit errors can silently corrupt results. You train for hours before realizing your loss mixed meters with kilometers.

**The solution:** dimtensor catches these errors at operation time:

- Native autograd (gradients flow through unit-aware ops)
- JAX pytree registration (jit, vmap, grad work)
- CUDA/MPS GPU support
- About 2-5x overhead vs raw tensors (negligible on GPU)

Example with JAX:

    import jax
    from dimtensor.jax import DimArray
    from dimtensor import units

    @jax.jit
    def kinetic_energy(mass, velocity):
        return 0.5 * mass * velocity**2

    m = DimArray(jnp.array([1.0]), units.kg)
    v = DimArray(jnp.array([10.0]), units.m / units.s)
    E = kinetic_energy(m, v)  # Returns Joules, JIT works

Would this be useful in your workflows? Looking for feedback on what features matter most for ML use cases.

Project is called dimtensor - happy to share links in comments.
```

---

## Alternative: Try r/pythontips or r/learnpython first

These subreddits have less strict filters. Build some post history, then try r/Python.

## Or: Comment on relevant threads

Search r/Python for posts about:
- "unit conversion"
- "physics simulation"
- "dimensional analysis"
- "pint library"

Add a helpful comment mentioning dimtensor as an alternative.
