# Hacker News Submission

Submit at: https://news.ycombinator.com/submit

---

## Title (copy this)

Show HN: dimtensor – Unit-aware tensors for PyTorch and JAX

---

## URL (copy this)

https://github.com/marcoloco23/dimtensor

---

## First Comment (post after submission)

Hi HN! I built dimtensor after wasting too many hours debugging physics simulations where the bug was a unit mismatch buried in the code.

It wraps NumPy/PyTorch/JAX arrays with physical units and catches dimensional errors immediately:

    velocity = DimArray([10], units.m / units.s)
    acceleration = DimArray([9.8], units.m / units.s**2)
    velocity + acceleration  # DimensionError!

Key difference from Pint/Astropy: native PyTorch autograd and JAX JIT support. Gradients flow through unit-aware operations.

Docs: https://marcoloco23.github.io/dimtensor

Happy to answer questions!
