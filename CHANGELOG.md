# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-01-08

### Added
- **Unit simplification**: Compound units now display as their SI derived equivalents
  - `kg·m/s²` → `N` (newton)
  - `kg·m²/s²` → `J` (joule)
  - `kg·m²/s³` → `W` (watt)
  - `m/s·s` → `m` (cancellation)
- **Format string support**: Use f-strings with DimArray
  - `f"{distance:.2f}"` → `"1234.57 m"`
  - `f"{energy:.2e}"` → `"1.23e+03 J"`
- **NumPy ufunc integration**: Use numpy functions directly
  - `np.sin(angle)`, `np.cos(angle)` - require dimensionless input
  - `np.exp(x)`, `np.log(x)` - require dimensionless input
  - `np.sqrt(area)` - halves dimension exponents
  - `np.abs(velocity)` - preserves units
  - `np.add(a, b)`, `np.multiply(a, b)` - dimension-aware arithmetic

### Changed
- Unit display now uses simplified symbols by default

## [0.1.2] - 2025-01-08

### Fixed
- Corrected GitHub repository URLs in PyPI metadata

## [0.1.1] - 2025-01-08

### Added
- `sqrt()` method on DimArray for square root with proper dimension handling

## [0.1.0] - 2025-01-08

### Added
- Initial release
- `DimArray` class wrapping numpy arrays with unit metadata
- `Dimension` class with full algebra (multiply, divide, power)
- SI base units: meter, kilogram, second, ampere, kelvin, mole, candela
- SI derived units: newton, joule, watt, pascal, volt, hertz, etc.
- Common non-SI units: km, mile, hour, eV, atm, etc.
- Unit conversion with `.to()` method
- Dimensional error catching at operation time
- Arithmetic operations with automatic dimension checking
- Comparison operations between compatible dimensions
- Array indexing, slicing, and iteration
- Reduction operations (sum, mean, std, min, max)
