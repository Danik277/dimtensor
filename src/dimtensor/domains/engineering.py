"""Engineering units for mechanical, thermal, and electrical engineering.

This module provides units commonly used in engineering disciplines,
including pressure units (MPa, ksi, psi), energy units (BTU, kWh),
power units (horsepower), and other practical engineering units.

Example:
    >>> from dimtensor import DimArray
    >>> from dimtensor.domains.engineering import MPa, hp, BTU
    >>> stress = DimArray([250], MPa)  # 250 MPa yield strength
    >>> power = DimArray([100], hp)  # 100 horsepower engine
    >>> heat = DimArray([1000], BTU)  # 1000 BTU of heat

Reference values from NIST and engineering handbooks.
"""

from __future__ import annotations

from ..core.dimensions import Dimension, DIMENSIONLESS
from ..core.units import Unit


# =============================================================================
# Pressure Units
# =============================================================================

# Kilopascal
kilopascal = Unit("kPa", Dimension(mass=1, length=-1, time=-2), 1e3)
kPa = kilopascal

# Megapascal
megapascal = Unit("MPa", Dimension(mass=1, length=-1, time=-2), 1e6)
MPa = megapascal

# Gigapascal
gigapascal = Unit("GPa", Dimension(mass=1, length=-1, time=-2), 1e9)
GPa = gigapascal

# Kilopound per square inch (ksi)
# 1 ksi = 1000 psi = 6.894757e6 Pa
ksi = Unit("ksi", Dimension(mass=1, length=-1, time=-2), 6.894757e6)

# Millibar (common in meteorology)
millibar = Unit("mbar", Dimension(mass=1, length=-1, time=-2), 100.0)
mbar = millibar


# =============================================================================
# Energy Units
# =============================================================================

# British Thermal Unit (IT definition)
# 1 BTU = 1055.05585262 J
BTU = Unit("BTU", Dimension(mass=1, length=2, time=-2), 1055.05585262)
btu = BTU

# Therm (100,000 BTU, common for natural gas)
therm = Unit("therm", Dimension(mass=1, length=2, time=-2), 1.0550559e8)

# Kilowatt-hour
kilowatt_hour = Unit("kWh", Dimension(mass=1, length=2, time=-2), 3.6e6)
kWh = kilowatt_hour

# Megawatt-hour
megawatt_hour = Unit("MWh", Dimension(mass=1, length=2, time=-2), 3.6e9)
MWh = megawatt_hour

# Foot-pound (energy, not torque)
foot_pound_energy = Unit("ft·lbf", Dimension(mass=1, length=2, time=-2), 1.3558179483314)
ft_lbf = foot_pound_energy


# =============================================================================
# Power Units
# =============================================================================

# Mechanical horsepower (550 ft*lbf/s)
# 1 hp = 745.69987158227022 W
horsepower = Unit("hp", Dimension(mass=1, length=2, time=-3), 745.69987158227022)
hp = horsepower

# Metric horsepower (PS, CV, pk)
# 1 PS = 75 kgf*m/s = 735.49875 W
metric_horsepower = Unit("PS", Dimension(mass=1, length=2, time=-3), 735.49875)
PS = metric_horsepower

# Ton of refrigeration (cooling power)
# 1 RT = 12000 BTU/h = 3516.8528420667 W
ton_refrigeration = Unit("RT", Dimension(mass=1, length=2, time=-3), 3516.8528420667)
RT = ton_refrigeration

# BTU per hour
BTU_per_hour = Unit("BTU/h", Dimension(mass=1, length=2, time=-3), 0.29307107017222)


# =============================================================================
# Volumetric Flow Rate Units
# =============================================================================

# Gallon per minute (US gallon)
# 1 gpm = 6.30902e-5 m^3/s
gallon_per_minute = Unit("gpm", Dimension(length=3, time=-1), 6.30901964e-5)
gpm = gallon_per_minute

# Cubic feet per minute
# 1 cfm = 4.71947e-4 m^3/s
cubic_feet_per_minute = Unit("cfm", Dimension(length=3, time=-1), 4.71947443e-4)
cfm = cubic_feet_per_minute

# Liters per minute
liters_per_minute = Unit("L/min", Dimension(length=3, time=-1), 1.66666667e-5)
lpm = liters_per_minute


# =============================================================================
# Torque Units
# =============================================================================

# Newton-meter (SI unit, but included for completeness)
newton_meter = Unit("N·m", Dimension(mass=1, length=2, time=-2), 1.0)
Nm = newton_meter

# Foot-pound (torque)
# 1 ft·lb = 1.3558179483314 N·m
foot_pound_torque = Unit("ft·lb", Dimension(mass=1, length=2, time=-2), 1.3558179483314)
ft_lb = foot_pound_torque

# Inch-pound
# 1 in·lb = 0.1129848290276 N·m
inch_pound = Unit("in·lb", Dimension(mass=1, length=2, time=-2), 0.1129848290276)
in_lb = inch_pound


# =============================================================================
# Angular Velocity Units
# =============================================================================

# Revolutions per minute
# 1 rpm = 2*pi/60 rad/s = 0.10471975511965977 rad/s
# Treated as frequency (1/time) like Hz
rpm = Unit("rpm", Dimension(time=-1), 0.016666666666666666)

# Revolutions per second
rps = Unit("rps", Dimension(time=-1), 1.0)


# =============================================================================
# Small Length Units (common in manufacturing)
# =============================================================================

# Mil (thousandth of an inch)
# 1 mil = 0.001 inch = 2.54e-5 m
mil = Unit("mil", Dimension(length=1), 2.54e-5)

# Micrometer/micron (also common in manufacturing)
micrometer = Unit("um", Dimension(length=1), 1e-6)
um = micrometer
micron = micrometer


# =============================================================================
# Thermal Units
# =============================================================================

# BTU per pound (specific energy, common in combustion)
BTU_per_pound = Unit(
    "BTU/lb", Dimension(length=2, time=-2), 2326.0
)

# BTU per hour per square foot (heat flux, common in HVAC)
BTU_per_hour_ft2 = Unit(
    "BTU/(h·ft²)",
    Dimension(mass=1, time=-3),
    3.15459074506
)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Pressure
    "kilopascal", "kPa",
    "megapascal", "MPa",
    "gigapascal", "GPa",
    "ksi",
    "millibar", "mbar",
    # Energy
    "BTU", "btu",
    "therm",
    "kilowatt_hour", "kWh",
    "megawatt_hour", "MWh",
    "foot_pound_energy", "ft_lbf",
    # Power
    "horsepower", "hp",
    "metric_horsepower", "PS",
    "ton_refrigeration", "RT",
    "BTU_per_hour",
    # Volumetric flow
    "gallon_per_minute", "gpm",
    "cubic_feet_per_minute", "cfm",
    "liters_per_minute", "lpm",
    # Torque
    "newton_meter", "Nm",
    "foot_pound_torque", "ft_lb",
    "inch_pound", "in_lb",
    # Angular velocity
    "rpm", "rps",
    # Small lengths
    "mil",
    "micrometer", "um", "micron",
    # Thermal
    "BTU_per_pound",
    "BTU_per_hour_ft2",
]
