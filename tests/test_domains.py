"""Tests for domain-specific unit modules."""

import pytest
import numpy as np
from dimtensor import DimArray
from dimtensor.core.dimensions import Dimension, DIMENSIONLESS


class TestAstronomyUnits:
    """Test astronomy units."""

    def test_parsec_dimension(self):
        """Parsec should have length dimension."""
        from dimtensor.domains.astronomy import parsec
        assert parsec.dimension == Dimension(length=1)

    def test_parsec_scale(self):
        """Parsec should be approximately 3.086e16 m."""
        from dimtensor.domains.astronomy import parsec
        assert pytest.approx(parsec.scale, rel=1e-3) == 3.0857e16

    def test_au_dimension(self):
        """AU should have length dimension."""
        from dimtensor.domains.astronomy import AU
        assert AU.dimension == Dimension(length=1)

    def test_au_scale(self):
        """AU should be approximately 1.496e11 m."""
        from dimtensor.domains.astronomy import AU
        assert pytest.approx(AU.scale, rel=1e-3) == 1.496e11

    def test_light_year_dimension(self):
        """Light-year should have length dimension."""
        from dimtensor.domains.astronomy import light_year
        assert light_year.dimension == Dimension(length=1)

    def test_light_year_scale(self):
        """Light-year should be approximately 9.461e15 m."""
        from dimtensor.domains.astronomy import light_year
        assert pytest.approx(light_year.scale, rel=1e-3) == 9.461e15

    def test_solar_mass_dimension(self):
        """Solar mass should have mass dimension."""
        from dimtensor.domains.astronomy import solar_mass
        assert solar_mass.dimension == Dimension(mass=1)

    def test_solar_mass_scale(self):
        """Solar mass should be approximately 1.989e30 kg."""
        from dimtensor.domains.astronomy import solar_mass
        assert pytest.approx(solar_mass.scale, rel=1e-3) == 1.989e30

    def test_solar_luminosity_dimension(self):
        """Solar luminosity should have power dimension."""
        from dimtensor.domains.astronomy import solar_luminosity
        assert solar_luminosity.dimension == Dimension(mass=1, length=2, time=-3)

    def test_arcsecond_dimensionless(self):
        """Arcsecond should be dimensionless (angular)."""
        from dimtensor.domains.astronomy import arcsecond
        assert arcsecond.dimension == DIMENSIONLESS

    def test_parsec_to_light_year_conversion(self):
        """Test parsec to light-year conversion."""
        from dimtensor.domains.astronomy import parsec, light_year
        # 1 parsec = 3.26156 light-years
        ratio = parsec.conversion_factor(light_year)
        assert pytest.approx(ratio, rel=1e-3) == 3.262

    def test_distance_calculation(self):
        """Test using astronomy units in calculations."""
        from dimtensor.domains.astronomy import parsec, AU
        # Proxima Centauri is about 1.3 parsec away
        distance_pc = DimArray([1.3], parsec)
        # Convert to AU
        distance_au = distance_pc.to(AU)
        # 1 pc ~ 206265 AU
        expected = 1.3 * 206265
        assert pytest.approx(distance_au.magnitude()[0], rel=1e-2) == expected

    def test_kiloparsec_megaparsec(self):
        """Test kpc and Mpc units."""
        from dimtensor.domains.astronomy import parsec, kiloparsec, megaparsec
        assert pytest.approx(kiloparsec.scale / parsec.scale) == 1000
        assert pytest.approx(megaparsec.scale / parsec.scale) == 1e6

    def test_earth_jupiter_masses(self):
        """Test Earth and Jupiter mass units."""
        from dimtensor.domains.astronomy import earth_mass, jupiter_mass
        # Jupiter is about 318 Earth masses
        ratio = jupiter_mass.scale / earth_mass.scale
        assert pytest.approx(ratio, rel=1e-2) == 318


class TestChemistryUnits:
    """Test chemistry units."""

    def test_dalton_dimension(self):
        """Dalton should have mass dimension."""
        from dimtensor.domains.chemistry import dalton
        assert dalton.dimension == Dimension(mass=1)

    def test_dalton_scale(self):
        """Dalton should be approximately 1.661e-27 kg."""
        from dimtensor.domains.chemistry import dalton
        assert pytest.approx(dalton.scale, rel=1e-3) == 1.661e-27

    def test_molar_dimension(self):
        """Molar should have amount/volume dimension."""
        from dimtensor.domains.chemistry import molar
        assert molar.dimension == Dimension(amount=1, length=-3)

    def test_molar_concentration_hierarchy(self):
        """Test mM, uM, nM scale relationships."""
        from dimtensor.domains.chemistry import molar, millimolar, micromolar, nanomolar
        # Scale factors relative to mol/m^3
        assert pytest.approx(molar.scale / millimolar.scale) == 1000
        assert pytest.approx(millimolar.scale / micromolar.scale) == 1000
        assert pytest.approx(micromolar.scale / nanomolar.scale) == 1000

    def test_ppm_dimensionless(self):
        """ppm should be dimensionless."""
        from dimtensor.domains.chemistry import ppm
        assert ppm.dimension == DIMENSIONLESS

    def test_ppm_scale(self):
        """ppm should have scale 1e-6."""
        from dimtensor.domains.chemistry import ppm
        assert ppm.scale == 1e-6

    def test_ppb_ppt_scales(self):
        """Test ppb and ppt scales."""
        from dimtensor.domains.chemistry import ppb, ppt
        assert ppb.scale == 1e-9
        assert ppt.scale == 1e-12

    def test_angstrom_dimension(self):
        """Angstrom should have length dimension."""
        from dimtensor.domains.chemistry import angstrom
        assert angstrom.dimension == Dimension(length=1)

    def test_angstrom_scale(self):
        """Angstrom should be 1e-10 m."""
        from dimtensor.domains.chemistry import angstrom
        assert angstrom.scale == 1e-10

    def test_molal_dimension(self):
        """Molality should have amount/mass dimension."""
        from dimtensor.domains.chemistry import molal
        assert molal.dimension == Dimension(amount=1, mass=-1)

    def test_hartree_energy_dimension(self):
        """Hartree should have energy dimension."""
        from dimtensor.domains.chemistry import hartree
        assert hartree.dimension == Dimension(mass=1, length=2, time=-2)

    def test_debye_dipole_moment(self):
        """Debye should have dipole moment dimension (charge * length)."""
        from dimtensor.domains.chemistry import debye
        # Dipole moment = charge * length = C * m = A * s * m
        assert debye.dimension == Dimension(current=1, time=1, length=1)

    def test_concentration_calculation(self):
        """Test using chemistry units in calculations."""
        from dimtensor.domains.chemistry import molar, millimolar
        conc = DimArray([0.001], molar)
        conc_mm = conc.to(millimolar)
        assert pytest.approx(conc_mm.magnitude()[0], rel=1e-6) == 1.0


class TestEngineeringUnits:
    """Test engineering units."""

    def test_megapascal_dimension(self):
        """MPa should have pressure dimension."""
        from dimtensor.domains.engineering import MPa
        assert MPa.dimension == Dimension(mass=1, length=-1, time=-2)

    def test_megapascal_scale(self):
        """MPa should be 1e6 Pa."""
        from dimtensor.domains.engineering import MPa
        assert MPa.scale == 1e6

    def test_ksi_dimension(self):
        """ksi should have pressure dimension."""
        from dimtensor.domains.engineering import ksi
        assert ksi.dimension == Dimension(mass=1, length=-1, time=-2)

    def test_ksi_scale(self):
        """ksi should be approximately 6.895e6 Pa."""
        from dimtensor.domains.engineering import ksi
        assert pytest.approx(ksi.scale, rel=1e-3) == 6.895e6

    def test_btu_dimension(self):
        """BTU should have energy dimension."""
        from dimtensor.domains.engineering import BTU
        assert BTU.dimension == Dimension(mass=1, length=2, time=-2)

    def test_btu_scale(self):
        """BTU should be approximately 1055 J."""
        from dimtensor.domains.engineering import BTU
        assert pytest.approx(BTU.scale, rel=1e-3) == 1055

    def test_horsepower_dimension(self):
        """Horsepower should have power dimension."""
        from dimtensor.domains.engineering import hp
        assert hp.dimension == Dimension(mass=1, length=2, time=-3)

    def test_horsepower_scale(self):
        """Horsepower should be approximately 745.7 W."""
        from dimtensor.domains.engineering import hp
        assert pytest.approx(hp.scale, rel=1e-3) == 745.7

    def test_kilowatt_hour_dimension(self):
        """kWh should have energy dimension."""
        from dimtensor.domains.engineering import kWh
        assert kWh.dimension == Dimension(mass=1, length=2, time=-2)

    def test_kilowatt_hour_scale(self):
        """kWh should be 3.6e6 J."""
        from dimtensor.domains.engineering import kWh
        assert kWh.scale == 3.6e6

    def test_rpm_dimension(self):
        """rpm should have frequency dimension."""
        from dimtensor.domains.engineering import rpm
        assert rpm.dimension == Dimension(time=-1)

    def test_gpm_dimension(self):
        """gpm should have volumetric flow dimension."""
        from dimtensor.domains.engineering import gpm
        assert gpm.dimension == Dimension(length=3, time=-1)

    def test_foot_pound_torque_dimension(self):
        """ft·lb should have torque/energy dimension."""
        from dimtensor.domains.engineering import ft_lb
        assert ft_lb.dimension == Dimension(mass=1, length=2, time=-2)

    def test_mil_dimension(self):
        """mil should have length dimension."""
        from dimtensor.domains.engineering import mil
        assert mil.dimension == Dimension(length=1)

    def test_mil_scale(self):
        """mil should be 2.54e-5 m (0.001 inch)."""
        from dimtensor.domains.engineering import mil
        assert pytest.approx(mil.scale, rel=1e-6) == 2.54e-5

    def test_pressure_conversion(self):
        """Test pressure unit conversions."""
        from dimtensor.domains.engineering import MPa, ksi
        from dimtensor.core.units import psi
        # 1 ksi = 1000 psi
        stress_ksi = DimArray([1.0], ksi)
        stress_mpa = stress_ksi.to(MPa)
        # 1 ksi ~ 6.895 MPa
        assert pytest.approx(stress_mpa.magnitude()[0], rel=1e-3) == 6.895

    def test_power_conversion(self):
        """Test power unit conversions."""
        from dimtensor.domains.engineering import hp
        from dimtensor.core.units import W
        power_hp = DimArray([1.0], hp)
        power_w = power_hp.to(W)
        assert pytest.approx(power_w.magnitude()[0], rel=1e-3) == 745.7

    def test_metric_vs_mechanical_horsepower(self):
        """Test metric vs mechanical horsepower."""
        from dimtensor.domains.engineering import hp, PS
        # Mechanical hp is about 1.4% more than metric hp
        ratio = hp.scale / PS.scale
        assert pytest.approx(ratio, rel=1e-3) == 1.014


class TestDomainImports:
    """Test that domain modules are importable."""

    def test_import_astronomy(self):
        """Test importing astronomy module."""
        from dimtensor.domains import astronomy
        assert hasattr(astronomy, "parsec")
        assert hasattr(astronomy, "AU")
        assert hasattr(astronomy, "solar_mass")

    def test_import_chemistry(self):
        """Test importing chemistry module."""
        from dimtensor.domains import chemistry
        assert hasattr(chemistry, "molar")
        assert hasattr(chemistry, "dalton")
        assert hasattr(chemistry, "ppm")

    def test_import_engineering(self):
        """Test importing engineering module."""
        from dimtensor.domains import engineering
        assert hasattr(engineering, "MPa")
        assert hasattr(engineering, "hp")
        assert hasattr(engineering, "BTU")

    def test_import_from_dimtensor(self):
        """Test importing domains from main package."""
        from dimtensor import domains
        assert hasattr(domains, "astronomy")
        assert hasattr(domains, "chemistry")
        assert hasattr(domains, "engineering")

    def test_direct_unit_imports(self):
        """Test importing units directly."""
        from dimtensor.domains.astronomy import parsec, AU, solar_mass
        from dimtensor.domains.chemistry import molar, dalton, ppm
        from dimtensor.domains.engineering import MPa, hp, BTU

        # Just verify they're Unit objects
        assert parsec.symbol == "pc"
        assert molar.symbol == "M"
        assert MPa.symbol == "MPa"


class TestCrossDomainCalculations:
    """Test calculations using units from multiple domains."""

    def test_stellar_density(self):
        """Calculate stellar density using astronomy units."""
        from dimtensor.domains.astronomy import solar_mass, solar_radius
        from dimtensor.core.units import kg, m
        import math

        mass = DimArray([1.0], solar_mass)
        radius = DimArray([1.0], solar_radius)

        # Convert to SI units first
        mass_kg = mass.to(kg)
        radius_m = radius.to(m)

        # Density = M / (4/3 * pi * R^3)
        volume = (4/3) * math.pi * (radius_m ** 3)
        density = mass_kg / volume

        # Sun's density is about 1.4 g/cm^3 = 1400 kg/m^3
        assert density.dimension == Dimension(mass=1, length=-3)
        # Check order of magnitude
        assert 1000 < density.magnitude()[0] < 2000

    def test_molecular_energy_conversion(self):
        """Test converting molecular energies."""
        from dimtensor.domains.chemistry import hartree
        from dimtensor.core.units import eV

        # 1 Hartree ~ 27.2 eV
        energy_ha = DimArray([1.0], hartree)
        energy_ev = energy_ha.to(eV)
        assert pytest.approx(energy_ev.magnitude()[0], rel=1e-2) == 27.2

    def test_engineering_energy_conversion(self):
        """Test energy unit conversions across domains."""
        from dimtensor.domains.engineering import BTU, kWh
        from dimtensor.core.units import J

        # Convert 1 kWh to BTU
        energy_kwh = DimArray([1.0], kWh)
        energy_btu = energy_kwh.to(BTU)
        # 1 kWh ~ 3412 BTU
        assert pytest.approx(energy_btu.magnitude()[0], rel=1e-2) == 3412
