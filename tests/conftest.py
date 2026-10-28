from importlib import resources

import gufe
import openfe
import openmm
import pytest
from openff.units import unit
from openmm import app
from rdkit import Chem


@pytest.fixture(scope="session")
def T4_pdb_path() -> str:
    with resources.as_file(resources.files("openfe.tests.data")) as d:
        return str(d / "181l_only.pdb")


@pytest.fixture(scope="session")
def T4_protein_component(T4_pdb_path) -> gufe.ProteinComponent:
    return gufe.ProteinComponent.from_pdb_file(T4_pdb_path, name="T4_protein")


@pytest.fixture(scope="session")
def T4_pdb(T4_pdb_path) -> app.PDBFile:
    return app.PDBFile(T4_pdb_path)


@pytest.fixture(scope="session")
def benzene_modifications() -> dict[str, gufe.SmallMoleculeComponent]:
    files = {}
    with resources.as_file(resources.files("openfe.tests.data")) as d:
        supp = Chem.SDMolSupplier(str(d / "benzene_modifications_am1bcc.sdf"), removeHs=False)
        for rdmol in supp:
            files[rdmol.GetProp("_Name")] = gufe.SmallMoleculeComponent(rdmol)
    return files


@pytest.fixture(scope="session")
def toluene(benzene_modifications) -> gufe.SmallMoleculeComponent:
    return benzene_modifications["toluene"]


@pytest.fixture
def toluene_complex_system(toluene, T4_protein_component) -> gufe.ChemicalSystem:
    return gufe.ChemicalSystem(
        {
            "ligand": toluene,
            "solvent": openfe.SolventComponent(
                positive_ion="Na", negative_ion="Cl", ion_concentration=0.15 * unit.molar
            ),
            "protein": T4_protein_component,
        }
    )


@pytest.fixture(scope="session")
def cpu_platform() -> openmm.Platform:
    return openmm.Platform.getPlatformByName("CPU")


@pytest.fixture(scope="session")
def reference_platform() -> openmm.Platform:
    """Double precision platform for exact energy comparisons."""
    return openmm.Platform.getPlatformByName("Reference")
