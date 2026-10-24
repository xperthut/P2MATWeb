######################################################
# Developer : Methun Kamruzzaman
# Date      : October, 2026
# Purpose   : Compute RDKit + Mordred molecular descriptors from SMILES.
#             Shared by the training pipelines (ModelTrain/featurize.py) and
#             the web app so training and inference prepare molecules
#             identically.
######################################################
"""
utils/featurizer.py
-------------------
RDKit + Mordred descriptor computation (replacement for PaDEL).

Every molecule goes through the same steps:

1. Parse the SMILES and re-parse its canonical form, so the atom order (and
   therefore the 3D embedding) does not depend on how the input was written.
   Disconnected SMILES (mixtures / salts, written with ``.``) are rejected:
   their melting/boiling point is not a property of one structure, and
   distance-based descriptors are undefined for them.
2. ``mode="2d"``: RDKit 2D descriptors (``rd_``) + Mordred 2D (``md_``).
3. ``mode="3d"``: additionally embed one conformer (ETKDGv3, fixed seed, MMFF
   optimisation with UFF fallback) and add RDKit 3D shape descriptors
   (``rd3_``) + Mordred 3D descriptors.

Failed descriptors are returned as NaN.  Column names are prefixed so RDKit
and Mordred never collide; value-level duplicates are removed later, on the
training set only.

Example::

    feats = featurize("CCO", mode="2d")   # dict: column -> float, or None
"""
import math
import warnings
from collections import OrderedDict

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

warnings.filterwarnings("ignore")
# Bump when the features change, so cached features are recomputed.
FEATURIZER_VERSION: int = 2
RDLogger.DisableLog("rdApp.*")

RANDOM_SEED: int = 42
EMBED_TIMEOUT_SECONDS: int = 60

# Scalar 3D shape descriptors from RDKit (the vector ones, e.g. RDF/MORSE/WHIM,
# overlap heavily with Mordred's 3D set and are left out).
_RDKIT_3D = OrderedDict([
    ("PMI1", rdMolDescriptors.CalcPMI1),
    ("PMI2", rdMolDescriptors.CalcPMI2),
    ("PMI3", rdMolDescriptors.CalcPMI3),
    ("NPR1", rdMolDescriptors.CalcNPR1),
    ("NPR2", rdMolDescriptors.CalcNPR2),
    ("RadiusOfGyration", rdMolDescriptors.CalcRadiusOfGyration),
    ("InertialShapeFactor", rdMolDescriptors.CalcInertialShapeFactor),
    ("Eccentricity", rdMolDescriptors.CalcEccentricity),
    ("Asphericity", rdMolDescriptors.CalcAsphericity),
    ("SpherocityIndex", rdMolDescriptors.CalcSpherocityIndex),
    ("PBF", rdMolDescriptors.CalcPBF),
])

_calculators: dict = {}   # mode -> mordred Calculator, built lazily


def _mordred_calculator(mode: str):
    """Return a cached Mordred calculator for *mode* (``"2d"`` or ``"3d"``)."""
    if mode not in _calculators:
        from mordred import Calculator, descriptors
        _calculators[mode] = Calculator(descriptors, ignore_3D=(mode == "2d"))
    return _calculators[mode]


def _to_float(value) -> float:
    """Convert a descriptor value to float; anything non-numeric becomes NaN."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


def is_disconnected(smiles: str) -> bool:
    """True if *smiles* parses to more than one fragment (mixture / salt)."""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None and len(Chem.GetMolFrags(mol)) > 1


def prepare_mol(smiles: str):
    """Parse *smiles* into a canonical RDKit Mol.

    Returns ``None`` if the SMILES is invalid or disconnected.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or len(Chem.GetMolFrags(mol)) > 1:
        return None
    return Chem.MolFromSmiles(Chem.MolToSmiles(mol))


def canonical_smiles(smiles: str):
    """Return RDKit canonical SMILES for *smiles*, or ``None`` if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else Chem.MolToSmiles(mol)


def embed_3d(mol):
    """Return a copy of *mol* with explicit Hs and one optimised conformer.

    Returns ``None`` when embedding fails.
    """
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = RANDOM_SEED
    params.timeout = EMBED_TIMEOUT_SECONDS
    if AllChem.EmbedMolecule(mol_h, params) != 0:
        # Retry with random starting coordinates (helps large / caged systems).
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol_h, params) != 0:
            return None
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol_h):
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=2000)
        else:
            AllChem.UFFOptimizeMolecule(mol_h, maxIters=2000)
    except Exception:
        pass   # keep the unoptimised ETKDG geometry
    return mol_h


def featurize(smiles: str, mode: str = "2d"):
    """Compute RDKit + Mordred descriptors for one SMILES.

    Args:
        smiles: Input SMILES string.
        mode:   ``"2d"`` or ``"3d"``.

    Returns:
        An ``OrderedDict`` mapping column name to float (NaN for failed
        descriptors), or ``None`` if the SMILES is invalid or disconnected,
        or (3D) the molecule cannot be embedded.
    """
    if mode not in ("2d", "3d"):
        raise ValueError(f"mode must be '2d' or '3d', got {mode!r}")

    mol = prepare_mol(smiles)
    if mol is None:
        return None

    feats = OrderedDict()
    for name, value in Descriptors.CalcMolDescriptors(mol).items():
        feats[f"rd_{name}"] = _to_float(value)
    # Raw Ipc grows exponentially with size (up to ~1e29 here) and swamps
    # linear models; the averaged form is the usual bounded replacement.
    feats["rd_Ipc"] = _to_float(Descriptors.Ipc(mol, avg=True))

    target = mol
    if mode == "3d":
        target = embed_3d(mol)
        if target is None:
            return None
        for name, fn in _RDKIT_3D.items():
            try:
                feats[f"rd3_{name}"] = _to_float(fn(target))
            except Exception:
                feats[f"rd3_{name}"] = math.nan

    result = _mordred_calculator(mode)(target)
    for name, value in zip(result.keys(), result.values()):
        feats[f"md_{name}"] = _to_float(value)

    return feats
