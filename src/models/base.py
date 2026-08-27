"""
Shared TabPFN factory used by all strategy modules.
Update this one function if the TabPFN API changes.
"""
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion


def _get_tabpfn(
    device: str = "auto",
    n_estimators: int = 16,
    seed: int = 42,
    ignore_pretraining_limits: bool = True,
) -> TabPFNClassifier:
    """
    Returns a TabPFNClassifier (v3) with the given settings.

    Args:
        device: Device string passed to TabPFN (e.g. 'auto', 'cuda:0', 'cpu').
        n_estimators: Number of ensemble estimators.
        seed: Random seed for reproducibility.
        ignore_pretraining_limits: Allow datasets larger than TabPFN pretraining limits.
    """
    return TabPFNClassifier.create_default_for_version(
        ModelVersion.V3,
        device=device,
        n_estimators=n_estimators,
        random_state=seed,
        ignore_pretraining_limits=ignore_pretraining_limits,
    )
