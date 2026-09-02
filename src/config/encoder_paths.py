from pathlib import Path


# Project root:
# Feline_Transfer_Learning/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXTERNAL_ROOT = PROJECT_ROOT / "external"
WEIGHTS_ROOT = PROJECT_ROOT / "weights"


# ------------------------------------------------------------------
# USFM
# ------------------------------------------------------------------

USFM_SOURCE = EXTERNAL_ROOT / "USFM"

USFM_CHECKPOINT = (
    WEIGHTS_ROOT
    / "usfm"
    / "USFM_latest.pth"
)


# ------------------------------------------------------------------
# OpenUS
# ------------------------------------------------------------------

OPENUS_SOURCE = EXTERNAL_ROOT / "OpenUS"

OPENUS_CHECKPOINT = (
    WEIGHTS_ROOT
    / "openus"
    / "OpenUS-S.pth"
)