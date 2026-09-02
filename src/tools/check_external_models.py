from pathlib import Path
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = PROJECT_ROOT / "external"


def print_status(name, path):
    exists = path.exists()

    status = "[OK]" if exists else "[MISSING]"
    print(f"{status:<10} {name}")
    print(f"           {path}")

    if exists and path.is_file():
        size_mb = path.stat().st_size / (1024 ** 2)
        print(f"           size: {size_mb:.2f} MB")

    return exists


def check_checkpoint(path):
    """Very lightweight checkpoint inspection."""
    print(f"\nInspecting checkpoint: {path}")

    try:
        ckpt = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        print(f"Checkpoint type: {type(ckpt)}")

        if isinstance(ckpt, dict):
            print("Top-level keys:")
            for key in ckpt.keys():
                print(f"  - {key}")

            # common checkpoint structures
            for key in [
                "model",
                "state_dict",
                "teacher",
                "student",
                "module",
            ]:
                if key in ckpt:
                    value = ckpt[key]

                    if isinstance(value, dict):
                        print(
                            f"'{key}' contains "
                            f"{len(value)} entries"
                        )

                        sample_keys = list(value.keys())[:10]

                        print("Sample parameter keys:")
                        for k in sample_keys:
                            print(f"    {k}")

        else:
            print(
                "Checkpoint is not a dictionary. "
                "Inspect manually."
            )

        print("[CHECKPOINT READ OK]")

    except Exception as e:
        print("[CHECKPOINT READ FAILED]")
        print(e)


def check_usfm():
    print("\n" + "=" * 80)
    print("USFM")
    print("=" * 80)

    root = EXTERNAL_ROOT / "USFM"

    repo_ok = print_status(
        "USFM repository",
        root,
    )

    print_status(
        "USFM model source",
        root / "usdsgen" / "model",
    )

    checkpoint = (
        root
        / "assets"
        / "FMweight"
        / "USFM_latest.pth"
    )

    ckpt_ok = print_status(
        "USFM checkpoint",
        checkpoint,
    )

    if repo_ok and ckpt_ok:
        check_checkpoint(checkpoint)


def check_openus():
    print("\n" + "=" * 80)
    print("OpenUS")
    print("=" * 80)

    root = EXTERNAL_ROOT / "OpenUS"

    repo_ok = print_status(
        "OpenUS repository",
        root,
    )

    print_status(
        "VMamba model source",
        root / "vmamba_models" / "vmamba.py",
    )

    # Official vanilla VMamba initialization
    vmamba_init = (
        root
        / "pretrained"
        / "vmamba"
        / "vssm_small_0229_ckpt_epoch_222.pth"
    )

    print_status(
        "VMamba-S ImageNet init",
        vmamba_init,
    )

    # Change this filename after downloading OpenUS-S
    openus_candidates = [
        root / "weights" / "OpenUS-S.pth",
        root / "weights" / "openus_s.pth",
        root / "OpenUS-S.pth",
    ]

    openus_ckpt = None

    for candidate in openus_candidates:
        if candidate.exists():
            openus_ckpt = candidate
            break

    if openus_ckpt is None:
        print("[MISSING]  OpenUS pretrained checkpoint")
        print("           Searched:")

        for path in openus_candidates:
            print(f"           - {path}")

    else:
        print_status(
            "OpenUS pretrained checkpoint",
            openus_ckpt,
        )

        check_checkpoint(openus_ckpt)

    return repo_ok


if __name__ == "__main__":
    print(f"Project root : {PROJECT_ROOT}")
    print(f"External root: {EXTERNAL_ROOT}")

    check_usfm()
    check_openus()