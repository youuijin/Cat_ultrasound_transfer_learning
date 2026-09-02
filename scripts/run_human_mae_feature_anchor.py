"""Screen fixed ImageNet feature anchoring for full Human MAE adaptation."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import statistics
from copy import deepcopy
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path: sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib.pyplot as plt
import torch

from src.human_ssl.data import build_ssl_loaders, discover_ssl_samples, split_ssl_samples
from src.human_ssl.feature_anchor import ANCHOR_BLOCKS, final_feature_drift
from src.human_ssl.mae import VisionMAE


RUNS = (("baseline_full", 0.0), ("lambda_0p01", 0.01),
        ("lambda_0p1", 0.1), ("lambda_1p0", 1.0))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--baseline-config", type=Path,
                        default=Path("checkpoints/human_mae_vit_b16_trajectory/config.json"))
    parser.add_argument("--runs-dir", type=Path,
                        default=Path("runs/human_mae_feature_anchor"))
    parser.add_argument("--results-dir", type=Path,
                        default=Path("results/human_mae_feature_anchor"))
    parser.add_argument("--repro-results-dir", type=Path,
                        default=Path("results/human_mae_feature_anchor_reproducibility"))
    parser.add_argument("--mode", choices=("screening", "reproduce-0p01", "layer-ablation",
                                           "reproduce-all-blocks"),
                        default="screening")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def command_run(command, label):
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}", flush=True)
    print(subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def read_json(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def config_matches(path: Path, expected: dict) -> bool:
    if not path.is_file(): return False
    config = read_json(path)
    return all((config.get(key, 0.0) if key == "feature_anchor_lambda" else config.get(key)) == value
               for key, value in expected.items())


def is_run_complete(root: Path, expected: dict, epochs: int,
                    final_artifact="final_feature_drift.csv") -> bool:
    """Validate an SSL run before reuse; never accept a partial checkpoint."""
    root = Path(root); expected_epoch = epochs - 1
    metrics_path = root / "ssl_metrics.csv"
    if not metrics_path.is_file():
        metrics_path = root / "metrics.csv"
    found_epoch = None
    if metrics_path.is_file():
        try:
            rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
            found_epoch = int(rows[-1]["epoch"]) if rows else None
        except (KeyError, ValueError, OSError):
            found_epoch = None
    checkpoint_epoch = None
    checkpoint = root / "last_encoder.pt"
    if checkpoint.is_file():
        try:
            checkpoint_epoch = int(torch.load(
                checkpoint, map_location="cpu", weights_only=False).get("epoch", -1))
        except (KeyError, TypeError, ValueError, OSError):
            checkpoint_epoch = None
    complete = (config_matches(root / "config.json", expected) and metrics_path.is_file() and
                found_epoch == expected_epoch and checkpoint.is_file() and
                checkpoint_epoch == expected_epoch and
                (final_artifact is None or (root / final_artifact).is_file()))
    if not complete and root.exists():
        print(f"[incomplete] {root.resolve()}")
        print(f"expected final epoch: {expected_epoch}")
        print(f"found final epoch: {found_epoch}")
        print("-> rerun required")
    return complete


def mae_expected(source, weight, args, seed=0, anchor_layers=ANCHOR_BLOCKS):
    expected = {"encoder": "vit_b16", "human1_root": source["human1_root"],
            "human2_root": source["human2_root"], "human3_root": source["human3_root"],
            "val_fraction": source["val_fraction"], "mask_ratio": source["mask_ratio"],
            "norm_pixel_loss": source["norm_pixel_loss"], "decoder_dim": source["decoder_dim"],
            "decoder_depth": source["decoder_depth"], "decoder_heads": source["decoder_heads"],
            "batch_size": source["batch_size"], "epochs": args.epochs, "lr": source["lr"],
            "weight_decay": source["weight_decay"], "warmup_epochs": source["warmup_epochs"],
            "seed": seed, "encoder_lr_scale": 1.0,
            "encoder_trainable_last_blocks": None, "feature_anchor_lambda": weight}
    if weight > 0:
        expected["feature_anchor_layers"] = list(anchor_layers)
        expected["feature_anchor_pooling"] = "mean patch tokens; CLS excluded"
        expected["feature_anchor_input"] = "same unmasked normalized Human image"
        expected["feature_anchor_teacher"] = "frozen ImageNet ViT-B/16"
    return expected


def probe_expected(init="human_mae", seed=0):
    reference = read_json("runs/human_ssl_trajectory_probe/epoch_000/config.json")
    return {"dataset": "human2", "data_root": reference["data_root"], "encoder": "vit_b16",
            "encoder_init": init, "transfer": "frozen", "val_fraction": 0.2,
            "split_seed": 42, "seed": seed, "batch_size": 8, "epochs": 50,
            "lr": 1e-4, "weight_decay": 1e-4,
            "train_subjects": reference["train_subjects"],
            "val_subjects": reference["val_subjects"]}


def mae_command(args, source, name, weight, output, seed=0, anchor_layers=ANCHOR_BLOCKS):
    command = [args.python, "train_human_ssl.py", "--method", "mae", "--encoder", "vit_b16",
            "--human1-root", source["human1_root"], "--human2-root", source["human2_root"],
            "--human3-root", source["human3_root"], "--val-fraction", str(source["val_fraction"]),
            "--mask-ratio", str(source["mask_ratio"]), "--no-norm-pixel-loss",
            "--decoder-dim", str(source["decoder_dim"]),
            "--decoder-depth", str(source["decoder_depth"]),
            "--decoder-heads", str(source["decoder_heads"]),
            "--batch-size", str(source["batch_size"]), "--epochs", str(args.epochs),
            "--lr", str(source["lr"]), "--encoder-lr-scale", "1.0",
            "--weight-decay", str(source["weight_decay"]),
            "--warmup-epochs", str(source["warmup_epochs"]),
            "--num-workers", str(args.num_workers), "--seed", str(seed), "--run-name", name,
            "--feature-anchor-lambda", str(weight), "--output-dir", str(output),
            "--amp" if args.amp else "--no-amp"]
    if weight > 0:
        command.extend(["--feature-anchor-layers", *map(str, anchor_layers)])
    return command


def probe_command(args, source, checkpoint, output, ssl_config, seed=0):
    return [args.python, "-m", "src.train_human_segmentation", "--dataset", "human2",
            "--data-root", source["human2_root"], "--encoder", "vit_b16",
            "--encoder-init", "human_mae", "--encoder-checkpoint", str(checkpoint),
            "--transfer", "frozen", "--val-fraction", "0.2", "--split-seed", "42",
            "--seed", str(seed), "--batch-size", "8", "--epochs", "50", "--lr", "1e-4",
            "--weight-decay", "1e-4", "--num-workers", str(args.num_workers),
            "--run-dir", str(output), "--ssl-reference-config", str(ssl_config),
            "--amp" if args.amp else "--no-amp"]


def compute_reused_baseline_drift(args, source, checkpoint, output, seed=0,
                                  blocks=ANCHOR_BLOCKS):
    model = VisionMAE("vit_b16", source["decoder_dim"], source["decoder_depth"],
                      source["decoder_heads"], source["norm_pixel_loss"])
    teacher = deepcopy(model.encoder).requires_grad_(False).eval()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.encoder.load_state_dict(payload["state_dict"], strict=True)
    samples = discover_ssl_samples({name: Path(source[f"{name}_root"])
                                    for name in ("human1", "human2", "human3")})
    _train, validation = split_ssl_samples(samples, source["val_fraction"], seed)
    _train_loader, val_loader = build_ssl_loaders(
        validation, validation, model.image_size, source["batch_size"],
        args.num_workers, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device); teacher.to(device)
    final_feature_drift(model, teacher, val_loader, device, output, tuple(blocks))


def best_validation(path):
    with Path(path).open(encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["phase"] == "validation"]
    return max(rows, key=lambda row: float(row["mean_dice"]))


def read_last(path):
    with Path(path).open(encoding="utf-8") as handle: return list(csv.DictReader(handle))[-1]


def read_drift(path):
    with Path(path).open(encoding="utf-8") as handle: return list(csv.DictReader(handle))


def resolve_baseline(args, source):
    target = args.runs_dir / "baseline_full"; target.mkdir(parents=True, exist_ok=True)
    existing = Path("runs/human_mae_recipe_ablation/baseline")
    if not config_matches(existing / "config.json", mae_expected(source, 0.0, args)):
        raise RuntimeError("Existing full baseline config does not exactly match")
    probe = existing / "human_frozen_probe"
    if not config_matches(probe / "config.json", probe_expected()):
        raise RuntimeError("Existing full baseline downstream config does not exactly match")
    (target / "reuse.json").write_text(json.dumps({"source": str(existing.resolve())}, indent=2),
                                        encoding="utf-8")
    drift = target / "final_feature_drift.csv"
    if args.force or not drift.is_file():
        compute_reused_baseline_drift(args, source, existing / "last_encoder.pt", drift)
    print(f"[reuse] baseline_full -> {existing.resolve()}")
    return existing, probe, drift


def resolve_anchor(args, source, name, weight):
    root = args.runs_dir / name; probe = root / "human_frozen_probe"
    valid = is_run_complete(root, mae_expected(source, weight, args), args.epochs)
    if args.force or not valid:
        command_run(mae_command(args, source, name, weight, root), f"Full MAE feature anchor {weight}")
    else: print(f"[reuse] {name} SSL -> {root.resolve()}")
    probe_valid = config_matches(probe / "config.json", probe_expected()) and (probe / "metrics.csv").is_file()
    if args.force or not probe_valid:
        command_run(probe_command(args, source, root / "last_encoder.pt", probe,
                                  root / "config.json"), f"Frozen Human2 probe {name}")
    else: print(f"[reuse] {name} probe -> {probe.resolve()}")
    return root, probe, root / "final_feature_drift.csv"


def collect(name, weight, ssl, probe, drift_path):
    metrics_path = ssl / ("ssl_metrics.csv" if (ssl / "ssl_metrics.csv").is_file() else "metrics.csv")
    metrics = read_last(metrics_path); best = best_validation(probe / "metrics.csv")
    drift = read_drift(drift_path); by_layer = {row["layer"]: row for row in drift}
    return {"run_name": name, "lambda_feature": weight,
            "ssl_final_train_mae_loss": metrics["train_mae_loss"],
            "ssl_final_val_mae_loss": metrics["validation_mae_loss"],
            "ssl_final_feature_loss": metrics.get("validation_feature_preserve_loss", ""),
            "ssl_final_total_loss": metrics.get("validation_total_loss", metrics["validation_mae_loss"]),
            "mean_feature_drift": sum(float(row["drift_1_minus_cka"]) for row in drift) / len(drift),
            **{f"block{block}_drift": by_layer[f"block_{block}"]["drift_1_minus_cka"]
               for block in ANCHOR_BLOCKS},
            "human_frozen_dice": best["mean_dice"], "human_frozen_iou": best["kidney_iou"],
            "human_frozen_loss": best["loss"],
            "final_checkpoint": str((ssl / "last_encoder.pt").resolve())}


def image_reference():
    root = Path("runs/human_ssl_trajectory_probe/epoch_000")
    if not config_matches(root / "config.json", probe_expected("imagenet")):
        raise RuntimeError("Existing ImageNet reference config does not exactly match")
    best = best_validation(root / "metrics.csv")
    return {"run_name": "imagenet_reference", "lambda_feature": "",
            "ssl_final_train_mae_loss": "", "ssl_final_val_mae_loss": "",
            "ssl_final_feature_loss": "", "ssl_final_total_loss": "",
            "mean_feature_drift": 0, **{f"block{block}_drift": 0 for block in ANCHOR_BLOCKS},
            "human_frozen_dice": best["mean_dice"], "human_frozen_iou": best["kidney_iou"],
            "human_frozen_loss": best["loss"], "final_checkpoint": "imagenet_pretrained"}


def write_summary(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def plots(args, rows):
    experiments = [row for row in rows if row["run_name"] != "imagenet_reference"]
    x = [float(row["lambda_feature"]) for row in experiments]
    reference = float(rows[0]["human_frozen_dice"])
    fig, ax = plt.subplots(); ax.plot(x, [float(r["human_frozen_dice"]) for r in experiments], marker="o")
    ax.axhline(reference, color="black", linestyle="--"); ax.set(xlabel="Feature anchor lambda", ylabel="Human frozen Dice")
    fig.tight_layout(); fig.savefig(args.results_dir / "lambda_vs_dice.png", dpi=200); plt.close(fig)
    fig, ax = plt.subplots(); ax.plot(x, [float(r["mean_feature_drift"]) for r in experiments], marker="o")
    ax.set(xlabel="Feature anchor lambda", ylabel="Mean feature drift")
    fig.tight_layout(); fig.savefig(args.results_dir / "lambda_vs_feature_drift.png", dpi=200); plt.close(fig)
    fig, ax = plt.subplots()
    ax.scatter([float(r["mean_feature_drift"]) for r in experiments],
               [float(r["human_frozen_dice"]) for r in experiments])
    for row in experiments: ax.annotate(row["run_name"], (float(row["mean_feature_drift"]), float(row["human_frozen_dice"])))
    ax.set(xlabel="Mean feature drift", ylabel="Human frozen Dice")
    fig.tight_layout(); fig.savefig(args.results_dir / "drift_vs_dice.png", dpi=200); plt.close(fig)


def require_run(root, expected, artifacts, label):
    root = Path(root)
    if not config_matches(root / "config.json", expected):
        raise RuntimeError(f"{label} config does not exactly match: {root / 'config.json'}")
    missing = [str(root / item) for item in artifacts if not (root / item).is_file()]
    if missing:
        raise RuntimeError(f"{label} is incomplete; missing: {', '.join(missing)}")
    print(f"[reuse] {label} -> {root.resolve()}")
    return root


def repro_paths(seed):
    if seed == 0:
        return {
            "imagenet": Path("runs/human_ssl_trajectory_probe/epoch_000"),
            "full": Path("runs/human_mae_recipe_ablation/baseline"),
            "full_probe": Path("runs/human_mae_recipe_ablation/baseline/human_frozen_probe"),
            "anchor": Path("runs/human_mae_feature_anchor/lambda_0p01"),
        }
    return {
        "imagenet": Path(f"runs/human_mae_adaptation_depth/imagenet/seed{seed}"),
        "full": Path(f"runs/human_mae_adaptation_depth/full/seed{seed}/mae"),
        "full_probe": Path(f"runs/human_mae_adaptation_depth/full/seed{seed}/human_frozen_probe"),
        "anchor": Path(f"runs/human_mae_feature_anchor/lambda_0p01/seed{seed}"),
    }


def collect_repro_method(method, seed, ssl, probe, drift_path=None):
    best = best_validation(probe / "metrics.csv")
    if method == "imagenet":
        mean_drift = 0.0
        checkpoint = "imagenet_pretrained"
    else:
        drift = read_drift(drift_path)
        mean_drift = sum(float(row["drift_1_minus_cka"]) for row in drift) / len(drift)
        checkpoint = str((ssl / "last_encoder.pt").resolve())
    return {"method": method, "seed": seed, "human_frozen_dice": float(best["mean_dice"]),
            "human_frozen_iou": float(best["kidney_iou"]),
            "human_frozen_loss": float(best["loss"]), "mean_feature_drift": mean_drift,
            "final_checkpoint": checkpoint, "probe_dir": str(probe.resolve())}


def mean_std(values):
    values = [float(value) for value in values]
    return statistics.mean(values), statistics.stdev(values)


def paired_rows(rows, comparator):
    indexed = {(row["method"], int(row["seed"])): row for row in rows}
    result = []
    for seed in (0, 1, 2):
        anchor = indexed[("feature_anchor_0p01", seed)]
        other = indexed[(comparator, seed)]
        result.append({"seed": seed, "anchor_dice": anchor["human_frozen_dice"],
                       f"{comparator}_dice": other["human_frozen_dice"],
                       "dice_delta": anchor["human_frozen_dice"] - other["human_frozen_dice"],
                       "anchor_iou": anchor["human_frozen_iou"],
                       f"{comparator}_iou": other["human_frozen_iou"],
                       "iou_delta": anchor["human_frozen_iou"] - other["human_frozen_iou"]})
    return result


def write_repro_outputs(args, rows):
    output = args.repro_results_dir
    output.mkdir(parents=True, exist_ok=True)
    write_summary(output / "all_seed_results.csv", rows)
    summary = []
    for method in ("imagenet", "full_mae", "feature_anchor_0p01"):
        selected = [row for row in rows if row["method"] == method]
        dice_mean, dice_std = mean_std(row["human_frozen_dice"] for row in selected)
        iou_mean, iou_std = mean_std(row["human_frozen_iou"] for row in selected)
        drift_mean, drift_std = mean_std(row["mean_feature_drift"] for row in selected)
        summary.append({"method": method, "n": len(selected), "dice_mean": dice_mean,
                        "dice_std": dice_std, "iou_mean": iou_mean, "iou_std": iou_std,
                        "feature_drift_mean": drift_mean, "feature_drift_std": drift_std})
    write_summary(output / "method_summary.csv", summary)
    comparisons = {}
    for comparator, stem in (("full_mae", "anchor_vs_full"), ("imagenet", "anchor_vs_imagenet")):
        paired = paired_rows(rows, comparator)
        write_summary(output / f"{stem}.csv", paired)
        dice_mean, dice_std = mean_std(row["dice_delta"] for row in paired)
        iou_mean, iou_std = mean_std(row["iou_delta"] for row in paired)
        aggregate = [{"comparison": stem, "n": len(paired), "dice_delta_mean": dice_mean,
                      "dice_delta_std": dice_std, "iou_delta_mean": iou_mean,
                      "iou_delta_std": iou_std}]
        write_summary(output / f"{stem}_summary.csv", aggregate)
        comparisons[comparator] = paired

    fig, ax = plt.subplots()
    for method in ("imagenet", "full_mae", "feature_anchor_0p01"):
        selected = sorted((row for row in rows if row["method"] == method), key=lambda row: row["seed"])
        ax.plot([row["seed"] for row in selected], [row["human_frozen_dice"] for row in selected],
                marker="o", label=method)
    ax.set(xlabel="Seed", ylabel="Human frozen Dice", xticks=[0, 1, 2]); ax.legend()
    fig.tight_layout(); fig.savefig(output / "feature_anchor_seed_reproducibility.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots()
    for comparator, paired in comparisons.items():
        ax.plot([row["seed"] for row in paired], [row["dice_delta"] for row in paired],
                marker="o", label=f"anchor - {comparator}")
    ax.axhline(0, color="black", linestyle="--"); ax.set(xlabel="Seed", ylabel="Paired Dice delta", xticks=[0, 1, 2]); ax.legend()
    fig.tight_layout(); fig.savefig(output / "anchor_paired_delta.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots()
    for method in ("full_mae", "feature_anchor_0p01"):
        selected = [row for row in rows if row["method"] == method]
        ax.scatter([row["mean_feature_drift"] for row in selected],
                   [row["human_frozen_dice"] for row in selected], label=method)
        for row in selected:
            ax.annotate(f"s{row['seed']}", (row["mean_feature_drift"], row["human_frozen_dice"]))
    ax.set(xlabel="Mean feature drift", ylabel="Human frozen Dice"); ax.legend()
    fig.tight_layout(); fig.savefig(output / "feature_drift_vs_dice_seeds.png", dpi=200); plt.close(fig)


def reproduce_0p01(args, source):
    rows = []
    diagnostic_root = args.runs_dir / "reproducibility_diagnostics"
    for seed in (0, 1, 2):
        paths = repro_paths(seed)
        imagenet = require_run(paths["imagenet"], probe_expected("imagenet", seed),
                               ["metrics.csv"], f"ImageNet seed {seed}")
        rows.append(collect_repro_method("imagenet", seed, None, imagenet))

        full = require_run(paths["full"], mae_expected(source, 0.0, args, seed),
                           ["last_encoder.pt"], f"Full MAE seed {seed}")
        full_probe = require_run(paths["full_probe"], probe_expected("human_mae", seed),
                                 ["metrics.csv"], f"Full MAE probe seed {seed}")
        full_drift = diagnostic_root / f"full_mae_seed{seed}_feature_drift.csv"
        if args.force or not full_drift.is_file():
            full_drift.parent.mkdir(parents=True, exist_ok=True)
            compute_reused_baseline_drift(args, source, full / "last_encoder.pt", full_drift, seed)
        rows.append(collect_repro_method("full_mae", seed, full, full_probe, full_drift))

        anchor = paths["anchor"]
        anchor_probe = anchor / "human_frozen_probe"
        anchor_valid = is_run_complete(anchor, mae_expected(source, 0.01, args, seed), args.epochs)
        if args.force or not anchor_valid:
            command_run(mae_command(args, source, f"lambda_0p01_seed{seed}", 0.01, anchor, seed),
                        f"Feature anchor 0.01 seed {seed}")
        else:
            print(f"[reuse] Feature anchor 0.01 seed {seed} -> {anchor.resolve()}")
        probe_valid = (config_matches(anchor_probe / "config.json", probe_expected("human_mae", seed)) and
                       (anchor_probe / "metrics.csv").is_file())
        if args.force or not probe_valid:
            command_run(probe_command(args, source, anchor / "last_encoder.pt", anchor_probe,
                                      anchor / "config.json", seed),
                        f"Feature anchor 0.01 frozen probe seed {seed}")
        else:
            print(f"[reuse] Feature anchor 0.01 probe seed {seed} -> {anchor_probe.resolve()}")
        rows.append(collect_repro_method("feature_anchor_0p01", seed, anchor, anchor_probe,
                                        anchor / "final_feature_drift.csv"))
    rows.sort(key=lambda row: (row["method"], row["seed"]))
    write_repro_outputs(args, rows)
    print(f"Results: {args.repro_results_dir.resolve()}")


LAYER_RUNS = (
    ("anchor_early", (3, 6)),
    ("anchor_late", (9, 11)),
    ("anchor_distributed", ANCHOR_BLOCKS),
    ("anchor_all_blocks", tuple(range(12))),
)


def layer_ssl_metrics(root):
    path = root / ("ssl_metrics.csv" if (root / "ssl_metrics.csv").is_file() else "metrics.csv")
    row = read_last(path)
    return row.get("validation_mae_loss", ""), row.get("validation_feature_preserve_loss", "")


def diagnostic_rows(args, source, run_name, checkpoint, destination):
    if args.force or not destination.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        compute_reused_baseline_drift(args, source, checkpoint, destination, 0, tuple(range(12)))
    rows = read_drift(destination)
    for row in rows:
        row["run_name"] = run_name
    return rows


def layer_ablation(args, source):
    output = Path("results/human_mae_anchor_layer_ablation")
    run_root = Path("runs/human_mae_anchor_layer_ablation")
    output.mkdir(parents=True, exist_ok=True); run_root.mkdir(parents=True, exist_ok=True)
    common = set(ANCHOR_BLOCKS)
    summary = []; all_drift_rows = []

    imagenet = require_run("runs/human_ssl_trajectory_probe/epoch_000",
                           probe_expected("imagenet", 0), ["metrics.csv"], "ImageNet seed 0")
    image_best = best_validation(imagenet / "metrics.csv")
    summary.append({"run_name": "imagenet_reference", "anchor_layers": "", "n_anchor_layers": 0,
                    "lambda_feature": "", "ssl_val_mae_loss": "", "ssl_val_feature_loss": "",
                    "human_frozen_dice": image_best["mean_dice"], "human_frozen_iou": image_best["kidney_iou"],
                    "human_frozen_loss": image_best["loss"], "mean_common_feature_drift": 0.0,
                    "block3_drift": 0.0, "block6_drift": 0.0, "block9_drift": 0.0,
                    "block11_drift": 0.0, "final_checkpoint": "imagenet_pretrained"})
    for block in range(12):
        all_drift_rows.append({"run_name": "imagenet_reference", "layer": f"block_{block}",
                               "cka_to_imagenet": 1.0, "drift_1_minus_cka": 0.0,
                               "cosine_similarity_to_imagenet": 1.0, "feature_preserve_loss": 0.0,
                               "n_validation_images": ""})

    full = require_run("runs/human_mae_recipe_ablation/baseline",
                       mae_expected(source, 0.0, args), ["last_encoder.pt"], "Full MAE seed 0")
    full_probe = require_run(full / "human_frozen_probe", probe_expected("human_mae", 0),
                             ["metrics.csv"], "Full MAE probe seed 0")
    full_drift_rows = diagnostic_rows(args, source, "full_mae", full / "last_encoder.pt",
                                      run_root / "diagnostics" / "full_mae_all_layers.csv")
    all_drift_rows.extend(full_drift_rows)

    def append_model_row(name, layers, ssl, probe, drift_rows):
        by_block = {int(row["layer"].split("_")[-1]): row for row in drift_rows}
        best = best_validation(probe / "metrics.csv")
        mae_loss, feature_loss = layer_ssl_metrics(ssl)
        summary.append({"run_name": name, "anchor_layers": ",".join(map(str, layers)),
                        "n_anchor_layers": len(layers), "lambda_feature": 0.01 if layers else 0.0,
                        "ssl_val_mae_loss": mae_loss, "ssl_val_feature_loss": feature_loss,
                        "human_frozen_dice": best["mean_dice"], "human_frozen_iou": best["kidney_iou"],
                        "human_frozen_loss": best["loss"],
                        "mean_common_feature_drift": sum(float(by_block[b]["drift_1_minus_cka"]) for b in common) / len(common),
                        **{f"block{b}_drift": by_block[b]["drift_1_minus_cka"] for b in ANCHOR_BLOCKS},
                        "final_checkpoint": str((ssl / "last_encoder.pt").resolve())})

    append_model_row("full_mae", (), full, full_probe, full_drift_rows)
    for name, layers in LAYER_RUNS:
        ssl = (Path("runs/human_mae_feature_anchor/lambda_0p01") if name == "anchor_distributed"
               else run_root / name)
        probe = ssl / "human_frozen_probe"
        valid = is_run_complete(ssl, mae_expected(source, 0.01, args, 0, layers), args.epochs)
        if args.force or not valid:
            print(f"anchor layer names: {[f'block_{b}' for b in layers]}")
            print("lambda: 0.01\nteacher frozen status: True\n"
                  "student trainable encoder params: full encoder\nseed: 0")
            command_run(mae_command(args, source, name, 0.01, ssl, 0, layers),
                        f"Layer-set ablation {name}")
        else:
            print(f"[reuse] {name} SSL -> {ssl.resolve()}")
        if args.force or not (config_matches(probe / "config.json", probe_expected("human_mae", 0)) and
                              (probe / "metrics.csv").is_file()):
            command_run(probe_command(args, source, ssl / "last_encoder.pt", probe,
                                      ssl / "config.json", 0), f"Frozen Human2 probe {name}")
        else:
            print(f"[reuse] {name} probe -> {probe.resolve()}")
        all_path = ssl / "all_layer_feature_drift.csv"
        drift_rows = diagnostic_rows(args, source, name, ssl / "last_encoder.pt", all_path)
        all_drift_rows.extend(drift_rows)
        append_model_row(name, layers, ssl, probe, drift_rows)

    write_summary(output / "layer_ablation_summary.csv", summary)
    by_name = {row["run_name"]: row for row in summary}
    full_row, image_row = by_name["full_mae"], by_name["imagenet_reference"]
    deltas = []
    for name, _layers in LAYER_RUNS:
        row = by_name[name]
        deltas.append({"run_name": name,
                       "dice_minus_full": float(row["human_frozen_dice"]) - float(full_row["human_frozen_dice"]),
                       "dice_minus_imagenet": float(row["human_frozen_dice"]) - float(image_row["human_frozen_dice"]),
                       "drift_reduction_vs_full": float(full_row["mean_common_feature_drift"]) - float(row["mean_common_feature_drift"]),
                       "ssl_loss_difference_vs_full": float(row["ssl_val_mae_loss"]) - float(full_row["ssl_val_mae_loss"])})
    write_summary(output / "anchor_layer_deltas.csv", deltas)
    write_summary(output / "all_layer_feature_drift.csv", all_drift_rows)

    labels = ["ImageNet", "Full", "Early", "Late", "Distributed", "All blocks"]
    dice = [float(row["human_frozen_dice"]) for row in summary]
    drift = [float(row["mean_common_feature_drift"]) for row in summary]
    fig, ax = plt.subplots(); ax.plot(labels, dice, marker="o"); ax.axhline(dice[0], color="black", linestyle="--")
    ax.set(ylabel="Human frozen Dice"); fig.autofmt_xdate(rotation=20); fig.tight_layout()
    fig.savefig(output / "anchor_layer_set_vs_dice.png", dpi=200); plt.close(fig)
    fig, ax = plt.subplots(); ax.plot(labels, drift, marker="o"); ax.set(ylabel="Mean common feature drift")
    fig.autofmt_xdate(rotation=20); fig.tight_layout(); fig.savefig(output / "anchor_layer_set_vs_drift.png", dpi=200); plt.close(fig)
    fig, ax = plt.subplots(); ax.scatter(drift, dice)
    for row, x, y in zip(summary, drift, dice): ax.annotate(row["run_name"], (x, y))
    ax.set(xlabel="Mean common feature drift", ylabel="Human frozen Dice"); fig.tight_layout()
    fig.savefig(output / "drift_vs_dice_layer_ablation.png", dpi=200); plt.close(fig)
    print(f"Results: {output.resolve()}")


def probe_complete(root, expected, epochs=50):
    root = Path(root); metrics = root / "metrics.csv"; found = None
    if metrics.is_file():
        try:
            rows = list(csv.DictReader(metrics.open(encoding="utf-8")))
            found = max(int(row["epoch"]) for row in rows) if rows else None
        except (KeyError, ValueError, OSError):
            found = None
    complete = config_matches(root / "config.json", expected) and found == epochs - 1
    if not complete and root.exists():
        print(f"[incomplete] {root.resolve()}")
        print(f"expected final epoch: {epochs - 1}")
        print(f"found final epoch: {found}")
        print("-> rerun required")
    return complete


def all_blocks_paths(seed):
    if seed == 0:
        return Path("runs/human_mae_anchor_layer_ablation/anchor_all_blocks")
    return Path(f"runs/human_mae_all_blocks_reproducibility/seed{seed}")


def completed_control_paths(method, seed):
    paths = repro_paths(seed)
    if method == "imagenet": return paths["imagenet"], paths["imagenet"]
    if method == "full_mae": return paths["full"], paths["full_probe"]
    return paths["anchor"], paths["anchor"] / "human_frozen_probe"


def ssl_final_mae(root):
    path = root / ("ssl_metrics.csv" if (root / "ssl_metrics.csv").is_file() else "metrics.csv")
    return float(read_last(path)["validation_mae_loss"])


def drift_values(path):
    rows = read_drift(path); indexed = {int(row["layer"].split("_")[-1]): row for row in rows}
    selected = [float(indexed[block]["drift_1_minus_cka"]) for block in ANCHOR_BLOCKS]
    return statistics.mean(selected), {block: float(indexed[block]["drift_1_minus_cka"])
                                       for block in ANCHOR_BLOCKS}


def descriptive_delta(rows, delta_key):
    values = [float(row[delta_key]) for row in rows]
    return {f"mean_{delta_key}": statistics.mean(values),
            f"std_{delta_key}": statistics.stdev(values),
            "positive_delta_count": sum(value > 0 for value in values),
            "negative_delta_count": sum(value < 0 for value in values),
            "total_seeds": len(values)}


def reproduce_all_blocks(args, source):
    output = Path("results/human_mae_all_blocks_reproducibility")
    diagnostics = Path("runs/human_mae_all_blocks_reproducibility/diagnostics")
    output.mkdir(parents=True, exist_ok=True); diagnostics.mkdir(parents=True, exist_ok=True)
    rows = []
    method_specs = (("imagenet", 0.0, ()), ("full_mae", 0.0, ()),
                    ("anchor_distributed_0p01", 0.01, ANCHOR_BLOCKS))
    for seed in (0, 1, 2):
        for method, weight, layers in method_specs:
            ssl, probe = completed_control_paths(method, seed)
            if method == "imagenet":
                if not probe_complete(probe, probe_expected("imagenet", seed)):
                    raise RuntimeError(f"ImageNet seed {seed} control is incomplete; controls are not retrained")
                drift, ssl_loss, checkpoint = 0.0, "", "imagenet_pretrained"
            else:
                expected = mae_expected(source, weight, args, seed, layers or ANCHOR_BLOCKS)
                expected_artifact = None if method == "full_mae" else "final_feature_drift.csv"
                if not is_run_complete(ssl, expected, args.epochs, expected_artifact):
                    raise RuntimeError(f"{method} seed {seed} control is incomplete; controls are not retrained")
                if not probe_complete(probe, probe_expected("human_mae", seed)):
                    raise RuntimeError(f"{method} seed {seed} probe is incomplete; controls are not retrained")
                drift_path = (ssl / "final_feature_drift.csv" if method.startswith("anchor_") else
                              diagnostics / f"full_mae_seed{seed}_feature_drift.csv")
                if not drift_path.is_file():
                    compute_reused_baseline_drift(args, source, ssl / "last_encoder.pt", drift_path, seed)
                drift, _ = drift_values(drift_path); ssl_loss = ssl_final_mae(ssl)
                checkpoint = str((ssl / "last_encoder.pt").resolve())
            best = best_validation(probe / "metrics.csv")
            rows.append({"method": method, "seed": seed, "human_frozen_dice": float(best["mean_dice"]),
                         "human_frozen_iou": float(best["kidney_iou"]),
                         "human_frozen_loss": float(best["loss"]), "ssl_val_mae_loss": ssl_loss,
                         "mean_feature_drift": drift, "checkpoint": checkpoint,
                         "reused_existing_run": True, "run_complete": True})

        ssl = all_blocks_paths(seed); probe = ssl / "human_frozen_probe"; layers = tuple(range(12))
        reused = is_run_complete(ssl, mae_expected(source, 0.01, args, seed, layers), args.epochs)
        if not reused:
            command_run(mae_command(args, source, f"anchor_all_blocks_seed{seed}", 0.01,
                                    ssl, seed, layers), f"All-block anchor seed {seed}")
            if not is_run_complete(ssl, mae_expected(source, 0.01, args, seed, layers), args.epochs):
                raise RuntimeError(f"All-block seed {seed} did not produce a complete SSL run")
        if not probe_complete(probe, probe_expected("human_mae", seed)):
            command_run(probe_command(args, source, ssl / "last_encoder.pt", probe,
                                      ssl / "config.json", seed), f"All-block frozen probe seed {seed}")
            if not probe_complete(probe, probe_expected("human_mae", seed)):
                raise RuntimeError(f"All-block seed {seed} downstream probe is incomplete")
        drift, _ = drift_values(ssl / "final_feature_drift.csv")
        best = best_validation(probe / "metrics.csv")
        rows.append({"method": "anchor_all_blocks_0p01", "seed": seed,
                     "human_frozen_dice": float(best["mean_dice"]),
                     "human_frozen_iou": float(best["kidney_iou"]),
                     "human_frozen_loss": float(best["loss"]), "ssl_val_mae_loss": ssl_final_mae(ssl),
                     "mean_feature_drift": drift, "checkpoint": str((ssl / "last_encoder.pt").resolve()),
                     "reused_existing_run": reused, "run_complete": True})

    rows.sort(key=lambda row: (row["method"], row["seed"])); write_summary(output / "all_seed_results.csv", rows)
    summaries = []
    methods = ("imagenet", "full_mae", "anchor_distributed_0p01", "anchor_all_blocks_0p01")
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        dm, ds = mean_std(row["human_frozen_dice"] for row in selected)
        im, ins = mean_std(row["human_frozen_iou"] for row in selected)
        lm, ls = mean_std(row["human_frozen_loss"] for row in selected)
        fm, fs = mean_std(row["mean_feature_drift"] for row in selected)
        summaries.append({"method": method, "n_seeds": 3, "mean_dice": dm, "std_dice": ds,
                          "mean_iou": im, "std_iou": ins, "mean_loss": lm, "std_loss": ls,
                          "mean_feature_drift": fm, "std_feature_drift": fs})
    write_summary(output / "method_summary.csv", summaries)
    index = {(row["method"], row["seed"]): row for row in rows}
    comparisons = {}
    for comparator, stem in (("imagenet", "all_blocks_vs_imagenet"),
                             ("full_mae", "all_blocks_vs_full"),
                             ("anchor_distributed_0p01", "all_blocks_vs_distributed")):
        paired = []
        for seed in (0, 1, 2):
            base=index[(comparator, seed)]; allb=index[("anchor_all_blocks_0p01", seed)]
            if comparator == "imagenet":
                paired.append({"seed": seed, "imagenet_dice": base["human_frozen_dice"],
                               "all_blocks_dice": allb["human_frozen_dice"],
                               "delta_dice": allb["human_frozen_dice"]-base["human_frozen_dice"],
                               "imagenet_iou": base["human_frozen_iou"], "all_blocks_iou": allb["human_frozen_iou"],
                               "delta_iou": allb["human_frozen_iou"]-base["human_frozen_iou"]})
            else:
                prefix = "full" if comparator == "full_mae" else "distributed"
                item={"seed": seed, f"{prefix}_dice": base["human_frozen_dice"],
                      "all_blocks_dice": allb["human_frozen_dice"],
                      "delta_dice": allb["human_frozen_dice"]-base["human_frozen_dice"],
                      f"{prefix}_feature_drift": base["mean_feature_drift"],
                      "all_blocks_feature_drift": allb["mean_feature_drift"]}
                if comparator == "full_mae": item["drift_reduction"] = base["mean_feature_drift"]-allb["mean_feature_drift"]
                paired.append(item)
        write_summary(output / f"{stem}.csv", paired); summary = descriptive_delta(paired, "delta_dice")
        if comparator == "imagenet":
            iou = descriptive_delta(paired, "delta_iou")
            summary.update({"mean_delta_iou": iou["mean_delta_iou"], "std_delta_iou": iou["std_delta_iou"]})
        write_summary(output / f"{stem}_summary.csv", [summary]); comparisons[comparator] = paired

    fig, ax = plt.subplots()
    labels=("ImageNet","Full MAE","Distributed","All blocks")
    for seed in (0,1,2): ax.plot(labels,[index[(m,seed)]["human_frozen_dice"] for m in methods],marker="o",label=f"seed {seed}")
    ax.set(ylabel="Human frozen Dice"); ax.legend(); fig.tight_layout(); fig.savefig(output/"all_blocks_seed_reproducibility.png",dpi=200); plt.close(fig)
    fig, ax = plt.subplots()
    for comparator, paired in comparisons.items(): ax.plot([r["seed"] for r in paired],[r["delta_dice"] for r in paired],marker="o",label=f"All-block - {comparator}")
    ax.axhline(0,color="black",linestyle="--"); ax.set(xlabel="Seed",ylabel="Paired Dice delta",xticks=[0,1,2]); ax.legend(); fig.tight_layout(); fig.savefig(output/"all_blocks_paired_delta.png",dpi=200); plt.close(fig)
    fig, ax = plt.subplots()
    for method in methods[1:]:
        selected=[index[(method,s)] for s in (0,1,2)]; ax.scatter([r["mean_feature_drift"] for r in selected],[r["human_frozen_dice"] for r in selected],label=method)
    ax.set(xlabel="Mean feature drift",ylabel="Human frozen Dice"); ax.legend(); fig.tight_layout(); fig.savefig(output/"drift_vs_dice_all_blocks.png",dpi=200); plt.close(fig)
    print("Method                      Mean Dice      SD\n------------------------------------------------")
    for row in summaries: print(f"{row['method']:<27} {row['mean_dice']:.6f}  {row['std_dice']:.6f}")
    for comparator, paired in comparisons.items(): print(f"All-block - {comparator}: " + ", ".join(f"seed{r['seed']}={r['delta_dice']:.6f}" for r in paired) + f", mean={statistics.mean(r['delta_dice'] for r in paired):.6f}")
    print("Mean feature drift: " + ", ".join(f"{r['method']}={r['mean_feature_drift']:.6f}" for r in summaries[1:]))


def main():
    args = parse_args(); source = read_json(args.baseline_config)
    if args.mode == "reproduce-0p01":
        reproduce_0p01(args, source)
        return
    if args.mode == "layer-ablation":
        layer_ablation(args, source)
        return
    if args.mode == "reproduce-all-blocks":
        reproduce_all_blocks(args, source)
        return
    args.runs_dir.mkdir(parents=True, exist_ok=True); args.results_dir.mkdir(parents=True, exist_ok=True)
    rows = [image_reference()]
    baseline, baseline_probe, baseline_drift = resolve_baseline(args, source)
    rows.append(collect("baseline_full", 0.0, baseline, baseline_probe, baseline_drift))
    for name, weight in RUNS[1:]:
        ssl, probe, drift = resolve_anchor(args, source, name, weight)
        rows.append(collect(name, weight, ssl, probe, drift))
    write_summary(args.results_dir / "summary.csv", rows); plots(args, rows)
    print(f"Summary: {(args.results_dir / 'summary.csv').resolve()}")


if __name__ == "__main__": main()
