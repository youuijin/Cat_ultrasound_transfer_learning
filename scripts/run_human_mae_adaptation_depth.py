"""Run Human MAE adaptation-depth screening and last4 reproducibility probes."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib.pyplot as plt


CORE_DEPTHS = ("last1", "last2", "last4", "last6", "full")
DEPTHS = ("last1", "last2", "last4", "last6", "last8", "last10", "full")
DEPTH_TO_BLOCKS = {"last1": 1, "last2": 2, "last4": 4, "last6": 6,
                   "last8": 8, "last10": 10, "full": None}
BASELINE_CONFIG = Path("checkpoints/human_mae_vit_b16_trajectory/config.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("screening", "reproduce-last4", "reproduce-depth",
                                            "extended-diagnostic"),
                        default="screening")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--depths", nargs="+", choices=DEPTHS)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--runs-dir", type=Path,
                        default=Path("runs/human_mae_adaptation_depth"))
    parser.add_argument("--results-dir", type=Path,
                        default=Path("results/human_mae_adaptation_depth"))
    parser.add_argument("--repro-results-dir", type=Path,
                        default=Path("results/human_mae_adaptation_depth_reproducibility"))
    parser.add_argument("--extended-results-dir", type=Path,
                        default=Path("results/human_mae_adaptation_depth_extended"))
    parser.add_argument("--baseline-config", type=Path, default=BASELINE_CONFIG)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.epochs < 2 or args.batch_size < 1:
        parser.error("--epochs must be >=2 and --batch-size must be positive")
    if args.seeds is None:
        args.seeds = [0] if args.mode == "screening" else [0, 1, 2]
    if args.depths is None:
        args.depths = (list(CORE_DEPTHS) if args.mode == "screening" else
                       ["last4"] if args.mode == "reproduce-last4" else
                       ["last2", "last4", "last6"] if args.mode == "reproduce-depth" else
                       ["last8", "last10", "full"])
    return args


def run(command: list[str], label: str) -> None:
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}", flush=True)
    print(subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def read_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def config_matches(path: Path, expected: dict) -> bool:
    if not path.is_file(): return False
    config = read_config(path)
    return all(config.get(key) == value for key, value in expected.items())


def mae_expected(source: dict, depth: str, seed: int, args) -> dict:
    return {"encoder": "vit_b16", "mask_ratio": source["mask_ratio"],
            "norm_pixel_loss": source["norm_pixel_loss"], "lr": source["lr"],
            "encoder_lr_scale": 1.0, "weight_decay": source["weight_decay"],
            "human1_root": source["human1_root"], "human2_root": source["human2_root"],
            "human3_root": source["human3_root"], "val_fraction": source["val_fraction"],
            "max_images": source.get("max_images"),
            "dataset_counts": source.get("dataset_counts"),
            "batch_size": args.batch_size, "epochs": args.epochs,
            "warmup_epochs": min(source["warmup_epochs"], args.epochs - 1), "seed": seed,
            "decoder_dim": source["decoder_dim"], "decoder_depth": source["decoder_depth"],
            "decoder_heads": source["decoder_heads"],
            "encoder_trainable_last_blocks": DEPTH_TO_BLOCKS[depth]}


def probe_expected(seed: int) -> dict:
    expected = {"dataset": "human2", "encoder": "vit_b16", "encoder_init": "human_mae",
            "transfer": "frozen", "val_fraction": 0.2, "split_seed": 42, "seed": seed,
            "batch_size": 8, "epochs": 50, "lr": 1e-4, "weight_decay": 1e-4}
    reference = Path("runs/human_ssl_trajectory_probe/epoch_000/config.json")
    if reference.is_file():
        config = read_config(reference)
        expected.update({"data_root": config.get("data_root"),
                         "train_subjects": config.get("train_subjects"),
                         "val_subjects": config.get("val_subjects")})
    return expected


def imagenet_expected(seed: int) -> dict:
    values = probe_expected(seed); values["encoder_init"] = "imagenet"
    return values


def external_mae_candidate(depth: str, seed: int) -> Path | None:
    if seed != 0: return None
    candidates = {"last4": Path("runs/human_mae_recipe_ablation/partial_last4"),
                  "full": Path("runs/human_mae_recipe_ablation/baseline")}
    return candidates.get(depth)


def external_imagenet_candidate(seed: int) -> Path | None:
    return Path("runs/human_ssl_trajectory_probe/epoch_000") if seed == 0 else None


def write_reuse(run_root: Path, kind: str, source: Path) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / f"{kind}_reuse.json").write_text(
        json.dumps({"reused": True, "source": str(source.resolve())}, indent=2),
        encoding="utf-8")
    print(f"[reuse] verified identical {kind}: {source.resolve()}")


def resolve_or_train_imagenet(args, seed: int) -> Path:
    target = args.runs_dir / "imagenet" / f"seed{seed}"
    if not args.force and config_matches(target / "config.json", imagenet_expected(seed)):
        print(f"[reuse] verified target ImageNet run: {target.resolve()}"); return target
    external = external_imagenet_candidate(seed)
    if (not args.force and external is not None and
            config_matches(external / "config.json", imagenet_expected(seed))):
        write_reuse(target, "imagenet", external); return external
    command = [args.python, "-m", "src.train_human_segmentation", "--dataset", "human2",
               "--encoder", "vit_b16", "--encoder-init", "imagenet", "--transfer", "frozen",
               "--val-fraction", "0.2", "--split-seed", "42", "--seed", str(seed),
               "--batch-size", "8", "--epochs", "50", "--lr", "1e-4",
               "--weight-decay", "1e-4", "--num-workers", str(args.num_workers),
               "--run-dir", str(target), "--amp" if args.amp else "--no-amp"]
    run(command, f"ImageNet frozen Human2 control | seed={seed}")
    return target


def mae_command(args, source: dict, depth: str, seed: int, output: Path) -> list[str]:
    command = [args.python, "train_human_ssl.py", "--method", "mae", "--encoder", "vit_b16",
               "--human1-root", str(source["human1_root"]),
               "--human2-root", str(source["human2_root"]),
               "--human3-root", str(source["human3_root"]),
               "--val-fraction", str(source["val_fraction"]),
               "--mask-ratio", str(source["mask_ratio"]),
               "--no-norm-pixel-loss", "--decoder-dim", str(source["decoder_dim"]),
               "--decoder-depth", str(source["decoder_depth"]),
               "--decoder-heads", str(source["decoder_heads"]),
               "--batch-size", str(args.batch_size), "--epochs", str(args.epochs),
               "--lr", str(source["lr"]), "--encoder-lr-scale", "1.0",
               "--weight-decay", str(source["weight_decay"]),
               "--warmup-epochs", str(min(source["warmup_epochs"], args.epochs - 1)),
               "--num-workers", str(args.num_workers), "--seed", str(seed),
               "--run-name", f"{depth}_seed{seed}", "--output-dir", str(output),
               "--amp" if args.amp else "--no-amp"]
    blocks = DEPTH_TO_BLOCKS[depth]
    if blocks is not None: command.extend(("--encoder-trainable-last-blocks", str(blocks)))
    return command


def probe_command(args, source: dict, seed: int, checkpoint: Path,
                  output: Path, ssl_config: Path) -> list[str]:
    return [args.python, "-m", "src.train_human_segmentation", "--dataset", "human2",
            "--data-root", str(source["human2_root"]), "--encoder", "vit_b16",
            "--encoder-init", "human_mae", "--encoder-checkpoint", str(checkpoint),
            "--transfer", "frozen", "--val-fraction", "0.2", "--split-seed", "42",
            "--seed", str(seed), "--batch-size", "8", "--epochs", "50", "--lr", "1e-4",
            "--weight-decay", "1e-4", "--num-workers", str(args.num_workers),
            "--run-dir", str(output), "--ssl-reference-config", str(ssl_config),
            "--amp" if args.amp else "--no-amp"]


def resolve_or_train_depth(args, source: dict, depth: str, seed: int) -> tuple[Path, Path]:
    root = args.runs_dir / depth / f"seed{seed}"
    mae_target, probe_target = root / "mae", root / "human_frozen_probe"
    expected_mae, expected_probe = mae_expected(source, depth, seed, args), probe_expected(seed)
    mae_source = mae_target
    if not (not args.force and config_matches(mae_target / "config.json", expected_mae)
            and (mae_target / "last_encoder.pt").is_file()):
        external = external_mae_candidate(depth, seed)
        if (not args.force and external is not None and
                config_matches(external / "config.json", expected_mae)
                and (external / "last_encoder.pt").is_file()):
            mae_source = external; write_reuse(root, "mae", external)
        else:
            run(mae_command(args, source, depth, seed, mae_target),
                f"Human MAE adaptation depth={depth} | seed={seed}")
            mae_source = mae_target
    else:
        print(f"[reuse] verified target MAE run: {mae_target.resolve()}")
    external_probe = mae_source / "human_frozen_probe"
    if not (not args.force and config_matches(probe_target / "config.json", expected_probe)
            and (probe_target / "metrics.csv").is_file()):
        if (mae_source != mae_target and not args.force and
                config_matches(external_probe / "config.json", expected_probe)
                and (external_probe / "metrics.csv").is_file()):
            write_reuse(root, "probe", external_probe); probe_source = external_probe
        else:
            run(probe_command(args, source, seed, mae_source / "last_encoder.pt", probe_target,
                              mae_source / "config.json"),
                f"Frozen Human2 probe depth={depth} | seed={seed}")
            probe_source = probe_target
    else:
        print(f"[reuse] verified target probe: {probe_target.resolve()}"); probe_source = probe_target
    return mae_source, probe_source


def best_validation(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["phase"] == "validation"]
    return max(rows, key=lambda row: float(row["mean_dice"]))


def depth_metadata(depth: str, config: dict) -> tuple[str, int, float]:
    blocks = config.get("trainable_blocks")
    if blocks is None:
        count = 12 if depth == "full" else int(depth.removeprefix("last"))
        blocks = list(range(12 - count, 12))
    params = config.get("trainable_encoder_params")
    percent = config.get("trainable_encoder_percent")
    if params is None or percent is None:
        # Exact torchvision ViT-B/16 component counts from the repository model.
        from src.encoders import get_encoder
        encoder = get_encoder("vit_b16_imagenet", pretrained=True)
        if depth != "full":
            encoder.freeze()
            for block in list(encoder.model.encoder.layers)[-int(depth.removeprefix("last")):]:
                block.requires_grad_(True)
            encoder.model.encoder.ln.requires_grad_(True)
        params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        total = sum(p.numel() for p in encoder.parameters())
        percent = 100 * params / total
    return json.dumps(blocks), int(params), float(percent)


def collect_depth(depth: str, seed: int, mae_dir: Path, probe_dir: Path) -> dict:
    config = read_config(mae_dir / "config.json")
    with (mae_dir / "metrics.csv").open(encoding="utf-8") as handle:
        ssl = list(csv.DictReader(handle))[-1]
    best = best_validation(probe_dir / "metrics.csv")
    blocks, params, percent = depth_metadata(depth, config)
    return {"adaptation_depth": depth, "seed": seed, "trainable_blocks": blocks,
            "trainable_encoder_params": params, "trainable_encoder_percent": percent,
            "ssl_train_loss": ssl["train_mae_loss"],
            "ssl_val_loss": ssl["validation_mae_loss"],
            "human_frozen_dice": best["mean_dice"], "human_frozen_iou": best["kidney_iou"],
            "human_frozen_loss": best["loss"],
            "mae_checkpoint": str((mae_dir / "last_encoder.pt").resolve())}


def collect_imagenet(seed: int, run_dir: Path) -> dict:
    best = best_validation(run_dir / "metrics.csv")
    return {"adaptation_depth": "imagenet", "seed": seed, "trainable_blocks": "[]",
            "trainable_encoder_params": 0, "trainable_encoder_percent": 0.0,
            "ssl_train_loss": "", "ssl_val_loss": "",
            "human_frozen_dice": best["mean_dice"], "human_frozen_iou": best["kidney_iou"],
            "human_frozen_loss": best["loss"], "mae_checkpoint": "imagenet_pretrained"}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows: return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_outputs(args, rows: list[dict]) -> None:
    order = {name: index for index, name in enumerate(("imagenet", *DEPTHS))}
    rows.sort(key=lambda row: (order[row["adaptation_depth"]], int(row["seed"])))
    write_csv(args.results_dir / "all_runs.csv", rows)
    pairs = []
    for seed in sorted({int(row["seed"]) for row in rows}):
        image = next((row for row in rows if row["adaptation_depth"] == "imagenet"
                      and int(row["seed"]) == seed), None)
        last4 = next((row for row in rows if row["adaptation_depth"] == "last4"
                      and int(row["seed"]) == seed), None)
        if image and last4:
            pairs.append({"seed": seed, "imagenet_dice": image["human_frozen_dice"],
                          "last4_dice": last4["human_frozen_dice"],
                          "delta_dice": float(last4["human_frozen_dice"]) - float(image["human_frozen_dice"]),
                          "imagenet_iou": image["human_frozen_iou"],
                          "last4_iou": last4["human_frozen_iou"],
                          "delta_iou": float(last4["human_frozen_iou"]) - float(image["human_frozen_iou"])})
    write_csv(args.results_dir / "last4_reproducibility.csv", pairs)
    reproducibility = []
    for method in ("imagenet", "last4"):
        selected = [row for row in rows if row["adaptation_depth"] == method
                    and any(int(row["seed"]) == int(pair["seed"]) for pair in pairs)]
        if selected:
            dice = [float(row["human_frozen_dice"]) for row in selected]
            iou = [float(row["human_frozen_iou"]) for row in selected]
            reproducibility.append({"method": method, "mean_dice": statistics.mean(dice),
                                    "std_dice": statistics.stdev(dice) if len(dice) > 1 else 0.0,
                                    "mean_iou": statistics.mean(iou),
                                    "std_iou": statistics.stdev(iou) if len(iou) > 1 else 0.0})
    write_csv(args.results_dir / "reproducibility_summary.csv", reproducibility)
    curve = [row for row in rows if int(row["seed"]) == 0]
    curve_rows = [{key: row[key] for key in (
        "adaptation_depth", "trainable_encoder_percent", "human_frozen_dice",
        "human_frozen_iou", "human_frozen_loss", "ssl_val_loss")} for row in curve]
    write_csv(args.results_dir / "depth_curve_seed0.csv", curve_rows)
    if len(curve) >= 2:
        fig, ax = plt.subplots(figsize=(9, 5))
        labels = [row["adaptation_depth"] for row in curve]
        values = [float(row["human_frozen_dice"]) for row in curve]
        ax.plot(labels, values, marker="o")
        reference = next((value for label, value in zip(labels, values) if label == "imagenet"), None)
        if reference is not None: ax.axhline(reference, color="black", linestyle="--")
        ax.set(xlabel="Adaptation depth", ylabel="Human frozen validation Dice")
        fig.tight_layout(); fig.savefig(args.results_dir / "adaptation_depth_vs_dice.png", dpi=200)
        plt.close(fig)
    if pairs:
        fig, ax = plt.subplots(figsize=(7, 5))
        for pair in pairs:
            ax.plot((0, 1), (float(pair["imagenet_dice"]), float(pair["last4_dice"])),
                    marker="o", label=f"seed {pair['seed']}")
        ax.set_xticks((0, 1), ("ImageNet", "Last4")); ax.set_ylabel("Human frozen validation Dice")
        ax.legend(); fig.tight_layout()
        fig.savefig(args.results_dir / "last4_seed_reproducibility.png", dpi=200); plt.close(fig)


def descriptive(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def existing_imagenet(args, seed: int) -> bool:
    target = args.runs_dir / "imagenet" / f"seed{seed}"
    external = external_imagenet_candidate(seed)
    return (config_matches(target / "config.json", imagenet_expected(seed)) or
            (external is not None and
             config_matches(external / "config.json", imagenet_expected(seed))))


def existing_depth(args, source: dict, depth: str, seed: int) -> bool:
    root = args.runs_dir / depth / f"seed{seed}"
    mae_target, probe_target = root / "mae", root / "human_frozen_probe"
    if (config_matches(mae_target / "config.json", mae_expected(source, depth, seed, args)) and
            config_matches(probe_target / "config.json", probe_expected(seed)) and
            (mae_target / "last_encoder.pt").is_file() and
            (probe_target / "metrics.csv").is_file()):
        return True
    external = external_mae_candidate(depth, seed)
    return bool(external is not None and
                config_matches(external / "config.json", mae_expected(source, depth, seed, args)) and
                config_matches(external / "human_frozen_probe" / "config.json",
                               probe_expected(seed)) and
                (external / "last_encoder.pt").is_file() and
                (external / "human_frozen_probe" / "metrics.csv").is_file())


def write_depth_reproducibility(args, rows: list[dict]) -> None:
    output = args.repro_results_dir
    output.mkdir(parents=True, exist_ok=True)
    order = {name: index for index, name in enumerate(("imagenet", "last2", "last4", "last6"))}
    rows.sort(key=lambda row: (order[row["adaptation_depth"]], int(row["seed"])))
    columns = ("adaptation_depth", "seed", "trainable_encoder_percent", "ssl_train_loss",
               "ssl_val_loss", "human_frozen_dice", "human_frozen_iou",
               "human_frozen_loss", "mae_checkpoint", "reused_existing_run")
    seed_rows = [{key: row.get(key, "") for key in columns} for row in rows]
    write_csv(output / "all_depth_seed_results.csv", seed_rows)
    summary = []
    for depth in ("imagenet", "last2", "last4", "last6"):
        selected = [row for row in rows if row["adaptation_depth"] == depth]
        if not selected: continue
        dice = [float(row["human_frozen_dice"]) for row in selected]
        iou = [float(row["human_frozen_iou"]) for row in selected]
        loss = [float(row["human_frozen_loss"]) for row in selected]
        ssl_loss = [float(row["ssl_val_loss"]) for row in selected if row["ssl_val_loss"] != ""]
        mean_dice, std_dice = descriptive(dice); mean_iou, std_iou = descriptive(iou)
        mean_loss, std_loss = descriptive(loss)
        mean_ssl, std_ssl = descriptive(ssl_loss) if ssl_loss else (float("nan"), float("nan"))
        summary.append({"adaptation_depth": depth, "n_seeds": len(selected),
                        "mean_dice": mean_dice, "std_dice": std_dice,
                        "mean_iou": mean_iou, "std_iou": std_iou,
                        "mean_loss": mean_loss, "std_loss": std_loss,
                        "mean_ssl_val_loss": mean_ssl, "std_ssl_val_loss": std_ssl})
    write_csv(output / "depth_reproducibility_summary.csv", summary)
    imagenet = {int(row["seed"]): row for row in rows if row["adaptation_depth"] == "imagenet"}
    paired = []
    for depth in ("last2", "last4", "last6"):
        for row in [value for value in rows if value["adaptation_depth"] == depth]:
            seed = int(row["seed"])
            if seed not in imagenet: continue
            base = imagenet[seed]
            paired.append({"adaptation_depth": depth, "seed": seed,
                           "imagenet_dice": base["human_frozen_dice"],
                           "adapted_dice": row["human_frozen_dice"],
                           "delta_dice": float(row["human_frozen_dice"]) - float(base["human_frozen_dice"]),
                           "imagenet_iou": base["human_frozen_iou"],
                           "adapted_iou": row["human_frozen_iou"],
                           "delta_iou": float(row["human_frozen_iou"]) - float(base["human_frozen_iou"])})
    write_csv(output / "paired_vs_imagenet.csv", paired)
    paired_summary = []
    for depth in ("last2", "last4", "last6"):
        selected = [row for row in paired if row["adaptation_depth"] == depth]
        if not selected: continue
        dice = [float(row["delta_dice"]) for row in selected]
        iou = [float(row["delta_iou"]) for row in selected]
        mean_dice, std_dice = descriptive(dice); mean_iou, std_iou = descriptive(iou)
        paired_summary.append({"adaptation_depth": depth,
                               "mean_delta_dice": mean_dice, "std_delta_dice": std_dice,
                               "positive_delta_count": sum(value > 0 for value in dice),
                               "total_seeds": len(dice), "mean_delta_iou": mean_iou,
                               "std_delta_iou": std_iou})
    write_csv(output / "paired_vs_imagenet_summary.csv", paired_summary)
    by_key = {(row["adaptation_depth"], int(row["seed"])): row for row in rows}
    local = []
    for seed in sorted(imagenet):
        if all((depth, seed) in by_key for depth in ("last2", "last4", "last6")):
            last2, last4, last6 = (by_key[(depth, seed)] for depth in
                                    ("last2", "last4", "last6"))
            local.append({"seed": seed, "last2_dice": last2["human_frozen_dice"],
                          "last4_dice": last4["human_frozen_dice"],
                          "last6_dice": last6["human_frozen_dice"],
                          "last4_minus_last2": float(last4["human_frozen_dice"]) - float(last2["human_frozen_dice"]),
                          "last4_minus_last6": float(last4["human_frozen_dice"]) - float(last6["human_frozen_dice"])})
    write_csv(output / "last2_last4_last6_comparison.csv", local)
    local_summary = []
    for label, column in (("last4 - last2", "last4_minus_last2"),
                          ("last4 - last6", "last4_minus_last6")):
        values = [float(row[column]) for row in local]
        if values:
            mean, std = descriptive(values)
            local_summary.append({"comparison": label, "mean_difference": mean,
                                  "std_difference": std,
                                  "positive_count": sum(value > 0 for value in values),
                                  "total_seeds": len(values)})
    write_csv(output / "local_peak_summary.csv", local_summary)
    if rows:
        fig, ax = plt.subplots(figsize=(9, 5))
        for seed in sorted(imagenet):
            selected = [by_key[(depth, seed)] for depth in
                        ("imagenet", "last2", "last4", "last6")
                        if (depth, seed) in by_key]
            ax.plot([row["adaptation_depth"] for row in selected],
                    [float(row["human_frozen_dice"]) for row in selected],
                    marker="o", label=f"seed {seed}")
        ax.set(xlabel="Adaptation depth", ylabel="Human frozen validation Dice")
        ax.legend(); fig.tight_layout()
        fig.savefig(output / "depth_reproducibility.png", dpi=200); plt.close(fig)
    if paired:
        fig, ax = plt.subplots(figsize=(8, 5))
        x_positions = {depth: index for index, depth in enumerate(("last2", "last4", "last6"))}
        for row in paired:
            ax.scatter(x_positions[row["adaptation_depth"]], float(row["delta_dice"]),
                       label=f"seed {row['seed']}" if row["adaptation_depth"] == "last2" else None)
        ax.axhline(0, color="black", linestyle="--")
        ax.set_xticks(range(3), ("Last2", "Last4", "Last6"))
        ax.set_ylabel("Dice difference vs matched ImageNet")
        ax.legend(); fig.tight_layout()
        fig.savefig(output / "paired_delta_vs_imagenet.png", dpi=200); plt.close(fig)
    print("\nDepth       Mean Dice     SD           Mean delta vs ImageNet")
    print("-" * 66)
    delta_by_depth = {row["adaptation_depth"]: row for row in paired_summary}
    for row in summary:
        delta = (0.0 if row["adaptation_depth"] == "imagenet" else
                 delta_by_depth.get(row["adaptation_depth"], {}).get("mean_delta_dice", float("nan")))
        print(f"{row['adaptation_depth']:<11} {row['mean_dice']:<13.6f} "
              f"{row['std_dice']:<12.6f} {delta:.6f}")
    for row in local_summary:
        print(f"{row['comparison']} mean: {row['mean_difference']:.6f}")


def write_extended_diagnostic(args, rows: list[dict]) -> None:
    output = args.extended_results_dir
    output.mkdir(parents=True, exist_ok=True)
    previous_path = args.results_dir / "all_runs.csv"
    previous = []
    if previous_path.is_file():
        with previous_path.open(encoding="utf-8") as handle:
            previous = list(csv.DictReader(handle))
    keys = {(row["adaptation_depth"], int(row["seed"])) for row in rows}
    rows.extend(row for row in previous
                if (row["adaptation_depth"], int(row["seed"])) not in keys)
    by_key = {(row["adaptation_depth"], int(row["seed"])): row for row in rows}
    curve_order = ("imagenet", "last1", "last2", "last4", "last6", "last8", "last10", "full")
    curve = [by_key[(depth, 0)] for depth in curve_order if (depth, 0) in by_key]
    curve_columns = ("adaptation_depth", "trainable_encoder_percent", "trainable_blocks",
                     "trainable_encoder_params", "human_frozen_dice", "human_frozen_iou",
                     "human_frozen_loss", "ssl_train_loss", "ssl_val_loss", "mae_checkpoint")
    write_csv(output / "extended_depth_curve_seed0.csv",
              [{key: row.get(key, "") for key in curve_columns} for row in curve])
    full_rows = []
    for seed in (0, 1, 2):
        image, full = by_key.get(("imagenet", seed)), by_key.get(("full", seed))
        if image and full:
            full_rows.append({"seed": seed, "imagenet_dice": image["human_frozen_dice"],
                              "full_dice": full["human_frozen_dice"],
                              "delta_dice": float(full["human_frozen_dice"]) - float(image["human_frozen_dice"]),
                              "imagenet_iou": image["human_frozen_iou"],
                              "full_iou": full["human_frozen_iou"],
                              "delta_iou": float(full["human_frozen_iou"]) - float(image["human_frozen_iou"]),
                              "ssl_val_loss": full["ssl_val_loss"],
                              "full_checkpoint": full["mae_checkpoint"]})
    write_csv(output / "full_reproducibility.csv", full_rows)
    method_summary = []
    for method in ("imagenet", "full"):
        selected = [by_key[(method, seed)] for seed in (0, 1, 2) if (method, seed) in by_key]
        if selected:
            dice = [float(row["human_frozen_dice"]) for row in selected]
            iou = [float(row["human_frozen_iou"]) for row in selected]
            loss = [float(row["human_frozen_loss"]) for row in selected]
            md, sd = descriptive(dice); mi, si = descriptive(iou); ml, sl = descriptive(loss)
            method_summary.append({"method": method, "n_seeds": len(selected),
                                   "mean_dice": md, "std_dice": sd, "mean_iou": mi,
                                   "std_iou": si, "mean_loss": ml, "std_loss": sl})
    write_csv(output / "full_reproducibility_summary.csv", method_summary)
    dice_delta = [float(row["delta_dice"]) for row in full_rows]
    iou_delta = [float(row["delta_iou"]) for row in full_rows]
    if dice_delta:
        md, sd = descriptive(dice_delta); mi, si = descriptive(iou_delta)
        write_csv(output / "full_vs_imagenet_summary.csv", [{
            "mean_delta_dice": md, "std_delta_dice": sd,
            "positive_delta_count": sum(value > 0 for value in dice_delta),
            "negative_delta_count": sum(value < 0 for value in dice_delta),
            "total_seeds": len(dice_delta), "mean_delta_iou": mi, "std_delta_iou": si}])
    high = [by_key[(depth, 0)] for depth in ("last6", "last8", "last10", "full")
            if (depth, 0) in by_key]
    high_columns = ("adaptation_depth", "trainable_encoder_percent", "human_frozen_dice",
                    "human_frozen_iou", "human_frozen_loss", "ssl_val_loss")
    write_csv(output / "high_depth_comparison_seed0.csv",
              [{key: row.get(key, "") for key in high_columns} for row in high])
    comparisons = (("last8 - last6", "last8", "last6"),
                   ("last10 - last8", "last10", "last8"),
                   ("full - last10", "full", "last10"),
                   ("full - last6", "full", "last6"))
    deltas = []
    for label, left, right in comparisons:
        if (left, 0) in by_key and (right, 0) in by_key:
            a, b = by_key[(left, 0)], by_key[(right, 0)]
            deltas.append({"comparison": label,
                           "dice_difference": float(a["human_frozen_dice"]) - float(b["human_frozen_dice"]),
                           "iou_difference": float(a["human_frozen_iou"]) - float(b["human_frozen_iou"]),
                           "ssl_val_loss_difference": float(a["ssl_val_loss"]) - float(b["ssl_val_loss"])})
    write_csv(output / "high_depth_deltas_seed0.csv", deltas)
    if curve:
        fig, ax = plt.subplots(figsize=(9, 5))
        x = [float(row["trainable_encoder_percent"]) for row in curve]
        y = [float(row["human_frozen_dice"]) for row in curve]
        ax.plot(x, y, marker="o")
        for px, py, row in zip(x, y, curve): ax.annotate(row["adaptation_depth"], (px, py))
        ax.axhline(float(by_key[("imagenet", 0)]["human_frozen_dice"]), color="black", linestyle="--")
        ax.set(xlabel="Trainable encoder (%)", ylabel="Human frozen validation Dice")
        fig.tight_layout(); fig.savefig(output / "extended_adaptation_depth_vs_dice.png", dpi=200)
        plt.close(fig)
        ssl_curve = [row for row in curve if row["ssl_val_loss"] != ""]
        fig, ax = plt.subplots(figsize=(9, 5))
        sx = [float(row["trainable_encoder_percent"]) for row in ssl_curve]
        sy = [float(row["ssl_val_loss"]) for row in ssl_curve]
        ax.plot(sx, sy, marker="o")
        for px, py, row in zip(sx, sy, ssl_curve): ax.annotate(row["adaptation_depth"], (px, py))
        ax.set(xlabel="Trainable encoder (%)", ylabel="SSL validation loss")
        fig.tight_layout(); fig.savefig(output / "extended_depth_vs_ssl_loss.png", dpi=200)
        plt.close(fig)
    if full_rows:
        fig, ax = plt.subplots(figsize=(7, 5))
        for row in full_rows:
            ax.plot((0, 1), (float(row["imagenet_dice"]), float(row["full_dice"])),
                    marker="o", label=f"seed {row['seed']}")
        ax.set_xticks((0, 1), ("ImageNet", "Full")); ax.set_ylabel("Human frozen validation Dice")
        ax.legend(); fig.tight_layout(); fig.savefig(output / "full_seed_reproducibility.png", dpi=200)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    source = read_config(args.baseline_config)
    if source["mask_ratio"] != 0.75 or source["norm_pixel_loss"] is not False:
        raise ValueError("Baseline config is not the expected mask=0.75/raw-pixel MAE recipe")
    args.runs_dir.mkdir(parents=True, exist_ok=True); args.results_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    image_seeds = [0, 1, 2] if args.mode == "extended-diagnostic" else args.seeds
    for seed in image_seeds:
        image_reused = existing_imagenet(args, seed) and not args.force
        image_dir = resolve_or_train_imagenet(args, seed)
        image_row = collect_imagenet(seed, image_dir)
        image_row["reused_existing_run"] = image_reused
        rows.append(image_row)
    depth_plan = ([("full", seed) for seed in (0, 1, 2)] +
                  [("last8", 0), ("last10", 0)] if args.mode == "extended-diagnostic" else
                  [(depth, seed) for seed in args.seeds for depth in args.depths])
    for depth, seed in depth_plan:
        depth_reused = existing_depth(args, source, depth, seed) and not args.force
        mae_dir, probe_dir = resolve_or_train_depth(args, source, depth, seed)
        depth_row = collect_depth(depth, seed, mae_dir, probe_dir)
        depth_row["reused_existing_run"] = depth_reused
        rows.append(depth_row)
    if args.mode == "extended-diagnostic":
        write_extended_diagnostic(args, rows)
        print(f"Results: {args.extended_results_dir.resolve()}")
        return
    if args.mode == "reproduce-depth":
        write_depth_reproducibility(args, rows)
        print(f"Results: {args.repro_results_dir.resolve()}")
        return
    # Merge rows from a previous mode so screening and reproducibility outputs coexist.
    previous = args.results_dir / "all_runs.csv"
    if previous.is_file():
        with previous.open(encoding="utf-8") as handle:
            old_rows = list(csv.DictReader(handle))
        keys = {(row["adaptation_depth"], int(row["seed"])) for row in rows}
        rows.extend(row for row in old_rows
                    if (row["adaptation_depth"], int(row["seed"])) not in keys)
    write_outputs(args, rows)
    print(f"Results: {args.results_dir.resolve()}")


if __name__ == "__main__":
    main()
