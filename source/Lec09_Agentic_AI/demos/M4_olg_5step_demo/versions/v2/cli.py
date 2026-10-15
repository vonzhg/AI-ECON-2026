from __future__ import annotations

import argparse
import json
from pathlib import Path

from plotting import write_all_figures, write_stage_map_svg
from solver import OLGParams, result_summary, solve_equilibrium, write_result_files


ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = ROOT / "build"
FIGURES_DIR = ROOT / "figures"
STAGES_DIR = ROOT / "stages"
PROMPTS_DIR = ROOT / "prompts"


STAGE_FILES = {
    "0": STAGES_DIR / "stage0_reverse_engineer_working_code.md",
    "1": STAGES_DIR / "stage1_model_spec.md",
    "2": STAGES_DIR / "stage2_equilibrium.md",
    "3": STAGES_DIR / "stage3_algorithm_choice.md",
    "4": STAGES_DIR / "stage4_pseudocode.md",
    "5": STAGES_DIR / "stage5_implementation_and_validation.md",
}

PROMPT_FILES = {
    "0": PROMPTS_DIR / "stage0_reverse_engineering_prompt.md",
    "1": PROMPTS_DIR / "stage1_prompt.md",
    "2": PROMPTS_DIR / "stage2_prompt.md",
    "3": PROMPTS_DIR / "stage3_prompt.md",
    "4": PROMPTS_DIR / "stage4_prompt.md",
    "5": PROMPTS_DIR / "stage5_prompt.md",
}


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def inspect(args: argparse.Namespace) -> None:
    print("========== OLG Five-Step Demo ==========")
    print(f"Root: {ROOT}")
    print("Stages:")
    for key, path in STAGE_FILES.items():
        print(f"  Stage {key}: {display_path(path)}")
    print("")
    print("Core commands:")
    print("  /usr/bin/python3 run.py stage-map")
    print("  /usr/bin/python3 run.py stage --stage 0")
    print("  /usr/bin/python3 run.py prompt --stage 0")
    print("  /usr/bin/python3 run.py solve")
    print("  /usr/bin/python3 run.py all")


def show_stage(args: argparse.Namespace) -> None:
    path = STAGE_FILES[str(args.stage)]
    print(path.read_text(encoding="utf-8"))


def show_prompt(args: argparse.Namespace) -> None:
    path = PROMPT_FILES[str(args.stage)]
    if not path.exists():
        raise FileNotFoundError(f"No prompt found for stage {args.stage}: {path}")
    print(path.read_text(encoding="utf-8"))


def stage_map(args: argparse.Namespace) -> None:
    write_stage_map_svg(args.figures_dir / "stage_map.svg")
    print("========== Stage Map ==========")
    print(f"Wrote: {display_path(args.figures_dir / 'stage_map.svg')}")


def solve(args: argparse.Namespace) -> None:
    params = OLGParams(
        asset_grid_size=args.asset_grid_size,
        asset_max=args.asset_max,
        ge_tolerance=args.ge_tolerance,
        ge_max_iter=args.ge_max_iter,
    )
    result = solve_equilibrium(params)
    write_result_files(result, args.build_dir)
    if args.figures:
        write_all_figures(result, args.figures_dir)

    summary = result_summary(result)
    print("========== OLG Equilibrium ==========")
    print(f"Interest rate r:        {summary['interest_rate']:.6f}")
    print(f"Wage w:                 {summary['wage']:.6f}")
    print(f"Capital supply:         {summary['capital_supply']:.6f}")
    print(f"Capital demand:         {summary['capital_demand']:.6f}")
    print(f"Excess assets:          {summary['excess_assets']:.3e}")
    print(f"Borrowing-constraint mass: {summary['borrowing_constraint_mass']:.2%}")
    print(f"Euler p95 log10 error:  {summary['euler_p95_log10']:.2f}")
    print(f"Summary:                {display_path(args.build_dir / 'summary.json')}")
    if args.figures:
        print(f"Figures:                {display_path(args.figures_dir)}")


def all_demo(args: argparse.Namespace) -> None:
    stage_map(argparse.Namespace(figures_dir=args.figures_dir))
    solve(
        argparse.Namespace(
            asset_grid_size=args.asset_grid_size,
            asset_max=args.asset_max,
            ge_tolerance=args.ge_tolerance,
            ge_max_iter=args.ge_max_iter,
            build_dir=args.build_dir,
            figures_dir=args.figures_dir,
            figures=True,
        )
    )
    print("")
    print("========== Five-Step Artifacts ==========")
    print(f"Stage docs: {display_path(STAGES_DIR)}")
    print(f"Claude prompts: {display_path(PROMPTS_DIR)}")
    print(f"Build outputs: {display_path(args.build_dir)}")
    print(f"Figures: {display_path(args.figures_dir)}")


def check(args: argparse.Namespace) -> None:
    summary_path = args.build_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("Run `/usr/bin/python3 run.py solve` first.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checks = [
        ("GE gap < 5e-3", abs(summary["excess_assets"]) < 5.0e-3),
        ("capital positive", summary["capital_supply"] > 0.0),
        ("wage positive", summary["wage"] > 0.0),
        ("distribution has borrowing-constraint mass in [0,1]", 0.0 <= summary["borrowing_constraint_mass"] <= 1.0),
        ("Euler p95 log10 error < 1", summary["euler_p95_log10"] < 1.0),
    ]
    print("========== Validation Checks ==========")
    for label, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'} | {label}")
    if not all(ok for _, ok in checks):
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Five-step Claude Code demo for a small OLG model."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.set_defaults(func=inspect)

    stage_parser = subparsers.add_parser("stage")
    stage_parser.add_argument("--stage", choices=list(STAGE_FILES.keys()), required=True)
    stage_parser.set_defaults(func=show_stage)

    prompt_parser = subparsers.add_parser("prompt")
    prompt_parser.add_argument("--stage", choices=list(PROMPT_FILES.keys()), required=True)
    prompt_parser.set_defaults(func=show_prompt)

    map_parser = subparsers.add_parser("stage-map")
    map_parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    map_parser.set_defaults(func=stage_map)

    solve_parser = subparsers.add_parser("solve")
    solve_parser.add_argument("--asset-grid-size", type=int, default=90)
    solve_parser.add_argument("--asset-max", type=float, default=8.0)
    solve_parser.add_argument("--ge-tolerance", type=float, default=5.0e-4)
    solve_parser.add_argument("--ge-max-iter", type=int, default=40)
    solve_parser.add_argument("--build-dir", type=Path, default=BUILD_DIR)
    solve_parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    solve_parser.add_argument("--no-figures", action="store_false", dest="figures")
    solve_parser.set_defaults(func=solve, figures=True)

    all_parser = subparsers.add_parser("all")
    all_parser.add_argument("--asset-grid-size", type=int, default=90)
    all_parser.add_argument("--asset-max", type=float, default=8.0)
    all_parser.add_argument("--ge-tolerance", type=float, default=5.0e-4)
    all_parser.add_argument("--ge-max-iter", type=int, default=40)
    all_parser.add_argument("--build-dir", type=Path, default=BUILD_DIR)
    all_parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    all_parser.set_defaults(func=all_demo)

    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--build-dir", type=Path, default=BUILD_DIR)
    check_parser.set_defaults(func=check)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
