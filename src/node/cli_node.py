import argparse
from typing import Optional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m project.src.node.cli_node",
        description="Node-based pipeline (Phase 0/1) for SEMCOG OD diagnostics",
    )
    sub = parser.add_subparsers(dest="command")
    sub.required = True

    # Phase 0: panel + geo
    sp = sub.add_parser("build-panel", help="Build H/W/In panel from LODES8 (2020–2021)")
    sp.add_argument(
        "--lodes-root",
        type=str,
        default="dataset/tract_data/data/LODES8/mi",
        help="Root directory of LODES8 MI data (contains by_year/{rac,wac,od})",
    )
    sp.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2020, 2021],
        help="Years to include in the panel (default: 2020 2021)",
    )
    sp.set_defaults(func=_cmd_build_panel)

    sp = sub.add_parser("attach-geo", help="Attach lat/lon/county to node panel tables")
    sp.add_argument(
        "--tracts",
        type=str,
        default="project/data/geo/tracts.geojson",
        help="GeoJSON with tract geometries/centroids",
    )
    sp.set_defaults(func=_cmd_attach_geo)

    # Phase 1 commands (implemented later)
    sp = sub.add_parser("gridify", help="Rasterize node signals to county grids")
    sp.add_argument(
        "--grid-res-km",
        type=float,
        default=1.5,
        help="Grid resolution in kilometers",
    )
    sp.set_defaults(func=_cmd_gridify)

    sp = sub.add_parser("nn-train", help="Train CNN for self-prediction of rho^H")
    sp.add_argument("--epochs", type=int, default=200, help="Training epochs")
    sp.set_defaults(func=_cmd_nn_train)

    sp = sub.add_parser("nn-saliency", help="Compute saliency-based locality curves")
    sp.add_argument(
        "--bands-km",
        type=float,
        nargs="+",
        default=[0, 2, 4, 6, 8, 10],
        help="Radial bands (km) for aggregating saliency",
    )
    sp.set_defaults(func=_cmd_nn_saliency)

    sp = sub.add_parser("report-phase1", help="Summarize Phase 1 findings")
    sp.set_defaults(func=_cmd_report_phase1)

    return parser


def _cmd_build_panel(args: argparse.Namespace) -> None:
    from . import build_panel as build_panel_mod

    build_panel_mod.build_panel(args.lodes_root, years=args.years)


def _cmd_attach_geo(args: argparse.Namespace) -> None:
    from . import io as node_io

    node_io.attach_geo(
        in_parquet="project/data/node/panel_node.parquet",
        tracts_geojson=args.tracts,
        out_parquet="project/data/node/panel_node_geo.parquet",
    )
    node_io.attach_geo(
        in_parquet="project/data/node/node_signals.parquet",
        tracts_geojson=args.tracts,
        out_parquet="project/data/node/node_signals_geo.parquet",
    )


def _cmd_gridify(args: argparse.Namespace) -> None:
    from . import grid as grid_mod

    grid_mod.to_grids(
        node_signals_geo_path="project/data/node/node_signals_geo.parquet",
        tracts_geojson="project/data/geo/tracts.geojson",
        grid_res_km=args.grid_res_km,
    )


def _cmd_nn_train(args: argparse.Namespace) -> None:
    from . import nn as nn_mod

    nn_mod.train_cnn(
        data_root="project/data/node/grids",
        inputs=["rhoH", "rhoW", "In"],
        target="drhoH",
        split="leave-county-out",
        epochs=args.epochs,
    )


def _cmd_nn_saliency(args: argparse.Namespace) -> None:
    from . import saliency as saliency_mod

    saliency_mod.compute_saliency_and_locality(
        data_root="project/data/node/grids",
        model_path="project/results/node/model_cnn.pt",
        bands_km=args.bands_km,
    )


def _cmd_report_phase1(args: argparse.Namespace) -> None:
    from . import report as report_mod

    report_mod.write_phase1_summary()


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

