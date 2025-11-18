import os
from typing import Any, Dict


def _load_json(path: str) -> Dict[str, Any]:
    import json

    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_phase1_summary() -> None:
    """Generate a lightweight Phase 1 summary markdown."""
    nn_metrics = _load_json(os.path.join("project", "results", "node", "nn_selfpred_metrics.json"))
    locality = _load_json(os.path.join("project", "results", "node", "saliency_locality_curves.json"))
    nri = _load_json(os.path.join("project", "results", "node", "nonreciprocity_index.json"))

    lines = []
    lines.append("# Phase 1 summary (node line)")
    lines.append("")

    # Self-prediction
    lines.append("## Self-prediction (ρ^H / ρ^W / In)")
    if nn_metrics:
        R2 = nn_metrics.get("R2_oos")
        if isinstance(R2, dict):
            for key, val in R2.items():
                lines.append(f"- OOS R² ({key}): {val}")
        else:
            lines.append(f"- OOS R²: {R2}")
    else:
        lines.append("- Metrics not found (run nn-train).")
    lines.append("")

    # Locality
    lines.append("## Locality (saliency radii)")
    if locality:
        for out_name, channels in locality.items():
            lines.append(f"- Output {out_name}:")
            for cname, payload in channels.items():
                bands = payload.get("bands_km", [])
                mean = payload.get("mean", [])
                lines.append(f"  - {cname}: bands_km={bands}, mean={mean}")
    else:
        lines.append("- Locality curves not found (run nn-saliency).")
    lines.append("")

    # Non-reciprocity
    lines.append("## Non-reciprocity (optional)")
    if nri and "NRI" in nri:
        lines.append(f"- NRI: {nri['NRI']}")
    else:
        lines.append("- NRI not computed.")
    lines.append("")

    out_dir = os.path.join("project", "results", "node")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "PHASE1_SUMMARY.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[report-phase1] Summary written to {out_path}")
