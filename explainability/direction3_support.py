"""
Direction 3 -- STEP 0: objective support-rule table (no RNA association, no modelling).

Computes, for every treatment class, the RNA-available support in
  * partA_trainval : RNA-available TRAIN+VAL rows (Part A CV universe), binary + multiclass
  * partB_pooled   : pooled RNA-available binary rows with non-missing line (Part B universe)
for the group AND its complement, applies the pre-specified rule
  primary     >= 150 RNA patients AND >= 150 events AND >= 150 non-events (group and complement)
  exploratory >=  80 RNA patients AND >=  75 events AND >=  75 non-events
  unsupported otherwise
  multiclass  additionally >= 20 rows per class (group and complement)
and STOPS if the result differs from the declared roles
(PI/IMiD/steroid primary; CD38/chemo exploratory; SLAMF7/BCMA unsupported).

Overlapping classes are NOT mutually exclusive: each class is compared with "not on that class".
Also writes the descriptive co-occurrence matrix and treatment-signature counts.

Run from the repo root:   python -m explainability.direction3_support [--smoke] [--force]
"""

import argparse

import pandas as pd

from explainability.direction3_common import (
    LABEL_A, LABEL_B, RNA_INTERPRETATION, TREATMENT_COLS, assert_predictors_clean, assert_support_matches_declared_roles,
    assert_treatment_fields_clean, get_dirs3, guard_no_overwrite, load_feature_sets, load_master, support_table,
    write_csv, write_json, write_output_manifest,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    dirs = get_dirs3(args.smoke)
    marker = f"{dirs['support']}/output_manifest.json"
    guard_no_overwrite(marker, args.smoke, args.force)

    fs = load_feature_sets()
    assert_predictors_clean(fs)
    assert_treatment_fields_clean(fs)
    m = load_master()

    tbl = support_table(m)
    files = []
    p = f"{dirs['support']}/support_rule_table.csv"
    write_csv(tbl, p)
    files.append(p)
    print(tbl[["treatment", "role_declared", "universe", "group_pts", "group_events", "group_non_events",
               "complement_pts", "tier", "tier_declared"]].to_string(index=False))
    assert_support_matches_declared_roles(tbl)            # STOP if the objective rule disagrees with the roles

    # descriptive overlap (binary RNA-available rows; NOT used for inference)
    b = m[m["rna_avail"] & m["eligible_for_binary"]]
    names = list(TREATMENT_COLS)
    co = pd.DataFrame({a: {c: int((b[TREATMENT_COLS[a]] & b[TREATMENT_COLS[c]]).sum()) for c in names} for a in names})
    co.index.name = "treatment"
    p = f"{dirs['support']}/treatment_cooccurrence_rows.csv"
    write_csv(co.reset_index(), p)
    files.append(p)
    sig = b[[TREATMENT_COLS[t] for t in names]].apply(
        lambda r: "+".join([t for t in names if r[TREATMENT_COLS[t]]]) or "none", axis=1)
    s = b.assign(signature=sig).groupby("signature").agg(
        rows=("pair_id", "size"), patients=("public_id", "nunique"), events=("improved", "sum")).reset_index()
    p = f"{dirs['support']}/treatment_signature_counts.csv"
    write_csv(s.sort_values("rows", ascending=False), p)
    files.append(p)

    p = f"{dirs['support']}/support_rule_spec.json"
    write_json({"labels": [LABEL_A, LABEL_B], "interpretation": RNA_INTERPRETATION,
                "declared_roles": {"primary": ["pi", "imid", "steroid"], "exploratory": ["cd38", "chemo"],
                                   "unsupported": ["slamf7", "bcma"]},
                "rule": "primary >=150 pts & >=150 events & >=150 non-events (group AND complement); "
                        "exploratory >=80 pts & >=75 events & >=75 non-events; multiclass >=20 rows/class",
                "all_declared_roles_reproduced": True}, p)
    files.append(p)
    man = write_output_manifest(dirs["support"], files, {"label": "support rule (no RNA association)"})
    print(f"\nDeclared roles reproduced by the objective support rule. Combined SHA256: {man['combined_sha256']}")


if __name__ == "__main__":
    main()
