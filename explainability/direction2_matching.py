"""
Direction 2 -- STEP 1: build and FREEZE the matched-pair tables.

*** RETROSPECTIVE MOLECULAR ASSOCIATION -- NOT PREDICTIVE VALIDATION ***

This script is OUTCOME-BLIND and RNA-BLIND by construction:
  * it reads ONLY the 22 Model-C similarity inputs, pair_id, public_id,
    eligible_for_binary (from task_eligibility.csv) and -- solely to build a
    boolean RNA-availability mask that defines the pre-specified RNA-available
    universe -- days_since_rna_sample, which is dropped immediately;
  * RNA availability defines ROW MEMBERSHIP ONLY. No RNA value and no RNA age ever
    influences a distance, a stratum, the caliper, the candidate ordering or the pair
    selection. This is enforced at runtime: the frame passed to the distance code must
    contain exactly the whitelisted Vt-side columns (assert_matching_frame_columns),
    no pathway/target/vt1_* column may be loaded, and no forbidden column may be present;
  * matches are cross-patient only and patient-once (M1), asserted;
  * the matched pairs are written to CSV and SHA256-hashed BEFORE any outcome
    or pathway value is merged (that happens only in direction2_association.py,
    which refuses to run unless the hashes still match).

Designs written (per family: ordinal = multiclass universe [PRIMARY],
binary = binary universe [SECONDARY]):
  primary_p75_M1   frozen primary: caliper = p75 of nearest-neighbour distances
  S1_p50_M1        sensitivity S1: caliper p50
  S1_p90_M1        sensitivity S1: caliper p90
  S4_random_M1     sensitivity S4: 200 randomised patient-once matchings (p75 caliper).
                   Seeded random VISITING ORDER; draw k uses seed 1042 + k (recorded per
                   row in the `seed` column and in the manifest).

MATCHING-COUNT QA (exact, no tolerance): the exact pair counts are recorded and compared
with the earlier audit (ordinal 259 pairs, binary 261 pairs). A difference is REPORTED with a
tie-breaking diagnosis (re-running the greedy step with the legacy stable-sort tie order and
counting exactly tied candidate distances); it is not silently tolerated and not used to alter
the frozen design. Structural violations (same-patient pair, patient/row reuse, stratum
mismatch, beyond caliper, too few jointly observed features) FAIL the run.

Run from the repo root:
    python -m explainability.direction2_matching
Options:
    --n-random N   number of S4 random matchings (frozen value 200)
    --smoke        write to .../direction2/_smoke/ with a tiny S4 (testing only)
    --force        overwrite an existing frozen manifest (NOT for real runs)
"""

import argparse
import datetime
import json
import os
import time

import numpy as np
import pandas as pd

from explainability.direction2_common import (
    ANALYSIS_LABEL, AUDIT_REFERENCE, CALIPER_PERCENTILE_PRIMARY, CALIPER_PERCENTILES_S1,
    CONTINUOUS_FEATURES, ELIGIBILITY_PATH, EXACT_KEYS, EXPECTED, FAMILIES, FAMILY_ROLE,
    FEATURE_SETS_PATH, FROZEN_SPEC, MASTER_PATH, MIN_JOINT_FEATURES, N_RANDOM_DRAWS_DEFAULT,
    S4_SEED_BASE, SIMILARITY_SOURCE_COLS, SPLIT_PATH, STALENESS_COL, assert_frame_blind,
    assert_loaded_columns_clean, assert_matching_frame_columns, assert_similarity_inputs_clean,
    combined_hash, continuous_matrix, expect, get_dirs, load_feature_sets, package_versions,
    require, select_universe, sha256_file, sha256_text, to_bool, write_csv, write_json,
)


# ---------------------------------------------------------------------------
# Loading (RNA-blind)
# ---------------------------------------------------------------------------
def load_matching_master(fs):
    assert_similarity_inputs_clean(fs)

    header = pd.read_csv(MASTER_PATH, nrows=0).columns.tolist()
    expect("master column count", len(header), EXPECTED["master_columns"])

    load_cols = ["pair_id", "public_id", STALENESS_COL] + SIMILARITY_SOURCE_COLS
    assert_loaded_columns_clean(load_cols)
    missing = sorted(set(load_cols) - set(header))
    require(not missing, f"columns missing from master table: {missing}")

    m = pd.read_csv(MASTER_PATH, usecols=load_cols, low_memory=False)
    elig = pd.read_csv(ELIGIBILITY_PATH, usecols=["pair_id", "eligible_for_binary"])
    m = m.merge(elig, on="pair_id", how="left", validate="one_to_one")
    require(m["eligible_for_binary"].notna().all(), "eligible_for_binary missing after merge")

    # RNA field -> boolean availability mask (universe membership only), then it leaves scope.
    m["rna_available"] = m[STALENESS_COL].notna()
    m = m.drop(columns=[STALENESS_COL])
    require(STALENESS_COL not in m.columns, "RNA age column still present after mask construction")
    m["eligible_for_binary"] = to_bool(m["eligible_for_binary"])

    expect("master rows", len(m), EXPECTED["master_rows"])
    expect("master patients", int(m["public_id"].nunique()), EXPECTED["master_patients"])
    expect("RNA-available rows", int(m["rna_available"].sum()), EXPECTED["rna_rows"])
    expect("RNA-available patients", int(m.loc[m["rna_available"], "public_id"].nunique()),
           EXPECTED["rna_patients"])
    expect("binary-eligible rows", int(m["eligible_for_binary"].sum()), EXPECTED["binary_eligible_rows"])
    require(m["pair_id"].is_unique, "pair_id is not unique in the master table")
    return m


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------
def stratum_distance(Z_s, pid_s):
    """RMS z-difference over jointly observed features; +inf if fewer than
    MIN_JOINT_FEATURES are jointly observed, for same-patient pairs, and on the diagonal."""
    n = Z_s.shape[0]
    ss = np.zeros((n, n))
    cnt = np.zeros((n, n), dtype=np.int16)
    for k in range(Z_s.shape[1]):
        col = Z_s[:, k]
        diff = col[:, None] - col[None, :]
        ok = ~np.isnan(diff)
        ss += np.where(ok, diff * diff, 0.0)
        cnt += ok
    D = np.sqrt(ss / np.maximum(cnt, 1))
    D[cnt < MIN_JOINT_FEATURES] = np.inf
    D[pid_s[:, None] == pid_s[None, :]] = np.inf      # SAME-PATIENT EXCLUSION
    np.fill_diagonal(D, np.inf)
    return D


class StratumCache:
    """Per exact-stratum distance matrices (computed once, reused by every design)."""

    def __init__(self, u, Z, pid_code):
        n = len(u)
        self.items = []                                  # (row_index_array, D or None)
        self.row_stratum = np.full(n, -1, dtype=np.int64)
        self.row_local = np.full(n, -1, dtype=np.int64)
        self.nn = np.full(n, np.inf)
        groups = u.groupby(EXACT_KEYS, sort=True).indices
        self.strata_table = []
        for key, idx in groups.items():
            idx = np.asarray(idx, dtype=np.int64)
            if len(idx) < 2:
                self.items.append((idx, None))
                self.strata_table.append((key, len(idx), len(np.unique(pid_code[idx])), 0))
                continue
            D = stratum_distance(Z[idx], pid_code[idx])
            sid = len(self.items)
            self.items.append((idx, D))
            self.row_stratum[idx] = sid
            self.row_local[idx] = np.arange(len(idx))
            nn = D.min(axis=1)
            self.nn[idx] = nn
            self.strata_table.append((key, len(idx), len(np.unique(pid_code[idx])),
                                      int(np.isfinite(nn).sum())))

    def all_candidates(self):
        I, J, d = [], [], []
        for idx, D in self.items:
            if D is None:
                continue
            iu = np.triu_indices(len(idx), 1)
            dd = D[iu]
            ok = np.isfinite(dd)
            I.append(idx[iu[0][ok]])
            J.append(idx[iu[1][ok]])
            d.append(dd[ok])
        if not I:
            return (np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([]))
        return np.concatenate(I), np.concatenate(J), np.concatenate(d)


# ---------------------------------------------------------------------------
# Matching algorithms
# ---------------------------------------------------------------------------
def greedy_patient_once(I, J, d, caliper, pair_id, pid_code, tie_break="pair_id"):
    """M1: global greedy pairing in ascending distance; each patient used at most once.
    tie_break="pair_id" (FROZEN): ties broken EXPLICITLY by (min pair_id, max pair_id).
    tie_break="legacy_generation_order" (QA DIAGNOSTIC ONLY, never written as a frozen table):
    stable sort on distance alone, i.e. ties resolved by candidate generation order."""
    keep = d <= caliper
    I, J, d = I[keep], J[keep], d[keep]
    if tie_break == "pair_id":
        pi, pj = pair_id[I], pair_id[J]
        lo, hi = np.minimum(pi, pj), np.maximum(pi, pj)
        order = np.lexsort((hi, lo, d))                  # primary key = distance
    elif tie_break == "legacy_generation_order":
        order = np.argsort(d, kind="stable")
    else:
        raise ValueError(f"unknown tie_break {tie_break!r}")
    used = [False] * (int(pid_code.max()) + 1)
    pc = pid_code.tolist()
    acc_i, acc_j, acc_d = [], [], []
    for i, j, dist in zip(I[order].tolist(), J[order].tolist(), d[order].tolist()):
        a, b = pc[i], pc[j]
        if used[a] or used[b]:
            continue
        used[a] = True
        used[b] = True
        acc_i.append(i)
        acc_j.append(j)
        acc_d.append(dist)
    return (np.array(acc_i, dtype=np.int64), np.array(acc_j, dtype=np.int64),
            np.array(acc_d, dtype=float))


def randomized_nn_matching(cache, caliper, pair_id, pid_code, seed):
    """S4: rows visited in a seeded random order; each row whose patient is still
    unused is paired with its nearest in-caliper partner from an unused patient
    (ties by smallest pair_id). Patient-once. Deterministic given `seed`."""
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(pair_id))
    used = np.zeros(int(pid_code.max()) + 1, dtype=bool)
    acc_i, acc_j, acc_d = [], [], []
    for i in order:
        if used[pid_code[i]]:
            continue
        sid = cache.row_stratum[i]
        if sid < 0:
            continue
        idx, D = cache.items[sid]
        drow = D[cache.row_local[i]]
        ok = (drow <= caliper) & (~used[pid_code[idx]])
        if not ok.any():
            continue
        cand = np.flatnonzero(ok)
        dmin = drow[cand].min()
        best = cand[drow[cand] == dmin]
        j_local = best[np.argmin(pair_id[idx[best]])]
        j = idx[j_local]
        used[pid_code[i]] = True
        used[pid_code[j]] = True
        acc_i.append(int(i))
        acc_j.append(int(j))
        acc_d.append(float(dmin))
    return (np.array(acc_i, dtype=np.int64), np.array(acc_j, dtype=np.int64),
            np.array(acc_d, dtype=float))


# ---------------------------------------------------------------------------
# Pair tables
# ---------------------------------------------------------------------------
def pairs_frame(u, Z, ii, jj, dd, pair_id, family, design, draw=-1, seed=-1):
    swap = pair_id[ii] > pair_id[jj]                     # canonical: smaller pair_id is member "a"
    a = np.where(swap, jj, ii)
    b = np.where(swap, ii, jj)
    n_joint = (~np.isnan(Z[a]) & ~np.isnan(Z[b])).sum(axis=1)
    pub = u["public_id"].to_numpy(dtype=object)
    strata = u.iloc[a][EXACT_KEYS].reset_index(drop=True).rename(
        columns={"currently_on_pi": "on_pi", "currently_on_imid": "on_imid",
                 "currently_on_cd38": "on_cd38"})
    P = pd.DataFrame({
        "family": family,
        "design": design,
        "draw": draw,
        "seed": seed,
        "pair_index": np.arange(len(a)),
        "pair_id_a": pair_id[a],
        "pair_id_b": pair_id[b],
        "public_id_a": pub[a],
        "public_id_b": pub[b],
        "distance": dd,
        "n_joint_features": n_joint,
    })
    return pd.concat([P, strata], axis=1)


def assert_pairs_valid(P, u, caliper, label):
    """PERMANENT checks: same-patient exclusion, patient-once, row-once, same exact
    stratum, caliper, min joint features. Applied per draw for S4."""
    groups = [P] if (P["draw"] == -1).all() else [g for _, g in P.groupby("draw")]
    key = u.set_index("pair_id")[EXACT_KEYS]
    for g in groups:
        if len(g) == 0:
            continue
        require((g["public_id_a"].to_numpy() != g["public_id_b"].to_numpy()).all(),
                f"{label}: same-patient pair present (same-patient exclusion violated)")
        require((g["pair_id_a"].to_numpy() < g["pair_id_b"].to_numpy()).all(),
                f"{label}: pair orientation not canonical")
        pats = pd.concat([g["public_id_a"], g["public_id_b"]])
        require(pats.is_unique, f"{label}: a patient appears in more than one pair (patient-once violated)")
        rows = pd.concat([g["pair_id_a"], g["pair_id_b"]])
        require(rows.is_unique, f"{label}: a visit row appears in more than one pair")
        require((g["distance"].to_numpy() <= caliper + 1e-9).all(), f"{label}: pair beyond caliper")
        require((g["n_joint_features"].to_numpy() >= MIN_JOINT_FEATURES).all(),
                f"{label}: pair with fewer than {MIN_JOINT_FEATURES} jointly observed features")
        ka = key.loc[g["pair_id_a"].to_numpy()].reset_index(drop=True)
        kb = key.loc[g["pair_id_b"].to_numpy()].reset_index(drop=True)
        require(ka.equals(kb), f"{label}: members are not in the same exact stratum")
    return True


def row_loss_reasons(cache, caliper, matched_rows_mask):
    """Outcome-blind attrition: why each universe row was not in a primary pair."""
    n = len(matched_rows_mask)
    size = np.zeros(n, dtype=np.int64)
    for idx, _ in cache.items:
        size[idx] = len(idx)
    reason = np.full(n, "matched", dtype=object)
    un = ~matched_rows_mask
    reason[un & (size < 2)] = "stratum_has_<2_rows"
    reason[un & (size >= 2) & ~np.isfinite(cache.nn)] = "no_cross_patient_candidate"
    reason[un & np.isfinite(cache.nn) & (cache.nn > caliper)] = "nearest_neighbour_beyond_caliper"
    reason[un & np.isfinite(cache.nn) & (cache.nn <= caliper)] = "partner_or_patient_already_used"
    return pd.Series(reason).value_counts().to_dict()


def pair_set(pair_id, ii, jj):
    lo = np.minimum(pair_id[ii], pair_id[jj]).tolist()
    hi = np.maximum(pair_id[ii], pair_id[jj]).tolist()
    return set(zip(lo, hi))


# ---------------------------------------------------------------------------
# One family
# ---------------------------------------------------------------------------
def run_family(master, fs, family, n_random, dirs, written):
    t0 = time.time()
    u = select_universe(master, family)
    # RNA availability has now done its ONLY job (defining the universe rows). Remove it so that the
    # frame handed to the distance code carries no RNA-derived column at all.
    require(bool(u["rna_available"].all()), f"[{family}] universe contains a non-RNA-available row")
    u = u.drop(columns=["rna_available"])
    assert_matching_frame_columns(u)                     # RUNTIME: exactly the whitelisted Vt-side columns
    assert_frame_blind(u, fs)                            # RUNTIME: no outcome / vt1_* / RNA / leakage column

    pair_id = u["pair_id"].to_numpy(dtype=np.int64)
    pid_code = pd.factorize(u["public_id"])[0].astype(np.int64)
    Z = continuous_matrix(u)
    cache = StratumCache(u, Z, pid_code)
    I, J, d = cache.all_candidates()

    finite_nn = cache.nn[np.isfinite(cache.nn)]
    require(len(finite_nn) > 0, f"[{family}] no row has any cross-patient same-stratum candidate")
    calipers = {f"p{p}": float(np.percentile(finite_nn, p))
                for p in sorted({CALIPER_PERCENTILE_PRIMARY, *CALIPER_PERCENTILES_S1})}
    cal_primary = calipers[f"p{CALIPER_PERCENTILE_PRIMARY}"]

    designs = [("primary_p75_M1", cal_primary),
               ("S1_p50_M1", calipers["p50"]),
               ("S1_p90_M1", calipers["p90"])]
    sizes = [len(i) for i, _ in cache.items]
    summary = {"family": family, "role": FAMILY_ROLE[family], "universe_rows": len(u),
               "universe_patients": int(u["public_id"].nunique()),
               "n_strata": len(cache.items),
               "strata_size": {"min": int(min(sizes)), "median": float(np.median(sizes)),
                               "max": int(max(sizes)), "n_lt2": int(sum(s < 2 for s in sizes))},
               "rows_with_any_candidate": int(np.isfinite(cache.nn).sum()),
               "nn_distance_percentiles": {str(p): float(np.percentile(finite_nn, p))
                                           for p in (25, 50, 75, 90, 95)},
               "calipers": calipers, "n_candidate_pairs_all_finite": int(len(d)),
               "designs": {}}

    for name, cal in designs:
        ii, jj, dd = greedy_patient_once(I, J, d, cal, pair_id, pid_code, tie_break="pair_id")
        P = pairs_frame(u, Z, ii, jj, dd, pair_id, family, name)
        assert_pairs_valid(P, u, cal, f"{family}/{name}")
        path = f"{dirs['frozen']}/pairs_{family}_{name}.csv"
        write_csv(P, path, float_format="%.10f")
        written.append(path)
        info = {"n_pairs": int(len(P)), "n_patients": int(2 * len(P)), "caliper": cal}

        if name == "primary_p75_M1":
            matched = np.zeros(len(u), dtype=bool)
            matched[ii] = True
            matched[jj] = True
            info["row_attrition_reasons"] = row_loss_reasons(cache, cal, matched)

            # ---- EXACT matching-count QA vs the earlier audit (no tolerance) ----
            ref = AUDIT_REFERENCE[family]["pairs"]
            ii_l, jj_l, _ = greedy_patient_once(I, J, d, cal, pair_id, pid_code,
                                                tie_break="legacy_generation_order")
            new_set, leg_set = pair_set(pair_id, ii, jj), pair_set(pair_id, ii_l, jj_l)
            dk = d[d <= cal]
            _, cnt = np.unique(dk, return_counts=True)
            info["count_qa"] = {
                "n_pairs": int(len(P)),
                "audit_reference_pairs": int(ref),
                "difference_vs_audit_reference": int(len(P) - ref),
                "identical_count_to_audit_reference": bool(len(P) == ref),
                "legacy_tie_order_pairs": int(len(ii_l)),
                "pair_set_identical_to_legacy_tie_order": bool(new_set == leg_set),
                "n_pairs_only_in_frozen_selection": int(len(new_set - leg_set)),
                "n_pairs_only_in_legacy_selection": int(len(leg_set - new_set)),
                "n_candidate_pairs_within_caliper": int(len(dk)),
                "n_candidate_pairs_sharing_an_exactly_tied_distance": int(cnt[cnt > 1].sum()),
                "note": ("Legacy tie order = stable sort on distance in candidate generation order. "
                         "Differences between the frozen (pair_id) and legacy selections can only arise "
                         "from exactly tied candidate distances."),
            }
            qa = info["count_qa"]
            print(f"  [{family}] EXACT pairs = {qa['n_pairs']} (audit reference {ref}; "
                  f"difference {qa['difference_vs_audit_reference']:+d}); "
                  f"selection identical to legacy tie order: {qa['pair_set_identical_to_legacy_tie_order']}; "
                  f"candidates sharing exactly tied distances: "
                  f"{qa['n_candidate_pairs_sharing_an_exactly_tied_distance']}")

            # outcome-blind balance: mean |z-difference| per frozen feature over matched pairs
            pos = pd.Series(np.arange(len(pair_id)), index=pair_id)
            ra = pos.loc[P["pair_id_a"].to_numpy()].to_numpy()
            rb = pos.loc[P["pair_id_b"].to_numpy()].to_numpy()
            absdiff = np.abs(Z[ra] - Z[rb])
            bal = pd.DataFrame({"feature": list(CONTINUOUS_FEATURES),
                                "mean_abs_z_difference": np.nanmean(absdiff, axis=0),
                                "share_pairs_observed": (~np.isnan(absdiff)).mean(axis=0)})
            bpath = f"{dirs['frozen']}/matching_balance_outcome_blind_{family}.csv"
            write_csv(bal, bpath, float_format="%.8f")
            written.append(bpath)
        summary["designs"][name] = info
        print(f"  [{family}] {name}: caliper={cal:.6f} pairs={len(P)}")

    # S4: 200 randomised patient-once matchings (p75 caliper); draw k uses seed S4_SEED_BASE + k
    parts = []
    for k in range(n_random):
        seed = S4_SEED_BASE + k
        ii, jj, dd = randomized_nn_matching(cache, cal_primary, pair_id, pid_code, seed=seed)
        parts.append(pairs_frame(u, Z, ii, jj, dd, pair_id, family, "S4_random_M1", draw=k, seed=seed))
    if parts:
        P4 = pd.concat(parts, ignore_index=True)
        assert_pairs_valid(P4, u, cal_primary, f"{family}/S4_random_M1")
        path = f"{dirs['frozen']}/pairs_{family}_S4_random_M1.csv"
        write_csv(P4, path, float_format="%.10f")
        written.append(path)
        per_draw = P4.groupby("draw").size()
        summary["designs"]["S4_random_M1"] = {
            "n_draws": n_random, "seed_base": S4_SEED_BASE,
            "seed_formula": f"seed_k = {S4_SEED_BASE} + k, k = 0..{n_random - 1}",
            "pairs_per_draw_min": int(per_draw.min()), "pairs_per_draw_max": int(per_draw.max())}

    st = pd.DataFrame([{"vt_disease_response": k[0], "line_group": k[1], "on_pi": k[2],
                        "on_imid": k[3], "on_cd38": k[4], "n_rows": n_r, "n_patients": n_p,
                        "n_rows_with_candidate": n_c}
                       for k, n_r, n_p, n_c in cache.strata_table])
    spath = f"{dirs['frozen']}/matching_strata_{family}.csv"
    write_csv(st, spath)
    written.append(spath)

    apath = f"{dirs['frozen']}/matching_attrition_{family}.json"
    write_json(summary, apath)
    written.append(apath)
    print(f"  [{family}] done in {time.time() - t0:.1f}s")
    return summary


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-random", type=int, default=N_RANDOM_DRAWS_DEFAULT)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    n_random = 3 if args.smoke else args.n_random
    if not args.smoke:
        require(n_random == N_RANDOM_DRAWS_DEFAULT,
                f"frozen S4 design requires --n-random {N_RANDOM_DRAWS_DEFAULT} (use --smoke to test)")

    print(f"DIRECTION 2 / STEP 1: matching  |  {ANALYSIS_LABEL}")
    dirs = get_dirs(args.smoke)
    manifest_path = f"{dirs['frozen']}/frozen_pair_manifest.json"
    if os.path.exists(manifest_path) and not args.force:
        raise SystemExit(f"[direction2] {manifest_path} already exists: frozen outputs are never "
                         f"overwritten silently. Delete them deliberately or pass --force (smoke only).")
    if args.force:
        require(args.smoke, "--force is only allowed together with --smoke (real frozen outputs are never overwritten)")
    os.makedirs(dirs["frozen"], exist_ok=True)

    fs = load_feature_sets()
    master = load_matching_master(fs)

    written = []
    per_family = {}
    for family in FAMILIES:
        print(f"[{family}] ({FAMILY_ROLE[family]})")
        per_family[family] = run_family(master, fs, family, n_random, dirs, written)

    spec_text = json.dumps(FROZEN_SPEC, indent=2, sort_keys=True, default=str)
    spec_path = f"{dirs['frozen']}/matching_spec.json"
    with open(spec_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(spec_text + "\n")
    written.append(spec_path)

    # ---- FREEZE: hash every frozen file, then the whole set ----
    def rel(p):
        return os.path.relpath(p, dirs["frozen"]).replace("\\", "/")

    file_hashes = {rel(p): sha256_file(p) for p in sorted(written)}
    here = os.path.dirname(os.path.abspath(__file__))
    manifest = {
        "label": ANALYSIS_LABEL,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "smoke": bool(args.smoke),
        "n_random_draws": n_random,
        "s4_seed_base": S4_SEED_BASE,
        "s4_seed_formula": f"seed_k = {S4_SEED_BASE} + k",
        "spec_sha256": sha256_text(spec_text + "\n"),
        "input_files_sha256": {MASTER_PATH: sha256_file(MASTER_PATH), SPLIT_PATH: sha256_file(SPLIT_PATH),
                               ELIGIBILITY_PATH: sha256_file(ELIGIBILITY_PATH),
                               FEATURE_SETS_PATH: sha256_file(FEATURE_SETS_PATH)},
        "code_sha256": {"direction2_common.py": sha256_file(os.path.join(here, "direction2_common.py")),
                        "direction2_matching.py": sha256_file(os.path.abspath(__file__))},
        "rna_usage_in_matching": FROZEN_SPEC["rna_usage_in_matching"],
        "rna_fields_read_during_matching": [f"{STALENESS_COL} (boolean availability mask only; dropped immediately)"],
        "pathway_columns_read_during_matching": 0,
        "outcome_columns_read_during_matching": 0,
        "calipers": {f: per_family[f]["calipers"] for f in FAMILIES},
        "exact_primary_pair_counts": {f: per_family[f]["designs"]["primary_p75_M1"]["n_pairs"] for f in FAMILIES},
        "count_qa": {f: per_family[f]["designs"]["primary_p75_M1"]["count_qa"] for f in FAMILIES},
        "files": file_hashes,
        "combined_sha256": combined_hash(file_hashes),
        "versions": package_versions(),
    }
    write_json(manifest, manifest_path)

    print("\nFROZEN PAIR TABLES written to:", dirs["frozen"])
    for k in sorted(file_hashes):
        print(f"  {file_hashes[k]}  {k}")
    print(f"\nCOMBINED FROZEN SHA256: {manifest['combined_sha256']}")
    print(f"S4 seeds: draw k uses {S4_SEED_BASE} + k (k = 0..{n_random - 1}); recorded per row in the 'seed' column.")
    print("Matching finished. No outcome or pathway value was read. Run STEP 2 next:")
    print("    python -m explainability.direction2_association" + (" --smoke --skip-regression" if args.smoke else ""))


if __name__ == "__main__":
    main()
