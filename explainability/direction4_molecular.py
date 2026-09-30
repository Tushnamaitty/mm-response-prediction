"""
Direction 4 -- PART B: RETROSPECTIVE molecular programs associated with longitudinal response trajectory.

NOT predictive validation and NOT causal.  RNA is baseline/as-of-sample tumour biology (latest sample at/before Vt).
Outcome is the Vt-side trajectory from previous Vt response -> current Vt response: improving / stable / worsening.
The six audited continuity-conflict rows and no-history rows never enter trajectory contrasts.

Primary analysis
  * pooled RNA-available history rows; stable is the reference outcome.
  * 50 Hallmark pathways, z-scored over distinct (patient, RNA-vector) samples using the frozen D3 helper.
  * per pathway NominalGEE (multinomial logit, independence working correlation, patient-cluster robust SE).
  * covariates: current response, line group, current PI/IMiD/steroid/CD38/chemo/other, RNA staleness,
    log history gap, and the frozen D3 clinical covariate set.
  * targets: 2-df pathway omnibus; improving-vs-stable and worsening-vs-stable pathway contrasts.
  * BH across pathways separately for each of the three pre-specified families.
  * global whole-RNA-vector permutation gate precedes pathway interpretation.  The global statistic is the sum of
    squared covariate-residualized state-mean contrasts across all pathways; whole RNA vectors are permuted across
    distinct patient/sample units while row sharing of an RNA vector is preserved.  This is a global screen only;
    pathway effect estimates/inference come from NominalGEE.

Sensitivities (separate, labelled families): first history pair per patient; gap<=180d; Vt=VGPR only; exclude Vt=PD;
RNA age<=730d; RNA age<=365d; at/above vs below best prior response; +plasma-cell percentage; +prior exposures.

Run from repo root:
    python -m explainability.direction4_molecular --smoke
    python -m explainability.direction4_molecular
"""

import argparse, datetime, json, os, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from explainability.direction2_common import RESPONSE_RANK, derive_line_group, transform_series
from explainability.direction3_common import (
    CONT_COVARIATES, GLOBAL_PERM_SEED, N_PERM_GLOBAL, PLASMA_COL, PRIOR_EXPOSURE_COLS,
    PRIMARY_TREATMENTS, STALENESS_COL, TREATMENT_COLS, bh_qvalues, build_pathway_z,
    load_feature_sets, pathway_columns, require, sha256_file, write_csv, write_json,
)
from explainability.direction4_common import (
    CONTRAST_STATES, REFERENCE_STATE, exclude_ambiguous, get_dirs4_all, guard_no_overwrite4,
    load_trajectory_master, run_selfchecks, write_manifest4,
)

N_PATHWAYS_SMOKE = 5
N_PERM_SMOKE = 20
STATE_CODE = {"improving": 0, "worsening": 1, "stable": 2}  # NominalGEE uses last category as reference
NONREF = [("improving", 0), ("worsening", 1)]


def _z(s):
    a = np.asarray(s, dtype=float)
    sd = np.nanstd(a)
    require(np.isfinite(sd) and sd > 0, "constant/non-finite continuous covariate")
    return (a - np.nanmean(a)) / sd


def primary_universe(master):
    u, n_amb = exclude_ambiguous(master)
    u = u[u["trajectory_state"].isin(CONTRAST_STATES)]
    u = u[u[STALENESS_COL].notna() & u["current_line_number"].notna()].copy()
    u["line_group"] = derive_line_group(u["current_line_number"])
    require(set(u["trajectory_state"].unique()) == set(CONTRAST_STATES), "a history trajectory state is absent")
    return u.sort_values("pair_id").reset_index(drop=True), n_amb


def add_best_prior_flag(master):
    """Vt-only history summary.  best prior excludes the current row."""
    d = master.sort_values(["public_id", "vt_days_to_visit", "pair_id"]).copy()
    rank = d["vt_disease_response"].map(RESPONSE_RANK).astype(float)
    d["_best_prior_rank"] = rank.groupby(d["public_id"]).transform(lambda x: x.shift().cummax())
    d["_at_or_above_best_prior"] = rank >= d["_best_prior_rank"]
    return d.set_index("pair_id")[["_best_prior_rank", "_at_or_above_best_prior"]]


def make_design(U, sens="primary"):
    cc = U.copy()
    extra = []
    if sens == "first_history_pair":
        first = cc.groupby("public_id", sort=False)["visit_index"].transform("min")
        cc = cc[cc["visit_index"] == first]
    elif sens == "gap_le_180d": cc = cc[cc["gap_prev_days"] <= 180]
    elif sens == "vt_VGPR_only": cc = cc[cc["vt_disease_response"] == "very_good_partial_response"]
    elif sens == "exclude_vt_PD": cc = cc[cc["vt_disease_response"] != "progressive_disease"]
    elif sens == "rna_age_le_730d": cc = cc[cc[STALENESS_COL] <= 730]
    elif sens == "rna_age_le_365d": cc = cc[cc[STALENESS_COL] <= 365]
    elif sens == "at_or_above_best_prior": cc = cc[cc["_at_or_above_best_prior"] == True]
    elif sens == "below_best_prior": cc = cc[cc["_at_or_above_best_prior"] == False]
    elif sens == "plus_plasma": extra = [PLASMA_COL]
    elif sens == "plus_prior_exposure": pass
    elif sens != "primary": raise ValueError(sens)

    miss = list(CONT_COVARIATES) + extra + ["gap_prev_days"]
    any_miss = cc[miss].isna().any(axis=1)
    before = len(cc); cc = cc[~any_miss].copy().reset_index(drop=True)
    require(len(cc) > 30 and cc["public_id"].nunique() > 10, f"{sens}: insufficient complete-case support")
    require(set(cc["trajectory_state"].unique()) == set(CONTRAST_STATES), f"{sens}: not all 3 states remain")

    blocks=[]; names=[]
    def add(a,nm):
        a=np.asarray(a,dtype=float); blocks.append(a if a.ndim==2 else a[:,None]); names.extend(nm)
    add(np.ones(len(cc)), ["const"])
    D=pd.get_dummies(cc["vt_disease_response"].astype(str), drop_first=True, dtype=float)
    add(D.to_numpy(), [f"vt_response[{x}]" for x in D.columns])
    L=pd.get_dummies(cc["line_group"].astype(str), drop_first=True, dtype=float)
    add(L.to_numpy(), [f"line_group[{x}]" for x in L.columns])
    for t in ["pi","imid","steroid","cd38","chemo"]:
        add(cc[TREATMENT_COLS[t]].to_numpy(float), [f"on_{t}"])
    add(cc["currently_on_other"].to_numpy(float), ["on_slamf7_or_bcma"])
    add(_z(np.log1p(cc[STALENESS_COL])), ["log1p_rna_staleness"])
    add(_z(np.log1p(cc["gap_prev_days"])), ["log1p_history_gap"])
    for name,kind in CONT_COVARIATES.items(): add(_z(transform_series(kind,cc[name])), [name])
    if sens == "plus_plasma": add(_z(cc[PLASMA_COL]), [PLASMA_COL])
    if sens == "plus_prior_exposure":
        for t in ["pi","imid","steroid","cd38","chemo"]:
            add(cc[PRIOR_EXPOSURE_COLS[t]].to_numpy(float), [f"prior_exposure_{t}"])
        add(cc["prior_exposure_other"].to_numpy(float), ["prior_exposure_other"])
    X=np.column_stack(blocks)
    require(np.linalg.matrix_rank(X)==X.shape[1], f"{sens}: rank-deficient covariate design")
    y=cc["trajectory_state"].map(STATE_CODE).to_numpy(int)
    return {"cc":cc,"X":X,"names":names,"y":y,"groups":pd.factorize(cc["public_id"])[0],
            "accounting":{"analysis":sens,"rows_before_complete_case":int(before),"rows_final":int(len(cc)),
                          "patients_final":int(cc.public_id.nunique()),"rows_lost_incomplete":int(before-len(cc)),
                          "state_counts":json.dumps(cc.trajectory_state.value_counts().to_dict())}}


def fit_pathway(design,z):
    from statsmodels.genmod.generalized_estimating_equations import NominalGEE
    from statsmodels.genmod.cov_struct import Independence
    X=pd.DataFrame(np.column_stack([z,design["X"]]), columns=["pathway"]+design["names"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod=NominalGEE(design["y"],X,groups=design["groups"],cov_struct=Independence())
        r=mod.fit(maxiter=100)
    if getattr(r,"converged",None) is False: raise RuntimeError("not_converged")
    pnames=list(r.params.index)
    idx={}
    for state,code in NONREF:
        hits=[i for i,n in enumerate(pnames) if n.startswith("pathway[") and str(float(code)) in n]
        require(len(hits)==1, f"cannot uniquely locate pathway coefficient for {state}: {hits}")
        idx[state]=hits[0]
    b=np.asarray(r.params); V=np.asarray(r.cov_params()); se=np.asarray(r.bse); pv=np.asarray(r.pvalues); ci=np.asarray(r.conf_int())
    ii=[idx["improving"],idx["worsening"]]; bb=b[ii]; VV=V[np.ix_(ii,ii)]
    W=float(bb @ np.linalg.solve(VV,bb)); out={"omnibus_chi2":W,"omnibus_df":2,"omnibus_p":float(chi2.sf(W,2))}
    for state,_ in NONREF:
        j=idx[state]; out.update({f"{state}_vs_stable_estimate":float(b[j]),f"{state}_vs_stable_se":float(se[j]),
            f"{state}_vs_stable_p":float(pv[j]),f"{state}_vs_stable_ci_lo":float(ci[j,0]),
            f"{state}_vs_stable_ci_hi":float(ci[j,1]),f"{state}_vs_stable_odds_ratio":float(np.exp(b[j]))})
    return out


def analyze(design,Z,path_cols,label):
    rows=[]
    for j,pw in enumerate(path_cols):
        row={"analysis":label,"pathway":pw,"status":"ok"}
        try: row.update(fit_pathway(design,Z.loc[design["cc"].pair_id,pw].to_numpy(float)))
        except AssertionError: raise
        except Exception as e: row["status"]=f"failed: {type(e).__name__}: {e}"
        rows.append(row)
        if (j+1)%10==0: print(f"    {label}: {j+1}/{len(path_cols)} pathways",flush=True)
    df=pd.DataFrame(rows); require(df.status.eq("ok").all(), f"{label}: one or more pathway models failed")
    df["omnibus_q_bh"]=bh_qvalues(df.omnibus_p.to_numpy())
    df["improving_vs_stable_q_bh"]=bh_qvalues(df.improving_vs_stable_p.to_numpy())
    df["worsening_vs_stable_q_bh"]=bh_qvalues(df.worsening_vs_stable_p.to_numpy())
    return df


def global_stat(design,zmat):
    """Covariate-residualized state separation. Used only as the pre-specified global permutation screen."""
    X=design["X"]; # residualize every pathway against the fixed clinical design
    B=np.linalg.lstsq(X,zmat,rcond=None)[0]; R=zmat-X@B
    s=0.0
    for code in (0,1):
        a=R[design["y"]==code].mean(axis=0); b=R[design["y"]==2].mean(axis=0)
        d=a-b; s += float(d@d)
    return s


def global_permutation(design,Z,path_cols,n_perm,seed):
    zp=Z.loc[design["cc"].pair_id,path_cols].to_numpy(float)
    # preserve repeated use of the same patient/RNA vector
    h=pd.util.hash_pandas_object(pd.DataFrame(np.round(zp,8)),index=False).to_numpy()
    key=pd.DataFrame({"g":design["groups"],"h":h}); sample_idx=key.groupby(["g","h"],sort=False).ngroup().to_numpy()
    first=~key.duplicated().to_numpy(); V=zp[first]
    require(V.shape[0]==sample_idx.max()+1,"sample-vector indexing mismatch")
    obs=global_stat(design,zp); rng=np.random.RandomState(seed); null=np.empty(n_perm)
    for b in range(n_perm):
        null[b]=global_stat(design,V[rng.permutation(len(V))][sample_idx])
        if (b+1)%50==0: print(f"    global permutation {b+1}/{n_perm}",flush=True)
    return {"statistic":"sum squared covariate-residualized pathway mean contrasts: improving-vs-stable + worsening-vs-stable",
        "observed":obs,"n_permutations":n_perm,"seed":seed,"n_distinct_samples_permuted":int(len(V)),
        "p_value":float((1+(null>=obs).sum())/(n_perm+1)),"null_mean":float(null.mean()),
        "null_p2_5":float(np.percentile(null,2.5)),"null_p97_5":float(np.percentile(null,97.5))}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--smoke",action="store_true"); ap.add_argument("--force",action="store_true"); args=ap.parse_args()
    t0=time.time(); dirs=get_dirs4_all(args.smoke); out=dirs["partB"]
    guard_no_overwrite4(f"{out}/output_manifest.json",args.smoke,args.force); os.makedirs(out,exist_ok=True)
    print("DIRECTION 4 / PART B | TRAJECTORY-ASSOCIATED MOLECULAR PROGRAMS")
    master,_=load_trajectory_master(args.smoke,keep_rna_values=True); run_selfchecks(master,args.smoke)
    bp=add_best_prior_flag(master); master=master.join(bp,on="pair_id")
    U,namb=primary_universe(master)
    fs=load_feature_sets(); allp=pathway_columns(fs); paths=allp[:N_PATHWAYS_SMOKE] if args.smoke else allp
    Z=build_pathway_z(master,allp)
    dp=make_design(U,"primary")
    nperm=N_PERM_SMOKE if args.smoke else N_PERM_GLOBAL
    print(f"[primary] rows={len(dp['cc'])}, patients={dp['cc'].public_id.nunique()}, pathways={len(paths)}")
    print(f"[global] {nperm} whole-RNA-vector permutations...")
    glob=global_permutation(dp,Z,paths,nperm,GLOBAL_PERM_SEED); write_json(glob,f"{out}/partB_global_permutation.json")
    print(f"  global p={glob['p_value']:.4g}")
    tables={"primary":analyze(dp,Z,paths,"primary")}; accounting=[dp["accounting"]]
    sens_names=["first_history_pair","gap_le_180d","vt_VGPR_only","exclude_vt_PD","rna_age_le_730d","rna_age_le_365d",
                "at_or_above_best_prior","below_best_prior","plus_plasma","plus_prior_exposure"]
    for s in sens_names:
        print(f"[sensitivity] {s}",flush=True)
        try:
            ds=make_design(U,s); accounting.append(ds["accounting"]); tables[s]=analyze(ds,Z,paths,s)
        except Exception as e:
            accounting.append({"analysis":s,"status":f"failed: {type(e).__name__}: {e}"}); print(f"  FAILED, reported: {e}")
    files=[f"{out}/partB_global_permutation.json"]
    for label,df in tables.items():
        p=f"{out}/partB_results_{label}.csv"; write_csv(df,p,float_format="%.8g"); files.append(p)
    acc=pd.DataFrame(accounting); p=f"{out}/partB_row_accounting.csv"; write_csv(acc,p); files.append(p)
    summary={"analysis":"Direction 4 Part B retrospective molecular trajectory association","smoke":bool(args.smoke),
        "completed_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"runtime_seconds":round(time.time()-t0,1),
        "reference_state":REFERENCE_STATE,"n_audited_ambiguous_rows_excluded":namb,"n_pathways":len(paths),"global_permutation":glob,
        "primary_significant_q_lt_0.05":{"omnibus":int((tables['primary'].omnibus_q_bh<.05).sum()),
        "improving_vs_stable":int((tables['primary'].improving_vs_stable_q_bh<.05).sum()),
        "worsening_vs_stable":int((tables['primary'].worsening_vs_stable_q_bh<.05).sum())},
        "interpretation":"retrospective association of baseline/as-of RNA biology with Vt-side longitudinal response trajectory; not predictive validation and not causal"}
    p=f"{out}/run_summary.json"; write_json(summary,p); files.append(p)
    man=write_manifest4(out,files,{"analysis":"direction4_partB_molecular","smoke":bool(args.smoke),
        "input_master_sha256":sha256_file("data/clinical/visit_pairs_with_rna.csv"),"global_permutation_seed":GLOBAL_PERM_SEED},args.smoke)
    print(f"\nDone in {time.time()-t0:.1f}s. Outputs in {out}\nCombined output SHA256: {man['combined_sha256']}")

if __name__=="__main__": main()
