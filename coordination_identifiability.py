"""
Identifiability of the coordination-consensus dimensions from behavioral signatures.

Constructive Monte-Carlo demonstration accompanying
    "The Coordination-Consensus Continuum" (A.-G. Adam, working paper).

------------------------------------------------------------------------------
STATUS OF THIS ANALYSIS
------------------------------------------------------------------------------
This is a CONSTRUCTIVE demonstration, not an empirical measurement. Synthetic
convergence episodes are generated from a transparent model in which the four
structural dimensions are known by construction. The result therefore shows
that, *under this model*, the latent attribution dimensions are non-identified
from the behavioral signature, and quantifies by how much. The magnitudes are
model-dependent; the robustness block shows the qualitative conclusion is
stable across confounder strength and an injected deception footprint. The
real-world magnitudes must be ESTIMATED FROM DATA, not assumed. Treat this as a
proof-of-concept that motivates an empirical study, not as a validated result.

------------------------------------------------------------------------------
WHY TWO MODEL CLASSES AND AN INFORMATION-THEORETIC BOUND
------------------------------------------------------------------------------
"Identifiability" here means: how much does the observable signature pin down a
latent dimension? A single learner's R^2 conflates "non-identified" with "this
learner underfit." To rule that out, recovery is assessed two independent ways:

  (1) cross-validated out-of-sample R^2 from two different learner classes
      (random forest and histogram gradient boosting), reported as mean +/- SD
      across folds. Two unrelated function approximators agreeing rules out an
      artefact of one model family.

  (2) a Kraskov-Stoegbauer-Grassberger (KSG) k-NN estimate of the mutual
      information I(signature ; y), in nats, reported also as a Gaussian-
      equivalent recoverable fraction  1 - exp(-2 I). Mutual information is
      model-free and captures dependence of any form, so it is an UPPER BOUND
      on what any estimator could recover. If even the MI bound is low, the
      information is not in the signature -- the failure is non-identification,
      not weak modelling.

If a dimension were identified from the signature, both would be high. The four
structural axes are low on both; the exogenous environmental factor is high on
both (a positive control that the pipeline recovers signal when it exists).

Author: A.-G. Adam.  Released for reproducibility under the MIT License.
"""

from __future__ import annotations
import argparse
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedKFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma
from sklearn.metrics import r2_score

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
SEED = 20240613
N = 30_000
N_SPLITS = 3
MI_K = 5
MI_SUBSAMPLE = 5_000
rng = np.random.default_rng(SEED)


# ----------------------------------------------------------------------------
# Generative model of convergence episodes
# ----------------------------------------------------------------------------
def generate(rng, n, conf_mult=1.0, deception_footprint=0.0):
    """Generate n synthetic episodes with known latent ground truth.

    Latent structural dimensions, drawn independently and uniformly on [0, 1]
    (independence is the conservative choice: any recoverable structure then
    comes from the signature map, not from prior correlation among the axes):

        I  intentionality          (0 emergent     ... 1 directed)
        C  control centralization  (0 distributed  ... 1 centralized)
        A  participant agency      (0 instrument   ... 1 autonomous)
        D  deceptiveness           (0 transparent  ... 1 concealed)

    Plus one EXOGENOUS environmental factor (not an attribution dimension):

        T  organic templating / algorithmic amplification

    The observable signature has five features, each a weighted combination of
    a latent structural coordination drive and T, with feature-specific
    loadings that overlap heavily (this overlap is signature convergence).

    Deceptiveness is modelled faithfully to its definition as a RELATIONAL
    property: by default it has no behavioral footprint. Its role is
    adversarial mimicry -- a deceptive operation fabricates one coherent
    organic-looking appearance and mixes its true structural signal toward it
    in proportion to D, so a maximally deceptive coordinated episode emits a
    trace indistinguishable from a genuinely organic one. The optional
    `deception_footprint` argument (used only in the robustness block) grants D
    a small DIRECT behavioral tell it does not have by definition, to show the
    non-identification of the OTHER axes does not depend on D being invisible.
    """
    I = rng.uniform(0, 1, n)
    C = rng.uniform(0, 1, n)
    A = rng.uniform(0, 1, n)
    D = rng.uniform(0, 1, n)
    T = rng.uniform(0, 1, n)
    instrumentation = 1.0 - A

    # single structural index, used both as the signature driver and as the
    # ground-truth coordination position (unified -- one definition, no drift)
    coord = 0.50 * C + 0.30 * instrumentation + 0.20 * I

    A_load = np.array([0.85, 0.80, 0.55, 0.60, 0.85])   # coordination loading
    B_load = np.array([0.45, 0.40, 0.60, 0.60, 0.20])   # templating loading
    D_tell = np.array([-0.5, 0.0, -0.7, 0.0, -0.4])     # direct deception tell

    mimic = rng.uniform(0, 1, n)                         # coherent fabricated appearance
    feats = np.empty((n, 5))
    for k in range(5):
        structural = A_load[k] * coord
        masked = (1.0 - D) * structural + D * (A_load[k] * mimic)
        linear = (masked
                  + conf_mult * B_load[k] * T
                  + deception_footprint * D_tell[k] * D
                  + rng.normal(0, 0.07, n))
        feats[:, k] = np.clip(linear, 0.0, 1.0)

    latents = {"Intentionality": I, "Centralization": C,
               "Participant agency": A, "Deceptiveness": D}
    return feats, latents, coord, T


# ----------------------------------------------------------------------------
# Recovery metric 1 -- cross-validated out-of-sample R^2
# ----------------------------------------------------------------------------
def cv_r2(S, y, estimator, n_splits=N_SPLITS, seed=0):
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=1, random_state=seed)
    scores = cross_val_score(estimator, S, y, cv=cv, scoring="r2", n_jobs=-1)
    return scores.mean(), scores.std()


def rf():
    return RandomForestRegressor(n_estimators=100, max_depth=12,
                                 n_jobs=-1, random_state=0)


def gbm():
    return HistGradientBoostingRegressor(max_depth=8, max_iter=120,
                                         learning_rate=0.1, random_state=0)


# ----------------------------------------------------------------------------
# Recovery metric 2 -- KSG mutual information I(S ; y), in nats
# Kraskov, Stoegbauer & Grassberger (2004), estimator 1, max-norm.
# Reported also as a Gaussian-equivalent recoverable fraction 1 - exp(-2 I),
# which puts the information bound on the same 0-1 scale as R^2.
# ----------------------------------------------------------------------------
def mi_to_r2(mi):
    return 1.0 - np.exp(-2.0 * mi)
def ksg_mutual_information(S, y, k=MI_K, n_sub=MI_SUBSAMPLE, seed=0):
    from scipy.spatial import cKDTree
    r = np.random.default_rng(seed)
    n = len(y)
    if n > n_sub:
        sel = r.choice(n, n_sub, replace=False)
        S, y = S[sel], y[sel]
        n = n_sub
    X = (S - S.mean(0)) / (S.std(0) + 1e-12)
    Y = ((y - y.mean()) / (y.std() + 1e-12)).reshape(-1, 1)
    X = X + 1e-10 * r.standard_normal(X.shape)   # break ties
    Y = Y + 1e-10 * r.standard_normal(Y.shape)
    Z = np.hstack([X, Y])

    tz, tx, ty = cKDTree(Z), cKDTree(X), cKDTree(Y)
    dist, _ = tz.query(Z, k=k + 1, p=np.inf)     # max-norm to k-th joint neighbour
    eps = dist[:, k]
    nx = np.array(tx.query_ball_point(X, eps - 1e-12, p=np.inf, return_length=True)) - 1
    ny = np.array(ty.query_ball_point(Y, eps - 1e-12, p=np.inf, return_length=True)) - 1
    mi = (digamma(k) + digamma(n)
          - np.mean(digamma(nx + 1) + digamma(ny + 1)))
    return max(mi, 0.0)


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------
def hline(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="optional path to write a results CSV")
    args = ap.parse_args()

    S, latents, coord, T = generate(rng, N)

    # clipping diagnostic (addresses the boundary-mass concern)
    clip_rate = np.mean((S <= 0.0) | (S >= 1.0))
    print(f"Episodes: {N:,}   features: {S.shape[1]}   "
          f"boundary-clipped feature values: {100*clip_rate:.2f}%")

    hline("RECOVERY OF EACH LATENT FROM THE SIGNATURE (independent metrics)")
    print(f"{'dimension':22s} {'R2 (RF)':>14s} {'R2 (GBM)':>14s} "
          f"{'MI (nats)':>10s} {'MI->R2':>8s}")
    rows = {}
    for name, y in latents.items():
        r_rf, s_rf = cv_r2(S, y, rf())
        r_gb, s_gb = cv_r2(S, y, gbm())
        mi = ksg_mutual_information(S, y)
        rows[name] = (r_rf, s_rf, r_gb, s_gb, mi, mi_to_r2(mi))
        print(f"{name:22s} {r_rf:6.3f}+/-{s_rf:4.3f} {r_gb:6.3f}+/-{s_gb:4.3f} "
              f"{mi:10.3f} {mi_to_r2(mi):8.3f}")

    # positive control: the exogenous environmental factor must be recoverable
    r_rf, s_rf = cv_r2(S, T, rf())
    mi = ksg_mutual_information(S, T)
    print(f"{'(control) Templating T':22s} {r_rf:6.3f}+/-{s_rf:4.3f} "
          f"{'':>14s} {mi:10.3f} {mi_to_r2(mi):8.3f}")

    hline("BINARY ATTRIBUTION FROM BEHAVIOUR ALONE")
    y_bin = (coord > np.median(coord)).astype(int)
    clf = RandomForestClassifier(n_estimators=300, max_depth=16,
                                 n_jobs=-1, random_state=0)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
    acc = cross_val_score(clf, S, y_bin, cv=skf, scoring="accuracy", n_jobs=-1)
    # ambiguity band on a held-out fold
    tr = np.arange(N) % N_SPLITS != 0
    clf.fit(S[tr], y_bin[tr])
    proba = clf.predict_proba(S[~tr])[:, 1]
    band = np.mean((proba >= 0.40) & (proba <= 0.60))
    print(f"  CV accuracy (coordination vs consensus side): "
          f"{100*acc.mean():.1f}% +/- {100*acc.std():.1f}   (chance = 50%)")
    print(f"  Episodes in the deep-ambiguous band P in [0.40, 0.60]: {100*band:.1f}%")

    hline("SAME SIGNATURE, DIFFERENT TRUTH")
    C = latents["Centralization"]
    Sz = (S - S.mean(0)) / S.std(0)
    nn = NearestNeighbors(n_neighbors=2).fit(Sz)
    dist, ind = nn.kneighbors(Sz)
    nn_dist, nn_idx = dist[:, 1], ind[:, 1]
    near = nn_dist <= np.percentile(nn_dist, 5)   # behaviourally near-identical pairs
    dC = np.abs(C[near] - C[nn_idx[near]])
    print(f"  Among behaviourally near-identical pairs (signature distance, bottom 5%):")
    print(f"    median |delta centralization| : {np.median(dC):.2f}  (scale 0-1)")
    print(f"    90th percentile               : {np.percentile(dC, 90):.2f}")

    hline("CONDITIONAL IDENTIFIABILITY OF CENTRALIZATION (poles vs interior)")
    clean = (T < 0.33) & (latents["Deceptiveness"] < 0.33)
    interior = (T > 0.5) | (latents["Deceptiveness"] > 0.5)
    rc_clean, sc_clean = cv_r2(S[clean], C[clean], rf())
    rc_int, sc_int = cv_r2(S[interior], C[interior], rf())
    print(f"  clean extremes (low templating & low deception):  "
          f"R2 = {rc_clean:.3f} +/- {sc_clean:.3f}")
    print(f"  confounded interior (high templating OR deception): "
          f"R2 = {rc_int:.3f} +/- {sc_int:.3f}")

    hline("ROBUSTNESS (independent draws per scenario)")
    print(f"{'scenario':30s} {'Cent.':>7s} {'Agency':>7s} {'Intent':>7s} {'Decept':>7s}")
    scenarios = [("baseline", 1.0, 0.0),
                 ("weak confounder", 0.4, 0.0),
                 ("strong confounder", 1.8, 0.0),
                 ("deception given a footprint", 1.0, 0.30)]

    def holdout_r2(S, y):
        n = len(y); cut = int(0.7 * n)
        m = RandomForestRegressor(n_estimators=100, max_depth=12,
                                  n_jobs=-1, random_state=0)
        m.fit(S[:cut], y[:cut])
        return r2_score(y[cut:], m.predict(S[cut:]))

    for i, (label, cm, dfp) in enumerate(scenarios):
        g = np.random.default_rng(SEED + 101 * (i + 1))
        Sx, lat_x, _, _ = generate(g, N, conf_mult=cm, deception_footprint=dfp)
        perm = g.permutation(N)
        Sx, lat_x = Sx[perm], {kk: vv[perm] for kk, vv in lat_x.items()}
        vals = [holdout_r2(Sx, lat_x[d])
                for d in ("Centralization", "Participant agency",
                          "Intentionality", "Deceptiveness")]
        print(f"{label:30s} {vals[0]:7.3f} {vals[1]:7.3f} {vals[2]:7.3f} {vals[3]:7.3f}")

    if args.out:
        import csv
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dimension", "r2_rf", "r2_rf_sd", "r2_gbm",
                        "r2_gbm_sd", "mi_nats", "mi_equiv_r2"])
            for name, vals in rows.items():
                w.writerow([name, *[f"{v:.4f}" for v in vals]])
        print(f"\nWrote {args.out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
