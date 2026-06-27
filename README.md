# The Coordination–Consensus Continuum — identifiability simulation

Reproducible Monte-Carlo simulation behind the identifiability result in
*The Coordination–Consensus Continuum: A Conceptual Framework and a Formal
Identifiability Result for the Attribution Problem in Social Listening*
(A.-G. Adam, working paper).

## What it does

It generates synthetic "convergence" episodes whose true position on four latent
attribution dimensions — intentionality, control centralization, participant
agency, deceptiveness — is known by construction, renders an observable
behavioral signature from them (plus an exogenous organic-templating
confounder), and measures how much of each latent dimension can be recovered
from that signature out-of-sample. The result: the structural dimensions are
non-identified from behavior, the signature is dominated by the environmental
factor, and the recovery of centralization collapses in the hybrid interior.

Recovery is assessed two independent ways — cross-validated out-of-sample R²
from two learner classes (random forest, gradient boosting), and a Kraskov
mutual-information bound — so the conclusion does not hinge on one estimator.

## Status

This is a **constructive demonstration, not an empirical measurement.** The
magnitudes follow from the model's assumptions; the robustness block shows the
qualitative conclusion is stable across confounder strength and even when
deception is granted a behavioral footprint it does not have by definition.
Real-world magnitudes must be estimated from data, not assumed.

## Run

```
pip install -r requirements.txt
python coordination_identifiability.py
```

Optional: `python coordination_identifiability.py --out results.csv` writes the
recovery table to CSV.

## Paper

https://www.researchgate.net/publication/405503551_The_Coordination-Consensus_Continuum_A_Conceptual_Framework_for_the_Attribution_Problem_in_Social_Listening

## License

MIT — see [LICENSE](LICENSE).
