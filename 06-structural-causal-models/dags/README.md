# DAGs, d-separation and do-calculus, as code

The graphical tools of the SCM lectures written as small, readable Python functions, and a notebook that checks every
claim by simulating the corresponding structural causal model.

## The problem

Which variables should we adjust for? When is a causal effect identifiable at all? And what can observations,
experiments and counterfactuals each tell us? The lectures answer by reading graphs; here the reading is automated and
then verified numerically.

## Notes: graphical identification in five lines

1. Association flows along open paths. A path is blocked by a conditioned mediator or confounder, or by a collider
   that is not conditioned on (nor any of its descendants).
2. d-separated ⇒ conditionally independent.
3. Backdoor: $Z$ blocks every path into $T$ and has no descendant of $T$ ⇒ $P(Y|do(t)) = \sum_z P(Y|t,z)P(z)$.
4. Do-calculus: three rules, each a d-separation test in a mutilated graph ($G_{\bar X}$ cuts edges into $X$,
   $G_{\underline X}$ edges out of $X$), complete for identification.
5. Rungs of the ladder are separated: same observations ≠ same interventions; same interventions ≠ same
   counterfactuals (PNS is only bounded by experiments).

## Start here

[`notebooks/dags_explained.ipynb`](notebooks/dags_explained.ipynb):

1. mediator, confounder, collider: correlations before and after conditioning;
2. $E[Y|T]$ vs $E[Y|do(T)]$ vs the backdoor formula on the hospital;
3. paths and their status on a 10-node graph for several conditioning sets;
4. all 256 adjustment sets regressed: valid backdoor sets land on the true effect, invalid ones (including "adjust for
   everything") do not;
5. the three rules of do-calculus on the front-door graph, and the front-door formula vs a simulated intervention;
6. two SCMs with identical observational distributions and different interventional ones;
7. PN, PS, PNS for two SCMs with identical experiments, and the Tian–Pearl bounds.

## Layout

```
dags/
├── src/
│   ├── dag.py       DAG: paths, blocking, d-separation, backdoor sets, do / cut_outgoing, drawing
│   ├── docalc.py    rule1, rule2, rule3
│   └── scm.py       SCM: sample, do, counterfactual; PN/PS/PNS and PNS bounds
├── notebooks/
│   └── dags_explained.ipynb
└── requirements.txt
```

Path enumeration is exponential in the worst case; that is fine for the lecture-sized graphs here and keeps the code a
direct transcription of the definitions.

## Run

```bash
cd 06-structural-causal-models/dags
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/dags_explained.ipynb
```
