"""
Tune D2_POOL_SIM: how similar two photos must look before the container treats them as the
same casualty. Embeds training photos with the assembled ensemble's first whole-image member,
clusters them at several thresholds, and scores the clusters against the true casualty ids.

    python scripts/tune_pool.py --images I --groups-file work/groups.csv --weights work/weights [--n 200]

Pick the highest-recall threshold whose pair precision stays >= ~0.95: a wrong merge spreads one
casualty's injuries onto another, which costs more than a missed merge.
Note: the embedding member was trained on these photos, so clusters here are somewhat cleaner
than on unseen test photos; prefer a slightly stricter threshold than the best one here.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from d2qual.data import load_image  # noqa: E402
from d2qual.predict import Ensemble, _select_device, cluster  # noqa: E402
from d2qual.train import load_groups  # noqa: E402


def pair_scores(pred: list[int], true: list[str]):
    tp = fp = fn = 0
    for i in range(len(pred)):
        for j in range(i + 1, len(pred)):
            same_p, same_t = pred[i] == pred[j], true[i] == true[j]
            tp += same_p and same_t
            fp += same_p and not same_t
            fn += same_t and not same_p
    return tp / max(1, tp + fp), tp / max(1, tp + fn)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--groups-file", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--n", type=int, default=200, help="photos to sample (test set is ~190)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    ids = sorted(p.name for p in Path(a.images).iterdir() if p.is_file())
    groups = load_groups(ids, None, a.groups_file)
    ids = [i for i in ids if not groups[i].startswith("__single__")]
    random.Random(a.seed).shuffle(ids)
    ids = ids[: a.n]
    ens = Ensemble(Path(a.weights), _select_device())
    emb = np.concatenate([ens.embed([load_image(Path(a.images) / i) for i in ids[k:k + 16]])
                          for k in range(0, len(ids), 16)])
    true = [groups[i] for i in ids]
    print(f"{len(ids)} photos, {len(set(true))} casualties")
    print("sim   clusters  pair_precision  pair_recall")
    for sim in (0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6):
        pred = cluster(emb, sim)
        p, r = pair_scores(pred, true)
        print(f"{sim:.2f}  {len(set(pred)):8d}  {p:14.3f}  {r:11.3f}")


if __name__ == "__main__":
    main()
