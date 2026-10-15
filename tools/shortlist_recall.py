"""DINOv2 shortlist recall@k over every field image of the 200 registered animals, plus recovery after a second capture."""
import sys, json, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from eval.dataset import make_split, build_gallery, load_feat
s = make_split(); g = build_gallery(s)
ranks, misses = [], []
for a in s.registered:
    for im in s.field_imgs[a]:
        q = load_feat(a, im)
        short = g.shortlist(q.embedding, k=8)
        ids = [x[0] for x in short]
        r = ids.index(a) + 1 if a in ids else 99
        ranks.append(r)
        if r > 2: misses.append({"query": im, "animal": a, "rank": r, "top2": ids[:2]})
ranks = np.array(ranks)
out = {"queries": int(len(ranks)), "animals": len(s.registered), "gallery_images": len(g.features),
       **{f"recall@{k}": round(float((ranks <= k).mean()), 4) for k in (1, 2, 3, 5, 8)},
       "misses_top2": misses}
print(json.dumps(out, indent=1))
json.dump(out, open('eval/out/shortlist_recall.json', 'w'), indent=1)
# Recapture recovery: a miss becomes "new", the policy asks for a straight-on re-look, identify runs on both.
rec = []
for m in misses:
    a = m["animal"]; q1 = load_feat(a, m["query"])
    for im in s.field_imgs[a]:
        if im == m["query"]: continue
        e = q1.embedding + load_feat(a, im).embedding; e /= np.linalg.norm(e)
        ids = [x[0] for x in g.shortlist(e, k=8)]
        rec.append(ids.index(a) + 1 if a in ids else 99)
rec = np.array(rec)
print("recapture pairs", len(rec), "recall@2", round(float((rec <= 2).mean()), 4), "recall@5", round(float((rec <= 5).mean()), 4))
