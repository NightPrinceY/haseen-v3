"""How separable are the varieties? (2026-10-14)

Experiment A, controlled: Alexandria only (same pipeline, same domains, 13 countries). Train on its train split,
test on its public test split. Source and domain are held constant, so confusions are dialect confusions.
Experiment B, cross-source: train on all human (non-synthetic) text except Casablanca / Habibi, test on the
spoken transcripts (Casablanca 8 countries, Habibi 5 + MSA). Measures generalization to speech.
Model: char 1-5-gram TF-IDF (word-boundary aware) + linear SVM, balanced classes, capped per variety.
Output: data/inventory/separability.json (per-class F1, confusion matrices, groups from average linkage)."""
import gzip, json, os, random, collections, sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from scipy.cluster.hierarchy import linkage, fcluster

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "HaSeen-V1", "src"))
from haseen.text import norm  # noqa: E402

INV = os.path.join(os.path.dirname(__file__), "..", "data", "inventory")
random.seed(13)
rows = [json.loads(l) for l in gzip.open(os.path.join(INV, "sentences.jsonl.gz"), "rt")]
clean = lambda t: " ".join(norm(w) for w in t.split())


def run(train, test, labels, name, cap=6000):
    by = collections.defaultdict(list)
    for r in train:
        by[r["var"]].append(r)
    tr = [x for v in labels for x in random.sample(by[v], min(cap, len(by[v])))]
    te = [r for r in test if r["var"] in labels]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), min_df=2, sublinear_tf=True, max_features=400000)
    X = vec.fit_transform([clean(r["text"]) for r in tr])
    clf = LinearSVC(C=0.5, class_weight="balanced").fit(X, [r["var"] for r in tr])
    pred = clf.predict(vec.transform([clean(r["text"]) for r in te]))
    gold = [r["var"] for r in te]
    rep = classification_report(gold, pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(gold, pred, labels=labels)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    # symmetric confusion -> distance -> groups
    sym = (cmn + cmn.T) / 2
    np.fill_diagonal(sym, 0)
    dist = 1 - sym / max(sym.max(), 1e-9)
    iu = np.triu_indices(len(labels), 1)
    Z = linkage(dist[iu], "average")
    groups = {str(k): {str(t): [labels[i] for i in np.where(fcluster(Z, k, "maxclust") == t)[0]]
                       for t in set(fcluster(Z, k, "maxclust"))} for k in (4, 6, 8, 10, 12)}
    top_conf = sorted(((float(cmn[i, j]), labels[i], labels[j]) for i in range(len(labels)) for j in range(len(labels))
                       if i != j), reverse=True)[:20]
    print(f"\n== {name}: train {len(tr)} test {len(te)} macro-F1 {rep['macro avg']['f1-score']:.3f} acc {rep['accuracy']:.3f}")
    for v in labels:
        print(f"  {v:4s} F1 {rep[v]['f1-score']:.3f}  P {rep[v]['precision']:.3f} R {rep[v]['recall']:.3f}  n={int(rep[v]['support'])}")
    print("  top confusions (gold -> pred, row share):", ", ".join(f"{a}->{b} {x:.2f}" for x, a, b in top_conf[:12]))
    return {"train": len(tr), "test": len(te), "macro_f1": rep["macro avg"]["f1-score"], "accuracy": rep["accuracy"],
            "per_class": {v: {k: rep[v][k] for k in ("precision", "recall", "f1-score", "support")} for v in labels},
            "labels": labels, "confusion": cm.tolist(), "top_confusions": top_conf, "groups": groups}


out = {}
alex = [r for r in rows if r["source"] == "alexandria"]
labs = sorted({r["var"] for r in alex})
out["A_alexandria_country"] = run([r for r in alex if r["split"] in ("train", "dev")],
                                  [r for r in alex if r["split"] == "test"], labs, "A: Alexandria, 13 countries")

human = [r for r in rows if r["kind"] != "synthetic" and r["source"] not in ("casablanca", "habibi")
         and r["source"] != "algerian_darja" and len(r["var"]) <= 3 and r["var"] not in ("LEV", "GLF", "MAG")]
spoken = [r for r in rows if r["source"] in ("casablanca", "habibi")]
labs_b = sorted({r["var"] for r in spoken} & {r["var"] for r in human})
out["B_cross_source_spoken"] = run(human, spoken, labs_b, "B: all written human text -> spoken transcripts")
labs_c = sorted(v for v, n in collections.Counter(r["var"] for r in human).items() if n >= 800)
test_c = [r for r in human if r["source"] == "alexandria" and r["split"] == "test"]
train_c = [r for r in human if not (r["source"] == "alexandria" and r["split"] == "test")]
out["C_all_human_countries"] = run(train_c, test_c + spoken, labs_c, "C: all human text, all countries with >=800 sentences")
json.dump(out, open(os.path.join(INV, "separability.json"), "w"), indent=1, default=float)
