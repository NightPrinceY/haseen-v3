"""Gold v3.3: remove colloquial MGB-2 sentences from the MSA sets (natural + challenge hosts), dev and test alike.

Why: build_test_v3.py meant to drop colloquial MGB-2 sentences (Al Jazeera talk shows), but its filter let through
sentences such as «مصر هي اللي صدرت الإسلام» and «زي ما قلت من شوية»; the gold then counts their dialect words as
conforming MSA. Found on dev (model false alarms), confirmed by reading every MGB-2 sentence of the MSA natural
dev and test sets (Claude, 2026-10-14) BEFORE any test scoring and without looking at model outputs for test.
Rule applied to every sentence: drop it if it holds at least one clearly colloquial word or phrase (اللي, زي, مش /
ما كنش / ما فيش, عم + verb, بدي / عايز, نحنا / احنا, عنا 'we have', شغلات, حتة, دا / بقى, colloquial numbers like
خمستاشر / تلاتة / تسعماية / في المية, verb plurals without nun such as يقولوا / يروحوا). Borderline formal speech
(والله, يعني, ولا في) is kept. The drop lists are by index over the MGB-2 sentences in file order."""
import hashlib, json, os

T3 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "test_v3")
DROP = {
    "dev": [12, 15, 16, 26, 31, 32, 34, 62, 89, 91, 93, 94, 113, 117, 120, 139, 142, 146, 147, 154, 159, 164, 176, 179],
    "test": [12, 14, 22, 27, 38, 45, 54, 56, 67, 74, 78, 81, 83, 86, 88, 93, 102, 124, 132, 142, 150, 154, 162, 163,
             165, 176, 177, 179, 184, 197, 207, 217, 228, 257, 259, 270, 280, 292, 304, 316, 335, 341, 344, 345, 349,
             365],
}
log = {}
for sp, idx in DROP.items():
    p = f"{T3}/natural_MSA_{sp}.jsonl"
    xs = [json.loads(l) for l in open(p)]
    mg = [x for x in xs if x["source"] == "mgb2"]
    gone = [mg[i] for i in idx]
    gone_ids = {x["id"] for x in gone}
    keep = [x for x in xs if x["id"] not in gone_ids]
    with open(p, "w") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in keep)
    # challenge hosts: same length and identical outside the swapped positions
    pc = f"{T3}/challenge_MSA_{sp}.jsonl"
    cs = [json.loads(l) for l in open(pc)]
    def host_of(c):
        sw = set(c.get("swapped", []))
        for g in gone:
            if len(g["tokens"]) == len(c["tokens"]) and all(a == b or i in sw for i, (a, b) in
                                                            enumerate(zip(g["tokens"], c["tokens"]))):
                return True
        return False
    ckeep = [c for c in cs if not (c["source"] == "mgb2" and host_of(c))]
    with open(pc, "w") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in ckeep)
    log[sp] = {"natural_dropped": len(gone), "natural_left": len(keep), "challenge_dropped": len(cs) - len(ckeep),
               "challenge_left": len(ckeep), "challenge_devs_left": sum(sum(c["dev"]) for c in ckeep),
               "dropped_text": [" ".join(g["tokens"]) for g in gone]}
    print(sp, {k: v for k, v in log[sp].items() if k != "dropped_text"})
json.dump(log, open(f"{T3}/review/msa_mgb2_review_v33.json", "w"), ensure_ascii=False, indent=1)
