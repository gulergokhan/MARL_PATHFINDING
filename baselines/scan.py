"""Tum (s1, s2, goal) konfigurasyonlarini tara — PLAN §0 ve §Asama 2.

Cikti:
  runs/feasibility.csv  — her konfig icin oracle sonucu
  runs/difficulty.csv   — curriculum sampler'in kullanacagi zorluk etiketleri

Calistir:  .venv\\Scripts\\python.exe -m baselines.scan
"""
import csv
import os
import time
from itertools import product

from baselines.bfs_oracle import manhattan, oracle
from config import DIFFICULTY_CSV, FEASIBILITY_CSV, GRID_N, RUNS_DIR

# PLAN §0'da BFS ile olculen beklenen degerler. Kod bunlari yeniden uretmeli.
EXPECTED = {
    "total": 15_625,
    "trivial": 1_225,
    "evaluated": 14_400,
    "same_start": 600,
    "a2_optimal": 14_120,
    "a2_detour": 280,
    "a2_dead": 0,
    "hard": 4_200,
}


def scan():
    cells = [(r, c) for r in range(GRID_N) for c in range(GRID_N)]
    rows = []
    t0 = time.time()

    trivial = same_start = 0
    a2_optimal = a2_detour = a2_dead = 0
    hard = 0
    detour_only_unique_path = 0
    block_rate_sum = 0.0          # konfig-agirlikli
    path_blocking = path_total = 0  # yol-agirlikli

    for s1, s2, g in product(cells, cells, cells):
        if s1 == g or s2 == g:
            trivial += 1
            continue
        if s1 == s2:
            same_start += 1

        orc = oracle(s1, s2, g)
        free2 = manhattan(s2, g)

        if orc.best_len2 is None:
            a2_dead += 1
            klass = "dead"
        elif orc.best_len2 == free2:
            a2_optimal += 1
            klass = "optimal"
        else:
            a2_detour += 1
            klass = "detour"
            if orc.n_paths == 1:
                detour_only_unique_path += 1

        if orc.is_hard:
            hard += 1
        block_rate_sum += orc.block_rate
        path_blocking += orc.n_blocking
        path_total += orc.n_paths

        rows.append({
            "s1_r": s1[0], "s1_c": s1[1],
            "s2_r": s2[0], "s2_c": s2[1],
            "g_r": g[0], "g_c": g[1],
            "d1": orc.len1,
            "free_len2": free2,
            "oracle_len2": orc.best_len2 if orc.best_len2 is not None else -1,
            "n_paths": orc.n_paths,
            "n_blocking": orc.n_blocking,
            "n_detour": orc.n_detour,
            "block_rate": round(orc.block_rate, 6),
            "class": klass,
            "difficulty": "hard" if orc.is_hard else "easy",
            "same_start": int(s1 == s2),
        })

    evaluated = len(rows)
    elapsed = time.time() - t0

    os.makedirs(RUNS_DIR, exist_ok=True)
    with open(FEASIBILITY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(DIFFICULTY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["s1_r", "s1_c", "s2_r", "s2_c",
                                          "g_r", "g_c", "d1", "difficulty",
                                          "block_rate", "oracle_len2"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})

    # ---------------------------------------------------------------- rapor
    print(f"=== 5x5 tam tarama ({elapsed:.1f}s) ===\n")
    print(f"Toplam (s1,s2,g)              : {trivial + evaluated}")
    print(f"  trivial (start==goal)       : {trivial}")
    print(f"  degerlendirilen             : {evaluated}")
    print(f"  ayni start (s1==s2)         : {same_start}")
    print()
    print("--- A1 EN IYI yolunu secerse A2'nin durumu (PLAN §0.2) ---")
    print(f"  A2 optimal                  : {a2_optimal:5d}  ({100*a2_optimal/evaluated:5.1f}%)")
    print(f"  A2 uzuyor (+2)              : {a2_detour:5d}  ({100*a2_detour/evaluated:5.1f}%)")
    print(f"  A2 HIC gidemiyor            : {a2_dead:5d}  ({100*a2_dead/evaluated:5.1f}%)")
    print(f"  -> uzayanlarin kaci tek-yollu: {detour_only_unique_path}/{a2_detour}"
          f"  ({100*detour_only_unique_path/max(a2_detour,1):.0f}%)")
    print()
    print("--- Koordinasyon ne kadar gerekli (PLAN §0.3) ---")
    print(f"  ZOR konfig (A1'in secimi onemli) : {hard:5d}  ({100*hard/evaluated:5.1f}%)")
    print(f"  kilitleme orani, YOL-agirlikli   : {100*path_blocking/path_total:5.2f}%"
          f"   ({path_blocking}/{path_total})")
    print(f"  kilitleme orani, KONFIG-agirlikli: {100*block_rate_sum/evaluated:5.2f}%"
          f"   <-- ortamin random-shortest baseline hedefi")
    print()
    print(f"yazildi: {FEASIBILITY_CSV}, {DIFFICULTY_CSV}")

    # ---------------------------------------------------------------- dogrulama
    got = {
        "total": trivial + evaluated, "trivial": trivial, "evaluated": evaluated,
        "same_start": same_start, "a2_optimal": a2_optimal,
        "a2_detour": a2_detour, "a2_dead": a2_dead, "hard": hard,
    }
    bad = {k: (v, EXPECTED[k]) for k, v in got.items() if v != EXPECTED[k]}
    print("\n=== PLAN §0 ile karsilastirma ===")
    if bad:
        for k, (v, e) in bad.items():
            print(f"  UYUSMUYOR {k}: {v} != beklenen {e}")
        raise SystemExit("Tarama PLAN §0 ile uyusmuyor — oracle'da bug var.")
    assert detour_only_unique_path == a2_detour, \
        "Uzayan konfiglerin hepsi tek-yollu olmaliydi (PLAN §0.2)"
    print("  TUM SAYILAR TUTUYOR ✓")
    return {"config_weighted_block_rate": block_rate_sum / evaluated,
            "path_weighted_block_rate": path_blocking / path_total}


if __name__ == "__main__":
    scan()
