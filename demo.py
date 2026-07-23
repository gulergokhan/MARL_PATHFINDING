"""Ortamin gozle dogrulanmasi: ayni konfig, iki farkli A1 yolu.

Calistir:  .venv\\Scripts\\python.exe demo.py
"""
from baselines.bfs_oracle import (all_shortest_paths, bfs_dist, bfs_path,
                                  forbidden_from, oracle)
from baselines.policies import _path_to_actions
from config import DOWN, RIGHT
from env.grid_env import MARLGridEnv


def play(env, config, a1_actions, baslik):
    env.reset(config=config)
    for a in a1_actions:                       # FAZ A: A1'in yolu scriptli
        _, _, done, info = env.step(a)
        if done:
            break
    print(f"\n### {baslik}")
    print(env.render())
    if info.get("blocked"):
        print("-> A2 KILITLENDI, episode faz sinirinda bitti.")
        print(f"   kacinilabilir miydi? {'HAYIR' if info['block_unavoidable'] else 'EVET'}")
        return
    # FAZ B: A2 BFS ile gider
    for a in _path_to_actions(bfs_path(env.s2, env.goal, env.forbidden)):
        _, _, done, info = env.step(a)
    print(f"-> basari={info['success']}  len1={info['len1']}  len2={info['len2']}"
          f"  (oracle={info['oracle_len2']}, serbest={info['free_len2']})"
          f"  zarar={info['harmed']}")


env = MARLGridEnv(seed=0)
cfg = ((0, 0), (0, 0), (4, 4))
orc = oracle(*cfg)
print(f"Konfig s1={cfg[0]} s2={cfg[1]} goal={cfg[2]}")
print(f"A1'in optimal yol sayisi: {orc.n_paths}, bunlardan {orc.n_blocking} tanesi A2'yi kilitliyor")

play(env, cfg, [DOWN, DOWN, DOWN, RIGHT, RIGHT, RIGHT, RIGHT, DOWN],
     "BENCIL A1: ilk hamle ASAGI, son hamle ASAGI -> kilitler")
play(env, cfg, [DOWN, DOWN, DOWN, DOWN, RIGHT, RIGHT, RIGHT, RIGHT],
     "KIBAR A1: ilk hamle ASAGI, son hamle SAG -> A2'ye yer birakir")

# Ayni start olmayan, A2'yi UZATAN bir ornek (asil sinyal bu)
print("\n" + "=" * 52)
cfg2 = ((0, 0), (1, 0), (2, 2))
orc2 = oracle(*cfg2)
print(f"Konfig s1={cfg2[0]} s2={cfg2[1]} goal={cfg2[2]}  "
      f"({orc2.n_paths} yol, {orc2.n_detour} tanesi A2'yi uzatiyor)")
for p in all_shortest_paths(cfg2[0], cfg2[2]):
    d = bfs_dist(cfg2[1], cfg2[2], forbidden_from(p, *cfg2))
    if d is not None and d > orc2.free_len2:
        play(env, cfg2, _path_to_actions(p), f"A1 yolu {p} -> A2 uzuyor")
        break
