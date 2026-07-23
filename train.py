"""Egitim giris noktasi.

Asama 3:  .venv\\Scripts\\python.exe train.py --algo dqn
"""
import argparse
import csv
import os
import sys
import time
from collections import deque

import numpy as np

# Windows'ta stdout bir dosyaya/boruya yonlendirilince (>, |) konsol degil
# sistem ANSI codepage'i (cp1252) kullaniliyor; Turkce karakterler orada
# gecersiz. UTF-8'e sabitleyip bu siniftaki tum betiklerde tekrarini onle.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from agents.dqn import DQNAgent
from config import DQN_EPISODES, DQN_EVAL_EVERY, RUNS_DIR, SEED
from env.single_agent import SingleAgentEnv, all_start_goal_pairs


# --------------------------------------------------------------------- eval

def evaluate_dqn(agent: DQNAgent, env: SingleAgentEnv) -> dict:
    """600 (start, goal) ciftinin TAMAMINDA deterministik greedy degerlendirme.

    Orneklem degil tam tarama: "gap 0.0" iddiasi boylece kanit olur.
    """
    gaps, fails = [], []
    for s, g in all_start_goal_pairs():
        obs = env.reset(config=(s, s, g))      # s2 = s1 (tek ajan icin ilgisiz)
        done = False
        while not done:
            a = agent.act(obs, env.action_mask(), eps=0.0)
            obs, _, done, info = env.step(a)
        if not info["reached"] or info["gap1"] != 0:
            fails.append((s, g, info["gap1"] if info["reached"] else "timeout"))
        gaps.append(info["gap1"] if info["reached"] else None)

    ok = [x for x in gaps if x is not None]
    return {
        "n": len(gaps),
        "reached": len(ok),
        "mean_gap": float(np.mean(ok)) if ok else float("nan"),
        "optimal_frac": sum(x == 0 for x in ok) / len(gaps),
        "fails": fails,
    }


# -------------------------------------------------------------------- train

def train_dqn(episodes: int = DQN_EPISODES, seed: int = SEED,
              log_every: int = 1_000, eval_every: int = DQN_EVAL_EVERY) -> DQNAgent:
    env = SingleAgentEnv(seed=seed)
    agent = DQNAgent(seed=seed)

    os.makedirs(f"{RUNS_DIR}/ckpt", exist_ok=True)
    log_path = f"{RUNS_DIR}/dqn_train_log.csv"
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    logger = csv.DictWriter(log_file, fieldnames=[
        "episode", "steps", "eps", "return", "len", "gap", "loss",
        "eval_optimal_frac", "eval_mean_gap"])
    logger.writeheader()

    ret_w, gap_w, len_w, loss_w = (deque(maxlen=500) for _ in range(4))
    t0 = time.time()

    for ep in range(1, episodes + 1):
        obs, done, ep_ret = env.reset(), False, 0.0
        while not done:
            mask = env.action_mask()
            a = agent.act(obs, mask)
            next_obs, r, done, info = env.step(a)
            next_mask = env.action_mask()
            # Zaman limitinde episode biter ama bootstrap DEVAM etmeli.
            # done=True yazmak "15. adimda hedefe uzaksan degerin 0" demektir;
            # gercek deger ~0.7'dir ve bu hata tam da ajanin kayboldugu
            # durumlara enjekte edilir. Klasik time-limit bootstrapping tuzagi.
            agent.push(obs, a, r, next_obs, done and not info["truncated"], next_mask)
            loss = agent.learn()
            if loss is not None:
                loss_w.append(loss)
            obs, ep_ret = next_obs, ep_ret + r

        ret_w.append(ep_ret)
        len_w.append(info["len1"])
        gap_w.append(info["gap1"] if info["reached"] else np.nan)

        row = None
        if ep % eval_every == 0:
            ev = evaluate_dqn(agent, env)
            row = {"eval_optimal_frac": round(ev["optimal_frac"], 4),
                   "eval_mean_gap": round(ev["mean_gap"], 4)}
            print(f"  [ep {ep:6d}] EVAL 600 cift: optimal %{100*ev['optimal_frac']:.1f}"
                  f"  ort.gap {ev['mean_gap']:+.3f}"
                  f"  basarisiz {len(ev['fails'])}", flush=True)

        if ep % log_every == 0 or row:
            mean_gap = float(np.nanmean(gap_w)) if len(gap_w) else float("nan")
            entry = {
                "episode": ep, "steps": agent.steps, "eps": round(agent.eps, 4),
                "return": round(float(np.mean(ret_w)), 3),
                "len": round(float(np.mean(len_w)), 3),
                "gap": round(mean_gap, 4),
                "loss": round(float(np.mean(loss_w)), 5) if loss_w else "",
                "eval_optimal_frac": "", "eval_mean_gap": "",
            }
            entry.update(row or {})
            logger.writerow(entry)
            log_file.flush()
            if row is None:
                print(f"  ep {ep:6d} | eps {agent.eps:.3f} | odul {entry['return']:+.2f}"
                      f" | uzunluk {entry['len']:.2f} | gap {mean_gap:+.3f}"
                      f" | loss {entry['loss']}", flush=True)

    log_file.close()
    ckpt = f"{RUNS_DIR}/ckpt/dqn.pt"
    agent.save(ckpt)

    print(f"\nEgitim bitti: {time.time()-t0:.0f}s, {agent.steps} adim -> {ckpt}")
    ev = evaluate_dqn(agent, env)
    print("\n=== ASAMA 3 KABUL KRITERI ===")
    print(f"  600 (start, goal) ciftinin tamami, deterministik greedy (eps=0)")
    print(f"  hedefe ulasan        : {ev['reached']}/{ev['n']}")
    print(f"  optimal (gap == 0)   : {100*ev['optimal_frac']:.2f}%")
    print(f"  ortalama gap         : {ev['mean_gap']:+.4f}   (hedef: 0.0000)")
    if ev["fails"]:
        print(f"  BASARISIZ {len(ev['fails'])} cift, ilk 10: {ev['fails'][:10]}")
        print("  -> KABUL KRITERI GECMEDI, Asama 4'e gecme.")
    else:
        print("  TUM 600 CIFT OPTIMAL  BFS ile birebir ayni. Asama 4'e gecilebilir.")
    return agent


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="dqn", choices=["dqn"])
    p.add_argument("--episodes", type=int, default=DQN_EPISODES)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    print(f"=== {args.algo.upper()} egitimi | {args.episodes} episode | seed {args.seed} ===")
    train_dqn(episodes=args.episodes, seed=args.seed)
