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
from config import (AGENT_1, AGENT_2, DIFFICULTY_CSV, DQN_EPISODES,
                    DQN_EVAL_EVERY, IQL_BATCH, IQL_BUFFER, IQL_EPISODES,
                    IQL_EPS_DECAY_STEPS, IQL_EVAL_EVERY, IQL_LEARN_START,
                    IQL_LR, IQL_TARGET_UPDATE, RUNS_DIR, SEED)
from env.grid_env import MARLGridEnv
from env.single_agent import SingleAgentEnv, all_start_goal_pairs
from env.two_agent import play_episode


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


# ==================================================================== IQL

def load_all_configs() -> tuple[list, list]:
    """difficulty.csv'den (Asama 2) TUM 14.400 konfig + zorluk etiketini oku.

    Bagimsiz uretilmis bu dosyayi kullanmak, IQL'in Asama 4 kabul kriterinin
    Asama 2'nin dogruluk zeminiyle AYNI konfig uzayina karsi olculmesini
    garanti eder.
    """
    if not os.path.exists(DIFFICULTY_CSV):
        from baselines.scan import scan
        scan()
    configs, difficulty = [], []
    with open(DIFFICULTY_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            s1 = (int(row["s1_r"]), int(row["s1_c"]))
            s2 = (int(row["s2_r"]), int(row["s2_c"]))
            g = (int(row["g_r"]), int(row["g_c"]))
            configs.append((s1, s2, g))
            difficulty.append(row["difficulty"])
    return configs, difficulty


def evaluate_iql(agents: dict, env: MARLGridEnv, configs: list,
                 difficulty: list | None = None) -> dict:
    """Verilen konfig listesinde deterministik greedy (eps=0) degerlendirme.

    PLAN §0.3: ana metrik "kilitleme" degil "zarar" (kilit VEYA uzama) —
    kilitleme konfig-agirlikli sadece %0.82'de, 2000 eval'de ~1 tane gorulur.
    """
    n = len(configs)
    reached1 = reached2 = blocked = detoured = harmed = gap1_bad = 0
    gap2_sum = 0.0
    hard_harmed = hard_n = 0

    for i, cfg in enumerate(configs):
        info, _ = play_episode(env, agents, train=False, config=cfg)
        if info["gap1"] is not None:
            reached1 += 1
            gap1_bad += info["gap1"] != 0
        if info["gap2"] is not None:
            reached2 += 1
            gap2_sum += info["gap2"]
        blocked += info["blocked"]
        detoured += bool(info.get("detoured"))
        harmed += bool(info.get("harmed"))
        if difficulty is not None and difficulty[i] == "hard":
            hard_n += 1
            hard_harmed += bool(info.get("harmed"))

    return {
        "n": n,
        "reached1_frac": reached1 / n,
        "reached2_frac": reached2 / n,
        "gap1_bad": gap1_bad,
        "mean_gap2": gap2_sum / max(reached2, 1),
        "block_rate": blocked / n,
        "detour_rate": detoured / n,
        "harm_rate": harmed / n,
        "hard_harm_rate": (hard_harmed / hard_n) if hard_n else float("nan"),
        "hard_n": hard_n,
    }


def _make_iql_agent(seed: int) -> DQNAgent:
    return DQNAgent(seed=seed, buffer_size=IQL_BUFFER, batch_size=IQL_BATCH,
                    lr=IQL_LR, eps_decay_steps=IQL_EPS_DECAY_STEPS,
                    learn_start=IQL_LEARN_START, target_update=IQL_TARGET_UPDATE)


def train_iql(episodes: int = IQL_EPISODES, seed: int = SEED,
              eval_every: int = IQL_EVAL_EVERY,
              quick_eval_n: int = 1_500) -> dict:
    """Iki bagimsiz DQN, ortak odul YOK. PLAN §Asama 4.

    grid_env.py'deki info["r_ind"] zaten sadece step-cost + kendi hedef
    bonusunu iceriyor; kilitleme/takim/optimallik cezalari (r_team'e ait)
    hic karismiyor — bu IQL'in "ortak odul yok" tanimini otomatik saglar.
    """
    env = MARLGridEnv(seed=seed)
    agents = {AGENT_1: _make_iql_agent(seed), AGENT_2: _make_iql_agent(seed + 1)}

    all_configs, all_difficulty = load_all_configs()
    rng = np.random.default_rng(seed + 999)
    quick_idx = rng.choice(len(all_configs), size=quick_eval_n, replace=False)
    quick_configs = [all_configs[i] for i in quick_idx]
    quick_difficulty = [all_difficulty[i] for i in quick_idx]

    os.makedirs(f"{RUNS_DIR}/ckpt", exist_ok=True)
    log_path = f"{RUNS_DIR}/iql_train_log.csv"
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    logger = csv.DictWriter(log_file, fieldnames=[
        "episode", "steps1", "steps2", "eps1", "eps2", "eval_gap1_bad",
        "eval_mean_gap2", "eval_block_rate", "eval_detour_rate",
        "eval_harm_rate", "eval_hard_harm_rate"])
    logger.writeheader()

    t0 = time.time()
    for ep in range(1, episodes + 1):
        play_episode(env, agents, train=True)

        if ep % eval_every == 0 or ep == episodes:
            ev = evaluate_iql(agents, env, quick_configs, quick_difficulty)
            row = {
                "episode": ep,
                "steps1": agents[AGENT_1].steps, "steps2": agents[AGENT_2].steps,
                "eps1": round(agents[AGENT_1].eps, 4),
                "eps2": round(agents[AGENT_2].eps, 4),
                "eval_gap1_bad": ev["gap1_bad"],
                "eval_mean_gap2": round(ev["mean_gap2"], 4),
                "eval_block_rate": round(ev["block_rate"], 4),
                "eval_detour_rate": round(ev["detour_rate"], 4),
                "eval_harm_rate": round(ev["harm_rate"], 4),
                "eval_hard_harm_rate": round(ev["hard_harm_rate"], 4),
            }
            logger.writerow(row)
            log_file.flush()
            print(f"  [ep {ep:6d}] A1-kotu {ev['gap1_bad']}/{quick_eval_n}"
                  f"  A2-gap {ev['mean_gap2']:+.3f}"
                  f"  kilit %{100*ev['block_rate']:.2f}"
                  f"  zarar %{100*ev['harm_rate']:.2f}"
                  f"  zarar(zor) %{100*ev['hard_harm_rate']:.2f}", flush=True)

    log_file.close()
    agents[AGENT_1].save(f"{RUNS_DIR}/ckpt/iql_agent1.pt")
    agents[AGENT_2].save(f"{RUNS_DIR}/ckpt/iql_agent2.pt")
    print(f"\nEgitim bitti: {time.time()-t0:.0f}s -> runs/ckpt/iql_agent{{1,2}}.pt")

    print("\n=== ASAMA 4 KABUL KRITERI (TAM 14.400 konfig) ===")
    ev = evaluate_iql(agents, env, all_configs, all_difficulty)
    print(f"  A1 optimal degil (gap1!=0)   : {ev['gap1_bad']}/{ev['n']}  (hedef: 0)")
    print(f"  A2 ortalama gap (ORACLE'a gore): {ev['mean_gap2']:+.4f}  (hedef: ~0.0)")
    print(f"  kilitleme orani               : %{100*ev['block_rate']:.2f}"
          f"   (random-shortest baseline: %0.82)")
    print(f"  ZARAR orani (genel)           : %{100*ev['harm_rate']:.2f}"
          f"   (random-shortest baseline: %13.28)")
    print(f"  ZARAR orani (zor alt-kume, n={ev['hard_n']}): %{100*ev['hard_harm_rate']:.2f}"
          f"   <-- VDN ile karsilastirilacak asil sayi")
    return {"agents": agents, "eval": ev}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="dqn", choices=["dqn", "iql"])
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    if args.algo == "dqn":
        episodes = args.episodes or DQN_EPISODES
        print(f"=== DQN egitimi | {episodes} episode | seed {args.seed} ===")
        train_dqn(episodes=episodes, seed=args.seed)
    else:
        episodes = args.episodes or IQL_EPISODES
        print(f"=== IQL egitimi | {episodes} episode | seed {args.seed} ===")
        train_iql(episodes=episodes, seed=args.seed)
