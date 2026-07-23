"""Tek ajan sarmalayicisi — PLAN §Asama 3.

MARLGridEnv'in FAZ A'sini tek basina oynatir: A1 hedefe varinca episode biter,
yasak bolge / A2 hic devreye girmez.

Gozlem MARL kurulumuyla BIREBIR AYNI (129 boyut, ayni kanallar). Bu bilincli:
Asama 4-5'te ayni ag mimarisi ve ayni gozlem borusu kullanilacak, dolayisiyla
Asama 3'te dogruladigin makine oldugu gibi tasiniyor. s2 kanali burada anlamsiz
bir sinyal — agin onu gormezden gelmeyi ogrenmesi de bir dogrulama.
"""
from typing import Optional

import numpy as np

from baselines.bfs_oracle import manhattan
from config import AGENT_1, DIRS, N_ACTIONS
from env.grid_env import MARLGridEnv

Cell = tuple[int, int]


class SingleAgentEnv:
    def __init__(self, seed: Optional[int] = None, **kwargs):
        self.env = MARLGridEnv(seed=seed, **kwargs)

    # ------------------------------------------------------------- ortam API

    def reset(self, config: Optional[tuple[Cell, Cell, Cell]] = None,
              seed: Optional[int] = None) -> np.ndarray:
        obs = self.env.reset(config=config, seed=seed)
        return obs[AGENT_1]

    def action_mask(self) -> np.ndarray:
        """Sadece duvar maskesi — tek ajan modunda yasak bolge yok.

        Bilerek MARLGridEnv.action_mask'i cagirmiyoruz: o, episode bitince
        yalniz NOOP'u aciyor. Kesilme (timeout) durumunda bootstrap icin
        gecerli bir maskeye ihtiyacimiz var, degenere NOOP maskesine degil.
        """
        e = self.env
        mask = np.zeros(N_ACTIONS, dtype=np.float32)
        r, c = e.pos[AGENT_1]
        for a, (dr, dc) in enumerate(DIRS):
            if 0 <= r + dr < e.n and 0 <= c + dc < e.n:
                mask[a] = 1.0
        return mask

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        obs, _, _, info = self.env.step(action)
        e = self.env
        reached = e.pos[AGENT_1] == e.goal
        done = reached or e.done          # e.done: timeout ya da faz sonu

        # Takim odulu DEGIL, A1'in kendi odulu: adim maliyeti + gecersiz hamle
        # cezasi + hedef odulu. Kilitleme/takim terimleri Asama 5'e ait.
        reward = info["r_ind"][AGENT_1]

        # KESILME (truncation) ile GERCEK terminali ayir. Zaman limitinde
        # episode biter ama durumun degeri 0 DEGILDIR; bootstrap edilmeli.
        out = {"reached": reached, "truncated": bool(done and not reached)}
        if done:
            len1 = len(e.path[AGENT_1]) - 1
            opt1 = manhattan(e.s1, e.goal)
            out.update({
                "len1": len1, "opt1": opt1, "gap1": len1 - opt1,
                "timeout": not reached,
                "config": (e.s1, e.s2, e.goal),
                "path1": tuple(e.path[AGENT_1]),
            })
        return obs[AGENT_1], reward, done, out

    def render(self) -> str:
        return self.env.render()

    # ---------------------------------------------------------- yardimcilar

    @property
    def s1(self) -> Cell:
        return self.env.s1

    @property
    def goal(self) -> Cell:
        return self.env.goal


def all_start_goal_pairs(n: int = 5) -> list[tuple[Cell, Cell]]:
    """25*24 = 600 (start, goal) cifti. Asama 3'un eval seti — orneklem degil,
    problem uzayinin TAMAMI."""
    cells = [(r, c) for r in range(n) for c in range(n)]
    return [(s, g) for s in cells for g in cells if s != g]
