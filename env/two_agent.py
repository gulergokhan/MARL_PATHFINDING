"""IQL (bagimsiz Q-learning) icin ortak episode calistiricisi — PLAN §Asama 4.

Egitim (train=True) ve degerlendirme (train=False, eps=0) AYNI fonksiyonu
kullanir; boylece iki kod yolu birbirinden sapmaz (Asama 3'te egitim/eval
tutarsizligi bir hata kaynagiydi, burada bastan onleniyor).
"""
from typing import Optional

from config import AGENT_1, AGENT_2
from env.grid_env import MARLGridEnv

Cell = tuple[int, int]


def play_episode(env: MARLGridEnv, agents: dict, train: bool,
                 config: Optional[tuple[Cell, Cell, Cell]] = None) -> tuple[dict, dict]:
    """Bir episode'u sonuna kadar oynat.

    agents = {AGENT_1: DQNAgent, AGENT_2: DQNAgent} — HER AJAN KENDI push()/
    learn()'unu SADECE kendi aktif oldugu adimlarda gorur (golge NOOP yok,
    IQL'in tanimi geregi PLAN §Asama 4).

    Donen: (terminal_info, losses). losses = {AGENT_1: [...], AGENT_2: [...]}
    (train=False ise ikisi de bos).
    """
    obs = env.reset(config=config)
    done = False
    losses = {AGENT_1: [], AGENT_2: []}
    info: dict = {}

    while not done:
        agent = env.active
        mask = env.action_mask(agent)
        eps = None if train else 0.0
        a = agents[agent].act(obs[agent], mask, eps=eps)

        prev_active = env.active
        next_obs, _r_team, done, info = env.step(a)

        if train:
            r = info["r_ind"][agent]
            # Bu ajanin KENDI gorevi bitti mi? Ya butun episode bitti (done)
            # ya da faz degisip aktif ajan degisti (A1 basariyla Faz B'ye
            # devretti — kendi episode'u burada biter, bir daha hic oynamaz).
            own_done = done or (env.active != prev_active)
            # done VE timeout ayni anda ise bu bir KESILME (truncation):
            # gercek terminal degil, deger 0 degildir, bootstrap gerekir.
            # (info["timeout"] sadece done=True'da set edilir; agent zaten
            # o an aktif oldugu icin baska ajanin timeout'uyla karismaz.)
            is_truncated = done and info.get("timeout", False)
            push_done = own_done and not is_truncated
            next_mask = (env.action_mask(agent) if push_done
                        else env.physical_mask(agent))
            agents[agent].push(obs[agent], a, r, next_obs[agent], push_done, next_mask)
            loss = agents[agent].learn()
            if loss is not None:
                losses[agent].append(loss)

        obs = next_obs

    return info, losses
