"""Sirali (turn-based) iki ajanli grid ortami — PLAN §1 ve §Asama 1.

Akis:

    t = 0 ................ T1        T1+1 ................ T
        |<--- FAZ A: A1 ---->|        |<---- FAZ B: A2 ---->|
         A1 hareket eder                A2 hareket eder
         A2 NOOP (golge)                A1 NOOP (golge)
                            ^
                            +-- yasak bolge B burada SABITLENIR

Golge NOOP (PLAN §2.2): sirasi gelmeyen ajan da her adimda gozlem uretir ve
NOOP basar; Q degeri VDN toplamina dahil olur. A2'nin gozlemi FAZ A boyunca
guncellenmeye devam eder (A1 ilerledikce yasak bolge buyur) — VDN'in kredi
atamasi tam olarak bu kanaldan calisir.
"""
from typing import Optional

import numpy as np

from baselines.bfs_oracle import bfs_dist, forbidden_from, manhattan, oracle
from config import (AGENT_1, AGENT_2, ALLOW_SAME_START, DIRS, GRID_N,
                    MAX_STEPS_PER_PHASE, MAX_STEPS_TOTAL, NOOP, N_ACTIONS,
                    OBS_DIM, R_AGENT_GOAL, R_BLOCKED, R_BOTH_GOAL, R_INVALID,
                    R_OPT_GAP, R_STEP, R_TIMEOUT, STATE_DIM)

Cell = tuple[int, int]


class MARLGridEnv:
    """Sirali iki fazli grid ortami.

    step() TAKIM odulunu (tek skaler) dondurur — VDN/QMIX boyle ister.
    Ajan basina odul IQL baseline'i icin info["r_ind"] icinde ayrica verilir.
    """

    def __init__(self, n: int = GRID_N,
                 max_steps_per_phase: int = MAX_STEPS_PER_PHASE,
                 allow_same_start: bool = ALLOW_SAME_START,
                 seed: Optional[int] = None):
        self.n = n
        self.max_steps_per_phase = max_steps_per_phase
        self.max_steps_total = 2 * max_steps_per_phase
        self.allow_same_start = allow_same_start
        self.rng = np.random.default_rng(seed)
        self._cells = [(r, c) for r in range(n) for c in range(n)]
        self.reset()

    # ------------------------------------------------------------- konfig

    def sample_config(self) -> tuple[Cell, Cell, Cell]:
        """Rastgele (s1, s2, goal). s1 == s2 serbest; start == goal degil."""
        while True:
            i, j, k = self.rng.integers(0, len(self._cells), size=3)
            s1, s2, g = self._cells[i], self._cells[j], self._cells[k]
            if s1 == g or s2 == g:
                continue
            if not self.allow_same_start and s1 == s2:
                continue
            return s1, s2, g

    # -------------------------------------------------------------- reset

    def reset(self, config: Optional[tuple[Cell, Cell, Cell]] = None,
              seed: Optional[int] = None) -> dict[int, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.s1, self.s2, self.goal = config if config is not None else self.sample_config()

        self.pos = {AGENT_1: self.s1, AGENT_2: self.s2}
        self.path = {AGENT_1: [self.s1], AGENT_2: [self.s2]}
        self.visited = {AGENT_1: {self.s1}, AGENT_2: {self.s2}}

        # Muaf hucreler — PLAN §1'in kritik kurali
        self.exempt = frozenset({self.s1, self.s2, self.goal})

        self.phase = 0            # 0 = FAZ A (A1), 1 = FAZ B (A2)
        self.t = 0                # global adim
        self.phase_t = 0          # icinde bulunulan fazdaki adim
        self.forbidden = frozenset()   # faz sinirinda sabitlenir
        self.done = False
        self.invalid_count = {AGENT_1: 0, AGENT_2: 0}
        self._blocked = False
        self._timeout = False
        return self.observations()

    # ------------------------------------------------------- yardimcilar

    @property
    def active(self) -> int:
        """Sirasi gelen ajan."""
        return AGENT_1 if self.phase == 0 else AGENT_2

    @property
    def passive(self) -> int:
        return AGENT_2 if self.phase == 0 else AGENT_1

    def forbidden_view(self) -> frozenset:
        """Yasak bolgenin O ANKI hali.

        FAZ A'da A1'in su ana kadarki izi (buyumeye devam ediyor), FAZ B'de
        sabitlenmis kume. Tek formul: iz eksi muaf hucreler.
        """
        return frozenset(self.visited[AGENT_1]) - self.exempt

    def _in_bounds(self, c: Cell) -> bool:
        return 0 <= c[0] < self.n and 0 <= c[1] < self.n

    def physical_mask(self, agent: int) -> np.ndarray:
        """Sadece FIZIKSEL gecerlilik: duvar + (A2 icin) yasak bolge.

        done/active durumunu HIC dikkate almaz. Egitim dongusunde bir ajanin
        kendi episode'u zaman asimiyla (timeout) bitince bootstrap icin
        next_state'in GERCEK maskesi lazim — action_mask() o anda sadece
        NOOP dondurur (asagida), bu da bootstrap'i bozar (bkz. Asama 3'teki
        SingleAgentEnv ile ayni tuzak). Bu metod o acigi kapatir.
        """
        mask = np.zeros(N_ACTIONS, dtype=np.float32)
        forb = self.forbidden if agent == AGENT_2 else frozenset()
        cur = self.pos[agent]
        for a, (dr, dc) in enumerate(DIRS):
            nxt = (cur[0] + dr, cur[1] + dc)
            if self._in_bounds(nxt) and nxt not in forb:
                mask[a] = 1.0
        if mask.sum() == 0:          # tamamen kapali kalirsa (olmamali) NOOP ac
            mask[NOOP] = 1.0
        return mask

    def action_mask(self, agent: int) -> np.ndarray:
        """Politika secimi icin: pasif/bitmis ajanda SADECE NOOP."""
        if self.done or agent != self.active:
            mask = np.zeros(N_ACTIONS, dtype=np.float32)
            mask[NOOP] = 1.0
            return mask
        return self.physical_mask(agent)

    # ------------------------------------------------------------ gozlem

    def observe(self, agent: int) -> np.ndarray:
        n = self.n
        ch = np.zeros((5, n, n), dtype=np.float32)
        own, other = self.pos[agent], self.pos[1 - agent]
        ch[0][own] = 1.0                      # kendi konumu
        ch[1][other] = 1.0                    # diger ajanin konumu
        ch[2][self.goal] = 1.0                # ortak hedef
        for c in self.forbidden_view():       # yasak bolge (FAZ A'da buyuyor)
            ch[3][c] = 1.0
        for c in self.visited[agent]:         # kendi izi
            ch[4][c] = 1.0

        max_man = 2 * (n - 1)
        scalars = np.array([
            float(agent),
            float(self.phase),
            self.t / self.max_steps_total,
            manhattan(own, self.goal) / max_man,
        ], dtype=np.float32)
        return np.concatenate([ch.ravel(), scalars])

    def observations(self) -> dict[int, np.ndarray]:
        return {AGENT_1: self.observe(AGENT_1), AGENT_2: self.observe(AGENT_2)}

    def state(self) -> np.ndarray:
        """QMIX mixer icin merkezi global state."""
        n = self.n
        ch = np.zeros((4, n, n), dtype=np.float32)
        ch[0][self.pos[AGENT_1]] = 1.0
        ch[1][self.pos[AGENT_2]] = 1.0
        ch[2][self.goal] = 1.0
        for c in self.forbidden_view():
            ch[3][c] = 1.0
        scalars = np.array([float(self.phase), self.t / self.max_steps_total],
                           dtype=np.float32)
        return np.concatenate([ch.ravel(), scalars])

    # -------------------------------------------------------------- step

    def step(self, actions) -> tuple[dict[int, np.ndarray], float, bool, dict]:
        """actions: {AGENT_1: a1, AGENT_2: a2} veya tek int (aktif ajanin aksiyonu).

        Pasif ajanin aksiyonu ne verilirse verilsin NOOP'a zorlanir.
        Donen odul TAKIM odulu (tek skaler).
        """
        if self.done:
            raise RuntimeError("Episode bitti — reset() cagir.")

        agent = self.active
        a = actions if isinstance(actions, (int, np.integer)) else actions[agent]
        a = int(a)

        r_team = R_STEP
        r_ind = {AGENT_1: 0.0, AGENT_2: 0.0}
        r_ind[agent] += R_STEP

        # --- hareket
        moved_to = None
        if a == NOOP:
            # aktif ajan icin NOOP maskeli; yine de gelirse gecersiz say
            r_team += R_INVALID
            r_ind[agent] += R_INVALID
            self.invalid_count[agent] += 1
        else:
            dr, dc = DIRS[a]
            cur = self.pos[agent]
            nxt = (cur[0] + dr, cur[1] + dc)
            forb = self.forbidden if agent == AGENT_2 else frozenset()
            if not self._in_bounds(nxt) or nxt in forb:
                r_team += R_INVALID              # duvar / yasak hucre: yerinde kal
                r_ind[agent] += R_INVALID
                self.invalid_count[agent] += 1
            else:
                moved_to = nxt

        if moved_to is not None:
            self.pos[agent] = moved_to
            self.path[agent].append(moved_to)
            self.visited[agent].add(moved_to)

        self.t += 1
        self.phase_t += 1
        info: dict = {"phase": self.phase, "active": agent}

        # --- terminal kontrolleri
        if self.phase == 0 and self.pos[AGENT_1] == self.goal:
            r_team += R_AGENT_GOAL
            r_ind[AGENT_1] += R_AGENT_GOAL
            r_team += self._close_phase_a(info)
        elif self.phase == 1 and self.pos[AGENT_2] == self.goal:
            r_team += R_AGENT_GOAL + R_BOTH_GOAL
            r_ind[AGENT_2] += R_AGENT_GOAL
            r_team += self._finish(info)
        elif self.phase_t >= self.max_steps_per_phase:
            r_team += R_TIMEOUT
            self._timeout = True
            self.done = True

        if self.done:
            info.update(self._terminal_info())
        info["r_ind"] = r_ind
        return self.observations(), float(r_team), self.done, info

    def _close_phase_a(self, info: dict) -> float:
        """A1 hedefe vardi: yasak bolgeyi sabitle, A2'nin fizibilitesini BFS ile kontrol et.

        Kilitliyse episode BURADA biter (PLAN §2.2 erken sonlandirma) — ceza
        A1'in hamlelerine gamma^T1 uzaklikta kalir, gamma^(T1+T2) degil.
        """
        self.forbidden = forbidden_from(tuple(self.path[AGENT_1]),
                                        self.s1, self.s2, self.goal)
        d2 = bfs_dist(self.s2, self.goal, self.forbidden, self.n)
        info["forbidden_size"] = len(self.forbidden)
        if d2 is None:
            self._blocked = True
            self.done = True
            return R_BLOCKED
        self.phase = 1
        self.phase_t = 0
        return 0.0

    def _finish(self, info: dict) -> float:
        """A2 hedefe vardi: optimallik cezasini ORACLE'a gore yaz (PLAN §5 kural 1)."""
        self.done = True
        orc = oracle(self.s1, self.s2, self.goal)
        len2 = len(self.path[AGENT_2]) - 1
        # ORACLE'a gore olc, serbest Manhattan'a gore DEGIL: 280 konfigde A1'in
        # alternatifi yok, oralarda serbest mesafeye gore ceza yazmak ajani
        # imkansiz bir sey icin cezalandirir.
        gap2 = max(0, len2 - (orc.best_len2 if orc.best_len2 is not None else len2))
        return R_OPT_GAP * gap2

    def _terminal_info(self) -> dict:
        orc = oracle(self.s1, self.s2, self.goal)
        len1 = len(self.path[AGENT_1]) - 1
        len2 = len(self.path[AGENT_2]) - 1
        reached1 = self.pos[AGENT_1] == self.goal
        reached2 = self.pos[AGENT_2] == self.goal
        return {
            "config": (self.s1, self.s2, self.goal),
            "success": bool(reached1 and reached2),
            "blocked": self._blocked,
            # A1 optimal oynasaydi da kilitlenir miydi? False ise suc A1'in.
            "block_unavoidable": self._blocked and orc.best_len2 is None,
            "timeout": self._timeout,
            "len1": len1, "len2": len2 if reached2 else None,
            "gap1": len1 - orc.len1 if reached1 else None,
            "gap2": (len2 - orc.best_len2) if (reached2 and orc.best_len2 is not None) else None,
            # A2 serbest optimumundan sapti mi? Kilitlemeden cok daha SIK olan
            # ve asil ogrenme sinyalini tasiyan olcut (bkz. PLAN §0.3).
            "detoured": bool(reached2 and len2 > orc.free_len2),
            "harmed": bool(self._blocked or (reached2 and len2 > orc.free_len2)),
            "oracle_len1": orc.len1,
            "oracle_len2": orc.best_len2,
            "free_len2": orc.free_len2,
            "is_hard": orc.is_hard,
            "invalid": dict(self.invalid_count),
            "path1": tuple(self.path[AGENT_1]),
            "path2": tuple(self.path[AGENT_2]),
        }

    # ------------------------------------------------------------ render

    def render(self) -> str:
        forb = self.forbidden_view()
        rows = []
        for r in range(self.n):
            row = []
            for c in range(self.n):
                cell = (r, c)
                if self.pos[AGENT_1] == cell and self.pos[AGENT_2] == cell:
                    row.append("*")
                elif self.pos[AGENT_1] == cell:
                    row.append("1")
                elif self.pos[AGENT_2] == cell:
                    row.append("2")
                elif cell == self.goal:
                    row.append("G")
                elif cell in forb:
                    row.append("#")
                else:
                    row.append(".")
            rows.append(" ".join(row))
        head = (f"faz={'A' if self.phase == 0 else 'B'} t={self.t} "
                f"aktif=A{self.active + 1} yasak={len(forb)}")
        return head + "\n" + "\n".join(rows)
