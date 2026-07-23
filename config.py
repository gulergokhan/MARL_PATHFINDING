"""Tum hiperparametreler ve sabitler burada. Koda deger gomme.

PLAN.md ile senkron tut: ozellikle odul degerleri (§5) ve max_steps (§Asama 1).
"""

# ---------------------------------------------------------------- grid / ortam
GRID_N = 5
N_AGENTS = 2

AGENT_1 = 0          # once giden, izi yasak bolge olan ajan
AGENT_2 = 1          # yasak bolgeden kacinarak giden ajan

# Aksiyonlar
UP, RIGHT, DOWN, LEFT, NOOP = range(5)
N_ACTIONS = 5
ACTION_NAMES = ("UP", "RIGHT", "DOWN", "LEFT", "NOOP")

# (d_satir, d_sutun) — NOOP dahil degil, o ayri ele aliniyor
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))

# Faz basina adim limiti. 5x5'te en uzun optimal yol 8; 15 fazlasiyla yeter.
MAX_STEPS_PER_PHASE = 15
MAX_STEPS_TOTAL = 2 * MAX_STEPS_PER_PHASE

# Ajanlar ayni hucreden baslayabilir mi (PLAN §0.1: 600 konfig)
ALLOW_SAME_START = True

# ---------------------------------------------------------------- odul (PLAN §5)
R_STEP = -0.05        # her timestep
R_INVALID = -0.10     # duvara / yasak hucreye hamle (ajan yerinde kalir)
R_AGENT_GOAL = +1.0   # bir ajan hedefe vardi (ara terminal)
R_BOTH_GOAL = +3.0    # ikisi de vardi (takim hedefi)
R_BLOCKED = -3.0      # A2 kilitlendi — faz sinirinda BFS ile tespit
R_TIMEOUT = -3.0      # herhangi bir fazda max_steps asildi
# Katsayi: R_OPT_GAP * (len2 - ORACLE(s1,s2,g)).
# Tarama sonucu (baselines/scan.py): random-shortest A1 altinda A2'nin
# KILITLENME olasiligi sadece %0.82, ama UZAMA olasiligi %13.3. Yani asil
# ogrenme sinyalini bu terim tasiyor, R_BLOCKED degil. Tipik uzama 2 adim ->
# -1.0 ceza, +3.0'lik takim bonusuna karsi anlamli bir agirlik.
R_OPT_GAP = -0.50

# ---------------------------------------------------------------- egitim (ortak)
SEED = 0
GAMMA = 0.99
LR = 5e-4
GRAD_CLIP = 10.0
HIDDEN = 128

# ---------------------------------------------------------------- MARL (Asama 4+)
EPS_START, EPS_END, EPS_DECAY_STEPS = 1.0, 0.05, 50_000
BUFFER_EPISODES = 5_000
BATCH_EPISODES = 32
TARGET_UPDATE_EVERY = 200      # episode
TOTAL_EPISODES = 100_000

# ---------------------------------------------------------------- DQN (Asama 3)
# Tek ajan, yasak bolge yok. Amac: MARL'a gecmeden once DQN makinesinin
# dogrulugunu kanitlamak. Episode'lar cok kisa (ortalama ~3.3 adim), o yuzden
# epsilon ADIM sayisina gore soner, episode sayisina gore degil.
DQN_EPISODES = 30_000
DQN_BUFFER = 100_000           # transition
DQN_BATCH = 128                # transition
DQN_EPS_DECAY_STEPS = 30_000   # adim
DQN_LEARN_START = 1_000        # bu kadar transition birikmeden ogrenme
DQN_LR = 1e-4                  # genel LR'den (5e-4) daha dusuk: Q-divergence onlemi
DQN_TARGET_UPDATE = 2_000       # adim (500 -> 2000: moving-target/divergence onlemi)
DQN_EVAL_EVERY = 2_000         # episode

# Curriculum (PLAN §Asama 6): zor konfig orani 0.2 -> 0.8
P_HARD_START, P_HARD_END, P_HARD_CAP = 0.20, 0.80, 0.80

# ---------------------------------------------------------------- gozlem boyutu
OBS_CHANNELS = 5               # own, other, goal, forbidden, own_visited
N_SCALARS = 4                  # agent_id, phase, t/max, kalan_manhattan/max
OBS_DIM = OBS_CHANNELS * GRID_N * GRID_N + N_SCALARS      # 129

STATE_CHANNELS = 4             # pos1, pos2, goal, forbidden
STATE_SCALARS = 2              # phase, t/max
STATE_DIM = STATE_CHANNELS * GRID_N * GRID_N + STATE_SCALARS   # 102

# ---------------------------------------------------------------- yollar
RUNS_DIR = "runs"
FEASIBILITY_CSV = "runs/feasibility.csv"
DIFFICULTY_CSV = "runs/difficulty.csv"
