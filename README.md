# MARL Pathfinding

5x5 gridde iki ajanlı sıralı yol bulma, Multi-Agent RL (IQL / VDN / QMIX).

Her episode'da `start1`, `start2` ve **ortak** `goal` rastgele seçilir.
Önce Ajan 1 hedefe gider; geçtiği hücreler **yasak bölge** olur; sonra Ajan 2
o bölgeye girmeden aynı hedefe gider. Ajan 1 kendi optimalliğinden ödün
vermeden Ajan 2'ye yer bırakmayı öğrenmelidir.

Tam plan, ölçülmüş istatistikler ve aşama aşama yol haritası: **[PLAN.md](PLAN.md)**

## Kurulum

```bash
python3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Çalıştırma

Tüm komutlar proje kökünden çalıştırılır.

Tam konfigürasyon taraması (14.400 konfig, ~1 sn) — `runs/*.csv` üretir:

```bash
.venv\Scripts\python.exe -m baselines.scan
```

Testler:

```bash
.venv\Scripts\python.exe -m tests.test_oracle
```

```bash
.venv\Scripts\python.exe -m tests.test_env
```

## Durum

| Aşama | Durum |
|---|---|
| 0 Kurulum | ✅ |
| 1 Ortam `env/grid_env.py` | ✅ |
| 2 BFS oracle `baselines/` | ✅ |
| 3 Tek ajan DQN | 🔲 |
| 4 IQL baseline | 🔲 |
| 5 VDN | 🔲 |
| 6 Curriculum | 🔲 |
| 7 QMIX | 🔲 |
| 8 Değerlendirme | 🔲 |
| 9 Görselleştirme | 🔲 |

## Ölçülmüş temel sayılar

Tam tarama (`baselines/scan.py`) ve 20.000 episode simülasyonu ile doğrulandı:

| | Değer |
|---|---:|
| Değerlendirilen konfigürasyon | 14.400 |
| Çözümsüz konfigürasyon | **0** |
| A2'nin optimal gidebildiği | %98.1 |
| A2'nin +2 uzamak zorunda kaldığı | %1.9 (hepsi A1'in tek yollu olduğu durumlar) |
| "Zor" konfig (A1'in seçimi önemli) | %29.2 |
| **Random-shortest baseline: A2 zarar oranı** | **%13.3** ← ana metrik |
| Random-shortest baseline: kilitleme | %0.82 |
| Random-shortest baseline: başarı | %99.16 |

## Ortam API'si

```python
from env.grid_env import MARLGridEnv

env = MARLGridEnv(seed=0)
obs = env.reset()                      # {0: (129,), 1: (129,)}
obs, r_team, done, info = env.step(action)   # r_team TEK skaler (VDN icin)
print(env.render())
```

- `env.action_mask(agent)` — aktif ajanda `NOOP` kapalı, pasif ajanda sadece `NOOP`
- `env.state()` — QMIX mixer için merkezi global state (102,)
- `info["r_ind"]` — IQL baseline'ı için ajan başına ödül
- `info` (terminal) — `success`, `blocked`, `harmed`, `detoured`, `gap1`, `gap2`,
  `oracle_len2`, `is_hard`, `path1`, `path2`
