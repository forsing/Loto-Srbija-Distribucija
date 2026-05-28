"""
9 Distributions

  → izbor prave verovatnoće za izbor odgovarajuće distribucije
    (upravo ovaj princip se koristi u Koraku 4 za NEXT predikciju)

Slika_1  → Distribucije.png       (tabela 9 distribucija: Normal, Bernoulli,
                                   Binomial, Geometric, Poisson, Exponential,
                                   Gamma, Beta, Uniform)

sve se računa nad CSV-om svih izvlačenja: loto7_4622_k42.csv.


recency-weighted beta-bernoulli posterior + recency boost
Per-broj P sad nije monotono u count-u: P(k) = (α + count_k + 5·recent_k) / (α + β + n_draws + 5·RECENT_W). Brojevi sa istim ukupnim count-om ali različitim brojem pojava u poslednjih 50 izvlačenja dobijaju različito P → ranking više nije puka frekvencija.
TOP_POOL = 13, enumeracija je sad C(13, 7) = 1716 kombinacija.
Display + TXT pokazuju Top 13 (ne 15).
Top 13 sad se razlikuje od raw count-a.
"""

import os
import time
from datetime import timedelta
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import warnings
warnings.filterwarnings('ignore')


T0 = time.time()

SEED = 42
np.random.seed(SEED)

CSV_PATH = "/Users/4c/Desktop/GHQ/data/loto7_4622_k42.csv"
HERE = os.path.dirname(os.path.abspath(__file__))
PNG_PATH = os.path.join(HERE, "distribucija_1.png")
TXT_PATH = os.path.join(HERE, "distribucija_1.txt")

POOL = 39       # opseg brojeva 1..39
DRAWS = 7       # 7 brojeva po izvlačenju
TARGET = 7      # fiksirani broj za Bernoulli / Geometric / Poisson signale
W = 20          # prozor (Poisson)
K_GAMMA = 5     # broj k za Gamma (vreme do k-te pojave)
RECENT_W = 50   # prozor za "recent" signal u koraku 4
RECENCY_GAMMA = 5  # težina recent prozora u Beta-Bernoulli posterioru
                   # (recent draw broji kao γ "običnih" izvlačenja)


# ═════════════════════════════════════════════════════════════════════
# Step 1 / KORAK 1: Dataset
# ═════════════════════════════════════════════════════════════════════
rows = []
with open(CSV_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) < DRAWS:
            continue
        try:
            rows.append(tuple(int(x) for x in parts[:DRAWS]))
        except ValueError:
            continue

n_draws = len(rows)
arr = np.array(rows, dtype=int)          # (n_draws, 7)
all_nums = arr.flatten()                  # svi izvučeni brojevi (flat)

print()
print(f"Učitano izvlačenja: {n_draws}  (CSV: {os.path.basename(CSV_PATH)})")
print(f"Opseg: 1..{POOL}   po izvlačenju: {DRAWS}")
print()


# ═════════════════════════════════════════════════════════════════════
# Step 2 / KORAK 2: Izvedeni feature-i za 9 distribucija
#   Svaki signal je prirodno povezan sa jednom teorijskom distribucijom
# ═════════════════════════════════════════════════════════════════════

# 1) NORMAL — suma 7 brojeva (CLT: zbir nezavisnih ~U(1..39) → Normal, μ ≈ 7·20 = 140)
sums = arr.sum(axis=1)

# 2) BERNOULLI — pojavio se TARGET u izvlačenju? (1/0)
contains_target = (arr == TARGET).any(axis=1).astype(int)

# 3) BINOMIAL — broj parnih u izvlačenju (n=7, p ≈ 19/39 ≈ 0.487)
parity_even = (arr % 2 == 0).sum(axis=1)

# 4) GEOMETRIC — gap između susednih pojava TARGET broja (broj izvlačenja do sled. pojave)
appearances = np.where(contains_target == 1)[0]
gaps_geom = np.diff(appearances) if len(appearances) >= 2 else np.array([1])

# 5) POISSON — broj pojava TARGET u kliznim prozorima dužine W
window_counts = np.array([
    contains_target[i:i+W].sum() for i in range(0, n_draws - W + 1)
])

# 6) EXPONENTIAL — gapovi između pojava bilo kog broja (svi gapovi 1..39 spojeni)
exp_gaps = []
for num in range(1, POOL + 1):
    idx = np.where((arr == num).any(axis=1))[0]
    if len(idx) >= 2:
        exp_gaps.extend(np.diff(idx).tolist())
exp_gaps = np.array(exp_gaps, dtype=float)

# 7) GAMMA — kumulativan broj izvlačenja do K-te pojave istog broja (sve brojeve agregiramo)
gamma_vals = []
for num in range(1, POOL + 1):
    idx = np.where((arr == num).any(axis=1))[0]
    for j in range(K_GAMMA, len(idx)):
        gamma_vals.append(idx[j] - idx[j - K_GAMMA])
gamma_vals = np.array(gamma_vals, dtype=float)

# 8) BETA — empirijske frekvencije svakog broja kao udeli u [0,1]
counts_per_num = np.array([(all_nums == k).sum() for k in range(1, POOL + 1)])
total_picks = counts_per_num.sum()
beta_freqs = counts_per_num / total_picks   # 39 vrednosti

# 9) UNIFORM — svi izvučeni brojevi 1..39 (treba da budu uniformni)
uniform_nums = all_nums.astype(float)


# Sažetak po izvlačenju
df = pd.DataFrame({
    "suma_7":  sums,
    f"ima_{TARGET}": contains_target,
    "parnih": parity_even,
})
print("Sažetak (prve kolone po izvlačenju):")
print(df.head())
print(f"\nShape: {df.shape}\n")
print("Basic Stats:\n", df.describe().round(2))
print()


# ═════════════════════════════════════════════════════════════════════
# Step 2b / KORAK 2b: Visualising Every Distribution
#   3x3 grid histograma 
#   Loto CSV-u → distribucija_1.png)
# ═════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle(f"Loto 7/39 — 9 distribucija nad CSV-om (n={n_draws})",
             fontsize=15, fontweight='bold')

panels = [
    (sums,             "Normal — Suma 7 brojeva",                       "steelblue",       40),
    (contains_target,  f"Bernoulli — Pojavio se broj {TARGET} (0/1)",   "coral",            2),
    (parity_even,      "Binomial — Broj parnih u izvlačenju (n=7)",     "mediumseagreen",   8),
    (gaps_geom,        f"Geometric — Gap do sled. pojave {TARGET}",     "orchid",          30),
    (window_counts,    f"Poisson — # pojava {TARGET} u prozoru {W}",    "tomato",          12),
    (exp_gaps,         "Exponential — Gap između pojava (svi brojevi)", "goldenrod",       40),
    (gamma_vals,       f"Gamma — Vreme do {K_GAMMA}. pojave",           "teal",            40),
    (beta_freqs,       "Beta — Empirijske frekvencije brojeva (1..39)", "slateblue",       20),
    (uniform_nums,     "Uniform — Svi izvučeni brojevi (1..39)",        "darkorange",    POOL),
]
for ax, (data, title, color, bins) in zip(axes.flatten(), panels):
    ax.hist(data, bins=bins, color=color, alpha=0.85, edgecolor='white')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_ylabel("Frequency")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(PNG_PATH, dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot saved → {PNG_PATH}\n")


# ═════════════════════════════════════════════════════════════════════
# Step 3 / KORAK 3: Fitting Distributions to Confirm Our Data
#   KS test (continuous: Normal/Exponential/Gamma/Beta) i chi-square
#   (discrete: Bernoulli/Binomial/Geometric/Poisson/Uniform).
#     p > 0.05 → NE odbacuje se H0 (distribucija fituje) → POGODNA
#     p ≤ 0.05 → odbacuje se → NEPOGODNA
# ═════════════════════════════════════════════════════════════════════
def kept(p):
    if not isinstance(p, float) or np.isnan(p):
        return "N/A"
    return "POGODNA" if p > 0.05 else "NEPOGODNA"


def safe_chisq(obs, exp, min_exp=5.0):
    """Chi-square sa kombinovanjem retkih bin-ova i rescale-om expected na sumu obs."""
    obs = np.asarray(obs, dtype=float)
    exp = np.asarray(exp, dtype=float)
    mask = exp >= min_exp
    if mask.sum() < 2:
        return np.nan
    o = obs[mask]
    e = exp[mask]
    # Rescale expected da suma odgovara observed (zahtev scipy.chisquare)
    if e.sum() > 0:
        e = e * (o.sum() / e.sum())
    _, p = stats.chisquare(o, e)
    return float(p)


test_results = []  # (Distribucija, Loto feature, Param, p-value, Status)

# 1) NORMAL — KS test na sumama
mu, sigma = stats.norm.fit(sums)
_, p_norm = stats.kstest(sums, "norm", args=(mu, sigma))
test_results.append(("Normal", "suma_7", f"μ={mu:.2f}, σ={sigma:.2f}", float(p_norm), kept(float(p_norm))))

# 2) BERNOULLI — chi-square (dva ishoda 0/1)
p_hat_b = float(contains_target.mean())
exp_bern = np.array([(1 - p_hat_b), p_hat_b]) * n_draws
obs_bern = np.array([(contains_target == 0).sum(), (contains_target == 1).sum()])
p_bern = safe_chisq(obs_bern, exp_bern, min_exp=1.0)
test_results.append(("Bernoulli", f"ima_{TARGET}", f"p̂={p_hat_b:.3f}", p_bern, kept(p_bern)))

# 3) BINOMIAL — chi-square (n=7, p̂)
p_bin = float(parity_even.mean()) / DRAWS
exp_bin = stats.binom.pmf(np.arange(DRAWS + 1), DRAWS, p_bin) * n_draws
obs_bin = np.array([(parity_even == k).sum() for k in range(DRAWS + 1)])
p_binom = safe_chisq(obs_bin, exp_bin)
test_results.append(("Binomial", "parnih", f"n={DRAWS}, p̂={p_bin:.3f}", p_binom, kept(p_binom)))

# 4) GEOMETRIC — chi-square na gaps_geom
if len(gaps_geom) >= 2:
    p_geom = 1.0 / float(gaps_geom.mean())
    max_g = int(gaps_geom.max())
    exp_geom = stats.geom.pmf(np.arange(1, max_g + 1), p_geom) * len(gaps_geom)
    obs_geom = np.array([(gaps_geom == k).sum() for k in range(1, max_g + 1)])
    p_geom_test = safe_chisq(obs_geom, exp_geom)
else:
    p_geom, p_geom_test = np.nan, np.nan
test_results.append(("Geometric", f"gap do {TARGET}", f"p̂={p_geom:.3f}", p_geom_test, kept(p_geom_test)))

# 5) POISSON — chi-square na window_counts
lam = float(window_counts.mean())
max_w = int(window_counts.max())
exp_pois = stats.poisson.pmf(np.arange(max_w + 1), lam) * len(window_counts)
obs_pois = np.array([(window_counts == k).sum() for k in range(max_w + 1)])
p_pois = safe_chisq(obs_pois, exp_pois)
test_results.append(("Poisson", f"# {TARGET} u W={W}", f"λ={lam:.3f}", p_pois, kept(p_pois)))

# 6) EXPONENTIAL — KS test na svim gapovima
loc_e, scale_e = stats.expon.fit(exp_gaps, floc=0)
_, p_exp = stats.kstest(exp_gaps, "expon", args=(loc_e, scale_e))
test_results.append(("Exponential", "gapovi 1..39", f"scale={scale_e:.2f}", float(p_exp), kept(float(p_exp))))

# 7) GAMMA — KS test na gamma_vals
try:
    sh, loc_g, sc_g = stats.gamma.fit(gamma_vals, floc=0)
    _, p_gamma = stats.kstest(gamma_vals, "gamma", args=(sh, loc_g, sc_g))
    p_gamma = float(p_gamma)
    gamma_par = f"shape={sh:.2f}, scale={sc_g:.2f}"
except Exception:
    p_gamma, gamma_par = np.nan, "fit failed"
test_results.append(("Gamma", f"vreme do {K_GAMMA}. pojave", gamma_par, p_gamma, kept(p_gamma)))

# 8) BETA — KS test na empirijskim frekvencijama brojeva (39 tačaka)
try:
    a_b, b_b, _loc, _sc = stats.beta.fit(beta_freqs, floc=0, fscale=1)
    _, p_beta = stats.kstest(beta_freqs, "beta", args=(a_b, b_b, 0, 1))
    p_beta = float(p_beta)
    beta_par = f"α={a_b:.2f}, β={b_b:.2f}"
except Exception:
    p_beta, beta_par = np.nan, "fit failed"
test_results.append(("Beta", "empir. freq. brojeva", beta_par, p_beta, kept(p_beta)))

# 9) UNIFORM — chi-square nad 1..39 (svaki broj treba ~total/39)
exp_unif = np.full(POOL, total_picks / POOL)
_, p_unif = stats.chisquare(counts_per_num, exp_unif)
p_unif = float(p_unif)
test_results.append(("Uniform", "svi brojevi 1..39", f"E≈{exp_unif[0]:.1f}/broj", p_unif, kept(p_unif)))


# Print tabela
print("KORAK 3 — testovi fita (p > 0.05 ⇒ POGODNA):")
print(f"{'Distribucija':<13}{'Loto feature':<26}{'Param':<28}{'p-value':<10}Status")
print("-" * 90)
for name, feat, par, p, st in test_results:
    p_str = f"{p:.4f}" if isinstance(p, float) and not np.isnan(p) else "N/A"
    print(f"{name:<13}{feat:<26}{par:<28}{p_str:<10}{st}")
print()


# ═════════════════════════════════════════════════════════════════════
# KORAK 4: Using Probability Right  →  NEXT predikcija iz PRAVIH
#          verovatnoća POGODNIH distribucija (zamenjuje polazne Steps 4–6).
#
#   Per-broj P(k pojavi u sledećem izvlačenju) — RECENCY-WEIGHTED:
#     • Ako je Beta POGODNA → Beta-Bernoulli posterior + recency boost:
#         P(k) = (α + appearances_k + γ·recent_k)
#                ─────────────────────────────────
#                (α + β + n_draws + γ·RECENT_W)
#       γ (RECENCY_GAMMA) podiže težinu poslednjih RECENT_W izvlačenja.
#       Time per-broj P više NIJE 1:1 sa raw frekvencijom: brojevi sa
#       istim ukupnim count-om ali različitom recent aktivnošću dobijaju
#       različito P. (princip iz polaznog "Step 5: Bayesian Thinking")
#     • Ako Beta nije pogodna → recency-weighted empirijska:
#         P(k) = (appearances_k + γ·recent_k) / (n_draws + γ·RECENT_W)
#
#   Kombinacijski log-skor (za 7-torke od TOP_POOL kandidata):
#     score = Σ log P(k)                                  ← per-broj prior
#           + log N(sum; μ, σ)              (ako Normal pogodna)
#           + log Binom(parnih; n=7, p̂)     (ako Binomial pogodna)
#           + Σ log Poisson(recent_k; λ_k)  (ako Poisson pogodna; analogno
#                                            polaznom Step 6: Poisson Regression)
#
#   Enumeracija C(TOP_POOL, 7) kandidata → najbolji log-skor = NEXT.
# ═════════════════════════════════════════════════════════════════════
good = [name for name, _, _, p, _ in test_results
        if isinstance(p, float) and not np.isnan(p) and p > 0.05]
print(f"Pogodne distribucije: {good if good else 'NIJEDNA'}\n")

# Per-broj broj pojavljivanja u izvlačenju (Bernoulli trial po izvlačenju)
appearances_per_num = np.array([
    int((arr == k).any(axis=1).sum()) for k in range(1, POOL + 1)
])

# Recent prozor (potrebno za Poisson log-skor i za prikaz)
recent_block = arr[-RECENT_W:] if n_draws >= RECENT_W else arr
recent_counts = np.zeros(POOL)
for r in recent_block:
    for v in r:
        if 1 <= v <= POOL:
            recent_counts[v - 1] += 1

# Gap signal (samo za prikaz; Geometric je memoryless pa direktno ne ulazi u skor)
last_seen = np.full(POOL, -1)
for i in range(n_draws):
    for v in arr[i]:
        last_seen[v - 1] = i
gap_now = (n_draws - 1) - last_seen

# ─────────────────────────────────────────────────────────────────────
# 1) Per-broj P(k pojavi u next) — RECENCY-WEIGHTED Beta-Bernoulli posterior
#    Recent prozor dobija težinu γ (RECENCY_GAMMA) → dva broja sa istim
#    ukupnim count-om dobijaju različito P ako im se razlikuje recent aktivnost.
# ─────────────────────────────────────────────────────────────────────
beta_fit_ok = ("Beta" in good) and (not np.isnan(p_beta))
gamma_w = RECENCY_GAMMA
if beta_fit_ok:
    p_per_num = (a_b + appearances_per_num + gamma_w * recent_counts) / (
                 a_b + b_b + n_draws + gamma_w * RECENT_W)
    p_source = (f"Beta-Bernoulli posterior + recency  "
                f"(α={a_b:.3f}, β={b_b:.3f}, γ={gamma_w}, W={RECENT_W})")
else:
    p_per_num = (appearances_per_num + gamma_w * recent_counts) / (
                 n_draws + gamma_w * RECENT_W)
    p_source = (f"recency-weighted empirijska  "
                f"(γ={gamma_w}, W={RECENT_W})")

# Uniform baseline za poređenje
p_uniform_baseline = DRAWS / POOL  # 7/39 ≈ 0.1795

# ─────────────────────────────────────────────────────────────────────
# 2) Kombinacijski log-skor (kombinuje per-broj prior + shape distribucije)
# ─────────────────────────────────────────────────────────────────────
def comb_log_score(combo):
    s = 0.0
    for k in combo:
        s += np.log(p_per_num[k - 1] + 1e-12)

    if "Normal" in good:
        s += float(stats.norm.logpdf(sum(combo), mu, sigma))

    if "Binomial" in good:
        ev = sum(1 for k in combo if k % 2 == 0)
        s += float(stats.binom.logpmf(ev, DRAWS, p_bin))

    if "Poisson" in good:
        # λ_k = očekivani broj pojava broja k u prozoru RECENT_W
        for k in combo:
            lam_k = appearances_per_num[k - 1] / n_draws * RECENT_W
            s += float(stats.poisson.logpmf(int(recent_counts[k - 1]), lam_k + 1e-9))

    return s


# ─────────────────────────────────────────────────────────────────────
# 3) Enumeracija C(TOP_POOL, 7) kandidata
# ─────────────────────────────────────────────────────────────────────
TOP_POOL = 13
top_candidates = sorted((np.argsort(p_per_num)[::-1][:TOP_POOL] + 1).tolist())

n_combos = 0
best_combo = None
best_score = -np.inf
for combo in combinations(top_candidates, DRAWS):
    sc = comb_log_score(combo)
    n_combos += 1
    if sc > best_score:
        best_score = sc
        best_combo = combo

next_pred = list(best_combo)

print(f"Per-broj P izvor: {p_source}")
print(f"Uniform baseline: {p_uniform_baseline:.5f}\n")

print(f"Enumeracija: C({TOP_POOL},{DRAWS}) = {n_combos} kandidata")
print()
print(f"NEXT predikcija: {next_pred}")
print(f"  log-skor       = {best_score:.4f}")
print(f"  suma           = {sum(next_pred)}   (μ={mu:.1f}, σ={sigma:.1f})")
print(f"  parnih         = {sum(1 for k in next_pred if k % 2 == 0)}/{DRAWS}")
print()

# Sumarni faktori za izabranu kombinaciju
factor_lines = []
factor_lines.append(f"Σ log P(k)            = {sum(np.log(p_per_num[k-1] + 1e-12) for k in next_pred):.4f}")
if "Normal" in good:
    factor_lines.append(f"log N(sum; μ, σ)      = {float(stats.norm.logpdf(sum(next_pred), mu, sigma)):.4f}")
if "Binomial" in good:
    ev = sum(1 for k in next_pred if k % 2 == 0)
    factor_lines.append(f"log Binom(parnih; n,p)= {float(stats.binom.logpmf(ev, DRAWS, p_bin)):.4f}")
if "Poisson" in good:
    pois_sum = 0.0
    for k in next_pred:
        lam_k = appearances_per_num[k-1] / n_draws * RECENT_W
        pois_sum += float(stats.poisson.logpmf(int(recent_counts[k-1]), lam_k + 1e-9))
    factor_lines.append(f"Σ log Poisson(rec;λ)  = {pois_sum:.4f}")
print("Faktori log-skora za izabranu kombinaciju:")
for line in factor_lines:
    print(f"   • {line}")
print()

# ─────────────────────────────────────────────────────────────────────
# 4) Top 15 brojeva po per-broj P
# ─────────────────────────────────────────────────────────────────────
order = np.argsort(p_per_num)[::-1]
print(f"Top 13 brojeva po per-broj P (baseline {p_uniform_baseline:.5f}):")
print(f"{'Broj':<6}{'P(next)':<12}{'P/baseline':<12}{'Count':<8}{'Recent':<8}{'Gap':<6}")
for k in order[:13]:
    print(f"{k+1:<6}{p_per_num[k]:<12.5f}{p_per_num[k]/p_uniform_baseline:<12.4f}"
          f"{int(appearances_per_num[k]):<8}{int(recent_counts[k]):<8}{int(gap_now[k]):<6}")
print()

elapsed = time.time() - T0
print(f"Ukupno vreme: {timedelta(seconds=int(elapsed))} ({elapsed:.1f} s)\n")


# ═════════════════════════════════════════════════════════════════════
# TXT izlaz (sve gore navedeno + parametri)
# ═════════════════════════════════════════════════════════════════════
with open(TXT_PATH, "w", encoding="utf-8") as f:
    f.write("Loto 7/39 — 9 distribucija nad CSV-om\n")
    f.write("=" * 60 + "\n")
    f.write(f"CSV: {CSV_PATH}\n")
    f.write(f"Broj izvlačenja: {n_draws}\n")
    f.write(f"Opseg: 1..{POOL}    po izvlačenju: {DRAWS}\n")
    f.write(f"Plot: {PNG_PATH}\n\n")

    f.write("KORAK 3 — fit testovi (p > 0.05 ⇒ POGODNA):\n")
    f.write(f"{'Distribucija':<13}{'Loto feature':<26}{'Param':<28}{'p-value':<10}Status\n")
    f.write("-" * 90 + "\n")
    for name, feat, par, p, st in test_results:
        p_str = f"{p:.4f}" if isinstance(p, float) and not np.isnan(p) else "N/A"
        f.write(f"{name:<13}{feat:<26}{par:<28}{p_str:<10}{st}\n")

    f.write(f"\nPogodne distribucije: {good if good else 'NIJEDNA'}\n")

    f.write(f"\nKORAK 4 — prave verovatnoće iz fitovanih distribucija:\n")
    f.write(f"   Per-broj P izvor:  {p_source}\n")
    f.write(f"   Uniform baseline:  {p_uniform_baseline:.5f}\n")
    f.write(f"   Enumeracija:       C({TOP_POOL},{DRAWS}) = {n_combos} kandidata\n")
    f.write(f"   TOP_POOL kandidati: {top_candidates}\n\n")

    f.write(f"NEXT predikcija: {next_pred}\n")
    f.write(f"   log-skor       = {best_score:.4f}\n")
    f.write(f"   suma           = {sum(next_pred)}   (μ={mu:.1f}, σ={sigma:.1f})\n")
    f.write(f"   parnih         = {sum(1 for k in next_pred if k % 2 == 0)}/{DRAWS}\n")

    f.write("\nFaktori log-skora za izabranu kombinaciju:\n")
    for line in factor_lines:
        f.write(f"   • {line}\n")

    f.write(f"\nTop 13 brojeva po per-broj P (baseline {p_uniform_baseline:.5f}):\n")
    f.write(f"{'Broj':<6}{'P(next)':<12}{'P/baseline':<12}{'Count':<8}{'Recent':<8}{'Gap':<6}\n")
    for k in order[:13]:
        f.write(f"{k+1:<6}{p_per_num[k]:<12.5f}{p_per_num[k]/p_uniform_baseline:<12.4f}"
                f"{int(appearances_per_num[k]):<8}{int(recent_counts[k]):<8}{int(gap_now[k]):<6}\n")

    f.write(f"\nUkupno vreme: {timedelta(seconds=int(elapsed))} ({elapsed:.1f} s)\n")

print(f"TXT saved → {TXT_PATH}")
print()
"""
Učitano izvlačenja: 4622  (CSV: loto7_4622_k42.csv)
Opseg: 1..39   po izvlačenju: 7

Sažetak (prve kolone po izvlačenju):
   suma_7  ima_7  parnih
0     143      0       4
1     115      0       2
2     154      0       3
3     195      0       4
4     124      0       3

Shape: (4622, 3)

Basic Stats:
         suma_7    ima_7   parnih
count  4622.00  4622.00  4622.00
mean    140.51     0.18     3.41
std      27.64     0.39     1.20
min      48.00     0.00     0.00
25%     121.00     0.00     3.00
50%     141.00     0.00     3.00
75%     160.00     0.00     4.00
max     231.00     1.00     7.00

Plot saved → /Users/4c/Desktop/GHQ/DISTRIBUCIJA/distribucija_1.png

KORAK 3 — testovi fita (p > 0.05 ⇒ POGODNA):
Distribucija Loto feature              Param                       p-value   Status
------------------------------------------------------------------------------------------
Normal       suma_7                    μ=140.51, σ=27.64           0.0593    POGODNA
Bernoulli    ima_7                     p̂=0.182                    1.0000    POGODNA
Binomial     parnih                    n=7, p̂=0.487               0.0000    NEPOGODNA
Geometric    gap do 7                  p̂=0.182                    0.5271    POGODNA
Poisson      # 7 u W=20                λ=3.638                     0.0000    NEPOGODNA
Exponential  gapovi 1..39              scale=5.57                  0.0000    NEPOGODNA
Gamma        vreme do 5. pojave        shape=6.33, scale=4.40      0.0000    NEPOGODNA
Beta         empir. freq. brojeva      α=664.03, β=25233.08        0.9715    POGODNA
Uniform      svi brojevi 1..39         E≈829.6/broj                0.1384    POGODNA

Pogodne distribucije: ['Normal', 'Bernoulli', 'Geometric', 'Beta', 'Uniform']

Per-broj P izvor: Beta-Bernoulli posterior + recency  (α=664.029, β=25233.082, γ=5, W=50)
Uniform baseline: 0.17949

Enumeracija: C(13,7) = 1716 kandidata

NEXT predikcija: [8, 11, 16, 22, 23, 24, 34]
  log-skor       = -25.0093
  suma           = 138   (μ=140.5, σ=27.6)
  parnih         = 5/7

Faktori log-skora za izabranu kombinaciju:
   • Σ log P(k)            = -20.7670
   • log N(sum; μ, σ)      = -4.2423

Top 13 brojeva po per-broj P (baseline 0.17949):
Broj  P(next)     P/baseline  Count   Recent  Gap   
8     0.05259     0.2930      909     9       9     
23    0.05207     0.2901      903     7       1     
34    0.05168     0.2879      871     11      0     
24    0.05145     0.2866      839     16      1     
38    0.05135     0.2861      841     15      0     
16    0.05116     0.2850      835     15      4     
33    0.05109     0.2847      853     11      5     
39    0.05096     0.2839      849     11      6     
29    0.05090     0.2836      847     11      8     
11    0.05073     0.2827      857     8       6     
22    0.05067     0.2823      850     9       3     
37    0.05064     0.2821      859     7       8     
35    0.05060     0.2819      843     10      7     

Ukupno vreme: 0:00:07 (7.9 s)

TXT saved → /Users/4c/Desktop/GHQ/DISTRIBUCIJA/distribucija_1.txt
"""



"""
KORAK 1 — učitava loto7_4622_k42.csv (4622 izvlačenja).

KORAK 2 — izvodi 9 feature-a, po jedan za svaku distribuciju:

Normal → suma 7 brojeva
Bernoulli → da li je broj 7 izvučen (1/0)
Binomial → broj parnih u izvlačenju
Geometric → gap do sledeće pojave broja 7
Poisson → broj pojava broja 7 u prozoru od 20 izvlačenja
Exponential → svi gapovi između pojava svih brojeva
Gamma → vreme do 5. pojave istog broja
Beta → empirijske frekvencije brojeva 1..39
Uniform → svi izvučeni brojevi 1..39
3x3 grid histograma → distribucija_1.png.

KORAK 3 — KS test (continuous: Normal/Exponential/Gamma/Beta) i chi-square (discrete: Bernoulli/Binomial/Geometric/Poisson/Uniform). Ispisuje tabelu sa p-value i statusom POGODNA / NEPOGODNA.

KORAK 4 — agregira signale samo iz POGODNIH distribucija:

Normal/Uniform/Beta → empirijska frekvencija
Poisson/Binomial → broj pojava u poslednjih 50 izvlačenja
Geometric/Exponential/Gamma → invertovan gap
Sve se z-normalizuje, sabira, vrh-7 brojeva → NEXT predikcija. Fallback je empirijska frekvencija ako nijedna distribucija ne prođe.

Izlazi: distribucija_1.png + distribucija_1.txt (sa NEXT predikcijom i Top-15 brojeva).
"""
