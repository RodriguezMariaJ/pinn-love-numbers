# paper_scan.py  --  the (s, l, m) Love-number table for the paper
# =============================================================================
# For every PHYSICAL row (s, l, m) the equation is SOLVED at spin weight
# s_solve = -|s|.  The Love number is identical (k(-s) = k(s) exactly, by the
# Gamma reflection formula, for integer s and any l, regulator included), but
# at -|s| the unphysical (outgoing) horizon branch z^(s_solve - 2 i m gamma)
# DIVERGES, so the bounded hard-BC ansatz cannot absorb it.  At s = +2 it
# vanishes like z^2 and the network imitates it at almost no cost.
# s = 0 is solved as is.
#
# Per row:  Adam (EPOCHS) -> L-BFGS on a grid refreshed every REFRESH_EVERY
# rounds, stopped early when the held-out loss stops improving.
# Extraction: Method 1 (integral residual), as in the paper.
#
# Everything is checkpointed:
#   paper_scan.json                  one record per finished row
#   nets/adam_*.pt, nets/final_*.pt  networks; nets/partial_*.pt during L-BFGS
# Re-running skips finished rows and resumes an interrupted one from its
# partial network.  At the end it prints the table and writes paper_table.tex.
#
# Needs love_numbers_pinn_v7.py in the same folder.  Run with
#     %run paper_scan.py
# =============================================================================

import json
import math
import os
import time

import numpy as np
import torch
import torch.optim as optim

import love_numbers_pinn_v7 as L

# ============================== settings =====================================
EPS = 1e-3
EPOCHS = 750
LBFGS_MAX = 40           # at most this many L-BFGS rounds
LBFGS_INNER = 20
LBFGS_PTS = 2000
REFRESH_EVERY = 10
STALL_ROUNDS = 3         # stop L-BFGS if the best held-out loss has not
STALL_GAIN = 0.05        # improved by 5% over the last 3 rounds
SAVE_EVERY = 5
SEED = 0
SOLVE_AT_MINUS_S = True  # False: solve at the physical s (old behaviour)

# Physical rows, in the order they are run (most informative first).
ROWS = [
    # the case that failed at s = +2: check this one first
    (2, 3, 2),
    # tensor, l = m and l != m
    (2, 2, 2), (2, 2, 1), (2, 3, 3), (2, 3, 1),
    (2, 4, 4), (2, 4, 3), (2, 4, 2), (2, 4, 1), (2, 5, 1),
    # vector and scalar, m = 1
    (1, 1, 1), (1, 2, 1), (1, 3, 1), (1, 4, 1), (1, 5, 1),
    (0, 1, 1), (0, 2, 1), (0, 3, 1), (0, 4, 1), (0, 5, 1),
]

NET_DIR = "nets"
CKPT = "paper_scan.json"
TEX = "paper_table.tex"
QUICK = False

if QUICK:
    EPOCHS, LBFGS_MAX, REFRESH_EVERY, SAVE_EVERY = 60, 3, 2, 1
    ROWS = [(2, 3, 2), (0, 2, 1)]
    NET_DIR, CKPT, TEX = "nets_quick", "paper_scan_quick.json", "paper_table_quick.tex"
    print(">>> QUICK: smoke test only, numbers meaningless.\n")

Z_REF = np.linspace(0.02, 0.99, 400)
Z_HOLD = np.linspace(0.01, 0.99, 4000)
os.makedirs(NET_DIR, exist_ok=True)


# ============================== helpers ======================================
def tag(s, l, m):
    return f"{s}_{l}_{m}".replace("-", "m")


def snapshot(model):
    return {k: t.clone() for k, t in model.state_dict().items()}


def new_model(cfg):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    return L.PINN(128, 4, *cfg.yp0()).to(L.device)


def holdout(model, pr):
    zh = torch.tensor(Z_HOLD.reshape(-1, 1), device=L.device)
    return L.sobolev_loss(model, pr, zh).item()


def stratified_grid(n, k):
    rng = np.random.default_rng(1000 * SEED + k)
    edges = 0.01 + 0.98 * np.arange(n) / n
    z = edges + rng.random(n) * 0.98 / n
    return torch.tensor(z.reshape(-1, 1), device=L.device)


def train_adam(cfg):
    pr = cfg.real_params()
    model = new_model(cfg)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    best, best_state, hl = float("inf"), None, float("nan")
    t0 = time.time()
    for ep in range(EPOCHS):
        z = torch.rand(400, 1, device=L.device) * 0.98 + 0.01
        loss = L.sobolev_loss(model, pr, z)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()
        if (ep + 1) % 100 == 0 or ep == 0:
            hl = holdout(model, pr)
            if hl < best:
                best, best_state = hl, snapshot(model)
        if (ep + 1) % 250 == 0 or ep == 0:
            el = time.time() - t0
            print(f"        ep {ep+1:6d}/{EPOCHS}  batch={loss.item():.3e}  "
                  f"holdout={hl:.3e}  best={best:.3e}  "
                  f"[{el:.0f}s, ETA {el/(ep+1)*(EPOCHS-ep-1)/60:.1f} min]",
                  flush=True)
    model.load_state_dict(best_state)
    return model


def train_lbfgs(model, cfg, partial_path):
    pr = cfg.real_params()
    best, best_state = holdout(model, pr), snapshot(model)
    hist = [best]
    print(f"        L-BFGS start: holdout={best:.3e}", flush=True)
    t0 = time.time()
    opt, z = None, None
    for k in range(LBFGS_MAX):
        if k % REFRESH_EVERY == 0:
            z = stratified_grid(LBFGS_PTS, k // REFRESH_EVERY)
            opt = optim.LBFGS(model.parameters(), lr=1.0,
                              max_iter=LBFGS_INNER, history_size=50,
                              line_search_fn="strong_wolfe",
                              tolerance_grad=1e-14, tolerance_change=1e-16)

        def closure():
            opt.zero_grad()
            loss = L.sobolev_loss(model, pr, z.clone())
            loss.backward()
            return loss

        loss = opt.step(closure).item()
        if not math.isfinite(loss):
            print("        non-finite loss; keeping best and stopping.")
            break
        hl = holdout(model, pr)
        if hl < best:
            best, best_state = hl, snapshot(model)
        hist.append(best)
        if (k + 1) % SAVE_EVERY == 0:
            torch.save(best_state, partial_path)
        el = time.time() - t0
        print(f"        lbfgs {k+1:3d}/{LBFGS_MAX}  grid={loss:.3e}  "
              f"holdout={hl:.3e}  best={best:.3e}  [{el:.0f}s]", flush=True)
        if (len(hist) > STALL_ROUNDS and
                hist[-1] > (1 - STALL_GAIN) * hist[-1 - STALL_ROUNDS]):
            print(f"        stalled (<{100*STALL_GAIN:.0f}% in {STALL_ROUNDS} "
                  f"rounds): stopping", flush=True)
            break
    model.load_state_dict(best_state)
    return model, best, k + 1


def load_rows():
    return json.load(open(CKPT)) if os.path.exists(CKPT) else []


def done(rows, s, l, m):
    for r in rows:
        if (r["s"], r["l"], r["m"], r["eps"]) == (s, l, m, EPS):
            return r
    return None


def sci_tex(x, digits=2):
    if x == 0 or not math.isfinite(x):
        return "$0$"
    e = int(math.floor(math.log10(abs(x))))
    mant = x / 10 ** e
    return f"${mant:.{digits}f}\\times 10^{{{e}}}$"


# ============================== run ==========================================
T0 = time.time()
print("=" * 104)
print(f"PAPER SCAN   eps={EPS}   device={L.device}   Adam {EPOCHS} + L-BFGS <= "
      f"{LBFGS_MAX} rounds   solve at {'-|s|' if SOLVE_AT_MINUS_S else 's'}")
print(f"{len(ROWS)} rows, checkpoint {os.path.abspath(CKPT)}")
print("=" * 104, flush=True)

rows = load_rows()
for (s, l, m) in ROWS:
    if done(rows, s, l, m):
        print(f"\n  (s,l,m)=({s},{l},{m}): done, skipping", flush=True)
        continue
    s_solve = -abs(s) if SOLVE_AT_MINUS_S else s
    cfg = L.Config(s_solve, l, m, EPS)
    cfg_phys = L.Config(s, l, m, EPS)
    if not L.parameters_supported(cfg):
        print(f"\n  (s,l,m)=({s},{l},{m}): not supported, skipping", flush=True)
        continue
    A_t, B_t = L.gamma_ratio_AB(cfg)
    kt = A_t / B_t
    if L.parameters_supported(cfg_phys):
        Ap, Bp = L.gamma_ratio_AB(cfg_phys)
        dk = abs(Ap / Bp - kt) / abs(kt)
    else:
        dk = float("nan")
    tg = tag(s_solve, l, m)
    adam_path = os.path.join(NET_DIR, f"adam_{tg}_e{EPOCHS}_s{SEED}.pt")
    final_path = os.path.join(NET_DIR, f"final_{tg}_e{EPOCHS}_s{SEED}.pt")
    partial_path = os.path.join(NET_DIR, f"partial_{tg}_e{EPOCHS}_s{SEED}.pt")

    print("\n" + "#" * 104)
    print(f"#  physical (s,l,m)=({s},{l},{m})   solved at s={s_solve}   "
          f"[{(time.time()-T0)/60:.0f} min elapsed]")
    print(f"#  exact k = {kt.real:+.6e} {kt.imag:+.6e} i   |B_grow| = {abs(B_t):.2f}"
          f"   k(s) vs k(s_solve): rel diff {dk:.1e}")
    print("#" * 104, flush=True)

    t0 = time.time()
    model = new_model(cfg)
    if os.path.exists(partial_path):
        model.load_state_dict(torch.load(partial_path, map_location=L.device))
        print(f"      resuming from {partial_path}", flush=True)
    elif os.path.exists(adam_path):
        model.load_state_dict(torch.load(adam_path, map_location=L.device))
        print(f"      Adam stage loaded from {adam_path}", flush=True)
    else:
        print(f"      Adam: {EPOCHS} epochs ...", flush=True)
        model = train_adam(cfg)
        torch.save(model.state_dict(), adam_path)
    model, best, n_lb = train_lbfgs(model, cfg, partial_path)
    torch.save(model.state_dict(), final_path)

    fr = L.Frobenius(cfg)
    yref = fr.y(Z_REF)
    ey = float(np.linalg.norm(L.eval_pinn(model, Z_REF) - yref) / np.linalg.norm(yref))
    y_of, dy_of, _ = L.pinn_callables(model, cfg)
    eA, eB, ek, k = L.errors(cfg, L.apply_op(L.op_integral(cfg), y_of, dy_of))
    aA = L.amplification(cfg, "1. Integral residual")[0]
    # baseline: the network switched off, ytilde = 1 + ytilde'(0) z only
    y1 = complex(*cfg.yp0())
    y0_of = lambda zz: 1 + y1 * np.asarray(zz, dtype=float)
    _, _, _, k0 = L.errors(cfg, L.apply_op(L.op_integral(cfg), y0_of, None))
    im_err_N0 = float(abs(k0.imag - kt.imag) / abs(kt.imag))
    rec = dict(s=s, l=l, m=m, s_solve=s_solve, eps=EPS, epochs=EPOCHS,
               lbfgs_rounds=n_lb, t=time.time() - t0, holdout=best,
               eps_y=ey, eps_A=float(eA), eps_B=float(eB), eps_k=float(ek),
               re_k=float(k.real), im_k=float(k.imag),
               re_k_ex=float(kt.real), im_k_ex=float(kt.imag),
               absB=float(abs(B_t)), A_A=float(aA),
               deg=l + s_solve, im_err_N0=im_err_N0)
    L.save_rows(rec, CKPT)
    rows.append(rec)
    if os.path.exists(partial_path):
        os.remove(partial_path)
    ire = abs(k.imag - kt.imag) / abs(kt.imag)
    print(f"\n  RESULT ({s},{l},{m}):  {rec['t']/60:.0f} min   holdout={best:.3e}   "
          f"eps_y={ey:.3e}   eps_A={eA:.3e}   Im k rel err={ire:.2e}", flush=True)
    print(f"     network switched off (1 + y'(0) z only): Im k rel err={im_err_N0:.2e}"
          f"   [polynomial degree l+s_solve = {l + s_solve}]", flush=True)
    print(f"     k PINN  = {k.real:+.4e} {k.imag:+.5e} i", flush=True)
    print(f"     k exact = {kt.real:+.4e} {kt.imag:+.5e} i", flush=True)

# ============================== summary ======================================
order = {r: i for i, r in enumerate(ROWS)}
res = sorted([r for r in rows if (r["s"], r["l"], r["m"]) in order and r["eps"] == EPS],
             key=lambda r: (-r["s"], r["l"], -r["m"]))
print("\n" + "=" * 140)
print(f"SUMMARY   eps={EPS}   Method 1   solved at {'-|s|' if SOLVE_AT_MINUS_S else 's'}")
print("=" * 140)
print(f"  {'s':<3}{'l':<3}{'m':<3}{'|B|':<8}{'A_A':<10}{'holdout':<11}{'eps_y':<11}"
      f"{'eps_A':<11}{'Re k PINN':<13}{'Re k exact':<13}{'Im k PINN':<14}"
      f"{'Im k exact':<14}{'Im rel err':<11}{'N=0 err':<10}{'deg':<5}{'min':<5}")
print("  " + "-" * 136)
for r in res:
    ire = abs(r["im_k"] - r["im_k_ex"]) / abs(r["im_k_ex"])
    print(f"  {r['s']:<3}{r['l']:<3}{r['m']:<3}{r['absB']:<8.2f}{r['A_A']:<10.3g}"
          f"{r['holdout']:<11.2e}{r['eps_y']:<11.2e}{r['eps_A']:<11.2e}"
          f"{r['re_k']:<+13.3e}{r['re_k_ex']:<+13.3e}{r['im_k']:<+14.4e}"
          f"{r['im_k_ex']:<+14.4e}{ire:<11.1e}{r.get('im_err_N0', float('nan')):<10.1e}"
          f"{r.get('deg', 0):<5}{r['t']/60:<5.0f}")

with open(TEX, "w") as f:
    f.write("% s & l & m & eps_y & Re k PINN & Im k PINN & Re k exact & Im k exact"
            " & Im k rel err & Im k rel err with network off\n")
    for r in res:
        f.write(f"{r['s']} & {r['l']} & {r['m']} & {sci_tex(r['eps_y'],1)} & "
                f"{sci_tex(r['re_k'])} & {sci_tex(r['im_k'])} & "
                f"{sci_tex(r['re_k_ex'])} & {sci_tex(r['im_k_ex'])} & "
                f"{sci_tex(abs(r['im_k']-r['im_k_ex'])/abs(r['im_k_ex']),1)} & "
                f"{sci_tex(r.get('im_err_N0', float('nan')),1)} \\\\\n")
print(f"\n  LaTeX rows written to {os.path.abspath(TEX)}")
print(f"  DONE in {(time.time()-T0)/60:.0f} min")
