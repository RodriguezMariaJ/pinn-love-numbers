# Black hole Love numbers with physics-informed neural networks

Code for the paper *Black Hole Love Numbers with Physics-Informed Neural Networks*
(P. Protopapas, L. F. Temoche, M. J. Rodríguez).

A physics-informed neural network (PINN) solves the static radial Teukolsky
equation on a Kerr background, written as a hypergeometric equation, and the
Love number $k_\ell = A_{\rm dec}/B_{\rm grow}$ is extracted from the trained
solution. Because the static problem has an exact solution, the error of the
solver and the error of the extracted Love number can be measured separately.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/majoquark/pinn-love-numbers/blob/main/run_scan.ipynb)

## Files

| File | Contents |
|---|---|
| `love_numbers_pinn_v7.py` | equation, network with hard horizon boundary condition, weighted $H^2$-Sobolev loss, exact reference (Frobenius series and Gamma-function connection coefficients), extraction methods |
| `paper_scan.py` | runs the $(s,\ell,m)$ table of the paper |
| `run_scan.ipynb` | Colab notebook that clones this repository and runs the scan |

## Running in Colab

Click the badge above and run the cells in order. The notebook mounts your
Google Drive, clones this repository into `MyDrive/pinn-love-numbers`, and runs
`paper_scan.py`. Results are written next to the code:

- `paper_scan.json`: one record per finished row,
- `paper_table.tex`: LaTeX rows of the table,
- `nets/`: trained networks.

A full run takes about 3 hours on a CPU runtime. If Colab disconnects, run the
cells again; finished rows are skipped and an interrupted row resumes from its
last saved network. Cell 3 of the notebook is an optional five-minute test.

This repository is private: to clone it from Colab, store a GitHub token in
Colab's *Secrets* panel under the name `GH_TOKEN`.

## Running locally

Requires Python 3 with `torch`, `numpy`, `mpmath` and `matplotlib`.

```bash
pip install torch numpy mpmath matplotlib
python paper_scan.py
```

## Method in brief

- **Equation.** Static radial Teukolsky equation in $z=(r-r_+)/(r-r_-)$,
  written as a hypergeometric equation for $\tilde y$, split into a coupled
  real system for its real and imaginary parts. The angular number is
  regulated, $\ell \to \ell_{\rm int} + \varepsilon$, with $\varepsilon = 10^{-3}$.
- **Network.** $\tilde y = 1 + \tilde y'(0)\,z + z^2\,\mathcal N(z)$, which
  imposes the horizon boundary condition exactly; $\mathcal N$ is a
  $4\times128$ tanh network, in double precision.
- **Loss.** Weighted $H^2$-Sobolev norm of the residual, weight $1/[z(1-z)]$.
- **Training.** Adam (750 epochs) followed by L-BFGS on a collocation grid
  that is redrawn every ten rounds. Model selection on a fixed held-out grid.
- **Spin weight.** Rows with $s \ge 1$ are solved at spin weight $-s$, which
  gives the same static Love number. At $-s$ the unphysical horizon solution
  diverges and cannot be absorbed by the network, and the solution has a much
  smaller amplitude (see the paper, Sections 2 and 3.2).
- **Extraction.** $A_{\rm dec}$ and $B_{\rm grow}$ by weighted least-squares
  projection onto the two local solutions at $z=1$ (Method 1 of the paper).

## Settings

The rows, epochs, regulator and L-BFGS settings are at the top of
`paper_scan.py`. Set `QUICK = True` for a short test run.
