from src.analysis_utils import (
    process_mc,
    z_mass_numpy,
)
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

mc_dy_high_data = "data/processed/mcDYhigh.root"
mc_dy_low_data = "data/processed/mcDYlow.root"

total_events = 85388673 
entry = 843234

wsumHigh = 8.70662e+10
wsumLow = 1.85149e+11

L_int = 8746231868.215154648 / 1e6; # pb^-1

sigmaDYhigh = 6422.0 # pb
sigmaDYlow = 20480.0 # pb
    
# DY total
reco_low, truth_low, weights_low = process_mc(mc_dy_low_data, sigmaDYlow, wsumLow, "DY low")
reco_high, truth_high, weights_high = process_mc(mc_dy_high_data, sigmaDYhigh, wsumHigh, "DY high")

weights = np.concatenate([weights_low, weights_high])
weights = ak.to_numpy(weights)
# --- merge ---
reco = ak.concatenate([reco_low, reco_high])
truth = ak.concatenate([truth_low, truth_high])

# --- convert to numpy ---
reco_lead = ak.to_numpy(reco["pt"][:, 0])
truth_lead = ak.to_numpy(truth["pt"][:, 0])

reco_mass  = z_mass_numpy(reco)
truth_mass = z_mass_numpy(truth)
reco_mass = ak.to_numpy(reco_mass)
truth_mass = ak.to_numpy(truth_mass)
# --- cut ---
mask = (truth_lead < 300) & (reco_lead < 300)

truth = truth_lead[mask]
reco  = reco_lead[mask]
w     = weights[mask]

truth = ak.to_numpy(truth)
reco = ak.to_numpy(reco)
w = ak.to_numpy(w)


H, xedges, yedges = np.histogram2d(
    truth_mass,
    reco_mass,
    bins=60,
    weights=weights
)

k = 0
i, j = np.indices(H.shape)
band = np.abs(i - j) <= k
D_mass = np.sum(H[band]) / np.sum(H)

plt.figure(figsize=(7,6))

plt.hist2d(
    truth_mass,
    reco_mass,
    bins=60,
    weights=weights,
    norm="log"
)

plt.xlabel(r"$m_Z^{truth}$ [GeV]")
plt.ylabel(r"$m_Z^{reco}$ [GeV]")

plt.colorbar(label="Events (weighted)")

plt.text(
    0.05, 0.93,
    f"Mass diagonality: {D_mass:.3f}",
    transform=plt.gca().transAxes,
    bbox=dict(facecolor='white', alpha=0.7)
)

plt.savefig("results/z_mass_diagonal.png")
plt.show()

# --- histogram ---
H, _, _ = np.histogram2d(
    truth,
    reco,
    bins=75,
    weights=w
)

# --- diagonality ---
k = 2
i, j = np.indices(H.shape)
band = np.abs(i - j) <= k
D = np.sum(H[band]) / np.sum(H)

# --- plot ---
plt.figure(figsize=(7,6))

plt.hist2d(
    truth,
    reco,
    bins=75,
    weights=w,
    norm="log"
)

plt.xlabel(r"$p_T^{truth}$ [GeV]")
plt.ylabel(r"$p_T^{reco}$ [GeV]")

plt.colorbar(label="Events (weighted)")

plt.text(
    0.05, 0.93,
    f"Diagonality: {D:.3f}",
    transform=plt.gca().transAxes,
    bbox=dict(facecolor='white', alpha=0.7)
)

plt.savefig("results/diagonality.png")
plt.show()