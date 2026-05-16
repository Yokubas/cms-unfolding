from src.analysis_utils import (
    process_mc,
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
dy_low = process_mc(mc_dy_low_data, sigmaDYlow, wsumLow, "DY low")
dy_high = process_mc(mc_dy_high_data, sigmaDYhigh, wsumHigh, "DY high")

dy_total = {
    "pt_reco": np.concatenate([dy_low["pt_reco"], dy_high["pt_reco"]]),
    "pt_truth": np.concatenate([dy_low["pt_truth"], dy_high["pt_truth"]]),
    "weights": np.concatenate([dy_low["weights"], dy_high["weights"]]),
}

reco_lead = ak.to_numpy(dy_total["pt_reco"][:, 0])
reco_sub  = ak.to_numpy(dy_total["pt_reco"][:, 1])

truth_lead = ak.to_numpy(dy_total["pt_truth"][:, 0])
truth_sub  = ak.to_numpy(dy_total["pt_truth"][:, 1])

weights = ak.to_numpy(dy_total["weights"])

mask = (truth_lead < 300) & (reco_lead < 300)

truth = truth_lead[mask]
reco  = reco_lead[mask]
w     = weights[mask]

H, _, _ = np.histogram2d(
    truth,
    reco,
    bins=75,
    weights = w
)
k = 2
i, j = np.indices(H.shape)
band = np.abs(i - j) <= k
D = np.sum(H[band]) / np.sum(H)
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
    0.05, 0.95,
    f"Diagonality: {D:.3f}",
    transform=plt.gca().transAxes,
    verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.7)
)
plt.show()