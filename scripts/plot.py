import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow import keras

# ─────────────────────────────────────────────
# CONFIG — edit these paths if needed, then hit run
# ─────────────────────────────────────────────
FILES = [
    "data/processed/mcDYhigh.root",
    "data/processed/mcDYlow.root",
]
SCALER_PATH = "models/pt_correction_scaler.pkl"
MODEL_PATH  = "models/pt_correction_best.keras"
OUTDIR      = "results"

PT_MIN = 15
PT_MAX = 300
BINS   = 75


# ─────────────────────────────────────────────
# Same feature construction as in train_nn.py
# (keep this in sync with train_nn.py!)
# ─────────────────────────────────────────────
def load_root(path):
    f = uproot.open(path)
    tree = f["Events"]
    return tree.arrays(library="ak")


def build_arrays(events, pt_min=PT_MIN):
    reco_pt  = ak.to_numpy(events["Electron_pt"])
    reco_eta = ak.to_numpy(events["Electron_eta"])
    reco_phi = ak.to_numpy(events["Electron_phi"])

    miniIso  = ak.to_numpy(events["Electron_miniPFRelIso_all"])
    sieie    = ak.to_numpy(events["Electron_sieie"])
    dxy      = ak.to_numpy(events["Electron_dxy"])
    dz       = ak.to_numpy(events["Electron_dz"])
    hoe      = ak.to_numpy(events["Electron_hoe"])
    scEtOverPt    = ak.to_numpy(events["Electron_scEtOverPt"])
    eInvMinusPInv = ak.to_numpy(events["Electron_eInvMinusPInv"])

    true_pt = ak.to_numpy(events["GenDressedLepton_pt"])

    mask = (reco_pt[:, 0] > pt_min) & (reco_pt[:, 1] > pt_min)

    reco_pt   = reco_pt[mask]
    reco_eta  = reco_eta[mask]
    reco_phi  = reco_phi[mask]

    miniIso   = miniIso[mask]
    sieie     = sieie[mask]
    dxy       = dxy[mask]
    dz        = dz[mask]
    hoe       = hoe[mask]
    scEtOverPt    = scEtOverPt[mask]
    eInvMinusPInv = eInvMinusPInv[mask]

    true_pt = true_pt[mask]

    X = np.column_stack([
        np.log(reco_pt[:, 0]),
        reco_eta[:, 0],
        reco_phi[:, 0],
        miniIso[:, 0],
        sieie[:, 0],
        dxy[:, 0],
        dz[:, 0],
        hoe[:, 0],
        scEtOverPt[:, 0],
        eInvMinusPInv[:, 0],
    ])

    y = np.log(true_pt[:, 0] / reco_pt[:, 0])

    return X, y, reco_pt


def compute_diagonality(H, k=2):
    i, j = np.indices(H.shape)
    band = np.abs(i - j) <= k
    return np.sum(H[band]) / np.sum(H)


# ─────────────────────────────────────────────
# Load saved scaler + model
# ─────────────────────────────────────────────
scaler = joblib.load(SCALER_PATH)
model = keras.models.load_model(MODEL_PATH)
print(f"Loaded scaler from  {SCALER_PATH}")
print(f"Loaded model  from  {MODEL_PATH}")

# ─────────────────────────────────────────────
# Load + concatenate data
# ─────────────────────────────────────────────
# X_list, y_list, rpt_list = [], [], []
# for path in FILES:
#     events = load_root(path)
#     X, y, rpt = build_arrays(events)
#     X_list.append(X)
#     y_list.append(y)
#     rpt_list.append(rpt)
#     print(f"  {path}: {len(X):,} events")

# X   = np.concatenate(X_list, axis=0)
# y   = np.concatenate(y_list, axis=0)
# rpt = np.concatenate(rpt_list, axis=0)
# print(f"Total: {len(X):,} events")

data = np.load("models/pt_correction_test_split.npz")
X   = data["X_test"]
y   = data["y_test"]
rpt = data["rpt_test"]
print(f"Loaded test split: {len(X):,} events")

# ─────────────────────────────────────────────
# Apply scaler + model
# ─────────────────────────────────────────────
X_s = scaler.transform(X)
y_pred = model.predict(X_s, batch_size=4096).flatten()

corrected_pt_lead = rpt[:, 0] * np.exp(y_pred)
truth_pt_lead     = rpt[:, 0] * np.exp(y)

mask = (
    (truth_pt_lead > PT_MIN) & (truth_pt_lead < PT_MAX) &
    (corrected_pt_lead > PT_MIN) & (corrected_pt_lead < PT_MAX)
)

bins = BINS

# ─────────────────────────────────────────────
# Diagonality 2D plots
# ─────────────────────────────────────────────
H_raw, _, _ = np.histogram2d(truth_pt_lead[mask], rpt[mask, 0], bins=bins)
D_raw = compute_diagonality(H_raw)

H_cor, _, _ = np.histogram2d(truth_pt_lead[mask], corrected_pt_lead[mask], bins=bins)
D_cor = compute_diagonality(H_cor)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].hist2d(truth_pt_lead[mask], rpt[mask, 0], bins=bins, norm="log")
axes[0].set_xlabel(r"$p_T^{truth}$ [GeV]")
axes[0].set_ylabel(r"$p_T^{reco}$ [GeV]")
axes[0].set_title(f"Before correction  (D = {D_raw:.3f})")

axes[1].hist2d(truth_pt_lead[mask], corrected_pt_lead[mask], bins=bins, norm="log")
axes[1].set_xlabel(r"$p_T^{truth}$ [GeV]")
axes[1].set_ylabel(r"$p_T^{corrected}$ [GeV]")
axes[1].set_title(f"After NN correction  (D = {D_cor:.3f})")

for ax in axes:
    lim = [0, PT_MAX]
    ax.plot(lim, lim, "r--", lw=1, label="diagonal")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/pt_correction_diagonal_loaded.png", dpi=150)
plt.show()

print(f"\nDiagonality  raw : {D_raw:.4f}")
print(f"Diagonality  NN  : {D_cor:.4f}")
print(f"Improvement      : {(D_cor - D_raw) / D_raw * 100:+.1f}%")

# ─────────────────────────────────────────────
# Response histograms
# ─────────────────────────────────────────────
response_raw  = rpt[:, 0] / truth_pt_lead
response_corr = corrected_pt_lead / truth_pt_lead

plt.figure()
plt.hist(response_raw, bins=100, range=(0.5, 1.5), density=True,
         histtype="step", label="raw")
plt.hist(response_corr, bins=100, range=(0.5, 1.5), density=True,
         histtype="step", label="NN")
plt.xlabel(r"$p_T^{reco \,/\, corrected} / p_T^{truth}$")
plt.legend()
plt.savefig(f"{OUTDIR}/hist_check_loaded.png")
plt.show()