import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.preprocessing
import joblib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ─────────────────────────────────────────────
# 1. Load data + build invariant mass
# ─────────────────────────────────────────────

def load_root(path):
    f = uproot.open(path)
    tree = f["Events"]
    return tree.arrays(library="ak")


def invariant_mass(pt1, eta1, phi1, pt2, eta2, phi2, m=0.000511):
    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1)
    e1  = np.sqrt(px1**2 + py1**2 + pz1**2 + m**2)

    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)
    pz2 = pt2 * np.sinh(eta2)
    e2  = np.sqrt(px2**2 + py2**2 + pz2**2 + m**2)

    E  = e1 + e2
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2

    m2 = E**2 - px**2 - py**2 - pz**2
    m2 = np.clip(m2, 0, None)  
    return np.sqrt(m2)


def build_arrays(events, pt_min=15):
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

    true_pt  = ak.to_numpy(events["GenDressedLepton_pt"])
    true_eta = ak.to_numpy(events["GenDressedLepton_eta"])
    true_phi = ak.to_numpy(events["GenDressedLepton_phi"])

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

    true_pt   = true_pt[mask]
    true_eta  = true_eta[mask]
    true_phi  = true_phi[mask]

    # --- reco and truth dilepton invariant mass ---
    m_reco = invariant_mass(
        reco_pt[:, 0], reco_eta[:, 0], reco_phi[:, 0],
        reco_pt[:, 1], reco_eta[:, 1], reco_phi[:, 1],
    )
    m_truth = invariant_mass(
        true_pt[:, 0], true_eta[:, 0], true_phi[:, 0],
        true_pt[:, 1], true_eta[:, 1], true_phi[:, 1],
    )

    # --- input features: BOTH electrons ---
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

        np.log(reco_pt[:, 1]),
        reco_eta[:, 1],
        reco_phi[:, 1],
        miniIso[:, 1],
        sieie[:, 1],
        dxy[:, 1],
        dz[:, 1],
        hoe[:, 1],
        scEtOverPt[:, 1],
        eInvMinusPInv[:, 1],
    ])

    # --- regression target: log(m_truth / m_reco) ---
    # guard against m_reco == 0
    safe = m_reco > 1e-6
    X = X[safe]
    m_reco = m_reco[safe]
    m_truth = m_truth[safe]

    y = np.log(m_truth / m_reco)

    return X, y, m_reco, m_truth


# ─────────────────────────────────────────────
# 2. Load files, split
# ─────────────────────────────────────────────

ev_high = load_root("data/processed/mcDYhigh.root")
ev_low  = load_root("data/processed/mcDYlow.root")

X_high, y_high, mreco_high, mtruth_high = build_arrays(ev_high)
X_low,  y_low,  mreco_low,  mtruth_low  = build_arrays(ev_low)

X      = np.concatenate([X_high,      X_low],      axis=0)
y      = np.concatenate([y_high,      y_low],      axis=0)
m_reco = np.concatenate([mreco_high,  mreco_low],  axis=0)
m_true = np.concatenate([mtruth_high, mtruth_low], axis=0)

X_train, X_temp, y_train, y_temp, mreco_train, mreco_temp, mtrue_train, mtrue_temp = \
    sklearn.model_selection.train_test_split(
        X, y, m_reco, m_true, test_size=0.30, random_state=42
    )
X_val, X_test, y_val, y_test, mreco_val, mreco_test, mtrue_val, mtrue_test = \
    sklearn.model_selection.train_test_split(
        X_temp, y_temp, mreco_temp, mtrue_temp, test_size=0.50, random_state=42
    )

print(f"Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")

# ─────────────────────────────────────────────
# 3. Feature scaling
# ─────────────────────────────────────────────

scaler = sklearn.preprocessing.StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

joblib.dump(scaler, "models/mass_correction_scaler.pkl")
print("Scaler saved → models/mass_correction_scaler.pkl")

# ─────────────────────────────────────────────
# 4. Build model
# ─────────────────────────────────────────────
#
#  Architecture: same dense network as train_nn.py, but
#   • 20 inputs (10 features x 2 electrons)
#   • 1 output: log(m_truth / m_reco)
#

def build_model(input_dim=X_train_s.shape[1], output_dim=1):
    inp = keras.Input(shape=(input_dim,), name="reco_features")

    x = layers.Dense(256, name="dense_1")(inp)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.10)(x)

    x = layers.Dense(128, name="dense_2")(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.10)(x)

    x = layers.Dense(64, name="dense_3")(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.BatchNormalization()(x)

    out = layers.Dense(output_dim, activation="linear", name="log_mass_correction")(x)

    model = keras.Model(inputs=inp, outputs=out)
    return model


model = build_model()
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.Huber(delta=0.3),
    metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")]
)

# ─────────────────────────────────────────────
# 5. Train
# ─────────────────────────────────────────────

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        "models/mass_correction_best.keras",
        monitor="val_loss", save_best_only=True, verbose=1
    ),
]

history = model.fit(
    X_train_s, y_train,
    validation_data=(X_val_s, y_val),
    epochs=30,
    batch_size=512,
    callbacks=callbacks,
    verbose=1,
)

# ─────────────────────────────────────────────
# 6. Training curves
# ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["loss"],     label="train loss")
axes[0].plot(history.history["val_loss"], label="val loss")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Huber loss")
axes[0].set_title("Loss curves"); axes[0].legend()

axes[1].plot(history.history["mae"],     label="train MAE")
axes[1].plot(history.history["val_mae"], label="val MAE")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MAE")
axes[1].set_title("MAE curves"); axes[1].legend()

plt.tight_layout()
plt.savefig("results/mass_training_curves.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────
# 7. Evaluate on test set — diagonality + response
# ─────────────────────────────────────────────

def compute_diagonality(H, k=2):
    i, j = np.indices(H.shape)
    band = np.abs(i - j) <= k
    return np.sum(H[band]) / np.sum(H)


y_pred_test = model.predict(X_test_s, batch_size=4096).flatten()

# Corrected mass = m_reco * exp(predicted log-ratio)
corrected_mass = mreco_test * np.exp(y_pred_test)
truth_mass     = mreco_test * np.exp(y_test)   # == mtrue_test, recomputed for consistency

mass_bins = np.array([
    40, 45, 50, 55, 60, 64, 68, 72, 76, 81, 86, 91, 96, 101, 106, 110,
    115, 120, 126, 133, 141, 150, 160, 171, 185, 200, 220, 243, 273,
    320, 380, 440, 510, 600, 700, 830, 1000, 1500, 2000, 3000
])

m_min, m_max = mass_bins[0], mass_bins[-1]
mask = (
    (truth_mass > m_min) & (truth_mass < m_max) &
    (corrected_mass > m_min) & (corrected_mass < m_max)
)

bins = [mass_bins, mass_bins]  # same custom edges on both axes

H_raw, xe, ye = np.histogram2d(truth_mass[mask], mreco_test[mask], bins=bins)
D_raw = compute_diagonality(H_raw)

H_cor, xe, ye = np.histogram2d(truth_mass[mask], corrected_mass[mask], bins=bins)
D_cor = compute_diagonality(H_cor)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].hist2d(truth_mass[mask], mreco_test[mask], bins=bins, norm="log")
axes[0].set_xlabel(r"$m_{ee}^{truth}$ [GeV]")
axes[0].set_ylabel(r"$m_{ee}^{reco}$ [GeV]")
axes[0].set_title(f"Before correction  (D = {D_raw:.3f})")

axes[1].hist2d(truth_mass[mask], corrected_mass[mask], bins=bins, norm="log")
axes[1].set_xlabel(r"$m_{ee}^{truth}$ [GeV]")
axes[1].set_ylabel(r"$m_{ee}^{corrected}$ [GeV]")
axes[1].set_title(f"After NN correction  (D = {D_cor:.3f})")

for ax in axes:
    lim = [m_min, m_max]
    ax.plot(lim, lim, "r--", lw=1, label="diagonal")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

plt.suptitle("Dilepton invariant mass correction", fontsize=13)
plt.tight_layout()
plt.savefig("results/mass_correction_diagonal.png", dpi=150)
plt.show()

print(f"\nDiagonality  raw : {D_raw:.4f}")
print(f"Diagonality  NN  : {D_cor:.4f}")
print(f"Improvement      : {(D_cor - D_raw) / D_raw * 100:+.1f}%")

# Response distributions
response_raw  = mreco_test / truth_mass
response_corr = corrected_mass / truth_mass

plt.figure()
plt.hist(response_raw, bins=100, range=(0.5, 1.5), density=True,
         histtype="step", label="raw")
plt.hist(response_corr, bins=100, range=(0.5, 1.5), density=True,
         histtype="step", label="NN")
plt.xlabel(r"$m_{ee}^{reco \,/\, corrected} / m_{ee}^{truth}$")
plt.legend()
plt.savefig("results/mass_hist_check.png")
plt.show()