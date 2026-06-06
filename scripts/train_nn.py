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
 
def load_root(path):
    f    = uproot.open(path)
    tree = f["Events"]
    return tree.arrays(library="ak")

def build_arrays(events):
    reco_pt  = ak.to_numpy(events["Electron_pt"])         # shape (N, 2)
    reco_eta = ak.to_numpy(events["Electron_eta"])
    reco_phi = ak.to_numpy(events["Electron_phi"])
    
    miniIso  = ak.to_numpy(events["Electron_miniPFRelIso_all"])
    sieie    = ak.to_numpy(events["Electron_sieie"])
    dxy      = ak.to_numpy(events["Electron_dxy"])
    dz       = ak.to_numpy(events["Electron_dz"])
    hoe      = ak.to_numpy(events["Electron_hoe"])
    scEtOverPt = ak.to_numpy(events["Electron_scEtOverPt"])
    eInvMinusPInv = ak.to_numpy(events["Electron_eInvMinusPInv"])
    # r9 = ak.to_numpy(events["Electron_r9"])
    # deltaEtaSC = ak.to_numpy(events["Electron_deltaEtaSC"])

    true_pt  = ak.to_numpy(events["GenDressedLepton_pt"])

    pt_min = 15
    
    mask = (reco_pt[:, 0] > pt_min) & (reco_pt[:, 1] > pt_min) 
    reco_pt   = reco_pt[mask]
    reco_eta  = reco_eta[mask]
    reco_phi  = reco_phi[mask]

    miniIso   = miniIso[mask]
    sieie     = sieie[mask]
    dxy       = dxy[mask]
    dz        = dz[mask]
    hoe       = hoe[mask]
    scEtOverPt = scEtOverPt[mask]
    eInvMinusPInv = eInvMinusPInv[mask]
    # r9 = r9[mask]
    # deltaEtaSC = deltaEtaSC[mask]

    true_pt   = true_pt[mask]
    
 
    # --- input features ---
    # Use log(pt) so the network sees a roughly Gaussian-shaped feature
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
        # r9[:, 0],
        # deltaEtaSC[:, 0],

        # np.log(reco_pt[:, 1]),
        # reco_eta[:, 1],
        # reco_phi[:, 1],

        # miniIso[:, 1],
        # sieie[:, 1],
        # dxy[:, 1],
        # dz[:, 1],
        # hoe[:, 1],
        # scEtOverPt[:, 1],
        # eInvMinusPInv[:, 1]
        # r9[:, 1],
        # deltaEtaSC[:, 1]
    ])
 
    # --- regression targets: correction factors (truth / reco) ---
    y = np.log(true_pt [:, 0]/ reco_pt[:, 0])   # shape (N, 2), values near 1.0
 
    # y = (true_pt - reco_pt) / reco_pt

    return X, y, reco_pt
 
ev_high = load_root("data/processed/mcDYhigh.root")
ev_low  = load_root("data/processed/mcDYlow.root")
 
X_high, y_high, rpt_high = build_arrays(ev_high)
X_low,  y_low,  rpt_low  = build_arrays(ev_low)

X   = np.concatenate([X_high,   X_low],   axis=0)
y   = np.concatenate([y_high,   y_low],   axis=0)
rpt = np.concatenate([rpt_high, rpt_low], axis=0)
 
# ─────────────────────────────────────────────
# 2. Train / val / test split
# ─────────────────────────────────────────────
 
X_train, X_temp, y_train, y_temp, rpt_train, rpt_temp = sklearn.model_selection.train_test_split(
    X, y, rpt, test_size=0.30, random_state=42
)
X_val, X_test, y_val, y_test, rpt_val, rpt_test = sklearn.model_selection.train_test_split(
    X_temp, y_temp, rpt_temp, test_size=0.50, random_state=42
)
 
print(f"Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")
 
# ─────────────────────────────────────────────
# 3. Feature scaling  (StandardScaler on X only)
# ─────────────────────────────────────────────
 
scaler = sklearn.preprocessing.StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)
 
joblib.dump(scaler, "models/pt_correction_scaler.pkl")
print("Scaler saved → models/pt_correction_scaler.pkl")
 
# ─────────────────────────────────────────────
# 4. Build model
# ─────────────────────────────────────────────
#
#  Architecture: small dense residual network
#   • 6 inputs
#   • 3 hidden layers (256 → 128 → 64 neurons), LeakyReLU + BatchNorm + Dropout
#   • 2 outputs (one correction factor per electron)
#   • Output activation: softplus  → ensures correction > 0
#     (softplus(x) = log(1 + e^x), always positive, smooth)
#
#  Loss: Huber (less sensitive to outlier events than plain MSE)
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
 
    # Output: predict correction factor — use softplus so it stays positive
    out = layers.Dense(output_dim, activation="linear", name="log_correction")(x)
 
    # x = layers.Dense(256)(inp)
    # x = layers.LeakyReLU(0.1)(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Dropout(0.1)(x)

    # x2 = layers.Dense(128)(x)
    # x2 = layers.LeakyReLU(0.1)(x2)
    # x2 = layers.BatchNormalization()(x2)
    # x2 = layers.Dropout(0.1)(x2)

    # # skip connection
    # x2 = layers.Add()([layers.Dense(128)(x), x2])

    # x3 = layers.Dense(64)(x2)
    # x3 = layers.LeakyReLU(0.1)(x3)
    # x3 = layers.BatchNormalization()(x3)

    # out = layers.Dense(2, activation="linear")(x3)

    model = keras.Model(inputs=inp, outputs=out)
    return model
 
 
model = build_model()
model.summary()
 
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.Huber(delta=0.3),   # Huber with small delta ≈ MAE-like near zero
    metrics=["mae",
             tf.keras.metrics.RootMeanSquaredError(name="rmse")]
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
        "models/pt_correction_best.keras",
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
plt.savefig("results/training_curves.png", dpi=150)
plt.show()
 
# ─────────────────────────────────────────────
# 7. Evaluate on test set — diagonality plot
# ─────────────────────────────────────────────
 
def compute_diagonality(H, k=2):
    i, j = np.indices(H.shape)
    band  = np.abs(i - j) <= k
    return np.sum(H[band]) / np.sum(H)
 
 
y_pred_test = model.predict(X_test_s, batch_size=4096).flatten()
 
# Corrected pT = reco_pt * predicted_correction_factor
corrected_pt_lead = rpt_test[:, 0] * np.exp(y_pred_test)
# corrected_pt_lead = rpt_test[:, 0] * y_pred_test[:, 0]
# corrected_pt_lead = rpt_test[:, 0] * (1.0 + y_pred_test[:, 0])
truth_pt_lead     = rpt_test[:, 0] * np.exp(y_test)   # truth_pt = reco_pt * true_ratio

# truth_pt_lead     = rpt_test[:, 0] * y_test[:, 0]   # truth_pt = reco_pt * true_ratio
# truth_pt_lead = rpt_test[:, 0] * (1.0 + y_test[:, 0])

pt_min, pt_max = 15, 300
mask = (
    (truth_pt_lead > pt_min) & (truth_pt_lead < pt_max) &
    (corrected_pt_lead > pt_min) & (corrected_pt_lead < pt_max)
)
 
bins = 75
 
# Raw reco vs truth
H_raw, xe, ye = np.histogram2d(
    truth_pt_lead[mask], rpt_test[mask, 0], bins=bins
)
D_raw = compute_diagonality(H_raw)
 
# Corrected vs truth
H_cor, xe, ye = np.histogram2d(
    truth_pt_lead[mask], corrected_pt_lead[mask], bins=bins
)
D_cor = compute_diagonality(H_cor)
 
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
 
axes[0].hist2d(truth_pt_lead[mask], rpt_test[mask, 0],
               bins=bins, norm="log")
axes[0].set_xlabel(r"$p_T^{truth}$ [GeV]")
axes[0].set_ylabel(r"$p_T^{reco}$ [GeV]")
axes[0].set_title(f"Before correction  (D = {D_raw:.3f})")
 
axes[1].hist2d(truth_pt_lead[mask], corrected_pt_lead[mask],
               bins=bins, norm="log")
axes[1].set_xlabel(r"$p_T^{truth}$ [GeV]")
axes[1].set_ylabel(r"$p_T^{corrected}$ [GeV]")
axes[1].set_title(f"After NN correction  (D = {D_cor:.3f})")
 
for ax in axes:
    lim = [0, pt_max]
    ax.plot(lim, lim, "r--", lw=1, label="diagonal")
    ax.legend(fontsize=8)
 
plt.suptitle("Leading electron  $p_T$  correction", fontsize=13)
plt.tight_layout()
plt.savefig("results/pt_correction_diagonal.png", dpi=150)
plt.show()
 
print(f"\nDiagonality  raw : {D_raw:.4f}")
print(f"Diagonality  NN  : {D_cor:.4f}")
print(f"Improvement      : {(D_cor - D_raw) / D_raw * 100:+.1f}%")

response_raw = rpt_test[:,0] / truth_pt_lead

response_corr = corrected_pt_lead / truth_pt_lead

plt.hist(response_raw,
         bins=100,
         range=(0.5,1.5),
         density=True,
         histtype="step",
         label="raw")

plt.hist(response_corr,
         bins=100,
         range=(0.5,1.5),
         density=True,
         histtype="step",
         label="NN")

plt.legend()
plt.savefig("results/hist_check.png")
plt.show()