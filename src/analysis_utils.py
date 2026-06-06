import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# ─────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────
 
def load_dataset(root_file, max_events=None):
    file = uproot.open(root_file)
    tree = file["Events"]
    data = tree.arrays(library="ak", entry_stop=max_events)
    return data
 
 
def build_electrons(events):
    electron = ak.zip(
        {
            "pt_reco":    events["Electron_pt"],
            "pt_truth":   events["GenDressedLepton_pt"],
            "eta_reco":   events["Electron_eta"],
            "eta_truth":  events["GenDressedLepton_eta"],
            "phi_reco":   events["Electron_phi"],
            "phi_truth":  events["GenDressedLepton_phi"],
            "energy_reco":  np.sqrt(events["Electron_pt"]**2         * np.cosh(events["Electron_eta"])**2),
            "energy_truth": np.sqrt(events["GenDressedLepton_pt"]**2 * np.cosh(events["GenDressedLepton_eta"])**2),
        },
        with_name="Momentum4D"
    )
    weights = events["genWeight"] if "genWeight" in events.fields else None
    return electron, weights
 
 
# ─────────────────────────────────────────────
# Kinematics
# ─────────────────────────────────────────────
 
def z_mass_numpy(leps):
    """Compute Z invariant mass from a 2-lepton array (using pt/eta/phi/energy fields)."""
    l0 = ak.to_numpy(leps[:, 0])
    l1 = ak.to_numpy(leps[:, 1])
 
    pxZ = l0["pt"] * np.cos(l0["phi"])  + l1["pt"] * np.cos(l1["phi"])
    pyZ = l0["pt"] * np.sin(l0["phi"])  + l1["pt"] * np.sin(l1["phi"])
    pzZ = l0["pt"] * np.sinh(l0["eta"]) + l1["pt"] * np.sinh(l1["eta"])
    EZ  = l0["energy"] + l1["energy"]
 
    return np.sqrt(np.maximum(EZ**2 - pxZ**2 - pyZ**2 - pzZ**2, 0))
 
 
# ─────────────────────────────────────────────
# NN — pT correction  (NEW)
# ─────────────────────────────────────────────

def build_correction_features(reco_pt, reco_eta, reco_phi):
    return np.column_stack([
        np.log(np.clip(reco_pt[:, 0], 1e-6, None)),
        reco_eta[:, 0],
        reco_phi[:, 0],
        np.log(np.clip(reco_pt[:, 1], 1e-6, None)),
        reco_eta[:, 1],
        reco_phi[:, 1],
    ])
 
 
def apply_pt_correction(reco_pt, reco_eta, reco_phi, model, scaler):
   
    X        = build_correction_features(reco_pt, reco_eta, reco_phi)
    X_scaled = scaler.transform(X)
    factors  = model.predict(X_scaled, batch_size=4096, verbose=0)   # shape (N, 2)
    return reco_pt * factors
 
 
# ─────────────────────────────────────────────
# MC processing pipeline
# ─────────────────────────────────────────────
 
def process_mc(
    file, sigma, wsum, threshold=0.5,
    L_int=8746231868.215154648 / 1e6,
    entry=843234,
    total_events=85388673,
    # pT correction kwargs
    apply_pt_correction_flag=False,
    pt_correction_model=None,
    pt_correction_scaler=None,
):
    
    events = load_dataset(file)

    electrons, weights = build_electrons(events)
 
    scale  = (sigma * L_int) * (entry / total_events) / wsum
    weight = weights * scale
 
    # ---- build reco array ------------------------------------------------
    reco_pt  = ak.to_numpy(electrons["pt_reco"])
    reco_eta = ak.to_numpy(electrons["eta_reco"])
    reco_phi = ak.to_numpy(electrons["phi_reco"])
 
    if apply_pt_correction_flag:
        if pt_correction_model is None or pt_correction_scaler is None:
            raise ValueError("Provide pt_correction_model and pt_correction_scaler.")
        reco_pt = apply_pt_correction(
            reco_pt, reco_eta, reco_phi,
            model=pt_correction_model,
            scaler=pt_correction_scaler,
        )
 
    # Recompute energy from (possibly corrected) pT
    reco_energy = np.sqrt(reco_pt**2 * np.cosh(reco_eta)**2)
 
    reco = ak.zip({
        "pt":     ak.from_numpy(reco_pt),
        "eta":    ak.from_numpy(reco_eta),
        "phi":    ak.from_numpy(reco_phi),
        "energy": ak.from_numpy(reco_energy),
    })
 
    truth = ak.zip({
        "pt":     electrons["pt_truth"],
        "eta":    electrons["eta_truth"],
        "phi":    electrons["phi_truth"],
        "energy": electrons["energy_truth"],
    })
 
    return reco, truth, weight
 
 


