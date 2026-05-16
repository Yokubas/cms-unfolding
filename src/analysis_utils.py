import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_dataset(root_file, max_events=None):
    
    file = uproot.open(root_file)
    tree = file["Events"]
    data = tree.arrays(library = "ak", entry_stop = max_events)

    return data

def build_electrons(events):
  
    electron = ak.zip(
        {
            "pt_reco": events["Electron_pt"],
            "pt_truth": events["GenDressedLepton_pt"]
            
        },
        with_name="Momentum4D"
    )
    weights = events["genWeight"] if "genWeight" in events.fields else None
    return electron, weights

def prepare_input(arr):
    df = pd.DataFrame()

    # df["nElectron"] = ak.to_numpy(ak.count_nonzero(arr["Electron_pt"] > 0, axis=1))
    # df["nJet"] = ak.to_numpy(ak.count_nonzero(arr["Jet_pt"] > 0, axis=1))

    # --- Electrons (leading 2) ---
    electron_features = ["pt", "eta", "phi", "sieie", "hoe", 
                         "dz", "dxy", "dr03TkSumPt", "scEtOverPt", 
                         "miniPFRelIso_all", "eInvMinusPInv"]
    
    for feature in electron_features:
        if feature == "pt":
            continue
        padded = ak.pad_none(arr[f"Electron_{feature}"], 2)
        for i in range(2):
            df[f"Electron{i+1}_{feature}"] = ak.to_numpy(ak.fill_none(padded[:, i], 0))

    padded_pt = ak.pad_none(arr["Electron_pt"], 2)

    pt1 = ak.to_numpy(ak.fill_none(padded_pt[:, 0], 0))
    pt2 = ak.to_numpy(ak.fill_none(padded_pt[:, 1], 0))

    df["Electron_pt_ratio"] = np.log(pt1/pt2)

    jet_features = ["pt", "eta", "phi", "btagDeepFlavB"]
    for feature in jet_features:
        padded = ak.pad_none(arr[f"Jet_{feature}"], 4)
        for i in range(4):
            df[f"Jet{i+1}_{feature}"] = ak.to_numpy(ak.fill_none(padded[:, i], 0))

    photon_features = ["pt", "eta", "phi", "sieie", "hoe", "pfRelIso03_all"]
    for feature in photon_features:
        padded = ak.pad_none(arr[f"Photon_{feature}"], 2)
        for i in range(2):
            df[f"Photon{i+1}_{feature}"] = ak.to_numpy(ak.fill_none(padded[:, i], 0))

    df["MET_phi"]          = ak.to_numpy(arr["MET_phi"])
    df["MET_sumEt"]        = ak.to_numpy(arr["MET_sumEt"])
    df["MET_significance"] = ak.to_numpy(arr["MET_significance"])

    return df.reset_index(drop=True)

def prepare_training(arr, label, process):
    n_events = len(arr["Electron_pt"])  # number of events in this dataset
    df = pd.DataFrame()
    
    df["label"] = [label] * n_events
    df["process"] = process

    # df["nElectron"] = ak.to_numpy(ak.count_nonzero(arr["Electron_pt"] > 0, axis=1))
    # df["nJet"] = ak.to_numpy(ak.count_nonzero(arr["Jet_pt"] > 0, axis=1))

    # --- Electrons (leading 2) ---
    electron_features = ["pt", "eta", "phi", "sieie", "hoe", 
                         "dz", "dxy", "dr03TkSumPt", "scEtOverPt", 
                         "miniPFRelIso_all", "eInvMinusPInv"]
    
    for feature in electron_features:
        if feature == "pt":
            continue
        padded = ak.pad_none(arr[f"Electron_{feature}"], 2)
        for i in range(2):
            df[f"Electron{i+1}_{feature}"] = ak.to_numpy(ak.fill_none(padded[:, i], 0))

    padded_pt = ak.pad_none(arr["Electron_pt"], 2)

    pt1 = ak.to_numpy(ak.fill_none(padded_pt[:, 0], 0))
    pt2 = ak.to_numpy(ak.fill_none(padded_pt[:, 1], 0))

    df["Electron_pt_ratio"] = np.log(pt1/pt2)

    jet_features = ["pt", "eta", "phi", "btagDeepFlavB"]
    for feature in jet_features:
        padded = ak.pad_none(arr[f"Jet_{feature}"], 4)
        for i in range(4):
            df[f"Jet{i+1}_{feature}"] = ak.to_numpy(ak.fill_none(padded[:, i], 0))

    photon_features = ["pt", "eta", "phi", "sieie", "hoe", "pfRelIso03_all"]
    for feature in photon_features:
        padded = ak.pad_none(arr[f"Photon_{feature}"], 2)
        for i in range(2):
            df[f"Photon{i+1}_{feature}"] = ak.to_numpy(ak.fill_none(padded[:, i], 0))
    
    df["MET_phi"]          = ak.to_numpy(arr["MET_phi"])
    df["MET_sumEt"]        = ak.to_numpy(arr["MET_sumEt"])
    df["MET_significance"] = ak.to_numpy(arr["MET_significance"])

    return df.reset_index(drop=True)

def process_mc(file, sigma, wsum, label, apply_nn_flag=False, model=None, scaler=None, threshold=0.5, L_int = 8746231868.215154648 / 1e6, entry = 843234, total_events = 85388673):
    events = load_dataset(file)

    if apply_nn_flag:
        events = apply_nn(events, model=model, scaler=scaler, threshold=threshold)

    electrons, weights = build_electrons(events)

    scale = (sigma * L_int) * (entry/total_events) / wsum

    return {
        "label": label,
        "pt_reco": electrons["pt_reco"],
        "pt_truth": electrons["pt_truth"],
        "weights": weights * scale,
        "events": events
    }

def apply_nn(events, model, scaler, threshold):
    prepared = prepare_input(events)
    X = prepared.values
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled, batch_size=1024)
    mask = y_pred.flatten() > threshold
    return events[mask]
