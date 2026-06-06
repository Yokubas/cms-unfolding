#include <TChain.h>
#include <TH1F.h>
#include <TFile.h>
#include <TTree.h>

#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <numeric> 
#include <algorithm>
#include <filesystem>
#include <cmath>
using namespace std;

TChain* createChainFromFileList(const char* txtFileName, int maxFiles = -1) {
    TChain* chain = new TChain("Events");
    ifstream infile(txtFileName);
    string line;
    int count = 0;
    while (getline(infile, line)) {
        if (maxFiles != -1 && count >= maxFiles) break;
        chain->Add(line.c_str());
        count++;
    }
    return chain;
}

float deltaPhi(float phi1, float phi2) {
    float dphi = phi1 - phi2;

    while (dphi > M_PI)  dphi -= 2*M_PI;
    while (dphi < -M_PI) dphi += 2*M_PI;

    return dphi;
}

float deltaR(float eta1, float phi1, float eta2, float phi2) {
    float dEta = eta1 - eta2;
    float dPhi = deltaPhi(phi1, phi2);

    return sqrt(dEta*dEta + dPhi*dPhi);
}

const float isoCut = 0.15;

void writeRootFile(TTree* tree, const char* outFileName, double& sumGenWeight){
    
    tree->SetBranchStatus("*", false);
    tree->SetBranchStatus("Electron_charge", true);
    tree->SetBranchStatus("Electron_cutBased", true);
    tree->SetBranchStatus("Electron_pfRelIso03_all", true);
    tree->SetBranchStatus("nElectron", true);
    tree->SetBranchStatus("Electron_phi", true);
    tree->SetBranchStatus("Electron_eta", true);
    tree->SetBranchStatus("Electron_pt", true);
    tree->SetBranchStatus("HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ", true);
    // Training
    tree->SetBranchStatus("Electron_miniPFRelIso_all", true);
    tree->SetBranchStatus("Electron_sieie", true);
    tree->SetBranchStatus("Electron_dxy", true);
    tree->SetBranchStatus("Electron_dz", true);
    tree->SetBranchStatus("Electron_hoe", true);
    tree->SetBranchStatus("Electron_scEtOverPt", true);
    tree->SetBranchStatus("Electron_eInvMinusPInv", true);
    tree->SetBranchStatus("Electron_r9", true);
    tree->SetBranchStatus("Electron_deltaEtaSC", true);
    
    tree->SetBranchStatus("GenDressedLepton_eta", true);
    tree->SetBranchStatus("GenDressedLepton_phi", true);
    tree->SetBranchStatus("GenDressedLepton_pt", true);
    tree->SetBranchStatus("nGenDressedLepton", true);

    Int_t Electron_charge[100];
    UInt_t nElectron;
    Float_t out_genWeight;
    Float_t Electron_phi[100], Electron_eta[100], Electron_pt[100], Electron_pfRelIso03_all[100];
    // Training
    Float_t Electron_miniPFRelIso_all[100], Electron_sieie[100], Electron_dxy[100], Electron_dz[100], Electron_hoe[100], Electron_scEtOverPt[100];
    Float_t Electron_eInvMinusPInv[100], Electron_r9[100], Electron_deltaEtaSC[100];
    
    Int_t Electron_cutBased[100];
    Float_t genWeight = 1.0;
    Bool_t HLT_Ele23_Ele12;

    Float_t GenDressedLepton_pt[100];
    Float_t GenDressedLepton_eta[100];
    Float_t GenDressedLepton_phi[100];
    UInt_t nGenDressedLepton;

    // Training out
    Float_t out_Electron_miniPFRelIso_all[2], out_Electron_sieie[2], out_Electron_dxy[2], out_Electron_dz[2], out_Electron_hoe[2], out_Electron_scEtOverPt[2];
    Float_t out_Electron_eInvMinusPInv[2], out_Electron_r9[2], out_Electron_deltaEtaSC[2];

    Float_t out_Electron_pt[2], out_Electron_phi[2], out_Electron_eta[2];
    Float_t out_GenDressedLepton_pt[2], out_GenDressedLepton_phi[2], out_GenDressedLepton_eta[2];
    sumGenWeight = 0;

    tree->SetBranchStatus("genWeight", true);
    tree->SetBranchAddress("genWeight", &genWeight);    

    tree->SetBranchAddress("Electron_charge", &Electron_charge);
    tree->SetBranchAddress("nElectron", &nElectron);
    tree->SetBranchAddress("Electron_pt", &Electron_pt);
    tree->SetBranchAddress("Electron_eta", &Electron_eta);
    tree->SetBranchAddress("Electron_phi", &Electron_phi);
    tree->SetBranchAddress("Electron_cutBased", &Electron_cutBased);
    tree->SetBranchAddress("Electron_pfRelIso03_all", &Electron_pfRelIso03_all);
    tree->SetBranchAddress("HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ", &HLT_Ele23_Ele12);

    tree->SetBranchAddress("Electron_miniPFRelIso_all", &Electron_miniPFRelIso_all);
    tree->SetBranchAddress("Electron_sieie", &Electron_sieie);
    tree->SetBranchAddress("Electron_dxy", &Electron_dxy);
    tree->SetBranchAddress("Electron_dz", &Electron_dz);
    tree->SetBranchAddress("Electron_hoe", &Electron_hoe);
    tree->SetBranchAddress("Electron_scEtOverPt", &Electron_scEtOverPt);
    tree->SetBranchAddress("Electron_eInvMinusPInv", &Electron_eInvMinusPInv);
    tree->SetBranchAddress("Electron_r9", &Electron_r9);
    tree->SetBranchAddress("Electron_deltaEtaSC", &Electron_deltaEtaSC);

    tree->SetBranchAddress("GenDressedLepton_eta", &GenDressedLepton_eta);
    tree->SetBranchAddress("GenDressedLepton_phi", &GenDressedLepton_phi);
    tree->SetBranchAddress("GenDressedLepton_pt", &GenDressedLepton_pt);
    tree->SetBranchAddress("nGenDressedLepton", &nGenDressedLepton);

    TFile *outFile = new TFile(outFileName,"RECREATE");
    TTree *outTree = new TTree("Events","Selected branches");

    outTree->Branch("Electron_pt", &out_Electron_pt, "Electron_pt[2]/F");
    outTree->Branch("GenDressedLepton_pt", &out_GenDressedLepton_pt, "GenDressedLepton_pt[2]/F");
    outTree->Branch("Electron_phi", &out_Electron_phi, "Electron_phi[2]/F");
    outTree->Branch("GenDressedLepton_phi", &out_GenDressedLepton_phi, "GenDressedLepton_phi[2]/F");
    outTree->Branch("Electron_eta", &out_Electron_eta, "Electron_eta[2]/F");
    outTree->Branch("GenDressedLepton_eta", &out_GenDressedLepton_eta, "GenDressedLepton_eta[2]/F");

    outTree->Branch("Electron_miniPFRelIso_all", &out_Electron_miniPFRelIso_all, "Electron_miniPFRelIso_all[2]/F");
    outTree->Branch("Electron_sieie", &out_Electron_sieie, "Electron_sieie[2]/F");
    outTree->Branch("Electron_dxy", &out_Electron_dxy, "Electron_dxy[2]/F");
    outTree->Branch("Electron_dz", &out_Electron_dz, "Electron_dz[2]/F");
    outTree->Branch("Electron_hoe", &out_Electron_hoe, "Electron_hoe[2]/F");
    outTree->Branch("Electron_scEtOverPt", &out_Electron_scEtOverPt, "Electron_scEtOverPt[2]/F");
    outTree->Branch("Electron_eInvMinusPInv", &out_Electron_eInvMinusPInv, "Electron_eInvMinusPInv[2]/F");
    outTree->Branch("Electron_r9", &out_Electron_r9, "Electron_r9[2]/F");
    outTree->Branch("Electron_deltaEtaSC", &out_Electron_deltaEtaSC, "Electron_deltaEtaSC[2]/F");

    outTree->Branch("genWeight", &out_genWeight, "genWeight/F");

    for (int iEntry = 0; tree->LoadTree(iEntry) >= 0; ++iEntry){    
        
        tree->GetEntry(iEntry);

        sumGenWeight += genWeight;
    
        if (!HLT_Ele23_Ele12) continue;

        vector<int> goodIndices;

        for (UInt_t i = 0; i < nElectron; ++i){
            Bool_t isGoodElectron = (Electron_cutBased[i] >= 3) && (Electron_pfRelIso03_all[i] < isoCut);
            if (isGoodElectron) {
                goodIndices.push_back(i);
            }
        }

        if (goodIndices.size() > 1){

            sort(goodIndices.begin(), goodIndices.end(), [&](int a, int b) {
                return Electron_pt[a] > Electron_pt[b];
            });
            
            int leading = goodIndices[0];
            int subleading = goodIndices[1];

            if (Electron_charge[leading] * Electron_charge[subleading] < 0) {

                if (Electron_pt[leading] > 28 && Electron_pt[subleading] > 20 && 
                    fabs(Electron_eta[leading]) < 2.5 && fabs(Electron_eta[subleading]) < 2.5 && 
                    !(fabs(Electron_eta[leading]) > 1.4442 && fabs(Electron_eta[leading]) < 1.566) &&
                    !(fabs(Electron_eta[subleading]) > 1.4442 && fabs(Electron_eta[subleading]) < 1.566)) {

                        float bestDR1 = 999.0;
                        int bestGen1 = -1;

                        float bestDR2 = 999.0;
                        int bestGen2 = -1;

                        for (UInt_t g = 0; g < nGenDressedLepton; g++) {

                            float dr1 = deltaR(
                                Electron_eta[leading], Electron_phi[leading],
                                GenDressedLepton_eta[g], GenDressedLepton_phi[g]
                            );

                            if (dr1 < bestDR1) {
                                bestDR1 = dr1;
                                bestGen1 = g;
                            }

                            float dr2 = deltaR(
                                Electron_eta[subleading], Electron_phi[subleading],
                                GenDressedLepton_eta[g], GenDressedLepton_phi[g]
                            );

                            if (dr2 < bestDR2) {
                                bestDR2 = dr2;
                                bestGen2 = g;
                            }
                        }
                        if (bestGen1 == bestGen2) continue;
                        if (bestGen1 < 0 || bestGen2 < 0) continue;
                        if (bestDR1 > 0.1 || bestDR2 > 0.1) continue;                        
                        
                        out_GenDressedLepton_pt[0] = GenDressedLepton_pt[bestGen1];
                        out_GenDressedLepton_pt[1] = GenDressedLepton_pt[bestGen2];

                        out_GenDressedLepton_eta[0] = GenDressedLepton_eta[bestGen1];
                        out_GenDressedLepton_eta[1] = GenDressedLepton_eta[bestGen2];

                        out_GenDressedLepton_phi[0] = GenDressedLepton_phi[bestGen1];
                        out_GenDressedLepton_phi[1] = GenDressedLepton_phi[bestGen2];

                        out_Electron_pt[0]  = Electron_pt[leading];
                        out_Electron_pt[1]  = Electron_pt[subleading];

                        out_Electron_eta[0]  = Electron_eta[leading];
                        out_Electron_eta[1]  = Electron_eta[subleading];

                        out_Electron_phi[0]  = Electron_phi[leading];
                        out_Electron_phi[1]  = Electron_phi[subleading];
                        
                        out_Electron_miniPFRelIso_all[0] = Electron_miniPFRelIso_all[leading];
                        out_Electron_miniPFRelIso_all[1] = Electron_miniPFRelIso_all[subleading];

                        out_Electron_sieie[0] = Electron_sieie[leading];
                        out_Electron_sieie[1] = Electron_sieie[subleading];

                        out_Electron_dxy[0] = Electron_dxy[leading];
                        out_Electron_dxy[1] = Electron_dxy[subleading];

                        out_Electron_dz[0] = Electron_dz[leading];
                        out_Electron_dz[1] = Electron_dz[subleading];

                        out_Electron_hoe[0] = Electron_hoe[leading];
                        out_Electron_hoe[1] = Electron_hoe[subleading];

                        out_Electron_scEtOverPt[0] = Electron_scEtOverPt[leading];
                        out_Electron_scEtOverPt[1] = Electron_scEtOverPt[subleading];

                        out_Electron_eInvMinusPInv[0] = Electron_eInvMinusPInv[leading];
                        out_Electron_eInvMinusPInv[1] = Electron_eInvMinusPInv[subleading];

                        out_Electron_r9[0] = Electron_r9[leading];
                        out_Electron_r9[1] = Electron_r9[subleading];
                        
                        out_Electron_deltaEtaSC[0] = Electron_deltaEtaSC[leading];
                        out_Electron_deltaEtaSC[1] = Electron_deltaEtaSC[subleading];

                        out_genWeight = genWeight;

                        outTree->Fill();
                }
            }    
        }
    }
    outFile->Write();
    outFile->Close();
}

int main() {
    double wsum = 0.0;

    TChain* mcTreeHigh = createChainFromFileList("../data/raw/CMS_mc_RunIISummer20UL16NanoAODv9_DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_30000_file_index.txt", 3);  
    TChain* mcTreeLow = createChainFromFileList("../data/raw/CMS_mc_RunIISummer20UL16NanoAODv9_DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_2520000_file_index.txt", 3);
    
    writeRootFile(mcTreeHigh, "../data/processed/mcDYhigh.root", wsum);
    cout << "Wsum for mcTreeHigh: " << wsum << endl;

    writeRootFile(mcTreeLow, "../data/processed/mcDYlow.root", wsum);
    cout << "Wsum for mcTreeLow: " << wsum << endl;

    return 0;
}