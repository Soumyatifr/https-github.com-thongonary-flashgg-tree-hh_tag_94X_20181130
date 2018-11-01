#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "flashgg/DataFormats/interface/Electron.h"
#include "flashgg/DataFormats/interface/Muon.h"
#include "flashgg/DataFormats/interface/Photon.h"
#include "flashgg/DataFormats/interface/Met.h"
#include "flashgg/Taggers/interface/LeptonSelection.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "flashgg/DataFormats/interface/Jet.h"
#include "flashgg/DataFormats/interface/DiPhotonCandidate.h"
#include "flashgg/DataFormats/interface/VertexCandidateMap.h"
#include "flashgg/DataFormats/interface/Jet.h"

#include "RecoJets/JetProducers/interface/PileupJetIdAlgo.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <TMath.h>

#include "DNN/Tensorflow/interface/Graph.h"
#include "DNN/Tensorflow/interface/Tensor.h"

#include "DataFormats/Math/interface/LorentzVector.h"
#include "TLorentzVector.h"

#define debug 0

using namespace std;
using namespace edm;

namespace flashgg {

    class DoubleHttHTagProducer80 : public EDProducer
    {

        public:
            typedef math::XYZTLorentzVector LorentzVector;

            DoubleHttHTagProducer80( const ParameterSet & );
            ~DoubleHttHTagProducer80(){};
            void InitEvent();
            void StandardizeInputVar();
            void SetNNVectorVar();
            double EvaluateNN();
        private:
            void produce( Event &, const EventSetup & ) override;
            double HelicityCosTheta(TLorentzVector Booster, TLorentzVector Boosted);
            double getCosThetaStar_CS(TLorentzVector h1, TLorentzVector h2, double ebeam);
            bool isExtraJet(const flashgg::Jet *jet,  std::vector<flashgg::Jet> DiJet);
            std::vector<double> XttCalculation(std::vector<flashgg::Jet> allJetsCol, std::vector<flashgg::Jet> DiJet);
            bool isclose(double a, double b, double rel_tol, double abs_tol);
            
            EDGetTokenT<View<flashgg::Jet> > jetToken_;
            EDGetTokenT<View<flashgg::DiPhotonCandidate> > diPhotonToken_;
            EDGetTokenT<View<flashgg::Electron> > electronToken_;
            EDGetTokenT<View<flashgg::Muon> > muonToken_;
            EDGetTokenT<View<flashgg::Photon> > photonToken_;
            EDGetTokenT<View<reco::Vertex> > vertexToken_;
            EDGetTokenT<View<reco::GenParticle> > genParticleToken_;
            EDGetTokenT<View<flashgg::Met> > METToken_;
            edm::EDGetTokenT<double> rhoToken_;        

            double minLeadPhoPt_, minSubleadPhoPt_;
            bool scalingPtCuts_, doPhotonId_, doMVAFlattening_, doCategorization_;
            double photonIDCut_;
            double vetoConeSize_;         
            double minJetPt_;
            double maxJetEta_;
            vector<double> mjjBoundaries_;
            vector<double> mjjBoundariesLower_;
            vector<double> mjjBoundariesUpper_;
            vector<int> photonElectronVeto_;
            std::string bTagType_;
            bool       useJetID_;
            string     JetIDLevel_;        

            string ttHWeightfileName_;
            std::vector<double> x_mean_, x_std_;

            dnn::tf::Graph NNgraph_;
            std::vector<double> HLF_VectorVar_; 
            std::vector<std::vector<double>> PL_VectorVar_; 
            std::vector<double> Xtt;

            //leptons selection
            double muPtThreshold;
            double muEtaThreshold;
            double muPFIsoSumRelThreshold; 

            double dRPhoElectronThreshold;
            double dRPhoMuonThreshold;
            double dRJetLeptonThreshold;

            double elecPtThreshold;  
            bool useElecMVARecipe; 
            bool useElecLooseId;
            std::vector<double> elecEtaThresholds;
            
            // NN variables
            float sumET;
            float MET;
            float phiMET;
            float dPhi1;
            float dPhi2;
            float PhoJetMinDr;
            float njets;
            float Xtt0;
            float Xtt1;
            float pte1;
            float pte2;
            float ptmu1;
            float ptmu2;
            float ptdipho;
            float etae1;
            float etae2;
            float etamu1;
            float etamu2;
            float etadipho;
            float phie1;
            float phie2;
            float phimu1;
            float phimu2;
            float phidipho;
            float fabs_CosThetaStar_CS;
            float fabs_CosTheta_bb;
    };


    DoubleHttHTagProducer80::DoubleHttHTagProducer80( const ParameterSet &iConfig ) :
        jetToken_ (consumes<View<flashgg::Jet> >(iConfig.getParameter<edm::InputTag>( "JetTag" ))),
        diPhotonToken_( consumes<View<flashgg::DiPhotonCandidate> >( iConfig.getParameter<InputTag> ( "DiPhotonTag" ) ) ),
        electronToken_( consumes<View<flashgg::Electron> >( iConfig.getParameter<InputTag>( "ElectronTag" ) ) ),
        muonToken_( consumes<View<flashgg::Muon> >( iConfig.getParameter<InputTag>( "MuonTag" ) ) ),
        vertexToken_( consumes<View<reco::Vertex> >( iConfig.getParameter<InputTag> ( "VertexTag" ) ) ),
        METToken_( consumes<View<flashgg::Met> >( iConfig.getParameter<InputTag> ( "METTag" ) ) ),
        rhoToken_( consumes<double>(iConfig.getParameter<edm::InputTag>( "rhoFixedGridCollection" ) ) ),

        minLeadPhoPt_( iConfig.getParameter<double> ( "MinLeadPhoPt" ) ),
        minSubleadPhoPt_( iConfig.getParameter<double> ( "MinSubleadPhoPt" ) ),
        scalingPtCuts_( iConfig.getParameter<bool> ( "ScalingPtCuts" ) ),
        vetoConeSize_( iConfig.getParameter<double> ( "VetoConeSize" ) ),
        minJetPt_( iConfig.getParameter<double> ( "MinJetPt" ) ),
        maxJetEta_( iConfig.getParameter<double> ( "MaxJetEta" ) ),
        bTagType_( iConfig.getUntrackedParameter<std::string>( "BTagType") ),
        useJetID_( iConfig.getParameter<bool>   ( "UseJetID"     ) ),
        JetIDLevel_( iConfig.getParameter<string> ( "JetIDLevel"   ) ),

        ttHWeightfileName_( iConfig.getUntrackedParameter<std::string>("ttHWeightfile"))
        
        {
            mjjBoundaries_ = iConfig.getParameter<vector<double > >( "MJJBoundaries" ); 
            mjjBoundariesLower_ = iConfig.getParameter<vector<double > >( "MJJBoundariesLower" ); 
            mjjBoundariesUpper_ = iConfig.getParameter<vector<double > >( "MJJBoundariesUpper" ); 
            doPhotonId_ = iConfig.getUntrackedParameter<bool>("ApplyEGMPhotonID");        
            photonIDCut_ = iConfig.getParameter<double>("PhotonIDCut");
            photonElectronVeto_ = iConfig.getUntrackedParameter<std::vector<int > >("PhotonElectronVeto");
            
            //leptons selection
            muPtThreshold = iConfig.getParameter<double>("leptonPtThreshold");
            muEtaThreshold = iConfig.getParameter<double>("muonEtaThreshold");
            muPFIsoSumRelThreshold = iConfig.getParameter<double>("muPFIsoSumRelThreshold");

            dRPhoElectronThreshold = iConfig.getParameter<double>("deltaRPhoElectronThreshold");
            dRPhoMuonThreshold = iConfig.getParameter<double>("deltaRPhoMuonThreshold");
            dRJetLeptonThreshold = iConfig.getParameter<double>("deltaRJetLepThreshold");

            elecPtThreshold  = iConfig.getParameter<double>("leptonPtThreshold");
            useElecMVARecipe = iConfig.getParameter<bool>("useElectronMVARecipe"); 
            useElecLooseId = iConfig.getParameter<bool>("useElectronLooseID");
            elecEtaThresholds = iConfig.getParameter<std::vector<double > >("electronEtaThresholds");
            
            x_mean_ = iConfig.getParameter<std::vector<double>> ("mean");
            x_std_ = iConfig.getParameter<std::vector<double>> ("std");
            
            NNgraph_ = *(new dnn::tf::Graph(ttHWeightfileName_.c_str())); 

            sumET = 0.;
            MET = 0.;
            phiMET = 0.;
            dPhi1 = 0.;
            dPhi2 = 0.;
            PhoJetMinDr = 0.;
            njets = 0.;
            Xtt0 = 0.;
            Xtt1 = 0.;
            pte1 = 0.;
            pte2 = 0.;
            ptmu1 = 0.;
            ptmu2 = 0.;
            ptdipho = 0.;
            etae1 = 0.;
            etae2 = 0.;
            etamu1 = 0.;
            etamu2 = 0.;
            etadipho = 0.;
            phie1 = 0.;
            phie2 = 0.;
            phimu1 = 0.;
            phimu2 = 0.;
            phidipho = 0.;
            fabs_CosThetaStar_CS = 0.;
            fabs_CosTheta_bb = 0.;
            
            HLF_VectorVar_.resize(9);  // High-level features. 9 at the moment
            PL_VectorVar_.resize(6);
            for (int i = 0; i < 6; i++)
                PL_VectorVar_[i].resize(7); // List of particles. 6 objects. Each object has 7 attributes.
            Xtt.resize(6);
            // produces<vector<flashgg::Jet> > (); // Don't know what it is
        }

    // Common functions
    double DoubleHttHTagProducer80::HelicityCosTheta(TLorentzVector Booster, TLorentzVector Boosted)
    {
        TVector3 BoostVector = Booster.BoostVector();
        Boosted.Boost(-BoostVector.x(), -BoostVector.y(), -BoostVector.z());
        return Boosted.CosTheta();
    }

    double DoubleHttHTagProducer80::getCosThetaStar_CS(TLorentzVector h1, TLorentzVector h2, double ebeam) 
    {
        TLorentzVector p1, p2;
        p1.SetPxPyPzE(0, 0,  ebeam, ebeam);
        p2.SetPxPyPzE(0, 0, -ebeam, ebeam);

        TLorentzVector hh;
        hh = h1 + h2;

        TVector3 boost = - hh.BoostVector();
        p1.Boost(boost);
        p2.Boost(boost);
        h1.Boost(boost);

        TVector3 CSaxis = p1.Vect().Unit() - p2.Vect().Unit();
        CSaxis.Unit();

        return cos(CSaxis.Angle(h1.Vect().Unit()));
    }

    bool DoubleHttHTagProducer80::isExtraJet(const flashgg::Jet *jet,  std::vector<flashgg::Jet> DiJet)
    {
        bool isSel = 1;
        if( deltaR(jet->p4(), DiJet[0].p4()) < 0.1
                || deltaR(jet->p4(), DiJet[1].p4()) < 0.1 ) 
            isSel = 0;

        return isSel;
    }

    std::vector<double> DoubleHttHTagProducer80::XttCalculation(std::vector<flashgg::Jet> allJetsCol, std::vector<flashgg::Jet> DiJet)
    {
        double mW = 80.3;
        double mt = 173.5;

        std::vector<flashgg::Jet> jetsCol;
        for( unsigned int jetIndex = 0; jetIndex < allJetsCol.size(); jetIndex++ )
        {
            flashgg::Jet *jet = &(allJetsCol[jetIndex]);
            bool bExtraJet = isExtraJet(jet, DiJet);

            if (!bExtraJet) continue;
            jetsCol.push_back(*jet );
        }


        std::vector<double> Xtt(6,0);
        Xtt[0] = 1000, Xtt[1] = 0, Xtt[2] = 0, Xtt[3] = 1000, Xtt[4] = 0, Xtt[5] = 0;

        unsigned int WJetIndex00 = 1000,  WJetIndex01 = 1000, bJetIndex0 = 1000;


        for( unsigned int jetIndex0 = 0; jetIndex0 < jetsCol.size(); jetIndex0++ ) {
            //flashgg::Jet jet = jetsCol[jetIndex];
            const flashgg::Jet *jet0 = &(jetsCol[jetIndex0]);
            for( unsigned int jetIndex1 = jetIndex0+1; jetIndex1 < jetsCol.size(); jetIndex1++ ) {
                const flashgg::Jet *jet1 = &(jetsCol[jetIndex1]);

                LorentzVector Wcand = jet0->p4() + jet1->p4();
                LorentzVector tcand0 = Wcand + DiJet[0].p4();
                LorentzVector tcand1 = Wcand + DiJet[1].p4();

                float XW = TMath::Power((Wcand.M() - mW)/(0.1*mW),2);
                float XT0 = XW + TMath::Power((tcand0.M() - mt)/(0.1*mt),2);
                float XT1 = XW + TMath::Power((tcand1.M() - mt)/(0.1*mt),2);

                if (Xtt[0] > XT0) {
                    Xtt[0] = XT0; Xtt[1] = Wcand.M(); Xtt[2] = tcand0.M(); 
                    WJetIndex00 = jetIndex0, WJetIndex01 = jetIndex1, bJetIndex0 = 0;
                }
                if (Xtt[0] > XT1) {
                    Xtt[0] = XT1; Xtt[1] = Wcand.M(); Xtt[2] = tcand1.M(); 
                    WJetIndex00 = jetIndex0, WJetIndex01 = jetIndex1, bJetIndex0 = 1;
                }
            }
        }

        if (jetsCol.size() < 4) return Xtt;

        int  bJetIndex = 0;
        if ( bJetIndex0 == 0) bJetIndex = 1;

        for (unsigned int jetIndex0 = 0; jetIndex0 < jetsCol.size(); jetIndex0++ ) {
            //flashgg::Jet jet = jetsCol[jetIndex];
            const flashgg::Jet *jet0 = &(jetsCol[jetIndex0]);

            for( unsigned int jetIndex1 = jetIndex0+1; jetIndex1 < jetsCol.size(); jetIndex1++ ) {

                if( ( WJetIndex00 == jetIndex0 && WJetIndex01 == jetIndex1 ) || 
                        ( WJetIndex01 == jetIndex0 && WJetIndex00 == jetIndex1 )) continue;

                const flashgg::Jet *jet1 = &(jetsCol[jetIndex1]);

                LorentzVector Wcand = jet0->p4() + jet1->p4();
                LorentzVector tcand = Wcand + DiJet[bJetIndex].p4();

                float XW = TMath::Power((Wcand.M() - mW)/(0.1*mW),2);
                float XT = XW + TMath::Power((tcand.M() - mt)/(0.1*mt),2);

                if (Xtt[3] > XT) {
                    Xtt[3] = XT;  Xtt[4] = Wcand.M(); Xtt[5] = tcand.M(); 
                }
            }
        }
        return Xtt;
    }

    bool DoubleHttHTagProducer80::isclose(double a, double b, double rel_tol=1e-09, double abs_tol=0.0)
    {
        return fabs(a-b) <= max(rel_tol * max(fabs(a), fabs(b)), abs_tol);   
    }


    void DoubleHttHTagProducer80::produce( Event &evt, const EventSetup & )
    {
        unique_ptr<double> NNscore;
        
        InitEvent();

        Handle<View<flashgg::DiPhotonCandidate> > diPhotons;
        evt.getByToken( diPhotonToken_, diPhotons );

        Handle<View<flashgg::Muon> > theMuons;
        evt.getByToken( muonToken_, theMuons );

        Handle<View<flashgg::Electron> > theElectrons;
        evt.getByToken( electronToken_, theElectrons );

        edm::Handle<double>  rho;
        evt.getByToken(rhoToken_,rho);

        Handle<View<reco::Vertex> > vertices;
        evt.getByToken( vertexToken_, vertices );

        Handle<View<flashgg::Met> > METs;
        evt.getByToken( METToken_, METs );
        if (METs->size() != 1) std::cout << "WARNING number of MET is not equal to 1" << std::endl;
        edm::Ptr<flashgg::Met> theMET = METs->ptrAt( 0 ); 

        // Select diphoton candidate with highest sclar sum PT
        // Do we need any smearing/scaling? Here I follow the HH tagger, which doesn't apply it but the bbgg selection has it.
        float sumPt_ref = 0;
        int maxId = -1;

        for ( unsigned int candIndex = 0; candIndex < diPhotons->size() ; candIndex++ ) 
        {
            edm::Ptr<flashgg::DiPhotonCandidate> dipho = diPhotons->ptrAt( candIndex );

            // kinematic cuts on diphotons
            auto & leadPho = *(dipho->leadingPhoton());
            auto & subleadPho = *(dipho->subLeadingPhoton());

            double leadPt = leadPho.pt();
            double subleadPt = subleadPho.pt();
            double _leadPt = leadPt; 
            double _subleadPt = subleadPt;

            if ( scalingPtCuts_ ) {
                _leadPt = leadPt/dipho->mass();
                _subleadPt = subleadPt/dipho->mass();
            }
            if (_leadPt <= minLeadPhoPt_ || _subleadPt <= minSubleadPhoPt_) continue; 

            //apply egm photon id with given working point
            if (doPhotonId_)
            {
                if (leadPho.userFloat("EGMPhotonMVA")<photonIDCut_ || subleadPho.userFloat("EGMPhotonMVA")<photonIDCut_){
                    continue;
                }
            }

            //electron veto
            if (leadPho.passElectronVeto()<photonElectronVeto_[0] || subleadPho.passElectronVeto()<photonElectronVeto_[1]){
                continue;
            }

            // select the candidate with highest scalar sum PT
            double sumPt = leadPt + subleadPt;
            if (sumPt > sumPt_ref)
            {
                sumPt_ref = sumPt;
                maxId = candIndex;
            }

        } // end of diphoton loop
        if (maxId == -1) return;

        edm::Ptr<flashgg::DiPhotonCandidate> dipho = diPhotons->ptrAt(int(maxId));

        // find vertex associated to diphoton object
        // size_t vtx = (size_t)dipho->jetCollectionIndex();
        // size_t vtx = (size_t)dipho->vertexIndex();
        //  if( vtx >= jetTokens_.size() ) { vtx = 0; }
        // and read corresponding jet collection

        Handle<View<flashgg::Jet> > jets;
        evt.getByToken( jetToken_, jets );
        unique_ptr<vector<flashgg::Jet> > jetColl( new vector<flashgg::Jet> );

        // photon-jet cross-cleaning and pt/eta/btag/jetid cuts for jets
        std::vector<flashgg::Jet> cleaned_jets;
        for (size_t ijet=0; ijet < jets->size(); ++ijet) 
        {
            //jets are ordered in pt
            auto jet = jets->ptrAt(ijet);
            if (jet->pt()<minJetPt_ || fabs(jet->eta())>maxJetEta_) continue;
            if (jet->bDiscriminator(bTagType_)<0) continue; //FIXME threshold might not be 0?
            if (useJetID_) 
            {
                if( JetIDLevel_ == "Loose" && !jet->passesJetID  ( flashgg::Loose ) ) continue;
                if( JetIDLevel_ == "Tight" && !jet->passesJetID  ( flashgg::Tight ) ) continue;
            }
            if( reco::deltaR( *jet, *(dipho->leadingPhoton()) ) > vetoConeSize_ && reco::deltaR( *jet, *(dipho->subLeadingPhoton()) ) > vetoConeSize_ ) 
            {
                flashgg::Jet *thisJetPointer = const_cast<flashgg::Jet*>(jet.get());
                cleaned_jets.push_back(*thisJetPointer);
            }
        }
        if (cleaned_jets.size() < 2) return;

        ///dijet pair selection. Do pair according to pt and choose the pair with highest b-tag
        double sumbtag_ref = -999;
        bool hasDijet = false;
        flashgg::Jet jet1, jet2;
        std::vector<flashgg::Jet> SelJets;
        for( size_t ijet=0; ijet < cleaned_jets.size()-1;++ijet){
            auto jet_1 = cleaned_jets[ijet];
            for( size_t kjet=ijet+1; kjet < cleaned_jets.size();++kjet){
                auto jet_2 = cleaned_jets[kjet];
                auto dijet_mass = (jet_1.p4()+jet_2.p4()).mass(); 
                if (dijet_mass<mjjBoundaries_[0] || dijet_mass>mjjBoundaries_[1]) continue;
                double sumbtag = jet_1.bDiscriminator(bTagType_) + jet_2.bDiscriminator(bTagType_);
                if (sumbtag > sumbtag_ref) {
                    hasDijet = true;
                    sumbtag_ref = sumbtag;
                    jet1 = jet_1; // convert edm::Ptr<flashgg::Jet> to flashgg::Jets*
                    jet2 = jet_2;
                }
            }
        }
        if (!hasDijet) return;             
        SelJets.push_back(jet1);
        SelJets.push_back(jet2);
        
        flashgg::Jet leadingJet = jet1; 
        flashgg::Jet subleadingJet = jet2; 
        njets = double(cleaned_jets.size());

        // VBF jets candidates for sumET calculations. Basically the same as cleaned_jets except eta and btags requirement. 
        // Implementation follows https://github.com/ResonantHbbHgg/bbggTools/blob/70fc6367a5c1dab593cc321f77192e7378b29ecc/src/bbggTools.cc#L1106-L1141
        std::vector<flashgg::Jet> VBF_jets;
        for( size_t ijet=0; ijet < jets->size(); ++ijet ) {//jets are ordered in pt
            auto jet = jets->ptrAt(ijet); // I think jet is now edm::Ptr<flashgg::Jet>
            flashgg::Jet *thisJetPointer = const_cast<flashgg::Jet *>(jet.get()); // convert it to flashgg::Jet*
            if (thisJetPointer->pt()<minJetPt_ || fabs(thisJetPointer->eta())>maxJetEta_) continue;
            if (useJetID_) {
                if (JetIDLevel_ == "Loose" && !thisJetPointer->passesJetID  ( flashgg::Loose )) continue;
                if (JetIDLevel_ == "Tight" && !thisJetPointer->passesJetID  ( flashgg::Tight )) continue;
            }
            bool bExtraJet = isExtraJet(thisJetPointer, SelJets);
            if (!bExtraJet) continue;
            if( reco::deltaR( *jet, *(dipho->leadingPhoton()) ) > vetoConeSize_ && reco::deltaR( *jet, *(dipho->subLeadingPhoton()) ) > vetoConeSize_ ) {
                VBF_jets.push_back(*thisJetPointer);
            }
        }

        for (unsigned int jetIndex = 0; jetIndex < VBF_jets.size(); jetIndex++) 
        {
            sumET += VBF_jets[jetIndex].pt();
        }

        // Get pt, eta, and phi of dipho
        //dipho->computeP4AndOrder();
        ptdipho = dipho->p4().pt();
        etadipho = dipho->p4().eta();
        phidipho = dipho->p4().phi();

        // Get the angles

        LorentzVector dijet = leadingJet.p4() + subleadingJet.p4();
        LorentzVector diHiggsCandidate = dipho->p4() + dijet;
        std::vector<float> helicityThetas;
        
        // cos theta star
        TLorentzVector BoostedHgg(0,0,0,0);
        BoostedHgg.SetPtEtaPhiE( dipho->p4().pt(), dipho->p4().eta(), dipho->p4().phi(), dipho->p4().energy());
        TLorentzVector HHforBoost(0,0,0,0);
        HHforBoost.SetPtEtaPhiE(diHiggsCandidate.Pt(), diHiggsCandidate.Eta(), diHiggsCandidate.Phi(), diHiggsCandidate.energy());
        // double CosThetaStar = HelicityCosTheta(HHforBoost, BoostedHgg);

        // cos theta bb
        TLorentzVector BoostedLeadingJet(0,0,0,0);
        BoostedLeadingJet.SetPtEtaPhiE( leadingJet.p4().pt(), leadingJet.p4().eta(), leadingJet.p4().phi(), leadingJet.p4().energy()); 
        TLorentzVector HbbforBoost(0,0,0,0);
        HbbforBoost.SetPtEtaPhiE(dijet.pt(), dijet.eta(), dijet.phi(), dijet.energy());
        double CosTheta_bb = HelicityCosTheta(HbbforBoost, BoostedLeadingJet);

        // Colin Sopper Frame CTS
        TLorentzVector djc, dpc; // TLorentzVector and LorentzVector are not compatible, thanks to the whoever wrote it
        djc.SetPtEtaPhiE( dijet.pt(), dijet.eta(), dijet.phi(), dijet.energy());
        dpc.SetPtEtaPhiE( dipho->p4().pt(), dipho->p4().eta(), dipho->p4().phi(), dipho->p4().energy());
        double CosTheta_CS = getCosThetaStar_CS(djc, dpc, 6500.);

        fabs_CosThetaStar_CS = fabs(CosTheta_CS);
        fabs_CosTheta_bb = fabs(CosTheta_bb);


        // Xtt variables
        if (cleaned_jets.size() > 2)
            Xtt = XttCalculation(cleaned_jets, SelJets);
        else
            Xtt[0] = 1000, Xtt[1] = 0, Xtt[2] = 0, Xtt[3] = 1000, Xtt[4] = 0, Xtt[5] = 0;

        Xtt0 = Xtt[0], Xtt1 = Xtt[3]; 
        
        // Select muons
        std::vector<edm::Ptr<flashgg::Muon> > selectedMuons = flashgg::selectAllMuons(theMuons->ptrs(), vertices->ptrs(), muEtaThreshold, muPtThreshold, muPFIsoSumRelThreshold);    

        std::vector<edm::Ptr<flashgg::Muon> > goodMuons;
        double Phi_Pho1 = dipho->leadingPhoton()->superCluster()->phi();
        double Phi_Pho2 = dipho->subLeadingPhoton()->superCluster()->phi();
        double Eta_Pho1 = dipho->leadingPhoton()->superCluster()->eta();
        double Eta_Pho2 = dipho->subLeadingPhoton()->superCluster()->eta();

        double Phi_Jet1 =  leadingJet.p4().Phi();
        double Phi_Jet2 =  subleadingJet.p4().Phi();
        double Eta_Jet1 =  leadingJet.p4().Eta();
        double Eta_Jet2 =  subleadingJet.p4().Eta();

        for (unsigned int muonIndex = 0; muonIndex < selectedMuons.size(); muonIndex++) {
            edm::Ptr<flashgg::Muon> muon = selectedMuons[muonIndex];
            double Eta_Lepton = muon->eta(), Phi_Lepton = muon->phi();
            float dRLeadPhoLepton      = deltaR( Eta_Lepton, Phi_Lepton, Eta_Pho1, Phi_Pho1);
            float dRSubLeadPhoLepton   = deltaR( Eta_Lepton, Phi_Lepton, Eta_Pho2, Phi_Pho2);
            float dRLeadJetLepton    = deltaR( Eta_Lepton, Phi_Lepton, Eta_Jet1, Phi_Jet1);
            float dRSubLeadJetLepton = deltaR( Eta_Lepton, Phi_Lepton, Eta_Jet2, Phi_Jet2);
            if (dRLeadPhoLepton < dRPhoMuonThreshold || dRSubLeadPhoLepton < dRPhoMuonThreshold || 
                    dRLeadJetLepton < dRJetLeptonThreshold || dRSubLeadJetLepton < dRJetLeptonThreshold) continue; 

            goodMuons.push_back( muon );
        }

        if (goodMuons.size() > 0) 
        {
            ptmu1 = goodMuons.at(0)->p4().pt();
            etamu1 = goodMuons.at(0)->p4().eta();
            phimu1 = goodMuons.at(0)->p4().phi();
        }

        if (goodMuons.size() > 1)
        {
            ptmu2 = goodMuons.at(1)->p4().pt();
            etamu2 = goodMuons.at(1)->p4().eta();
            phimu2 = goodMuons.at(1)->p4().phi();
        } 

        // Select electrons
        std::vector<edm::Ptr<flashgg::Electron> > selectedElectrons = flashgg::selectStdAllElectrons( theElectrons->ptrs(), vertices->ptrs(), elecPtThreshold, elecEtaThresholds, useElecMVARecipe, useElecLooseId, *rho, evt.isRealData());
        std::vector<edm::Ptr<flashgg::Electron> > goodElectrons;
        for (unsigned int electronIndex = 0; electronIndex < selectedElectrons.size(); electronIndex++ ) {
            edm::Ptr<flashgg::Electron> electron = selectedElectrons[electronIndex];
            double Eta_Lepton = electron->superCluster()->eta(), Phi_Lepton = electron->superCluster()->phi();
            float dRLeadPhoLepton      = deltaR( Eta_Lepton, Phi_Lepton, Eta_Pho1, Phi_Pho1);
            float dRSubLeadPhoLepton   = deltaR( Eta_Lepton, Phi_Lepton, Eta_Pho2, Phi_Pho2);
            float dRLeadJetLepton    = deltaR( Eta_Lepton, Phi_Lepton, Eta_Jet1, Phi_Jet1);
            float dRSubLeadJetLepton = deltaR( Eta_Lepton, Phi_Lepton, Eta_Jet2, Phi_Jet2);

            if (dRLeadPhoLepton < dRPhoElectronThreshold || dRSubLeadPhoLepton < dRPhoElectronThreshold || 
                    dRLeadJetLepton < dRJetLeptonThreshold || dRSubLeadJetLepton < dRJetLeptonThreshold) continue; 

            goodElectrons.push_back( electron );
        }

        if (goodElectrons.size() > 0) 
        {
            pte1 = goodElectrons.at(0)->p4().pt();
            etae1 = goodElectrons.at(0)->p4().eta();
            phie1 = goodElectrons.at(0)->p4().phi();
        }

        if (goodElectrons.size() > 1)
        {
            pte2 = goodElectrons.at(1)->p4().pt();
            etae2 = goodElectrons.at(1)->p4().eta();
            phie2 = goodElectrons.at(1)->p4().phi();
        } 

        // Get MET info
        LorentzVector p4MET = theMET->p4();
        MET = p4MET.Pt();
        phiMET = p4MET.Phi();
        dPhi1 = deltaPhi(p4MET.Phi(), leadingJet.p4().Phi());
        dPhi2 = deltaPhi(p4MET.Phi(), subleadingJet.p4().Phi());

        PhoJetMinDr = min( min( reco::deltaR( *(dipho->leadingPhoton()), leadingJet ), reco::deltaR( *(dipho->leadingPhoton()), subleadingJet ) ),
                min(reco::deltaR( *(dipho->subLeadingPhoton()), leadingJet ), reco::deltaR( *(dipho->subLeadingPhoton()), subleadingJet ) ) );

        // Evaluate the network
        *NNscore = -999.;
        
        StandardizeInputVar();
        SetNNVectorVar();
        *NNscore = EvaluateNN();
        HLF_VectorVar_.clear();
        PL_VectorVar_.clear();

        evt.put(std::move(NNscore), "ttHTaggerScore");
    }

    void DoubleHttHTagProducer80::InitEvent(){

        sumET = 0.;
        MET = 0.;
        phiMET = 0.;
        dPhi1 = 0.;
        dPhi2 = 0.;
        PhoJetMinDr = 0.;
        njets = 0.;
        Xtt0 = 0.;
        Xtt1 = 0.;
        pte1 = 0.;
        pte2 = 0.;
        ptmu1 = 0.;
        ptmu2 = 0.;
        ptdipho = 0.;
        etae1 = 0.;
        etae2 = 0.;
        etamu1 = 0.;
        etamu2 = 0.;
        etadipho = 0.;
        phie1 = 0.;
        phie2 = 0.;
        phimu1 = 0.;
        phimu2 = 0.;
        phidipho = 0.;
        fabs_CosThetaStar_CS = 0.;
        fabs_CosTheta_bb = 0.;
        Xtt[0] = 1000, Xtt[1] = 0, Xtt[2] = 0, Xtt[3] = 1000, Xtt[4] = 0, Xtt[5] = 0;


    }//end InitEvent

    void DoubleHttHTagProducer80::StandardizeInputVar()
    {
        if (!isclose(sumET,0)) sumET = (sumET - x_mean_[0])/x_std_[0];
        if (!isclose(phiMET,0)) phiMET = (phiMET - x_mean_[1])/x_std_[1];
        if (!isclose(dPhi1,0)) dPhi1 = (dPhi1 - x_mean_[2])/x_std_[2];
        if (!isclose(dPhi2,0)) dPhi2 = (dPhi2 - x_mean_[3])/x_std_[3];
        if (!isclose(PhoJetMinDr,0)) PhoJetMinDr = (PhoJetMinDr - x_mean_[4])/x_std_[4];
        if (!isclose(njets,0)) njets = (njets - x_mean_[5])/x_std_[5];
        if (!isclose(Xtt0,0)) Xtt0 = (Xtt0 - x_mean_[6])/x_std_[6];
        if (!isclose(Xtt1,0)) Xtt1 = (Xtt1 - x_mean_[7])/x_std_[7];
        if (!isclose(pte1,0)) pte1 = (pte1 - x_mean_[8])/x_std_[8];
        if (!isclose(pte2,0)) pte2 = (pte2 - x_mean_[9])/x_std_[9];
        if (!isclose(ptmu1,0)) ptmu1 = (ptmu1 - x_mean_[10])/x_std_[10];
        if (!isclose(ptmu2,0)) ptmu2 = (ptmu2 - x_mean_[11])/x_std_[11];
        if (!isclose(ptdipho,0)) ptdipho = (ptdipho - x_mean_[12])/x_std_[12];
        if (!isclose(etae1,0)) etae1 = (etae1 - x_mean_[13])/x_std_[13];
        if (!isclose(etae2,0)) etae2 = (etae2 - x_mean_[14])/x_std_[14];
        if (!isclose(etamu1,0)) etamu1 = (etamu1 - x_mean_[15])/x_std_[15];
        if (!isclose(etamu2,0)) etamu2 = (etamu2 - x_mean_[16])/x_std_[16];
        if (!isclose(etadipho,0)) etadipho = (etadipho - x_mean_[17])/x_std_[17];
        if (!isclose(phie1,0)) phie1 = (phie1 - x_mean_[18])/x_std_[18];
        if (!isclose(phie2,0)) phie2 = (phie2 - x_mean_[19])/x_std_[19];
        if (!isclose(phimu1,0)) phimu1 = (phimu1 - x_mean_[20])/x_std_[20];
        if (!isclose(phimu2,0)) phimu2 = (phimu2 - x_mean_[21])/x_std_[21];
        if (!isclose(phidipho,0)) phidipho = (phidipho - x_mean_[22])/x_std_[22];
        if (!isclose(fabs_CosThetaStar_CS,0)) fabs_CosThetaStar_CS = (fabs_CosThetaStar_CS - x_mean_[23])/x_std_[23];
        if (!isclose(fabs_CosTheta_bb,0)) fabs_CosTheta_bb = (fabs_CosTheta_bb - x_mean_[24])/x_std_[24];
    }

    void DoubleHttHTagProducer80::SetNNVectorVar()
    {
        //9 HLFs: 'sumEt','dPhi1','dPhi2','PhoJetMinDr','njets','Xtt0',
        //'Xtt1','fabs_CosThetaStar_CS','fabs_CosTheta_bb'
        HLF_VectorVar_[0] = sumET;
        HLF_VectorVar_[1] = dPhi1;
        HLF_VectorVar_[2] = dPhi2;
        HLF_VectorVar_[3] = PhoJetMinDr;
        HLF_VectorVar_[4] = njets;
        HLF_VectorVar_[5] = Xtt0;
        HLF_VectorVar_[6] = Xtt1;
        HLF_VectorVar_[7] = fabs_CosThetaStar_CS;
        HLF_VectorVar_[8] = fabs_CosTheta_bb;
     
        // 6 objects: ele1, ele2, mu1, mu2, dipho, MET
        // Each object has 7 attributes: pt, eta, phi, isele, ismuon, isdipho, isMET
        //
        // 0: leading ele
        PL_VectorVar_[0][0] = pte1;
        PL_VectorVar_[0][1] = etae1;
        PL_VectorVar_[0][2] = phie1;
        PL_VectorVar_[0][3] = (isclose(pte1,0)) ? 0 : 1; // isEle
        PL_VectorVar_[0][4] = 0; // isMuon
        PL_VectorVar_[0][5] = 0; // isDiPho
        PL_VectorVar_[0][6] = 0; // isMET

        // 1: subleading ele
        PL_VectorVar_[1][0] = pte2;
        PL_VectorVar_[1][1] = etae2;
        PL_VectorVar_[1][2] = phie2;
        PL_VectorVar_[1][3] = (isclose(pte2,0)) ? 0 : 1; // isEle
        PL_VectorVar_[1][4] = 0; // isMuon
        PL_VectorVar_[1][5] = 0; // isDiPho
        PL_VectorVar_[1][6] = 0; // isMET

        // 2: leading muon
        PL_VectorVar_[2][0] = ptmu1;
        PL_VectorVar_[2][1] = etamu1;
        PL_VectorVar_[2][2] = phimu1;
        PL_VectorVar_[2][3] = 0; // isEle
        PL_VectorVar_[2][4] = (isclose(ptmu1,0)) ? 0 : 1; // isMuon
        PL_VectorVar_[2][5] = 0; // isDiPho
        PL_VectorVar_[2][6] = 0; // isMET

        // 3: subleading muon
        PL_VectorVar_[3][0] = ptmu2;
        PL_VectorVar_[3][1] = etamu2;
        PL_VectorVar_[3][2] = phimu2;
        PL_VectorVar_[3][3] = 0; //isEle
        PL_VectorVar_[3][4] = (isclose(ptmu2,0)) ? 0 : 1; // isMuon
        PL_VectorVar_[3][5] = 0; // isDiPho
        PL_VectorVar_[3][6] = 0; // isMET

        // 4: dipho
        PL_VectorVar_[4][0] = ptdipho;
        PL_VectorVar_[4][1] = etadipho;
        PL_VectorVar_[4][2] = phidipho;
        PL_VectorVar_[4][3] = 0; // isEle
        PL_VectorVar_[4][4] = 0; // isMuon
        PL_VectorVar_[4][5] = (isclose(ptdipho,0)) ? 0 : 1; // isDiPho
        PL_VectorVar_[4][6] = 0; // isMET

        // 5: MET
        PL_VectorVar_[5][0] = MET;
        PL_VectorVar_[5][1] = 0; // MET eta
        PL_VectorVar_[5][2] = phiMET;
        PL_VectorVar_[5][3] = 0; //isEle
        PL_VectorVar_[5][4] = 0; // isMuon
        PL_VectorVar_[5][5] = 0; // isDiPho
        PL_VectorVar_[5][6] = (isclose(MET,0)) ? 0 : 1; // isMET

        // Sort by pT
        std::sort(PL_VectorVar_.rbegin(), PL_VectorVar_.rend()); 
    }


    double DoubleHttHTagProducer80::EvaluateNN()
    {
        unsigned int shape = HLF_VectorVar_.size();
        dnn::tf::Shape xShape[] = { 1, shape }; 
        dnn::tf::Tensor* hlf_input = NNgraph_.defineInput(new dnn::tf::Tensor("HLF:0", 2, xShape));
        unsigned int plshape1 = PL_VectorVar_.size();
        unsigned int plshape2 = PL_VectorVar_[0].size();
        dnn::tf::Shape plShape[] = { 1, plshape1, plshape2 };
        dnn::tf::Tensor* pl_input = NNgraph_.defineInput(new dnn::tf::Tensor("input:0", 3, plShape));
        
        dnn::tf::Tensor* y = NNgraph_.defineOutput(new dnn::tf::Tensor("Final_output/Sigmoid:0"));
        for (int i = 0; i < hlf_input->getShape(1); i++){
            hlf_input->setValue<float>(0, i, HLF_VectorVar_[i]);
        }
        for (int i = 0; i < pl_input->getShape(1); i++)
            for (int j = 0; j < pl_input->getShape(2); j++)
                pl_input->setValue<float>(0, i, j, PL_VectorVar_[i][j]);       
        
        NNgraph_.eval();
        double NNscore = y->getValue<float>(0);
        return NNscore;
    }//end EvaluateNN

}

typedef flashgg::DoubleHttHTagProducer80 DoubleHttHTagProducer80;
DEFINE_FWK_MODULE( DoubleHttHTagProducer80);
// Local Variables:
// mode:c++
// indent-tabs-mode:nil
// tab-width:4
// c-basic-offset:4
// End:
// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4


