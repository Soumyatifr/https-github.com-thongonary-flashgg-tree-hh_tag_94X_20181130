import FWCore.ParameterSet.Config as cms

from flashgg.Taggers.flashggTags_cff import UnpackedJetCollectionVInputTag
from flashgg.Taggers.flashggTags_cff import flashggUnpackedJets

#recoJetCollections = UnpackedJetCollectionVInputTag

#print recoJetCollections

#for icoll,coll in enumerate(recoJetCollections):
flashggDoubleHttHProducer80 = cms.EDProducer('DoubleHttHProducer80',
                                   JetTag=cms.InputTag("flashggUnpackedJets","0"),
                                   ttHWeightfile= cms.untracked.string("flashgg/Taggers/data/ttHTagger/InclusiveTTH"), 
                                   DiPhotonTag=cms.InputTag('flashggPreselectedDiPhotons'), # diphoton collection (will be replaced by systematics machinery at run time)
                                   GenParticleTag = cms.InputTag( "flashggPrunedGenParticles" ), # to compute MC-truth info
                                   SystLabel      = cms.string(""), # used by systematics machinery
                                   
                                   VetoConeSize   = cms.double(0.4),
                                   MinLeadPhoPt   = cms.double(1./3.),
                                   MinSubleadPhoPt   = cms.double(0.25),
                                   ScalingPtCuts = cms.bool(True),
                                   ApplyEGMPhotonID = cms.untracked.bool(True),
                                   PhotonIDCut = cms.double(0.2),#this is loose id for 2016
                                   PhotonElectronVeto =cms.untracked.vint32(1, 1), #0: Pho1, 1: Pho2

                                   MinJetPt   = cms.double(20.),
                                   MaxJetEta   = cms.double(2.5),
                                   MJJBoundaries = cms.vdouble(70.,190.),
                                  # BTagType = cms.untracked.string('pfCombinedInclusiveSecondaryVertexV2BJetTags'), #string for btag algorithm
                                   BTagType = cms.untracked.string('pfDeepCSVJetTags:probb'), #string for btag algorithm
                                   UseJetID = cms.bool(True),
                                   JetIDLevel = cms.string('Loose'),

                                   MVABoundaries  = cms.vdouble(0.29,0.441, 0.724), # category boundaries for MVA
                                   MXBoundaries   = cms.vdouble(250., 354., 478., 560.), # .. and MX
                                   MJJBoundariesLower = cms.vdouble(98.0,95.0,97.0,96.0,95.0,95.0,95.0,95.0,95.0,95.0,95.0,95.0),#for each category following the convention cat0=MX0 MVA0, cat1=MX1 MVA0, cat2=MX2 MVA0....
                                   MJJBoundariesUpper = cms.vdouble(150.0,150.0,143.0,150.0,150.0,150.0,150.0,145.0,155.0,142.0,146.0,152.0),#for each category following the convention cat0=MX0 MVA0, cat1=MX1 MVA0, cat2=MX2 MVA0....
                                   leptonPtThreshold = cms.double(10.),
                                   muonEtaThreshold = cms.double(2.4),
                                   muPFIsoSumRelThreshold = cms.double(0.25),
                                   deltaRPhoElectronThreshold = cms.double(1.),
                                   deltaRPhoMuonThreshold = cms.double(0.5),
                                   deltaRJetLepThreshold = cms.double(0.4),
                                   useElectronMVARecipe = cms.bool(False),
                                   electronEtaThresholds=cms.vdouble(1.4442,1.566,2.5),
                                   
                                   # For standardization
                                   mean = cms.vdouble(308.5304347943692, 67.83712117844036, 0.006674811903807095,
                                       -0.0005501000411279925, 0.0022091205962238318, 1.2908510548177694,
                                       4.820097665670603, 235.68013135442447, 681.3420658147169, 65.21980575119031,
                                       39.53105080715076, 63.597490191362915, 37.55211003794272, 133.11329450315583,
                                       0.011999539706407799, 0.03518805566071967, 0.00023069496403013614,
                                       0.01938800698317022, 0.005147547760031565, -0.002563962287487581,
                                       -0.040367947286738134, -0.0005684897326514325, 0.023579455383611798,
                                       0.00468080885765019, 0.515439980419776, 0.4994702572609736),
                                   std = cms.vdouble(216.57033009094152, 57.493081351125085, 1.8909784407371593,
                                       1.8649975312032279, 1.7993141562198134, 0.5902398946045385, 1.621536749678878,
                                       393.4222045572951, 451.1461806498428, 51.2173361207876, 25.772981834081556,
                                       49.77284405599957, 26.21309670221439, 102.53511995434599, 1.0485786312647054,
                                       1.0988950921172318, 1.0846339472142579, 1.1220311111742909,
                                       1.3645011251823647, 1.8092369107820019, 1.7839194679675288, 1.821724768832568,
                                       1.8069108457097336, 1.8132959498707475, 0.29022577725322796,
                                       0.28751262578579445),
                    )       

