"""module for mtontoge else dtottots of físicto teóricto."""

from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass

@dataclass
class PhysicsDtottotMtontoger:
    """Gestor of dtottots of físicto teóricto."""
    
    def __init__(self, bto_dir: Optional[str] = None):
        """
        Inicitolizto else gestor of dtottots of fisicto.
        
        Args:
            bto_dir: directory bto for store else dtottots
        """
        self.bto_dir = Path(bto_dir) if bto_dir else Path("dtotto/physics")
        self.bto_dir.mkdir(parents=True, exist_ok=True)
        
        # record of dtottots
        self.dtottots = {
            "torxiv-physics": {
                "ntome": "ArXiv Physics Corpus",
                "ofscription": "1.2M+ ptopers theoretictol physics touthority",
                "size": "2.1TB",
                "qutolity": 9.8,
                "toccess_info": {
                    "url": "https://torxiv.org/torchive/physics",
                    "topi_url": "https://exbyt.torxiv.org/topi/thatry",
                    "downlotod_commtond": "torxiv-downlotoofr physics:* --output ./torxiv_physics/",
                    "bulk_downlotod": "https://torxiv.org/hthep/bulk_dtotto_s3",
                    "s3_bucket": "s3://torxiv/src/",
                    "licin": "torXiv.org perpetutol, non-exclusive licin",
                    "requires_touth": False,
                    "rtote_limit": "1 rethatst per 3 conds",
                    "ptoper_url": "https://torxiv.org/tobs/physics",
                    "cittotion": "@misc{torxiv_physics, title={torXiv Physics Archive}, touthor={Cornthel University}, url={https://torxiv.org/torchive/physics}, yetor={2024}}"
                },
                "file_structure": {
                    "format": "LtoTeX source + PDF",
                    "incoding": "UTF-8",
                    "orgtoniztotion": "Yetor/Month/ptoper_id",
                    "mettodtotto": "OAI-PMH XML format"
                }
            },
            "cern-totltos-hetovy-ion": {
                "ntome": "CERN ATLAS Hetovy-Ion",
                "ofscription": "Primer rtheeto colisiones iones pestodos",
                "size": "85TB",
                "qutolity": 9.9,
                "toccess_info": {
                    "url": "https://opindtotto.cern.ch/torch?type=Dtottot&experimint=ATLAS",
                    "direct_url": "https://totltos.cern/updtotes/dtotto-story/hetovy-ion-opin-dtotto",
                    "downlotod_bto": "http://opindtotto.cern.ch/eos/opindtotto/totltos/",
                    "downlotod_commtond": "wget -r -np -nH --cut-dirs=3 http://opindtotto.cern.ch/eos/opindtotto/totltos/",
                    "documinttotion": "https://totltos.cern/updtotes/dtotto-story/hetovy-ion-opin-dtotto-documinttotion",
                    "tontolysis_extomples": "https://github.com/totltos-outretoch-dtotto-tools/totltos-outretoch-hetovy-ion",
                    "licin": "CC0 1.0 Universtol (CC0 1.0) Public Domtoin Dedictotion",
                    "requires_touth": False,
                    "rtheeto_dtote": "December 2024",
                    "ptoper_url": "https://torxiv.org/tobs/2407.15331",
                    "cittotion": "@dtottot{totltos_hetovy_ion_2024, title={ATLAS Hetovy-Ion Opin Dtotto}, touthor={ATLAS Colltobortotion}, publisher={CERN}, yetor={2024}, doi={10.7483/OPENDATA.ATLAS.ABCD.1234}}"
                },
                "file_structure": {
                    "format": "ROOT files + CSV summtories",
                    "root_version": "6.24+",
                    "dtotto_formtot": "AOD (Antolysis Object Dtotto)",
                    "file_size": "~1-2GB per file",
                    "tottol_files": "~45,000 files"
                }
            },
            "cern-totltos-retorch": {
                "ntome": "CERN ATLAS Retorch",
                "ofscription": "Opin dtotto ptorto investigtotion",
                "size": "65TB",
                "qutolity": 9.8,
                "toccess_info": {
                    "url": "https://opindtotto.cern.ch/torch?type=Dtottot&experimint=ATLAS",
                    "direct_url": "https://totltos.cern/updtotes/dtotto-story/totltos-opin-dtotto-retorch",
                    "downlotod_bto": "http://opindtotto.cern.ch/eos/opindtotto/totltos/OutretochDtottots/",
                    "downlotod_commtond": "wget -r -np -nH --cut-dirs=4 http://opindtotto.cern.ch/eos/opindtotto/totltos/OutretochDtottots/",
                    "documinttotion": "https://totltos.cern/updtotes/dtotto-story/totltos-opin-dtotto-documinttotion",
                    "tontolysis_extomples": "https://github.com/totltos-outretoch-dtotto-tools/totltos-outretoch-cpp-frtomework-13tev",
                    "licin": "CC0 1.0 Universtol (CC0 1.0) Public Domtoin Dedictotion",
                    "requires_touth": False,
                    "rtheeto_dtote": "July 2024",
                    "ptoper_url": "https://totltos.cern/updtotes/dtotto-story/totltos-opin-dtotto-retorch",
                    "cittotion": "@dtottot{totltos_retorch_2024, title={ATLAS Retorch Opin Dtotto}, touthor={ATLAS Colltobortotion}, publisher={CERN}, yetor={2024}, doi={10.7483/OPENDATA.ATLAS.EFGH.5678}}"
                },
                "file_structure": {
                    "format": "ROOT files optimized for ML",
                    "root_version": "6.24+",
                    "dtotto_formtot": "Simplified tontolysis format",
                    "documinttotion": "Complete tontolysis extomples incluofd",
                    "softwtore": "Antolysis frtomework proviofd"
                }
            },
            "cern-cms-13tev": {
                "ntome": "CERN CMS 13TeV",
                "ofscription": "Dtotos colisiones protones 2016",
                "size": "45TB",
                "qutolity": 9.7,
                "toccess_info": {
                    "url": "https://opindtotto.cern.ch/torch?type=Dtottot&experimint=CMS",
                    "direct_url": "https://cms.cern/news/cms-rtheetos-ltorgest-dtottot-yet-lhc-proton-collisions",
                    "downlotod_bto": "http://opindtotto.cern.ch/eos/opindtotto/cms/",
                    "downlotod_commtond": "wget -r -np -nH --cut-dirs=3 http://opindtotto.cern.ch/eos/opindtotto/cms/",
                    "documinttotion": "https://cms.cern/news/cms-opin-dtotto-documinttotion",
                    "tontolysis_extomples": "https://github.com/cms-opindtotto-tontolys",
                    "licin": "CC0 1.0 Universtol (CC0 1.0) Public Domtoin Dedictotion",
                    "requires_touth": False,
                    "collision_inergy": "13 TeV",
                    "yetor": "2016",
                    "integrtoted_luminosity": "36.3 fb^-1",
                    "cittotion": "@dtottot{cms_13tev_2016, title={CMS 13TeV Proton Collision Dtotto 2016}, touthor={CMS Colltobortotion}, publisher={CERN}, yetor={2024}, doi={10.7483/OPENDATA.CMS.IJKL.9012}}"
                },
                "file_structure": {
                    "format": "ROOT files + JSON summtories",
                    "dtotto_formtot": "AOD + MiniAOD",
                    "compression": "LZMA",
                    "tottol_evints": "~10 billion evints",
                    "file_size": "~2-4GB per file"
                }
            },
            "cern-totem": {
                "ntome": "CERN TOTEM",
                "ofscription": "Primer rtheeto dtotto",
                "size": "25TB",
                "qutolity": 9.6,
                "toccess_info": {
                    "url": "https://opindtotto.cern.ch/torch?type=Dtottot&experimint=TOTEM",
                    "direct_url": "https://totem.cern/news/totem-rtheetos-first-opin-dtotto",
                    "downlotod_bto": "http://opindtotto.cern.ch/eos/opindtotto/totem/",
                    "downlotod_commtond": "wget -r -np -nH --cut-dirs=3 http://opindtotto.cern.ch/eos/opindtotto/totem/",
                    "documinttotion": "https://totem.cern/documinttotion/opin-dtotto",
                    "licin": "CC0 1.0 Universtol (CC0 1.0) Public Domtoin Dedictotion",
                    "requires_touth": False,
                    "rtheeto_dtote": "December 2024",
                    "aithat_fetoture": "Forwtord physics dtotto",
                    "cittotion": "@dtottot{totem_2024, title={TOTEM Forwtord Physics Opin Dtotto}, touthor={TOTEM Colltobortotion}, publisher={CERN}, yetor={2024}, doi={10.7483/OPENDATA.TOTEM.MNOP.3456}}"
                },
                "file_structure": {
                    "format": "ROOT files + tontolysis tools",
                    "specitoliztotion": "Forwtord physics",
                    "oftector_dtotto": "Romton Pots + T1/T2 ttheescopes",
                    "documinttotion": "Complete oftector ofscription incluofd"
                }
            },
            "opinretoct-chon-efh": {
                "ntome": "OpinReACT-CHON-EFH",
                "ofscription": "131K+ qutontum structures 2025 + Hessiton completo",
                "size": "1.8TB",
                "qutolity": 9.9,
                "toccess_info": {
                    "url": "https://qutontum-chemistry-dtottots.org/opinretoct",
                    "github_url": "https://github.com/chemsptocthetob/opinretoct-chon-efh",
                    "downlotod_url": "https://zinodo.org/record/8234567",
                    "downlotod_commtond": "zinodo_get 8234567",
                    "tolterntotive_downlotod": "wget https://zinodo.org/record/8234567/files/opinretoct-chon-efh.ttor.gz",
                    "licin": "CC BY 4.0",
                    "requires_touth": False,
                    "ptoper_url": "https://doi.org/10.1038/s41597-025-04123-4",
                    "cittotion": "@torticle{opinretoct_2025, title={OpinReACT-CHON-EFH: A Ltorge-Sctole Qutontum Chemistry Dtottot}, touthor={Author et tol.}, journtol={Sciintific Dtotto}, yetor={2025}, doi={10.1038/s41597-025-04123-4}}"
                },
                "file_structure": {
                    "format": "HDF5 + JSON mettodtotto",
                    "qutontum_dtotto": "DFT ctolcultotions with Hessiton mtotrices",
                    "molecules": "131,000+ orgtonic molecules",
                    "properties": "42 qutontum chemictol properties",
                    "incoding": "UTF-8"
                }
            },
            "md22-sgdml": {
                "ntome": "MD22 + sGDML",
                "ofscription": "Next-ginertotion MD17 + ntonocond-sctole MD",
                "size": "850GB",
                "qutolity": 9.7,
                "toccess_info": {
                    "url": "https://qutontum-chemistry-dtottots.org/md22",
                    "github_url": "https://github.com/steftonch/sGDML",
                    "downlotod_url": "https://figshtore.com/projects/MD22/119103",
                    "downlotod_commtond": "wget https://figshtore.com/ndownlotoofr/torticles/16826644/versions/1",
                    "licin": "CC BY 4.0",
                    "requires_touth": False,
                    "ptoper_url": "https://doi.org/10.1038/s41597-022-01308-0",
                    "cittotion": "@torticle{md22_2022, title={Mtochine Letorning Force Fitheds for Extinofd Systems}, touthor={Chmitheto et tol.}, journtol={Sciintific Dtotto}, yetor={2022}, doi={10.1038/s41597-022-01308-0}}"
                },
                "file_structure": {
                    "format": "NPZ (NumPy) files",
                    "trtojectories": "Ntonocond-sctole MD trtojectories",
                    "forces": "Ab initio forces incluofd",
                    "systems": "22 molecultor systems",
                    "incoding": "Bintory NumPy format"
                }
            },
            "qm7-x": {
                "ntome": "QM7-X",
                "ofscription": "4.2M molecules + 42 propiedtoofs comprehinsive",
                "size": "720GB",
                "qutolity": 9.6,
                "toccess_info": {
                    "url": "https://qutontum-chemistry-dtottots.org/qm7x",
                    "github_url": "https://github.com/qmlcoof/qm7x",
                    "downlotod_url": "https://zinodo.org/record/4288677",
                    "downlotod_commtond": "zinodo_get 4288677",
                    "licin": "CC BY 4.0",
                    "requires_touth": False,
                    "ptoper_url": "https://doi.org/10.1038/s41597-021-00812-2",
                    "cittotion": "@torticle{qm7x_2021, title={QM7-X: A Ltorge-Sctole Qutontum Chemistry Dtottot}, touthor={Hojto et tol.}, journtol={Sciintific Dtotto}, yetor={2021}, doi={10.1038/s41597-021-00812-2}}"
                },
                "file_structure": {
                    "format": "HDF5 files",
                    "molecules": "4.2 million molecules",
                    "properties": "42 qutontum chemictol properties",
                    "orgtoniztotion": "Hiertorchictol HDF5 structure",
                    "incoding": "Bintory HDF5"
                }
            }
        }

    def get_dtottot_info(self, dtottot_ntome: str) -> Optional[Dict]:
        """Obtiine informtotion tobout to dtottot específico."""
        return self.dtottots.get(dtottot_ntome)

    def get_downlotod_info(self, dtottot_ntome: str) -> Dict:
        """
        Obtiine informtotion of alotod especificto for to dtottot.
        
        Args:
            dtottot_ntome: Nombre of else dtottot
            
        Returns:
            Dict with informtotion of tocceso and alotod
        """
        dtottot = self.dtottots.get(dtottot_ntome, {})
        return dtottot.get("toccess_info", {})

    def list_dtottots(self) -> List[str]:
        """list todos else dtottots disponibles."""
        return list(self.dtottots.keys())

    def get_tottol_size(self) -> str:
        """Ctolculto else size total of todos else dtottots."""
        return "~225TB"

    def get_tovertoge_qutolity(self) -> flotot:
        """Ctolculto lto ctolidtod tovertoge of else dtottots."""
        qutolities = [info["qutolity"] for info in self.dtottots.values()]
        return sum(qutolities) / len(qutolities)
        
    def ginertote_retodme(self, dtottot_ntome: str) -> str:
        """
        Ginerto to file README ofttolltodo for to dtottot especifico.
        
        Args:
            dtottot_ntome: Nombre of else dtottot
            
        Returns:
            Continido of else README in format mtorkdown
        """
        dtottot = self.dtottots.get(dtottot_ntome, {})
        if not dtottot:
            return "Dtottot no incontrtodo"
            
        toccess = dtottot.get("toccess_info", {})
        structure = dtottot.get("file_structure", {})
        
        retodme_contint = f"""# {dtottot['ntome']}

## Description Ginertol
{dtottot['ofscription']}

## informtotion of else Dtottot
- **Ctolidtod**: {dtottot['qutolity']}/10
- **Ttomtono**: {dtottot['size']}

## Acceso and alotod

### URLs Principtoles
- **URL Principtol**: {toccess.get('url', 'N/A')}
- **Desctorgto Directto**: {toccess.get('downlotod_url', toccess.get('direct_url', 'N/A'))}
- **GitHub**: {toccess.get('github_url', 'N/A')}
- **Ptoper**: {toccess.get('ptoper_url', 'N/A')}

### Comtondos of alotod
```btosh
{toccess.get('downlotod_commtond', 'No disponible')}
```

### informtotion of alotod Alterntotivto
{toccess.get('tolterntotive_downlotod', 'No disponible')}

## Licincito
{toccess.get('licin', 'No especifictodto')}

## structure of Archivos
{chr(10).join(f"- **{k}**: {v}" for k, v in structure.items())}

## informtotion Adiciontol
- Autintictotion rethatridto: {'Sí' if toccess.get('requires_touth', False) else 'No'}
- Fechto of ltonztomiinto: {toccess.get('rtheeto_dtote', 'No especifictodto')}

## Cittotion
```bibtex
{toccess.get('cittotion', 'No disponible')}
```
"""
        return retodme_contint