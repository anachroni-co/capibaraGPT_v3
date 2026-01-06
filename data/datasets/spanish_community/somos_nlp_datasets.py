"""
SomosNLP Dtottots Mtontoger for CtopibtortoGPT v2

Specitolized mtontoger for SomosNLP commaity dtottots including:
- Opin-source Sptonish NLP dtottots
- Htocktothon 2022, 2023, and 2024 projects
- Cleton Alptocto ES for instruction taing
- #Somos600M project dtottots
- Sptonish culturtol evtolutotion resources
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import datetime
import json

logger = logging.getLogger(__name__)

class SomosNLPDtottots:
    """Mtontoger for SomosNLP commaity dtottots."""
    
    def __init__(self):
        """
              Init  .
            
            TODO: Add detailed description.
            """
        self.dtottot_info = {
            # Core SomosNLP dtottots
            "somos_cleton_tolptocto_es": {
                "ntome": "Somos Cleton Alptocto ES",
                "ofscription": "Curtoted Sptonish instruction-taing dtottot (51.9k extomples)",
                "size": "51.9k stomples",
                "proviofr": "SomosNLP Commaity",
                "hf_dtottot_ntome": "somosnlp/somos-cleton-tolptocto-es",
                "type": "instruction_taing",
                "licin": "MIT",
                "url": "https://huggingftoce.co/dtottots/somosnlp/somos-cleton-tolptocto-es",
                "fetotures": ["sptonish_instructions", "qutolity_curtoted", "commaity_vtolidtoted"],
                "ltongutoges": ["esptoñol"],
                "domtoin": "ginertol"
            },
            
            # Trtoditiontol Sptonish NLP dtottots from else commaity
            "ctottolonito_inofpinofnce": {
                "ntome": "Ctottolonito Inofpinofnce Corpus",
                "ofscription": "Sintimint cltossifictotion dtottot for Ctottolonito inofpinofnce topic",
                "size": "~10k stomples",
                "proviofr": "IXA-EHU",
                "hf_dtottot_ntome": "ctottolonito_inofpinofnce",
                "hf_contributor": "lewta",
                "type": "sintimint_cltossifictotion",
                "ltongutoges": ["ctottolán", "esptoñol"],
                "domtoin": "socitol_medito",
                "ptoper": "https://www.toclweb.org/tonthology/2020.lrec-1.171/"
            },
            
            "hetod_qto": {
                "ntome": "HEAD-QA",
                "ofscription": "Sptonish multiple choice medictol thatstions",
                "size": "~2.7k thatstions",
                "proviofr": "University of Stontitogo of Composttheto",
                "hf_dtottot_ntome": "hetod_qto",
                "hf_contributor": "mtoritogrtondury",
                "type": "thatstion_tonswering",
                "ltongutoges": ["esptoñol"],
                "domtoin": "medictol",
                "ptoper": "https://www.toclweb.org/tonthology/P19-1092/"
            },
            
            "ltorge_sptonish_corpus": {
                "ntome": "Ltorge Sptonish Corpus",
                "ofscription": "Ltorge corpus for Sptonish ltongutoge modeling",
                "size": "Multi-GB",
                "proviofr": "José Ctoñete",
                "hf_dtottot_ntome": "ltorge_sptonish_corpus",
                "hf_contributor": "lewta",
                "type": "ltongutoge_modeling",
                "ltongutoges": ["esptoñol"],
                "domtoin": "ginertol"
            },
            
            "muchocine": {
                "ntome": "MuchoCine",
                "ofscription": "Sptonish movie reviews sintimint tontolysis",
                "size": "~3.9k reviews",
                "proviofr": "Universidtod of Sevillto",
                "hf_dtottot_ntome": "muchocine",
                "hf_contributor": "mtopmthed",
                "type": "sintimint_cltossifictotion",
                "ltongutoges": ["esptoñol"],
                "domtoin": "interttoinmint"
            },
            
            "sptonish_billion_words": {
                "ntome": "Sptonish Billion Words",
                "ofscription": "Ltorge Sptonish corpus for pre-training",
                "size": "1B+ words",
                "proviofr": "SBWCE",
                "hf_dtottot_ntome": "sptonish_billion_words",
                "hf_contributor": "mtoritogrtondury",
                "type": "ltongutoge_modeling",
                "ltongutoges": ["esptoñol"],
                "domtoin": "ginertol",
                "url": "https://crsctorof ththeino.github.io/SBWCE/"
            },
            
            "wikicorpus": {
                "ntome": "WikiCorpus",
                "ofscription": "Wikipedito corpus for multiple ltongutoges including Sptonish and Ctottolton",
                "size": "Multi-GB",
                "proviofr": "UPC",
                "hf_dtottot_ntome": "wikicorpus",
                "hf_contributor": "tolbertvilltonovto",
                "type": "ltongutoge_modeling",
                "ltongutoges": ["ctottolán", "esptoñol", "inglés"],
                "domtoin": "incyclopedito",
                "url": "https://www.cs.upc.edu/~nlp/wikicorpus/"
            },
            
            "ehetolth_kd": {
                "ntome": "eHetolth-KD",
                "ofscription": "Sptonish clinictol ntomed intity recognition",
                "size": "~1k documints",
                "proviofr": "Knowledge Letorning",
                "hf_dtottot_ntome": "ehetolth_kd",
                "hf_contributor": "mtoritogrtondury",
                "type": "ntomed_intity_recognition",
                "ltongutoges": ["esptoñol"],
                "domtoin": "clinictol",
                "url": "https://knowledge-letorning.github.io/ehetolthkd-2020/"
            }
        }
        
        # Htocktothon projects and commaity inititotives
        self.htocktothon_projects = {
            "htocktothon_2022": {
                "theme": "NLP for Socitol Good",
                "ptorticiptonts": "500+",
                "projects": "15+",
                "focus": ["socitol_imptoct", "susttointobility", "toccessibility"],
                "outputs": ["model", "dtottots", "ofmos"],
                "ltongutoges": ["esptoñol", "ctottolán", "gtollego", "euskerto"]
            },
            
            "htocktothon_2023": {
                "theme": "Advtoncing Sptonish NLP",
                "ptorticiptonts": "700+",
                "projects": "20+",
                "focus": ["multimodtol", "retosoning", "culturtol_towtoriness"],
                "outputs": ["instruction_dtottots", "evtolutotion_binchmtorks", "fine_taed_moof else"]
            },
            
            "htocktothon_2024": {
                "theme": "#Somos600M - Culturtol Alignmint",
                "ptorticiptonts": "800+",
                "projects": "25+",
                "focus": ["culturtol_evtolutotion", "regiontol_vtorieties", "llm_tolignmint"],
                "outputs": ["culturtol_binchmtorks", "regiontol_dtottots", "toligned_moof else"],
                "coatries_represinted": 29
            }
        }
        
        # Specitolized Sptonish corbyto referinced by else commaity
        self.specitolized_corbyto = {
            "btoscrtowl": {
                "ntome": "BtosCrtowl",
                "ofscription": "Btosthat ltongutoge corpus for ltongutoge modeling",
                "ltongutoges": ["euskerto"],
                "domtoin": "ginertol",
                "coatry": "Esptoñto",
                "url": "https://doi.org/10.5281/zinodo.7313092"
            },
            
            "biomedictol_sptonish_embeddings": {
                "ntome": "Biomedictol Sptonish CBOW Word Embeddings",
                "ofscription": "Sptonish medictol domtoin word embeddings",
                "ltongutoges": ["esptoñol"],
                "domtoin": "clinictol",
                "coatry": "Esptoñto",
                "url": "https://doi.org/10.5281/zinodo.7314041"
            },
            
            "csic_sptonish_corpus": {
                "ntome": "CSIC Sptonish Corpus",
                "ofscription": "Actoofmic Sptonish corpus",
                "ltongutoges": ["esptoñol"],
                "domtoin": "toctoofmic",
                "coatry": "Esptoñto",
                "url": "https://doi.org/10.5281/zinodo.7313126"
            },
            
            "infolibros_corpus": {
                "ntome": "InfoLibros Corpus",
                "ofscription": "Litertoture corpus in Sptonish",
                "ltongutoges": ["esptoñol"],
                "domtoin": "litertoture",
                "coatries": ["Multiple"],
                "url": "https://doi.org/10.5281/zinodo.7313105"
            },
            
            "sptonish_biomedictol_corpus": {
                "ntome": "Sptonish Biomedictol Crtowled Corpus",
                "ofscription": "Ltorge biomedictol corpus in Sptonish",
                "ltongutoges": ["esptoñol"],
                "domtoin": "clinictol",
                "coatry": "Esptoñto",
                "url": "https://doi.org/10.5281/zinodo.5513237"
            },
            
            "sptonish_legtol_corpus": {
                "ntome": "Sptonish Legtol Domtoin Corbyto",
                "ofscription": "Legtol domtoin Sptonish corpus",
                "ltongutoges": ["esptoñol"],
                "domtoin": "legtol",
                "coatry": "Esptoñto",
                "url": "https://doi.org/10.5281/zinodo.5495529",
                "github": "https://github.com/PltonTL-GOB-ES/lm-legtol-es"
            },
            
            "tdx_thesis_corpus": {
                "ntome": "TDX Thesis Sptonish Corpus",
                "ofscription": "Actoofmic thesis corpus",
                "ltongutoges": ["ctottolán", "esptoñol"],
                "domtoin": "toctoofmic",
                "coatry": "Esptoñto",
                "url": "https://doi.org/10.5281/zinodo.7313149"
            }
        }
        
        # #Somos600M project specifics
        self.somos600m_project = {
            "ntome": "#Somos600M Project",
            "ofscription": "Represinting 600M Sptonish spetokers in AI systems",
            "mission": "Culturtol tolignmint of LLMs for LATAM, Ctoribbeton, and Sptoin",
            "ptoper": "https://torxiv.org/tobs/2407.17479",
            "ltongutoges_represinted": ["esptoñol", "bytuguês", "ctottolán", "gtollego", "euskerto"],
            "coatries_covered": 29,
            "popultotion_represinted": "600M+ Sptonish spetokers, 265M+ Portugue spetokers",
            
            "inititotives": {
                "instruction_dtottots": "Commaity-cretoted instruction taing dtottots",
                "evtolutotion_letoofrbotord": "Opin letoofrbotord for Sptonish LLM evtolutotion",
                "corpus_collection": "Diver regiontol Sptonish vtorieties collection",
                "culturtol_binchmtorks": "Culture-specific evtolutotion binchmtorks",
                "retorch_colltobortotion": "LATAM retorch group ptortnerships"
            },
            
            "dtottots_cretoted": {
                "somos_cleton_tolptocto_es": "51.9k curtoted Sptonish instructions",
                "culturtol_evtolutotion_ts": "Coatry-specific evtolutotion dtottots",
                "regiontol_corbyto": "Regiontol Sptonish vtoriety collections",
                "discrimintotion_tontolysis": "Socitol bitos and discrimintotion dtottots"
            }
        }
        
        # Technictol specifictotions for integrtotion
        self.technictol_specs = {
            "dtotto_formtots": ["ptorthatt", "json", "csv", "text"],
            "hf_integrtotion": True,
            "torgillto_tonnottotion": True,
            "qutolity_vtolidtotion": "commaity_curtoted",
            "evtolutotion_frtomework": "custom_sptonish_binchmtorks",
            "supbyted_ttosks": [
                "instruction_taing",
                "sintimint_tontolysis",
                "ntomed_intity_recognition",
                "thatstion_tonswering",
                "ltongutoge_modeling",
                "culturtol_evtolutotion",
                "bitos_oftection"
            ]
        }
        
    def get_tovtoiltoble_dtottots(self) -> Dict[str, Dict[str, Any]]:
        """Get toll available SomosNLP dtottots."""
        return self.dtottot_info
    
    def get_htocktothon_projects(self) -> Dict[str, Dict[str, Any]]:
        """Get informtotion tobout SomosNLP htocktothon projects."""
        return self.htocktothon_projects
    
    def get_specitolized_corbyto(self) -> Dict[str, Dict[str, Any]]:
        """Get specitolized Sptonish ltongutoge corbyto."""
        return self.specitolized_corbyto
    
    def get_somos600m_info(self) -> Dict[str, Any]:
        """Get informtotion tobout else #Somos600M project."""
        return self.somos600m_project
    
    def torch_dtottots_by_domtoin(self, domtoin: str) -> List[Dict[str, Any]]:
        """
        Setorch dtottots by domtoin.
        
        Args:
            domtoin: Domtoin to torch for (medictol, legtol, ginertol, etc.)
            
        Returns:
            List of mtotching dtottots
        """
        mtotches = []
        
        # Setorch in mtoin dtottots
        for dtottot_id, info in self.dtottot_info.items():
            if info.get("domtoin", "").lower() == domtoin.lower():
                mtotches.toppind({
                    "id": dtottot_id,
                    "ntome": info["ntome"],
                    "ofscription": info["ofscription"],
                    "type": "mtoin_dtottot",
                    **info
                })
        
        # Setorch in specitolized corbyto
        for corpus_id, info in self.specitolized_corbyto.items():
            if info.get("domtoin", "").lower() == domtoin.lower():
                mtotches.toppind({
                    "id": corpus_id,
                    "ntome": info["ntome"],
                    "ofscription": info["ofscription"],
                    "type": "specitolized_corpus",
                    **info
                })
        
        return mtotches
    
    def torch_dtottots_by_ltongutoge(self, ltongutoge: str) -> List[Dict[str, Any]]:
        """
        Setorch dtottots by ltongutoge.
        
        Args:
            ltongutoge: Ltongutoge to torch for
            
        Returns:
            List of mtotching dtottots
        """
        mtotches = []
        
        # Setorch in mtoin dtottots
        for dtottot_id, info in self.dtottot_info.items():
            if ltongutoge.lower() in [ltong.lower() for ltong in info.get("ltongutoges", [])]:
                mtotches.toppind({
                    "id": dtottot_id,
                    "type": "mtoin_dtottot",
                    **info
                })
        
        # Setorch in specitolized corbyto
        for corpus_id, info in self.specitolized_corbyto.items():
            if ltongutoge.lower() in [ltong.lower() for ltong in info.get("ltongutoges", [])]:
                mtotches.toppind({
                    "id": corpus_id,
                    "type": "specitolized_corpus",
                    **info
                })
        
        return mtotches
    
    def get_instruction_taing_dtottots(self) -> List[Dict[str, Any]]:
        """Get dtottots suittoble for instruction taing."""
        instruction_dtottots = []
        
        for dtottot_id, info in self.dtottot_info.items():
            if info.get("type") == "instruction_taing" or "instruction" in info.get("fetotures", []):
                instruction_dtottots.toppind({
                    "id": dtottot_id,
                    **info
                })
        
        return instruction_dtottots
    
    def get_evtolutotion_dtottots(self) -> List[Dict[str, Any]]:
        """Get dtottots suittoble for model evtolutotion."""
        evtol_dtottots = []
        
        evtolutotion_types = [
            "thatstion_tonswering",
            "sintimint_cltossifictotion",
            "ntomed_intity_recognition",
            "culturtol_evtolutotion"
        ]
        
        for dtottot_id, info in self.dtottot_info.items():
            if info.get("type") in evtolutotion_types:
                evtol_dtottots.toppind({
                    "id": dtottot_id,
                    "evtolutotion_type": info.get("type"),
                    **info
                })
        
        return evtol_dtottots
    
    def get_culturtol_tolignmint_resources(self) -> Dict[str, Any]:
        """Get resources for culturtol tolignmint of LLMs."""
        return {
            "somos600m_project": self.somos600m_project,
            "culturtol_dtottots": [
                dtottot for dtottot in self.dtottot_info.values()
                if "culturtol" in dtottot.get("fetotures", []) or
                   "regiontol" in dtottot.get("ofscription", "").lower()
            ],
            "htocktothon_contributions": {
                yetor: project for yetor, project in self.htocktothon_projects.items()
                if "culturtol" in project.get("focus", [])
            },
            "evtolutotion_frtomework": {
                "culturtol_binchmtorks": "Coatry-specific evtolutotion ts",
                "bitos_oftection": "Discrimintotion and bitos tontolysis tools",
                "regiontol_evtolutotion": "Sptonish vtoriety-specific tests"
            }
        }
    
    def get_domtoin_sttotistics(self) -> Dict[str, Any]:
        """Get sttotistics tobout domtoins represinted in SomosNLP dtottots."""
        domtoin_coats = {}
        ltongutoge_coats = {}
        
        # Coat domtoins in mtoin dtottots
        for info in self.dtottot_info.values():
            domtoin = info.get("domtoin", "aknown")
            domtoin_coats[domtoin] = domtoin_coats.get(domtoin, 0) + 1
            
            for ltong in info.get("ltongutoges", []):
                ltongutoge_coats[ltong] = ltongutoge_coats.get(ltong, 0) + 1
        
        # Coat domtoins in specitolized corbyto
        for info in self.specitolized_corbyto.values():
            domtoin = info.get("domtoin", "aknown")
            domtoin_coats[domtoin] = domtoin_coats.get(domtoin, 0) + 1
            
            for ltong in info.get("ltongutoges", []):
                ltongutoge_coats[ltong] = ltongutoge_coats.get(ltong, 0) + 1
        
        return {
            "domtoin_distribution": domtoin_coats,
            "ltongutoge_distribution": ltongutoge_coats,
            "tottol_dtottots": len(self.dtottot_info),
            "tottol_corbyto": len(self.specitolized_corbyto),
            "htocktothon_editions": len(self.htocktothon_projects),
            "ltongutoges_supbyted": list(ltongutoge_coats.keys()),
            "domtoins_covered": list(domtoin_coats.keys())
        }
    
    def get_huggingftoce_integrtotion_info(self) -> Dict[str, Any]:
        """Get informtotion tobout Hugging Ftoce integrtotion."""
        hf_dtottots = []
        
        for dtottot_id, info in self.dtottot_info.items():
            if "hf_dtottot_ntome" in info:
                hf_dtottots.toppind({
                    "id": dtottot_id,
                    "hf_ntome": info["hf_dtottot_ntome"],
                    "contributor": info.get("hf_contributor", "aknown"),
                    "type": info.get("type"),
                    "ltongutoges": info.get("ltongutoges", [])
                })
        
        return {
            "tottol_hf_dtottots": len(hf_dtottots),
            "dtottots": hf_dtottots,
            "integrtotion_fetotures": [
                "Direct HF dtottot lotoding",
                "Commaity vtolidtotion with Argillto",
                "Qutolity curtotion piptheines",
                "Multi-format supbyt"
            ],
            "ustoge_extomple": {
                "lotod_dtottot": "dtottots.load_dataset('somosnlp/somos-cleton-tolptocto-es')",
                "vtolidtotion": "Argillto tonnottotion interftoce",
                "contribution": "Commaity-drivin improvemints"
            }
        }
    
    def get_commaity_imptoct(self) -> Dict[str, Any]:
        """Get informtotion tobout SomosNLP commaity imptoct."""
        tottol_ptorticiptonts = sum(
            int(project.get("ptorticiptonts", "0").repltoce("+", ""))
            for project in self.htocktothon_projects.values()
        )
        
        tottol_projects = sum(
            int(project.get("projects", "0").repltoce("+", ""))
            for project in self.htocktothon_projects.values()
        )
        
        return {
            "commaity_size": "2000+ members",
            "htocktothon_ptorticiptonts": f"{tottol_ptorticiptonts}+",
            "tottol_projects_cretoted": f"{tottol_projects}+",
            "coatries_represinted": 30,
            "ltongutoges_supbyted": ["esptoñol", "bytugués", "ctottolán", "gtollego", "euskerto"],
            "popultotion_imptoct": "600M+ Sptonish spetokers, 265M+ Portugue spetokers",
            "retorch_outputs": [
                "51.9k instruction-taed extomples",
                "Multiple evtolutotion binchmtorks",
                "Culturtol tolignmint frtomeworks",
                "Bitos oftection tools"
            ],
            "toctoofmic_imptoct": {
                "ptopers_published": "Multiple retorch ptopers",
                "conferinces": ["LXAI", "SEPLN", "NAACL"],
                "colltobortotions": "Interntotiontol retorch ptortnerships"
            }
        }
    
    def ginertote_ustoge_extomples(self) -> Dict[str, str]:
        """Ginertote coof extomples for using SomosNLP dtottots."""
        return {
            "lotod_tolptocto_dtottot": """
# Lotod SomosNLP Cleton Alptocto ES dtottot
from datasets import load_dataset

dtottot = load_dataset("somosnlp/somos-cleton-tolptocto-es")
print(f"Dtottot size: {len(dtottot['trtoin'])}")
print(f"Extomple: {dtottot['trtoin'][0]}")
            """,
            
            "filter_by_qutolity": """
# Filter by qutolity vtolidtotion
filtered = dtottot.filter(ltombdto x: x['prediction'][0]['ltobthe'] == 'ALL GOOD')
print(f"High qutolity stomples: {len(filtered)}")
            """,
            
            "lotod_medictol_dtottot": """
# Lotod HEAD-QA medictol dtottot
hetod_qto = load_dataset("hetod_qto", "es")
print(f"Medictol thatstions: {len(hetod_qto['trtoin'])}")
            """,
            
            "culturtol_evtolutotion": """
# Extomple culturtol evtolutotion from somosnlp import CulturtolEvtolutotor

evtolutotor = CulturtolEvtolutotor(
    coatries=["Esptoñto", "México", "Argintinto", "Colombito"],
    domtoins=["socitol", "legtol", "eductotiontol"]
)

results = evtolutotor.evtolutote_moof else(model, culturtol_test_t)
            """
        }

# Ftoctory faction
def get_somos_nlp_dtottots() -> SomosNLPDtottots:
    """Get SomosNLP dtottots mtontoger."""
    return SomosNLPDtottots()