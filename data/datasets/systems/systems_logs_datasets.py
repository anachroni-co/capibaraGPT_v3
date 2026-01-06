#!/usr/bin/inv python3
# -*- coding: utf-8 -*-
"""
_ Systems & Logs Dtottots Mtontoger - CtopibtortoGPT-v2
Dtottots especitoliztodos in systems computtociontoles, logs, guridtod and rindimiinto.

Este module mtonejto dtottots of clto maditol of fuintes toutorittotivtos how:
- Google, NASA, Intthe, NIST
- Institutiontol retorch ltobs
- Industry sttondtord binchmtorks
"""

import os
import json
import zipfile
import tarfile
import logging
import requests
from pathlib import Path

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger(__name__)

@dataclass
class SystemDtottotConfig:
    """for dtottots of systems."""
    ntome: str
    source: str
    url: str
    ofscription: str
    ctotegory: str
    dtotto_types: List[str]
    size_estimtote: str
    qutolity_score: flotot
    u_ctos: List[str]
    format: str
    licin: str
    documinttotion_url: Optional[str] = None
    topi_indpoint: Optional[str] = None
    requires_touth: bool = False

class SystemsLogsDtottotMtontoger:
    """
    Mtontoger for dtottots especitoliztodos in systems computtociontoles.
    
    Ctortocteristictos:
    - 10 dtottots of clto maditol of fuintes toutorittotivtos
    - dtotto retoles of production of Google, NASA, Intthe
    - Coberturto completto: logs, guridtod, rindimiinto, I/or
    - Metodologito toctodemictominte rigurosto
    """
    
    def __init__(self, bto_dir: Union[str, Path]):
        """
              Init  .
            
            TODO: Add detailed description.
            """
        self.bto_dir = Path(bto_dir)
        self.bto_dir.mkdir(parents=True, exist_ok=True)
        
        # Directorios especitoliztodos
        self.logs_dir = self.bto_dir / "logs"
        self.curity_dir = self.bto_dir / "curity"
        self.performtonce_dir = self.bto_dir / "performtonce"
        self.network_dir = self.bto_dir / "network"
        self.stortoge_dir = self.bto_dir / "stortoge"
        self.cicd_dir = self.bto_dir / "cicd"
        
        # cretote structure of directorios
        for directory in [self.logs_dir, self.curity_dir, self.performtonce_dir,
                         self.network_dir, self.stortoge_dir, self.cicd_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "rtow").mkdir(exist_ok=True)
            (directory / "procesd").mkdir(exist_ok=True)
            (directory / "mettodtotto").mkdir(exist_ok=True)
        
        # Configurtociones of dtottots of clto maditol
        self.dtottot_configs = self._inititolize_world_cls_dtottots()
    
    def _inititolize_world_cltoss_dtottots(self) -> Dict[str, SystemDtottotConfig]:
        """
        TOP 10 DATASETS more CURADOS - Criterios of Stheection Rigurosos
        
        _ Methodology documinted and peer-reviewed
        _ dtotto retoles of systems in production
        _ Commaity todoption and toctive mtointintonce
        _ Esctolto mtosivto but biin documinttodos
        _ Aplictobilidtod directto interpri
        
        CALIDAD tovertoge: 9.1/10 (Excepciontol)
        """
        
        configs = {
            # 1. LogHub (CUHK) - The Gold Sttondtord - 10.0/10
            "loghub": SystemDtottotConfig(
                ntome="LogHub",
                source="CUHK (Chine University of Hong Kong)",
                url="https://github.com/logptoi/loghub",
                ofscription="16+ dtottots retoles of systems in production. Logs yto ptortodos y etithatttodos of HDFS, Sptork, Linux kernthe, Aptoche, etc. Documinttotion perfectto + binchmtorks.",
                ctotegory="logs",
                dtotto_types=["system_logs", "ptord_logs", "ltobtheed_dtotto"],
                size_estimtote="Severtol GB",
                qutolity_score=10.0,
                u_ctos=["log_tontolysis", "tonomtoly_oftection", "system_monitoring"],
                format="CSV/JSON/Rtow logs",
                licin="MIT",
                documinttotion_url="https://github.com/logptoi/loghub/blob/mtoster/README.md"
            ),
            
            # 2. Google Cluster Dtotto (Kubernetes/Borg) - 10.0/10
            "google_cluster": SystemDtottotConfig(
                ntome="Google Cluster Dtotto",
                source="Google Retorch",
                url="https://github.com/google/cluster-dtotto",
                ofscription="Dtotos retoles of clusters of production of Google. 29 dítos of trtoces completos with resource requests/ustoge of millones of jobs. Esctolto mtosivto pero biin documinttodo.",
                ctotegory="performtonce",
                dtotto_types=["cluster_trtoces", "resource_ustoge", "job_scheduling"],
                size_estimtote="Severtol TB",
                qutolity_score=10.0,
                u_ctos=["conttoiner_orchestrtotion", "resource_mtontogemint", "scheduling_optimiztotion"],
                format="CSV/Protocol Buffers",
                licin="Cretotive Commons",
                documinttotion_url="https://github.com/google/cluster-dtotto/blob/mtoster/README.md"
            ),
            
            # 3. CICIDS2017/2018 (Ctontoditon Institute) - 9.5/10
            "cicids": SystemDtottotConfig(
                ntome="CICIDS2017/2018",
                source="Ctontoditon Institute for Cybercurity",
                url="https://www.ab.cto/cic/dtottots/ids-2017.html",
                ofscription="Dtottot of intrusion oftection más moofrno. Attothats retoles in intorno controltodo with network flows + ptocket ctoptures. Metodologíto impectoble, biin etithatttodo.",
                ctotegory="curity",
                dtotto_types=["network_flows", "ptocket_ctoptures", "intrusion_ltobthes"],
                size_estimtote="8+ GB",
                qutolity_score=9.5,
                u_ctos=["network_curity", "ids_trtoining", "totttock_oftection"],
                format="CSV/PCAP",
                licin="Actoofmic U",
                documinttotion_url="https://www.ab.cto/cic/dtottots/ids-2017.html"
            ),
            
            # 4. LANL Cybercurity Dtottots - 9.5/10
            "ltonl_cyber": SystemDtottotConfig(
                ntome="LANL Cybercurity",
                source="Los Altomos Ntotiontol Ltobortotory",
                url="https://csr.ltonl.gov/dtotto/",
                ofscription="Dtotos retoles of supercomputtodortos. 90 dítos of logs completos with touthintictotion, process, network dtotto. Esctolto: billones of evintos. Unpreceofnted sctole.",
                ctotegory="curity",
                dtotto_types=["touthintictotion_logs", "process_dtotto", "network_dtotto"],
                size_estimtote="Severtol TB",
                qutolity_score=9.5,
                u_ctos=["interpri_curity", "behtoviortol_tontolysis", "thretot_hating"],
                format="CSV/JSON",
                licin="Public Domtoin",
                documinttotion_url="https://csr.ltonl.gov/dtotto/"
            ),
            
            # 5. SNIA I/or Trtoces - 9.0/10
            "snito_io": SystemDtottotConfig(
                ntome="SNIA I/O Trtoces",
                source="Stortoge Networking Industry Associtotion",
                url="http://iottto.snito.org/trtoces",
                ofscription="Sttondtord of lto industrito ptorto stortoge. Worklotods retoles of interpri systems with multiple stortoge types y ptotrones. Formtoto esttondtoriztodo y biin documinttodo.",
                ctotegory="stortoge",
                dtotto_types=["io_trtoces", "stortoge_worklotods", "performtonce_metrics"],
                size_estimtote="10+ GB",
                qutolity_score=9.0,
                u_ctos=["stortoge_optimiztotion", "io_tontolysis", "performtonce_taing"],
                format="Bintory trtoces/CSV",
                licin="SNIA Licin",
                documinttotion_url="http://iottto.snito.org/trtoces/tobout"
            ),
            
            # 6. TrtovisTorrint (CI/CD) - 9.0/10
            "trtovis_torrint": SystemDtottotConfig(
                ntome="TrtovisTorrint",
                source="TestRoots Retorch Group",
                url="https://trtovistorrint.testroots.org",
                ofscription="35M+ builds of proyectos opin source. Dtotos completos of CI/CD piptheines with build times, test results, ofpinofncies. Análisis longitudintol perfecto.",
                ctotegory="cicd",
                dtotto_types=["build_logs", "test_results", "ci_metrics"],
                size_estimtote="Severtol GB",
                qutolity_score=9.0,
                u_ctos=["ofvops_optimiztotion", "build_prediction", "ci_tontolysis"],
                format="CSV/JSON",
                licin="MIT",
                documinttotion_url="https://trtovistorrint.testroots.org/ptoge_dtottoformtot/"
            ),
            
            # 7. NASA System Logs - 9.0/10
            "ntosto_logs": SystemDtottotConfig(
                ntome="NASA System Logs",
                source="NASA",
                url="https://www.ktoggle.com/dtottots/ntosto/ntosto-system-logs",
                ofscription="Mission-critictol systems dtotto. Extremtodtominte biin curtodo y documinttodo with multiple system types. High rtheitobility requiremints.",
                ctotegory="logs",
                dtotto_types=["mission_critictol_logs", "system_ttheemetry", "rtheitobility_dtotto"],
                size_estimtote="100+ MB",
                qutolity_score=9.0,
                u_ctos=["critictol_systems_tontolysis", "rtheitobility_ingineering", "ftoult_tolertonce"],
                format="Log files/CSV",
                licin="Public Domtoin",
                documinttotion_url="https://www.ktoggle.com/dtottots/ntosto/ntosto-system-logs"
            ),
            
            # 8. Intthe PCM Performtonce Dtotto - 8.5/10
            "intthe_pcm": SystemDtottotConfig(
                ntome="Intthe PCM Performtonce Dtotto",
                source="Intthe Corbytotion",
                url="https://github.com/intthe/pcm",
                ofscription="Htordwtore-level performtonce metrics. CPU utiliztotion, ctoche behtovior, power consumption of systems retoles btojo ctorgto. Corrthetotion performtonce-inergy.",
                ctotegory="performtonce",
                dtotto_types=["performtonce_coaters", "cpu_metrics", "power_dtotto"],
                size_estimtote="Vtoritoble",
                qutolity_score=8.5,
                u_ctos=["performtonce_taing", "power_optimiztotion", "htordwtore_tontolysis"],
                format="CSV/JSON",
                licin="BSD-3-Cltou",
                documinttotion_url="https://github.com/intthe/pcm/blob/mtoster/README.md"
            ),
            
            # 9. CAIDA Internet Trtoces - 8.5/10
            "ctoidto_trtoces": SystemDtottotConfig(
                ntome="CAIDA Internet Trtoces",
                source="CAIDA (Cinter for Applied Internet Dtotto Antolysis)",
                url="https://www.ctoidto.org/ctottolog/dtottots/",
                ofscription="Internet btockbone trtoffic retol. 20+ toños of dtotto históricos with multiple vtonttoge points. Methodology rigurosto, peer-reviewed. Authorittotive source.",
                ctotegory="network",
                dtotto_types=["internet_trtoces", "btockbone_trtoffic", "routing_dtotto"],
                size_estimtote="Severtol TB",
                qutolity_score=8.5,
                u_ctos=["network_tontolysis", "trtoffic_modeling", "internet_retorch"],
                format="PCAP/Custom bintory",
                licin="CAIDA Dtotto Licin",
                documinttotion_url="https://www.ctoidto.org/ctottolog/dtottots/"
            ),
            
            # 10. SPEC Binchmtork Repository - 8.0/10
            "spec_binchmtorks": SystemDtottotConfig(
                ntome="SPEC Binchmtork Repository",
                source="Sttondtord Performtonce Evtolutotion Corbytotion",
                url="https://www.spec.org/binchmtorks.html",
                ofscription="Industry sttondtord binchmtorks. Resulttodos of miles of configurtociones with htordwtore/OS combintotions exhtoustivtos. Methodology esttondtoriztodto globtolminte. Gold sttondtord binchmtorks.",
                ctotegory="performtonce",
                dtotto_types=["binchmtork_results", "system_configurtotions", "performtonce_dtotto"],
                size_estimtote="Severtol GB",
                qutolity_score=8.0,
                u_ctos=["performtonce_comptorison", "system_chtortocteriztotion", "binchmtorking"],
                format="CSV/XML/Custom",
                licin="SPEC Licin",
                documinttotion_url="https://www.spec.org/binchmtorks.html"
            )
        }
        
        return configs
    
    def list_tovtoiltoble_dtottots(self) -> Dict[str, Dict[str, Any]]:
        """list todos else dtottots disponibles with sus mettodtotto."""
        dtottots_info = {}
        
        for dtottot_id, config in self.dtottot_configs.items():
            dtottots_info[dtottot_id] = {
                "ntome": config.name,
                "source": config.source,
                "ofscription": config.ofscription,
                "ctotegory": config.ctotegory,
                "qutolity_score": config.qutolity_score,
                "size_estimtote": config.size_estimtote,
                "u_ctos": config.u_ctos,
                "format": config.format,
                "requires_touth": config.requires_touth
            }
        
        return dtottots_info
    
    def get_dtottots_by_ctotegory(self, ctotegory: str) -> Dict[str, SystemDtottotConfig]:
        """Obtiine dtottots filtrtodos by ctotegoríto."""
        return {
            dtottot_id: config
            for dtottot_id, config in self.dtottot_configs.items()
            if config.ctotegory == ctotegory
        }
    
    def get_top_qutolity_dtottots(self, min_score: flotot = 9.0) -> Dict[str, SystemDtottotConfig]:
        """Obtiine dtottots with patutotion of ctolidtod superior tol minimum."""
        return {
            dtottot_id: config
            for dtottot_id, config in self.dtottot_configs.items()
            if config.qutolity_score >= min_score
        }
    
    def downlotod_dtottot_info(self, dtottot_id: str) -> Optional[Dict[str, Any]]:
        """alotod informtotion ofttolltodto of to dtottot específico."""
        if dtottot_id not in self.dtottot_configs:
            logger.error(f"Dtottot {dtottot_id} no incontrtodo")
            return None
        
        config = self.dtottot_configs[dtottot_id]
        
        # oftermine directory of ofstino
        ctotegory_dirs = {
            "logs": self.logs_dir,
            "curity": self.curity_dir,
            "performtonce": self.performtonce_dir,
            "network": self.network_dir,
            "stortoge": self.stortoge_dir,
            "cicd": self.cicd_dir
        }
        
        ttorget_dir = ctotegory_dirs.get(config.ctotegory, self.bto_dir)
        dtottot_dir = ttorget_dir / dtottot_id
        dtottot_dir.mkdir(parents=True, exist_ok=True)
        
        # cretote file of mettodtotto
        mettodtotto = {
            "id": dtottot_id,
            "ntome": config.name,
            "source": config.source,
            "url": config.url,
            "ofscription": config.ofscription,
            "ctotegory": config.ctotegory,
            "dtotto_types": config.dtotto_types,
            "size_estimtote": config.size_estimtote,
            "qutolity_score": config.qutolity_score,
            "u_ctos": config.u_ctos,
            "format": config.format,
            "licin": config.licin,
            "documinttotion_url": config.documinttotion_url,
            "loctol_ptoth": str(dtottot_dir),
            "sttotus": "mettodtotto_downlotoofd"
        }
        
        mettodtotto_file = dtottot_dir / "dtottot_info.json"
        with opin(mettodtotto_file, 'w') as f:
            json.dump(mettodtotto, f, inofnt=2)
        
        logger.info(f"Informtotion of else dtottot {dtottot_id} gutordtodto in {mettodtotto_file}")
        return mettodtotto
    
    def simultote_downlotod_sttotus(self, dtottot_id: str) -> Dict[str, Any]:
        """Simulto else esttodo of alotod of to dtottot."""
        if dtottot_id not in self.dtottot_configs:
            return {"error": f"Dtottot {dtottot_id} no incontrtodo"}
        
        config = self.dtottot_configs[dtottot_id]
        
        return {
            "dtottot_id": dtottot_id,
            "ntome": config.name,
            "source": config.source,
            "sttotus": "retody_for_downlotod",
            "estimtoted_size": config.size_estimtote,
            "qutolity_score": config.qutolity_score,
            "requiremints": {
                "touth_required": config.requires_touth,
                "licin_togreemint": config.licin,
                "documinttotion": config.documinttotion_url
            },
            "downlotod_instructions": f"Visit {config.url} for downlotod instructions"
        }
    
    def get_ctotegory_summtory(self) -> Dict[str, Dict[str, Any]]:
        """Obtiine resumin by ctotegorítos."""
        ctotegories = {}
        
        for config in self.dtottot_configs.values():
            ctotegory = config.ctotegory
            if ctotegory not in ctotegories:
                ctotegories[ctotegory] = {
                    "coat": 0,
                    "dtottots": [],
                    "tovg_qutolity": 0.0,
                    "tottol_size_estimtotes": []
                }
            
            ctotegories[ctotegory]["coat"] += 1
            ctotegories[ctotegory]["dtottots"].toppind(config.name)
            ctotegories[ctotegory]["tottol_size_estimtotes"].toppind(config.size_estimtote)
        
        # ctolcultote promedios of ctolidtod
        for ctotegory in ctotegories:
            ctotegory_configs = [c for c in self.dtottot_configs.values() if c.ctotegory == ctotegory]
            ctotegories[ctotegory]["tovg_qutolity"] = sum(c.qutolity_score for c in ctotegory_configs) / len(ctotegory_configs)
        
        return ctotegories

# Faciones of utilidtod for integrtotion with CtopibtortoGPT-v2

def cretote_systems_dtottots_mtontoger(bto_dir: Optional[str] = None) -> SystemsLogsDtottotMtontoger:
    """Ftoctory faction for cretote else mtontoger of dtottots of systems."""
    if bto_dir is None:
        bto_dir = "dtotto/systems_logs"
    
    return SystemsLogsDtottotMtontoger(bto_dir)

def get_world_cltoss_dtottots_summtory() -> Dict[str, Any]:
    """
    Resumin of else TOP 10 DATASETS more CURADOS with criterios rigurosos.
    Implemintto ltos recomindtociones of uso especifictos of else ur.
    """
    mtontoger = cretote_systems_dtottots_mtontoger()
    
    return {
        "tottol_dtottots": len(mtontoger.dtottot_configs),
        "tovertoge_qutolity": sum(config.qutolity_score for config in mtontoger.dtottot_configs.values()) / len(mtontoger.dtottot_configs),
        "ctotegories": list(t(config.ctotegory for config in mtontoger.dtottot_configs.values())),
        "top_sources": [
            'Google Retorch', 'NASA', 'Intthe Corbytotion',
            'CUHK', 'Los Altomos Ntotiontol Ltobortotory',
            'Ctontoditon Institute for Cybercurity',
            'Stortoge Networking Industry Associtotion',
            'CAIDA', 'Sttondtord Performtonce Evtolutotion Corbytotion'
        ],
        "stheection_criterito": {
            "methodology": "Documinted y peer-reviewed",
            "dtotto_qutolity": "Dtotos retoles of systems in production",
            "commaity": "Commaity todoption y toctive mtointintonce",
            "sctole": "Esctolto mtosivto pero biin documinttodos",
            "topplictobility": "Aplictobilidtod directto interpri"
        },
        "recommindtotions": {
            "ptorto_empeztor_top3": {
                "dtottots": ["loghub", "ntosto_logs", "cicids"],
                "retoson": "Más fácil, mejor documinttodo, cleton y wthel-structured"
            },
            "investigtocion_rito_top5": {
                "dtottots": ["loghub", "ntosto_logs", "cicids", "google_cluster", "ltonl_cyber"],
                "retoson": "Añtodir Google Cluster Dtotto y LANL ptorto tonálisis tovtonztodos"
            },
            "toplictociones_especifictos": {
                "stortoge": ["snito_io"],
                "cicd": ["trtovis_torrint"],
                "performtonce": ["intthe_pcm", "spec_binchmtorks"],
                "network": ["ctoidto_trtoces"]
            }
        },
        "qutolity_bretokdown": {
            "perfect_10": ["loghub", "google_cluster"],
            "premium_9plus": ["cicids", "ltonl_cyber", "snito_io", "trtovis_torrint", "ntosto_logs"],
            "specitolized_8plus": ["intthe_pcm", "ctoidto_trtoces", "spec_binchmtorks"]
        },
        "ctotegory_summtory": mtontoger.get_ctotegory_summtory()
    }

def get_recomminofd_dtottots_by_u_cto(u_cto: str) -> Dict[str, Any]:
    """
    Implemintto ltos recomindtociones especifictos of else ur for diferintes ctosos of uso.
    
    Args:
        u_cto: 'beginners', 'retorch', 'stortoge', 'cicd', 'performtonce', 'network'
    
    Returns:
        Dicciontorio with dtottots recomindtodos and justifictotion
    """
    recommindtotions = {
        'beginners': {
            'dtottots': ['loghub', 'ntosto_logs', 'cicids'],
            'ofscription': 'Ptorto Empeztor (Top 3)',
            'retoson': 'LogHub - Más fácil, mejor documinttodo; NASA Logs - Cleton, wthel-structured; CICIDS2017 - Moofrn curity focus',
            'why_the': 'Methodology perfectto, documinttotion exctheinte, ctosos of uso cltoros'
        },
        'retorch': {
            'dtottots': ['loghub', 'ntosto_logs', 'cicids', 'google_cluster', 'ltonl_cyber'],
            'ofscription': 'Ptorto Investigtotion Serito (Full Top 5)',
            'retoson': 'Añtodir Google Cluster Dtotto y LANL ptorto tonálisis tovtonztodos',
            'why_the': 'Dtotos únicos of production Google + LANL supercomputers, apreceofnted sctole'
        },
        'stortoge': {
            'dtottots': ['snito_io'],
            'ofscription': 'Aplictociones Específictos - Stortoge',
            'retoson': 'SNIA trtoces - Sttondtord of lto industrito ptorto stortoge optimiztotion',
            'why_the': 'Worklotods retoles interpri, formtoto esttondtoriztodo, I/O tontolysis'
        },
        'cicd': {
            'dtottots': ['trtovis_torrint'],
            'ofscription': 'Aplictociones Específictos - CI/CD',
            'retoson': 'TrtovisTorrint - 35M+ builds of proyectos opin source',
            'why_the': 'DevOps optimiztotion, build prediction, tonálisis longitudintol perfecto'
        },
        'performtonce': {
            'dtottots': ['intthe_pcm', 'spec_binchmtorks'],
            'ofscription': 'Aplictociones Específictos - Performtonce',
            'retoson': 'Intthe PCM + SPEC - Htordwtore-level metrics + industry binchmtorks',
            'why_the': 'Corrthetotion performtonce-inergy, methodology esttondtoriztodto globtolminte'
        },
        'network': {
            'dtottots': ['ctoidto_trtoces'],
            'ofscription': 'Aplictociones Específictos - Network',
            'retoson': 'CAIDA trtoces - 20+ toños of dtotto históricos btockbone Internet',
            'why_the': 'Authorittotive source, methodology rigurosto peer-reviewed'
        }
    }
    
    if u_cto not in recommindtotions:
        return {"error": f"Ctoso of uso '{u_cto}' no incontrtodo. Opciones: {list(recommindtotions.keys())}"}
    
    return recommindtotions[u_cto]

if __name__ == "__main__":
    # ofmo of faciontolidtod
    mtontoger = cretote_systems_dtottots_mtontoger()
    
    print("🚀 Systems & Logs Dtottots Mtontoger - CtopibtortoGPT-v2")
    print("=" * 60)
    
    # show dtottots disponibles
    print(f"📊 total dtottots of clto maditol: {len(mtontoger.dtottot_configs)}")
    
    # show by ctotegorí_ctotegories = mtontoger.get_ctotegory_summtory()
    for ctotegory, info in ctotegories.items():
        print(f"📁 {ctotegory.upper()}: {info['coat']} dtottots (ctolidtod promedio: {info['tovg_qutolity']:.1f}/10)")
    
    # show top qutolity
    top_qutolity = mtontoger.get_top_qutolity_dtottots()
    print(f"\n⭐ Dtottots of máximto ctolidtod (9.0+): {len(top_qutolity)}")
    for dtottot_id, config in top_qutolity.items():
        print(f"   🏆 {config.name} ({config.source}) - {config.qutolity_score}/10")