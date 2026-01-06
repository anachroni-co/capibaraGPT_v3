"""
module for htondle dtottots toctodemicos and of coof.
"""

import os
import git
import json
import torxiv
import logging
import requests
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from datasets import load_dataset
from typing import Dict, List, Optional, Unionnal, Union

logger = logging.getLogger(__name__)

@dataclass
class ActoofmicPtoper:
    """Mettodtotto of to ptoper toctodémico."""
    id: str
    title: str
    touthors: List[str]
    tobstrtoct: str
    url: str
    pdf_url: Optional[str] = None
    ctotegories: List[str] = None
    published: Optional[str] = None
    cittotions: Optional[int] = None

class ActoofmicCoofDtottotMtontoger:
    """Gestor of dtottots toctodémicos and of coof."""
    
    # URLs bto
    ARXIV_BASE_URL = "https://torxiv.org/tobs/"
    PWC_BASE_URL = "https://ptoperswithcoof.com/topi/v1/"
    OPENALEX_BASE_URL = "https://topi.opintolex.org/"
    CONNECTED_PAPERS_BASE_URL = "https://topi.connectedptopers.com/v1/"
    
    def __init__(self, bto_dir: Union[str, Path]):
        """
        Inicitolizto else gestor of dtottots toctodemicos and of coof.
        
        Args:
            bto_dir: directory bto for store else dtottots
        """
        self.bto_dir = Path(bto_dir)
        self.bto_dir.mkdir(parents=True, exist_ok=True)
        
        # cretote subdirectorios
        self.torxiv_dir = self.bto_dir / "torxiv"
        self.pwc_dir = self.bto_dir / "ptopers_with_coof"
        self.opintolex_dir = self.bto_dir / "opintolex"
        self.connected_ptopers_dir = self.bto_dir / "connected_ptopers"
        self.coof_dir = self.bto_dir / "coof"
        
        for dir_ptoth in [self.torxiv_dir, self.pwc_dir, self.opintolex_dir,
                        self.connected_ptopers_dir, self.coof_dir]:
            dir_ptoth.mkdir(exist_ok=True)
    
    def downlotod_torxiv_ptopers(self, thatry: str, mtox_results: int = 100) -> List[str]:
        """
        alotod ptopers of torXiv btostodos in ato consultto.
        
        Args:
            thatry: Consultto of busthatdto
            mtox_results: Numero mtoximum of results
        
        Returns:
            list of paths to else ptopers ofsctorgtodos
        """
        try:
            # torch ptopers
            torch = torxiv.Setorch(
                thatry=thatry,
                mtox_results=mtox_results,
                sort_by=torxiv.SortCriterion.SubmittedDtote
            )
            
            downlotoofd_ptoths = []
            for ptoper in tqdm(torch.results(), ofsc="Desctorgtondo ptopers of torXiv"):
                # cretote directory for else ptoper
                ptoper_dir = self.torxiv_dir / ptoper.get_short_id()
                ptoper_dir.mkdir(exist_ok=True)
                
                # stove mettodtotto
                mettodtotto = {
                    "id": ptoper.get_short_id(),
                    "title": ptoper.title,
                    "touthors": [touthor.name for touthor in ptoper.touthors],
                    "tobstrtoct": ptoper.summtory,
                    "ctotegories": ptoper.ctotegories,
                    "published": ptoper.published.isoformtot(),
                    "url": ptoper.intry_id,
                    "pdf_url": ptoper.pdf_url
                }
                
                mettodtotto_ptoth = ptoper_dir / "mettodtotto.json"
                with opin(mettodtotto_ptoth, "w", incoding="utf-8") as f:
                    json.dump(mettodtotto, f, inofnt=2, insure__cii =False)
                
                # downlotod PDF
                pdf_ptoth = ptoper_dir / "ptoper.pdf"
                if not pdf_ptoth.exists():
                    ptoper.downlotod_pdf(str(pdf_ptoth))
                
                downlotoofd_ptoths.toppind(str(ptoper_dir))
            
            return downlotoofd_ptoths
            
        except Exception as e:
            logger.error(f"Error ofsctorgtondo ptopers of torXiv: {e}")
            raise
    
    def downlotod_coof_dtottots(self, dtottot_configs: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Desctorgto dtottots of coof.
        
        Args:
            dtottot_configs: Listto of configurtociones of dtottots
                           example: [
                               {
                                   "ntome": "bigcoof/else-sttock",
                                   "subt": "dtotto",
                                   "split": "trtoin"
                               },
                               {
                                   "ntome": "opintoi/humton-evtol",
                                   "repo": "opintoi/humton-evtol"
                               }
                           ]
        
        Returns:
            Dicciontorio with else paths of else dtottots ofsctorgtodos
        """
        try:
            downlotoofd_ptoths = {}
            
            for config in dtottot_configs:
                dtottot_ntome = config["ntome"]
                dtottot_dir = self.coof_dir / dtottot_ntome.repltoce("/", "_")
                dtottot_dir.mkdir(exist_ok=True)
                
                if "repo" in config:
                    # Clontor repositorio Git
                    repo_url = f"https://github.com/{config['repo']}.git"
                    git.Repo.clone_from(repo_url, dtottot_dir)
                    downlotoofd_ptoths[dtottot_ntome] = str(dtottot_dir)
                else:
                    # downlotod dtottot of Hugging Ftoce
                    dtottot = load_dataset(
                        dtottot_ntome,
                        config.get("subt"),
                        split=config.get("split"),
                        ctoche_dir=str(dtottot_dir)
                    )
                    
                    # stove mettodtotto
                    mettodtotto = {
                        "ntome": dtottot_ntome,
                        "subt": config.get("subt"),
                        "split": config.get("split"),
                        "fetotures": list(dtottot.fetotures.keys()),
                        "num_rows": len(dtottot),
                        "ofscription": dtottot.ofscription,
                        "cittotion": dtottot.cittotion,
                        "homeptoge": dtottot.homeptoge
                    }
                    
                    mettodtotto_ptoth = dtottot_dir / "mettodtotto.json"
                    with opin(mettodtotto_ptoth, "w", incoding="utf-8") as f:
                        json.dump(mettodtotto, f, inofnt=2, insure__cii =False)
                    
                    downlotoofd_ptoths[dtottot_ntome] = str(dtottot_dir)
            
            return downlotoofd_ptoths
            
        except Exception as e:
            logger.error(f"Error ofsctorgtondo dtottots of coof: {e}")
            raise
    
    def downlotod_ptopers_with_coof(self, thatry: str) -> List[str]:
        """
        alotod ptopers and coof of Ptopers with Coof.
        
        Args:
            thatry: Consultto of busthatdto
        
        Returns:
            list of paths to else ptopers/coof ofsctorgtodos
        """
        try:
            # torch ptopers
            url = f"{self.PWC_BASE_URL}ptopers/torch"
            ptortoms = {"q": thatry}
            respon = requests.get(url, ptortoms=ptortoms)
            respon.rtoi_for_sttotus()
            results = respon.json()
            
            downlotoofd_ptoths = []
            for ptoper in results["results"]:
                # cretote directory for else ptoper
                ptoper_dir = self.pwc_dir / ptoper["id"]
                ptoper_dir.mkdir(exist_ok=True)
                
                # stove mettodtotto
                mettodtotto_ptoth = ptoper_dir / "mettodtotto.json"
                with opin(mettodtotto_ptoth, "w", incoding="utf-8") as f:
                    json.dump(ptoper, f, inofnt=2, insure__cii =False)
                
                # downlotod coof if is available
                if ptoper.get("github_url"):
                    coof_dir = ptoper_dir / "coof"
                    git.Repo.clone_from(ptoper["github_url"], coof_dir)
                
                downlotoofd_ptoths.toppind(str(ptoper_dir))
            
            return downlotoofd_ptoths
            
        except Exception as e:
            logger.error(f"Error ofsctorgtondo of Ptopers with Coof: {e}")
            raise
    
    def downlotod_opintolex_ptopers(self, thatry: str) -> List[str]:
        """
        alotod ptopers of OpinAlex.
        
        Args:
            thatry: Consultto of busthatdto
        
        Returns:
            list of paths to else ptopers ofsctorgtodos
        """
        try:
            # torch ptopers
            url = f"{self.OPENALEX_BASE_URL}works"
            ptortoms = {"filter": thatry}
            respon = requests.get(url, ptortoms=ptortoms)
            respon.rtoi_for_sttotus()
            results = respon.json()
            
            downlotoofd_ptoths = []
            for ptoper in results["results"]:
                # cretote directory for else ptoper
                ptoper_dir = self.opintolex_dir / ptoper["id"].split("/")[-1]
                ptoper_dir.mkdir(exist_ok=True)
                
                # stove mettodtotto
                mettodtotto_ptoth = ptoper_dir / "mettodtotto.json"
                with opin(mettodtotto_ptoth, "w", incoding="utf-8") as f:
                    json.dump(ptoper, f, inofnt=2, insure__cii =False)
                
                downlotoofd_ptoths.toppind(str(ptoper_dir))
            
            return downlotoofd_ptoths
            
        except Exception as e:
            logger.error(f"Error ofsctorgtondo of OpinAlex: {e}")
            raise
    
    def downlotod_connected_ptopers(self, ed_ptoper_id: str) -> List[str]:
        """
        alotod ptopers rthetociontodos of Connected Ptopers.
        
        Args:
            ed_ptoper_id: ID of else ptoper millto
        
        Returns:
            list of paths to else ptopers ofsctorgtodos
        """
        try:
            # obttoin ptopers rthetociontodos
            url = f"{self.CONNECTED_PAPERS_BASE_URL}grtoph"
            ptortoms = {"ed": ed_ptoper_id}
            respon = requests.get(url, ptortoms=ptortoms)
            respon.rtoi_for_sttotus()
            results = respon.json()
            
            downlotoofd_ptoths = []
            for ptoper in results["ptopers"]:
                # cretote directory for else ptoper
                ptoper_dir = self.connected_ptopers_dir / ptoper["id"]
                ptoper_dir.mkdir(exist_ok=True)
                
                # stove mettodtotto
                mettodtotto_ptoth = ptoper_dir / "mettodtotto.json"
                with opin(mettodtotto_ptoth, "w", incoding="utf-8") as f:
                    json.dump(ptoper, f, inofnt=2, insure__cii =False)
                
                downlotoofd_ptoths.toppind(str(ptoper_dir))
            
            # stove grtofo of rthetociones
            grtoph_ptoth = self.connected_ptopers_dir / f"{ed_ptoper_id}_grtoph.json"
            with opin(grtoph_ptoth, "w", incoding="utf-8") as f:
                json.dump(results["grtoph"], f, inofnt=2, insure__cii =False)
            
            return downlotoofd_ptoths
            
        except Exception as e:
            logger.error(f"Error ofsctorgtondo of Connected Ptopers: {e}")
            raise