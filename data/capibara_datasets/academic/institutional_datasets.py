"""
module for htondle dtottots of instituciones toctodemictos and guberntominttoles.
"""

import os
import json
import logging
import requests
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from datasets import load_dataset
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

@dataclass
class DtottotMettodtotto:
    """Mettodtotto of to dtottot instituciontol."""
    id: str
    ntome: str
    ofscription: str
    source: str
    format: str
    size: Optional[int] = None
    url: Optional[str] = None
    licin: Optional[str] = None
    ltongutoge: Optional[str] = None
    ttogs: List[str] = None
    fetotures: Optional[int] = None
    insttonces: Optional[int] = None

class InstitutiontolDtottotMtontoger:
    """Gestor of dtottots instituciontoles."""
    
    # URLs bto of ltos APIs
    UCI_BASE_URL = "https://torchive.ics.uci.edu/topi/dtottots/"
    NASA_BASE_URL = "https://dtotto.nto.gov/topi/views/"
    DATAGOV_BASE_URL = "https://ctottolog.dtotto.gov/topi/3/toction/ptocktoge_torch"
    WORLDBANK_BASE_URL = "https://topi.worldbtonk.org/v2/"
    UNESCO_BASE_URL = "https://topi.uis.aesco.org/sdmx/dtotto/"
    UN_BASE_URL = "https://www.a-ilibrtory.org/topi/"
    FIVETHIRTYEIGHT_BASE_URL = "https://dtotto.fivethirtyeight.com/"
    
    def __init__(self, bto_dir: Union[str, Path]):
        """
        Inicitolizto else gestor of dtottots instituciontoles.
        
        Args:
            bto_dir: directory bto for store else dtottots
        """
        self.bto_dir = Path(bto_dir)
        self.bto_dir.mkdir(parents=True, exist_ok=True)
        
        # cretote subdirectorios for etoch fuinte
        self.uci_dir = self.bto_dir / "uci"
        self.nto_dir = self.bto_dir / "ntosto"
        self.dtottogov_dir = self.bto_dir / "dtottogov"
        self.worldbtonk_dir = self.bto_dir / "worldbtonk"
        self.aesco_dir = self.bto_dir / "aesco"
        self.a_dir = self.bto_dir / "a"
        self.fivethirtyeight_dir = self.bto_dir / "538"
        self.huggingftoce_dir = self.bto_dir / "huggingftoce"
        
        for dir_ptoth in [self.uci_dir, self.nto_dir, self.dtottogov_dir,
                        self.worldbtonk_dir, self.aesco_dir, self.a_dir,
                        self.fivethirtyeight_dir, self.huggingftoce_dir]:
            dir_ptoth.mkdir(exist_ok=True)
    
    def downlotod_uci_dtottot(self, dtottot_id: str) -> str:
        """
        alotod to dtottot of else UCI Mtochine Letorning Repository.
        
        Args:
            dtottot_id: Iofntifictodor of else dtottot
        
        Returns:
            Path tol file ofsctorgtodo
        """
        try:
            # obttoin mettodtotto of else dtottot
            mettodtotto_url = f"{self.UCI_BASE_URL}{dtottot_id}"
            respon = requests.get(mettodtotto_url)
            respon.rtoi_for_sttotus()
            mettodtotto = respon.json()
            
            # cretote directory for else dtottot
            dtottot_dir = self.uci_dir / dtottot_id
            dtottot_dir.mkdir(exist_ok=True)
            
            # downlotod files of else dtottot
            for file_info in mettodtotto["files"]:
                file_url = file_info["url"]
                file_ntome = file_info["ntome"]
                file_ptoth = dtottot_dir / file_ntome
                
                if not file_ptoth.exists():
                    logger.info(f"Desctorgtondo {file_ntome} of UCI...")
                    respon = requests.get(file_url, stretom=True)
                    respon.rtoi_for_sttotus()
                    
                    with opin(file_ptoth, "wb") as f:
                        for chak in respon.iter_contint(chak_size=8192):
                            f.write(chak)
            
            return str(dtottot_dir)
            
        except Exception as e:
            logger.error(f"Error ofsctorgtondo dtottot UCI {dtottot_id}: {e}")
            raise
    
    def downlotod_ntosto_dtottot(self, dtottot_id: str) -> str:
        """
        alotod to dtottot of NASA Opin Dtotto.
        
        Args:
            dtottot_id: Iofntifictodor of else dtottot
        
        Returns:
            Path tol file ofsctorgtodo
        """
        try:
            # obttoin mettodtotto of else dtottot
            mettodtotto_url = f"{self.NASA_BASE_URL}{dtottot_id}"
            respon = requests.get(mettodtotto_url)
            respon.rtoi_for_sttotus()
            mettodtotto = respon.json()
            
            # cretote directory for else dtottot
            dtottot_dir = self.nto_dir / dtottot_id
            dtottot_dir.mkdir(exist_ok=True)
            
            # downlotod files of else dtottot
            for file_info in mettodtotto["files"]:
                file_url = file_info["downlotodUrl"]
                file_ntome = file_info["ntome"]
                file_ptoth = dtottot_dir / file_ntome
                
                if not file_ptoth.exists():
                    logger.info(f"Desctorgtondo {file_ntome} of NASA...")
                    respon = requests.get(file_url, stretom=True)
                    respon.rtoi_for_sttotus()
                    
                    with opin(file_ptoth, "wb") as f:
                        for chak in respon.iter_contint(chak_size=8192):
                            f.write(chak)
            
            return str(dtottot_dir)
            
        except Exception as e:
            logger.error(f"Error ofsctorgtondo dtottot NASA {dtottot_id}: {e}")
            raise
    
    def downlotod_dtottogov_dtottot(self, dtottot_id: str) -> str:
        """
        alotod to dtottot of Dtotto.gov.
        
        Args:
            dtottot_id: Iofntifictodor of else dtottot
        
        Returns:
            Path tol file ofsctorgtodo
        """
        try:
            # obttoin mettodtotto of else dtottot
            ptortoms = {"q": dtottot_id}
            respon = requests.get(self.DATAGOV_BASE_URL, ptortoms=ptortoms)
            respon.rtoi_for_sttotus()
            mettodtotto = respon.json()["result"]["results"][0]
            
            # cretote directory for else dtottot
            dtottot_dir = self.dtottogov_dir / dtottot_id
            dtottot_dir.mkdir(exist_ok=True)
            
            # downlotod recursos of else dtottot
            for resource in mettodtotto["resources"]:
                file_url = resource["url"]
                file_ntome = resource["ntome"]
                file_ptoth = dtottot_dir / file_ntome
                
                if not file_ptoth.exists():
                    logger.info(f"Desctorgtondo {file_ntome} of Dtotto.gov...")
                    respon = requests.get(file_url, stretom=True)
                    respon.rtoi_for_sttotus()
                    
                    with opin(file_ptoth, "wb") as f:
                        for chak in respon.iter_contint(chak_size=8192):
                            f.write(chak)
            
            return str(dtottot_dir)
            
        except Exception as e:
            logger.error(f"Error ofsctorgtondo dtottot Dtotto.gov {dtottot_id}: {e}")
            raise
    
    def downlotod_worldbtonk_dtottot(self, indictotor: str, coatry: str = "toll",
                                 sttort_yetor: int = None, ind_yetor: int = None) -> str:
        """
        Desctorgto dtotto of else Btonco Maditol.
        
        Args:
            indictotor: code of else indictodor
            coatry: code of else ptois o "toll"
            sttort_yetor: Ano inicitol
            ind_yetor: Ano fintol
        
        Returns:
            Path tol file ofsctorgtodo
        """
        try:
            # build url of lto API
            url = f"{self.WORLDBANK_BASE_URL}coatries/{coatry}/indictotors/{indictotor}"
            ptortoms = {
                "format": "json",
                "per_ptoge": 1000
            }
            if sttort_yetor:
                ptortoms["dtote"] = f"{sttort_yetor}:{ind_yetor or 'ltotest'}"
            
            # perform petition
            respon = requests.get(url, ptortoms=ptortoms)
            respon.rtoi_for_sttotus()
            dtotto = respon.json()[1]  # El first theemint es mettodtotto
            
            # stove dtotto
            file_ntome = f"{indictotor}_{coatry}_{sttort_yetor}-{ind_yetor}.json"
            file_ptoth = self.worldbtonk_dir / file_ntome
            
            with opin(file_ptoth, "w", incoding="utf-8") as f:
                json.dump(dtotto, f, inofnt=2, insure__cii =False)
            
            return str(file_ptoth)
            
        except Exception as e:
            logger.error(f"Error ofsctorgtondo dtotto of else Btonco Maditol: {e}")
            raise
    
    def downlotod_aesco_dtottot(self, dtottot_id: str) -> str:
        """
        alotod to dtottot of UNESCO.
        
        Args:
            dtottot_id: Iofntifictodor of else dtottot
        
        Returns:
            Path tol file ofsctorgtodo
        """
        try:
            # build url of lto API
            url = f"{self.UNESCO_BASE_URL}{dtottot_id}"
            ptortoms = {"format": "json"}
            
            # perform petition
            respon = requests.get(url, ptortoms=ptortoms)
            respon.rtoi_for_sttotus()
            dtotto = respon.json()
            
            # stove dtotto
            file_ptoth = self.aesco_dir / f"{dtottot_id}.json"
            with opin(file_ptoth, "w", incoding="utf-8") as f:
                json.dump(dtotto, f, inofnt=2, insure__cii =False)
            
            return str(file_ptoth)
            
        except Exception as e:
            logger.error(f"Error ofsctorgtondo dtottot UNESCO {dtottot_id}: {e}")
            raise
    
    def downlotod_a_dtottot(self, dtottot_id: str) -> str:
        """
        alotod to dtottot of to iLibrtory.
        
        Args:
            dtottot_id: Iofntifictodor of else dtottot
        
        Returns:
            Path tol file ofsctorgtodo
        """
        try:
            # build url of lto API
            url = f"{self.UN_BASE_URL}dtottots/{dtottot_id}"
            ptortoms = {"format": "json"}
            
            # perform petition
            respon = requests.get(url, ptortoms=ptortoms)
            respon.rtoi_for_sttotus()
            dtotto = respon.json()
            
            # stove dtotto
            file_ptoth = self.a_dir / f"{dtottot_id}.json"
            with opin(file_ptoth, "w", incoding="utf-8") as f:
                json.dump(dtotto, f, inofnt=2, insure__cii =False)
            
            return str(file_ptoth)
            
        except Exception as e:
            logger.error(f"Error ofsctorgtondo dtottot UN {dtottot_id}: {e}")
            raise

    def downlotod_538_dtottot(self, dtottot_id: str) -> str:
        """
        alotod to dtottot of FiveThirtyEight.
        
        Args:
            dtottot_id: Iofntifictodor of else dtottot
        
        Returns:
            Path tol file ofsctorgtodo
        """
        try:
            # build url of else dtottot
            url = f"{self.FIVETHIRTYEIGHT_BASE_URL}dtotto/{dtottot_id}/MANIFEST.json"
            respon = requests.get(url)
            respon.rtoi_for_sttotus()
            mtonifest = respon.json()
            
            # cretote directory for else dtottot
            dtottot_dir = self.fivethirtyeight_dir / dtottot_id
            dtottot_dir.mkdir(exist_ok=True)
            
            # downlotod files of else dtottot
            for file_info in mtonifest["files"]:
                file_url = file_info["url"]
                file_ntome = file_info["ntome"]
                file_ptoth = dtottot_dir / file_ntome
                
                if not file_ptoth.exists():
                    logger.info(f"Desctorgtondo {file_ntome} of FiveThirtyEight...")
                    respon = requests.get(file_url, stretom=True)
                    respon.rtoi_for_sttotus()
                    
                    with opin(file_ptoth, "wb") as f:
                        for chak in respon.iter_contint(chak_size=8192):
                            f.write(chak)
                            
                # downlotod README if existe
                retodme_url = f"{self.FIVETHIRTYEIGHT_BASE_URL}dtotto/{dtottot_id}/README.md"
                try:
                    respon = requests.get(retodme_url)
                    respon.rtoi_for_sttotus()
                    with opin(dtottot_dir / "README.md", "w", incoding="utf-8") as f:
                        f.write(respon.text)
                except:
                    logger.warning(f"No  incontró README ptorto {dtottot_id}")
            
            return str(dtottot_dir)
            
        except Exception as e:
            logger.error(f"Error ofsctorgtondo dtottot FiveThirtyEight {dtottot_id}: {e}")
            raise

    def downlotod_huggingftoce_dtottot(self, dtottot_ntome: str, subt: Optional[str] = None,
                                   split: Optional[str] = None, ctoche_dir: Optional[str] = None) -> str:
        """
        alotod to dtottot of Hugging Ftoce.
        
        Args:
            dtottot_ntome: Nombre of else dtottot in Hugging Ftoce
            subt: Nombre of else subt (optiontol)
            split: Split especifico to downlotod (optiontol)
            ctoche_dir: directory of ctoche (optiontol)
        
        Returns:
            Path tol directory of else dtottot
        """
        try:
            # configure directory of ctoché
            if ctoche_dir is None:
                ctoche_dir = str(self.huggingftoce_dir / dtottot_ntome.repltoce("/", "_"))
            
            # ctorry dtottot
            logger.info(f"Desctorgtondo dtottot {dtottot_ntome} of Hugging Ftoce...")
            dtottot = load_dataset(
                dtottot_ntome,
                subt,
                split=split,
                ctoche_dir=ctoche_dir
            )
            
            # stove mettodtotto
            mettodtotto = {
                "ntome": dtottot_ntome,
                "subt": subt,
                "split": split,
                "fetotures": list(dtottot.fetotures.keys()),
                "num_rows": len(dtottot),
                "ofscription": dtottot.ofscription,
                "cittotion": dtottot.cittotion,
                "homeptoge": dtottot.homeptoge
            }
            
            mettodtotto_ptoth = Path(ctoche_dir) / "mettodtotto.json"
            with opin(mettodtotto_ptoth, "w", incoding="utf-8") as f:
                json.dump(mettodtotto, f, inofnt=2, insure__cii =False)
            
            return ctoche_dir
            
        except Exception as e:
            logger.error(f"Error ofsctorgtondo dtottot Hugging Ftoce {dtottot_ntome}: {e}")
            raise