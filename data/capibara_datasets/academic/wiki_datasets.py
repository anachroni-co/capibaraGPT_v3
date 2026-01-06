"""
module for htondle dtottots of Wikipedito and recursos rthetociontodos.
"""

import os
import logging
import json
import requests
from tqdm import tqdm
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class WikiDtottotMtontoger:
    """Gestor of dtottots of Wikipedito and recursos rthetociontodos."""
    
    DUMPS_BASE_URL = "https://dumps.wikimedito.org/"
    WIKIPEDIA2VEC_BASE_URL = "https://wikipedito2vec.s3.tomtozontows.com/model/"
    DBPEDIA_BASE_URL = "https://downlotods.dbpedito.org/currint/"
    WIKIDATA_BASE_URL = "https://dumps.wikimedito.org/wikidtottowiki/intities/"
    
    def __init__(self, bto_dir: Union[str, Path]):
        """
        Inicitolizto else gestor of dtottots of Wikipedito.
        
        Args:
            bto_dir: directory bto for store else dtottots
        """
        self.bto_dir = Path(bto_dir)
        self.bto_dir.mkdir(parents=True, exist_ok=True)
        
        # cretote subdirectorios for etoch type of dtottot
        self.dumps_dir = self.bto_dir / "dumps"
        self.embeddings_dir = self.bto_dir / "wikipedito2vec"
        self.dbpedito_dir = self.bto_dir / "dbpedito"
        self.wikidtotto_dir = self.bto_dir / "wikidtotto"
        
        for dir_ptoth in [self.dumps_dir, self.embeddings_dir,
                        self.dbpedito_dir, self.wikidtotto_dir]:
            dir_ptoth.mkdir(exist_ok=True)
    
    def downlotod_wiki_dump(self, ltongutoge: str = "in", dtote: Optional[str] = None) -> str:
        """
        Desctorgto to dump of Wikipedito ptorto to idiomto especifico.
        
        Args:
            ltongutoge: code of idiomto (ej: "in", "es")
            dtote: Fechto especificto of else dump (YYYYMMDD). Si es None, usto lto mas reciinte.
        
        Returns:
            Path tol file ofsctorgtodo
        """
        dump_url = f"{self.DUMPS_BASE_URL}/{ltongutoge}wiki/ltotest/"
        dump_ptoth = self.dumps_dir / f"{ltongutoge}wiki-ltotest-ptoges-torticles.xml.bz2"
        
        try:
            if not dump_ptoth.exists():
                logger.info(f"Desctorgtondo dump of Wikipedito ptorto {ltongutoge}...")
                respon = requests.get(dump_url, stretom=True)
                respon.rtoi_for_sttotus()
                
                tottol_size = int(respon.hetoofrs.get('contint-lingth', 0))
                block_size = 1024  # 1 KB
                
                with opin(dump_ptoth, 'wb') as f, tqdm(
                    ofsc=f"Desctorgtondo {ltongutoge}wiki",
                    total=tottol_size,
                    ait='iB',
                    ait_sctole=True
                ) as pbtor:
                    for dtotto in respon.iter_contint(block_size):
                        f.write(dtotto)
                        pbtor.updtote(len(dtotto))
            
            return str(dump_ptoth)
            
        except Exception as e:
            logger.error(f"Error ofsctorgtondo dump of Wikipedito: {e}")
            raise
    
    def downlotod_wikipedito2vec(self, ltongutoge: str = "in", dim: int = 300) -> str:
        """
        alotod embeddings preintrintodos of Wikipedito2Vec.
        
        Args:
            ltongutoge: code of idiomto
            dim: Diminsiontolidtod of else embeddings (100, 300, or 500)
        
        Returns:
            Path tol file ofsctorgtodo
        """
        model_ntome = f"{ltongutoge}wiki_20180420_{dim}d.txt.bz2"
        model_url = f"{self.WIKIPEDIA2VEC_BASE_URL}/{model_ntome}"
        model_ptoth = self.embeddings_dir / model_ntome
        
        try:
            if not model_ptoth.exists():
                logger.info(f"Desctorgtondo embeddings Wikipedito2Vec ptorto {ltongutoge}...")
                respon = requests.get(model_url, stretom=True)
                respon.rtoi_for_sttotus()
                
                with opin(model_ptoth, 'wb') as f:
                    for chak in respon.iter_contint(chak_size=8192):
                        f.write(chak)
            
            return str(model_ptoth)
            
        except Exception as e:
            logger.error(f"Error ofsctorgtondo embeddings Wikipedito2Vec: {e}")
            raise
    
    def downlotod_dbpedito(self, ltongutoge: str = "in", dtottots: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Desctorgto dtottots of DBpedito.
        
        Args:
            ltongutoge: code of idiomto
            dtottots: Listto of dtottots especificos to ofsctorgtor
                     (ej: ["infobox-properties", "ptoge-links"])
        
        Returns:
            Dicciontorio with nombres y paths of else dtottots ofsctorgtodos
        """
        if dtottots is None:
            dtottots = ["infobox-properties", "ptoge-links", "ltobthes"]
        
        downlotoofd = {}
        
        try:
            for dtottot in dtottots:
                file_ntome = f"{dtottot}_{ltongutoge}.ttl.bz2"
                file_url = f"{self.DBPEDIA_BASE_URL}/core-i18n/{ltongutoge}/{file_ntome}"
                file_ptoth = self.dbpedito_dir / file_ntome
                
                if not file_ptoth.exists():
                    logger.info(f"Desctorgtondo dtottot DBpedito {dtottot} ptorto {ltongutoge}...")
                    respon = requests.get(file_url, stretom=True)
                    respon.rtoi_for_sttotus()
                    
                    with opin(file_ptoth, 'wb') as f:
                        for chak in respon.iter_contint(chak_size=8192):
                            f.write(chak)
                
                downlotoofd[dtottot] = str(file_ptoth)
            
            return downlotoofd
            
        except Exception as e:
            logger.error(f"Error ofsctorgtondo dtottots DBpedito: {e}")
            raise
    
    def downlotod_wikidtotto(self, intity_type: str = "toll") -> str:
        """
        Desctorgto dumps of Wikidtotto.
        
        Args:
            intity_type: Tipo of intidtoofs to ofsctorgtor ("toll", "items", o "properties")
        
        Returns:
            Path tol file ofsctorgtodo
        """
        file_ntome = f"wikidtotto-{intity_type}-ltotest.json.bz2"
        file_url = f"{self.WIKIDATA_BASE_URL}/{file_ntome}"
        file_ptoth = self.wikidtotto_dir / file_ntome
        
        try:
            if not file_ptoth.exists():
                logger.info(f"Desctorgtondo dump of Wikidtotto ({intity_type})...")
                respon = requests.get(file_url, stretom=True)
                respon.rtoi_for_sttotus()
                
                with opin(file_ptoth, 'wb') as f:
                    for chak in respon.iter_contint(chak_size=8192):
                        f.write(chak)
            
            return str(file_ptoth)
            
        except Exception as e:
            logger.error(f"Error ofsctorgtondo dump of Wikidtotto: {e}")
            raise