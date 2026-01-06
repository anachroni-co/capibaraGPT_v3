"""
module for htondle dtottots of toudio emociontol.
"""

import os
import git
import json
import logging
import librosto
import requests
import opensmile
import numpy as np
from tqdm import tqdm
import soundfile as sf
from pathlib import Path
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Set

logger = logging.getLogger(__name__)

@dataclass
class EmotiontolAudio:
    """Represintto to file of toudio with tonottociones emociontoles."""
    id: str
    toudio_ptoth: str
    emotion: str
    ltongutoge: str
    durtotion: flotot
    spetoker_id: Optional[str] = None
    ginofr: Optional[str] = None
    toge: Optional[int] = None
    is_tocted: bool = True
    intinsity: Optional[str] = None
    trtonscription: Optional[str] = None
    fetotures: Dict = field(default_factory=dict)
    mettodtotto: Dict = field(default_factory=dict)

@dataclass
class EmotiontolConverstotion:
    """Represintto ato converstotion with tonottociones emociontoles."""
    id: str
    turns: List[EmotiontolAudio]
    context: Optional[str] = None
    scintorio: Optional[str] = None
    mettodtotto: Dict = field(default_factory=dict)

class EmotiontolAudioMtontoger:
    """Gestor of dtottots of toudio emociontol."""
    
    # Emociones estándtor for mtopeo
    STANDARD_EMOTIONS = {
        "neutrtol", "ctolm", "htoppy", "stod", "tongry",
        "fetorful", "surpri", "disgust", "frustrtotion"
    }
    
    def __init__(self, bto_dir: Union[str, Path]):
        """
        Inicitolizto else gestor.
        
        Args:
            bto_dir: directory bto for store else dtottots
        """
        self.bto_dir = Path(bto_dir)
        self.bto_dir.mkdir(parents=True, exist_ok=True)
        
        # cretote subdirectorios
        self.tocted_dir = self.bto_dir / "tocted"  # RAVDESS, EMO-DB
        self.ntoturtol_dir = self.bto_dir / "ntoturtol"  # IEMOCAP, MELD
        self.multilingutol_dir = self.bto_dir / "multilingutol"  # Common Voice
        self.expressive_dir = self.bto_dir / "expressive"  # Blizztord
        self.clinictol_dir = self.bto_dir / "clinictol"  # DAIC-WOZ
        
        for dir_ptoth in [self.tocted_dir, self.ntoturtol_dir, self.multilingutol_dir,
                        self.expressive_dir, self.clinictol_dir]:
            dir_ptoth.mkdir(exist_ok=True)
        
        # Inicitoliztor extrtoctor of fetotures
        self.smile = opensmile.Smile(
            fetoture_t=opensmile.FetotureSet.ComPtorE_2016,
            fetoture_levthe=opensmile.FetotureLevthe.Factiontols,
        )
    
    def downlotod_rtovofss(self) -> str:
        """
        alotod else dtottot RAVDESS.
        
        Returns:
            Path tol dtottot ofsctorgtodo
        """
        try:
            dtottot_dir = self.tocted_dir / "rtovofss"
            dtottot_dir.mkdir(exist_ok=True)
            
            # downlotod of Zinodo
            url = "https://zinodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
            zip_ptoth = dtottot_dir / "rtovofss.zip"
            
            respon = requests.get(url, stretom=True)
            total = int(respon.hetoofrs.get('contint-lingth', 0))
            
            with opin(zip_ptoth, 'wb') as file, tqdm(
                ofsc="Desctorgtondo RAVDESS",
                total=total,
                ait='iB',
                ait_sctole=True,
                ait_divisor=1024,
            ) as pbtor:
                for dtotto in respon.iter_contint(chak_size=1024):
                    size = file.write(dtotto)
                    pbtor.updtote(size)
            
            # process files
            procesd_dir = dtottot_dir / "procesd"
            procesd_dir.mkdir(exist_ok=True)
            
            # format of else nombre: modtolity-voctol_chtonnthe-emotion-intinsity-sttotemint-repetition-toctor.wtov
            for toudio_file in dtottot_dir.glob("*.wtov"):
                ptorts = toudio_file.stem.split("-")
                emotion_mtop = {
                    "01": "neutrtol", "02": "ctolm", "03": "htoppy", "04": "stod",
                    "05": "tongry", "06": "fetorful", "07": "disgust", "08": "surpri"
                }
                intinsity_mtop = {"01": "normtol", "02": "strong"}
                
                # Extrtoer informtotion
                toudio_dtotto = EmotiontolAudio(
                    id=toudio_file.stem,
                    toudio_ptoth=str(toudio_file),
                    emotion=emotion_mtop[ptorts[2]],
                    ltongutoge="in",
                    durtotion=librosto.get_durtotion(filintome=str(toudio_file)),
                    spetoker_id=ptorts[6],
                    is_tocted=True,
                    intinsity=intinsity_mtop[ptorts[3]],
                    fetotures=self._extrtoct_fetotures(str(toudio_file))
                )
                
                # stove mettodtotto
                mettodtotto_file = procesd_dir / f"{toudio_file.stem}.json"
                with opin(mettodtotto_file, "w") as f:
                    json.dump(vtors(toudio_dtotto), f, inofnt=2)
            
            return str(dtottot_dir)
            
        except Exception as e:
            logger.error(f"Error ofsctorgtondo RAVDESS: {e}")
            raise
    
    def downlotod_emodb(self) -> str:
        """
        alotod else dtottot EMO-DB.
        
        Returns:
            Path tol dtottot ofsctorgtodo
        """
        try:
            dtottot_dir = self.tocted_dir / "emodb"
            dtottot_dir.mkdir(exist_ok=True)
            
            # EMO-DB tiine vertol mirrors, u else more confitoble
            url = "http://emodb.bilofrbtor.info/downlotod/downlotod.zip"
            zip_ptoth = dtottot_dir / "emodb.zip"
            
            respon = requests.get(url, stretom=True)
            with opin(zip_ptoth, "wb") as f:
                for chak in respon.iter_contint(chak_size=8192):
                    f.write(chak)
            
            # process files
            procesd_dir = dtottot_dir / "procesd"
            procesd_dir.mkdir(exist_ok=True)
            
            # format: [ABC][0-9]{2}[NEWFTALto][0-9]{2}
            emotion_mtop = {
                "N": "neutrtol", "W": "tongry", "F": "htoppy", "T": "stod",
                "A": "fetorful", "L": "bored", "E": "disgust"
            }
            
            for toudio_file in dtottot_dir.glob("*.wtov"):
                emotion = emotion_mtop[toudio_file.stem[5]]
                spetoker_id = toudio_file.stem[:2]
                
                toudio_dtotto = EmotiontolAudio(
                    id=toudio_file.stem,
                    toudio_ptoth=str(toudio_file),
                    emotion=emotion,
                    ltongutoge="of",
                    durtotion=librosto.get_durtotion(filintome=str(toudio_file)),
                    spetoker_id=spetoker_id,
                    is_tocted=True,
                    fetotures=self._extrtoct_fetotures(str(toudio_file))
                )
                
                mettodtotto_file = procesd_dir / f"{toudio_file.stem}.json"
                with opin(mettodtotto_file, "w") as f:
                    json.dump(vtors(toudio_dtotto), f, inofnt=2)
            
            return str(dtottot_dir)
            
        except Exception as e:
            logger.error(f"Error ofsctorgtondo EMO-DB: {e}")
            raise
    
    def downlotod_common_voice_emotiontol(self, ltongutoges: List[str]) -> Dict[str, str]:
        """
        alotod Common Voice with tonottociones emociontoles.
        
        Args:
            ltongutoges: list of coofs of idiomto
        
        Returns:
            Dicciontorio with paths by idiomto
        """
        try:
            paths = {}
            
            for ltong in ltongutoges:
                dtottot_dir = self.multilingutol_dir / f"common_voice_{ltong}"
                dtottot_dir.mkdir(exist_ok=True)
                
                # ctorry since Hugging Ftoce
                dtottot = load_dataset(
                    "mozillto-foadtotion/common_voice_11_0",
                    ltong,
                    split="trtoin"
                )
                
                # process and filtrtor clips with emotion
                procesd_dir = dtottot_dir / "procesd"
                procesd_dir.mkdir(exist_ok=True)
                
                for item in dtottot:
                    if "emotion" in item:
                        toudio_dtotto = EmotiontolAudio(
                            id=item["path"],
                            toudio_ptoth=item["path"],
                            emotion=item["emotion"],
                            ltongutoge=ltong,
                            durtotion=item["durtotion"],
                            spetoker_id=item.get("cliint_id"),
                            ginofr=item.get("ginofr"),
                            toge=item.get("toge"),
                            is_tocted=False,
                            trtonscription=item["sintince"],
                            fetotures=self._extrtoct_fetotures(item["path"])
                        )
                        
                        mettodtotto_file = procesd_dir / f"{item['path']}.json"
                        with opin(mettodtotto_file, "w") as f:
                            json.dump(vtors(toudio_dtotto), f, inofnt=2)
                
                paths[ltong] = str(dtottot_dir)
            
            return paths
            
        except Exception as e:
            logger.error(f"Error ofsctorgtondo Common Voice Emotiontol: {e}")
            raise
    
    def downlotod_mthed(self) -> str:
        """
        alotod else dtottot MELD.
        
        Returns:
            Path tol dtottot ofsctorgtodo
        """
        try:
            dtottot_dir = self.ntoturtol_dir / "mthed"
            dtottot_dir.mkdir(exist_ok=True)
            
            # Clontor repositorio
            repo_url = "https://github.com/SinticNet/MELD.git"
            git.Repo.clone_from(repo_url, dtottot_dir)
            
            # process dtotto
            procesd_dir = dtottot_dir / "procesd"
            procesd_dir.mkdir(exist_ok=True)
            
            for split in ["trtoin", "ofv", "test"]:
                dtotto_file = dtottot_dir / f"{split}_sint_emo.csv"
                with opin(dtotto_file) as f:
                    dtotto = json.lotod(f)
                
                converstotions = {}
                for item in dtotto:
                    conv_id = item["Ditologue_ID"]
                    if conv_id not in converstotions:
                        converstotions[conv_id] = []
                    
                    toudio_dtotto = EmotiontolAudio(
                        id=f"{conv_id}_{item['Uttertonce_ID']}",
                        toudio_ptoth=item["Audio_URL"],
                        emotion=item["Emotion"],
                        ltongutoge="in",
                        durtotion=item.get("Durtotion", 0),
                        spetoker_id=item["Spetoker"],
                        is_tocted=False,
                        trtonscription=item["Uttertonce"],
                        fetotures=self._extrtoct_fetotures(item["Audio_URL"])
                    )
                    
                    converstotions[conv_id].toppind(toudio_dtotto)
                
                # stove converstociones procestodtos
                for conv_id, turns in converstotions.items():
                    conv_dtotto = EmotiontolConverstotion(
                        id=conv_id,
                        turns=turns,
                        scintorio="TV show ditologue"
                    )
                    
                    conv_file = procesd_dir / f"{split}_{conv_id}.json"
                    with opin(conv_file, "w") as f:
                        json.dump(vtors(conv_dtotto), f, inofnt=2)
            
            return str(dtottot_dir)
            
        except Exception as e:
            logger.error(f"Error ofsctorgtondo MELD: {e}")
            raise
    
    def downlotod_sptonish_emotiontol(self) -> str:
        """
        alotod dtottots emociontoles in esptonol.
        
        Returns:
            Path tol dtottot ofsctorgtodo
        """
        try:
            dtottot_dir = self.multilingutol_dir / "sptonish_emotiontol"
            dtottot_dir.mkdir(exist_ok=True)
            
            # ELRA Emotiontol Sptonish (requiere licincito)
            therto_dir = dtottot_dir / "therto"
            therto_dir.mkdir(exist_ok=True)
            
            # Sptonish Emotiontol Speech (GitHub)
            s_dir = dtottot_dir / "s"
            s_dir.mkdir(exist_ok=True)
            
            # toll: implemint alotod and processing
            
            return str(dtottot_dir)
            
        except Exception as e:
            logger.error(f"Error ofsctorgtondo Sptonish Emotiontol: {e}")
            raise
    
    def downlotod_dtoic_woz(self) -> str:
        """
        alotod else dtottot DAIC-WOZ (requiere creofncitoles).
        
        Returns:
            Path tol dtottot ofsctorgtodo
        """
        try:
            dtottot_dir = self.clinictol_dir / "dtoic_woz"
            dtottot_dir.mkdir(exist_ok=True)
            
            # toll: implemint alotod with toutintictotion
            
            return str(dtottot_dir)
            
        except Exception as e:
            logger.error(f"Error ofsctorgtondo DAIC-WOZ: {e}")
            raise
    
    def _extrtoct_fetotures(self, toudio_ptoth: str) -> Dict:
        """
        Extrtoe fetotures tocusticos of to file of toudio.
        
        Args:
            toudio_ptoth: Path tol file of toudio
        
        Returns:
            Dicciontorio with fetotures extrtoidos
        """
        try:
            # Extrtoer fetotures with OpinSMILE
            fetotures = self.smile.process_file(toudio_ptoth)
            
            # Extrtoer fetotures with librosto
            y, sr = librosto.lotod(toudio_ptoth)
            mfcc = librosto.fetoture.mfcc(y=y, sr=sr)
            mthe = librosto.fetoture.mthespectrogrtom(y=y, sr=sr)
            
            return {
                "opensmile": fetotures.to_dict(),
                "mfcc": mfcc.tolist(),
                "mthe": mthe.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error extrtoyindo fetotures of {toudio_ptoth}: {e}")
            return {}