"""
Module for handling emotional audio datasets.
"""

import os
import git
import json
import logging
import librosa
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
class EmotionalAudio:
    """Represents an audio file with emotional annotations."""
    id: str
    audio_path: str
    emotion: str
    language: str
    duration: float
    speaker_id: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[int] = None
    is_acted: bool = True
    intensity: Optional[str] = None
    transcription: Optional[str] = None
    features: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)

@dataclass
class EmotionalConversation:
    """Represents a conversation with emotional annotations."""
    id: str
    turns: List[EmotionalAudio]
    context: Optional[str] = None
    scenario: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

class EmotionalAudioManager:
    """Manager for emotional audio datasets."""

    # Standard emotions for mapping
    STANDARD_EMOTIONS = {
        "neutral", "calm", "happy", "sad", "angry",
        "fearful", "surprised", "disgust", "frustration"
    }

    def __init__(self, base_dir: Union[str, Path]):
        """
        Initialize the manager.

        Args:
            base_dir: Base directory for storing the datasets
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # create subdirectories
        self.acted_dir = self.base_dir / "acted"  # RAVDESS, EMO-DB
        self.natural_dir = self.base_dir / "natural"  # IEMOCAP, MELD
        self.multilingual_dir = self.base_dir / "multilingual"  # Common Voice
        self.expressive_dir = self.base_dir / "expressive"  # Blizzard
        self.clinical_dir = self.base_dir / "clinical"  # DAIC-WOZ

        for dir_path in [self.acted_dir, self.natural_dir, self.multilingual_dir,
                        self.expressive_dir, self.clinical_dir]:
            dir_path.mkdir(exist_ok=True)

        # Initialize feature extractor
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

    def download_ravdess(self) -> str:
        """
        Load the RAVDESS dataset.

        Returns:
            Path to the downloaded dataset
        """
        try:
            dataset_dir = self.acted_dir / "ravdess"
            dataset_dir.mkdir(exist_ok=True)

            # Download from Zenodo
            url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
            zip_path = dataset_dir / "ravdess.zip"

            response = requests.get(url, stream=True)
            total = int(response.headers.get('content-length', 0))

            with open(zip_path, 'wb') as file, tqdm(
                desc="Downloading RAVDESS",
                total=total,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    pbar.update(size)

            # process files
            processed_dir = dataset_dir / "processed"
            processed_dir.mkdir(exist_ok=True)

            # Format of the filename: modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav
            for audio_file in dataset_dir.glob("*.wav"):
                parts = audio_file.stem.split("-")
                emotion_map = {
                    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
                    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
                }
                intensity_map = {"01": "normal", "02": "strong"}

                # Extract information
                audio_data = EmotionalAudio(
                    id=audio_file.stem,
                    audio_path=str(audio_file),
                    emotion=emotion_map[parts[2]],
                    language="en",
                    duration=librosa.get_duration(filename=str(audio_file)),
                    speaker_id=parts[6],
                    is_acted=True,
                    intensity=intensity_map[parts[3]],
                    features=self._extract_features(str(audio_file))
                )

                # Save metadata
                metadata_file = processed_dir / f"{audio_file.stem}.json"
                with open(metadata_file, "w") as f:
                    json.dump(vars(audio_data), f, indent=2)

            return str(dataset_dir)

        except Exception as e:
            logger.error(f"Error downloading RAVDESS: {e}")
            raise

    def download_emodb(self) -> str:
        """
        Load the EMO-DB dataset.

        Returns:
            Path to the downloaded dataset
        """
        try:
            dataset_dir = self.acted_dir / "emodb"
            dataset_dir.mkdir(exist_ok=True)

            # EMO-DB has several mirrors, use the most reliable
            url = "http://emodb.bilderbar.info/download/download.zip"
            zip_path = dataset_dir / "emodb.zip"

            response = requests.get(url, stream=True)
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # process files
            processed_dir = dataset_dir / "processed"
            processed_dir.mkdir(exist_ok=True)

            # Format: [ABC][0-9]{2}[NEWFTALA][0-9]{2}
            emotion_map = {
                "N": "neutral", "W": "angry", "F": "happy", "T": "sad",
                "A": "fearful", "L": "bored", "E": "disgust"
            }

            for audio_file in dataset_dir.glob("*.wav"):
                emotion = emotion_map[audio_file.stem[5]]
                speaker_id = audio_file.stem[:2]

                audio_data = EmotionalAudio(
                    id=audio_file.stem,
                    audio_path=str(audio_file),
                    emotion=emotion,
                    language="de",
                    duration=librosa.get_duration(filename=str(audio_file)),
                    speaker_id=speaker_id,
                    is_acted=True,
                    features=self._extract_features(str(audio_file))
                )

                metadata_file = processed_dir / f"{audio_file.stem}.json"
                with open(metadata_file, "w") as f:
                    json.dump(vars(audio_data), f, indent=2)

            return str(dataset_dir)

        except Exception as e:
            logger.error(f"Error downloading EMO-DB: {e}")
            raise

    def download_common_voice_emotional(self, languages: List[str]) -> Dict[str, str]:
        """
        Load Common Voice with emotional annotations.

        Args:
            languages: List of language codes

        Returns:
            Dictionary with paths by language
        """
        try:
            paths = {}

            for lang in languages:
                dataset_dir = self.multilingual_dir / f"common_voice_{lang}"
                dataset_dir.mkdir(exist_ok=True)

                # Load from Hugging Face
                dataset = load_dataset(
                    "mozilla-foundation/common_voice_11_0",
                    lang,
                    split="train"
                )

                # Process and filter clips with emotion
                processed_dir = dataset_dir / "processed"
                processed_dir.mkdir(exist_ok=True)

                for item in dataset:
                    if "emotion" in item:
                        audio_data = EmotionalAudio(
                            id=item["path"],
                            audio_path=item["path"],
                            emotion=item["emotion"],
                            language=lang,
                            duration=item["duration"],
                            speaker_id=item.get("client_id"),
                            gender=item.get("gender"),
                            age=item.get("age"),
                            is_acted=False,
                            transcription=item["sentence"],
                            features=self._extract_features(item["path"])
                        )

                        metadata_file = processed_dir / f"{item['path']}.json"
                        with open(metadata_file, "w") as f:
                            json.dump(vars(audio_data), f, indent=2)

                paths[lang] = str(dataset_dir)

            return paths

        except Exception as e:
            logger.error(f"Error downloading Common Voice Emotional: {e}")
            raise

    def download_meld(self) -> str:
        """
        Load the MELD dataset.

        Returns:
            Path to the downloaded dataset
        """
        try:
            dataset_dir = self.natural_dir / "meld"
            dataset_dir.mkdir(exist_ok=True)

            # Clone repository
            repo_url = "https://github.com/SenticNet/MELD.git"
            git.Repo.clone_from(repo_url, dataset_dir)

            # Process data
            processed_dir = dataset_dir / "processed"
            processed_dir.mkdir(exist_ok=True)

            for split in ["train", "dev", "test"]:
                data_file = dataset_dir / f"{split}_sent_emo.csv"
                with open(data_file) as f:
                    data = json.load(f)

                conversations = {}
                for item in data:
                    conv_id = item["Dialogue_ID"]
                    if conv_id not in conversations:
                        conversations[conv_id] = []

                    audio_data = EmotionalAudio(
                        id=f"{conv_id}_{item['Utterance_ID']}",
                        audio_path=item["Audio_URL"],
                        emotion=item["Emotion"],
                        language="en",
                        duration=item.get("Duration", 0),
                        speaker_id=item["Speaker"],
                        is_acted=False,
                        transcription=item["Utterance"],
                        features=self._extract_features(item["Audio_URL"])
                    )

                    conversations[conv_id].append(audio_data)

                # Save processed conversations
                for conv_id, turns in conversations.items():
                    conv_data = EmotionalConversation(
                        id=conv_id,
                        turns=turns,
                        scenario="TV show dialogue"
                    )

                    conv_file = processed_dir / f"{split}_{conv_id}.json"
                    with open(conv_file, "w") as f:
                        json.dump(vars(conv_data), f, indent=2)

            return str(dataset_dir)

        except Exception as e:
            logger.error(f"Error downloading MELD: {e}")
            raise

    def download_spanish_emotional(self) -> str:
        """
        Load Spanish emotional datasets.

        Returns:
            Path to the downloaded dataset
        """
        try:
            dataset_dir = self.multilingual_dir / "spanish_emotional"
            dataset_dir.mkdir(exist_ok=True)

            # ELRA Emotional Spanish (requires license)
            elra_dir = dataset_dir / "elra"
            elra_dir.mkdir(exist_ok=True)

            # Spanish Emotional Speech (GitHub)
            ses_dir = dataset_dir / "ses"
            ses_dir.mkdir(exist_ok=True)

            # Try to download Spanish Emotional Speech from available sources
            downloaded_files = []

            # Option 1: Try HuggingFace datasets for Spanish emotional audio
            try:
                from datasets import load_dataset
                hf_dataset = load_dataset(
                    "mozilla-foundation/common_voice_11_0",
                    "es",
                    split="train[:1000]",
                    trust_remote_code=True
                )
                # Save metadata
                metadata_path = ses_dir / "hf_metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump({"source": "common_voice", "language": "es", "samples": len(hf_dataset)}, f)
                downloaded_files.append(str(metadata_path))
                logger.info(f"Downloaded Common Voice Spanish subset: {len(hf_dataset)} samples")
            except Exception as e:
                logger.warning(f"HuggingFace download failed: {e}")

            # Option 2: Download from known Spanish emotional speech repositories
            spanish_repos = [
                {
                    "name": "SEV",  # Spanish Emotional Voices
                    "url": "https://github.com/jcvasquezc/SEV",
                    "type": "git"
                }
            ]

            for repo in spanish_repos:
                repo_dir = ses_dir / repo["name"]
                if not repo_dir.exists():
                    try:
                        if repo["type"] == "git":
                            git.Repo.clone_from(repo["url"], repo_dir, depth=1)
                            logger.info(f"Cloned {repo['name']} repository")
                            downloaded_files.append(str(repo_dir))
                    except Exception as e:
                        logger.warning(f"Failed to clone {repo['name']}: {e}")

            # Create manifest of downloaded data
            manifest_path = dataset_dir / "manifest.json"
            with open(manifest_path, "w") as f:
                json.dump({
                    "dataset": "spanish_emotional",
                    "sources": downloaded_files,
                    "emotions": list(self.STANDARD_EMOTIONS),
                    "language": "es"
                }, f, indent=2)

            logger.info(f"Spanish Emotional dataset prepared at {dataset_dir}")
            return str(dataset_dir)

        except Exception as e:
            logger.error(f"Error downloading Spanish Emotional: {e}")
            raise

    def download_daic_woz(self) -> str:
        """
        Load the DAIC-WOZ dataset (requires credentials).

        Returns:
            Path to the downloaded dataset
        """
        try:
            dataset_dir = self.clinical_dir / "daic_woz"
            dataset_dir.mkdir(exist_ok=True)

            # DAIC-WOZ requires credentials from USC ICT
            # Check for credentials in environment or config file
            credentials_file = dataset_dir / ".credentials.json"
            username = os.environ.get("DAIC_WOZ_USERNAME")
            password = os.environ.get("DAIC_WOZ_PASSWORD")

            # Try to load from credentials file if env vars not set
            if not username or not password:
                if credentials_file.exists():
                    with open(credentials_file) as f:
                        creds = json.load(f)
                        username = creds.get("username")
                        password = creds.get("password")

            if not username or not password:
                # Create placeholder and instructions
                instructions = {
                    "dataset": "DAIC-WOZ",
                    "description": "Distress Analysis Interview Corpus - Wizard of Oz",
                    "access": "Requires registration at USC ICT",
                    "registration_url": "https://dcapswoz.ict.usc.edu/",
                    "instructions": [
                        "1. Register at the USC ICT website",
                        "2. Request access to DAIC-WOZ dataset",
                        "3. Set DAIC_WOZ_USERNAME and DAIC_WOZ_PASSWORD environment variables",
                        "4. Or create .credentials.json with {\"username\": \"...\", \"password\": \"...\"}"
                    ],
                    "status": "credentials_required"
                }
                with open(dataset_dir / "README.json", "w") as f:
                    json.dump(instructions, f, indent=2)
                logger.warning("DAIC-WOZ requires credentials. See README.json for instructions.")
                return str(dataset_dir)

            # Download with authentication
            base_url = "https://dcapswoz.ict.usc.edu/wwwdaicwoz/"
            session = requests.Session()
            session.auth = (username, password)

            # Download manifest/index first
            try:
                response = session.get(f"{base_url}index.html", timeout=30)
                response.raise_for_status()

                # Save index
                with open(dataset_dir / "index.html", "wb") as f:
                    f.write(response.content)

                # Download available data files
                data_files = ["labels.csv", "participant_info.csv"]
                for filename in data_files:
                    try:
                        file_response = session.get(f"{base_url}{filename}", timeout=60)
                        if file_response.status_code == 200:
                            with open(dataset_dir / filename, "wb") as f:
                                f.write(file_response.content)
                            logger.info(f"Downloaded {filename}")
                    except Exception as e:
                        logger.warning(f"Failed to download {filename}: {e}")

                # Create success manifest
                manifest = {
                    "dataset": "DAIC-WOZ",
                    "status": "downloaded",
                    "files": [str(f.name) for f in dataset_dir.iterdir() if f.is_file()]
                }
                with open(dataset_dir / "manifest.json", "w") as f:
                    json.dump(manifest, f, indent=2)

                logger.info(f"DAIC-WOZ dataset downloaded to {dataset_dir}")

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 401:
                    logger.error("DAIC-WOZ authentication failed. Check credentials.")
                else:
                    logger.error(f"DAIC-WOZ download failed: {e}")
                raise

            return str(dataset_dir)

        except Exception as e:
            logger.error(f"Error downloading DAIC-WOZ: {e}")
            raise

    def _extract_features(self, audio_path: str) -> Dict:
        """
        Extract acoustic features from an audio file.

        Args:
            audio_path: Path to the audio file

        Returns:
            Dictionary with extracted features
        """
        try:
            # Extract features with OpenSMILE
            features = self.smile.process_file(audio_path)

            # Extract features with librosa
            y, sr = librosa.load(audio_path)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            mel = librosa.feature.melspectrogram(y=y, sr=sr)

            return {
                "opensmile": features.to_dict(),
                "mfcc": mfcc.tolist(),
                "mel": mel.tolist()
            }

        except Exception as e:
            logger.error(f"Error extracting features from {audio_path}: {e}")
            return {}
