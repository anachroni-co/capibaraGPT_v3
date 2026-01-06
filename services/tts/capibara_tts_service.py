"""
Capibara TTS module that integrates FastSpeech and HiFi-GAN for text-to-speech,
with a fallback using pyttsx3. Includes a WebSocket server for streaming audio.
"""

import asyncio
import ssl
import json
import os
import logging
from typing import Optional, Set, Dict, Any

import numpy as np  # type: ignore
import websockets  # type: ignore
import onnxruntime as ort  # type: ignore
import pyttsx3  # type: ignore
from dotenv import load_dotenv  # type: ignore

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Retrieve configurations from .env
FASTSPEECH_PATH = os.getenv("FASTSPEECH_MODEL_PATH")
HIFIGAN_PATH = os.getenv("HIFIGAN_MODEL_PATH")
SAMPLE_RATE = int(os.getenv("CAPIBARA_TTS_SAMPLE_RATE", "22050"))
HOST = os.getenv("CAPIBARA_TTS_HOST", "localhost")
PORT = int(os.getenv("CAPIBARA_TTS_PORT", "8765"))
CERT_FILE = os.getenv("CAPIBARA_TTS_CERT_FILE")
KEY_FILE = os.getenv("CAPIBARA_TTS_KEY_FILE")

# Set up SSL context if cert and key files are available
SSL_CONTEXT = None
if CERT_FILE and KEY_FILE:
    SSL_CONTEXT = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    SSL_CONTEXT.load_cert_chain(certfile=CERT_FILE, keyfile=KEY_FILE)

# Track connected clients
connected_clients: Set[Any] = set()


class CapibaraTextToSpeech:
    """
    Main class for Capibara text-to-speech functionality.

    Provides:
      - FastSpeech-based spectrogram generation
      - HiFi-GAN-based audio wave synthesis
      - Fallback TTS with pyttsx3
      - A WebSocket server to serve audio to clients
    """

    def __init__(
        self,
        fastspeech_model_path: Optional[str] = FASTSPEECH_PATH,
        hifigan_model_path: Optional[str] = HIFIGAN_PATH,
        sample_rate: int = SAMPLE_RATE
    ):
        """
        Initialize TTS with paths to ONNX models and sample rate.

        Args:
            fastspeech_model_path (Optional[str]): Path to the FastSpeech ONNX model.
            hifigan_model_path (Optional[str]): Path to the HiFi-GAN ONNX model.
            sample_rate (int): Audio sample rate for generated waveforms.
        """
        self.sample_rate = sample_rate

        # Validate paths
        if not fastspeech_model_path or not hifigan_model_path:
            raise ValueError("FastSpeech and HiFi-GAN model paths must be provided.")

        # Load ONNX models
        try:
            self.fastspeech_session = ort.InferenceSession(fastspeech_model_path)
            self.hifigan_session = ort.InferenceSession(hifigan_model_path)
        except Exception as e:
            raise RuntimeError(f"Error loading ONNX models: {e}")

    def text_to_spectrogram(self, text: str) -> np.ndarray:
        """
        Convert text to a spectrogram using FastSpeech ONNX.

        Args:
            text (str): The text to convert.

        Returns:
            np.ndarray: A spectrogram array with shape (1, T, features), clipped between [-4.0, 4.0].
        """
        input_text = np.array([text], dtype=object)
        inputs = {self.fastspeech_session.get_inputs()[0].name: input_text}
        spectrogram = self.fastspeech_session.run(None, inputs)[0]
        return np.clip(spectrogram, -4.0, 4.0)

    def spectrogram_to_audio(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        Convert a spectrogram into audio using HiFi-GAN ONNX.

        Args:
            spectrogram (np.ndarray): The spectrogram array.

        Returns:
            np.ndarray: The generated waveform array.
        """
        inputs = {self.hifigan_session.get_inputs()[0].name: spectrogram}
        return self.hifigan_session.run(None, inputs)[0]

    async def handle_connection(self, websocket, path):
        """
        Asynchronous handler for each connected WebSocket client.

        Args:
            websocket: The WebSocket connection.
            path: The path for the connection (unused here).
        """
        connected_clients.add(websocket)
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    user_text = data.get("text")
                    if not user_text:
                        await websocket.send(json.dumps({"error": "No valid text provided."}))
                        continue

                    # Convert text to spectrogram and then to audio
                    spectrogram = self.text_to_spectrogram(user_text)
                    audio = self.spectrogram_to_audio(spectrogram)

                    # Send audio data as JSON
                    await websocket.send(json.dumps({
                        "audio": audio.tolist(),
                        "sample_rate": self.sample_rate
                    }))
                except Exception as e:
                    await websocket.send(json.dumps({"error": str(e)}))
        except websockets.ConnectionClosed:
            logger.info("WebSocket client disconnected.")
        finally:
            connected_clients.remove(websocket)

    def start_websocket_server(self, host: str = HOST, port: int = PORT, ssl_context: Optional[ssl.SSLContext] = SSL_CONTEXT):
        """
        Start an asynchronous WebSocket server to deliver synthesized audio.

        Args:
            host (str): Host IP or hostname.
            port (int): Port for the server.
            ssl_context: Optional SSL context for secure connections.
        """
        start_server = websockets.serve(self.handle_connection, host, port, ssl=ssl_context)
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(start_server)
            logger.info(f"TTS WebSocket server started at ws://{host}:{port}")
            loop.run_forever()
        except KeyboardInterrupt:
            logger.info("Shutting down TTS WebSocket server.")
        finally:
            loop.stop()

    def synthesize(self, text: str) -> bytes:
        """
        Generate audio from text using pyttsx3 as a fallback synthesizer.

        Args:
            text (str): The text to convert.

        Returns:
            bytes: The resulting audio data as a byte string.
        """
        synthesizer = pyttsx3.init()
        synthesizer.setProperty('rate', 150)
        synthesizer.setProperty('volume', 1.0)
        audio_file = "output_audio.wav"
        try:
            synthesizer.save_to_file(text, audio_file)
            synthesizer.runAndWait()
            with open(audio_file, 'rb') as f:
                return f.read()
        finally:
            if os.path.exists(audio_file):
                os.remove(audio_file)

    def generate_audio(self, text: str) -> bytes:
        """
        High-level method to generate audio bytes from text.

        Args:
            text (str): The text to convert to speech.

        Returns:
            bytes: Raw audio data (e.g., WAV contents).
        """
        return self.synthesize(text)


if __name__ == "__main__":
    try:
        tts = CapibaraTextToSpeech()
        tts.start_websocket_server()
    except Exception as e:
        logger.error(f"Error starting TTS server: {e}")
        print(f"Error starting TTS server: {e}")
