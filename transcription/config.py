"""
Configuration settings for the sermon transcription pipeline.
"""
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sermon_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try to load from .env file if it exists, otherwise use system environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("Loaded configuration from .env file")
except ImportError:
    logger.info("python-dotenv not installed, using system environment variables")

# API Keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")
else:
    logger.info("Found OpenAI API key in environment variables")

# Proof of Concept Mode
POC_MODE = os.environ.get("POC_MODE", "true").lower() == "true"
POC_LIMIT = int(os.environ.get("POC_LIMIT", "5"))

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Canonical directory for all sermon data
DATA_DIR = os.path.join(BASE_DIR, "data")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
TRANSCRIPT_DIR = os.path.join(DATA_DIR, "transcripts")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")
VIDEO_LIST_PATH = os.path.join(DATA_DIR, "video_list.csv")

# Ensure directories exist
for directory in [DATA_DIR, AUDIO_DIR, TRANSCRIPT_DIR, METADATA_DIR]:
    os.makedirs(directory, exist_ok=True)

# YouTube Download Settings
YTDLP_FORMAT = "bestaudio/best"
YTDLP_POSTPROCESSORS = [{
    'key': 'FFmpegExtractAudio',
    'preferredcodec': 'mp3',
    'preferredquality': '128',
}]

# OpenAI Whisper Settings
WHISPER_MODEL = "whisper-1"  # OpenAI's Whisper model
WHISPER_LANGUAGE = "en"      # Language code (English)

# Log configuration details
logger.info(f"POC Mode: {POC_MODE}, Limit: {POC_LIMIT}")
logger.info(f"Data directory: {DATA_DIR}")
logger.info(f"Using Whisper model: {WHISPER_MODEL}")