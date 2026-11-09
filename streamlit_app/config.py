import os

class Config:
    """
    Configuration class to manage file paths and settings for the application.
    """

    # --- YouTube Configuration ---
    # URL of the YouTube video to be processed.
    YOUTUBE_VIDEO_URL: str = "https://www.youtube.com/watch?v=uFhDGagZzjs"
    
    # Directory where downloaded YouTube audio files will be saved.
    YOUTUBE_AUDIO_SAVE_DIRECTORY: str = "docs/youtube/"

    # --- PDF Configuration ---
    # Directory where source PDF documents are located.
    PDF_SOURCE_DIRECTORY: str = "data"
    # Directory where ChromaDB embeddings will be persisted.
    CHROMA_PERSIST_DIRECTORY: str = "docs/chroma/"


    # ---Embedding Model Configuration ---
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # "intfloat/multilingual-e5-large" # "all-MiniLM-L6-v2"
    CHUNK_SIZE = 2028
    CHUNK_OVERLAP = 250

    def __init__(self):
        # Ensure the PDF source directory exists upon initialization.
        os.makedirs(self.PDF_SOURCE_DIRECTORY, exist_ok=True)
        print(f"Configuration loaded. PDF documents should be placed in '{self.PDF_SOURCE_DIRECTORY}'.")

# Create a global instance of the Config class for easy access.
# You can import `config` from this module in other files.
config = Config()