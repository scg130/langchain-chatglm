import uvicorn

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app", 
        host="0.0.0.0", 
        port=8800, 
        reload=True
    )
