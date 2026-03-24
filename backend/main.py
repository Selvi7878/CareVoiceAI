"""
CareVoice AI — Elder care voice companion built on Microsoft Agent Framework.

Start with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

from api import app  # noqa: E402

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
