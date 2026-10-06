"""Step 1: sanity-check that the project is set up correctly."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Reads key=value pairs from .env into environment variables.
load_dotenv()

key = os.getenv("ANTHROPIC_API_KEY")
if not key or key.startswith("sk-ant-your-key"):
    print("❌ ANTHROPIC_API_KEY is missing. Copy .env.example to .env and add your key.")
else:
    print(f"✅ ANTHROPIC_API_KEY found (starts with {key[:10]}...)")

docs = sorted(Path("docs").glob("*"))
print(f"✅ Found {len(docs)} files in ./docs:")
for doc in docs:
    print(f"   - {doc.name} ({doc.stat().st_size:,} bytes)")
