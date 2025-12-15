import os

# Prefer new adapter, fallback to legacy
try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma

# === Config ===
CHROMA_DIR = os.getenv("CHROMA_DIR", "./.chroma")
COLLECTION = os.getenv("CHROMA_COLLECTION", "default")

# === Load the Chroma collection ===
print(f"[INFO] Connecting to Chroma collection='{COLLECTION}' at: {CHROMA_DIR}")
vs = Chroma(collection_name=COLLECTION, persist_directory=CHROMA_DIR)

# === Access the underlying collection ===
try:
    coll = vs._collection  # underlying Chroma client
    print(f"[INFO] Collection name: {coll.name}")
    print(f"[INFO] Total items in collection: {coll.count()}")
except Exception as e:
    print(f"[ERROR] Could not get count: {e}")

print("[DONE]")