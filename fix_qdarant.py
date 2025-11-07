"""Fix Qdrant collection dimension mismatch."""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from app.core.config import settings

def fix_collection():
    """Delete and recreate Qdrant collection with correct dimension."""

    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    collection_name = settings.qdrant_collection_name

    # Check if collection exists
    collections = client.get_collections().collections
    exists = any(col.name == collection_name for col in collections)

    if exists:
        print(f"Deleting existing collection: {collection_name}")
        client.delete_collection(collection_name)
        print("✓ Collection deleted")

    # Create new collection with correct dimension (384 for sentence-transformers)
    print(f"Creating collection with dimension 384...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=384,  # Correct dimension for all-MiniLM-L6-v2
            distance=Distance.COSINE
        )
    )
    print("✓ Collection created successfully!")

    # Verify
    collection_info = client.get_collection(collection_name)
    print(f"\nCollection info:")
    print(f"  Name: {collection_info.config.params.vectors.size}")
    print(f"  Dimension: {collection_info.config.params.vectors.size}")
    print(f"  Distance: {collection_info.config.params.vectors.distance}")

if __name__ == "__main__":
    try:
        fix_collection()
        print("\n✅ Qdrant collection fixed! You can now upload documents.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        exit(1)