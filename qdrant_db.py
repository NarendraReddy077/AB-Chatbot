from qdrant_client import QdrantClient
from config import settings

qdrant_client = QdrantClient(
    url= settings.qdrant_url, 
    api_key= settings.qdrant_api_key,
)

print(qdrant_client.get_collections())
# print(settings.qdrant_url)