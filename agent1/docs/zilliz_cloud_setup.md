# Zilliz Cloud Setup

Zilliz Cloud can host both the research-document collection and the user-memory collection.

```powershell
$env:ZILLIZ_CLOUD_URI="https://your-endpoint"
$env:ZILLIZ_CLOUD_TOKEN="your-token"
python agent_main.py check-milvus
python agent_main.py rebuild-index
```

The application uses separate collection names for research evidence and user memory. Ensure the configured embedding model matches the stored vector dimension.
