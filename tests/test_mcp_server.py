from unittest.mock import MagicMock

from starlette.testclient import TestClient

from notes_rag.mcp_server import MCPServer


def test_health_endpoint():
    storage = MagicMock()
    server = MCPServer(storage)

    with TestClient(server.mcp.streamable_http_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert storage.mock_calls == []
