from typing import List

import uvicorn
from fastapi import FastAPI
from llama_index.core.schema import TextNode
from pydantic import BaseModel

from notes_rag.storage import Storage


class QueryBody(BaseModel):
    query: str
    num_docs: int


class HTTPServer:
    def __init__(self, storage: Storage, host: str = "0.0.0.0", port: int = 9099):

        self._storage = storage
        self.app = FastAPI()
        self.host = host
        self.port = port

        self._register_routes()

    def _register_routes(self):

        @self.app.post("/query_rag")
        async def query_rag(query_body: QueryBody):
            nodes = await self._storage.search(query_body.query, query_body.num_docs)
            additional_context = self._build_llm_context(nodes)
            return {"additional_context": additional_context}

    def _build_llm_context(self, nodes: List[TextNode]):
        """Builds text returnted to LLM including all retrieved Nodes"""
        out = ""
        for node in nodes:
            out += f"--- [source: {node.metadata['source_file']}] ---\n{node.text}\n"

        return out

    async def run(self):
        config = uvicorn.Config(self.app, host=self.host, port=self.port)
        server = uvicorn.Server(config)
        await server.serve()
