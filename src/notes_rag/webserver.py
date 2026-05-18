from typing import List

from llama_index.core.schema import TextNode
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from notes_rag.storage import Storage


class QueryBody(BaseModel):
    query: str
    num_docs: int


class MCPServer:
    def __init__(self, storage: Storage, host: str = "0.0.0.0", port: int = 9099):
        self._storage = storage
        self.mcp = FastMCP("NotesRAG", stateless_http=True, json_response=True, host=host, port=port)

        self._register_tools()

    def _register_tools(self):
        @self.mcp.tool()
        async def query_notes(query_body: QueryBody):
            """
            Query the user's notes knowledge base for additional context.
            """
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
        await self.mcp.run_streamable_http_async()
