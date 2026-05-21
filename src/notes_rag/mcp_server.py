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
            """Query the user's own notes for further context.

            Allows to tap in the knowledge base of the user to avoid making assumptions when answering the user's
            prompts.
            Use this tool when the user explicitly requests you look at the notes.

            The query body (tool argument) consists of:
            - query: string containing the summary of the information you are looking for. Think of it like a Google
              search, so try to be concise and use mostly keywords.
            - num_docs: number of documents you would like to retrieve. The higher the value, the more context you will
              get, but keep in mind that too many documents may result in non-relevant information.
              Typical values are:
              - 5-10: initial search, this will be the most relevant information
              - 10-20: if more information is needed, but not required in most cases
              - 20+: used very rarely
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
