from fastapi import FastAPI
from pydantic import BaseModel

from notes_rag.storage import Storage

class QueryBody(BaseModel):
    query: str
    num_docs: int

class HTTPServer:

    def __init__(self, storage: Storage):

        self._storage = storage
        self.app = FastAPI()

        self._register_routes()

    def _register_routes(self):

        @self.app.post('/query_rag')
        async def query_rag(query_body: QueryBody):
            docs = self._storage.search(query_body.query, query_body.num_docs)
            return {
                "documents": docs
            }
