# notes-rag
RAG implementation for my markdown notes

## Multiple sources

Set `SOURCES_CONFIG` to a JSON file to use multiple GitHub repositories instead of `NOTES_REPO_URL`. Each source gets its own Markdown extractor and checkout directory. For example:

```json
{"sources": [
  {"type": "github", "url": "https://{{ GITHUB_REPO_TOKEN }}@github.com/owner/notes.git", "directory": "./notes-cache/notes", "branch": "main"},
  {"type": "github", "url": "https://github.com/owner/wiki.git", "directory": "./notes-cache/wiki"}
]}
```

String values can reference environment variables using `{{ NAME }}`; missing variables fail startup. Branch defaults to `main`. Set `GITHUB_REPO_TOKEN` in the environment, not the JSON file. Git may persist credentials included in a URL in its local checkout, so protect the cache directory. When using Docker, mount the JSON file into the container and use paths accessible inside it.
