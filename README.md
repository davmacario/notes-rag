# notes-rag
RAG implementation for my markdown notes

## Git refresh

The service checks the configured branch for new commits every 60 seconds and rebuilds when it detects a change. Set `GIT_POLL_INTERVAL` to a positive number of seconds to change the interval. The first successful check also starts a rebuild; scheduled rebuilds continue independently. Remote checks require Git credentials to be available to the service.
