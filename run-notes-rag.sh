#!/bin/bash
# ==========================================================
# Ensure the notes-rag MCP server container is running.
#
# notes-rag (https://github.com/gmacario/notes-rag) indexes this repository's
# Markdown notes into a local ChromaDB vector store and exposes them to MCP
# clients - e.g. Claude Code - as a "query_notes" tool over streamable HTTP.
#
# This script is idempotent: it does nothing if the container is already up,
# starts it again if it exists but is stopped, and otherwise creates it. Run it
# before "claude mcp list" if notes-rag shows as "Failed to connect".
#
# Register the server once with:
#   claude mcp add --transport http notes-rag http://localhost:9099/mcp
#
# Usage: scripts/notes-rag.sh [--status] [--logs] [--stop] [--restart]
# ==========================================================

set -e

# All overridable so a locally built image or a second instance can be tested
# without editing this file.
IMAGE="${NOTES_RAG_IMAGE:-ghcr.io/davmacario/notes-rag:latest}"
CONTAINER="${NOTES_RAG_CONTAINER:-notes-rag}"
PORT="${NOTES_RAG_PORT:-9099}"
# Notes cache and vector store live outside the container so a restart does not
# re-download the embedding model or rebuild the index from scratch.
DATA_DIR="${NOTES_RAG_DATA_DIR:-$HOME/.cache/notes-rag}"
REBUILD_CRON="${NOTES_RAG_REBUILD_CRON:-0 */6 * * *}"
TIMEZONE="${TZ:-Europe/Rome}"
LOG_LEVEL="${NOTES_RAG_LOG_LEVEL:-info}"
# Seconds to wait for the HTTP endpoint to accept connections.
STARTUP_TIMEOUT="${NOTES_RAG_STARTUP_TIMEOUT:-120}"

# The notes indexed are this repository. NOTES_REPO_URL is a git remote as far
# as notes-rag is concerned, but git clones happily from a plain path, so the
# repo is bind-mounted read-only instead of needing a URL plus access token.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NOTES_REPO_URL="${NOTES_REPO_URL:-/notes-src}"
NOTES_BRANCH="${NOTES_BRANCH:-main}"

usage() {
	echo "Usage: $0 [--status] [--logs] [--stop] [--restart]"
	echo "Starts the $CONTAINER container (from $IMAGE) unless it is already running."
	echo
	echo "  --status    Report whether the container is running, then exit"
	echo "  --logs      Follow the container logs (Ctrl-C to detach)"
	echo "  --stop      Stop and remove the container"
	echo "  --restart   Recreate the container even if it is already running"
	echo "  -h, --help  Show this help"
	echo
	echo "Environment overrides: NOTES_RAG_IMAGE, NOTES_RAG_CONTAINER,"
	echo "NOTES_RAG_PORT, NOTES_RAG_DATA_DIR, NOTES_RAG_REBUILD_CRON,"
	echo "NOTES_RAG_LOG_LEVEL, NOTES_RAG_STARTUP_TIMEOUT, NOTES_REPO_URL,"
	echo "NOTES_BRANCH, TZ."
}

die() {
	echo "ERROR: $*" >&2
	exit 1
}

# Exact-match against the container name; "docker ps --filter name=" is a
# substring match, which would also match e.g. "notes-rag-old".
is_running() {
	[ -n "$(docker ps --quiet --filter "name=^/${CONTAINER}\$")" ]
}

exists() {
	[ -n "$(docker ps --all --quiet --filter "name=^/${CONTAINER}\$")" ]
}

# The MCP server starts listening before the first index rebuild finishes, so an
# open port means "reachable", not "ready to answer queries".
wait_for_port() {
	local waited=0
	while [ "$waited" -lt "$STARTUP_TIMEOUT" ]; do
		if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
			exec 3>&- 2>/dev/null || true
			return 0
		fi
		# Fail fast rather than wait out the timeout on a container that died.
		is_running || die "container $CONTAINER exited during startup; see: $0 --logs"
		sleep 1
		waited=$((waited + 1))
	done
	return 1
}

ACTION=start
while [ $# -gt 0 ]; do
	case "$1" in
		--status) ACTION=status ;;
		--logs) ACTION=logs ;;
		--stop) ACTION=stop ;;
		--restart) ACTION=restart ;;
		-h|--help) usage; exit 0 ;;
		*) usage >&2; die "unknown option: $1" ;;
	esac
	shift
done

command -v docker >/dev/null 2>&1 || die "docker is not installed or not on PATH"
docker info >/dev/null 2>&1 || die "cannot talk to the docker daemon - is Docker running?"

case "$ACTION" in
	status)
		if is_running; then
			echo "$CONTAINER is running on http://localhost:$PORT/mcp"
		elif exists; then
			echo "$CONTAINER exists but is stopped"
			exit 1
		else
			echo "$CONTAINER does not exist"
			exit 1
		fi
		exit 0
		;;
	logs)
		exists || die "$CONTAINER does not exist"
		exec docker logs --follow "$CONTAINER"
		;;
	stop)
		if exists; then
			docker rm --force "$CONTAINER" >/dev/null
			echo "Stopped and removed $CONTAINER"
		else
			echo "$CONTAINER is not running"
		fi
		exit 0
		;;
	restart)
		exists && docker rm --force "$CONTAINER" >/dev/null
		;;
esac

if is_running; then
	echo "$CONTAINER is already running on http://localhost:$PORT/mcp"
	exit 0
fi

# A stopped container still has its indexed state; restarting beats recreating.
if exists; then
	echo "Starting existing container $CONTAINER ..."
	docker start "$CONTAINER" >/dev/null
else
	# uid/gid 1000 inside the image, so a bind mount owned by the invoking user
	# stays writable without a chown.
	mkdir -p "$DATA_DIR"
	echo "Creating container $CONTAINER from $IMAGE ..."
	echo "  notes:   $REPO_ROOT (branch $NOTES_BRANCH)"
	echo "  data:    $DATA_DIR"
	docker run --detach \
		--name "$CONTAINER" \
		--restart unless-stopped \
		--publish "127.0.0.1:$PORT:9099" \
		--volume "$REPO_ROOT:/notes-src:ro" \
		--volume "$DATA_DIR:/app/data" \
		--env "NOTES_REPO_URL=$NOTES_REPO_URL" \
		--env "NOTES_BRANCH=$NOTES_BRANCH" \
		--env "NOTES_DIRECTORY=/app/data/notes-cache" \
		--env "CHROMA_PATH=/app/data/.chromadb" \
		--env "REBUILD_CRON=$REBUILD_CRON" \
		--env "TZ=$TIMEZONE" \
		--env "LOG_LEVEL=$LOG_LEVEL" \
		"$IMAGE" \
		--rebuild-on-start >/dev/null
fi

if wait_for_port; then
	echo "$CONTAINER is up on http://localhost:$PORT/mcp"
	echo "The first rebuild may still be indexing; follow it with: $0 --logs"
else
	die "$CONTAINER did not open port $PORT within ${STARTUP_TIMEOUT}s; see: $0 --logs"
fi
