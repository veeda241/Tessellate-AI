#!/bin/sh
# Two processes in one container, mirroring the documented split-process design:
# the FastAPI server and the replay ingest poller, which talk only over HTTP and
# the atomic bridge file. If either exits, the container exits so the platform
# restarts it instead of quietly serving a dead Live feed.
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &
server=$!

sleep 5
python -m api.openf1_streamer &
streamer=$!

while kill -0 "$server" 2>/dev/null && kill -0 "$streamer" 2>/dev/null; do
    sleep 5
done

kill "$server" "$streamer" 2>/dev/null
exit 1
