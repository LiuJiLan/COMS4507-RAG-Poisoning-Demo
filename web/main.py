"""
Entry point: launch uvicorn + auto-open the browser.
启动入口:跑 uvicorn,顺手打开浏览器。

Usage:
    python -m web.main             # production-ish, single worker, browser-open
    python -m web.main --reload    # dev: hot-reload on file changes, NO auto-open

We delay the webbrowser.open by ~1.5s so uvicorn has time to bind the port
before the browser hits it. Reload mode skips the auto-open: uvicorn's
reloader spawns child workers and a naive open() would fire twice.

延迟 1.5 秒再打开浏览器,等 uvicorn 监听端口。reload 模式不自动开浏览器
(reloader 会 spawn 子进程,naive open 会触发两次)。
"""
import sys
import threading
import webbrowser

import uvicorn

HOST = "127.0.0.1"
PORT = 8000


def _open_browser_when_ready() -> None:
    """Open the dashboard URL after a short delay (best-effort)."""
    threading.Timer(
        1.5, lambda: webbrowser.open(f"http://{HOST}:{PORT}/"),
    ).start()


def main() -> None:
    reload = "--reload" in sys.argv
    if not reload:
        _open_browser_when_ready()
    uvicorn.run(
        "web.api:app",
        host=HOST,
        port=PORT,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
