"""
Локальные CLI-агенты для лаборатории: Claude Code (`claude`) и Codex (`codex`).
Порт NexTask apps/api/src/harness/{spawn,claude,codex}.ts: тот же запуск без shell, те же флаги, те же события.

CLI работает в песочнице (каталог от lab.ai_run), правит там agent.py; его вывод (JSON по строке)
превращается в события {kind: text|tool|log|result|error} для NDJSON-стрима в UI.

    python3 harness.py   # self-check парсеров + статус CLI на этой машине
"""

import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path

HOME = Path.home()
# GUI/Tauri-процесс получает урезанный PATH — CLI из npm/brew/nvm ищем сами
EXTRA_DIRS = [HOME / ".claude" / "local", HOME / ".local" / "bin", Path("/opt/homebrew/bin"), Path("/usr/local/bin"),
              *sorted(HOME.glob(".nvm/versions/node/*/bin"), reverse=True)]
# ключи и секреты приложения агенту не нужны: agent.py без ключа не зовёт LLM, а доступ к БД ему не положен
SECRETS = ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "DATABASE_URL", "ENV_FILE")

EVAL_SCRIPTS = ("local_eval.py", "stress_eval.py", "lab_check.py")  # что агент Claude может запускать в песочнице
READ_ONLY = ("tail", "head", "grep", "ls", "cat", "wc", "sort")  # для `python3 stress_eval.py | tail` и чтения вывода

HARNESSES = {
    "claude": {"title": "Claude Code", "bin": "claude", "models": ["sonnet", "opus", "haiku"],
               "login": "claude auth login", "install": "npm i -g @anthropic-ai/claude-code"},
    "codex": {"title": "Codex", "bin": "codex", "models": [],  # пусто — модель по умолчанию из ~/.codex/config.toml
              "login": "codex login", "install": "npm i -g @openai/codex"},
}


def find_bin(name):
    if p := shutil.which(name):
        return p
    for d in EXTRA_DIRS:
        if os.access(d / name, os.X_OK):
            return str(d / name)
    return None


def child_env(bin_):
    env = {k: v for k, v in os.environ.items() if k not in SECRETS and not k.startswith("AUTH_") and k != "FORCE_COLOR"}
    # python3 агента — тот же, что у сервера (с pandas/numpy), а не первый попавшийся из Homebrew;
    # claude/codex — node-скрипты с `#!/usr/bin/env node`: node лежит рядом с ними (nvm) или в тех же каталогах
    py = [] if getattr(sys, "frozen", False) else [str(Path(sys.executable).parent)]
    env["PATH"] = os.pathsep.join([*py, str(Path(bin_).parent), *map(str, EXTRA_DIRS), env.get("PATH", "")])
    return {**env, "NO_COLOR": "1"}


# --- статус: установлен ли, залогинен ли --------------------------------------
def _claude_auth(bin_):
    try:
        r = subprocess.run([bin_, "auth", "status", "--json"], capture_output=True, text=True, timeout=10, env=child_env(bin_))
        return "ready" if json.loads(r.stdout).get("loggedIn") else "login"
    except (OSError, subprocess.TimeoutExpired, ValueError, AttributeError):
        return "unknown"  # старый CLI без `auth status` — пусть пробует


def _codex_auth(_bin):
    try:
        a = json.loads((Path(os.environ.get("CODEX_HOME", HOME / ".codex")) / "auth.json").read_text())
        return "ready" if a.get("tokens") or a.get("OPENAI_API_KEY") else "login"
    except (OSError, ValueError):
        return "login"


_AUTH = {"claude": _claude_auth, "codex": _codex_auth}
_cache = {"at": 0.0, "data": None}


def _probe(hid, h):
    info = {"id": hid, "title": h["title"], "models": h["models"], "login_cmd": h["login"], "install_cmd": h["install"],
            "installed": False, "version": None, "auth": "missing", "ready": False}
    bin_ = find_bin(h["bin"])
    if not bin_:
        return info
    try:
        r = subprocess.run([bin_, "--version"], capture_output=True, text=True, timeout=10, env=child_env(bin_))
        info["version"] = (r.stdout.strip().splitlines() or [None])[0]
    except (OSError, subprocess.TimeoutExpired):
        return {**info, "auth": "broken"}
    auth = _AUTH[hid](bin_)
    return {**info, "installed": True, "auth": auth, "ready": auth != "login"}


def status(refresh=False):
    if refresh or _cache["data"] is None or time.time() - _cache["at"] > 60:
        _cache.update(at=time.time(), data=[_probe(k, h) for k, h in HARNESSES.items()])
    return _cache["data"]


# --- запуск ---------------------------------------------------------------------
def _codex_mcp_off():
    """`-c mcp_servers={}` не снимает серверы из ~/.codex/config.toml — выключаем каждый по имени."""
    try:
        cfg = tomllib.loads((Path(os.environ.get("CODEX_HOME", HOME / ".codex")) / "config.toml").read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return []
    return [a for name in cfg.get("mcp_servers") or {} for a in ("-c", f"mcp_servers.{name}.enabled=false")]


def command(harness, model, prompt, budget_usd=None):
    if harness == "claude":
        # acceptEdits: правки файлов без вопросов; из Bash — только наши прогоны
        # (не `python3:*`: так проходит `python3 -m pip install` в окружение сервера)
        return ["-p", prompt, "--output-format", "stream-json", "--verbose", "--permission-mode", "acceptEdits",
                "--allowedTools", ",".join([*(f"Bash(python3 {f}:*)" for f in EVAL_SCRIPTS), *(f"Bash({c}:*)" for c in READ_ONLY)]),
                "--strict-mcp-config", "--no-session-persistence", *(["--model", model] if model else []),
                *(["--max-budget-usd", f"{budget_usd:.2f}"] if budget_usd else [])]  # стоп после хода, где дошли до суммы
    return ["exec", "--json", "--ephemeral", "--skip-git-repo-check",
            "-c", 'sandbox_mode="workspace-write"', "-c", 'approval_policy="never"',
            *_codex_mcp_off(), "-c", "features.plugins=false",
            *(["-c", f'model="{model}"'] if model else []), prompt]


def _json(line):
    try:
        ev = json.loads(line)
    except ValueError:
        return None
    return ev if isinstance(ev, dict) else None


def _short(inp):
    for k in ("file_path", "path", "command", "pattern", "description"):
        if isinstance(inp.get(k), str):
            return inp[k][:300]
    return ""


def _text(content):
    return content if isinstance(content, str) else " ".join(c.get("text", "") for c in content or [] if isinstance(c, dict))


def parse_claude(line):
    """stream-json Claude Code → события; tool_result не пересылаем (там целые файлы)."""
    ev = _json(line)
    if ev is None:
        return [{"kind": "log", "text": line}] if line else []
    if ev.get("type") == "result":
        u = ev.get("usage") or {}
        return [{"kind": "result", "ok": not ev.get("is_error"), "text": ev.get("result") or "; ".join(ev.get("errors") or []),
                 "budget": ev.get("subtype") == "error_max_budget_usd", "cost_usd": ev.get("total_cost_usd"),
                 "tokens_in": sum(u.get(k) or 0 for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")),
                 "tokens_out": u.get("output_tokens") or 0}]
    if ev.get("type") == "user":  # отказ в разрешении и падения инструментов — видно, почему агент мечется
        return [{"kind": "log", "text": "отказ: " + _text(c.get("content"))[:300]}
                for c in (ev.get("message") or {}).get("content") or [] if isinstance(c, dict) and c.get("is_error")]
    if ev.get("type") != "assistant":
        return []
    out = []
    for c in (ev.get("message") or {}).get("content") or []:
        if c.get("type") == "text" and c.get("text", "").strip():
            out.append({"kind": "text", "text": c["text"]})
        elif c.get("type") == "tool_use":
            out.append({"kind": "tool", "tool": c.get("name", ""), "detail": _short(c.get("input") or {})})
    return out


def parse_codex(line):
    """codex exec --json: новый формат item.*/turn.*, старый {msg:{type:agent_message}}."""
    ev = _json(line)
    if ev is None:
        return [{"kind": "log", "text": line}] if line else []
    msg = ev.get("msg") if isinstance(ev.get("msg"), dict) else ev
    t = msg.get("type")
    if t == "agent_message":
        return [{"kind": "text", "text": msg.get("message", "")}]
    if t == "item.completed":
        it = msg.get("item") or {}
        if it.get("type") == "agent_message":
            return [{"kind": "text", "text": it.get("text", "")}]
        if it.get("type") == "command_execution":
            return [{"kind": "tool", "tool": "shell", "detail": str(it.get("command", ""))[:300]}]
        if it.get("type") == "file_change":
            return [{"kind": "tool", "tool": "edit", "detail": ", ".join(str(c.get("path", "")) for c in it.get("changes") or [])[:300]}]
        return []
    if t == "turn.completed":  # цены codex не сообщает — только токены
        u = msg.get("usage") or {}
        return [{"kind": "result", "ok": True, "text": "", "cost_usd": None,
                 "tokens_in": u.get("input_tokens") or 0, "tokens_out": u.get("output_tokens") or 0}]
    if t in ("turn.failed", "error"):
        err = msg.get("error") if isinstance(msg.get("error"), dict) else {}
        return [{"kind": "result", "ok": False, "text": msg.get("message") or err.get("message") or t}]
    return []


PARSE = {"claude": parse_claude, "codex": parse_codex}


def kill_tree(pid):
    """
    CLI запускает shell/python дочерними процессами — proc.kill() оставил бы их сиротами.
    Группа процессов не помогает: Claude Code запускает Bash-команды в своих группах. Поэтому — всё дерево по ppid.
    """
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
        return
    kids = {}
    for line in subprocess.run(["ps", "-A", "-o", "pid=,ppid="], capture_output=True, text=True).stdout.splitlines():
        child, parent = map(int, line.split())
        kids.setdefault(parent, []).append(child)
    tree, todo = [], [pid]
    while todo:
        tree.append(todo.pop())
        todo += kids.get(tree[-1], [])
    for p in tree:  # сначала CLI — чтобы не успел породить новых
        try:
            os.kill(p, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def alive_cli(pid):
    """pid ещё принадлежит claude/codex — после перезапуска сервера pid мог достаться другому процессу."""
    if os.name == "nt":
        return False  # ponytail: на Windows сирот после падения сервера не ищем
    cmd = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True).stdout
    return any(Path(w).name in ("claude", "codex") for w in cmd.split()[:2])


async def run(harness, model, prompt, cwd, timeout, on_start=None, budget_usd=None):
    """События CLI по мере вывода. Отмена генератора (клиент ушёл, Stop) или таймаут — процесс убит.
    on_start(pid) — чтобы после падения сервера было кого добить (lab.ai_recover). budget_usd — только Claude."""
    bin_ = find_bin(HARNESSES[harness]["bin"])
    if not bin_:
        yield {"kind": "error", "message": f"{harness} не установлен: {HARNESSES[harness]['install']}"}
        return
    # stderr в тот же поток: «not logged in» и прочее видно в логе; limit — строки stream-json бывают по мегабайту
    proc = await asyncio.create_subprocess_exec(bin_, *command(harness, model, prompt, budget_usd), cwd=cwd, env=child_env(bin_),
                                                stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.STDOUT, limit=1 << 24)
    if on_start:
        on_start(proc.pid)
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    try:
        result = False
        while line := await asyncio.wait_for(proc.stdout.readline(), max(0.1, end - loop.time())):
            for ev in PARSE[harness](line.decode(errors="replace").strip()):
                result = result or ev["kind"] == "result"
                yield ev
        if (code := await proc.wait()) and not result:  # итог с причиной уже был — код выхода ничего не добавит
            yield {"kind": "error", "message": f"{harness} завершился с кодом {code}"}
    except asyncio.TimeoutError:
        yield {"kind": "error", "message": f"таймаут {timeout // 60} мин"}
    finally:
        if proc.returncode is None or os.name != "nt":  # дети могут жить и после выхода CLI
            kill_tree(proc.pid)


if __name__ == "__main__":
    a = '{"type":"assistant","message":{"content":[{"type":"text","text":"смотрю"},{"type":"tool_use","name":"Edit","input":{"file_path":"agent.py"}}]}}'
    assert parse_claude(a) == [{"kind": "text", "text": "смотрю"}, {"kind": "tool", "tool": "Edit", "detail": "agent.py"}]
    r = parse_claude('{"type":"result","is_error":false,"result":"готово","total_cost_usd":0.02,'
                     '"usage":{"input_tokens":10,"cache_read_input_tokens":90,"output_tokens":5}}')[0]
    assert (r["ok"], r["text"], r["cost_usd"], r["tokens_in"], r["tokens_out"]) == (True, "готово", 0.02, 100, 5), r
    r = parse_claude('{"type":"result","subtype":"error_max_budget_usd","is_error":true,"result":null,"errors":["Reached maximum budget ($1)"]}')[0]
    assert r["budget"] and not r["ok"] and r["text"] == "Reached maximum budget ($1)", r
    assert command("claude", "", "x", 0.5)[-2:] == ["--max-budget-usd", "0.50"] and "--max-budget-usd" not in command("claude", "", "x")
    r = parse_codex('{"type":"turn.completed","usage":{"input_tokens":200,"output_tokens":7}}')[0]
    assert (r["kind"], r["cost_usd"], r["tokens_in"], r["tokens_out"]) == ("result", None, 200, 7), r
    deny = '{"type":"user","message":{"content":[{"type":"tool_result","is_error":true,"content":"Permission to use Bash has been denied"}]}}'
    assert parse_claude(deny) == [{"kind": "log", "text": "отказ: Permission to use Bash has been denied"}]
    assert parse_claude('{"type":"user","message":{}}') == [] and parse_claude("Error: not logged in")[0]["kind"] == "log"
    assert parse_codex('{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}') == [{"kind": "text", "text": "ok"}]
    assert parse_codex('{"type":"item.completed","item":{"type":"command_execution","command":"python3 local_eval.py"}}')[0]["detail"] == "python3 local_eval.py"
    assert parse_codex('{"type":"item.completed","item":{"type":"file_change","changes":[{"path":"agent.py"}]}}')[0]["detail"] == "agent.py"
    assert parse_codex('{"id":"0","msg":{"type":"agent_message","message":"old"}}') == [{"kind": "text", "text": "old"}]
    assert parse_codex('{"type":"turn.failed","error":{"message":"quota"}}') == [{"kind": "result", "ok": False, "text": "quota"}]
    async def _cancel_kills_children():
        import tempfile
        HARNESSES["_t"], PARSE["_t"] = {"bin": "bash", "install": ""}, lambda line: [{"kind": "log", "text": line}]
        command_orig = globals()["command"]
        globals()["command"] = lambda h, m, p, b=None: ["-c", "set -m; sleep 300 & echo $!; wait"]  # внук в своей группе, как Bash у claude
        try:
            gen = run("_t", "", "", tempfile.gettempdir(), 60)
            child = int((await gen.__anext__())["text"])
            await gen.aclose()
            await asyncio.sleep(0.2)
            try:
                os.kill(child, 0)
                return False
            except ProcessLookupError:
                return True
        finally:
            globals()["command"] = command_orig
            del HARNESSES["_t"], PARSE["_t"]
    if os.name != "nt":
        assert asyncio.run(_cancel_kills_children()), "дочерний процесс CLI пережил отмену"
    env = child_env("/usr/bin/true")
    assert "OPENAI_API_KEY" not in env and env["PATH"].startswith(str(Path(sys.executable).parent))
    for h in status():
        print(f"{h['id']}: installed={h['installed']} auth={h['auth']} version={h['version']}")
    print("ok")
