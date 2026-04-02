Droid Code – AI OS Terminal Browser

Droid Code is a powerful AI‑assisted terminal application that combines an autonomous web browser, file system access, shell execution, Git integration, memory, and advanced features like subagent coordination and memory consolidation. It uses a local LLM (via Ollama) to understand user goals and decide actions, making it a self‑contained AI agent that can browse, search, edit files, run commands, and more—all from your terminal.

The script was inspired by the leaked source code of Anthropic’s Claude Code, implementing many of its architectural concepts (subagents, “dream” memory, Yolo auto‑approval, persistent background mode) in a clean‑room, open‑source manner.
🚀 Features
Feature	Description
Web Browsing & Search	Fetches pages using lynx or requests, caches results, extracts links, and performs DuckDuckGo Lite searches.
File System	Reads, writes, lists, and searches files within configurable trusted directories (strict sandbox).
Shell Commands	Executes shell commands with a configurable allowlist. Supports Yolo mode for auto‑approval of safe commands via a simple ML classifier (or regex fallback).
Git Integration	Basic Git operations: status, diff, commit, branch, log.
Project Awareness	Parses codebases (Python, JS, etc.) to build a symbol index for better context.
Memory	Stores key‑value memories in JSON. Also maintains conversation history.
Dream / Memory Consolidation	Periodically summarizes recent conversations and stores the insights as memories.
Subagents	Spawns background threads that run separate LLM tasks (parallel / isolated).
Scheduling	Allows scheduling of shell commands or web fetches at specific times (daily).
Push Notifications	Sends desktop notifications for completed tasks (optional).
Persistent Background Mode	Watches file changes and can trigger actions (requires watchdog).
Chat Mode	Direct conversation with the AI without triggering web browsing or tools (chat: your message).
Manual Tool Invocation	Call any tool directly with !tool tool_name key=value ....
API Server	Exposes a REST endpoint (/chat) for programmatic use.
Model Management	Lists, selects, and pulls models via Ollama.
📦 Installation
Prerequisites

    Python 3.8 or higher

    Ollama installed and running (for LLM inference)

Install dependencies
bash

pip install requests pyyaml beautifulsoup4 rich ollama
# Optional extras
pip install aiohttp schedule watchdog plyer scikit-learn flask

Download the script

Save the latest version of droid_code.py from this repository.
Make it executable (optional)
bash

chmod +x droid_code.py

🖥️ Usage
Interactive mode
bash

python droid_code.py

API server mode
bash

python droid_code.py --serve

Configuration

Settings are stored in ~/.droid_code_config.yaml. Example:
yaml

model: llama3.2
yolo_enabled: true
shell_allowed_commands:
  - ls
  - pwd
  - echo
  - cat
  - grep
  - git

Commands in interactive mode
Command	Description
!model	Switch models (list available, pull new)
!memory	View stored memories
!tool tool_name [key=value ...]	Call any tool directly
chat: your message	Have a direct conversation with the AI
exit	Quit

When you enter a goal (e.g., “find the latest AI news”), Droid Code will autonomously search, fetch pages, and execute tools until the goal is achieved.
🧠 How It Works

    User input (goal) → process_query()

    If needed, search the web (DuckDuckGo Lite) or fetch a page.

    AI decision (decide_next_action()) → returns JSON with action (search, visit, extract, tool, multi‑edit, stop).

    Execute action: run tools, navigate, or extract answer.

    Loop until goal achieved or max steps reached.

    Optionally, user can interrupt, change model, or view memory.

Subagents

Subagents are background threads that run separate LLM tasks. They are not separate models—they use the same model as the main session. They allow parallel tasks and isolation of subtasks.
Yolo Mode

If yolo_enabled: true in the config, commands deemed “safe” (by a simple ML classifier or regex) are auto‑approved. This is inspired by Claude Code’s “Yolo” feature.
Dream / Memory Consolidation

A background thread periodically summarizes recent conversations and stores the insights as memories, allowing the AI to “remember” across sessions.
🧪 Example Session
text

What do you want to do?: What is the weather in Paris?
Searching for: What is the weather in Paris?
AI decision: This page contains the weather forecast for Paris.
Following link: https://weather.com/weather/tenday/l/Paris+France...
AI decision: The current weather in Paris is 18°C and partly cloudy.
Done!

Chat mode
text

What do you want to do?: chat: What do you think about AI safety?
╭────────────────────── Chat Response ───────────────────────╮
│ AI safety is a critical field... (AI’s response)           │
╰────────────────────────────────────────────────────────────╯

Manual tool invocation
text

What do you want to do?: !tool list_dir path="."
╭──────────────────────── Directory: . ──────────────────────╮
│ droid_code.py
│ README.md
│ ...
╰────────────────────────────────────────────────────────────╯

🔧 Advanced Configuration

    trusted_dirs: List of allowed directories for file operations.

    shell_allowed_commands: Commands that can be executed via the shell tool.

    yolo_enabled: Auto‑approve safe shell commands.

    dream_interval: Seconds between memory consolidation runs.

    persistent_mode: Enable background file‑watching agent.

    notification_enabled: Send desktop notifications for completed tasks.

🛡️ Safety & Sandboxing

    All file operations are restricted to trusted_dirs (default: ~/Documents, ~/Downloads, current directory).

    Shell commands are limited to an allowlist (shell_allowed_commands).

    Yolo mode auto‑approves commands based on safety heuristics (disabled by default).

🤝 Credits

Droid Code is heavily inspired by the Claude Code leak, which revealed an internal architecture with subagents, dream memory, Yolo auto‑approval, and more. This project is a clean‑room implementation, using only publicly available information about those concepts.

Special thanks to the open‑source community and the maintainers of:

    Ollama

    Rich

    Beautiful Soup

    Requests

📝 License

MIT License – feel free to use, modify, and share.
🧩 Contributing

Issues, suggestions, and pull requests are welcome! Please keep the code aligned with the “leak‑inspired” spirit—clean‑room implementations only.

Enjoy exploring the capabilities of your own AI‑driven terminal!

#!/usr/bin/env python3
"""
Droid Code – AI OS Terminal Browser (Enhanced with Claude Code leak features)
Features: Web browsing, file operations, shell, memory, Git, project awareness,
multi-file editing, summarization, comparison, scheduling, subagent coordination,
memory consolidation ("dream"), Yolo auto-approval, persistent background mode,
chat mode, and chatbot API.
"""

import os
import sys
import re
import json
import subprocess
import difflib
import threading
import time
import argparse
import ast
import logging
import hashlib
import asyncio
import queue
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs, unquote
from typing import Dict, List, Optional, Tuple, Any, Union, Callable

import requests
import yaml
from bs4 import BeautifulSoup
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import ollama

# ---------- Optional imports ----------
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    import schedule
    HAS_SCHEDULE = True
except ImportError:
    HAS_SCHEDULE = False

try:
    import watchdog
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False

try:
    from plyer import notification
    HAS_NOTIFY = True
except ImportError:
    HAS_NOTIFY = False

try:
    from flask import Flask, request, jsonify
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ------------------- Logging -------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("droid_code")

# ------------------- Configuration -------------------
CONFIG_FILE = Path.home() / ".droid_code_config.yaml"
DEFAULT_CONFIG = {
    "model": None,
    "trusted_dirs": [
        "~/Documents",
        "~/Downloads",
        "."
    ],
    "user_agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "memory_file": "~/.droid_code_memory.json",
    "conversation_file": "~/.droid_code_conversation.json",
    "cache_ttl": 3600,
    "max_conversation_turns": 20,
    "max_page_chars": 8000,
    "shell_allowed_commands": ["ls", "pwd", "echo", "cat", "grep", "git"],
    "api_host": "127.0.0.1",
    "api_port": 5000,
    "yolo_enabled": False,
    "dream_interval": 3600,          # seconds between memory consolidation
    "persistent_mode": False,         # run background agent
    "notification_enabled": True
}

class Config:
    def __init__(self):
        self._data = DEFAULT_CONFIG.copy()
        self.load()

    def load(self):
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r') as f:
                    user_cfg = yaml.safe_load(f)
                    if user_cfg:
                        self._data.update(user_cfg)
            except Exception as e:
                logger.warning(f"Could not load config: {e}")

    def save(self):
        try:
            with open(CONFIG_FILE, 'w') as f:
                yaml.dump(self._data, f)
        except Exception as e:
            logger.warning(f"Could not save config: {e}")

    def get(self, key, default=None):
        return self._data.get(key, default)

    def set(self, key, value):
        self._data[key] = value
        self.save()

config = Config()

# Expand user paths in trusted dirs
TRUSTED_DIRS = [Path(p).expanduser().resolve() for p in config.get("trusted_dirs")]

# ---------- Session and cache ----------
session = requests.Session()
session.headers.update({"User-Agent": config.get("user_agent")})
console = Console()

# Simple cache for web pages
_cache = {}

def get_cache(url: str) -> Optional[str]:
    if url in _cache:
        timestamp, content = _cache[url]
        if datetime.now() - timestamp < timedelta(seconds=config.get("cache_ttl")):
            return content
        else:
            del _cache[url]
    return None

def set_cache(url: str, content: str):
    _cache[url] = (datetime.now(), content)

# ------------------- Model Management -------------------
def get_available_models():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split("\n")
        if len(lines) < 2:
            return []
        return [line.split()[0] for line in lines[1:] if line.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

def pull_model(model_name):
    console.print(f"[yellow]Pulling model {model_name}...[/yellow]")
    try:
        subprocess.run(["ollama", "pull", model_name], check=True)
        console.print(f"[green]Model {model_name} pulled successfully.[/green]")
        return True
    except subprocess.CalledProcessError:
        console.print(f"[red]Failed to pull model {model_name}.[/red]")
        return False

def select_model():
    console.print(Panel("[bold cyan]Model Selection[/bold cyan]", style="bold"))
    models = get_available_models()
    has_table = bool(models)

    if has_table:
        table = Table(title="Available Models")
        table.add_column("#", style="cyan")
        table.add_column("Model Name", style="green")
        for idx, m in enumerate(models, 1):
            table.add_row(str(idx), m)
        console.print(table)
        console.print("\nYou can:")
        console.print("  • Enter a [green]number[/green] to select a model from the table above")
        console.print("  • Type a [green]model name[/green] to pull it (e.g., llama3.2, qwen2.5:7b)")
        console.print("  • Choose one of the options below")
    else:
        console.print("[yellow]No models found. You can pull a new model.[/yellow]")

    while True:
        if has_table:
            prompt_text = "Enter your choice (number, model name, or [Choose from list/Pull new model/Use default]): "
        else:
            prompt_text = "Enter your choice (model name, or [Pull new model/Use default]): "

        choice = Prompt.ask(prompt_text)

        if has_table and choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                model = models[idx]
                console.print(f"[bold green]Selected model: {model}[/bold green]")
                if Confirm.ask(f"Update '{model}' to the latest version?", default=False):
                    pull_model(model)
                return model
            else:
                console.print("[red]Invalid number.[/red]")
                continue

        choice_lower = choice.lower()
        if any(phrase in choice_lower for phrase in ["choose", "list", "table"]):
            if has_table:
                continue
            else:
                console.print("[red]No models available. Please pull one.[/red]")
        elif any(phrase in choice_lower for phrase in ["pull", "new model"]):
            new_model = Prompt.ask("Enter model name (e.g., llama3.2, qwen2.5:7b)")
            if new_model and pull_model(new_model):
                return new_model
            else:
                console.print("[red]Pull failed or cancelled. Using default model.[/red]")
                return "llama3.2"
        elif any(phrase in choice_lower for phrase in ["default", "llama3.2"]):
            return "llama3.2"
        else:
            model_name = choice.strip()
            existing = next((m for m in models if m.lower() == model_name.lower()), None)
            if existing:
                model = existing
                console.print(f"[bold green]Using existing model: {model}[/bold green]")
                if Confirm.ask(f"Update '{model}' to the latest version?", default=False):
                    pull_model(model)
                return model
            else:
                console.print(f"[yellow]Model '{model_name}' not found. Attempting to pull...[/yellow]")
                if pull_model(model_name):
                    return model_name
                else:
                    console.print("[red]Pull failed. Using default.[/red]")
                    return "llama3.2"

# ------------------- Memory Management -------------------
def load_memory():
    mem_path = Path(config.get("memory_file")).expanduser()
    if mem_path.exists():
        try:
            with open(mem_path, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_memory(memory):
    mem_path = Path(config.get("memory_file")).expanduser()
    try:
        with open(mem_path, 'w') as f:
            json.dump(memory, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save memory: {e}")

def memory_save(key, value):
    mem = load_memory()
    mem[key] = value
    save_memory(mem)
    return f"Saved memory: {key} = {value}"

def memory_recall(key):
    mem = load_memory()
    value = mem.get(key)
    if value is None:
        return f"No memory found for key: {key}"
    return value

def memory_list():
    mem = load_memory()
    if not mem:
        return "No memories stored."
    return "\n".join(f"- {k}: {v}" for k, v in mem.items())

# ------------------- Conversation Memory -------------------
def load_conversation():
    conv_path = Path(config.get("conversation_file")).expanduser()
    if conv_path.exists():
        try:
            with open(conv_path, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_conversation(conv):
    conv_path = Path(config.get("conversation_file")).expanduser()
    max_turns = config.get("max_conversation_turns")
    if max_turns and len(conv) > max_turns:
        conv = conv[-max_turns:]
    try:
        with open(conv_path, 'w') as f:
            json.dump(conv, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save conversation: {e}")

def add_to_conversation(role, content):
    conv = load_conversation()
    conv.append({"role": role, "content": content, "timestamp": datetime.now().isoformat()})
    save_conversation(conv)

def get_conversation_context(limit=10):
    conv = load_conversation()
    return conv[-limit:] if limit else conv

# ------------------- Dream / Memory Consolidation -------------------
def dream_cycle():
    """Consolidate recent conversations into long-term memory."""
    conv = load_conversation()
    if not conv:
        return
    recent = conv[-5:]
    text = "\n".join(f"{item['role']}: {item['content']}" for item in recent)
    prompt = f"Extract key facts, learnings, and insights from this conversation in a concise bullet list:\n\n{text}"
    summary = ask_llm(prompt)
    if summary:
        memory_save(f"dream_{datetime.now().isoformat()}", summary)
        logger.info("Dream cycle completed: consolidated memories")

def start_dream_worker():
    """Periodically run dream_cycle in background."""
    if not config.get("dream_interval"):
        return
    def worker():
        while True:
            time.sleep(config.get("dream_interval"))
            try:
                dream_cycle()
            except Exception as e:
                logger.error(f"Dream cycle error: {e}")
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    logger.info("Dream worker started")

# ------------------- Yolo Mode Auto-Approval -------------------
class YoloClassifier:
    """Simple ML-based classifier to auto-approve safe commands."""
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.model = None
        self.trained = False
        if HAS_SKLEARN:
            self._train()

    def _train(self):
        examples = [
            ("ls -la", 1), ("pwd", 1), ("echo hello", 1), ("cat file.txt", 1),
            ("git status", 1), ("grep pattern", 1),
            ("rm -rf /", 0), ("sudo", 0), ("chmod 777", 0), ("dd if=/dev/zero", 0)
        ]
        texts, labels = zip(*examples)
        X = self.vectorizer.fit_transform(texts)
        self.model = MultinomialNB()
        self.model.fit(X, labels)
        self.trained = True

    def predict(self, command):
        if not self.trained or not HAS_SKLEARN:
            dangerous = ["rm", "sudo", "dd", "mkfs", "chmod 777", "chown"]
            return not any(part in command for part in dangerous)
        X = self.vectorizer.transform([command])
        proba = self.model.predict_proba(X)[0]
        return proba[1] > 0.7

yolo_classifier = YoloClassifier()

def auto_approve_command(cmd: str) -> bool:
    if config.get("yolo_enabled"):
        return yolo_classifier.predict(cmd)
    return False

# ------------------- Push Notifications -------------------
def send_notification(title: str, message: str):
    if config.get("notification_enabled") and HAS_NOTIFY:
        try:
            notification.notify(title=title, message=message, timeout=5)
        except Exception as e:
            logger.warning(f"Notification failed: {e}")

# ------------------- Subagent / Coordinator System -------------------
class Subagent:
    def __init__(self, task: str, context: str = "", parent=None):
        self.task = task
        self.context = context
        self.parent = parent
        self.result = None
        self.status = "pending"

    def run(self):
        self.status = "running"
        prompt = f"Task: {self.task}\n\nContext: {self.context}\n\nPlease accomplish this task and return the result in a concise way."
        self.result = ask_llm(prompt)
        self.status = "completed"
        if self.parent:
            self.parent.receive_subagent_result(self.result)
        return self.result

def spawn_subagent(task: str, context: str = "") -> Subagent:
    sub = Subagent(task, context)
    thread = threading.Thread(target=sub.run, daemon=True)
    thread.start()
    return sub

class Coordinator:
    def __init__(self):
        self.subagents = []
        self.results = []
        self.lock = threading.Lock()

    def spawn(self, task: str, context: str = ""):
        sub = Subagent(task, context, parent=self)
        self.subagents.append(sub)
        sub.run()
        return sub

    def receive_subagent_result(self, result):
        with self.lock:
            self.results.append(result)

    def wait_all(self, timeout=30):
        start = time.time()
        while len(self.results) < len(self.subagents) and (time.time() - start) < timeout:
            time.sleep(0.1)
        return self.results

# ------------------- File System (Strict Sandbox) -------------------
def _sanitize_path(path: Union[str, Path]) -> Path:
    try:
        p = Path(path).expanduser().resolve()
    except Exception:
        raise PermissionError(f"Invalid path: {path}")
    for trusted in TRUSTED_DIRS:
        try:
            p.relative_to(trusted)
            return p
        except ValueError:
            continue
    raise PermissionError(f"Access denied: {path} is outside trusted directories.")

def read_file(path: Union[str, Path]) -> str:
    try:
        safe_path = _sanitize_path(path)
        return safe_path.read_text(encoding='utf-8')
    except Exception as e:
        return f"Error reading file: {e}"

def write_file(path: Union[str, Path], content: str, append: bool = False) -> str:
    try:
        safe_path = _sanitize_path(path)
        mode = 'a' if append else 'w'
        with open(safe_path, mode, encoding='utf-8') as f:
            f.write(content)
        return f"File {'appended to' if append else 'written'}: {safe_path}"
    except Exception as e:
        return f"Error writing file: {e}"

def list_dir(path: Union[str, Path] = ".") -> str:
    try:
        safe_path = _sanitize_path(path)
        return "\n".join(str(x) for x in safe_path.iterdir())
    except Exception as e:
        return f"Error listing directory: {e}"

def search_files(pattern: str, root: Union[str, Path] = ".") -> str:
    try:
        safe_root = _sanitize_path(root)
        matches = []
        for file_path in safe_root.rglob("*"):
            if file_path.is_file() and pattern in file_path.name:
                matches.append(str(file_path))
        return "\n".join(matches) if matches else "No matching files found."
    except Exception as e:
        return f"Error searching: {e}"

# ------------------- Codebase Awareness (Project Index) -------------------
class ProjectIndex:
    def __init__(self, root_path: Union[str, Path]):
        self.root = Path(root_path).resolve()
        self.files = []
        self.imports = {}
        self.definitions = {}

    def build(self):
        for file_path in self.root.rglob("*"):
            if file_path.suffix in ('.py', '.js', '.ts', '.go', '.java', '.c', '.cpp'):
                rel = str(file_path.relative_to(self.root))
                self.files.append(rel)
                if file_path.suffix == '.py':
                    self._parse_python(file_path, rel)

    def _parse_python(self, file_path: Path, rel: str):
        try:
            tree = ast.parse(file_path.read_text(encoding='utf-8'))
        except (SyntaxError, Exception):
            return
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        self.imports[rel] = imports
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                name = node.name
                self.definitions.setdefault(name, []).append((rel, node.lineno))

    def get_summary(self):
        lines = [f"Project root: {self.root}"]
        lines.append(f"Files: {len(self.files)}")
        top_files = [f for f in self.files if '/' not in f][:10]
        lines.append(f"Top-level files: {', '.join(top_files)}")
        top_symbols = list(self.definitions.keys())[:20]
        lines.append(f"Top symbols: {', '.join(top_symbols)}")
        return "\n".join(lines)

    def find_definition(self, symbol):
        return self.definitions.get(symbol, [])

    def find_imports_of(self, module):
        results = []
        for file, imps in self.imports.items():
            if any(module in imp for imp in imps):
                results.append(file)
        return results

_project_index = None

def get_project_index(root: Union[str, Path] = ".") -> ProjectIndex:
    global _project_index
    safe_root = _sanitize_path(root)
    if _project_index is None or _project_index.root != safe_root:
        _project_index = ProjectIndex(safe_root)
        _project_index.build()
    return _project_index

def project_info(root: Union[str, Path] = ".") -> str:
    index = get_project_index(root)
    return index.get_summary()

# ------------------- Git Integration -------------------
def git_status(repo_path: Union[str, Path] = ".") -> str:
    safe_path = _sanitize_path(repo_path)
    try:
        result = subprocess.run(
            ["git", "-C", str(safe_path), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return f"Git error: {result.stderr}"
        return result.stdout if result.stdout.strip() else "Working tree clean"
    except Exception as e:
        return f"Error: {e}"

def git_diff(repo_path: Union[str, Path] = ".", staged: bool = False) -> str:
    safe_path = _sanitize_path(repo_path)
    cmd = ["git", "-C", str(safe_path), "diff"]
    if staged:
        cmd.append("--staged")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.stdout if result.stdout.strip() else "No changes to show"
    except Exception as e:
        return f"Error: {e}"

def git_commit(repo_path: Union[str, Path] = ".", message: str = "") -> str:
    if not message:
        return "Commit message required"
    safe_path = _sanitize_path(repo_path)
    try:
        result = subprocess.run(
            ["git", "-C", str(safe_path), "commit", "-m", message],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return f"Commit failed: {result.stderr}"
        return result.stdout
    except Exception as e:
        return f"Error: {e}"

def git_branch(repo_path: Union[str, Path] = ".", new_branch: Optional[str] = None) -> str:
    safe_path = _sanitize_path(repo_path)
    if new_branch:
        cmd = ["git", "-C", str(safe_path), "checkout", "-b", new_branch]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return result.stdout.strip() or result.stderr
        except Exception as e:
            return f"Error: {e}"
    else:
        try:
            result = subprocess.run(
                ["git", "-C", str(safe_path), "branch", "--show-current"],
                capture_output=True, text=True, timeout=10
            )
            return result.stdout.strip()
        except Exception as e:
            return f"Error: {e}"

def git_log(repo_path: Union[str, Path] = ".", n: int = 10) -> str:
    safe_path = _sanitize_path(repo_path)
    try:
        result = subprocess.run(
            ["git", "-C", str(safe_path), "log", f"-{n}", "--oneline"],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout if result.stdout.strip() else "No commits"
    except Exception as e:
        return f"Error: {e}"

# ------------------- Summarization -------------------
def summarize_text(text: str, max_length: int = 500) -> str:
    prompt = f"Summarize the following text concisely (max {max_length} words):\n\n{text[:8000]}"
    response = ask_llm(prompt)
    return response or "Summarization failed."

# ------------------- Comparison Tools -------------------
def compare_files(file1: Union[str, Path], file2: Union[str, Path]) -> str:
    try:
        content1 = read_file(file1).splitlines()
        content2 = read_file(file2).splitlines()
    except Exception as e:
        return f"Error reading files: {e}"
    diff = difflib.unified_diff(content1, content2, fromfile=str(file1), tofile=str(file2))
    diff_text = '\n'.join(diff)
    return diff_text if diff_text else "Files are identical."

def compare_webpages(url1: str, url2: str) -> str:
    text1 = fetch_page(url1)
    text2 = fetch_page(url2)
    lines1 = text1.splitlines()
    lines2 = text2.splitlines()
    diff = difflib.unified_diff(lines1, lines2, fromfile=url1, tofile=url2)
    diff_text = '\n'.join(diff)
    return diff_text if diff_text else "Pages are identical."

# ------------------- Scheduled Tasks -------------------
scheduled_jobs = []
_scheduler_running = False
_scheduler_thread = None

def start_scheduler():
    global _scheduler_running, _scheduler_thread
    if _scheduler_running or not HAS_SCHEDULE:
        return
    _scheduler_running = True

    def run_loop():
        while _scheduler_running:
            schedule.run_pending()
            time.sleep(1)

    _scheduler_thread = threading.Thread(target=run_loop, daemon=True)
    _scheduler_thread.start()

def stop_scheduler():
    global _scheduler_running
    _scheduler_running = False
    if _scheduler_thread and _scheduler_thread.is_alive():
        _scheduler_thread.join(timeout=2)

def _execute_scheduled_task(job):
    command = job['command']
    tool_args = job.get('tool_args', {})
    if command.startswith('shell:'):
        cmd = command[6:]
        result = _safe_shell_command(cmd, auto_confirm=True)
        console.print(f"[yellow]Scheduled task executed: {cmd}[/yellow]\n{result}")
        send_notification("Scheduled Task", f"Executed: {cmd}")
    elif command.startswith('fetch:'):
        url = command[6:]
        content = fetch_page(url)
        output = tool_args.get('output', 'schedule_output.txt')
        write_file(output, content, append=True)
        console.print(f"[yellow]Scheduled fetch saved to {output}[/yellow]")
        send_notification("Scheduled Fetch", f"Saved {url} to {output}")
    else:
        console.print(f"[red]Unknown scheduled command: {command}[/red]")

def schedule_task(command: str, time_str: str, tool_args: Optional[Dict] = None) -> str:
    if not HAS_SCHEDULE:
        return "Schedule library not installed. Install with: pip install schedule"
    start_scheduler()
    job = {
        'command': command,
        'tool_args': tool_args or {}
    }
    try:
        schedule.every().day.at(time_str).do(_execute_scheduled_task, job)
        return f"Scheduled {command} at {time_str} daily"
    except Exception as e:
        return f"Failed to schedule: {e}"

def list_scheduled_tasks() -> str:
    if not HAS_SCHEDULE:
        return "Schedule library not installed."
    jobs = schedule.get_jobs()
    if not jobs:
        return "No scheduled tasks."
    lines = []
    for i, job in enumerate(jobs, 1):
        lines.append(f"{i}. {job}")
    return "\n".join(lines)

def cancel_scheduled_task(index: int) -> str:
    if not HAS_SCHEDULE:
        return "Schedule library not installed."
    jobs = schedule.get_jobs()
    if 1 <= index <= len(jobs):
        schedule.cancel_job(jobs[index-1])
        return f"Cancelled task {index}"
    else:
        return "Invalid task index"

# ------------------- Shell Commands (with allowlist) -------------------
def _safe_shell_command(cmd: str, auto_confirm: bool = False) -> str:
    parts = cmd.split()
    if not parts:
        return "Empty command."
    executable = parts[0]
    allowed = config.get("shell_allowed_commands")
    if executable not in allowed:
        return f"Command '{executable}' is not allowed. Allowed: {', '.join(allowed)}"
    if not auto_confirm and not auto_approve_command(cmd):
        console.print(Panel(f"[yellow]Shell command: {cmd}[/yellow]", title="⚠️ Shell Execution", border_style="red"))
        if not Confirm.ask("Execute this command?", default=False):
            return "Command cancelled by user."
    try:
        result = subprocess.run(parts, capture_output=True, text=True, timeout=60)
        output = result.stdout + result.stderr
        return output if output.strip() else "(no output)"
    except subprocess.TimeoutExpired:
        return "Command timed out after 60 seconds."
    except Exception as e:
        return f"Error executing command: {e}"

# ------------------- Web Helpers -------------------
def resolve_redirect(url: str) -> str:
    """Resolve DuckDuckGo redirects and add scheme to // URLs."""
    # Handle relative scheme URLs
    if url.startswith('//'):
        url = 'https:' + url
    # Check for DuckDuckGo redirect link
    if 'duckduckgo.com/l/' in url:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        if 'uddg' in qs:
            target = qs['uddg'][0]
            # Sometimes target is URL-encoded
            try:
                target = unquote(target)
            except:
                pass
            return target
    # Follow redirects (HTTP 301/302)
    try:
        resp = session.get(url, allow_redirects=False, timeout=5)
        if resp.status_code in (301, 302) and 'Location' in resp.headers:
            return resp.headers['Location']
    except:
        pass
    return url

def fetch_page(url: str) -> str:
    cached = get_cache(url)
    if cached is not None:
        return cached
    # Normalize URL
    url = resolve_redirect(url)
    try:
        proc = subprocess.run(["lynx", "-dump", "-nolist", url],
                              capture_output=True, text=True, timeout=30)
        if proc.returncode == 0:
            content = proc.stdout
            set_cache(url, content)
            return content
        else:
            logger.warning("lynx returned error, falling back to requests")
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for script in soup(["script", "style"]):
                script.decompose()
            content = soup.get_text(separator="\n")
            set_cache(url, content)
            return content
    except FileNotFoundError:
        logger.info("lynx not found, using requests")
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for script in soup(["script", "style"]):
                script.decompose()
            content = soup.get_text(separator="\n")
            set_cache(url, content)
            return content
        except Exception as e:
            return f"Error fetching page: {e}"
    except Exception as e:
        return f"Error fetching page: {e}"

async def fetch_page_async(url: str) -> str:
    cached = get_cache(url)
    if cached is not None:
        return cached
    if not HAS_AIOHTTP:
        return fetch_page(url)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers={"User-Agent": config.get("user_agent")}) as resp:
                text = await resp.text()
                soup = BeautifulSoup(text, "html.parser")
                for script in soup(["script", "style"]):
                    script.decompose()
                content = soup.get_text(separator="\n")
                set_cache(url, content)
                return content
    except Exception as e:
        return f"Error fetching page asynchronously: {e}"

def extract_links(page_text: str, base_url: str) -> List[Tuple[str, str]]:
    try:
        # Use the actual base URL to fetch the page again for link extraction
        resp = session.get(base_url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            full_url = urljoin(base_url, href)
            # Normalize scheme
            if full_url.startswith('//'):
                full_url = 'https:' + full_url
            text = a.get_text(strip=True)
            links.append((text if text else full_url, full_url))
        return links
    except Exception as e:
        logger.error(f"Error extracting links: {e}")
        return []

def search_duckduckgo_lite(query: str) -> str:
    url = "https://lite.duckduckgo.com/lite/"
    params = {"q": query}
    try:
        resp = session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        tables = soup.find_all("table")
        result_table = None
        for tbl in tables:
            if tbl.find("a", href=True):
                result_table = tbl
                break
        if not result_table:
            return "No results found."
        rows = result_table.find_all("tr")
        for row in rows:
            link_cell = row.find("a")
            if link_cell and link_cell.get("href"):
                title = link_cell.get_text(strip=True)
                href = link_cell["href"]
                if any(phrase in title.lower() for phrase in ["ad", "sponsored", "promoted"]):
                    continue
                if "y.js" in href or "ad_domain" in href:
                    continue
                href = resolve_redirect(href)
                results.append((title, href))
        if not results:
            return "No results found."
        result_text = f"Search results for: {query}\n\n"
        for i, (title, href) in enumerate(results[:10], 1):
            result_text += f"{i}. {title}\n   {href}\n\n"
        return result_text
    except Exception as e:
        return f"Search failed: {e}"

# ------------------- AI Decision -------------------
def ask_llm(prompt: str, context: str = "", stream: bool = False) -> Union[str, Any]:
    full_prompt = f"{context}\n\n{prompt}" if context else prompt
    try:
        if stream:
            return ollama.chat(model=config.get("model"), messages=[{"role": "user", "content": full_prompt}], stream=True)
        else:
            response = ollama.chat(model=config.get("model"), messages=[{"role": "user", "content": full_prompt}])
            return response["message"]["content"]
    except Exception as e:
        logger.error(f"Error calling Ollama: {e}")
        return None

def decide_next_action(user_query: str, current_url: str, page_text: str, available_links: List[Tuple[str, str]]) -> Dict:
    max_chars = config.get("max_page_chars")
    if len(page_text) > max_chars:
        page_text = page_text[:max_chars] + "... [truncated]"
    links_text = "\n".join([f"- {text}: {url}" for text, url in available_links[:20]])
    tools_description = """
You have access to additional tools:
- **File system**: read_file(path), write_file(path, content, append=False), list_dir(path="."), search_files(pattern, root=".")
- **Shell**: shell(command) – executes a system command (requires confirmation). Allowed commands: ls, pwd, echo, cat, grep, git.
- **Memory**: save_memory(key, value), recall_memory(key), list_memory()
- **Git**: git_status(repo_path="."), git_diff(repo_path=".", staged=False), git_commit(repo_path=".", message=""), git_branch(repo_path=".", new_branch=None), git_log(repo_path=".", n=10)
- **Project**: project_info(root=".") – gives overview of codebase
- **Summarization**: summarize(text) – returns concise summary
- **Comparison**: compare_files(file1, file2), compare_webpages(url1, url2)
- **Scheduling**: schedule_task(command, time_str, tool_args=None), list_scheduled_tasks(), cancel_scheduled_task(index)
- **Web browsing**: search(query), visit_link(url), extract(answer)
- **Subagent**: spawn_subagent(task, context) – starts a new subagent to handle a subtask in parallel.
- **Multi‑edit**: multi_edit(edits) – where edits is a list of {"path": "...", "content": "...", "append": false} objects.
- **Notification**: send_notification(title, message) – sends a desktop alert.

When you need to use a tool, respond with a JSON object that includes "action": "tool", "tool_name": <name>, and "tool_args": <arguments>.
For example:
{"action": "tool", "tool_name": "write_file", "tool_args": {"path": "~/Documents/note.txt", "content": "Hello"}}
{"action": "tool", "tool_name": "shell", "tool_args": {"command": "ls -la"}}
{"action": "tool", "tool_name": "spawn_subagent", "tool_args": {"task": "summarize the latest news", "context": "focus on AI"}}
{"action": "multi_edit", "edits": [{"path": "main.py", "content": "print('hello')"}, {"path": "utils.py", "content": "def foo(): pass"}], "reason": "Add new functions"}
"""
    prompt = f"""
You are Droid Code, an AI OS Terminal Browser. The user's goal is: "{user_query}"
You are currently on: {current_url}

Page content (first {max_chars} chars):
{page_text}

Available links on this page:
{links_text}

{tools_description}

**IMPORTANT INSTRUCTIONS:**
- If the current page is a search results page, the goal is NOT yet achieved. You must choose a relevant link to visit and then extract the actual information from that page.
- Only stop if the page already contains the exact answer the user wants, or if you have already extracted it.
- If you are on a search results page, use "visit_link" with one of the URLs from the list.
- If you are on a page that likely contains the answer, use "extract" to pull out the relevant information.
- Only use "search" again if the current page is completely irrelevant.

Decide what to do next. Choose one action from:
- "search" (with query)
- "visit_link" (with url)
- "extract" (with answer)
- "tool" (with tool_name and tool_args)
- "multi_edit" (with edits list)
- "stop"

Respond with a JSON object. Example responses:
{{"action": "visit_link", "url": "https://techcrunch.com/ai/", "reason": "This looks like a major AI news source"}}
{{"action": "extract", "answer": "The latest AI news: ...", "reason": "Found the answer on this page"}}
{{"action": "search", "query": "latest AI news 2026", "reason": "No relevant results found"}}
{{"action": "tool", "tool_name": "write_file", "tool_args": {{"path": "~/Documents/notes.txt", "content": "Meeting notes"}}, "reason": "User asked to save notes"}}
{{"action": "multi_edit", "edits": [{{"path": "main.py", "content": "print('hello')"}}, {{"path": "utils.py", "content": "def foo(): pass"}}], "reason": "Add new functions"}}
{{"action": "stop", "reason": "Goal achieved"}}

Do NOT stop on a search results page. Always try to get to the actual content.
"""
    response = ask_llm(prompt)
    if not response:
        return {"action": "stop", "reason": "LLM error"}

    # Uncomment to debug raw response
    # console.print(f"[dim]Raw LLM response: {response}[/dim]")

    # Try to extract JSON from the response
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: try to fix common issues (e.g., missing quotes, etc.)
    try:
        return json.loads(response)
    except:
        return {"action": "stop", "reason": "Could not parse LLM response"}

# ------------------- Tool Execution Helper -------------------
def execute_tool(tool_name: str, tool_args: Dict, interactive: bool) -> Optional[str]:
    """Execute a tool and return the result as string (or None if interactive handled)."""
    if tool_name == "read_file":
        path = tool_args.get("path")
        if path:
            result = read_file(path)
            if interactive:
                console.print(Panel(result, title=f"File: {path}", border_style="cyan"))
            return result
        else:
            if interactive:
                console.print("[red]Missing path for read_file[/red]")
            return "Missing path for read_file"
    elif tool_name == "write_file":
        path = tool_args.get("path")
        content = tool_args.get("content", "")
        append = tool_args.get("append", False)
        if path:
            result = write_file(path, content, append)
            if interactive:
                console.print(Panel(result, title="Write Result", border_style="green"))
            return result
        else:
            if interactive:
                console.print("[red]Missing path for write_file[/red]")
            return "Missing path for write_file"
    elif tool_name == "list_dir":
        path = tool_args.get("path", ".")
        result = list_dir(path)
        if interactive:
            console.print(Panel(result, title=f"Directory: {path}", border_style="cyan"))
        return result
    elif tool_name == "search_files":
        pattern = tool_args.get("pattern")
        root = tool_args.get("root", ".")
        if pattern:
            result = search_files(pattern, root)
            if interactive:
                console.print(Panel(result, title=f"Search for '{pattern}'", border_style="cyan"))
            return result
        else:
            if interactive:
                console.print("[red]Missing pattern for search_files[/red]")
            return "Missing pattern for search_files"
    elif tool_name == "shell":
        cmd = tool_args.get("command")
        if cmd:
            result = _safe_shell_command(cmd, auto_confirm=not interactive)
            if interactive:
                console.print(Panel(result, title="Shell Output", border_style="yellow"))
            return result
        else:
            if interactive:
                console.print("[red]Missing command for shell[/red]")
            return "Missing command for shell"
    elif tool_name == "save_memory":
        key = tool_args.get("key")
        value = tool_args.get("value")
        if key and value:
            result = memory_save(key, value)
            if interactive:
                console.print(Panel(result, title="Memory Saved", border_style="green"))
            return result
        else:
            if interactive:
                console.print("[red]Missing key or value for save_memory[/red]")
            return "Missing key or value for save_memory"
    elif tool_name == "recall_memory":
        key = tool_args.get("key")
        if key:
            result = memory_recall(key)
            if interactive:
                console.print(Panel(result, title=f"Memory: {key}", border_style="cyan"))
            return result
        else:
            if interactive:
                console.print("[red]Missing key for recall_memory[/red]")
            return "Missing key for recall_memory"
    elif tool_name == "list_memory":
        result = memory_list()
        if interactive:
            console.print(Panel(result, title="Memory List", border_style="cyan"))
        return result
    elif tool_name == "project_info":
        root = tool_args.get("root", ".")
        result = project_info(root)
        if interactive:
            console.print(Panel(result, title="Project Info", border_style="green"))
        return result
    elif tool_name == "git_status":
        repo = tool_args.get("repo_path", ".")
        result = git_status(repo)
        if interactive:
            console.print(Panel(result, title="Git Status", border_style="green"))
        return result
    elif tool_name == "git_diff":
        repo = tool_args.get("repo_path", ".")
        staged = tool_args.get("staged", False)
        result = git_diff(repo, staged)
        if interactive:
            console.print(Panel(result, title="Git Diff", border_style="green"))
        return result
    elif tool_name == "git_commit":
        repo = tool_args.get("repo_path", ".")
        message = tool_args.get("message", "")
        result = git_commit(repo, message)
        if interactive:
            console.print(Panel(result, title="Git Commit", border_style="green"))
        return result
    elif tool_name == "git_branch":
        repo = tool_args.get("repo_path", ".")
        new_branch = tool_args.get("new_branch", None)
        result = git_branch(repo, new_branch)
        if interactive:
            console.print(Panel(result, title="Git Branch", border_style="green"))
        return result
    elif tool_name == "git_log":
        repo = tool_args.get("repo_path", ".")
        n = tool_args.get("n", 10)
        result = git_log(repo, n)
        if interactive:
            console.print(Panel(result, title="Git Log", border_style="green"))
        return result
    elif tool_name == "summarize":
        text = tool_args.get("text", "")
        if text:
            result = summarize_text(text)
            if interactive:
                console.print(Panel(result, title="Summary", border_style="cyan"))
            return result
        else:
            if interactive:
                console.print("[red]Missing text for summarize[/red]")
            return "Missing text for summarize"
    elif tool_name == "compare_files":
        file1 = tool_args.get("file1")
        file2 = tool_args.get("file2")
        if file1 and file2:
            result = compare_files(file1, file2)
            if interactive:
                console.print(Panel(result, title="File Comparison", border_style="cyan"))
            return result
        else:
            if interactive:
                console.print("[red]Missing file1 or file2 for compare_files[/red]")
            return "Missing file1 or file2 for compare_files"
    elif tool_name == "compare_webpages":
        url1 = tool_args.get("url1")
        url2 = tool_args.get("url2")
        if url1 and url2:
            result = compare_webpages(url1, url2)
            if interactive:
                console.print(Panel(result, title="Webpage Comparison", border_style="cyan"))
            return result
        else:
            if interactive:
                console.print("[red]Missing url1 or url2 for compare_webpages[/red]")
            return "Missing url1 or url2 for compare_webpages"
    elif tool_name == "schedule_task":
        command = tool_args.get("command")
        time_str = tool_args.get("time_str")
        task_tool_args = tool_args.get("tool_args", {})
        if command and time_str:
            result = schedule_task(command, time_str, task_tool_args)
            if interactive:
                console.print(Panel(result, title="Schedule Task", border_style="green"))
            return result
        else:
            if interactive:
                console.print("[red]Missing command or time_str for schedule_task[/red]")
            return "Missing command or time_str for schedule_task"
    elif tool_name == "list_scheduled_tasks":
        result = list_scheduled_tasks()
        if interactive:
            console.print(Panel(result, title="Scheduled Tasks", border_style="cyan"))
        return result
    elif tool_name == "cancel_scheduled_task":
        index = tool_args.get("index")
        if index is not None:
            result = cancel_scheduled_task(index)
            if interactive:
                console.print(Panel(result, title="Cancel Task", border_style="green"))
            return result
        else:
            if interactive:
                console.print("[red]Missing index for cancel_scheduled_task[/red]")
            return "Missing index for cancel_scheduled_task"
    elif tool_name == "spawn_subagent":
        task = tool_args.get("task")
        context = tool_args.get("context", "")
        if task:
            sub = spawn_subagent(task, context)
            if interactive:
                console.print(Panel(f"Subagent spawned for task: {task}", title="Subagent", border_style="blue"))
                # Optionally wait for result
                if Confirm.ask("Wait for subagent result?", default=True):
                    timeout = 30
                    start = time.time()
                    while sub.status != "completed" and (time.time() - start) < timeout:
                        time.sleep(0.5)
                    if sub.result:
                        console.print(Panel(sub.result, title="Subagent Result", border_style="green"))
                        return sub.result
                    else:
                        console.print("[red]Subagent did not complete in time[/red]")
                        return "Subagent timed out"
                else:
                    return f"Subagent spawned (task: {task})"
            else:
                # Non-interactive: just spawn and return immediately
                return f"Subagent spawned (task: {task})"
        else:
            if interactive:
                console.print("[red]Missing task for spawn_subagent[/red]")
            return "Missing task for spawn_subagent"
    elif tool_name == "send_notification":
        title = tool_args.get("title", "Droid Code")
        message = tool_args.get("message", "")
        if message:
            send_notification(title, message)
            if interactive:
                console.print(f"[green]Notification sent: {message}[/green]")
            return "Notification sent"
        else:
            if interactive:
                console.print("[red]Missing message for send_notification[/red]")
            return "Missing message for send_notification"
    else:
        if interactive:
            console.print(f"[red]Unknown tool: {tool_name}[/red]")
        return f"Unknown tool: {tool_name}"

# ------------------- Chat Feature -------------------
def chat_with_ai(user_message: str) -> str:
    """Handle a free‑form chat message (not a goal)."""
    query = user_message.strip()
    if query.lower().startswith("chat:"):
        query = query[5:].strip()
    if not query:
        return "Please ask something after 'chat:'."
    # Use a simple prompt (no tool instructions)
    prompt = f"You are Droid Code, a helpful AI assistant. Respond concisely and naturally to the user's message.\nUser: {query}\nAssistant:"
    response = ask_llm(prompt)
    if response:
        return response
    else:
        return "Sorry, I couldn't generate a response."

# ------------------- Persistent Background Agent -------------------
class BackgroundAgent(FileSystemEventHandler):
    def __init__(self, callback: Callable):
        self.callback = callback
        self.observer = None

    def on_modified(self, event):
        if not event.is_directory:
            self.callback(f"File modified: {event.src_path}")

    def start(self, path="."):
        if not HAS_WATCHDOG:
            logger.warning("Watchdog not installed, background agent disabled")
            return
        self.observer = Observer()
        self.observer.schedule(self, path, recursive=True)
        self.observer.start()
        logger.info(f"Background agent watching {path}")

    def stop(self):
        if self.observer:
            self.observer.stop()
            self.observer.join()

def start_background_agent():
    if not config.get("persistent_mode"):
        return
    def on_change(message):
        logger.info(f"Background event: {message}")
        send_notification("Droid Code Background", message)
    agent = BackgroundAgent(on_change)
    agent.start(".")
    return agent

# ------------------- Autonomous Query Processing -------------------
def process_query(user_query: str, interactive: bool = False) -> Optional[str]:
    """
    Process a single query autonomously, returning the final answer.
    If interactive is True, it will show output and ask for confirmation at each step.
    """
    current_url = None
    page_text = ""
    available_links = []
    final_answer = None
    step_count = 0
    max_steps = 20

    # Helper to handle special commands in interactive mode
    def handle_special(cmd: str) -> bool:
        if cmd.strip() == "!model":
            new_model = select_model()
            config.set("model", new_model)
            console.print(f"[green]Switched to model: {new_model}[/green]")
            return True
        elif cmd.strip() == "!memory":
            mem_list = memory_list()
            console.print(Panel(mem_list, title="Memory Contents", border_style="green"))
            return True
        # Manual tool invocation: !tool tool_name key=value ...
        elif cmd.strip().startswith("!tool"):
            parts = cmd.strip().split()
            if len(parts) < 2:
                console.print("[red]Usage: !tool tool_name [key=value ...][/red]")
                return True
            tool_name = parts[1]
            tool_args = {}
            for part in parts[2:]:
                if '=' in part:
                    key, val = part.split('=', 1)
                    # Try to convert numeric or bool
                    if val.lower() == 'true':
                        val = True
                    elif val.lower() == 'false':
                        val = False
                    elif val.isdigit():
                        val = int(val)
                    else:
                        try:
                            val = float(val)
                        except:
                            pass
                    tool_args[key] = val
                else:
                    # assume positional? not supported; ignore
                    pass
            # Execute the tool
            execute_tool(tool_name, tool_args, interactive=True)
            return True
        elif cmd.lower().startswith("chat:"):
            answer = chat_with_ai(cmd)
            console.print(Panel(Markdown(answer), title="Chat Response", border_style="cyan"))
            return True
        return False

    # If we're in interactive mode and the initial query is a special command, handle it directly
    if interactive and handle_special(user_query):
        return None

    while step_count < max_steps:
        step_count += 1

        # Check for special commands again inside the loop (for manual input after continue)
        if interactive and handle_special(user_query):
            user_query = Prompt.ask("[bold yellow]What do you want to do?[/bold yellow]")
            if user_query.lower() in ("exit", "quit"):
                break
            continue

        # If no web context and not a special command, start with search
        if current_url is None and not user_query.startswith("!"):
            if interactive:
                console.print(f"[green]Searching for: {user_query}[/green]")
            page_text = search_duckduckgo_lite(user_query)
            # Extract links from search results
            url = "https://lite.duckduckgo.com/lite/"
            params = {"q": user_query}
            try:
                resp = session.get(url, params=params, timeout=30)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                tables = soup.find_all("table")
                result_table = None
                for tbl in tables:
                    if tbl.find("a", href=True):
                        result_table = tbl
                        break
                if result_table:
                    available_links = []
                    rows = result_table.find_all("tr")
                    for row in rows:
                        link_cell = row.find("a")
                        if link_cell and link_cell.get("href"):
                            title = link_cell.get_text(strip=True)
                            href = link_cell["href"]
                            if any(phrase in title.lower() for phrase in ["ad", "sponsored", "promoted"]):
                                continue
                            if "y.js" in href or "ad_domain" in href:
                                continue
                            href = resolve_redirect(href)
                            available_links.append((title, href))
            except Exception as e:
                if interactive:
                    console.print(f"[red]Error extracting links: {e}[/red]")
                available_links = []
        elif current_url is not None:
            if interactive:
                console.print(f"[green]Fetching: {current_url}[/green]")
            page_text = fetch_page(current_url)
            if interactive:
                console.print("[dim]Extracting links from page...[/dim]")
            available_links = extract_links(page_text, current_url)

        # Show preview if interactive
        if interactive and page_text and page_text != "No results found." and not user_query.startswith("!"):
            preview = page_text[:1000] + "..." if len(page_text) > 1000 else page_text
            console.print(Panel(preview, title=f"Current page: {current_url if current_url else 'Search Results'}", border_style="blue"))

        if interactive:
            console.print("[yellow]AI is thinking...[/yellow]")
        decision = decide_next_action(user_query, current_url or "Search Results page", page_text, available_links)
        if interactive:
            console.print(f"[bold]AI decision:[/bold] {decision.get('reason', 'No reason')}")

        action = decision.get("action")
        if action == "stop":
            final_answer = decision.get("answer")
            if interactive:
                console.print("[green]Done![/green]")
                if final_answer:
                    console.print(Markdown(final_answer))
            break
        elif action == "search":
            new_query = decision.get("query")
            if not new_query:
                if interactive:
                    console.print("[red]No search query provided. Exiting.[/red]")
                break
            if interactive:
                console.print(f"[blue]Searching for: {new_query}[/blue]")
            current_url = None
            user_query = new_query
        elif action == "visit_link":
            url = decision.get("url")
            if not url:
                if interactive:
                    console.print("[red]No URL provided. Exiting.[/red]")
                break
            url = resolve_redirect(url)
            if current_url and not urlparse(url).netloc:
                url = urljoin(current_url, url)
            if interactive:
                console.print(f"[blue]Following link: {url}[/blue]")
            current_url = url
        elif action == "extract":
            answer = decision.get("answer")
            if answer:
                final_answer = answer
                if interactive:
                    console.print(Panel(Markdown(answer), title="Extracted Answer", border_style="green"))
            else:
                if interactive:
                    console.print("[red]No answer extracted. Exiting.[/red]")
            break
        elif action == "multi_edit":
            edits = decision.get("edits", [])
            for edit in edits:
                path = edit.get("path")
                content = edit.get("content", "")
                append = edit.get("append", False)
                if path:
                    result = write_file(path, content, append)
                    if interactive:
                        console.print(Panel(result, title=f"Write {path}", border_style="green"))
                else:
                    if interactive:
                        console.print("[red]Missing path in edit[/red]")
        elif action == "tool":
            tool_name = decision.get("tool_name")
            tool_args = decision.get("tool_args", {})
            if not tool_name:
                if interactive:
                    console.print("[red]No tool name provided.[/red]")
                break
            # Execute the tool using the helper
            execute_tool(tool_name, tool_args, interactive)
        else:
            if interactive:
                console.print(f"[red]Unknown action: {action}. Exiting.[/red]")
            break

        if interactive:
            if not Confirm.ask("[bold yellow]Continue with AI?[/bold yellow]", default=True):
                manual = Prompt.ask("Enter command (URL, !model, !memory, !tool, chat:, 'exit')")
                if manual.lower() == "exit":
                    break
                else:
                    user_query = manual
                    current_url = None
            else:
                pass

    if step_count >= max_steps:
        logger.warning("Reached max steps, stopping")
        final_answer = final_answer or "Process exceeded maximum steps."

    return final_answer

# ------------------- Flask API Server -------------------
def start_api_server(host: str = None, port: int = None):
    if not HAS_FLASK:
        console.print("[red]Flask not installed. Cannot start API server.[/red]")
        return
    host = host or config.get("api_host")
    port = port or config.get("api_port")
    app = Flask(__name__)

    @app.route('/chat', methods=['POST'])
    def chat_endpoint():
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing "query" field'}), 400
        query = data['query']
        answer = process_query(query, interactive=False)
        add_to_conversation('user', query)
        add_to_conversation('assistant', answer)
        return jsonify({'response': answer})

    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({'status': 'ok'})

    console.print(f"[green]Starting API server on http://{host}:{port}[/green]")
    app.run(host=host, port=port)

# ------------------- Main Interactive Loop -------------------
def interactive_main():
    console.print(Panel.fit("[bold cyan]Droid Code[/bold cyan] - AI-powered assistant with web, file, shell, Git, scheduling, subagents, dream memory, and more", style="bold"))

    try:
        subprocess.run(["ollama", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        console.print("[red]Ollama not found. Please install Ollama and make sure it's running.[/red]")
        sys.exit(1)

    model = config.get("model")
    if not model:
        model = select_model()
        config.set("model", model)
    else:
        console.print(f"[green]Using model from config: {model}[/green]")

    # Start background workers
    start_dream_worker()
    start_scheduler()
    background_agent = start_background_agent()

    console.print("\n[bold yellow]Enter your goal. You can ask me to browse the web, read/write files, run commands, manage Git, schedule tasks, spawn subagents, etc.[/bold yellow]")
    console.print("Type 'exit' to quit. During session, type '!model' to switch models, '!memory' to view memory, '!tool' to call a tool manually, or 'chat: ...' to have a conversation.\n")

    user_query = Prompt.ask("[bold yellow]What do you want to do?[/bold yellow]")
    while user_query.lower() not in ("exit", "quit"):
        result = process_query(user_query, interactive=True)
        if result:
            console.print(Panel(Markdown(result), title="Final Answer", border_style="green"))
        user_query = Prompt.ask("[bold yellow]What do you want to do?[/bold yellow]")

    # Cleanup
    stop_scheduler()
    if background_agent:
        background_agent.stop()
    console.print("[bold green]Goodbye![/bold green]")

# ------------------- Command-line Entry -------------------
def main():
    parser = argparse.ArgumentParser(description="Droid Code – AI OS Terminal Browser")
    parser.add_argument('--serve', action='store_true', help="Start API server instead of interactive mode")
    parser.add_argument('--host', default=None, help="Host for API server")
    parser.add_argument('--port', type=int, default=None, help="Port for API server")
    args = parser.parse_args()

    if args.serve:
        start_api_server(host=args.host, port=args.port)
    else:
        interactive_main()

if __name__ == "__main__":
    main()
