"""
tools/github_tools.py — GitHub API tools for the coding agent.
All tools return (result: str, error: str | None).
Files > 100 KB are refused. Outputs are capped at 8 KB.
"""
import base64
import json
import urllib.request
import urllib.error
from typing import Optional

MAX_FILE_BYTES  = 100 * 1024   # 100 KB — refuse larger files
MAX_OUTPUT_BYTES = 8  * 1024   #   8 KB — truncate tool output

HEADERS = {"Accept": "application/vnd.github+json",
           "User-Agent": "coding-research-agent/1.0"}


def _get(url: str) -> tuple[dict | list | None, Optional[str]]:
    """Raw GitHub API GET. Returns (data, error)."""
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read()), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.reason} — {url}"
    except Exception as e:
        return None, f"Request failed: {e}"


def _cap(text: str) -> str:
    """Truncate output to MAX_OUTPUT_BYTES with explicit notice."""
    encoded = text.encode("utf-8")
    if len(encoded) <= MAX_OUTPUT_BYTES:
        return text
    truncated = encoded[:MAX_OUTPUT_BYTES].decode("utf-8", errors="ignore")
    return truncated + f"\n\n[OUTPUT TRUNCATED — showed {MAX_OUTPUT_BYTES//1024} KB of {len(encoded)//1024} KB total]"


# ── Tool 1: List repo structure ───────────────────────────────────────────────
def list_repo_structure(owner: str, repo: str, path: str = "") -> tuple[str, Optional[str]]:
    """
    List files and directories at a given path in the repo.
    Returns a tree-style text listing.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    data, err = _get(url)
    if err:
        return "", err
    if not isinstance(data, list):
        return "", f"Expected directory listing, got: {type(data)}"

    lines = []
    for item in sorted(data, key=lambda x: (x["type"] != "dir", x["name"])):
        icon = "📁" if item["type"] == "dir" else "📄"
        size = f" ({item.get('size', 0):,} bytes)" if item["type"] == "file" else ""
        lines.append(f"{icon} {item['name']}{size}")

    result = f"Contents of /{path or '(root)'}:\n" + "\n".join(lines)
    return _cap(result), None


# ── Tool 2: Read file ─────────────────────────────────────────────────────────
def read_file(owner: str, repo: str, path: str) -> tuple[str, Optional[str]]:
    """
    Read a file from the repo. Refuses files > 100 KB.
    Returns file content as text.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    data, err = _get(url)
    if err:
        return "", err
    if isinstance(data, list):
        return "", f"{path} is a directory, not a file. Use list_repo_structure."

    size = data.get("size", 0)
    if size > MAX_FILE_BYTES:
        return "", (f"File too large: {size:,} bytes > {MAX_FILE_BYTES:,} bytes limit. "
                    f"Refusing to read {path} to avoid burning tokens on large files.")

    content_b64 = data.get("content", "")
    try:
        content = base64.b64decode(content_b64).decode("utf-8", errors="replace")
    except Exception as e:
        return "", f"Failed to decode file content: {e}"

    result = f"File: {path} ({size:,} bytes)\n{'='*50}\n{content}"
    return _cap(result), None


# ── Tool 3: Search code ───────────────────────────────────────────────────────
def search_code(owner: str, repo: str, query: str) -> tuple[str, Optional[str]]:
    """
    Search for a string/pattern in the repo using GitHub code search.
    Returns matching file names and snippets.
    """
    url = f"https://api.github.com/search/code?q={urllib.parse.quote(query)}+repo:{owner}/{repo}&per_page=10"
    
    import urllib.parse
    url = f"https://api.github.com/search/code?q={urllib.parse.quote(query)}+repo:{owner}/{repo}&per_page=10"
    
    data, err = _get(url)
    if err:
        return "", err

    items = data.get("items", [])
    if not items:
        return f"No results found for '{query}' in {owner}/{repo}", None

    lines = [f"Search results for '{query}' in {owner}/{repo} ({data.get('total_count', 0)} total matches):"]
    for item in items[:10]:
        lines.append(f"\n📄 {item['path']}")

    return _cap("\n".join(lines)), None


# ── Tool 4: Get repo metadata ─────────────────────────────────────────────────
def get_repo_info(owner: str, repo: str) -> tuple[str, Optional[str]]:
    """
    Get basic metadata about the repo: description, language, stars, topics, etc.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}"
    data, err = _get(url)
    if err:
        return "", err

    info = {
        "name"            : data.get("full_name"),
        "description"     : data.get("description"),
        "language"        : data.get("language"),
        "stars"           : data.get("stargazers_count"),
        "forks"           : data.get("forks_count"),
        "open_issues"     : data.get("open_issues_count"),
        "topics"          : data.get("topics", []),
        "default_branch"  : data.get("default_branch"),
        "created_at"      : data.get("created_at"),
        "updated_at"      : data.get("updated_at"),
        "license"         : data.get("license", {}).get("name") if data.get("license") else None,
        "size_kb"         : data.get("size"),
    }

    lines = ["Repository Info:"]
    for k, v in info.items():
        if v is not None:
            lines.append(f"  {k}: {v}")

    return _cap("\n".join(lines)), None


# ── Tool 5: List commits ──────────────────────────────────────────────────────
def list_recent_commits(owner: str, repo: str, limit: int = 10) -> tuple[str, Optional[str]]:
    """
    List the most recent commits with author and message.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/commits?per_page={min(limit, 20)}"
    data, err = _get(url)
    if err:
        return "", err

    lines = [f"Recent commits for {owner}/{repo}:"]
    for commit in data:
        sha     = commit["sha"][:7]
        message = commit["commit"]["message"].split("\n")[0][:80]
        author  = commit["commit"]["author"]["name"]
        date    = commit["commit"]["author"]["date"][:10]
        lines.append(f"  [{sha}] {date} {author}: {message}")

    return _cap("\n".join(lines)), None


# ── Tool registry ─────────────────────────────────────────────────────────────
TOOLS = {
    "list_repo_structure" : list_repo_structure,
    "read_file"           : read_file,
    "search_code"         : search_code,
    "get_repo_info"       : get_repo_info,
    "list_recent_commits" : list_recent_commits,
}

TOOL_SCHEMAS = [
    {
        "name"       : "list_repo_structure",
        "description": "List files and directories at a path in the GitHub repo. Use this first to explore the structure.",
        "parameters" : {
            "owner": {"type": "string", "description": "GitHub username or org"},
            "repo" : {"type": "string", "description": "Repository name"},
            "path" : {"type": "string", "description": "Directory path (empty string for root)", "default": ""},
        },
        "required"   : ["owner", "repo"],
    },
    {
        "name"       : "read_file",
        "description": "Read a file from the repo. Refuses files > 100 KB. Returns file content.",
        "parameters" : {
            "owner": {"type": "string", "description": "GitHub username or org"},
            "repo" : {"type": "string", "description": "Repository name"},
            "path" : {"type": "string", "description": "File path from repo root (e.g. 'src/main.py')"},
        },
        "required"   : ["owner", "repo", "path"],
    },
    {
        "name"       : "search_code",
        "description": "Search for a string or pattern across all files in the repo.",
        "parameters" : {
            "owner": {"type": "string", "description": "GitHub username or org"},
            "repo" : {"type": "string", "description": "Repository name"},
            "query": {"type": "string", "description": "Search query string"},
        },
        "required"   : ["owner", "repo", "query"],
    },
    {
        "name"       : "get_repo_info",
        "description": "Get metadata about the repo: description, language, stars, topics, license.",
        "parameters" : {
            "owner": {"type": "string", "description": "GitHub username or org"},
            "repo" : {"type": "string", "description": "Repository name"},
        },
        "required"   : ["owner", "repo"],
    },
    {
        "name"       : "list_recent_commits",
        "description": "List the most recent commits with author, date, and message.",
        "parameters" : {
            "owner": {"type": "string", "description": "GitHub username or org"},
            "repo" : {"type": "string", "description": "Repository name"},
            "limit": {"type": "integer", "description": "Number of commits to fetch (max 20)", "default": 10},
        },
        "required"   : ["owner", "repo"],
    },
]
