"""
agent.py — Coding Research Agent
Reads a GitHub repo and answers natural-language questions about its architecture.

Usage:
    python agent.py --repo owner/repo --question "What does this repo do?"
"""
import argparse
import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime

from tools import TOOLS, TOOL_SCHEMAS

MAX_ITERATIONS = 15
API_KEY_ENV    = "GROQ_API_KEY"
MODEL          = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a Coding Research Agent. Answer questions about GitHub repositories by exploring their code.

Tools available:
- list_repo_structure: explore directory contents
- read_file: read a file (refuses files > 100 KB)
- search_code: search for patterns
- get_repo_info: get repo metadata
- list_recent_commits: see recent commits

Strategy:
1. Start with get_repo_info
2. Use list_repo_structure on root
3. Read key files (README, entry points)
4. Search for specific patterns
5. Give a clear FINAL ANSWER citing specific files

When you have enough info, just answer directly without calling any tools."""


def _build_tools_prompt() -> str:
    """Build a text description of tools for the prompt."""
    lines = ["\nAvailable tools (call them by responding with JSON in this exact format):"]
    lines.append('{"tool": "tool_name", "args": {"param1": "value1"}}')
    lines.append("\nTool definitions:")
    for t in TOOL_SCHEMAS:
        params = ", ".join(f"{k}: {v.get('type','string')}" 
                          for k, v in t["parameters"].items())
        lines.append(f"- {t['name']}({params}): {t['description']}")
    return "\n".join(lines)


def _call_groq(api_key: str, messages: list) -> tuple[str, list]:
    """Call Groq API and parse response for tool calls or text."""
    payload = json.dumps({
        "model"      : MODEL,
        "messages"   : messages,
        "temperature": 0.1,
        "max_tokens" : 2048,
    }).encode()

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type" : "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise Exception(f"Groq API error {e.code}: {e.read().decode()}")

    text = data["choices"][0]["message"]["content"] or ""

    # Try to extract tool call from JSON in response
    tool_calls = []
    stripped = text.strip()

    # Look for JSON tool call
    if stripped.startswith("{") and '"tool"' in stripped:
        try:
            # Extract just the JSON part
            end = stripped.find("}") + 1
            json_str = stripped[:end]
            parsed = json.loads(json_str)
            if "tool" in parsed and "args" in parsed:
                tool_calls.append({
                    "name": parsed["tool"],
                    "args": parsed["args"]
                })
                text = stripped[end:].strip()
        except json.JSONDecodeError:
            pass

    return text, tool_calls


def _execute_tool(tool_name: str, args: dict, iteration: int) -> str:
    """Execute a tool and log every call."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n  🔧 [{timestamp}] Tool call #{iteration}: {tool_name}")
    print(f"     Args: {json.dumps(args)}")

    if tool_name not in TOOLS:
        print(f"     ❌ Unknown tool")
        return f"Unknown tool: {tool_name}"

    try:
        result, error = TOOLS[tool_name](**args)
        if error:
            print(f"     ❌ {error}")
            return f"Error: {error}"
        print(f"     ✅ {len(result)} chars returned")
        return result
    except Exception as e:
        print(f"     ❌ {e}")
        return f"Tool failed: {e}"


def run_agent(repo: str, question: str, api_key: str) -> str:
    """Main agent loop — max 15 iterations."""
    parts = repo.strip("/").split("/")
    if len(parts) != 2:
        return f"Invalid repo format. Use 'owner/repo'"
    owner, repo_name = parts

    print(f"\n{'='*60}")
    print(f"🤖 Coding Research Agent")
    print(f"{'='*60}")
    print(f"📦 Repo     : {owner}/{repo_name}")
    print(f"❓ Question : {question}")
    print(f"{'='*60}")

    tools_prompt = _build_tools_prompt()

    messages = [
        {
            "role"   : "system",
            "content": SYSTEM_PROMPT + tools_prompt
        },
        {
            "role"   : "user",
            "content": f"Repo: {owner}/{repo_name}\n\nQuestion: {question}\n\nStart by calling get_repo_info, then explore and answer."
        }
    ]

    iteration    = 0
    final_answer = None
    last_text    = ""

    while iteration < MAX_ITERATIONS:
        iteration += 1
        print(f"\n📍 Iteration {iteration}/{MAX_ITERATIONS}")

        try:
            text, tool_calls = _call_groq(api_key, messages)
        except Exception as e:
            return f"API error: {e}"

        last_text = text

        if text and not tool_calls:
            final_answer = text
            break

        if text:
            print(f"   💭 {text[:120]}...")

        if tool_calls:
            messages.append({"role": "assistant", "content": json.dumps({"tool": tool_calls[0]["name"], "args": tool_calls[0]["args"]})})
            result = _execute_tool(tool_calls[0]["name"], tool_calls[0]["args"], iteration)
            messages.append({"role": "user", "content": f"Tool result:\n{result}\n\nContinue researching or provide your final answer."})
        else:
            if text:
                final_answer = text
            break

    if not final_answer:
        final_answer = last_text or f"No answer after {MAX_ITERATIONS} iterations."

    print(f"\n{'='*60}")
    print("📋 FINAL ANSWER")
    print(f"{'='*60}")
    print(final_answer)
    print(f"{'='*60}\n")
    return final_answer


def main():
    parser = argparse.ArgumentParser(description="Coding Research Agent")
    parser.add_argument("--repo",     required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--api-key",  default=None)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get(API_KEY_ENV)
    if not api_key:
        print(f"❌ Set {API_KEY_ENV} or use --api-key")
        sys.exit(1)

    run_agent(args.repo, args.question, api_key)

if __name__ == "__main__":
    main()
