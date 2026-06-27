"""
agent.py — Coding Research Agent
Reads a GitHub repo and answers natural-language questions about its architecture.

Usage:
    python agent.py --repo owner/repo --question "What does this repo do?"
    python agent.py --repo fastapi/fastapi --question "How is routing implemented?"

Rules:
    - No LangChain / LangGraph — pure agent loop from scratch
    - Max 15 iterations
    - Files > 100 KB refused
    - Tool output capped at 8 KB
    - Every tool call logged with arguments
"""
import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from typing import Optional

from tools import TOOLS, TOOL_SCHEMAS

MAX_ITERATIONS = 15
MODEL          = "gemini-2.0-flash"
API_KEY_ENV    = "GEMINI_API_KEY"

SYSTEM_PROMPT = """You are a Coding Research Agent. Your job is to answer questions about GitHub repositories by exploring their code.

You have access to these tools:
- list_repo_structure: explore directory contents
- read_file: read a specific file (refuses files > 100 KB)
- search_code: search for patterns across the repo
- get_repo_info: get repo metadata
- list_recent_commits: see recent commit history

Strategy:
1. Start with get_repo_info to understand the project
2. Use list_repo_structure on root to see the layout
3. Read key files (README, main entry points, config files)
4. Search for specific patterns when needed
5. Synthesize findings into a clear answer

Rules:
- Never read files > 100 KB (the tool will refuse anyway)
- Always explore before answering — don't guess
- Cite specific files when making claims
- If you cannot find the answer after thorough exploration, say so honestly
- When you have enough information, provide your FINAL ANSWER clearly"""


def _gemini_request(api_key: str, messages: list, tools: list) -> dict:
    """Call Gemini API with tool support."""
    
    # Convert to Gemini format
    gemini_tools = [{
        "function_declarations": [
            {
                "name"       : t["name"],
                "description": t["description"],
                "parameters" : {
                    "type"      : "object",
                    "properties": {k: {"type": v.get("type", "string"), 
                                       "description": v.get("description", "")}
                                   for k, v in t["parameters"].items()},
                    "required"  : t.get("required", [])
                }
            }
            for t in tools
        ]
    }]

    # Convert messages to Gemini format
    gemini_messages = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        if isinstance(msg["content"], str):
            gemini_messages.append({
                "role"  : role,
                "parts" : [{"text": msg["content"]}]
            })
        elif isinstance(msg["content"], list):
            parts = []
            for part in msg["content"]:
                if part.get("type") == "text":
                    parts.append({"text": part["text"]})
                elif part.get("type") == "tool_use":
                    parts.append({
                        "functionCall": {
                            "name": part["name"],
                            "args": part["input"]
                        }
                    })
                elif part.get("type") == "tool_result":
                    parts.append({
                        "functionResponse": {
                            "name"    : part.get("tool_use_id", "tool"),
                            "response": {"result": part["content"]}
                        }
                    })
            gemini_messages.append({"role": role, "parts": parts})

    payload = json.dumps({
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents"          : gemini_messages,
        "tools"             : gemini_tools,
        "generationConfig"  : {"temperature": 0.1, "maxOutputTokens": 2048}
    }).encode()

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={api_key}"
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _parse_gemini_response(response: dict) -> tuple[str, list]:
    """Parse Gemini response into (text, tool_calls)."""
    candidate = response.get("candidates", [{}])[0]
    content   = candidate.get("content", {})
    parts     = content.get("parts", [])

    text       = ""
    tool_calls = []

    for part in parts:
        if "text" in part:
            text += part["text"]
        if "functionCall" in part:
            fc = part["functionCall"]
            tool_calls.append({
                "name" : fc["name"],
                "args" : fc.get("args", {})
            })

    return text.strip(), tool_calls


def _execute_tool(tool_name: str, args: dict, iteration: int) -> str:
    """Execute a tool and return result string. Logs every call."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n  🔧 [{timestamp}] Tool call #{iteration}: {tool_name}")
    print(f"     Args: {json.dumps(args, indent=8)}")

    if tool_name not in TOOLS:
        result = f"Unknown tool: {tool_name}"
        print(f"     ❌ {result}")
        return result

    tool_fn = TOOLS[tool_name]
    try:
        result, error = tool_fn(**args)
        if error:
            print(f"     ❌ Error: {error}")
            return f"Tool error: {error}"
        print(f"     ✅ Success ({len(result)} chars)")
        return result
    except TypeError as e:
        err = f"Invalid arguments for {tool_name}: {e}"
        print(f"     ❌ {err}")
        return err
    except Exception as e:
        err = f"Tool execution failed: {e}"
        print(f"     ❌ {err}")
        return err


def run_agent(repo: str, question: str, api_key: str) -> str:
    """
    Main agent loop.
    - Max 15 iterations
    - Logs every tool call
    - Returns final answer
    """
    # Parse owner/repo
    parts = repo.strip("/").split("/")
    if len(parts) != 2:
        return f"Invalid repo format '{repo}'. Use 'owner/repo' e.g. 'fastapi/fastapi'"
    owner, repo_name = parts

    print(f"\n{'='*60}")
    print(f"🤖 Coding Research Agent")
    print(f"{'='*60}")
    print(f"📦 Repo     : {owner}/{repo_name}")
    print(f"❓ Question : {question}")
    print(f"{'='*60}")

    # Initial user message
    messages = [{
        "role"   : "user",
        "content": f"Repo: {owner}/{repo_name}\n\nQuestion: {question}\n\nPlease research this repo and answer the question."
    }]

    iteration    = 0
    final_answer = None

    while iteration < MAX_ITERATIONS:
        iteration += 1
        print(f"\n📍 Iteration {iteration}/{MAX_ITERATIONS}")

        # Call Gemini
        try:
            response   = _gemini_request(api_key, messages, TOOL_SCHEMAS)
            text, tool_calls = _parse_gemini_response(response)
        except urllib.error.HTTPError as e:
            error_body = e.read().decode()
            return f"Gemini API error {e.code}: {error_body}"
        except Exception as e:
            return f"API call failed: {e}"

        # Check for final answer
        if text and not tool_calls:
            print(f"\n✅ Agent provided final answer (iteration {iteration})")
            final_answer = text
            break

        if text:
            print(f"   💭 Agent thinking: {text[:150]}...")

        # Execute tool calls
        if tool_calls:
            # Add assistant message with tool calls
            assistant_parts = []
            if text:
                assistant_parts.append({"type": "text", "text": text})
            for tc in tool_calls:
                assistant_parts.append({
                    "type" : "tool_use",
                    "name" : tc["name"],
                    "input": tc["args"],
                    "id"   : f"call_{iteration}"
                })
            messages.append({"role": "assistant", "content": assistant_parts})

            # Execute each tool and collect results
            tool_results = []
            for tc in tool_calls:
                result = _execute_tool(tc["name"], tc["args"], iteration)
                tool_results.append({
                    "type"       : "tool_result",
                    "tool_use_id": f"call_{iteration}",
                    "content"    : result
                })

            # Add tool results as user message
            messages.append({"role": "user", "content": tool_results})

        else:
            # No tool calls, no text — something went wrong
            print("   ⚠️  No tool calls or text in response")
            break

    if not final_answer:
        final_answer = f"Agent reached max iterations ({MAX_ITERATIONS}) without a final answer. Last response: {text}"

    print(f"\n{'='*60}")
    print("📋 FINAL ANSWER")
    print(f"{'='*60}")
    print(final_answer)
    print(f"{'='*60}\n")

    return final_answer


def main():
    parser = argparse.ArgumentParser(
        description="Coding Research Agent — answers questions about GitHub repos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent.py --repo fastapi/fastapi --question "How is routing implemented?"
  python agent.py --repo pallets/flask --question "What is the main entry point?"
  python agent.py --repo psf/requests --question "How does the session handling work?"
        """
    )
    parser.add_argument("--repo",     required=True, help="GitHub repo in owner/repo format")
    parser.add_argument("--question", required=True, help="Natural language question about the repo")
    parser.add_argument("--api-key",  help="Gemini API key (or set GEMINI_API_KEY env var)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get(API_KEY_ENV)
    if not api_key:
        print(f"❌ No API key found. Set {API_KEY_ENV} environment variable or use --api-key")
        sys.exit(1)

    answer = run_agent(args.repo, args.question, api_key)
    return answer


if __name__ == "__main__":
    main()
