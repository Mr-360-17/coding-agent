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
from datetime import datetime

from tools import TOOLS, TOOL_SCHEMAS

MAX_ITERATIONS = 15
API_KEY_ENV    = "GEMINI_API_KEY"

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
5. Give a clear FINAL ANSWER citing specific files"""


def _call_gemini(api_key: str, messages: list) -> tuple[str, list]:
    """Call Gemini API using google-genai library."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    # Build function declarations
    func_decls = []
    for schema in TOOL_SCHEMAS:
        props = {}
        for name, info in schema["parameters"].items():
            t = info.get("type", "string").upper()
            props[name] = types.Schema(type=t, description=info.get("description", ""))

        func_decls.append(types.FunctionDeclaration(
            name=schema["name"],
            description=schema["description"],
            parameters=types.Schema(
                type="OBJECT",
                properties=props,
                required=schema.get("required", [])
            )
        ))

    tools = [types.Tool(function_declarations=func_decls)]
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=tools,
        temperature=0.1
    )

    # Convert messages
    contents = []
    for msg in messages:
        role    = "user" if msg["role"] == "user" else "model"
        content = msg["content"]
        if isinstance(content, str):
            contents.append(types.Content(role=role, parts=[types.Part(text=content)]))
        else:
            contents.append(types.Content(role=role, parts=[types.Part(text=str(content))]))

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents,
        config=config
    )

    text       = ""
    tool_calls = []

    for part in response.candidates[0].content.parts:
        if hasattr(part, "text") and part.text:
            text += part.text
        if hasattr(part, "function_call") and part.function_call:
            fc = part.function_call
            tool_calls.append({
                "name": fc.name,
                "args": dict(fc.args) if fc.args else {}
            })

    return text.strip(), tool_calls


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

    messages = [{
        "role"   : "user",
        "content": f"Repo: {owner}/{repo_name}\n\nQuestion: {question}\n\nResearch this repo and answer the question."
    }]

    iteration    = 0
    final_answer = None
    last_text    = ""

    while iteration < MAX_ITERATIONS:
        iteration += 1
        print(f"\n📍 Iteration {iteration}/{MAX_ITERATIONS}")

        try:
            text, tool_calls = _call_gemini(api_key, messages)
        except Exception as e:
            return f"API error: {e}"

        last_text = text

        if text and not tool_calls:
            final_answer = text
            break

        if text:
            print(f"   💭 {text[:120]}...")

        if tool_calls:
            messages.append({"role": "model", "content": text or "Calling tools..."})
            results = ""
            for tc in tool_calls:
                result = _execute_tool(tc["name"], tc["args"], iteration)
                results += f"\n[{tc['name']}]:\n{result}\n"
            messages.append({"role": "user", "content": results})
        else:
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
