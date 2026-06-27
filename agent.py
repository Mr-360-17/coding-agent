"""
agent.py — Coding Research Agent
Reads a GitHub repo and answers natural-language questions about its architecture.

Usage:
    python agent.py --repo owner/repo --question "What does this repo do?"
    python agent.py --repo fastapi/fastapi --question "How is routing implemented?"
"""
import argparse
import json
import os
import sys
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime
from typing import Optional

from tools import TOOLS, TOOL_SCHEMAS

MAX_ITERATIONS = 15
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
- Never read files > 100 KB
- Always explore before answering
- Cite specific files when making claims
- When you have enough information, provide your FINAL ANSWER"""


def _call_gemini(api_key: str, conversation: list) -> tuple[str, list]:
    """Call Gemini API using google-generativeai library."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)

    # Build tools for Gemini
    tools = []
    for schema in TOOL_SCHEMAS:
        props = {}
        for param_name, param_info in schema["parameters"].items():
            props[param_name] = genai.protos.Schema(
                type=genai.protos.Type[param_info.get("type", "string").upper()],
                description=param_info.get("description", "")
            )

        func_decl = genai.protos.FunctionDeclaration(
            name=schema["name"],
            description=schema["description"],
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties=props,
                required=schema.get("required", [])
            )
        )
        tools.append(func_decl)

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=SYSTEM_PROMPT,
        tools=[genai.protos.Tool(function_declarations=tools)]
    )

    # Convert conversation to Gemini format
    gemini_history = []
    for msg in conversation[:-1]:
        gemini_history.append({
            "role" : "user" if msg["role"] == "user" else "model",
            "parts": [msg["content"]] if isinstance(msg["content"], str) else msg["content"]
        })

    chat    = model.start_chat(history=gemini_history)
    last    = conversation[-1]
    content = last["content"] if isinstance(last["content"], str) else last["content"]
    response = chat.send_message(content)

    # Parse response
    text       = ""
    tool_calls = []

    for part in response.parts:
        if hasattr(part, "text") and part.text:
            text += part.text
        if hasattr(part, "function_call") and part.function_call.name:
            fc = part.function_call
            tool_calls.append({
                "name": fc.name,
                "args": dict(fc.args)
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
    except Exception as e:
        err = f"Tool execution failed: {e}"
        print(f"     ❌ {err}")
        return err


def run_agent(repo: str, question: str, api_key: str) -> str:
    """Main agent loop."""
    parts = repo.strip("/").split("/")
    if len(parts) != 2:
        return f"Invalid repo format '{repo}'. Use 'owner/repo'"
    owner, repo_name = parts

    print(f"\n{'='*60}")
    print(f"🤖 Coding Research Agent")
    print(f"{'='*60}")
    print(f"📦 Repo     : {owner}/{repo_name}")
    print(f"❓ Question : {question}")
    print(f"{'='*60}")

    conversation = [{
        "role"   : "user",
        "content": f"Repo: {owner}/{repo_name}\n\nQuestion: {question}\n\nPlease research this repo thoroughly and answer the question."
    }]

    iteration    = 0
    final_answer = None

    while iteration < MAX_ITERATIONS:
        iteration += 1
        print(f"\n📍 Iteration {iteration}/{MAX_ITERATIONS}")

        try:
            text, tool_calls = _call_gemini(api_key, conversation)
        except Exception as e:
            return f"API call failed: {e}"

        if text and not tool_calls:
            print(f"\n✅ Agent provided final answer (iteration {iteration})")
            final_answer = text
            break

        if text:
            print(f"   💭 Thinking: {text[:150]}...")

        if tool_calls:
            # Add assistant turn
            conversation.append({"role": "model", "content": text or "Using tools..."})

            # Execute tools and collect results
            results_text = ""
            for tc in tool_calls:
                result = _execute_tool(tc["name"], tc["args"], iteration)
                results_text += f"\n[{tc['name']} result]:\n{result}\n"

            # Add tool results as user turn
            conversation.append({"role": "user", "content": results_text})
        else:
            print("   ⚠️  No tool calls or text")
            break

    if not final_answer:
        final_answer = f"Agent reached max iterations ({MAX_ITERATIONS}). Last response: {text}"

    print(f"\n{'='*60}")
    print("📋 FINAL ANSWER")
    print(f"{'='*60}")
    print(final_answer)
    print(f"{'='*60}\n")

    return final_answer


def main():
    parser = argparse.ArgumentParser(description="Coding Research Agent")
    parser.add_argument("--repo",     required=True, help="GitHub repo in owner/repo format")
    parser.add_argument("--question", required=True, help="Natural language question about the repo")
    parser.add_argument("--api-key",  help="Gemini API key (or set GEMINI_API_KEY env var)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get(API_KEY_ENV)
    if not api_key:
        print(f"❌ No API key. Set {API_KEY_ENV} env var or use --api-key")
        sys.exit(1)

    run_agent(args.repo, args.question, api_key)


if __name__ == "__main__":
    main()
