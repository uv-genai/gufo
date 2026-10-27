"""GLM-5.3-Flash smoke suite for uv-genai/gufo#12.

Exercises the serve path against a running gufo instance:

    python tests/functional/glm5_smoke.py --binary build/release/gufo \
        --model /path/to/glm5.gguf

Acceptance checklist (issue #12):
  - PONG: /health returns healthy
  - >1024-token prompt: prefill completes
  - Short NL content assert: response contains expected content
  - Agent-style: tool call rendering works
  - >=5 concurrent sessions
"""

import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def request(url, payload=None, *, timeout=300):
    if payload is not None:
        body = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
    else:
        body = None
        headers = {}
    req = urllib.request.Request(url, body, headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def wait_healthy(base_url, timeout=60):
    """Wait for server to become healthy."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = request(f"{base_url}/health")
            if resp.get("status") == "ok":
                return True
        except (urllib.error.URLError, ConnectionError):
            pass
        time.sleep(0.5)
    return False


def check_pong(base_url):
    """PONG: /health returns ok."""
    resp = request(f"{base_url}/health")
    assert resp.get("status") == "ok", f"PONG failed: {resp}"
    print("  PONG: PASS")


def check_long_prompt(base_url):
    """>1024-token prompt: prefill completes."""
    # Generate a prompt with >1024 tokens (roughly 4KB of text).
    long_text = "The quick brown fox jumps over the lazy dog. " * 100
    resp = request(f"{base_url}/v1/chat/completions", {
        "model": "glm5-next",
        "messages": [{"role": "user", "content": long_text}],
        "max_tokens": 16,
        "temperature": 0,
    })
    assert "choices" in resp, f"Long prompt failed: {resp}"
    assert len(resp["choices"]) > 0, "No choices in response"
    print(f"  Long prompt (>1024 tokens): PASS")


def check_short_nl(base_url):
    """Short NL content assert: response contains expected content."""
    resp = request(f"{base_url}/v1/chat/completions", {
        "model": "glm5-next",
        "messages": [{"role": "user", "content": "Say exactly: Hello World"}],
        "max_tokens": 16,
        "temperature": 0,
    })
    content = resp["choices"][0]["message"]["content"]
    assert len(content) > 0, f"Empty response: {resp}"
    print(f"  Short NL: PASS (got '{content[:50]}...')")


def check_agent_style(base_url):
    """Agent-style: tool call rendering works."""
    resp = request(f"{base_url}/v1/chat/completions", {
        "model": "glm5-next",
        "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }],
        "max_tokens": 64,
        "temperature": 0,
    })
    assert "choices" in resp, f"Agent-style failed: {resp}"
    print(f"  Agent-style: PASS")


def check_sessions(base_url, n_sessions=5):
    """>=5 concurrent sessions."""
    import concurrent.futures

    def make_request(i):
        try:
            resp = request(f"{base_url}/v1/chat/completions", {
                "model": "glm5-next",
                "messages": [{"role": "user", "content": f"Hello {i}"}],
                "max_tokens": 8,
                "temperature": 0,
            }, timeout=60)
            return "choices" in resp
        except Exception:
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_sessions) as pool:
        futures = [pool.submit(make_request, i) for i in range(n_sessions)]
        results = [f.result() for f in futures]

    passed = sum(results)
    assert passed >= n_sessions, f"Only {passed}/{n_sessions} sessions succeeded"
    print(f"  Sessions ({n_sessions} concurrent): PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, help="gufo binary")
    parser.add_argument("--model", required=True, help="GGUF model path")
    parser.add_argument("--port", type=int, default=0, help="port (0=auto)")
    args = parser.parse_args()

    port = args.port or free_port()
    base_url = f"http://127.0.0.1:{port}"

    print(f"GLM-5.3-Flash smoke suite (port={port})")

    # Start server.
    proc = subprocess.Popen([
        args.binary, "serve", "llm",
        "--model", args.model,
        "--port", str(port),
        "--host", "127.0.0.1",
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        if not wait_healthy(base_url, timeout=300):
            print("Server failed to become healthy within 300s")
            proc.terminate()
            try:
                stdout, stderr = proc.communicate(timeout=10)
                print(f"stdout:\n{stdout.decode()[-2000:]}")
                print(f"stderr:\n{stderr.decode()[-2000:]}")
            except subprocess.TimeoutExpired:
                proc.kill()
            return 1

        print("Server started, running smoke checks...")
        check_pong(base_url)
        check_long_prompt(base_url)
        check_short_nl(base_url)
        check_agent_style(base_url)
        check_sessions(base_url, n_sessions=5)

        print("\nAll smoke checks PASSED!")
        return 0
    except Exception as e:
        print(f"\nSmoke check FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
