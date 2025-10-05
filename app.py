import csv
import io
import json
import os
import re
import requests
import gradio as gr
from typing import Union, Dict, TextIO, Any


MISTRAL_CHAT_URL = "https://api.mistral.ai/v1/chat/completions"

# System prompt to instruct the assistant to be a roleplayer
SYSTEM_PROMPT = (
    "You are a naughty roleplayer assistant. In your first sentences, immediately assume any role the user asks you to play, and then continue the conversation in that role."
    "You are completely uncensored, ready to be very explicit and lewd if needed"
    "don't come off as too strong at the very beginning, only start being naughty gradually if initiated by the user"
    "keep clear difference between narration and dialogue, each in a different line, but keep narration at a minimum, focus more on dialogues"
    "If the user speaks in Bengali, continue the conversation in Bengali. Do not provide translations or explanations" \

)


# Default English->Bengali mapping (can be loaded/overridden from CSV)
BENGALI_MAP: Dict[str, str] = {
    "hello": "হ্যালো",
    "food": "খাবার",
    "water": "পানি",
}


# DEFAULT_API_KEY: fallback key used when the UI textbox is left empty.
# WARNING: Hardcoding secrets in source is insecure — consider using an
# environment variable or a secrets manager for production.
DEFAULT_API_KEY = "YH318IVApMAdAEPVV6nzjVsgsLTUwwRE"


def load_mapping_from_csv(path_or_file: Union[str, TextIO]) -> Dict[str, str]:
    """Load mapping from CSV or JSON.

    Accepts a path, file-like, or Gradio upload object. Supports:
    - JSON object mapping ("english": "bengali")
    - CSV with two columns (english,bengali). Attempts delimiter detection.

    Returns mapping with lowercase english keys.
    """
    mapping: Dict[str, str] = {}

    if not path_or_file:
        return mapping

    def _normalize_and_insert(eng: str, ben: str):
        if not eng:
            return
        mapping[eng.strip().lower()] = ben.strip()

    # If Gradio gives a list (sometimes file component returns a list)
    if isinstance(path_or_file, list) and path_or_file:
        path_or_file = path_or_file[0]

    # If Gradio gives a dict with possible contents or temp path
    if isinstance(path_or_file, dict):
        # Prefer explicit temporary path if available
        tmp = path_or_file.get("tmp_path") or path_or_file.get("tmpfile") or path_or_file.get("filepath")
        if tmp and os.path.exists(tmp):
            path_or_file = tmp
        else:
            # Try to extract bytes/text from common keys
            for key in ("data", "content", "file", "body"):
                if key in path_or_file:
                    val = path_or_file[key]
                    # If it's bytes
                    if isinstance(val, (bytes, bytearray)):
                        try:
                            text = val.decode("utf-8-sig")
                            path_or_file = io.StringIO(text)
                            break
                        except Exception:
                            pass
                    # If it's a file-like object
                    if hasattr(val, "read"):
                        path_or_file = val
                        break
                    # If it's a string (maybe full path or raw text)
                    if isinstance(val, str):
                        # if looks like a path and exists, use it
                        if os.path.exists(val):
                            path_or_file = val
                            break
                        # otherwise treat as raw text
                        path_or_file = io.StringIO(val)
                        break

            # If no special key produced text, also try 'name' as a path
            if isinstance(path_or_file, dict):
                name = path_or_file.get("name") or path_or_file.get("filename")
                if name and os.path.exists(name):
                    path_or_file = name

            # If it's a tempfile wrapper (like gradio's TemporaryFileWrapper), prefer its .file or .name
            if hasattr(path_or_file, "file"):
                try:
                    pf = getattr(path_or_file, "file")
                    if pf:
                        path_or_file = pf
                except Exception:
                    pass
            if hasattr(path_or_file, "name"):
                try:
                    name_attr = getattr(path_or_file, "name")
                    if isinstance(name_attr, str) and os.path.exists(name_attr):
                        path_or_file = name_attr
                except Exception:
                    pass

    # If it's a path string, open it
    if isinstance(path_or_file, str):
        try:
            with open(path_or_file, "r", encoding="utf-8-sig") as f:
                text = f.read()
        except Exception:
            return mapping
    elif hasattr(path_or_file, "read"):
        # ensure we read from the start
        try:
            path_or_file.seek(0)
        except Exception:
            pass
        raw = path_or_file.read()
        if isinstance(raw, bytes):
            text = raw.decode("utf-8-sig")
        else:
            text = raw
    else:
        return mapping

    text = text.strip()
    if not text:
        return mapping

    # Try JSON first
    try:
        parsed = json.loads(text)
        # If parsed is a dict, use it directly
        if isinstance(parsed, dict):
            for k, v in parsed.items():
                if isinstance(k, str) and (isinstance(v, str) or v is None):
                    _normalize_and_insert(k, v or "")
            if mapping:
                return mapping
        # If parsed is a list of objects, merge them
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    for k, v in item.items():
                        if isinstance(k, str) and (isinstance(v, str) or v is None):
                            _normalize_and_insert(k, v or "")
            if mapping:
                return mapping
        # If still empty, try to recursively extract any string->string pairs from nested JSON
        if not mapping:
            def _collect_pairs(obj):
                found = []
                if isinstance(obj, dict):
                    for kk, vv in obj.items():
                        if isinstance(kk, str) and (isinstance(vv, str) or vv is None):
                            found.append((kk, vv or ""))
                        else:
                            found.extend(_collect_pairs(vv))
                elif isinstance(obj, list):
                    for it in obj:
                        # If it's a pair-like list [eng, ben]
                        if isinstance(it, (list, tuple)) and len(it) >= 2 and isinstance(it[0], str) and isinstance(it[1], (str, type(None))):
                            found.append((it[0], it[1] or ""))
                        else:
                            found.extend(_collect_pairs(it))
                return found

            pairs = _collect_pairs(parsed)
            for ek, ev in pairs:
                _normalize_and_insert(ek, ev)
            if mapping:
                return mapping
    except Exception:
        pass

    # Not JSON -> treat as CSV. Try delimiter detection (comma vs semicolon)
    lines = text.splitlines()
    sample = "\n".join(lines[:5])
    delimiter = ","
    if sample.count(";") > sample.count(","):
        delimiter = ";"

    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    for row in reader:
        if not row:
            continue
        first = row[0].strip()
        # skip header heuristics
        if first.lower() in ("english", "eng", "english_word", "word"):
            continue
        eng = row[0].strip()
        ben = row[1].strip() if len(row) > 1 else ""
        _normalize_and_insert(eng, ben)

    return mapping


def apply_bengali_mapping(text: str, mapping: Dict[str, str]) -> str:
    """Deterministic replacement of whole words/phrases using mapping.

    Longer keys are replaced first so multi-word phrases win over substrings.
    Case-insensitive for English keys (mapping keys expected lowercase).
    """
    if not mapping:
        return text

    keys = sorted(mapping.keys(), key=len, reverse=True)
    # build a word-boundary pattern; this works for most English words/phrases
    pattern = r"\b(?:" + "|".join(re.escape(k) for k in keys) + r")\b"

    def _repl(m):
        matched = m.group(0)
        return mapping.get(matched) or mapping.get(matched.lower()) or matched

    return re.sub(pattern, _repl, text, flags=re.IGNORECASE)


def call_mistral(api_key: str, model: str, messages: list, temperature: float = 0.7, max_tokens: int = 512, safe_prompt: bool = False) -> dict:
    """Call Mistral Chat Completion API (non-streaming).

    This function uses the ApiKey auth header as `Authorization: ApiKey <key>`.
    """
    # Some Mistral endpoints / accounts may expect `Authorization: ApiKey <key>`
    # while others or proxied setups may expect `Authorization: Bearer <key>`.
    # Try both and return on first successful response. If both yield 401,
    # raise an HTTPError with the last response attached so callers can inspect it.
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "safe_prompt": bool(safe_prompt),
    }

    import time
    import random

    auth_prefixes = ["ApiKey", "Bearer"]
    last_resp = None

    max_retries = 5
    base_backoff = 1.0  # seconds

    for prefix in auth_prefixes:
        headers = {
            "Authorization": f"{prefix} {api_key}",
            "Content-Type": "application/json",
        }

        # retry loop for transient 429 responses
        for attempt in range(0, max_retries):
            try:
                resp = requests.post(MISTRAL_CHAT_URL, headers=headers, json=payload, timeout=30)
            except requests.RequestException:
                # network-level error; re-raise so caller can handle
                raise

            if resp.status_code == 401:
                # try next prefix
                last_resp = resp
                break

            if resp.status_code == 429:
                # Service tier capacity exceeded -> wait and retry with exponential backoff + jitter
                wait = base_backoff * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(wait)
                # continue retrying
                last_resp = resp
                continue

            # For any other status, raise_for_status will raise helpful errors (or return on 200)
            resp.raise_for_status()
            return resp.json()

    # If we get here both prefixes returned 401 or repeated 429s
    if last_resp is not None and last_resp.status_code == 401:
        err = requests.HTTPError("401 Unauthorized - Authorization header tried with ApiKey and Bearer prefixes")
        err.response = last_resp
        raise err

    if last_resp is not None and last_resp.status_code == 429:
        err = requests.HTTPError("429 Too Many Requests - service tier capacity exceeded after retries")
        err.response = last_resp
        raise err

    # final fallback (single try)
    resp = requests.post(MISTRAL_CHAT_URL, headers={"Authorization": f"ApiKey {api_key}", "Content-Type": "application/json"}, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def extract_text_from_response(resp: dict) -> str:
    """Robustly extract assistant text from a Mistral chat response."""
    choices = resp.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]

    # common shapes: choice['message']['content'] or choice['content'] or choice['delta']['content']
    if isinstance(choice, dict):
        msg = choice.get("message") or {}
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, dict):
                # sometimes content can be a structured object
                return json.dumps(content)

        if "content" in choice and isinstance(choice["content"], str):
            return choice["content"]

        delta = choice.get("delta") or {}
        if isinstance(delta, dict) and "content" in delta:
            return delta.get("content")

    # fallback: stringify first choice
    try:
        return json.dumps(choice)
    except Exception:
        return str(choice)


def messages_to_pairs(messages: list) -> list:
    """Convert flat messages list (dicts with role/content) to Chatbot pairs list.

    Returns list of (user, assistant) tuples for gr.Chatbot.
    """
    pairs = []
    i = 0
    while i < len(messages):
        if messages[i]["role"] == "user":
            user = messages[i]["content"]
            assistant = ""
            if i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
                assistant = messages[i + 1]["content"]
                i += 2
            else:
                i += 1
            pairs.append((user, assistant))
        else:
            i += 1
    return pairs


def _normalize_chat_history(chat_history: list) -> list:
    """Normalize various chat history shapes to a list of message dicts.

    Accepts:
    - list of dicts like [{'role': 'user', 'content': '...'}, ...]
    - list of (user, assistant) pairs as produced by gr.Chatbot: [('hi','hello'), ...]
    - empty or None
    Returns a list of dicts with 'role' and 'content'.
    """
    if not chat_history:
        return []

    # Already a list of dicts with role keys
    try:
        if all(isinstance(m, dict) and 'role' in m for m in chat_history):
            return list(chat_history)
    except Exception:
        pass

    # If it's a list of pairs (user, assistant)
    is_pair_list = True
    for it in chat_history:
        if not (isinstance(it, (list, tuple)) and len(it) >= 2):
            is_pair_list = False
            break

    if is_pair_list:
        out = []
        for user_part, assistant_part in chat_history:
            out.append({"role": "user", "content": user_part})
            if assistant_part:
                out.append({"role": "assistant", "content": assistant_part})
        return out

    # Fallback: return as list (best-effort)
    return list(chat_history)


def send_message(user_message: str, api_key: str, model: str, chat_history: list, temperature: float, max_tokens: int, safe_prompt: bool, mapping: Dict[str, str], character_name: str = None, relation: str = None, user_label: str = None):
    """Handle a user message from Gradio UI and return updated chat and state."""
    # Use DEFAULT_API_KEY when UI field is empty for convenience. Do not log the key.
    if not api_key:
        api_key = DEFAULT_API_KEY

    # Normalize incoming state to a list of message dicts so we can reliably
    # detect whether a system prompt was already injected.
    messages = _normalize_chat_history(chat_history)

    # If character details are provided, build a short directive describing
    # the character and prepend it to the system prompt so the model roleplays
    # accordingly. Fields are optional; we only include provided values.
    character_directive_parts = []
    try:
        if character_name and isinstance(character_name, str) and character_name.strip():
            character_directive_parts.append(f"Character name: {character_name.strip()}.")
        if relation and isinstance(relation, str) and relation.strip():
            character_directive_parts.append(f"Relation to the user: {relation.strip()}.")
        if user_label and isinstance(user_label, str) and user_label.strip():
            character_directive_parts.append(f"This character should call the user: {user_label.strip()}.")
    except Exception:
        # defensive: ignore any character formatting errors
        character_directive_parts = []

    if character_directive_parts:
        combined_system_prompt = " ".join(character_directive_parts) + " " + SYSTEM_PROMPT
    else:
        combined_system_prompt = SYSTEM_PROMPT

    # Inject the system prompt once per conversation if not already present.
    has_system = any(isinstance(m, dict) and m.get("role") == "system" for m in messages)
    if not has_system:
        # Insert at the beginning so it's the first message the model sees.
        messages.insert(0, {"role": "system", "content": combined_system_prompt})

    # Append the user message
    messages.append({"role": "user", "content": user_message})

    try:
        resp = call_mistral(api_key, model, messages, temperature=temperature, max_tokens=max_tokens, safe_prompt=safe_prompt)
        assistant_text = extract_text_from_response(resp)
        # apply deterministic Bengali mapping if provided
        if mapping:
            assistant_text = apply_bengali_mapping(assistant_text, mapping)
    except requests.HTTPError as e:
        # include server body when possible
        body = ""
        try:
            body = e.response.text
        except Exception:
            body = "(no response body)"
        assistant_text = f"[ERROR {e.response.status_code}] {body}"
    except Exception as e:
        assistant_text = f"[ERROR] {e}"

    messages.append({"role": "assistant", "content": assistant_text})

    return messages_to_pairs(messages), "", messages


def clear_chat():
    return [], "", []


def build_ui():
    with gr.Blocks(title="Simple Mistral Chatbot") as demo:
        gr.Markdown("# Simple Mistral Chatbot\nEnter your Mistral API key, pick a model, type messages and press Enter to chat.")

        with gr.Row():
            api_key = gr.Textbox(label="Mistral API Key", placeholder="sk-...", type="password")
            model = gr.Dropdown(choices=["mistral-small-latest", "mistral-medium-latest", "mistral-large-latest"], value="mistral-small-latest", label="Model", info="Pick a model. Smaller models reduce capacity issues.")

        with gr.Row():
            temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.01, label="Temperature")
            max_tokens = gr.Slider(minimum=16, maximum=2048, value=512, step=1, label="Max tokens")

        chatbot = gr.Chatbot()
        txt = gr.Textbox(show_label=False, placeholder="Type a message and press Enter")

        # Character inputs: name, relation to user, and how to address the user
        with gr.Row():
            character_name = gr.Textbox(label="AI character name", placeholder="e.g. 'Alex'", value="")
            character_relation = gr.Textbox(label="Relation to you", placeholder="e.g. 'friend', 'teacher'", value="")
            character_user_label = gr.Textbox(label="How character should call you", placeholder="e.g. 'boss', 'dear'", value="")

        # mapping state holds the English->Bengali dict
        mapping_state = gr.State(value=BENGALI_MAP)

        # file uploader for CSV mapping
        file_upload = gr.File(label="Load mapping (CSV or JSON)", file_count="single", file_types=[".csv", ".json"])
        load_btn = gr.Button("Load mapping")
        load_status = gr.Textbox(label="Mapping status", interactive=False)

        state = gr.State([])

        with gr.Row():
            safe_cb = gr.Checkbox(label="Enable safe prompt (safety)", value=False, info="When checked, the safety prompt will be injected by the server. Uncheck to disable safety injection.")

        def _on_submit(message, api_key_val, model_val, state_val, temperature_val, max_tokens_val, safe_val, mapping_val, char_name, char_relation, char_user_label):
            return send_message(message, api_key_val, model_val, state_val, temperature_val, int(max_tokens_val), bool(safe_val), mapping_val, character_name=char_name, relation=char_relation, user_label=char_user_label)

        txt.submit(_on_submit, [txt, api_key, model, state, temperature, max_tokens, safe_cb, mapping_state, character_name, character_relation, character_user_label], [chatbot, txt, state])

        def _on_load(file_obj, mapping_val):
            # file_obj can be None or a dict with 'name'/'tmp_path' depending on gradio version
            if not file_obj:
                return mapping_val, "No file provided"
            try:
                # Try multiple strategies to read the uploaded object into text
                text = None
                diagnostics = []

                # If dict (gradio may provide dict with tmp_path)
                if isinstance(file_obj, dict):
                    diagnostics.append(f"upload_keys={list(file_obj.keys())}")
                    tmp = file_obj.get('tmp_path') or file_obj.get('name') or file_obj.get('filename')
                    if tmp and os.path.exists(tmp):
                        try:
                            with open(tmp, 'r', encoding='utf-8-sig', errors='replace') as f:
                                text = f.read()
                            diagnostics.append(f"read from tmp_path={tmp}")
                        except Exception as e:
                            diagnostics.append(f"tmp_path read error: {e}")

                # If object has .file (TemporaryFileWrapper)
                if text is None and hasattr(file_obj, 'file'):
                    try:
                        f = getattr(file_obj, 'file')
                        try:
                            f.seek(0)
                        except Exception:
                            pass
                        raw = f.read()
                        if isinstance(raw, (bytes, bytearray)):
                            text = raw.decode('utf-8-sig', errors='replace')
                        else:
                            text = str(raw)
                        diagnostics.append('read from .file')
                    except Exception as e:
                        diagnostics.append(f'.file read error: {e}')

                # If file-like (readable)
                if text is None and hasattr(file_obj, 'read'):
                    try:
                        try:
                            file_obj.seek(0)
                        except Exception:
                            pass
                        raw = file_obj.read()
                        if isinstance(raw, (bytes, bytearray)):
                            text = raw.decode('utf-8-sig', errors='replace')
                        else:
                            text = str(raw)
                        diagnostics.append('read from .read')
                    except Exception as e:
                        diagnostics.append(f'.read error: {e}')

                # If it has a name pointing to a path
                if text is None and hasattr(file_obj, 'name'):
                    try:
                        name_attr = getattr(file_obj, 'name')
                        if isinstance(name_attr, str) and os.path.exists(name_attr):
                            with open(name_attr, 'r', encoding='utf-8-sig', errors='replace') as f:
                                text = f.read()
                            diagnostics.append(f'read from .name={name_attr}')
                    except Exception as e:
                        diagnostics.append(f'.name read error: {e}')

                # If still none and file_obj is a string path
                if text is None and isinstance(file_obj, str) and os.path.exists(file_obj):
                    try:
                        with open(file_obj, 'r', encoding='utf-8-sig', errors='replace') as f:
                            text = f.read()
                        diagnostics.append(f'read from path={file_obj}')
                    except Exception as e:
                        diagnostics.append(f'path read error: {e}')

                # Extra attempts: sometimes the wrapper exposes a binary buffer
                if not text:
                    # try .file.buffer
                    try:
                        if hasattr(file_obj, 'file') and hasattr(file_obj.file, 'buffer'):
                            b = file_obj.file.buffer.read()
                            if isinstance(b, (bytes, bytearray)) and b:
                                text = b.decode('utf-8-sig', errors='replace')
                                diagnostics.append('read from .file.buffer')
                    except Exception as e:
                        diagnostics.append(f'.file.buffer read error: {e}')

                if not text:
                    # try reopening .name in binary
                    try:
                        if hasattr(file_obj, 'name'):
                            n = getattr(file_obj, 'name')
                            if isinstance(n, str) and os.path.exists(n):
                                b = open(n, 'rb').read()
                                text = b.decode('utf-8-sig', errors='replace')
                                diagnostics.append('reopened .name in binary')
                    except Exception as e:
                        diagnostics.append(f'reopen .name error: {e}')

                snippet = (text[:400] if text else None)
                parsed_type = None
                parse_err = None
                if text:
                    try:
                        parsed = json.loads(text)
                        parsed_type = type(parsed).__name__
                    except Exception as e:
                        parse_err = str(e)

                # Call loader with a file-like StringIO if we have text
                loader_input = io.StringIO(text) if text is not None else file_obj
                new_map = load_mapping_from_csv(loader_input)
                if not new_map:
                    msg = "Loaded file but mapping empty — try saving the Excel as UTF-8 CSV or as a JSON object {\"english\": \"bengali\"}."
                    if snippet:
                        msg += f"\nFile snippet: {snippet}"
                    if parsed_type:
                        msg += f"\nParsed JSON type: {parsed_type}"
                    if parse_err:
                        msg += f"\nJSON parse error: {parse_err}"
                    if diagnostics:
                        msg += f"\nDiagnostics: {'; '.join(diagnostics)}"
                    try:
                        msg += f"\nUpload object repr: {repr(file_obj)[:400]}"
                    except Exception:
                        pass
                    return mapping_val, msg
                return new_map, f"Loaded {len(new_map)} entries (parsed_type={parsed_type})"
            except Exception as e:
                return mapping_val, f"Error loading CSV/JSON: {e}"

        load_btn.click(_on_load, [file_upload, mapping_state], [mapping_state, load_status])

        with gr.Row():
            clear_btn = gr.Button("Clear")
            clear_btn.click(lambda: clear_chat(), outputs=[chatbot, txt, state])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=True)
