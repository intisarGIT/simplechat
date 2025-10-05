# normalize_mapping.py
# Reads mapping.json (or a filename you pass), merges arrays/objects into one mapping,
# lowercases keys, strips whitespace, and writes mapping_merged.json (UTF-8).
# Safe: does not print or upload the file anywhere.

import json
import sys
from pathlib import Path
from typing import Any, Dict

def collect_pairs(obj: Any) -> Dict[str, str]:
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and (isinstance(v, str) or v is None):
                out[k.strip().lower()] = (v or "").strip()
            else:
                # recurse into values
                nested = collect_pairs(v)
                out.update(nested)
    elif isinstance(obj, list):
        for item in obj:
            out.update(collect_pairs(item))
    return out

def normalize_file(path: Path) -> Dict[str, str]:
    text = path.read_text(encoding="utf-8-sig")
    data = json.loads(text)
    mapping = {}
    # If dict, take direct pairs and also recursively extract
    if isinstance(data, dict):
        mapping.update(collect_pairs(data))
    elif isinstance(data, list):
        for item in data:
            mapping.update(collect_pairs(item))
    else:
        # fallback: try recursive extraction
        mapping.update(collect_pairs(data))
    return mapping

# normalize_mapping.py
# Reads mapping.json (or a filename you pass), merges arrays/objects into one mapping,
# lowercases keys, strips whitespace, and writes mapping_merged.json (UTF-8).
# Safe: does not print or upload the file anywhere.

import json
import sys
from pathlib import Path
from typing import Any, Dict


def collect_pairs(obj: Any) -> Dict[str, str]:
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and (isinstance(v, str) or v is None):
                out[k.strip().lower()] = (v or "").strip()
            else:
                # recurse into values
                nested = collect_pairs(v)
                out.update(nested)
    elif isinstance(obj, list):
        for item in obj:
            out.update(collect_pairs(item))
    return out


def normalize_file(path: Path) -> Dict[str, str]:
    text = path.read_text(encoding="utf-8-sig")
    data = json.loads(text)
    mapping = {}
    # If dict, take direct pairs and also recursively extract
    if isinstance(data, dict):
        mapping.update(collect_pairs(data))
    elif isinstance(data, list):
        for item in data:
            mapping.update(collect_pairs(item))
    else:
        # fallback: try recursive extraction
        mapping.update(collect_pairs(data))
    return mapping


def main():
    fname = sys.argv[1] if len(sys.argv) > 1 else "mapping.json"
    p = Path(fname)
    if not p.exists():
        print("File not found: {}".format(p))
        sys.exit(2)
    mapping = normalize_file(p)
    out_path = p.with_name("mapping_merged.json")
    out_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote: {} ({} entries).".format(out_path, len(mapping)))  # only prints filename & count, not contents


if __name__ == '__main__':
    main()