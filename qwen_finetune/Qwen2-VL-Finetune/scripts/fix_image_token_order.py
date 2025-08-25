#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import shutil
from typing import Any, Dict, List, Union


def normalize_image_token(text: str) -> str:
    """Ensure that if '<image>' appears, it's moved to the first line as '<image>\n'.
    If already at start, keep it; otherwise, remove all occurrences and place one at top.
    """
    if text is None or '<image>' not in text:
        return text

    # Remove all '<image>' occurrences and surrounding extra spaces/newlines
    without = text.replace('<image>', '').strip()

    # If there was nothing else
    if not without:
        return '<image>'

    # Put one '<image>' at start followed by newline and the remaining text
    return '<image>\n' + without


def process_conversations(obj: Dict[str, Any]) -> bool:
    """Process a single data item with 'conversations' list.
    Returns True if any modification was made.
    """
    modified = False
    convs = obj.get('conversations')
    if not isinstance(convs, list):
        return modified

    for turn in convs:
        if isinstance(turn, dict) and turn.get('from') == 'human' and isinstance(turn.get('value'), str):
            old_val = turn['value']
            new_val = normalize_image_token(old_val)
            if new_val != old_val:
                turn['value'] = new_val
                modified = True
    return modified


def process_file(path: str) -> None:
    print(f'Processing: {path}')
    if not os.path.exists(path):
        print(f'  ! File not found, skipping: {path}')
        return

    # Read JSON (list or dict)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    modified = False

    # Handle list of items
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                if process_conversations(item):
                    modified = True
    # Handle dict with nested datasets
    elif isinstance(data, dict):
        # Try common key containers
        keys = list(data.keys())
        for k in keys:
            v = data[k]
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        if process_conversations(item):
                            modified = True
            elif isinstance(v, dict):
                if process_conversations(v):
                    modified = True
        # Also process top-level if it itself is one entry
        if process_conversations(data):
            modified = True
    else:
        print('  ! Unrecognized JSON structure, skipping modifications')

    if not modified:
        print('  No changes needed.')
        return

    # Backup original
    backup_path = path + '.bak'
    if not os.path.exists(backup_path):
        shutil.copy2(path, backup_path)
        print(f'  Backup created: {backup_path}')
    else:
        print(f'  Backup already exists: {backup_path}')

    # Write back pretty-printed JSON
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print('  File updated.')


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print('Usage: python fix_image_token_order.py <json_file1> [<json_file2> ...]')
        return 1

    for p in argv[1:]:
        process_file(p)
    print('Done.')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))