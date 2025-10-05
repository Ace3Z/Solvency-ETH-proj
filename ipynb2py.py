import json
from pathlib import Path
from typing import Any, Dict, List, Optional

def ipynb_to_py_with_outputs(ipynb_path: str, out_path: Optional[str] = None, include_headers: bool = True):


    ipynb_path = Path(ipynb_path)
    if out_path is None:
        out_path = ipynb_path.with_suffix(".py")
    else:
        out_path = Path(out_path)

    with ipynb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    nb_cells = nb.get("cells", [])
    out_lines: List[str] = []
    out_lines.append(f"# Auto-generated from: {ipynb_path.name}")
    out_lines.append("# This file contains code cells followed by their saved outputs as comments.\n")

    def cell_header(idx: int, exec_count: Optional[int]) -> str:
        if exec_count is None:
            return f"# In[{idx}]:"
        return f"# In[{exec_count}]:  (cell {idx})"

    def format_output_block(outputs: List[Dict[str, Any]], idx: int) -> str:
        if not outputs:
            return ""
        lines: List[str] = []
        lines.append(f"# --- Output [{idx}] ---")
        for out_i, out in enumerate(outputs, start=1):
            otype = out.get("output_type", "unknown")

            def add_comment_block(block_lines: List[str]) -> None:
                for ln in block_lines:
                    lines.append("# " + ln.rstrip("\n"))

            lines.append(f"# [{out_i}] type: {otype}")

            if otype == "stream":
                name = out.get("name", "stdout")
                text = out.get("text", "")
                lines.append(f"# ({name})")
                add_comment_block(text.splitlines() if isinstance(text, str) else [repr(text)])

            elif otype in ("execute_result", "display_data"):
                data = out.get("data", {})
                if "text/plain" in data:
                    text = data["text/plain"]
                    add_comment_block(text.splitlines() if isinstance(text, str) else [repr(text)])
                elif "text" in data:
                    text = data["text"]
                    add_comment_block(text.splitlines() if isinstance(text, str) else [repr(text)])
                else:
                    available = ", ".join(sorted(data.keys()))
                    lines.append(f"# [non-text output: {available} omitted]")

            elif otype == "error":
                ename = out.get("ename", "Error")
                evalue = out.get("evalue", "")
                tb = out.get("traceback", [])
                lines.append(f"# ERROR: {ename}: {evalue}")
                if tb and isinstance(tb, list):
                    add_comment_block([t for t in tb])
                else:
                    lines.append("# (no traceback available)")

            else:
                lines.append(f"# [unhandled output type: {otype}]")
                try:
                    add_comment_block(json.dumps(out, indent=2).splitlines())
                except Exception:
                    add_comment_block([repr(out)])

        lines.append("# --- End Output ---")
        return "\n".join(lines)

    code_index = 0
    for i, cell in enumerate(nb_cells):
        if cell.get("cell_type") != "code":
            continue
        code_index += 1
        exec_count = cell.get("execution_count", None)
        src = cell.get("source", [])
        code_text = "".join(src) if isinstance(src, list) else str(src)

        if include_headers:
            out_lines.append(cell_header(code_index, exec_count))

        out_lines.append(code_text.rstrip("\n"))
        out_lines.append("")

        outputs = cell.get("outputs", [])
        out_block = format_output_block(outputs, code_index)
        if out_block:
            out_lines.append(out_block)
            out_lines.append("")

    if code_index == 0:
        out_lines.append("# (No code cells found.)")

    out_text = "\n".join(out_lines).rstrip() + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out_text, encoding="utf-8")
    print(f"Wrote: {out_path}")

ipynb_path = "code.ipynb"
ipynb_to_py_with_outputs(ipynb_path)