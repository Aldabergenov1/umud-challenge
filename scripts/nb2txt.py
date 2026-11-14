import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
for nb in root.rglob("*.ipynb"):
    d = json.loads(nb.read_text(encoding="utf-8"))
    out = []
    for c in d.get("cells", []):
        src = "".join(c.get("source", []))
        if c["cell_type"] == "markdown":
            out.append("# [MD] " + src.replace("\n", "\n# "))
        else:
            out.append(src)
            for o in c.get("outputs", []):
                t = "".join(o.get("text", [])) if "text" in o else "".join(o.get("data", {}).get("text/plain", []))
                if t:
                    out.append("# [OUT] " + t[:1500].replace("\n", "\n# "))
    p = nb.with_suffix(".txt")
    p.write_text("\n\n".join(out), encoding="utf-8")
    print(p.name, len(p.read_text(encoding="utf-8")))
