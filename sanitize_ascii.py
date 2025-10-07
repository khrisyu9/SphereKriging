# sanitize_ascii.py
REPLACEMENTS = {
    "θ": "theta",
    "ℓ": "l",
    "π": "pi",
    "×": "x",
    "→": "->",
    "∈": "in",
    "≤": "<=",
    "≥": ">=",
    "—": "-",
    "–": "-",
    "-": "-",   # non-breaking hyphen
    "√": "sqrt",
}

fname = "SDK_RBF_Sim.py"
with open(fname, "r", encoding="utf-8") as f:
    s = f.read()

for u, a in REPLACEMENTS.items():
    s = s.replace(u, a)

with open(fname, "w", encoding="utf-8") as f:
    f.write(s)

print("Sanitized non-ASCII characters in", fname)
