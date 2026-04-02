from pathlib import Path
import sys

def clean_one_leading_space(path: str):
    p = Path(path)
    text = p.read_text(encoding="utf-8")

    cleaned = "".join(
        line[1:] if line.startswith(" ") else line
        for line in text.splitlines(keepends=True)
    )

    p.write_text(cleaned, encoding="utf-8")
    print(f"Cleaned: {p}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clean_indent.py <file>")
        sys.exit(1)

    clean_one_leading_space(sys.argv[1])