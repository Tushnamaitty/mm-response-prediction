import os

ROOT_DIR = "."
OUTPUT_FILE = "repo_structure.txt"

EXCLUDE_DIRS = {
    ".venv",
    "venv",
    "env",
    ".env",
    "node_modules",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
    "dist",
    "build",
    "site-packages",
    ".next",
    ".cache",
}

EXCLUDE_FILES = {
    ".DS_Store",
    OUTPUT_FILE,
}

def write_tree(directory, file, prefix=""):
    entries = sorted(
        entry for entry in os.listdir(directory)
        if entry not in EXCLUDE_DIRS
        and entry not in EXCLUDE_FILES
    )

    for index, entry in enumerate(entries):
        path = os.path.join(directory, entry)
        is_last = index == len(entries) - 1

        connector = "└── " if is_last else "├── "
        file.write(prefix + connector + entry + "\n")

        if os.path.isdir(path):
            extension = "    " if is_last else "│   "
            write_tree(path, file, prefix + extension)


repo_name = os.path.basename(os.path.abspath(ROOT_DIR))

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(repo_name + "/\n")
    write_tree(ROOT_DIR, f)

print(f"Repo structure saved to {OUTPUT_FILE}")