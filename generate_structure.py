"""
generate_structure.py
Run this once from the repo root: python generate_structure.py
Creates the full folder structure with empty __init__.py placeholders
so imports resolve immediately, before any real code is dropped in.
"""

import os

STRUCTURE = {
    "apps/backend/src/rag_harness": [
        "domain/models",
        "domain/ports",
        "application/ingestion",
        "application/retrieval",
        "application/generation",
        "application/verification",
        "application/agent",
        "infrastructure/llm",
        "infrastructure/embeddings",
        "infrastructure/vector_stores",
        "infrastructure/sparse_search",
        "infrastructure/document_parsers",
        "infrastructure/cache",
        "api/v1/routers",
        "api/v1/schemas",
        "api/middleware",
        "config",
        "shared",
    ],
    "apps/backend/tests": ["unit/domain", "unit/application", "unit/infrastructure", "integration", "e2e"],
    "apps/backend": ["scripts"],
    "apps/frontend/src": ["app/documents", "components/chat", "components/upload", "components/ui", "lib", "hooks", "store"],
    "apps/frontend": ["public"],
    "packages": ["shared-types"],
    "infra": ["k8s", "terraform"],
    ".github": ["workflows"],
    "docs": ["adr"],
}

PACKAGE_DIRS_NEEDING_INIT = [
    "domain", "domain/models", "domain/ports",
    "application", "application/ingestion", "application/retrieval",
    "application/generation", "application/verification", "application/agent",
    "infrastructure", "infrastructure/llm", "infrastructure/embeddings",
    "infrastructure/vector_stores", "infrastructure/sparse_search",
    "infrastructure/document_parsers", "infrastructure/cache",
    "api", "api/v1", "api/v1/routers", "api/v1/schemas", "api/middleware",
    "config", "shared",
]


def create_structure():
    for base, subdirs in STRUCTURE.items():
        os.makedirs(base, exist_ok=True)
        for sub in subdirs:
            full_path = os.path.join(base, sub)
            os.makedirs(full_path, exist_ok=True)
            print(f"Created: {full_path}")

    backend_root = "apps/backend/src/rag_harness"
    for pkg_dir in PACKAGE_DIRS_NEEDING_INIT:
        init_path = os.path.join(backend_root, pkg_dir, "__init__.py")
        if not os.path.exists(init_path):
            with open(init_path, "w") as f:
                f.write("")
            print(f"Created: {init_path}")

    root_init = os.path.join(backend_root, "__init__.py")
    if not os.path.exists(root_init):
        open(root_init, "w").close()
        print(f"Created: {root_init}")

    for test_dir in ["unit", "unit/domain", "unit/application", "unit/infrastructure", "integration", "e2e"]:
        init_path = f"apps/backend/tests/{test_dir}/__init__.py"
        if not os.path.exists(init_path):
            open(init_path, "w").close()
            print(f"Created: {init_path}")

    root_files = [".gitignore", ".editorconfig", "README.md"]
    for f in root_files:
        if not os.path.exists(f):
            open(f, "w").close()
            print(f"Created placeholder: {f}")

    print("\nProject structure generated successfully.")


if __name__ == "__main__":
    create_structure()