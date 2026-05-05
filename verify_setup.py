"""Setup verification script. Run after installing requirements."""
import importlib
import sys


def check(label: str, module: str, attr: str | None = None) -> bool:
    """Try to import a module and optionally access an attribute."""
    try:
        mod = importlib.import_module(module)
        if attr:
            getattr(mod, attr)
        print(f"  [OK]  {label}")
        return True
    except ImportError as e:
        print(f"  [FAIL] {label}: {e}")
        return False
    except AttributeError as e:
        print(f"  [FAIL] {label} (attr): {e}")
        return False


def main() -> None:
    print(f"\nPython {sys.version}\n")

    checks = [
        ("mediapipe", "mediapipe", "solutions"),
        ("opencv-python", "cv2", None),
        ("numpy", "numpy", "ndarray"),
        ("scipy", "scipy", "version"),
        ("fastdtw", "fastdtw", "fastdtw"),
        ("anthropic SDK", "anthropic", "Anthropic"),
        ("google-generativeai", "google.generativeai", None),
        ("openai (DeepSeek compat)", "openai", "OpenAI"),
        ("python-dotenv", "dotenv", "load_dotenv"),
        ("jupyter / notebook", "notebook", None),
    ]

    print("=== Dependency check ===")
    results = [check(label, module, attr) for label, module, attr in checks]

    print()
    passed = sum(results)
    total = len(results)
    if passed == total:
        print(f"All {total} checks passed.")
    else:
        print(f"{passed}/{total} checks passed. Install missing packages with:")
        print("  .venv\\Scripts\\pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
