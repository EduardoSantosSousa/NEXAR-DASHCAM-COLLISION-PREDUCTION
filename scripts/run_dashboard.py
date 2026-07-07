import subprocess
import sys


def main() -> None:
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"],
        check=True,
    )


if __name__ == "__main__":
    main()
