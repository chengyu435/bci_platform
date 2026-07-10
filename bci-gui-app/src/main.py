import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARENT_DIR = PROJECT_ROOT.parent
for candidate in (str(PROJECT_ROOT), str(PARENT_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from tkinter import Tk
from app import BCIApp


def main():
    root = Tk()
    window = BCIApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()