from pathlib import Path

from terminal_gui import run_gui

if __name__ == "__main__":
    run_gui(config_path=Path(r"research_config.yaml").absolute())
