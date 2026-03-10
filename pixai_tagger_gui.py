__version__ = "1.5.0"

import sys
from PySide6.QtWidgets import QApplication
from main_window import MainWindow


def main():
    """main entry point."""
    app = QApplication(sys.argv)

    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
