__version__ = "0.1.0"

import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from main_window import MainWindow
from utils import write_debug_log

def main():
    """main entry point."""
    app = QApplication(sys.argv)

    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

