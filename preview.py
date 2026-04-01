"""
Neural Network Classifier — preview entry point (no login).

Launches the GUI directly without the login window.
"""

import tkinter as tk
from gui import NeuralNetworkGUI

if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralNetworkGUI(root)
    root.mainloop()
