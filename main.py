"""
Neural Network Classifier — entry point.

University: Karadeniz Teknik University
Department: Computer Engineering
Course: Artificial Neural Systems (BIL5050)
Supervisor: Prof. Dr. Murat Ekinci
Student: Mohamed Hassan (440638)
"""

import tkinter as tk
from gui import NeuralNetworkGUI, LoginWindow

if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralNetworkGUI(root)
    login = LoginWindow(root)
    root.mainloop()
