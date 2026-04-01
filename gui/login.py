"""Login window for the Neural Network Classifier."""

import os
import sys
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


def _resource_path(filename):
    """Resolve path to a bundled resource (PyInstaller or normal)."""
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", filename)


class LoginWindow:
    def __init__(self, main_root):
        self.root = tk.Toplevel()
        self.main_root = main_root
        self.root.title("Login")
        self.main_root.withdraw()

        window_width = 400
        window_height = 400
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        self.root.resizable(False, False)
        self.root.attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Logo and header
        header_frame = ttk.Frame(self.root, padding="20")
        header_frame.pack(fill=tk.X)

        try:
            logo_img = Image.open(_resource_path("ktu_logo.png"))
            logo_img = logo_img.resize((100, 100), Image.Resampling.LANCZOS)
            self.logo_photo = ImageTk.PhotoImage(logo_img)
            logo_label = ttk.Label(header_frame, image=self.logo_photo)
            logo_label.pack()
        except (FileNotFoundError, IOError):
            ttk.Label(
                header_frame, text="KTU", font=("Helvetica", 24, "bold")
            ).pack(pady=20)

        ttk.Label(
            header_frame,
            text="Karadeniz Technical University 1955",
            font=("Helvetica", 14, "bold"),
        ).pack(pady=5)
        ttk.Label(
            header_frame,
            text="Computer Engineering Department",
            font=("Helvetica", 12),
        ).pack()

        # Login form
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        username_frame = ttk.Frame(main_frame)
        username_frame.pack(fill=tk.X, pady=5)
        ttk.Label(username_frame, text="Username:", width=10).pack(side=tk.LEFT)
        self.username_var = tk.StringVar()
        self.username_entry = ttk.Entry(username_frame, textvariable=self.username_var)
        self.username_entry.pack(side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True)

        password_frame = ttk.Frame(main_frame)
        password_frame.pack(fill=tk.X, pady=5)
        ttk.Label(password_frame, text="Password:", width=10).pack(side=tk.LEFT)
        self.password_var = tk.StringVar()
        self.password_entry = ttk.Entry(
            password_frame, textvariable=self.password_var, show="*"
        )
        self.password_entry.pack(side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True)

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=15)
        center_frame = ttk.Frame(button_frame)
        center_frame.pack(expand=True)
        self.login_button = ttk.Button(
            center_frame, text="Login", command=self.validate_login, width=15
        )
        self.login_button.pack(pady=5)

        self.root.bind("<Return>", lambda e: self.validate_login())
        self.username_entry.focus()

    def validate_login(self):
        username = self.username_var.get().strip()
        password = self.password_var.get().strip()

        if username == "admin" and password == "440638":
            self.root.destroy()
            self.main_root.deiconify()
        else:
            error_window = tk.Toplevel(self.root)
            error_window.title("Error")
            error_window.attributes("-topmost", True)
            error_window.geometry(
                f"+{self.root.winfo_x() + 50}+{self.root.winfo_y() + 50}"
            )

            ttk.Label(error_window, text="Invalid username or password", padding=20).pack()
            ttk.Button(error_window, text="OK", command=error_window.destroy).pack(
                pady=(0, 10)
            )

            self.password_var.set("")
            self.password_entry.focus()

            error_window.transient(self.root)
            error_window.grab_set()
            self.root.wait_window(error_window)

    def on_closing(self):
        self.main_root.destroy()
