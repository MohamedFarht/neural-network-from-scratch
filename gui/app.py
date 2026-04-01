"""Main GUI application for the Neural Network Classifier."""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from network import NeuralNetwork
from gui.visualization import (
    get_class_colors, draw_architecture, draw_decision_boundaries,
    draw_weights, draw_error_plot,
)
from gui.report import generate_report


class NeuralNetworkGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Classifier")

        self.loss_function = tk.StringVar(value="MSE")
        self.points = np.array([])
        self.labels = np.array([])
        self.sample_count = 0
        self.error_history = []
        self.mean = np.zeros(2)
        self.std = np.ones(2)
        self.nn = None

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.setup_gui()

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            plt.close("all")
            self.root.destroy()

    def setup_gui(self):
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)

        self.left_frame = ttk.Frame(main_container)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_frame = ttk.Frame(main_container)
        right_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)

        self.main_plot_frame = ttk.Frame(self.left_frame)
        self.arch_view_frame = ttk.Frame(self.left_frame)

        self.setup_main_plot(self.main_plot_frame)
        self.setup_architecture_view(self.arch_view_frame)

        self.setup_network_architecture(right_frame)
        self.setup_training_parameters(right_frame)
        self.setup_optimizer_parameters(right_frame)
        self.setup_action_buttons(right_frame)
        self.setup_results_frame(right_frame)
        self.setup_visualization(right_frame)

        self.show_main_plot()

    # -- view switching ----------------------------------------------------

    def show_main_plot(self):
        if hasattr(self, "main_view_button"):
            self.main_view_button.state(["pressed"])
        if hasattr(self, "arch_view_button"):
            self.arch_view_button.state(["!pressed"])
        self.arch_view_frame.pack_forget()
        self.main_plot_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas.draw()

    def show_architecture_view(self):
        if hasattr(self, "main_view_button"):
            self.main_view_button.state(["!pressed"])
        if hasattr(self, "arch_view_button"):
            self.arch_view_button.state(["pressed"])
        self.main_plot_frame.pack_forget()
        self.arch_view_frame.pack(fill=tk.BOTH, expand=True)
        self.visualize_architecture()
        self.arch_canvas.draw()

    # -- plot areas --------------------------------------------------------

    def setup_architecture_view(self, parent):
        self.arch_fig = Figure(figsize=(6, 6))
        self.arch_fig.subplots_adjust(
            left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0,
        )
        self.arch_ax = self.arch_fig.add_subplot(111)
        self.arch_fig.tight_layout(pad=0.1, rect=[0.05, 0.05, 0.95, 0.95])

        self.arch_canvas = FigureCanvasTkAgg(self.arch_fig, parent)
        self.arch_canvas.draw()
        self.arch_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        parent.bind("<Configure>", self.on_frame_resize)

    def setup_main_plot(self, parent):
        self.fig = Figure(figsize=(6, 6))
        self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0)

        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.grid(False)
        self.ax.axhline(y=0, color="k", linewidth=1)
        self.ax.axvline(x=0, color="k", linewidth=1)
        self.fig.tight_layout(pad=0.1)

        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.canvas.mpl_connect("button_press_event", self.on_click)

    # -- right panel controls ----------------------------------------------

    def setup_network_architecture(self, parent):
        arch_frame = ttk.LabelFrame(parent, text="Network Architecture", padding=(5, 5))
        arch_frame.pack(fill=tk.X, pady=(0, 5))

        main_controls = ttk.Frame(arch_frame)
        main_controls.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(main_controls, text="Number of Classes:").pack(side=tk.LEFT)
        self.class_var = tk.StringVar(value="5")
        self.class_var.trace_add("write", self.update_label_options)
        ttk.Entry(main_controls, textvariable=self.class_var, width=5).pack(side=tk.LEFT, padx=5)

        ttk.Label(main_controls, text="Training Method:").pack(side=tk.LEFT, padx=(10, 5))
        self.training_methods = ttk.Combobox(
            main_controls, values=["Single layer", "Multi layer"], width=15,
        )
        self.training_methods.pack(side=tk.LEFT)
        self.training_methods.set("Single layer")

        ttk.Label(main_controls, text="Label:").pack(side=tk.LEFT, padx=(20, 5))
        self.current_label = tk.StringVar(value="1")
        self.label_combo = ttk.Combobox(
            main_controls, textvariable=self.current_label, values=["1"], width=5,
        )
        self.label_combo.pack(side=tk.LEFT)

        output_activation_frame = ttk.Frame(arch_frame)
        output_activation_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(output_activation_frame, text="Output Activation:").pack(side=tk.LEFT)
        self.output_activation = ttk.Combobox(
            output_activation_frame, values=["sigmoid", "relu", "tanh"], width=10,
        )
        self.output_activation.pack(side=tk.LEFT, padx=5)
        self.output_activation.set("sigmoid")

        self.use_softmax = tk.BooleanVar(value=False)
        self.softmax_check = ttk.Checkbutton(
            output_activation_frame, text="Use Softmax", variable=self.use_softmax,
        )
        self.softmax_check.pack(side=tk.LEFT, padx=10)
        self.softmax_check.state(["disabled"])

        self.hidden_layers_frame = ttk.Frame(arch_frame)
        self.hidden_layers_frame.pack(fill=tk.X, pady=5)

        self.hidden_sizes_frame = ttk.Frame(self.hidden_layers_frame)
        self.hidden_sizes_frame.pack(fill=tk.X, padx=5, pady=5)

        self.hidden_size_vars = []
        self.hidden_activation_vars = []
        self.setup_hidden_layers_controls()

        self.softmax_check.configure(command=self.on_softmax_change)
        self.training_methods.bind("<<ComboboxSelected>>", self.on_method_change)
        self.hidden_layers_frame.pack_forget()
        self.update_label_options()

    def on_method_change(self, event=None):
        if self.training_methods.get() == "Multi layer":
            self.softmax_check.state(["!disabled"])
            self.hidden_layers_frame.pack(fill=tk.X, pady=5)
            if not hasattr(self, "hidden_sizes_frame") or not self.hidden_sizes_frame.winfo_children():
                self.setup_hidden_layers_controls()
                self.update_hidden_layers()
        else:
            self.softmax_check.state(["disabled"])
            self.use_softmax.set(False)
            self.hidden_layers_frame.pack_forget()

        if hasattr(self, "arch_view_frame") and str(self.arch_view_frame.winfo_manager()) == "pack":
            self.visualize_architecture()

    def setup_hidden_layers_controls(self):
        for widget in self.hidden_sizes_frame.winfo_children():
            widget.destroy()

        self.hidden_size_vars = []
        self.hidden_activation_vars = []

        first_line = ttk.Frame(self.hidden_sizes_frame)
        first_line.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(first_line, text="Hidden Layers:").pack(side=tk.LEFT)
        self.num_hidden_layers = ttk.Combobox(first_line, values=["1", "2", "3"], width=5)
        self.num_hidden_layers.pack(side=tk.LEFT, padx=(5, 15))
        self.num_hidden_layers.set("1")
        self.num_hidden_layers.bind("<<ComboboxSelected>>", self.update_hidden_layers)

        self.setup_hidden_layer(first_line, 1)

        second_line = ttk.Frame(self.hidden_sizes_frame)
        second_line.pack(fill=tk.X, pady=(0, 5))
        second_line.pack_forget()

    def update_hidden_layers(self, event=None):
        try:
            if not hasattr(self, "num_hidden_layers"):
                return
            num_layers = int(self.num_hidden_layers.get())
            frames = self.hidden_sizes_frame.winfo_children()
            if len(frames) < 2:
                return

            first_line = frames[0]
            second_line = frames[1]

            for widget in second_line.winfo_children():
                widget.destroy()

            while len(self.hidden_size_vars) > 1:
                self.hidden_size_vars.pop()
            while len(self.hidden_activation_vars) > 1:
                self.hidden_activation_vars.pop()

            first_line.pack(fill=tk.X, pady=(0, 5))

            if num_layers > 1:
                second_line.pack(fill=tk.X, pady=(0, 5))
                self.setup_hidden_layer(second_line, 2)
                if num_layers > 2:
                    ttk.Label(second_line, text="  ").pack(side=tk.LEFT)
                    self.setup_hidden_layer(second_line, 3)
            else:
                second_line.pack_forget()

            if hasattr(self, "arch_view_frame") and str(self.arch_view_frame.winfo_manager()) == "pack":
                self.visualize_architecture()
        except ValueError:
            pass

    def setup_hidden_layer(self, parent, layer_num):
        ttk.Label(parent, text=f"Layer {layer_num}:").pack(side=tk.LEFT)
        size_var = tk.StringVar(value="5")
        self.hidden_size_vars.append(size_var)
        ttk.Entry(parent, textvariable=size_var, width=5).pack(side=tk.LEFT, padx=(5, 15))

        ttk.Label(parent, text="Activation:").pack(side=tk.LEFT)
        activation_var = ttk.Combobox(parent, values=["sigmoid", "relu", "tanh"], width=10)
        activation_var.set("sigmoid")
        activation_var.pack(side=tk.LEFT, padx=5)
        self.hidden_activation_vars.append(activation_var)

        size_var.trace_add("write", lambda *args: self.visualize_architecture())

    def setup_training_parameters(self, parent):
        param_frame = ttk.LabelFrame(parent, text="Training Parameters", padding=(5, 5))
        param_frame.pack(fill=tk.X, pady=5)

        param_content = ttk.Frame(param_frame)
        param_content.pack(fill=tk.X, padx=5, pady=5)

        param_row1 = ttk.Frame(param_content)
        param_row1.pack(fill=tk.X, pady=2)

        ttk.Label(param_row1, text="Target Error:").pack(side=tk.LEFT)
        self.target_error = tk.StringVar(value="0.001")
        ttk.Entry(param_row1, textvariable=self.target_error, width=8).pack(side=tk.LEFT, padx=5)

        ttk.Label(param_row1, text="Max Epochs:").pack(side=tk.LEFT, padx=(10, 0))
        self.max_epochs = tk.StringVar(value="1000")
        ttk.Entry(param_row1, textvariable=self.max_epochs, width=8).pack(side=tk.LEFT, padx=5)

        ttk.Label(param_row1, text="Batch Size:").pack(side=tk.LEFT, padx=(10, 0))
        self.batch_size = tk.StringVar(value="32")
        ttk.Entry(param_row1, textvariable=self.batch_size, width=8).pack(side=tk.LEFT, padx=5)

    def setup_optimizer_parameters(self, parent):
        self.optimizer_frame = ttk.LabelFrame(parent, text="Optimizer Parameters", padding=(5, 5))
        self.optimizer_frame.pack(fill=tk.X, pady=5)

        top_frame = ttk.Frame(self.optimizer_frame)
        top_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(top_frame, text="Loss Function:").pack(side=tk.LEFT)
        self.loss_combo = ttk.Combobox(
            top_frame, textvariable=self.loss_function,
            values=["MSE", "Binary Cross Entropy", "Categorical Cross Entropy"], width=20,
        )
        self.loss_combo.pack(side=tk.LEFT, padx=5)

        ttk.Label(top_frame, text="Optimizer:").pack(side=tk.LEFT, padx=(20, 5))
        self.optimizer = ttk.Combobox(top_frame, values=["SGD", "Momentum"], width=10)
        self.optimizer.set("Momentum")
        self.optimizer.pack(side=tk.LEFT, padx=5)
        self.optimizer.bind("<<ComboboxSelected>>", self.on_optimizer_change)

        params_frame = ttk.Frame(self.optimizer_frame)
        params_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(params_frame, text="Learning Rate:").pack(side=tk.LEFT)
        self.learning_rate = ttk.Combobox(
            params_frame,
            values=["1.0", "0.3", "0.1", "0.03", "0.01", "0.003", "0.001"], width=8,
        )
        self.learning_rate.set("0.03")
        self.learning_rate.pack(side=tk.LEFT, padx=5)

        self.momentum_label = ttk.Label(params_frame, text="Momentum:")
        self.momentum_label.pack(side=tk.LEFT, padx=(10, 0))
        self.momentum = tk.StringVar(value="0.9")
        self.momentum_entry = ttk.Entry(params_frame, textvariable=self.momentum, width=8)
        self.momentum_entry.pack(side=tk.LEFT, padx=5)

        ttk.Label(params_frame, text="L2 Lambda:").pack(side=tk.LEFT, padx=(10, 0))
        self.l2_lambda = tk.StringVar(value="0.0")
        ttk.Entry(params_frame, textvariable=self.l2_lambda, width=8).pack(side=tk.LEFT, padx=5)

    def on_optimizer_change(self, event=None):
        if self.optimizer.get() == "SGD":
            self.momentum_label.pack_forget()
            self.momentum_entry.pack_forget()
        else:
            self.momentum_label.pack(side=tk.LEFT, padx=(10, 0))
            self.momentum_entry.pack(side=tk.LEFT, padx=5)

    def setup_action_buttons(self, parent):
        button_frame = ttk.LabelFrame(parent, text="Actions", padding=(5, 5))
        button_frame.pack(fill=tk.X, pady=5)

        row1 = ttk.Frame(button_frame)
        row1.pack(fill=tk.X, padx=5, pady=2)

        left_buttons = ttk.Frame(row1)
        left_buttons.pack(side=tk.LEFT)

        ttk.Button(left_buttons, text="Train Network", command=self.train_network).pack(side=tk.LEFT, padx=2)
        ttk.Button(left_buttons, text="Clear Data", command=self.clear_data).pack(side=tk.LEFT, padx=2)
        ttk.Button(left_buttons, text="Generate Report", command=lambda: generate_report(self)).pack(side=tk.LEFT, padx=2)

        right_controls = ttk.Frame(row1)
        right_controls.pack(side=tk.RIGHT)

        ttk.Label(right_controls, text="Boundary Style:").pack(side=tk.LEFT, padx=(10, 2))
        self.viz_style = ttk.Combobox(
            right_controls, values=["Lines Only", "Regions Only", "Both"],
            width=12, state="readonly",
        )
        self.viz_style.set("Regions Only")
        self.viz_style.pack(side=tk.LEFT, padx=2)
        self.viz_style.bind("<<ComboboxSelected>>", lambda e: self.update_visualization())

        row2 = ttk.Frame(button_frame)
        row2.pack(fill=tk.X, padx=5, pady=2)

        ttk.Button(row2, text="Save Samples", command=self.save_samples).pack(side=tk.LEFT, padx=2)
        ttk.Button(row2, text="Load Samples", command=self.load_samples).pack(side=tk.LEFT, padx=2)
        ttk.Button(row2, text="Save Weights", command=self.save_weights).pack(side=tk.LEFT, padx=2)
        ttk.Button(row2, text="Load Weights", command=self.load_weights).pack(side=tk.LEFT, padx=2)

    def setup_results_frame(self, parent):
        results_frame = ttk.LabelFrame(parent, text="Results", padding=(5, 5))
        results_frame.pack(fill=tk.X, pady=5)

        results_content = ttk.Frame(results_frame)
        results_content.pack(fill=tk.X, padx=5, pady=5)

        self.total_samples_var = tk.StringVar(value="Samples: 0")
        self.epochs_run_var = tk.StringVar(value="Epochs: 0")
        self.current_error_var = tk.StringVar(value="Error: 0.0")

        ttk.Label(results_content, textvariable=self.total_samples_var).pack(side=tk.LEFT, padx=5)
        ttk.Label(results_content, textvariable=self.epochs_run_var).pack(side=tk.LEFT, padx=5)
        ttk.Label(results_content, textvariable=self.current_error_var).pack(side=tk.LEFT, padx=5)

    def setup_visualization(self, parent):
        viz_frame = ttk.LabelFrame(parent, text="Visualization", padding=(5, 5))
        viz_frame.pack(fill=tk.X, pady=5)

        nav_frame = ttk.Frame(viz_frame)
        nav_frame.pack(fill=tk.X, padx=5, pady=2)

        button_frame = ttk.Frame(nav_frame)
        button_frame.pack(fill=tk.X, expand=True)
        button_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.main_view_button = ttk.Button(button_frame, text="Main Plot", command=self.show_main_plot)
        self.main_view_button.grid(row=0, column=0, padx=2, sticky="ew")

        self.arch_view_button = ttk.Button(button_frame, text="Network Architecture", command=self.show_architecture_view)
        self.arch_view_button.grid(row=0, column=1, padx=2, sticky="ew")

        self.error_button = ttk.Button(button_frame, text="Training Error", command=lambda: self.show_graph("error"))
        self.error_button.grid(row=0, column=2, padx=2, sticky="ew")

        self.weights_button = ttk.Button(button_frame, text="Network Weights", command=lambda: self.show_graph("weights"))
        self.weights_button.grid(row=0, column=3, padx=2, sticky="ew")

        self.error_frame = ttk.Frame(viz_frame, height=300)
        self.error_frame.pack_propagate(False)
        self.weights_frame = ttk.Frame(viz_frame, height=300)
        self.weights_frame.pack_propagate(False)

        self.error_fig = Figure(figsize=(8, 4), dpi=100)
        self.error_ax = self.error_fig.add_subplot(111)
        self.error_ax.set_xlabel("Epoch")
        self.error_ax.set_ylabel("Error")
        self.error_canvas = FigureCanvasTkAgg(self.error_fig, self.error_frame)
        self.error_canvas.draw()
        self.error_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.weights_fig = Figure(figsize=(8, 4), dpi=100)
        self.weights_ax = self.weights_fig.add_subplot(111)
        self.weights_canvas = FigureCanvasTkAgg(self.weights_fig, self.weights_frame)
        self.weights_canvas.draw()
        self.weights_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        viz_frame.bind("<Configure>", self.on_viz_frame_resize)
        self.show_graph("error")

    # -- event handlers ----------------------------------------------------

    def on_viz_frame_resize(self, event=None):
        if event is not None:
            width = event.width / 100
            height = (event.height - 50) / 100
            if hasattr(self, "error_fig"):
                self.error_fig.set_size_inches(width, height)
                self.error_canvas.draw()
            if hasattr(self, "weights_fig"):
                self.weights_fig.set_size_inches(width, height)
                self.weights_canvas.draw()

    def on_frame_resize(self, event=None):
        if hasattr(self, "arch_fig"):
            width = self.arch_view_frame.winfo_width() / 100
            height = self.arch_view_frame.winfo_height() / 100
            self.arch_fig.set_size_inches(width, height)
            if str(self.arch_view_frame.winfo_manager()) == "pack":
                self.visualize_architecture()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        try:
            current_label = int(self.current_label.get())
            num_classes = int(self.class_var.get())

            if current_label > num_classes:
                messagebox.showwarning(
                    "Invalid Label",
                    f"Label {current_label} cannot be greater than number of classes {num_classes}",
                )
                return

            x = np.array([event.xdata, event.ydata])
            self.sample_count += 1

            if len(self.points) == 0:
                self.points = np.array([x])
                self.labels = np.array([current_label])
            else:
                self.points = np.vstack([self.points, x])
                self.labels = np.append(self.labels, current_label)

            _, point_colors = get_class_colors(num_classes)
            color = point_colors[current_label - 1]

            circle = plt.Circle((event.xdata, event.ydata), 0.05, color=color, alpha=0.6)
            self.ax.add_patch(circle)
            edge_circle = plt.Circle((event.xdata, event.ydata), 0.05, color=color, alpha=0.3, fill=False, linewidth=1)
            self.ax.add_patch(edge_circle)

            self.canvas.draw()
            self.total_samples_var.set(f"Samples: {self.sample_count}")
        except Exception as e:
            messagebox.showerror("Error", f"Error adding point: {e}")

    def on_softmax_change(self, event=None):
        if self.use_softmax.get():
            self.output_activation.state(["disabled"])
            self.output_activation.set("sigmoid")
        else:
            self.output_activation.state(["!disabled"])

    def update_label_options(self, *args):
        try:
            num_classes = int(self.class_var.get())
            if num_classes < 1:
                num_classes = 1
                self.class_var.set("1")
            elif num_classes > 100:
                num_classes = 100
                self.class_var.set("100")

            new_values = [str(i) for i in range(1, num_classes + 1)]
            self.label_combo["values"] = new_values

            current_label = int(self.current_label.get())
            if current_label > num_classes:
                self.current_label.set("1")
        except ValueError:
            self.class_var.set("5")
            self.label_combo["values"] = [str(i) for i in range(1, 6)]
            self.current_label.set("1")

    # -- graph switching ---------------------------------------------------

    def show_graph(self, graph_type):
        self.error_button.state(["!pressed"])
        self.weights_button.state(["!pressed"])

        if graph_type == "error":
            self.error_button.state(["pressed"])
            self.error_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            self.weights_frame.pack_forget()
            self.error_fig.tight_layout()
            self.error_canvas.draw()
        elif graph_type == "weights":
            self.weights_button.state(["pressed"])
            self.weights_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            self.error_frame.pack_forget()
            self.weights_fig.tight_layout()
            self.weights_canvas.draw()

    # -- delegated visualization -------------------------------------------

    def visualize_architecture(self):
        draw_architecture(self)

    def visualize_decision_boundaries(self):
        draw_decision_boundaries(self)

    def visualize_weights(self):
        draw_weights(self)

    def update_error_plot(self):
        draw_error_plot(self)

    def update_visualization(self):
        style_map = {"Lines Only": "lines", "Regions Only": "regions", "Both": "both"}
        if not hasattr(self, "nn") or self.nn is None:
            draw_decision_boundaries(self)
            return
        self.viz_style_internal = style_map[self.viz_style.get()]
        self.visualize_decision_boundaries()

    # -- data management ---------------------------------------------------

    def clear_data(self):
        if messagebox.askyesno("Clear Data", "Are you sure you want to clear all data?"):
            self.points = np.array([])
            self.labels = np.array([])
            self.sample_count = 0
            self.error_history = []
            self.mean = np.zeros(2)
            self.std = np.ones(2)
            self.nn = None

            self.ax.clear()
            self.ax.set_xlim(-5, 5)
            self.ax.set_ylim(-5, 5)
            self.ax.grid(False)
            self.ax.set_xticklabels([])
            self.ax.set_yticklabels([])
            self.ax.axhline(y=0, color="k", linewidth=0.5)
            self.ax.axvline(x=0, color="k", linewidth=0.5)
            self.canvas.draw()

            self.error_ax.clear()
            self.error_ax.set_xlabel("Epoch")
            self.error_ax.set_ylabel("Error")
            self.error_canvas.draw()

            self.weights_ax.clear()
            self.weights_canvas.draw()

            self.total_samples_var.set("Samples: 0")
            self.epochs_run_var.set("Epochs: 0")
            self.current_error_var.set("Error: 0.0")

            self.show_graph("error")

    def validate_data(self):
        if len(self.points) == 0:
            messagebox.showerror("Data Error", "No training data available")
            return False

        num_classes = int(self.class_var.get())
        for i in range(1, num_classes + 1):
            class_samples = np.sum(self.labels == i)
            if class_samples < 2:
                messagebox.showerror(
                    "Data Error",
                    f"Class {i} needs at least 2 samples (has {class_samples})",
                )
                return False
        return True

    def normalize_data(self):
        self.mean = np.mean(self.points, axis=0)
        self.std = np.std(self.points, axis=0)
        self.std[self.std == 0] = 1

        normalized_points = (self.points - self.mean) / self.std

        num_classes = int(self.class_var.get())
        one_hot_labels = np.zeros((len(self.labels), num_classes))
        for i, label in enumerate(self.labels):
            one_hot_labels[i, int(label) - 1] = 1

        return normalized_points, one_hot_labels

    # -- training ----------------------------------------------------------

    def train_network(self):
        loss_map = {
            "MSE": "mse",
            "Binary Cross Entropy": "binary_crossentropy",
            "Categorical Cross Entropy": "categorical_crossentropy",
        }
        selected_loss = loss_map[self.loss_function.get()]

        optimizer_map = {"SGD": "sgd", "Momentum": "momentum"}
        selected_optimizer = optimizer_map[self.optimizer.get()]

        num_classes = int(self.class_var.get())
        if selected_loss == "binary_crossentropy" and num_classes != 2:
            messagebox.showerror(
                "Error", "Binary Cross Entropy can only be used with binary classification",
            )
            return

        params = self.validate_parameters()
        if params is None or not self.validate_data():
            return

        X, y = self.normalize_data()
        if X is None:
            return

        try:
            if self.training_methods.get() == "Multi layer":
                hidden_activations = [var.get() for var in self.hidden_activation_vars]
            else:
                hidden_activations = "sigmoid"

            output_activation = self.output_activation.get()
            use_softmax = self.use_softmax.get()

            self.nn = NeuralNetwork(
                input_size=2,
                hidden_sizes=params["hidden_sizes"] if self.training_methods.get() == "Multi layer" else None,
                output_size=params["num_classes"],
                training_method=self.training_methods.get(),
                hidden_activations=hidden_activations,
                output_activation=output_activation,
                use_softmax=use_softmax,
                loss_function=selected_loss,
                learning_rate=params["learning_rate"],
                optimizer=selected_optimizer,
                momentum=params["momentum"] if selected_optimizer == "momentum" else 0.0,
                l2_lambda=params["l2_lambda"],
            )
            self.nn.reset_optimizer_state()

            best_epoch, best_error = self.nn.train(
                X, y,
                max_epochs=params["max_epochs"],
                target_error=params["target_error"],
                batch_size=params["batch_size"],
            )

            self.error_history = self.nn.error_history
            self.epochs_run_var.set(f"Epochs: {best_epoch}")
            self.current_error_var.set(f"Error: {best_error:.6f}")

            self.update_error_plot()
            self.visualize_weights()
            self.visualize_decision_boundaries()

        except Exception as e:
            messagebox.showerror("Training Error", str(e))

    def validate_parameters(self):
        try:
            num_classes = int(self.class_var.get())
            if num_classes < 2:
                raise ValueError("Number of classes must be at least 2")

            max_epochs = int(self.max_epochs.get())
            if max_epochs < 1:
                raise ValueError("Maximum epochs must be positive")

            target_error = float(self.target_error.get())
            if target_error <= 0:
                raise ValueError("Target error must be positive")

            batch_size = int(self.batch_size.get())
            if batch_size < 1:
                raise ValueError("Batch size must be positive")

            learning_rate = float(self.learning_rate.get())
            if learning_rate <= 0 or learning_rate > 1:
                raise ValueError("Learning rate must be between 0 and 1")

            momentum = float(self.momentum.get())
            if momentum < 0 or momentum >= 1:
                raise ValueError("Momentum must be between 0 and 1")

            optimizer = self.optimizer.get()
            if optimizer == "Momentum" and momentum == 0:
                raise ValueError("Momentum value cannot be 0 when using Momentum optimizer")

            l2_lambda = float(self.l2_lambda.get())
            if l2_lambda < 0:
                raise ValueError("L2 lambda must be non-negative")

            valid_activations = ["sigmoid", "relu", "tanh"]
            output_activation = self.output_activation.get()
            if output_activation not in valid_activations:
                raise ValueError(f"Invalid output activation function: {output_activation}")

            hidden_sizes = []
            if self.training_methods.get() == "Multi layer":
                num_hidden_layers = int(self.num_hidden_layers.get())
                if not 1 <= num_hidden_layers <= 3:
                    raise ValueError("Number of hidden layers must be between 1 and 3")

                for i in range(num_hidden_layers):
                    size = int(self.hidden_size_vars[i].get())
                    if size < 1:
                        raise ValueError(f"Hidden layer {i+1} size must be positive")
                    hidden_sizes.append(size)

                    activation = self.hidden_activation_vars[i].get()
                    if activation not in valid_activations:
                        raise ValueError(f"Invalid activation function for layer {i+1}: {activation}")

            return {
                "num_classes": num_classes,
                "max_epochs": max_epochs,
                "target_error": target_error,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "momentum": momentum,
                "l2_lambda": l2_lambda,
                "hidden_sizes": hidden_sizes,
            }
        except ValueError as e:
            messagebox.showerror("Parameter Error", str(e))
            return None

    # -- save / load -------------------------------------------------------

    def save_samples(self):
        if len(self.points) == 0:
            messagebox.showwarning("Warning", "No samples to save.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".npz", filetypes=[("NumPy files", "*.npz")],
            title="Save Samples",
        )
        if not file_path:
            return

        try:
            np.savez(
                file_path, points=self.points, labels=self.labels,
                mean=self.mean, std=self.std,
                sample_count=np.array([self.sample_count]),
            )
            messagebox.showinfo("Saved", f"Samples saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving samples: {e}")

    def load_samples(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("NumPy files", "*.npz")], title="Load Samples",
        )
        if not file_path:
            return

        try:
            data = np.load(file_path)
            self.points = data["points"]
            self.labels = data["labels"]
            self.mean = data["mean"]
            self.std = data["std"]
            self.sample_count = int(data["sample_count"][0])

            self.total_samples_var.set(f"Samples: {self.sample_count}")

            self.ax.clear()
            self.ax.set_xlim(-5, 5)
            self.ax.set_ylim(-5, 5)
            self.ax.grid(False)
            self.ax.axhline(y=0, color="k")
            self.ax.axvline(x=0, color="k")

            num_classes = int(self.class_var.get())
            _, point_colors = get_class_colors(num_classes)

            for i in range(self.sample_count):
                x = self.points[i, 0]
                y = self.points[i, 1]
                label = int(self.labels[i]) - 1
                color = point_colors[label]
                circle = plt.Circle((x, y), 0.05, color=color, alpha=0.6)
                self.ax.add_patch(circle)
                edge_circle = plt.Circle((x, y), 0.05, color=color, alpha=0.3, fill=False, linewidth=1)
                self.ax.add_patch(edge_circle)

            self.canvas.draw()
            messagebox.showinfo("Loaded", f"Samples loaded from {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading samples: {e}")

    def save_weights(self):
        if self.nn is None:
            messagebox.showwarning("Warning", "No trained network to save.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".npz", filetypes=[("NumPy files", "*.npz")],
            title="Save Network Weights",
        )
        if not file_path:
            return

        try:
            config = self.nn.get_config()
            np.savez(
                file_path,
                weights=[w for w in self.nn.weights],
                biases=[b for b in self.nn.biases],
                config=np.array([config], dtype=object),
                error_history=np.array(self.error_history),
            )
            messagebox.showinfo("Saved", f"Weights saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving weights: {e}")

    def load_weights(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("NumPy files", "*.npz")], title="Load Network Weights",
        )
        if not file_path:
            return

        try:
            data = np.load(file_path, allow_pickle=True)
            weights = [w for w in data["weights"]]
            biases = [b for b in data["biases"]]
            config = data["config"][0]
            self.error_history = data["error_history"].tolist()

            self.nn = NeuralNetwork(
                input_size=config["input_size"],
                hidden_sizes=config["hidden_sizes"],
                output_size=config["output_size"],
                training_method=config["training_method"],
                hidden_activations=config["hidden_activations"],
                output_activation=config["output_activation"],
                learning_rate=config["learning_rate"],
                optimizer=config.get("optimizer", "momentum"),
                momentum=config["momentum"],
                l2_lambda=config["l2_lambda"],
            )
            self.nn.weights = weights
            self.nn.biases = biases

            self.training_methods.set(config["training_method"])
            self.learning_rate.set(str(config["learning_rate"]))
            self.momentum.set(str(config["momentum"]))
            self.l2_lambda.set(str(config["l2_lambda"]))
            self.class_var.set(str(config["output_size"]))

            if "optimizer" in config:
                self.optimizer.set(config["optimizer"])
                self.on_optimizer_change()

            if len(self.error_history) > 0:
                self.current_error_var.set(f"Error: {self.error_history[-1]:.6f}")
                self.epochs_run_var.set(f"Epochs: {len(self.error_history)}")
                self.update_error_plot()
                self.visualize_weights()

            if len(self.points) > 0:
                self.visualize_decision_boundaries()

            messagebox.showinfo("Loaded", f"Weights loaded from {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading weights: {e}")
