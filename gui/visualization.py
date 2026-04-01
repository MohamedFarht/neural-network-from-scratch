"""Decision boundary, architecture diagram, weight, and error plot rendering."""

import colorsys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def get_class_colors(num_classes):
    """Return (region_colors, point_colors) lists for *num_classes* classes."""
    base_region = ["lightblue", "#ffcc99", "#ccffcc", "#ffccff", "#ffffcc"]
    base_point = ["#0066cc", "#ff6600", "#00cc00", "#cc00cc", "#cccc00"]

    if num_classes > len(base_region):
        for i in range(len(base_region), num_classes):
            hue = (i * 0.618033988749895) % 1
            rgb = colorsys.hsv_to_rgb(hue, 0.3, 0.95)
            base_region.append(
                f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"
            )
            rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.8)
            base_point.append(
                f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"
            )

    return base_region[:num_classes], base_point[:num_classes]



def draw_architecture(gui):
    """Render the network architecture onto *gui.arch_ax*."""
    if not hasattr(gui, "arch_ax"):
        return

    gui.arch_ax.clear()

    fig_height_px = gui.arch_fig.get_figheight() * gui.arch_fig.dpi
    margin_fraction = 20 / fig_height_px

    gui.arch_fig.subplots_adjust(
        left=0.02, right=0.98, bottom=0.05, top=1 - margin_fraction,
        wspace=0, hspace=0,
    )

    input_size = 2
    output_size = int(gui.class_var.get())
    is_multi = gui.training_methods.get() == "Multi layer"

    hidden_sizes = []
    hidden_activations = []
    if is_multi:
        try:
            hidden_sizes = [int(var.get()) for var in gui.hidden_size_vars]
            hidden_activations = [var.get() for var in gui.hidden_activation_vars]
        except (ValueError, AttributeError):
            hidden_sizes = [5]
            hidden_activations = ["sigmoid"]

    num_classes = int(gui.class_var.get())
    region_colors, point_colors = get_class_colors(num_classes)
    node_colors = {
        "input": "#87CEEB",
        "hidden": "#722F37",
        "output": point_colors,
    }

    activation_info = {
        "sigmoid": {
            "equation": r"$f(x)=\frac{1}{1+e^{-x}}$",
            "range": "Range: (0, 1)",
        },
        "relu": {"equation": r"$f(x)=max(0,x)$", "range": "Range: [0, \u221e]"},
        "tanh": {
            "equation": r"$f(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$",
            "range": "Range: (-1, 1)",
        },
        "softmax": {
            "equation": r"$f(x_i)=\frac{e^{x_i}}{\sum_{j} e^{x_j}}$",
            "range": "Range: (0, 1), \u03a3 = 1",
        },
    }

    loss_info = {
        "MSE": r"$L=\frac{1}{n}\sum_{i=1}^n(y_i-\hat{y_i})^2$",
        "Binary Cross Entropy": r"$L=-\frac{1}{n}\sum_{i=1}^n[y_i\log(\hat{y_i})+(1-y_i)\log(1-\hat{y_i})]$",
        "Categorical Cross Entropy": r"$L=-\frac{1}{n}\sum_{i=1}^n\sum_{j=1}^c y_{ij}\log(\hat{y_{ij}})$",
    }

    layer_spacing = 2.5
    node_spacing = 1.0
    node_radius = 0.25
    total_layers = len(hidden_sizes) + 2 if is_multi else 2
    max_nodes = max([input_size] + hidden_sizes + [output_size])

    width = gui.arch_view_frame.winfo_width() / 100
    height = gui.arch_view_frame.winfo_height() / 100
    gui.arch_fig.set_size_inches(width, height)

    gui.arch_ax.set_xlim(-1, (total_layers - 1) * layer_spacing + 1.5)
    gui.arch_ax.set_ylim(
        -max_nodes * node_spacing / 2 - 2.5,
        max_nodes * node_spacing / 2 + 3,
    )

    title = "Multi-Layer Neural Network" if is_multi else "Single-Layer Neural Network"
    loss_name = gui.loss_function.get()
    max_y = max_nodes * node_spacing / 2 + 2.8

    gui.arch_ax.text(
        (total_layers - 1) * layer_spacing / 2, max_y, title,
        ha="center", va="bottom", fontsize=14, fontweight="bold",
    )
    gui.arch_ax.text(
        (total_layers - 1) * layer_spacing / 2, max_y - 0.5,
        f"Loss Function: {loss_name}",
        ha="center", va="bottom", fontsize=12,
    )

    def draw_layer(pos_x, size, color, layer_name="", activation=None, layer_weights=None):
        nodes = []
        start_y = -(size - 1) * node_spacing / 2

        gui.arch_ax.text(
            pos_x, max_nodes * node_spacing / 2 + 1.1, layer_name,
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

        if activation and activation in activation_info:
            info = activation_info[activation]
            gui.arch_ax.text(
                pos_x, max_nodes * node_spacing / 2 + 0.85,
                f"Activation: {activation}",
                ha="center", va="bottom", fontsize=12,
            )
            gui.arch_ax.text(
                pos_x, max_nodes * node_spacing / 2 + 0.15,
                info["equation"], ha="center", va="bottom", fontsize=14,
            )
            gui.arch_ax.text(
                pos_x, max_nodes * node_spacing / 2.1,
                info["range"], ha="center", va="bottom", fontsize=10,
            )

        for i in range(size):
            y = start_y + i * node_spacing
            node_color = color[i] if isinstance(color, list) else color
            circle = plt.Circle((pos_x, y), node_radius, color=node_color, alpha=0.8)
            gui.arch_ax.add_patch(circle)

            text_color = "white" if node_color == node_colors["hidden"] else "black"
            gui.arch_ax.text(
                pos_x, y, str(i + 1),
                ha="center", va="center", color=text_color, fontsize=8,
            )

            if layer_weights is not None and i < len(layer_weights):
                gui.arch_ax.text(
                    pos_x, y - node_spacing / 2,
                    f"w: {layer_weights[i]:.3f}",
                    ha="center", va="top", fontsize=8, color="darkred",
                )

            nodes.append((pos_x, y))

        gui.arch_ax.text(
            pos_x, -max_nodes * node_spacing / 2 - 0.3,
            f"Size: {size}", ha="center", va="top", fontsize=8,
        )
        return nodes

    def draw_connections(layer1_nodes, layer2_nodes, weights=None):
        for i, n1 in enumerate(layer1_nodes):
            for j, n2 in enumerate(layer2_nodes):
                alpha = 0.2
                linewidth = 0.5
                if weights is not None:
                    try:
                        if weights.shape[1] > i:
                            weight = abs(weights[j][i])
                            linewidth = 0.5 + weight * 0.5
                            alpha = min(0.8, 0.2 + weight * 0.2)
                    except (IndexError, AttributeError):
                        pass
                gui.arch_ax.plot(
                    [n1[0], n2[0]], [n1[1], n2[1]],
                    color="gray", alpha=alpha, linewidth=linewidth,
                )

    weights = gui.nn.weights if gui.nn is not None else None
    biases = gui.nn.biases if gui.nn is not None else None

    input_nodes = draw_layer(0, input_size, node_colors["input"], "Input Layer\n(Features)")
    gui.arch_ax.text(-0.5, input_nodes[0][1], "X", ha="right", va="center", fontsize=10)
    gui.arch_ax.text(-0.5, input_nodes[1][1], "Y", ha="right", va="center", fontsize=10)

    prev_nodes = input_nodes
    layer_idx = 0
    for i, (size, activation) in enumerate(zip(hidden_sizes, hidden_activations)):
        pos_x = (i + 1) * layer_spacing
        layer_bias = biases[i] if biases is not None and i < len(biases) else None
        hidden_nodes = draw_layer(
            pos_x, size, node_colors["hidden"],
            f"Hidden Layer {i+1}", activation, layer_bias,
        )

        if weights is not None and layer_idx < len(weights):
            draw_connections(prev_nodes, hidden_nodes, weights[layer_idx])
            layer_idx += 1
        else:
            draw_connections(prev_nodes, hidden_nodes)

        prev_nodes = hidden_nodes

    output_x = (total_layers - 1) * layer_spacing
    output_activation = (
        "softmax" if gui.use_softmax.get() else gui.output_activation.get()
    )
    output_bias = biases[-1] if biases is not None and len(biases) > 0 else None
    output_nodes = draw_layer(
        output_x, output_size, node_colors["output"],
        "Output Layer\n(Classes)", output_activation, output_bias,
    )

    for i, node in enumerate(output_nodes):
        gui.arch_ax.text(
            node[0] + 0.5, node[1], f"Class {i+1}",
            ha="left", va="center", fontsize=10,
        )

    if weights is not None and layer_idx < len(weights):
        draw_connections(prev_nodes, output_nodes, weights[layer_idx])
    else:
        draw_connections(prev_nodes, output_nodes)

    if hasattr(gui, "error_history") and gui.error_history:
        bottom_y = -max_nodes * node_spacing / 2 - 1.5
        stats_text = (
            f"Training Error: {gui.error_history[-1]:.6f}\n"
            f"Epochs: {len(gui.error_history)}\n"
            f"Current Loss Function:\n"
            f"{loss_info[gui.loss_function.get()]}"
        )
        gui.arch_ax.text(0, bottom_y, stats_text, ha="left", va="top", fontsize=12)
        current_ylim = gui.arch_ax.get_ylim()
        gui.arch_ax.set_ylim(current_ylim[0] - 0.5, current_ylim[1])

    gui.arch_ax.set_aspect("auto")
    gui.arch_ax.axis("off")
    gui.arch_canvas.draw()



def draw_decision_boundaries(gui):
    """Render decision boundaries, regions, and training points onto *gui.ax*."""
    if gui.nn is None:
        _draw_no_network(gui)
        return

    try:
        x_min, x_max = gui.ax.get_xlim()
        y_min, y_max = gui.ax.get_ylim()

        resolution = 300
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution),
        )
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        num_classes = int(gui.class_var.get())
        region_colors, point_colors = get_class_colors(num_classes)

        gui.ax.clear()
        gui.ax.set_xlim(x_min, x_max)
        gui.ax.set_ylim(y_min, y_max)

        grid_points_norm = (grid_points - gui.mean) / gui.std
        outputs = gui.nn.forward(grid_points_norm)
        probabilities = outputs[0]
        predictions = np.argmax(probabilities, axis=1).reshape(xx.shape)

        viz_style = gui.viz_style.get()
        is_multi_layer = gui.training_methods.get() == "Multi layer"

        if viz_style in ["Regions Only", "Both"]:
            custom_cmap = ListedColormap(region_colors)
            gui.ax.pcolormesh(xx, yy, predictions, cmap=custom_cmap, alpha=0.3, shading="auto")

        if viz_style in ["Lines Only", "Both"]:
            weights = gui.nn.weights[0]
            biases = gui.nn.biases[0]
            for i in range(len(weights)):
                w = weights[i]
                b = biases[i]
                if abs(w[1]) > abs(w[0]):
                    x_norm = np.array([-2, 2])
                    y_norm = -(w[0] * x_norm + b) / (w[1] + 1e-10)
                else:
                    y_norm = np.array([-2, 2])
                    x_norm = -(w[1] * y_norm + b) / (w[0] + 1e-10)

                x_line = x_norm * gui.std[0] + gui.mean[0]
                y_line = y_norm * gui.std[1] + gui.mean[1]

                if is_multi_layer:
                    gui.ax.plot(x_line, y_line, "--", color=point_colors[i], linewidth=1.0, alpha=0.5)
                else:
                    gui.ax.plot(x_line, y_line, "-", color=point_colors[i], linewidth=1.5)

        misclassified_points = []
        if len(gui.points) > 0:
            for i in range(gui.sample_count):
                x = gui.points[i, 0]
                y = gui.points[i, 1]
                label = int(gui.labels[i]) - 1
                gui.ax.plot(x, y, "o", color=point_colors[label], markersize=6)

                point_norm = (np.array([x, y]) - gui.mean) / gui.std
                pred_outputs = gui.nn.forward(point_norm.reshape(1, -1))
                pred = np.argmax(pred_outputs[0], axis=1)[0]

                if pred != label:
                    misclassified_points.append((x, y))

            for x, y in misclassified_points:
                gui.ax.plot(
                    x, y, "o", color="red", fillstyle="none",
                    markersize=10, markeredgewidth=2,
                )

        legend_elements = []
        for i in range(num_classes):
            legend_elements.append(
                plt.Line2D(
                    [0], [0], marker="o", color=point_colors[i],
                    label=f"Class {i+1}", markersize=6, linestyle="None",
                )
            )
        if viz_style in ["Lines Only", "Both"]:
            line_style = "--" if is_multi_layer else "-"
            legend_elements.append(
                plt.Line2D(
                    [0], [0], color="gray", linestyle=line_style,
                    label="Decision Boundaries", linewidth=1.5,
                )
            )
        if misclassified_points:
            legend_elements.append(
                plt.Line2D(
                    [0], [0], marker="o", color="red", fillstyle="none",
                    label="Misclassified", markersize=10, markeredgewidth=2,
                    linestyle="None",
                )
            )

        gui.ax.legend(handles=legend_elements, loc="best")
        gui.ax.set_xticklabels([])
        gui.ax.set_yticklabels([])
        gui.ax.grid(False)
        gui.ax.axhline(y=0, color="k", linewidth=0.5)
        gui.ax.axvline(x=0, color="k", linewidth=0.5)

    except Exception as e:
        gui.ax.clear()
        gui.ax.set_xlim(-5, 5)
        gui.ax.set_ylim(-5, 5)
        gui.ax.grid(False)
        gui.ax.axhline(y=0, color="k", linewidth=0.5)
        gui.ax.axvline(x=0, color="k", linewidth=0.5)

        num_classes = int(gui.class_var.get())
        _, point_colors = get_class_colors(num_classes)
        if len(gui.points) > 0:
            for i in range(gui.sample_count):
                x = gui.points[i, 0]
                y = gui.points[i, 1]
                label = int(gui.labels[i]) - 1
                gui.ax.plot(x, y, "o", color=point_colors[label], markersize=6)
            for i in range(num_classes):
                gui.ax.plot([], [], "o", color=point_colors[i], label=f"Class {i+1}", markersize=6)
            gui.ax.legend()
        else:
            gui.ax.text(
                0.5, 0.5, "No data available",
                ha="center", va="center", transform=gui.ax.transAxes,
            )
        print(f"viz error: {e}")

    finally:
        gui.canvas.draw()


def _draw_no_network(gui):
    """Redraw points when no trained network exists."""
    gui.ax.clear()
    gui.ax.set_xlim(-5, 5)
    gui.ax.set_ylim(-5, 5)
    gui.ax.grid(False)
    gui.ax.axhline(y=0, color="k", linewidth=0.5)
    gui.ax.axvline(x=0, color="k", linewidth=0.5)

    if len(gui.points) > 0:
        num_classes = int(gui.class_var.get())
        _, point_colors = get_class_colors(num_classes)
        for i in range(gui.sample_count):
            x = gui.points[i, 0]
            y = gui.points[i, 1]
            label = int(gui.labels[i]) - 1
            gui.ax.plot(x, y, "o", color=point_colors[label], markersize=6)
        for i in range(num_classes):
            gui.ax.plot([], [], "o", color=point_colors[i], label=f"Class {i+1}", markersize=6)
        gui.ax.legend()
    else:
        gui.ax.text(
            0.5, 0.5, "No training data or network available",
            ha="center", va="center", transform=gui.ax.transAxes,
        )

    gui.canvas.draw()



def draw_weights(gui):
    """Render weight values onto *gui.weights_ax*."""
    if gui.nn is None:
        return

    gui.weights_ax.clear()
    num_classes = int(gui.class_var.get())
    _, colors = get_class_colors(num_classes)

    if gui.nn.is_multi:
        total_layers = len(gui.nn.weights)
        hidden_colors = plt.cm.Blues(np.linspace(0.3, 0.7, total_layers - 1))

        for layer in range(total_layers):
            weights = gui.nn.weights[layer]
            biases = gui.nn.biases[layer]

            if layer == total_layers - 1:
                for i in range(num_classes):
                    start_idx = sum(len(w) for w in gui.nn.weights[:layer]) * 3
                    indices = np.array([start_idx + i * 3, start_idx + i * 3 + 1, start_idx + i * 3 + 2])
                    values = np.array([weights[i, 0], weights[i, 1], biases[i]])
                    gui.weights_ax.plot(
                        indices, values, "-o", color=colors[i],
                        label=f"Output {i+1}", linewidth=2, markersize=6,
                    )
            else:
                for i in range(len(biases)):
                    start_idx = sum(len(w) for w in gui.nn.weights[:layer]) * 3
                    indices = np.array([start_idx + i * 3, start_idx + i * 3 + 1, start_idx + i * 3 + 2])
                    values = np.array([weights[i, 0], weights[i, 1], biases[i]])
                    gui.weights_ax.plot(
                        indices, values, "-o", color=hidden_colors[layer],
                        label=f"Hidden {layer+1}" if i == 0 else "",
                        linewidth=2, markersize=6, alpha=0.7,
                    )
    else:
        weights = gui.nn.weights[0]
        biases = gui.nn.biases[0]
        for i in range(num_classes):
            indices = np.array([i * 3, i * 3 + 1, i * 3 + 2])
            values = np.array([weights[i, 0], weights[i, 1], biases[i]])
            gui.weights_ax.plot(
                indices, values, "-o", color=colors[i],
                label=f"Class {i+1}", linewidth=2, markersize=6,
            )

    gui.weights_ax.set_xlabel("Weight Index")
    gui.weights_ax.set_ylabel("Weight Value")
    gui.weights_ax.grid(True)
    gui.weights_ax.legend(
        loc="upper right", bbox_to_anchor=(0.98, 0.98),
        framealpha=0.8, edgecolor="none",
    )
    gui.weights_fig.tight_layout()
    gui.weights_canvas.draw()



def draw_error_plot(gui):
    """Render training error history onto *gui.error_ax*."""
    gui.error_ax.clear()
    gui.error_ax.set_xlabel("Epoch")
    gui.error_ax.set_ylabel("Error")

    epochs = range(1, len(gui.error_history) + 1)
    gui.error_ax.plot(epochs, gui.error_history, "b-", label="Training Error")

    min_error_idx = np.argmin(gui.error_history)
    min_error = gui.error_history[min_error_idx]
    gui.error_ax.plot(min_error_idx + 1, min_error, "ro", label="Best Error")

    gui.error_ax.legend()
    gui.error_ax.grid(True)
    gui.error_fig.tight_layout()
    gui.error_canvas.draw()
