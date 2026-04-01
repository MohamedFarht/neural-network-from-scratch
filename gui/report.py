"""PDF report generation for trained neural networks."""

import io
import datetime

from tkinter import filedialog, messagebox

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
)


def fig_to_img(fig, dpi=300):
    """Convert a matplotlib Figure to a reportlab Image scaled to fit the page."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    img = Image(buf)

    max_width = 6.5 * inch
    max_height = 4 * inch
    aspect = img.imageWidth / img.imageHeight
    if img.imageWidth > max_width:
        img.drawWidth = max_width
        img.drawHeight = max_width / aspect
    if img.drawHeight > max_height:
        img.drawHeight = max_height
        img.drawWidth = max_height * aspect

    return img


def generate_report(gui):
    """Build and save a PDF report from the current GUI state."""
    if gui.nn is None:
        messagebox.showerror("Error", "No network has been trained yet.")
        return

    file_path = filedialog.asksaveasfilename(
        defaultextension=".pdf",
        filetypes=[("PDF files", "*.pdf")],
        title="Save Report As",
    )
    if not file_path:
        return

    try:
        doc = SimpleDocTemplate(
            file_path, pagesize=letter,
            rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72,
        )
        elements = []

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "CustomTitle", parent=styles["Heading1"], fontSize=24, spaceAfter=30,
        )
        heading_style = ParagraphStyle(
            "CustomHeading", parent=styles["Heading2"],
            fontSize=16, spaceBefore=20, spaceAfter=12,
        )
        subheading_style = ParagraphStyle(
            "CustomSubHeading", parent=styles["Heading3"],
            fontSize=14, spaceBefore=15, spaceAfter=10,
        )
        normal_style = styles["Normal"]

        elements.append(Paragraph("Neural Network Analysis Report", title_style))
        elements.append(Paragraph(
            f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            normal_style,
        ))
        elements.append(Spacer(1, 20))

        # 1. Architecture
        elements.append(Paragraph("1. Network Architecture", heading_style))
        config = gui.nn.get_config()
        arch_data = [
            ["Parameter", "Value"],
            ["Training Method", config["training_method"]],
            ["Input Size", str(config["input_size"])],
            ["Hidden Layers", str(config["hidden_sizes"]) if config["hidden_sizes"] else "None"],
            ["Output Size", str(config["output_size"])],
            ["Use Softmax", str(gui.use_softmax.get())],
        ]
        arch_table = Table(arch_data, colWidths=[2.5 * inch, 3.5 * inch])
        arch_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(arch_table)
        elements.append(Spacer(1, 15))

        # 2. Training parameters
        elements.append(Paragraph("2. Training Parameters", heading_style))
        train_data = [
            ["Parameter", "Value"],
            ["Learning Rate", f"{config['learning_rate']:.6f}"],
            ["Momentum", f"{config['momentum']:.4f}"],
            ["L2 Lambda", f"{config['l2_lambda']:.6f}"],
            ["Batch Size", str(gui.batch_size.get())],
            ["Max Epochs", str(gui.max_epochs.get())],
            ["Target Error", str(gui.target_error.get())],
            ["Loss Function", gui.loss_function.get()],
        ]
        train_table = Table(train_data, colWidths=[2.5 * inch, 3.5 * inch])
        train_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(train_table)
        elements.append(Spacer(1, 15))

        # 3. Math details
        elements.append(Paragraph("3. Mathematical Details", heading_style))
        elements.append(Paragraph("Activation Functions:", subheading_style))

        act_descriptions = {
            "sigmoid": r"Sigmoid: f(x) = 1/(1 + e^(-x))",
            "relu": r"ReLU: f(x) = max(0, x)",
            "tanh": r"Tanh: f(x) = (e^x - e^(-x))/(e^x + e^(-x))",
            "softmax": r"Softmax: f(x_i) = e^(x_i)/\u03a3(e^(x_j))",
        }
        used_activations = set()
        if config["hidden_activations"]:
            used_activations.update(config["hidden_activations"])
        used_activations.add(config["output_activation"])
        if gui.use_softmax.get():
            used_activations.add("softmax")
        for act in used_activations:
            elements.append(Paragraph(act_descriptions.get(act, ""), normal_style))
        elements.append(Spacer(1, 10))

        elements.append(Paragraph("Loss Function:", subheading_style))
        loss_descriptions = {
            "MSE": r"Mean Squared Error: L = (1/n)\u03a3(y - \u0177)\u00b2",
            "Binary Cross Entropy": r"Binary Cross-Entropy: L = -(1/n)\u03a3[y log(\u0177) + (1-y)log(1-\u0177)]",
            "Categorical Cross Entropy": r"Categorical Cross-Entropy: L = -(1/n)\u03a3[\u03a3 yij log(\u0177ij)]",
        }
        elements.append(Paragraph(loss_descriptions[gui.loss_function.get()], normal_style))
        elements.append(Spacer(1, 15))

        # 4. Final parameters
        elements.append(Paragraph("4. Final Network Parameters", heading_style))
        elements.append(Paragraph("Final Weights:", subheading_style))
        for i, weights in enumerate(gui.nn.weights):
            elements.append(Paragraph(f"Layer {i+1}:", normal_style))
            weight_data = [[f"{val:.6f}" for val in row] for row in weights]
            wt = Table(weight_data)
            wt.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]))
            elements.append(wt)
            elements.append(Spacer(1, 10))

        elements.append(Paragraph("Final Biases:", subheading_style))
        for i, biases in enumerate(gui.nn.biases):
            elements.append(Paragraph(
                f"Layer {i+1}: {[f'{b:.6f}' for b in biases]}", normal_style,
            ))

        # 5. Results
        elements.append(Paragraph("5. Training Results", heading_style))
        results_data = [
            ["Metric", "Value"],
            ["Total Samples", str(gui.sample_count)],
            ["Training Epochs", str(len(gui.error_history))],
            ["Final Error", f"{gui.error_history[-1]:.6f}"],
            ["Best Error", f"{min(gui.error_history):.6f}"],
            ["Data Mean", f"[{gui.mean[0]:.4f}, {gui.mean[1]:.4f}]"],
            ["Data Std", f"[{gui.std[0]:.4f}, {gui.std[1]:.4f}]"],
        ]
        rt = Table(results_data, colWidths=[2.5 * inch, 3.5 * inch])
        rt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(rt)
        elements.append(Spacer(1, 20))

        # 6. Visualizations
        elements.append(Paragraph("6. Visualizations", heading_style))
        elements.append(Paragraph("Decision Boundaries and Training Data:", subheading_style))
        elements.append(fig_to_img(gui.fig))
        elements.append(Spacer(1, 15))
        elements.append(Paragraph("Training Error History:", subheading_style))
        elements.append(fig_to_img(gui.error_fig))
        elements.append(Spacer(1, 15))
        elements.append(Paragraph("Network Architecture:", subheading_style))
        elements.append(fig_to_img(gui.arch_fig))
        elements.append(Spacer(1, 15))
        elements.append(Paragraph("Network Weights Visualization:", subheading_style))
        elements.append(fig_to_img(gui.weights_fig))

        doc.build(elements)
        messagebox.showinfo("Report saved", f"Report written to {file_path}")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to generate report: {e}")
