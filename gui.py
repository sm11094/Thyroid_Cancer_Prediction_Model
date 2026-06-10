import os
import tkinter as tk
from tkinter import ttk, messagebox

from crud import *
from kNN import get_knn
from predict import *


class ThyroidPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Thyroid Cancer Prediction System")
        self.root.geometry("980x720")
        self.root.minsize(850, 600)
        self.root.configure(bg="#eef3f8")

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.features_path = os.path.join(self.base_dir, "features.csv")
        self.labels_path = os.path.join(self.base_dir, "labels.csv")
        self.ids_path = os.path.join(self.base_dir, "ids.csv")

        self.colors = {
            "bg": "#eef3f8",
            "panel": "#ffffff",
            "primary": "#0f4c81",
            "primary_dark": "#0b355b",
            "accent": "#12a4d9",
            "success": "#168a52",
            "danger": "#c0392b",
            "warning": "#b7791f",
            "text": "#1f2937",
            "muted": "#6b7280",
            "border": "#d9e2ec",
            "entry": "#f9fbfd",
        }

        self.setup_styles()

        try:
            self.tree, self.id_database = initialize_tree(
                self.features_path,
                self.labels_path,
                self.ids_path
            )
            self.db_status_text = "Database connected"
            self.db_status_color = self.colors["success"]

        except FileNotFoundError:
            messagebox.showwarning(
                "Warning",
                "Data files not found. Make sure features.csv, labels.csv, and ids.csv are in the same folder as gui.py."
            )
            self.tree = None
            self.id_database = {}
            self.db_status_text = "Database files missing"
            self.db_status_color = self.colors["warning"]

        except ValueError:
            messagebox.showwarning(
                "Configuration Error",
                "Make sure initialize_tree in crud.py is returning BOTH the tree and id_database!"
            )
            self.tree = None
            self.id_database = {}
            self.db_status_text = "Configuration issue"
            self.db_status_color = self.colors["danger"]

        self.last_predicted_point = None
        self.last_predicted_label = None

        self.build_layout()
        self.build_predict_tab()
        self.build_search_tab()
        self.build_delete_tab()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("Main.TFrame", background=self.colors["bg"])
        style.configure("Card.TFrame", background=self.colors["panel"])
        style.configure("Header.TFrame", background=self.colors["primary"])

        style.configure("TNotebook", background=self.colors["bg"], borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            padding=(22, 12),
            font=("Segoe UI", 10, "bold"),
            background="#dbe7f2",
            foreground=self.colors["primary_dark"],
            borderwidth=0
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", self.colors["panel"]), ("active", "#edf6fb")],
            foreground=[("selected", self.colors["primary"]), ("active", self.colors["primary_dark"])]
        )

        style.configure(
            "Title.TLabel",
            background=self.colors["primary"],
            foreground="white",
            font=("Segoe UI", 22, "bold")
        )
        style.configure(
            "Subtitle.TLabel",
            background=self.colors["primary"],
            foreground="#d8ecf8",
            font=("Segoe UI", 10)
        )
        style.configure(
            "Section.TLabel",
            background=self.colors["panel"],
            foreground=self.colors["text"],
            font=("Segoe UI", 15, "bold")
        )
        style.configure(
            "Hint.TLabel",
            background=self.colors["panel"],
            foreground=self.colors["muted"],
            font=("Segoe UI", 9)
        )
        style.configure(
            "Form.TLabel",
            background=self.colors["panel"],
            foreground=self.colors["text"],
            font=("Segoe UI", 10, "bold")
        )
        style.configure(
            "Status.TLabel",
            background=self.colors["primary"],
            foreground="white",
            font=("Segoe UI", 9, "bold")
        )

        style.configure(
            "TEntry",
            fieldbackground=self.colors["entry"],
            background=self.colors["entry"],
            foreground=self.colors["text"],
            bordercolor=self.colors["border"],
            lightcolor=self.colors["border"],
            darkcolor=self.colors["border"],
            padding=8
        )
        style.configure(
            "TCombobox",
            fieldbackground=self.colors["entry"],
            background=self.colors["entry"],
            foreground=self.colors["text"],
            bordercolor=self.colors["border"],
            lightcolor=self.colors["border"],
            darkcolor=self.colors["border"],
            padding=8
        )

        style.configure(
            "Primary.TButton",
            background=self.colors["primary"],
            foreground="white",
            font=("Segoe UI", 10, "bold"),
            padding=(18, 10),
            borderwidth=0
        )
        style.map(
            "Primary.TButton",
            background=[("active", self.colors["primary_dark"]), ("pressed", self.colors["primary_dark"])]
        )

        style.configure(
            "Accent.TButton",
            background=self.colors["accent"],
            foreground="white",
            font=("Segoe UI", 10, "bold"),
            padding=(16, 9),
            borderwidth=0
        )
        style.map("Accent.TButton", background=[("active", "#0c8fbd"), ("pressed", "#0c8fbd")])

        style.configure(
            "Danger.TButton",
            background=self.colors["danger"],
            foreground="white",
            font=("Segoe UI", 10, "bold"),
            padding=(16, 9),
            borderwidth=0
        )
        style.map("Danger.TButton", background=[("active", "#962d22"), ("pressed", "#962d22")])

        style.configure(
            "Soft.TButton",
            background="#e8eef5",
            foreground=self.colors["primary_dark"],
            font=("Segoe UI", 10, "bold"),
            padding=(16, 9),
            borderwidth=0
        )
        style.map("Soft.TButton", background=[("active", "#d9e4ef"), ("pressed", "#d9e4ef")])

    def make_scrollable_tab(self, parent):
        canvas = tk.Canvas(parent, bg=self.colors["bg"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas, style="Main.TFrame")

        canvas_window = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

        def configure_scroll_region(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def resize_frame(event):
            canvas.itemconfig(canvas_window, width=event.width)

        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def on_linux_scroll_up(event):
            canvas.yview_scroll(-1, "units")

        def on_linux_scroll_down(event):
            canvas.yview_scroll(1, "units")

        scroll_frame.bind("<Configure>", configure_scroll_region)
        canvas.bind("<Configure>", resize_frame)

        canvas.bind_all("<MouseWheel>", on_mousewheel)
        canvas.bind_all("<Button-4>", on_linux_scroll_up)
        canvas.bind_all("<Button-5>", on_linux_scroll_down)

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        return scroll_frame

    def build_layout(self):
        self.main = ttk.Frame(self.root, style="Main.TFrame")
        self.main.pack(fill="both", expand=True)

        header = ttk.Frame(self.main, style="Header.TFrame")
        header.pack(fill="x")

        header_inner = ttk.Frame(header, style="Header.TFrame")
        header_inner.pack(fill="x", padx=28, pady=22)

        ttk.Label(
            header_inner,
            text="Thyroid Cancer Prediction System",
            style="Title.TLabel"
        ).pack(anchor="w")

        ttk.Label(
            header_inner,
            text="KD-Tree based patient risk evaluation, record search, and secure deletion",
            style="Subtitle.TLabel"
        ).pack(anchor="w", pady=(4, 0))

        status_row = ttk.Frame(header_inner, style="Header.TFrame")
        status_row.pack(anchor="w", pady=(12, 0))

        self.status_dot = tk.Canvas(
            status_row,
            width=12,
            height=12,
            bg=self.colors["primary"],
            highlightthickness=0
        )
        self.status_dot.pack(side="left", padx=(0, 7))
        self.status_dot.create_oval(2, 2, 10, 10, fill=self.db_status_color, outline=self.db_status_color)

        ttk.Label(status_row, text=self.db_status_text, style="Status.TLabel").pack(side="left")

        body = ttk.Frame(self.main, style="Main.TFrame")
        body.pack(fill="both", expand=True, padx=28, pady=22)

        self.notebook = ttk.Notebook(body)
        self.notebook.pack(expand=True, fill="both")

        self.tab_predict_outer = ttk.Frame(self.notebook, style="Main.TFrame")
        self.tab_search_outer = ttk.Frame(self.notebook, style="Main.TFrame")
        self.tab_delete_outer = ttk.Frame(self.notebook, style="Main.TFrame")

        self.notebook.add(self.tab_predict_outer, text="  Predict & Insert  ")
        self.notebook.add(self.tab_search_outer, text="  Search Record  ")
        self.notebook.add(self.tab_delete_outer, text="  Delete Record  ")

        self.tab_predict = self.make_scrollable_tab(self.tab_predict_outer)
        self.tab_search = self.make_scrollable_tab(self.tab_search_outer)
        self.tab_delete = self.make_scrollable_tab(self.tab_delete_outer)

    def make_card(self, parent, padx=26, pady=22):
        card = tk.Frame(
            parent,
            bg=self.colors["panel"],
            highlightbackground=self.colors["border"],
            highlightthickness=1
        )
        card.pack(fill="both", expand=True, padx=padx, pady=pady)
        return card

    def make_result_box(self, parent):
        return tk.Frame(
            parent,
            bg="#f7fafc",
            highlightbackground=self.colors["border"],
            highlightthickness=1
        )

    def normalize_input(self, raw_features):
        means = [
            44.918496786417855,
            0.40041186510007476,
            0.3000832193181658,
            0.14979477269842165,
            0.24927241867309854,
            0.19949598243461172,
            0.3003700203581722,
            5.045102002435458,
            2.5034032469639054
        ]
        stds = [
            21.632763673991214,
            0.48998183985445243,
            0.4582938803887866,
            0.3568701427279254,
            0.4325918167995807,
            0.399621490196737,
            0.4584188818408372,
            2.8602573449788182,
            1.4446273346433622
        ]

        normalized_point = []
        for i in range(9):
            z_score = (raw_features[i] - means[i]) / stds[i]
            normalized_point.append(z_score)

        return normalized_point

    def add_field(self, parent, row, label, key, widget_type="entry", values=None):
        ttk.Label(parent, text=label, style="Form.TLabel").grid(
            row=row,
            column=0,
            sticky="w",
            padx=(0, 14),
            pady=8
        )

        if widget_type == "combo":
            widget = ttk.Combobox(parent, values=values, state="readonly", width=24)
        else:
            widget = ttk.Entry(parent, width=27)

        widget.grid(row=row, column=1, sticky="ew", pady=8)
        self.inputs[key] = widget

    def build_predict_tab(self):
        card = self.make_card(self.tab_predict)
        card.grid_columnconfigure(0, weight=1)
        card.grid_columnconfigure(1, weight=1)

        left = tk.Frame(card, bg=self.colors["panel"])
        left.grid(row=0, column=0, sticky="nsew", padx=(24, 14), pady=22)

        right = tk.Frame(card, bg=self.colors["panel"])
        right.grid(row=0, column=1, sticky="nsew", padx=(14, 24), pady=22)

        ttk.Label(left, text="Patient Details", style="Section.TLabel").pack(anchor="w")
        ttk.Label(
            left,
            text="Fill all fields before generating the prediction.",
            style="Hint.TLabel"
        ).pack(anchor="w", pady=(3, 14))

        input_frame = tk.Frame(left, bg=self.colors["panel"])
        input_frame.pack(fill="x")
        input_frame.grid_columnconfigure(1, weight=1)

        self.inputs = {}

        self.add_field(input_frame, 0, "Age", "age")
        self.add_field(input_frame, 1, "Gender", "gender", "combo", ["Male", "Female"])
        self.add_field(input_frame, 2, "Family History", "Family History", "combo", ["Yes", "No"])
        self.add_field(input_frame, 3, "Radiation Exposure", "Radiation Exposure", "combo", ["Yes", "No"])
        self.add_field(input_frame, 4, "Iodine Deficiency", "Iodine Deficiency", "combo", ["Yes", "No"])
        self.add_field(input_frame, 5, "Smoking", "Smoking", "combo", ["Yes", "No"])
        self.add_field(input_frame, 6, "Obesity", "Obesity", "combo", ["Yes", "No"])
        self.add_field(input_frame, 7, "TSH Level (mIU/L)", "tsh")
        self.add_field(input_frame, 8, "Nodule Size (cm)", "nodule")

        button_row = tk.Frame(left, bg=self.colors["panel"])
        button_row.pack(anchor="w", pady=(18, 0))

        ttk.Button(
            button_row,
            text="Generate Prediction",
            style="Primary.TButton",
            command=self.run_prediction
        ).pack(side="left")

        ttk.Button(
            button_row,
            text="Clear",
            style="Soft.TButton",
            command=self.clear_prediction_tab
        ).pack(side="left", padx=(10, 0))

        ttk.Label(right, text="Prediction Output", style="Section.TLabel").pack(anchor="w")
        ttk.Label(
            right,
            text="Result will appear here after analysis.",
            style="Hint.TLabel"
        ).pack(anchor="w", pady=(3, 14))

        self.result_box = self.make_result_box(right)
        self.result_box.pack(fill="both", expand=True)

        self.lbl_result = tk.Label(
            self.result_box,
            text="No prediction generated yet.\nEnter patient details and click Generate Prediction.",
            bg="#f7fafc",
            fg=self.colors["muted"],
            font=("Segoe UI", 12),
            justify="center",
            wraplength=360
        )
        self.lbl_result.pack(expand=True, fill="both", padx=18, pady=20)

        self.post_prediction_frame = tk.Frame(right, bg=self.colors["panel"])

        self.btn_insert = ttk.Button(
            self.post_prediction_frame,
            text="Confirm & Insert Record",
            style="Accent.TButton",
            command=self.confirm_and_insert
        )
        self.btn_insert.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_clear = ttk.Button(
            self.post_prediction_frame,
            text="Clear Details",
            style="Soft.TButton",
            command=self.clear_prediction_tab
        )
        self.btn_clear.pack(side=tk.LEFT)

    def validate_common_inputs(self):
        required_dropdowns = [
            "gender",
            "Family History",
            "Radiation Exposure",
            "Iodine Deficiency",
            "Smoking",
            "Obesity"
        ]

        for field in required_dropdowns:
            if not self.inputs[field].get():
                messagebox.showerror("Input Error", f"Please select a value for {field}.")
                return False

        return True

    def run_prediction(self):
        if not self.validate_common_inputs():
            return

        try:
            patient_age = float(self.inputs["age"].get())
            tsh_level = float(self.inputs["tsh"].get())
            nodule_size = float(self.inputs["nodule"].get())

            if patient_age <= 0:
                messagebox.showerror("Input Error", "Age must be a positive number greater than 0.")
                return

            if tsh_level < 0 or nodule_size < 0:
                messagebox.showerror("Input Error", "TSH level and nodule size cannot be negative.")
                return

            raw_data = [
                patient_age,
                0.0 if self.inputs["gender"].get() == "Male" else 1.0,
                1.0 if self.inputs["Family History"].get() == "Yes" else 0.0,
                1.0 if self.inputs["Radiation Exposure"].get() == "Yes" else 0.0,
                1.0 if self.inputs["Iodine Deficiency"].get() == "Yes" else 0.0,
                1.0 if self.inputs["Smoking"].get() == "Yes" else 0.0,
                1.0 if self.inputs["Obesity"].get() == "Yes" else 0.0,
                tsh_level,
                nodule_size
            ]

            normalized_point = self.normalize_input(raw_data)

        except ValueError:
            messagebox.showerror(
                "Input Error",
                "Please enter valid numbers for Age, TSH Level, and Nodule Size."
            )
            return

        if not self.tree:
            messagebox.showerror("Error", "KD-Tree is not initialized.")
            return

        neighbors = get_knn(self.tree, normalized_point, k=5)
        closest_distance = neighbors[0]["distance"]

        if closest_distance > 4.0:
            self.lbl_result.config(
                text=(
                    "⚠ Extreme Outlier Detected\n\n"
                    "Patient data cannot be safely matched with known database profiles.\n"
                    "Please review manually before making a decision."
                ),
                fg=self.colors["danger"],
                font=("Segoe UI", 12, "bold")
            )

            self.last_predicted_point = normalized_point
            self.last_predicted_label = None
            self.btn_insert.pack_forget()
            self.post_prediction_frame.pack(pady=(14, 0), anchor="w")
            return

        prediction = generate_prediction(neighbors)
        cancer_risk, tumor_diagnosis = deduce_label(prediction)
        confidence = confidence_score(neighbors, prediction)

        self.lbl_result.config(
            text=(
                "Prediction Complete\n\n"
                f"Risk Level: {cancer_risk}\n"
                f"Diagnosis: {tumor_diagnosis}\n"
                f"Confidence: {confidence:.2f}%"
            ),
            fg=self.colors["primary"],
            font=("Segoe UI", 13, "bold")
        )

        self.last_predicted_point = normalized_point
        self.last_predicted_label = prediction

        self.btn_insert.pack_forget()
        self.btn_insert.pack(side=tk.LEFT, padx=(0, 10))
        self.post_prediction_frame.pack(pady=(14, 0), anchor="w")

    def confirm_and_insert(self):
        if self.last_predicted_point and self.last_predicted_label is not None:
            new_id = max(self.id_database.keys()) + 1 if self.id_database else 1
            self.id_database[new_id] = self.last_predicted_point

            self.tree = insert_record(
                tree=self.tree,
                point=self.last_predicted_point,
                label=self.last_predicted_label,
                new_id=new_id,
                features_file=self.features_path,
                labels_file=self.labels_path,
                ids_file=self.ids_path
            )

            messagebox.showinfo(
                "Success",
                f"Patient record permanently saved!\nAssigned Patient ID: {new_id}"
            )
            self.clear_prediction_tab()

        else:
            messagebox.showerror(
                "Insert Error",
                "Please generate a valid prediction before inserting a record."
            )

    def clear_prediction_tab(self):
        for key in ["age", "tsh", "nodule"]:
            self.inputs[key].delete(0, tk.END)

        for key in [
            "gender",
            "Family History",
            "Radiation Exposure",
            "Iodine Deficiency",
            "Smoking",
            "Obesity"
        ]:
            self.inputs[key].set("")

        self.lbl_result.config(
            text="No prediction generated yet.\nEnter patient details and click Generate Prediction.",
            fg=self.colors["muted"],
            font=("Segoe UI", 12)
        )

        self.post_prediction_frame.pack_forget()
        self.last_predicted_point = None
        self.last_predicted_label = None

    def build_search_tab(self):
        card = self.make_card(self.tab_search, padx=90, pady=45)

        content = tk.Frame(card, bg=self.colors["panel"])
        content.pack(expand=True, fill="both", padx=40, pady=35)

        ttk.Label(content, text="Search Patient Record", style="Section.TLabel").pack(anchor="w")
        ttk.Label(
            content,
            text="Enter a Patient ID to retrieve the saved diagnosis.",
            style="Hint.TLabel"
        ).pack(anchor="w", pady=(3, 20))

        form = tk.Frame(content, bg=self.colors["panel"])
        form.pack(anchor="w")

        ttk.Label(form, text="Patient ID", style="Form.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 12))
        self.search_entry = ttk.Entry(form, width=30)
        self.search_entry.grid(row=0, column=1, sticky="ew")

        ttk.Button(
            form,
            text="Search Database",
            style="Primary.TButton",
            command=self.run_search
        ).grid(row=0, column=2, padx=(12, 0))

        self.search_result_box = self.make_result_box(content)
        self.search_result_box.pack(fill="x", pady=(28, 0))

        self.lbl_search_result = tk.Label(
            self.search_result_box,
            text="Search result will appear here.",
            bg="#f7fafc",
            fg=self.colors["muted"],
            font=("Segoe UI", 12),
            justify="center"
        )
        self.lbl_search_result.pack(fill="x", padx=20, pady=28)

    def run_search(self):
        try:
            target_id = int(self.search_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid numeric Patient ID.")
            return

        if target_id in self.id_database:
            target_point = self.id_database[target_id]
            node = search(self.tree, target_point)

            if node:
                risk, diagnosis = deduce_label(node["label"])
                self.lbl_search_result.config(
                    text=(
                        "Record Found\n\n"
                        f"Patient ID: {target_id}\n"
                        f"Diagnosis on File: {risk} Risk, {diagnosis}"
                    ),
                    fg=self.colors["success"],
                    font=("Segoe UI", 12, "bold")
                )
            else:
                self.lbl_search_result.config(
                    text="Tree inconsistency: coordinates found, but node is missing.",
                    fg=self.colors["danger"],
                    font=("Segoe UI", 12, "bold")
                )
        else:
            self.lbl_search_result.config(
                text=f"Patient ID {target_id} does not exist.",
                fg=self.colors["danger"],
                font=("Segoe UI", 12, "bold")
            )

    def build_delete_tab(self):
        card = self.make_card(self.tab_delete, padx=90, pady=45)

        content = tk.Frame(card, bg=self.colors["panel"])
        content.pack(expand=True, fill="both", padx=40, pady=35)

        ttk.Label(content, text="Delete Patient Record", style="Section.TLabel").pack(anchor="w")
        ttk.Label(
            content,
            text="Enter a Patient ID to remove it from the system.",
            style="Hint.TLabel"
        ).pack(anchor="w", pady=(3, 20))

        warning = tk.Frame(
            content,
            bg="#fff7ed",
            highlightbackground="#fed7aa",
            highlightthickness=1
        )
        warning.pack(fill="x", pady=(0, 22))

        tk.Label(
            warning,
            text="Deletion removes the selected patient from the active KD-Tree and the reference dictionary.",
            bg="#fff7ed",
            fg="#9a3412",
            font=("Segoe UI", 10, "bold"),
            anchor="w"
        ).pack(fill="x", padx=16, pady=12)

        form = tk.Frame(content, bg=self.colors["panel"])
        form.pack(anchor="w")

        ttk.Label(form, text="Patient ID", style="Form.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 12))
        self.delete_entry = ttk.Entry(form, width=30)
        self.delete_entry.grid(row=0, column=1, sticky="ew")

        ttk.Button(
            form,
            text="Delete Record",
            style="Danger.TButton",
            command=self.run_delete
        ).grid(row=0, column=2, padx=(12, 0))

        self.delete_result_box = self.make_result_box(content)
        self.delete_result_box.pack(fill="x", pady=(28, 0))

        self.lbl_delete_result = tk.Label(
            self.delete_result_box,
            text="Deletion status will appear here.",
            bg="#f7fafc",
            fg=self.colors["muted"],
            font=("Segoe UI", 12),
            justify="center"
        )
        self.lbl_delete_result.pack(fill="x", padx=20, pady=28)

    def run_delete(self):
        try:
            target_id = int(self.delete_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid numeric Patient ID.")
            return

        if target_id in self.id_database:
            target_point = self.id_database[target_id]
            self.tree = delete(self.tree, target_point)
            del self.id_database[target_id]

            self.lbl_delete_result.config(
                text=f"Patient ID {target_id} successfully deleted from all active records.",
                fg=self.colors["success"],
                font=("Segoe UI", 12, "bold")
            )
            self.delete_entry.delete(0, tk.END)
        else:
            self.lbl_delete_result.config(
                text=f"Cannot delete. Patient ID {target_id} does not exist.",
                fg=self.colors["danger"],
                font=("Segoe UI", 12, "bold")
            )


if __name__ == "__main__":
    root = tk.Tk()
    app = ThyroidPredictionApp(root)
    root.mainloop()
