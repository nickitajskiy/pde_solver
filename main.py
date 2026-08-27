import tkinter as tk
from tkinter import messagebox, ttk
from solvers import classify_pde

root = tk.Tk()

root.title("Численное решение УЧП")
root.geometry("760x500")
root.minsize(600, 400)

window_background = "#c0c0c0"
root.configure(background=window_background)

style = ttk.Style()
style.theme_use("clam")

style.configure(
    "TNotebook",
    background=window_background,
    borderwidth=2,
)

style.configure(
    "TNotebook.Tab",
    background=window_background,
    foreground="black",
    padding=(10, 4),
)

style.map(
    "TNotebook.Tab",
    background=[("selected", "#dfdfdf")],    
)

style.configure(
    "Win95.TFrame",
    background=window_background,
)

style.configure(
    "Win95.TEntry",
    fieldbackground="white",
    foreground="black",
    borderwidth=2,
)

notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True, padx=8, pady=8)

classification_tab = ttk.Frame(notebook, style="Win95.TFrame")
notebook.add(classification_tab, text="КЛАССИФИКАЦИЯ")

equation_label = tk.Label(
    classification_tab,
    text="A·∂²u/∂x² + B·∂²u/∂x∂y + C·∂²u/∂y² + … = 0",
    background=window_background,
    foreground="black",
    font=("TkDefaultFont", 10, "bold"),
)

equation_label.pack(anchor="w", padx=16, pady=(16, 8))

input_frame = tk.LabelFrame(
    classification_tab,
    text=" Коэффициенты второго порядка ",
    background=window_background,
    foreground="black",
    relief="groove",
    borderwidth=2,
    padx=12,
    pady=12,
)

input_frame.pack(anchor="w", padx=16, pady=8)

coef_a_label = tk.Label(
    input_frame,
    text="A:",
    background=window_background,
    foreground="black",
)

coef_a_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)

coef_a_entry = ttk.Entry(
    input_frame,
    width=15,
    style="Win95.TEntry",
)

coef_a_entry.grid(row=0, column=1, padx=5, pady=5)

coef_b_label = tk.Label(
    input_frame,
    text="B:",
    background=window_background,
    foreground="black",
)

coef_b_label.grid(row=1, column=0, sticky="w", padx=5, pady=5)

coef_b_entry = ttk.Entry(
    input_frame,
    width=15,
    style="Win95.TEntry",
)

coef_b_entry.grid(row=1, column=1, padx=5, pady=5)

coef_c_label = tk.Label(
    input_frame,
    text="C:",
    background=window_background,
    foreground="black",
)

coef_c_label.grid(row=2, column=0, sticky="w", padx=5, pady=5)

coef_c_entry = ttk.Entry(
    input_frame,
    width=15,
    style="Win95.TEntry",
)

coef_c_entry.grid(row=2, column=1, padx=5, pady=5)

result_frame = tk.LabelFrame(
    classification_tab,
    text=" Результат ",
    background=window_background,
    foreground="black",
    relief="groove",
    borderwidth=2,
    padx=12,
    pady=12,
)

result_frame.pack(
    anchor="w",
    fill="x",
    padx=16,
    pady=8,
)

discriminant_result_label = tk.Label(
    result_frame,
    text="Δ = —",
    background=window_background,
    foreground="black",
)

discriminant_result_label.pack(
    anchor="w",
    pady=2,
)

pde_type_result_label = tk.Label(
    result_frame,
    text="Тип: —",
    background=window_background,
    foreground="black",
    font=("TkDefaultFont", 10, "bold"),
)

pde_type_result_label.pack(anchor="w", pady=2)

example_result_label = tk.Label(
    result_frame,
    text="Пример: —",
    background=window_background,
    foreground="black",
)

example_result_label.pack(anchor="w", pady=2)

def handle_classification():
    discriminant_result_label.config(text="Δ = —")
    pde_type_result_label.config(text="Тип: —")
    example_result_label.config(text="Пример: —")

    try:
        coef_a = float(coef_a_entry.get())
        coef_b = float(coef_b_entry.get())
        coef_c = float(coef_c_entry.get())
    except ValueError:
        messagebox.showerror(
            "Ошибка ввода", 
            "Введите числовые значения A, B и C",
            parent=root,
            )
        return

    try:
        discriminant, pde_type = classify_pde(
            coef_a,
            coef_b,
            coef_c,
        )
    except ValueError as error:
        messagebox.showerror(
            "Ошибка классификации",
            str(error),
            parent=root,
            )
        return

    examples = {
        "ЭЛЛИПТИЧЕСКИЙ": "уравнение Лапласа",
        "ПАРАБОЛИЧЕСКИЙ": "уравнение теплопроводности",
        "ГИПЕРБОЛИЧЕСКИЙ": "волновое уравнение",
    }

    discriminant_result_label.config(
        text=f"Δ = {discriminant:g}"
    )

    pde_type_result_label.config(
        text=f"Тип: {pde_type}"
    )

    example_result_label.config(
        text=f"Пример: {examples[pde_type]}"
    )
    
classification_button = tk.Button(
    input_frame,
    text="ОПРЕДЕЛИТЬ ТИП",
    command=handle_classification,
    background=window_background,
    foreground="black",
    activebackground="#dfdfdf",
    relief="raised",
    borderwidth=2,
    padx=10,
    pady=4,
)

classification_button.grid(
    row=3,
    column=0,
    columnspan=2,
    pady=(12, 5),
)
root.mainloop()