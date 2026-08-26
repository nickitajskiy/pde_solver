import tkinter as tk
from tkinter import ttk

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

notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True, padx=8, pady=8)

classification_tab = ttk.Frame(notebook, style="Win95.TFrame")
notebook.add(classification_tab, text="КЛАССИФИКАЦИЯ")

root.mainloop()