"""Tkinter-интерфейс учебного решателя УЧП. Запуск: python main.py."""

import tkinter as tk
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from solvers import (
    HEAT_METHODS, calculate_max_error, classify_pde, exact_heat_solution,
    exact_wave_solution, heat_convergence, solve_heat, solve_laplace, solve_wave,
)

BACKGROUND = "#c0c0c0"


def label(parent, text, **options):
    return tk.Label(parent, text=text, background=BACKGROUND, foreground="black", **options)


def button(parent, text, command):
    return tk.Button(parent, text=text, command=command, background=BACKGROUND,
                     foreground="black", activebackground="#dfdfdf", relief="raised",
                     borderwidth=2, padx=8, pady=4)


def fields(parent, specifications):
    """Поля формы: (ключ, подпись, начальное значение)."""
    entries = {}
    for row, (key, title, default) in enumerate(specifications):
        label(parent, title).grid(row=row, column=0, sticky="w", padx=4, pady=4)
        entry = ttk.Entry(parent, width=13)
        entry.insert(0, str(default))
        entry.grid(row=row, column=1, padx=4, pady=4)
        entries[key] = entry
    return entries


def read_numbers(entries):
    try:
        return {key: float(entry.get().replace(",", ".")) for key, entry in entries.items()}
    except ValueError as error:
        raise ValueError("Введите числа во все поля параметров.") from error


class PlotPanel:
    """Один встроенный Figure, переиспользуемый при каждом расчёте."""

    def __init__(self, parent):
        self.figure = Figure(figsize=(6.5, 4.5), dpi=100, layout="constrained")
        self.axes = self.figure.add_subplot()
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, parent, pack_toolbar=False)
        self.toolbar.pack(fill="x")
        self.colorbar = None

    def clear(self):
        if self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None
        self.axes.clear()
        self.toolbar.update()

    def draw(self):
        self.canvas.draw_idle()


class EvolutionTab:
    """Общая форма теплопроводности и струны; расчёт хранит все слои u[n,i]."""

    def __init__(self, notebook, wave=False):
        self.wave = wave
        self.solutions = []
        self.parameters = None
        self.frame = ttk.Frame(notebook)
        notebook.add(self.frame, text="СТРУНА" if wave else "ТЕПЛОПРОВОДНОСТЬ")
        equation = ("u_tt = c²·u_xx;  u(0,t)=u(L,t)=0;  u(x,0)=A·sin(πx/L);  u_t(x,0)=0"
                    if wave else "u_t = a·u_xx;  0 ≤ x ≤ 1;  u(0,t)=u(1,t)=0;  u(x,0)=sin(2πx)")
        label(self.frame, equation, anchor="w").pack(fill="x", padx=12, pady=10)
        content = ttk.Frame(self.frame)
        content.pack(fill="both", expand=True, padx=10)
        controls = ttk.Frame(content)
        controls.pack(side="left", fill="y", padx=(0, 10))
        specifications = [("c", "c — скорость", 1), ("length", "L — длина", 1),
                          ("amplitude", "A — амплитуда", 1)] if wave else [("a", "a — коэффициент", 0.1)]
        specifications += [("final_time", "T — время", 1 if wave else 0.2),
                           ("h", "h — шаг x", 0.05), ("tau", "tau — шаг t", 0.01)]
        self.entries = fields(controls, specifications)
        self.method = tk.StringVar(value=HEAT_METHODS[0])
        row = len(specifications)
        if not wave:
            label(controls, "Схема:").grid(row=row, column=0, sticky="w", pady=4)
            ttk.Combobox(controls, values=HEAT_METHODS, textvariable=self.method,
                         state="readonly", width=20).grid(row=row + 1, column=0, columnspan=2, pady=4)
            row += 2
        button(controls, "РАССЧИТАТЬ", self.calculate).grid(row=row, column=0, columnspan=2, pady=8)
        if not wave:
            button(controls, "СРАВНИТЬ 3 СХЕМЫ", lambda: self.calculate(compare=True)).grid(
                row=row + 1, column=0, columnspan=2, pady=4)
            button(controls, "ИССЛЕДОВАТЬ ОШИБКУ", self.study).grid(
                row=row + 2, column=0, columnspan=2, pady=4)
        label(controls, "h делит длину, tau делит T.\nМинимум 4 интервала по x.\n"
              + ("Устойчивость: c·tau/h ≤ 1" if wave else "Явная схема: a·tau/h² ≤ 0.5"),
              justify="left").grid(row=row + 3, column=0, columnspan=2, pady=12, sticky="w")
        plot_frame = ttk.Frame(content)
        plot_frame.pack(side="left", fill="both", expand=True)
        self.plot = PlotPanel(plot_frame)
        self.status = label(self.frame, "Введите параметры и нажмите РАССЧИТАТЬ.", anchor="w", justify="left")
        self.status.pack(fill="x", padx=12, pady=6)
        self.layer = tk.Scale(self.frame, from_=0, to=1, orient="horizontal", resolution=1,
                              command=self.draw_layer, background=BACKGROUND, foreground="black",
                              highlightthickness=0, label="Временной слой n", state="disabled")
        self.layer.pack(fill="x", padx=12, pady=(0, 8))

    def reset(self):
        self.solutions = []
        self.parameters = None
        self.layer.configure(state="disabled")
        self.status.config(text="Результат: —")
        self.plot.clear()
        self.plot.draw()

    def calculate(self, compare=False):
        self.reset()
        try:
            parameters = read_numbers(self.entries)
            if self.wave:
                x, t, u, ratio = solve_wave(**parameters)
                solutions = [("Численное", u)]
            else:
                methods = HEAT_METHODS if compare else (self.method.get(),)
                solutions = []
                for method in methods:
                    x, t, u, ratio = solve_heat(**parameters, method=method)
                    solutions.append((method, u))
        except ValueError as error:
            messagebox.showerror("Ошибка расчёта", str(error), parent=self.frame)
            return
        self.parameters = parameters
        self.x, self.t, self.ratio = x, t, ratio
        self.solutions = solutions
        self.layer.configure(state="normal", to=len(t) - 1)
        self.layer.set(len(t) - 1)
        self.draw_layer(len(t) - 1)

    def draw_layer(self, value):
        if not self.solutions:
            return
        n = min(int(float(value)), len(self.t) - 1)
        p = self.parameters
        if self.wave:
            exact = exact_wave_solution(self.x, self.t[n], p["c"], p["length"], p["amplitude"])
        else:
            exact = exact_heat_solution(self.x, self.t[n], p["a"])
        self.plot.clear()
        ax = self.plot.axes
        ax.plot(self.x, exact, "k--", label="Точное", linewidth=2)
        errors = []
        for name, u in self.solutions:
            ax.plot(self.x, u[n], marker=".", markersize=4, label=name)
            errors.append(f"{name}: {calculate_max_error(u[n], exact):.4g}")
        ax.set(xlabel="x", ylabel="u(x,t)", title=f"t = {self.t[n]:.6g}; слой {n} / {len(self.t) - 1}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        ratio_name = "c·tau/h" if self.wave else "r = a·tau/h²"
        self.status.config(text=f"{ratio_name} = {self.ratio:.6g}; сетка {len(self.t)} × {len(self.x)}\n"
                                "Макс. ошибка по узлам выбранного слоя: " + "; ".join(errors))
        self.plot.draw()

    def study(self):
        try:
            rows = heat_convergence(**read_numbers(self.entries), method=self.method.get())
        except ValueError as error:
            messagebox.showerror("Ошибка исследования", str(error), parent=self.frame)
            return
        window = tk.Toplevel(self.frame)
        window.title("Исследование ошибки: " + self.method.get())
        window.geometry("760x510")
        window.configure(background=BACKGROUND)
        label(window, "При каждом сгущении: h → h/2, tau → tau/4; r сохраняется. Ошибка при t=T.").pack(pady=8)
        table = ttk.Treeview(window, columns=("h", "tau", "r", "error"), show="headings", height=3)
        for column, title in zip(("h", "tau", "r", "error"), ("h", "tau", "r", "Макс. ошибка")):
            table.heading(column, text=title)
            table.column(column, width=150, anchor="center")
        for row in rows:
            table.insert("", "end", values=[f"{value:.6g}" for value in row])
        table.pack(fill="x", padx=12)
        panel = PlotPanel(window)
        panel.axes.loglog([r[0] for r in rows], [max(r[3], 1e-16) for r in rows], "o-")
        panel.axes.set(xlabel="h", ylabel="Макс. ошибка при t=T", title="Совместное сгущение сетки")
        panel.axes.grid(True, which="both", alpha=0.3)
        panel.draw()
        # Сохраняем ссылку, пока открыто окно с Tk-canvas.
        window.plot_panel = panel


class LaplaceTab:
    def __init__(self, notebook):
        self.frame = ttk.Frame(notebook)
        notebook.add(self.frame, text="ЛАПЛАС")
        label(self.frame, "u_xx + u_yy = 0; единичный квадрат; одинаковые шаги по x и y.").pack(pady=10)
        content = ttk.Frame(self.frame)
        content.pack(fill="both", expand=True, padx=10)
        controls = ttk.Frame(content)
        controls.pack(side="left", fill="y", padx=(0, 10))
        self.entries = fields(controls, [
            ("h", "h — шаг сетки", 0.05), ("left", "Слева", 0), ("right", "Справа", 0),
            ("bottom", "Снизу", 0), ("top", "Сверху", 100), ("epsilon", "ε — допуск невязки", "1e-6"),
            ("max_iterations", "Предел итераций", 10000),
        ])
        button(controls, "РАССЧИТАТЬ", self.calculate).grid(row=7, column=0, columnspan=2, pady=10)
        label(controls, "Метод Гаусса—Зейделя.\nВ углах — среднее\nсмежных температур.", justify="left").grid(
            row=8, column=0, columnspan=2, sticky="w", pady=10)
        plot_frame = ttk.Frame(content)
        plot_frame.pack(side="left", fill="both", expand=True)
        self.plot = PlotPanel(plot_frame)
        self.status = label(self.frame, "Результат: —", anchor="w")
        self.status.pack(fill="x", padx=12, pady=10)

    def calculate(self):
        self.plot.clear()
        self.plot.draw()
        self.status.config(text="Результат: —")
        try:
            p = read_numbers(self.entries)
            limit = p["max_iterations"]
            if not limit.is_integer():
                raise ValueError("Предел итераций должен быть целым числом.")
            p["max_iterations"] = int(limit)
            x, y, u, iterations, residual = solve_laplace(**p)
        except (ValueError, OverflowError) as error:
            messagebox.showerror("Ошибка расчёта", str(error), parent=self.frame)
            return
        image = self.plot.axes.pcolormesh(x, y, u, shading="auto", cmap="inferno")
        self.plot.colorbar = self.plot.figure.colorbar(image, ax=self.plot.axes, label="Температура u")
        self.plot.axes.set(xlabel="x", ylabel="y", title="Стационарное распределение температуры", aspect="equal")
        self.status.config(text=f"Сетка {len(y)} × {len(x)}; итераций: {iterations}; "
                                f"макс. |Δₕu| = {residual:.4g} ≤ {p['epsilon']:g}")
        self.plot.draw()


def classification_tab(notebook):
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="КЛАССИФИКАЦИЯ")
    label(frame, "A·∂²u/∂x² + B·∂²u/∂x∂y + C·∂²u/∂y² + … = 0",
          font=("TkDefaultFont", 12, "bold")).pack(anchor="w", padx=16, pady=16)
    form = ttk.Frame(frame)
    form.pack(anchor="w", padx=16)
    entries = fields(form, [("coef_a", "A:", 1), ("coef_b", "B:", 0), ("coef_c", "C:", 1)])
    result = label(frame, "Δ = —\nТип: —\nПример: —", justify="left", anchor="w")
    result.pack(fill="x", padx=20, pady=20)

    def classify():
        result.config(text="Δ = —\nТип: —\nПример: —")
        try:
            discriminant, kind = classify_pde(**read_numbers(entries))
        except ValueError as error:
            messagebox.showerror("Ошибка классификации", str(error), parent=frame)
            return
        examples = {"ЭЛЛИПТИЧЕСКИЙ": "уравнение Лапласа", "ПАРАБОЛИЧЕСКИЙ": "уравнение теплопроводности",
                    "ГИПЕРБОЛИЧЕСКИЙ": "волновое уравнение"}
        result.config(text=f"Δ = {discriminant:.6g}\nТип: {kind}\nПример: {examples[kind]}")

    button(form, "ОПРЕДЕЛИТЬ ТИП", classify).grid(row=3, column=0, columnspan=2, pady=10)
    label(frame, "Δ = B² − 4AC:  Δ < 0 — эллиптический;  Δ = 0 — параболический;  Δ > 0 — гиперболический.\n"
          "Классификация локальная, по старшим производным; смешанный член записан с B, без множителя 2.",
          justify="left").pack(anchor="w", padx=20, pady=8)


def create_app():
    root = tk.Tk()
    root.title("Численное решение УЧП")
    root.geometry("1120x720")
    root.minsize(1000, 650)
    root.configure(background=BACKGROUND)
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TFrame", background=BACKGROUND)
    style.configure("TNotebook", background=BACKGROUND, borderwidth=2)
    style.configure("TNotebook.Tab", background=BACKGROUND, foreground="black", padding=(12, 6))
    style.map("TNotebook.Tab", background=[("selected", "#dfdfdf")])
    style.configure("TEntry", fieldbackground="white", foreground="black", borderwidth=2)
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=8, pady=8)
    classification_tab(notebook)
    root.heat_tab = EvolutionTab(notebook)
    root.wave_tab = EvolutionTab(notebook, wave=True)
    root.laplace_tab = LaplaceTab(notebook)
    return root


if __name__ == "__main__":
    create_app().mainloop()
