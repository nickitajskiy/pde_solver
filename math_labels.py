"""LaTeX-подобные надписи для Tkinter средствами Matplotlib MathText."""

import base64
from functools import lru_cache
from io import BytesIO
import tkinter as tk

from matplotlib import rc_context
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import math_to_image


@lru_cache(maxsize=128)
def render_formula(text, size, dpi):
    """PNG с прозрачным фоном; системная установка LaTeX не требуется."""
    buffer = BytesIO()
    font = FontProperties(family="DejaVu Sans", size=size, math_fontfamily="cm")
    with rc_context({"savefig.transparent": True}):
        math_to_image(text, buffer, prop=font, dpi=dpi, format="png", color="black")
    return buffer.getvalue()


class MathLabel(tk.Label):
    """Обычная Tk-надпись с формулой; ссылка на PhotoImage хранится в виджете."""

    def __init__(self, parent, text, size=14, **options):
        super().__init__(parent, **options)
        self.size = size
        self.set_formula(text)

    def set_formula(self, text):
        self.formula = text
        dpi = max(96, round(self.winfo_fpixels("1i")))
        png = render_formula(text, self.size, dpi)
        self.formula_image = tk.PhotoImage(master=self, data=base64.b64encode(png), format="png")
        self.configure(image=self.formula_image)


def format_number(value, precision=4, *, latex=False):
    """Научная запись с показателем степени вместо технического 1e-06."""
    text = f"{value:.{precision}g}"
    if "e" not in text:
        return text
    mantissa, exponent = text.split("e")
    exponent = str(int(exponent))
    if latex:
        return rf"{mantissa}\times 10^{{{exponent}}}"
    superscripts = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"{mantissa} × 10{exponent.translate(superscripts)}"
