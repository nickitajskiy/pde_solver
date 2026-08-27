def classify_pde(coef_a, coef_b, coef_c):
    tolerance = 1e-12

    if (
        abs(coef_a) < tolerance 
        and abs(coef_b) < tolerance
        and abs(coef_c) < tolerance
    ):
        raise ValueError("Все коэффициенты второго порядка равны нулю")

    discriminant = (
        coef_b ** 2 - 4 * coef_a * coef_c
    )

    if discriminant < -tolerance:
        pde_type = "ЭЛЛИПТИЧЕСКИЙ"
    elif discriminant > tolerance:
        pde_type = "ГИПЕРБОЛИЧЕСКИЙ"
    else:
        discriminant = 0.0
        pde_type = "ПАРАБОЛИЧЕСКИЙ"

    return discriminant, pde_type