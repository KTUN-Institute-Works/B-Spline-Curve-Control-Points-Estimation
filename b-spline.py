import os
import tkinter as tk
import numpy as np

from model import CPClassifier

root = tk.Tk()
root.title("B-Spline Eğrisi")

canvas = tk.Canvas(root, width=600, height=600, bg="white")
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

canvas2 = tk.Canvas(root, width=600, height=600, bg="white")
canvas2.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

k_var = tk.IntVar(value=3)
knot_type_var = tk.StringVar(value="clamped")  # "clamped" veya "uniform"

dot_positions = []
dots = []

cpc = CPClassifier()
cpc.set_inout_data()
cpc.create_model()

def blending_function(u, i, k, knot_vector) -> float:
    if k == 1:
        if knot_vector[i] <= u < knot_vector[i + 1]:
            return 1.0
        else:
            return 0.0

    else:

        denom1 = knot_vector[i + k - 1] - knot_vector[i]
        denom2 = knot_vector[i + k] - knot_vector[i + 1]

        term1 = 0.0
        term2 = 0.0

        if denom1 != 0:
            term1 = (u - knot_vector[i]) / denom1 * blending_function(u, i, k - 1, knot_vector)
        if denom2 != 0:
            term2 = (knot_vector[i + k] - u) / denom2 * blending_function(u, i + 1, k - 1, knot_vector)

        return term1 + term2


def draw_bspline():
    k = k_var.get()
    n = len(dot_positions) - 1
    if n < k:
        return

    elif n == k:
        # Burada bezier dönmesi gerek ama şimdilik kalsın
        return

    m = n + k + 1

    clamped_knot_vector = [0] * k + list(range(1, n - k + 2)) + [n - k + 2] * k

    uniform_knot_vector = list(range(m))

    if knot_type_var.get() == "clamped":
        knot_vector = clamped_knot_vector
    elif knot_type_var.get() == "uniform":
        knot_vector = uniform_knot_vector
    else:
        return

    print(clamped_knot_vector)
    u_vector = np.linspace(0, knot_vector[-1], 100)

    for u in u_vector:
        x = 0
        y = 0
        for i in range(n+1):
            blend_func = blending_function(u, i, k, knot_vector)
            x += blend_func * dot_positions[i][0]
            y += blend_func * dot_positions[i][1]

        dots.append([x, y])

    print(dot_positions)

    print(dots)

    dots.pop(-1)

    for i in range(1, len(dots)):
        canvas.create_line(dots[i - 1][0], dots[i - 1][1],
                           dots[i][0], dots[i][1], fill="blue")


def complete():
    draw_bspline()

def clear():
    dot_positions.clear()
    canvas.delete("all")
    # draw_button.config(state=tk.DISABLED)


def on_canvas_click(event):
    x = event.x
    y = event.y
    dot_positions.append((x, y))
    canvas.delete("all")
    for dp in dot_positions:
        canvas.create_oval(dp[0] - 5, dp[1] - 5, dp[0] + 5, dp[1] + 5, fill="red", outline="red")


def predict():
    input_data = [int(coord) for dot in dots for coord in dot]
    print(input_data)
    valid_points = cpc.predict(input_data)
    print(valid_points)

    for dp in valid_points:
        canvas2.create_oval(dp[0] - 5, dp[1] - 5, dp[0] + 5, dp[1] + 5, fill="red", outline="red")

    dots.clear()

    k = k_var.get()
    n = len(dot_positions) - 1
    if n < k:
        return

    elif n == k:
        # Burada bezier dönmesi gerek ama şimdilik kalsın
        return

    m = n + k + 1

    clamped_knot_vector = [0] * k + list(range(1, n - k + 2)) + [n - k + 2] * k

    uniform_knot_vector = list(range(m))

    if knot_type_var.get() == "clamped":
        knot_vector = clamped_knot_vector
    elif knot_type_var.get() == "uniform":
        knot_vector = uniform_knot_vector
    else:
        return

    print(clamped_knot_vector)
    u_vector = np.linspace(0, knot_vector[-1], 100)

    for u in u_vector:
        x = 0
        y = 0
        for i in range(n+1):
            blend_func = blending_function(u, i, k, knot_vector)
            x += blend_func * valid_points[i][0]
            y += blend_func * valid_points[i][1]

        dots.append([x, y])

    print(valid_points)

    print(dots)

    dots.pop(-1)

    for i in range(1, len(dots)):
        canvas2.create_line(dots[i - 1][0], dots[i - 1][1],
                           dots[i][0], dots[i][1], fill="blue")

control_frame = tk.Frame(root)
control_frame.pack(pady=10)

tk.Label(control_frame, text="k (Derece):").pack(side=tk.LEFT)
tk.Spinbox(control_frame, from_=1, to=10, textvariable=k_var, width=5).pack(side=tk.LEFT, padx=5)

tk.Button(control_frame, text="Tamamla", command=complete).pack(side=tk.LEFT, padx=10)

tk.Button(control_frame, text="Temizle", command=clear).pack(side=tk.LEFT, padx=10)

tk.Button(control_frame, text="Tahmin", command=predict).pack(side=tk.LEFT, padx=10)

canvas.bind("<Button-1>", on_canvas_click)

root.mainloop()
