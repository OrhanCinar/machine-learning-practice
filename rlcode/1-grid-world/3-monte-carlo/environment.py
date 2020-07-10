import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image

np.random.seed(1)
PhotoImage = ImageTk.PhotoImage
UNIT = 100
HEIGHT = 5
WIDTH = 5


class Env(tk.Tk):
    def __init__(self):
        super(Env, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('Monte Carlo')
        self.geometry(f'{HEIGHT*UNIT}x{HEIGHT * UNIT}')
        self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        self.texts = []

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white', height=HEIGHT *
                           UNIT, width=WIDTH * UNIT)

        for c in range(0, WIDTH * UNIT, UNIT):
            x0, y0, x1, y2 = c, 0, c, HEIGHT*UNIT
            canvas.create_line(x0, y0, x1, y1)

        for r in range(0, HEIGHT * UNIT, UNIT):
            x0, y0, x1, y2 = r, 0, r, HEIGHT*UNIT
            canvas.create_line(x0, y0, x1, y1)

        self.rectangle = canvas.create_image(50, 50, image=self.shapes[0])
        self.triangle1 = canvas.create_image(250, 150, image=self.shapes[1])
        self.triangle2 = canvas.create_image(150, 250, image=self.shapes[2])
        self.circle = canvas.create_image(250, 250, image=self.shapes[3])

        canvas.pack()
        return canvas

    def load_images(self):
        rectangle = PhotoImage(Image.open(
            "../img/rectangle.png").resize((65, 65)))
        triangle = PhotoImage(Image.open(
            "../img/triangle.png").resize((65, 65)))
        circle = PhotoImage(Image.open("../img/circle.png").resize((65, 65)))
        return rectangle, triangle, circle
