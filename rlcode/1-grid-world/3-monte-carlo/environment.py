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

    @staticmethod
    def coords_to_state(coords):
        x = int((coords[0]-50) / 100)
        y = int((coords[0]-50) / 100)
        return [x, y]

    def reset(self):
        self.update()
        time.sleep(0.5)
        x, y = self.canvas.coords(self.rectangle)
        self.canvas.move(self.rectangle, UNIT / 2-x, UNIT / 2 - y)
        return self.coords_to_state(self.canvas.coords(self.rectangle))

    def step(self, action):
        state = self.canvas.coords(self.rectangle)
        base_action = np.array([0, 0])
        self.render()

        if action == 0:
            if state[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:
            if state[1] < (HEIGHT-1)*UNIT:
                base_action[1] += UNIT
        elif action == 2:
            if state[0] > UNIT:
                base_action[0] -= UNIT
        elif action == 3:
            if state[0] < (WIDTH-1) * UNIT:
                base_action[0] += UNIT

        self.canvas.move(self.rectangle, base_action[0], base_action[1])
        self.canvas.tag_raise(self.rectangle)

        next_state = self.canvas.coords(self.rectangle)

        if next_state == self.canvas.coords(self.circle):
            reward = 100
            done = True
        elif next_state in [self.canvas.coords(self.triangle1),
                            self.canvas.coords(self.triangle2)]:
            reward = -100
            done = True
        else:
            reward = 0
            done = False

        next_state = self.coords_to_state(next_state)
        return next_state, reward, done

    def render(self):
        time.sleep(0.03)
        self.update()
