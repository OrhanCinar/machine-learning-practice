import tkinter as tk
from tkinter import Button
import time
import numpy as np
from PIL import ImageTk, Image
# https://github.com/rlcode/reinforcement-learning/blob/master/1-grid-world/1-policy-iteration/environment.py
PhotoImage = ImageTk.PhotoImage
UNIT = 100
HEIGHT = 5
WIDTH = 5
TRANSITION_PROB = 1
POSSIBLE_ACTIONS = [0, 1, 2, 3]  # up, down, left, right
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # actions in coordinates
REWARDS = []


class GraphicDisplay(tk.Tk):
    def __init__(self, agent):
        super(GraphicDisplay, self).__init__()
        self.title('Policy Iteration')
        self.geometry('{0}x{1}'.format(HEIGHT*UNIT, HEIGHT*UNIT+50))
        self.texts = []
        self.arrows = []
        self.env = Env()
        self.agent = agent
        self.evaluation_count = 0
        self.improvement_count = 0
        self.is_moving = 0
        (self.up, self.down, self.left, self.right), self.shapes = self.load_images()
        self.canvas = self.__build_canvas()
        self.text_reward(2, 2, "R : 1.0")
        self.text_reward(1, 2, "R : -1.0")
        self.text_reward(2, 1, "R : -1.0")

    def __build_canvas(self):
        canvas = tk.Canvas(self, bg='white',
                           height=HEIGHT * UNIT,
                           width=WIDTH * UNIT)

        iteration_button = Button(
            self, text='Evaluate', command=self.evaluate_policy)
        iteration_button.configure(width=10, activebackground="#33B5E5")
        canvas.create_window(WIDTH * UNIT * 0.13, HEIGHT *
                             UNIT + 10, window=iteration_button)

        policy_button = Button(self, text="Improve",
                               command=self.improve_policy)
        policy_button.configure(width=10, activebackground="#33B5E5")
        canvas.create_window(WIDTH * UNIT * 0.37, HEIGHT *
                             UNIT + 10, window=policy_button)

        policy_button = Button(self, text="move", command=self.move_by_policy)
        policy_button.configure(width=10, activebackground="#33B5E5")
        canvas.create_window(WIDTH * UNIT * 0.62, HEIGHT *
                             UNIT + 10, window=policy_button)

        policy_button = Button(self, text="reset", command=self.reset)
        policy_button.configure(width=10, activebackground="#33B5E5")
        canvas.create_window(WIDTH * UNIT * 0.87, HEIGHT *
                             UNIT + 10, window=policy_button)

        # create grids
        for col in range(0, WIDTH*UNIT, UNIT):
            x0, y0, x1, y1 = col, 0, col, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1)

        for row in range(0, HEIGHT * UNIT, UNIT):
            x0, y0, x1, y1 = 0, row, HEIGHT*UNIT, row
            canvas.create_line(x0, y0, x1, y1)

        self.rectangle = canvas.create_image(50, 50, image=self.shapes[0])
        canvas.create_image(250, 150, image=self.shapes[1])
        canvas.create_image(150, 250, image=self.shapes[1])
        canvas.create_image(250, 250, image=self.shapes[2])

        canvas.pack()
        return canvas

    def load_images(self):
        up = PhotoImage(Image.open("../img/up.png").resize(13, 13))
        right = PhotoImage(Image.open("../img/right.png").resize(13, 13))
        left = PhotoImage(Image.open("../img/left.png").resize(13, 13))
        down = PhotoImage(Image.open("../img/down.png").resize(13, 13))
        rectangle = PhotoImage(Image.open(
            "../img/rectangle.png").resize(65, 65))
        triangle = PhotoImage(Image.open("../img/triangle.png").resize(65, 65))
        circle = PhotoImage(Image.open("../img/circle.png").resize(65, 65))
        return (up, down, left, right), (rectangle, triangle, circle)

    def reset(self):
        if self.is_moving == 0:
            self.evaluation_count = 0
            self.improvement_count = 0
            for i in self.texts:
                self.canvas.delete(i)

            for i in self.arrows:
                self.canvas.delete(i)
            self.agent.value_table = [[0.0] * WIDTH for _ in range(HEIGHT)]
            self.agent_policy_table = (
                [[[0.25, 0.25, 0.25, 0.25]] * WIDTH for _ in range(HEIGHT)])
            self.agent.policy_table[2][2] = []
            x, y = self.canvas.coords(self.rectangle)
            self.canvas.move(self.rectangle, UNIT / 2 - x, UNIT / 2 - y)

    def text_value(self, row, col, contents, font='Helvetiva', size=10, style='normal', anchor='nw'):
        origin_x, origin_y = 85, 70
        x, y = origin_y + (UNIT * col), origin_x + (UNIT * row)
        font = (font, str(size), style)
        text = self.canvas.create_text(
            x, y, fill="black", text=contents, font=font, anchor=anchor)
        return self.texts.append(text)

    def text_reward(self, row, col, contents, font='Helvetiva', size=10, style='normal', anchor='nw'):
        origin_x, origin_y = 5, 5
        x, y = origin_y + (UNIT * col), origin_x + (UNIT * row)
        font = (font, str(size), style)
        text = self.canvas.create_text(
            x, y, fill="black", text=contents, font=font, anchor=anchor)
        return self.texts.append(text)

    def rectangle_move(self, action):
        base_action = np.array([0, 0])
        location = self.find_rectangle()
        self.render()

        if action == 0 and location[0] > 0:
            base_action[1] -= UNIT
        elif action == 1 and location[0] < HEIGHT - 1:
            base_action[1] += UNIT
        elif action == 2 and location[1] > 0:
            base_action[0] -= UNIT
        elif action == 3 and location[1] < WIDTH-1:
            base_action[0] += UNIT
        self.canvas.move(self.rectangle, base_action[0], base_action[1])

    def find_rectangle(self):
        temp = self.canvas.coords(self.rectangle)
        x = (temp[0] / 100) - 0.5
        y = (temp[1] / 100) - 0.5
        return int(y), int(x)
