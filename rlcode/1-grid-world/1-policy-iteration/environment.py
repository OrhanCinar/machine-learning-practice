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
