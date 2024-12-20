import heapq
import random
import sys
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

##########################################################
# Maze and Pathfinding Logic
##########################################################


class Maze:
    def __init__(self, rows=20, cols=20):
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols), dtype=int)  # 0 - path, 1 - wall
        self.start = (0, 0)
        self.goal = (rows - 1, cols - 1)

    def generate_random_maze(self):
        # Use recursive division or another method
        self.grid = np.ones((self.rows, self.cols), dtype=int)
        self._recursive_division(0, 0, self.rows, self.cols)
        # Ensure start and goal are paths
        self.grid[self.start[0], self.start[1]] = 0
        self.grid[self.goal[0], self.goal[1]] = 0

    def _recursive_division(self, r, c, h, w):
        if h <= 2 or w <= 2:
            # carve out all cells in this region as paths
            self.grid[r : r + h, c : c + w] = 0
            return

        # choose a horizontal line
        horizontal = r + random.randint(1, h - 2)
        # choose a vertical line
        vertical = c + random.randint(1, w - 2)

        # Draw horizontal line of walls
        self.grid[horizontal, c : c + w] = 1
        # Draw vertical line of walls
        self.grid[r : r + h, vertical] = 1

        # Create passage in random places
        holes = [
            (horizontal, random.randint(c, c + w - 1)),
            (random.randint(r, r + h - 1), vertical),
        ]

        # Add two more holes in random walls to ensure connectivity
        holes.append((horizontal, random.randint(c, vertical - 1)))
        holes.append((horizontal, random.randint(vertical + 1, c + w - 1)))

        for hr, hc in holes:
            # Ensure indices are within the grid
            if 0 <= hr < self.rows and 0 <= hc < self.cols:
                self.grid[hr, hc] = 0

        # Now recursively divide the four sub-mazes
        # top-left
        self._recursive_division(r, c, horizontal - r, vertical - c)
        # top-right
        self._recursive_division(
            r, vertical + 1, horizontal - r, c + w - (vertical + 1)
        )
        # bottom-left
        self._recursive_division(
            horizontal + 1, c, r + h - (horizontal + 1), vertical - c
        )
        # bottom-right
        self._recursive_division(
            horizontal + 1,
            vertical + 1,
            r + h - (horizontal + 1),
            c + w - (vertical + 1),
        )

    def load_from_text(self, text):
        lines = text.strip().split("\n")
        grid_data = []
        for line in lines:
            row = [int(x) for x in line.strip().split()]
            grid_data.append(row)
        arr = np.array(grid_data)
        self.rows, self.cols = arr.shape
        self.grid = arr
        self.start = (0, 0)
        self.goal = (self.rows - 1, self.cols - 1)


def dfs_solver(maze):
    start = maze.start
    goal = maze.goal
    stack = [start]
    visited = set()
    parent = dict()
    visited_order = []

    while stack:
        current = stack.pop()
        visited_order.append(current)
        if current == goal:
            return reconstruct_path(parent, start, goal), visited_order
        if current not in visited:
            visited.add(current)
            for nbr in neighbors(maze, current):
                if nbr not in visited:
                    parent[nbr] = current
                    stack.append(nbr)
    return [], visited_order


def ucs_solver(maze):
    # Uniform Cost Search (Dijkstra)
    start = maze.start
    goal = maze.goal
    visited_order = []
    dist = {start: 0}
    parent = {}
    pq = [(0, start)]
    visited = set()
    while pq:
        cost, current = heapq.heappop(pq)
        if current in visited:
            continue
        visited.add(current)
        visited_order.append(current)
        if current == goal:
            return reconstruct_path(parent, start, goal), visited_order
        for nbr in neighbors(maze, current):
            new_cost = cost + 1
            if nbr not in dist or new_cost < dist[nbr]:
                dist[nbr] = new_cost
                parent[nbr] = current
                heapq.heappush(pq, (new_cost, nbr))
    return [], visited_order


def heuristic(a, b):
    # Manhattan distance
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar_solver(maze):
    start = maze.start
    goal = maze.goal
    visited_order = []
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    parent = {}
    open_set = [(f_score[start], start)]
    visited = set()
    while open_set:
        _, current = heapq.heappop(open_set)
        visited_order.append(current)
        if current == goal:
            return reconstruct_path(parent, start, goal), visited_order
        visited.add(current)
        for nbr in neighbors(maze, current):
            tent_g = g_score[current] + 1
            if nbr in visited and tent_g >= g_score.get(nbr, float("inf")):
                continue
            if tent_g < g_score.get(nbr, float("inf")):
                parent[nbr] = current
                g_score[nbr] = tent_g
                f_score[nbr] = tent_g + heuristic(nbr, goal)
                heapq.heappush(open_set, (f_score[nbr], nbr))
    return [], visited_order


def greedy_best_first_solver(maze):
    # Similar to A* but ignoring g_score, only heuristic
    start = maze.start
    goal = maze.goal
    visited_order = []
    parent = {}
    open_set = [(heuristic(start, goal), start)]
    visited = set()

    while open_set:
        _, current = heapq.heappop(open_set)
        visited_order.append(current)
        if current == goal:
            return reconstruct_path(parent, start, goal), visited_order
        visited.add(current)
        for nbr in neighbors(maze, current):
            if nbr not in visited:
                parent[nbr] = current
                heapq.heappush(open_set, (heuristic(nbr, goal), nbr))
    return [], visited_order


def neighbors(maze, cell):
    r, c = cell
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    nbrs = []
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < maze.rows and 0 <= nc < maze.cols and maze.grid[nr, nc] == 0:
            nbrs.append((nr, nc))
    return nbrs


def reconstruct_path(parent, start, goal):
    if goal not in parent and goal != start:
        return []
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path


##########################################################
# GUI and Visualization
##########################################################


class MazeView(QGraphicsView):
    def __init__(self, maze, cell_size=20, parent=None):
        super().__init__(parent)
        self.maze = maze
        self.cell_size = cell_size
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.init_scene()
        # Use QPainter render hints
        self.setRenderHint(QPainter.Antialiasing)
        self.setStyleSheet("background-color: #EEEEEE;")
        self.setFixedSize(
            self.maze.cols * cell_size + 5, self.maze.rows * cell_size + 5
        )
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.NoAnchor)

        self.interactive_mode = False
        self.dynamic_obstacles_mode = False

    def init_scene(self):
        self.scene.clear()
        for r in range(self.maze.rows):
            for c in range(self.maze.cols):
                rect = QGraphicsRectItem(
                    c * self.cell_size,
                    r * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                if self.maze.grid[r, c] == 1:
                    rect.setBrush(QBrush(QColor("#666666")))
                else:
                    rect.setBrush(QBrush(Qt.white))
                rect.setPen(QPen(Qt.gray))
                self.scene.addItem(rect)

        # Mark start and goal
        self.highlight_cell(self.maze.start, color=QColor("#00FF00"))  # start green
        self.highlight_cell(self.maze.goal, color=QColor("#FF0000"))  # goal red

    def highlight_cell(self, cell, color=Qt.yellow):
        r, c = cell
        highlight_rect = QGraphicsRectItem(
            c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size
        )
        highlight_rect.setBrush(QBrush(color))
        highlight_rect.setPen(QPen(Qt.black))
        self.scene.addItem(highlight_rect)

    def mousePressEvent(self, event):
        if self.interactive_mode:
            pos = event.position()
            c = int(pos.x() // self.cell_size)
            r = int(pos.y() // self.cell_size)
            if 0 <= r < self.maze.rows and 0 <= c < self.maze.cols:
                # Toggle wall
                self.maze.grid[r, c] = 1 - self.maze.grid[r, c]
                self.init_scene()
        super().mousePressEvent(event)

    def random_toggle_wall(self):
        if self.dynamic_obstacles_mode:
            # randomly pick a cell and toggle it
            r = random.randint(0, self.maze.rows - 1)
            c = random.randint(0, self.maze.cols - 1)
            if (r, c) not in [self.maze.start, self.maze.goal]:
                self.maze.grid[r, c] = 1 - self.maze.grid[r, c]
                self.init_scene()


##########################################################
# Main Window
##########################################################


class MazeSolverApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Maze Solver Using AI Pathfinding Algorithms")
        self.setStyleSheet("background-color: #DDDDDD; color: #000000;")

        self.maze = Maze(20, 20)
        self.maze.generate_random_maze()

        self.maze_view = MazeView(self.maze)
        self.start_button = QPushButton("Run")
        self.reset_button = QPushButton("Reset Maze")
        self.load_button = QPushButton("Load Custom Maze")
        self.toggle_interactive_button = QPushButton("Interactive Mode: OFF")
        self.dynamic_obstacles_button = QPushButton("Dynamic Obstacles: OFF")
        self.algo_select_buttons = [
            QPushButton("DFS"),
            QPushButton("UCS"),
            QPushButton("A*"),
            QPushButton("Greedy"),
        ]
        self.current_algo = dfs_solver

        self.compare_label = QLabel("No data yet")

        self.playing = False
        self.timer = QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.visualize_step)

        self.step_index = 0
        self.path = []
        self.visited_order = []
        self.start_time = 0
        self.end_time = 0

        # Multiplayer mode
        self.multiplayer_button = QPushButton("Multiplayer Mode")
        self.multiplayer_active = False
        self.maze2 = Maze(20, 20)
        self.maze2.generate_random_maze()
        self.maze_view2 = MazeView(self.maze2)
        self.path2 = []
        self.visited_order2 = []
        self.current_algo_2 = ucs_solver

        layout = QVBoxLayout()
        top_layout = QHBoxLayout()

        algo_layout = QHBoxLayout()
        for btn in self.algo_select_buttons:
            btn.clicked.connect(self.select_algorithm)
            algo_layout.addWidget(btn)

        top_layout.addWidget(self.maze_view)
        top_layout.addWidget(self.maze_view2)

        control_layout = QHBoxLayout()
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.reset_button)
        control_layout.addWidget(self.load_button)
        control_layout.addWidget(self.toggle_interactive_button)
        control_layout.addWidget(self.dynamic_obstacles_button)
        control_layout.addWidget(self.multiplayer_button)

        layout.addLayout(top_layout)
        layout.addLayout(algo_layout)
        layout.addLayout(control_layout)
        layout.addWidget(self.compare_label)

        self.setLayout(layout)

        self.start_button.clicked.connect(self.run_solver)
        self.reset_button.clicked.connect(self.reset_maze)
        self.load_button.clicked.connect(self.load_custom_maze)
        self.toggle_interactive_button.clicked.connect(self.toggle_interactive_mode)
        self.dynamic_obstacles_button.clicked.connect(self.toggle_dynamic_obstacles)
        self.multiplayer_button.clicked.connect(self.toggle_multiplayer_mode)

    def toggle_multiplayer_mode(self):
        self.multiplayer_active = not self.multiplayer_active
        if self.multiplayer_active:
            self.multiplayer_button.setText("Multiplayer Mode: ON")
        else:
            self.multiplayer_button.setText("Multiplayer Mode")

    def toggle_interactive_mode(self):
        self.maze_view.interactive_mode = not self.maze_view.interactive_mode
        if self.maze_view.interactive_mode:
            self.toggle_interactive_button.setText("Interactive Mode: ON")
        else:
            self.toggle_interactive_button.setText("Interactive Mode: OFF")

    def toggle_dynamic_obstacles(self):
        self.maze_view.dynamic_obstacles_mode = (
            not self.maze_view.dynamic_obstacles_mode
        )
        self.maze_view2.dynamic_obstacles_mode = (
            not self.maze_view2.dynamic_obstacles_mode
        )
        if self.maze_view.dynamic_obstacles_mode:
            self.dynamic_obstacles_button.setText("Dynamic Obstacles: ON")
        else:
            self.dynamic_obstacles_button.setText("Dynamic Obstacles: OFF")

    def select_algorithm(self):
        sender = self.sender().text()
        if sender == "DFS":
            self.current_algo = dfs_solver
        elif sender == "UCS":
            self.current_algo = ucs_solver
        elif sender == "A*":
            self.current_algo = astar_solver
        else:
            self.current_algo = greedy_best_first_solver

        # If in multiplayer mode, choose a second algorithm automatically
        if self.multiplayer_active:
            # Just pick a different algorithm from the chosen one for player 2
            algos = [dfs_solver, ucs_solver, astar_solver, greedy_best_first_solver]
            if self.current_algo in algos:
                idx = algos.index(self.current_algo)
                self.current_algo_2 = algos[(idx + 1) % 4]

    def run_solver(self):
        if self.playing:
            self.stop_visualization()
            return
        self.start_button.setText("Stop")
        self.playing = True

        # Solve maze
        self.maze_view.init_scene()
        self.path, self.visited_order = self.current_algo(self.maze)
        self.start_time = time.time()

        if self.multiplayer_active:
            self.maze_view2.init_scene()
            self.path2, self.visited_order2 = self.current_algo_2(self.maze2)

        self.step_index = 0
        self.timer.start()

    def stop_visualization(self):
        self.playing = False
        self.start_button.setText("Run")
        self.timer.stop()

    def visualize_step(self):
        if self.step_index < len(self.visited_order):
            cell = self.visited_order[self.step_index]
            self.maze_view.highlight_cell(cell, color=QColor("#cccc00"))
        if self.multiplayer_active and self.step_index < len(self.visited_order2):
            cell2 = self.visited_order2[self.step_index]
            self.maze_view2.highlight_cell(cell2, color=QColor("#cccc00"))

        # Randomly toggle walls if dynamic mode
        self.maze_view.random_toggle_wall()
        self.maze_view2.random_toggle_wall()

        self.step_index += 1

        # If done visiting all
        if self.step_index >= max(
            len(self.visited_order),
            len(self.visited_order2) if self.multiplayer_active else 0,
        ):
            # Highlight path
            for p in self.path:
                self.maze_view.highlight_cell(p, color=QColor("#00cccc"))
            if self.multiplayer_active:
                for p2 in self.path2:
                    self.maze_view2.highlight_cell(p2, color=QColor("#00cccc"))
            self.stop_visualization()
            self.end_time = time.time()
            # Show metrics
            time_taken = self.end_time - self.start_time
            path_len = len(self.path)
            algo_name = self.get_algo_name(self.current_algo)

            if self.multiplayer_active:
                path_len2 = len(self.path2)
                algo_name2 = self.get_algo_name(self.current_algo_2)
                result_text = (
                    f"Single Maze:\nAlgorithm: {algo_name}\nPath length: {path_len}, Time: {time_taken:.4f}s\n"
                    f"Multiplayer Maze:\nAlgorithm: {algo_name2}\nPath length: {path_len2}"
                )
            else:
                result_text = f"Algorithm: {algo_name}\nPath length: {path_len}, Time: {time_taken:.4f}s"
            self.compare_label.setText(result_text)

    def get_algo_name(self, algo_func):
        if algo_func == dfs_solver:
            return "DFS"
        elif algo_func == ucs_solver:
            return "UCS"
        elif algo_func == astar_solver:
            return "A*"
        else:
            return "Greedy Best-First"

    def reset_maze(self):
        self.stop_visualization()
        self.maze = Maze(20, 20)
        self.maze.generate_random_maze()
        self.maze_view.maze = self.maze
        self.maze_view.init_scene()

        self.maze2 = Maze(20, 20)
        self.maze2.generate_random_maze()
        self.maze_view2.maze = self.maze2
        self.maze_view2.init_scene()
        self.compare_label.setText("No data yet")

    def load_custom_maze(self):
        dialog = TextInputDialog()
        if dialog.exec_():
            text = dialog.text_edit.toPlainText()
            try:
                self.maze.load_from_text(text)
                self.maze_view.maze = self.maze
                self.maze_view.init_scene()
                self.compare_label.setText("Custom Maze Loaded")
            except:
                QMessageBox.warning(self, "Error", "Invalid maze format")


class TextInputDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Load Custom Maze")
        self.text_edit = QTextEdit()
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Enter maze rows (0=path,1=wall):"))
        layout.addWidget(self.text_edit)
        layout.addWidget(button_box)
        self.setLayout(layout)


##########################################################
# Main Entry
##########################################################

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MazeSolverApp()
    window.show()
    sys.exit(app.exec())
