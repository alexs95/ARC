#!/usr/bin/python
"""NUI Galway CT5132/CT5148 Programming and Tools for AI (James McDermott)

Solution for Assignment 3: ARC
Student name(s): Alexey Shapovalov
Student ID(s): XXXXXXXX

To make a decision on which challenge to code up, I looked through
about 50 - 75 different challenges and tried to come up with and idea on how to solve them.
What I found is that challenges that I found the hardest to solve were ones where
I could not figure out the pattern to the solution. However, once I found the pattern, I
almost always figured they would be fairly easy to implement. As examples to this point,
these were the challenges I found the hardest to solve:
6d0160f0 - this took one took the longest to figure out, was at it for at least 15 mins
44f52bb0 - still not sure if I what I think the answer to this is correct
68b16354 - convinced myself this one was a bug (looked purely random) before I finally figured it out
99b1bc43 - I came up with very imaginative theories on what dictates
           the amount of lines coming out of the triangle
6d58a25d - not sure why it took so long but could not figure it out

Instead of choosing any of the ones I found difficult to solve, the criteria I judged
the challenges "difficulty" was how hard I thought it would be to solve them programmatically.
The main feel for this difficulty was when I had to take a minute to figure out what
it is I actually did to come up with a solution. Judging this criteria, I would consider
all of the ones I struggled with simple as once the pattern was found it would be easy to
implement. For example, 68b16354 would be to swap the rows.

Similarities:
    concept of a "background"

"""

from itertools import combinations, product
from collections import defaultdict
from numpy import linalg
import numpy as np
import json
import os
import re


class ColouredPoint:
    """An abstraction of a point in the input array providing utility methods and transformations"""

    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color

    def euclidean(self, other):
        """Euclidean distance between this and other"""
        return linalg.norm([self.x - other.x, self.y - other.y])

    def rotate(self, degrees):
        # Based on https://www.onlinemathlearning.com/transformation-review.html
        if degrees == 90:
            return ColouredPoint(x=-self.y, y=-self.x, color=self.color)
        elif degrees == 180:
            return ColouredPoint(x=-self.x, y=-self.y, color=self.color)
        elif degrees == 270:
            return ColouredPoint(x=self.y, y=-self.x, color=self.color)
        else:
            raise ValueError("Unsupported degrees: {}".format(degrees))

    def reflect(self, axis):
        # Based on https://www.onlinemathlearning.com/transformation-review.html
        if axis == "x":
            return ColouredPoint(x=self.x, y=-self.y, color=self.color)
        elif axis == "y":
            return ColouredPoint(x=-self.x, y=self.y, color=self.color)
        elif axis == "x=y":
            return ColouredPoint(x=self.y, y=self.x, color=self.color)
        else:
            raise ValueError("Unsupported axis: {}".format(axis))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.color == other.color

    def __str__(self):
        return "({}: ({}, {}))".format(self.color, self.x, self.y)


class Collection:
    """An abstraction representing a collection of points: provides information about the points
    and methods that operate the collection as a whole

    """

    def __init__(self, points):
        points = list(points)
        self.points = points
        self.min_x = min(p.x for p in points)
        self.min_y = min(p.y for p in points)
        self.max_x = max(p.x for p in points)
        self.max_y = max(p.y for p in points)
        self.shape = (self.max_x - self.min_x + 1, self.max_y - self.min_y + 1)
        self.colors = set([p.color for p in points])

    def asnumpy(self, background=0):
        arr = np.full(self.shape, background)
        for p in self.points:
            arr[p.x - self.min_x][p.y - self.min_y] = p.color
        return arr

    def fill(self, arr, color=None):
        """Fills arr (numpy array) using the points in this collection. If color is provided each point
        will be filled with color rather than the color of the point.
        """
        for point in self.points:
            arr[point.x][point.y] = color if color is not None else point.color

    def translate(self, source, destination):
        """Translates each point in the collection defined by the translation
        from source (point) to destination (point)

        """

        x_diff = destination.x - source.x
        y_diff = destination.y - source.y
        return Collection(
            ColouredPoint(x=p.x+x_diff, y=p.y+y_diff, color=p.color) for p in self.points
        )

    def rotate(self, degrees):
        return Collection(p.rotate(degrees) for p in self.points)

    def reflect(self, axis):
        return Collection(p.reflect(axis) for p in self.points)

    def __str__(self):
        return str(self.asnumpy())

    def __eq__(self, other):
        if len(other.points) == len(self.points):
            other.points.sort(key=lambda p: (p.x, p.y, p.color))
            self.points.sort(key=lambda p: (p.x, p.y, p.color))
            return all(a == b for (a, b) in zip(self.points, other.points))
        else:
            return False


class Grid:
    """An abstraction representing the entire input grid - provides information about the input and
    contains methods to obtain Collections and ColouredPoints"""
    def __init__(self, X):
        self.arr = X
        self.visited_mark = -1
        self.shape = X.shape

    def colors(self):
        unique, counts = np.unique(self.arr, return_counts=True)
        return {k: v for (k, v) in zip(unique, counts)}

    def get_points_by_color(self, color=None):
        points = []
        for x in range(self.arr.shape[0]):
            for y in range(self.arr.shape[1]):
                if self.arr[x][y] == color:
                    points.append(ColouredPoint(x, y, color))
        return Collection(points)

    def get_adjacency_collections(self, avoid_color, match_color=None):
        collections = []
        visited = np.zeros(self.arr.shape)
        for x in range(self.arr.shape[0]):
            for y in range(self.arr.shape[1]):
                collection = self._traverse(x, y, visited, avoid_color, match_color)
                if collection is not None:
                    collections.append(Collection(collection))

        return collections

    def _traverse(self, x, y, visited, avoid_color, match_color):
        stop_condition = (
                x == -1 or x == self.arr.shape[0] or
                y == -1 or y == self.arr.shape[1] or
                self.arr[x][y] == avoid_color or
                visited[x][y] == self.visited_mark or
                (match_color is not None and self.arr[x][y] != match_color)
        )

        if not stop_condition:
            visited[x][y] = self.visited_mark
            point = ColouredPoint(x, y, self.arr[x][y])
            points = [point]
            for neighbour in ((x, y-1), (x, y+1), (x+1, y), (x-1, y)):
                nx, ny = neighbour
                new = self._traverse(nx, ny, visited, avoid_color, match_color)
                if new is not None:
                    points += new

            return points

        return None


def solve_0e206a2e(X, background=0):
    """I would consider this one the hardest of all my choices. Firstly it was not super easy
    to solve as a human (in comparison to others). But the real difficulty, even after thinking
    about it, I am not fully sure of the exact steps I take to solve it.
    The steps in my head for solving it are:
    1. Find the shape(s) and the "dots" that the shapes need to be placed on.
    2. Figure out which shape belongs to which set of "dots".
    3. Move the shape(s) into the correct position, judging it by the dots.
    4. Remove the old position of the shape(s).

    """

    # 4) The solution will be on a blank canvas
    Y = np.full(X.shape, background)

    # Extract what is needed from the input
    X = Grid(X)
    collections = X.get_adjacency_collections(background)
    colors = X.colors()

    # 1.a) Find the target collections, these need to be transformed to match the location
    # of the corresponding reference points
    targets = [s for s in collections if len(s.colors) >= len(colors) - 1]

    # 1.b) Find the reference collections, the target collections
    # will be transformed based on these points
    references = find_references(s for s in collections if len(s.colors) < len(colors) - 1)

    # 2) Brute force search on all (target, reference) possibilities
    for target, reference in product(targets, references):
        # 3) Try to find a transformation that places the target on reference
        transformation = find_transformation(target, reference)
        if transformation is not None:
            # Draw the transformed collection in the correct location
            # if a valid transformation is found
            transformation.fill(Y)
            
    return Y


def find_references(collections):
    """The reference points will be the set of points of unique colors that are nearest to each other.
    I defined "nearest to each other" as the nearest summed euclidean distance between each point
    
    collections - list of Collection - points to find references from (the collections
        will be flattened)

    """
    
    references = []
    
    # Flatten collections to points
    points = [p for s in collections for p in s.points]
    
    # Greedily find the set of points that are unique in color and have minimum summed distance
    while len(points) != 0:
        support = find_best_reference_set(points)
        if len(support) > 1:
            references.append(Collection(support))
            points = [p for p in points if p not in support]

    return references


def find_best_reference_set(points):
    """Finds the best set of points that have a minimum summed distance between each point"""

    # Group points by color
    grouped = defaultdict(list)
    for point in points:
        grouped[point.color].append(point)

    # Brute force search on all combinations of points with unique colors
    possibilities = product(*[grouped[key] for key in grouped])
    return min(possibilities, key=summed_distances)


def summed_distances(points):
    return sum(a.euclidean(b) for a, b in combinations(points, 2))


def find_transformation(target, reference):
    """Finds a transformation of collection onto the points in the reference collection"""

    # Brute force search on all possible transformations
    for axis in (None, "x", "y", "x=y"):
        curr = target
        for degrees in (None, 90, 180, 270):
            if axis is not None:
                curr = curr.reflect(axis)
            if degrees is not None:
                curr = curr.rotate(degrees)

            # Find the reference points in target by matching them with the colors in the
            # reference collection
            target_reference = Collection([p for p in curr.points if p.color in reference.colors])

            # Apply the transformation of one point, to the rest of the points in the target collection
            corresponding = next(p for p in reference.points if p.color == target_reference.points[0].color)
            transformed = target_reference.translate(target_reference.points[0], corresponding)

            # If the resulting collection is identical to the reference point collection it is a match
            if transformed == reference:
                return curr.translate(target_reference.points[0], corresponding)

    return None


def solve_b782dc8a(X, wall=8):
    """This challenge is very simple to solve as a human but will probably be hard to solve programmatically.
    To solve this one, all the black cells are coloured in following an "every second one" pattern.

    """
    # Solution will be on a canvas identical to the input
    Y = X.copy()

    # Extract what is needed from the input
    X = Grid(X)
    colors = X.colors()
    # There will only be one center point, so only once cell will be colored
    # with the center color. Ideally there should be more than one outward points, otherwise
    # it is ambiguous which is the center. The code should work either way
    # by picking one at random if this happens.
    sorted_colors = sorted(colors.items(), key=lambda x: x[1])
    center_color, out_color = [c for c, _ in sorted_colors[:2]]
    center_point = X.get_points_by_color(center_color).points[0]
    color_transitions = {center_color: out_color, out_color: center_color}

    # Paint the pattern starting from the center point
    visited = np.zeros(X.shape)
    paint(Y, (center_point.x, center_point.y), center_color, color_transitions, visited, wall)

    return Y


def paint(X, point, color, color_transitions, visited, wall):
    """Recursively paints non-wall points choosing the next color using the color_transitions dict.
    Points that are already visited are not re-painted.

    """
    visited_mark = -1
    x, y = point

    stop_condition = (
        x == -1 or x == X.shape[0] or
        y == -1 or y == X.shape[1] or
        X[x][y] == wall or
        visited[x][y] == visited_mark
    )

    if not stop_condition:
        X[x][y] = color
        visited[x][y] = visited_mark
        for neighbour in ((x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y)):
            paint(X, neighbour, color_transitions[color], color_transitions, visited, wall)


def solve_5ad4f10b(X, background=0):
    """This challenge was one that was mentioned as difficult in the examples provided.
    As a human it is very simple to solve but the reason I decided to do this attempt this one was
    because it would be interesting to figure out how to identify the square blocks programmatically.
    All of the examples have an output of shape (3, 3) but as a human I would know how
    to solve the same problem with a (4, 4) or a (5, 5), etc. solution. I will try to make my code
    generalise to all output sizes.
    The steps to solve this one is to create an array from the "box" coloured squares pattern.

    """

    # Extract what is needed from the input
    X = Grid(X)
    colors = X.colors()
    del colors[background]

    # Get the bounding boxes of each unique color
    bounding_boxes = (Grid(X.get_points_by_color(c).asnumpy(background)) for c in colors.keys())

    for box, color in zip(bounding_boxes, colors.keys()):
        # Find adjacency collections for both the target color and the background
        targets = box.get_adjacency_collections(background, color)
        backgrounds = box.get_adjacency_collections(color, background)

        # The length of the sides of the squares will be the greatest common divisor
        # of all the sides of the collections
        sides = [t.shape[0] for t in targets] + [t.shape[1] for t in targets] + [box.shape[0]]
        sides += [b.shape[0] for b in backgrounds] + [b.shape[1] for b in backgrounds]
        side = np.gcd.reduce(sides)

        # The fill color of the solution will be the only other color
        # (and not the background color)
        fill_color = next(c for c in colors.keys() if c != color)

        Y = find_solution(box.arr, side, fill_color, background)
        if Y is not None:
            return Y


def find_solution(box, side, fill_color, background):
    w, h = int(box.shape[0] / side), int(box.shape[1] / side)
    Y = np.full((w, h), background)

    for x in range(w):
        for y in range(h):
            square_color = get_square_color(box, x*side, y*side, side)
            if square_color is not None:
                if square_color != background:
                    Y[x][y] = fill_color
            else:
                return None
    return Y


def get_square_color(box, x, y, side):
    colors = set()
    for i in range(side-1):
        for j in range(side-1):
            colors.add(box[x + i][y + j])
    if len(colors) == 1:
        return colors.pop()
    else:
        return None



def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})" 
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals(): 
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1) # just the task ID
            solve_fn = globals()[name] # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)
    
def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""
    
    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)

        
def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    # if yhat has the right shape, then (y == yhat) is a bool array
    # and we test whether it is True everywhere. if yhat has the wrong
    # shape, then y == yhat is just a single bool.
    print(np.all(y == yhat))


if __name__ == "__main__":
    main()
