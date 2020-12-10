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
from collections import defaultdict
from itertools import combinations, product
from numpy import linalg, radians
from math import cos, sin
import numpy as np
import json
import os
import re


def summed_distances(pillars):
    distances = [
        linalg.norm(np.array(a) - np.array(b))
        for a, b in combinations(pillars, 2)
    ]
    return sum(distances)


def expand(cell, X, background, marker, visited):
    # targets will always have more than 3 elems in it
    x, y = cell
    if x == -1 or x == X.shape[0] or y == -1 or y == X.shape[1] or X[x][y] == background or visited[x][y] == marker:
        return None
    else:
        visited[x][y] = marker
        colours = [X[x][y]]
        cells = [cell]
        for neighbour in ((x, y-1), (x, y+1), (x+1, y), (x-1, y)):
            expansion = expand(neighbour, X, background, marker, visited)
            if expansion is not None:
                colours += expansion[0]
                cells += expansion[1]
        return colours, cells


def find_best_support(grouped):
    supports = [
        (summed_distances(dimension for (color, dimension) in pillar), pillar)
        for pillar in product(*[grouped[key] for key in grouped])
    ]
    return min(supports, key=lambda x: x[0])[1]


def remove_points(colored_points, grouped):
    for color, point in colored_points:
        grouped[color].remove((color, point))
        if len(grouped[color]) == 0:
            del grouped[color]


def find_supports(supports):
    # Group by color
    grouped = defaultdict(list)
    for colours, points in supports:
        for color, point in zip(colours, points):
            grouped[color].append((color, point))
    grouped = dict(grouped)

    supports = []
    while len(grouped) != 0:
        support = find_best_support(grouped)
        if support is not None:
            supports.append(support)
            remove_points(support, grouped)

    return supports


def rotate_point(point, angle, center_point=(0, 0)):
    angle = radians(angle)
    # Shift the point so that center_point becomes the origin
    rotated = (point[0] - center_point[0], point[1] - center_point[1])
    rotated = (
        rotated[0] * cos(angle) - rotated[1] * sin(angle),
        rotated[0] * sin(angle) + rotated[1] * cos(angle)
    )
    # Reverse the shifting we have done
    rotated = (rotated[0] + center_point[0], rotated[1] + center_point[1])
    rotated = (round(rotated[0]), round(rotated[1]))
    return rotated


def rotate_colored_polygon(polygon, angle, center_point=(0, 0)):
    rotated = []
    for color, point in polygon:
        rotated_point = rotate_point(point, angle, center_point)
        rotated.append((color, rotated_point))
    return rotated


def find_supporting_points(shape, support):
    return [
        next((x for x in shape if x[0] == color), None)
        for color, _ in support
    ]


def flip_colored_polygon(polygon, axis):
    flipped = []
    for color, point in polygon:
        if axis == 0:
            flipped.append((color, (point[0] * -1, point[1])))
        elif axis == 1:
            flipped.append((color, (point[0], point[1] * -1)))
        elif axis == 2:
            flipped.append((color, (point[1] * -1, point[0])))
        else:
            flipped.append((color, (point[1] * -1, point[0])))

    return flipped


def find_transformation(shape, support):
    # Matching support should be equidistant
    for axis in (None, 0, 1, 2, 3):
        target = shape
        for degrees in (None, 90, 180, 270):
            if axis is not None:
                target = flip_colored_polygon(target, axis)
            if degrees is not None:
                target = rotate_colored_polygon(target, degrees)
            supporting_points = find_supporting_points(target, support)
            x_diff = support[0][1][0] - supporting_points[0][1][0]
            y_diff = support[0][1][1] - supporting_points[0][1][1]
            transformation = [
                (color, (x + x_diff, y + y_diff))
                for (color, (x, y)) in target
            ]
            if set([t[1] for t in support]) == set([t[1] for t in transformation if t[0] in [c[0] for c in support]]):
                return transformation

    return None


def solve_0e206a2e(X):
    """I would consider this one the hardest of all my choices. Firstly it was not super easy
    to solve as a human (in comparison to others). But the real difficulty, even after thinking
    about it, I am not fully sure of the exact steps I take to solve it.
    The steps in my head for solving it are:
    1. Find the shape(s) and the "dots" that the shapes need to be placed on.
    2. Figure out which shape belongs to which set of "dots".
    3. Move the shape(s) into the correct position, judging it by the dots.
    4. Remove the old position of the shape(s).

    """

    # Min distance + different color until x colors
    # Pick the ones where all colours found are in it
    # To create supports choose sets which contain all colours of the remaining colors
    background = 0
    marker = -1
    shapes = []
    visited = np.zeros(X.shape)
    for x in range(X.shape[0]):
        for y in range(X.shape[1]):
            found = expand((x, y), X, background, marker, visited)
            if found is not None:
                shapes.append(found)

    supports = find_supports([(colors, cells) for (colors, cells) in shapes if len(set(colors)) < 4])
    shapes = list((colors, cells) for (colors, cells) in shapes if len(set(colors)) >= 4)
    shapes = [
        [(color, point) for (color, point) in zip(colors, points)]
        for (colors, points) in shapes
    ]
    Y = X.copy()
    targets = list((shape, support, find_transformation(shape, support)) for shape, support in product(shapes, supports))
    targets = list(t[2] for t in targets if t[2] is not None)
    for shape in shapes:
        for _, (x, y) in shape:
            Y[x][y] = background
    for shape in targets:
        for color, (x, y) in shape:
            Y[x][y] = color
    return Y


def color_cells(X, curr, prev, color_transitions, visited, wall=8):
    color = color_transitions[X[prev]]
    x, y = curr
    if not(x == -1 or x == X.shape[0] or y == -1 or y == X.shape[1] or X[x][y] == wall or visited[x][y] == 1):
        X[x][y] = color
        visited[x][y] = 1
        for neighbour in ((x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y)):
            color_cells(X, neighbour, curr, color_transitions, visited)


def find_cells_by_color(X, color):
    cells = []
    for x in range(X.shape[0]):
        for y in range(X.shape[1]):
            if X[x][y] == color:
                cells.append((x, y))
    return cells


def solve_b782dc8a(X):
    """This challenge is very simple to solve as a human but will probably be hard to solve programmatically.
    To solve this one, all the black cells are coloured in following an "every second one" pattern.

    """
    # need to handle case where there is only one path to take,
    # which one is the first one?
    unique, counts = np.unique(X, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    colors = frequencies[frequencies[:, 1].argsort()].T[0, :2]
    color_transitions = {colors[0]: colors[1], colors[1]: colors[0]}
    visited = np.zeros(X.shape)
    Y = X.copy()
    for cell in find_cells_by_color(X, colors[1]):
        color_cells(Y, cell, find_cells_by_color(X, colors[0])[0], color_transitions, visited)
    return Y


def solve_5ad4f10b(x):
    """This challenge was one that was mentioned as difficult in the examples provided.
    As a human it is very simple to solve but the reason I decided to do this attempt this one was
    because it would be interesting to figure out how to identify the square blocks programmatically.
    All of the examples have an output of shape (3, 3) but as a human I would know how
    to solve the same problem with a (4, 4) or a (5, 5), etc. solution. I will try to make my code
    generalise to all output sizes.
    The steps to solve this one is to create an array from the "box" coloured squares pattern.

    """

    return x


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

if __name__ == "__main__": main()

