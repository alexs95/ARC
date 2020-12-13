#!/usr/bin/python
"""NUI Galway CT5148 Programming and Tools for AI (James McDermott)

CT5148 Assignment 3: ARC
Student name(s): Alexey Shapovalov
Student ID(s): XXXXXXXX
GitHub: https://github.com/alexs95/ARC

Choice of Task
---------------------------------------------------------------
I looked through about 50 - 75 different tasks to decide which ones to implement.
What I found out is that tasks that I found the hardest to solve were ones I could not
figure out the pattern to the solution. However, once I figured out the pattern, I
almost always figured they would be fairly easy to implement.
As examples to this point, these were the tasks I found the hardest to solve:
6d0160f0 - this took one took the longest to figure out, was at it for at least 15 mins
44f52bb0 - still not sure if I what I think the answer to this is correct
68b16354 - convinced myself this one was a bug (looked purely random) before I finally figured it out
6d58a25d - not sure why it took so long but could not figure it out
99b1bc43 - I came up with very imaginative theories on what dictates
           the amount of lines coming out of the triangle

Instead of choosing any of these, the criteria I judged the tasks "difficulty" was how hard
I thought it would be to solve them programmatically.
The main "feel" for this difficulty was when I had to take a minute to figure out what
it is I actually did to come up with a solution. Judging by this criteria, I would consider
much easier to implement. For example, 68b16354 would be to swap rows.


Code structure & GitHub
---------------------------------------------------------------
I did all my work on the assignment branch which I merged to master when I was complete.
The README contains contains comments about the purpose of this fork.

The code is laid out as follows:
  At first there are three classes representing different levels of abstraction
  about objects in the task. These provide an API for querying the input.
  Then each solve_ function with functions that are used in the solutions below
  each solve_ function.


Reflection
---------------------------------------------------------------
I would consider the main similarity between these (and all the other) tasks
pattern matching and interpretation of colors.
All of the challenges require you to figure out (the pattern) a set of transformations
to the input grid by interpreting what the colors mean.
You have a set of examples to figure out this pattern. In my solution the
three classes contain transformations (and also the function downscale if it was
implemented more generally), could potentially be used for more than one task.
The method get_adjacency_collections would be an example of pattern matching.
The actual solutions would in a way contain the interpretation of the colors.

Similarities:
Interpretation: All three solutions had a concept of a "background" on which
stuff was happening.

Pattern matching: All three required understanding of the idea of the significance of
cells of equal being color next to each other.

Transformation: 0e206a2e and required understanding geometrical transformation concepts
(but the transformations themselves were different).

Differences:
The most obvious difference is that the pattern to solve each task was different.

There was a concept of a path in b782dc8a.


Relationship to Chollet paper
---------------------------------------------------------------
Sadly as this was our second last assignment I was really stuck for time (could not start as soon
as I liked as I had to finish previous ones). I only had a chance to skim through the paper so
this section might be taken with a pinch of salt.. possibly a lot of salt.
I did want to give it a go. I will read it over Christmas properly for sure,
I found it very interesting!

My interpretation of the goal of ARC dataset is to provide a dataset that if solved would represent
a more general artificial intelligence. It explains how state of the art applications of machine
learning are usually very specific to one narrow task, e.g. playing Go or chess,
but are not generally intelligent. The ARC dataset sets out to contain general
tasks that would need to be a solved by an AI that is more generally intelligent.
The Chollet paper describes a set of priors that an entity can have to solve
these general tasks. I am thinking these would correspond to the similarities
in my solutions. For example the concept of rotating, moving, recognising squares
etc. This would loosely correspond to the three classes at the start of my solution.
The actual solve_ functions would correspond to the use of these priors to solve the
tasks, I guess this is what the AI would actually need to have understand.

"""

from itertools import combinations, product
from collections import defaultdict
from numpy import linalg
import numpy as np
import json
import os
import re


class ColouredPoint:
    """An abstraction of a point in the input array providing utility methods and
    transformations on the point"""

    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color

    def euclidean(self, other):
        """Euclidean distance between this and other"""
        return linalg.norm([self.x - other.x, self.y - other.y])

    def rotate(self, degrees):
        """Returns a new point by rotating this one about the origin (0, 0) by the given degrees"""

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
        """Returns a new point by reflecting this one across the given axis"""

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
    and methods (transformations) that operate the collection as a whole.
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
        """Fills arr (numpy array) with the points in this collection. If color is provided each point
        will be filled with color rather than the color of the point
        """

        for point in self.points:
            arr[point.x][point.y] = color if color is not None else point.color

    def translate(self, source, destination):
        """Translates each point in the collection defined by the translation
        from source (point) to destination (point)
        """

        # Based on https://www.onlinemathlearning.com/transformation-review.html
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
            sorted_points = sorted(other.points, key=lambda p: (p.x, p.y, p.color))
            other_sorted_points = sorted(self.points, key=lambda p: (p.x, p.y, p.color))
            return all(a == b for (a, b) in zip(sorted_points, other_sorted_points))
        else:
            return False


class Grid:
    """An abstraction representing a grid - provides information about arr and
    contains methods that return Collections and ColouredPoints"""
    def __init__(self, X):
        self.arr = X
        self.visited_mark = -1

    def colors(self):
        """Returns a color frequency distribution as a dict"""
        unique, counts = np.unique(self.arr, return_counts=True)
        return {k: v for (k, v) in zip(unique, counts)}

    def get_points_by_color(self, color=None):
        """Returns a Collection of all the points that have the same color"""
        points = []
        for x in range(self.arr.shape[0]):
            for y in range(self.arr.shape[1]):
                if self.arr[x][y] == color:
                    points.append(ColouredPoint(x, y, color))
        return Collection(points)

    def get_adjacency_collections(self, avoid_color, match_color=None):
        """Returns a list of Collection consisting of points that are not avoid_color and have
        at least one other neighbouring point on either the x-axis or the y-axis

        If match_color is set the points must be that color
        """

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
    """I would consider this one to be the hardest of all my choices. It was not super easy
    to solve as a human (in comparison to others) but also the real difficulty was that
    even after thinking about it for a while, I was not fully sure of the exact steps
    I took in my head to solve it.

    The steps in my head for solving it are:
    1. Find the shape(s) and the "dots" that the shapes need to be placed on.
    2. Figure out which shape belongs to which set of "dots".
    3. Move the shape(s) into the correct position, judging it by the dots.
    4. Remove the old position of the shape(s).

    Algorithm:
    1. The pattern here was that shapes would need to be connected by at least
    one neighbour on the horizontal or vertical. The way I figured out the
    reference "dots" was by creating sets of that that were closest together,
    my metric for closest was summed distance.
    2. I brute forced this, just tried every possible combination
    of shape and set of "dots".
    3. Move actually corresponded to transform.. you could also rotate, or flip
    before moving. Also brute forced this step by trying all possibilities
    of rotating and flipping until I found any match.
    4. This was easily done by starting with a blank canvas for the solution.

    Tasks solved: All
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
    """This task is very simple to solve as a human but harder to solve programmatically.

    The steps in my head for solving it are:
    1. Identify the center point from which I would start the "painting".
    2. Identify the second color in the every second one pattern judging it
       by the neighbouring cells that were not the colour of the wall.
    3. Paint the background until I could not paint anymore.

    Algorithm:
    1. The center point would be the only point with one color.
    2. The pattern color would be the second least used color in the grid.
    3. Painting is terminated when a wall is hit or the limits of the grid are reached.

    Tasks solved: All
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
    visited = np.zeros(X.arr.shape)
    paint(Y, (center_point.x, center_point.y), center_color, color_transitions, visited, wall)

    return Y


def paint(X, point, color, color_transitions, visited, wall):
    """Recursively paints non-wall points starting from point by choosing the next color
    using the color_transitions dict. Points that are already visited are not re-painted.
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
    """This task was not far off the difficulty of the first task done.
    It was mentioned as one of the examples provided. As a human it is very simple
    to solve but I struggled to come up with a pattern that was always true
    when differentiate which color was which.

    Note:
    All of the examples have an output of shape (3, 3) but as a human I would know how
    to solve the same problem with a (4, 4) or a (5, 5), etc. solution. I tried
    to make the code generalize to this.

    The steps in my head for solving it are:
    1. Find the square of squares of equal size pattern.
    2. Downscale the pattern so that each square is of size 1. Also change the color
    to the other color that is not the background.

    Algorithm:
    This one did not transfer as smoothly to the same steps when its solved programmatically,
    I will describe how the steps are achieved though:
    1. This is handled by treating both colours as the color that contains the square of squares,
       some condition would fail in the process and nothing would be returned when the
       wrong color is attempted.
    2. The size of each side would be the greatest common divisor of all the sides in the
       square of squares. Downscaling is done pretty much exactly as you would think,
       all squares are made a size of one.

    Tasks solved: All
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
        # of all the sides in the square of squares
        sides = [t.shape[0] for t in targets] + [t.shape[1] for t in targets] + [box.arr.shape[0]]
        sides += [b.shape[0] for b in backgrounds] + [b.shape[1] for b in backgrounds]
        side = np.gcd.reduce(sides)

        # The fill color of the solution will be the only other color
        # (and not the background color)
        fill_color = next(c for c in colors.keys() if c != color)

        Y = downscale(box.arr, side, fill_color, background)
        if Y is not None:
            return Y


def downscale(box, side, fill_color, background):
    """Downscales box by considering each box of length side as one pixel.
    If a downscaled pixel contains more than on color this returns None"""

    if side == 1:
        return None

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
    """Gets the color of the square with its top left corner at (x, y) and with the sides being a length of side
    Returns None if they are not all the same color or if side == 1"""

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
