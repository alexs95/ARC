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

"""


import numpy as np
import json
import os
import re

def solve_0e206a2e(x):
    """I would consider this one the hardest of all my choices. Firstly it was not super easy
    to solve as a human (in comparison to others). But the real difficulty, even after thinking
    about it, I am not fully sure of the exact steps I take to solve it.
    The steps in my head for solving it are:
    1. Find the shape(s) and the "dots" that the shapes need to be placed on.
    2. Figure out which shape belongs to which set of "dots".
    3. Move the shape(s) into the correct position, judging it by the dots.
    4. Remove the old position of the shape(s).

    """

    return x

def solve_b782dc8a(x):
    """This challenge is very simple to solve as a human but will probably be hard to solve programmatically.
    To solve this one, all the black cells are coloured in following an "every second one" pattern.

    """

    return x

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

