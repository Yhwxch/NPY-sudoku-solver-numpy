import numpy as np
from sudoku import (
    adapt_long_sudoku_line_to_array, read_sudokus_from_csv,
    detect_invalid_solutions)


def test_adapt_long_sudoku_lines():
    line = '067050010084309000003080040090000205000621790700093600300400000020007153500800076'
    expected = np.array([
        [0, 6, 7, 0, 5, 0, 0, 1, 0],
        [0, 8, 4, 3, 0, 9, 0, 0, 0],
        [0, 0, 3, 0, 8, 0, 0, 4, 0],
        [0, 9, 0, 0, 0, 0, 2, 0, 5],
        [0, 0, 0, 6, 2, 1, 7, 9, 0],
        [7, 0, 0, 0, 9, 3, 6, 0, 0],
        [3, 0, 0, 4, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 7, 1, 5, 3],
        [5, 0, 0, 8, 0, 0, 0, 7, 6]])
    assert np.array_equal(adapt_long_sudoku_line_to_array(line), expected)


def test_read_csv_puzzles():
    expected = np.array([[
        [0, 8, 3, 2, 0, 0, 0, 9, 6],
        [2, 0, 0, 0, 3, 0, 7, 0, 4],
        [0, 0, 7, 9, 1, 5, 0, 0, 0],
        [4, 0, 2, 3, 9, 0, 0, 0, 8],
        [0, 1, 0, 0, 0, 4, 0, 6, 0],
        [0, 6, 9, 8, 7, 0, 0, 0, 0],
        [0, 0, 0, 4, 0, 0, 0, 0, 7],
        [5, 0, 0, 0, 6, 0, 2, 8, 0],
        [0, 7, 0, 0, 5, 0, 9, 0, 0]],

       [[8, 0, 3, 0, 0, 0, 2, 7, 0],
        [4, 0, 9, 0, 0, 8, 0, 0, 0],
        [7, 0, 0, 0, 2, 4, 0, 9, 6],
        [0, 0, 0, 0, 0, 6, 9, 1, 5],
        [0, 0, 1, 8, 0, 2, 0, 0, 0],
        [0, 3, 0, 7, 5, 0, 0, 0, 0],
        [0, 5, 4, 0, 0, 0, 0, 6, 0],
        [6, 0, 8, 1, 0, 0, 0, 0, 3],
        [3, 7, 2, 0, 0, 9, 1, 4, 0]]])
    assert np.array_equal(read_sudokus_from_csv('data/sudoku_test_valid.csv'), expected)


def test_read_csv_solutions():
    expected = np.array([[
        [1, 8, 3, 2, 4, 7, 5, 9, 6],
        [2, 9, 5, 6, 3, 8, 7, 1, 4],
        [6, 4, 7, 9, 1, 5, 8, 3, 2],
        [4, 5, 2, 3, 9, 6, 1, 7, 8],
        [7, 1, 8, 5, 2, 4, 3, 6, 9],
        [3, 6, 9, 8, 7, 1, 4, 2, 5],
        [9, 2, 1, 4, 8, 3, 6, 5, 7],
        [5, 3, 4, 7, 6, 9, 2, 8, 1],
        [8, 7, 6, 1, 5, 2, 9, 4, 3]],

       [[8, 6, 3, 9, 1, 5, 2, 7, 4],
        [4, 2, 9, 6, 7, 8, 3, 5, 1],
        [7, 1, 5, 3, 2, 4, 8, 9, 6],
        [2, 8, 7, 4, 3, 6, 9, 1, 5],
        [5, 4, 1, 8, 9, 2, 6, 3, 7],
        [9, 3, 6, 7, 5, 1, 4, 8, 2],
        [1, 5, 4, 2, 8, 3, 7, 6, 9],
        [6, 9, 8, 1, 4, 7, 5, 2, 3],
        [3, 7, 2, 5, 6, 9, 1, 4, 8]]])

    assert np.array_equal(read_sudokus_from_csv('data/sudoku_test_valid.csv', read_solutions=True), expected)


def test_detect_invalid_solutions():
    expected = np.array([[
        [8, 6, 3, 9, 1, 5, 5, 7, 4],
        [4, 2, 9, 6, 7, 8, 3, 5, 1],
        [7, 1, 5, 3, 2, 4, 8, 9, 6],
        [2, 8, 7, 4, 3, 6, 9, 1, 5],
        [5, 4, 1, 8, 9, 2, 6, 3, 7],
        [9, 3, 6, 7, 5, 1, 4, 8, 2],
        [1, 5, 4, 2, 8, 3, 7, 6, 9],
        [6, 9, 8, 1, 4, 7, 5, 2, 3],
        [3, 7, 2, 5, 6, 9, 1, 4, 8]]])
    assert np.array_equal(detect_invalid_solutions('data/sudoku_test_invalid.csv'), expected)