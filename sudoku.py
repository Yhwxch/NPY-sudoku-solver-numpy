import numpy as np
import collections
import itertools
import csv

# Part 1:
def string_puzzle_to_arr(puzzle):
    return np.array([list(line.strip()) for line in puzzle.split('\n') if line.strip()], dtype=np.int)

class Board:
    def __init__(self, puzzle):
        if isinstance(puzzle, str):
            self.puzzle = self.string_puzzle_to_arr(puzzle)
        elif isinstance(puzzle, np.ndarray):
            self.puzzle = puzzle
        else:
            raise ValueError("Invalid input type. Must be either a string or a numpy array.")
        
        # Add arr attribute for backward compatibility with the tests
        self.arr = self.puzzle

    def string_puzzle_to_arr(self, puzzle_str):
        return np.array([[int(char) for char in line.strip()] for line in puzzle_str.split('\n') if line.strip()], dtype=int)

    def get_row(self, row_index):
        return self.puzzle[row_index]

    def get_column(self, col_index):
        return self.puzzle[:, col_index]

    def get_block(self, pos_1, pos_2):
        return self.puzzle[pos_1 * 3: (pos_1 * 3) + 3, pos_2 * 3: (pos_2 * 3) + 3]
   
    def iter_rows(self):
        return [self.get_row(i) for i in range(self.puzzle.shape[0])]

    def iter_columns(self):
        return [self.get_column(i) for i in range(self.puzzle.shape[1])]

    def iter_blocks(self):
        return [self.get_block(i // 3, i % 3) for i in range(9)]


# Part 2:
def is_subset_valid(arr):
    filtered_arr = arr[arr != 0]
    return len(filtered_arr) == len(set(filtered_arr))

def is_valid(board):
    for row in board.iter_rows():
        if not is_subset_valid(row):
            return False
    
    for col in board.iter_columns():
        if not is_subset_valid(col):
            return False
        
    for block in board.iter_blocks():
        if not is_subset_valid(block):
            return False
    return True


# Part 3:
def find_empty(board):
    empty_cells = []
    
    for row_index, row in enumerate(board.iter_rows()):  # Iterate over rows with index
        for col_index, value in enumerate(row):  # Iterate over values in the row with index
            if value == 0:  # Check if the cell is empty
                empty_cells.append((row_index, col_index))  # Append the position as a tuple
    
    if len(empty_cells) == 0:
        return None  # Return None if no empty cells
    else:
        return np.array(empty_cells)  # Return the list of empty cell positions

def is_full(board):
    return len(board.puzzle[board.puzzle == 0]) == 0

def find_possibilities(board, x, y):
    row = set(board.get_row(x))
    col = set(board.get_column(y))
    block = set(board.get_block(x // 3, y // 3).flatten())

    used_values = row | col | block
    possible_values = {1, 2, 3, 4, 5, 6, 7, 8, 9} - used_values
    return possible_values


# Part 4:
def adapt_long_sudoku_line_to_array(line):
    rows = []
    int_list = [int(i) for i in line]
    for i in range(0, 81, 9):
        rows.append(int_list[i: i+9])
    return rows

def read_sudokus_from_csv(filename, read_solutions=False):
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)  # Read the file as a dictionary
        sudokus = []  # List to store the 9x9 grids
        
        for row in reader:
            try:
                # Get the relevant column (quizzes or solutions)
                sudoku_str = row['solutions'] if read_solutions else row['quizzes']
                
                # Ensure the string has exactly 81 characters
                if len(sudoku_str.strip()) != 81:
                    print(f"Skipping invalid sudoku with length {len(sudoku_str.strip())}: {sudoku_str}")
                    continue
                
                # Convert the string to a list of integers
                sudoku_list = [int(char) for char in sudoku_str.strip()]
                
                # Break the flat list into a 9x9 grid
                sudoku_grid = [sudoku_list[i:i+9] for i in range(0, 81, 9)]
                
                # Append the 9x9 grid to the list of sudokus
                sudokus.append(sudoku_grid)
                
            except KeyError:
                # Handle the case where the expected column is missing
                print(f"Column missing in row: {row}")
                continue
            except ValueError:
                # Handle the case where conversion to integer fails
                print(f"Invalid character in sudoku: {sudoku_str}")
                continue
        
        return sudokus  # Return the list of 9x9 grids

def detect_invalid_solutions(file_path):
    """ Detect and return invalid Sudoku solutions from a CSV file. """
    sudoku_solutions = []

    # Read the CSV file using the csv module
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sudoku_str = row['solutions']  # Extract the solution string
            # Convert the string to a 9x9 grid
            grid = [list(map(int, sudoku_str[i:i+9])) for i in range(0, 81, 9)]
            sudoku_solutions.append(grid)

    invalid_solutions = []

    for grid in sudoku_solutions:
        board_array = np.array(grid)

        # Check rows for duplicates
        valid = True
        for row in board_array:
            if len(set(row)) != 9:
                valid = False
                break

        # Check columns for duplicates
        if valid:
            for col in board_array.T:  # Transpose to get columns
                if len(set(col)) != 9:
                    valid = False
                    break

        # Check 3x3 subgrids for duplicates
        if valid:
            for i in range(0, 9, 3):
                for j in range(0, 9, 3):
                    block = board_array[i:i+3, j:j+3].flatten()  # Get the 3x3 block
                    if len(set(block)) != 9:
                        valid = False
                        break

        # If the solution is invalid, add it to the invalid_solutions list
        if not valid:
            invalid_solutions.append(grid)

    return invalid_solutions