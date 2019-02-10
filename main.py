import sys
import sudoku


if __name__ == '__main__':
    file = 'map/base3_0.sudoku'
    max_step = 100
    if len(sys.argv) == 2:
        file = sys.argv[1]
    elif len(sys.argv) == 3:
        file = sys.argv[1]
        max_step = int(sys.argv[2])

    solver = sudoku.solver(file=file, max_step=max_step)
    solver.run()
