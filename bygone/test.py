import sys
import sudoku


def main(file=None):
    if not file:
        raise ValueError('no table is given to solve')

    a = sudoku.solver(file=file)

    step = 100
    for x in range(step):
        print(f'Step : {x}, Level : {a.level}')
        print(a)
        a.run()
        if a.is_solved:
            print(f'Solved in step {x + 1}, level {a.level}')
            print(a)
            break
        elif not a.is_updated:
            print(f'Not solvable.')
            print(a.board_from_candi())
            break
    else:
        print(f'Not solved in step {step}')


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        file = 'map/base3_0.sudoku'
    else:
        file = sys.argv[1]

    main(file)
