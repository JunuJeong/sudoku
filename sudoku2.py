# -*- coding: utf-8 -*-
"""Sudoku solver.

This module solves any sudoku puzzels. A puzzel should be
given by a file consisting with comma-separated columns and
line_break-separated rows. The number of rows and columns
should be the same and they should be a square of an integer.

Example:
    >>> import sudoku
    >>> solver = sudoku.solver(file, max_iteration)
    >>> solver.run()
"""

import itertools
import numpy as np


class sudoku:
    """a sudoku solver class."""
    def __init__(self, file=None, max_iteration=100):
        """initialize the solver."""
        self.table = self.table_from_file(file)
        self.notes = self.notes_from_table(self.table)

        self.base = self.base_from_table(self.table)
        self.line = np.array([x for x in range(self.base**2)])
        self.permutations = self.permutations_from_line(self.line)
        self.candidates = self.candidates_from_notes(self.notes)

        self.max_iteration = max_iteration
        self.step = 0
        self.depth = 0

    def __len__(self):
        """return the base."""
        return self.base

    def __str__(self):
        """return the board of the table."""
        return self.board_from_table(self.table)

    def run(self, table=[], notes=[], max_iteration=0, log=True):
        """run the solver."""
        if table == []:
            table = self.table
        if notes == []:
            notes = self.notes
        if max_iteration == 0:
            max_iteration = self.max_iteration
        is_solved, is_collided = False, False

        if self.step >= max_iteration:
            if self.depth == 0:
                self.log(log, 'not-solved', self.depth)
            return is_solved, is_collided

        if self.depth == 0:
            complexity = self.complexity(notes)
            self.log(log, 'update', self.depth, table, complexity)

        for step in itertools.count():
            is_solved, is_updated, is_collided = self.solve(table, notes)
            complexity = self.complexity(notes)
            self.step += 1
            if not is_solved:
                self.log(log, 'update', self.depth, table, complexity)

            if self.step >= max_iteration:
                if self.depth == 0:
                    self.log(log, 'not-solved', self.depth)
                return is_solved, is_collided

            if is_collided:
                self.log(log, 'collision', self.depth, table)
                break

            if not is_updated and not is_solved:
                min_index, complexity = self.complexity_min(notes)
                trial_table = table.copy()
                trials_notes = self.trial_notes(notes, min_index)
                self.depth += 1
                for trial_notes in trials_notes:
                    is_solved, is_collided = self.run(
                        trial_table, trial_notes, log=log)
                    if self.step >= max_iteration:
                        self.depth -= 1
                        if self.depth == 0:
                            self.log(log, 'not-solved', self.depth)
                        return is_solved, is_collided

                    if is_collided:
                        trial_table = table.copy()
                        continue

                    if is_solved:
                        table[:, :] = trial_table
                        notes[:, :] = trial_notes
                        complexity = self.complexity(notes)
                        self.depth -= 1
                        return is_solved, is_collided
                else:
                    if self.step > max_iteration:
                        self.depth -= 1
                        break
                    self.log(log, 'collision', self.depth, table)
                    self.depth -= 1
                    break

            if is_solved:
                self.log(log, 'solved', self.depth, table, complexity)
                break
        return is_solved, is_collided

    def solve(self, table, notes):
        """
        update tables and notes.

        :return: return whether is solved, updated, and collided.
        """
        bygone = notes.copy()
        self.update_table(table, notes)
        self.update_notes(table, notes)
        self.update_table(table, notes)
        is_updated = not np.all(bygone == notes)
        is_solved = np.all(table)
        is_collided = self.is_collided(table, notes)
        return is_solved, is_updated, is_collided

    def log(self, log, stat, depth=0, table=[], complexity=0):
        """print logs and the board."""
        if stat == 'solved':
            print(f'Step {self.step:d}, depth {depth:d}')
            print(f'{complexity} possibility, solved.')
            print(self.board_from_table(table))
        if not log:
            return log
        elif stat == 'update':
            print(f'Step {self.step:d}, depth {depth:d}')
            if complexity > 1:
                print(f'{complexity} possibilities.')
            else:
                print(f'{complexity} possibility.')
            print(self.board_from_table(table))
        elif stat == 'not-solved':
            print(f'Not solved in step {self.step:d}')
        elif stat == 'collision':
            print(f'Step {self.step:d}, depth {depth:d}')
            print(f'There is a contradiction.')
            print(self.board_from_table(table))
        return log

    def is_power_of_2(self, bit):
        """return whether bit is power of 2."""
        return bit & (bit - 1) == 0

    def ilog2(self, n):
        """return log based 2."""
        n = int(n)
        if not self.is_power_of_2(n):
            raise ValueError('not a power of 2')
        return n.bit_length() - 1

    def is_sqrt_int(self, n):
        """return whether square root is integer."""
        return np.sqrt(n) == int(np.sqrt(n))

    def isqrt(self, n):
        """return the integer square root."""
        if not self.is_sqrt_int(n):
            raise ValueError('not a square of integer')
        return int(np.sqrt(n))

    def inot(self, n, base):
        """return the inverted byte."""
        return (1 << (base**2)) - 1 - n

    def is_collided(self, table, notes):
        """return whether there is a contradiction."""
        for wise in self.positions_in(self.base):
            for xs, ys in wise:
                for p in range(self.base**2):
                    x, y = xs[p], ys[p]
                    xo, yo = self.positions_other(xs, ys, p)
                    element = table[x, y]
                    others = table[xo, yo]
                    if element and element in others:
                        return True
        if self.complexity(notes) == 0:
            return True
        return False

    def table_from_file(self, file):
        """return a table from the file."""
        table = []
        with open(file, 'r') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    break
                table.append(line.split(','))
        table = np.array(table, dtype=np.uint8)
        return table

    def base_from_table(self, table):
        """return a base from the table."""
        nrow, ncol = table.shape
        if nrow != ncol:
            raise ValueError('table is not a right-square.')
        if self.is_sqrt_int(nrow):
            base = self.isqrt(nrow)
        else:
            raise ValueError('table is not matched to solve.')
        return base

    def notes_from_table(self, table):
        """return notes from the table."""
        nrow, ncol = table.shape
        notes = ((1 << nrow) - 1) * np.ones((nrow, ncol), dtype=np.uint16)
        for x, y in itertools.product(range(nrow), range(ncol)):
            element = table[x, y]
            if element:
                notes[x, y] = 1 << (element - 1)
        return notes

    def update_notes(self, table, notes):
        """update the notes from the table."""
        for wise in self.positions_in(self.base):
            for xs, ys in wise:
                for p in range(self.base**2):
                    x, y = xs[p], ys[p]
                    element = table[x, y]
                    xo, yo = self.positions_other(xs, ys, p)
                    if element:
                        notes[xo, yo] = self.note_extracted(
                            notes[xo, yo], 1 << (element - 1))
        for wise in self.positions_in(self.base):
            for xs, ys in wise:
                for p in range(self.base**2):
                    x, y = xs[p], ys[p]
                    xo, yo = self.positions_other(xs, ys, p)
                    note = self.inot(
                        np.bitwise_or.reduce(notes[xo, yo]), self.base)
                    if self.is_power_of_2(note) and note:
                        notes[x, y] = note

    def update_table(self, table, notes):
        """update the table from the notes."""
        nrow, ncol = table.shape
        for x, y in itertools.product(range(nrow), range(ncol)):
            note = notes[x, y]
            if self.is_power_of_2(note):
                table[x, y] = self.ilog2(note) + 1

    def trial_notes(self, notes, index):
        """yield possible candidates with the index-th block try."""
        candidates = list(self.candidates_from_notes(notes))[index]
        cx, cy = list(self.positions_in_blockwise(self.base))[index]
        for candidate in candidates:
            trial = notes.copy()
            trial[cx, cy] = candidate
            yield trial

    def positions_other(self, xs, ys, p):
        """return positions except the p-th."""
        xo = np.concatenate((xs[:p], xs[p + 1:]))
        yo = np.concatenate((ys[:p], ys[p + 1:]))
        return xo, yo

    def positions_in(self, base):
        """yield positions in blockwise, row-wise and col-wise."""
        yield self.positions_in_blockwise(base)
        yield self.positions_in_row_wise(base)
        yield self.positions_in_col_wise(base)

    def positions_in_blockwise(self, base):
        """yield positions in blockwise."""
        for i in range(int(base**2)):
            ps = self.line.copy()
            xs = base * (i // base) + ps // base
            ys = base * (i % base) + ps % base
            yield xs, ys

    def positions_in_row_wise(self, base):
        """yield positions in row-wise."""
        for i in range(int(base**2)):
            ps = self.line.copy()
            xs = i + ps // (base ** 2)
            ys = ps + i // (base ** 2)
            yield xs, ys

    def positions_in_col_wise(self, base):
        """yield positions in col-wise."""
        for i in range(int(base**2)):
            ps = self.line.copy()
            xs = ps + i // (base ** 2)
            ys = i + ps // (base ** 2)
            yield xs, ys

    def note_extracted(self, note, other):
        """return a note from the note except the other."""
        note = note - (note & other)
        return note

    def permutations_from_line(self, line):
        """return all possible permutations from the line."""
        return 1 << np.array(
            list(itertools.permutations(line)),
            dtype=np.uint16)

    def candidates_from_notes(self, notes):
        """yield all possible candidates from the note."""
        for xs, ys in self.positions_in_blockwise(self.base):
            yield self.candidates_from_block(notes[xs, ys])

    def candidates_from_block(self, block):
        """yield all possible candidates from the block."""
        candidates = self.permutations[
            np.bitwise_and(self.permutations, block).all(axis=1)]
        return candidates

    def complexity(self, notes):
        """return a complexity of the notes."""
        complexity = 1
        for candidates in self.candidates_from_notes(notes):
            complexity *= candidates.shape[0]
        return complexity

    def complexity_min(self, notes):
        """return an index and a complexity of the minimum's from the notes."""
        complexity = np.math.factorial(self.base**2)
        min_index = 0
        for i, candidates in enumerate(self.candidates_from_notes(notes)):
            if complexity > candidates.shape[0] and candidates.shape[0] != 1:
                min_index, complexity = i, candidates.shape[0]
        return min_index, complexity

    def board_from_table(self, table):
        """return a board which can be printed from the table."""
        digit = 2
        wall_h = ' -'
        wall_v = ' |'
        wall_c = ' +'
        board_length = (self.base + 1) * self.base + 1
        board = ''
        for bx in range(board_length):
            if bx % (self.base + 1) == 0:
                for p in range(board_length):
                    if p % (self.base + 1) == 0:
                        board += wall_c
                    else:
                        board += wall_h
                board += '\n'
                continue
            for by in range(board_length):
                if by % (self.base + 1) == 0:
                    board += wall_v
                    continue
                x = self.base * (bx // (self.base + 1)) \
                    + bx % (self.base + 1) - 1
                y = self.base * (by // (self.base + 1)) \
                    + by % (self.base + 1) - 1
                board += f'{table[x, y]: {digit}x}'
            board += '\n'
        return board

    def board_from_notes(self, notes):
        """return a board from the candidates."""
        digit = self.base**2
        wall = '*'
        board_length = (self.base + 1) * self.base + 1
        board = ''
        for bx in range(board_length):
            if bx % (self.base + 1) == 0:
                board += wall*((self.base**2 + 1)*self.base**2 \
                         + self.base + 1) + '\n'
                continue
            for by in range(board_length):
                if by % (self.base + 1) == 0:
                    board += wall
                    continue
                x = self.base * (bx // (self.base + 1)) \
                    + bx % (self.base + 1) - 1
                y = self.base * (by // (self.base + 1)) \
                    + by % (self.base + 1) - 1
                board += f'{notes[x, y]:0{digit}b} '
            board += '\n'
        return board
