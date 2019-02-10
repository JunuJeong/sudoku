import itertools
import numpy as np


class solver:
    def __init__(self, file=None, string=None):
        if file:
            self.table = self.table_from_file(file)
        elif string:
            self.table = self.table_from_string(string)
        else:
            raise ValueError('no table is given to solve')

        self.length = len(self.table)
        self.base = self.base_of_table()
        self.candi = self.candi_from_base()
        self.bygone = np.copy(self.candi)

        self.level = 0

        self.is_updated = True
        self.is_solved = False

    def __len__(self):
        return self.length

    def __str__(self):
        board = self.board_from_table()
        return board

    def table_from_string(self, string):
        """return a table from a string formatted 'csv\n...'."""
        rows = string.strip().split('\n')
        len_rows = len(rows)
        table = np.zeros((len_rows, len_rows), dtype=np.uint8)
        for x, row in enumerate(rows):
            cells = row.strip().split(',')
            for y, cell in enumerate(cells):
                table[x, y] = cell
        return table

    def table_from_file(self, file):
        """return a table from a file consisting 'csv\n...'."""
        len_cols = self.length_of_file(file)
        table = np.zeros((len_cols, len_cols), dtype=np.uint8)
        with open(file, 'r') as f:
            for x, row in enumerate(f):
                cells = row.strip().split(',')
                if cells == ['']:
                    break
                for y, cell in enumerate(cells):
                    table[x, y] = cell
        return table

    def length_of_file(self, file):
        """return a column length of a file."""
        with open(file, 'r') as f:
            len_cols = len(f.readline().strip().split(','))
        return len_cols

    def base_of_table(self):
        """check the table is sudoku-ish and return a base."""
        for base in range(1, 9):
            if self.length == base**2:
                break
        else:
            raise ValueError("table length is not matched")
        return base

    def board_from_table(self):
        """return a board which can be printed from the table."""
        if self.base <= 3:
            digit = 2
            wall = ' +'
        else:
            digit = 3
            wall = ' ++'
        board_length = (self.base + 1) * self.base + 1
        board = ''
        for bx in range(board_length):
            if bx % (self.base + 1) == 0:
                board += wall*board_length + '\n'
                continue
            for by in range(board_length):
                if by % (self.base + 1) == 0:
                    board += wall
                    continue
                x = self.base * (bx // (self.base + 1)) + bx % (self.base + 1) - 1
                y = self.base * (by // (self.base + 1)) + by % (self.base + 1) - 1
                board += f'{self.table[x, y]: {digit}d}'
            board += '\n'
        return board

    def board_from_candi(self):
        """return a board from the candidates."""
        digit = self.length
        wall = '*'
        board_length = (self.base + 1) * self.base + 1
        board = ''
        for bx in range(board_length):
            if bx % (self.base + 1) == 0:
                board += wall*((self.length + 1)*self.length + self.base + 1) + '\n'
                continue
            for by in range(board_length):
                if by % (self.base + 1) == 0:
                    board += wall
                    continue
                x = self.base * (bx // (self.base + 1)) + bx % (self.base + 1) - 1
                y = self.base * (by // (self.base + 1)) + by % (self.base + 1) - 1
                board += f'{self.candi[x, y]:0{digit}b} '
            board += '\n'
        return board

    def candi_from_base(self):
        """return candidates from the base"""
        if self.base <= 2:
            candi_type = np.uint8
        elif self.base <= 4:
            candi_type = np.uint16
        elif self.base <= 5:
            candi_type = np.uint32
        else:
            candi_type = np.uint64
        candi = np.ones(self.table.shape, dtype=candi_type) * (2**self.length - 1)
        return candi

    def blockwise_table(self, table):
        """yield a block in blockwise."""
        for i in range(self.length):
            block = np.array(
                [table[self.position_in_blockwise(i, p)] for p in range(self.length)],
                dtype=type(table))
            yield block

    def block_from_position(self, table, row, col):
        """return a block from the given row and col."""
        i = self.base * row + col
        block = np.array(
            [table[self.position_in_blockwise(i, p)] for p in range(self.length)],
            dtype=type(table))
        return block

    def blocks_from_positions(self, table, rows, cols):
        if len(rows) == 1:
            row = rows[0]
            blocks = np.zeros((len(cols), self.length), dtype=type(table))
            for q, col in enumerate(cols):
                blocks[q] = self.block_from_position(table, row, col)
        elif len(cols) == 1:
            col = cols[0]
            blocks = np.zeros((len(rows), self.length), dtype=type(table))
            for q, row in enumerate(rows):
                blocks[q] = self.block_from_position(table, row, col)
        return blocks

    def position_in_blockwise(self, i, p):
        """return x, y position form the given blockwise position."""
        x = self.base * (i // self.base) + p // self.base
        y = self.base * (i % self.base) + p % self.base
        return x, y

    def erase_trivial(self):
        """set candidates for the known elements."""
        for x, y in itertools.product(range(self.length), range(self.length)):
            element = self.table[x, y]
            if element:
                self.candi[x, y] = 1 << element - 1

    def erase_single(self):
        """erase candidates for the known elements in the block."""
        for i, block in enumerate(self.blockwise_table(self.table)):
            for p, element in enumerate(block):
                for q in range(self.length):
                    if not element:
                        break
                    if p == q:
                        continue
                    nx, ny = self.position_in_blockwise(i, q)
                    self.erase_single_wise(nx, ny, element)
        for x, row in enumerate(self.table):
            for y, element in enumerate(row):
                for q in range(self.length):
                    if not element:
                        break
                    if y == q:
                        continue
                    nx, ny = x, q
                    self.erase_single_wise(nx, ny, element)
        for y, col in enumerate(np.transpose(self.table)):
            for x, element in enumerate(col):
                for q in range(self.length):
                    if not element:
                        break
                    if x == q:
                        continue
                    nx, ny = q, y
                    self.erase_single_wise(nx, ny, element)

    def erase_single_wise(self, nx, ny, element):
        note = self.candi[nx, ny]
        if (note >> (element - 1)) % 2 and note & (note - 1) != 0:
            self.candi[nx, ny] -= 2**(element - 1)

    def erase_unique(self):
        """erase candidates for the uniquely determined."""
        for i in range(self.length):
            for p in range(self.length):
                block = list(self.blockwise_table(self.candi))[i]

                others = np.concatenate((block[:p], block[p + 1:]))
                nx, ny = self.position_in_blockwise(i, p)
                self.erase_unique_wise(nx, ny, others)
        for x in range(self.length):
            for y in range(self.length):
                row = self.candi[x, :]

                others = np.concatenate((row[:y], row[y + 1:]))
                nx, ny = x, y
                self.erase_unique_wise(nx, ny, others)
        for y in range(self.length):
            for x in range(self.length):
                col = self.candi[:, y]

                others = np.concatenate((col[:x], col[x + 1:]))
                nx, ny = x, y
                self.erase_unique_wise(nx, ny, others)
                
    def erase_unique_wise(self, nx, ny, others):
        others_note = np.bitwise_or.reduce(others)
        for q in range(self.length):
            note = self.candi[nx, ny]
            if (note >> q) % 2 == 1 and (others_note >> q) % 2 == 0:
                self.candi[nx, ny] = 1 << q
                self.table[nx, ny] = q + 1

    def erase_blockline(self):
        """erase candidates from block-line interactions."""
        for i, block in enumerate(self.blockwise_table(self.candi)):
            for p, note in enumerate(block):
                brow_i = [q + self.base * (p // self.base) for q in range(self.base)]
                bcol_i = [q * self.base + (p % self.base) for q in range(self.base)]
                brow, bcol = block[brow_i], block[bcol_i]

                brow_note = np.bitwise_or.reduce(brow)
                bcol_note = np.bitwise_or.reduce(bcol)

                borow_i = [q
                    if (q // self.base) - (p // self.base) < 0
                    else q + self.base
                    for q in range(self.length - self.base)]
                bocol_i = [self.base * (q // (self.base - 1)) + q % (self.base - 1)
                    if (q % (self.base - 1)) - (p % self.base) < 0
                    else self.base * (q // (self.base - 1)) + q % (self.base - 1) + 1
                    for q in range(self.length - self.base)]
                borow, bocol = block[borow_i], block[bocol_i]
                borow_note, bocol_note = np.bitwise_or.reduce(borow), np.bitwise_or.reduce(bocol)

                burow_note = 0
                for q in range(self.length):
                    if (brow_note >> q) % 2 and not (borow_note >> q) % 2:
                        burow_note += 2**q
                bucol_note = 0
                for q in range(self.length):
                    if (bcol_note >> q) % 2 and not (bocol_note >> q) % 2:
                        bucol_note += 2**q

                for o in range(self.length - self.base):
                    nx, ny = self.position_in_blockwise(i, p)
                    ny = o \
                        if (o // self.base) - (ny // self.base) < 0 \
                        else o + self.base
                    for q in range(self.length):
                        note = self.candi[nx, ny]
                        if (burow_note >> q) % 2 and (note >> q) % 2 and note & (note - 1) != 0:
                            self.candi[nx, ny] -= 2**q
                for o in range(self.length - self.base):
                    nx, ny = self.position_in_blockwise(i, p)
                    nx = o \
                        if (o // self.base) - (nx // self.base) < 0 \
                        else o + self.base
                    for q in range(self.length):
                        note = self.candi[nx, ny]
                        if (bucol_note >> q) % 2 and (note >> q) % 2 and note & (note - 1) != 0:
                            self.candi[nx, ny] -= 2**q

                nx, ny = self.position_in_blockwise(i, p)
                orow_i = [q
                    if (q // self.base) - (ny // self.base) < 0
                    else q + self.base
                    for q in range(self.length - self.base)]
                ocol_i = [q
                    if (q // self.base) - (nx // self.base) < 0
                    else q + self.base
                    for q in range(self.length - self.base)]
                orow, ocol = self.candi[nx, orow_i], self.candi[ocol_i, ny]
                orow_note, ocol_note = np.bitwise_or.reduce(orow), np.bitwise_or.reduce(ocol)

                ourow_note = 0
                for q in range(self.length):
                    if (brow_note >> q) % 2 and not (orow_note >> q) % 2:
                        ourow_note += 2**q
                oucol_note = 0
                for q in range(self.length):
                    if (bcol_note >> q) % 2 and not (ocol_note >> q) % 2:
                        oucol_note += 2**q

                for o in range(self.length - self.base):
                    op = o \
                        if (o // self.base) - (p // self.base) < 0 \
                        else o + self.base
                    nx, ny = self.position_in_blockwise(i, op)
                    for q in range(self.length):
                        note = self.candi[nx, ny]
                        if (ourow_note >> q) % 2 and (note >> q) % 2 and note & (note - 1) != 0:
                            self.candi[nx, ny] -= 2**q
                for o in range(self.length - self.base):
                    op = self.base * (o // (self.base - 1)) + o % (self.base - 1) \
                        if (o % (self.base - 1)) - (p % self.base) < 0 \
                        else self.base * (o // (self.base - 1)) + o % (self.base - 1) + 1
                    nx, ny = self.position_in_blockwise(i, op)
                    for q in range(self.length):
                        note = self.candi[nx, ny]
                        if (oucol_note >> q) % 2 and (note >> q) % 2 and note & (note - 1) != 0:
                            self.candi[nx, ny] -= 2**q

    def erase_blockblock(self):
        """erase candidates from the block-block interactions."""
        self.erase_blockblock_rowwise()
        self.erase_blockblock_colwise()

    def erase_blockblock_rowwise(self):
        """erase candidates from the block-block interactions in row-wise."""
        for r in range(2, self.base):
            for row_i in range(self.base):
                for cols_i in itertools.combinations(range(self.base), r):
                    blocks = self.blocks_from_positions(self.candi, [row_i], cols_i)
                    ocols_i = list(set(range(self.base)) - set(cols_i))
                    for rows_i in itertools.combinations(range(self.base), r):
                        orows_i = list(set(range(self.base)) - set(rows_i))

                        row_q = [self.base * rows_i[q1 % r] + q2 % self.base
                            for q1, q2 in itertools.product(range(r), range(self.base))]
                        orow_q = [self.base * orows_i[q1 % (self.base - r)] + q2 % self.base
                            for q1, q2 in itertools.product(range(self.base - r), range(self.base))]
                        row = blocks[:, row_q].flatten()
                        orow = blocks[:, orow_q].flatten()

                        row_note = np.bitwise_or.reduce(row)
                        orow_note = np.bitwise_or.reduce(orow)
                        urow_note = 0
                        for q in range(self.length):
                            if (row_note >> q) % 2 and not (orow_note >> q) % 2:
                                urow_note += 2**q

                        orow_i = row_i
                        for ocol_i in ocols_i:
                            oblock = self.block_from_position(self.candi, orow_i, ocol_i)
                            for o in range(len(row_q)):
                                i = self.base * orow_i + ocol_i
                                op = row_q[o]
                                nx, ny = self.position_in_blockwise(i, op)
                                for q in range(self.length):
                                    note = self.candi[nx, ny]
                                    if (urow_note >> q) % 2 and (note >> q) % 2 and note & (note - 1) != 0:
                                        self.candi[nx, ny] -= 2**q

    def erase_blockblock_colwise(self):
        """erase candidates from the block-block interactions in col-wise."""
        for r in range(2, self.base):
            for col_i in range(self.base):
                for rows_i in itertools.combinations(range(self.base), r):
                    blocks = self.blocks_from_positions(self.candi, rows_i, [col_i])
                    orows_i = list(set(range(self.base)) - set(rows_i))
                    for cols_i in itertools.combinations(range(self.base), r):
                        ocols_i = list(set(range(self.base)) - set(cols_i))

                        col_q = [self.base * q2 + cols_i[q1 % r] % self.base
                            for q1, q2 in itertools.product(range(r), range(self.base))]
                        ocol_q = [self.base * q2 + ocols_i[q1 % (self.base - r)] % self.base
                            for q1, q2 in itertools.product(range(self.base - r), range(self.base))]
                        col = blocks[:, col_q].flatten()
                        ocol = blocks[:, ocol_q].flatten()

                        col_note = np.bitwise_or.reduce(col)
                        ocol_note = np.bitwise_or.reduce(ocol)
                        ucol_note = 0
                        for q in range(self.length):
                            if (col_note >> q) % 2 and not (ocol_note >> q) % 2:
                                ucol_note += 2**q

                        ocol_i = col_i
                        for orow_i in orows_i:
                            oblock = self.block_from_position(self.candi, orow_i, ocol_i)
                            for o in range(len(col_q)):
                                i = self.base * orow_i + ocol_i
                                op = col_q[o]
                                nx, ny = self.position_in_blockwise(i, op)
                                for q in range(self.length):
                                    note = self.candi[nx, ny]
                                    if (ucol_note >> q) % 2 and (note >> q) % 2 and note & (note - 1) != 0:
                                        self.candi[nx, ny] -= 2**q

    def erase_naked(self):
        """erase candidates from the naked note."""
        self.erase_naked_rowwise()
        self.erase_naked_colwise()

    def erase_naked_rowwise(self):
        """erase candidates from the naked note in row-wise."""
        for x in range(self.length):
            for r in range(2, self.length):
                row = self.candi[x, :]
                for notes in itertools.combinations(zip(range(self.length), row), r):
                    ys = np.array(notes)[:, 0]
                    rows = np.array(notes)[:, 1]
                    row_note = np.bitwise_or.reduce(rows)
                    row_count = bin(row_note).count('1')
                    if row_count == r:
                        for y in range(len(row)):
                            if np.any(ys == y):
                                continue
                            nx, ny = x, y
                            for q in range(self.length):
                                note = self.candi[nx, ny]
                                if (row_note >> q) % 2 and (note >> q) % 2 and note & (note - 1) != 0:
                                    self.candi[nx, ny] -= 2**q

    def erase_naked_colwise(self):
        """erase candidates from the naked note in col-wise."""
        for y in range(self.length):
            for r in range(2, self.length):
                col = self.candi[:, y]
                for notes in itertools.combinations(zip(range(self.length), col), r):
                    xs = np.array(notes)[:, 0]
                    cols = np.array(notes)[:, 1]
                    col_note = np.bitwise_or.reduce(cols)
                    col_count = bin(col_note).count('1')
                    if col_count == r:
                        for x in range(len(col)):
                            if np.any(xs == x):
                                continue
                            nx, ny = x, y
                            for q in range(self.length):
                                note = self.candi[nx, ny]
                                if (col_note >> q) % 2 and (note >> q) % 2 and note & (note - 1) != 0:
                                    self.candi[nx, ny] -= 2**q

    def erase_hidden(self):
        """erase candidates from the hidden note."""
        self.erase_hidden_rowwise()
        self.erase_hidden_colwise()

    def erase_hidden_rowwise(self):
        """erase candidates from the hidden note in row-wise."""
        for x in range(self.length):
            for col_i in range(self.base):
                for r in range(2, self.base + 1):
                    for bcols_i in itertools.combinations(range(self.base), r):
                        row = self.candi[x, :]
                        bocols_i = list(set(range(self.base)) - set(bcols_i))
                        bcols_q = [self.base * (x % self.base) + bcols_i[q % r]
                            for q in range(r)]
                        bocols_q = [self.base * (x % self.base) + bocols_i[q % r]
                            for q in range(self.base - r)]
                        ocols = np.concatenate(
                            (row[:self.base * col_i], row[self.base * (col_i + 1):]))
                        bcols = self.candi[x, bcols_q]
                        bocols = self.candi[x, bocols_q]
                        bcol_note = np.bitwise_or.reduce(bcols)
                        ocol_note = np.bitwise_or.reduce(np.concatenate((ocols, bocols)))
                        ucol_note = 0
                        for q in range(self.length):
                            if (bcol_note >> q) % 2 and not (ocol_note >> q) % 2:
                                ucol_note += 2**q
                        ucol_count = bin(ucol_note).count('1')
                        if ucol_count == r:
                            for q in bcols_q:
                                note = self.candi[x, q]
                                if note & (note - 1) != 0:
                                    self.candi[x, q] = ucol_note

    def erase_hidden_colwise(self):
        """erase candidates from the hidden note in col-wise."""
        for y in range(self.length):
            for row_i in range(self.base):
                for r in range(2, self.base + 1):
                    for brows_i in itertools.combinations(range(self.base), r):
                        col = self.candi[:, y]
                        borows_i = list(set(range(self.base)) - set(brows_i))
                        brows_q = [self.base * (y % self.base) + brows_i[q % r]
                            for q in range(r)]
                        borows_q = [self.base * (y % self.base) + borows_i[q % r]
                            for q in range(self.base - r)]
                        orows = np.concatenate(
                            (col[:self.base * row_i], col[self.base * (row_i + 1):]))
                        brows = self.candi[brows_q, y]
                        borows = self.candi[borows_q, y]
                        brow_note = np.bitwise_or.reduce(brows)
                        orow_note = np.bitwise_or.reduce(np.concatenate((orows, borows)))
                        urow_note = 0
                        for q in range(self.length):
                            if (brow_note >> q) % 2 and not (orow_note >> q) % 2:
                                urow_note += 2**q
                        urow_count = bin(urow_note).count('1')
                        if urow_count == r:
                            for q in brows_q:
                                note = self.candi[q, y]
                                if note & (note - 1) != 0:
                                    self.candi[q, y] = urow_note

    def erase_wing(self):
        """erase candidates from the x-wing technique."""
        r = 2
        for xs in itertools.combinations(range(self.length), r):
            for ys in itertools.combinations(range(self.length), r):
                xm = np.min(np.diff(np.sort([x // self.base for x in xs])))
                ym = np.min(np.diff(np.sort([y // self.base for y in ys])))
                if xm == 0 or ym == 0:
                    continue
                wing_notes = [self.candi[x, y] for x, y in itertools.product(xs, ys)]
                xwing_notes = np.array(
                    [self.candi[xs, y]
                    if y not in ys else [0]*r
                    for y in range(self.length)]).flatten()
                ywing_notes = np.array(
                    [self.candi[x, ys]
                    if x not in xs else [0]*r
                    for x in range(self.length)]).flatten()
                wing_note = np.bitwise_and.reduce(wing_notes)
                xwing_note = np.bitwise_or.reduce(xwing_notes)
                ywing_note = np.bitwise_or.reduce(ywing_notes)
                wing_note_count = bin(wing_note).count('1')
                xwing_note_count = bin(wing_note & xwing_note).count('1')
                ywing_note_count = bin(wing_note & ywing_note).count('1')
                if wing_note_count - xwing_note_count == r:
                    for x in xs:
                        for y in range(self.length):
                            if y in ys:
                                continue
                            note = self.candi[x, y]
                            if note != wing_note:
                                self.candi[x, y] -= note & wing_note
                if wing_note_count - ywing_note_count == r:
                    for y in ys:
                        for x in range(self.length):
                            if x in xs:
                                continue
                            note = self.candi[x, y]
                            if note != wing_note:
                                self.candi[x, y] -= note & wing_note

    def erase_swordfish(self):
        """erase candidates from the swordfish technique."""
        self.erase_swordfish_rowwise()
        self.erase_swordfish_colwise()

    def erase_swordfish_rowwise(self):
        """erase candidates from the swordfish technique in row-wise."""
        r = 4
        for xs in itertools.combinations(range(self.length), r):
            xm = np.diff(np.sort([x // self.base for x in xs]))
            if len(xm) - np.count_nonzero(xm) >= 2:
                continue
            for yss in itertools.combinations(range(self.length), r):
                ym = np.diff(np.sort([y // self.base for y in yss]))
                if len(ym) - np.count_nonzero(ym) >= 2:
                    continue
                for path in itertools.permutations(range(r), r):
                    for xr in range(r):
                        x = xs[xr]
                        ys = (path[xr], path[(xr + 1) % r])
                        wing_notes = [self.candi[x, y] for y in ys]
                        xwing_notes = [self.candi[x, y]
                            if y not in ys else 0
                            for y in range(self.length)]
                        wing_note = np.bitwise_and.reduce(wing_notes)
                        xwing_note = np.bitwise_or.reduce(xwing_notes)
                        wing_note_count = bin(wing_note).count('1')
                        xwing_note_count = bin(wing_note & xwing_note).count('1')
                        if wing_note_count - xwing_note_count == int(r/2):
                            for y in range(self.length):
                                if y in ys:
                                    continue
                                note = self.candi[x, y]
                                if note != wing_note:
                                    self.candi[x, y] -= note & wing_note

    def erase_swordfish_colwise(self):
        """erase candidates from the swordfish technique in col-wise."""
        r = 4
        for ys in itertools.combinations(range(self.length), r):
            ym = np.diff(np.sort([y // self.base for y in ys]))
            if len(ym) - np.count_nonzero(ym) >= 2:
                continue
            for xss in itertools.combinations(range(self.length), r):
                xm = np.diff(np.sort([x // self.base for x in xss]))
                if len(xm) - np.count_nonzero(xm) >= 2:
                    continue
                for path in itertools.permutations(range(r), r):
                    for yr in range(r):
                        y = ys[yr]
                        xs = (path[yr], path[(yr + 1) % r])
                        wing_notes = [self.candi[x, y] for x in xs]
                        ywing_notes = [self.candi[x, y]
                            if x not in xs else 0
                            for x in range(self.length)]
                        wing_note = np.bitwise_and.reduce(wing_notes)
                        ywing_note = np.bitwise_or.reduce(ywing_notes)
                        wing_note_count = bin(wing_note).count('1')
                        ywing_note_count = bin(wing_note & ywing_note).count('1')
                        if wing_note_count - ywing_note_count == int(r/2):
                            for x in range(self.length):
                                if x in xs:
                                    continue
                                note = self.candi[x, y]
                                if note != wing_note:
                                    self.candi[x, y] -= note & wing_note

    def check_consistency(self):
        """check self-consistency of the table"""
        for i, block in enumerate(self.blockwise_table(self.table)):
            for p in range(self.length):
                p_count = np.count_nonzero(block == p + 1)
                if p_count > 1:
                    raise ValueError(f'{i + 1:d}-th block has {p + 1:d}s of {p_count:d}.')
        for x, row in enumerate(self.table):
            for p in range(self.length):
                p_count = np.count_nonzero(row == p + 1)
                if p_count > 1:
                    raise ValueError(f'{x + 1:d}-th row has {p + 1:d}s of {p_count:d}.')
        for y, col in enumerate(np.transpose(self.table)):
            for p in range(self.length):
                p_count = np.count_nonzero(col == p + 1)
                if p_count > 1:
                    raise ValueError(f'{y + 1:d}-th col has {p + 1:d}s of {p_count:d}.')

    def update_table(self):
        """update the table from the candidates and check updated."""
        for x, y in itertools.product(range(self.length), range(self.length)):
            note = self.candi[x, y]
            element = self.table[x, y]
            if not element and note & (note - 1) == 0:
                for p in range(self.length):
                    if (note >> p) % 2 == 1:
                        value = p + 1
                        break
                self.table[x, y] = value

        if np.all(self.bygone == self.candi) and self.level >= 3:
            self.is_updated = False
        elif np.all(self.bygone == self.candi):
            self.level += 1
            self.is_updated = True
        else:
            self.is_updated = True

    def run(self):
        """run algorithms and check solved."""
        self.bygone = np.copy(self.candi)

        self.erase_trivial()
        self.erase_single()
        self.erase_unique()

        if self.level >= 1:
            self.erase_blockline()
            self.erase_blockblock()
            pass
        
        if self.level >= 2:
            self.erase_naked()
            # self.erase_hidden()
            pass

        if self.level >= 3:
            # self.erase_wing()
            # self.erase_swordfish()
            pass

        self.update_table()
        self.check_consistency()

        if np.all(self.table):
            self.is_solved = True
        else:
            self.is_solved = False
