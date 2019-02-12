import itertools
import numpy as np

import sudoku2

class solver:
	def __init__(self, arg):
		if type(arg) == sudoku2.sudoku:
			self.board = arg
		elif type(arg) == str:
			self.board = sudoku2.sudoku(file = arg)
		else:
			raise ValueError('Wrong argument.')

	def solve_naked_single(self):
		cand = 0
		for x, y in itertools.product(*map(range, [self.board.base**2]*2)):
			note = self.board.notes[x, y]
			if self.board.is_power_of_2(note):
				self.board.table[x, y] = self.board.ilog2(note)+1
				cand += 1
		if cand > 0:
			print(f'{cand: d} naked singles are evaluated.')

	def solve_hidden_single(self):
		cand = 0
		def test_single(blocks):
			for block in blocks:
				for i in range(9):
					if 0 << i in block:
						continue
					for 


				1 << i
				map(lambda x, y: x & b, list)
				for element in block:


	def solve_naked_pair(self):
		cand = 0	
		virtual_board = self.board.notes.copy()
		def test_pair(blocks):
			for row in self.board.notes:
				for element in row:
					if bin(element).count('1') != 2
						continue





if __name__ == '__main__':
	solv = solver('map/base3_3_0.sudoku')
	solv.solve_naked_single()