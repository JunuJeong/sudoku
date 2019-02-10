class board:
    def __init__(self):
        self.numbers = [[1234567890]*9 for i in range(9)]

    def __str__(self):
        boardstr = " "
        flag1 = 0
        for row in range(9):
            if flag1 % 3 == 0:
                boardstr += "+"+"-"*3+"+"+"-"*3+"+"+"-"*3+"+\n "
            flag1 += 1
            flag2 = 0
            for col in range(9):
                if flag2 % 3 == 0:
                    boardstr += "|"
                if self.numbers[row][col] % 10 == 0:
                    boardstr += " "
                else:
                    boardstr += "%1d" % self.numbers[row][col]
                flag2 += 1
            boardstr += "|\n "
        boardstr += "+"+"-"*3+"+"+"-"*3+"+"+"-"*3+"+"
        return boardstr

    def set_num(self, row, col, num):
        if type(num) == int and num > 0:
            self.numbers[row-1][col-1] = num

    def block(self, row, col):
        ret = []
        for i in range(3):
            for j in range(3):
                ret.append(self.numbers[i+(row-1)/3*3][j+(col-1)/3*3])
        return ret

    def row(self, row):
        return self.numbers[row-1]

    def column(self, col):
        ret = []
        for i in range(9):
            ret.append(self.numbers[i][col-1])
        return ret

    def unknowns(self):
        flag = 0
        for i in range(9):
            for j in range(9):
                if self.numbers[i][j] > 9:
                    flag += 1
        return flag


# Sub functions #

# All candidates list from given list-type block
    def get_cans(self, block):
        cans = []
        for can in block:
            if can < 10:
                continue
            else:
                cans += list({int(i) for i in str(can).replace('0', '')})
        return cans

# Unique candidates list from given list-type block
    def get_unique_can(self, block):
        can_str = ""
        unqs = []
        for can in block:
            if can < 10:
                continue
            else:
                can_str += str(can)
        for n in self.get_cans(block):
            if can_str.count(str(n)) == 1:
                unqs.append(n)
        return unqs

    def bounded_triple(self, i, j):
        flag = 10000
        # The SOMETHING including (i,j)-cell is "not" BOUNDED-TRIPLE
        for k in range(1, 10):
            # SOMETHING = row by ROW
            if (k-1)/3 != (i-1)/3 and\
                    self.row(i)[k] > 10 and str(flag)[4] == '0':
                flag += 1
            # SOMETHING = column by COL
            if (k-1)/3 != (j-1)/3 and\
                    self.col(j)[k] > 10 and str(flag)[3] == '0':
                flag += 10
            # SOMETHING = row by BLOCK
            if (k-1)/3 != (i-1)/3*(j % 3) and\
                    self.block(i, j)[k] > 10 and str(flag)[2] == '0':
                flag += 100
            # SOMETHING = column by BLOCK
            if (k-1)/3 != (j-1)/3*(i % 3) and\
                    self.block(i, j)[k] > 10 and str(flag)[1] == '0':
                flag += 1000
        print("Bounded_triple is checking... (%d, %d) : %d" % (i, j, flag))
        '''
        This function focuses on the specific three cells including (i,j)-cell.
        That three cells is determined by an intersection between row(column)
        and block, both sets contain (i,j)-cell.
        The first condition related to 'k' excludes three cells and remained
        cells will be checked whether it is bounded or not.
        If the set of three cells is bounded, the return is 0.
        '''

# GENERATE or REMOVE CANDIDATE #

# Direct generating.
    def can_a(self):
        for i in range(1, 10):
            for j in range(1, 10):
                if self.numbers[i-1][j-1] < 10:
                    continue
                candidates = ""
                for k in range(1, 10):
                    if k not in self.row(i)+self.block(i, j)+self.column(j):
                        candidates += str(k)
                if len(candidates) == 0:
                    print("The puzzle has a problem type CAN_A.")
                self.set_num(i, j, int(candidates+"0"))
        # print "The candidates are generated directly."

# Indriect generating.
    def can_b(self):
        for i in range(1, 4):
            for j in range(1, 4):
                continue


# GET NUMBER #

# Direct evaluation.
    def sol_a(self):
        flag = 0
        for i in range(9):
            for j in range(9):
                if len(str(self.numbers[i][j])) == 2:
                    self.set_num(i+1, j+1, int(str(self.numbers[i][j])[0]))
                    flag += 1
        if flag > 0:
            print("Single candidates are evaluated.")

# Unique candidate.
    def sol_b(self):
        flag = 0
        temp_cand = []
        for i in range(1, 10):
            for n in range(1, 10):
                if n in self.get_unique_can(self.row(i)):
                    for j in range(1, 10):
                        if str(n) in str(self.row(i)[j-1]):
                            temp_cand.append([i, j, n])
                            flag += 1
                            break
                if n in self.get_unique_can(self.block((i-1)/3*3+1, (i-1) % 3*3+1)):
                    for j in range(1, 10):
                        if str(n) in str(self.block((i-1)/3*3+1, (i-1) % 3*3+1)[j-1]):
                            temp_cand.append([(i-1)/3*3+(j-1)/3+1, (i-1) % 3*3+(j-1) % 3+1,n])
                            flag += 1
                            break
                if n in self.get_unique_can(self.column(i)):
                    for j in range(1,10):
                        if str(n) in str(self.column(i)[j-1]):
                            #self.set_num(j,i,n)
                            temp_cand.append([j,i,n])
                            flag+=1
                            break
        if flag>0:
            for i in range(len(temp_cand)):
                self.set_num(temp_cand[i][0],temp_cand[i][1],temp_cand[i][2])
            print("Unique candidates are evaluated.")


a=board()
a.set_num(1,1,5)
a.set_num(1,4,6)
a.set_num(1,5,7)
a.set_num(1,7,8)
a.set_num(2,2,4)
a.set_num(2,4,8)
a.set_num(3,1,8)
a.set_num(3,4,5)
a.set_num(3,7,6)
a.set_num(3,8,1)
a.set_num(3,9,3)
a.set_num(4,2,6)
a.set_num(4,3,2)
a.set_num(4,4,4)
a.set_num(4,8,7)
a.set_num(5,1,1)
a.set_num(5,6,3)
a.set_num(5,8,2)
a.set_num(6,1,3)
a.set_num(6,2,7)
a.set_num(6,3,4)
a.set_num(6,4,9)
a.set_num(6,6,8)
a.set_num(7,2,9)
a.set_num(7,3,6)
a.set_num(7,4,1)
a.set_num(7,6,7)
a.set_num(7,7,8)
a.set_num(7,9,2)
a.set_num(8,1,2)
a.set_num(8,2,1)
a.set_num(8,3,8)
a.set_num(8,6,6)
a.set_num(8,8,4)
a.set_num(8,9,5)
a.set_num(9,2,5)
a.set_num(9,5,8)
a.set_num(9,8,9)





'''
a.set_num(1,2,4)
a.set_num(1,3,7)
a.set_num(1,6,2)
a.set_num(1,7,3)
a.set_num(1,8,9)
a.set_num(2,1,9)
a.set_num(2,3,3)
a.set_num(2,4,1)
a.set_num(2,6,6)
a.set_num(2,7,7)
a.set_num(3,1,5)
a.set_num(3,4,3)
a.set_num(3,5,7)
a.set_num(3,9,4)
a.set_num(4,1,8)
a.set_num(4,2,9)
a.set_num(4,3,1)
a.set_num(5,2,7)
a.set_num(5,8,3)
a.set_num(6,7,2)
a.set_num(6,8,4)
a.set_num(6,9,9)
a.set_num(7,1,6)
a.set_num(7,5,2)
a.set_num(7,6,7)
a.set_num(7,9,1)
a.set_num(8,3,9)
a.set_num(8,4,8)
a.set_num(8,6,3)
a.set_num(8,7,6)
a.set_num(8,9,2)
a.set_num(9,2,5)
a.set_num(9,3,2)
a.set_num(9,4,6)
a.set_num(9,7,4)
a.set_num(9,8,8)
'''
'''

#Unique Candidate
for i in range(5,10):
    a.set_num(1,i,i)
a.set_num(4,1,2)
a.set_num(4,2,3)
a.set_num(5,4,2)
a.set_num(6,4,3)
a.set_num(7,2,2)
a.set_num(7,3,3)
'''
"""
#Indirect generating
for i in range(4,9):
    a.set_num(1,i,i)
    a.set_num(2,i,13-i)

a.set_num(1,9,9)
a.set_num(4,3,4)
a.set_num(5,3,3)

print a

print "Unknown numbers: %2d\nThe solver works now:\n"%a.unknowns()
step=0
while(a.unknowns()>0):
    print ""
    a.can_a()
    a.sol_a()
    a.can_a()
    a.sol_b()
    print a,"\nUnknown numbers: %2d"%a.unknowns()
    step+=1
    if a.unknowns()==0:
        print "Total steps: %d"%step
        break
    b=raw_input("Continue?")
    if b!='y': break

print "="*40
"""