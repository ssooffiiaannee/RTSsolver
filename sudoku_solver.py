import time
# a = [[5,3,0,0,7,0,0,0,0],
# 	 [6,0,0,1,9,5,0,0,0],
# 	 [0,9,8,0,0,0,0,6,0],
# 	 [8,0,0,0,6,0,0,0,3],
# 	 [4,0,0,8,0,3,0,0,1],
# 	 [7,0,0,0,2,0,0,0,6],
# 	 [0,6,0,0,0,0,2,8,0],
# 	 [0,0,0,4,1,9,0,0,5],
# 	 [0,0,0,0,8,0,0,7,9]]

# a1 = [[0,0,0,0,0,0,0,0,8],
# 	  [1,8,0,0,0,2,3,0,0],
# 	  [0,6,0,0,5,7,0,0,1],
# 	  [0,7,0,9,6,0,0,0,0],
# 	  [0,9,0,7,0,4,0,1,0],
# 	  [0,0,0,0,8,1,0,4,0],
# 	  [6,0,0,2,4,0,0,8,0],
# 	  [0,0,4,5,0,0,0,9,3],
#  	  [5,0,0,0,0,0,0,0,0]]

# a3 =   [[0, 0, 0, 0, 0, 0, 0, 0, 0],
# 		[0, 3, 0, 0, 0, 0, 1, 6, 0],
# 		[0, 6, 7, 0, 3, 5, 0, 0, 4],
# 		[6, 0, 8, 1, 2, 0, 9, 0, 0],
# 		[0, 9, 0, 0, 8, 0, 0, 3, 0],
# 		[0, 0, 2, 0, 7, 9, 8, 0, 6],
# 		[8, 0, 0, 6, 9, 0, 3, 5, 0],
# 		[0, 2, 6, 0, 0, 0, 0, 9, 0],
# 		[0, 0, 0, 0, 0, 0, 0, 0, 0]]

a4 =   [[0, 0, 0, 0, 0, 0, 0, 0, 8],
		[1, 8, 0, 0, 0, 2, 3, 0, 0],
		[0, 6, 0, 0, 5, 7, 0, 0, 1],
		[0, 7, 0, 9, 6, 0, 0, 0, 0],
		[0, 9, 0, 7, 0, 4, 0, 1, 0],
		[0, 0, 0, 0, 8, 1, 0, 4, 0],
		[6, 0, 0, 2, 4, 0, 0, 8, 0],
		[0, 0, 4, 5, 0, 0, 0, 9, 3],
		[5, 0, 0, 0, 0, 0, 0, 0, 0]]

def print_sudoku(sudoku):
	for i in range(9):
		print(sudoku[i])

def solve(a, c, d, k):
	a[c][d] = k
	for i in range(9):
		for j in range(9):
			if(not a[i][j]):
				for k in range(1, 10):
					ans = True
					b = False
					for l in range(9):
						if(a[i][l] == k or a[l][j] == k or a[(i//3)*3 + l//3][(j//3)*3 + l%3] == k):
							ans = False
					if(ans):
						b = solve(a, i, j, k)
						if(b):
							more = input("Enter something:")
							print_sudoku(a)
							return False
							# return b
				a[c][d] = 0
				return False
	return a


b = solve(a4, 0, 0, a4[0][0])

if(b):
	print_sudoku(b)
else:
	print(False)