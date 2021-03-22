+import GeneticAlgorithm as GA
import xlsxwriter
import NQueens as NQ
import random


with xlsxwriter.Workbook('exel files/nqueens_select_3.xlsx') as workbook:
    worksheet_00 = workbook.add_worksheet(name="0_0")
    worksheet_10 = workbook.add_worksheet(name="1_0")
    worksheet_01 = workbook.add_worksheet(name="0_1")
    worksheet_11 = workbook.add_worksheet(name="1_1")
    for i in range(100):
        #mut = random.choice([0, 1])
        #select = random.choice([0, 3])
        #cross = random.choice([0, 1])

        print("choices:", 0, 0, 3)
        iter, time = NQ.main_nqueens(0, 0, 3)
        worksheet_00.write_row(i+1, 0, [iter, time])
        print(i, iter, time)

        print("choices:", 1, 0, 3)
        iter, time = NQ.main_nqueens(1, 0, 3)
        worksheet_10.write_row(i+1, 0, [iter, time])
        print(i, iter, time)

        print("choices:", 0, 1, 3)
        iter, time = NQ.main_nqueens(0, 1, 3)
        worksheet_01.write_row(i+1, 0, [iter, time])
        print(i, iter, time)

        print("choices:", 1, 1, 3)
        iter, time = NQ.main_nqueens(1, 1, 3)
        worksheet_11.write_row(i+1, 0, [iter, time])
