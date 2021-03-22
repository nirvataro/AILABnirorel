import GeneticAlgorithm as GA
import xlsxwriter
import NQueens as NQ
import random


with xlsxwriter.Workbook('exel files/nqueens0age.xlsx') as workbook:
    worksheet = workbook.add_worksheet()
    for i in range(100):
        mut = random.choice([0, 1])
        select = random.choice([0, 3])
        cross = random.choice([0, 1])
        print("choices:", mut, cross, select)
        iter, time = NQ.main_nqueens(mut, cross, select)
        worksheet.write_row(i+1, 0, [iter, time])
        print(i, iter, time)