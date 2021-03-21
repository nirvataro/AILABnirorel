import GeneticAlgorithm as GA
import xlsxwriter
import NQueens as NQ
import random


results_Nqueens_500pop = []

for i in range(100):
    mut = random.choice([0,1])
    select = random.choice([0, 3])
    cross = random.choice([0, 1])
    print("choices:", mut, cross, select)
    results_Nqueens_500pop.append(NQ.main_nqueens(mut, cross, select))
    print(i)


with xlsxwriter.Workbook('exel files/nqueens500pop.xlsx') as workbook:
    worksheet = workbook.add_worksheet()
    for row_num, data in enumerate(results_Nqueens_500pop):
        worksheet.write_row(row_num, 0, data)


