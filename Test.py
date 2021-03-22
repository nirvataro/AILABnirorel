import GeneticAlgorithm as GA
import xlsxwriter
import NQueens as NQ
import random


with xlsxwriter.Workbook('exel files/hit.xlsx') as workbook:
    worksheet_Uniform = workbook.add_worksheet(name="Uniform")
    worksheet_one = workbook.add_worksheet(name="one")
    worksheet_two = workbook.add_worksheet(name="two")

    x, y, z = 1, 1, 1
    for i in range(100):
        print("choices:", 0, 1, 0)
        iter, time = GA.gen_alg(0, 1, 0)
        if iter != -1:
            worksheet_Uniform.write_row(x, 0, [iter, time])
            x += 1
        print(i, iter, time)

        print("choices:", 1, 1, 0)
        iter, time = GA.gen_alg(1, 1, 0)
        if iter != -1:
            worksheet_one.write_row(y, 0, [iter, time])
            y += 1
        print(i, iter, time)

        print("choices:", 2, 1, 0)
        iter, time = GA.gen_alg(2, 1, 0)
        if iter != -1:
            worksheet_two.write_row(z, 0, [iter, time])
            z += 1
        print(i, iter, time)
