import random


results_Nqueens_1000pop = []
results_Nqueens_MUTATIONRATE_05 = []

for i in range(100):
    mut = random.choice([0,1])
    mut = random.choice([0, 1])
    select = random.choice([0, 3])
    cross = random.choice([0, 1])
    print("choices:", mut, cross, select)
    results_Nqueens_1000pop.append(NQ.main_nqueens(mut, cross, select))
    results_Nqueens_MUTATIONRATE_05.append(NQ.main_nqueens(mut, cross, select))
    print(i)


with xlsxwriter.Workbook('exel files/nqueens1000pop.xlsx') as workbook:
with xlsxwriter.Workbook('exel files/results_Nqueens_MUTATIONRATE_05.xlsx') as workbook:
    worksheet = workbook.add_worksheet()
    for row_num, data in enumerate(results_Nqueens_1000pop):
    for row_num, data in enumerate(results_Nqueens_MUTATIONRATE_05):
        worksheet.write_row(row_num, 0, data)