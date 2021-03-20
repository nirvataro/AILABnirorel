import GeneticAlgorithm as GA
import xlsxwriter


results_reg_003 = []
results_pso_0 = []
results_reg_013 = []
results_pso_1 = []
for i in range(100):
    results_reg_003.append(GA.gen_alg(0, 0, 3))
    results_pso_0.append(GA.gen_alg_PSO(0))
    results_reg_013.append(GA.gen_alg(0, 1, 3))
    results_pso_1.append(GA.gen_alg_PSO(1))
    print(i)


with xlsxwriter.Workbook('exel files/results_30.xlsx') as workbook:
    worksheet = workbook.add_worksheet()
    worksheet.write_row(0, 0, "reg_003")
    worksheet.write_row(0, 4, "pso_0")
    worksheet.write_row(0, 7, "reg_013")
    worksheet.write_row(0, 10, "pso_1")
    for row_num, data in enumerate(results_reg_003):
        worksheet.write_row(row_num+1, 0, data)
    for row_num, data in enumerate(results_pso_0):
        worksheet.write_row(row_num+1, 4, data)
    for row_num, data in enumerate(results_reg_013):
        worksheet.write_row(row_num+1, 7, data)
    for row_num, data in enumerate(results_pso_1):
        worksheet.write_row(row_num+1, 10, data)





