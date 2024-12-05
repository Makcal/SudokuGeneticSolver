import subprocess
from time import time
import matplotlib.pyplot as plt
import random
from dokusan import generators

# C++ коды
code = ["./build/sudoku"]
# Java коды
# code = ["java", "./submit.java"]
# Python коды
# code = ["python", "./submit.py"]

N_TESTS = 100

def read_sudoku(file):
    sudoku = []
    i = 0
    for line in file:
        i += 1
        if (i == 10):
            break
        row = list(map(int, line.split()))
        sudoku.append(row)
    return sudoku

def is_valid_sudoku(sudoku, input_file):
    # Проверка строк
    for row in sudoku:
        if len(set(row)) != 9 or any(num < 1 or num > 9 for num in row):
            print('строка', row)
            return False

    # Проверка столбцов
    for col in range(9):
        column = [sudoku[row][col] for row in range(9)]
        if len(set(column)) != 9:
            print('столбец', col)
            return False

    # Проверка 3x3 квадратов
    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):
            square = []
            for i in range(3):
                for j in range(3):
                    square.append(sudoku[box_row + i][box_col + j])
            if len(set(square)) != 9:
                print('квадрат')
                return False

    # Проверка совпадения с input
    row = 0
    for line in input_file:
        a = line.split()
        for column in range(9):
            if a[column] != '-' and int(a[column]) != sudoku[row][column]:
                print('строка', row)
                return False
        row += 1

    return True

def mapgen(numbers, input_file):
    # Сгенерировать полный решённый Судоку
    full_sudoku = list(map(int, str(generators.random_sudoku(avg_rank=0))))
    grid = [full_sudoku[i:i+9] for i in range(0, 81, 9)]

    # Составить список всех координат
    coords = [(i, j) for i in range(9) for j in range(9)]
    random.shuffle(coords)

    # Удаление чисел с проверкой на уникальность решения
    while sum(row.count(0) for row in grid) < (81 - numbers) and coords:
        x, y = coords.pop()
        grid[x][y] = 0

    # Записать результат в файл
    for row in grid:
        input_file.write(" ".join(map(str, row)).replace('0', '-') + "\n")

def main():
    exec_time_avg_easy = []
    avg_fitness_avg_easy = []
    max_fitness_avg_easy = []
    exec_time_avg_medium = []
    avg_fitness_avg_medium = []
    max_fitness_avg_medium = []
    exec_time_avg_hard = []
    avg_fitness_avg_hard = []
    max_fitness_avg_hard = []

    exec_time_avg = []
    avg_fitness_avg = []
    max_fitness_avg = []
    number_of_cells = []
    a = 20
    b = 41
    for cells in range(a, b):
        exec_time = []
        avg_fitness = []
        max_fitness = []
        for maps in range(N_TESTS):
            number_of_cells.append(cells)

            # генерация карты
            with open("input.txt", "w") as input_file:
                mapgen(cells, input_file)

            # запуск алгоритма
            with open("input.txt", "r") as input_file, open("output.txt", "w") as output_file:
                start = time()
                process1 = subprocess.Popen(code, stdin=input_file, stdout=output_file, stderr=subprocess.PIPE, text=True)
                process1.wait()
                exec_time.append(round(time() - start, 2))
                if maps == 0:
                    print('Тест', cells, maps, 'пройден за', exec_time[-1])

            # проверка на корректность решения
            with open("input.txt", "r") as input_file, open("output.txt", "r") as output_file:
                read = output_file.readlines()
                avg_fitness.append(float(read[1]))
                max_fitness.append(float(read[0]))
                read.pop(1)
                read.pop(0)
                sudoku = read_sudoku(read)
                if not is_valid_sudoku(sudoku, input_file):
                    print("Решение судоку некорректное.")
                    exit()
        if (30 <= cells <= 40):
            exec_time_avg_easy += exec_time
            avg_fitness_avg_easy += avg_fitness
            max_fitness_avg_easy += max_fitness
        elif (26 <= cells <= 29):
            exec_time_avg_medium += exec_time
            avg_fitness_avg_medium += avg_fitness
            max_fitness_avg_medium += max_fitness
        else:
            exec_time_avg_hard += exec_time
            avg_fitness_avg_hard += avg_fitness
            max_fitness_avg_hard += max_fitness
        exec_time_avg.append(sum(exec_time) / len(exec_time))
        avg_fitness_avg.append(sum(avg_fitness) / len(avg_fitness))
        max_fitness_avg.append(sum(max_fitness) / len(max_fitness))

    print('EASY')
    print('average time', sum(exec_time_avg_easy) / len(exec_time_avg_easy))
    print('maximum fitness', sum(max_fitness_avg_easy) / len(max_fitness_avg_easy))
    print('average fitness', sum(avg_fitness_avg_easy) / len(avg_fitness_avg_easy))
    print()
    print('MEDIUM')
    print('average time', sum(exec_time_avg_medium) / len(exec_time_avg_medium))
    print('maximum fitness', sum(max_fitness_avg_medium) / len(max_fitness_avg_medium))
    print('average fitness', sum(avg_fitness_avg_medium) / len(avg_fitness_avg_medium))
    print()
    print('HARD')
    print('average time', sum(exec_time_avg_hard) / len(exec_time_avg_hard))
    print('maximum fitness', sum(max_fitness_avg_hard) / len(max_fitness_avg_hard))
    print('average fitness', sum(avg_fitness_avg_hard) / len(avg_fitness_avg_hard))
    plt.figure(1)
    plt.plot([i for i in range(a, b)], avg_fitness_avg, linestyle='-', color='b')
    plt.title(f'Average avg fitness on last generation among {N_TESTS} tests per each N')
    plt.xlabel('Numbers provided (N)')
    plt.ylabel('Average avg fitness on last generation')
    plt.grid()
    plt.savefig(f"avgfit{N_TESTS}.png", dpi=400)

    plt.figure(2)
    plt.plot([i for i in range(a, b)], exec_time_avg, linestyle='-', color='b')
    plt.title(f'Average execution time among {N_TESTS} tests per each N')
    plt.xlabel('Numbers provided (N)')
    plt.ylabel('Average execution time, sec')
    plt.grid()
    plt.savefig(f"exec{N_TESTS}.png", dpi=400)

    plt.figure(3)
    plt.plot([i for i in range(a, b)], max_fitness_avg, linestyle='-', color='b')
    plt.title(f'Average max fitness on last generation among {N_TESTS} tests per each N')
    plt.xlabel('Numbers provided (N)')
    plt.ylabel('Average max fitness on last generation')
    plt.grid()
    plt.savefig(f"maxfit{N_TESTS}.png", dpi=400)

    plt.show()

if __name__ == "__main__":
    main()
