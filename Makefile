all: build/sudoku

build/sudoku: sudoku.cpp
	g++ --std=c++23 sudoku.cpp -o build/sudoku -Ofast

run: build/sudoku
	./build/sudoku

clean:
	rm ./build/sudoku
