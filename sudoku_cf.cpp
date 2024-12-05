#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <iostream>
#include <memory>
#include <optional>
#include <ostream>
#include <random>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#define POPULATION_SIZE 2000
#define ITERATION_COUNT 250
#define MUTATION_PROBABILITY 0.9
// #define DEBUG
#define FILE_INPUT

namespace GeneticSudoku {
constexpr std::size_t SUDOKU_SIZE = 9;

using digit_t = unsigned char;
using Field = std::array<std::array<digit_t, SUDOKU_SIZE>, SUDOKU_SIZE>;
using fitness_t = unsigned int;
struct Individual;
using IndividualPtr = std::shared_ptr<Individual>;
using Population = std::vector<IndividualPtr>;

struct Individual {
    Field field;

  private:
    // Statistics about the initial field
    inline static Field initial_field;
    inline static std::array<std::size_t, SUDOKU_SIZE> free_position_count_by_rows;
    inline static std::array<std::array<std::size_t, SUDOKU_SIZE>, SUDOKU_SIZE> free_positions_by_rows;
    inline static std::array<std::array<digit_t, SUDOKU_SIZE>, SUDOKU_SIZE> available_digits_by_rows;
    // Info whether there is a digit in a row/column
    inline static std::array<std::array<bool, SUDOKU_SIZE>, SUDOKU_SIZE> locked_digits_by_columns,
        locked_digits_by_subgrids;

  public:
    static void set_initial_data(const Field &initial_field) {
        Individual::initial_field = initial_field;

        // Init structures
        std::array<std::array<bool, SUDOKU_SIZE>, SUDOKU_SIZE> digit_counter{};
        for (std::size_t i = 0; i < SUDOKU_SIZE; ++i) {
            free_position_count_by_rows[i] = 0;
            locked_digits_by_columns[i].fill(false);
            locked_digits_by_subgrids[i].fill(false);
        }

        // Count free positions and locked digits
        for (int i = 0; i < SUDOKU_SIZE; ++i) {
            for (int j = 0; j < SUDOKU_SIZE; ++j) {
                auto digit = initial_field[i][j];
                if (digit == 0) {
                    free_positions_by_rows[i][free_position_count_by_rows[i]++] = j;
                    continue;
                }
                digit--;
                digit_counter[i][digit] = true;
                locked_digits_by_columns[j][digit] = true;
                locked_digits_by_subgrids[(i / 3) * 3 + (j / 3)][digit] = true;
            }
        }

        // Compute available digits
        for (std::size_t i = 0; i < SUDOKU_SIZE; ++i) {
            int n = 0;
            for (digit_t digit = 0; digit < SUDOKU_SIZE; ++digit) {
                if (digit_counter[i][digit])
                    continue;
                available_digits_by_rows[i][n++] = digit;
            }
        }
    }

    Individual() = default;

    // Generate a random instance
    // Invariant: rows are correct
    template <class URBG>
        requires std::uniform_random_bit_generator<std::remove_cvref_t<URBG>>
    explicit Individual(URBG &&randomizer) : field{initial_field} { // NOLINT(*-pro-type-member-init)
        for (std::size_t i = 0; i < SUDOKU_SIZE; ++i) {
            std::shuffle(available_digits_by_rows[i].begin(),
                         available_digits_by_rows[i].begin() + free_position_count_by_rows[i], randomizer);
            for (std::size_t j = 0; j < free_position_count_by_rows[i]; ++j)
                field[i][free_positions_by_rows[i][j]] = available_digits_by_rows[i][j] + 1;
        }
    }

    [[nodiscard]] fitness_t fitness() const {
        fitness_t error = 0;

        std::array<std::array<unsigned int, SUDOKU_SIZE>, SUDOKU_SIZE> digits_by_columns{}, digits_by_subgrids{};
        for (std::size_t i = 0; i < SUDOKU_SIZE; ++i) {
            for (std::size_t j = 0; j < SUDOKU_SIZE; ++j) {
                if (initial_field[i][j] != 0)
                    continue;
                const auto digit = field[i][j] - 1;
                digits_by_columns[j][digit]++;
                digits_by_subgrids[(i / 3) * 3 + (j / 3)][digit]++;
            }
        }

        for (std::size_t i = 0; i < SUDOKU_SIZE; ++i) {
            for (digit_t digit = 0; digit < SUDOKU_SIZE; ++digit) {
                error +=
                    std::max(0, static_cast<int>(digits_by_columns[i][digit]) + locked_digits_by_columns[i][digit] - 1);
                error += std::max(0, static_cast<int>(digits_by_subgrids[i][digit]) +
                                         locked_digits_by_subgrids[i][digit] - 1);
            }
        }

        return error;
    }

    template <class URBG>
        requires std::uniform_random_bit_generator<std::remove_cvref_t<URBG>>
    void mutate(URBG &&randomizer) {
        // Choose a row
        std::uniform_int_distribution<std::size_t> row_choice{0, SUDOKU_SIZE - 1};
        std::size_t row;
        do {
            row = row_choice(randomizer);
        } while (free_position_count_by_rows[row] <= 1);

        // Generate two distinct indices for choosing cells in the row to swap
        std::uniform_int_distribution<std::size_t> cell_choice{0, free_position_count_by_rows[row] - 1};
        auto i = cell_choice(randomizer);
        auto j = cell_choice(randomizer, decltype(cell_choice)::param_type{0, free_position_count_by_rows[row] - 2});
        if (j >= i)
            ++j;

        std::swap(field[row][free_positions_by_rows[row][i]], field[row][free_positions_by_rows[row][j]]);
    }

    template <class URBG>
        requires std::uniform_random_bit_generator<std::remove_cvref_t<URBG>>
    static IndividualPtr makeChild(const Individual &mother, const Individual &father, URBG &&randomizer) {
        const auto child = std::make_shared<Individual>();
        Field &field = child->field;
        std::uniform_int_distribution coin{0, 1};

        for (int i = 0; i < SUDOKU_SIZE; ++i) {
            field[i] = coin(randomizer) ? mother.field[i] : father.field[i];
        }
        return child;
    }
};

const auto seed = 4203717821;//std::random_device{}();
std::mt19937 randomizer{seed};
class Solver {
    const std::size_t population_size;
    const unsigned int iteration_count;
    std::bernoulli_distribution mutation_decider;

    using FitnessCache = std::unordered_map<IndividualPtr, fitness_t>;

    template <class T>
        requires std::same_as<std::remove_cvref_t<T>, FitnessCache>
    static auto getCachingComparator(T &&cache) {
        return [&cache](const IndividualPtr &a, const IndividualPtr &b) mutable {
            typename std::remove_cvref_t<T>::iterator it;
            fitness_t temp;
            const fitness_t fitness_a =
                (it = cache.find(a)) != cache.end() ? it->second : (cache.emplace(a, temp = a->fitness()), temp);
            const fitness_t fitness_b =
                (it = cache.find(b)) != cache.end() ? it->second : (cache.emplace(b, temp = b->fitness()), temp);
            return fitness_a < fitness_b;
        };
    };

  public:
    Solver(const std::size_t population_size, const unsigned int iteration_count, const double mutation_probability)
        : population_size{population_size}, iteration_count{iteration_count}, mutation_decider{mutation_probability} {}

    std::optional<Field> solve(const Field &initial_field) {
        Individual::set_initial_data(initial_field);
        Population population = getInitialPopulation();

        FitnessCache cache{};
        std::ranges::sort(population, getCachingComparator(cache));
        for (int i = 0; i < iteration_count; ++i) {
            if (cache[population[0]] == 0) {
                return population[0]->field;
            }

            crossover(population);
            std::ranges::sort(population, getCachingComparator(cache));

            // Collect garbage (dead individuals)
            for (auto it = cache.begin(), end = cache.end(); it != end;) {
                if (it->first == population[0] || it->first.use_count() != 1) {
                    ++it;
                    continue;
                }
                cache.erase(it++);
            }
        }

#ifdef DEBUG
        std::cout << (cache.contains(population[0]) ? cache[population[0]] : population[0]->fitness()) << std::endl;
#endif
        return std::nullopt;
    }

  private:
    Population getInitialPopulation() {
        Population population;
        population.reserve(population_size);
        for (int i = 0; i < population_size; ++i) {
            population.emplace_back(std::make_shared<Individual>(randomizer));
        }
        return population;
    }

    void crossover(Population &population) {
        const auto child_count = static_cast<std::size_t>(0.5L * population_size);
        std::uniform_int_distribution<std::size_t> selector{0, population.size() - 1 - child_count};

        for (int i = 0; i < child_count; ++i) {
            IndividualPtr child =
                Individual::makeChild(*population[selector(randomizer)], *population[selector(randomizer)], randomizer);
            if (mutation_decider(randomizer)) {
                child->mutate(randomizer);
            }
            population[population_size - 1 - i] = std::move(child);
        }
    }
};
} // namespace GeneticSudoku

int main() {
    using namespace GeneticSudoku;

    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);

#ifdef FILE_INPUT
    std::freopen("input.txt", "r", stdin);
#endif
#ifdef FILE_OUTPUT
    std::freopen("output.txt", "w", stdout);
#endif

    std::cout << seed << std::endl;
    Field input_field;
    for (auto &row : input_field) {
        for (auto &cell : row) {
            std::cin >> cell;
            if (cell == '-')
                cell = 0;
            else
                cell -= '0';
        }
    }

    Solver solver{POPULATION_SIZE, ITERATION_COUNT, MUTATION_PROBABILITY};
    std::optional<Field> solution;
    for (int i = 0; !solution.has_value(); ++i) {
        solution = solver.solve(input_field);
    }

    for (std::size_t i = 0; i < SUDOKU_SIZE; ++i) {
        if (i > 0)
            std::cout << '\n';
        for (std::size_t j = 0; j < SUDOKU_SIZE; ++j) {
            const auto &answer_field = input_field[i][j] == 0 ? solution.value() : input_field;
            if (j > 0)
                std::cout << ' ';
            std::cout << static_cast<char>(answer_field[i][j] + '0');
        }
    }

    return 0;
}

