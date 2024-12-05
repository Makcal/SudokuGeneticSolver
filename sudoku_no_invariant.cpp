#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <ostream>
#include <random>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace GeneticSudoku {
constexpr std::size_t SUDOKU_SIZE = 9;

// Integer exponentiation
constexpr int ipow(int base, int exp) {
    int result = 1;
    for (;;) {
        if (exp & 1)
            result *= base;
        exp >>= 1;
        if (!exp)
            break;
        base *= base;
    }

    return result;
}

struct Individual;

using Field = std::array<std::array<unsigned char, SUDOKU_SIZE>, SUDOKU_SIZE>;
using fitness_t = unsigned int;
using IndividualPtr = std::shared_ptr<Individual>;
using Population = std::vector<IndividualPtr>;

struct Individual {
    Field field;

  private:
    // Statistics about the initial field
    inline static Field initial_field;
    inline static std::size_t free_position_count;
    inline static std::array<std::pair<std::size_t, std::size_t>, SUDOKU_SIZE * SUDOKU_SIZE> free_positions;
    inline static std::array<unsigned char, SUDOKU_SIZE * SUDOKU_SIZE> available_digits;
    // Info whether there is a digit in a row/column
    inline static std::array<std::array<bool, SUDOKU_SIZE>, SUDOKU_SIZE> locked_digits_by_rows,
        locked_digits_by_columns, locked_digits_by_subgrids;

  public:
    static void set_initial_data(const Field &initial_field) {
        Individual::initial_field = initial_field;

        // Init structures
        free_position_count = 0;
        std::array<int, SUDOKU_SIZE> digit_counter{};
        for (std::size_t i = 0; i < SUDOKU_SIZE; ++i) {
            locked_digits_by_rows[i].fill(false);
            locked_digits_by_columns[i].fill(false);
            locked_digits_by_subgrids[i].fill(false);
        }

        // Count free positions and locked digits
        for (int i = 0; i < SUDOKU_SIZE; ++i) {
            for (int j = 0; j < SUDOKU_SIZE; ++j) {
                auto digit = initial_field[i][j];
                if (digit == 0) {
                    free_positions[free_position_count++] = {i, j};
                    continue;
                }
                digit--;
                digit_counter[digit]++;
                locked_digits_by_rows[i][digit] = true;
                locked_digits_by_columns[j][digit] = true;
                locked_digits_by_subgrids[(i / 3) * 3 + (j / 3)][digit] = true;
            }
        }

        // Compute available digits
        for (int digit = 1, n = 0; digit <= SUDOKU_SIZE; ++digit) {
            for (int i = 0; i < SUDOKU_SIZE - digit_counter[digit - 1]; ++i) {
                available_digits[n++] = digit;
            }
        }
    }

    Individual() = default;

    // Generate a random instance
    template <class URBG>
        requires std::uniform_random_bit_generator<std::remove_cvref_t<URBG>>
    explicit Individual(URBG &&randomizer) { // NOLINT(*-pro-type-member-init)
        std::shuffle(available_digits.begin(), available_digits.begin() + free_position_count, randomizer);

        auto it = available_digits.cbegin();
        for (std::size_t pos_i = 0; pos_i < free_position_count; ++pos_i) {
            const auto [i, j]{free_positions[pos_i]};
            field[i][j] = *it++;
        }
    }

    [[nodiscard]] fitness_t fitness() const {
        fitness_t error = 0;

        std::array<std::array<unsigned int, SUDOKU_SIZE>, SUDOKU_SIZE> digits_by_rows{}, digits_by_columns{},
            digits_by_subgrids{};
        for (std::size_t i = 0; i < SUDOKU_SIZE; ++i) {
            for (std::size_t j = 0; j < SUDOKU_SIZE; ++j) {
                if (initial_field[i][j] != 0)
                    continue;
                const auto digit = field[i][j] - 1;
                digits_by_rows[i][digit]++;
                digits_by_columns[j][digit]++;
                digits_by_subgrids[(i / 3) * 3 + (j / 3)][digit]++;
            }
        }

        for (unsigned char digit = 0; digit < SUDOKU_SIZE; ++digit) {
            constexpr int repeat_penalty_with_locked = 5, repeat_penalty = 3;
            // by rows
            for (std::size_t i = 0; i < SUDOKU_SIZE; ++i) {
                const auto digits_in_row = digits_by_rows[i][digit];
                if (!digits_in_row || digits_in_row == 1 && !locked_digits_by_rows[i][digit])
                    continue;
                error += ipow(locked_digits_by_rows[i][digit] ? repeat_penalty_with_locked : repeat_penalty,
                              static_cast<int>(digits_in_row));
                error += digits_in_row - 1;
            }
            // by columns
            for (std::size_t j = 0; j < SUDOKU_SIZE; ++j) {
                const auto digits_in_column = digits_by_columns[j][digit];
                if (!digits_in_column || digits_in_column == 1 && !locked_digits_by_columns[j][digit])
                    continue;
                error += ipow(locked_digits_by_columns[j][digit] ? repeat_penalty_with_locked : repeat_penalty,
                              static_cast<int>(digits_in_column));
                error += digits_in_column - 1;
            }
            // by subgrids
            for (std::size_t subgrid_i = 0; subgrid_i < SUDOKU_SIZE; ++subgrid_i) {
                const auto digits_in_subgrid = digits_by_subgrids[subgrid_i][digit];
                if (!digits_in_subgrid || digits_in_subgrid == 1 && !locked_digits_by_subgrids[subgrid_i][digit])
                    continue;
                error += ipow(locked_digits_by_columns[subgrid_i][digit] ? repeat_penalty_with_locked : repeat_penalty,
                              static_cast<int>(digits_in_subgrid));
                error += digits_in_subgrid - 1;
            }
        }

        return error;
    }

    template <class URBG>
        requires std::uniform_random_bit_generator<std::remove_cvref_t<URBG>>
    void mutate(URBG &&randomizer) {
        // Generate two distinct indices for choosing cells to swap
        std::uniform_int_distribution<std::size_t> cell_choice{0, free_position_count - 1};
        auto i = cell_choice(randomizer);
        auto j = cell_choice(randomizer, decltype(cell_choice)::param_type{0, free_position_count - 2});
        if (j >= i)
            ++j;

        std::swap(field[free_positions[i].first][free_positions[i].second],
                  field[free_positions[j].first][free_positions[j].second]);
    }

    template <class URBG>
        requires std::uniform_random_bit_generator<std::remove_cvref_t<URBG>>
    static IndividualPtr makeChild(const Individual &mother, const Individual &father, URBG &&randomizer) {
        const auto child = std::make_shared<Individual>();
        Field &field = child->field;
        std::uniform_int_distribution coin{0, 1};

        for (int i = 0; i < SUDOKU_SIZE; ++i) {
            for (int j = 0; j < SUDOKU_SIZE; ++j) {
                if (initial_field[i][j] != 0)
                    continue;
                field[i][j] = coin(randomizer) ? mother.field[i][j] : father.field[i][j];
            }
        }
        return child;
    }

  private:
    static unsigned int computeErrorForSet(std::array<unsigned char, SUDOKU_SIZE> &&set) {
        std::ranges::sort(set);
        int sum = 0;
        for (int i = 0; i < SUDOKU_SIZE; ++i) {
            sum += std::abs(i + 1 - set[i]);
        }
        return sum;
    }
};

class Solver {
    const std::size_t population_size;
    const unsigned int iteration_count;
    /*std::mt19937 randomizer{std::random_device{}()};*/
    std::mt19937 randomizer{1812};
    std::bernoulli_distribution mutation_decider;
    std::uniform_int_distribution<> mutation_count_generator{0, 20};

    using FitnessCache = std::unordered_map<IndividualPtr, int>;

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

    Field solve(const Field &initial_field) {
        Individual::set_initial_data(initial_field);
        Population population = getInitialPopulation();

        FitnessCache cache{};
        std::ranges::sort(population, getCachingComparator(cache));
        for (int i = 0; i < iteration_count; ++i) {
            if (i % 100 == 1)
                std::cout << "Iteration " << i << '\n';
            if (cache[population[0]] == 0) {
                std::cout << "Good\n";
                return population[0]->field;
            }

            /*std::ranges::shuffle(population, randomizer);*/
            crossover(population, cache);
            std::ranges::sort(population, getCachingComparator(cache));
            population.resize(population_size);

            // Collect garbage (dead individuals)
            for (auto it = cache.begin(), end = cache.end(); it != end;) {
                if (it->first.use_count() == 1)
                    cache.erase(it++);
                else
                    ++it;
            }
        };

        std::cout << cache[population[0]] << std::endl;
        return population[0]->field;
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

    void crossover(Population &population, FitnessCache &fitness_cache) {
        const std::size_t child_count = population.size();
        population.reserve(population.size() + child_count);
        std::uniform_int_distribution<std::size_t> selector{0, population.size() - 1};

        for (int i = 0; i < child_count; ++i) {
            std::array<IndividualPtr, 10> random;
            std::ranges::generate(random, [&] { return population[selector(randomizer)]; });
            std::ranges::sort(random, getCachingComparator(fitness_cache));
            IndividualPtr child = Individual::makeChild(*random[0], *random[1], randomizer);
            if (mutation_decider(randomizer)) {
                const int n = mutation_count_generator(randomizer);
                for (int j = 0; j < n; ++j)
                    child->mutate(randomizer);
            }
            population.emplace_back(std::move(child));
        }
    }
};
} // namespace GeneticSudoku

int main() {
    using namespace GeneticSudoku;

    std::freopen("input.txt", "r", stdin);
    /*std::freopen("output.txt", "w", stdout);*/

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

    Solver solver{5000, 4000, 0.10};
    const Field solution = solver.solve(input_field);
    for (std::size_t i = 0; i < SUDOKU_SIZE; ++i) {
        for (std::size_t j = 0; j < SUDOKU_SIZE; ++j) {
            const auto &answer_field = input_field[i][j] == 0 ? solution : input_field;
            std::cout << static_cast<char>(answer_field[i][j] + '0') << ' ';
        }
        std::cout << '\n';
    }

    return 0;
}
