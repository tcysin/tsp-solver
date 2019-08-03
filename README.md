# TSP Solver
Simple module to solve instances of [Travelling Salesman Problem](https://en.wikipedia.org/wiki/Travelling_salesman_problem). Finds shortest tours 
for graphs like [these](https://wwwproxy.iwr.uni-heidelberg.de/groups/comopt/software/TSPLIB95/tsp/).

## About
Currently implemented approaches include:

**Exact algorithms**
- [Brute force search](https://en.wikipedia.org/wiki/Brute-force_search)
- [Branch and bound algorithm](https://en.wikipedia.org/wiki/Branch_and_bound)
- [Held-Karp (dynamic) algorithm](https://en.wikipedia.org/wiki/Held%E2%80%93Karp_algorithm)

**Approximation algorithms**
- [Nearest neighbour (greedy) algorithm](https://en.wikipedia.org/wiki/Nearest_neighbour_algorithm)
- [Genetic algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm)

## Usage
```
>>> from src.solver import Solver
>>> s = Solver(algorithm='greedy')
>>> path = 'a4_.tsp'
>>> shortest_tour = s.shortest_tour(path)
>>> shortest_tour
['1', '2', '4', '3']
```

## Dependencies
- [pytest](https://docs.pytest.org/en/latest/): 4.5.0 or higher for tests


## Tests
Tests are implemented using `pytest`.

## Documentation


## Authors


## License
[MIT](LICENSE.md)
