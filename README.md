# Travelling Salesman Problem Solver

Simple library to solve instances of Symmetric [Travelling Salesman Problem](https://en.wikipedia.org/wiki/Travelling_salesman_problem).

## Installation
Copy `src` and `tests` folders to your project.

## Usage
```python
>>> from src import Solver
>>> s = Solver(algorithm='dynamic')
>>> shortest_tour = s.solve(path='a4_.tsp')
>>> shortest_tour
[0, 1, 3, 2]
```

## About
The description of *supported file format* can be found [here](https://wwwproxy.iwr.uni-heidelberg.de/groups/comopt/software/TSPLIB95/tsp95.pdf).

**Exact algorithms**

- [Brute force](https://en.wikipedia.org/wiki/Brute-force_search)
- [Branch and bound](https://en.wikipedia.org/wiki/Branch_and_bound)
- [Dynamic programming](https://en.wikipedia.org/wiki/Held%E2%80%93Karp_algorithm)

**Approximation algorithms**

- [Nearest neighbour (greedy)](https://en.wikipedia.org/wiki/Nearest_neighbour_algorithm)
- [Genetic](https://en.wikipedia.org/wiki/Genetic_algorithm)


## Dependencies
- [Python](https://www.python.org/downloads/): 3.5.1 or higher
- [pytest](https://docs.pytest.org/en/latest/): 4.5.0 or higher for tests


## Tests
Tests are implemented using `pytest`. To run all tests, please refer to [this piece](https://docs.pytest.org/en/latest/getting-started.html#run-multiple-tests) of *pytest* documentation.


## License
[MIT](LICENSE)

## Authors
- Aleksei Tcysin
- Malshani Ranchagoda
- Sivasundar Karthikeyan

