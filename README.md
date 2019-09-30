# TSP Solver

![Just some random picture.](aco_tsp.png)

Simple library to solve instances of Symmetric [Travelling Salesman Problem](https://en.wikipedia.org/wiki/Travelling_salesman_problem).

Comes in two flavours:

- **src** python module
- **walkthrough** is simple demo-GUI that walks you through solving the problem


## Usage
```python
>>> from src.solver import Solver
>>> s = Solver(algorithm='greedy')
>>> path = 'a4_.tsp'
>>> shortest_tour = s.shortest_tour(path)
>>> shortest_tour
['1', '3', '4', '2']
```

To run **walkthorugh**, download the repository and double-click 
*Solver GUI* icon.


## About
The description of *supported file format* can be found [here](https://wwwproxy.iwr.uni-heidelberg.de/groups/comopt/software/TSPLIB95/tsp95.pdf).

Implemented approaches are listed below.

**Exact algorithms**

- [Brute force search](https://en.wikipedia.org/wiki/Brute-force_search)
- [Branch and bound algorithm](https://en.wikipedia.org/wiki/Branch_and_bound)

**Approximation algorithms**

- [Nearest neighbour (greedy) algorithm](https://en.wikipedia.org/wiki/Nearest_neighbour_algorithm)
- [Genetic algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm)


## Dependencies
- [pytest](https://docs.pytest.org/en/latest/): 4.5.0 or higher for tests


## Tests
Tests are implemented using `pytest`. To run all tests, please refer to [this piece](https://docs.pytest.org/en/latest/getting-started.html#run-multiple-tests) of *pytest* documentation.


## Documentation


## License
[MIT](LICENSE)
