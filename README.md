# GRAVI
A gravity simulator using graph neural networks. Currently in development.

## Scope
This is not intended to be a scientific tool. It is being developed to support video games and entertainment.

## Approach
The bottleneck for a simulation of N newtonian gravitating point masses is that each body interacts with every other body - the complexity is quadratic with N. GRAVI hopes to find a "passable" solution that is linear with N using graph neural networks and a sparse interaction graph.