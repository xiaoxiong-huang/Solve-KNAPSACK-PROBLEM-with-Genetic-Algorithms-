# Homework 3 for CS131
Genetic Algorithms
Implemention of Genetic Algorithms

## Usage
```bash
#   Install numpy if you don't have it:
pip install numpy
#   Run the program:
python3 Informed_Search.py
```
## input format
After run the program, it will ask you for some config we need to run the program.
There are four ways to stop running a GA:
1. Default: when a GA finished run 1000 iteration.
2. MaxIteration: When you pass in the maxIteration, GA stop when finish run maxIterations iteration.
3. ConverageLimit: When you pass in the converageLimit, GA stop when result does not getting better
for converageLimit's iteration.
4. MaxTime: When you pass in the maxTime, GA stop after it run maxTime sec.
You don't have to enter any config, program will run and stop with default stopper/config. You also can enter more 
than one config, program will stop as soon as one of stopper had been trigger.

## Chromosome
Each chromosome is called indiv in program, which is a list of bool. each element in the individual 
list: True if we want this box in our backpack, False otherwise.

## Fitness
Each chromosome have a fitness value can be caculate. A weight rate is caculated, larger weight will have a smaller rate, 
then a exp function is used to caculate the penalty for over weight. Of crouse, for indiv that is not over weighted,
weight rate will be zero.

## Cull the population
Every generation will cull the population by half, where top half indiv with heigher fitness will be kept.

## Fringe operation
I used both crossover and mutation. Probility of indiv muation is fixed at 0.05 and crossover will random draw from top half of population.

## Detail
Detail about the code can be found in comment. I am assuming that the box set is fixed, so I hard coded the box object and the max weight of the backpack.
During testing, I have never seen the solution get stuck in a local max, but there could be a small chance. So I asked user to input number of GA user wish 
to run, help to avoid the local max stuck, it has a dafult value of 6. When enter the population size, PLEASE only only enter EVEN NUMBER!!!
