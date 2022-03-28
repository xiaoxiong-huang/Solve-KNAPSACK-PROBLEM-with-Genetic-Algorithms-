import random as rd
import time
import numpy as np
import math

##-------------------------------------------------------------
##  Box object, represent box we going to put into our backpack
##  id:         the id for each box.
##  weight:     the weight of the box.
##  importance: the importance value of the box
##-------------------------------------------------------------
class Box:
    def __init__(self, id: int, weight: int, importance: int) -> None:
        self.id = id
        self.weight = weight
        self.importance = importance

    def __str__(self) -> str:
        return "Box ID: " + str(self.id) + ", Box weight: " \
            + str(self.weight) + ", Box importance: " + str(self.importance)

##-------------------------------------------------------------
##   This is a genetic algo, with all the configration and population
##   needed to run a genetic algorithm as class varbile. First population
##   is generated and sorted when this object init.
##
##   maxIteration:      Max number of iteration we will run for 1 problem.
##                      Could not be None, has a defult value of 1000.
##
##   converageLimit:    Max number of iteration we will run for 1 problem,
##                      before we stop when the result stop getting better.
##
##   maxTime:           Max time we will run the problem, once the operation
##                      time is bigger than maxtime, iteration will stop.
##
##   populationSize:    Size of the population.
##
##   population:        Population is a list of all individual. 
##
##   individual:        Each individual is a list of bool, each element                   
##                      in the individual list: True if we want this box 
##                      in our backpack, False otherwise.
##-------------------------------------------------------------
class GenAlgo:
    def __init__(self,maxIteration: int = 1000, converageLimit: int = None,
                 maxTime: float = None, populationSize: int = 256) -> None:
        self.maxIteration = maxIteration
        self.converageLimit = converageLimit
        self.maxTime = maxTime
        self.boxSize = len(Boxs)
        self.populationSize = populationSize
        self.population = self.generate_population()
        self.sort_population()

    #   return the weight for a backpack.
    def get_weight(self, boxSet: list[Box]) -> int :
        weight = 0
        for box in boxSet:
            weight += box.weight
        return weight

    #   Generate a individual list, index with type boolean.
    #   True is when we want to use that box, False otherwise.
    #   Reason for make index type bool is we can use numpy filter.
    def generate_individual(self) -> list[bool]:
        individual: list[bool] = []

        # we want individual have same size with box set.
        for i in range(len(Boxs)):
            individual.append(rd.choice([True, False]))

        return individual

    #   generate the population, 
    #   used function : generate_individual()
    def generate_population(self) -> list[list[bool]]:
        population: list[list[bool]] = []

        for i in range(self.populationSize):
            population.append(self.generate_individual())

        return population

    #   return the fitness value
    def fitnes_func(self, individual: list[bool]) -> int:
        # use filter to get boxs list only contains 
        # box item described in individual list
        filtedBoxs: list[Box] = Boxs[individual]

        weight: int = 0
        importance: int = 0
        fitness: float = 0

        # get the weight and importance for a backpack
        for box in filtedBoxs:
            importance += box.importance
            weight += box.weight
        
        # weightRate is for individuals that exceed the weight limit
        # larger weight will have a smaller rate, which will cause the exp
        # function return a smaller result.
        # individuals that did not exceed the weight limit will have
        # rate of 0.
        weightRate = -max(0, (weight - maxWeight) / maxWeight) * 3
        fitness = importance * math.exp(weightRate)
        return fitness

    #   Sort the population using the fitness value.
    def sort_population(self) -> None:
        # create a new list of (fitness, indiv) so we can sort population
        # by fitness value.
        populationWithFitness: list[(float, list[bool])] = [] 
        # for each indiv in current population
        for individual in self.population:
            # get the fitness value by calling fitness function.
            fitness = self.fitnes_func(individual)
            # add the (fitness, indiv) to our unsorted list
            populationWithFitness.append((fitness, individual))
        # sort the unsorted list 
        populationWithFitness.sort(key = lambda x: x[0], reverse = True)
        # update the population with only indiv
        self.population = [i[1] for i in populationWithFitness]
    
    #   Fringe operation, crossover two indiv to product two new indiv
    #   only return one of then.
    def cross_product(self, indiv1: list[bool], indiv2: list[bool]) -> list[bool]:
        newIndiv1, newIndiv2 = [], []
        # randomly select a cut index where we will "crossover".
        curIndex = rd.randint(0, 12)
        # generate two new indivs use old indivs
        newIndiv1 = indiv1[curIndex:] + indiv2[:curIndex]
        newIndiv2 = indiv2[curIndex:] + indiv1[:curIndex]

        # random select which new indiv we will add 
        # to our population
        if rd.choice([True, False]):
            return newIndiv1
        return newIndiv2

    #   Fringe operation: mutate the population.
    def population_mutation(self) -> None:
        for indiv in self.population:
            # Each indiv have 0.05 chance to become mutated.
            if(np.random.uniform(0, 1) < 0.05):
                # Random select a index to mutate
                mutateIndex = rd.randint(0, len(Boxs)-1)
                # convert true to false, false to true
                indiv[mutateIndex] = not indiv[mutateIndex]

    def cull_Reproduct(self) -> None:
        # cull the population by half
        self.populationSize = int(self.populationSize / 2)
        self.population = self.population[:self.populationSize]
        

        # sicne we culled population by half,
        # we need to reproduct same amount
        for i in range(self.populationSize):
            # randomly draw two indiv from our culled population
            indiv1, indiv2 = rd.sample(self.population, 2)

            # call cross product give us a new born to put into the 
            # culled population
            self.population.append(
                self.cross_product(indiv1, indiv2))
        # reset the populstion size
        self.populationSize = self.populationSize * 2
        # mutate the population
        self.population_mutation()

    def run(self) -> list[Box]:
        # start the timmer
        start = time.time()
        # keep track the number of iteration
        count = 0
        # keep track number of iteration since result stop getting better
        converageCount = 0
        # keep the best indiv(i.e. indiv with height fitness)
        bestIndiv: list[Box] = Boxs[self.population[0]].copy()
        # keep the best fitness value we got
        maxFitness: float = self.fitnes_func(self.population[0])

        while True:
            # start a new iteration by cull and reproduct the population
            self.cull_Reproduct()
            # sort current population
            self.sort_population()
            # keep the current hieghtest fitness score, which will be the very first 
            # indiv in our population
            tempMaxFitness = self.fitnes_func(self.population[0])
            # compare the current indiv to the over all heightest indiv,
            # update the result if we got better
            if maxFitness < tempMaxFitness:
                maxFitness = tempMaxFitness
                bestIndiv = Boxs[self.population[0]].copy()
                # reset the number of iteration for converage,
                # because we just found a better solution.
                converageCount = 0
            # add the converage count and number of iteration by one
            # for each iteration.
            converageCount += 1
            count += 1

            # default iteration stoper, break out the loop when number of
            # iteration is exceed the max.
            if self.maxIteration is not None and self.maxIteration <= count:
                print("Find the fitest set with fitness of : " + str(maxFitness))
                print("Weight for this set is : " + str(self.get_weight(bestIndiv)))
                print("Number of iteration: " + str(count))
                print("Total time used: " + str(time.time() - start))
                return bestIndiv
            
            # if user passed in converage limit as config, stop the loop when number of
            # converaged iteration is exceed the max.
            if self.converageLimit is not None and self.converageLimit <= converageCount:
                print("Find the fitest set with fitness of : " + str(maxFitness))
                print("Weight for this set is : " + str(self.get_weight(bestIndiv)))
                print("Number of iteration: " + str(count))
                print("Total time used: " + str(time.time() - start))
                return bestIndiv

            # if user passed in max time as config, stop the loop when we exceed the 
            # time limit.
            if self.maxTime is not None and self.maxTime <= (time.time() - start):
                print("Find the fitest set with fitness of : " + str(maxFitness))
                print("Weight for this set is : " + str(self.get_weight(bestIndiv)))
                print("Number of iteration: " + str(count))
                print("Total time used: " + str(time.time() - start))
                return bestIndiv
        


def main():
    global Boxs, maxWeight
    maxWeight = 250
    Boxs = np.array([Box(1, 20, 6), Box(2, 30, 5),
                                 Box(3, 60, 8), Box(4, 90, 7),
                                 Box(5, 50, 6), Box(6, 70, 9),
                                 Box(7, 30, 4), Box(8, 30, 5),
                                 Box(9, 70, 4), Box(10, 20, 9),
                                 Box(11, 20, 2), Box(12, 60, 1)])
    print("---------------------------------------------------")
    print("--------------Welcome to Genetic algo!-------------")
    print("Before we start, please answer some question below!")
    print("---------------------------------------------------")
    numberOfRun = 6
    maxIteration = 1000
    maxTime = None
    converageLimit = None
    populationSize = 256
    temp = input("How many times do you want to run the Genetic algo?(int) ")
    if temp:
        numberOfRun = int(temp)
    print("---------------------------------------------------")
    temp = input("What is the max number of iteration do you want to do for each run?(int) ")
    if temp:
        maxIteration = int(temp)
    print("---------------------------------------------------")
    temp = input("What is the max amount of time you want to spend on each run?(float")
    if temp:
        maxTime = float(temp)
    print("---------------------------------------------------")
    temp = input("If the algo start to show evidence of converage, how many more iteration will convence you?(int) ")
    if temp:
        converageLimit = int(temp)
    print("---------------------------------------------------")
    temp = input("Please give me a population size(int): ")
    if temp:
        populationSize = int(temp)
    input("--------------Press enter to start!----------------")
    print("===================================================")
    for i in range(numberOfRun):
        print("NO. " + str(i+1))
        solution = GenAlgo(maxIteration=maxIteration, maxTime=maxTime, converageLimit=converageLimit, populationSize=populationSize)
        maxindiv = solution.run()

        for box in maxindiv:
            print(box)
        print("===================================================")
        



if __name__ == "__main__":
    main()