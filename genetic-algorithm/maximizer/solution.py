import random
import math
from typing import Tuple

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# POP_SIZE = 20
# INTERVAL = (-1, 2)
# COEF = (-1, 1, 2)  # a, b, c pentru f(x) = ax^2 + bx + c
# PRECIZIE = 6
# P_CROSSOVER = 0.25
# P_MUTATION = 0.01
# GENERATII = 50
class GeneticAlgorithmForMaximization:
    def __init__(self,
                 populationSize: int,
                 domain: Tuple[float, float],
                 parameters: Tuple[float, float, float],
                 precision: int,
                 probCrossover: int,
                 probMutation: int,
                 generations: int):
        self.populationSize = populationSize
        self.domain = domain
        self.parameters = parameters
        self.precision = precision
        self.probCrossover = probCrossover
        self.probMutation = probMutation
        self.generations = generations

        # chromosome
        self.intervalCount = (domain[1] - domain[0]) * (10 ** precision)  # (b-a) * 10 ^ p
        self.chromosomeLength = math.ceil(math.log2(self.intervalCount))  # l = [log2((b-a) * 10^p)]
        self.population = self.GenerateRandomPopulation()

    def GenerateRandomPopulation(self):
        population = []
        for _ in range(self.populationSize):
            chromosome = ''.join(random.choice(['0', '1']) for _ in range(self.chromosomeLength))
            population.append(chromosome)
        return population

    # conversions
    def BinaryStringToInt(self, binaryString: str) -> int:
        return int(binaryString, 2)

    def ChromosomeToNumber(self, chromosome: str, domain: Tuple[float, float]) -> float:
        a = domain[0]
        b = domain[1]
        intVal = self.BinaryStringToInt(chromosome)
        maxVal = 2 ** self.chromosomeLength - 1
        realVal = a + (b - a) * (intVal / maxVal)
        return realVal

    def FitnessFunction(self, chromosome: str):
        # valoare nuemrica
        number = self.ChromosomeToNumber(chromosome, self.domain)

        ecuationRes = self.parameters[0] * number ** 2 + self.parameters[1] * number + self.parameters[2]

        return ecuationRes

    def PopulationAfterFitness(self):
        fitnessValues = [self.FitnessFunction(x) for x in self.population]
        fitnessSum = sum(fitnessValues)
        return fitnessValues, fitnessSum

    def SelectionProbabilities(self, fitnessValeus, fitnessSum):
        probabilities = [p / fitnessSum for p in fitnessValeus]
        cumulatedProbabilities = [0]
        for p in probabilities:
            cumulatedProbabilities.append(cumulatedProbabilities[-1] + p)
        return probabilities, cumulatedProbabilities

    def BinarySearchInterval(self, cumulatedProbabilities, value):
        left = 0
        high = len(cumulatedProbabilities) - 2
        while left <= high:
            mid = (left + high) // 2
            if cumulatedProbabilities[mid] <= value < cumulatedProbabilities[mid+1]:
                return mid
            elif value < cumulatedProbabilities[mid]:
                high = mid - 1
            else:
                left = mid + 1
        return len(cumulatedProbabilities) - 2 # esec

    def SelectNewPopulation(self):
        fitnessValues, totalFitness = self.PopulationAfterFitness()
        probabilities, cumulative = self.SelectionProbabilities(fitnessValues, totalFitness)

        newPopulation = []

        for _ in range(self.populationSize - 1):
            u = random.random()
            index = self.BinarySearchInterval(cumulative, u)
            newPopulation.append(self.population[index])

        # val elitista
        maxIndex = fitnessValues.index(max(fitnessValues))
        best = self.population[maxIndex]
        newPopulation.append(best)

        return newPopulation

    def Crossover(self, population):
        new_population = []
        i = 0

        while i < self.populationSize - 1:
            parent1 = population[i]
            parent2 = population[i + 1]

            #recombinare
            if random.random() < self.probCrossover / 100:
                cut = random.randint(1, self.chromosomeLength - 1)
                child1 = parent1[:cut] + parent2[cut:]
                child2 = parent2[:cut] + parent1[cut:]

            else:
                child1 = parent1
                child2 = parent2

            new_population.extend([child1, child2])
            i += 2

        if self.populationSize % 2 == 1:
            new_population.append(population[-1])

        return new_population

    def Mutation(self, population):
        new_population = []
        for chromosome in population:
            if random.random() < self.probMutation / 100:
                mutation_point = random.randint(0, self.chromosomeLength - 1)
                chromosome = list(chromosome)
                chromosome[mutation_point] = '1' if chromosome[mutation_point] == '0' else '0'
                chromosome = ''.join(chromosome)
            new_population.append(chromosome)
        return new_population

    def UpdatePopulation(self):
        # selectie
        new_population = self.SelectNewPopulation()
        # incrcisare 1 pct
        new_population = self.Crossover(new_population)
        # mutation
        new_population = self.Mutation(new_population)

        return new_population

    def RunAlgorithm(self):
        bestFitnessValues = []
        with open("Evolutie.txt", "w") as file:
            file.write(
                f"Pentru functia ax^2 + bx + c, unde a={self.parameters[0]} b={self.parameters[1]} c={self.parameters[2]}\n"
                f"domeniul[{self.domain[0]},{self.domain[1]}], "
                f"dimensiunea populatiei: {self.populationSize}, "
                f"precizia: {self.precision}, "
                f"probabilitatea de recombinare: {self.probCrossover}%, "
                f"probabilitatea de mutatie: {self.probMutation}% si {self.generations} de etape\n")

            # date initiale
            file.write("Populatia Initiala:\n")
            for i, chrom in enumerate(self.population):
                xi = self.ChromosomeToNumber(chrom, self.domain)
                fitness_val = self.FitnessFunction(chrom)
                file.write(f"Individ {i + 1}: B[i] = {chrom}, X[i] = {xi}, f(Xi) = {fitness_val}\n")

            # prob si prob cumulativa
            file.write("\nProbabilitatile de selectie:\n")
            fitness_values, total_fitness = self.PopulationAfterFitness()
            probabilities, cumulated = self.SelectionProbabilities(fitness_values, total_fitness)
            for i in range(len(self.population)):
                file.write(f"Individ {i + 1}: p = {probabilities[i]}, q = {cumulated[i + 1]}\n")

            # selectie
            firstSelection = []

            file.write("\nProcesul de selectie:\n")
            for _ in range(self.populationSize - 1):
                u = random.random()
                index = self.BinarySearchInterval(cumulated, u)
                file.write(
                    f"Selectie: U = {u}, interval [{cumulated[index]}, {cumulated[index + 1]}], "
                    f"selectat {index + 1}\n")
                firstSelection.append(self.population[index])

            # selectie cea mai buna sol
            maxIndex = fitness_values.index(max(fitness_values))
            best = self.population[maxIndex]
            file.write(
                f"\nSelectia elitista: Individul {maxIndex + 1} este selectat automat pentru urmatoarea generatie.\n")
            firstSelection.append(best)

            # recombinare
            new_population = []

            file.write("\nRecombinarile:\n")
            for i in range(0, len(firstSelection) - 1, 2):
                parent1 = firstSelection[i]
                parent2 = firstSelection[i + 1]
                if random.random() < self.probCrossover / 100:
                    cut = random.randint(1, self.chromosomeLength - 1)
                    child1 = parent1[:cut] + parent2[cut:]
                    child2 = parent2[:cut] + parent1[cut:]
                    file.write(f"Recombinare intre {parent1} si {parent2} la pozitia {cut}: {child1}, {child2}\n")
                else:
                    child1 = parent1
                    child2 = parent2
                    file.write(f"Fara recombinare intre {parent1} si {parent2}\n")
                new_population.extend([child1, child2])

            # ultimul individ
            if len(firstSelection) % 2 == 1:
                new_population.append(firstSelection[-1])

            # afisare dupa recombinare
            file.write("\nPopulatia dupa recombinare:\n")
            for i, chrom in enumerate(new_population):
                xi = self.ChromosomeToNumber(chrom, self.domain)
                fitness_val = self.FitnessFunction(chrom)
                file.write(f"Individ {i + 1}: Bi = {chrom}, Xi = {xi}, f(Xi) = {fitness_val}\n")

            # mutatii
            new_population = self.Mutation(new_population)

            # pop dupa mutatii
            file.write("\nPopulatia dupa mutatii:\n")
            for i, chrom in enumerate(new_population):
                xi = self.ChromosomeToNumber(chrom, self.domain)
                fitness_val = self.FitnessFunction(chrom)
                file.write(f"Individ {i + 1}: Bi = {chrom}, Xi = {xi}, f(Xi) = {fitness_val}\n")

            self.population = new_population


            # gen 2 <-> last_gen
            for generation in range(1, self.generations):
                self.population = self.UpdatePopulation()
                fitness_values, total_fitness = self.PopulationAfterFitness()
                best_fitness = max(fitness_values)
                mean_fitness = sum(fitness_values) / len(fitness_values)
                bestFitnessValues.append(best_fitness)

                file.write(
                    f"\nGeneratia {generation + 1}: Max Fitness = {best_fitness}, Mean Fitness = {mean_fitness}\n")

        return bestFitnessValues


algoritm = GeneticAlgorithmForMaximization(
    20, (-1, 2), (-1, 1, 2), 6, 40, 1, 50
)

bestFitnessValues = algoritm.RunAlgorithm()
print(f"Fitness maxim pe parcursul generatiilor: {bestFitnessValues}")

root = tk.Tk()
root.title("Evoluția Algoritmului Genetic")


fig = plt.Figure(figsize=(6, 4), dpi=100)
ax = fig.add_subplot(111)
ax.plot(range(1, len(bestFitnessValues) + 1), bestFitnessValues, label='Max Fitness', color='blue')
ax.set_xlabel('Generație')
ax.set_ylabel('Fitness')
ax.set_title('Evoluția Fitnessului pe parcursul Generațiilor')
ax.legend()

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()

canvas.get_tk_widget().pack()

button_exit = ttk.Button(root, text="Închide", command=root.quit)
button_exit.pack()

root.mainloop()