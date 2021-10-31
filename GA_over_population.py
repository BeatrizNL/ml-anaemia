from data_processing_binary import x_data, y_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from random import choices, randint, randrange, random
from typing import List, Optional, Callable, Tuple
from collections import namedtuple
from functools import partial
import time
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
from matplotlib.backends.backend_pdf import PdfPages


features = x_data.columns
for index, i in enumerate(features):
    for e in features[index:]:
        x_data[i+"*"+e] = x_data[i] * x_data[e]
        if i != e:
            x_data[i + "/" + e] = x_data[i] / x_data[e]


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=0)

logis_model = LogisticRegression()
'''logis_model.fit(x_train,y_train)
accuracy = accuracy_score(y_test, logis_model.predict(x_test))
print("accuracy:", accuracy)
'''

Genome = List[int]
Population = List[Genome]
FitnessFunc = Callable[[Genome], int]
PopulateFunc = Callable[[], Population]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]

Thing = namedtuple('Thing', ['name'])
features = list(x_train.columns)
things = [Thing(i) for i in features]


def generate_genome(length:int) -> Genome:
    return choices([0, 1], weights=[0.8, 0.2], k=length)        #<---- na inicialização é mais provável começar com 0


def generate_population(size: int, genome_length: int) -> Population:
    return [generate_genome(genome_length) for _ in range(size)]

def fitness(genome: Genome, data: x_train, data_solution: y_train, test_data: x_test, test_solution: y_test, model: logis_model) -> float:
    if len(genome) != len(data.columns):
        raise ValueError("genome and data must be of the same length")
    accuracy = 0

    '''if sum(np.array(genome)) >= 6:          #<----- alterar de modo a penalizar pela quantidade de features
        accuracy = 0
    else:'''
    data = np.array(genome).T * data
    test_data = np.array(genome).T * test_data
    accuracy = accuracy_score(test_solution, model.fit(data, data_solution).predict(test_data)) - sum(np.array(genome)) * 0.005

    return accuracy

def selection_pair(population: Population, fitness_func: FitnessFunc) -> Population:
    indexes = np.argsort([fitness_func(genome) for genome in population])
    return [population[i] for i in indexes[-2:]]

def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    if len(a) != len(b):
        raise ValueError("Genomes a and b must be of the same lenght")

    length = len(a)
    if length < 2:
        return a,b

    p = randint(1, length -1)
    return a[0:p] + b[p:], b[0:p] + a[p:]

def mutation(genome: Genome, num: int = 1, probability: float = 0.5) -> Genome:
    for _ in range(num):
        index = randrange(len(genome))
        genome[index] = genome[index] if random() > probability else abs(genome[index]-1)
    return genome

def run_evolution(
        populate_func: PopulateFunc,
        fitness_func: FitnessFunc,
        fitness_limit: int,
        selection_func: SelectionFunc = selection_pair,
        crossover_func: CrossoverFunc = single_point_crossover,
        mutation_func: MutationFunc = mutation,
        generation_limit: int = 100
) -> Tuple[Population, int]:
    population = populate_func()
    for i in range(generation_limit):
        population = sorted(
            population,
            key=lambda genome: fitness_func(genome),
            reverse=True
        )
        best_accuracy = fitness_func(population[0])
        if best_accuracy >= fitness_limit:            #<------- fazer em cima uma lista com as accuracys para não estar sempre a fazer fit dos modelos
            break
        print("accuracy", best_accuracy)
        print("number of features", sum(population[0]))
        next_generation = population[0:2]

        for _ in range(int(len(population) / 2) -1):
            parents = selection_func(population, fitness_func)
            offspring_a, offspring_b = crossover_func(parents[0], parents[1])
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
            next_generation += [offspring_a, offspring_b]

        population = next_generation

        population = sorted(
        population,
        key=lambda genome: fitness_func(genome),
        reverse=True
    )
    return population, i

start = time.time()
accuracy_population = []
population_array = np.arange(5, 150, 10)
for i in population_array:
    population, generations = run_evolution(
    populate_func=partial(
        generate_population, size=i, genome_length=len(x_train.columns)
    ),
    fitness_func=partial(
        fitness, data=x_train, data_solution=y_train, test_data=x_test, test_solution=y_test,  model=logis_model
    ),
    fitness_limit=1,
    generation_limit=20)
    final_best_accuracy = accuracy_score(y_test, logis_model.fit(np.array(population[0]).T * x_train,y_train).predict(x_test))
    accuracy_population.append(final_best_accuracy)

end = time.time()
'''
def genome_to_things(genome: Genome, things: [Thing]) -> [Thing]:
    result = []
    for i, thing in enumerate(things):
        if genome[i] == 1:
            result += [thing.name]

    return result

accuracy_final = accuracy_score(y_test, logis_model.fit(np.array(population[0]).T * x_train,y_train).predict(x_test))

print(f"number of generations: {generations}")
print(f"Time: {end-start}s")
print(f"best solution: {genome_to_things(population[0], things)}")
print("Final accuracy", accuracy_final)
'''
accuracy_population = [i * 100 for i in accuracy_population]
accuracy_population = [round(num, 1) for num in accuracy_population]


fig = plt.figure(figsize=(8,5.25))
bax = brokenaxes(ylims=((0, 1), (90, 100)), hspace=.1)
bax.set_xlim([0, population_array[-1]])
bax.plot(population_array, accuracy_population, c='blue', marker='o')
bax.set_title("Accuracy over population size")
bax.set_xlabel("Population size")
bax.set_ylabel("Accuracy (%)")
bax.grid()
plt.show()
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Calibri'

with PdfPages('test.pdf') as pdf:
    pdf.savefig()

