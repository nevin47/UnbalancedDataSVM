from pyevolve import G1DList
from pyevolve import GSimpleGA
import math

def eval_func(chromosome):
   score = chromosome[0] * math.sin(chromosome[0])
   return score

genome = G1DList.G1DList(20)
genome.evaluator.set(eval_func)
ga = GSimpleGA.GSimpleGA(genome)
ga.evolve(freq_stats=10)
print ga.bestIndividual()