#Generic Algorithm using binary strings as genotypes to evolve fitter individuals
#using trained neural network model to evaluate fitness of individuals automatically

import random
import numpy as np
import copy
from keras.models import load_model
#copy midi_pitches.csv, GANNsig5.h5 and records.txt together with this script, chose to use model GANN4.h5

all_genes = []

class NLU: #individual binary genotype 
    
    def __init__(self, binstring: str):#binstring could come from anywhere
        self.genes = binstring
        self.fitness = 0
        self.instrument = binstring[-2:-1]#added for instrument
        self.pitchbend = binstring[-1] #added for pitch bend yes/no
        n_notes = 4 #4 notes per sound
        n_bits_pernote = 11
        self.notes = []
        self.selection_count = 0
        for i in range(0,n_notes*n_bits_pernote,n_bits_pernote):
            self.notes.append(Note(self.genes[i:i+n_bits_pernote]))#instance of the Note class
    
    def offspring(self, othernlu): #not using
        crosspoint = random.randint(1,len(self.genes)-1)
        newgenes = self.genes[:crosspoint] + othernlu.genes[crosspoint:]
        mut_prob = 1/len(newgenes) #not used
        for index, gene in enumerate(newgenes):
            if random.random() < mut_prob:
                newgenes[index] = '0' if gene =='1' else '1'
        return NLU(newgenes)
    
    def hexname(self):
        x = int(self.genes,2)
        return hex(x)
        
class Note: 
    
    def __init__(self, bits: str): #need to include pitches lookup function
        vol_prob = 0.7 #changed from 0.7 to 0.9 to have less silent notes
        self.note = bits[0:]
        self.pitch = self.pitch_lookup(int(bits[0:5],2))#lookup pitch from designated values
        self.duration = int(bits[5:8],2)+1#int value of bits + 1
        self.pitch_bend = int(bits[9:11],2)
        #if random.random() < vol_prob:#rework
        #    self.volume = 100
        #else:
        self.volume = int(bits[8:9],2)*100
        
    def pitch_lookup(self, value):
        pitches = np.genfromtxt('midi_pitches.csv',delimiter=',', skip_header = 1) 
        for pitch in pitches:
            if int(pitch[0]) == value:
                return int(pitch[1])
        
class Population: #play sound and get rating (fitness of one sound)
    
    def __init__(self, individuals):
        self.individuals = []
        self.generation = 0 #change later
    
    def randNLU(self): # generate random NLU in binary string format (77 digits)
        bits = ""
        num_notes = 4
        bits_per_note = 11
        while len(bits) < (num_notes*bits_per_note): #took out extra gene at sound level for deinterleave setting
            bits=bits+str(random.randint(0,1))
        bits = bits + str(random.randint(0,1)) #for instrument
        #bits = bits + str(random.randint(0,1)) #for pitch bend
        return NLU(bits)   #changed added .notes          
    
    def with_randoms(self, population_size: int):
        self.population_size = population_size
        while len(self.individuals)<self.population_size: #just move to new function and call it instead of constructor
            self.individuals.append(self.randNLU())
        return self #took out.individuals
    
    def with_individuals(self, individuals): 
        for genotype in individuals:
            self.individuals.append(genotype)
        return self #took out .individuals
                
    def evaluate(self, nlu: NLU):
        neuralnet = NeuralNet('GANN4.h5')
        nlu.fitness = np.round(np.squeeze(neuralnet.predict(nlu.genes)),2)

    def evaluate_population(self, nlu: NLU):
        neuralnet = NeuralNet('GANN4.h5')
        predictions = neuralnet.predict_on_population(self)
        for i in range(len(predictions)):
            self.individuals[i].fitness = np.round(np.squeeze(predictions[i]),2)
    
    def get_the_fittest(self, n: int):
        self.__sort_by_fitness()
        return self.individuals[:n] #NLU for individual?
        
    def __sort_by_fitness(self):
        self.individuals.sort(key=lambda nlu: nlu.fitness, reverse=True)#change
    
    def fittest(self, population, n: int): #not used
        population.sort(key=lambda nlu: nlu.fitness, reverse=True)
        return population[:n]

class NeuralNet:

    def __init__(self, modelpath):
        self.model = load_model(modelpath)
    
    def predict(self, binstring):
        genes = [int(x) for x in binstring]
        genes = np.array(genes)
        genes = np.reshape(genes,[-1,34])
        return self.model.predict(genes, verbose = 0)
    
    def predict_on_population(self, population: Population):
        population_genes = []
        for individual in population.individuals:
            population_genes.append([int(gene) for gene in individual.genes])
        population_genes = np.array(population_genes)
        print(population_genes.shape) #to test
        return self.model.predict(population_genes, verbose = 0)

            
class ParentSelector:
        
    def select_parents(self, population: Population):
        total_fitness = 0
        fitness_scale = []
        for index, individual in enumerate(population.individuals):#removed .population.individuals
            total_fitness += individual.fitness
            if index == 0:
                fitness_scale.append(individual.fitness)
            else:
                fitness_scale.append(individual.fitness + fitness_scale[index - 1])
        # Store the selected parents
        mating_pool = []
        # Equal to the size of the population
        number_of_parents = len(population.individuals)#removed .population.individuals
        # How fast we move along the fitness scale
        fitness_step = total_fitness / number_of_parents
        random_offset = random.uniform(0, fitness_step)
        # Iterate over the parents size range and for each:
        # - generate pointer position on the fitnss scale
        # - pick the parent who corresponds to the current pointer position and add them to the mating pool
        current_fitess_pointer = random_offset
        last_fitness_scale_position = 0
        for index in range(len(population.individuals)):#removed .population.individuals
            for fitness_scale_position in range(last_fitness_scale_position, len(fitness_scale)):
                if fitness_scale[fitness_scale_position] >= current_fitess_pointer:
                    mating_pool.append(population.individuals[fitness_scale_position])#removed .population.individuals
                    last_fitness_scale_position = fitness_scale_position
                    break
            current_fitess_pointer += fitness_step
        return mating_pool
    
class RouletteWheelSelector:

    def select_parents(self, population: Population):
        total_fitness = 0
        fitness_values = []
        for individual in population.individuals:
            total_fitness += individual.fitness
            fitness_values.append(individual.fitness)
        selection_probs = [f/total_fitness for f in fitness_values]
        cumulative_probs = [sum(selection_probs[:i+1]) for i in range(len(selection_probs))]
        mating_pool = []
        number_of_parents = len(population.individuals)
        for i in range(number_of_parents):
            r = random.uniform(0, 1)
            selected_idx = 0
            for j in range(len(cumulative_probs)):
                if r <= cumulative_probs[j]:
                    selected_idx = j
                    break
            mating_pool.append(population.individuals[selected_idx])
        return mating_pool

class TournamentSelector(ParentSelector): #add a limit so that individuals cant be chosen again and again

    def __init__(self, tournament_size):
        self.tournament_size = tournament_size
        
    def select_parents(self, population: Population):
        mating_pool = []
        for i in range(len(population.individuals)):
            tournament = random.sample(population.individuals, self.tournament_size)
            winner = max(tournament, key=lambda x: x.fitness)
            mating_pool.append(winner)
        return mating_pool

class HybridSelector(ParentSelector): #combine stochastic universal sampling and tournament-based selection

    def __init__(self, tournament_size: int, max_tournament_repeat: int, max_selection_count: int):
        self.tournament_size = tournament_size
        self.max_tournament_repeat = max_tournament_repeat
        self.max_selection_count = max_selection_count

    def select_parents(self, population: Population):
        # Initialize a dictionary to keep track of previously selected individuals
        selected_dict = {}
        # Initialize an empty list to store the selected parents
        selected_parents = []
        # Perform multiple tournaments until the mating pool is filled
        while len(selected_parents) < len(population.individuals):
            # Select a random subset of the population to compete in the tournament
            tournament = random.sample(population.individuals, self.tournament_size)
            # Sort the individuals in the tournament by fitness, highest to lowest
            tournament.sort(key=lambda x: x.fitness, reverse=True)
            # Calculate the total fitness of the tournament
            total_fitness = sum([individual.fitness for individual in tournament])
            # Calculate the distance between pointers for stochastic universal sampling
            pointer_distance = total_fitness / self.tournament_size
            # Generate random starting positions for the pointers
            pointers = [random.uniform(0, pointer_distance) + (i * pointer_distance) for i in range(self.tournament_size)]
            # Iterate over the pointers and select the corresponding individual for each pointer
            for pointer in pointers:
                cumulative_fitness = 0
                for individual in tournament:
                    cumulative_fitness += individual.fitness
                    if cumulative_fitness > pointer:
                        # Select the individual if it has not been selected before
                        if individual not in selected_dict.keys():
                            selected_dict[individual] = 1
                            selected_parents.append(individual)
                            break
                        # If the individual has been selected before, increment its count
                        else:
                            if selected_dict[individual] < self.tournament_size:
                                selected_dict[individual] += 1
                                selected_parents.append(individual)
                                break
            # Check if the selected individuals have filled the mating pool
            if len(selected_parents) >= len(population.individuals):
                break
            # Check if the selected individuals have been selected too many times across all tournaments
            for individual in selected_dict.keys():
                if selected_dict[individual] >= self.max_selection_count:
                    # If so, remove the individual from the population and select a new individual
                    population.individuals.remove(individual)
                    break
            # Limit the number of times the same genetic sequence can be chosen within each tournament
            if len(selected_dict) >= self.max_tournament_repeat:
                selected_dict.clear()
        # Reset the selection count of all individuals in the selected pool
        for individual in selected_parents:
            individual.selection_count = 0
        return selected_parents

class LimitedTournamentSelector(ParentSelector): #add a limit so that individuals cant be chosen again and again

    def __init__(self, tournament_size, limit):
        self.tournament_size = tournament_size
        self.limit = limit
        
    def select_parents(self, population: Population):
        mating_pool = []
        while len(mating_pool) < (len(population.individuals)):
            tournament = random.sample(population.individuals, self.tournament_size)
            tournament.sort(key=lambda ind: ind.fitness, reverse=True)
            for ind in tournament:
                if all_genes.count(ind.genes) <= self.limit:
                    mating_pool.append(ind)
                    all_genes.append(ind)
        return mating_pool
    
class TournamentSelectorLimit(ParentSelector):

    def __init__(self, tournament_size: int, max_repeat: int):
        self.tournament_size = tournament_size
        self.max_repeat = max_repeat

    def select_parents(self, population: Population):
        mating_pool = []
        selected = {}
        
        # Keep selecting parents until the mating pool is full
        while len(mating_pool) < len(population.individuals):
            # Randomly select k individuals for the tournament
            tournament = random.sample(population.individuals, self.tournament_size)

            # Sort individuals in the tournament by fitness
            tournament.sort(key=lambda ind: ind.fitness, reverse=True)

            # Choose the fittest individual from the tournament
            # that hasn't been selected more than max_repeat times
            for ind in tournament:
                if ind not in selected or selected[ind] < self.max_repeat:
                    mating_pool.append(ind)
                    if ind in selected:
                        selected[ind] += 1
                    else:
                        selected[ind] = 1
                    break

        # Reset selection counts for individuals that are not in the mating pool
        for ind in population.individuals:
            if ind not in mating_pool:
                ind.selection_count = 0

        return mating_pool

class RankBasedSelector(ParentSelector):

    def select_parents(self, population: Population):
        fitness_sum = sum(range(1, len(population.individuals)+1))
        selection_probs = [(len(population.individuals)-i)/fitness_sum for i in range(len(population.individuals))]
        mating_pool = []
        for i in range(len(population.individuals)):
            parent1 = random.choices(population.individuals, weights=selection_probs)[0]
            parent2 = random.choices(population.individuals, weights=selection_probs)[0]
            mating_pool.append((parent1, parent2))
        return mating_pool
    
class ElitistSelector(ParentSelector):

    def __init__(self, elite_size):
        self.elite_size = elite_size
        
    def select_parents(self, population: Population):
        sorted_population = sorted(population.individuals, key=lambda x: x.fitness, reverse=True)
        elite = sorted_population[:self.elite_size]
        mating_pool = elite.copy()
        for i in range(len(population.individuals) - self.elite_size):
            tournament = random.sample(population.individuals, 2)
            winner = max(tournament, key=lambda x: x.fitness)
            mating_pool.append(winner)
        return mating_pool
    
class SinglePointCrossover: 
      
    def crossover(self, parent_1: NLU, parent_2: NLU):
        #crossover_point = random.randint(1, len(parent_1.genes)-1) #changed so that parent cant create exact copy of itself
        #new crossover_point definition: I want to crossover but keep intact pitches, durations, volumes, of each note
        indices = [x for x in range(5, 30, 8)] + [x for x in range(7, 32, 8)] + [x for x in range(8, 25, 8)] + [32]
        indices.sort()
        crossover_point = random.choice(indices)
        genotype_1 = self.__new_genotype(crossover_point, parent_1, parent_2)
        genotype_2 = self.__new_genotype(crossover_point, parent_2, parent_1)
        child_1 = NLU(genotype_1)
        child_2 = NLU(genotype_2)
        return child_1, child_2
    
    def __new_genotype(self, crossover_point: int, parent_1: NLU, parent_2: NLU):
        return parent_1.genes[:crossover_point] + parent_2.genes[crossover_point:]	

class TwoPointCrossoverAny:

    def crossover(self, parent_1: NLU, parent_2: NLU):
        crossover_point_1 = random.randint(1,len(parent_1.genes)-1)
        crossover_point_2 = random.randint(1,len(parent_1.genes)-1)
        
        if crossover_point_1 < crossover_point_2:
            genotype_1 = self.__new_genotype(crossover_point_1, crossover_point_2, parent_1, parent_2)
            genotype_2 = self.__new_genotype(crossover_point_1, crossover_point_2, parent_2, parent_1)
        else:
            genotype_1 = self.__new_genotype(crossover_point_2, crossover_point_1, parent_1, parent_2)
            genotype_2 = self.__new_genotype(crossover_point_2, crossover_point_1, parent_2, parent_1)
            
        child_1 = NLU(genotype_1)
        child_2 = NLU(genotype_2)
        return child_1, child_2
    
    def __new_genotype(self, crossover_point_1: int, crossover_point_2: int ,parent_1: NLU, parent_2: NLU):
        return parent_1.genes[:crossover_point_1] + parent_2.genes[crossover_point_1:crossover_point_2] + parent_1.genes[crossover_point_2:]

class TwoPointCrossover: 
      
    def crossover(self, parent_1: NLU, parent_2: NLU):
        #new crossover_point definition: I want to crossover but keep intact pitches, durations, volumes, of each note
        indices = [x for x in range(5, 30, 8)] + [x for x in range(7, 32, 8)] + [x for x in range(8, 25, 8)] + [32]
        indices.sort()
        crossover_point_1 = random.choice(indices)
        indices.remove(crossover_point_1)
        crossover_point_2 = random.choice(indices)
        
        if crossover_point_1 < crossover_point_2:
            genotype_1 = self.__new_genotype(crossover_point_1, crossover_point_2, parent_1, parent_2)
            genotype_2 = self.__new_genotype(crossover_point_1, crossover_point_2, parent_2, parent_1)
        else:
            genotype_1 = self.__new_genotype(crossover_point_2, crossover_point_1, parent_1, parent_2)
            genotype_2 = self.__new_genotype(crossover_point_2, crossover_point_1, parent_2, parent_1)
            
        child_1 = NLU(genotype_1)
        child_2 = NLU(genotype_2)
        return child_1, child_2
    
    def __new_genotype(self, crossover_point_1: int, crossover_point_2: int ,parent_1: NLU, parent_2: NLU):
        return parent_1.genes[:crossover_point_1] + parent_2.genes[crossover_point_1:crossover_point_2] + parent_1.genes[crossover_point_2:]

class Mutator:
           
    def mutate(self, nlu: NLU):
        mutated_genotype = nlu.genes
        mutation_probability = (1 / len(nlu.genes))*2 #doubled mutation rate
        for index, gene in enumerate(nlu.genes):
            if random.random() < mutation_probability:
                if (mutated_genotype[index] == "1"):
                    mutated_genotype = mutated_genotype[:index] + "0" + mutated_genotype[index+1:]
                else:
                    mutated_genotype = mutated_genotype[:index] + "1" + mutated_genotype[index+1:]
        return NLU(mutated_genotype)
    

class Breeder:
    
    def __init__(self, two_point_crossover: TwoPointCrossoverAny, mutator: Mutator): #change TwoPointCrossover to TwoPointCrossoverAny
        self.two_point_crossover = two_point_crossover
        self.mutator = mutator
        
    def produce_offspring(self, parents):
        crossover_rate = 0.99 #only perform crossover if random number is less than the crossover rate else pass on parents as is
        offspring = []
        number_of_parents = len(parents)
        for index in range(int(number_of_parents / 2)):
            parent_1, parent_2 = self.__pick_random_parents(parents, number_of_parents)
            if random.random() < crossover_rate:
                child_1, child_2 = self.two_point_crossover.crossover(parent_1, parent_2)
            else: 
                child_1, child_2 = copy.deepcopy(parent_1), copy.deepcopy(parent_2)
            child_1_mutated = mutator.mutate(child_1)
            child_2_mutated = mutator.mutate(child_2)
            offspring.extend((child_1_mutated, child_2_mutated))
        return offspring
    
    def __pick_random_parents(self, parents, number_of_parents: int):
        parent_1 = parents[random.randint(0, number_of_parents - 1)]
        parent_2 = parents[random.randint(0, number_of_parents - 1)]
        return parent_1, parent_2
		
class LimitedBreeder(Breeder): #check if child exists, create and add to selected list

    def produce_offspring(self, parents):
        limit = 5000 #how many instances of the same genotype are allowed in the gene pool
        crossover_rate = 0.95 #only perform crossover if random number is less than the crossover rate else pass on parents as is
        offspring = []
        number_of_parents = len(parents)
        for index in range(int(number_of_parents / 2)):
            parent_1, parent_2 = self.__pick_random_parents(parents, number_of_parents)
            if random.random() < crossover_rate:
                count = 0
                while True:
                    child_1, child_2 = self.two_point_crossover.crossover(parent_1, parent_2)
                    if all_genes.count(child_1.genes) <= limit and all_genes.count(child_2.genes) <= limit:
                        break
                    count += 1
                    if count > 1000: # if count exceeds 10 attempts, select new parents and try again
                        parent_1, parent_2 = self.__pick_random_parents(parents, number_of_parents)
                        count = 0
                        continue
                    #print("children genes over limit - entering in loop {0}, trying to create new children...".format(str(count)))
            else: 
                child_1, child_2 = copy.deepcopy(parent_1), copy.deepcopy(parent_2)
                all_genes.append(child_1.genes)
                all_genes.append(child_2.genes)
            child_1_mutated = mutator.mutate(child_1)
            child_2_mutated = mutator.mutate(child_2)
            offspring.extend((child_1_mutated, child_2_mutated))
            all_genes.append(child_1_mutated.genes)
            all_genes.append(child_2_mutated.genes)
        return offspring
    
    def __pick_random_parents(self, parents, number_of_parents: int):
        parent_1 = parents[random.randint(0, number_of_parents - 1)]
        parent_2 = parents[random.randint(0, number_of_parents - 1)]
        return parent_1, parent_2

class Environment:
    
    def __init__(self, population_size: int, parent_selector: TournamentSelector, population: Population, breeder: LimitedBreeder): #changed from parentselector: ParentSelector
        self.population = population #took out.individuals to make it object instead of list
        self.parent_selector = parent_selector#(8) #added tournament_size argument
        self.breeder = breeder
        #self.selected = []
    
    def update(self): #after evaluating fitness of population call this to update population and start next generation
        parents = self.parent_selector.select_parents(self.population) #this returns the mating pool
        next_generation = breeder.produce_offspring(parents)#changed breeder to self.breeder, changed it back after
        #self.population = self.population.with_individuals(next_generation)#put back with_individuals returns population object not list
        self.population.individuals.clear() #make sure population list is empty before creating new one?
        self.population = population.with_individuals(next_generation)#put back with_individuals returns population object not list
        
    def get_the_fittest(self, n: int): #change this the nlu object contains fitness. so you need to loop through members of the population to find the largest value of fitness
        return Population.get_the_fittest(n) 
    
    def get_fittest(self, n: int):
        self.population.individuals.sort(key=lambda nlu: nlu.fitness, reverse=True)
        return self.population.individuals[:n] #returns sorted list

if __name__ == '__main__':
    TOTAL_GENERATIONS = 20000
    POPULATION_SIZE = 400 #increased to create more genetic variation
    current_generation = 1
    records = []
    recordstxt = "records.txt"
    population = Population(individuals = [])
    population.generation = current_generation
    two_point_crossover = TwoPointCrossoverAny() #changed to any points in sequence "..Any"
    mutator = Mutator()
    breeder = LimitedBreeder(two_point_crossover, mutator)
    parent_selector = TournamentSelector(16) #choose selector method tournament size and genetic sequence repeat limit, higher tournament size = more selection pressure
    #parent_selector = RouletteWheelSelector
    environment = Environment(POPULATION_SIZE, parent_selector, population, breeder) #changed parent_selector to tournament_selector
    pop = population.with_randoms(POPULATION_SIZE)
    #rewrite to evaluate entire population:
    population.evaluate_population(pop)
    for member in pop.individuals: #added .individuals to get list
        temp = []
        population_fitness = []
        #population.evaluate(member)
        population_fitness.append(member.fitness)
        temp.append((current_generation, member.genes, member.fitness))
        records.append((current_generation, member.genes, member.fitness))
        with open(recordstxt, 'a') as g:
            np.savetxt(g, temp, delimiter=",", fmt='%s')
    print("current_generation: ", current_generation, " average fitness: ", str(sum(population_fitness)/POPULATION_SIZE))
    current_generation += 1
    # from 2nd generation onwards:
    highest_fitness_list = []
    while current_generation <= TOTAL_GENERATIONS:  # whats telling the program to run evaluate on each member of the current population? #its not stopping
        records = []
        fitsum = 0
        fittest = environment.get_fittest(1)[0]  # returns list
        highest_fitness_list.append(fittest.fitness) 
        environment.update()  # population has been updated right? self.population = population.with_individuals(next_generation)
        population.evaluate_population(population)
        for member in population.individuals:  # so if I call population now its referring to the updated population? #error popluation object is not iterable
            temp = []
            population_fitness = []
            #population.evaluate(member)
            population_fitness.append(member.fitness)
            fitsum = fitsum + member.fitness
            temp.append((current_generation, member.genes, member.fitness))
            records.append((current_generation, member.genes, member.fitness))
            with open(recordstxt, 'a') as g:
                np.savetxt(g, temp, delimiter=",", fmt='%s')
        print("current_generation: ", current_generation, " average fitness: ", fitsum/POPULATION_SIZE)
        current_generation += 1
        population.generation = current_generation
    print("GA complete")    
