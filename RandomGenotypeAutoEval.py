#Genetic Algorithm using binary strings as genotype
#midi data played back and manually evalauted fitness of individuals - just using the first generation (random individuals) only

import random
import numpy as np
import mido
import pandas as pd
import os
import glob
from midiutil import MIDIFile
import shutil
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import copy
import msvcrt

all_genes = []

class NLU: #individual binary genotype 
    
    def __init__(self, binstring: str):#binstring could come from anywhere
        self.genes = binstring
        self.fitness = 0
        self.instrument = binstring[-1]#added for instrument
        #self.pitchbend = binstring[-1] #added for pitch bend yes/no
        n_notes = 4 #4 notes per sound
        n_bits_pernote = 11
        self.notes = []
        for i in range(0,n_notes*n_bits_pernote,n_bits_pernote):
            self.notes.append(Note(self.genes[i:i+n_bits_pernote]))#instance of the Note class
    
    def offspring(self, othernlu): #not using
        crosspoint = random.randint(1,len(self.genes)-1)
        newgenes = self.genes[:crosspoint] + othernlu.genes[crosspoint:]
        mut_prob = 1/len(newgenes)
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
                
    def save_midi(self, nlu: NLU, generation: int): #can change deinterleave=True if want notes not to overlap
        instrument = int(nlu.instrument,2)
        instruments = [80, 85]
        myMIDI = MIDIFile(1, adjust_origin=False, deinterleave=False, removeDuplicates=True)
        bpm = 480 #can change later if needed
        myMIDI.addTempo(0,0,bpm)
        myMIDI.addProgramChange(0, 0, 0, instruments[instrument])#synth square wave instrument change later
        track = 0
        time = 0 # each note sounds at next time increment
        channel = 0
        filenotes = ""
        for note in nlu.notes: 
            myMIDI.addNote(track, channel, note.pitch, time, note.duration, note.volume)
            time += note.duration
            if note.pitch_bend == 1:
                myMIDI.addPitchWheelEvent(0, 0, time, -8000)
                myMIDI.addPitchWheelEvent(0, 0, note.duration/2, 8000)
            elif note.pitch_bend == 2:
                myMIDI.addPitchWheelEvent(0, 0, time, 8000)
                myMIDI.addPitchWheelEvent(0, 0, note.duration/2, -8000)
            filenotes += note.note
        dest = "F:/GA_midis/{0}/".format(str(generation))#added generation folder
        filename = nlu.hexname().replace("0x",str(generation)) +".mid"  
        with open(os.path.join(dest,filename), 'wb') as binfile: #need to increment filename later in order to save multiple files per generation
            myMIDI.writeFile(binfile)
        return os.path.join(dest,filename)
    
    def play(self, nlu:NLU, midifile): #when is this called?
        port = mido.open_output()
        mid = mido.MidiFile(midifile)
        for msg in mid.play():
            port.send(msg)
            
    def evaluate(self, midifile, nlu: NLU): # to play and get rating
        self.play(nlu,midifile)
        while True:
            print("Please rate the sound from 1-5: ")
            ch = msvcrt.getch()
            ch = ch.decode('utf-8') # decode the byte string to Unicode string
            if ch.isdigit() and int(ch) in range(1, 6):
                # user input is valid
                nlu.fitness = int(ch)
                break
            else:
                print("Invalid input. Please enter a number between 1 and 5.")
    
    def get_the_fittest(self, n: int):
        self.__sort_by_fitness()
        return self.individuals[:n] #NLU for individual?
        
    def __sort_by_fitness(self):
        self.individuals.sort(key=lambda nlu: nlu.fitness, reverse=True)#change
    
    def __nlu_fitness_sort_key(self, nlu: NLU):
        return nlu.fitness
    
    def fittest(self, population, n: int): #not used
        population.sort(key=lambda nlu: nlu.fitness, reverse=True)
        return population[:n]
            
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

class SinglePointCrossover: #try twopointcrossover
      
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


class TwoPointCrossover: #try twopointcrossover
      
    def crossover(self, parent_1: NLU, parent_2: NLU):
        #crossover_point = random.randint(1, len(parent_1.genes)-1) #changed so that parent cant create exact copy of itself
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

class Mutator:
           
    def mutate(self, nlu: NLU):
        mutated_genotype = nlu.genes
        mutation_multiplier = 3
        mutation_probability = (1 / len(nlu.genes))*mutation_multiplier #added to increase probability of mutation
        for index, gene in enumerate(nlu.genes):
            if random.random() < mutation_probability:
                if (mutated_genotype[index] == "1"):
                    mutated_genotype = mutated_genotype[:index] + "0" + mutated_genotype[index+1:]
                else:
                    mutated_genotype = mutated_genotype[:index] + "1" + mutated_genotype[index+1:]
        return NLU(mutated_genotype)

class Breeder:
    
    def __init__(self, 
                 two_point_crossover: TwoPointCrossover,
                 mutator: Mutator):
        self.two_point_crossover = two_point_crossover
        self.mutator = mutator
        
    def produce_offspring(self, parents):
        offspring = []
        number_of_parents = len(parents)
        for index in range(int(number_of_parents / 2)):
            parent_1, parent_2 = self.__pick_random_parents(parents, number_of_parents)
            child_1, child_2 = self.two_point_crossover.crossover(parent_1, parent_2)
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
        limit = 10 #how many instances of the same genotype are allowed in the gene pool
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
    
    def __init__(self, population_size: int, parent_selector: TournamentSelector, population: Population, breeder: Breeder):
        self.population = population #took out.individuals to make it object instead of list
        self.parent_selector = parent_selector
        self.breeder = breeder
    
    def update(self): #after evaluating fitness of population call this to update population and start next generation
        parents = self.parent_selector.select_parents(self.population) #add.individuals?
        next_generation = breeder.produce_offspring(parents)#changed breeder to self.breeder, changed it back after
        #self.population = self.population.with_individuals(next_generation)#put back with_individuals returns population object not list
        self.population.individuals.clear() #make sure population list is empty before creating new one?
        self.population = population.with_individuals(next_generation)#put back with_individuals returns population object not list
        
    def get_the_fittest(self, n: int): #change this the nlu object contains fitness. so you need to loop through members of the population to find the largest value of fitness
        return Population.get_the_fittest(n) 
    
    def get_fittest(self, n: int):
        self.population.individuals.sort(key=lambda nlu: nlu.fitness, reverse=True)
        return self.population.individuals[:n] #returns sorted list
    		
# test block including loading

if __name__ == '__main__':

    TOTAL_GENERATIONS = 1
    POPULATION_SIZE = 1000

    current_generation = 1
    #recordsrecords = []
    records = []

    population = Population(individuals = [])
    population.generation = current_generation
    two_point_crossover = TwoPointCrossover()#keep pitches, volumes and durations intact
    mutator = Mutator()
    breeder = Breeder(two_point_crossover, mutator)
    parent_selector = TournamentSelector(5)
    environment = Environment(POPULATION_SIZE, parent_selector, population, breeder)

    pop = population.with_randoms(POPULATION_SIZE)
    for member in pop.individuals: #added .individuals to get list
        print("population: ", member.genes)
    for member in pop.individuals: #added .individuals to get list
        temp = []
        midifile = population.save_midi(member, current_generation)
        population.evaluate(midifile, member)
        temp.append((current_generation, midifile, member.genes, member.fitness))
        records.append((current_generation, midifile, member.genes, member.fitness))
        with open("F:/GA_midis/records.txt", 'a') as g:
            np.savetxt(g, temp, delimiter=",", fmt='%s')

    current_generation += 1

    # from 2nd generation onwards:
    # need to start this environment thing and use parentselector etc until generations all completed
    highest_fitness_list = []
    while current_generation <= TOTAL_GENERATIONS:  # whats telling the program to run evaluate on each member of the current population? #its not stopping
        records = []
        print("ENTERED REPRODUCTION")
        print("current_generation: ", current_generation)
        fittest = environment.get_fittest(1)[0]  # returns list
        highest_fitness_list.append(fittest.fitness) 
        environment.update()  # population has been updated right? self.population = population.with_individuals(next_generation)
        for member in population.individuals: #print population members
            print("population: ", member.genes)
        for member in population.individuals:  # so if I call population now its referring to the updated population? #error popluation object is not iterable
            # print("member: ",member)
            temp = []
            midifile = population.save_midi(member, current_generation)
            population.evaluate(midifile, member)
            temp.append((current_generation, midifile, member.genes, member.fitness))
            records.append((current_generation, midifile, member.genes, member.fitness))
            with open("F:/GA_midis/records.txt", 'a') as g:
                np.savetxt(g, temp, delimiter=",", fmt='%s')

        records = np.array(records)
        np.savetxt("F:/GA_midis/records_{0}.txt".format(str(current_generation)), records, delimiter=",", fmt='%s')
        current_generation += 1
        population.generation = current_generation

    #recordsrecords = np.array(recordsrecords)
    #np.savetxt("F:/GA_midis/records.txt", recordsrecords, delimiter=",", fmt='%s')
    print("GA complete")    

    
