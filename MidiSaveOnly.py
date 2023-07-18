import mido
import os
from midiutil import MIDIFile
import random
import numpy as np
import time
import pandas as pd

class NLU: #individual binary genotype 
    
    def __init__(self, binstring: str):#binstring could come from anywhere
        self.genes = binstring
        self.fitness = 0
        self.valence = 0
        self.arousal = 0
        self.instrument = binstring[-1]#added for instrument
        #self.pitchbend = binstring[-1] #added for pitch bend yes/no
        n_notes = 4 #4 notes per sound
        n_bits_pernote = 11
        self.notes = []
        for i in range(0,n_notes*n_bits_pernote,n_bits_pernote):
            self.notes.append(Note(self.genes[i:i+n_bits_pernote]))#instance of the Note class
    
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

class MidiFile:

    def __init__(self, binstring):
        self.binstring = binstring

    def save_midi(self, nlu: NLU): #can change deinterleave=True if want notes not to overlap
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


        dest = "F:/GA_midis/test/allmidis/100000/"#added generation folder
        filename = nlu.hexname() +".mid"  
        with open(os.path.join(dest,filename), 'wb') as binfile: #need to increment filename later in order to save multiple files per generation
            print("saving ", str(os.path.join(dest,filename)))
            myMIDI.writeFile(binfile)
        return os.path.join(dest,filename)
    
    def play(self, nlu): #no longer needed
        midifile = self.save_midi(nlu)
        port = mido.open_output()
        mid = mido.MidiFile(midifile)
        for msg in mid.play():
            port.send(msg)
        time.sleep(1)
            

    def evaluate(self, nlu: NLU): # to play and get rating
        self.play(nlu)
        while True:
            try:
                nlu.fitness = int(input("please rate the sound from 1-5"))
            except ValueError:
                print("Invalid input. Please enter a value between 1-5 ")
                continue
            else:
                break

if __name__ == '__main__':

    records = []
    data = pd.read_csv("BinsforVA.csv")
    print(data)
    data = np.squeeze(np.array(data.values.tolist()))
    print(data[0])
    for binstring in data:
        nlu = NLU(binstring)
        midifile = MidiFile(nlu)
        midifile.save_midi(nlu)
