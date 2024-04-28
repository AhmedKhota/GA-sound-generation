A Genetic Algorithm (GA) was designed to optimally generate non-linguistic utterances. These are sounds similar to those made by the android R2D2 in Star Wars and are popularly used by such robots to communicate and convey emotion.

For the GA, a 45-bit binary genotype was designed that contained four notes of 11 bits each, with one extra bit at the end of the genotype representing the instrument.
Sounds were generated using MIDI encoding and the python packages mido and midiUTIL from pypi. In each of the four 11-bit notes, 5 bits were for pitch, 3 bits were for duration in beats, 2 bits were for pitch-bend (bending a single pitch up or down for a given note), and 1 bit was for volume (0 or 100).

![Genotype](/images/genotype.jpg)

The GA uses the Tournament Selection method with a Tournament size of 16, and two-point crossover, to evolve successive generations. The population size was set to 400 and with each new generation, the children produced by the selected parents completely replaced them in the new generation.

Firstly, randomly generated sounds, using the same genotype design, were evaluated manually, not using the GA, by listening to them and assigning a fitness value from 1 to 5. 3000 such sounds were evaluated and used as training data for a neural network (GANNkfold.py) to learn how to evaluate the fitness of such genotypes.

Two neural network models were used, one using a linear activation function for the output layer and one using a sigmoid activation function for the output layer. The
neural network layout is shown below.

![Neural Network Layout](/images/GANNlayout.jpg)

Using 10-fold cross-validation, the neural network was trained to evaluate the fitness of the genotypes and achieved an average Mean Absolute Error (MAE) of 0.136 on the normalized rating scale, and a correlation between actual and predicted values of 0.725. A single run of the neural network is shown below to demonstrate it training correctly and reducing loss for both the training and validation data sets.

![Neural Network Results](/images/NN2.png)

The neural network model was then used to automatically evaluate the fitness of individuals produced by the GA over successive generations.

Using a population size of 400, the GA was run and reached convergence within 30 generations. 

After the GA generated fit individuals (sounds), the fittest individuals were selected and midi files were generated and saved using the MidiSaveOnly.py script. The midi files were converted to wav files using the midi_to_wav.py script. 

The RandomForestModel.py script was then used to predict the emotional Valence and Arousal of the GA-generated sounds. The RandomForestModel.py script is from the Valence Arousal Inference Project. An experiment was conducted, using 80 of the GA sounds, where human raters evaluated the emotional Valence and Arousal of the sounds on a seven-point scale. The full flow of operations is shown below.

![GA model](/images/GAmodel.jpg)


After comparing the Valence and Arousal ratings from the human subjects and the inferences from the random forest model, the Mean Absolute Error was found to be 0.117 for Valence and 0.067 for Arousal. Correlation coefficients between experiment ratings and inferences were 0.22 for Valence and 0.63 for Arousal.

Since the random forest model was trained on sampled sounds from robots like R2D2 and WallE etc. the fact that the model inferences and human ratings corresponded reasonably well implies that the generated sounds were at least comparable if not useful candidates to be used as robot non-linguistic utterances.

