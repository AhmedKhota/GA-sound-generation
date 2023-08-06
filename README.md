A Genetic Algorithm (GA) was designed to optimally generate non-linguistic utterances. These are sounds similar to those made by the android R2D2 in Star Wars and are popularly used by such robots to communicate and convey emotion.

For the GA, a 45-bit binary genotype was designed that contained four notes of 11 bits each, with one extra bit at the end of the genotype representing the instrument.
Sounds were generated using MIDI encoding and the python packages mido and midiUTIL from pypi. In each of the four 11-bit notes, 5 bits were for pitch, 3 bits were for duration in beats, 2 bits were for pitch-bend (bending a single pitch up or down for a given note), and 1 bit was for volume (0 or 100).

![genotype](https://github.com/AhmedKhota/Data-Science-Projects/assets/139664971/d872f573-8cd6-4c04-a80e-0f75d9ffab60)

The GA uses the Tournament Selection method with a Tournament size of 16, and two-point crossover, to evolve successive generations. The population size was set to 400 and with each new generation, the children produced by the selected parents completely replaced them in the new generation.

Firstly, randomly generated sounds, using the same genotype design, were evaluated manually, not using the GA, by listening to them and assigning a fitness value from 1 to 5. 3000 such sounds were evaluated and used as training data for a neural network (GANNkfold.py) to learn how to evaluate the fitness of such genotypes.

Two neural network models were used, one using a linear activation function for the output layer and one using a sigmoid activation function for the output layer. The
neural network layout is shown below.

![GANNlayout](https://github.com/AhmedKhota/GA-sound-generation/assets/139664971/e82eea15-8f51-4c77-88d6-9c08a62aad5d)

Using 10-fold cross-validation, the neural network was trained to evaluate the fitness of the genotypes and achieved an average Mean Absolute Error (MAE) of 0.136 on the normalized rating scale, and a correlation between actual and predicted values of 0.725. A single run of the neural network is shown below to demonstrate it training correctly and reducing loss for both the training and validation data sets.

![NewNNv2](https://github.com/AhmedKhota/GA-sound-generation/assets/139664971/c16b7f54-60d1-42dc-850f-8b7b90ab379b)

The neural network model was then used to automatically evaluate the fitness of individuals produced by the GA over successive generations.

Using a population size of 400, the GA was run and reached convergence within 30 generations. 

After the GA generated fit individuals (sounds), the fittest individuals were selected and midi files were generated and saved using the MidiSaveOnly.py script. The midi files were converted to wav files using the midi_to_wav.py script. 

The RandomForestModel.py script was then used to predict the emotional Valence and Arousal of the GA-generated sounds. The RandomForestModel.py script is from the Valence Arousal Inference Project. An experiment was conducted, using 80 of the GA sounds, where human raters evaluated the emotional Valence and Arousal of the sounds on a seven-point scale. The full flow of operations is shown below.

![GAmodel](https://github.com/AhmedKhota/GA-sound-generation/assets/139664971/17d48b81-aab6-4113-b3aa-c8ff6be7ffbc)

The coverage of the Valence Arousal 2d space by the generated GA sounds is shown in the below hex-plot. The figure shows 48000 fit individuals. 

![snapshot_4](https://github.com/AhmedKhota/GA-sound-generation/assets/139664971/0ff2b8da-8888-4bd6-92f7-77a83b45d30c)

After comparing the Valence and Arousal ratings from the human subjects and the inferences from the random forest model, the Mean Absolute Error was found to be 0.117 for Valence and 0.067 for Arousal. Correlation coefficients between experiment ratings and inferences were 0.22 for Valence and 0.63 for Arousal.

Since the random forest model was trained on sampled sounds from robots like R2D2 and WallE etc. the fact that the model inferences and human ratings corresponded reasonably well implies that the generated sounds were at least comparable if not useful candidates to be used as robot non-linguistic utterances.

