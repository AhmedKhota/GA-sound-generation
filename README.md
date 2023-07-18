A Genetic Algorithm (GA) was designed to optimally generate non-linguistic utterances. These are sounds similar to those made by the android R2D2 in Star Wars and are popularly used by such robots to communicate and convey emotion.

For the GA, a 45-bit binary genotype was designed that contained four notes of 11 bits each, with one extra bit at the end of the genotype representing the instrument.
Sounds were generated using MIDI encoding and the python packages mido and midiUTIL from pypi. In each of the four 11-bit notes, 5 bits were for pitch, 3 bits were for duration in beats, 2 bits were for pitch-bend (bending a single pitch up or down for a given note), and 1 bit was for volume (0 or 100).

![genotype](https://github.com/AhmedKhota/Data-Science-Projects/assets/139664971/d872f573-8cd6-4c04-a80e-0f75d9ffab60)

The GA uses the Tournament Selection method with a Tournament size of 16, and two-point crossover, to evolve successive generations. The population size was set to 400 and with each new generation, the children produced by the selected parents completely replaced them in the new generation.

Firstly, randomly generated sounds, using the same genotype design, were evaluated manually, not using the GA, by listening to them and assigning a fitness value from 1 to 5. 3000 such sounds were evaluated and used as training data for a neural network to learn how to evaluate fitness of such genotypes.

Two neural network models were used, one using a linear activation function for the output layer and one using a sigmoid activation function for the output layer. The
neural network layout is shown below.

![GANNlayout](https://github.com/AhmedKhota/GA-sound-generation/assets/139664971/e82eea15-8f51-4c77-88d6-9c08a62aad5d)

Using 10-fold cross-validation, the neural network was trained to evaluate the fitness of the genotypes and achieved an average Mean Absolute Error (MAE) of 0.136 on the normalized rating scale, and a correlation between actual and predicted values of 0.725. 

The neural network model was then used to automatically evaluate the fitness of individuals produced by the GA over successive generations.

After the GA generated fit individuals (sounds), the fittest individuals were selected and the midi files were converted to wav files 
