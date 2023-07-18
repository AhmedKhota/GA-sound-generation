import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import glob
import random
from IPython.display import Audio, Image
from scipy.io import wavfile
import soundfile
import math
import pandas as pd
import sklearn.preprocessing as skp
import shutil
from sklearn.decomposition import FactorAnalysis
from pyAudioAnalysis import MidTermFeatures as aF
from pyAudioAnalysis import audioBasicIO as aIO 
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import opensmile
import audiofile

class Model_VA:
    def __init__(self,mono,splitby):
        #variables needed:
        #self.wavs = [] declare here or outside the class?
        #self.mono = mono is the main source folder for audio files
        self.TrainV_dir = f'Feb2022/TrainV'
        self.TestV_dir = f'Feb2022/TestV'
        self.TrainA_dir = f'Feb2022/TrainA'
        self.TestA_dir = f'Feb2022/TestA'
        self.csv_file = 'labelsVAv5.csv'    
        self.badfilescsv = 'BadFiles.csv'
        self.splitby = splitby
        self.mono = mono
        self.Train_dir_V = glob.glob('Feb2022/TrainV/*.wav')
        self.Test_dir_V = glob.glob('Feb2022/TestV/*.wav')
        self.Train_dir_A = glob.glob('Feb2022/TrainA/*.wav')
        self.Test_dir_A = glob.glob('Feb2022/TestA/*.wav')
        self.Ext_dir = glob.glob("F:/GA_midis/test/allmidis/100000/batchwavs/*.wav")
        self.All_Train_dir = glob.glob('Feb2022/All_Train/*.wav')
    
    def shuffle_split_files_labels_PA(self, mwl, mws,swl, sws): #new function for splitting-checked-working
        
        wavs =[]
        
        for file in self.mono:
            wavs.append(file)
        
        #Audio_data = Audio_data
        labels = pd.read_csv(self.csv_file)
        filenames_mono = []

        for filename in self.mono:
            filenames_mono.append(os.path.basename(filename))

        df = pd.DataFrame(filenames_mono, columns = ['Filename'])
        df['V'] = df['Filename'].map(labels.set_index('Filename')['V'])
        df['A'] = df['Filename'].map(labels.set_index('Filename')['A'])
        Labels = df
        Vlabels = Labels.pop('V')
        Alabels = Labels.pop('A')
        scaler = skp.MinMaxScaler(feature_range=(0, 1))
        LabelsV = scaler.fit_transform(np.array(Vlabels).reshape(-1, 1))
        LabelsA = scaler.fit_transform(np.array(Alabels).reshape(-1, 1))

        def shuffle_in_unison(a, b):
            rng_state = np.random.get_state()
            np.random.shuffle(a)
            np.random.set_state(rng_state)
            np.random.shuffle(b)

        LabelsAll = np.hstack((LabelsV,LabelsA))
        shuffle_in_unison(wavs,LabelsAll)

        split = int(len(wavs)*self.splitby)
        train_files = wavs[:split]
        train_labels = LabelsAll[:split] #use in labels fit to frames function
        test_files = wavs[split:]
        test_labels = LabelsAll[split:]
        
        train_data = self.Build_data(train_files,mwl,mws,swl,sws)
        test_data = self.Build_data(test_files,mwl,mws,swl,sws)
        
        return train_data, test_data, train_labels, test_labels
    
    
    def shuffle_split_files_labels(self, Audio_data): #new function for splitting-checked-working
        
        Audio_data = Audio_data
        labels = pd.read_csv(self.csv_file)
        filenames_mono = []

        for filename in self.mono:
            filenames_mono.append(os.path.basename(filename))

        df = pd.DataFrame(filenames_mono, columns = ['Filename'])
        df['V'] = df['Filename'].map(labels.set_index('Filename')['V'])
        df['A'] = df['Filename'].map(labels.set_index('Filename')['A'])
        Labels = df
        Vlabels = Labels.pop('V')
        Alabels = Labels.pop('A')
        scaler = skp.MinMaxScaler(feature_range=(0, 1))
        LabelsV = scaler.fit_transform(np.array(Vlabels).reshape(-1, 1))
        LabelsA = scaler.fit_transform(np.array(Alabels).reshape(-1, 1))

        def shuffle_in_unison(a, b):
            rng_state = np.random.get_state()
            np.random.shuffle(a)
            np.random.set_state(rng_state)
            np.random.shuffle(b)

        LabelsAll = np.hstack((LabelsV,LabelsA))
        shuffle_in_unison(Audio_data,LabelsAll)

        split = int(len(Audio_data)*self.splitby)
        train_data = Audio_data[:split]
        train_labels = LabelsAll[:split] #use in labels fit to frames function
        test_data = Audio_data[split:]
        test_labels = LabelsAll[split:]
        
        return train_data, test_data, train_labels, test_labels
    
    def split_shuffle_copy_files_labels(self):

        wavs = []

        for filename in glob.glob('Mono/*.wav'):
            #print(os.path.basename(filename))
            #if os.path.basename(filename) not in self.check_bad_files(self.badfilescsv):
            wavs.append(filename)
        print(len(wavs))

        LabelsV = pd.read_csv(self.csv_file).pop('V')
        LabelsA = pd.read_csv(self.csv_file).pop('A')

        labelsV = np.asarray(LabelsV)
        labelsA = np.asarray(LabelsA)

        labels_filenames = np.array(pd.read_csv(self.csv_file))

        #get order to match between wavs and labels fix this not working first!
        files = []

        for i in range(len(labelsV)):
            for file in wavs:
                if labels_filenames[i][0] == os.path.basename(file):
                    files.append(file)

        print(len(files))  

        #now files is sorted the same order as LabelsV and labelsA

        #normalize label data
        #labels = np.delete(labels,0,1)
        scaler = skp.MinMaxScaler(feature_range=(0, 1))
        Vlabels = scaler.fit_transform(np.array(labelsV).reshape(-1, 1))
        Alabels = scaler.fit_transform(np.array(labelsA).reshape(-1, 1))    

        # shuffle the two arrays shuffle function
        def shuffle_in_unison_scary(a, b):
            rng_state = np.random.get_state()
            np.random.shuffle(a)
            np.random.set_state(rng_state)
            np.random.shuffle(b)

        #perform shuffle    
        shuffle_in_unison_scary(files, Vlabels) #Alabels?

        #do 70-30 and 90-10 splits

        #splitby = 0.7
        #splitby2 = 0.9

        split = int(len(files)*self.splitby)
        train_files = files[:split]
        train_labelsV = Vlabels[:split] #use in labels fit to frames function
        test_files = files[split:]
        test_labelsV = Vlabels[split:]
        train_labelsA = Alabels[:split]
        test_labelsA = Alabels[split:]

        TrainV_dir = f'Feb2022/TrainV'
        TestV_dir = f'Feb2022/TestV'
        TrainA_dir = f'Feb2022/TrainA'
        TestA_dir = f'Feb2022/TestA'

        for file in train_files:
            for label in train_labelsV:
                shutil.copyfile(file, os.path.join(TrainV_dir, os.path.basename(file)))

        for file in test_files:
            for label in test_labelsV:
                shutil.copyfile(file, os.path.join(TestV_dir, os.path.basename(file)))

        for file in train_files:
            for label in train_labelsA:
                shutil.copyfile(file, os.path.join(TrainA_dir, os.path.basename(file)))

        for file in test_files:
            for label in test_labelsA:
                shutil.copyfile(file, os.path.join(TestA_dir, os.path.basename(file)))
                
        print("all files copied")

        New_dirs = []

        #Path_Main = Nov2021

        Root1 = f'Feb2022/'

        TrainV = 'TrainV/'
        TestV = 'TestV/'
        TrainA = 'TrainA/'
        TestA = 'TestA/'

        New_dirs.append(os.path.join(Root1,TrainV))
        New_dirs.append(os.path.join(Root1,TestV))
        New_dirs.append(os.path.join(Root1,TrainA))
        New_dirs.append(os.path.join(Root1,TestA))

        #if pcm16 is needed run this, it will overwrite existing file and data formats

        for dir in New_dirs:
            for filename in glob.glob(dir):
                data, samplerate = soundfile.read(filename)
                soundfile.write(filename, data, samplerate, subtype='PCM_16')
                
    def Delete_files(self, folder):
        
        folder = folder
        for file in folder:
            os.remove(file)
                
    def Build_data(self,Train_path, mid_win_length, mid_win_step, short_win_length, short_win_step ): #try access values from split shuffle function to determine folders

        #mid_win_length = 0.3 #was 0.25
        #mid_win_step = 0.3 #was 0.15
        #short_win_length = 0.05
        #short_win_step = 0.01

        #dirs = [Class_path_1, Class_path_2, Class_path_3] 

        mt_arr = []
        st_arr = []

        for file in Train_path:
            #if os.path.basename(file) not in self.check_bad_files(self.badfilescsv):
            #print(file)
            fs, s = aIO.read_audio_file(file)
            mt, st, mt_n = aF.mid_feature_extraction(s, fs, mid_win_length * fs, mid_win_step * fs, short_win_length * fs, short_win_step * fs)
            #class_Name = os.path.split(os.path.split(directory)[0])[1]
            mt_arr.append(mt)

        return mt_arr
    
        #build frame arrays from Train_data and Test_data 
    #make function repeatable so it returns one array at a time and you can use it the same way for Train V, Test V, Train A, Test A
    
    def opensmile_Functionals_Build(self, Train_path): 
        
        smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,)
        
        feats_df = smile.process_files(Train_path)
        
        return feats_df
    
    def opensmile_LLDs_Build(self, Train_path): 
        
        data = []
        
        smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,)
        
        for file in Train_path:
            signal, sampling_rate = audiofile.read(file)
            data.append(smile.process_signal(signal,sampling_rate))
            #data.append(smile.process_signal(signal,sampling_rate).to_numpy())
        
        return data
        #return np.array(data)
        
    def opensmile_LLDs_Build_df(self, Train_path): 
            
        smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,)
        
        data = smile.process_files(Train_path)
        #data.append(smile.process_signal(signal,sampling_rate).to_numpy())
        
        return data
        #return np.array(data)

    def scaled_frames_opensmileLLD(self, Train_data, Test_data, Train_dir_V, Test_dir_V):

        All_X = pd.concat([Train_data,Test_data])
        from sklearn import preprocessing
        x = All_X.values #returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        All_X_scaled = pd.DataFrame(x_scaled)
        Train_data_scaled = All_X_scaled.iloc[:len(Train_data),:]
        Test_data_scaled = All_X_scaled.iloc[len(Train_data):,:]

        #how to get numframes?
        Train_data_list = self.opensmile_LLDs_Build(Train_dir_V)
        Test_data_list = self.opensmile_LLDs_Build(Test_dir_V)

        numframes_Train_LLD = []
        numframes_Test_LLD = []

        for datalist in Train_data_list:
            numframes_Train_LLD.append(len(datalist))

        for datalist in Test_data_list:
            numframes_Test_LLD.append(len(datalist))

        j = 0
        k = 0
        Train_frames_LLD = []

        for i in range(len(numframes_Train_LLD)):
            j = numframes_Train_LLD[i]
            Train_frames_LLD.append(Train_data_scaled[k:(j+k)])
            #print('k: ', k, ': ', 'j+k: ', j+k)
            k = j + k

        j = 0
        k = 0
        Test_frames_LLD = []

        for i in range(len(numframes_Test_LLD)):
            j = numframes_Test_LLD[i]
            Test_frames_LLD.append(Test_data_scaled[k:(j+k)])
            #print('k: ', k, ': ', 'j+k: ', j+k)
            k = j + k

        return Train_frames_LLD, Test_frames_LLD
    
    def frames_from_data(self, data): #pass in Train_data_V, then run same for Test_data_V, Train_data_A, Test_data_A made from Build_data()

        frames = []

        for i in range(len(data)):
            frames.append(np.hsplit(data[i], len(data[i][0])))

        _X_ = []

        for subarray in frames:
            for elem in subarray:
                _X_.append(elem)

        _X_ = np.array(_X_)

        return _X_
    
    def scaled_frames_opensmileFunc(self, Train_data, Test_data, Train_dir_V, Test_dir_V):

        All_X = pd.concat([Train_data,Test_data])
        from sklearn import preprocessing
        x = All_X.values #returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        All_X_scaled = pd.DataFrame(x_scaled)
        Train_data_scaled = All_X_scaled.iloc[:len(Train_data),:]
        Test_data_scaled = All_X_scaled.iloc[len(Train_data):,:]

        #how to get numframes?
        Train_data_list = self.opensmile_Functionals_Build(Train_dir_V)
        Test_data_list = self.opensmile_Functionals_Build(Test_dir_V)

        numframes_Train_Func = []
        numframes_Test_Func = []

        for i in range(len(Train_data)):
            numframes_Train_Func.append(1)

        for i in range(len(Test_data)):
            numframes_Test_Func.append(1)

        j = 0
        k = 0
        Train_frames_Func = []

        for i in range(len(numframes_Train_Func)):
            j = numframes_Train_Func[i]
            Train_frames_Func.append(Train_data_scaled[k:(j+k)])
            #print('k: ', k, ': ', 'j+k: ', j+k)
            k = j + k

        j = 0
        k = 0
        Test_frames_Func = []

        for i in range(len(numframes_Test_Func)):
            j = numframes_Test_Func[i]
            Test_frames_Func.append(Test_data_scaled[k:(j+k)])
            #print('k: ', k, ': ', 'j+k: ', j+k)
            k = j + k

        return Train_frames_Func, Test_frames_Func
    

    def scaled_frames(self, Train_X, Test_X):#pass in frames_from_data(Train_data_V etc. for each of Train_X_V etc.) repeat function for A

        All_X = np.concatenate((Train_X, Test_X),axis=0) 

        All_X_split = np.split(All_X,[1,2,3,4,5,6,7,8,21,34,35,36,37,38,39,40,41,42,55,68,69,70,71,72,73,74,75,76,89,102,103,104,105,106,107,108,109,110,123,136],axis=1)
#maybe use the feature name to define the split instead
        scaler = MinMaxScaler()
        Scaled_data = []

        for array in All_X_split:
            if array.shape[1] != 0:
                scaler.fit(np.squeeze(array,axis = 2))
                Scaled_data.append(scaler.transform(np.squeeze(array,axis = 2)))        

        Scaled_frames = []

        for array in Scaled_data:
            arr = np.array(array)
            split = np.split(arr,len(array[0]),axis=1)
            for frame in split:
                Scaled_frames.append(frame)

        Scaled_frames = np.array(Scaled_frames)

        return Scaled_frames

        #rejoin all the data and split into train and test for training NN again

        #for each file, num_frames = length of file in ms / window size in ms and add to 
        #total framecount for Train and Test to get those numbers in those brackets

        #Set up variables that relate to the num files and window length so you dont need to plug in actual numbers

        #Scaled_frames_Split = np.split(Scaled_frames,[5321,8040],axis=1)

    def frames_split_train_test(self, Train_X, Test_X, Scaled_frames):#call for V and A seperately, pass in frames_from_data(_X_) and scaled_frames(frames_from_data(Train_data_V)) etc.
    #run for V and A seperately
        split_start = Train_X.shape[0]
        split_end = Train_X.shape[0]+Test_X.shape[0]

        Scaled_frames_Split = np.split(Scaled_frames,[split_start,split_end],axis=1)

        Train_frames_Scaled = np.squeeze(Scaled_frames_Split[0],axis = 2).T
        Test_frames_Scaled = np.squeeze(Scaled_frames_Split[1],axis = 2).T

        return Train_frames_Scaled, Test_frames_Scaled

    def num_frames(self, Train_data, Test_data):#pass in Train_data_V, Test_data_V

        #build numframes array
        numframes_Train = []
        numframes_Test = []

        for subarrays in Train_data:
            numframes_Train.append(len(subarrays[0][:]))

        for subarrays in Test_data:
            numframes_Test.append(len(subarrays[0][:]))

        return numframes_Train, numframes_Test #run for V and A seperately
    
    def normalized_labels_frames_Ext(self, csv_file,Train_dir_V, Train_dir_A): # shouldnt be needed with new method

        wavs = []
        for filename in glob.glob('Mono/*.wav'):
            wavs.append(filename)
        print(len(wavs))
        LabelsV = pd.read_csv(csv_file).pop('V')
        LabelsA = pd.read_csv(csv_file).pop('A')
        labelsV = np.array(LabelsV)
        labelsA = np.array(LabelsA)
        labels_filenames = np.array(pd.read_csv(csv_file))
        files = []

        for i in range(len(labelsV)):
            for file in wavs:
                #if os.path.basename(file) not in self.check_bad_files(self.badfilescsv):
                if labels_filenames[i][0] == os.path.basename(file):
                    files.append(file)
        print(len(files))  
        scaler = skp.MinMaxScaler(feature_range=(0, 1))
        Vlabels = scaler.fit_transform(np.array(labelsV).reshape(-1, 1))
        Alabels = scaler.fit_transform(np.array(labelsA).reshape(-1, 1))
        data_dir = 'Mono' #are you using this variable anywhere?
        Train_Y_V_arr = []
        Train_Y_A_arr = []

        for file in Train_dir_V:
                #match filename and return label
            for i in range(len(labels_filenames)):
                #if os.path.basename(file) not in self.check_bad_files(self.badfilescsv):
                if labels_filenames[i][0] == os.path.basename(file):
                    #print('match')
                    Train_Y_V_arr.append(labels_filenames[i][1])
                    Train_Y_A_arr.append(labels_filenames[i][2])
        Train_Y_V_arr = np.array(Train_Y_V_arr)
        Train_Y_A_arr = np.array(Train_Y_A_arr)
        All_Y_V = Train_Y_V_arr
        All_Y_A = Train_Y_A_arr
        #normalize labels
        All_Y_V_scaled = scaler.fit_transform(np.array(All_Y_V).reshape(-1, 1))
        All_Y_A_scaled = scaler.fit_transform(np.array(All_Y_A).reshape(-1, 1))
        Train_Y_V_scaled = All_Y_V_scaled
        Train_Y_A_scaled = All_Y_A_scaled
        
        return Train_Y_V_scaled, Train_Y_A_scaled

    def normalized_labels_frames(self, csv_file,Train_dir_V, Test_dir_V, Train_dir_A, Test_dir_A): # shouldnt be needed with new method

        wavs = []

        for filename in glob.glob('Mono/*.wav'):
            #if os.path.basename(filename) not in self.check_bad_files(self.badfilescsv):
            #print(os.path.basename(filename))
            wavs.append(filename)
        print(len(wavs))

        #Change to LabelsVADec2021.csv and make sure filenames in csv are correct for output#.wav files and WallE_34.wav file

        LabelsV = pd.read_csv(csv_file).pop('V')
        LabelsA = pd.read_csv(csv_file).pop('A')

        labelsV = np.array(LabelsV)
        labelsA = np.array(LabelsA)

        labels_filenames = np.array(pd.read_csv(csv_file))

        files = []

        for i in range(len(labelsV)):
            for file in wavs:
                #if os.path.basename(file) not in self.check_bad_files(self.badfilescsv):
                if labels_filenames[i][0] == os.path.basename(file):
                    files.append(file)

        print(len(files))  

        #now files is sorted the same order as LabelsV and labelsA

        #normalize label data
        #labels = np.delete(labels,0,1)
        scaler = skp.MinMaxScaler(feature_range=(0, 1))
        Vlabels = scaler.fit_transform(np.array(labelsV).reshape(-1, 1))
        Alabels = scaler.fit_transform(np.array(labelsA).reshape(-1, 1))

        #seperate function 

        data_dir = 'Mono' #are you using this variable anywhere?

        #Train_dir_V = glob.glob('Nov2021/70_30/TrainV/*.wav')
        #Test_dir_V = glob.glob('Nov2021/70_30/TestV/*.wav')
        #Train_dir_A = glob.glob('Nov2021/70_30/TrainA/*.wav')
        #Test_dir_A = glob.glob('Nov2021/70_30/TestA/*.wav')

        Train_Y_V_arr = []
        Test_Y_V_arr = []
        Train_Y_A_arr = []
        Test_Y_A_arr = []

        #for file in Train_dir_V:
            #print(os.path.basename(file))

        for file in Train_dir_V:
                #match filename and return label
            for i in range(len(labels_filenames)):
                #if os.path.basename(file) not in self.check_bad_files(self.badfilescsv):
                if labels_filenames[i][0] == os.path.basename(file):
                    #print('match')
                    Train_Y_V_arr.append(labels_filenames[i][1])
                    Train_Y_A_arr.append(labels_filenames[i][2])

        for file in Test_dir_V:
                #match filename and return label
            for i in range(len(labels_filenames)):
                #if os.path.basename(file) not in self.check_bad_files(self.badfilescsv):
                if labels_filenames[i][0] == os.path.basename(file):
                    #print('match')
                    Test_Y_V_arr.append(labels_filenames[i][1])
                    Test_Y_A_arr.append(labels_filenames[i][2])

        Train_Y_V_arr = np.array(Train_Y_V_arr)
        Train_Y_A_arr = np.array(Train_Y_A_arr)
        Test_Y_V_arr = np.array(Test_Y_V_arr)
        Test_Y_A_arr = np.array(Test_Y_A_arr)

        All_Y_V = np.hstack((Train_Y_V_arr,Test_Y_V_arr))
        All_Y_A = np.hstack((Train_Y_A_arr,Test_Y_A_arr))

        #normalize labels
        All_Y_V_scaled = scaler.fit_transform(np.array(All_Y_V).reshape(-1, 1))
        All_Y_A_scaled = scaler.fit_transform(np.array(All_Y_A).reshape(-1, 1))

        splitby = self.splitby
        print(splitby)
        split = int(len(files)*splitby)
        Train_Y_V_scaled = All_Y_V_scaled[:split]
        Test_Y_V_scaled = All_Y_V_scaled[split:]
        Train_Y_A_scaled = All_Y_A_scaled[:split]
        Test_Y_A_scaled = All_Y_A_scaled[split:]
        
        return Train_Y_V_scaled, Test_Y_V_scaled, Train_Y_A_scaled, Test_Y_A_scaled

    def labels_to_frames(self, Data_frames, labels):#use frames_from_data(Train_data_V)

        #apply labels to frames:
        num_frames = []

        for subarray in Data_frames:#use frames_from_data(Train_data_V) etc. variable should have come from Build_Data()
            num_frames.append(subarray.shape[1])

        Data_Y = np.repeat(labels,num_frames)

        Data_Y = np.array(Data_Y)

        return Data_Y
    
    def labels_to_frames_opensmileLLD(self, labels, _dir):#use Train_Y_V_scaled, Train_dir_V
        
        #how to get numframes?
        data_list = self.opensmile_LLDs_Build(_dir)
   
        numframes_LLD = []

        for datalist in data_list:
            numframes_LLD.append(len(datalist))

        Data_Y = np.repeat(labels,numframes_LLD)

        Data_Y = np.array(Data_Y)

        return Data_Y
    
    def labels_to_frames_opensmileFunc(self, labels, _dir):#use Train_Y_V_scaled, Train_dir_V
        
        #how to get numframes?
        data_list = self.opensmile_Functionals_Build(_dir)
   
        numframes_Func = []

        for i in range(len(data_list)):
            numframes_Func.append(1)

        Data_Y = np.repeat(labels,numframes_Func)

        Data_Y = np.array(Data_Y)

        return Data_Y
    
    def numframes_opensmileLLD(self, _dir):
    
        #how to get numframes?
        data_list = self.opensmile_LLDs_Build(_dir)
   
        numframes_LLD = []

        for datalist in data_list:
            numframes_LLD.append(len(datalist))
                               
        return numframes_LLD
    
    def numframes_opensmileFunc(self, _dir):
    
        #how to get numframes?
        data_list = self.opensmile_Functionals_Build(_dir)
   
        numframes_Func = []

        for i in range(len(data_list)):
            numframes_Func.append(1)
                               
        return numframes_Func
    
    
    def join_frames_for_ragged(self, frames_Scaled, numframes):

    # build the arrays of frames per file that you will use as ragged tensor input to the NN
    # also needed for featureFrame matrix to use for Factor Analysis

        data_Scaled = []
        j = 0

        for nums in numframes:
            data_Scaled.append(frames_Scaled[j:nums+j])
            j = nums + j
        
        
        return data_Scaled
    
    def features_frames_data(self, Train_data_V_Scaled, Test_data_V_Scaled): #what to pass in here? pass in join_frames_for_ragged(Train_frames_Scaled_V, Test_frames_Scaled_V, Train_frames_Scaled_A, Train_frames_Scaled_A)[0], join_frames_for_ragged(Train_frames_Scaled_V, Test_frames_Scaled_V, Train_frames_Scaled_A, Train_frames_Scaled_A)[1]

        Feat_names = pd.read_csv('FeatNames.csv')
        Featnames = np.array(Feat_names)
        Features_data = []
        fcount = 0

        X_data_V = np.concatenate((Train_data_V_Scaled, Test_data_V_Scaled),axis=0)
        Features_data.append(Featnames)
        for i in range(len(X_data_V)):
            #labelV = Y_data_V[i]
            #labelA = Y_data_A[i]
            for row in X_data_V[i]:
                Features_data.append(row)

        Features_flat = []

        for elem in Features_data:
            Features_flat.append(list(itertools.chain.from_iterable(elem)))

        for line in Features_flat:
            self.writecsv(line,'FeaturesFramesData.csv')

        FeatFrame_data = pd.read_csv('FeaturesFramesData.csv')

        return FeatFrame_data

    def features_frames_data_1(self, data_Scaled): #what to pass in here?

        Feat_names = pd.read_csv('FeatNames.csv')
        Featnames = np.array(Feat_names)
        Features_data = []
        fcount = 0

        for i in range(len(data_Scaled)):
            #labelV = Y_data_V[i]
            #labelA = Y_data_A[i]
            for row in data_Scaled[i]:
                if fcount == 136:
                    fcount = 0
                fcount = fcount + 1
                #print(fcount-1)
                Features_data.append((Featnames[fcount-1],row))

        Features_flat = []

        for elem in Features_data:
            Features_flat.append(list(itertools.chain.from_iterable(elem)))

        FeatFrame_data = Features_flat

        return FeatFrame_data
    
    def features_frames_dataframe(self,Train_frames_Scaled, Test_frames_Scaled):#only use this one not the two functions above
        
        Feat_names = pd.read_csv('FeatNames.csv')
        Featnames = np.array(Feat_names)
        
        X_frames = np.concatenate((Train_frames_Scaled, Test_frames_Scaled),axis=0)
        df = pd.DataFrame(data = X_frames, index = None, columns = Featnames)
        
        return df
    
    def features_frames_dataframe_opensmileLLD(self,Train_data, Test_data):#only use this one not the two functions above
        
        Feat_names = pd.read_csv('LLD_Features.csv')
        Featnames = np.array(Feat_names)
        
        X_frames = np.concatenate((Train_data, Test_data),axis=0)
        df = pd.DataFrame(data = X_frames, index = None, columns = Featnames)
        
        return df
    
    def factor_analysis(self, n_components, data): #pass features_frames_data(Train_data_V_Scaled, Test_data_V_Scaled) for data

        #data = features_frames_data(Train_data_V_Scaled, Test_data_V_Scaled)

        feature_names = list(data.columns)#dont need this

        FA_2 = FactorAnalysis(n_components=n_components, random_state=0, rotation="varimax")
        FA_2.fit(data)
        FATF_data = FA_2.transform(data)

        return FATF_data     
    
    def factor_eigenvals(self, data):
    
        from factor_analyzer import FactorAnalyzer
        factorA = FactorAnalyzer(rotation=None)
        factorA.fit(data)
        # Check Eigenvalues
        ev, v = factorA.get_eigenvalues()
        evs = []
        for eigen in ev:
            if eigen >= 0:
                evs.append(eigen)
        
        return evs
    
    def scree_plot(self, data):
                   
        from factor_analyzer import FactorAnalyzer
        factorA = FactorAnalyzer(rotation=None)
        factorA.fit(data)
        # Check Eigenvalues
        ev, v = factorA.get_eigenvalues()
        # Create scree plot using matplotlib
        plt.scatter(range(1,data.shape[1]+1),ev)
        plt.plot(range(1,data.shape[1]+1),ev)
        plt.title('Scree Plot')
        plt.xlabel('Factors')
        plt.ylabel('Eigenvalue')
        plt.grid()
        plt.show()
    
    def writecsv(self, data, filename):
        import csv
        with open(filename,'a', newline = '', encoding = 'UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(data)
    
    def writecsv_rows(self, data, filename):
        import csv
        with open(filename,'a', newline = '', encoding = 'UTF8') as f:
            writer = csv.writer(f)
            writer.writerows(data)
            
    def Random_Forest_Infer_opensmileLLD_Ext(self, Train_data_scaled, Test_data_scaled, TrainYV_Scaled_frames, TrainYA_Scaled_frames):

        TrainY_frame = np.vstack((TrainYV_Scaled_frames, TrainYA_Scaled_frames)).T
        #Then use in forest.fit(FATF_Train, Training Labels per frame for both V and A (from Build_Scaled_Labels_per_frame)
        from sklearn.ensemble import RandomForestRegressor
        forest = RandomForestRegressor(random_state=0)
        forest.fit(Train_data_scaled, TrainY_frame)
        predictions = forest.predict(Test_data_scaled)
        predictionsV = np.squeeze(np.hsplit(predictions,2)[0],axis=1)
        predictionsA = np.squeeze(np.hsplit(predictions,2)[1],axis=1)
        
        return predictionsV, predictionsA
            
    def Random_Forest_Infer_opensmileLLD(self, Train_data_scaled, Test_data_scaled, TrainYV_Scaled_frames, TrainYA_Scaled_frames, TestYV_Scaled_frames, TestYA_Scaled_frames):

        TrainY_frame = np.vstack((TrainYV_Scaled_frames, TrainYA_Scaled_frames)).T
        
        #Then use in forest.fit(FATF_Train, Training Labels per frame for both V and A (from Build_Scaled_Labels_per_frame)
        from sklearn.ensemble import RandomForestRegressor
        forest = RandomForestRegressor(random_state=0)
        forest.fit(Train_data_scaled, TrainY_frame)
        
        predictions = forest.predict(Test_data_scaled)
        TestY_frame = np.vstack((TestYV_Scaled_frames, TestYA_Scaled_frames)).T

        errors = abs(predictions - TestY_frame)

        #print('Metrics for Random Forest Trained on Expanded Data')
        #print('Average absolute error:', round(np.mean(errors), 2))
        #mape = np.mean(100 * (errors / TestY_frame.T))
        #accuracy = 100 - mape
        #print('Accuracy:', round(accuracy, 2), '%.')

        predictionsV = np.squeeze(np.hsplit(predictions,2)[0],axis=1)
        predictionsA = np.squeeze(np.hsplit(predictions,2)[1],axis=1)
        
        return predictionsV, predictionsA
    
    def Random_Forest_Infer_opensmileFunc(self, data, Train_YV_Scaled, Train_YA_Scaled, Test_YV_Scaled, Test_YA_Scaled, Train_frames_Scaled, Test_frames_Scaled):

        TrainX_FA = np.split(data, [len(Train_frames_Scaled), len(Train_frames_Scaled)+len(Test_frames_Scaled)])[0]
        TestX_FA = np.split(data, [len(Train_frames_Scaled), len(Train_frames_Scaled)+len(Test_frames_Scaled)])[1]
        
        TrainY_frame = np.hstack((Train_YV_Scaled, Train_YA_Scaled)) 
        
        #Then use in forest.fit(FATF_Train, Training Labels per frame for both V and A (from Build_Scaled_Labels_per_frame)
        from sklearn.ensemble import RandomForestRegressor
        forest = RandomForestRegressor(random_state=0)
        forest.fit(TrainX_FA, TrainY_frame)
        
        predictions = forest.predict(TestX_FA)
        TestY_frame = np.hstack((Test_YV_Scaled, Test_YA_Scaled)) 

        errors = abs(predictions - TestY_frame)

        #print('Metrics for Random Forest Trained on Expanded Data')
        #print('Average absolute error:', round(np.mean(errors), 2))
        #mape = np.mean(100 * (errors / TestY_frame.T))
        #accuracy = 100 - mape
        #print('Accuracy:', round(accuracy, 2), '%.')

        predictionsV = np.squeeze(np.hsplit(predictions,2)[0],axis=1)
        predictionsA = np.squeeze(np.hsplit(predictions,2)[1],axis=1)
        
        return predictionsV, predictionsA
    
    def Random_Forest_Infer_opensmileFunc_Feat(self, Train_data_df, Test_data_df, Train_YV_Scaled, Train_YA_Scaled, Test_YV_Scaled, Test_YA_Scaled):
  
        TrainY_frame = np.hstack((Train_YV_Scaled, Train_YA_Scaled)) 
        
        #Then use in forest.fit(FATF_Train, Training Labels per frame for both V and A (from Build_Scaled_Labels_per_frame)
        from sklearn.ensemble import RandomForestRegressor
        forest = RandomForestRegressor(random_state=0)
        forest.fit(Train_data_df, TrainY_frame)
        
        predictions = forest.predict(Test_data_df)
        TestY_frame = np.hstack((Test_YV_Scaled, Test_YA_Scaled)) 

        errors = abs(predictions - TestY_frame)

        #print('Metrics for Random Forest Trained on Expanded Data')
        #print('Average absolute error:', round(np.mean(errors), 2))
        #mape = np.mean(100 * (errors / TestY_frame.T))
        #accuracy = 100 - mape
        #print('Accuracy:', round(accuracy, 2), '%.')

        predictionsV = np.squeeze(np.hsplit(predictions,2)[0],axis=1)
        predictionsA = np.squeeze(np.hsplit(predictions,2)[1],axis=1)
        
        return predictionsV, predictionsA
    
    def Random_Forest_Infer(self, data, TrainYV_Scaled_frames, TrainYA_Scaled_frames, TestYV_Scaled_frames, TestYA_Scaled_frames, Train_frames_Scaled, Test_frames_Scaled):

        TrainX_FA = np.split(data, [len(Train_frames_Scaled), len(Train_frames_Scaled)+len(Test_frames_Scaled)])[0]
        TestX_FA = np.split(data, [len(Train_frames_Scaled), len(Train_frames_Scaled)+len(Test_frames_Scaled)])[1]
        
        TrainY_frame = np.vstack((TrainYV_Scaled_frames, TrainYA_Scaled_frames)).T
        
        #Then use in forest.fit(FATF_Train, Training Labels per frame for both V and A (from Build_Scaled_Labels_per_frame)
        from sklearn.ensemble import RandomForestRegressor
        forest = RandomForestRegressor(random_state=0)
        forest.fit(TrainX_FA, TrainY_frame)
        
        predictions = forest.predict(TestX_FA)
        TestY_frame = np.vstack((TestYV_Scaled_frames, TestYA_Scaled_frames)).T

        errors = abs(predictions - TestY_frame)

        #print('Metrics for Random Forest Trained on Expanded Data')
        #print('Average absolute error:', round(np.mean(errors), 2))
        #mape = np.mean(100 * (errors / TestY_frame.T))
        #accuracy = 100 - mape
        #print('Accuracy:', round(accuracy, 2), '%.')

        predictionsV = np.squeeze(np.hsplit(predictions,2)[0],axis=1)
        predictionsA = np.squeeze(np.hsplit(predictions,2)[1],axis=1)
        
        return predictionsV, predictionsA
    
    def NeuralNet_Compile(self, inputs, lr):
        
        model = tf.keras.Sequential()
        #model.add(tf.keras.layers.Flatten(input_shape=(35,)))
        #model.add(tf.keras.layers.Dense(128, input_dim=35, activation='relu'))
        #model.add(tf.keras.layers.Input(shape=(None,35)))
        model.add(tf.keras.layers.Dense(64, input_dim = inputs, activation='relu'))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(1, kernel_initializer='normal'))
        
        Optim = tf.keras.optimizers.Adam(learning_rate=lr)

        model.compile(loss='mean_absolute_error', optimizer=Optim, metrics=['accuracy'])
        
        return model
    
    def CNN_model_Compile(self, lr):
    
        model = tf.keras.models.Sequential()
        #model.add(tf.keras.layers.Input(shape=(None, 136),ragged = True))
        model.add(tf.keras.layers.Input(shape=(None, None, 136)))
        #model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', name='Conv2DL1'))
        model.add(tf.keras.layers.MaxPooling2D((2,2), name='MaxP1'))
        model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu', name='Conv2DL2'))
        model.add(tf.keras.layers.Dense(32, activation='relu', name='dense1'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid', name='denseoutput2'))
        # Compile model
        Optim = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(
            loss='mae',
            optimizer=Optim,
            metrics=['accuracy'],
        )
        return model
    
    def CNN_Infer(self, model, Train_data_Scaled, Test_data_Scaled, TrainY_Scaled, TestY_Scaled):
    
        # shuffle the two arrays shuffle function
        def shuffle_in_unison_scary(a, b):
            rng_state = np.random.get_state()
            np.random.shuffle(a)
            np.random.set_state(rng_state)
            np.random.shuffle(b)
                          
        #shuffle Train and Test data and labels for V and A:
        shuffle_in_unison_scary(Train_data_Scaled, TrainY_Scaled)
        shuffle_in_unison_scary(Test_data_Scaled, TestY_Scaled)
        
        Xtrain = tf.ragged.constant(Train_data_Scaled)
        Xtest = tf.ragged.constant(Test_data_Scaled)
        
        model.fit(Xtrain, TrainY_Scaled, validation_data=(Xtest, TestY_Scaled), batch_size = 32, epochs=1000, verbose=2, shuffle=True)
        
        predictions = model.predict(Xtest)
        
        return predictions
                  
    def NeuralNet_Infer(self, model, n_factors, data, TrainY_Scaled_frames, TestY_Scaled_frames, Train_frames_Scaled, Test_frames_Scaled):
        
        scaler = skp.MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(np.array(data).reshape(-1,n_factors))
        
        TrainX_FA = np.split(data, [len(Train_frames_Scaled), len(Train_frames_Scaled)+len(Test_frames_Scaled)])[0]
        TestX_FA = np.split(data, [len(Train_frames_Scaled), len(Train_frames_Scaled)+len(Test_frames_Scaled)])[1]
        
        self.shuffle_in_unison_scary(TrainX_FA, TrainY_Scaled_frames)
        self.shuffle_in_unison_scary(TestX_FA, TestY_Scaled_frames)
        
        model.fit(TrainX_FA, TrainY_Scaled_frames, validation_data=(TestX_FA, TestY_Scaled_frames),
          batch_size = 32, epochs=1000, verbose=2, shuffle=True)
                   
        predictions = model.predict(TestX_FA)
                   
        return predictions
    
    def NeuralNet_Infer_Func(self, model, Train_data_df, Test_data_df, Train_Y_Scaled, Test_Y_Scaled):
        
        Train_data = Train_data_df.to_numpy()
        Test_data = Test_data_df.to_numpy()
        
        self.shuffle_in_unison_scary(Train_data, Train_Y_Scaled)
        self.shuffle_in_unison_scary(Test_data, Test_Y_Scaled)
        
        model.fit(Train_data, Train_Y_Scaled, validation_data=(Test_data, Test_Y_Scaled),
          batch_size = 32, epochs=100, verbose=2, shuffle=True)
                   
        predictions = model.predict(Test_data)
                   
        return predictions
        
    def shuffle_in_unison_scary(self, a, b):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        
    def Avg_over_frames_Ext(self, predictions, num_frames):
        
        start = 0
        stop = 0
        indices = []
        predavg = []

        for framecount in num_frames:
            stop = framecount + start
            for i in range(start,stop):
                indices.append(i)
            predavg.append(np.average(predictions[indices]))
            start = start + framecount
            indices.clear()
            
        return predavg
    
    def Avg_over_frames(self, TestY_Scaled_frames, predictions, num_frames):
        
        #averaged over number of frames - not weighted average
        y_test = TestY_Scaled_frames
        #predavg
        start = 0
        stop = 0
        indices = []
        predavg = []
        actavg = []

        for framecount in num_frames:
            stop = framecount + start
            #print('start1: ', start)
            for i in range(start,stop):
                indices.append(i)
            #print('indices: ', indices)
            #print('predictions: ', predictions[indices])
            #print('predavg: ', np.average(predictions[indices]))
            predavg.append(np.average(predictions[indices]))
            #print('actavg: ', np.average(y_test[indices]))
            actavg.append(np.average(y_test[indices]))
            start = start + framecount
            indices.clear()
            #print('start2: ', start)
            #print('stop: ', stop)
            #print('framecount: ', framecount)
            #print('i:', i)
            #print(" ")
            
        return predavg, actavg
    
    def R2_score(self, predavg, actavg):
        
        from sklearn.metrics import r2_score

        return r2_score(actavg,predavg)
    
    def model_error_accuracy(self, predavg, actavg):
        
        # Performance metrics
        errors = abs(predavg - actavg)

        #print('Metrics for Random Forest Trained on Expanded Data')
        #print('Average absolute error:', round(np.mean(errors), 2))

        mape = np.mean(100 * (errors / actavg))

        accuracy = 100 - mape
        #print('Accuracy:', round(accuracy, 2), '%.')
        
        return accuracy
    
    def ttest_correl(self, pred, act):
        
        from scipy.stats import ttest_ind
        import scipy.stats
        
        return ttest_ind(act,pred),scipy.stats.pearsonr(act,pred)
    
    def check_bad_files(self, badfilescsv):
        
        badfiles = np.array(pd.read_csv(badfilescsv))
        return badfiles
    
    def remove_bad_files(self, badfilescsv):
        
        badfiles = np.array(pd.read_csv(badfilescsv))
        #check every folder for audio file 
        #check through labels for audio file
        #remove all records and files
        
    def Random_Forest_KFold(self, data, TrainYV_Scaled_frames, TrainYA_Scaled_frames, TestYV_Scaled_frames, TestYA_Scaled_frames, Train_frames_Scaled, Test_frames_Scaled):
        
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold
        
        TrainX_FA = np.split(data, [len(Train_frames_Scaled), len(Train_frames_Scaled)+len(Test_frames_Scaled)])[0]
        TestX_FA = np.split(data, [len(Train_frames_Scaled), len(Train_frames_Scaled)+len(Test_frames_Scaled)])[1]
        TrainY_frame = np.vstack((TrainYV_Scaled_frames, TrainYA_Scaled_frames)).T
        # Create an instance of Pipeline
        pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, max_depth=4))
        # Create an instance of StratifiedKFold which can be used to get indices of different training and test folds
        strtfdKFold = StratifiedKFold(n_splits=10)
        kfold = strtfdKFold.split(TrainX_FA, TrainY_frame)
        scores = []

        for k, (train, test) in enumerate(kfold):
            pipeline.fit(TrainX_FA.iloc[train, :], TrainY_frame.iloc[train])
            score = pipeline.score(TrainX_FA.iloc[test, :], TrainY_frame.iloc[test])
            scores.append(score)
            print('Fold: %2d, Training/Test Split Distribution: %s, Accuracy: %.3f' % (k+1, np.bincount(y_train.iloc[train]), score))

        print('\n\nCross-Validation accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))
        
        TestY_frame = np.vstack((TestYV_Scaled_frames, TestYA_Scaled_frames)).T
        
        predictions = pipeline.predict(TestX_FA)
        
        predictionsV = np.squeeze(np.hsplit(predictions,2)[0],axis=1)
        predictionsA = np.squeeze(np.hsplit(predictions,2)[1],axis=1)
        
        return predictionsV, predictionsA
    
    def create_folds(self, df, n_s=10, n_grp=None):
        df['Fold'] = -1

        if n_grp is None:
            skf = KFold(n_splits=n_s)
            target = df.target
        else:
            skf = StratifiedKFold(n_splits=n_s)
            df['grp'] = pd.cut(df.target, n_grp, labels=False)
            target = df.grp

        for fold_no, (t, v) in enumerate(skf.split(target, target)):
            df.loc[v, 'Fold'] = fold_no
        return df
    
    def Random_Forest_KFold_1(self, data, TrainY_Scaled_frames, TestY_Scaled_frames, Train_frames_Scaled, Test_frames_Scaled):
        
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold, KFold
        import warnings
        warnings.filterwarnings('ignore')
        
        TrainX_FA = pd.DataFrame(np.split(data, [len(Train_frames_Scaled), len(Train_frames_Scaled)+len(Test_frames_Scaled)])[0],columns=None)
        TestX_FA = np.split(data, [len(Train_frames_Scaled), len(Train_frames_Scaled)+len(Test_frames_Scaled)])[1]
        TrainY_frame = pd.DataFrame(TrainY_Scaled_frames.T,columns=None)
        
        from sklearn import preprocessing
        from sklearn import utils

        #convert labels to one-hot encoding before Kfold
        
        #lab_enc = preprocessing.LabelEncoder()
        #encoded = lab_enc.fit_transform(trainingScores)

        
        ##these are for using stratified create_folds function
        ##join TrainX_FA and TrainY_frame to create df for create_folds function
        ##TrainY_frame = np.reshape(TrainY_frame,(-1,1))
        ##joindf = np.concatenate((TrainX_FA,TrainY_frame),axis=1)
        ##Train_df = self.create_folds(joindf,n_s=10,n_grp=None)
        ##Train_X = np.split(Train_df,np.shape(TrainX_FA)[1],np.shape(TrainX_FA)[1]+1,axis=1)[0]
        ##Train_Y = np.split(Train_df,np.shape(TrainX_FA)[1],np.shape(TrainX_FA)[1]+1,axis=1)[1]
        # Create an instance of Pipeline
        pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, max_depth=4))
        # Create an instance of StratifiedKFold which can be used to get indices of different training and test folds
        KFold = KFold(n_splits=10)
        kfold = KFold.split(TrainX_FA, TrainY_frame)
        scores = []

        for k, (train, test) in enumerate(kfold):
            pipeline.fit(TrainX_FA.iloc[train, :], TrainY_frame.iloc[train])
            score = pipeline.score(TrainX_FA.iloc[test, :], TrainY_frame.iloc[test])
            scores.append(score)
            print('Fold: %2d, Training/Test Split Distribution: %s, Accuracy: %.3f' % (k+1, np.bincount(y_train.iloc[train]), score))

        print('\n\nCross-Validation accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))
        
        TestY_frame = TestY_Scaled_frames.T
        
        predictions = pipeline.predict(TestX_FA)
        
        return predictions
    
    def pyAudio_pipeline_RF_1(self, mwl, n_factors):
        
        ResultsV = []
        ResultsA = []

        mwl = mwl
        n_factors = n_factors

        mws = mwl
        swl = 0.05
        sws = 0.01
        
        train_data = self.shuffle_split_files_labels_PA(mwl, mws,swl, sws)[0]
        test_data = self.shuffle_split_files_labels_PA(mwl, mws,swl, sws)[1]
        train_labels = self.shuffle_split_files_labels_PA(mwl, mws,swl, sws)[2]
        test_labels = self.shuffle_split_files_labels_PA(mwl, mws,swl, sws)[3]
        #Audio_data = self.Build_data(self.mono,mwl,mws,swl,sws)
        #train_data = self.shuffle_split_files_labels(Audio_data)[0]
        #test_data = self.shuffle_split_files_labels(Audio_data)[1]
        #train_labels = self.shuffle_split_files_labels(Audio_data)[2]
        #test_labels = self.shuffle_split_files_labels(Audio_data)[3]
        Train_frames = self.frames_from_data(train_data)
        Test_frames = self.frames_from_data(test_data)
        Train_frames_Scaled, Test_frames_Scaled = self.frames_split_train_test(Train_frames,Test_frames, self.scaled_frames(Train_frames, Test_frames))
        numframes_Train, numframes_Test = self.num_frames(train_data,test_data)
        Train_YV_Scaled, Train_YA_Scaled = np.hsplit(train_labels,2)
        Test_YV_Scaled, Test_YA_Scaled = np.hsplit(test_labels,2)
        TrainYV_Scaled_frames = self.labels_to_frames(train_data, Train_YV_Scaled)
        TestYV_Scaled_frames = self.labels_to_frames(test_data, Test_YV_Scaled)
        TrainYA_Scaled_frames = self.labels_to_frames(train_data, Train_YA_Scaled)
        TestYA_Scaled_frames = self.labels_to_frames(test_data, Test_YA_Scaled)
        Train_data_Scaled = self.join_frames_for_ragged(Train_frames_Scaled, numframes_Train)
        Test_data_Scaled = self.join_frames_for_ragged(Test_frames_Scaled, numframes_Test)
        Featframe_df = self.features_frames_dataframe(Train_frames_Scaled, Test_frames_Scaled)
        FATF_data = self.factor_analysis(n_factors, Featframe_df)
        predictionsV, predictionsA = self.Random_Forest_Infer(FATF_data, TrainYV_Scaled_frames, TrainYA_Scaled_frames, TestYV_Scaled_frames, TestYA_Scaled_frames, Train_frames_Scaled, Test_frames_Scaled)
        predavgV, actavgV = np.array(self.Avg_over_frames(TestYV_Scaled_frames, predictionsV, numframes_Test))
        predavgA, actavgA = np.array(self.Avg_over_frames(TestYA_Scaled_frames, predictionsA, numframes_Test))
        err_accV = self.model_error_accuracy(predavgV, actavgV)
        err_accA = self.model_error_accuracy(predavgA, actavgA)
        ResultsV.append((mwl,n_factors,err_accV, np.average(np.absolute(np.subtract(predavgV,actavgV))),self.R2_score(predavgV,actavgV)))
        ResultsA.append((mwl,n_factors,err_accA, np.average(np.absolute(np.subtract(predavgA,actavgA))),self.R2_score(predavgA,actavgA)))

        return ResultsV, ResultsA
    
    def pyAudio_pipeline_RF(self, mwl, n_factors):

        #straight run RF with specified number of factors
        ResultsV = []
        ResultsA = []

        mwl = mwl
        n_factors = n_factors

        mws = mwl
        swl = 0.05
        sws = 0.01

        Train_data_V = self.Build_data(self.Train_dir_V,mwl,mws,swl,sws)
        Test_data_V = self.Build_data(self.Test_dir_V,mwl,mws,swl,sws)
        Train_data_A = Train_data_V
        Test_data_A = Test_data_V
        Train_frames_V = self.frames_from_data(Train_data_V)
        Test_frames_V = self.frames_from_data(Test_data_V)
        Train_frames_A = Train_frames_V
        Test_frames_A = Test_frames_V
        Train_frames_Scaled, Test_frames_Scaled = self.frames_split_train_test(Train_frames_V,Test_frames_V, self.scaled_frames(Train_frames_V, Test_frames_V))
        numframes_Train, numframes_Test = self.num_frames(Train_data_V,Test_data_V)
        Train_YV_Scaled = self.normalized_labels_frames(self.csv_file,self.Train_dir_V, self.Test_dir_V, self.Train_dir_A, self.Test_dir_A)[0]#not needed in new method
        Test_YV_Scaled = self.normalized_labels_frames(self.csv_file,self.Train_dir_V, self.Test_dir_V, self.Train_dir_A, self.Test_dir_A)[1]
        Train_YA_Scaled = self.normalized_labels_frames(self.csv_file,self.Train_dir_V, self.Test_dir_V, self.Train_dir_A, self.Test_dir_A)[2]
        Test_YA_Scaled = self.normalized_labels_frames(self.csv_file,self.Train_dir_V, self.Test_dir_V, self.Train_dir_A, self.Test_dir_A)[3]
        TrainYV_Scaled_frames = self.labels_to_frames(Train_data_V, Train_YV_Scaled)
        TestYV_Scaled_frames = self.labels_to_frames(Test_data_V, Test_YV_Scaled)
        TrainYA_Scaled_frames = self.labels_to_frames(Train_data_A, Train_YA_Scaled)
        TestYA_Scaled_frames = self.labels_to_frames(Test_data_A, Test_YA_Scaled)
        Train_data_Scaled = self.join_frames_for_ragged(Train_frames_Scaled, numframes_Train)
        Test_data_Scaled = self.join_frames_for_ragged(Test_frames_Scaled, numframes_Test)
        Featframe_df = self.features_frames_dataframe(Train_frames_Scaled, Test_frames_Scaled)
        FATF_data = self.factor_analysis(n_factors, Featframe_df)
        predictionsV, predictionsA = self.Random_Forest_Infer(FATF_data, TrainYV_Scaled_frames, TrainYA_Scaled_frames, TestYV_Scaled_frames, TestYA_Scaled_frames, Train_frames_Scaled, Test_frames_Scaled)
        predavgV, actavgV = np.array(self.Avg_over_frames(TestYV_Scaled_frames, predictionsV, numframes_Test))
        predavgA, actavgA = np.array(self.Avg_over_frames(TestYA_Scaled_frames, predictionsA, numframes_Test))
        err_accV = self.model_error_accuracy(predavgV, actavgV)
        err_accA = self.model_error_accuracy(predavgA, actavgA)
        ttest_correl_V = self.ttest_correl(predavgV, actavgV)
        ttest_correl_A = self.ttest_correl(predavgA, actavgA)
        ResultsV.append((mwl,n_factors,err_accV, np.average(np.absolute(np.subtract(predavgV,actavgV))),self.R2_score(predavgV,actavgV),ttest_correl_V))
        ResultsA.append((mwl,n_factors,err_accA, np.average(np.absolute(np.subtract(predavgA,actavgA))),self.R2_score(predavgA,actavgA),ttest_correl_A))
        
        self.scatter_plot(actavgV, predavgV, 'Valence')
        self.scatter_plot(actavgA, predavgA, 'Arousal')
        
        return ResultsV, ResultsA
    
    def opensmileLLD_pipeline_RF_Ext(self, n_factors, batch):

        #openSmile LLD block 1: determine number of factors to use

        ResultsV = []
        ResultsA = []

        Train_data_V = self.opensmile_LLDs_Build_df(self.All_Train_dir)
        Test_data_V = self.opensmile_LLDs_Build_df(batch)
        Train_data_A = Train_data_V
        Test_data_A = Test_data_V
        Train_data_scaled, Test_data_scaled = self.scaled_frames_opensmileLLD(Train_data_V, Test_data_V, self.All_Train_dir, batch)
        Train_YV_Scaled = self.normalized_labels_frames_Ext(self.csv_file,self.All_Train_dir, self.All_Train_dir)[0]
        Train_YA_Scaled = self.normalized_labels_frames_Ext(self.csv_file,self.All_Train_dir, self.All_Train_dir)[1]
        TrainYV_Scaled_frames = self.labels_to_frames_opensmileLLD(Train_YV_Scaled, self.All_Train_dir)
        TrainYA_Scaled_frames = self.labels_to_frames_opensmileLLD(Train_YA_Scaled, self.All_Train_dir)
        numframes_Train = self.numframes_opensmileLLD(self.All_Train_dir)
        numframes_Test = self.numframes_opensmileLLD(batch)

        df_collection_Train = []
        df_collection_Test = []

        for array in Train_data_scaled:
            df = pd.DataFrame(array)
            df_collection_Train.append(df)

        for array in Test_data_scaled:
            df = pd.DataFrame(array)
            df_collection_Test.append(df)

        Train_data_df = pd.concat(df_collection_Train)
        Test_data_df = pd.concat(df_collection_Test)

        Feat_names = pd.read_csv('LLD_Features.csv')
        Featnames = np.array(Feat_names)

        X_frames = pd.concat((Train_data_df, Test_data_df))
        #df = pd.DataFrame(data = X_frames, index = None, columns = Featnames)

        n_factors = n_factors

        if n_factors != 0:
            FATF_data = self.factor_analysis(n_factors, X_frames)
            predictionsV, predictionsA = self.Random_Forest_Infer(FATF_data, TrainYV_Scaled_frames, TrainYA_Scaled_frames, TestYV_Scaled_frames, TestYA_Scaled_frames, Train_data_df, Test_data_df)
            predavgV, actavgV = np.array(self.Avg_over_frames(TestYV_Scaled_frames, predictionsV, numframes_Test))
            predavgA, actavgA = np.array(self.Avg_over_frames(TestYA_Scaled_frames, predictionsA, numframes_Test))
            err_accV = self.model_error_accuracy(predavgV, actavgV)
            err_accA = self.model_error_accuracy(predavgA, actavgA)
            R2_V = self.R2_score(predavgV, actavgV)
            R2_A = self.R2_score(predavgA, actavgA)
            ttest_correl_V = self.ttest_correl(predavgV, actavgV)
            ttest_correl_A = self.ttest_correl(predavgA, actavgA)
            ResultsV.append((err_accV, np.average(np.absolute(np.subtract(predavgV,actavgV))),R2_V,ttest_correl_V))
            ResultsA.append((err_accA, np.average(np.absolute(np.subtract(predavgA,actavgA))),R2_A,ttest_correl_A))
            
            self.scatter_plot(actavgV, predavgV, 'Valence')
            self.scatter_plot(actavgA, predavgA, 'Arousal')
            
        else:
            predictionsV = self.Random_Forest_Infer_opensmileLLD_Ext(Train_data_df, Test_data_df, TrainYV_Scaled_frames, TrainYA_Scaled_frames)[0]
            predictionsA = self.Random_Forest_Infer_opensmileLLD_Ext(Train_data_df, Test_data_df, TrainYV_Scaled_frames, TrainYA_Scaled_frames)[1]
            predavgV = np.array(self.Avg_over_frames_Ext(predictionsV, numframes_Test))
            predavgA = np.array(self.Avg_over_frames_Ext(predictionsA, numframes_Test))
          
            #self.scatter_plot(predavgV, predavgA, 'Predicted Valence vs Predicted Arousal')
            
        return predavgV, predavgA
    
    def opensmileLLD_pipeline_RF_504_56(self, n_factors):

        #openSmile LLD block 1: determine number of factors to use
        predavgV_all = []
        predavgA_all = []
        
        for j in range(1,11):
            test_dir = glob.glob("Feb2022/504_56_{0}/Test/*.wav".format(j)) 
            train_dir = glob.glob("Feb2022/504_56_{0}/Train/*.wav".format(j))

            ResultsV = []
            ResultsA = []

            Train_data_V = self.opensmile_LLDs_Build_df(train_dir)
            Test_data_V = self.opensmile_LLDs_Build_df(test_dir)
            Train_data_A = Train_data_V
            Test_data_A = Test_data_V
            Train_data_scaled, Test_data_scaled = self.scaled_frames_opensmileLLD(Train_data_V, Test_data_V, train_dir, test_dir)
            Train_YV_Scaled = self.normalized_labels_frames_Ext(self.csv_file,train_dir, test_dir)[0]
            Train_YA_Scaled = self.normalized_labels_frames_Ext(self.csv_file,train_dir, train_dir)[1]
            TrainYV_Scaled_frames = self.labels_to_frames_opensmileLLD(Train_YV_Scaled, train_dir)
            TrainYA_Scaled_frames = self.labels_to_frames_opensmileLLD(Train_YA_Scaled, train_dir)
            numframes_Train = self.numframes_opensmileLLD(train_dir)
            numframes_Test = self.numframes_opensmileLLD(test_dir)

            df_collection_Train = []
            df_collection_Test = []

            for array in Train_data_scaled:
                df = pd.DataFrame(array)
                df_collection_Train.append(df)

            for array in Test_data_scaled:
                df = pd.DataFrame(array)
                df_collection_Test.append(df)

            Train_data_df = pd.concat(df_collection_Train)
            Test_data_df = pd.concat(df_collection_Test)

            Feat_names = pd.read_csv('LLD_Features.csv')
            Featnames = np.array(Feat_names)

            X_frames = pd.concat((Train_data_df, Test_data_df))
            #df = pd.DataFrame(data = X_frames, index = None, columns = Featnames)

            n_factors = n_factors

            if n_factors != 0:
                FATF_data = self.factor_analysis(n_factors, X_frames)
                predictionsV, predictionsA = self.Random_Forest_Infer(FATF_data, TrainYV_Scaled_frames, TrainYA_Scaled_frames, TestYV_Scaled_frames, TestYA_Scaled_frames, Train_data_df, Test_data_df)
                predavgV, actavgV = np.array(self.Avg_over_frames(TestYV_Scaled_frames, predictionsV, numframes_Test))
                predavgA, actavgA = np.array(self.Avg_over_frames(TestYA_Scaled_frames, predictionsA, numframes_Test))
                err_accV = self.model_error_accuracy(predavgV, actavgV)
                err_accA = self.model_error_accuracy(predavgA, actavgA)
                R2_V = self.R2_score(predavgV, actavgV)
                R2_A = self.R2_score(predavgA, actavgA)
                ttest_correl_V = self.ttest_correl(predavgV, actavgV)
                ttest_correl_A = self.ttest_correl(predavgA, actavgA)
                ResultsV.append((err_accV, np.average(np.absolute(np.subtract(predavgV,actavgV))),R2_V,ttest_correl_V))
                ResultsA.append((err_accA, np.average(np.absolute(np.subtract(predavgA,actavgA))),R2_A,ttest_correl_A))

                self.scatter_plot(actavgV, predavgV, 'Valence')
                self.scatter_plot(actavgA, predavgA, 'Arousal')

            else:
                predictionsV = self.Random_Forest_Infer_opensmileLLD_Ext(Train_data_df, Test_data_df, TrainYV_Scaled_frames, TrainYA_Scaled_frames)[0]
                predictionsA = self.Random_Forest_Infer_opensmileLLD_Ext(Train_data_df, Test_data_df, TrainYV_Scaled_frames, TrainYA_Scaled_frames)[1]
                predavgV = np.array(self.Avg_over_frames_Ext(predictionsV, numframes_Test))
                predavgA = np.array(self.Avg_over_frames_Ext(predictionsA, numframes_Test))
                predavgV_all.append(predavgV)
                predavgA_all.append(predavgA)

                self.scatter_plot(predavgV, predavgA, 'Predicted Valence vs Predicted Arousal')
            
        return predavgV_all, predavgA_all
    
    def opensmileLLD_pipeline_RF(self, n_factors):

        #openSmile LLD block 1: determine number of factors to use

        ResultsV = []
        ResultsA = []

        Train_data_V = self.opensmile_LLDs_Build_df(self.Train_dir_V)
        Test_data_V = self.opensmile_LLDs_Build_df(self.Test_dir_V)
        Train_data_A = Train_data_V
        Test_data_A = Test_data_V
        Train_data_scaled, Test_data_scaled = self.scaled_frames_opensmileLLD(Train_data_V, Test_data_V, self.Train_dir_V, self.Test_dir_V)
        Train_YV_Scaled = self.normalized_labels_frames(self.csv_file,self.Train_dir_V, self.Test_dir_V, self.Train_dir_A, self.Test_dir_A)[0]
        Test_YV_Scaled = self.normalized_labels_frames(self.csv_file,self.Train_dir_V, self.Test_dir_V, self.Train_dir_A, self.Test_dir_A)[1]
        Train_YA_Scaled = self.normalized_labels_frames(self.csv_file,self.Train_dir_V, self.Test_dir_V, self.Train_dir_A, self.Test_dir_A)[2]
        Test_YA_Scaled = self.normalized_labels_frames(self.csv_file,self.Train_dir_V, self.Test_dir_V, self.Train_dir_A, self.Test_dir_A)[3]
        TrainYV_Scaled_frames = self.labels_to_frames_opensmileLLD(Train_YV_Scaled, self.Train_dir_V)
        TestYV_Scaled_frames = self.labels_to_frames_opensmileLLD(Test_YV_Scaled, self.Test_dir_V)
        TrainYA_Scaled_frames = self.labels_to_frames_opensmileLLD(Train_YA_Scaled, self.Train_dir_A)
        TestYA_Scaled_frames = self.labels_to_frames_opensmileLLD(Test_YA_Scaled, self.Test_dir_A)
        numframes_Train = self.numframes_opensmileLLD(self.Train_dir_V)
        numframes_Test = self.numframes_opensmileLLD(self.Test_dir_V)

        df_collection_Train = []
        df_collection_Test = []

        for array in Train_data_scaled:
            df = pd.DataFrame(array)
            df_collection_Train.append(df)

        for array in Test_data_scaled:
            df = pd.DataFrame(array)
            df_collection_Test.append(df)

        Train_data_df = pd.concat(df_collection_Train)
        Test_data_df = pd.concat(df_collection_Test)

        Feat_names = pd.read_csv('LLD_Features.csv')
        Featnames = np.array(Feat_names)

        X_frames = pd.concat((Train_data_df, Test_data_df))
        #df = pd.DataFrame(data = X_frames, index = None, columns = Featnames)

        n_factors = n_factors

        if n_factors != 0:
            FATF_data = self.factor_analysis(n_factors, X_frames)
            predictionsV, predictionsA = self.Random_Forest_Infer(FATF_data, TrainYV_Scaled_frames, TrainYA_Scaled_frames, TestYV_Scaled_frames, TestYA_Scaled_frames, Train_data_df, Test_data_df)
            predavgV, actavgV = np.array(self.Avg_over_frames(TestYV_Scaled_frames, predictionsV, numframes_Test))
            predavgA, actavgA = np.array(self.Avg_over_frames(TestYA_Scaled_frames, predictionsA, numframes_Test))
            err_accV = self.model_error_accuracy(predavgV, actavgV)
            err_accA = self.model_error_accuracy(predavgA, actavgA)
            R2_V = self.R2_score(predavgV, actavgV)
            R2_A = self.R2_score(predavgA, actavgA)
            ttest_correl_V = self.ttest_correl(predavgV, actavgV)
            ttest_correl_A = self.ttest_correl(predavgA, actavgA)
            ResultsV.append((err_accV, np.average(np.absolute(np.subtract(predavgV,actavgV))),R2_V,ttest_correl_V))
            ResultsA.append((err_accA, np.average(np.absolute(np.subtract(predavgA,actavgA))),R2_A,ttest_correl_A))
            
            self.scatter_plot(actavgV, predavgV, 'Valence')
            self.scatter_plot(actavgA, predavgA, 'Arousal')
            
        else:
            predictionsV = self.Random_Forest_Infer_opensmileLLD(Train_data_df, Test_data_df, TrainYV_Scaled_frames, TrainYA_Scaled_frames, TestYV_Scaled_frames, TestYA_Scaled_frames)[0]
            predictionsA = self.Random_Forest_Infer_opensmileLLD(Train_data_df, Test_data_df, TrainYV_Scaled_frames, TrainYA_Scaled_frames, TestYV_Scaled_frames, TestYA_Scaled_frames)[1]
            predavgV = np.array(self.Avg_over_frames(TestYV_Scaled_frames, predictionsV, numframes_Test)[0])
            actavgV = np.array(self.Avg_over_frames(TestYV_Scaled_frames, predictionsV, numframes_Test)[1])
            predavgA = np.array(self.Avg_over_frames(TestYA_Scaled_frames, predictionsA, numframes_Test)[0])
            actavgA = np.array(self.Avg_over_frames(TestYA_Scaled_frames, predictionsA, numframes_Test)[1])
            err_accV = self.model_error_accuracy(predavgV, actavgV)
            err_accA = self.model_error_accuracy(predavgA, actavgA)
            R2_V = self.R2_score(predavgV, actavgV)
            R2_A = self.R2_score(predavgA, actavgA)
            ttest_correl_V = self.ttest_correl(predavgV, actavgV)
            ttest_correl_A = self.ttest_correl(predavgA, actavgA)
            ResultsV.append((err_accV, np.average(np.absolute(np.subtract(predavgV,actavgV))),R2_V,ttest_correl_V))
            ResultsA.append((err_accA, np.average(np.absolute(np.subtract(predavgA,actavgA))),R2_A,ttest_correl_A))
            
            self.scatter_plot(actavgV, predavgV, 'Valence')
            self.scatter_plot(actavgA, predavgA, 'Arousal')
            
        return ResultsV, ResultsA
    
    def opensmileLLD_pipeline_RF_errorvals(self, n_factors):

        #openSmile LLD block 1: determine number of factors to use

        ResultsV = []
        ResultsA = []
        errorValV = []
        errorValA = []

        Train_data_V = self.opensmile_LLDs_Build_df(self.Train_dir_V)
        Test_data_V = self.opensmile_LLDs_Build_df(self.Test_dir_V)
        Train_data_A = Train_data_V
        Test_data_A = Test_data_V
        Train_data_scaled, Test_data_scaled = self.scaled_frames_opensmileLLD(Train_data_V, Test_data_V, self.Train_dir_V, self.Test_dir_V)
        Train_YV_Scaled = self.normalized_labels_frames(self.csv_file,self.Train_dir_V, self.Test_dir_V, self.Train_dir_A, self.Test_dir_A)[0]
        Test_YV_Scaled = self.normalized_labels_frames(self.csv_file,self.Train_dir_V, self.Test_dir_V, self.Train_dir_A, self.Test_dir_A)[1]
        Train_YA_Scaled = self.normalized_labels_frames(self.csv_file,self.Train_dir_V, self.Test_dir_V, self.Train_dir_A, self.Test_dir_A)[2]
        Test_YA_Scaled = self.normalized_labels_frames(self.csv_file,self.Train_dir_V, self.Test_dir_V, self.Train_dir_A, self.Test_dir_A)[3]
        TrainYV_Scaled_frames = self.labels_to_frames_opensmileLLD(Train_YV_Scaled, self.Train_dir_V)
        TestYV_Scaled_frames = self.labels_to_frames_opensmileLLD(Test_YV_Scaled, self.Test_dir_V)
        TrainYA_Scaled_frames = self.labels_to_frames_opensmileLLD(Train_YA_Scaled, self.Train_dir_A)
        TestYA_Scaled_frames = self.labels_to_frames_opensmileLLD(Test_YA_Scaled, self.Test_dir_A)
        numframes_Train = self.numframes_opensmileLLD(self.Train_dir_V)
        numframes_Test = self.numframes_opensmileLLD(self.Test_dir_V)

        df_collection_Train = []
        df_collection_Test = []

        for array in Train_data_scaled:
            df = pd.DataFrame(array)
            df_collection_Train.append(df)

        for array in Test_data_scaled:
            df = pd.DataFrame(array)
            df_collection_Test.append(df)

        Train_data_df = pd.concat(df_collection_Train)
        Test_data_df = pd.concat(df_collection_Test)

        Feat_names = pd.read_csv('LLD_Features.csv')
        Featnames = np.array(Feat_names)

        X_frames = pd.concat((Train_data_df, Test_data_df))
        #df = pd.DataFrame(data = X_frames, index = None, columns = Featnames)

        n_factors = n_factors

        if n_factors != 0:
            FATF_data = self.factor_analysis(n_factors, X_frames)
            predictionsV, predictionsA = self.Random_Forest_Infer(FATF_data, TrainYV_Scaled_frames, TrainYA_Scaled_frames, TestYV_Scaled_frames, TestYA_Scaled_frames, Train_data_df, Test_data_df)
            predavgV, actavgV = np.array(self.Avg_over_frames(TestYV_Scaled_frames, predictionsV, numframes_Test))
            predavgA, actavgA = np.array(self.Avg_over_frames(TestYA_Scaled_frames, predictionsA, numframes_Test))
            err_accV = self.model_error_accuracy(predavgV, actavgV)
            err_accA = self.model_error_accuracy(predavgA, actavgA)
            R2_V = self.R2_score(predavgV, actavgV)
            R2_A = self.R2_score(predavgA, actavgA)
            ttest_correl_V = self.ttest_correl(predavgV, actavgV)
            ttest_correl_A = self.ttest_correl(predavgA, actavgA)
            ResultsV.append((err_accV, np.average(np.absolute(np.subtract(predavgV,actavgV))),R2_V,ttest_correl_V))
            ResultsA.append((err_accA, np.average(np.absolute(np.subtract(predavgA,actavgA))),R2_A,ttest_correl_A))
            errorValV.append(np.absolute(np.subtract(predavgV,actavgV)))
            errorValA.append(np.absolute(np.subtract(predavgA,actavgA)))
            
            self.scatter_plot(actavgV, predavgV, 'Valence')
            self.scatter_plot(actavgA, predavgA, 'Arousal')
            
        else:
            predictionsV = self.Random_Forest_Infer_opensmileLLD(Train_data_df, Test_data_df, TrainYV_Scaled_frames, TrainYA_Scaled_frames, TestYV_Scaled_frames, TestYA_Scaled_frames)[0]
            predictionsA = self.Random_Forest_Infer_opensmileLLD(Train_data_df, Test_data_df, TrainYV_Scaled_frames, TrainYA_Scaled_frames, TestYV_Scaled_frames, TestYA_Scaled_frames)[1]
            predavgV = np.array(self.Avg_over_frames(TestYV_Scaled_frames, predictionsV, numframes_Test)[0])
            actavgV = np.array(self.Avg_over_frames(TestYV_Scaled_frames, predictionsV, numframes_Test)[1])
            predavgA = np.array(self.Avg_over_frames(TestYA_Scaled_frames, predictionsA, numframes_Test)[0])
            actavgA = np.array(self.Avg_over_frames(TestYA_Scaled_frames, predictionsA, numframes_Test)[1])
            err_accV = self.model_error_accuracy(predavgV, actavgV)
            err_accA = self.model_error_accuracy(predavgA, actavgA)
            R2_V = self.R2_score(predavgV, actavgV)
            R2_A = self.R2_score(predavgA, actavgA)
            ttest_correl_V = self.ttest_correl(predavgV, actavgV)
            ttest_correl_A = self.ttest_correl(predavgA, actavgA)
            ResultsV.append((err_accV, np.average(np.absolute(np.subtract(predavgV,actavgV))),R2_V,ttest_correl_V))
            ResultsA.append((err_accA, np.average(np.absolute(np.subtract(predavgA,actavgA))),R2_A,ttest_correl_A))
            errorValV.append(np.absolute(np.subtract(predavgV,actavgV)))
            errorValA.append(np.absolute(np.subtract(predavgA,actavgA)))
            
            self.scatter_plot(actavgV, predavgV, 'Valence')
            self.scatter_plot(actavgA, predavgA, 'Arousal')
            
        return errorValV, errorValA
    
    def opensmileFunctionals_pipeline_RF(self, n_factors):

        ResultsV = []
        ResultsA = []

        Train_data_V = self.opensmile_Functionals_Build(self.Train_dir_V)
        Test_data_V = self.opensmile_Functionals_Build(self.Test_dir_V)
        Train_data_A = Train_data_V
        Test_data_A = Test_data_V
        Train_data_scaled, Test_data_scaled = self.scaled_frames_opensmileFunc(Train_data_V, Test_data_V, self.Train_dir_V, self.Test_dir_V)
        Train_YV_Scaled = self.normalized_labels_frames(self.csv_file,self.Train_dir_V, self.Test_dir_V, self.Train_dir_A, self.Test_dir_A)[0]
        Test_YV_Scaled = self.normalized_labels_frames(self.csv_file,self.Train_dir_V, self.Test_dir_V, self.Train_dir_A, self.Test_dir_A)[1]
        Train_YA_Scaled = self.normalized_labels_frames(self.csv_file,self.Train_dir_V, self.Test_dir_V, self.Train_dir_A, self.Test_dir_A)[2]
        Test_YA_Scaled = self.normalized_labels_frames(self.csv_file,self.Train_dir_V, self.Test_dir_V, self.Train_dir_A, self.Test_dir_A)[3]
        numframes_Train = self.numframes_opensmileFunc(self.Train_dir_V)
        numframes_Test = self.numframes_opensmileFunc(self.Test_dir_V)

        df_collection_Train = []
        df_collection_Test = []

        for array in Train_data_scaled:
            df = pd.DataFrame(array)
            df_collection_Train.append(df)

        for array in Test_data_scaled:
            df = pd.DataFrame(array)
            df_collection_Test.append(df)

        Train_data_df = pd.concat(df_collection_Train)
        Test_data_df = pd.concat(df_collection_Test)

        Feat_names = pd.read_csv('Features_Func.csv')
        Featnames = np.array(Feat_names)

        X_frames = pd.concat((Train_data_df, Test_data_df))

        n_factors = n_factors

        if n_factors != 0:
            FATF_data = self.factor_analysis(n_factors, X_frames)
            predictionsV, predictionsA = self.Random_Forest_Infer_opensmileFunc(FATF_data, Train_YV_Scaled, Train_YA_Scaled, Test_YV_Scaled, Test_YA_Scaled, Train_data_df, Test_data_df)
            predavgV, actavgV = np.array(self.Avg_over_frames(Test_YV_Scaled, predictionsV, numframes_Test))
            predavgA, actavgA = np.array(self.Avg_over_frames(Test_YA_Scaled, predictionsA, numframes_Test))
            err_accV = self.model_error_accuracy(predavgV, actavgV)
            err_accA = self.model_error_accuracy(predavgA, actavgA)
            R2_V = self.R2_score(predavgV, actavgV)
            R2_A = self.R2_score(predavgA, actavgA)
            ttest_correl_V = self.ttest_correl(predavgV, actavgV)
            ttest_correl_A = self.ttest_correl(predavgA, actavgA)
            ResultsV.append((err_accV, np.average(np.absolute(np.subtract(predavgV,actavgV))),R2_V,ttest_correl_V))
            ResultsA.append((err_accA, np.average(np.absolute(np.subtract(predavgA,actavgA))),R2_A,ttest_correl_A))
            
            self.scatter_plot(actavgV, predavgV, 'Valence')
            self.scatter_plot(actavgA, predavgA, 'Arousal')

        else:
            predictionsV, predictionsA = self.Random_Forest_Infer_opensmileFunc_Feat(Train_data_df, Test_data_df, Train_YV_Scaled, Train_YA_Scaled, Test_YV_Scaled, Test_YA_Scaled)
            predavgV, actavgV = np.array(self.Avg_over_frames(Test_YV_Scaled, predictionsV, numframes_Test))
            predavgA, actavgA = np.array(self.Avg_over_frames(Test_YA_Scaled, predictionsA, numframes_Test))
            err_accV = self.model_error_accuracy(predavgV, actavgV)
            err_accA = self.model_error_accuracy(predavgA, actavgA)
            R2_V = self.R2_score(predavgV, actavgV)
            R2_A = self.R2_score(predavgA, actavgA)
            ttest_correl_V = self.ttest_correl(predavgV, actavgV)
            ttest_correl_A = self.ttest_correl(predavgA, actavgA)
            ResultsV.append((err_accV, np.average(np.absolute(np.subtract(predavgV,actavgV))),R2_V,ttest_correl_V))
            ResultsA.append((err_accA, np.average(np.absolute(np.subtract(predavgA,actavgA))),R2_A,ttest_correl_A))
            
            self.scatter_plot(actavgV, predavgV, 'Valence')
            self.scatter_plot(actavgA, predavgA, 'Arousal')

        return ResultsV, ResultsA
    
    def opensmileFunctionals_pipeline_RF_errorvals(self, n_factors): #fastest pipeline, good accuracy, using for errors per file with filename

        ResultsV = []
        ResultsA = []
        errorValV = []
        errorValA = []
        #Train_Filenames = []
        #Test_Filenames = []
        #errorsVreport = []
        #errorsAreport = []
        
        #for file in self.Test_dir_V:
        #    Test_Filenames.append(os.path.basename(file))
        #for file in self.Train_dir_V:
        #    Train_Filenames.append(os.path.basename(file))
        Train_data_V = self.opensmile_Functionals_Build(self.Train_dir_V)
        Test_data_V = self.opensmile_Functionals_Build(self.Test_dir_V)
        Train_data_A = Train_data_V
        Test_data_A = Test_data_V
        print('Features extracted')
        Train_data_scaled, Test_data_scaled = self.scaled_frames_opensmileFunc(Train_data_V, Test_data_V, self.Train_dir_V, self.Test_dir_V)
        Train_YV_Scaled = self.normalized_labels_frames(self.csv_file,self.Train_dir_V, self.Test_dir_V, self.Train_dir_A, self.Test_dir_A)[0]
        Test_YV_Scaled = self.normalized_labels_frames(self.csv_file,self.Train_dir_V, self.Test_dir_V, self.Train_dir_A, self.Test_dir_A)[1]
        Train_YA_Scaled = self.normalized_labels_frames(self.csv_file,self.Train_dir_V, self.Test_dir_V, self.Train_dir_A, self.Test_dir_A)[2]
        Test_YA_Scaled = self.normalized_labels_frames(self.csv_file,self.Train_dir_V, self.Test_dir_V, self.Train_dir_A, self.Test_dir_A)[3]
        numframes_Train = self.numframes_opensmileFunc(self.Train_dir_V)
        numframes_Test = self.numframes_opensmileFunc(self.Test_dir_V)

        df_collection_Train = []
        df_collection_Test = []

        for array in Train_data_scaled:
            df = pd.DataFrame(array)
            df_collection_Train.append(df)

        for array in Test_data_scaled:
            df = pd.DataFrame(array)
            df_collection_Test.append(df)

        Train_data_df = pd.concat(df_collection_Train)
        Test_data_df = pd.concat(df_collection_Test)

        Feat_names = pd.read_csv('Features_Func.csv')
        Featnames = np.array(Feat_names)

        X_frames = pd.concat((Train_data_df, Test_data_df))

        n_factors = n_factors

        if n_factors != 0:
            FATF_data = self.factor_analysis(n_factors, X_frames)
            print('Factor analysis completed')
            predictionsV, predictionsA = self.Random_Forest_Infer_opensmileFunc(FATF_data, Train_YV_Scaled, Train_YA_Scaled, Test_YV_Scaled, Test_YA_Scaled, Train_data_df, Test_data_df)
            print('Random Forest Inference completed')
            predavgV, actavgV = np.array(self.Avg_over_frames(Test_YV_Scaled, predictionsV, numframes_Test))
            predavgA, actavgA = np.array(self.Avg_over_frames(Test_YA_Scaled, predictionsA, numframes_Test))
            print('Post processing compelted')
            err_accV = self.model_error_accuracy(predavgV, actavgV)
            err_accA = self.model_error_accuracy(predavgA, actavgA)
            R2_V = self.R2_score(predavgV, actavgV)
            R2_A = self.R2_score(predavgA, actavgA)
            ttest_correl_V = self.ttest_correl(predavgV, actavgV)
            ttest_correl_A = self.ttest_correl(predavgA, actavgA)
            ResultsV.append((err_accV, np.average(np.absolute(np.subtract(predavgV,actavgV))),R2_V,ttest_correl_V))
            ResultsA.append((err_accA, np.average(np.absolute(np.subtract(predavgA,actavgA))),R2_A,ttest_correl_A))
            errorValV.append(np.absolute(np.subtract(predavgV,actavgV)))
            errorValA.append(np.absolute(np.subtract(predavgA,actavgA)))
            
            print('compiling error values for Valence and Arousal')
            
            #for i in range(len(Test_Filenames)):
            #        errorsVreport.append((Test_Filenames[i],errorValV[i]))
            #        errorsAreport.append((Test_Filenames[i],errorValA[i]))
            
            self.scatter_plot(actavgV, predavgV, 'Valence')
            self.scatter_plot(actavgA, predavgA, 'Arousal')

        else:
            predictionsV, predictionsA = self.Random_Forest_Infer_opensmileFunc_Feat(Train_data_df, Test_data_df, Train_YV_Scaled, Train_YA_Scaled, Test_YV_Scaled, Test_YA_Scaled)
            predavgV, actavgV = np.array(self.Avg_over_frames(Test_YV_Scaled, predictionsV, numframes_Test))
            predavgA, actavgA = np.array(self.Avg_over_frames(Test_YA_Scaled, predictionsA, numframes_Test))
            err_accV = self.model_error_accuracy(predavgV, actavgV)
            err_accA = self.model_error_accuracy(predavgA, actavgA)
            R2_V = self.R2_score(predavgV, actavgV)
            R2_A = self.R2_score(predavgA, actavgA)
            ttest_correl_V = self.ttest_correl(predavgV, actavgV)
            ttest_correl_A = self.ttest_correl(predavgA, actavgA)
            ResultsV.append((err_accV, np.average(np.absolute(np.subtract(predavgV,actavgV))),R2_V,ttest_correl_V))
            ResultsA.append((err_accA, np.average(np.absolute(np.subtract(predavgA,actavgA))),R2_A,ttest_correl_A))
            errorValV.append(np.absolute(np.subtract(predavgV,actavgV)))
            errorValA.append(np.absolute(np.subtract(predavgA,actavgA)))
            
            self.scatter_plot(actavgV, predavgV, 'Valence')
            self.scatter_plot(actavgA, predavgA, 'Arousal')

        return errorValV, errorValA
    
    def opensmileFunctionals_pipeline_NN(self, n_factors, lr):

        ResultsV = []
        ResultsA = []

        Train_data_V = self.opensmile_Functionals_Build(self.Train_dir_V)
        Test_data_V = self.opensmile_Functionals_Build(self.Test_dir_V)
        Train_data_A = Train_data_V
        Test_data_A = Test_data_V
        Train_data_scaled, Test_data_scaled = self.scaled_frames_opensmileFunc(Train_data_V, Test_data_V, self.Train_dir_V, self.Test_dir_V)
        Train_YV_Scaled = self.normalized_labels_frames(self.csv_file,self.Train_dir_V, self.Test_dir_V, self.Train_dir_A, self.Test_dir_A)[0]
        Test_YV_Scaled = self.normalized_labels_frames(self.csv_file,self.Train_dir_V, self.Test_dir_V, self.Train_dir_A, self.Test_dir_A)[1]
        Train_YA_Scaled = self.normalized_labels_frames(self.csv_file,self.Train_dir_V, self.Test_dir_V, self.Train_dir_A, self.Test_dir_A)[2]
        Test_YA_Scaled = self.normalized_labels_frames(self.csv_file,self.Train_dir_V, self.Test_dir_V, self.Train_dir_A, self.Test_dir_A)[3]
        numframes_Train = self.numframes_opensmileFunc(self.Train_dir_V)
        numframes_Test = self.numframes_opensmileFunc(self.Test_dir_V)

        df_collection_Train = []
        df_collection_Test = []

        for array in Train_data_scaled:
            df = pd.DataFrame(array)
            df_collection_Train.append(df)

        for array in Test_data_scaled:
            df = pd.DataFrame(array)
            df_collection_Test.append(df)

        Train_data_df = pd.concat(df_collection_Train)
        Test_data_df = pd.concat(df_collection_Test)

        Feat_names = pd.read_csv('Features_Func.csv')
        Featnames = np.array(Feat_names)

        X_frames = pd.concat((Train_data_df, Test_data_df))

        n_factors = n_factors
        lr = lr

        if n_factors != 0:
            FATF_data = self.factor_analysis(n_factors, X_frames)
            model = self.NeuralNet_Compile(n_factors,lr)
            predictionsV = self.NeuralNet_Infer(model, n_factors, FATF_data, Train_YV_Scaled, Test_YV_Scaled, Train_data_df, Test_data_df)
            predictionsA = self.NeuralNet_Infer(model, n_factors, FATF_data, Train_YA_Scaled, Test_YA_Scaled, Train_data_df, Test_data_df)
            predavgV, actavgV = np.array(self.Avg_over_frames(Test_YV_Scaled, predictionsV, numframes_Test))
            predavgA, actavgA = np.array(self.Avg_over_frames(Test_YA_Scaled, predictionsA, numframes_Test))
            err_accV = self.model_error_accuracy(predavgV, actavgV)
            err_accA = self.model_error_accuracy(predavgA, actavgA)
            R2_V = self.R2_score(predavgV, actavgV)
            R2_A = self.R2_score(predavgA, actavgA)
            ttest_correl_V = self.ttest_correl(predavgV, actavgV)
            ttest_correl_A = self.ttest_correl(predavgA, actavgA)
            ResultsV.append((err_accV, np.average(np.absolute(np.subtract(predavgV,actavgV))),R2_V,ttest_correl_V))
            ResultsA.append((err_accA, np.average(np.absolute(np.subtract(predavgA,actavgA))),R2_A,ttest_correl_A))
            
            self.scatter_plot(actavgV, predavgV, 'Valence')
            self.scatter_plot(actavgA, predavgA, 'Arousal')

        else:
            model = self.NeuralNet_Compile(88,lr)
            predictionsV = self.NeuralNet_Infer_Func(model, Train_data_df, Test_data_df, Train_YV_Scaled, Test_YV_Scaled)
            predictionsA = self.NeuralNet_Infer_Func(model, Train_data_df, Test_data_df, Train_YA_Scaled, Test_YA_Scaled)
            predavgV, actavgV = np.array(self.Avg_over_frames(Test_YV_Scaled, predictionsV, numframes_Test))
            predavgA, actavgA = np.array(self.Avg_over_frames(Test_YA_Scaled, predictionsA, numframes_Test))
            err_accV = self.model_error_accuracy(predavgV, actavgV)
            err_accA = self.model_error_accuracy(predavgA, actavgA)
            R2_V = self.R2_score(predavgV, actavgV)
            R2_A = self.R2_score(predavgA, actavgA)
            ttest_correl_V = self.ttest_correl(predavgV, actavgV)
            ttest_correl_A = self.ttest_correl(predavgA, actavgA)
            ResultsV.append((err_accV, np.average(np.absolute(np.subtract(predavgV,actavgV))),R2_V,ttest_correl_V))
            ResultsA.append((err_accA, np.average(np.absolute(np.subtract(predavgA,actavgA))),R2_A,ttest_correl_A))
            
            self.scatter_plot(actavgV, predavgV, 'Valence')
            self.scatter_plot(actavgA, predavgA, 'Arousal')

        return ResultsV, ResultsA
    
    def scatter_plot(self, xvals, yvals, title):
            
        plt.scatter(xvals, yvals)
        plt.plot()
        plt.title(title)
        plt.show()
        #plt.savefig(title)
    
    def set_dirs(self):
        
            self.Train_dir_V = glob.glob('Feb2022/TrainV/*.wav')
            self.Test_dir_V = glob.glob('Feb2022/TestV/*.wav')
            self.Train_dir_A = glob.glob('Feb2022/TrainA/*.wav')
            self.Test_dir_A = glob.glob('Feb2022/TestA/*.wav')
    
    def set_files(self): #obsolete

        if self.splitby == 0.7:
            self.Train_dir_V = glob.glob('Nov2021/70_30/TrainV/*.wav')
            self.Test_dir_V = glob.glob('Nov2021/70_30/TestV/*.wav')
            self.Train_dir_A = glob.glob('Nov2021/70_30/TrainA/*.wav')
            self.Test_dir_A = glob.glob('Nov2021/70_30/TestA/*.wav')
            print('directories set for ', self.splitby, ' split')
        if self.splitby == 0.8:
            self.Train_dir_V = glob.glob('Feb2022/80_19/TrainV/*.wav')
            self.Test_dir_V = glob.glob('Feb2022/80_19/TestV/*.wav')
            self.Train_dir_A = glob.glob('Feb2022/80_19/TrainA/*.wav')
            self.Test_dir_A = glob.glob('Feb2022/80_19/TestA/*.wav')
            print('directories set for ', self.splitby, ' split')
        if self.splitby == 0.5:
            self.Train_dir_V = glob.glob('Jan2022/50_50/TrainV/*.wav')
            self.Test_dir_V = glob.glob('Jan2022/50_50/TestV/*.wav')
            self.Train_dir_A = glob.glob('Jan2022/50_50/TrainA/*.wav')
            self.Test_dir_A = glob.glob('Jan2022/50_50/TestA/*.wav')
            print('directories set for ', self.splitby, ' split')
        if self.splitby == 0.9:
            self.Train_dir_V = glob.glob('Feb2022/90_9/TrainV/*.wav')
            self.Test_dir_V = glob.glob('Feb2022/90_9/TestV/*.wav')
            self.Train_dir_A = glob.glob('Feb2022/90_9/TrainA/*.wav')
            self.Test_dir_A = glob.glob('Feb2022/90_9/TestA/*.wav')
            print('directories set for ', self.splitby, ' split')
        if self.splitby == 0.95:
            self.Train_dir_V = glob.glob('Jan2022/95_5/TrainV/*.wav')
            self.Test_dir_V = glob.glob('Jan2022/95_5/TestV/*.wav')
            self.Train_dir_A = glob.glob('Jan2022/95_5/TrainA/*.wav')
            self.Test_dir_A = glob.glob('Jan2022/95_5/TestA/*.wav')
            print('directories set for ', self.splitby, ' split')
            
if __name__ == '__main__':
    #RUN RF LLD model to Train on all 560 NLUs and Predict on AudioSet data to produce predictions and V vs A scatter plots
    main_dir = glob.glob('Mono/*.wav')
    split = 1
    InferModel = Model_VA(main_dir,split)
    n_factors_LLD = 0
    batch_size = 1000
    batches = int(len(InferModel.Ext_dir)/batch_size)
    
    for i in range(61,batches):
        filenames_test_data = []
        batch = InferModel.Ext_dir[i*batch_size:(i+1)*batch_size]
        for filepath in batch:
            filenames_test_data.append(filepath)

        pred_files = []
        print("running RF model on batch {0}...".format(i))
        predV, predA = InferModel.opensmileLLD_pipeline_RF_Ext(n_factors_LLD, batch)
            
        for j in range(len(filenames_test_data)):
            pred_files.append((filenames_test_data[j],predV[j],predA[j]))

        with open("midiVApreds.csv", 'a') as g:
            np.savetxt(g, pred_files, delimiter=",", fmt='%s')
