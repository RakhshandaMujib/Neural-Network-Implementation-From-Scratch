#Import the following modules: 
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import statistics as stats


'''********************* MODEL TRAINING *********************'''


class Nuron:
    
    alpha = 0.01 #Learning rate.
    mu = 0.5 #Momenutum of learning.
    
    def __init__(self, output_lines, my_index):
        self.output_lines = output_lines
        self.my_index = my_index 
        self.connection = list() #list of tuples of the form: (weights, delta weights)
        self.activation = float() #Output activation of the neuron.
        self.gradient = float()

        for i in range(output_lines):
            weight = np.random.randint(-np.sqrt(len(train)), np.sqrt(len(train)))
            self.connection.append([weight, 0]) #Let the delta weigth be 0 for now. 
        print(self.connection)
    
    def sigmoid_activation(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def calc_activation(self, prev_layer):
        weighted_sum = 0.0
        for i in range(len(prev_layer)):
            weighted_sum += prev_layer[i].activation * prev_layer[i].connection[self.my_index][0]
        self.activation = self.sigmoid_activation(weighted_sum)
        
    def calculate_output_gradient(self, target_value):
        a = self.activation
        delta = target_value - a
        self.gradient = delta * self.sigmoid_derivative(a) #(t-a)a(1-a)
    
    def sumDOW(self, next_layer):
        sum_is = 0.0
        
        #Add up all the output weights going from the current nuron
        #to the nurons in the next layer times the gradient of the 
        #corresponding nurons in the next layer.
        for i in range(len(next_layer)):
            sum_is += self.connection[i][0] * next_layer[i].gradient
        return sum_is #sum(w(t-a)a(1-a))
            
        
    def calculate_hidden_gradient(self, next_layer):
        dow = self.sumDOW(next_layer)
        #sum(w(t-a)a(1-a))* aj(1-aj) =
        self.gradient = dow * self.sigmoid_derivative(self.activation)
        
   
    def update_input_weights(self, previous_layer):
        #The weights to be updated are stored in the connection list of tuples
        #in the neurons in the preceding layer.
        for i in range(len(previous_layer)):
            nuron = previous_layer[i]
            old_del_weight = nuron.connection[self.my_index][1]
            new_del_weight = (self.mu * old_del_weight) + (self.alpha * self.activation * self.gradient) 
            nuron.connection[self.my_index][0] += new_del_weight
            nuron.connection[self.my_index][1] = new_del_weight


class NeuralNetwork:
    
    layer_info = list() #2-D list containg required number of nurons in 
                        #each layer.
    total_error = float() #Error information after prediction. 
    
    
    def __init__(self, topology):
        '''
        topology: List of positive integers where the index number
                represents the layer number and the value at a 
                particular index determines how many neurons are 
                present in that layer. 
        '''
        self.topology = topology    
        self.num_layers = len(topology) #Get the total number of layers
        
        #For each layer, create an empty list to occupy the nurons
        #of that layer. 
        for i in range(self.num_layers):
            self.layer_info.append(list()) 
            
            #Define the number of output paths of a neuron in the ith layer.
            #It is equal to the number of nurons in the successive layer
            #for every layer that is not the output layer. 
            output_lines = 0 if i == self.num_layers - 1 else self.topology[i + 1]
            
            #Now, fill in the empty list for each layer with the neurons: 
            for j in range(self.topology[i]):
                     
                #Add the jth nuron to the ith layer:
                self.layer_info[i].append(Nuron(output_lines, j))
                
                #If we're adding the 0th nuron to the input layer,
                #we call it the "bias", else
                #we label it with the (j + 1)th number itself.
                nuron_number = "bias" if not i and not j else j + 1  
                
                layer_num = i
                if not i: #If the layer number is 0, 
                    layer_num = "input" #we call it the input layer.
                elif i == self.num_layers - 1: #If the layer is the last layer,
                    layer_num = "output" #we call it the output layer.
                
                #Print the success message:
                print(f"Adding {nuron_number} neuron to {layer_num} layer...")
              
            
    def feedForward(self, input_values):
        '''
        input_values: List of the data of one training instance.
        '''
        #Initialize the neurons of layer 0 or the input layer with the
        #input values.
        for i in range(len(input_values)):
            self.layer_info[0][i].activation = input_values[i]
            
            
        #Forward propagation begins...
        for i in range(1, self.num_layers): #From the 1st hidden layer...
                
            #Store the information of the previous layer: 
            previous_layer = self.layer_info[i - 1]
                
            #For each neuron in the ith hidden layer...
            for neuron in range(self.topology[i]):
                   
                #Use the knowledge of the previous layer to perform
                #forward propagation. 
                self.layer_info[i][neuron].calc_activation(previous_layer)

    
    def backPropagation(self, target_values):
        '''
        target_values: List of results in the training dataset. 
        '''
        
        #Initialize the error:
        error = 0.0
        
        #Get the output layer (a layer of nurons):
        output_layer = self.layer_info[self.num_layers - 1]
        
        
        #Calculate the total error:
        for i in range(len(output_layer)):
            error += pow((target_values[i] - output_layer[i].activation), 2)
        error /= 2
        self.total_error += error
        
        #Calculate the output gradient:
        for i in range(len(output_layer)): 
            output_layer[i].calculate_output_gradient(target_values[i])
            
            
        #Calculate the hidden-layer (from the last hidden-layer to the input
        #layer) gradients:
        last_hidden_layer_index = self.num_layers - 2
        for i in range(last_hidden_layer_index, 0, -1):
            hidden_layer = self.layer_info[i]
            next_layer = self.layer_info[i + 1]
            
            #For each neuron in the hidden layer, calculate its gradient.
            for nuron in hidden_layer:
                nuron.calculate_hidden_gradient(next_layer)
                
        #Update the connection weights backward:
        for i in range(1, self.num_layers): #No input weights for the input layer. 
            current_layer = self.layer_info[i]
            previous_layer = self.layer_info[i - 1]
            
            #For each neuron in the current layer, update its weight. 
            for nuron in current_layer:
                nuron.update_input_weights(previous_layer)
                
    
    def prediction(self):
        result = list()
        output_layer = self.layer_info[self.num_layers - 1]
        for i in range(len(output_layer)):
            result.append(output_layer[i].activation)
        return result


def main():
    '''
    Driver function:
    Creates the topology required depending on the data. 
    Designs and trains the neural network and computes the accuracy. 
    '''
    
    '''******************** TRAINING THE MODEL ********************'''
    
    training_pass = 0
    cols = len(train.columns) - 1
    input_values = list()
    target_values = list()
    result = list()
    
    #Get the topology of the network:
    topology = list()
    while(cols):
        topology.append(cols)
        cols //= 2
    
    #Create the neural network object:
    my_ann = NeuralNetwork(topology)
    print(f"\nTopology for the network: {topology}")
    
    #tolerance_value = 0.01
    epoch = 1
    while(epoch < 6): #or my_ann.total_error < tolerance_value):
        
        print(f"***************** EPOCH {epoch} *****************")
        
        for i in range(len(train)):
            training_pass += 1
            print(f"\nTraining pass: {training_pass}")
        
            #Get the inputs for one training instance:
            input_values = train.iloc[i][: -1].values
            print(f"Inputs: {input_values}")
            my_ann.feedForward(input_values) #Feed it to the network.
        
            #Get the target for the particular training instance:
            target_values = list()
            target_values.append(train.iloc[i][-1])
            print(f"Traget: {target_values}")
        
            #Collect the result of the forward pass:
            result = my_ann.prediction()
            print(f"Predicted output: {result}")   
        
            #Backward pass:
            my_ann.backPropagation(target_values)
        
            #Print the average error so far:
            print(f"Average error: {my_ann.total_error / training_pass}")
        
        #Reset the following values:
        my_ann.total_error = 0
        epoch += 1
        training_pass = 0
    print("Training completed!")
    
    '''******************** TESTING THE MODEL ********************'''
    data_num = 0
    prediction = list()
    
    print(f"***************** TESTING BEGINS *****************")
    print(f"\n\nTotal number of test samples: {len(test_data)}")
    
    for i in range(len(test_data)):
        data_num += 1
        print(f"Testing data {data_num}...")
        
        #Feed the test data:
        input_values = test_data.iloc[i][: -1].values
        my_ann.feedForward(input_values)
        
        #Get the prediction:
        prediction.append(round(my_ann.prediction()[0]))
        
    #Get the accuracy:
    correctly_classified = 0
    for i in range(len(test_data)):
        if test_data.iloc[i][-1] == prediction[i]:
            correctly_classified += 1
    print(f"Total accuracy = {correctly_classified / len(test_data) * 100}")