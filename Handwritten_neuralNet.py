import numpy
import scipy.special
import csv
class NeuralNetwork:   
	    
	    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):  
	             
	        self.inodes = inputnodes   
	        self.hnodes = hiddennodes     
	        self.onodes = outputnodes    
	        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
	        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)) 
	        self.lr = learningrate       
	        self.activation_function = lambda x: scipy.special.expit(x)     
	    pass 

	    def train(self, inputs_list, targets_list):

	        inputs = numpy.array(inputs_list, ndmin=2).T 
	        targets = numpy.array(targets_list, ndmin=2).T  
	        hidden_inputs = numpy.dot(self.wih, inputs)    
	        hidden_outputs = self.activation_function(hidden_inputs)    
	        final_inputs = numpy.dot(self.who, hidden_outputs) 
	        final_outputs = self.activation_function(final_inputs) 

	        output_errors = targets - final_outputs    
	        hidden_errors = numpy.dot(self.who.T, output_errors)    
	        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))    
	        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))   
	    pass
	    
	    def query(self, inputs_list):
	   
	        inputs = numpy.array(inputs_list, ndmin=2).T     
	        hidden_inputs = numpy.dot(self.wih, inputs)    
	        hidden_outputs = self.activation_function(hidden_inputs)   
	        final_inputs = numpy.dot(self.who, hidden_outputs)  
	        final_outputs = self.activation_function(final_inputs)
	        return final_outputs


input_nodes = 784 
hidden_nodes = 200
output_nodes = 52
learning_rate = 0.1
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
data_file = open("DataSets/train.csv", 'r')
data_list = data_file.readlines()
data_file.close()

epochs = 5

for e in range(epochs):
	for record in data_list:
			all_values = record.split(',')
			inputs = (numpy.asfarray(all_values[1:])/255 * 0.99) + 0.01
			targets = numpy.zeros(output_nodes) + 0.01
			targets[int(all_values[0])] = 0.99
			n.train(inputs, targets)
	
pass		
	
test_data_file = open("DataSets/test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

score = []

for record in test_data_list:
	  all_values = record.split(',')
	  inputs = (numpy.asfarray(all_values[1:])/255 * 0.99) + 0.01
	  correct_label = int(all_values[0])
	  outputs = n.query(inputs)
	  label = numpy.argmax(outputs)
	  if (label == correct_label):
	    score.append(1)
	  else:
	    score.append(0)
	    pass
pass

score_array = numpy.asarray(score)
print ("Performance = ",(score_array.sum()/score_array.size) * 100)



