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
			if all_values[0] == 'A' or 'a':
				all_values[0] = 1
			elif all_values[0] == 'B' or 'b':
				all_values[0] = 2
			elif all_values[0] == 'C' or 'c':
				all_values[0] = 3
			elif all_values[0] == 'D' or 'd':
				all_values[0] = 4
			elif all_values[0] == 'E' or 'e':
				all_values[0] = 5					
            
            #elif all_vlaues[0] == 'F':
            	#all_values = 6
			elif all_values[0] == 'G' or 'g':
				all_values[0] = 7
			elif all_values[0] == 'H' or 'h':
				all_values[0] = 8		
			elif all_values[0] == 'I' or 'i':
				all_values[0] = 9		
			elif all_values[0] == 'J' or 'j':
				all_values[0] = 10		
			elif all_values[0] == 'K' or 'k':
				all_values[0] = 11		
			elif all_values[0] == 'L' or 'l':
				all_values[0] = 12		
			elif all_values[0] == 'M' or 'm':
				all_values[0] = 13		
			elif all_values[0] == 'N' or 'n':
				all_values[0] = 14		
			elif all_values[0] == 'O' or 'o':
				all_values[0] = 15
			elif all_values[0] == 'P' or 'p':
				all_values[0] = 16		
			elif all_values[0] == 'Q' or 'q':
				all_values[0] = 17		
			elif all_values[0] == 'R' or 'r':
				all_values[0] = 18		
			elif all_values[0] == 'S' or 's':
				all_values[0] = 19
			elif all_values[0] == 'T' or 't':
				all_values[0] = 20				
			elif all_values[0] == 'U' or 'u':
				all_values[0] = 21
			elif all_values[0] == 'V' or 'v':
				all_values[0] = 22		
			elif all_values[0] == 'W' or 'w':
				all_values[0] = 23
			elif all_values[0] == 'X' or 'x':
				all_values[0] = 24		
			elif all_values[0] == 'Y' or 'y':
				all_values[0] = 25
			elif all_values[0] == 'Z' or 'z':
				all_values[0] = 26
			elif all_values[0] == 'F' or 'f':
				all_values[0] = 6	
			
			
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



