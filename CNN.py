import scipy.special
import numpy
import json
import os

class NeuralNetwork:
    def __init__(self, inputNodes = None, hiddenNodes = None, outputNodes = None, learningRate = None, fromJson = None):
        if fromJson:
            self.importFromJSON(fromJson)
            return
        
        if not inputNodes or not hiddenNodes or not outputNodes or not learningRate:
            raise RuntimeError('\n\n\n > INVALID PARAMETERS. CORRECT USAGE IS AS FOLLOWS:\nn = NeuralNetwork([int], [int], [int], [float])\nor\nn = NeuralNetwork(jsonFile=[json file path])\n')
        
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes

        self.weightsInputToHidden = numpy.random.normal(
            0.0,
            pow(self.hiddenNodes, -0.5),
            (self.hiddenNodes, self.inputNodes)
        )
        self.weightsHiddenToOutput = numpy.random.normal(
            0.0,
            pow(self.outputNodes, -0.5),
            (self.outputNodes, self.hiddenNodes)
        )

        self.learningRate = learningRate
    

    def activationFunction(self, x):
        return scipy.special.expit(x) # sigmoid function
    

    def train(self, inputList, targetList):
        inputs = numpy.array(inputList, ndmin=2).T
        targets = numpy.array(targetList, ndmin=2).T

        hiddenInputs = numpy.dot(self.weightsInputToHidden, inputs)
        hiddenOutputs = self.activationFunction(hiddenInputs)

        finalInputs = numpy.dot(self.weightsHiddenToOutput, hiddenOutputs)
        finalOutputs = self.activationFunction(finalInputs)

        outputErrors = targets - finalOutputs
        hiddenErrors = numpy.dot(self.weightsHiddenToOutput.T, outputErrors)

        self.weightsHiddenToOutput += self.learningRate * numpy.dot((outputErrors * finalOutputs * (1.0 - finalOutputs)), numpy.transpose(hiddenOutputs))
        self.weightsInputToHidden += self.learningRate * numpy.dot((hiddenErrors * hiddenOutputs * (1.0 - hiddenOutputs)), numpy.transpose(inputs))


    def query(self, inputs):
        inputs = numpy.array(inputs, ndmin=2).T

        hidden_inputs = numpy.dot(self.weightsInputToHidden, inputs)
        hidden_outputs = self.activationFunction(hidden_inputs)

        final_inputs = numpy.dot(self.weightsHiddenToOutput, hidden_outputs)
        final_outputs = self.activationFunction(final_inputs)

        return final_outputs


    def importFromJSON(self, jsonFile:str):
        if not jsonFile.endswith('.json'):
            jsonFile += '.json'
        
        try:
            with open(jsonFile, "r") as read_file:
                # json.load() returns a python dictionary
                data = json.load(read_file)
        except Exception as e:
            raise FileExistsError(f"\n\n > FILE '{jsonFile}' COULDN'T BE LOADED.\nMAKE SURE THAT THE FILE EXISTS AND IS PROPERLY FORMATTED.\nIF THE FILE EXISTS AND IS PROPERLY FORMATTED, CHECK FOR TYPOS OR TRY TYPING THE FULL PATH.\n\nFULL EXCEPTION:\n\n{e}")
        
        self.inputNodes = data['inputNodes']
        self.hiddenNodes = data['hiddenNodes']
        self.outputNodes = data['outputNodes']

        self.weightsInputToHidden = numpy.asarray(data['weightsInputToHidden'])
        self.weightsHiddenToOutput = numpy.asarray(data['weightsHiddenToOutput'])

        self.learningRate = data['learningRate']

    
    def export(self, outputName = 'model.json'):
        if not os.path.exists('models'):
            os.makedirs('models')
        
        vals = {
            'inputNodes': self.inputNodes,
            'hiddenNodes': self.hiddenNodes,
            'outputNodes': self.outputNodes,
            'learningRate': self.learningRate,
            'weightsInputToHidden': self.weightsInputToHidden.tolist(),
            'weightsHiddenToOutput': self.weightsHiddenToOutput.tolist()
        }

        if not outputName.endswith('json'):
            outputName += '.json'
        
        outputPath = f"models/{outputName}"

        c = 0
        while os.path.exists(outputPath):
            outputName = outputName[:len(outputName) - 5] # gets rid of the '.json' substring
            c += 1
            outputPath = f"models/{outputName}({c}).json"
        
        with open(outputPath, "w") as outfile: 
            json.dump(vals, outfile, indent = 4)
