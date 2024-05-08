using FileIO;
using DelimitedFiles;
using Statistics;
using Flux;
using Flux.Losses;

using Random
using Random:seed!


using ScikitLearn: @sk_import, fit!, predict

@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier


# Importamos los datos
dataset = readdlm("datasets/breastCancer/wdbc.data", ',')
dataset
include("58002749S_35591688Q_58024091J.jl")




# --------------------------------- PREPARACIÓN DE LOS DATOS. --------------------------------- #
inputs = convert(Matrix{Float64}, dataset[:, 3:32]);
targets =  dataset[:, 2]

# CONVERTIMOS CATEGÓRICAS A BOOLEANOS.
targets = oneHotEncoding(targets, ["M", "B"])[:,1];


# NORMALIZAMOS LAS VARIABLES CONTÍNUAS.
# HAY QUE DECIDIR SI ES MEJOR ZEROMEAN O MINMAX. DE PRIMERAS MINMAX
normalizeZeroMean!(inputs)

mean(inputs[:,4])
std(inputs[:,4])



# ------------------------------------ CREACIÓN DE LOS MODELOS. ------------------------------------ #
# EMPLEAREMOS CROSSVALIDATION CON VARIOS MODELOS DISTINTOS.
# CREAMOS HYPERPARÁMETROS.

indicesCross = crossvalidation(targets, 10) # creamos 10 k grupos en crossvalidation
    # En esta parte toca probar con distintos valores para los hiperparámetros.
modelHyperparameters = Dict("topology" => [5,3], "learningRate" => 0.01,"validationRatio" => 0.2, "numExecutions" => 50, "maxEpochs" => 1000,"maxEpochsVal" => 6);


ANN = modelCrossValidation(:ANN, modelHyperparameters, inputs, targets, indicesCross)





# ------------------------------------ EVALUACIÓN DE LOS MODELOS. ------------------------------------ #

# SVM Y ÁRBOLES DE DECISIÓN.






