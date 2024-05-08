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


# Importar los datos
dataset = readdlm("wdbc.data", ',')
dataset
include("funciones.jl")


# Separar los datos
inputs = dataset[:, 3:30];
targets = dataset[:, 2];

# Indices de crossvalidation
indices = crossvalidation(targets, 10);

# Convertir inputs a un formato adecuado para trabajar con ellos
inputs2 = convert(Matrix{Float32}, inputs)

# Aplicar oneHotEncoding sobre los targets
targets = oneHotEncoding(targets, ["M", "B"]);

# Normalizar los datos
parametros = calculateZeroMeanNormalizationParameters(inputs2);
normalizado = normalizeZeroMean(inputs2, parametros)

# Crear distintos modelos y comparar resultados
modelHyperparameters = Dict("topology" => [5,3], "learningRate" => 0.01,"validationRatio" => 0.2, "numExecutions" => 50, "maxEpochs" => 1000,"maxEpochsVal" => 6);

ANN = modelCrossValidation(:ANN, modelHyperparameters, inputs2, targets, indices)