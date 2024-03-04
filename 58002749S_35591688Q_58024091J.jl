# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 2 ---------------------------------------------
# ----------------------------------------------------------------------------------------------

using Statistics
using Flux
using Flux.Losses


# -------------------------------------------------------------------------
# Funciones para codificar entradas y salidas categóricas

# Funcion para realizar la codificacion, recibe el vector de caracteristicas (uno por patron), y las clases
function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    numClasses = length(classes);
    @assert(numClasses>1);
    if numClasses==2
        # Si solo hay dos clases, se devuelve una matriz con una columna
        targets = reshape(feature.==classes[1], :, 1);
    else
        # Si hay mas de dos clases se devuelve una matriz con una columna por clase
        # Cualquiera de estos dos tipos (Array{Bool,2} o BitArray{2}) vale perfectamente
        oneHot = Array{Bool,2}(undef, length(feature), numClasses);
        # oneHot =   BitArray{2}(undef, length(targets), numClasses);
        for numClass = 1:numClasses
            oneHot[:,numClass] .= (feature.==classes[numClass]);
        end;
        targets = oneHot;
    end
    return targets
end

# Esta función es similar a la anterior, pero si no se especifican las clases, se toman de la propia variable
function oneHotEncoding(feature::AbstractArray{<:Any,1})
    classes = unique(feature);
    return oneHotEncoding(feature, classes)
end

#Sin palabra function 
#oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature))

# Sobrecargamos la función oneHotEncoding por si acaso se pasa un vector de valores booleanos
# En este caso, el propio vector ya está codificado, simplemente lo convertimos a una matriz columna
function oneHotEncoding(feature::AbstractArray{Bool,1})
    return reshape(feature, :, 1)
end

# Cuando se llame a la funcion oneHotEncoding, según el tipo del argumento pasado, Julia realizará
#  la llamada a la función correspondiente


# -------------------------------------------------------------------------
# Funciones para calcular los parametros de normalizacion y normalizar

# Para calcular los parametros de normalizacion, segun la forma de normalizar que se desee:
# Para calcular los parámetros de normalización min-max
function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    minValues = minimum(dataset, dims=1)
    maxValues = maximum(dataset, dims=1)
    return minValues, maxValues
end

# Para calcular los parámetros de normalización zero-mean
function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    avgValues = mean(dataset, dims=1)
    stdValues = std(dataset, dims=1)
    return avgValues, stdValues
end


# 4 versiones de la funcion para normalizar entre 0 y 1:

# Nos dan los parámetros de normalización y se quiere modificar el array de entradas (el nombre de la función acaba en '!')
function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    minValues, maxValues = normalizationParameters
    dataset .-= minValues
    dataset ./= (maxValues .- minValues)
end

# No nos dan los parámetros de normalización y se quiere modificar el array de entradas (el nombre de la función acaba en '!')
function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    minValues, maxValues = calculateMinMaxNormalizationParameters(dataset)
    dataset .-= minValues
    dataset ./= (maxValues .- minValues)
end

# Nos dan los parámetros de normalización y no se quiere modificar el array de entradas (se crea uno nuevo)
function normalizeMinMax(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    minValues, maxValues = normalizationParameters
    normalized_dataset = (dataset .- minValues) ./ (maxValues .- minValues)
    return normalized_dataset
end

# No nos dan los parámetros de normalización y no se quiere modificar el array de entradas (se crea uno nuevo)
function normalizeMinMax(dataset::AbstractArray{<:Real,2})
    minValues, maxValues = calculateMinMaxNormalizationParameters(dataset)
    normalized_dataset = (dataset .- minValues) ./ (maxValues .- minValues)
    return normalized_dataset
end


# 4 versiones similares de la funcion para normalizar de media 0:
#  - Nos dan los parametros de normalizacion, y se quiere modificar el array de entradas (el nombre de la funcion acaba en '!')
function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    avgValues, stdValues = normalizationParameters
    dataset .-= avgValues;
    dataset ./= stdValues;
end;
#  - No nos dan los parametros de normalizacion, y se quiere modificar el array de entradas (el nombre de la funcion acaba en '!')
function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    avgValues, stdValues = calculateZeroMeanNormalizationParameters(dataset)
    dataset .-= avgValues;
    dataset ./= stdValues;
end;

#  - Nos dan los parametros de normalizacion, y no se quiere modificar el array de entradas (se crea uno nuevo)
function normalizeZeroMean( dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    avgValues, stdValues = normalizationParameters
    normalized_dataset = (dataset .- avgValues) ./ stdValues
    return normalized_dataset
end;

#  - No nos dan los parametros de normalizacion, y no se quiere modificar el array de entradas (se crea uno nuevo)

function normalizeZeroMean( dataset::AbstractArray{<:Real,2})
    avgValues, stdValues = calculateZeroMeanNormalizationParameters(dataset)
    normalized_dataset = (dataset .- avgValues) ./ stdValues
    return normalized_dataset
end;


# -------------------------------------------------------
# Funcion que permite transformar una matriz de valores reales con las salidas del clasificador o clasificadores en una matriz de valores booleanos con la clase en la que sera clasificada

# Función para transformar un vector de valores reales en una matriz de valores booleanos con la clase en la que será clasificada

function classifyOutputs(outputs::AbstractArray{<:Real,1}; threshold::Real=0.5)
    classified_outputs = outputs .>= threshold
    return classified_outputs
end

# Función para transformar una matriz de valores reales en una matriz de valores booleanos que indique la clase a la que se clasifica cada patrón
function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    if size(outputs, 2) == 1 #si tiene 1 columna
        vector_outputs = outputs[:]
        classified_outputs = classifyOutputs(vector_outputs, threshold=threshold)
        return reshape(classified_outputs, :, 1)
    else 
        (_,indicesMaxEachInstance) = findmax(outputs, dims=2);
        outputs = falses(size(outputs));
        outputs[indicesMaxEachInstance] .= true;
        return outputs
    end
end


# -------------------------------------------------------
# Función para calcular la precisión para un vector de salidas binarias y un vector de objetivos binarios
function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    return sum(outputs .== targets) / length(targets)
end

# Función para calcular la precisión para matrices de salidas binarias y objetivos binarios

# MIS CAMBIOS ANTÓN.
function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    if size(outputs,2) == 1
        return accuracy(outputs[:], targets[:])
    else
        classComparison = targets .== outputs
        correctClassifications = all(classComparison, dims=2)
        accuracy3 = mean(correctClassifications)
        return accuracy3
    end
end

# Función para calcular la precisión para un vector de salidas reales y un vector de objetivos binarios
function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    outputs_bool = classifyOutputs(outputs, threshold=threshold)  
    return accuracy(outputs_bool, targets)
end

# Función para calcular la precisión para matrices de salidas reales y objetivos binarios
function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)

    if size(output,2) == 1
        vector_outputs = outputs[:]
        vector_targets = outputs[:]

        return accuracy(vector_outputs, vector_targets, threshold)

    
    else

        outputs_bool = classifyOutputs(outputs, threshold=threshold)  
        return accuracy(outputs_bool, targets)
    end
end
# -------------------------------------------------------

# Funciones para crear y entrenar una RNA

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    ann = Chain()
    numInputsLayer = numInputs

    for numOutputsLayer in topology
        transferFunction = pop!(transferFunctions) 
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, transferFunction))
        numInputsLayer = numOutputsLayer
    end
    
    # Agregar la última capa de salida
    if numOutputs == 2
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, σ))
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity))
        ann = Chain(ann...,softmax)
    end

    return ann
end



function trainClassANN(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    # Crear RNA
    ann = buildClassANN(size(dataset[1], 1), topology, size(dataset[2], 1))
    losses = Float64[]
    # Obtener número de entradas y salidas
    n_inputs = size(dataset[1], 1)
    n_outputs = size(dataset[2], 1)
    # Entrenamiento
    for epoch in 1:maxEpochs
        # Entrenar un ciclo
        Flux.train!(loss, Flux.params(ann), zip(eachrow(dataset[1]), eachrow(dataset[2])), ADAM(learningRate))
        # Calcular la pérdida en esta época
        loss_value = Flux.Losses.mse(ann(dataset[1]'), dataset[2]')
        # Almacenar el valor de pérdida
        push!(losses, loss_value)
        # Criterio de parada
        if loss_value <= minLoss
            break
        end
    end
    return ann, losses
end
 
function trainClassANN(topology::AbstractArray{<:Int,1}, (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    # Convertir vector de salidas en matriz
    dataset_mat = (inputs, reshape(targets, (length(targets), 1)))
    return trainClassANN(topology, dataset_mat; transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate)
end

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 3 ---------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random

function holdOut(N::Int, P::Real)
     indices = randperm(N)

     num_test = round(Int, N * P)
     
     test_indices = indices[1:num_test]
     train_indices = indices[num_test+1:end]
     
     return (train_indices, test_indices)
end;


function holdOut(N::Int, Pval::Real, Ptest::Real)
    indices, test_indices = holdOut(N, Ptest)
    train_indices, val_indices = holdOut(round(Int, (1 - Ptest) * (N - length(test_indices))), Pval / (1 - Ptest))
    
    return train_indices, val_indices, test_indices
end



# Funcion para entrenar RR.NN.AA. con conjuntos de entrenamiento, validacion y test. Estos dos ultimos son opcionales
# Es la funcion anterior, modificada para calcular errores en los conjuntos de validacion y test y realizar parada temprana si es necesario
function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)
    #
    # Codigo a desarrollar
    #
end;


function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)
    #
    # Codigo a desarrollar
    #
end;

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 4 ---------------------------------------------
# ----------------------------------------------------------------------------------------------


function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    TP = sum(outputs .& targets)  # Verdaderos positivos
    TN = sum((.!outputs) .& (.!targets))  # Verdaderos negativos
    FP = sum(outputs .& (.!targets))  # Falsos positivos
    FN = sum((.!outputs) .& targets)  # Falsos negativos

    accuracy = (TP + TN) / (TP + TN + FP + FN)  # Precisión
    error_rate = 1 - accuracy  # Tasa de error
    sensitivity = TP / (TP + FN)  # Sensibilidad
    specificity = TN / (TN + FP)  # Especificidad

    return accuracy, error_rate, sensitivity, specificity
end;

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    # Convertir las salidas a valores booleanos según el umbral
    outputs_bool = outputs .≥ threshold

    # Calcular las métricas de clasificación
    TP = sum(outputs_bool .& targets)  # Verdaderos positivos
    TN = sum((.!outputs_bool) .& (.!targets))  # Verdaderos negativos
    FP = sum(outputs_bool .& (.!targets))  # Falsos positivos
    FN = sum((.!outputs_bool) .& targets)  # Falsos negativos

    accuracy = (TP + TN) / (TP + TN + FP + FN)  # Precisión
    error_rate = 1 - accuracy  # Tasa de error
    sensitivity = TP / (TP + FN)  # Sensibilidad
    specificity = TN / (TN + FP)  # Especificidad

    return accuracy, error_rate, sensitivity, specificity
end;

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    num_classes = size(outputs, 2)  # Número de clases
    num_samples = size(outputs, 1)  # Número de muestras

    # Inicializar las métricas de clasificación para cada clase
    TP = zeros(Int, num_classes)
    TN = zeros(Int, num_classes)
    FP = zeros(Int, num_classes)
    FN = zeros(Int, num_classes)

    # Calcular las métricas de clasificación para cada clase
    for c in 1:num_classes
        for s in 1:num_samples
            if outputs[s, c] && targets[s, c]
                TP[c] += 1  # Verdaderos positivos
            elseif !outputs[s, c] && !targets[s, c]
                TN[c] += 1  # Verdaderos negativos
            elseif outputs[s, c] && !targets[s, c]
                FP[c] += 1  # Falsos positivos
            elseif !outputs[s, c] && targets[s, c]
                FN[c] += 1  # Falsos negativos
            end
        end
    end

    # Calcular las métricas globales
    accuracy = sum(TP) / sum(TP + TN + FP + FN)  # Precisión global
    error_rate = 1 - accuracy  # Tasa de error global

    # Calcular las métricas de forma macro o weighted según el parámetro
    if weighted
        weights = sum(targets, dims=1)  # Ponderación por clase
        sensitivity = sum(TP ./ (TP + FN) .* weights) / sum(weights)  # Sensibilidad ponderada
        specificity = sum(TN ./ (TN + FP) .* weights) / sum(weights)  # Especificidad ponderada
    else
        sensitivity = mean(TP ./ (TP + FN))  # Sensibilidad macro
        specificity = mean(TN ./ (TN + FP))  # Especificidad macro
    end

    return accuracy, error_rate, sensitivity, specificity, TP, TN, FP, FN
end;

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    num_classes = size(outputs, 2)  # Número de clases
    num_samples = size(outputs, 1)  # Número de muestras
    
    # Inicializar variables para almacenar métricas por clase
    TP = zeros(Int, num_classes)  # Verdaderos positivos por clase
    TN = zeros(Int, num_classes)  # Verdaderos negativos por clase
    FP = zeros(Int, num_classes)  # Falsos positivos por clase
    FN = zeros(Int, num_classes)  # Falsos negativos por clase
    
    # Convertir las salidas a valores booleanos
    outputs_bool = outputs .≥ 0.5
    
    # Calcular las métricas por clase
    for i in 1:num_classes
        for j in 1:num_samples
            if outputs_bool[j, i] == true && targets[j, i] == true
                TP[i] += 1
            elseif outputs_bool[j, i] == false && targets[j, i] == false
                TN[i] += 1
            elseif outputs_bool[j, i] == true && targets[j, i] == false
                FP[i] += 1
            elseif outputs_bool[j, i] == false && targets[j, i] == true
                FN[i] += 1
            end
        end
    end
    
    # Calcular las métricas globales
    accuracy = sum(TP) / (sum(TP) + sum(FP))  # Precisión global
    error_rate = 1 - accuracy  # Tasa de error global
    
    if weighted
        weights = sum(targets, dims=1)  # Ponderación por clase
        sensitivity = sum(TP ./ (TP .+ FN) .* weights) / sum(weights)  # Sensibilidad ponderada
        specificity = sum(TN ./ (TN .+ FP) .* weights) / sum(weights)  # Especificidad ponderada
    else
        sensitivity = sum(TP) / (sum(TP) + sum(FN))  # Sensibilidad global
        specificity = sum(TN) / (sum(TN) + sum(FP))  # Especificidad global
    end
    
    return accuracy, error_rate, sensitivity, specificity, TP, TN, FP, FN
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    # Verificar que los vectores tengan la misma longitud
    length(outputs) == length(targets) || throw(ArgumentError("Los vectores de salida y los vectores de destino deben tener la misma longitud."))

    # Contadores para verdaderos positivos, verdaderos negativos,
    # falsos positivos y falsos negativos
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    # Calcular las métricas
    for (output, target) in zip(outputs, targets)
        if output == target
            if output == true
                TP += 1
            else
                TN += 1
            end
        else
            if output == true
                FP += 1
            else
                FN += 1
            end
        end
    end

    # Calcular las métricas globales
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    error_rate = 1 - accuracy

    if weighted
        total_positive = sum(targets)
        total_negative = length(targets) - total_positive
        sensitivity = TP / total_positive
        specificity = TN / total_negative
    else
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
    end

    return accuracy, error_rate, sensitivity, specificity, TP, TN, FP, FN
end;

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 5 ---------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random
using Random:seed!

function crossvalidation(N::Int64, k::Int64)
    #
    # Codigo a desarrollar
    #
    ordenado = collect(1:k)
    
    return 
end;

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    #
    # Codigo a desarrollar
    #
end;

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    #
    # Codigo a desarrollar
    #
end;

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    #
    # Codigo a desarrollar
    #
end;







function ANNCrossValidation(topology::AbstractArray{<:Int,1},
    inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1},
    crossValidationIndices::Array{Int64,1};
    numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, validationRatio::Real=0, maxEpochsVal::Int=20, showText::Bool=false)
    #
    # Codigo a desarrollar
    #
end;


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 6 ---------------------------------------------
# ----------------------------------------------------------------------------------------------

using ScikitLearn: @sk_import, fit!, predict

@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier


function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1}, crossValidationIndices::Array{Int64,1})
    #
    # Codigo a desarrollar
    #
end;
