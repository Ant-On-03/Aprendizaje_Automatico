# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 2 ---------------------------------------------
# --------------------------OneHotEncoding, Normalización de Parámetros-------------------------
# -----------------------------------------Entrenamiento----------------------------------------
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
# -------------------------------------------------------
# Función para calcular la precisión para un vector de salidas binarias y un vector de objetivos binarios
function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    return sum(outputs .== targets) / length(targets)
end
 
# Función para calcular la precisión para matrices de salidas binarias y objetivos binarios
 
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
 
    if size(outputs,2) == 1 
        return accuracy(outputs[:], targets[:]; threshold)
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

    for i = 1:length(topology)
        numOutputsLayer = topology[i]
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, transferFunctions[i]));
        numInputsLayer = numOutputsLayer;
    end;

    # Agregar la última capa de salida
    if numOutputs == 1
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, σ))
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity))
        ann = Chain(ann...,softmax)
    end

    return ann
end



function trainClassANN(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    # Crear RNA
    ann = buildClassANN(size(dataset[1], 1), topology, size(dataset[2], 1), transferFunctions)
    losses = Float64[]
    # Definir la función de pérdida
    loss(model, x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x),y) : Losses.crossentropy(model(x),y);
    # Entrenamiento
    for epoch in 1:maxEpochs
        # Entrenar un ciclo
        Flux.train!(loss_function, Flux.params(ann), zip(eachrow(dataset[1]), eachrow(dataset[2])), ADAM(learningRate))
        # Calcular la pérdida en este ciclo
        loss_value = loss_function(dataset[1]', dataset[2]')
        # Almacenar el valor de pérdida
        push!(losses, loss_value)
        # Criterio de parada
        if loss_value <= minLoss
            break
        end
    end
    return (ann, losses)
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
    
    test = indices[1:num_test]
    train = indices[num_test+1:end]
    
    return (train, test)
end;


function holdOut(N::Int, Pval::Real, Ptest::Real)
   indices, test = holdOut(N, Ptest)
   Pval = Pval * N / length(indices)
   val, train = holdOut(length(indices), Pval)

   val = indices[1:length(train)]
   train = indices[length(train)+1:end]

   return (train, val, test)
end


# Funcion para entrenar RR.NN.AA. con conjuntos de entrenamiento, validacion y test. Estos dos ultimos son opcionales
# Es la funcion anterior, modificada para calcular errores en los conjuntos de validacion y test y realizar parada temprana si es necesario
function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)
   
    # Crear RNA
    ann = buildClassANN(size(trainingDataset[1], 1), topology, size(trainingDataset[2], 1); transferFunctions=transferFunctions)
    losses_train = Float64[]
    losses_validation = Float64[]
    losses_test = Float64[]

    # Definir la función de pérdida
    loss_function(ann, x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);

    loss_train = loss_function(ann, trainingDataset[1]', trainingDataset[2]')
    push!(losses_train, loss_train)

    if !(isempty(validationDataset[1]) && isempty(validationDataset[2]))
        loss_validation = loss_function(ann, validationDataset[1]', validationDataset[2]')
        push!(losses_validation, loss_validation)
        
        best_loss_validation = loss_validation
        best_ann = deepcopy(ann)
    end

    if !(isempty(testDataset[1]) && isempty(testDataset[2]))
        loss_test = loss_function(ann, testDataset[1]', testDataset[2]')
        push!(losses_test, loss_test)
    end

    
    counter = 0
    # Se ha proporcionado un conjunto de validación
    for epoch in 1:maxEpochs

        # Entrenar un ciclo
        Flux.train!(loss_function, Flux.params(ann), zip(eachrow(trainingDataset[1]), eachrow(trainingDataset[2])), ADAM(learningRate))
    
        # Calcular la pérdida en este ciclo para el conjunto de entrenamiento
        loss_train = loss_function(ann, trainingDataset[1]', trainingDataset[2]')
        push!(losses_train, loss_train)

        if !(isempty(testDataset[1]) && isempty(testDataset[2]))
            # Calcular la pérdida en este ciclo para el conjunto de prueba
            loss_test = loss_function(ann, testDataset[1]', testDataset[2]')
            push!(losses_test, loss_test)
        end
    
        if !(isempty(validationDataset[1]) && isempty(validationDataset[2]))
            # Calcular la pérdida en este ciclo para el conjunto de validación
            loss_validation = loss_function(ann, validationDataset[1]', validationDataset[2]')
            push!(losses_validation, loss_validation)

            counter = counter + 1
            # Actualizar el modelo si se encuentra una pérdida de validación más baja
            if loss_validation < best_loss_validation
                best_loss_validation = loss_validation
                best_ann = deepcopy(ann)
                counter = 0
            end

            # Criterio de parada temprana basado en el número de épocas sin mejorar la validación
            if counter >= maxEpochsVal
                break
            end

        end

        if loss_train <= minLoss
            break
        end
        
    end

    if isempty(validationDataset[1]) && isempty(validationDataset[2])
        best_ann = deepcopy(ann)
    end

    # Devolver la mejor RNA y los vectores de pérdidas
    return (best_ann, losses_train, losses_validation, losses_test)
end;
 
 
function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)
   
    # Convertir las salidas deseadas a formato de matriz
    trainingDataset = (trainingDataset[1], reshape(trainingDataset[2], (length(trainingDataset[2]), 1)))
    validationDataset = (validationDataset[1], reshape(validationDataset[2], (length(validationDataset[2]), 1)))
    testDataset = (testDataset[1], reshape(testDataset[2], (length(testDataset[2]), 1)))
   
    # Llamar a la función original trainClassAnn con los nuevos datos
    return trainClassANN(topology, trainingDataset; validationDataset=validationDataset, testDataset=testDataset, transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, maxEpochsVal=maxEpochsVal)
end;

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 4 ---------------------------------------------
# ---------------------------------------------------------------------------------------------

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    VP = sum((outputs .== true) .& (targets .== true))
    FP = sum((outputs .== true) .& (targets .== false))
    VN = sum((outputs .== false) .& (targets .== false))
    FN = sum((outputs .== false) .& (targets .== true))

    # Casos particulares
    sensitivity = VP == 0 && FN == 0 ? 1 : VP / (VP + FN) 
    VPP = VP == 0 && FP == 0 ? 1 : VP / (VP + FP)
    specificity = VN == 0 && FP == 0 ? 1 : VN / (VN + FP)
    VPN = VN == 0 && FN == 0 ? 1 : VN / (VN + FN)
    f1_score = sensitivity == 0 && VPP == 0 ? 0 : 2 * (VPP * sensitivity) / (VPP + sensitivity)

    accuracy = (VP + VN) / (VP + VN + FP + FN)  # valor de precisión
    error_rate = 1 - accuracy  # Tasa de fallo
    conf_mat = [VN FP; FN VP]

    return (accuracy, error_rate, sensitivity, specificity, VPP, VPN, f1_score, conf_mat)
end


function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    # Convertir las salidas a valores booleanos según el umbral
    outputs_bool = outputs .≥ threshold

    return confusionMatrix(outputs_bool, targets)
end;

function printConfusionMatrix(outputs::AbstractArray{Bool,1},targets::AbstractArray{Bool,1})
    accuracy, error_rate, sensitivity, specificity, VPP, VPN, f1_score, conf_mat = confusionMatrix(outputs, targets)
        
    println("Precisión: $accuracy")
    println("Tasa de fallo: $error_rate")
    println("Sensibilidad: $sensitivity")
    println("Especifidad: $specificity")
    println("Valor predictivo positivo: $VPP")
    println("Valor predictivo negativo: $VPN")
    println("F1-score: $f1_score")
    println("Matriz de confusión:\n$conf_mat")
end


function printConfusionMatrix(outputs::AbstractArray{<:Real,1},targets::AbstractArray{Bool,1}; threshold::Real=0.5) 
    accuracy, error_rate, sensitivity, specificity, VPP, VPN, f1_score, conf_mat = confusionMatrix(outputs, targets)
        
    println("Precisión: $accuracy")
    println("Tasa de fallo: $error_rate")
    println("Sensibilidad: $sensitivity")
    println("Especifidad: $specificity")
    println("Valor predictivo positivo: $VPP")
    println("Valor predictivo negativo: $VPN")
    println("F1-score: $f1_score")
    println("Matriz de confusión:\n$conf_mat")
end

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    # Verificar que el número de columnas es distinto de 2
    size(outputs, 2) == size(targets, 2) != 2 || throw(ArgumentError("El número de columnas de ambas matrices debe ser igual y distinto de 2"))

    if size(outputs,2) == 1
        return confusionMatrix(outputs, targets)
    else 
        num_classes = size(outputs, 2)
        num_samples = size(outputs, 1)

        # Inicializar vectores para métricas por clase
        sensitivity = zeros(Float64, num_classes)
        specificity = zeros(Float64, num_classes)
        VPP = zeros(Float64, num_classes)
        VPN = zeros(Float64, num_classes)
        f1_score = zeros(Float64, num_classes)
    
        # Calcular métricas por clase
        for c in 1:num_classes
            outputs_class = outputs[:, c]
            targets_class = targets[:, c]
            metrics = confusionMatrix(outputs_class, targets_class)

            sensitivity[c] = metrics[3]
            specificity[c] = metrics[4]
            VPP[c] = metrics[5]
            VPN[c] = metrics[6]
            f1_score[c] = metrics[7]
        end

        # Inicializar la matriz de confusión
        conf_mat = zeros(Int, num_classes, num_classes)

        # Iterar sobre las muestras
        for i in 1:size(outputs, 1)
            # Obtener la clase predicha y la clase real para esta muestra
            predicted_class = argmax(outputs[i, :])
            actual_class = argmax(targets[i, :])
            
            # Incrementar el recuento en la matriz de confusión
            conf_mat[actual_class, predicted_class] += 1
        end
    
    
        # Calcular métricas macro o weighted
        if weighted
            weights = sum(targets, dims=1)[:]
            sensitivity_weighted = sum(sensitivity .* weights) / sum(weights)
            specificity_weighted = sum(specificity .* weights) / sum(weights)
            VPP_weighted = sum(VPP .* weights) / sum(weights)
            VPN_weighted = sum(VPN .* weights) / sum(weights)
            f1_score_weighted = sum(f1_score .* weights) / sum(weights)
        else
            sensitivity_weighted = mean(sensitivity)
            specificity_weighted = mean(specificity)
            VPP_weighted = mean(VPP)
            VPN_weighted = mean(VPN)
            f1_score_weighted = mean(f1_score)
        end

        # Calcular precisión y tasa de error
        accuracy1 = accuracy(outputs, targets)
        error_rate = 1 - accuracy1

    end
    
        return (accuracy1, error_rate, sensitivity_weighted, specificity_weighted, VPP_weighted, VPN_weighted, f1_score_weighted, conf_mat)
        
    end

    
function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    outputs_bool = classifyOutputs(outputs)

    return confusionMatrix(outputs_bool, targets; weighted)
end;


function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    # Verificar que los vectores tengan la misma longitud
    length(outputs) == length(targets) || throw(ArgumentError("Los vectores de salida y los vectores de destino deben tener la misma longitud."))

     # Obtener las clases únicas presentes en los vectores outputs y targets
     classes = unique(vcat(outputs, targets))

     # Codificar los vectores outputs y targets utilizando one-hot encoding
     encoded_outputs = oneHotEncoding(outputs, classes)
     encoded_targets = oneHotEncoding(targets, classes)
 
     # Llamar a la función confusionMatrix con las codificaciones resultantes
     return confusionMatrix(encoded_outputs, encoded_targets; weighted=weighted)
end;

function printConfusionMatrix(outputs::AbstractArray{Bool,2},targets::AbstractArray{Bool,2}; weighted::Bool=true)
    accuracy, error_rate, sensitivity, specificity, VPP, VPN, f1_score, conf_mat = confusionMatrix(outputs, targets; weighted = weighted)
        
    println("Precisión: $accuracy")
    println("Tasa de fallo: $error_rate")
    println("Sensibilidad: $sensitivity")
    println("Especifidad: $specificity")
    println("Valor predictivo positivo: $VPP")
    println("Valor predictivo negativo: $VPN")
    println("F1-score: $f1_score")
    println("Matriz de confusión:\n$conf_mat")
end; 

function printConfusionMatrix(outputs::AbstractArray{<:Real,2},targets::AbstractArray{Bool,2}; weighted::Bool=true)
    accuracy, error_rate, sensitivity, specificity, VPP, VPN, f1_score, conf_mat = confusionMatrix(outputs, targets; weighted = weighted)
        
    println("Precisión: $accuracy")
    println("Tasa de fallo: $error_rate")
    println("Sensibilidad: $sensitivity")
    println("Especifidad: $specificity")
    println("Valor predictivo positivo: $VPP")
    println("Valor predictivo negativo: $VPN")
    println("F1-score: $f1_score")
    println("Matriz de confusión:\n$conf_mat")

end;

function printConfusionMatrix(outputs::AbstractArray{<:Any,1},targets::AbstractArray{<:Any,1}; weighted::Bool=true) 
    accuracy, error_rate, sensitivity, specificity, VPP, VPN, f1_score, conf_mat = confusionMatrix(outputs, targets; weighted = weighted)
        
    println("Precisión: $accuracy")
    println("Tasa de fallo: $error_rate")
    println("Sensibilidad: $sensitivity")
    println("Especifidad: $specificity")
    println("Valor predictivo positivo: $VPP")
    println("Valor predictivo negativo: $VPN")
    println("F1-score: $f1_score")
    println("Matriz de confusión:\n$conf_mat")
end;

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 5 ---------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random
using Random:seed!

function crossvalidation(N::Int64, k::Int64)
    # muestreo de forma estratificada en k subconjuntos.
    # devolver N elementos ordenados en k subconjuntos 
    ordenado = collect(1:k)
    # obtener un vector de longitud N que repita 1,2,3,...k, 1,2,3,...k hasta la longitud N
    repetido = repeat(ordenado, ceil(Int64, N/k))[1:N]    
    # devolverlo desordenado
    return shuffle!(repetido) 
    
end;

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    # muestreo de forma estratificada en k subconjuntos.
    # para un vector devolver un índice asignando a cada posición un elemento mediante muestreo estratificado.
    indices = Vector{Int64}(undef,length(targets))
    # asignar en el índice a los elementos positivos, sus respectivos subconjuntos
    indices[findall(targets)] = crossvalidation(count(targets), k)
    # asignar por separado en el índice de elementos negativos, sus respectivos subconjuntos
    indices[findall(!,targets)] = crossvalidation(count(!,targets), k)

    return indices
end;


function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    # muestreo de forma estratificada en k subconjuntos.
    indices = Vector{Int64}(undef, size(targets,1))
    # por cada columna (atributo) de targets
    for col in 1:size(targets,2)
        n_positives = sum(targets[:,col])
        # coloca en índice a los elementos positivos, sus respectivos subconjuntos.
        indices[findall(targets[:,col])] = crossvalidation(n_positives, k)
    end
    # repite esto por cada columna de manera que todos los elementos quedan con un subconjunto asociado
    return indices
end;


function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)    

    # Ambas soluciones funcionan. Quizás la segunda sea más rápida.

    #Sol1.
    #return crossvalidation(oneHotEncoding(targets), k)


    #Sol2.
    # without onehotencoding (reto Profe):
    indice = Vector{Int64}(undef, length(targets))
    for class in unique(targets)

        println(class)
        n_class = length(findall(x -> x==class, targets))
        println(n_class)
        indice[findall(x -> x==class, targets)] = crossvalidation(n_class, k)
    end
    return indice
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
    if modelType == SVC
        model = SVC(kernel="rbf", degree=3, gamma=2, C=1);
    elseif modelType == DecisionTreeClassifier
        model = DecisionTreeClassifier(max_depth=4, random_state=1) 
    else 
        model = KNeighborsClassifier(3); 
    end 

    testOutputs = predict(model, inputs); 

end;
