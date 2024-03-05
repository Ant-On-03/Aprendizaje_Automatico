using FileIO;
using DelimitedFiles;
using Statistics;
using Flux;


include("58002749S_35591688Q_58024091J.jl")


# Leer los datos y definir targets
dataset = readdlm("iris.data", ',')
inputs = dataset[:, 1:4]
targets = dataset[:, 5]

# Llamar a la función oneHotEncoding con el vector y las clases
oneHotEncoding(targets)

holdOut(10,0.4,0.5)

outputs_bool = [false, true, true, false]
outputs = [2,0.1,0.6,0.3]
targets_bool = [true, true, false, false]

accuracy(outputs,targets_bool)

printConfusionMatrix(outputs_bool,targets_bool)

confusionMatrix(outputs,targets_bool)

outputs_array = [0 0 1; 1 0 0]
targets_array = [false false true; true false false]

confusionMatrix(outputs_array, targets_array)






# Define la función sigmoide para las capas ocultas
σ(x) = 1 / (1 + exp(-x))

# Definir la topología de la red neuronal
numInputs = size(inputs, 2) # Número de características de entrada
topology = [8, 8]            # Dos capas ocultas con 8 neuronas cada una
numOutputs = 3               # Tres clases en el conjunto de datos Iris

# Definir las funciones de activación para cada capa oculta
transferFunctions = fill(σ, length(topology))

# Construir la red neuronal
buildClassANN(numInputs, topology, numOutputs)

buildClassANN_1(numInputs, topology, numOutputs)

function buildClassANN_1(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    # Num de entradas, num de neuronas de cada capa oculta, num de salidas y función de activación
    ann = Chain();
    numInputsLayer = numInputs
    if length(topology) != 0
        for i = 1:length(topology)
            numOutputsLayer = topology[i]
            ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, transferFunctions[i]));
            numInputsLayer = numOutputsLayer;
        end;
    end;
    numclasses = numOutputs
    if numclasses == 1
        ann = Chain(ann..., Dense(topology[end], numclasses, σ))
    else
        ann = Chain(ann..., Dense(topology[end], numclasses, identity))
        ann = Chain(ann..., softmax)
    end;
end;


############################################################################################
############################################################################################

# Cargar el conjunto de datos Iris desde el archivo
dataset = readdlm("iris.data", ',')

# Extraer los inputs y targets
inputs = convert(Array{Float64, 2}, dataset[:, 1:4])
# Extraer los targets como cadenas de texto
targets = convert(Vector{String}, dataset[:, 5])


# Convertir las etiquetas de clase a números enteros
label_dict = Dict("Iris-setosa" => 1, "Iris-versicolor" => 2, "Iris-virginica" => 3)
targets = map(x -> label_dict[x], targets)

# Convertir los targets a one-hot encoding
using Flux: onehotbatch
targets = oneHotEncoding(targets,classes)

# Definir la topología de la red neuronal
topology = [8, 8]  # Dos capas ocultas con 8 neuronas cada una
num_outputs = 3     # Tres clases en el conjunto de datos

# Entrenar la red neuronal
trainClassANN1(topology, (inputs, targets))

#function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    #outputs_bool = classifyOutputs(outputs, threshold=threshold)  
    #return sum(outputs_bool .== targets) / length(targets)
#end

#########################################################################################################
#########################################################################################################
#########################################################################################################

# Definir una matriz de ejemplo
 matriz = [1 2 3; 4 5 6; 7 8 9]

# Método 1: Usar la función show
println("Usando la función show:")
show(stdout, "text/plain", matriz)
println()




