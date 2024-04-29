

# Ahora vamos a medir y comparar la eficiencia de distintos métodos de clasificación sobre la base de datos
# de car + evaluation.

# Primero cargamos la base de datos y la preprocesamos

using CSV

# download DataFrames package
using Pkg
Pkg.add("DataFrames")


using DataFrames
using Random
using DecisionTree
using ScikitLearn: fit!, predict, @sk_import
using MLBase: cross_entropy
using StatsBase: sample, mean
using Plots
using Statistics

# Cargamos la base de datos
# import the car.data file

# Creamos un DataFrame con la base de datos
# Creamos un DataFrame con la base de datos




# Cargamos la base de datos
# import the car.data file

using Random
using DelimitedFiles;

using Statistics
dataset = readdlm("datasets/iris.data", ','); # me pasas por wasap ou teams este archivo de iris? 

df = CSV.read("datasets/iris.data", DataFrame; header = false)


# Check if the file exists and is accessible
import Base.Filesystem: isfile

if isfile("datasets/iris.data")
    df = CSV.read("datasets/iris.data", DataFrame; header = false)
else
    println("File does not exist or is not accessible")
end


datasets/car_evaluation/car.data
# Preprocesamos la base de datos
# Convertimos las variables categóricas a numéricas
