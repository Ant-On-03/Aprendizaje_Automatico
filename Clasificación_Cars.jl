

# Ahora vamos a medir y comparar la eficiencia de distintos métodos de clasificación sobre la base de datos
# de car + evaluation.

# Primero cargamos la base de datos y la preprocesamos
# download DataFrames package
using Pkg
Pkg.add("DataFrames")


using DataFrames
using Random
using ScikitLearn: fit!, predict, @sk_import
using Plots
using Statistics
using Random
using DelimitedFiles;



dataset = readdlm("datasets/car_evaluation/car.data", ','); # me pasas por wasap ou teams este archivo de iris? 

# Creamos un DataFrame con la base de datos
df = DataFrame(dataset, :auto)
println(df)