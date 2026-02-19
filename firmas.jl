
# Tened en cuenta que en este archivo todas las funciones tienen puesta la palabra reservada 'function' y 'end' al final
# Según cómo las defináis, podrían tener que llevarlas o no

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 2 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Statistics
using Flux
using Flux.Losses


function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    numClasses = length(classes)
    numInstances = length(feature)

    if numClasses == 0
        error("No se han proporcionado clases para la codificación one-hot.")
    elseif numClasses <= 2
        binary_result = feature .== classes[1] # Codificación binaria
        return reshape(binary_result, numInstances, 1) # Convertir a matriz de una columna
    else
        encoded = fill(false ,numInstances, numClasses) # Codificación one-hot
        for (i, class) in enumerate(classes)
            is_this_class = feature .== class   
            encoded[is_this_class, i] .= true
        end
        return encoded
    end
end;

function oneHotEncoding(feature::AbstractArray{<:Any,1})
    classes = sort(unique(feature))
    return oneHotEncoding(feature, classes)
end;

function oneHotEncoding(feature::AbstractArray{Bool,1})
    return reshape(feature, length(feature), 1)
end;

function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    min_vals = minimum(dataset, dims=1)
    max_vals = maximum(dataset, dims=1)
    return (min_vals, max_vals)
end;

function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    mean_vals = mean(dataset, dims=1)
    std_vals = std(dataset, dims=1)
    return (mean_vals, std_vals)
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    min_vals, max_vals = normalizationParameters
    ranges = max_vals .- min_vals

    dataset .-= min_vals
    dataset ./= ranges

    #Columnas constantes a 0
    dataset[:, vec(ranges .== 0)] .= 0.0

    return dataset
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    paramas = calculateMinMaxNormalizationParameters(dataset)
    return normalizeMinMax!(dataset, paramas)
end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    newDataset = copy(dataset)
    normalizeMinMax!(newDataset, normalizationParameters)   
    return newDataset
end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2})
    newDataset = copy(dataset)
    normalizeMinMax!(newDataset)
    return newDataset
end;

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    mean_vals, std_vals = normalizationParameters
    dataset .-= mean_vals
    dataset ./= std_vals

    #Columnas constantes a 0
    dataset[:, vec(std_vals .== 0)] .= 0.0

    return dataset
end;

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    paramas = calculateZeroMeanNormalizationParameters(dataset)
    return normalizeZeroMean!(dataset, paramas)
end;

function normalizeZeroMean(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    newDataset = copy(dataset)
    normalizeZeroMean!(newDataset, normalizationParameters)   
    return newDataset
end;

function normalizeZeroMean(dataset::AbstractArray{<:Real,2})
    newDataset = copy(dataset)
    normalizeZeroMean!(newDataset)
    return newDataset
end;

function classifyOutputs(outputs::AbstractArray{<:Real,1}; threshold::Real=0.5)
    return outputs .>= threshold
end;

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    numCols = size(outputs, 2)

    # Caso binario
    if numCols == 1
        classifiedVector = classifyOutputs(outputs[:]; threshold=threshold)
        return reshape(classifiedVector, :, 1)
    end

    # Caso multiclase
    # Indice de la clase con mayor valor
    (_, maxIndices) = findmax(outputs, dims=2)

    # Creamos matriz booleana del mismo tamaño
    classified = fill(false,size(outputs))

    # Ponemos a true el indice de la clase con mayor valor
    classified[maxIndices] .= true

    return classified
end;

function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    mean(outputs .== targets)
end;

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    numCols = size(outputs, 2)
    # Solo 1 columna
    if numCols == 1
        return accuracy(outputs[:, 1], targets[:, 1])
    end
    # Multiclase
    matches = eachrow(outputs) .== eachrow(targets)
    return mean(matches)
end;

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    classifiedOutputs = classifyOutputs(outputs; threshold=threshold)
    return accuracy(classifiedOutputs, targets)
end;

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    numCols = size(outputs, 2)

    # Solo 1 columna
    if numCols == 1
        return accuracy(outputs[:, 1], targets[:, 1]; threshold=threshold)
    else
        classifiedOutputs = classifyOutputs(outputs)
        return accuracy(classifiedOutputs, targets)
    end
end;

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    # Red Inicial Vacía
    ann = Chain()

    # Numero entradas para la primera capa
    numInputsLayer = numInputs

    # Capas ocultas
    for (i, numNeurons) in enumerate(topology)
        ann = Chain(ann..., Dense(numInputsLayer, numNeurons, transferFunctions[i]))
        numInputsLayer = numNeurons
    end

    # Funcion de activacion de la capa de salida según numero de clases
    if numOutputs == 1
        # Para clasificación binaria
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, σ))
    else 
        # Clasificacion multiclase: capa lineal y luego softmax
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity), softmax)
    end

    return ann
end;

function trainClassANN(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    inputs, targets = dataset

    # Covertir entradas a Float32
    inputs = Float32.(inputs)
    # Covertir salidas a Float32
    targets = Float32.(targets)

    # Numero de neuronas de entrada y salida
    numInputs = size(inputs, 2)
    numOutputs = size(targets, 2)

    # Construir la red
    ann = buildClassANN(numInputs, topology, numOutputs; transferFunctions=transferFunctions)

    function loss_function(model, x, y)
        if numOutputs == 1
            return Flux.Losses.binarycrossentropy(model(x), y)
        else
            return Flux.Losses.crossentropy(model(x), y)
        end
    end

    # Optimizador
    opt = Flux.setup(Adam(learningRate), ann)

    # Vector para almacenar losses 
    losses = Float32[]

    # Evaluar loss ciclo 0
    push!(losses, Float32(loss_function(ann, inputs', targets')))

    # Entrenamiento
    for epoch in 1:maxEpochs
        # Entrena un ciclo
        Flux.train!(loss_function, ann, [(inputs', targets')], opt)

        # Evaluar loss actual 
        current_loss = loss_function(ann, inputs', targets')
        push!(losses, current_loss)

        # Comprobamos criterios de parada
        if current_loss <= minLoss 
            break
        end 
    end 

    return ann, losses
end;

function trainClassANN(topology::AbstractArray{<:Int,1}, (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01) 
    # Convertir vector  a matriz columna
    targets_matrix = reshape(targets, :, 1)

    # Llamamos a la version matriz 
    return trainClassANN(topology, (inputs, targets_matrix); transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate)
end;


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 3 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random

function holdOut(N::Int, P::Real)
    # Validar parametros
    @assert 0.0 <= P <= 1.0 "P debe estar entre 0 y 1"

    # Calcular número elementos para test
    n_test = round(Int, N * P)

    # Permutar aleatoriamente los indices
    perm = randperm(N)

    # Dividir indices en train y test
    idx_test = perm[1:n_test]
    idx_train = perm[n_test+1:end]

    return idx_train, idx_test
end;

function holdOut(N::Int, Pval::Real, Ptest::Real)
    # Validar parametros
    @assert 0.0 <= Pval <= 1.0 "Pval debe estar entre 0 y 1"
    @assert 0.0 <= Ptest <= 1.0 "Ptest debe estar entre 0 y 1"
    @assert Pval + Ptest <= 1.0 "La suma de Pval y Ptest no puede ser mayor que 1"

    # Primero separamos el conjunto de test
    idx_train_val, idx_test = holdOut(N, Ptest)
    
    # Calcular Pval ajustado para el conjunto restante
    n_train_val = length(idx_train_val)
    n_val_target = round(Int, N * Pval)

    if n_train_val > 0
        Pval_adjusted = n_val_target / n_train_val
        # Asegurar que Pval_adjusted no es mayor que 1
        Pval_adjusted = min(Pval_adjusted, 1.0)
    else 
        Pval_adjusted = 0.0
    end

    # Separar train y validation del conjunto restante
    idx_train, idx_val = holdOut(n_train_val, Pval_adjusted)

    # Mapear indices de idx_train e idx_val al conjunto original
    idx_train = idx_train_val[idx_train]
    idx_val = idx_train_val[idx_val]

    return idx_train, idx_val, idx_test
end;

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0,size(trainingDataset[2],2))),
    testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0,size(trainingDataset[2],2))),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)
    
    # Extraer datos Entrenamiento
    train_inputs, train_targets = trainingDataset
    
    # Preparar datos validación y test
    val_inputs, val_targets = validationDataset
    test_inputs, test_targets = testDataset
    
    # Obtener dimensiones 
    numInputs = size(train_inputs, 2)
    numOutputs = size(train_targets, 2)
    
    # Construir la red
    ann = buildClassANN(numInputs, topology, numOutputs; transferFunctions=transferFunctions)
    
    # Definir función de pérdida
    function loss_function(model, x, y)
        if numOutputs == 1
            return Flux.Losses.binarycrossentropy(model(x), y)
        else
            return Flux.Losses.crossentropy(model(x), y)
        end
    end
    
    # Optimizador
    opt_state = Flux.setup(Adam(learningRate), ann)
    
    # Vector para almacenar losses
    train_losses = Float32[]
    val_losses = Float32[]
    test_losses = Float32[]
    
    # Preparar datos para flux 
    train_inputs_t = Float32.(train_inputs)'
    train_targets_t = Float32.(train_targets)'
    
    # Preparar datos de validación
    has_validation = size(val_inputs, 1) > 0 && size(val_targets, 1) > 0
    if has_validation
        val_inputs_t = Float32.(val_inputs)'
        val_targets_t = Float32.(val_targets)'
    end
    
    # Preparar datos de test
    has_test = size(test_inputs, 1) > 0 && size(test_targets, 1) > 0
    if has_test
        test_inputs_t = Float32.(test_inputs)'
        test_targets_t = Float32.(test_targets)'
    end
    
    # Calcular loss inicial
    initial_train_loss = Float32(loss_function(ann, train_inputs_t, train_targets_t))
    push!(train_losses, initial_train_loss)
    
    if has_validation
        initial_val_loss = Float32(loss_function(ann, val_inputs_t, val_targets_t))
        push!(val_losses, initial_val_loss)
    end
    
    if has_test
        initial_test_loss = Float32(loss_function(ann, test_inputs_t, test_targets_t))
        push!(test_losses, initial_test_loss)
    end
    
    # Variables para early stopping
    if has_validation
        best_val_loss = val_losses[1]
        best_ann = deepcopy(ann)
        epochs_without_improvement = 0
    else
        best_ann = nothing
        epochs_without_improvement = 0
    end
    
    # Bucle principal Entrenamiento
    for epoch in 1:maxEpochs
        # Entrena un ciclo con nueva API de Flux
        Flux.train!(loss_function, ann, [(train_inputs_t, train_targets_t)], opt_state)
        
        # Evaluar loss actual en entrenamiento
        current_train_loss = Float32(loss_function(ann, train_inputs_t, train_targets_t))
        push!(train_losses, current_train_loss)
        
        # Evaluar loss actual en validación
        if has_validation
            current_val_loss = Float32(loss_function(ann, val_inputs_t, val_targets_t))
            push!(val_losses, current_val_loss)
            
            # Comprobar mejora en validación para early stopping
            if current_val_loss < best_val_loss
                best_val_loss = current_val_loss
                best_ann = deepcopy(ann)
                epochs_without_improvement = 0
            else
                epochs_without_improvement += 1
            end
        end
        
        # Evaluar loss actual en test
        if has_test
            current_test_loss = Float32(loss_function(ann, test_inputs_t, test_targets_t))
            push!(test_losses, current_test_loss)
        end
        
        # Comprobar criterio de parada por early stopping (después de evaluar test)
        if has_validation && epochs_without_improvement >= maxEpochsVal
            break
        end
        
        # Comprobar criterio de parada por loss mínimo alcanzado
        if current_train_loss <= minLoss 
            break
        end 
    end

    # Determinar qué modelo devolver
    if has_validation
        return best_ann, train_losses, val_losses, test_losses
    else
        return ann, train_losses, val_losses, test_losses
    end
end;

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)
    
    # Convertir targets a matriz columna
    train_inputs, train_targets = trainingDataset
    train_targets_matrix = reshape(train_targets, :, 1)
    
    # Para validación
    val_inputs, val_targets = validationDataset
    if length(val_targets) > 0
        val_targets_matrix = reshape(val_targets, :, 1)
        new_validationDataset = (val_inputs, val_targets_matrix)
    else
        new_validationDataset = (Array{eltype(val_inputs),2}(undef,0,size(val_inputs,2)), falses(0,1))
    end
    
    # Para test
    test_inputs, test_targets = testDataset
    if length(test_targets) > 0
        test_targets_matrix = reshape(test_targets, :, 1)
        new_testDataset = (test_inputs, test_targets_matrix)
    else
        new_testDataset = (Array{eltype(test_inputs),2}(undef,0,size(test_inputs,2)), falses(0,1))
    end
    
    # Llamar a la versión matriz
    return trainClassANN(topology, 
        (train_inputs, train_targets_matrix); 
        validationDataset=new_validationDataset, 
        testDataset=new_testDataset, 
        transferFunctions=transferFunctions, 
        maxEpochs=maxEpochs, 
        minLoss=minLoss, 
        learningRate=learningRate, 
        maxEpochsVal=maxEpochsVal)
end;


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 4 --------------------------------------------
# ----------------------------------------------------------------------------------------------


function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
end;

using SymDoME
using GeneticProgramming


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    #
    # Codigo a desarrollar
    #
end;

function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    #
    # Codigo a desarrollar
    #
end;


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    #
    # Codigo a desarrollar
    #
end;




# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 5 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random
using Random:seed!

function crossvalidation(N::Int64, k::Int64)
    #
    # Codigo a desarrollar
    #
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
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
    crossValidationIndices::Array{Int64,1};
    numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, validationRatio::Real=0, maxEpochsVal::Int=20)
    #
    # Codigo a desarrollar
    #
end;


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 6 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using MLJ
using LIBSVM, MLJLIBSVMInterface
using NearestNeighborModels, MLJDecisionTreeInterface

SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
kNNClassifier = MLJ.@load KNNClassifier pkg=NearestNeighborModels verbosity=0
DTClassifier  = MLJ.@load DecisionTreeClassifier pkg=DecisionTree verbosity=0


function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, crossValidationIndices::Array{Int64,1})
    #
    # Codigo a desarrollar
    #
end;



