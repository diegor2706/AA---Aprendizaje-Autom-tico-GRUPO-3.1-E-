using Statistics
using Flux
using Flux.Losses
using Random

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 2 --------------------------------------------
# ----------------------------------------------------------------------------------------------

# 1. ONE HOT ENCODING
function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    numClasses = length(classes)
    
    if numClasses <= 2
        return reshape(feature .== classes[1], :, 1)
    else
        oneHot = BitArray{2}(undef, length(feature), numClasses)
        for i = 1:numClasses
            oneHot[:, i] .= (feature .== classes[i])
        end
        return oneHot
    end
end

oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature))
oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, :, 1)


# 2. NORMALIZACIÓN
function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    minValues = minimum(dataset, dims=1)
    maxValues = maximum(dataset, dims=1)
    return (minValues, maxValues)
end
    
function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    avgValues = mean(dataset, dims=1)
    stdValues = std(dataset, dims=1)
    return (avgValues, stdValues)
end

function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    dataset .-= normalizationParameters[1]
    dataset ./= (normalizationParameters[2] .- normalizationParameters[1])
    dataset[:, vec(normalizationParameters[1] .== normalizationParameters[2])] .= 0
    return dataset
end

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    mins_maxs = calculateMinMaxNormalizationParameters(dataset)
    return normalizeMinMax!(dataset, mins_maxs)
end

function normalizeMinMax(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    dataset2 = copy(dataset)
    normalizeMinMax!(dataset2, normalizationParameters)
    return dataset2
end

function normalizeMinMax(dataset::AbstractArray{<:Real,2}) 
    dataset2 = copy(dataset)
    normalizeMinMax!(dataset2)
    return dataset2
end

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    dataset .-= normalizationParameters[1]
    dataset ./= normalizationParameters[2]
    dataset[:, vec(normalizationParameters[2] .== 0)] .= 0
    return dataset
end

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    avgs_stds = calculateZeroMeanNormalizationParameters(dataset)
    return normalizeZeroMean!(dataset, avgs_stds)
end

function normalizeZeroMean(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    dataset2 = copy(dataset)
    normalizeZeroMean!(dataset2, normalizationParameters)
    return dataset2
end

function normalizeZeroMean(dataset::AbstractArray{<:Real,2})
    dataset2 = copy(dataset)
    normalizeZeroMean!(dataset2)
    return dataset2
end


# 3. accuracy y classifyOutputs
function classifyOutputs(outputs::AbstractArray{<:Real,1}; threshold::Real=0.5)
    return outputs .>= threshold
end

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    if size(outputs, 2) == 1 
        vector = classifyOutputs(outputs[:]; threshold=threshold)
        return reshape(vector, :, 1)
    else 
        (_, indicesMaxEachInstance) = findmax(outputs, dims=2)
        outputs_bool = falses(size(outputs))
        outputs_bool[indicesMaxEachInstance] .= true 
        return outputs_bool
    end
end

function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    return mean(targets .== outputs)
end

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    num_cols_targets = size(targets, 2)
    num_cols_outputs = size(outputs, 2)
    @assert (num_cols_targets == num_cols_outputs) "las matrices no tienen el mismo numero de columnas"
    
    if num_cols_targets == 1
        return accuracy(targets[:, 1], outputs[:, 1])
    else
        return mean(all(targets .== outputs, dims=2))
    end
end

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    predicted_classes = outputs .>= threshold
    return accuracy(predicted_classes, targets)
end

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    num_cols_targets = size(targets, 2)
    num_cols_outputs = size(outputs, 2)
    @assert (num_cols_targets == num_cols_outputs) "las matrices no tienen el mismo numero de columnas"
    
    if num_cols_targets == 1
        return accuracy(outputs[:, 1], targets[:, 1]; threshold=threshold)
    else
        predicted_classes = classifyOutputs(outputs; threshold=threshold)
        return accuracy(predicted_classes, targets)    
    end
end


# 4. CONSTRUCCIÓN DE LA RED
function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    ann = Chain()
    numInputsLayer = numInputs
    for i in eachindex(topology)
        ann = Chain(ann..., Dense(numInputsLayer, topology[i], transferFunctions[i]))
        numInputsLayer = topology[i]
    end 
    if numOutputs > 2
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity))
        ann = Chain(ann..., softmax)
    else
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ))
    end
    return ann 
end

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 3 --------------------------------------------
# ----------------------------------------------------------------------------------------------

# 5. HOLD OUT
function holdOut(N::Int, P::Real)
    indices = randperm(N)
    num_train = round(Int, N * (1 - P))
    training_indices = indices[1:num_train]
    test_indices = indices[(num_train + 1):N]
    return (training_indices, test_indices)
end

function holdOut(N::Int, Pval::Real, Ptest::Real)
    training_indices, remaining_indices = holdOut(N, Pval + Ptest)
    val_indices, test_indices = holdOut(length(remaining_indices), Ptest / (Pval + Ptest))
    val_indices = remaining_indices[val_indices]
    test_indices = remaining_indices[test_indices]
    return (training_indices, val_indices, test_indices)
end

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=
    (Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0,size(trainingDataset[2],2))),
    testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=
    (Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0,size(trainingDataset[2],2))),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    maxEpochsVal::Int=20)
    
    inputs = Float32.(trainingDataset[1])
    targets = trainingDataset[2]

    numInputs = size(inputs, 2)
    numOutputs = size(targets, 2)
    
    rna = buildClassANN(numInputs, topology, numOutputs; transferFunctions=transferFunctions)
    loss(m, x, y) = (size(y,1) == 1) ? Flux.binarycrossentropy(m(x),y) : Flux.crossentropy(m(x),y)
    opt_state = Flux.setup(Adam(learningRate), rna)

    training_loss = Float32[]
    validation_loss = Float32[]
    test_loss = Float32[]

    best_rna = deepcopy(rna)
    best_val_loss = Inf32
    epochs_with_no_better_val_loss = 0

    has_validation = !isempty(validationDataset[1])
    has_test = !isempty(testDataset[1])

    # Ciclo 0
    push!(training_loss, loss(rna, inputs', targets'))
    
    if has_validation
        val_loss = loss(rna, Float32.(validationDataset[1])', validationDataset[2]')
        push!(validation_loss, val_loss)
        best_val_loss = val_loss
    end
    
    if has_test
        push!(test_loss, loss(rna, Float32.(testDataset[1])', testDataset[2]'))
    end

    # Bucle
    for i in 1:maxEpochs
        Flux.train!(loss, rna, [(inputs', targets')], opt_state)
        push!(training_loss, loss(rna, inputs', targets'))
        
        if has_test
            push!(test_loss, loss(rna, Float32.(testDataset[1])', testDataset[2]'))
        end

        if has_validation
            val_loss = loss(rna, Float32.(validationDataset[1])', validationDataset[2]')
            push!(validation_loss, val_loss)
            
            if val_loss < best_val_loss
                best_rna = deepcopy(rna)
                best_val_loss = val_loss
                epochs_with_no_better_val_loss = 0
            else
                epochs_with_no_better_val_loss += 1
            end
            
            if epochs_with_no_better_val_loss >= maxEpochsVal
                break
            end
        end
        
        if training_loss[end] <= minLoss
            break
        end
    end
    
    final_rna = has_validation ? best_rna : rna
    return final_rna, training_loss, validation_loss, test_loss
end

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=
    (Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=
    (Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    maxEpochsVal::Int=20) 
    
    training_matrix = reshape(trainingDataset[2], :, 1)
    validation_matrix = reshape(validationDataset[2], :, 1)
    test_matrix = reshape(testDataset[2], :, 1)
    
    return trainClassANN(topology, (trainingDataset[1], training_matrix); 
        validationDataset=(validationDataset[1], validation_matrix), 
        testDataset=(testDataset[1], test_matrix), 
        transferFunctions=transferFunctions, 
        maxEpochs=maxEpochs, 
        minLoss=minLoss, 
        learningRate=learningRate, 
        maxEpochsVal=maxEpochsVal)
end


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 4 --------------------------------------------
# ----------------------------------------------------------------------------------------------


function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    # Se calculan los 4 valores base comparando los vectores lógicos
    VN = sum(.!outputs .& .!targets) # Verdaderos Negativos
    FP = sum(outputs .& .!targets)   # Falsos Positivos
    FN = sum(.!outputs .& targets)  # Falsos Negativos
    VP = sum(outputs .& targets)    # Verdaderos Positivos

    # Precision y Tasa de Error
    acc = (VN + VP) / length(outputs) # Accuracy
    err = (FP + FN) / length(outputs) # Error

    # Sensibilidad y Especificidad
    sens = (VP + FN == 0) ? 1.0 : VP / (VP + FN) # Sensibilidad
    spec = (VN + FP == 0) ? 1.0 : VN / (VN + FP) # Especificidad
    vpp = (VP + FP == 0) ? 1.0 : VP / (VP + FP) # Valor Predictivo Positivo
    vnn = (VN + FN == 0) ? 1.0 : VN / (VN + FN) # Valor Predictivo Negativo

    # F1-Score con control por si sensibilidad y VPP son 0
    f1 = (sens == 0 && vpp == 0) ? 0.0 : (2 * sens * vpp) / (sens + vpp)

    # Construccion de la matriz de confusion (2x2)
    conf_matrix = [VN FP; FN VP]    

    return (acc, err, sens, spec, vpp, vnn, f1, conf_matrix)
end;


function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    # Convertimos los valores reales a booleanos usando el umbral
    return confusionMatrix(classifyOutputs(outputs; threshold=threshold), targets)
end;

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    numClasses = size(outputs, 2)

    # Comprobacion de que no tenga una sola columna
    if numClasses == 1
        return confusionMatrix(outputs[:, 1], targets[:, 1])
    end

    # Reservar memoria para las metricas
    sens = zeros(numClasses)
    spec = zeros(numClasses)
    vpp = zeros(numClasses)
    vnn = zeros(numClasses)
    f1 = zeros(numClasses)

    # Iterar por cada clase y calcular metricas individualmente
    for c in 1:numClasses
        _, _, sens[c], spec[c], vpp[c], vnn[c], f1[c], _ = confusionMatrix(outputs[:, c], targets[:, c])
    end

    # Calculo de la matriz de confusion 
    conf_matrix = targets' * outputs # Multiplicamos la matriz

    # Agregación macro o weighted
    if weighted
        weights = vec(sum(targets, dims=1)) # Peso por número de instancias de cada clase
        total_instances = sum(weights)
        sens_val = sum(sens .* weights) / total_instances
        spec_val = sum(spec .* weights) / total_instances
        vpp_val = sum(vpp .* weights) / total_instances
        vnn_val = sum(vnn .* weights) / total_instances
        f1_val = sum(f1 .* weights) / total_instances
    else
        sens_val = mean(sens)
        spec_val = mean(spec)
        vpp_val = mean(vpp)
        vnn_val = mean(vnn)
        f1_val = mean(f1)
    end

    acc = accuracy(outputs, targets)
    err = 1.0 - acc

    return (acc, err, sens_val, spec_val, vpp_val, vnn_val, f1_val, conf_matrix)
end;

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)
    # Llama a classifyOutputs y luego a la version multiclase booleana
    return confusionMatrix(classifyOutputs(outputs; threshold=threshold), targets; weighted=weighted)
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    # Asegura que todas las etiquetas esten en "classes"
    @assert(all([in(label, classes) for label in vcat(targets, outputs)])) 

    # Codifica las matrices y llama a la version multiclase
    encoded_outputs = oneHotEncoding(outputs, classes)
    encoded_targets = oneHotEncoding(targets, classes)

    return confusionMatrix(encoded_outputs, encoded_targets; weighted=weighted)
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    # Extrae las clases unicas de los targets y outputs y llama a la version anterior
    classes = unique(vcat(targets, outputs))

    return confusionMatrix(outputs, targets, classes; weighted=weighted)
end;

# Caso base: Booleano, 1D (Binaria)
function printConfusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    acc, err, sens, spec, vpp, vnn, f1, conf_matrix = confusionMatrix(outputs, targets)
    println("Matriz de Confusión (Binaria):")
    println("   $(conf_matrix[1,1]) (VN)   $(conf_matrix[1,2]) (FP)")
    println("   $(conf_matrix[2,1]) (FN)   $(conf_matrix[2,2]) (VP)")
    println("----------------------------------------")
    println("Precisión (Accuracy):     $(round(acc, digits=4))")
    println("Tasa de Error:            $(round(err, digits=4))")
    println("Sensibilidad:             $(round(sens, digits=4))")
    println("Especificidad:            $(round(spec, digits=4))")
    println("VPP (Precisión):          $(round(vpp, digits=4))")
    println("VPN:                      $(round(vnn, digits=4))")
    println("F1-Score:                 $(round(f1, digits=4))")
end;

# Caso: Real, 1D (Binaria con umbral)
function printConfusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    # Reutilizamos la lógica booleana tras aplicar el umbral
    printConfusionMatrix(classifyOutputs(outputs; threshold=threshold), targets)
end;

# Caso: Booleano, 2D (Multiclase)
function printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    acc, err, sens, spec, vpp, vnn, f1, conf_matrix = confusionMatrix(outputs, targets; weighted=weighted)
    
    println("Matriz de Confusión Multiclase ($(weighted ? "Weighted" : "Macro")):")
    # display() es ideal para matrices de Julia
    display(conf_matrix)
    
    println("\n----------------------------------------")
    println("Métricas Globales/Promedio:")
    println("Precisión Global:         $(round(acc, digits=4))")
    println("Tasa de Error Global:     $(round(err, digits=4))")
    println("Sensibilidad Media:       $(round(sens, digits=4))")
    println("Especificidad Media:      $(round(spec, digits=4))")
    println("VPP Medio:                $(round(vpp, digits=4))")
    println("VPN Medio:                $(round(vnn, digits=4))")
    println("F1-Score Medio:           $(round(f1, digits=4))")
end;

# Caso: Real, 2D (Multiclase con umbral)
function printConfusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)
    # Clasificamos las salidas continuas y llamamos a la versión de matrices booleanas
    printConfusionMatrix(classifyOutputs(outputs; threshold=threshold), targets; weighted=weighted)
end;

# Caso: Any (Etiquetas), con clases especificadas
function printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    # Obtenemos las métricas delegando en la función de cálculo
    acc, err, sens, spec, vpp, vnn, f1, conf_matrix = confusionMatrix(outputs, targets, classes; weighted=weighted)
    
    println("Matriz de Confusión para clases: ", classes)
    display(conf_matrix)
    
    println("\n----------------------------------------")
    println("Precisión: $(round(acc, digits=4)) | Error: $(round(err, digits=4)) | F1: $(round(f1, digits=4))")
end;

# Caso: Any (Etiquetas), clases automáticas
function printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    # Extraemos las clases y llamamos a la versión anterior
    classes = unique(vcat(targets, outputs))
    printConfusionMatrix(outputs, targets, classes; weighted=weighted)
end;


using SymDoME
using GeneticProgramming


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    trainingInputs, trainingTargets = trainingDataset

    # Convertir a Float64 para mayor precision en DoME
    train_in_f64 = Float64.(trainingInputs)
    test_in_f64 = Float64.(testInputs)

    # Entrenar el modelo
    model, _, _, _ = dome(train_in_f64, trainingTargets; maximumNodes=maximumNodes)

    # Evaluar el modelo
    testOutputs = evaluateTree(model, test_in_f64)
    
    #Control por si el modelo devuelve una constante
    if isa(testOutputs, Real)
        testOutputs = repeat([testOutputs], size(test_in_f64, 1))
    end

    return Float64.(testOutputs)
end;

function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    trainingInputs, trainingTargets = trainingDataset
    numClasses = size(trainingTargets, 2)

    # Si solo hay una columna, es clasificacion binaria, llamamos a la version anterior
    if numClasses == 1
        out = trainClassDoME((trainingInputs, trainingTargets[:, 1]), testInputs, maximumNodes)
        return reshape(out, :, 1)
    end

    # Estrategia "Uno contra todos": matriz para guardar salidas continuas
    testOutputs = Array{Float64,2}(undef, size(testInputs, 1), numClasses)

    # Iterar por clase
    for c in 1:numClasses
        testOutputs[:, c] = trainClassDoME((trainingInputs, trainingTargets[:, c]), testInputs, maximumNodes)
    end
    
    return testOutputs
end;


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    trainingInputs, trainingTargets = trainingDataset

    # Conocer las clases y reservar memoria para la salida final
    classes = unique(trainingTargets)
    testOutputs = Array{eltype(trainingTargets),1}(undef, size(testInputs, 1))

    # Codificar y entrenar en multiclase
    encoded_targets = oneHotEncoding(trainingTargets, classes)
    testOutputsDoME = trainClassDoME((trainingInputs, encoded_targets), testInputs, maximumNodes)

    # Clasificar salidas aplicando umbral 0
    testOutputsBool = classifyOutputs(testOutputsDoME; threshold=0.0)

    # Mapear salidas booleanas a etiquetas originales
    if length(classes) <= 2
        testOutputsBool_vec = vec(testOutputsBool)
        testOutputs[testOutputsBool_vec] .= classes[1]
        if length(classes) == 2
            testOutputs[.!testOutputsBool_vec] .= classes[2]
        end
    else
        for (i, class) in enumerate(classes)
            testOutputs[testOutputsBool[:, i]] .= class
        end
    end

    return testOutputs
end;


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 5 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random
using Random:seed!

function crossvalidation(N::Int64, k::Int64)
    # Creamos un vector con los indices de los folds (1, 2, ..., k)
    indices = repeat(1:k, outer=Int(ceil(Int, N/k)))[1:N]

    # Los desordenamos aleatoriamente
    shuffled_indices = shuffle!(indices)

    return shuffled_indices
end;

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    indices = zeros(Int64, length(targets))

    # Buscamos los indices de cada clase
    idx_true = findall(targets)
    idx_false = findall(.!targets)

    # Asignamos los folds de forma independiente para mantener la proporción
    indices[idx_true] = crossvalidation(length(idx_true), k)
    indices[idx_false] = crossvalidation(length(idx_false), k)

    return indices

end;

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    N, numClasses = size(targets)

    # Caso especial: Si es binaria con una sola columna
    if numClasses == 1
        return crossvalidation(targets[:, 1], k)
    end

    # Para multiclase, aplicamos la estrategia "Uno contra todos" y luego combinamos los resultados
    indices = zeros(Int64, N)
    for c in 1:numClasses
        idx_class = findall(targets[:, c])
        indices[idx_class] = crossvalidation(length(idx_class), k)
    end

    return indices
end;

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    return crossvalidation(oneHotEncoding(targets), k)
end;

function ANNCrossValidation(topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
    crossValidationIndices::Array{Int64,1};
    numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, validationRatio::Real=0, maxEpochsVal::Int=20)

    inputs, targets_raw = dataset
    classes = unique(targets_raw)
    numClasses = length(classes)
    k = maximum(crossValidationIndices)

    # Para almacenar las métricas de cada fold
    # (Acc, Error, Sens, Spec, VPP, VPN, F1)
    metrics_per_fold = [zeros(Float64, k) for _ in 1:7]
    global_conf_matrix = zeros(Float64, numClasses, numClasses)

    for i in 1:k
        # Separar Test (flold actual) de Entrenamiento + Validacion (resto de folds)
        idx_test = (crossValidationIndices .== i)
        idx_train_val = .!idx_test

        test_data = (inputs[idx_test, :], targets_raw[idx_test])
        train_val_inputs = inputs[idx_train_val, :]
        train_val_targets = targets_raw[idx_train_val]

        # Si hay validacion, calculamos el ratio ajustado respecto al train_val
        # Pval_adj = N * validationRatio / length(train_val)
        actual_val_ratio = (validationRatio > 0) ? (length(targets_raw)*validationRatio) / sum(idx_train_val) : 0.0

        # Almacenes para las N ejecuciones de este fold
        exec_metrics = [zeros(numExecutions) for _ in 1:7]
        exec_conf_matrices = Array{Float64, 3}(undef, numClasses, numClasses, numExecutions)

        for j in 1:numExecutions
            # Dividir en Train y Val usando holdOut
            # Necesitamos convertir targets a bool/OneHot para TrainClassANN
            if actual_val_ratio > 0
                idx_t, idx_v = holdOut(sum(idx_train_val), actual_val_ratio)
                train_set = (train_val_inputs[idx_t, :], oneHotEncoding(train_val_targets[idx_t], classes))
                val_set = (train_val_inputs[idx_v, :], oneHotEncoding(train_val_targets[idx_v], classes))
            else
                train_set = (train_val_inputs, oneHotEncoding(train_val_targets, classes))
                val_set = (Array{Float32, 2}(undef, 0, size(inputs, 2)), falses(0, size(train_set[2], 2)))
            end

            ann, _, _, _ = trainClassANN(topology, train_set; 
                validationDataset=val_set, transferFunctions=transferFunctions, 
                maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, maxEpochsVal=maxEpochsVal)
            
            outputs_test = ann(inputs[idx_test, :]')'
            targets_test = oneHotEncoding(targets_raw[idx_test], classes)

            res = confusionMatrix(outputs_test, targets_test)

            for m in 1:7; exec_metrics[m][j] = res[m]; end
            exec_conf_matrices[:, :, j] = res[8]
        end

        # Promediamos las ejecuciones del fold
        for m in 1:7; metrics_per_fold[m][i] = mean(exec_metrics[m]); end
        global_conf_matrix .+= mean(exec_conf_matrices, dims=3)[:, :, 1]
    end

    # Formatear salida
    final_result = []
    for m in 1:7
        push!(final_result, (mean(metrics_per_fold[m]), std(metrics_per_fold[m])))
    end
        
    return (final_result..., global_conf_matrix)
            
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
    
    # Función auxiliar para extraer parámetros admitiendo claves String o Symbol
    function get_hp(key_str::String, default_val)
        key_sym = Symbol(key_str)
        if haskey(modelHyperparameters, key_str)
        return modelHyperparameters[key_str]
        elseif haskey(modelHyperparameters, key_sym)
            return modelHyperparameters[key_sym]
        else
            return default_val
        end
    end
    
    if modelType == :ANN
        # Extraemos la topología, que es obligatoria y posicional
        topology = get_hp("topology", [4, 3])
        
        # Llamamos a la función respetando el orden y pasando los demás como keywords (;)
        return ANNCrossValidation(
            topology, 
            dataset, 
            crossValidationIndices;
            numExecutions   = get_hp("numExecutions", 50),
            maxEpochs       = get_hp("maxEpochs", 1000),
            learningRate    = get_hp("learningRate", 0.01),
            validationRatio = get_hp("validationRatio", 0.0),
            maxEpochsVal    = get_hp("maxEpochsVal", 20)
        )
    end

    inputs, targets, = dataset

    # Preparación de las salidas deseadas y clases
    targets_str = string.(targets)
    classes = unique(targets_str)
    num_classes = length(classes)
    num_folds = maximum(crossValidationIndices)

    # Preparación de las salidas deseadas y clases
    precisions = zeros(Float64, num_folds)
    error_rates = zeros(Float64, num_folds)
    sensitivities = zeros(Float64, num_folds)
    specificities = zeros(Float64, num_folds)
    ppvs = zeros(Float64, num_folds)
    npvs = zeros(Float64, num_folds)
    f1s = zeros(Float64, num_folds)
    
    global_cm = zeros(Int64, num_classes, num_classes)

    # Bucle de validacion cruzada
    for fold in 1:num_folds
        # Division de indices
        test_idx = (crossValidationIndices .== fold)
        train_idx = .!test_idx

        # Separar conjuntos
        train_inputs = inputs[train_idx, :]
        train_targets = targets_str[train_idx]  
        test_inputs = inputs[test_idx, :]
        test_targets = targets_str[test_idx]

        if modelType == :DoME
            max_nodes = get_hp("maximumNodes", 10)
            test_outputs = trainClassDoME((train_inputs, train_targets), test_inputs, max_nodes)
            test_outputs_str = string.(test_outputs)
        else
            # Creacion del modelo MLJ segun el hiperparametro
            if modelType == :SVC
                kernel_str = get_hp("kernel", "rbf")
                C = Float64(get_hp("C", 1.0))
                gamma = Float64(get_hp("gamma", 0.1))
                degree = Int32(get_hp("degree", 3))
                coef0 = Float64(get_hp("coef0", 0.0))

                if kernel_str == "linear"
                    kernel_obj = LIBSVM.Kernel.Linear
                elseif kernel_str == "rbf"
                    kernel_obj = LIBSVM.Kernel.RadialBasis
                elseif kernel_str == "sigmoid"
                    kernel_obj = LIBSVM.Kernel.Sigmoid
                else # poly
                    kernel_obj = LIBSVM.Kernel.Polynomial
                end

                model = SVMClassifier(kernel=kernel_obj, cost=C, gamma=gamma, degree=degree, coef0=coef0)

            elseif modelType == :DecisionTreeClassifier
                max_depth = get_hp("max_depth", 4)
                model = DTClassifier(max_depth=max_depth, rng=Random.MersenneTwister(1))
            elseif modelType == :KNeighborsClassifier
                k = get_hp("n_neighbors", 3)
                model = kNNClassifier(K=k)
            else
                error("Modelo no soportado: ", modelType)
            end

            # Creacion de la maquina y entrenamiento
            mach = machine(model, MLJ.table(train_inputs), categorical(train_targets))
            MLJ.fit!(mach, verbosity=0)

            # Prediccion y post-procesamiento
            test_outputs = MLJ.predict(mach, MLJ.table(test_inputs))
            
            if modelType == :DecisionTreeClassifier || modelType == :KNeighborsClassifier
                test_outputs_str = string.(mode.(test_outputs))
            else
                test_outputs_str = string.(test_outputs)
            end
        end

        # Calculo de metricas y acumulación de matriz de confusion
        acc, err, sens, spec, ppv, npv, f1, cm = confusionMatrix(test_outputs_str, test_targets, classes)

        precisions[fold] = acc
        error_rates[fold] = err
        sensitivities[fold] = sens
        specificities[fold] = spec
        ppvs[fold] = ppv
        npvs[fold] = npv
        f1s[fold] = f1

        global_cm .+= cm
    end

    # Retorno de la tupla con medias, desviaciones típicas y la matriz global
    return (
        (mean(precisions), std(precisions)),
        (mean(error_rates), std(error_rates)),
        (mean(sensitivities), std(sensitivities)),
        (mean(specificities), std(specificities)),
        (mean(ppvs), std(ppvs)),
        (mean(npvs), std(npvs)),
        (mean(f1s), std(f1s)),
        global_cm
    )

end;