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