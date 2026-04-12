using FileIO
using Images
using Random # Necesario para mezclar las imágenes aleatoriamente

include("ejercicio6.jl")

#  Función para extraer las características de una imagen
function extract_features(img, img_size=(32, 32))
    img_resized = imresize(img, img_size)
    
    matrizR = Float32.(red.(img_resized))
    matrizG = Float32.(green.(img_resized))
    matrizB = Float32.(blue.(img_resized))
    
    return vcat(vec(matrizR), vec(matrizG), vec(matrizB))
end

# Función para leer las carpetas y balancear las clases
function load_simpsons_binary(flanders_folder::String, apu_folder::String, img_size=(32, 32))
    flanders_inputs = []
    apu_inputs = []
    
    println("Cargando imágenes de Ned Flanders...")
    for filename in readdir(flanders_folder)
        if endswith(lowercase(filename), ".jpg") || endswith(lowercase(filename), ".png")
            img = FileIO.load(joinpath(flanders_folder, filename)) 
            push!(flanders_inputs, extract_features(img, img_size))
        end
    end
    
    println("Cargando imágenes de Apu...")
    for filename in readdir(apu_folder)
        if endswith(lowercase(filename), ".jpg") || endswith(lowercase(filename), ".png")
            img = FileIO.load(joinpath(apu_folder, filename)) 
            push!(apu_inputs, extract_features(img, img_size))
        end
    end
    
    min_imgs = min(length(flanders_inputs), length(apu_inputs))
    println("\nBalanceando el dataset a $min_imgs imágenes por clase...")
    
    # Mezclamos aleatoriamente para no coger siempre las primeras
    shuffle!(flanders_inputs)
    shuffle!(apu_inputs)
    
    # Recortamos ambos arrays para que tengan exactamente el mismo tamaño (en cas de añadir más imágenes
    # a alguna de las carpetas en el futuro)
    flanders_inputs = flanders_inputs[1:min_imgs]
    apu_inputs = apu_inputs[1:min_imgs]
    
    # Juntamos los inputs
    inputs_list = vcat(flanders_inputs, apu_inputs)
    # Creamos los targets (true = Flanders, false = Apu)
    targets_list = vcat(trues(min_imgs), falses(min_imgs))
    
    # Convertimos a matriz 2D (Patrones x Atributos)
    inputs_matrix = Matrix{Float32}(undef, length(inputs_list), length(inputs_list[1]))
    for i in 1:length(inputs_list)
        inputs_matrix[i, :] = inputs_list[i]
    end
    
    println("¡Dataset final listo! Imágenes totales: ", length(targets_list))
    return inputs_matrix, targets_list
end

# Cargamos el dataset desde las carpetas
inputs, targets = load_simpsons_binary("ned_flanders", "apu_nahasapeemapetilon", (32, 32))

# Dividimos en 5 particiones para Validación Cruzada
println("\nCalculando particiones para validación cruzada...")
indices_cv = crossvalidation(targets, 5)

# Entrenamos y evaluamos usando KNN
println("\nIniciando Validación Cruzada con KNN...")
hyper_knn = Dict("n_neighbors" => 3)
resultados_knn = modelCrossValidation(:KNeighborsClassifier, hyper_knn, (inputs, targets), indices_cv)

println("\n¡Validación Cruzada Terminada!")
println("==================================================")
println("Resultados (Media, Desviación Típica):")
println("1. Precisión (Accuracy): ", resultados_knn[1])
println("2. Tasa de Error:        ", resultados_knn[2])
println("3. Sensibilidad:         ", resultados_knn[3])
println("4. Especificidad:        ", resultados_knn[4])
println("5. Valor Pred. Positivo: ", resultados_knn[5])
println("6. Valor Pred. Negativo: ", resultados_knn[6])
println("7. F1-Score:             ", resultados_knn[7])
println("==================================================")
