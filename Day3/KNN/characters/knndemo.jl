#Pkg.add("Images")
#Pkg.add("DataFrames")
using Images, Colors
using DataFrames

#typeData could be either "train" or "test.
#labelsInfo should contain the IDs of each image to be read
#The images in the trainResized and testResized data files
#are 20x20 pixels, so imageSize is set to 400.
#path should be set to the llabelsInfoTrain = readtable("$(path)/trainLabels.csv")ocation of the data files.
function read_data_sv(typeData, labelsInfo, imageSize, path)
    x = zeros(size(labelsInfo, 1), imageSize)
    for (index, idImage) in enumerate(labelsInfo[:ID])
        nameFile = "$(path)/$(typeData)Resized/$(idImage).Bmp"
        println(idImage)
        img = imread(nameFile)
        imgg = convert(Image{Gray}, img)
        temp = float(imgg.data)
        x[index, :] = reshape(temp, 1, imageSize)
    end
    return x
end

@everywhere function euclidean_distance(a, b)
 distance = 0.0 
 for index in 1:size(a, 1) 
  distance += (a[index]-b[index]) * (a[index]-b[index])
 end
 return distance
end

@everywhere function get_k_nearest_neighbors(x, i, k)
 nRows, nCols = size(x)
 imageI = Array(Float32, nRows)
 for index in 1:nRows
  imageI[index] = x[index, i]
 end
 imageJ = Array(Float32, nRows)
 distances = Array(Float32, nCols) 
 for j in 1:nCols
  for index in 1:nRows
   imageJ[index] = x[index, j]
  end
  distances[j] = euclidean_distance(imageI, imageJ)
 end
 sortedNeighbors = sortperm(distances)
 kNearestNeighbors = sortedNeighbors[2:k+1]
 return kNearestNeighbors
end 

@everywhere function assign_label(x, y, k, i)
 kNearestNeighbors = get_k_nearest_neighbors(x, i, k) 
 counts = Dict{Int, Int}() 
 highestCount = 0
 mostPopularLabel = 0
 for n in kNearestNeighbors
  labelOfN = y[n]
  if !haskey(counts, labelOfN)
   counts[labelOfN] = 0
  end
  counts[labelOfN] += 1 
  if counts[labelOfN] > highestCount
   highestCount = counts[labelOfN]
   mostPopularLabel = labelOfN
  end 
 end
 return mostPopularLabel
end

#Similar to function assign_label.
#Only changes are commented
@everywhere function assign_label_each_k(x, y, maxK, i)
 kNearestNeighbors = get_k_nearest_neighbors(x, i, maxK) 

 #The next array will keep the labels for each value of k
 labelsK = zeros(Int, 1, maxK) 

 counts = Dict{Int, Int}()
 highestCount = 0
 mostPopularLabel = 0

 #We need to keep track of the current value of k
 for (k, n) in enumerate(kNearestNeighbors)
  labelOfN = y[n]
  if !haskey(counts, labelOfN)
   counts[labelOfN] = 0
  end
  counts[labelOfN] += 1
  if counts[labelOfN] > highestCount
   highestCount = counts[labelOfN]
   mostPopularLabel = labelOfN  
  end
  #Save current most popular label 
  labelsK[k] = mostPopularLabel
 end
 #Return vector of labels for each k
 return labelsK
end

@everywhere function get_k_nearest_neighbors(xTrain, imageI, k)
 nRows, nCols = size(xTrain) 
 imageJ = Array(Float32, nRows)
 distances = Array(Float32, nCols) 
 for j in 1:nCols
  for index in 1:nRows
   imageJ[index] = xTrain[index, j]
  end
  distances[j] = euclidean_distance(imageI, imageJ)
 end
 sortedNeighbors = sortperm(distances)
 kNearestNeighbors = sortedNeighbors[1:k]
 return kNearestNeighbors
end 

@everywhere function assign_label(xTrain, yTrain, k, imageI)
 kNearestNeighbors = get_k_nearest_neighbors(xTrain, imageI, k) 
 counts = Dict{Int, Int}() 
 highestCount = 0
 mostPopularLabel = 0
 for n in kNearestNeighbors
  labelOfN = yTrain[n]
  if !haskey(counts, labelOfN)
   counts[labelOfN] = 0
  end
  counts[labelOfN] += 1 #add one to the count
  if counts[labelOfN] > highestCount
   highestCount = counts[labelOfN]
   mostPopularLabel = labelOfN
  end 
 end
 return mostPopularLabel
end
