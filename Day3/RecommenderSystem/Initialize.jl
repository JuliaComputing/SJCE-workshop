function readData()

	training = readdlm("../data/train.txt",'\t';has_header=false)
	test = readdlm("../data/test.txt",'\t';has_header=false)
	println("size of training data", size(training))
	println("size of test data", size(test))

	userCol = int(training[:,1])
	movieCol = int(training[:,2])
	ratingsCol = int(training[:,3])
	training_sparse =sparse(userCol,movieCol,ratingsCol)

	QuserCol = int(test[:,1])
	QmovieCol = int(test[:,2])
	QratingsCol = int(test[:,3])
	test_sparse=sparse(QuserCol,QmovieCol,QratingsCol)

	return (training_sparse,test_sparse)
end

