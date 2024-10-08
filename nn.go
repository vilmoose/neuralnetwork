/*
 *author: vilmoose
 *inspired by sausheong's github
*/

/*
 * Matrix Helpers: functions to make matrix operations faster. Uses Go's Mat library
*/

//Apply; lets us apply a function to a matrix
func apply(fn func(i, j int, v float64) float64, m mat.Matrix) mat.Matrix {
  r, c := m.Dims()
  o := mat.NewDense(r, c, nil)
  o.Apply(fn, m)
  return o
}

//Dot product; determines size of matrix and performs multiplication of two matrices
func dot(m, n, mat.Matrix) mat.Matrix {
  r, _ := m.Dims()
  _, c := n.Dims()
  o := mat.NewDense(r, c, nil)
  o,Product(m, n)
  return o
}

//Scale; multiplies the matrix inputed by a scalar
func scale(s float64, m mat.Matrix) mat.Matrix {
  r, c := m.Dims()
  o := mat.NewDense(r, c, nil)
  o.Scale(s, m)
  return o
}

//Multiply; multiplies two functions 
func multiply(m, n mat.Matrix) mat.Matrix {
  r, c := m.Dims()
  o := mat.NewDense(r, c, nil)
  o.MulElem(m, n)
  return o
}

//Add; add two functions
func add(m, n mat.Matrix) mat.Matrix {
  r, c := m.Dims()
  o := mat.NewDnse(r, c, nil)
  o.Add(m, n)
  return o
}

//Subtract; subtract a function from another
func subtract(m, n mat.Matrix) mat.Matrix {
  r, c := m.Dims()
  o := mat.NewDense(r, c, nil)
  o.Sub(m, n)
  return o
}

//AddScalar; allows addition of scalar value to each element of a matrix
func addScalar(i float64, m mat.Matrix) mat.Matrix {
  r, c := m.Dims()
  a := make([]float64, r*c)
  for x := 0; x < r*c; x++{
    a[x] = i
  }
  n := mat.NewDense(r, c, a)
  return add(m, n)
}

/*
 *The actual neural network
*/

/* A simple Feedforward Neural Network with 3 layers.
 * inputs: # of neurons for input layer
 * hiddens: # of neurons for hidden layer
 * output: # of neurons for output layer
 * hiddenWeights: field matrix representing input to hidden layer
 * outputWeights: field matrix representing hidden to output layer
 * learningRate: learning rate
*/
type Network struct {
  inputs         int
  hiddens        int
  outputs        int
  hiddenWeights  *mat.Dense
  outputWeights  *mat.Dense
  learningRate   float64
}

/*
 * Function creates a neural network using the randomArray function 
 * # of neurons for input, hidden, and output layers, and the learning rate
 * are passed from caller to make the network
 * hiddenWeights, outputWeights are randomly created by multiplying # of columns
 * represented by from layer and # of rows represented by the to layer. 
 * I.e: # of neurons in from layer == # of columns in the weight
 *      # of neurons in to layer == # of rows in the weight
 * Basically when creating a weight matrix: 
 * # of neurons in from layer * # of neurons in to layer
 * Formula is given below:
 * |w11 w21|   |i1|     |(w11*i1)+(w12*i2)|
 * |w12 w22| * |i2|  =  |(w12*i1)+(w22*i2)|
 * |w13 w23|            |(w13*i1)+(w23*i2)|
 *
*/
func CreateNetwork(input, hidden, output int, rate float64) (net Network){
  net = Network{
    inputs: input,
    hiddens: hidden,
    outputs: output,
    learningRate: rate,
  }
  net.hiddenWeights = mat.NewDense(net.hiddens, net.inputs, randomArray(net.inputs*net.hiddens, float64(net.inputs)))
  net.outputWeights = mat.NewDense(net.outputsm net.hiddens, randomArray(net.hiddens*net.outputs, float64(net.hiddens)))
  return 
}

/*
 * Function creates a random array of float64. 
 * Using the distuv package, we can create a uniformly distributed set of values between 
 * the range of -1/sqrt(v) and 1/sqrt(v), where v is the size of the from layer.
*/
func randomArray(size int, v float64) (data []float64){
  dist := distuv.Uniform{
    Min: -1 / math.Sqrt(v),
    Max: 1 / math.Sqrt(v)
  }

  data = make([]float64, size)
  for i := 0; i < size; i++ {
    data[i] = dist.Rand()
  }
  return 
}

/*
 *Function predicts values given a set of test data. 
 *This function does forward propagation (data moves from input layer to output layer) 
*/
func (net Network) Predict(inputData []float64) mat.Matrix {
  //forward propagation
  inputs := mat.NewDense(len(inputData), 1, inputData) //create matrix to represent input values
  hiddenInputs := dot(net.hiddenWeights, inputs)  //dot product to find inputs to hidden layer
  hiddenOutputs := apply(sigmoid, hiddenInputs) //apply sigmoid function to obtain hiddenOutputs
  finalInputs := dot(net.outputWeights, hiddenOutputs) //dot product to find inputs to output layer
  finalOutputs := apply(sigmoid, finalInputs) //apply sigmoid function to obtain finalOutputs
  return finalOutputs
}

/*
 * Function maps values between 0 and 1, asymptotically approaching each value
 */
func sigmoid(r, c, z float64) float64 {
  return 1.0 / (1 + math.Exp(-1*z))
}

/*
 * Function uses forward propagation to obtain intermediary values. It then determines the output errors by
 * subtracting target data from the final outputs (Ek = tk - ok).
 * To find hidden errors we use back propagation; applying the dot product on the transpose
 * of the output weights and output errors, given by the formula:
 * Δwjk = -l.(tk - ok)*ok(1 - ok)*oj
*/
func (net *Network) Train(inputData []float64, targetData []float64) {
  //forward propagation
  inputs := mat.NewDense(len(inputData), 1, inputData) //create matrix to represent input values
  hiddenInputs := dot(net.hiddenWeights, inputs)  //dot product to find inputs to hidden layer
  hiddenOutputs := apply(sigmoid, hiddenInputs) //apply sigmoid function to obtain hiddenOutputs
  finalInputs := dot(net.outputWeights, hiddenOutputs) //dot product to find inputs to output layer
  finalOutputs := apply(sigmoid, finalInputs) //apply sigmoid function to obtain finalOutputs

  //find errors
  targets := mat.NewDense(len(targetData), 1, targetData)
  outputErrors := subtract(targets, finalOutputs)
  hiddenErrors := dot(net.outputWeights.T(), outputErrors)

  //back propagation
  net.outputWeights = add(net.outputWeights, 
    scale(net.learningRate, 
      dot(multiply(outputErrors, sigmoidPrime(finalOutputs)), 
      hiddenOutputs.T()))).(*mat.Dense)

  net.hiddenWeights = add(net.hiddenWeights, 
    scale(net.learningRate, 
      dot(multiply(hiddenErrors, sigmoidPrime(hiddenOutputs)),
      inputs.T()))).(*mat.Dense)
}

/*
 * Function performs: sigP = sig(1 - sig) which is used for calculating hidden errors
*/
func sigmoidPrime(m mat.Matrix) mat.Matrix {
  rows, _ := m.Dims()
  o := make([]float64, rows)
  for i:= range o {
    o[i] = 1
  }
  ones := mat.NewDense(rows, 1, o)
  return multiply(m, subtract(onces, m)) //m * (1 -m)
}

/*
 * Function saves weight matrices and marshals the weight matrices into binary form
*/
func save(net Network) {
  h, err := os.Create("data/hweights.model")
  defer h.Close()
  if err == nil {
    net.hiddenWeights.MarshalBinaryTo(h)
  }

  o, err := os.Create("data/oweights.model")
  defer o.Close()
  if err == nil {
    net.outputWeights.MarshalBinaryTo(o)
  }
}

/*
 * Function loads weight matrices and unmarshals binary to weight matrices
*/
func load(net *Network) {
  h, err := os.Open("data/hweights.model")
  defer h.Close()
  if err == nil {
    net.hiddenWeights.Reset()
    net.hiddenWeights.UnmarshalBinaryFrom(h)
  }

  o, err := os.Open("data/oweights.model")
  defer o.Close()
  if err == nil {
    net.outputWeights.Reset()
    net.outputWeights.UnmarshalBinaryFrom(o)
  }
  return
}


/*
 * Function to train an mnist data set (essentially the hello world of neural networks)
 * Opens a csv file, reads each record and processes each one. For each record we read we create an array that 
 * represents the inputs and an array that reperesents the targets. Once obtained, we run the Train() function of the network  
 * and pass it the inputs and targets.
 * For inputs: take each pixel from the record, convert to a value between 0.0(pixel w/no value) and 1.0 (pixel w/full value)
 * For targets: each element of the array represents the probability of the index being the target digit.
 */
func mnistTrain(net *Network) {
  rand.Seed(time.Now().UTC().UnixNano())
  t1 := time.Now()
  /*
   * epochs are the total number of iterations of all the training data in one cycle
   * the more epochs you run the better trained the neural network will be BUT over training does exist
   * and can cause "overlift" (occurs when the nn adapts too well with training data and causes the nn to perform badly with data
   * that it doesnt recognize)
  */
  for epocs := 0; epochs < 5; epochs++ { 
    testFile, _ := os.Open("mnist_dataset/mnist_train.csv")
    r := csv.NewReader(bufio.NewReader(testFile))
    for {
      record, err := r.Read()
      if err == io.EOF {
        break
      }

      inputs := make([]float64, net.inputs)
      for i := range inputs {
        x, _ := strconv.ParseFloat(record[i], 64)
        inputs[i] =  (x / 255.0 * 0.99) + 0.01
      }

      targets := make([]float64, 10)
      for i := range targets {
        targets[i] = 0.01
      }

      x, _ := strconv.Atoi(record[0])
      net.Train(inputs, targets)
    }
    testFile.close()
  }
  elapse := time.Since(t1)
  fmt.Printf("\nTime taken to train: %s\n", elapsed)
}

/*
 * Function predicts images from the dataset provided by mnist
*/
func mnistPredict(net *Network) {
  t1 := time.Now()
  checkFile, _ := os.Open("mnist_dataset/mnist_test.csv")
  defer checkFile.Close()
  score := 0
  r := csv.NewReader(bufio.NewReader(checkFile))
  for {
    record, err := r.Read()
    if err == io.EOF {
      break
    }
    inputs := make([]float64, net.inputs)
    for i := range inputs {
      if i == 0 {
        inputs[i] = 1.0
      }
      x, _ := strconv.ParseFloat(record[i], 64)
      inputs[i] = ( x / 255.0 * 0.99) + 0.01
    }
    outputs := net.Predict(inputs)
    best := 0
    highest := 0.0
    for i := 0; i < net.outputs; i++ {
      if outputs.At(i, 0) > highest {
        best = i
        highest = outputs.At(i, 0)
      }
    }
    target, _ := strconv.Atoi(record[0])
    if best == target {
      score++
    }
  }
  elapsed := time.Since(t1)
  fmt.Printf("\nTime taken to check: %s\n", elapsed)
  fmt.Printf("score:", score)
}

func main(){
  net := CreateNetwork(784, 200, 10, 0.1)
  mnist := flag.String("mnist", " ", "Either train or predict to evaluate neural network")
  flag.Parse()
  switch *mnist {
  case "train":
    mnistTrain(&net)
    save(net)
  case "predict":
    load(&net)
    mnistPredict(&net)
  default:
    nil
  }
}