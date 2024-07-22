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
  outputWieghts  *mat.Dense
  learningRate   float64
}
  
