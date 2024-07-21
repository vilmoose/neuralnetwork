//Neural Network with 3 layers
type Network struct {
  inputs         int
  hiddens        int
  outputs        int
  hiddenWeights  *mat.Dense
  outputWieghts  *mat.Dense
  learningRate   float64
}
  
