//
// !!!THIS CODE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND!!!
//
// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

public enum Raw {

static let generatedTensorFlowVersion = "1.8.0"
static let generatedTensorFlowGitVersion = "v1.8.0-0-g93bc2e2072"

public enum A: String {
  case apples = "apples"
  case oranges = "oranges"
}

public enum DataFormat: String {
  case nchw = "NCHW"
  case nhwc = "NHWC"
}

public enum DataFormat1: String {
  case ncdhw = "NCDHW"
  case ndhwc = "NDHWC"
}

public enum DataFormat2: String {
  case nchw = "NCHW"
  case nchwVectC = "NCHW_VECT_C"
  case nhwc = "NHWC"
}

public enum DensityUnit: String {
  case cm = "cm"
  case in_ = "in"
}

public enum Direction: String {
  case bidirectional = "bidirectional"
  case unidirectional = "unidirectional"
}

public enum FinalOp: String {
  case div = "Div"
  case id = "Id"
}

public enum Format: String {
  case empty = ""
  case grayscale = "grayscale"
  case rgb = "rgb"
}

public enum InputMode: String {
  case autoSelect = "auto_select"
  case linearInput = "linear_input"
  case skipInput = "skip_input"
}

public enum LossType: String {
  case hingeLoss = "hinge_loss"
  case logisticLoss = "logistic_loss"
  case smoothHingeLoss = "smooth_hinge_loss"
  case squaredLoss = "squared_loss"
}

public enum MergeOp: String {
  case add = "Add"
  case max = "Max"
  case min = "Min"
  case mul = "Mul"
}

public enum Method: String {
  case bilinear = "bilinear"
}

public enum Mode: String {
  case minCombined = "MIN_COMBINED"
  case minFirst = "MIN_FIRST"
  case scaled = "SCALED"
}

public enum Mode3: String {
  case reflect = "REFLECT"
  case symmetric = "SYMMETRIC"
}

public enum Padding: String {
  case same = "SAME"
  case valid = "VALID"
}

public enum RnnMode: String {
  case gru = "gru"
  case lstm = "lstm"
  case rnnRelu = "rnn_relu"
  case rnnTanh = "rnn_tanh"
}

public enum RoundMode: String {
  case halfAwayFromZero = "HALF_AWAY_FROM_ZERO"
  case halfToEven = "HALF_TO_EVEN"
}

@_inlineable @inline(__always)
public static func a(
) -> Tensor<Float> {
  return #tfop("A")
}

// Raise a exception to abort the process when called.
//
// If exit_without_error is true, the process will exit normally,
// otherwise it will exit with a SIGABORT signal.
//
// Returns nothing but an exception.
//
// - Attr error_msg: A string which is the message associated with the exception.
@_inlineable @inline(__always)
public static func abort(
  errorMsg: String,
  exitWithoutError: Bool = false
) {
  return #tfop("Abort",
    error_msg: errorMsg,
    exit_without_error: exitWithoutError)
}

// Computes the absolute value of a tensor.
//
// Given a tensor `x`, this operation returns a tensor containing the absolute
// value of each element in `x`. For example, if x is an input element and y is
// an output element, this operation computes \\(y = |x|\\).
@_inlineable @inline(__always)
public static func abs<T: Numeric>(
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Abs",
    x,
    T: T.self)
}

// Computes acos of x element-wise.
@_inlineable @inline(__always)
public static func acos<T: Numeric>(
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Acos",
    x,
    T: T.self)
}

// Computes inverse hyperbolic cosine of x element-wise.
@_inlineable @inline(__always)
public static func acosh<T: BinaryFloatingPoint>(
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Acosh",
    x,
    T: T.self)
}

// Returns x + y element-wise.
//
// *NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@_inlineable @inline(__always)
public static func add<T: Numeric>(
  x: Tensor<T>,
  y: Tensor<T>
) -> Tensor<T> {
  return #tfop("Add",
    x,
    y,
    T: T.self)
}

// Add an `N`-minibatch `SparseTensor` to a `SparseTensorsMap`, return `N` handles.
//
// A `SparseTensor` of rank `R` is represented by three tensors: `sparse_indices`,
// `sparse_values`, and `sparse_shape`, where
//
// ```sparse_indices.shape[1] == sparse_shape.shape[0] == R```
//
// An `N`-minibatch of `SparseTensor` objects is represented as a `SparseTensor`
// having a first `sparse_indices` column taking values between `[0, N)`, where
// the minibatch size `N == sparse_shape[0]`.
//
// The input `SparseTensor` must have rank `R` greater than 1, and the first
// dimension is treated as the minibatch dimension.  Elements of the `SparseTensor`
// must be sorted in increasing order of this first dimension.  The stored
// `SparseTensor` objects pointed to by each row of the output `sparse_handles`
// will have rank `R-1`.
//
// The `SparseTensor` values can then be read out as part of a minibatch by passing
// the given keys as vector elements to `TakeManySparseFromTensorsMap`.  To ensure
// the correct `SparseTensorsMap` is accessed, ensure that the same
// `container` and `shared_name` are passed to that Op.  If no `shared_name`
// is provided here, instead use the *name* of the Operation created by calling
// `AddManySparseToTensorsMap` as the `shared_name` passed to
// `TakeManySparseFromTensorsMap`.  Ensure the Operations are colocated.
//
// - Parameters:
//   - sparse_indices: 2-D.  The `indices` of the minibatch `SparseTensor`.
//     `sparse_indices[:, 0]` must be ordered values in `[0, N)`.
//   - sparse_values: 1-D.  The `values` of the minibatch `SparseTensor`.
//   - sparse_shape: 1-D.  The `shape` of the minibatch `SparseTensor`.
//     The minibatch size `N == sparse_shape[0]`.
//
// - Attrs:
//   - container: The container name for the `SparseTensorsMap` created by this op.
//   - shared_name: The shared name for the `SparseTensorsMap` created by this op.
//     If blank, the new Operation's unique name is used.
//
// - Output sparse_handles: 1-D.  The handles of the `SparseTensor` now stored in the
//   `SparseTensorsMap`.  Shape: `[N]`.
@_inlineable @inline(__always)
public static func addManySparseToTensorsMap<T: Numeric>(
  sparseIndices: Tensor<Int64>,
  sparseValues: Tensor<T>,
  sparseShape: Tensor<Int64>,
  container: String,
  sharedName: String
) -> Tensor<Int64> {
  return #tfop("AddManySparseToTensorsMap",
    sparseIndices,
    sparseValues,
    sparseShape,
    T: T.self,
    container: container,
    shared_name: sharedName)
}

// Add all input tensors element wise.
//
// - Parameter inputs: Must all be the same size and shape.
@_inlineable @inline(__always)
public static func addN<T: Numeric>(
  inputs: [Tensor<T>]
) -> Tensor<T> {
  return #tfop("AddN",
    inputs,
    T: T.self)
}

// Add a `SparseTensor` to a `SparseTensorsMap` return its handle.
//
// A `SparseTensor` is represented by three tensors: `sparse_indices`,
// `sparse_values`, and `sparse_shape`.
//
// This operator takes the given `SparseTensor` and adds it to a container
// object (a `SparseTensorsMap`).  A unique key within this container is generated
// in the form of an `int64`, and this is the value that is returned.
//
// The `SparseTensor` can then be read out as part of a minibatch by passing
// the key as a vector element to `TakeManySparseFromTensorsMap`.  To ensure
// the correct `SparseTensorsMap` is accessed, ensure that the same
// `container` and `shared_name` are passed to that Op.  If no `shared_name`
// is provided here, instead use the *name* of the Operation created by calling
// `AddSparseToTensorsMap` as the `shared_name` passed to
// `TakeManySparseFromTensorsMap`.  Ensure the Operations are colocated.
//
// - Parameters:
//   - sparse_indices: 2-D.  The `indices` of the `SparseTensor`.
//   - sparse_values: 1-D.  The `values` of the `SparseTensor`.
//   - sparse_shape: 1-D.  The `shape` of the `SparseTensor`.
//
// - Attrs:
//   - container: The container name for the `SparseTensorsMap` created by this op.
//   - shared_name: The shared name for the `SparseTensorsMap` created by this op.
//     If blank, the new Operation's unique name is used.
//
// - Output sparse_handle: 0-D.  The handle of the `SparseTensor` now stored in the
//   `SparseTensorsMap`.
@_inlineable @inline(__always)
public static func addSparseToTensorsMap<T: Numeric>(
  sparseIndices: Tensor<Int64>,
  sparseValues: Tensor<T>,
  sparseShape: Tensor<Int64>,
  container: String,
  sharedName: String
) -> Tensor<Int64> {
  return #tfop("AddSparseToTensorsMap",
    sparseIndices,
    sparseValues,
    sparseShape,
    T: T.self,
    container: container,
    shared_name: sharedName)
}

// Returns x + y element-wise.
//
// *NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@_inlineable @inline(__always)
public static func addV2<T: Numeric>(
  x: Tensor<T>,
  y: Tensor<T>
) -> Tensor<T> {
  return #tfop("AddV2",
    x,
    y,
    T: T.self)
}

// Deprecated. Disallowed in GraphDef version >= 2.
@_inlineable @inline(__always)
public static func adjustContrast<T: Numeric>(
  images: Tensor<T>,
  contrastFactor: Tensor<Float>,
  minValue: Tensor<Float>,
  maxValue: Tensor<Float>
) -> Tensor<Float> {
  return #tfop("AdjustContrast",
    images,
    contrastFactor,
    minValue,
    maxValue,
    T: T.self)
}

// Adjust the contrast of one or more images.
//
// `images` is a tensor of at least 3 dimensions.  The last 3 dimensions are
// interpreted as `[height, width, channels]`.  The other dimensions only
// represent a collection of images, such as `[batch, height, width, channels].`
//
// Contrast is adjusted independently for each channel of each image.
//
// For each channel, the Op first computes the mean of the image pixels in the
// channel and then adjusts each component of each pixel to
// `(x - mean) * contrast_factor + mean`.
//
// - Parameters:
//   - images: Images to adjust.  At least 3-D.
//   - contrast_factor: A float multiplier for adjusting contrast.
//
// - Output output: The contrast-adjusted image or images.
@_inlineable @inline(__always)
public static func adjustContrastv2(
  images: Tensor<Float>,
  contrastFactor: Tensor<Float>
) -> Tensor<Float> {
  return #tfop("AdjustContrastv2",
    images,
    contrastFactor)
}

// Adjust the hue of one or more images.
//
// `images` is a tensor of at least 3 dimensions.  The last dimension is
// interpretted as channels, and must be three.
//
// The input image is considered in the RGB colorspace. Conceptually, the RGB
// colors are first mapped into HSV. A delta is then applied all the hue values,
// and then remapped back to RGB colorspace.
//
// - Parameters:
//   - images: Images to adjust.  At least 3-D.
//   - delta: A float delta to add to the hue.
//
// - Output output: The hue-adjusted image or images.
@_inlineable @inline(__always)
public static func adjustHue(
  images: Tensor<Float>,
  delta: Tensor<Float>
) -> Tensor<Float> {
  return #tfop("AdjustHue",
    images,
    delta)
}

// Adjust the saturation of one or more images.
//
// `images` is a tensor of at least 3 dimensions.  The last dimension is
// interpretted as channels, and must be three.
//
// The input image is considered in the RGB colorspace. Conceptually, the RGB
// colors are first mapped into HSV. A scale is then applied all the saturation
// values, and then remapped back to RGB colorspace.
//
// - Parameters:
//   - images: Images to adjust.  At least 3-D.
//   - scale: A float scale to add to the saturation.
//
// - Output output: The hue-adjusted image or images.
@_inlineable @inline(__always)
public static func adjustSaturation(
  images: Tensor<Float>,
  scale: Tensor<Float>
) -> Tensor<Float> {
  return #tfop("AdjustSaturation",
    images,
    scale)
}

// Computes the "logical and" of elements across dimensions of a tensor.
//
// Reduces `input` along the dimensions given in `axis`. Unless
// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
// `axis`. If `keep_dims` is true, the reduced dimensions are
// retained with length 1.
//
// - Parameters:
//   - input: The tensor to reduce.
//   - reduction_indices: The dimensions to reduce. Must be in the range
//     `[-rank(input), rank(input))`.
//
// - Attr keep_dims: If true, retain reduced dimensions with length 1.
//
// - Output output: The reduced tensor.
@_inlineable @inline(__always)
public static func all<Tidx: BinaryInteger>(
  input: Tensor<Bool>,
  reductionIndices: Tensor<Tidx>,
  keepDims: Bool = false
) -> Tensor<Bool> {
  return #tfop("All",
    input,
    reductionIndices,
    Tidx: Tidx.self,
    keep_dims: keepDims)
}

// Generates labels for candidate sampling with a learned unigram distribution.
//
// See explanations of candidate sampling and the data formats at
// go/candidate-sampling.
//
// For each batch, this op picks a single set of sampled candidate labels.
//
// The advantages of sampling candidates per-batch are simplicity and the
// possibility of efficient dense matrix multiplication. The disadvantage is that
// the sampled candidates must be chosen independently of the context and of the
// true labels.
//
// - Parameter true_classes: A batch_size * num_true matrix, in which each row contains the
//   IDs of the num_true target_classes in the corresponding original label.
//
// - Attrs:
//   - num_true: Number of true labels per context.
//   - num_sampled: Number of candidates to produce.
//   - unique: If unique is true, we sample with rejection, so that all sampled
//     candidates in a batch are unique. This requires some approximation to
//     estimate the post-rejection sampling probabilities.
//   - seed: If either seed or seed2 are set to be non-zero, the random number
//     generator is seeded by the given seed.  Otherwise, it is seeded by a
//     random seed.
//   - seed2: An second seed to avoid seed collision.
//
// - Outputs:
//   - sampled_candidates: A vector of length num_sampled, in which each element is
//     the ID of a sampled candidate.
//   - true_expected_count: A batch_size * num_true matrix, representing
//     the number of times each candidate is expected to occur in a batch
//     of sampled candidates. If unique=true, then this is a probability.
//   - sampled_expected_count: A vector of length num_sampled, for each sampled
//     candidate representing the number of times the candidate is expected
//     to occur in a batch of sampled candidates.  If unique=true, then this is a
//     probability.
@_inlineable @inline(__always)
public static func allCandidateSampler(
  trueClasses: Tensor<Int64>,
  numTrue: Int,
  numSampled: Int,
  unique: Bool,
  seed: Int = 0,
  seed2: Int = 0
) -> (Tensor<Int64>, Tensor<Float>, Tensor<Float>) {
  return #tfop("AllCandidateSampler",
    trueClasses,
    num_true: numTrue,
    num_sampled: numSampled,
    unique: unique,
    seed: seed,
    seed2: seed2)
}

// Returns the argument of a complex number.
//
// Given a tensor `input` of complex numbers, this operation returns a tensor of
// type `float` that is the argument of each element in `input`. All elements in
// `input` must be complex numbers of the form \\(a + bj\\), where *a*
// is the real part and *b* is the imaginary part.
//
// The argument returned by this operation is of the form \\(atan2(b, a)\\).
//
// For example:
//
// ```
// # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
// tf.angle(input) ==> [2.0132, 1.056]
// ```
//
// @compatibility(numpy)
// Equivalent to np.angle.
// @end_compatibility
@_inlineable @inline(__always)
public static func angle<T: Numeric, Tout: BinaryFloatingPoint>(
  input: Tensor<T>
) -> Tensor<Tout> {
  return #tfop("Angle",
    input,
    T: T.self,
    Tout: Tout.self)
}

// Computes the "logical or" of elements across dimensions of a tensor.
//
// Reduces `input` along the dimensions given in `axis`. Unless
// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
// `axis`. If `keep_dims` is true, the reduced dimensions are
// retained with length 1.
//
// - Parameters:
//   - input: The tensor to reduce.
//   - reduction_indices: The dimensions to reduce. Must be in the range
//     `[-rank(input), rank(input))`.
//
// - Attr keep_dims: If true, retain reduced dimensions with length 1.
//
// - Output output: The reduced tensor.
@_inlineable @inline(__always)
public static func any<Tidx: BinaryInteger>(
  input: Tensor<Bool>,
  reductionIndices: Tensor<Tidx>,
  keepDims: Bool = false
) -> Tensor<Bool> {
  return #tfop("Any",
    input,
    reductionIndices,
    Tidx: Tidx.self,
    keep_dims: keepDims)
}

// Update '*var' according to the adadelta scheme.
//
// accum = rho() * accum + (1 - rho()) * grad.square();
// update = (update_accum + epsilon).sqrt() * (accum + epsilon()).rsqrt() * grad;
// update_accum = rho() * update_accum + (1 - rho()) * update.square();
// var -= update;
//
// - Parameters:
//   - var: Should be from a Variable().
//   - accum: Should be from a Variable().
//   - accum_update: Should be from a Variable().
//   - lr: Scaling factor. Must be a scalar.
//   - rho: Decay factor. Must be a scalar.
//   - epsilon: Constant factor. Must be a scalar.
//   - grad: The gradient.
//
// - Attr use_locking: If True, updating of the var, accum and update_accum tensors will be protected by
//   a lock; otherwise the behavior is undefined, but may exhibit less contention.
//
// - Output out: Same as "var".
@_inlineable @inline(__always)
public static func applyAdadelta<T: Numeric>(
  var_: Tensor<T>,
  accum: Tensor<T>,
  accumUpdate: Tensor<T>,
  lr: Tensor<T>,
  rho: Tensor<T>,
  epsilon: Tensor<T>,
  grad: Tensor<T>,
  useLocking: Bool = false
) -> Tensor<T> {
  return #tfop("ApplyAdadelta",
    var_,
    accum,
    accumUpdate,
    lr,
    rho,
    epsilon,
    grad,
    T: T.self,
    use_locking: useLocking)
}

// Update '*var' according to the adagrad scheme.
//
// accum += grad * grad
// var -= lr * grad * (1 / sqrt(accum))
//
// - Parameters:
//   - var: Should be from a Variable().
//   - accum: Should be from a Variable().
//   - lr: Scaling factor. Must be a scalar.
//   - grad: The gradient.
//
// - Attr use_locking: If `True`, updating of the var and accum tensors will be protected
//   by a lock; otherwise the behavior is undefined, but may exhibit less
//   contention.
//
// - Output out: Same as "var".
@_inlineable @inline(__always)
public static func applyAdagrad<T: Numeric>(
  var_: Tensor<T>,
  accum: Tensor<T>,
  lr: Tensor<T>,
  grad: Tensor<T>,
  useLocking: Bool = false
) -> Tensor<T> {
  return #tfop("ApplyAdagrad",
    var_,
    accum,
    lr,
    grad,
    T: T.self,
    use_locking: useLocking)
}

// Update '*var' according to the proximal adagrad scheme.
//
// - Parameters:
//   - var: Should be from a Variable().
//   - gradient_accumulator: Should be from a Variable().
//   - gradient_squared_accumulator: Should be from a Variable().
//   - grad: The gradient.
//   - lr: Scaling factor. Must be a scalar.
//   - l1: L1 regularization. Must be a scalar.
//   - l2: L2 regularization. Must be a scalar.
//   - global_step: Training step number. Must be a scalar.
//
// - Attr use_locking: If True, updating of the var and accum tensors will be protected by
//   a lock; otherwise the behavior is undefined, but may exhibit less contention.
//
// - Output out: Same as "var".
@_inlineable @inline(__always)
public static func applyAdagradDA<T: Numeric>(
  var_: Tensor<T>,
  gradientAccumulator: Tensor<T>,
  gradientSquaredAccumulator: Tensor<T>,
  grad: Tensor<T>,
  lr: Tensor<T>,
  l1: Tensor<T>,
  l2: Tensor<T>,
  globalStep: Tensor<Int64>,
  useLocking: Bool = false
) -> Tensor<T> {
  return #tfop("ApplyAdagradDA",
    var_,
    gradientAccumulator,
    gradientSquaredAccumulator,
    grad,
    lr,
    l1,
    l2,
    globalStep,
    T: T.self,
    use_locking: useLocking)
}

// Update '*var' according to the Adam algorithm.
//
// $$lr_t := \text{learning_rate} * \sqrt{(1 - beta_2^t) / (1 - beta_1^t)}$$
// $$m_t := beta_1 * m_{t-1} + (1 - beta_1) * g$$
// $$v_t := beta_2 * v_{t-1} + (1 - beta_2) * g * g$$
// $$variable := variable - lr_t * m_t / (\sqrt{v_t} + \epsilon)$$
//
// - Parameters:
//   - var: Should be from a Variable().
//   - m: Should be from a Variable().
//   - v: Should be from a Variable().
//   - beta1_power: Must be a scalar.
//   - beta2_power: Must be a scalar.
//   - lr: Scaling factor. Must be a scalar.
//   - beta1: Momentum factor. Must be a scalar.
//   - beta2: Momentum factor. Must be a scalar.
//   - epsilon: Ridge term. Must be a scalar.
//   - grad: The gradient.
//
// - Attrs:
//   - use_locking: If `True`, updating of the var, m, and v tensors will be protected
//     by a lock; otherwise the behavior is undefined, but may exhibit less
//     contention.
//   - use_nesterov: If `True`, uses the nesterov update.
//
// - Output out: Same as "var".
@_inlineable @inline(__always)
public static func applyAdam<T: Numeric>(
  var_: Tensor<T>,
  m: Tensor<T>,
  v: Tensor<T>,
  beta1Power: Tensor<T>,
  beta2Power: Tensor<T>,
  lr: Tensor<T>,
  beta1: Tensor<T>,
  beta2: Tensor<T>,
  epsilon: Tensor<T>,
  grad: Tensor<T>,
  useLocking: Bool = false,
  useNesterov: Bool = false
) -> Tensor<T> {
  return #tfop("ApplyAdam",
    var_,
    m,
    v,
    beta1Power,
    beta2Power,
    lr,
    beta1,
    beta2,
    epsilon,
    grad,
    T: T.self,
    use_locking: useLocking,
    use_nesterov: useNesterov)
}

// Update '*var' according to the AddSign update.
//
// m_t <- beta1 * m_{t-1} + (1 - beta1) * g
// update <- (alpha + sign_decay * sign(g) *sign(m)) * g
// variable <- variable - lr_t * update
//
// - Parameters:
//   - var: Should be from a Variable().
//   - m: Should be from a Variable().
//   - lr: Scaling factor. Must be a scalar.
//   - alpha: Must be a scalar.
//   - sign_decay: Must be a scalar.
//   - beta: Must be a scalar.
//   - grad: The gradient.
//
// - Attr use_locking: If `True`, updating of the var and m tensors is
//   protected by a lock; otherwise the behavior is undefined, but may exhibit less
//   contention.
//
// - Output out: Same as "var".
@_inlineable @inline(__always)
public static func applyAddSign<T: Numeric>(
  var_: Tensor<T>,
  m: Tensor<T>,
  lr: Tensor<T>,
  alpha: Tensor<T>,
  signDecay: Tensor<T>,
  beta: Tensor<T>,
  grad: Tensor<T>,
  useLocking: Bool = false
) -> Tensor<T> {
  return #tfop("ApplyAddSign",
    var_,
    m,
    lr,
    alpha,
    signDecay,
    beta,
    grad,
    T: T.self,
    use_locking: useLocking)
}

// Update '*var' according to the centered RMSProp algorithm.
//
// The centered RMSProp algorithm uses an estimate of the centered second moment
// (i.e., the variance) for normalization, as opposed to regular RMSProp, which
// uses the (uncentered) second moment. This often helps with training, but is
// slightly more expensive in terms of computation and memory.
//
// Note that in dense implementation of this algorithm, mg, ms, and mom will
// update even if the grad is zero, but in this sparse implementation, mg, ms,
// and mom will not update in iterations during which the grad is zero.
//
// mean_square = decay * mean_square + (1-decay) * gradient ** 2
// mean_grad = decay * mean_grad + (1-decay) * gradient
//
// Delta = learning_rate * gradient / sqrt(mean_square + epsilon - mean_grad ** 2)
//
// mg <- rho * mg_{t-1} + (1-rho) * grad
// ms <- rho * ms_{t-1} + (1-rho) * grad * grad
// mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms - mg * mg + epsilon)
// var <- var - mom
//
// - Parameters:
//   - var: Should be from a Variable().
//   - mg: Should be from a Variable().
//   - ms: Should be from a Variable().
//   - mom: Should be from a Variable().
//   - lr: Scaling factor. Must be a scalar.
//   - rho: Decay rate. Must be a scalar.
//   - epsilon: Ridge term. Must be a scalar.
//   - grad: The gradient.
//
// - Attr use_locking: If `True`, updating of the var, mg, ms, and mom tensors is
//   protected by a lock; otherwise the behavior is undefined, but may exhibit less
//   contention.
//
// - Output out: Same as "var".
@_inlineable @inline(__always)
public static func applyCenteredRMSProp<T: Numeric>(
  var_: Tensor<T>,
  mg: Tensor<T>,
  ms: Tensor<T>,
  mom: Tensor<T>,
  lr: Tensor<T>,
  rho: Tensor<T>,
  momentum: Tensor<T>,
  epsilon: Tensor<T>,
  grad: Tensor<T>,
  useLocking: Bool = false
) -> Tensor<T> {
  return #tfop("ApplyCenteredRMSProp",
    var_,
    mg,
    ms,
    mom,
    lr,
    rho,
    momentum,
    epsilon,
    grad,
    T: T.self,
    use_locking: useLocking)
}

// Update '*var' according to the Ftrl-proximal scheme.
//
// accum_new = accum + grad * grad
// linear += grad + (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
// quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
// var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
// accum = accum_new
//
// - Parameters:
//   - var: Should be from a Variable().
//   - accum: Should be from a Variable().
//   - linear: Should be from a Variable().
//   - grad: The gradient.
//   - lr: Scaling factor. Must be a scalar.
//   - l1: L1 regulariation. Must be a scalar.
//   - l2: L2 regulariation. Must be a scalar.
//   - lr_power: Scaling factor. Must be a scalar.
//
// - Attr use_locking: If `True`, updating of the var and accum tensors will be protected
//   by a lock; otherwise the behavior is undefined, but may exhibit less
//   contention.
//
// - Output out: Same as "var".
@_inlineable @inline(__always)
public static func applyFtrl<T: Numeric>(
  var_: Tensor<T>,
  accum: Tensor<T>,
  linear: Tensor<T>,
  grad: Tensor<T>,
  lr: Tensor<T>,
  l1: Tensor<T>,
  l2: Tensor<T>,
  lrPower: Tensor<T>,
  useLocking: Bool = false
) -> Tensor<T> {
  return #tfop("ApplyFtrl",
    var_,
    accum,
    linear,
    grad,
    lr,
    l1,
    l2,
    lrPower,
    T: T.self,
    use_locking: useLocking)
}

// Update '*var' according to the Ftrl-proximal scheme.
//
// grad_with_shrinkage = grad + 2 * l2_shrinkage * var
// accum_new = accum + grad_with_shrinkage * grad_with_shrinkage
// linear += grad_with_shrinkage +
//     (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
// quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
// var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
// accum = accum_new
//
// - Parameters:
//   - var: Should be from a Variable().
//   - accum: Should be from a Variable().
//   - linear: Should be from a Variable().
//   - grad: The gradient.
//   - lr: Scaling factor. Must be a scalar.
//   - l1: L1 regulariation. Must be a scalar.
//   - l2: L2 shrinkage regulariation. Must be a scalar.
//   - lr_power: Scaling factor. Must be a scalar.
//
// - Attr use_locking: If `True`, updating of the var and accum tensors will be protected
//   by a lock; otherwise the behavior is undefined, but may exhibit less
//   contention.
//
// - Output out: Same as "var".
@_inlineable @inline(__always)
public static func applyFtrlV2<T: Numeric>(
  var_: Tensor<T>,
  accum: Tensor<T>,
  linear: Tensor<T>,
  grad: Tensor<T>,
  lr: Tensor<T>,
  l1: Tensor<T>,
  l2: Tensor<T>,
  l2Shrinkage: Tensor<T>,
  lrPower: Tensor<T>,
  useLocking: Bool = false
) -> Tensor<T> {
  return #tfop("ApplyFtrlV2",
    var_,
    accum,
    linear,
    grad,
    lr,
    l1,
    l2,
    l2Shrinkage,
    lrPower,
    T: T.self,
    use_locking: useLocking)
}

// Update '*var' by subtracting 'alpha' * 'delta' from it.
//
// - Parameters:
//   - var: Should be from a Variable().
//   - alpha: Scaling factor. Must be a scalar.
//   - delta: The change.
//
// - Attr use_locking: If `True`, the subtraction will be protected by a lock;
//   otherwise the behavior is undefined, but may exhibit less contention.
//
// - Output out: Same as "var".
@_inlineable @inline(__always)
public static func applyGradientDescent<T: Numeric>(
  var_: Tensor<T>,
  alpha: Tensor<T>,
  delta: Tensor<T>,
  useLocking: Bool = false
) -> Tensor<T> {
  return #tfop("ApplyGradientDescent",
    var_,
    alpha,
    delta,
    T: T.self,
    use_locking: useLocking)
}

// Update '*var' according to the momentum scheme. Set use_nesterov = True if you
//
// want to use Nesterov momentum.
//
// accum = accum * momentum + grad
// var -= lr * accum
//
// - Parameters:
//   - var: Should be from a Variable().
//   - accum: Should be from a Variable().
//   - lr: Scaling factor. Must be a scalar.
//   - grad: The gradient.
//   - momentum: Momentum. Must be a scalar.
//
// - Attrs:
//   - use_locking: If `True`, updating of the var and accum tensors will be protected
//     by a lock; otherwise the behavior is undefined, but may exhibit less
//     contention.
//   - use_nesterov: If `True`, the tensor passed to compute grad will be
//     var - lr * momentum * accum, so in the end, the var you get is actually
//     var - lr * momentum * accum.
//
// - Output out: Same as "var".
@_inlineable @inline(__always)
public static func applyMomentum<T: Numeric>(
  var_: Tensor<T>,
  accum: Tensor<T>,
  lr: Tensor<T>,
  grad: Tensor<T>,
  momentum: Tensor<T>,
  useLocking: Bool = false,
  useNesterov: Bool = false
) -> Tensor<T> {
  return #tfop("ApplyMomentum",
    var_,
    accum,
    lr,
    grad,
    momentum,
    T: T.self,
    use_locking: useLocking,
    use_nesterov: useNesterov)
}

// Update '*var' according to the AddSign update.
//
// m_t <- beta1 * m_{t-1} + (1 - beta1) * g
// update <- exp(logbase * sign_decay * sign(g) * sign(m_t)) * g
// variable <- variable - lr_t * update
//
// - Parameters:
//   - var: Should be from a Variable().
//   - m: Should be from a Variable().
//   - lr: Scaling factor. Must be a scalar.
//   - logbase: Must be a scalar.
//   - sign_decay: Must be a scalar.
//   - beta: Must be a scalar.
//   - grad: The gradient.
//
// - Attr use_locking: If `True`, updating of the var and m tensors is
//   protected by a lock; otherwise the behavior is undefined, but may exhibit less
//   contention.
//
// - Output out: Same as "var".
@_inlineable @inline(__always)
public static func applyPowerSign<T: Numeric>(
  var_: Tensor<T>,
  m: Tensor<T>,
  lr: Tensor<T>,
  logbase: Tensor<T>,
  signDecay: Tensor<T>,
  beta: Tensor<T>,
  grad: Tensor<T>,
  useLocking: Bool = false
) -> Tensor<T> {
  return #tfop("ApplyPowerSign",
    var_,
    m,
    lr,
    logbase,
    signDecay,
    beta,
    grad,
    T: T.self,
    use_locking: useLocking)
}

// Update '*var' and '*accum' according to FOBOS with Adagrad learning rate.
//
// accum += grad * grad
// prox_v = var - lr * grad * (1 / sqrt(accum))
// var = sign(prox_v)/(1+lr*l2) * max{|prox_v|-lr*l1,0}
//
// - Parameters:
//   - var: Should be from a Variable().
//   - accum: Should be from a Variable().
//   - lr: Scaling factor. Must be a scalar.
//   - l1: L1 regularization. Must be a scalar.
//   - l2: L2 regularization. Must be a scalar.
//   - grad: The gradient.
//
// - Attr use_locking: If True, updating of the var and accum tensors will be protected by
//   a lock; otherwise the behavior is undefined, but may exhibit less contention.
//
// - Output out: Same as "var".
@_inlineable @inline(__always)
public static func applyProximalAdagrad<T: Numeric>(
  var_: Tensor<T>,
  accum: Tensor<T>,
  lr: Tensor<T>,
  l1: Tensor<T>,
  l2: Tensor<T>,
  grad: Tensor<T>,
  useLocking: Bool = false
) -> Tensor<T> {
  return #tfop("ApplyProximalAdagrad",
    var_,
    accum,
    lr,
    l1,
    l2,
    grad,
    T: T.self,
    use_locking: useLocking)
}

// Update '*var' as FOBOS algorithm with fixed learning rate.
//
// prox_v = var - alpha * delta
// var = sign(prox_v)/(1+alpha*l2) * max{|prox_v|-alpha*l1,0}
//
// - Parameters:
//   - var: Should be from a Variable().
//   - alpha: Scaling factor. Must be a scalar.
//   - l1: L1 regularization. Must be a scalar.
//   - l2: L2 regularization. Must be a scalar.
//   - delta: The change.
//
// - Attr use_locking: If True, the subtraction will be protected by a lock;
//   otherwise the behavior is undefined, but may exhibit less contention.
//
// - Output out: Same as "var".
@_inlineable @inline(__always)
public static func applyProximalGradientDescent<T: Numeric>(
  var_: Tensor<T>,
  alpha: Tensor<T>,
  l1: Tensor<T>,
  l2: Tensor<T>,
  delta: Tensor<T>,
  useLocking: Bool = false
) -> Tensor<T> {
  return #tfop("ApplyProximalGradientDescent",
    var_,
    alpha,
    l1,
    l2,
    delta,
    T: T.self,
    use_locking: useLocking)
}

// Update '*var' according to the RMSProp algorithm.
//
// Note that in dense implementation of this algorithm, ms and mom will
// update even if the grad is zero, but in this sparse implementation, ms
// and mom will not update in iterations during which the grad is zero.
//
// mean_square = decay * mean_square + (1-decay) * gradient ** 2
// Delta = learning_rate * gradient / sqrt(mean_square + epsilon)
//
// ms <- rho * ms_{t-1} + (1-rho) * grad * grad
// mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
// var <- var - mom
//
// - Parameters:
//   - var: Should be from a Variable().
//   - ms: Should be from a Variable().
//   - mom: Should be from a Variable().
//   - lr: Scaling factor. Must be a scalar.
//   - rho: Decay rate. Must be a scalar.
//   - epsilon: Ridge term. Must be a scalar.
//   - grad: The gradient.
//
// - Attr use_locking: If `True`, updating of the var, ms, and mom tensors is protected
//   by a lock; otherwise the behavior is undefined, but may exhibit less
//   contention.
//
// - Output out: Same as "var".
@_inlineable @inline(__always)
public static func applyRMSProp<T: Numeric>(
  var_: Tensor<T>,
  ms: Tensor<T>,
  mom: Tensor<T>,
  lr: Tensor<T>,
  rho: Tensor<T>,
  momentum: Tensor<T>,
  epsilon: Tensor<T>,
  grad: Tensor<T>,
  useLocking: Bool = false
) -> Tensor<T> {
  return #tfop("ApplyRMSProp",
    var_,
    ms,
    mom,
    lr,
    rho,
    momentum,
    epsilon,
    grad,
    T: T.self,
    use_locking: useLocking)
}

// Returns the truth value of abs(x-y) < tolerance element-wise.
@_inlineable @inline(__always)
public static func approximateEqual<T: Numeric>(
  x: Tensor<T>,
  y: Tensor<T>,
  tolerance: Double = 1e-05
) -> Tensor<Bool> {
  return #tfop("ApproximateEqual",
    x,
    y,
    T: T.self,
    tolerance: tolerance)
}

// Returns the index with the largest value across dimensions of a tensor.
//
// Note that in case of ties the identity of the return value is not guaranteed.
//
// - Parameter dimension: int32 or int64, must be in the range `[-rank(input), rank(input))`.
//   Describes which dimension of the input Tensor to reduce across. For vectors,
//   use dimension = 0.
@_inlineable @inline(__always)
public static func argMax<T: Numeric, Tidx: BinaryInteger, Output_type: BinaryInteger>(
  input: Tensor<T>,
  dimension: Tensor<Tidx>
) -> Tensor<Output_type> {
  return #tfop("ArgMax",
    input,
    dimension,
    T: T.self,
    Tidx: Tidx.self,
    Output_type: Output_type.self)
}

// Returns the index with the smallest value across dimensions of a tensor.
//
// Note that in case of ties the identity of the return value is not guaranteed.
//
// - Parameter dimension: int32 or int64, must be in the range `[-rank(input), rank(input))`.
//   Describes which dimension of the input Tensor to reduce across. For vectors,
//   use dimension = 0.
@_inlineable @inline(__always)
public static func argMin<T: Numeric, Tidx: BinaryInteger, Output_type: BinaryInteger>(
  input: Tensor<T>,
  dimension: Tensor<Tidx>
) -> Tensor<Output_type> {
  return #tfop("ArgMin",
    input,
    dimension,
    T: T.self,
    Tidx: Tidx.self,
    Output_type: Output_type.self)
}

// Computes asin of x element-wise.
@_inlineable @inline(__always)
public static func asin<T: Numeric>(
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Asin",
    x,
    T: T.self)
}

// Computes inverse hyperbolic sine of x element-wise.
@_inlineable @inline(__always)
public static func asinh<T: BinaryFloatingPoint>(
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Asinh",
    x,
    T: T.self)
}

// Asserts that the given condition is true.
//
// If `condition` evaluates to false, print the list of tensors in `data`.
// `summarize` determines how many entries of the tensors to print.
//
// - Parameters:
//   - condition: The condition to evaluate.
//   - data: The tensors to print out when condition is false.
//
// - Attr summarize: Print this many entries of each tensor.
@_inlineable @inline(__always)
public static func assert<T: Numeric>(
  condition: Tensor<Bool>,
  data: [Tensor<T>],
  summarize: Int = 3
) {
  return #tfop("Assert",
    condition,
    data,
    summarize: summarize)
}

// Update 'ref' by assigning 'value' to it.
//
// This operation outputs "ref" after the assignment is done.
// This makes it easier to chain operations that need to use the reset value.
//
// - Parameters:
//   - ref: Should be from a `Variable` node. May be uninitialized.
//   - value: The value to be assigned to the variable.
//
// - Attrs:
//   - validate_shape: If true, the operation will validate that the shape
//     of 'value' matches the shape of the Tensor being assigned to.  If false,
//     'ref' will take on the shape of 'value'.
//   - use_locking: If True, the assignment will be protected by a lock;
//     otherwise the behavior is undefined, but may exhibit less contention.
//
// - Output output_ref: = Same as "ref".  Returned as a convenience for operations that want
//   to use the new value after the variable has been reset.
@_inlineable @inline(__always)
public static func assign<T: Numeric>(
  ref: Tensor<T>,
  value: Tensor<T>,
  validateShape: Bool = true,
  useLocking: Bool = true
) -> Tensor<T> {
  return #tfop("Assign",
    ref,
    value,
    T: T.self,
    validate_shape: validateShape,
    use_locking: useLocking)
}

// Update 'ref' by adding 'value' to it.
//
// This operation outputs "ref" after the update is done.
// This makes it easier to chain operations that need to use the reset value.
//
// - Parameters:
//   - ref: Should be from a `Variable` node.
//   - value: The value to be added to the variable.
//
// - Attr use_locking: If True, the addition will be protected by a lock;
//   otherwise the behavior is undefined, but may exhibit less contention.
//
// - Output output_ref: = Same as "ref".  Returned as a convenience for operations that want
//   to use the new value after the variable has been updated.
@_inlineable @inline(__always)
public static func assignAdd<T: Numeric>(
  ref: Tensor<T>,
  value: Tensor<T>,
  useLocking: Bool = false
) -> Tensor<T> {
  return #tfop("AssignAdd",
    ref,
    value,
    T: T.self,
    use_locking: useLocking)
}

// Update 'ref' by subtracting 'value' from it.
//
// This operation outputs "ref" after the update is done.
// This makes it easier to chain operations that need to use the reset value.
//
// - Parameters:
//   - ref: Should be from a `Variable` node.
//   - value: The value to be subtracted to the variable.
//
// - Attr use_locking: If True, the subtraction will be protected by a lock;
//   otherwise the behavior is undefined, but may exhibit less contention.
//
// - Output output_ref: = Same as "ref".  Returned as a convenience for operations that want
//   to use the new value after the variable has been updated.
@_inlineable @inline(__always)
public static func assignSub<T: Numeric>(
  ref: Tensor<T>,
  value: Tensor<T>,
  useLocking: Bool = false
) -> Tensor<T> {
  return #tfop("AssignSub",
    ref,
    value,
    T: T.self,
    use_locking: useLocking)
}

// Computes atan of x element-wise.
@_inlineable @inline(__always)
public static func atan<T: Numeric>(
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Atan",
    x,
    T: T.self)
}

// Computes arctangent of `y/x` element-wise, respecting signs of the arguments.
//
// This is the angle \( \theta \in [-\pi, \pi] \) such that
// \[ x = r \cos(\theta) \]
// and
// \[ y = r \sin(\theta) \]
// where \(r = \sqrt(x^2 + y^2) \).
@_inlineable @inline(__always)
public static func atan2<T: BinaryFloatingPoint>(
  y: Tensor<T>,
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Atan2",
    y,
    x,
    T: T.self)
}

// Computes inverse hyperbolic tangent of x element-wise.
@_inlineable @inline(__always)
public static func atanh<T: BinaryFloatingPoint>(
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Atanh",
    x,
    T: T.self)
}

@_inlineable @inline(__always)
public static func attr(
  a: Int
) {
  return #tfop("Attr",
    a: a)
}

@_inlineable @inline(__always)
public static func attrBool(
  a: Bool
) {
  return #tfop("AttrBool",
    a: a)
}

@_inlineable @inline(__always)
public static func attrBoolList(
  a: [Bool]
) {
  return #tfop("AttrBoolList",
    a: a)
}

@_inlineable @inline(__always)
public static func attrDefault(
  a: String = "banana"
) {
  return #tfop("AttrDefault",
    a: a)
}

@_inlineable @inline(__always)
public static func attrEmptyListDefault(
  a: [Double]
) {
  return #tfop("AttrEmptyListDefault",
    a: a)
}

@_inlineable @inline(__always)
public static func attrEnum(
  a: A
) {
  return #tfop("AttrEnum",
    a: a.rawValue)
}

@_inlineable @inline(__always)
public static func attrEnumList(
  a: [String]
) {
  return #tfop("AttrEnumList",
    a: a)
}

@_inlineable @inline(__always)
public static func attrFloat(
  a: Double
) {
  return #tfop("AttrFloat",
    a: a)
}

@_inlineable @inline(__always)
public static func attrListDefault(
  a: [Int]
) {
  return #tfop("AttrListDefault",
    a: a)
}

@_inlineable @inline(__always)
public static func attrListMin(
  a: [Int]
) {
  return #tfop("AttrListMin",
    a: a)
}

@_inlineable @inline(__always)
public static func attrListTypeDefault<T: Numeric>(
  a: [Tensor<T>],
  b: [Tensor<T>]
) {
  return #tfop("AttrListTypeDefault",
    a,
    b,
    T: T.self)
}

@_inlineable @inline(__always)
public static func attrMin(
  a: Int
) {
  return #tfop("AttrMin",
    a: a)
}

@_inlineable @inline(__always)
public static func attrTypeDefault<T: Numeric>(
  a: Tensor<T>
) {
  return #tfop("AttrTypeDefault",
    a,
    T: T.self)
}

// Produces a visualization of audio data over time.
//
// Spectrograms are a standard way of representing audio information as a series of
// slices of frequency information, one slice for each window of time. By joining
// these together into a sequence, they form a distinctive fingerprint of the sound
// over time.
//
// This op expects to receive audio data as an input, stored as floats in the range
// -1 to 1, together with a window width in samples, and a stride specifying how
// far to move the window between slices. From this it generates a three
// dimensional output. The lowest dimension has an amplitude value for each
// frequency during that time slice. The next dimension is time, with successive
// frequency slices. The final dimension is for the channels in the input, so a
// stereo audio input would have two here for example.
//
// This means the layout when converted and saved as an image is rotated 90 degrees
// clockwise from a typical spectrogram. Time is descending down the Y axis, and
// the frequency decreases from left to right.
//
// Each value in the result represents the square root of the sum of the real and
// imaginary parts of an FFT on the current window of samples. In this way, the
// lowest dimension represents the power of each frequency in the current window,
// and adjacent windows are concatenated in the next dimension.
//
// To get a more intuitive and visual look at what this operation does, you can run
// tensorflow/examples/wav_to_spectrogram to read in an audio file and save out the
// resulting spectrogram as a PNG image.
//
// - Parameter input: Float representation of audio data.
//
// - Attrs:
//   - window_size: How wide the input window is in samples. For the highest efficiency
//     this should be a power of two, but other values are accepted.
//   - stride: How widely apart the center of adjacent sample windows should be.
//   - magnitude_squared: Whether to return the squared magnitude or just the
//     magnitude. Using squared magnitude can avoid extra calculations.
//
// - Output spectrogram: 3D representation of the audio frequencies as an image.
@_inlineable @inline(__always)
public static func audioSpectrogram(
  input: Tensor<Float>,
  windowSize: Int,
  stride: Int,
  magnitudeSquared: Bool = false
) -> Tensor<Float> {
  return #tfop("AudioSpectrogram",
    input,
    window_size: windowSize,
    stride: stride,
    magnitude_squared: magnitudeSquared)
}

// Performs average pooling on the input.
//
// Each entry in `output` is the mean of the corresponding size `ksize`
// window in `value`.
//
// - Parameter value: 4-D with shape `[batch, height, width, channels]`.
//
// - Attrs:
//   - ksize: The size of the sliding window for each dimension of `value`.
//   - strides: The stride of the sliding window for each dimension of `value`.
//   - padding: The type of padding algorithm to use.
//   - data_format: Specify the data format of the input and output data. With the
//     default format "NHWC", the data is stored in the order of:
//         [batch, in_height, in_width, in_channels].
//     Alternatively, the format could be "NCHW", the data storage order of:
//         [batch, in_channels, in_height, in_width].
//
// - Output output: The average pooled output tensor.
@_inlineable @inline(__always)
public static func avgPool<T: BinaryFloatingPoint>(
  value: Tensor<T>,
  ksize: [Int],
  strides: [Int],
  padding: Padding,
  dataFormat: DataFormat = .nhwc
) -> Tensor<T> {
  return #tfop("AvgPool",
    value,
    T: T.self,
    ksize: ksize,
    strides: strides,
    padding: padding.rawValue,
    data_format: dataFormat.rawValue)
}

// Performs 3D average pooling on the input.
//
// - Parameter input: Shape `[batch, depth, rows, cols, channels]` tensor to pool over.
//
// - Attrs:
//   - ksize: 1-D tensor of length 5. The size of the window for each dimension of
//     the input tensor. Must have `ksize[0] = ksize[4] = 1`.
//   - strides: 1-D tensor of length 5. The stride of the sliding window for each
//     dimension of `input`. Must have `strides[0] = strides[4] = 1`.
//   - padding: The type of padding algorithm to use.
//   - data_format: The data format of the input and output data. With the
//     default format "NDHWC", the data is stored in the order of:
//         [batch, in_depth, in_height, in_width, in_channels].
//     Alternatively, the format could be "NCDHW", the data storage order is:
//         [batch, in_channels, in_depth, in_height, in_width].
//
// - Output output: The average pooled output tensor.
@_inlineable @inline(__always)
public static func avgPool3D<T: BinaryFloatingPoint>(
  input: Tensor<T>,
  ksize: [Int],
  strides: [Int],
  padding: Padding,
  dataFormat: DataFormat1 = .ndhwc
) -> Tensor<T> {
  return #tfop("AvgPool3D",
    input,
    T: T.self,
    ksize: ksize,
    strides: strides,
    padding: padding.rawValue,
    data_format: dataFormat.rawValue)
}

// Computes gradients of average pooling function.
//
// - Parameters:
//   - orig_input_shape: The original input dimensions.
//   - grad: Output backprop of shape `[batch, depth, rows, cols, channels]`.
//
// - Attrs:
//   - ksize: 1-D tensor of length 5. The size of the window for each dimension of
//     the input tensor. Must have `ksize[0] = ksize[4] = 1`.
//   - strides: 1-D tensor of length 5. The stride of the sliding window for each
//     dimension of `input`. Must have `strides[0] = strides[4] = 1`.
//   - padding: The type of padding algorithm to use.
//   - data_format: The data format of the input and output data. With the
//     default format "NDHWC", the data is stored in the order of:
//         [batch, in_depth, in_height, in_width, in_channels].
//     Alternatively, the format could be "NCDHW", the data storage order is:
//         [batch, in_channels, in_depth, in_height, in_width].
//
// - Output output: The backprop for input.
@_inlineable @inline(__always)
public static func avgPool3DGrad<T: BinaryFloatingPoint>(
  origInputShape: Tensor<Int32>,
  grad: Tensor<T>,
  ksize: [Int],
  strides: [Int],
  padding: Padding,
  dataFormat: DataFormat1 = .ndhwc
) -> Tensor<T> {
  return #tfop("AvgPool3DGrad",
    origInputShape,
    grad,
    T: T.self,
    ksize: ksize,
    strides: strides,
    padding: padding.rawValue,
    data_format: dataFormat.rawValue)
}

// Computes gradients of the average pooling function.
//
// - Parameters:
//   - orig_input_shape: 1-D.  Shape of the original input to `avg_pool`.
//   - grad: 4-D with shape `[batch, height, width, channels]`.  Gradients w.r.t.
//     the output of `avg_pool`.
//
// - Attrs:
//   - ksize: The size of the sliding window for each dimension of the input.
//   - strides: The stride of the sliding window for each dimension of the input.
//   - padding: The type of padding algorithm to use.
//   - data_format: Specify the data format of the input and output data. With the
//     default format "NHWC", the data is stored in the order of:
//         [batch, in_height, in_width, in_channels].
//     Alternatively, the format could be "NCHW", the data storage order of:
//         [batch, in_channels, in_height, in_width].
//
// - Output output: 4-D.  Gradients w.r.t. the input of `avg_pool`.
@_inlineable @inline(__always)
public static func avgPoolGrad<T: BinaryFloatingPoint>(
  origInputShape: Tensor<Int32>,
  grad: Tensor<T>,
  ksize: [Int],
  strides: [Int],
  padding: Padding,
  dataFormat: DataFormat = .nhwc
) -> Tensor<T> {
  return #tfop("AvgPoolGrad",
    origInputShape,
    grad,
    T: T.self,
    ksize: ksize,
    strides: strides,
    padding: padding.rawValue,
    data_format: dataFormat.rawValue)
}

@_inlineable @inline(__always)
public static func b(
) -> Tensor<Float> {
  return #tfop("B")
}

// Batches all input tensors nondeterministically.
//
// When many instances of this Op are being run concurrently with the same
// container/shared_name in the same device, some will output zero-shaped Tensors
// and others will output Tensors of size up to max_batch_size.
//
// All Tensors in in_tensors are batched together (so, for example, labels and
// features should be batched with a single instance of this operation.
//
// Each invocation of batch emits an `id` scalar which will be used to identify
// this particular invocation when doing unbatch or its gradient.
//
// Each op which emits a non-empty batch will also emit a non-empty batch_index
// Tensor, which, is a [K, 3] matrix where each row contains the invocation's id,
// start, and length of elements of each set of Tensors present in batched_tensors.
//
// Batched tensors are concatenated along the first dimension, and all tensors in
// in_tensors must have the first dimension of the same size.
//
// in_tensors: The tensors to be batched.
// num_batch_threads: Number of scheduling threads for processing batches of work.
//  Determines the number of batches processed in parallel.
// max_batch_size: Batch sizes will never be bigger than this.
// batch_timeout_micros: Maximum number of microseconds to wait before outputting
//  an incomplete batch.
// allowed_batch_sizes: Optional list of allowed batch sizes. If left empty, does
//  nothing. Otherwise, supplies a list of batch sizes, causing the op to pad
//  batches up to one of those sizes. The entries must increase monotonically, and
//  the final entry must equal max_batch_size.
// grad_timeout_micros: The timeout to use for the gradient. See Unbatch.
// batched_tensors: Either empty tensors or a batch of concatenated Tensors.
// batch_index: If out_tensors is non-empty, has information to invert it.
// container: Controls the scope of sharing of this batch.
// id: always contains a scalar with a unique ID for this invocation of Batch.
// shared_name: Concurrently running instances of batch in the same device with the
//  same container and shared_name will batch their elements together. If left
//  empty, the op name will be used as the shared name.
// T: the types of tensors to be batched.
@_inlineable @inline(__always)
public static func batch<T: Numeric>(
  inTensors: [Tensor<T>],
  numBatchThreads: Int,
  maxBatchSize: Int,
  maxEnqueuedBatches: Int = 10,
  batchTimeoutMicros: Int,
  allowedBatchSizes: [Int],
  gradTimeoutMicros: Int,
  container: String,
  sharedName: String,
  batchingQueue: String
) -> ([Tensor<T>], Tensor<Int64>, Tensor<Int64>) {
  return #tfop("Batch",
    inTensors,
    num_batch_threads: numBatchThreads,
    max_batch_size: maxBatchSize,
    max_enqueued_batches: maxEnqueuedBatches,
    batch_timeout_micros: batchTimeoutMicros,
    allowed_batch_sizes: allowedBatchSizes,
    grad_timeout_micros: gradTimeoutMicros,
    container: container,
    shared_name: sharedName,
    batching_queue: batchingQueue)
}

@_inlineable @inline(__always)
public static func batchCholesky<T: BinaryFloatingPoint>(
  input: Tensor<T>
) -> Tensor<T> {
  return #tfop("BatchCholesky",
    input,
    T: T.self)
}

@_inlineable @inline(__always)
public static func batchCholeskyGrad<T: BinaryFloatingPoint>(
  l: Tensor<T>,
  grad: Tensor<T>
) -> Tensor<T> {
  return #tfop("BatchCholeskyGrad",
    l,
    grad,
    T: T.self)
}

// Multiplies slices of two tensors in batches.
//
// Multiplies all slices of `Tensor` `x` and `y` (each slice can be
// viewed as an element of a batch), and arranges the individual results
// in a single output tensor of the same batch size. Each of the
// individual slices can optionally be adjointed (to adjoint a matrix
// means to transpose and conjugate it) before multiplication by setting
// the `adj_x` or `adj_y` flag to `True`, which are by default `False`.
//
// The input tensors `x` and `y` are 2-D or higher with shape `[..., r_x, c_x]`
// and `[..., r_y, c_y]`.
//
// The output tensor is 2-D or higher with shape `[..., r_o, c_o]`, where:
//
//     r_o = c_x if adj_x else r_x
//     c_o = r_y if adj_y else c_y
//
// It is computed as:
//
//     output[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :])
//
// - Parameters:
//   - x: 2-D or higher with shape `[..., r_x, c_x]`.
//   - y: 2-D or higher with shape `[..., r_y, c_y]`.
//
// - Attrs:
//   - adj_x: If `True`, adjoint the slices of `x`. Defaults to `False`.
//   - adj_y: If `True`, adjoint the slices of `y`. Defaults to `False`.
//
// - Output output: 3-D or higher with shape `[..., r_o, c_o]`
@_inlineable @inline(__always)
public static func batchMatMul<T: Numeric>(
  x: Tensor<T>,
  y: Tensor<T>,
  adjX: Bool = false,
  adjY: Bool = false
) -> Tensor<T> {
  return #tfop("BatchMatMul",
    x,
    y,
    T: T.self,
    adj_x: adjX,
    adj_y: adjY)
}

@_inlineable @inline(__always)
public static func batchMatrixBandPart<T: Numeric>(
  input: Tensor<T>,
  numLower: Tensor<Int64>,
  numUpper: Tensor<Int64>
) -> Tensor<T> {
  return #tfop("BatchMatrixBandPart",
    input,
    numLower,
    numUpper,
    T: T.self)
}

@_inlineable @inline(__always)
public static func batchMatrixDeterminant<T: BinaryFloatingPoint>(
  input: Tensor<T>
) -> Tensor<T> {
  return #tfop("BatchMatrixDeterminant",
    input,
    T: T.self)
}

@_inlineable @inline(__always)
public static func batchMatrixDiag<T: Numeric>(
  diagonal: Tensor<T>
) -> Tensor<T> {
  return #tfop("BatchMatrixDiag",
    diagonal,
    T: T.self)
}

@_inlineable @inline(__always)
public static func batchMatrixDiagPart<T: Numeric>(
  input: Tensor<T>
) -> Tensor<T> {
  return #tfop("BatchMatrixDiagPart",
    input,
    T: T.self)
}

@_inlineable @inline(__always)
public static func batchMatrixInverse<T: BinaryFloatingPoint>(
  input: Tensor<T>,
  adjoint: Bool = false
) -> Tensor<T> {
  return #tfop("BatchMatrixInverse",
    input,
    T: T.self,
    adjoint: adjoint)
}

@_inlineable @inline(__always)
public static func batchMatrixSetDiag<T: Numeric>(
  input: Tensor<T>,
  diagonal: Tensor<T>
) -> Tensor<T> {
  return #tfop("BatchMatrixSetDiag",
    input,
    diagonal,
    T: T.self)
}

@_inlineable @inline(__always)
public static func batchMatrixSolve<T: BinaryFloatingPoint>(
  matrix: Tensor<T>,
  rhs: Tensor<T>,
  adjoint: Bool = false
) -> Tensor<T> {
  return #tfop("BatchMatrixSolve",
    matrix,
    rhs,
    T: T.self,
    adjoint: adjoint)
}

@_inlineable @inline(__always)
public static func batchMatrixSolveLs<T: BinaryFloatingPoint>(
  matrix: Tensor<T>,
  rhs: Tensor<T>,
  l2Regularizer: Tensor<Double>,
  fast: Bool = true
) -> Tensor<T> {
  return #tfop("BatchMatrixSolveLs",
    matrix,
    rhs,
    l2Regularizer,
    T: T.self,
    fast: fast)
}

@_inlineable @inline(__always)
public static func batchMatrixTriangularSolve<T: BinaryFloatingPoint>(
  matrix: Tensor<T>,
  rhs: Tensor<T>,
  lower: Bool = true,
  adjoint: Bool = false
) -> Tensor<T> {
  return #tfop("BatchMatrixTriangularSolve",
    matrix,
    rhs,
    T: T.self,
    lower: lower,
    adjoint: adjoint)
}

// Batch normalization.
//
// This op is deprecated. Prefer `tf.nn.batch_normalization`.
//
// - Parameters:
//   - t: A 4D input Tensor.
//   - m: A 1D mean Tensor with size matching the last dimension of t.
//     This is the first output from tf.nn.moments,
//     or a saved moving average thereof.
//   - v: A 1D variance Tensor with size matching the last dimension of t.
//     This is the second output from tf.nn.moments,
//     or a saved moving average thereof.
//   - beta: A 1D beta Tensor with size matching the last dimension of t.
//     An offset to be added to the normalized tensor.
//   - gamma: A 1D gamma Tensor with size matching the last dimension of t.
//     If "scale_after_normalization" is true, this tensor will be multiplied
//     with the normalized tensor.
//
// - Attrs:
//   - variance_epsilon: A small float number to avoid dividing by 0.
//   - scale_after_normalization: A bool indicating whether the resulted tensor
//     needs to be multiplied with gamma.
@_inlineable @inline(__always)
public static func batchNormWithGlobalNormalization<T: Numeric>(
  t: Tensor<T>,
  m: Tensor<T>,
  v: Tensor<T>,
  beta: Tensor<T>,
  gamma: Tensor<T>,
  varianceEpsilon: Double,
  scaleAfterNormalization: Bool
) -> Tensor<T> {
  return #tfop("BatchNormWithGlobalNormalization",
    t,
    m,
    v,
    beta,
    gamma,
    T: T.self,
    variance_epsilon: varianceEpsilon,
    scale_after_normalization: scaleAfterNormalization)
}

// Gradients for batch normalization.
//
// This op is deprecated. See `tf.nn.batch_normalization`.
//
// - Parameters:
//   - t: A 4D input Tensor.
//   - m: A 1D mean Tensor with size matching the last dimension of t.
//     This is the first output from tf.nn.moments,
//     or a saved moving average thereof.
//   - v: A 1D variance Tensor with size matching the last dimension of t.
//     This is the second output from tf.nn.moments,
//     or a saved moving average thereof.
//   - gamma: A 1D gamma Tensor with size matching the last dimension of t.
//     If "scale_after_normalization" is true, this Tensor will be multiplied
//     with the normalized Tensor.
//   - backprop: 4D backprop Tensor.
//
// - Attrs:
//   - variance_epsilon: A small float number to avoid dividing by 0.
//   - scale_after_normalization: A bool indicating whether the resulted tensor
//     needs to be multiplied with gamma.
//
// - Outputs:
//   - dx: 4D backprop tensor for input.
//   - dm: 1D backprop tensor for mean.
//   - dv: 1D backprop tensor for variance.
//   - db: 1D backprop tensor for beta.
//   - dg: 1D backprop tensor for gamma.
@_inlineable @inline(__always)
public static func batchNormWithGlobalNormalizationGrad<T: Numeric>(
  t: Tensor<T>,
  m: Tensor<T>,
  v: Tensor<T>,
  gamma: Tensor<T>,
  backprop: Tensor<T>,
  varianceEpsilon: Double,
  scaleAfterNormalization: Bool
) -> (Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>) {
  return #tfop("BatchNormWithGlobalNormalizationGrad",
    t,
    m,
    v,
    gamma,
    backprop,
    T: T.self,
    variance_epsilon: varianceEpsilon,
    scale_after_normalization: scaleAfterNormalization)
}

@_inlineable @inline(__always)
public static func batchSelfAdjointEig<T: BinaryFloatingPoint>(
  input: Tensor<T>
) -> Tensor<T> {
  return #tfop("BatchSelfAdjointEig",
    input,
    T: T.self)
}

@_inlineable @inline(__always)
public static func batchSelfAdjointEigV2<T: BinaryFloatingPoint>(
  input: Tensor<T>,
  computeV: Bool = true
) -> (Tensor<T>, Tensor<T>) {
  return #tfop("BatchSelfAdjointEigV2",
    input,
    T: T.self,
    compute_v: computeV)
}

@_inlineable @inline(__always)
public static func batchSvd<T: BinaryFloatingPoint>(
  input: Tensor<T>,
  computeUv: Bool = true,
  fullMatrices: Bool = false
) -> (Tensor<T>, Tensor<T>, Tensor<T>) {
  return #tfop("BatchSvd",
    input,
    T: T.self,
    compute_uv: computeUv,
    full_matrices: fullMatrices)
}

// BatchToSpace for 4-D tensors of type T.
//
// This is a legacy version of the more general BatchToSpaceND.
//
// Rearranges (permutes) data from batch into blocks of spatial data, followed by
// cropping. This is the reverse transformation of SpaceToBatch. More specifically,
// this op outputs a copy of the input tensor where values from the `batch`
// dimension are moved in spatial blocks to the `height` and `width` dimensions,
// followed by cropping along the `height` and `width` dimensions.
//
// - Parameters:
//   - input: 4-D tensor with shape
//     `[batch*block_size*block_size, height_pad/block_size, width_pad/block_size,
//       depth]`. Note that the batch size of the input tensor must be divisible by
//     `block_size * block_size`.
//   - crops: 2-D tensor of non-negative integers with shape `[2, 2]`. It specifies
//     how many elements to crop from the intermediate result across the spatial
//     dimensions as follows:
//
//         crops = [[crop_top, crop_bottom], [crop_left, crop_right]]
//
// - Output output: 4-D with shape `[batch, height, width, depth]`, where:
//
//         height = height_pad - crop_top - crop_bottom
//         width = width_pad - crop_left - crop_right
//
//   The attr `block_size` must be greater than one. It indicates the block size.
//
//   Some examples:
//
//   (1) For the following input of shape `[4, 1, 1, 1]` and block_size of 2:
//
//   ```
//   [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
//   ```
//
//   The output tensor has shape `[1, 2, 2, 1]` and value:
//
//   ```
//   x = [[[[1], [2]], [[3], [4]]]]
//   ```
//
//   (2) For the following input of shape `[4, 1, 1, 3]` and block_size of 2:
//
//   ```
//   [[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]]
//   ```
//
//   The output tensor has shape `[1, 2, 2, 3]` and value:
//
//   ```
//   x = [[[[1, 2, 3], [4, 5, 6]],
//         [[7, 8, 9], [10, 11, 12]]]]
//   ```
//
//   (3) For the following input of shape `[4, 2, 2, 1]` and block_size of 2:
//
//   ```
//   x = [[[[1], [3]], [[9], [11]]],
//        [[[2], [4]], [[10], [12]]],
//        [[[5], [7]], [[13], [15]]],
//        [[[6], [8]], [[14], [16]]]]
//   ```
//
//   The output tensor has shape `[1, 4, 4, 1]` and value:
//
//   ```
//   x = [[[1],   [2],  [3],  [4]],
//        [[5],   [6],  [7],  [8]],
//        [[9],  [10], [11],  [12]],
//        [[13], [14], [15],  [16]]]
//   ```
//
//   (4) For the following input of shape `[8, 1, 2, 1]` and block_size of 2:
//
//   ```
//   x = [[[[1], [3]]], [[[9], [11]]], [[[2], [4]]], [[[10], [12]]],
//        [[[5], [7]]], [[[13], [15]]], [[[6], [8]]], [[[14], [16]]]]
//   ```
//
//   The output tensor has shape `[2, 2, 4, 1]` and value:
//
//   ```
//   x = [[[[1], [3]], [[5], [7]]],
//        [[[2], [4]], [[10], [12]]],
//        [[[5], [7]], [[13], [15]]],
//        [[[6], [8]], [[14], [16]]]]
//   ```
@_inlineable @inline(__always)
public static func batchToSpace<T: Numeric, Tidx: BinaryInteger>(
  input: Tensor<T>,
  crops: Tensor<Tidx>,
  blockSize: Int
) -> Tensor<T> {
  return #tfop("BatchToSpace",
    input,
    crops,
    T: T.self,
    Tidx: Tidx.self,
    block_size: blockSize)
}

// BatchToSpace for N-D tensors of type T.
//
// This operation reshapes the "batch" dimension 0 into `M + 1` dimensions of shape
// `block_shape + [batch]`, interleaves these blocks back into the grid defined by
// the spatial dimensions `[1, ..., M]`, to obtain a result with the same rank as
// the input.  The spatial dimensions of this intermediate result are then
// optionally cropped according to `crops` to produce the output.  This is the
// reverse of SpaceToBatch.  See below for a precise description.
//
// - Parameters:
//   - input: N-D with shape `input_shape = [batch] + spatial_shape + remaining_shape`,
//     where spatial_shape has M dimensions.
//   - block_shape: 1-D with shape `[M]`, all values must be >= 1.
//   - crops: 2-D with shape `[M, 2]`, all values must be >= 0.
//       `crops[i] = [crop_start, crop_end]` specifies the amount to crop from input
//       dimension `i + 1`, which corresponds to spatial dimension `i`.  It is
//       required that
//       `crop_start[i] + crop_end[i] <= block_shape[i] * input_shape[i + 1]`.
//
//     This operation is equivalent to the following steps:
//
//     1. Reshape `input` to `reshaped` of shape:
//          [block_shape[0], ..., block_shape[M-1],
//           batch / prod(block_shape),
//           input_shape[1], ..., input_shape[N-1]]
//
//     2. Permute dimensions of `reshaped` to produce `permuted` of shape
//          [batch / prod(block_shape),
//
//           input_shape[1], block_shape[0],
//           ...,
//           input_shape[M], block_shape[M-1],
//
//           input_shape[M+1], ..., input_shape[N-1]]
//
//     3. Reshape `permuted` to produce `reshaped_permuted` of shape
//          [batch / prod(block_shape),
//
//           input_shape[1] * block_shape[0],
//           ...,
//           input_shape[M] * block_shape[M-1],
//
//           input_shape[M+1],
//           ...,
//           input_shape[N-1]]
//
//     4. Crop the start and end of dimensions `[1, ..., M]` of
//        `reshaped_permuted` according to `crops` to produce the output of shape:
//          [batch / prod(block_shape),
//
//           input_shape[1] * block_shape[0] - crops[0,0] - crops[0,1],
//           ...,
//           input_shape[M] * block_shape[M-1] - crops[M-1,0] - crops[M-1,1],
//
//           input_shape[M+1], ..., input_shape[N-1]]
//
//     Some examples:
//
//     (1) For the following input of shape `[4, 1, 1, 1]`, `block_shape = [2, 2]`, and
//         `crops = [[0, 0], [0, 0]]`:
//
//     ```
//     [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
//     ```
//
//     The output tensor has shape `[1, 2, 2, 1]` and value:
//
//     ```
//     x = [[[[1], [2]], [[3], [4]]]]
//     ```
//
//     (2) For the following input of shape `[4, 1, 1, 3]`, `block_shape = [2, 2]`, and
//         `crops = [[0, 0], [0, 0]]`:
//
//     ```
//     [[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]]
//     ```
//
//     The output tensor has shape `[1, 2, 2, 3]` and value:
//
//     ```
//     x = [[[[1, 2, 3], [4, 5, 6]],
//           [[7, 8, 9], [10, 11, 12]]]]
//     ```
//
//     (3) For the following input of shape `[4, 2, 2, 1]`, `block_shape = [2, 2]`, and
//         `crops = [[0, 0], [0, 0]]`:
//
//     ```
//     x = [[[[1], [3]], [[9], [11]]],
//          [[[2], [4]], [[10], [12]]],
//          [[[5], [7]], [[13], [15]]],
//          [[[6], [8]], [[14], [16]]]]
//     ```
//
//     The output tensor has shape `[1, 4, 4, 1]` and value:
//
//     ```
//     x = [[[1],   [2],  [3],  [4]],
//          [[5],   [6],  [7],  [8]],
//          [[9],  [10], [11],  [12]],
//          [[13], [14], [15],  [16]]]
//     ```
//
//     (4) For the following input of shape `[8, 1, 3, 1]`, `block_shape = [2, 2]`, and
//         `crops = [[0, 0], [2, 0]]`:
//
//     ```
//     x = [[[[0], [1], [3]]], [[[0], [9], [11]]],
//          [[[0], [2], [4]]], [[[0], [10], [12]]],
//          [[[0], [5], [7]]], [[[0], [13], [15]]],
//          [[[0], [6], [8]]], [[[0], [14], [16]]]]
//     ```
//
//     The output tensor has shape `[2, 2, 4, 1]` and value:
//
//     ```
//     x = [[[[1],   [2],  [3],  [4]],
//           [[5],   [6],  [7],  [8]]],
//          [[[9],  [10], [11],  [12]],
//           [[13], [14], [15],  [16]]]]
//     ```
@_inlineable @inline(__always)
public static func batchToSpaceND<T: Numeric, Tblock_shape: BinaryInteger, Tcrops: BinaryInteger>(
  input: Tensor<T>,
  blockShape: Tensor<Tblock_shape>,
  crops: Tensor<Tcrops>
) -> Tensor<T> {
  return #tfop("BatchToSpaceND",
    input,
    blockShape,
    crops,
    T: T.self,
    Tblock_shape: Tblock_shape.self,
    Tcrops: Tcrops.self)
}

// Compute the regularized incomplete beta integral \\(I_x(a, b)\\).
//
// The regularized incomplete beta integral is defined as:
//
//
// \\(I_x(a, b) = \frac{B(x; a, b)}{B(a, b)}\\)
//
// where
//
//
// \\(B(x; a, b) = \int_0^x t^{a-1} (1 - t)^{b-1} dt\\)
//
//
// is the incomplete beta function and \\(B(a, b)\\) is the *complete*
// beta function.
@_inlineable @inline(__always)
public static func betainc<T: BinaryFloatingPoint>(
  a: Tensor<T>,
  b: Tensor<T>,
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Betainc",
    a,
    b,
    x,
    T: T.self)
}

// Adds `bias` to `value`.
//
// This is a special case of `tf.add` where `bias` is restricted to be 1-D.
// Broadcasting is supported, so `value` may have any number of dimensions.
//
// - Parameters:
//   - value: Any number of dimensions.
//   - bias: 1-D with size the last dimension of `value`.
//
// - Attr data_format: Specify the data format of the input and output data. With the
//   default format "NHWC", the bias tensor will be added to the last dimension
//   of the value tensor.
//   Alternatively, the format could be "NCHW", the data storage order of:
//       [batch, in_channels, in_height, in_width].
//   The tensor will be added to "in_channels", the third-to-the-last
//       dimension.
//
// - Output output: Broadcasted sum of `value` and `bias`.
@_inlineable @inline(__always)
public static func biasAdd<T: Numeric>(
  value: Tensor<T>,
  bias: Tensor<T>,
  dataFormat: DataFormat = .nhwc
) -> Tensor<T> {
  return #tfop("BiasAdd",
    value,
    bias,
    T: T.self,
    data_format: dataFormat.rawValue)
}

// The backward operation for "BiasAdd" on the "bias" tensor.
//
// It accumulates all the values from out_backprop into the feature dimension.
// For NHWC data format, the feature dimension is the last. For NCHW data format,
// the feature dimension is the third-to-last.
//
// - Parameter out_backprop: Any number of dimensions.
//
// - Attr data_format: Specify the data format of the input and output data. With the
//   default format "NHWC", the bias tensor will be added to the last dimension
//   of the value tensor.
//   Alternatively, the format could be "NCHW", the data storage order of:
//       [batch, in_channels, in_height, in_width].
//   The tensor will be added to "in_channels", the third-to-the-last
//       dimension.
//
// - Output output: 1-D with size the feature dimension of `out_backprop`.
@_inlineable @inline(__always)
public static func biasAddGrad<T: Numeric>(
  outBackprop: Tensor<T>,
  dataFormat: DataFormat = .nhwc
) -> Tensor<T> {
  return #tfop("BiasAddGrad",
    outBackprop,
    T: T.self,
    data_format: dataFormat.rawValue)
}

// Adds `bias` to `value`.
//
// This is a deprecated version of BiasAdd and will be soon removed.
//
// This is a special case of `tf.add` where `bias` is restricted to be 1-D.
// Broadcasting is supported, so `value` may have any number of dimensions.
//
// - Parameters:
//   - value: Any number of dimensions.
//   - bias: 1-D with size the last dimension of `value`.
//
// - Output output: Broadcasted sum of `value` and `bias`.
@_inlineable @inline(__always)
public static func biasAddV1<T: Numeric>(
  value: Tensor<T>,
  bias: Tensor<T>
) -> Tensor<T> {
  return #tfop("BiasAddV1",
    value,
    bias,
    T: T.self)
}

@_inlineable @inline(__always)
public static func binary<T: Numeric>(
  a: Tensor<T>,
  b: Tensor<T>
) -> Tensor<T> {
  return #tfop("Binary",
    a,
    b,
    T: T.self)
}

// Counts the number of occurrences of each value in an integer array.
//
// Outputs a vector with length `size` and the same dtype as `weights`. If
// `weights` are empty, then index `i` stores the number of times the value `i` is
// counted in `arr`. If `weights` are non-empty, then index `i` stores the sum of
// the value in `weights` at each index where the corresponding value in `arr` is
// `i`.
//
// Values in `arr` outside of the range [0, size) are ignored.
//
// - Parameters:
//   - arr: int32 `Tensor`.
//   - size: non-negative int32 scalar `Tensor`.
//   - weights: is an int32, int64, float32, or float64 `Tensor` with the same
//     shape as `arr`, or a length-0 `Tensor`, in which case it acts as all weights
//     equal to 1.
//
// - Output bins: 1D `Tensor` with length equal to `size`. The counts or summed weights for
//   each value in the range [0, size).
@_inlineable @inline(__always)
public static func bincount<T: Numeric>(
  arr: Tensor<Int32>,
  size: Tensor<Int32>,
  weights: Tensor<T>
) -> Tensor<T> {
  return #tfop("Bincount",
    arr,
    size,
    weights,
    T: T.self)
}

// Bitcasts a tensor from one type to another without copying data.
//
// Given a tensor `input`, this operation returns a tensor that has the same buffer
// data as `input` with datatype `type`.
//
// If the input datatype `T` is larger than the output datatype `type` then the
// shape changes from [...] to [..., sizeof(`T`)/sizeof(`type`)].
//
// If `T` is smaller than `type`, the operator requires that the rightmost
// dimension be equal to sizeof(`type`)/sizeof(`T`). The shape then goes from
// [..., sizeof(`type`)/sizeof(`T`)] to [...].
//
// *NOTE*: Bitcast is implemented as a low-level cast, so machines with different
// endian orderings will give different results.
@_inlineable @inline(__always)
public static func bitcast<T: Numeric, Type: Numeric>(
  input: Tensor<T>
) -> Tensor<Type> {
  return #tfop("Bitcast",
    input,
    T: T.self,
    Type: Type.self)
}

// Elementwise computes the bitwise AND of `x` and `y`.
//
// The result will have those bits set, that are set in both `x` and `y`. The
// computation is performed on the underlying representations of `x` and `y`.
@_inlineable @inline(__always)
public static func bitwiseAnd<T: BinaryInteger>(
  x: Tensor<T>,
  y: Tensor<T>
) -> Tensor<T> {
  return #tfop("BitwiseAnd",
    x,
    y,
    T: T.self)
}

// Elementwise computes the bitwise OR of `x` and `y`.
//
// The result will have those bits set, that are set in `x`, `y` or both. The
// computation is performed on the underlying representations of `x` and `y`.
@_inlineable @inline(__always)
public static func bitwiseOr<T: BinaryInteger>(
  x: Tensor<T>,
  y: Tensor<T>
) -> Tensor<T> {
  return #tfop("BitwiseOr",
    x,
    y,
    T: T.self)
}

// Elementwise computes the bitwise XOR of `x` and `y`.
//
// The result will have those bits set, that are different in `x` and `y`. The
// computation is performed on the underlying representations of `x` and `y`.
@_inlineable @inline(__always)
public static func bitwiseXor<T: BinaryInteger>(
  x: Tensor<T>,
  y: Tensor<T>
) -> Tensor<T> {
  return #tfop("BitwiseXor",
    x,
    y,
    T: T.self)
}

// Computes the LSTM cell forward propagation for all the time steps.
//
// This is equivalent to applying LSTMBlockCell in a loop, like so:
//
// ```python
// for x1 in unpack(x):
//   i1, cs1, f1, o1, ci1, co1, h1 = LSTMBlock(
//     x1, cs_prev, h_prev, w, wci, wcf, wco, b)
//   cs_prev = cs1
//   h_prev = h1
//   i.append(i1)
//   cs.append(cs1)
//   f.append(f1)
//   o.append(o1)
//   ci.append(ci1)
//   co.append(co1)
//   h.append(h1)
// return pack(i), pack(cs), pack(f), pack(o), pack(ci), pack(ch), pack(h)
// ```
//
// - Parameters:
//   - seq_len_max: Maximum time length actually used by this input. Outputs are padded
//     with zeros beyond this length.
//   - x: The sequence input to the LSTM, shape (timelen, batch_size, num_inputs).
//   - cs_prev: Value of the initial cell state.
//   - h_prev: Initial output of cell (to be used for peephole).
//   - w: The weight matrix.
//   - wci: The weight matrix for input gate peephole connection.
//   - wcf: The weight matrix for forget gate peephole connection.
//   - wco: The weight matrix for output gate peephole connection.
//   - b: The bias vector.
//
// - Attrs:
//   - forget_bias: The forget gate bias.
//   - cell_clip: Value to clip the 'cs' value to.
//   - use_peephole: Whether to use peephole weights.
//
// - Outputs:
//   - i: The input gate over the whole time sequence.
//   - cs: The cell state before the tanh over the whole time sequence.
//   - f: The forget gate over the whole time sequence.
//   - o: The output gate over the whole time sequence.
//   - ci: The cell input over the whole time sequence.
//   - co: The cell after the tanh over the whole time sequence.
//   - h: The output h vector over the whole time sequence.
@_inlineable @inline(__always)
public static func blockLSTM<T: BinaryFloatingPoint>(
  seqLenMax: Tensor<Int64>,
  x: Tensor<T>,
  csPrev: Tensor<T>,
  hPrev: Tensor<T>,
  w: Tensor<T>,
  wci: Tensor<T>,
  wcf: Tensor<T>,
  wco: Tensor<T>,
  b: Tensor<T>,
  forgetBias: Double = 1,
  cellClip: Double = 3,
  usePeephole: Bool = false
) -> (Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>) {
  return #tfop("BlockLSTM",
    seqLenMax,
    x,
    csPrev,
    hPrev,
    w,
    wci,
    wcf,
    wco,
    b,
    T: T.self,
    forget_bias: forgetBias,
    cell_clip: cellClip,
    use_peephole: usePeephole)
}

// Computes the LSTM cell backward propagation for the entire time sequence.
//
// This implementation is to be used in conjunction of LSTMBlock.
//
// - Parameters:
//   - seq_len_max: Maximum time length actually used by this input. Outputs are padded
//     with zeros beyond this length.
//   - x: The sequence input to the LSTM, shape (timelen, batch_size, num_inputs).
//   - cs_prev: Value of the initial cell state.
//   - h_prev: Initial output of cell (to be used for peephole).
//   - w: The weight matrix.
//   - wci: The weight matrix for input gate peephole connection.
//   - wcf: The weight matrix for forget gate peephole connection.
//   - wco: The weight matrix for output gate peephole connection.
//   - b: The bias vector.
//   - i: The input gate over the whole time sequence.
//   - cs: The cell state before the tanh over the whole time sequence.
//   - f: The forget gate over the whole time sequence.
//   - o: The output gate over the whole time sequence.
//   - ci: The cell input over the whole time sequence.
//   - co: The cell after the tanh over the whole time sequence.
//   - h: The output h vector over the whole time sequence.
//   - cs_grad: The current gradient of cs.
//   - h_grad: The gradient of h vector.
//
// - Attr use_peephole: Whether to use peephole weights.
//
// - Outputs:
//   - x_grad: The gradient of x to be back-propped.
//   - cs_prev_grad: The gradient of cs_prev to be back-propped.
//   - h_prev_grad: The gradient of h_prev to be back-propped.
//   - w_grad: The gradient for w to be back-propped.
//   - wci_grad: The gradient for wci to be back-propped.
//   - wcf_grad: The gradient for wcf to be back-propped.
//   - wco_grad: The gradient for wco to be back-propped.
//   - b_grad: The gradient for w to be back-propped.
@_inlineable @inline(__always)
public static func blockLSTMGrad<T: BinaryFloatingPoint>(
  seqLenMax: Tensor<Int64>,
  x: Tensor<T>,
  csPrev: Tensor<T>,
  hPrev: Tensor<T>,
  w: Tensor<T>,
  wci: Tensor<T>,
  wcf: Tensor<T>,
  wco: Tensor<T>,
  b: Tensor<T>,
  i: Tensor<T>,
  cs: Tensor<T>,
  f: Tensor<T>,
  o: Tensor<T>,
  ci: Tensor<T>,
  co: Tensor<T>,
  h: Tensor<T>,
  csGrad: Tensor<T>,
  hGrad: Tensor<T>,
  usePeephole: Bool
) -> (Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>) {
  return #tfop("BlockLSTMGrad",
    seqLenMax,
    x,
    csPrev,
    hPrev,
    w,
    wci,
    wcf,
    wco,
    b,
    i,
    cs,
    f,
    o,
    ci,
    co,
    h,
    csGrad,
    hGrad,
    T: T.self,
    use_peephole: usePeephole)
}

// Calculates gains for each feature and returns the best possible split information for the feature.
//
// The split information is the best threshold (bucket id), gains and left/right node contributions per node for each feature.
//
// It is possible that not all nodes can be split on each feature. Hence, the list of possible nodes can differ between the features. Therefore, we return `node_ids_list` for each feature, containing the list of nodes that this feature can be used to split.
//
// In this manner, the output is the best split per features and per node, so that it needs to be combined later to produce the best split for each node (among all possible features).
//
// The length of output lists are all of the same length, `num_features`.
// The output shapes are compatible in a way that the first dimension of all tensors of all lists are the same and equal to the number of possible split nodes for each feature.
//
// - Parameters:
//   - node_id_range: A Rank 1 tensor (shape=[2]) to specify the range [first, last) of node ids to process within `stats_summary_list`. The nodes are iterated between the two nodes specified by the tensor, as like `for node_id in range(node_id_range[0], node_id_range[1])` (Note that the last index node_id_range[1] is exclusive).
//   - stats_summary_list: A list of Rank 3 tensor (#shape=[max_splits, bucket, 2]) for accumulated stats summary (gradient/hessian) per node per buckets for each feature. The first dimension of the tensor is the maximum number of splits, and thus not all elements of it will be used, but only the indexes specified by node_ids will be used.
//   - l1: l1 regularization factor on leaf weights, per instance based.
//   - l2: l2 regularization factor on leaf weights, per instance based.
//   - tree_complexity: adjustment to the gain, per leaf based.
//   - min_node_weight: mininum avg of hessians in a node before required for the node to be considered for splitting.
//
// - Attrs:
//   - max_splits: the number of nodes that can be split in the whole tree. Used as a dimension of output tensors.
//   - num_features: inferred from the size of `stats_summary_list`; the number of total features.
//
// - Outputs:
//   - node_ids_list: An output list of Rank 1 tensors indicating possible split node ids for each feature. The length of the list is num_features, but each tensor has different size as each feature provides different possible nodes. See above for details like shapes and sizes.
//   - gains_list: An output list of Rank 1 tensors indicating the best gains for each feature to split for certain nodes. See above for details like shapes and sizes.
//   - thresholds_list: An output list of Rank 1 tensors indicating the bucket id to compare with (as a threshold) for split in each node. See above for details like shapes and sizes.
//   - left_node_contribs_list: A list of Rank 2 tensors indicating the contribution of the left nodes when branching from parent nodes (given by the tensor element in the output node_ids_list) to the left direction by the given threshold for each feature. This value will be used to make the left node value by adding to the parent node value. Second dimension size is 1 for 1-dimensional logits, but would be larger for multi-class problems. See above for details like shapes and sizes.
//   - right_node_contribs_list: A list of Rank 2 tensors, with the same shape/conditions as left_node_contribs_list, but just that the value is for the right node.
@_inlineable @inline(__always)
public static func boostedTreesCalculateBestGainsPerFeature(
  nodeIdRange: Tensor<Int32>,
  statsSummaryList: [Tensor<Float>],
  l1: Tensor<Float>,
  l2: Tensor<Float>,
  treeComplexity: Tensor<Float>,
  minNodeWeight: Tensor<Float>,
  maxSplits: Int
) -> ([Tensor<Int32>], [Tensor<Float>], [Tensor<Int32>], [Tensor<Float>], [Tensor<Float>]) {
  return #tfop("BoostedTreesCalculateBestGainsPerFeature",
    nodeIdRange,
    statsSummaryList,
    l1,
    l2,
    treeComplexity,
    minNodeWeight,
    max_splits: maxSplits)
}

// Makes the summary of accumulated stats for the batch.
//
// The summary stats contains gradients and hessians accumulated into the corresponding node and bucket for each example.
//
// - Parameters:
//   - node_ids: int32 Rank 1 Tensor containing node ids, which each example falls into for the requested layer.
//   - gradients: float32; Rank 2 Tensor (shape=[#examples, 1]) for gradients.
//   - hessians: float32; Rank 2 Tensor (shape=[#examples, 1]) for hessians.
//   - bucketized_features_list: int32 list of Rank 1 Tensors, each containing the bucketized feature (for each feature column).
//
// - Attrs:
//   - max_splits: int; the maximum number of splits possible in the whole tree.
//   - num_buckets: int; equals to the maximum possible value of bucketized feature.
//   - num_features: int; inferred from the size of bucketized_features_list; the number of features.
//
// - Output stats_summary: output Rank 4 Tensor (shape=[#features, #splits, #buckets, 2]) containing accumulated stats put into the corresponding node and bucket. The first index of 4th dimension refers to gradients, and the second to hessians.
@_inlineable @inline(__always)
public static func boostedTreesMakeStatsSummary(
  nodeIds: Tensor<Int32>,
  gradients: Tensor<Float>,
  hessians: Tensor<Float>,
  bucketizedFeaturesList: [Tensor<Int32>],
  maxSplits: Int,
  numBuckets: Int
) -> Tensor<Float> {
  return #tfop("BoostedTreesMakeStatsSummary",
    nodeIds,
    gradients,
    hessians,
    bucketizedFeaturesList,
    max_splits: maxSplits,
    num_buckets: numBuckets)
}

// Return the shape of s0 op s1 with broadcast.
//
// Given `s0` and `s1`, tensors that represent shapes, compute `r0`, the
// broadcasted shape. `s0`, `s1` and `r0` are all integer vectors.
@_inlineable @inline(__always)
public static func broadcastArgs<T: BinaryInteger>(
  s0: Tensor<T>,
  s1: Tensor<T>
) -> Tensor<T> {
  return #tfop("BroadcastArgs",
    s0,
    s1,
    T: T.self)
}

// Return the reduction indices for computing gradients of s0 op s1 with broadcast.
//
// This is typically used by gradient computations for a broadcasting operation.
@_inlineable @inline(__always)
public static func broadcastGradientArgs<T: BinaryInteger>(
  s0: Tensor<T>,
  s1: Tensor<T>
) -> (Tensor<T>, Tensor<T>) {
  return #tfop("BroadcastGradientArgs",
    s0,
    s1,
    T: T.self)
}

// Bucketizes 'input' based on 'boundaries'.
//
// For example, if the inputs are
//     boundaries = [0, 10, 100]
//     input = [[-5, 10000]
//              [150,   10]
//              [5,    100]]
//
// then the output will be
//     output = [[0, 3]
//               [3, 2]
//               [1, 3]]
//
// - Parameter input: Any shape of Tensor contains with int or float type.
//
// - Attr boundaries: A sorted list of floats gives the boundary of the buckets.
//
// - Output output: Same shape with 'input', each value of input replaced with bucket index.
//
//   @compatibility(numpy)
//   Equivalent to np.digitize.
//   @end_compatibility
@_inlineable @inline(__always)
public static func bucketize<T: Numeric>(
  input: Tensor<T>,
  boundaries: [Double]
) -> Tensor<Int32> {
  return #tfop("Bucketize",
    input,
    T: T.self,
    boundaries: boundaries)
}

// Performs beam search decoding on the logits given in input.
//
// A note about the attribute merge_repeated: For the beam search decoder,
// this means that if consecutive entries in a beam are the same, only
// the first of these is emitted.  That is, when the top path is "A B B B B",
// "A B" is returned if merge_repeated = True but "A B B B B" is
// returned if merge_repeated = False.
//
// - Parameters:
//   - inputs: 3-D, shape: `(max_time x batch_size x num_classes)`, the logits.
//   - sequence_length: A vector containing sequence lengths, size `(batch)`.
//
// - Attrs:
//   - beam_width: A scalar >= 0 (beam search beam width).
//   - top_paths: A scalar >= 0, <= beam_width (controls output size).
//   - merge_repeated: If true, merge repeated classes in output.
//
// - Outputs:
//   - decoded_indices: A list (length: top_paths) of indices matrices.  Matrix j,
//     size `(total_decoded_outputs[j] x 2)`, has indices of a
//     `SparseTensor<int64, 2>`.  The rows store: [batch, time].
//   - decoded_values: A list (length: top_paths) of values vectors.  Vector j,
//     size `(length total_decoded_outputs[j])`, has the values of a
//     `SparseTensor<int64, 2>`.  The vector stores the decoded classes for beam j.
//   - decoded_shape: A list (length: top_paths) of shape vector.  Vector j,
//     size `(2)`, stores the shape of the decoded `SparseTensor[j]`.
//     Its values are: `[batch_size, max_decoded_length[j]]`.
//   - log_probability: A matrix, shaped: `(batch_size x top_paths)`.  The
//     sequence log-probabilities.
@_inlineable @inline(__always)
public static func cTCBeamSearchDecoder(
  inputs: Tensor<Float>,
  sequenceLength: Tensor<Int32>,
  beamWidth: Int,
  topPaths: Int,
  mergeRepeated: Bool = true
) -> ([Tensor<Int64>], [Tensor<Int64>], [Tensor<Int64>], Tensor<Float>) {
  return #tfop("CTCBeamSearchDecoder",
    inputs,
    sequenceLength,
    beam_width: beamWidth,
    top_paths: topPaths,
    merge_repeated: mergeRepeated)
}

// Performs greedy decoding on the logits given in inputs.
//
// A note about the attribute merge_repeated: if enabled, when
// consecutive logits' maximum indices are the same, only the first of
// these is emitted.  Labeling the blank '*', the sequence "A B B * B B"
// becomes "A B B" if merge_repeated = True and "A B B B B" if
// merge_repeated = False.
//
// Regardless of the value of merge_repeated, if the maximum index of a given
// time and batch corresponds to the blank, index `(num_classes - 1)`, no new
// element is emitted.
//
// - Parameters:
//   - inputs: 3-D, shape: `(max_time x batch_size x num_classes)`, the logits.
//   - sequence_length: A vector containing sequence lengths, size `(batch_size)`.
//
// - Attr merge_repeated: If True, merge repeated classes in output.
//
// - Outputs:
//   - decoded_indices: Indices matrix, size `(total_decoded_outputs x 2)`,
//     of a `SparseTensor<int64, 2>`.  The rows store: [batch, time].
//   - decoded_values: Values vector, size: `(total_decoded_outputs)`,
//     of a `SparseTensor<int64, 2>`.  The vector stores the decoded classes.
//   - decoded_shape: Shape vector, size `(2)`, of the decoded SparseTensor.
//     Values are: `[batch_size, max_decoded_length]`.
//   - log_probability: Matrix, size `(batch_size x 1)`, containing sequence
//     log-probabilities.
@_inlineable @inline(__always)
public static func cTCGreedyDecoder(
  inputs: Tensor<Float>,
  sequenceLength: Tensor<Int32>,
  mergeRepeated: Bool = false
) -> (Tensor<Int64>, Tensor<Int64>, Tensor<Int64>, Tensor<Float>) {
  return #tfop("CTCGreedyDecoder",
    inputs,
    sequenceLength,
    merge_repeated: mergeRepeated)
}

// Calculates the CTC Loss (log probability) for each batch entry.  Also calculates
//
// the gradient.  This class performs the softmax operation for you, so inputs
// should be e.g. linear projections of outputs by an LSTM.
//
// - Parameters:
//   - inputs: 3-D, shape: `(max_time x batch_size x num_classes)`, the logits.
//   - labels_indices: The indices of a `SparseTensor<int32, 2>`.
//     `labels_indices(i, :) == [b, t]` means `labels_values(i)` stores the id for
//     `(batch b, time t)`.
//   - labels_values: The values (labels) associated with the given batch and time.
//   - sequence_length: A vector containing sequence lengths (batch).
//
// - Attrs:
//   - preprocess_collapse_repeated: Scalar, if true then repeated labels are
//     collapsed prior to the CTC calculation.
//   - ctc_merge_repeated: Scalar.  If set to false, *during* CTC calculation
//     repeated non-blank labels will not be merged and are interpreted as
//     individual labels.  This is a simplified version of CTC.
//   - ignore_longer_outputs_than_inputs: Scalar. If set to true, during CTC
//     calculation, items that have longer output sequences than input sequences
//     are skipped: they don't contribute to the loss term and have zero-gradient.
//
// - Outputs:
//   - loss: A vector (batch) containing log-probabilities.
//   - gradient: The gradient of `loss`.  3-D, shape:
//     `(max_time x batch_size x num_classes)`.
@_inlineable @inline(__always)
public static func cTCLoss(
  inputs: Tensor<Float>,
  labelsIndices: Tensor<Int64>,
  labelsValues: Tensor<Int32>,
  sequenceLength: Tensor<Int32>,
  preprocessCollapseRepeated: Bool = false,
  ctcMergeRepeated: Bool = true,
  ignoreLongerOutputsThanInputs: Bool = false
) -> (Tensor<Float>, Tensor<Float>) {
  return #tfop("CTCLoss",
    inputs,
    labelsIndices,
    labelsValues,
    sequenceLength,
    preprocess_collapse_repeated: preprocessCollapseRepeated,
    ctc_merge_repeated: ctcMergeRepeated,
    ignore_longer_outputs_than_inputs: ignoreLongerOutputsThanInputs)
}

// Cast x of type SrcT to y of DstT.
@_inlineable @inline(__always)
public static func cast<Srct: Numeric, Dstt: Numeric>(
  x: Tensor<Srct>
) -> Tensor<Dstt> {
  return #tfop("Cast",
    x,
    Srct: Srct.self,
    Dstt: Dstt.self)
}

// Returns element-wise smallest integer in not less than x.
@_inlineable @inline(__always)
public static func ceil<T: BinaryFloatingPoint>(
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Ceil",
    x,
    T: T.self)
}

// Checks a tensor for NaN and Inf values.
//
// When run, reports an `InvalidArgument` error if `tensor` has any values
// that are not a number (NaN) or infinity (Inf). Otherwise, passes `tensor` as-is.
//
// - Attr message: Prefix of the error message.
@_inlineable @inline(__always)
public static func checkNumerics<T: BinaryFloatingPoint>(
  tensor: Tensor<T>,
  message: String
) -> Tensor<T> {
  return #tfop("CheckNumerics",
    tensor,
    T: T.self,
    message: message)
}

// Computes the Cholesky decomposition of one or more square matrices.
//
// The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
// form square matrices.
//
// The input has to be symmetric and positive definite. Only the lower-triangular
// part of the input will be used for this operation. The upper-triangular part
// will not be read.
//
// The output is a tensor of the same shape as the input
// containing the Cholesky decompositions for all input submatrices `[..., :, :]`.
//
// **Note**: The gradient computation on GPU is faster for large matrices but
// not for large batch dimensions when the submatrices are small. In this
// case it might be faster to use the CPU.
//
// - Parameter input: Shape is `[..., M, M]`.
//
// - Output output: Shape is `[..., M, M]`.
@_inlineable @inline(__always)
public static func cholesky<T: BinaryFloatingPoint>(
  input: Tensor<T>
) -> Tensor<T> {
  return #tfop("Cholesky",
    input,
    T: T.self)
}

// Computes the reverse mode backpropagated gradient of the Cholesky algorithm.
//
// For an explanation see "Differentiation of the Cholesky algorithm" by
// Iain Murray http://arxiv.org/abs/1602.07527.
//
// - Parameters:
//   - l: Output of batch Cholesky algorithm l = cholesky(A). Shape is `[..., M, M]`.
//     Algorithm depends only on lower triangular part of the innermost matrices of
//     this tensor.
//   - grad: df/dl where f is some scalar function. Shape is `[..., M, M]`.
//     Algorithm depends only on lower triangular part of the innermost matrices of
//     this tensor.
//
// - Output output: Symmetrized version of df/dA . Shape is `[..., M, M]`
@_inlineable @inline(__always)
public static func choleskyGrad<T: BinaryFloatingPoint>(
  l: Tensor<T>,
  grad: Tensor<T>
) -> Tensor<T> {
  return #tfop("CholeskyGrad",
    l,
    grad,
    T: T.self)
}

// Clips tensor values to a specified min and max.
//
// Given a tensor `t`, this operation returns a tensor of the same type and
// shape as `t` with its values clipped to `clip_value_min` and `clip_value_max`.
// Any values less than `clip_value_min` are set to `clip_value_min`. Any values
// greater than `clip_value_max` are set to `clip_value_max`.
//
// - Parameters:
//   - t: A `Tensor`.
//   - clip_value_min: A 0-D (scalar) `Tensor`, or a `Tensor` with the same shape
//     as `t`. The minimum value to clip by.
//   - clip_value_max: A 0-D (scalar) `Tensor`, or a `Tensor` with the same shape
//     as `t`. The maximum value to clip by.
//
// - Output output: A clipped `Tensor` with the same shape as input 't'.
@_inlineable @inline(__always)
public static func clipByValue<T: Numeric>(
  t: Tensor<T>,
  clipValueMin: Tensor<T>,
  clipValueMax: Tensor<T>
) -> Tensor<T> {
  return #tfop("ClipByValue",
    t,
    clipValueMin,
    clipValueMax,
    T: T.self)
}

// Mutually reduces multiple tensors of identical type and shape.
@_inlineable @inline(__always)
public static func collectiveReduce<T: Numeric>(
  input: Tensor<T>,
  groupSize: Int,
  groupKey: Int,
  instanceKey: Int,
  mergeOp: MergeOp,
  finalOp: FinalOp,
  subdivOffsets: [Int]
) -> Tensor<T> {
  return #tfop("CollectiveReduce",
    input,
    T: T.self,
    group_size: groupSize,
    group_key: groupKey,
    instance_key: instanceKey,
    merge_op: mergeOp.rawValue,
    final_op: finalOp.rawValue,
    subdiv_offsets: subdivOffsets)
}

// Compare values of `input` to `threshold` and pack resulting bits into a `uint8`.
//
// Each comparison returns a boolean `true` (if `input_value > threshold`)
// or and `false` otherwise.
//
// This operation is useful for Locality-Sensitive-Hashing (LSH) and other
// algorithms that use hashing approximations of cosine and `L2` distances;
// codes can be generated from an input via:
//
// ```python
// codebook_size = 50
// codebook_bits = codebook_size * 32
// codebook = tf.get_variable('codebook', [x.shape[-1].value, codebook_bits],
//                            dtype=x.dtype,
//                            initializer=tf.orthogonal_initializer())
// codes = compare_and_threshold(tf.matmul(x, codebook), threshold=0.)
// codes = tf.bitcast(codes, tf.int32)  # go from uint8 to int32
// # now codes has shape x.shape[:-1] + [codebook_size]
// ```
//
// **NOTE**: Currently, the innermost dimension of the tensor must be divisible
// by 8.
//
// Given an `input` shaped `[s0, s1, ..., s_n]`, the output is
// a `uint8` tensor shaped `[s0, s1, ..., s_n / 8]`.
//
// - Parameters:
//   - input: Values to compare against `threshold` and bitpack.
//   - threshold: Threshold to compare against.
//
// - Attr T: The type of the input and threshold.
//
// - Output output: The bitpacked comparisons.
@_inlineable @inline(__always)
public static func compareAndBitpack<T: Numeric>(
  input: Tensor<T>,
  threshold: Tensor<T>
) -> Tensor<UInt8> {
  return #tfop("CompareAndBitpack",
    input,
    threshold,
    T: T.self)
}

// Converts two real numbers to a complex number.
//
// Given a tensor `real` representing the real part of a complex number, and a
// tensor `imag` representing the imaginary part of a complex number, this
// operation returns complex numbers elementwise of the form \\(a + bj\\), where
// *a* represents the `real` part and *b* represents the `imag` part.
//
// The input tensors `real` and `imag` must have the same shape.
//
// For example:
//
// ```
// # tensor 'real' is [2.25, 3.25]
// # tensor `imag` is [4.75, 5.75]
// tf.complex(real, imag) ==> [[2.25 + 4.75j], [3.25 + 5.75j]]
// ```
@_inlineable @inline(__always)
public static func complex<T: BinaryFloatingPoint, Tout: Numeric>(
  real: Tensor<T>,
  imag: Tensor<T>
) -> Tensor<Tout> {
  return #tfop("Complex",
    real,
    imag,
    T: T.self,
    Tout: Tout.self)
}

// Computes the complex absolute value of a tensor.
//
// Given a tensor `x` of complex numbers, this operation returns a tensor of type
// `float` or `double` that is the absolute value of each element in `x`. All
// elements in `x` must be complex numbers of the form \\(a + bj\\). The absolute
// value is computed as \\( \sqrt{a^2 + b^2}\\).
@_inlineable @inline(__always)
public static func complexAbs<T: Numeric, Tout: BinaryFloatingPoint>(
  x: Tensor<T>
) -> Tensor<Tout> {
  return #tfop("ComplexAbs",
    x,
    T: T.self,
    Tout: Tout.self)
}

@_inlineable @inline(__always)
public static func complexStruct<T_c: Numeric>(
  nA: Int,
  nB: Int
) -> ([Tensor<Int32>], [Tensor<Int64>], [Tensor<T_c>]) {
  return #tfop("ComplexStruct",
    n_a: nA,
    n_b: nB)
}

// Computes the ids of the positions in sampled_candidates that match true_labels.
//
// When doing log-odds NCE, the result of this op should be passed through a
// SparseToDense op, then added to the logits of the sampled candidates. This has
// the effect of 'removing' the sampled labels that match the true labels by
// making the classifier sure that they are sampled labels.
//
// - Parameters:
//   - true_classes: The true_classes output of UnpackSparseLabels.
//   - sampled_candidates: The sampled_candidates output of CandidateSampler.
//
// - Attrs:
//   - num_true: Number of true labels per context.
//   - seed: If either seed or seed2 are set to be non-zero, the random number
//     generator is seeded by the given seed.  Otherwise, it is seeded by a
//     random seed.
//   - seed2: An second seed to avoid seed collision.
//
// - Outputs:
//   - indices: A vector of indices corresponding to rows of true_candidates.
//   - ids: A vector of IDs of positions in sampled_candidates that match a true_label
//     for the row with the corresponding index in indices.
//   - weights: A vector of the same length as indices and ids, in which each element
//     is -FLOAT_MAX.
@_inlineable @inline(__always)
public static func computeAccidentalHits(
  trueClasses: Tensor<Int64>,
  sampledCandidates: Tensor<Int64>,
  numTrue: Int,
  seed: Int = 0,
  seed2: Int = 0
) -> (Tensor<Int32>, Tensor<Int64>, Tensor<Float>) {
  return #tfop("ComputeAccidentalHits",
    trueClasses,
    sampledCandidates,
    num_true: numTrue,
    seed: seed,
    seed2: seed2)
}

// Concatenates tensors along one dimension.
//
// - Parameters:
//   - concat_dim: 0-D.  The dimension along which to concatenate.  Must be in the
//     range [0, rank(values)).
//   - values: The `N` Tensors to concatenate. Their ranks and types must match,
//     and their sizes must match in all dimensions except `concat_dim`.
//
// - Output output: A `Tensor` with the concatenation of values stacked along the
//   `concat_dim` dimension.  This tensor's shape matches that of `values` except
//   in `concat_dim` where it has the sum of the sizes.
@_inlineable @inline(__always)
public static func concat<T: Numeric>(
  concatDim: Tensor<Int32>,
  values: [Tensor<T>]
) -> Tensor<T> {
  return #tfop("Concat",
    concatDim,
    values,
    T: T.self)
}

// Computes offsets of concat inputs within its output.
//
// For example:
//
// ```
// # 'x' is [2, 2, 7]
// # 'y' is [2, 3, 7]
// # 'z' is [2, 5, 7]
// concat_offset(2, [x, y, z]) => [0, 0, 0], [0, 2, 0], [0, 5, 0]
// ```
//
// This is typically used by gradient computations for a concat operation.
//
// - Parameters:
//   - concat_dim: The dimension along which to concatenate.
//   - shape: The `N` int32 vectors representing shape of tensors being concatenated.
//
// - Output offset: The `N` int32 vectors representing the starting offset
//   of input tensors within the concatenated output.
@_inlineable @inline(__always)
public static func concatOffset(
  concatDim: Tensor<Int32>,
  shape: [Tensor<Int32>]
) -> [Tensor<Int32>] {
  return #tfop("ConcatOffset",
    concatDim,
    shape)
}

// Concatenates tensors along one dimension.
//
// - Parameters:
//   - values: List of `N` Tensors to concatenate. Their ranks and types must match,
//     and their sizes must match in all dimensions except `concat_dim`.
//   - axis: 0-D.  The dimension along which to concatenate.  Must be in the
//     range [-rank(values), rank(values)).
//
// - Output output: A `Tensor` with the concatenation of values stacked along the
//   `concat_dim` dimension.  This tensor's shape matches that of `values` except
//   in `concat_dim` where it has the sum of the sizes.
@_inlineable @inline(__always)
public static func concatV2<T: Numeric, Tidx: BinaryInteger>(
  values: [Tensor<T>],
  axis: Tensor<Tidx>
) -> Tensor<T> {
  return #tfop("ConcatV2",
    values,
    axis,
    T: T.self,
    Tidx: Tidx.self)
}

// Returns the complex conjugate of a complex number.
//
// Given a tensor `input` of complex numbers, this operation returns a tensor of
// complex numbers that are the complex conjugate of each element in `input`. The
// complex numbers in `input` must be of the form \\(a + bj\\), where *a* is the
// real part and *b* is the imaginary part.
//
// The complex conjugate returned by this operation is of the form \\(a - bj\\).
//
// For example:
//
// ```
// # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
// tf.conj(input) ==> [-2.25 - 4.75j, 3.25 - 5.75j]
// ```
@_inlineable @inline(__always)
public static func conj<T: Numeric>(
  input: Tensor<T>
) -> Tensor<T> {
  return #tfop("Conj",
    input,
    T: T.self)
}

// Shuffle dimensions of x according to a permutation and conjugate the result.
//
// The output `y` has the same rank as `x`. The shapes of `x` and `y` satisfy:
//   `y.shape[i] == x.shape[perm[i]] for i in [0, 1, ..., rank(x) - 1]`
//   `y[i,j,k,...,s,t,u] == conj(x[perm[i], perm[j], perm[k],...,perm[s], perm[t], perm[u]])`
@_inlineable @inline(__always)
public static func conjugateTranspose<T: Numeric, Tperm: BinaryInteger>(
  x: Tensor<T>,
  perm: Tensor<Tperm>
) -> Tensor<T> {
  return #tfop("ConjugateTranspose",
    x,
    perm,
    T: T.self,
    Tperm: Tperm.self)
}

@_inlineable @inline(__always)
public static func constructionFails(
) {
  return #tfop("ConstructionFails")
}

// Does nothing. Serves as a control trigger for scheduling.
//
// Only useful as a placeholder for control edges.
@_inlineable @inline(__always)
public static func controlTrigger(
) {
  return #tfop("ControlTrigger")
}

// Computes a 2-D convolution given 4-D `input` and `filter` tensors.
//
// Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
// and a filter / kernel tensor of shape
// `[filter_height, filter_width, in_channels, out_channels]`, this op
// performs the following:
//
// 1. Flattens the filter to a 2-D matrix with shape
//    `[filter_height * filter_width * in_channels, output_channels]`.
// 2. Extracts image patches from the input tensor to form a *virtual*
//    tensor of shape `[batch, out_height, out_width,
//    filter_height * filter_width * in_channels]`.
// 3. For each patch, right-multiplies the filter matrix and the image patch
//    vector.
//
// In detail, with the default NHWC format,
//
//     output[b, i, j, k] =
//         sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
//                         filter[di, dj, q, k]
//
// Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
// horizontal and vertices strides, `strides = [1, stride, stride, 1]`.
//
// - Parameters:
//   - input: A 4-D tensor. The dimension order is interpreted according to the value
//     of `data_format`, see below for details.
//   - filter: A 4-D tensor of shape
//     `[filter_height, filter_width, in_channels, out_channels]`
//
// - Attrs:
//   - strides: 1-D tensor of length 4.  The stride of the sliding window for each
//     dimension of `input`. The dimension order is determined by the value of
//     `data_format`, see below for details.
//   - padding: The type of padding algorithm to use.
//   - data_format: Specify the data format of the input and output data. With the
//     default format "NHWC", the data is stored in the order of:
//         [batch, height, width, channels].
//     Alternatively, the format could be "NCHW", the data storage order of:
//         [batch, channels, height, width].
//   - dilations: 1-D tensor of length 4.  The dilation factor for each dimension of
//     `input`. If set to k > 1, there will be k-1 skipped cells between each
//     filter element on that dimension. The dimension order is determined by the
//     value of `data_format`, see above for details. Dilations in the batch and
//     depth dimensions must be 1.
//
// - Output output: A 4-D tensor. The dimension order is determined by the value of
//   `data_format`, see below for details.
@_inlineable @inline(__always)
public static func conv2D<T: BinaryFloatingPoint>(
  input: Tensor<T>,
  filter: Tensor<T>,
  strides: [Int],
  useCudnnOnGpu: Bool = true,
  padding: Padding,
  dataFormat: DataFormat = .nhwc,
  dilations: [Int]
) -> Tensor<T> {
  return #tfop("Conv2D",
    input,
    filter,
    T: T.self,
    strides: strides,
    use_cudnn_on_gpu: useCudnnOnGpu,
    padding: padding.rawValue,
    data_format: dataFormat.rawValue,
    dilations: dilations)
}

// Computes the gradients of convolution with respect to the filter.
//
// - Parameters:
//   - input: 4-D with shape `[batch, in_height, in_width, in_channels]`.
//   - filter_sizes: An integer vector representing the tensor shape of `filter`,
//     where `filter` is a 4-D
//     `[filter_height, filter_width, in_channels, out_channels]` tensor.
//   - out_backprop: 4-D with shape `[batch, out_height, out_width, out_channels]`.
//     Gradients w.r.t. the output of the convolution.
//
// - Attrs:
//   - strides: The stride of the sliding window for each dimension of the input
//     of the convolution. Must be in the same order as the dimension specified with
//     format.
//   - padding: The type of padding algorithm to use.
//   - data_format: Specify the data format of the input and output data. With the
//     default format "NHWC", the data is stored in the order of:
//         [batch, in_height, in_width, in_channels].
//     Alternatively, the format could be "NCHW", the data storage order of:
//         [batch, in_channels, in_height, in_width].
//   - dilations: 1-D tensor of length 4.  The dilation factor for each dimension of
//     `input`. If set to k > 1, there will be k-1 skipped cells between each filter
//     element on that dimension. The dimension order is determined by the value of
//     `data_format`, see above for details. Dilations in the batch and depth
//     dimensions must be 1.
//
// - Output output: 4-D with shape
//   `[filter_height, filter_width, in_channels, out_channels]`.  Gradient w.r.t.
//   the `filter` input of the convolution.
@_inlineable @inline(__always)
public static func conv2DBackpropFilter<T: BinaryFloatingPoint>(
  input: Tensor<T>,
  filterSizes: Tensor<Int32>,
  outBackprop: Tensor<T>,
  strides: [Int],
  useCudnnOnGpu: Bool = true,
  padding: Padding,
  dataFormat: DataFormat = .nhwc,
  dilations: [Int]
) -> Tensor<T> {
  return #tfop("Conv2DBackpropFilter",
    input,
    filterSizes,
    outBackprop,
    T: T.self,
    strides: strides,
    use_cudnn_on_gpu: useCudnnOnGpu,
    padding: padding.rawValue,
    data_format: dataFormat.rawValue,
    dilations: dilations)
}

// Computes the gradients of convolution with respect to the input.
//
// - Parameters:
//   - input_sizes: An integer vector representing the shape of `input`,
//     where `input` is a 4-D `[batch, height, width, channels]` tensor.
//   - filter: 4-D with shape
//     `[filter_height, filter_width, in_channels, out_channels]`.
//   - out_backprop: 4-D with shape `[batch, out_height, out_width, out_channels]`.
//     Gradients w.r.t. the output of the convolution.
//
// - Attrs:
//   - strides: The stride of the sliding window for each dimension of the input
//     of the convolution. Must be in the same order as the dimension specified with
//     format.
//   - padding: The type of padding algorithm to use.
//   - data_format: Specify the data format of the input and output data. With the
//     default format "NHWC", the data is stored in the order of:
//         [batch, in_height, in_width, in_channels].
//     Alternatively, the format could be "NCHW", the data storage order of:
//         [batch, in_channels, in_height, in_width].
//   - dilations: 1-D tensor of length 4.  The dilation factor for each dimension of
//     `input`. If set to k > 1, there will be k-1 skipped cells between each filter
//     element on that dimension. The dimension order is determined by the value of
//     `data_format`, see above for details. Dilations in the batch and depth
//     dimensions must be 1.
//
// - Output output: 4-D with shape `[batch, in_height, in_width, in_channels]`.  Gradient
//   w.r.t. the input of the convolution.
@_inlineable @inline(__always)
public static func conv2DBackpropInput<T: BinaryFloatingPoint>(
  inputSizes: Tensor<Int32>,
  filter: Tensor<T>,
  outBackprop: Tensor<T>,
  strides: [Int],
  useCudnnOnGpu: Bool = true,
  padding: Padding,
  dataFormat: DataFormat = .nhwc,
  dilations: [Int]
) -> Tensor<T> {
  return #tfop("Conv2DBackpropInput",
    inputSizes,
    filter,
    outBackprop,
    T: T.self,
    strides: strides,
    use_cudnn_on_gpu: useCudnnOnGpu,
    padding: padding.rawValue,
    data_format: dataFormat.rawValue,
    dilations: dilations)
}

// Computes a 3-D convolution given 5-D `input` and `filter` tensors.
//
// In signal processing, cross-correlation is a measure of similarity of
// two waveforms as a function of a time-lag applied to one of them. This
// is also known as a sliding dot product or sliding inner-product.
//
// Our Conv3D implements a form of cross-correlation.
//
// - Parameters:
//   - input: Shape `[batch, in_depth, in_height, in_width, in_channels]`.
//   - filter: Shape `[filter_depth, filter_height, filter_width, in_channels,
//     out_channels]`. `in_channels` must match between `input` and `filter`.
//
// - Attrs:
//   - strides: 1-D tensor of length 5. The stride of the sliding window for each
//     dimension of `input`. Must have `strides[0] = strides[4] = 1`.
//   - padding: The type of padding algorithm to use.
//   - data_format: The data format of the input and output data. With the
//     default format "NDHWC", the data is stored in the order of:
//         [batch, in_depth, in_height, in_width, in_channels].
//     Alternatively, the format could be "NCDHW", the data storage order is:
//         [batch, in_channels, in_depth, in_height, in_width].
//   - dilations: 1-D tensor of length 5.  The dilation factor for each dimension of
//     `input`. If set to k > 1, there will be k-1 skipped cells between each
//     filter element on that dimension. The dimension order is determined by the
//     value of `data_format`, see above for details. Dilations in the batch and
//     depth dimensions must be 1.
@_inlineable @inline(__always)
public static func conv3D<T: BinaryFloatingPoint>(
  input: Tensor<T>,
  filter: Tensor<T>,
  strides: [Int],
  padding: Padding,
  dataFormat: DataFormat1 = .ndhwc,
  dilations: [Int]
) -> Tensor<T> {
  return #tfop("Conv3D",
    input,
    filter,
    T: T.self,
    strides: strides,
    padding: padding.rawValue,
    data_format: dataFormat.rawValue,
    dilations: dilations)
}

// Computes the gradients of 3-D convolution with respect to the filter.
//
// - Parameters:
//   - input: Shape `[batch, depth, rows, cols, in_channels]`.
//   - filter: Shape `[depth, rows, cols, in_channels, out_channels]`.
//     `in_channels` must match between `input` and `filter`.
//   - out_backprop: Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
//     out_channels]`.
//
// - Attrs:
//   - strides: 1-D tensor of length 5. The stride of the sliding window for each
//     dimension of `input`. Must have `strides[0] = strides[4] = 1`.
//   - padding: The type of padding algorithm to use.
@_inlineable @inline(__always)
public static func conv3DBackpropFilter<T: BinaryFloatingPoint>(
  input: Tensor<T>,
  filter: Tensor<T>,
  outBackprop: Tensor<T>,
  strides: [Int],
  padding: Padding
) -> Tensor<T> {
  return #tfop("Conv3DBackpropFilter",
    input,
    filter,
    outBackprop,
    T: T.self,
    strides: strides,
    padding: padding.rawValue)
}

// Computes the gradients of 3-D convolution with respect to the filter.
//
// - Parameters:
//   - input: Shape `[batch, depth, rows, cols, in_channels]`.
//   - filter_sizes: An integer vector representing the tensor shape of `filter`,
//     where `filter` is a 5-D
//     `[filter_depth, filter_height, filter_width, in_channels, out_channels]`
//     tensor.
//   - out_backprop: Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
//     out_channels]`.
//
// - Attrs:
//   - strides: 1-D tensor of length 5. The stride of the sliding window for each
//     dimension of `input`. Must have `strides[0] = strides[4] = 1`.
//   - padding: The type of padding algorithm to use.
//   - data_format: The data format of the input and output data. With the
//     default format "NDHWC", the data is stored in the order of:
//         [batch, in_depth, in_height, in_width, in_channels].
//     Alternatively, the format could be "NCDHW", the data storage order is:
//         [batch, in_channels, in_depth, in_height, in_width].
//   - dilations: 1-D tensor of length 5.  The dilation factor for each dimension of
//     `input`. If set to k > 1, there will be k-1 skipped cells between each
//     filter element on that dimension. The dimension order is determined by the
//     value of `data_format`, see above for details. Dilations in the batch and
//     depth dimensions must be 1.
@_inlineable @inline(__always)
public static func conv3DBackpropFilterV2<T: BinaryFloatingPoint>(
  input: Tensor<T>,
  filterSizes: Tensor<Int32>,
  outBackprop: Tensor<T>,
  strides: [Int],
  padding: Padding,
  dataFormat: DataFormat1 = .ndhwc,
  dilations: [Int]
) -> Tensor<T> {
  return #tfop("Conv3DBackpropFilterV2",
    input,
    filterSizes,
    outBackprop,
    T: T.self,
    strides: strides,
    padding: padding.rawValue,
    data_format: dataFormat.rawValue,
    dilations: dilations)
}

// Computes the gradients of 3-D convolution with respect to the input.
//
// - Parameters:
//   - input: Shape `[batch, depth, rows, cols, in_channels]`.
//   - filter: Shape `[depth, rows, cols, in_channels, out_channels]`.
//     `in_channels` must match between `input` and `filter`.
//   - out_backprop: Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
//     out_channels]`.
//
// - Attrs:
//   - strides: 1-D tensor of length 5. The stride of the sliding window for each
//     dimension of `input`. Must have `strides[0] = strides[4] = 1`.
//   - padding: The type of padding algorithm to use.
@_inlineable @inline(__always)
public static func conv3DBackpropInput<T: BinaryFloatingPoint>(
  input: Tensor<T>,
  filter: Tensor<T>,
  outBackprop: Tensor<T>,
  strides: [Int],
  padding: Padding
) -> Tensor<T> {
  return #tfop("Conv3DBackpropInput",
    input,
    filter,
    outBackprop,
    T: T.self,
    strides: strides,
    padding: padding.rawValue)
}

// Computes the gradients of 3-D convolution with respect to the input.
//
// - Parameters:
//   - input_sizes: An integer vector representing the tensor shape of `input`,
//     where `input` is a 5-D
//     `[batch, depth, rows, cols, in_channels]` tensor.
//   - filter: Shape `[depth, rows, cols, in_channels, out_channels]`.
//     `in_channels` must match between `input` and `filter`.
//   - out_backprop: Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
//     out_channels]`.
//
// - Attrs:
//   - strides: 1-D tensor of length 5. The stride of the sliding window for each
//     dimension of `input`. Must have `strides[0] = strides[4] = 1`.
//   - padding: The type of padding algorithm to use.
//   - data_format: The data format of the input and output data. With the
//     default format "NDHWC", the data is stored in the order of:
//         [batch, in_depth, in_height, in_width, in_channels].
//     Alternatively, the format could be "NCDHW", the data storage order is:
//         [batch, in_channels, in_depth, in_height, in_width].
//   - dilations: 1-D tensor of length 5.  The dilation factor for each dimension of
//     `input`. If set to k > 1, there will be k-1 skipped cells between each
//     filter element on that dimension. The dimension order is determined by the
//     value of `data_format`, see above for details. Dilations in the batch and
//     depth dimensions must be 1.
@_inlineable @inline(__always)
public static func conv3DBackpropInputV2<T: BinaryFloatingPoint>(
  inputSizes: Tensor<Int32>,
  filter: Tensor<T>,
  outBackprop: Tensor<T>,
  strides: [Int],
  padding: Padding,
  dataFormat: DataFormat1 = .ndhwc,
  dilations: [Int]
) -> Tensor<T> {
  return #tfop("Conv3DBackpropInputV2",
    inputSizes,
    filter,
    outBackprop,
    T: T.self,
    strides: strides,
    padding: padding.rawValue,
    data_format: dataFormat.rawValue,
    dilations: dilations)
}

// Copy Op.
//
// Performs CPU-to-CPU or GPU-to-GPU deep-copying of tensor, depending on the
// device on which the tensor is allocated.
// N.B.: If the all downstream attached debug ops are disabled given the current
// gRPC gating status, the output will simply forward the input tensor without
// deep-copying. See the documentation of Debug* ops for more details.
//
// Unlike the CopyHost Op, this op does not have HostMemory constraint on its
// input or output.
//
// - Parameter input: Input tensor.
//
// - Attrs:
//   - tensor_name: The name of the input tensor.
//   - debug_ops_spec: A list of debug op spec (op, url, gated_grpc) for attached debug
//     ops. Each element of the list has the format
//     <debug_op>;<grpc_url>;<gated_grpc>, wherein gated_grpc is boolean represented
//     as 0/1. E.g., "DebugIdentity;grpc://foo:3333;1",
//     "DebugIdentity;file:///tmp/tfdbg_1;0".
//
// - Output output: Output tensor, deep-copied from input.
@_inlineable @inline(__always)
public static func copy<T: Numeric>(
  input: Tensor<T>,
  tensorName: String,
  debugOpsSpec: [String]
) -> Tensor<T> {
  return #tfop("Copy",
    input,
    T: T.self,
    tensor_name: tensorName,
    debug_ops_spec: debugOpsSpec)
}

// Copy Host Op.
//
// Performs CPU-to-CPU deep-copying of tensor.
// N.B.: If the all downstream attached debug ops are disabled given the current
// gRPC gating status, the output will simply forward the input tensor without
// deep-copying. See the documentation of Debug* ops for more details.
//
// Unlike the Copy Op, this op has HostMemory constraint on its input or output.
//
// - Parameter input: Input tensor.
//
// - Attrs:
//   - tensor_name: The name of the input tensor.
//   - debug_ops_spec: A list of debug op spec (op, url, gated_grpc) for attached debug
//     ops. Each element of the list has the format
//     <debug_op>;<grpc_url>;<gated_grpc>, wherein gated_grpc is boolean represented
//     as 0/1. E.g., "DebugIdentity;grpc://foo:3333;1",
//     "DebugIdentity;file:///tmp/tfdbg_1;0".
//
// - Output output: Output tensor, deep-copied from input.
@_inlineable @inline(__always)
public static func copyHost<T: Numeric>(
  input: Tensor<T>,
  tensorName: String,
  debugOpsSpec: [String]
) -> Tensor<T> {
  return #tfop("CopyHost",
    input,
    T: T.self,
    tensor_name: tensorName,
    debug_ops_spec: debugOpsSpec)
}

@_inlineable @inline(__always)
public static func copyOp<T: Numeric>(
  a: Tensor<T>
) -> Tensor<T> {
  return #tfop("CopyOp",
    a,
    T: T.self)
}

// Computes cos of x element-wise.
@_inlineable @inline(__always)
public static func cos<T: BinaryFloatingPoint>(
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Cos",
    x,
    T: T.self)
}

// Computes hyperbolic cosine of x element-wise.
@_inlineable @inline(__always)
public static func cosh<T: BinaryFloatingPoint>(
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Cosh",
    x,
    T: T.self)
}

// Increments 'ref' until it reaches 'limit'.
//
// - Parameter ref: Should be from a scalar `Variable` node.
//
// - Attr limit: If incrementing ref would bring it above limit, instead generates an
//   'OutOfRange' error.
//
// - Output output: A copy of the input before increment. If nothing else modifies the
//   input, the values produced will all be distinct.
@_inlineable @inline(__always)
public static func countUpTo<T: BinaryInteger>(
  ref: Tensor<T>,
  limit: Int
) -> Tensor<T> {
  return #tfop("CountUpTo",
    ref,
    T: T.self,
    limit: limit)
}

// Extracts crops from the input image tensor and resizes them.
//
// Extracts crops from the input image tensor and resizes them using bilinear
// sampling or nearest neighbor sampling (possibly with aspect ratio change) to a
// common output size specified by `crop_size`. This is more general than the
// `crop_to_bounding_box` op which extracts a fixed size slice from the input image
// and does not allow resizing or aspect ratio change.
//
// Returns a tensor with `crops` from the input `image` at positions defined at the
// bounding box locations in `boxes`. The cropped boxes are all resized (with
// bilinear or nearest neighbor interpolation) to a fixed
// `size = [crop_height, crop_width]`. The result is a 4-D tensor
// `[num_boxes, crop_height, crop_width, depth]`. The resizing is corner aligned.
// In particular, if `boxes = [[0, 0, 1, 1]]`, the method will give identical
// results to using `tf.image.resize_bilinear()` or
// `tf.image.resize_nearest_neighbor()`(depends on the `method` argument) with
// `align_corners=True`.
//
// - Parameters:
//   - image: A 4-D tensor of shape `[batch, image_height, image_width, depth]`.
//     Both `image_height` and `image_width` need to be positive.
//   - boxes: A 2-D tensor of shape `[num_boxes, 4]`. The `i`-th row of the tensor
//     specifies the coordinates of a box in the `box_ind[i]` image and is specified
//     in normalized coordinates `[y1, x1, y2, x2]`. A normalized coordinate value of
//     `y` is mapped to the image coordinate at `y * (image_height - 1)`, so as the
//     `[0, 1]` interval of normalized image height is mapped to
//     `[0, image_height - 1]` in image height coordinates. We do allow `y1` > `y2`, in
//     which case the sampled crop is an up-down flipped version of the original
//     image. The width dimension is treated similarly. Normalized coordinates
//     outside the `[0, 1]` range are allowed, in which case we use
//     `extrapolation_value` to extrapolate the input image values.
//   - box_ind: A 1-D tensor of shape `[num_boxes]` with int32 values in `[0, batch)`.
//     The value of `box_ind[i]` specifies the image that the `i`-th box refers to.
//   - crop_size: A 1-D tensor of 2 elements, `size = [crop_height, crop_width]`. All
//     cropped image patches are resized to this size. The aspect ratio of the image
//     content is not preserved. Both `crop_height` and `crop_width` need to be
//     positive.
//
// - Attrs:
//   - method: A string specifying the sampling method for resizing. It can be either
//     `"bilinear"` or `"nearest"` and default to `"bilinear"`. Currently two sampling
//     methods are supported: Bilinear and Nearest Neighbor.
//   - extrapolation_value: Value used for extrapolation, when applicable.
//
// - Output crops: A 4-D tensor of shape `[num_boxes, crop_height, crop_width, depth]`.
@_inlineable @inline(__always)
public static func cropAndResize<T: Numeric>(
  image: Tensor<T>,
  boxes: Tensor<Float>,
  boxInd: Tensor<Int32>,
  cropSize: Tensor<Int32>,
  method: Method = .bilinear,
  extrapolationValue: Double = 0
) -> Tensor<Float> {
  return #tfop("CropAndResize",
    image,
    boxes,
    boxInd,
    cropSize,
    T: T.self,
    method: method.rawValue,
    extrapolation_value: extrapolationValue)
}

// Computes the gradient of the crop_and_resize op wrt the input boxes tensor.
//
// - Parameters:
//   - grads: A 4-D tensor of shape `[num_boxes, crop_height, crop_width, depth]`.
//   - image: A 4-D tensor of shape `[batch, image_height, image_width, depth]`.
//     Both `image_height` and `image_width` need to be positive.
//   - boxes: A 2-D tensor of shape `[num_boxes, 4]`. The `i`-th row of the tensor
//     specifies the coordinates of a box in the `box_ind[i]` image and is specified
//     in normalized coordinates `[y1, x1, y2, x2]`. A normalized coordinate value of
//     `y` is mapped to the image coordinate at `y * (image_height - 1)`, so as the
//     `[0, 1]` interval of normalized image height is mapped to
//     `[0, image_height - 1] in image height coordinates. We do allow y1 > y2, in
//     which case the sampled crop is an up-down flipped version of the original
//     image. The width dimension is treated similarly. Normalized coordinates
//     outside the `[0, 1]` range are allowed, in which case we use
//     `extrapolation_value` to extrapolate the input image values.
//   - box_ind: A 1-D tensor of shape `[num_boxes]` with int32 values in `[0, batch)`.
//     The value of `box_ind[i]` specifies the image that the `i`-th box refers to.
//
// - Attr method: A string specifying the interpolation method. Only 'bilinear' is
//   supported for now.
//
// - Output output: A 2-D tensor of shape `[num_boxes, 4]`.
@_inlineable @inline(__always)
public static func cropAndResizeGradBoxes<T: Numeric>(
  grads: Tensor<Float>,
  image: Tensor<T>,
  boxes: Tensor<Float>,
  boxInd: Tensor<Int32>,
  method: Method = .bilinear
) -> Tensor<Float> {
  return #tfop("CropAndResizeGradBoxes",
    grads,
    image,
    boxes,
    boxInd,
    T: T.self,
    method: method.rawValue)
}

// Computes the gradient of the crop_and_resize op wrt the input image tensor.
//
// - Parameters:
//   - grads: A 4-D tensor of shape `[num_boxes, crop_height, crop_width, depth]`.
//   - boxes: A 2-D tensor of shape `[num_boxes, 4]`. The `i`-th row of the tensor
//     specifies the coordinates of a box in the `box_ind[i]` image and is specified
//     in normalized coordinates `[y1, x1, y2, x2]`. A normalized coordinate value of
//     `y` is mapped to the image coordinate at `y * (image_height - 1)`, so as the
//     `[0, 1]` interval of normalized image height is mapped to
//     `[0, image_height - 1] in image height coordinates. We do allow y1 > y2, in
//     which case the sampled crop is an up-down flipped version of the original
//     image. The width dimension is treated similarly. Normalized coordinates
//     outside the `[0, 1]` range are allowed, in which case we use
//     `extrapolation_value` to extrapolate the input image values.
//   - box_ind: A 1-D tensor of shape `[num_boxes]` with int32 values in `[0, batch)`.
//     The value of `box_ind[i]` specifies the image that the `i`-th box refers to.
//   - image_size: A 1-D tensor with value `[batch, image_height, image_width, depth]`
//     containing the original image size. Both `image_height` and `image_width` need
//     to be positive.
//
// - Attr method: A string specifying the interpolation method. Only 'bilinear' is
//   supported for now.
//
// - Output output: A 4-D tensor of shape `[batch, image_height, image_width, depth]`.
@_inlineable @inline(__always)
public static func cropAndResizeGradImage<T: BinaryFloatingPoint>(
  grads: Tensor<Float>,
  boxes: Tensor<Float>,
  boxInd: Tensor<Int32>,
  imageSize: Tensor<Int32>,
  method: Method = .bilinear
) -> Tensor<T> {
  return #tfop("CropAndResizeGradImage",
    grads,
    boxes,
    boxInd,
    imageSize,
    T: T.self,
    method: method.rawValue)
}

// Compute the pairwise cross product.
//
// `a` and `b` must be the same shape; they can either be simple 3-element vectors,
// or any shape where the innermost dimension is 3. In the latter case, each pair
// of corresponding 3-element vectors is cross-multiplied independently.
//
// - Parameters:
//   - a: A tensor containing 3-element vectors.
//   - b: Another tensor, of same type and shape as `a`.
//
// - Output product: Pairwise cross product of the vectors in `a` and `b`.
@_inlineable @inline(__always)
public static func cross<T: Numeric>(
  a: Tensor<T>,
  b: Tensor<T>
) -> Tensor<T> {
  return #tfop("Cross",
    a,
    b,
    T: T.self)
}

// A RNN backed by cuDNN.
//
// Computes the RNN from the input and initial states, with respect to the params
// buffer.
//
// rnn_mode: Indicates the type of the RNN model.
// input_mode: Indicate whether there is a linear projection between the input and
//   the actual computation before the first layer. 'skip_input' is only allowed
//   when input_size == num_units; 'auto_select' implies 'skip_input' when
//   input_size == num_units; otherwise, it implies 'linear_input'.
// direction: Indicates whether a bidirectional model will be used. Should be
//   "unidirectional" or "bidirectional".
// dropout: Dropout probability. When set to 0., dropout is disabled.
// seed: The 1st part of a seed to initialize dropout.
// seed2: The 2nd part of a seed to initialize dropout.
// input: A 3-D tensor with the shape of [seq_length, batch_size, input_size].
// input_h: A 3-D tensor with the shape of [num_layer * dir, batch_size,
//     num_units].
// input_c: For LSTM, a 3-D tensor with the shape of
//     [num_layer * dir, batch, num_units]. For other models, it is ignored.
// params: A 1-D tensor that contains the weights and biases in an opaque layout.
//     The size must be created through CudnnRNNParamsSize, and initialized
//     separately. Note that they might not be compatible across different
//     generations. So it is a good idea to save and restore
// output: A 3-D tensor with the shape of [seq_length, batch_size,
//     dir * num_units].
// output_h: The same shape has input_h.
// output_c: The same shape as input_c for LSTM. An empty tensor for other models.
// is_training: Indicates whether this operation is used for inferenece or
//   training.
// reserve_space: An opaque tensor that can be used in backprop calculation. It
//   is only produced if is_training is false.
@_inlineable @inline(__always)
public static func cudnnRNN<T: BinaryFloatingPoint>(
  input: Tensor<T>,
  inputH: Tensor<T>,
  inputC: Tensor<T>,
  params: Tensor<T>,
  rnnMode: RnnMode = .lstm,
  inputMode: InputMode = .linearInput,
  direction: Direction = .unidirectional,
  dropout: Double = 0,
  seed: Int = 0,
  seed2: Int = 0,
  isTraining: Bool = true
) -> (Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>) {
  return #tfop("CudnnRNN",
    input,
    inputH,
    inputC,
    params,
    T: T.self,
    rnn_mode: rnnMode.rawValue,
    input_mode: inputMode.rawValue,
    direction: direction.rawValue,
    dropout: dropout,
    seed: seed,
    seed2: seed2,
    is_training: isTraining)
}

// Backprop step of CudnnRNN.
//
// Compute the backprop of both data and weights in a RNN.
//
// rnn_mode: Indicates the type of the RNN model.
// input_mode: Indicate whether there is a linear projection between the input and
//     the actual computation before the first layer. 'skip_input' is only allowed
//     when input_size == num_units; 'auto_select' implies 'skip_input' when
//     input_size == num_units; otherwise, it implies 'linear_input'.
// direction: Indicates whether a bidirectional model will be used. Should be
//   "unidirectional" or "bidirectional".
// dropout: Dropout probability. When set to 0., dropout is disabled.
// seed: The 1st part of a seed to initialize dropout.
// seed2: The 2nd part of a seed to initialize dropout.
// input: A 3-D tensor with the shape of [seq_length, batch_size, input_size].
// input_h: A 3-D tensor with the shape of [num_layer * dir, batch_size,
//     num_units].
// input_c: For LSTM, a 3-D tensor with the shape of
//     [num_layer * dir, batch, num_units]. For other models, it is ignored.
// params: A 1-D tensor that contains the weights and biases in an opaque layout.
//     The size must be created through CudnnRNNParamsSize, and initialized
//     separately. Note that they might not be compatible across different
//     generations. So it is a good idea to save and restore
// output: A 3-D tensor with the shape of [seq_length, batch_size,
//     dir * num_units].
// output_h: The same shape has input_h.
// output_c: The same shape as input_c for LSTM. An empty tensor for other models.
// output_backprop: A 3-D tensor with the same shape as output in the forward pass.
// output_h_backprop: A 3-D tensor with the same shape as output_h in the forward
//     pass.
// output_c_backprop: A 3-D tensor with the same shape as output_c in the forward
//     pass.
// reserve_space: The same reserve_space produced in for forward operation.
// input_backprop: The backprop to input in the forward pass. Has the same shape
//     as input.
// input_h_backprop: The backprop to input_h in the forward pass. Has the same
//     shape as input_h.
// input_c_backprop: The backprop to input_c in the forward pass. Has the same
//     shape as input_c.
// params_backprop: The backprop to the params buffer in the forward pass. Has the
//     same shape as params.
@_inlineable @inline(__always)
public static func cudnnRNNBackprop<T: BinaryFloatingPoint>(
  input: Tensor<T>,
  inputH: Tensor<T>,
  inputC: Tensor<T>,
  params: Tensor<T>,
  output: Tensor<T>,
  outputH: Tensor<T>,
  outputC: Tensor<T>,
  outputBackprop: Tensor<T>,
  outputHBackprop: Tensor<T>,
  outputCBackprop: Tensor<T>,
  reserveSpace: Tensor<T>,
  rnnMode: RnnMode = .lstm,
  inputMode: InputMode = .linearInput,
  direction: Direction = .unidirectional,
  dropout: Double = 0,
  seed: Int = 0,
  seed2: Int = 0
) -> (Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>) {
  return #tfop("CudnnRNNBackprop",
    input,
    inputH,
    inputC,
    params,
    output,
    outputH,
    outputC,
    outputBackprop,
    outputHBackprop,
    outputCBackprop,
    reserveSpace,
    T: T.self,
    rnn_mode: rnnMode.rawValue,
    input_mode: inputMode.rawValue,
    direction: direction.rawValue,
    dropout: dropout,
    seed: seed,
    seed2: seed2)
}

// Converts CudnnRNN params from canonical form to usable form.
//
// Writes a set of weights into the opaque params buffer so they can be used in
// upcoming training or inferences.
//
// Note that the params buffer may not be compatible across different GPUs. So any
// save and restoration should be converted to and from the canonical weights and
// biases.
//
// num_layers: Specifies the number of layers in the RNN model.
// num_units: Specifies the size of the hidden state.
// input_size: Specifies the size of the input state.
// weights: the canonical form of weights that can be used for saving
//     and restoration. They are more likely to be compatible across different
//     generations.
// biases: the canonical form of biases that can be used for saving
//     and restoration. They are more likely to be compatible across different
//     generations.
// num_params: number of parameter sets for all layers.
//     Each layer may contain multiple parameter sets, with each set consisting of
//     a weight matrix and a bias vector.
// rnn_mode: Indicates the type of the RNN model.
// input_mode: Indicate whether there is a linear projection between the input and
//     The actual computation before the first layer. 'skip_input' is only allowed
//     when input_size == num_units; 'auto_select' implies 'skip_input' when
//     input_size == num_units; otherwise, it implies 'linear_input'.
// direction: Indicates whether a bidirectional model will be used.
//     dir = (direction == bidirectional) ? 2 : 1
// dropout: dropout probability. When set to 0., dropout is disabled.
// seed: the 1st part of a seed to initialize dropout.
// seed2: the 2nd part of a seed to initialize dropout.
@_inlineable @inline(__always)
public static func cudnnRNNCanonicalToParams<T: BinaryFloatingPoint>(
  numLayers: Tensor<Int32>,
  numUnits: Tensor<Int32>,
  inputSize: Tensor<Int32>,
  weights: [Tensor<T>],
  biases: [Tensor<T>],
  rnnMode: RnnMode = .lstm,
  inputMode: InputMode = .linearInput,
  direction: Direction = .unidirectional,
  dropout: Double = 0,
  seed: Int = 0,
  seed2: Int = 0
) -> Tensor<T> {
  return #tfop("CudnnRNNCanonicalToParams",
    numLayers,
    numUnits,
    inputSize,
    weights,
    biases,
    T: T.self,
    rnn_mode: rnnMode.rawValue,
    input_mode: inputMode.rawValue,
    direction: direction.rawValue,
    dropout: dropout,
    seed: seed,
    seed2: seed2)
}

// Computes size of weights that can be used by a Cudnn RNN model.
//
// Return the params size that can be used by the Cudnn RNN model. Subsequent
// weight allocation and initialization should use this size.
//
// num_layers: Specifies the number of layers in the RNN model.
// num_units: Specifies the size of the hidden state.
// input_size: Specifies the size of the input state.
// rnn_mode: Indicates the type of the RNN model.
// input_mode: Indicate whether there is a linear projection between the input and
//   The actual computation before the first layer. 'skip_input' is only allowed
//   when input_size == num_units; 'auto_select' implies 'skip_input' when
//   input_size == num_units; otherwise, it implies 'linear_input'.
// direction: Indicates whether a bidirectional model will be used.
//   dir = (direction == bidirectional) ? 2 : 1
// dropout: dropout probability. When set to 0., dropout is disabled.
// seed: the 1st part of a seed to initialize dropout.
// seed2: the 2nd part of a seed to initialize dropout.
// params_size: The size of the params buffer that should be allocated and
//   initialized for this RNN model. Note that this params buffer may not be
//   compatible across GPUs. Please use CudnnRNNParamsWeights and
//   CudnnRNNParamsBiases to save and restore them in a way that is compatible
//   across different runs.
@_inlineable @inline(__always)
public static func cudnnRNNParamsSize<T: BinaryFloatingPoint, S: BinaryInteger>(
  numLayers: Tensor<Int32>,
  numUnits: Tensor<Int32>,
  inputSize: Tensor<Int32>,
  rnnMode: RnnMode = .lstm,
  inputMode: InputMode = .linearInput,
  direction: Direction = .unidirectional,
  dropout: Double = 0,
  seed: Int = 0,
  seed2: Int = 0,
  typeT: T.Type
) -> Tensor<S> {
  return #tfop("CudnnRNNParamsSize",
    numLayers,
    numUnits,
    inputSize,
    T: T.self,
    S: S.self,
    rnn_mode: rnnMode.rawValue,
    input_mode: inputMode.rawValue,
    direction: direction.rawValue,
    dropout: dropout,
    seed: seed,
    seed2: seed2)
}

// Retrieves CudnnRNN params in canonical form.
//
// Retrieves a set of weights from the opaque params buffer that can be saved and
// restored in a way compatible with future runs.
//
// Note that the params buffer may not be compatible across different GPUs. So any
// save and restoration should be converted to and from the canonical weights and
// biases.
//
// num_layers: Specifies the number of layers in the RNN model.
// num_units: Specifies the size of the hidden state.
// input_size: Specifies the size of the input state.
// num_params: number of parameter sets for all layers.
//     Each layer may contain multiple parameter sets, with each set consisting of
//     a weight matrix and a bias vector.
// weights: the canonical form of weights that can be used for saving
//     and restoration. They are more likely to be compatible across different
//     generations.
// biases: the canonical form of biases that can be used for saving
//     and restoration. They are more likely to be compatible across different
//     generations.
// rnn_mode: Indicates the type of the RNN model.
// input_mode: Indicate whether there is a linear projection between the input and
//     The actual computation before the first layer. 'skip_input' is only allowed
//     when input_size == num_units; 'auto_select' implies 'skip_input' when
//     input_size == num_units; otherwise, it implies 'linear_input'.
// direction: Indicates whether a bidirectional model will be used.
//     dir = (direction == bidirectional) ? 2 : 1
// dropout: dropout probability. When set to 0., dropout is disabled.
// seed: the 1st part of a seed to initialize dropout.
// seed2: the 2nd part of a seed to initialize dropout.
@_inlineable @inline(__always)
public static func cudnnRNNParamsToCanonical<T: BinaryFloatingPoint>(
  numLayers: Tensor<Int32>,
  numUnits: Tensor<Int32>,
  inputSize: Tensor<Int32>,
  params: Tensor<T>,
  numParams: Int,
  rnnMode: RnnMode = .lstm,
  inputMode: InputMode = .linearInput,
  direction: Direction = .unidirectional,
  dropout: Double = 0,
  seed: Int = 0,
  seed2: Int = 0
) -> ([Tensor<T>], [Tensor<T>]) {
  return #tfop("CudnnRNNParamsToCanonical",
    numLayers,
    numUnits,
    inputSize,
    params,
    T: T.self,
    num_params: numParams,
    rnn_mode: rnnMode.rawValue,
    input_mode: inputMode.rawValue,
    direction: direction.rawValue,
    dropout: dropout,
    seed: seed,
    seed2: seed2)
}

// Compute the cumulative product of the tensor `x` along `axis`.
//
// By default, this op performs an inclusive cumprod, which means that the first
// element of the input is identical to the first element of the output:
//
// ```python
// tf.cumprod([a, b, c])  # => [a, a * b, a * b * c]
// ```
//
// By setting the `exclusive` kwarg to `True`, an exclusive cumprod is
// performed instead:
//
// ```python
// tf.cumprod([a, b, c], exclusive=True)  # => [1, a, a * b]
// ```
//
// By setting the `reverse` kwarg to `True`, the cumprod is performed in the
// opposite direction:
//
// ```python
// tf.cumprod([a, b, c], reverse=True)  # => [a * b * c, b * c, c]
// ```
//
// This is more efficient than using separate `tf.reverse` ops.
//
// The `reverse` and `exclusive` kwargs can also be combined:
//
// ```python
// tf.cumprod([a, b, c], exclusive=True, reverse=True)  # => [b * c, c, 1]
// ```
//
// - Parameters:
//   - x: A `Tensor`. Must be one of the following types: `float32`, `float64`,
//     `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`,
//     `complex128`, `qint8`, `quint8`, `qint32`, `half`.
//   - axis: A `Tensor` of type `int32` (default: 0). Must be in the range
//     `[-rank(x), rank(x))`.
//
// - Attrs:
//   - exclusive: If `True`, perform exclusive cumprod.
//   - reverse: A `bool` (default: False).
@_inlineable @inline(__always)
public static func cumprod<T: Numeric, Tidx: BinaryInteger>(
  x: Tensor<T>,
  axis: Tensor<Tidx>,
  exclusive: Bool = false,
  reverse: Bool = false
) -> Tensor<T> {
  return #tfop("Cumprod",
    x,
    axis,
    T: T.self,
    Tidx: Tidx.self,
    exclusive: exclusive,
    reverse: reverse)
}

// Compute the cumulative sum of the tensor `x` along `axis`.
//
// By default, this op performs an inclusive cumsum, which means that the first
// element of the input is identical to the first element of the output:
//
// ```python
// tf.cumsum([a, b, c])  # => [a, a + b, a + b + c]
// ```
//
// By setting the `exclusive` kwarg to `True`, an exclusive cumsum is
// performed instead:
//
// ```python
// tf.cumsum([a, b, c], exclusive=True)  # => [0, a, a + b]
// ```
//
// By setting the `reverse` kwarg to `True`, the cumsum is performed in the
// opposite direction:
//
// ```python
// tf.cumsum([a, b, c], reverse=True)  # => [a + b + c, b + c, c]
// ```
//
// This is more efficient than using separate `tf.reverse` ops.
//
// The `reverse` and `exclusive` kwargs can also be combined:
//
// ```python
// tf.cumsum([a, b, c], exclusive=True, reverse=True)  # => [b + c, c, 0]
// ```
//
// - Parameters:
//   - x: A `Tensor`. Must be one of the following types: `float32`, `float64`,
//     `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`,
//     `complex128`, `qint8`, `quint8`, `qint32`, `half`.
//   - axis: A `Tensor` of type `int32` (default: 0). Must be in the range
//     `[-rank(x), rank(x))`.
//
// - Attrs:
//   - exclusive: If `True`, perform exclusive cumsum.
//   - reverse: A `bool` (default: False).
@_inlineable @inline(__always)
public static func cumsum<T: Numeric, Tidx: BinaryInteger>(
  x: Tensor<T>,
  axis: Tensor<Tidx>,
  exclusive: Bool = false,
  reverse: Bool = false
) -> Tensor<T> {
  return #tfop("Cumsum",
    x,
    axis,
    T: T.self,
    Tidx: Tidx.self,
    exclusive: exclusive,
    reverse: reverse)
}

// Returns the dimension index in the destination data format given the one in
//
// the source data format.
//
// - Parameter x: A Tensor with each element as a dimension index in source data format.
//   Must be in the range [-4, 4).
//
// - Attrs:
//   - src_format: source data format.
//   - dst_format: destination data format.
//
// - Output y: A Tensor with each element as a dimension index in destination data format.
@_inlineable @inline(__always)
public static func dataFormatDimMap<T: BinaryInteger>(
  x: Tensor<T>,
  srcFormat: String = "NHWC",
  dstFormat: String = "NCHW"
) -> Tensor<T> {
  return #tfop("DataFormatDimMap",
    x,
    T: T.self,
    src_format: srcFormat,
    dst_format: dstFormat)
}

// Returns the permuted vector/tensor in the destination data format given the
//
// one in the source data format.
//
// - Parameter x: Vector of size 4 or Tensor of shape (4, 2) in source data format.
//
// - Attrs:
//   - src_format: source data format.
//   - dst_format: destination data format.
//
// - Output y: Vector of size 4 or Tensor of shape (4, 2) in destination data format.
@_inlineable @inline(__always)
public static func dataFormatVecPermute<T: BinaryInteger>(
  x: Tensor<T>,
  srcFormat: String = "NHWC",
  dstFormat: String = "NCHW"
) -> Tensor<T> {
  return #tfop("DataFormatVecPermute",
    x,
    T: T.self,
    src_format: srcFormat,
    dst_format: dstFormat)
}

// Identity op for gradient debugging.
//
// This op is hidden from public in Python. It is used by TensorFlow Debugger to
// register gradient tensors for gradient debugging.
// This op operates on non-reference-type tensors.
@_inlineable @inline(__always)
public static func debugGradientIdentity<T: Numeric>(
  input: Tensor<T>
) -> Tensor<T> {
  return #tfop("DebugGradientIdentity",
    input,
    T: T.self)
}

// Identity op for gradient debugging.
//
// This op is hidden from public in Python. It is used by TensorFlow Debugger to
// register gradient tensors for gradient debugging.
// This op operates on reference-type tensors.
@_inlineable @inline(__always)
public static func debugGradientRefIdentity<T: Numeric>(
  input: Tensor<T>
) -> Tensor<T> {
  return #tfop("DebugGradientRefIdentity",
    input,
    T: T.self)
}

// Debug Identity Op.
//
// Provides an identity mapping of the non-Ref type input tensor for debugging.
//
// - Parameter input: Input tensor, non-Reference type.
//
// - Attrs:
//   - tensor_name: Name of the input tensor.
//   - debug_urls: List of URLs to debug targets, e.g.,
//     file:///foo/tfdbg_dump, grpc:://localhost:11011
//   - gated_grpc: Whether this op will be gated. If any of the debug_urls of this
//     debug node is of the grpc:// scheme, when the value of this attribute is set
//     to True, the data will not actually be sent via the grpc stream unless this
//     debug op has been enabled at the debug_url. If all of the debug_urls of this
//     debug node are of the grpc:// scheme and the debug op is enabled at none of
//     them, the output will be an empty Tensor.
//
// - Output output: Output tensor that equals the input tensor.
@_inlineable @inline(__always)
public static func debugIdentity<T: Numeric>(
  input: Tensor<T>,
  deviceName: String,
  tensorName: String,
  debugUrls: [String],
  gatedGrpc: Bool = false
) -> Tensor<T> {
  return #tfop("DebugIdentity",
    input,
    T: T.self,
    device_name: deviceName,
    tensor_name: tensorName,
    debug_urls: debugUrls,
    gated_grpc: gatedGrpc)
}

// Debug NaN Value Counter Op
//
// Counts number of NaNs in the input tensor, for debugging.
//
// - Parameter input: Input tensor, non-Reference type.
//
// - Attrs:
//   - tensor_name: Name of the input tensor.
//   - debug_urls: List of URLs to debug targets, e.g.,
//     file:///foo/tfdbg_dump, grpc:://localhost:11011.
//   - gated_grpc: Whether this op will be gated. If any of the debug_urls of this
//     debug node is of the grpc:// scheme, when the value of this attribute is set
//     to True, the data will not actually be sent via the grpc stream unless this
//     debug op has been enabled at the debug_url. If all of the debug_urls of this
//     debug node are of the grpc:// scheme and the debug op is enabled at none of
//     them, the output will be an empty Tensor.
//
// - Output output: An integer output tensor that is the number of NaNs in the input.
@_inlineable @inline(__always)
public static func debugNanCount<T: Numeric>(
  input: Tensor<T>,
  deviceName: String,
  tensorName: String,
  debugUrls: [String],
  gatedGrpc: Bool = false
) -> Tensor<Int64> {
  return #tfop("DebugNanCount",
    input,
    T: T.self,
    device_name: deviceName,
    tensor_name: tensorName,
    debug_urls: debugUrls,
    gated_grpc: gatedGrpc)
}

// Debug Numeric Summary Op.
//
// Provide a basic summary of numeric value types, range and distribution.
//
// - Parameter input: Input tensor, non-Reference type, float or double.
//
// - Attrs:
//   - tensor_name: Name of the input tensor.
//   - debug_urls: List of URLs to debug targets, e.g.,
//     file:///foo/tfdbg_dump, grpc:://localhost:11011
//   - lower_bound: (float) The lower bound <= which values will be included in the
//     generalized -inf count. Default: -inf.
//   - upper_bound: (float) The upper bound >= which values will be included in the
//     generalized +inf count. Default: +inf.
//   - mute_if_healthy: (bool) Do not send data to the debug URLs unless at least one
//     of elements [2], [3] and [7] (i.e., the nan count and the generalized -inf and
//     inf counts) is non-zero.
//   - gated_grpc: Whether this op will be gated. If any of the debug_urls of this
//     debug node is of the grpc:// scheme, when the value of this attribute is set
//     to True, the data will not actually be sent via the grpc stream unless this
//     debug op has been enabled at the debug_url. If all of the debug_urls of this
//     debug node are of the grpc:// scheme and the debug op is enabled at none of
//     them, the output will be an empty Tensor.
//
// - Output output: A double tensor of shape [14 + nDimensions], where nDimensions is the
//     the number of dimensions of the tensor's shape. The elements of output are:
//     [0]: is initialized (1.0) or not (0.0).
//     [1]: total number of elements
//     [2]: NaN element count
//     [3]: generalized -inf count: elements <= lower_bound. lower_bound is -inf by
//       default.
//     [4]: negative element count (excluding -inf), if lower_bound is the default
//       -inf. Otherwise, this is the count of elements > lower_bound and < 0.
//     [5]: zero element count
//     [6]: positive element count (excluding +inf), if upper_bound is the default
//       -inf. Otherwise, this is the count of elements < upper_bound and > 0.
//     [7]: generalized +inf count, elements >= upper_bound. upper_bound is +inf by
//       default.
//   Output elements [1:8] are all zero, if the tensor is uninitialized.
//     [8]: minimum of all non-inf and non-NaN elements.
//          If uninitialized or no such element exists: +inf.
//     [9]: maximum of all non-inf and non-NaN elements.
//          If uninitialized or no such element exists: -inf.
//     [10]: mean of all non-inf and non-NaN elements.
//           If uninitialized or no such element exists: NaN.
//     [11]: variance of all non-inf and non-NaN elements.
//           If uninitialized or no such element exists: NaN.
//     [12]: Data type of the tensor encoded as an enum integer. See the DataType
//           proto for more details.
//     [13]: Number of dimensions of the tensor (ndims).
//     [14+]: Sizes of the dimensions.
@_inlineable @inline(__always)
public static func debugNumericSummary<T: Numeric>(
  input: Tensor<T>,
  deviceName: String,
  tensorName: String,
  debugUrls: [String],
  lowerBound: Double = -Double.infinity,
  upperBound: Double = Double.infinity,
  muteIfHealthy: Bool = false,
  gatedGrpc: Bool = false
) -> Tensor<Double> {
  return #tfop("DebugNumericSummary",
    input,
    T: T.self,
    device_name: deviceName,
    tensor_name: tensorName,
    debug_urls: debugUrls,
    lower_bound: lowerBound,
    upper_bound: upperBound,
    mute_if_healthy: muteIfHealthy,
    gated_grpc: gatedGrpc)
}

// Makes a copy of `x`.
//
// - Parameter x: The source tensor of type `T`.
//
// - Output y:     y: A `Tensor` of type `T`. A copy of `x`. Guaranteed that `y`
//         is not an alias of `x`.
@_inlineable @inline(__always)
public static func deepCopy<T: Numeric>(
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("DeepCopy",
    x,
    T: T.self)
}

// Applies set operation along last dimension of 2 `Tensor` inputs.
//
// See SetOperationOp::SetOperationFromContext for values of `set_operation`.
//
// Output `result` is a `SparseTensor` represented by `result_indices`,
// `result_values`, and `result_shape`. For `set1` and `set2` ranked `n`, this
// has rank `n` and the same 1st `n-1` dimensions as `set1` and `set2`. The `nth`
// dimension contains the result of `set_operation` applied to the corresponding
// `[0...n-1]` dimension of `set`.
//
// - Parameters:
//   - set1: `Tensor` with rank `n`. 1st `n-1` dimensions must be the same as `set2`.
//     Dimension `n` contains values in a set, duplicates are allowed but ignored.
//   - set2: `Tensor` with rank `n`. 1st `n-1` dimensions must be the same as `set1`.
//     Dimension `n` contains values in a set, duplicates are allowed but ignored.
//
// - Outputs:
//   - result_indices: 2D indices of a `SparseTensor`.
//   - result_values: 1D values of a `SparseTensor`.
//   - result_shape: 1D `Tensor` shape of a `SparseTensor`. `result_shape[0...n-1]` is
//     the same as the 1st `n-1` dimensions of `set1` and `set2`, `result_shape[n]`
//     is the max result set size across all `0...n-1` dimensions.
@_inlineable @inline(__always)
public static func denseToDenseSetOperation<T: BinaryInteger>(
  set1: Tensor<T>,
  set2: Tensor<T>,
  setOperation: String,
  validateIndices: Bool = true
) -> (Tensor<Int64>, Tensor<T>, Tensor<Int64>) {
  return #tfop("DenseToDenseSetOperation",
    set1,
    set2,
    T: T.self,
    set_operation: setOperation,
    validate_indices: validateIndices)
}

// Applies set operation along last dimension of `Tensor` and `SparseTensor`.
//
// See SetOperationOp::SetOperationFromContext for values of `set_operation`.
//
// Input `set2` is a `SparseTensor` represented by `set2_indices`, `set2_values`,
// and `set2_shape`. For `set2` ranked `n`, 1st `n-1` dimensions must be the same
// as `set1`. Dimension `n` contains values in a set, duplicates are allowed but
// ignored.
//
// If `validate_indices` is `True`, this op validates the order and range of `set2`
// indices.
//
// Output `result` is a `SparseTensor` represented by `result_indices`,
// `result_values`, and `result_shape`. For `set1` and `set2` ranked `n`, this
// has rank `n` and the same 1st `n-1` dimensions as `set1` and `set2`. The `nth`
// dimension contains the result of `set_operation` applied to the corresponding
// `[0...n-1]` dimension of `set`.
//
// - Parameters:
//   - set1: `Tensor` with rank `n`. 1st `n-1` dimensions must be the same as `set2`.
//     Dimension `n` contains values in a set, duplicates are allowed but ignored.
//   - set2_indices: 2D `Tensor`, indices of a `SparseTensor`. Must be in row-major
//     order.
//   - set2_values: 1D `Tensor`, values of a `SparseTensor`. Must be in row-major
//     order.
//   - set2_shape: 1D `Tensor`, shape of a `SparseTensor`. `set2_shape[0...n-1]` must
//     be the same as the 1st `n-1` dimensions of `set1`, `result_shape[n]` is the
//     max set size across `n-1` dimensions.
//
// - Outputs:
//   - result_indices: 2D indices of a `SparseTensor`.
//   - result_values: 1D values of a `SparseTensor`.
//   - result_shape: 1D `Tensor` shape of a `SparseTensor`. `result_shape[0...n-1]` is
//     the same as the 1st `n-1` dimensions of `set1` and `set2`, `result_shape[n]`
//     is the max result set size across all `0...n-1` dimensions.
@_inlineable @inline(__always)
public static func denseToSparseSetOperation<T: BinaryInteger>(
  set1: Tensor<T>,
  set2Indices: Tensor<Int64>,
  set2Values: Tensor<T>,
  set2Shape: Tensor<Int64>,
  setOperation: String,
  validateIndices: Bool = true
) -> (Tensor<Int64>, Tensor<T>, Tensor<Int64>) {
  return #tfop("DenseToSparseSetOperation",
    set1,
    set2Indices,
    set2Values,
    set2Shape,
    T: T.self,
    set_operation: setOperation,
    validate_indices: validateIndices)
}

// DepthToSpace for tensors of type T.
//
// Rearranges data from depth into blocks of spatial data.
// This is the reverse transformation of SpaceToDepth. More specifically,
// this op outputs a copy of the input tensor where values from the `depth`
// dimension are moved in spatial blocks to the `height` and `width` dimensions.
// The attr `block_size` indicates the input block size and how the data is moved.
//
//   * Chunks of data of size `block_size * block_size` from depth are rearranged
//     into non-overlapping blocks of size `block_size x block_size`
//   * The width the output tensor is `input_depth * block_size`, whereas the
//     height is `input_height * block_size`.
//   * The Y, X coordinates within each block of the output image are determined
//     by the high order component of the input channel index.
//   * The depth of the input tensor must be divisible by
//     `block_size * block_size`.
//
// The `data_format` attr specifies the layout of the input and output tensors
// with the following options:
//   "NHWC": `[ batch, height, width, channels ]`
//   "NCHW": `[ batch, channels, height, width ]`
//   "NCHW_VECT_C":
//       `qint8 [ batch, channels / 4, height, width, 4 ]`
//
// It is useful to consider the operation as transforming a 6-D Tensor.
// e.g. for data_format = NHWC,
//      Each element in the input tensor can be specified via 6 coordinates,
//      ordered by decreasing memory layout significance as:
//      n,iY,iX,bY,bX,oC  (where n=batch index, iX, iY means X or Y coordinates
//                         within the input image, bX, bY means coordinates
//                         within the output block, oC means output channels).
//      The output would be the input transposed to the following layout:
//      n,iY,bY,iX,bX,oC
//
// This operation is useful for resizing the activations between convolutions
// (but keeping all data), e.g. instead of pooling. It is also useful for training
// purely convolutional models.
//
// For example, given an input of shape `[1, 1, 1, 4]`, data_format = "NHWC" and
// block_size = 2:
//
// ```
// x = [[[[1, 2, 3, 4]]]]
//
// ```
//
// This operation will output a tensor of shape `[1, 2, 2, 1]`:
//
// ```
//    [[[[1], [2]],
//      [[3], [4]]]]
// ```
//
// Here, the input has a batch of 1 and each batch element has shape `[1, 1, 4]`,
// the corresponding output will have 2x2 elements and will have a depth of
// 1 channel (1 = `4 / (block_size * block_size)`).
// The output element shape is `[2, 2, 1]`.
//
// For an input tensor with larger depth, here of shape `[1, 1, 1, 12]`, e.g.
//
// ```
// x = [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
// ```
//
// This operation, for block size of 2, will return the following tensor of shape
// `[1, 2, 2, 3]`
//
// ```
//    [[[[1, 2, 3], [4, 5, 6]],
//      [[7, 8, 9], [10, 11, 12]]]]
//
// ```
//
// Similarly, for the following input of shape `[1 2 2 4]`, and a block size of 2:
//
// ```
// x =  [[[[1, 2, 3, 4],
//        [5, 6, 7, 8]],
//       [[9, 10, 11, 12],
//        [13, 14, 15, 16]]]]
// ```
//
// the operator will return the following tensor of shape `[1 4 4 1]`:
//
// ```
// x = [[[ [1],   [2],  [5],  [6]],
//       [ [3],   [4],  [7],  [8]],
//       [ [9],  [10], [13],  [14]],
//       [ [11], [12], [15],  [16]]]]
//
// ```
//
// - Attr block_size: The size of the spatial block, same as in Space2Depth.
@_inlineable @inline(__always)
public static func depthToSpace<T: Numeric>(
  input: Tensor<T>,
  blockSize: Int,
  dataFormat: DataFormat2 = .nhwc
) -> Tensor<T> {
  return #tfop("DepthToSpace",
    input,
    T: T.self,
    block_size: blockSize,
    data_format: dataFormat.rawValue)
}

// Computes a 2-D depthwise convolution given 4-D `input` and `filter` tensors.
//
// Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
// and a filter / kernel tensor of shape
// `[filter_height, filter_width, in_channels, channel_multiplier]`, containing
// `in_channels` convolutional filters of depth 1, `depthwise_conv2d` applies
// a different filter to each input channel (expanding from 1 channel to
// `channel_multiplier` channels for each), then concatenates the results
// together. Thus, the output has `in_channels * channel_multiplier` channels.
//
// ```
// for k in 0..in_channels-1
//   for q in 0..channel_multiplier-1
//     output[b, i, j, k * channel_multiplier + q] =
//       sum_{di, dj} input[b, strides[1] * i + di, strides[2] * j + dj, k] *
//                         filter[di, dj, k, q]
// ```
//
// Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
// horizontal and vertices strides, `strides = [1, stride, stride, 1]`.
//
// - Attrs:
//   - strides: 1-D of length 4.  The stride of the sliding window for each dimension
//     of `input`.
//   - padding: The type of padding algorithm to use.
//   - data_format: Specify the data format of the input and output data. With the
//     default format "NHWC", the data is stored in the order of:
//         [batch, height, width, channels].
//     Alternatively, the format could be "NCHW", the data storage order of:
//         [batch, channels, height, width].
//   - dilations: 1-D tensor of length 4.  The dilation factor for each dimension of
//     `input`. If set to k > 1, there will be k-1 skipped cells between each filter
//     element on that dimension. The dimension order is determined by the value of
//     `data_format`, see above for details. Dilations in the batch and depth
//     dimensions must be 1.
@_inlineable @inline(__always)
public static func depthwiseConv2dNative<T: BinaryFloatingPoint>(
  input: Tensor<T>,
  filter: Tensor<T>,
  strides: [Int],
  padding: Padding,
  dataFormat: DataFormat = .nhwc,
  dilations: [Int]
) -> Tensor<T> {
  return #tfop("DepthwiseConv2dNative",
    input,
    filter,
    T: T.self,
    strides: strides,
    padding: padding.rawValue,
    data_format: dataFormat.rawValue,
    dilations: dilations)
}

// Computes the gradients of depthwise convolution with respect to the filter.
//
// - Parameters:
//   - input: 4-D with shape based on `data_format`.  For example, if
//     `data_format` is 'NHWC' then `input` is a 4-D `[batch, in_height,
//     in_width, in_channels]` tensor.
//   - filter_sizes: An integer vector representing the tensor shape of `filter`,
//     where `filter` is a 4-D
//     `[filter_height, filter_width, in_channels, depthwise_multiplier]` tensor.
//   - out_backprop: 4-D with shape  based on `data_format`.
//     For example, if `data_format` is 'NHWC' then
//     out_backprop shape is `[batch, out_height, out_width, out_channels]`.
//     Gradients w.r.t. the output of the convolution.
//
// - Attrs:
//   - strides: The stride of the sliding window for each dimension of the input
//     of the convolution.
//   - padding: The type of padding algorithm to use.
//   - data_format: Specify the data format of the input and output data. With the
//     default format "NHWC", the data is stored in the order of:
//         [batch, height, width, channels].
//     Alternatively, the format could be "NCHW", the data storage order of:
//         [batch, channels, height, width].
//   - dilations: 1-D tensor of length 4.  The dilation factor for each dimension of
//     `input`. If set to k > 1, there will be k-1 skipped cells between each filter
//     element on that dimension. The dimension order is determined by the value of
//     `data_format`, see above for details. Dilations in the batch and depth
//     dimensions must be 1.
//
// - Output output: 4-D with shape
//   `[filter_height, filter_width, in_channels, out_channels]`.  Gradient w.r.t.
//   the `filter` input of the convolution.
@_inlineable @inline(__always)
public static func depthwiseConv2dNativeBackpropFilter<T: BinaryFloatingPoint>(
  input: Tensor<T>,
  filterSizes: Tensor<Int32>,
  outBackprop: Tensor<T>,
  strides: [Int],
  padding: Padding,
  dataFormat: DataFormat = .nhwc,
  dilations: [Int]
) -> Tensor<T> {
  return #tfop("DepthwiseConv2dNativeBackpropFilter",
    input,
    filterSizes,
    outBackprop,
    T: T.self,
    strides: strides,
    padding: padding.rawValue,
    data_format: dataFormat.rawValue,
    dilations: dilations)
}

// Computes the gradients of depthwise convolution with respect to the input.
//
// - Parameters:
//   - input_sizes: An integer vector representing the shape of `input`, based
//     on `data_format`.  For example, if `data_format` is 'NHWC' then
//      `input` is a 4-D `[batch, height, width, channels]` tensor.
//   - filter: 4-D with shape
//     `[filter_height, filter_width, in_channels, depthwise_multiplier]`.
//   - out_backprop: 4-D with shape  based on `data_format`.
//     For example, if `data_format` is 'NHWC' then
//     out_backprop shape is `[batch, out_height, out_width, out_channels]`.
//     Gradients w.r.t. the output of the convolution.
//
// - Attrs:
//   - strides: The stride of the sliding window for each dimension of the input
//     of the convolution.
//   - padding: The type of padding algorithm to use.
//   - data_format: Specify the data format of the input and output data. With the
//     default format "NHWC", the data is stored in the order of:
//         [batch, height, width, channels].
//     Alternatively, the format could be "NCHW", the data storage order of:
//         [batch, channels, height, width].
//   - dilations: 1-D tensor of length 4.  The dilation factor for each dimension of
//     `input`. If set to k > 1, there will be k-1 skipped cells between each filter
//     element on that dimension. The dimension order is determined by the value of
//     `data_format`, see above for details. Dilations in the batch and depth
//     dimensions must be 1.
//
// - Output output: 4-D with shape according to `data_format`.  For example, if
//   `data_format` is 'NHWC', output shape is `[batch, in_height,
//   in_width, in_channels]`.  Gradient w.r.t. the input of the
//   convolution.
@_inlineable @inline(__always)
public static func depthwiseConv2dNativeBackpropInput<T: BinaryFloatingPoint>(
  inputSizes: Tensor<Int32>,
  filter: Tensor<T>,
  outBackprop: Tensor<T>,
  strides: [Int],
  padding: Padding,
  dataFormat: DataFormat = .nhwc,
  dilations: [Int]
) -> Tensor<T> {
  return #tfop("DepthwiseConv2dNativeBackpropInput",
    inputSizes,
    filter,
    outBackprop,
    T: T.self,
    strides: strides,
    padding: padding.rawValue,
    data_format: dataFormat.rawValue,
    dilations: dilations)
}

// Dequantize the 'input' tensor into a float Tensor.
//
// [min_range, max_range] are scalar floats that specify the range for
// the 'input' data. The 'mode' attribute controls exactly which calculations are
// used to convert the float values to their quantized equivalents.
//
// In 'MIN_COMBINED' mode, each value of the tensor will undergo the following:
//
// ```
// if T == qint8, in[i] += (range(T) + 1)/ 2.0
// out[i] = min_range + (in[i]* (max_range - min_range) / range(T))
// ```
// here `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`
//
// *MIN_COMBINED Mode Example*
//
// If the input comes from a QuantizedRelu6, the output type is
// quint8 (range of 0-255) but the possible range of QuantizedRelu6 is
// 0-6.  The min_range and max_range values are therefore 0.0 and 6.0.
// Dequantize on quint8 will take each value, cast to float, and multiply
// by 6 / 255.
// Note that if quantizedtype is qint8, the operation will additionally add
// each value by 128 prior to casting.
//
// If the mode is 'MIN_FIRST', then this approach is used:
//
// ```c++
// num_discrete_values = 1 << (# of bits in T)
// range_adjust = num_discrete_values / (num_discrete_values - 1)
// range = (range_max - range_min) * range_adjust
// range_scale = range / num_discrete_values
// const double offset_input = static_cast<double>(input) - lowest_quantized;
// result = range_min + ((input - numeric_limits<T>::min()) * range_scale)
// ```
//
// *SCALED mode Example*
//
// `SCALED` mode matches the quantization approach used in
// `QuantizeAndDequantize{V2|V3}`.
//
// If the mode is `SCALED`, we do not use the full range of the output type,
// choosing to elide the lowest possible value for symmetry (e.g., output range is
// -127 to 127, not -128 to 127 for signed 8 bit quantization), so that 0.0 maps to
// 0.
//
// We first find the range of values in our tensor. The
// range we use is always centered on 0, so we find m such that
// ```c++
//   m = max(abs(input_min), abs(input_max))
// ```
//
// Our input tensor range is then `[-m, m]`.
//
// Next, we choose our fixed-point quantization buckets, `[min_fixed, max_fixed]`.
// If T is signed, this is
// ```
//   num_bits = sizeof(T) * 8
//   [min_fixed, max_fixed] =
//       [-(1 << (num_bits - 1) - 1), (1 << (num_bits - 1)) - 1]
// ```
//
// Otherwise, if T is unsigned, the fixed-point range is
// ```
//   [min_fixed, max_fixed] = [0, (1 << num_bits) - 1]
// ```
//
// From this we compute our scaling factor, s:
// ```c++
//   s = (2 * m) / (max_fixed - min_fixed)
// ```
//
// Now we can dequantize the elements of our tensor:
// ```c++
// result = input * s
// ```
//
// - Parameters:
//   - min_range: The minimum scalar value possibly produced for the input.
//   - max_range: The maximum scalar value possibly produced for the input.
@_inlineable @inline(__always)
public static func dequantize<T: Numeric>(
  input: Tensor<T>,
  minRange: Tensor<Float>,
  maxRange: Tensor<Float>,
  mode: Mode = .minCombined
) -> Tensor<Float> {
  return #tfop("Dequantize",
    input,
    minRange,
    maxRange,
    T: T.self,
    mode: mode.rawValue)
}

// Deserialize `SparseTensor` objects.
//
// The input `serialized_sparse` must have the shape `[?, ?, ..., ?, 3]` where
// the last dimension stores serialized `SparseTensor` objects and the other N
// dimensions (N >= 0) correspond to a batch. The ranks of the original
// `SparseTensor` objects must all match. When the final `SparseTensor` is
// created, its rank is the rank of the incoming `SparseTensor` objects plus N;
// the sparse tensors have been concatenated along new dimensions, one for each
// batch.
//
// The output `SparseTensor` object's shape values for the original dimensions
// are the max across the input `SparseTensor` objects' shape values for the
// corresponding dimensions. The new dimensions match the size of the batch.
//
// The input `SparseTensor` objects' indices are assumed ordered in
// standard lexicographic order.  If this is not the case, after this
// step run `SparseReorder` to restore index ordering.
//
// For example, if the serialized input is a `[2 x 3]` matrix representing two
// original `SparseTensor` objects:
//
//     index = [ 0]
//             [10]
//             [20]
//     values = [1, 2, 3]
//     shape = [50]
//
// and
//
//     index = [ 2]
//             [10]
//     values = [4, 5]
//     shape = [30]
//
// then the final deserialized `SparseTensor` will be:
//
//     index = [0  0]
//             [0 10]
//             [0 20]
//             [1  2]
//             [1 10]
//     values = [1, 2, 3, 4, 5]
//     shape = [2 50]
//
// - Parameter serialized_sparse: The serialized `SparseTensor` objects. The last dimension
//   must have 3 columns.
//
// - Attr dtype: The `dtype` of the serialized `SparseTensor` objects.
@_inlineable @inline(__always)
public static func deserializeSparse<Dtype: Numeric, Tserialized: Numeric>(
  serializedSparse: Tensor<Tserialized>
) -> (Tensor<Int64>, Tensor<Dtype>, Tensor<Int64>) {
  return #tfop("DeserializeSparse",
    serializedSparse,
    Dtype: Dtype.self,
    Tserialized: Tserialized.self)
}

// Destroys the temporary variable and returns its final value.
//
// Sets output to the value of the Tensor pointed to by 'ref', then destroys
// the temporary variable called 'var_name'.
// All other uses of 'ref' *must* have executed before this op.
// This is typically achieved by chaining the ref through each assign op, or by
// using control dependencies.
//
// Outputs the final value of the tensor pointed to by 'ref'.
//
// - Parameter ref: A reference to the temporary variable tensor.
//
// - Attr var_name: Name of the temporary variable, usually the name of the matching
//   'TemporaryVariable' op.
@_inlineable @inline(__always)
public static func destroyTemporaryVariable<T: Numeric>(
  ref: Tensor<T>,
  varName: String
) -> Tensor<T> {
  return #tfop("DestroyTemporaryVariable",
    ref,
    T: T.self,
    var_name: varName)
}

// Returns a diagonal tensor with a given diagonal values.
//
// Given a `diagonal`, this operation returns a tensor with the `diagonal` and
// everything else padded with zeros. The diagonal is computed as follows:
//
// Assume `diagonal` has dimensions [D1,..., Dk], then the output is a tensor of
// rank 2k with dimensions [D1,..., Dk, D1,..., Dk] where:
//
// `output[i1,..., ik, i1,..., ik] = diagonal[i1, ..., ik]` and 0 everywhere else.
//
// For example:
//
// ```
// # 'diagonal' is [1, 2, 3, 4]
// tf.diag(diagonal) ==> [[1, 0, 0, 0]
//                        [0, 2, 0, 0]
//                        [0, 0, 3, 0]
//                        [0, 0, 0, 4]]
// ```
//
// - Parameter diagonal: Rank k tensor where k is at most 1.
@_inlineable @inline(__always)
public static func diag<T: Numeric>(
  diagonal: Tensor<T>
) -> Tensor<T> {
  return #tfop("Diag",
    diagonal,
    T: T.self)
}

// Returns the diagonal part of the tensor.
//
// This operation returns a tensor with the `diagonal` part
// of the `input`. The `diagonal` part is computed as follows:
//
// Assume `input` has dimensions `[D1,..., Dk, D1,..., Dk]`, then the output is a
// tensor of rank `k` with dimensions `[D1,..., Dk]` where:
//
// `diagonal[i1,..., ik] = input[i1, ..., ik, i1,..., ik]`.
//
// For example:
//
// ```
// # 'input' is [[1, 0, 0, 0]
//               [0, 2, 0, 0]
//               [0, 0, 3, 0]
//               [0, 0, 0, 4]]
//
// tf.diag_part(input) ==> [1, 2, 3, 4]
// ```
//
// - Parameter input: Rank k tensor where k is even and not zero.
//
// - Output diagonal: The extracted diagonal.
@_inlineable @inline(__always)
public static func diagPart<T: Numeric>(
  input: Tensor<T>
) -> Tensor<T> {
  return #tfop("DiagPart",
    input,
    T: T.self)
}

// Computes Psi, the derivative of Lgamma (the log of the absolute value of
//
// `Gamma(x)`), element-wise.
@_inlineable @inline(__always)
public static func digamma<T: BinaryFloatingPoint>(
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Digamma",
    x,
    T: T.self)
}

// Computes the grayscale dilation of 4-D `input` and 3-D `filter` tensors.
//
// The `input` tensor has shape `[batch, in_height, in_width, depth]` and the
// `filter` tensor has shape `[filter_height, filter_width, depth]`, i.e., each
// input channel is processed independently of the others with its own structuring
// function. The `output` tensor has shape
// `[batch, out_height, out_width, depth]`. The spatial dimensions of the output
// tensor depend on the `padding` algorithm. We currently only support the default
// "NHWC" `data_format`.
//
// In detail, the grayscale morphological 2-D dilation is the max-sum correlation
// (for consistency with `conv2d`, we use unmirrored filters):
//
//     output[b, y, x, c] =
//        max_{dy, dx} input[b,
//                           strides[1] * y + rates[1] * dy,
//                           strides[2] * x + rates[2] * dx,
//                           c] +
//                     filter[dy, dx, c]
//
// Max-pooling is a special case when the filter has size equal to the pooling
// kernel size and contains all zeros.
//
// Note on duality: The dilation of `input` by the `filter` is equal to the
// negation of the erosion of `-input` by the reflected `filter`.
//
// - Parameters:
//   - input: 4-D with shape `[batch, in_height, in_width, depth]`.
//   - filter: 3-D with shape `[filter_height, filter_width, depth]`.
//
// - Attrs:
//   - strides: The stride of the sliding window for each dimension of the input
//     tensor. Must be: `[1, stride_height, stride_width, 1]`.
//   - rates: The input stride for atrous morphological dilation. Must be:
//     `[1, rate_height, rate_width, 1]`.
//   - padding: The type of padding algorithm to use.
//
// - Output output: 4-D with shape `[batch, out_height, out_width, depth]`.
@_inlineable @inline(__always)
public static func dilation2D<T: Numeric>(
  input: Tensor<T>,
  filter: Tensor<T>,
  strides: [Int],
  rates: [Int],
  padding: Padding
) -> Tensor<T> {
  return #tfop("Dilation2D",
    input,
    filter,
    T: T.self,
    strides: strides,
    rates: rates,
    padding: padding.rawValue)
}

// Computes the gradient of morphological 2-D dilation with respect to the filter.
//
// - Parameters:
//   - input: 4-D with shape `[batch, in_height, in_width, depth]`.
//   - filter: 3-D with shape `[filter_height, filter_width, depth]`.
//   - out_backprop: 4-D with shape `[batch, out_height, out_width, depth]`.
//
// - Attrs:
//   - strides: 1-D of length 4. The stride of the sliding window for each dimension of
//     the input tensor. Must be: `[1, stride_height, stride_width, 1]`.
//   - rates: 1-D of length 4. The input stride for atrous morphological dilation.
//     Must be: `[1, rate_height, rate_width, 1]`.
//   - padding: The type of padding algorithm to use.
//
// - Output filter_backprop: 3-D with shape `[filter_height, filter_width, depth]`.
@_inlineable @inline(__always)
public static func dilation2DBackpropFilter<T: Numeric>(
  input: Tensor<T>,
  filter: Tensor<T>,
  outBackprop: Tensor<T>,
  strides: [Int],
  rates: [Int],
  padding: Padding
) -> Tensor<T> {
  return #tfop("Dilation2DBackpropFilter",
    input,
    filter,
    outBackprop,
    T: T.self,
    strides: strides,
    rates: rates,
    padding: padding.rawValue)
}

// Computes the gradient of morphological 2-D dilation with respect to the input.
//
// - Parameters:
//   - input: 4-D with shape `[batch, in_height, in_width, depth]`.
//   - filter: 3-D with shape `[filter_height, filter_width, depth]`.
//   - out_backprop: 4-D with shape `[batch, out_height, out_width, depth]`.
//
// - Attrs:
//   - strides: 1-D of length 4. The stride of the sliding window for each dimension of
//     the input tensor. Must be: `[1, stride_height, stride_width, 1]`.
//   - rates: 1-D of length 4. The input stride for atrous morphological dilation.
//     Must be: `[1, rate_height, rate_width, 1]`.
//   - padding: The type of padding algorithm to use.
//
// - Output in_backprop: 4-D with shape `[batch, in_height, in_width, depth]`.
@_inlineable @inline(__always)
public static func dilation2DBackpropInput<T: Numeric>(
  input: Tensor<T>,
  filter: Tensor<T>,
  outBackprop: Tensor<T>,
  strides: [Int],
  rates: [Int],
  padding: Padding
) -> Tensor<T> {
  return #tfop("Dilation2DBackpropInput",
    input,
    filter,
    outBackprop,
    T: T.self,
    strides: strides,
    rates: rates,
    padding: padding.rawValue)
}

// Returns x / y element-wise.
//
// *NOTE*: `Div` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@_inlineable @inline(__always)
public static func div<T: Numeric>(
  x: Tensor<T>,
  y: Tensor<T>
) -> Tensor<T> {
  return #tfop("Div",
    x,
    y,
    T: T.self)
}

// Draw bounding boxes on a batch of images.
//
// Outputs a copy of `images` but draws on top of the pixels zero or more bounding
// boxes specified by the locations in `boxes`. The coordinates of the each
// bounding box in `boxes` are encoded as `[y_min, x_min, y_max, x_max]`. The
// bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
// height of the underlying image.
//
// For example, if an image is 100 x 200 pixels (height x width) and the bounding
// box is `[0.1, 0.2, 0.5, 0.9]`, the upper-left and bottom-right coordinates of
// the bounding box will be `(40, 10)` to `(100, 50)` (in (x,y) coordinates).
//
// Parts of the bounding box may fall outside the image.
//
// - Parameters:
//   - images: 4-D with shape `[batch, height, width, depth]`. A batch of images.
//   - boxes: 3-D with shape `[batch, num_bounding_boxes, 4]` containing bounding
//     boxes.
//
// - Output output: 4-D with the same shape as `images`. The batch of input images with
//   bounding boxes drawn on the images.
@_inlineable @inline(__always)
public static func drawBoundingBoxes<T: BinaryFloatingPoint>(
  images: Tensor<T>,
  boxes: Tensor<Float>
) -> Tensor<T> {
  return #tfop("DrawBoundingBoxes",
    images,
    boxes,
    T: T.self)
}

// Partitions `data` into `num_partitions` tensors using indices from `partitions`.
//
// For each index tuple `js` of size `partitions.ndim`, the slice `data[js, ...]`
// becomes part of `outputs[partitions[js]]`.  The slices with `partitions[js] = i`
// are placed in `outputs[i]` in lexicographic order of `js`, and the first
// dimension of `outputs[i]` is the number of entries in `partitions` equal to `i`.
// In detail,
//
// ```python
//     outputs[i].shape = [sum(partitions == i)] + data.shape[partitions.ndim:]
//
//     outputs[i] = pack([data[js, ...] for js if partitions[js] == i])
// ```
//
// `data.shape` must start with `partitions.shape`.
//
// For example:
//
// ```python
//     # Scalar partitions.
//     partitions = 1
//     num_partitions = 2
//     data = [10, 20]
//     outputs[0] = []  # Empty with shape [0, 2]
//     outputs[1] = [[10, 20]]
//
//     # Vector partitions.
//     partitions = [0, 0, 1, 1, 0]
//     num_partitions = 2
//     data = [10, 20, 30, 40, 50]
//     outputs[0] = [10, 20, 50]
//     outputs[1] = [30, 40]
// ```
//
// See `dynamic_stitch` for an example on how to merge partitions back.
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="https://www.tensorflow.org/images/DynamicPartition.png" alt>
// </div>
//
// - Parameter partitions: Any shape.  Indices in the range `[0, num_partitions)`.
//
// - Attr num_partitions: The number of partitions to output.
@_inlineable @inline(__always)
public static func dynamicPartition<T: Numeric>(
  data: Tensor<T>,
  partitions: Tensor<Int32>,
  numPartitions: Int
) -> [Tensor<T>] {
  return #tfop("DynamicPartition",
    data,
    partitions,
    T: T.self,
    num_partitions: numPartitions)
}

// Interleave the values from the `data` tensors into a single tensor.
//
// Builds a merged tensor such that
//
// ```python
//     merged[indices[m][i, ..., j], ...] = data[m][i, ..., j, ...]
// ```
//
// For example, if each `indices[m]` is scalar or vector, we have
//
// ```python
//     # Scalar indices:
//     merged[indices[m], ...] = data[m][...]
//
//     # Vector indices:
//     merged[indices[m][i], ...] = data[m][i, ...]
// ```
//
// Each `data[i].shape` must start with the corresponding `indices[i].shape`,
// and the rest of `data[i].shape` must be constant w.r.t. `i`.  That is, we
// must have `data[i].shape = indices[i].shape + constant`.  In terms of this
// `constant`, the output shape is
//
//     merged.shape = [max(indices)] + constant
//
// Values are merged in order, so if an index appears in both `indices[m][i]` and
// `indices[n][j]` for `(m,i) < (n,j)` the slice `data[n][j]` will appear in the
// merged result. If you do not need this guarantee, ParallelDynamicStitch might
// perform better on some devices.
//
// For example:
//
// ```python
//     indices[0] = 6
//     indices[1] = [4, 1]
//     indices[2] = [[5, 2], [0, 3]]
//     data[0] = [61, 62]
//     data[1] = [[41, 42], [11, 12]]
//     data[2] = [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]
//     merged = [[1, 2], [11, 12], [21, 22], [31, 32], [41, 42],
//               [51, 52], [61, 62]]
// ```
//
// This method can be used to merge partitions created by `dynamic_partition`
// as illustrated on the following example:
//
// ```python
//     # Apply function (increments x_i) on elements for which a certain condition
//     # apply (x_i != -1 in this example).
//     x=tf.constant([0.1, -1., 5.2, 4.3, -1., 7.4])
//     condition_mask=tf.not_equal(x,tf.constant(-1.))
//     partitioned_data = tf.dynamic_partition(
//         x, tf.cast(condition_mask, tf.int32) , 2)
//     partitioned_data[1] = partitioned_data[1] + 1.0
//     condition_indices = tf.dynamic_partition(
//         tf.range(tf.shape(x)[0]), tf.cast(condition_mask, tf.int32) , 2)
//     x = tf.dynamic_stitch(condition_indices, partitioned_data)
//     # Here x=[1.1, -1., 6.2, 5.3, -1, 8.4], the -1. values remain
//     # unchanged.
// ```
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="https://www.tensorflow.org/images/DynamicStitch.png" alt>
// </div>
@_inlineable @inline(__always)
public static func dynamicStitch<T: Numeric>(
  indices: [Tensor<Int32>],
  data: [Tensor<T>]
) -> Tensor<T> {
  return #tfop("DynamicStitch",
    indices,
    data,
    T: T.self)
}

// Eagerly executes a python function to compute func(input)->output. The
//
// semantics of the input, output, and attributes are the same as those for
// PyFunc.
@_inlineable @inline(__always)
public static func eagerPyFunc<Tin: Numeric, Tout: Numeric>(
  input: [Tensor<Tin>],
  token: String
) -> [Tensor<Tout>] {
  return #tfop("EagerPyFunc",
    input,
    token: token)
}

// Computes the (possibly normalized) Levenshtein Edit Distance.
//
// The inputs are variable-length sequences provided by SparseTensors
//   (hypothesis_indices, hypothesis_values, hypothesis_shape)
// and
//   (truth_indices, truth_values, truth_shape).
//
// The inputs are:
//
// - Parameters:
//   - hypothesis_indices: The indices of the hypothesis list SparseTensor.
//     This is an N x R int64 matrix.
//   - hypothesis_values: The values of the hypothesis list SparseTensor.
//     This is an N-length vector.
//   - hypothesis_shape: The shape of the hypothesis list SparseTensor.
//     This is an R-length vector.
//   - truth_indices: The indices of the truth list SparseTensor.
//     This is an M x R int64 matrix.
//   - truth_values: The values of the truth list SparseTensor.
//     This is an M-length vector.
//   - truth_shape: truth indices, vector.
//
// - Attr normalize: boolean (if true, edit distances are normalized by length of truth).
//
//   The output is:
//
// - Output output: A dense float tensor with rank R - 1.
//
//   For the example input:
//
//       // hypothesis represents a 2x1 matrix with variable-length values:
//       //   (0,0) = ["a"]
//       //   (1,0) = ["b"]
//       hypothesis_indices = [[0, 0, 0],
//                             [1, 0, 0]]
//       hypothesis_values = ["a", "b"]
//       hypothesis_shape = [2, 1, 1]
//
//       // truth represents a 2x2 matrix with variable-length values:
//       //   (0,0) = []
//       //   (0,1) = ["a"]
//       //   (1,0) = ["b", "c"]
//       //   (1,1) = ["a"]
//       truth_indices = [[0, 1, 0],
//                        [1, 0, 0],
//                        [1, 0, 1],
//                        [1, 1, 0]]
//       truth_values = ["a", "b", "c", "a"]
//       truth_shape = [2, 2, 2]
//       normalize = true
//
//   The output will be:
//
//       // output is a 2x2 matrix with edit distances normalized by truth lengths.
//       output = [[inf, 1.0],  // (0,0): no truth, (0,1): no hypothesis
//                 [0.5, 1.0]]  // (1,0): addition, (1,1): no hypothesis
@_inlineable @inline(__always)
public static func editDistance<T: Numeric>(
  hypothesisIndices: Tensor<Int64>,
  hypothesisValues: Tensor<T>,
  hypothesisShape: Tensor<Int64>,
  truthIndices: Tensor<Int64>,
  truthValues: Tensor<T>,
  truthShape: Tensor<Int64>,
  normalize: Bool = true
) -> Tensor<Float> {
  return #tfop("EditDistance",
    hypothesisIndices,
    hypothesisValues,
    hypothesisShape,
    truthIndices,
    truthValues,
    truthShape,
    T: T.self,
    normalize: normalize)
}

// Computes exponential linear: `exp(features) - 1` if < 0, `features` otherwise.
//
// See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
// ](http://arxiv.org/abs/1511.07289)
@_inlineable @inline(__always)
public static func elu<T: BinaryFloatingPoint>(
  features: Tensor<T>
) -> Tensor<T> {
  return #tfop("Elu",
    features,
    T: T.self)
}

// Computes gradients for the exponential linear (Elu) operation.
//
// - Parameters:
//   - gradients: The backpropagated gradients to the corresponding Elu operation.
//   - outputs: The outputs of the corresponding Elu operation.
//
// - Output backprops: The gradients: `gradients * (outputs + 1)` if outputs < 0,
//   `gradients` otherwise.
@_inlineable @inline(__always)
public static func eluGrad<T: BinaryFloatingPoint>(
  gradients: Tensor<T>,
  outputs: Tensor<T>
) -> Tensor<T> {
  return #tfop("EluGrad",
    gradients,
    outputs,
    T: T.self)
}

// Creates a tensor with the given shape.
//
// This operation creates a tensor of `shape` and `dtype`.
//
// - Parameter shape: 1-D. Represents the shape of the output tensor.
//
// - Attr init: If True, initialize the returned tensor with the default value of dtype.  Otherwise, the implementation is free not to initializethe tensor's content.
//
// - Output output: A `Tensor` of type `T`.
@_inlineable @inline(__always)
public static func empty<Dtype: Numeric>(
  shape: Tensor<Int32>,
  init_: Bool = false
) -> Tensor<Dtype> {
  return #tfop("Empty",
    shape,
    Dtype: Dtype.self,
    init: init_)
}

// Creates or finds a child frame, and makes `data` available to the child frame.
//
// This op is used together with `Exit` to create loops in the graph.
// The unique `frame_name` is used by the `Executor` to identify frames. If
// `is_constant` is true, `output` is a constant in the child frame; otherwise
// it may be changed in the child frame. At most `parallel_iterations` iterations
// are run in parallel in the child frame.
//
// - Parameter data: The tensor to be made available to the child frame.
//
// - Attrs:
//   - frame_name: The name of the child frame.
//   - is_constant: If true, the output is constant within the child frame.
//   - parallel_iterations: The number of iterations allowed to run in parallel.
//
// - Output output: The same tensor as `data`.
@_inlineable @inline(__always)
public static func enter<T: Numeric>(
  data: Tensor<T>,
  frameName: String,
  isConstant: Bool = false,
  parallelIterations: Int = 10
) -> Tensor<T> {
  return #tfop("Enter",
    data,
    T: T.self,
    frame_name: frameName,
    is_constant: isConstant,
    parallel_iterations: parallelIterations)
}

// Returns the truth value of (x == y) element-wise.
//
// *NOTE*: `Equal` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@_inlineable @inline(__always)
public static func equal<T: Numeric>(
  x: Tensor<T>,
  y: Tensor<T>
) -> Tensor<Bool> {
  return #tfop("Equal",
    x,
    y,
    T: T.self)
}

// Computes the Gauss error function of `x` element-wise.
@_inlineable @inline(__always)
public static func erf<T: BinaryFloatingPoint>(
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Erf",
    x,
    T: T.self)
}

// Computes the complementary error function of `x` element-wise.
@_inlineable @inline(__always)
public static func erfc<T: BinaryFloatingPoint>(
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Erfc",
    x,
    T: T.self)
}

// Exits the current frame to its parent frame.
//
// Exit makes its input `data` available to the parent frame.
//
// - Parameter data: The tensor to be made available to the parent frame.
//
// - Output output: The same tensor as `data`.
@_inlineable @inline(__always)
public static func exit<T: Numeric>(
  data: Tensor<T>
) -> Tensor<T> {
  return #tfop("Exit",
    data,
    T: T.self)
}

// Computes exponential of x element-wise.  \\(y = e^x\\).
@_inlineable @inline(__always)
public static func exp<T: BinaryFloatingPoint>(
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Exp",
    x,
    T: T.self)
}

// Inserts a dimension of 1 into a tensor's shape.
//
// Given a tensor `input`, this operation inserts a dimension of 1 at the
// dimension index `axis` of `input`'s shape. The dimension index `axis` starts at
// zero; if you specify a negative number for `axis` it is counted backward from
// the end.
//
// This operation is useful if you want to add a batch dimension to a single
// element. For example, if you have a single image of shape `[height, width,
// channels]`, you can make it a batch of 1 image with `expand_dims(image, 0)`,
// which will make the shape `[1, height, width, channels]`.
//
// Other examples:
//
// ```
// # 't' is a tensor of shape [2]
// shape(expand_dims(t, 0)) ==> [1, 2]
// shape(expand_dims(t, 1)) ==> [2, 1]
// shape(expand_dims(t, -1)) ==> [2, 1]
//
// # 't2' is a tensor of shape [2, 3, 5]
// shape(expand_dims(t2, 0)) ==> [1, 2, 3, 5]
// shape(expand_dims(t2, 2)) ==> [2, 3, 1, 5]
// shape(expand_dims(t2, 3)) ==> [2, 3, 5, 1]
// ```
//
// This operation requires that:
//
// `-1-input.dims() <= dim <= input.dims()`
//
// This operation is related to `squeeze()`, which removes dimensions of
// size 1.
//
// - Parameter dim: 0-D (scalar). Specifies the dimension index at which to
//   expand the shape of `input`. Must be in the range
//   `[-rank(input) - 1, rank(input)]`.
//
// - Output output: Contains the same data as `input`, but its shape has an additional
//   dimension of size 1 added.
@_inlineable @inline(__always)
public static func expandDims<T: Numeric, Tdim: BinaryInteger>(
  input: Tensor<T>,
  dim: Tensor<Tdim>
) -> Tensor<T> {
  return #tfop("ExpandDims",
    input,
    dim,
    T: T.self,
    Tdim: Tdim.self)
}

// Computes exponential of x - 1 element-wise.
//
// I.e., \\(y = (\exp x) - 1\\).
@_inlineable @inline(__always)
public static func expm1<T: BinaryFloatingPoint>(
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Expm1",
    x,
    T: T.self)
}

// Extracts a glimpse from the input tensor.
//
// Returns a set of windows called glimpses extracted at location
// `offsets` from the input tensor. If the windows only partially
// overlaps the inputs, the non overlapping areas will be filled with
// random noise.
//
// The result is a 4-D tensor of shape `[batch_size, glimpse_height,
// glimpse_width, channels]`. The channels and batch dimensions are the
// same as that of the input tensor. The height and width of the output
// windows are specified in the `size` parameter.
//
// The argument `normalized` and `centered` controls how the windows are built:
//
// * If the coordinates are normalized but not centered, 0.0 and 1.0
//   correspond to the minimum and maximum of each height and width
//   dimension.
// * If the coordinates are both normalized and centered, they range from
//   -1.0 to 1.0. The coordinates (-1.0, -1.0) correspond to the upper
//   left corner, the lower right corner is located at (1.0, 1.0) and the
//   center is at (0, 0).
// * If the coordinates are not normalized they are interpreted as
//   numbers of pixels.
//
// - Parameters:
//   - input: A 4-D float tensor of shape `[batch_size, height, width, channels]`.
//   - size: A 1-D tensor of 2 elements containing the size of the glimpses
//     to extract.  The glimpse height must be specified first, following
//     by the glimpse width.
//   - offsets: A 2-D integer tensor of shape `[batch_size, 2]` containing
//     the y, x locations of the center of each window.
//
// - Attrs:
//   - centered: indicates if the offset coordinates are centered relative to
//     the image, in which case the (0, 0) offset is relative to the center
//     of the input images. If false, the (0,0) offset corresponds to the
//     upper left corner of the input images.
//   - normalized: indicates if the offset coordinates are normalized.
//   - uniform_noise: indicates if the noise should be generated using a
//     uniform distribution or a Gaussian distribution.
//
// - Output glimpse: A tensor representing the glimpses `[batch_size,
//   glimpse_height, glimpse_width, channels]`.
@_inlineable @inline(__always)
public static func extractGlimpse(
  input: Tensor<Float>,
  size: Tensor<Int32>,
  offsets: Tensor<Float>,
  centered: Bool = true,
  normalized: Bool = true,
  uniformNoise: Bool = true
) -> Tensor<Float> {
  return #tfop("ExtractGlimpse",
    input,
    size,
    offsets,
    centered: centered,
    normalized: normalized,
    uniform_noise: uniformNoise)
}

// Extract `patches` from `images` and put them in the "depth" output dimension.
//
// - Parameter images: 4-D Tensor with shape `[batch, in_rows, in_cols, depth]`.
//
// - Attrs:
//   - ksizes: The size of the sliding window for each dimension of `images`.
//   - strides: 1-D of length 4. How far the centers of two consecutive patches are in
//     the images. Must be: `[1, stride_rows, stride_cols, 1]`.
//   - rates: 1-D of length 4. Must be: `[1, rate_rows, rate_cols, 1]`. This is the
//     input stride, specifying how far two consecutive patch samples are in the
//     input. Equivalent to extracting patches with
//     `patch_sizes_eff = patch_sizes + (patch_sizes - 1) * (rates - 1)`, followed by
//     subsampling them spatially by a factor of `rates`. This is equivalent to
//     `rate` in dilated (a.k.a. Atrous) convolutions.
//   - padding: The type of padding algorithm to use.
//
//     We specify the size-related attributes as:
//
//     ```python
//           ksizes = [1, ksize_rows, ksize_cols, 1]
//           strides = [1, strides_rows, strides_cols, 1]
//           rates = [1, rates_rows, rates_cols, 1]
//     ```
//
// - Output patches: 4-D Tensor with shape `[batch, out_rows, out_cols, ksize_rows *
//   ksize_cols * depth]` containing image patches with size
//   `ksize_rows x ksize_cols x depth` vectorized in the "depth" dimension. Note
//   `out_rows` and `out_cols` are the dimensions of the output patches.
@_inlineable @inline(__always)
public static func extractImagePatches<T: Numeric>(
  images: Tensor<T>,
  ksizes: [Int],
  strides: [Int],
  rates: [Int],
  padding: Padding
) -> Tensor<T> {
  return #tfop("ExtractImagePatches",
    images,
    T: T.self,
    ksizes: ksizes,
    strides: strides,
    rates: rates,
    padding: padding.rawValue)
}

// Fake-quantize the 'inputs' tensor, type float to 'outputs' tensor of same type.
//
// Attributes `[min; max]` define the clamping range for the `inputs` data.
// `inputs` values are quantized into the quantization range (`[0; 2^num_bits - 1]`
// when `narrow_range` is false and `[1; 2^num_bits - 1]` when it is true) and
// then de-quantized and output as floats in `[min; max]` interval.
// `num_bits` is the bitwidth of the quantization; between 2 and 16, inclusive.
//
// Quantization is called fake since the output is still in floating point.
@_inlineable @inline(__always)
public static func fakeQuantWithMinMaxArgs(
  inputs: Tensor<Float>,
  min: Double = -6,
  max: Double = 6,
  numBits: Int = 8,
  narrowRange: Bool = false
) -> Tensor<Float> {
  return #tfop("FakeQuantWithMinMaxArgs",
    inputs,
    min: min,
    max: max,
    num_bits: numBits,
    narrow_range: narrowRange)
}

// Compute gradients for a FakeQuantWithMinMaxArgs operation.
//
// - Parameters:
//   - gradients: Backpropagated gradients above the FakeQuantWithMinMaxArgs operation.
//   - inputs: Values passed as inputs to the FakeQuantWithMinMaxArgs operation.
//
// - Output backprops: Backpropagated gradients below the FakeQuantWithMinMaxArgs operation:
//   `gradients * (inputs >= min && inputs <= max)`.
@_inlineable @inline(__always)
public static func fakeQuantWithMinMaxArgsGradient(
  gradients: Tensor<Float>,
  inputs: Tensor<Float>,
  min: Double = -6,
  max: Double = 6,
  numBits: Int = 8,
  narrowRange: Bool = false
) -> Tensor<Float> {
  return #tfop("FakeQuantWithMinMaxArgsGradient",
    gradients,
    inputs,
    min: min,
    max: max,
    num_bits: numBits,
    narrow_range: narrowRange)
}

// Fake-quantize the 'inputs' tensor of type float via global float scalars `min`
//
// and `max` to 'outputs' tensor of same shape as `inputs`.
//
// `[min; max]` define the clamping range for the `inputs` data.
// `inputs` values are quantized into the quantization range (`[0; 2^num_bits - 1]`
// when `narrow_range` is false and `[1; 2^num_bits - 1]` when it is true) and
// then de-quantized and output as floats in `[min; max]` interval.
// `num_bits` is the bitwidth of the quantization; between 2 and 16, inclusive.
//
// This operation has a gradient and thus allows for training `min` and `max`
// values.
@_inlineable @inline(__always)
public static func fakeQuantWithMinMaxVars(
  inputs: Tensor<Float>,
  min: Tensor<Float>,
  max: Tensor<Float>,
  numBits: Int = 8,
  narrowRange: Bool = false
) -> Tensor<Float> {
  return #tfop("FakeQuantWithMinMaxVars",
    inputs,
    min,
    max,
    num_bits: numBits,
    narrow_range: narrowRange)
}

// Compute gradients for a FakeQuantWithMinMaxVars operation.
//
// - Parameters:
//   - gradients: Backpropagated gradients above the FakeQuantWithMinMaxVars operation.
//   - inputs: Values passed as inputs to the FakeQuantWithMinMaxVars operation.
//     min, max: Quantization interval, scalar floats.
//
// - Attrs:
//   - num_bits: The bitwidth of the quantization; between 2 and 8, inclusive.
//   - narrow_range: Whether to quantize into 2^num_bits - 1 distinct values.
//
// - Outputs:
//   - backprops_wrt_input: Backpropagated gradients w.r.t. inputs:
//     `gradients * (inputs >= min && inputs <= max)`.
//   - backprop_wrt_min: Backpropagated gradients w.r.t. min parameter:
//     `sum(gradients * (inputs < min))`.
//   - backprop_wrt_max: Backpropagated gradients w.r.t. max parameter:
//     `sum(gradients * (inputs > max))`.
@_inlineable @inline(__always)
public static func fakeQuantWithMinMaxVarsGradient(
  gradients: Tensor<Float>,
  inputs: Tensor<Float>,
  min: Tensor<Float>,
  max: Tensor<Float>,
  numBits: Int = 8,
  narrowRange: Bool = false
) -> (Tensor<Float>, Tensor<Float>, Tensor<Float>) {
  return #tfop("FakeQuantWithMinMaxVarsGradient",
    gradients,
    inputs,
    min,
    max,
    num_bits: numBits,
    narrow_range: narrowRange)
}

// Fake-quantize the 'inputs' tensor of type float and one of the shapes: `[d]`,
//
// `[b, d]` `[b, h, w, d]` via per-channel floats `min` and `max` of shape `[d]`
// to 'outputs' tensor of same shape as `inputs`.
//
// `[min; max]` define the clamping range for the `inputs` data.
// `inputs` values are quantized into the quantization range (`[0; 2^num_bits - 1]`
// when `narrow_range` is false and `[1; 2^num_bits - 1]` when it is true) and
// then de-quantized and output as floats in `[min; max]` interval.
// `num_bits` is the bitwidth of the quantization; between 2 and 16, inclusive.
//
// This operation has a gradient and thus allows for training `min` and `max`
// values.
@_inlineable @inline(__always)
public static func fakeQuantWithMinMaxVarsPerChannel(
  inputs: Tensor<Float>,
  min: Tensor<Float>,
  max: Tensor<Float>,
  numBits: Int = 8,
  narrowRange: Bool = false
) -> Tensor<Float> {
  return #tfop("FakeQuantWithMinMaxVarsPerChannel",
    inputs,
    min,
    max,
    num_bits: numBits,
    narrow_range: narrowRange)
}

// Compute gradients for a FakeQuantWithMinMaxVarsPerChannel operation.
//
// - Parameters:
//   - gradients: Backpropagated gradients above the FakeQuantWithMinMaxVars operation,
//     shape one of: `[d]`, `[b, d]`,  `[b, h, w, d]`.
//   - inputs: Values passed as inputs to the FakeQuantWithMinMaxVars operation, shape
//       same as `gradients`.
//     min, max: Quantization interval, floats of shape `[d]`.
//
// - Attrs:
//   - num_bits: The bitwidth of the quantization; between 2 and 16, inclusive.
//   - narrow_range: Whether to quantize into 2^num_bits - 1 distinct values.
//
// - Outputs:
//   - backprops_wrt_input: Backpropagated gradients w.r.t. inputs, shape same as
//     `inputs`:
//       `gradients * (inputs >= min && inputs <= max)`.
//   - backprop_wrt_min: Backpropagated gradients w.r.t. min parameter, shape `[d]`:
//     `sum_per_d(gradients * (inputs < min))`.
//   - backprop_wrt_max: Backpropagated gradients w.r.t. max parameter, shape `[d]`:
//     `sum_per_d(gradients * (inputs > max))`.
@_inlineable @inline(__always)
public static func fakeQuantWithMinMaxVarsPerChannelGradient(
  gradients: Tensor<Float>,
  inputs: Tensor<Float>,
  min: Tensor<Float>,
  max: Tensor<Float>,
  numBits: Int = 8,
  narrowRange: Bool = false
) -> (Tensor<Float>, Tensor<Float>, Tensor<Float>) {
  return #tfop("FakeQuantWithMinMaxVarsPerChannelGradient",
    gradients,
    inputs,
    min,
    max,
    num_bits: numBits,
    narrow_range: narrowRange)
}

// Creates a tensor filled with a scalar value.
//
// This operation creates a tensor of shape `dims` and fills it with `value`.
//
// For example:
//
// ```
// # Output tensor has shape [2, 3].
// fill([2, 3], 9) ==> [[9, 9, 9]
//                      [9, 9, 9]]
// ```
//
// - Parameters:
//   - dims: 1-D. Represents the shape of the output tensor.
//   - value: 0-D (scalar). Value to fill the returned tensor.
//
//     @compatibility(numpy)
//     Equivalent to np.full
//     @end_compatibility
@_inlineable @inline(__always)
public static func fill<T: Numeric, Index_type: BinaryInteger>(
  dims: Tensor<Index_type>,
  value: Tensor<T>
) -> Tensor<T> {
  return #tfop("Fill",
    dims,
    value,
    T: T.self,
    Index_type: Index_type.self)
}

@_inlineable @inline(__always)
public static func fiveFloatOutputs(
) -> (Tensor<Float>, Tensor<Float>, Tensor<Float>, Tensor<Float>, Tensor<Float>) {
  return #tfop("FiveFloatOutputs")
}

// Generates labels for candidate sampling with a learned unigram distribution.
//
// A unigram sampler could use a fixed unigram distribution read from a
// file or passed in as an in-memory array instead of building up the distribution
// from data on the fly. There is also an option to skew the distribution by
// applying a distortion power to the weights.
//
// The vocabulary file should be in CSV-like format, with the last field
// being the weight associated with the word.
//
// For each batch, this op picks a single set of sampled candidate labels.
//
// The advantages of sampling candidates per-batch are simplicity and the
// possibility of efficient dense matrix multiplication. The disadvantage is that
// the sampled candidates must be chosen independently of the context and of the
// true labels.
//
// - Parameter true_classes: A batch_size * num_true matrix, in which each row contains the
//   IDs of the num_true target_classes in the corresponding original label.
//
// - Attrs:
//   - num_true: Number of true labels per context.
//   - num_sampled: Number of candidates to randomly sample.
//   - unique: If unique is true, we sample with rejection, so that all sampled
//     candidates in a batch are unique. This requires some approximation to
//     estimate the post-rejection sampling probabilities.
//   - range_max: The sampler will sample integers from the interval [0, range_max).
//   - vocab_file: Each valid line in this file (which should have a CSV-like format)
//     corresponds to a valid word ID. IDs are in sequential order, starting from
//     num_reserved_ids. The last entry in each line is expected to be a value
//     corresponding to the count or relative probability. Exactly one of vocab_file
//     and unigrams needs to be passed to this op.
//   - distortion: The distortion is used to skew the unigram probability distribution.
//     Each weight is first raised to the distortion's power before adding to the
//     internal unigram distribution. As a result, distortion = 1.0 gives regular
//     unigram sampling (as defined by the vocab file), and distortion = 0.0 gives
//     a uniform distribution.
//   - num_reserved_ids: Optionally some reserved IDs can be added in the range [0,
//     ..., num_reserved_ids) by the users. One use case is that a special unknown
//     word token is used as ID 0. These IDs will have a sampling probability of 0.
//   - num_shards: A sampler can be used to sample from a subset of the original range
//     in order to speed up the whole computation through parallelism. This parameter
//     (together with 'shard') indicates the number of partitions that are being
//     used in the overall computation.
//   - shard: A sampler can be used to sample from a subset of the original range
//     in order to speed up the whole computation through parallelism. This parameter
//     (together with 'num_shards') indicates the particular partition number of a
//     sampler op, when partitioning is being used.
//   - unigrams: A list of unigram counts or probabilities, one per ID in sequential
//     order. Exactly one of vocab_file and unigrams should be passed to this op.
//   - seed: If either seed or seed2 are set to be non-zero, the random number
//     generator is seeded by the given seed.  Otherwise, it is seeded by a
//     random seed.
//   - seed2: An second seed to avoid seed collision.
//
// - Outputs:
//   - sampled_candidates: A vector of length num_sampled, in which each element is
//     the ID of a sampled candidate.
//   - true_expected_count: A batch_size * num_true matrix, representing
//     the number of times each candidate is expected to occur in a batch
//     of sampled candidates. If unique=true, then this is a probability.
//   - sampled_expected_count: A vector of length num_sampled, for each sampled
//     candidate representing the number of times the candidate is expected
//     to occur in a batch of sampled candidates.  If unique=true, then this is a
//     probability.
@_inlineable @inline(__always)
public static func fixedUnigramCandidateSampler(
  trueClasses: Tensor<Int64>,
  numTrue: Int,
  numSampled: Int,
  unique: Bool,
  rangeMax: Int,
  vocabFile: String,
  distortion: Double = 1,
  numReservedIds: Int = 0,
  numShards: Int = 1,
  shard: Int = 0,
  unigrams: [Double],
  seed: Int = 0,
  seed2: Int = 0
) -> (Tensor<Int64>, Tensor<Float>, Tensor<Float>) {
  return #tfop("FixedUnigramCandidateSampler",
    trueClasses,
    num_true: numTrue,
    num_sampled: numSampled,
    unique: unique,
    range_max: rangeMax,
    vocab_file: vocabFile,
    distortion: distortion,
    num_reserved_ids: numReservedIds,
    num_shards: numShards,
    shard: shard,
    unigrams: unigrams,
    seed: seed,
    seed2: seed2)
}

@_inlineable @inline(__always)
public static func floatInput(
  a: Tensor<Float>
) {
  return #tfop("FloatInput",
    a)
}

@_inlineable @inline(__always)
public static func floatOutput(
) -> Tensor<Float> {
  return #tfop("FloatOutput")
}

// Returns element-wise largest integer not greater than x.
@_inlineable @inline(__always)
public static func floor<T: BinaryFloatingPoint>(
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Floor",
    x,
    T: T.self)
}

// Returns x // y element-wise.
//
// *NOTE*: `FloorDiv` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@_inlineable @inline(__always)
public static func floorDiv<T: Numeric>(
  x: Tensor<T>,
  y: Tensor<T>
) -> Tensor<T> {
  return #tfop("FloorDiv",
    x,
    y,
    T: T.self)
}

// Returns element-wise remainder of division. When `x < 0` xor `y < 0` is
//
// true, this follows Python semantics in that the result here is consistent
// with a flooring divide. E.g. `floor(x / y) * y + mod(x, y) = x`.
//
// *NOTE*: `FloorMod` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@_inlineable @inline(__always)
public static func floorMod<T: Numeric>(
  x: Tensor<T>,
  y: Tensor<T>
) -> Tensor<T> {
  return #tfop("FloorMod",
    x,
    y,
    T: T.self)
}

@_inlineable @inline(__always)
public static func foo1(
  a: Tensor<Float>,
  b: Tensor<Int32>,
  c: Tensor<Int32>
) -> (Tensor<Float>, Tensor<Int32>) {
  return #tfop("Foo1",
    a,
    b,
    c)
}

// Performs fractional average pooling on the input.
//
// Fractional average pooling is similar to Fractional max pooling in the pooling
// region generation step. The only difference is that after pooling regions are
// generated, a mean operation is performed instead of a max operation in each
// pooling region.
//
// - Parameter value: 4-D with shape `[batch, height, width, channels]`.
//
// - Attrs:
//   - pooling_ratio: Pooling ratio for each dimension of `value`, currently only
//     supports row and col dimension and should be >= 1.0. For example, a valid
//     pooling ratio looks like [1.0, 1.44, 1.73, 1.0]. The first and last elements
//     must be 1.0 because we don't allow pooling on batch and channels
//     dimensions. 1.44 and 1.73 are pooling ratio on height and width dimensions
//     respectively.
//   - pseudo_random: When set to True, generates the pooling sequence in a
//     pseudorandom fashion, otherwise, in a random fashion. Check paper [Benjamin
//     Graham, Fractional Max-Pooling](http://arxiv.org/abs/1412.6071) for
//     difference between pseudorandom and random.
//   - overlapping: When set to True, it means when pooling, the values at the boundary
//     of adjacent pooling cells are used by both cells. For example:
//
//     `index  0  1  2  3  4`
//
//     `value  20 5  16 3  7`
//
//     If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
//     The result would be [41/3, 26/3] for fractional avg pooling.
//   - deterministic: When set to True, a fixed pooling region will be used when
//     iterating over a FractionalAvgPool node in the computation graph. Mainly used
//     in unit test to make FractionalAvgPool deterministic.
//   - seed: If either seed or seed2 are set to be non-zero, the random number
//     generator is seeded by the given seed.  Otherwise, it is seeded by a
//     random seed.
//   - seed2: An second seed to avoid seed collision.
//
// - Outputs:
//   - output: output tensor after fractional avg pooling.
//   - row_pooling_sequence: row pooling sequence, needed to calculate gradient.
//   - col_pooling_sequence: column pooling sequence, needed to calculate gradient.
@_inlineable @inline(__always)
public static func fractionalAvgPool<T: Numeric>(
  value: Tensor<T>,
  poolingRatio: [Double],
  pseudoRandom: Bool = false,
  overlapping: Bool = false,
  deterministic: Bool = false,
  seed: Int = 0,
  seed2: Int = 0
) -> (Tensor<T>, Tensor<Int64>, Tensor<Int64>) {
  return #tfop("FractionalAvgPool",
    value,
    T: T.self,
    pooling_ratio: poolingRatio,
    pseudo_random: pseudoRandom,
    overlapping: overlapping,
    deterministic: deterministic,
    seed: seed,
    seed2: seed2)
}

// Computes gradient of the FractionalAvgPool function.
//
// Unlike FractionalMaxPoolGrad, we don't need to find arg_max for
// FractionalAvgPoolGrad, we just need to evenly back-propagate each element of
// out_backprop to those indices that form the same pooling cell. Therefore, we
// just need to know the shape of original input tensor, instead of the whole
// tensor.
//
// - Parameters:
//   - orig_input_tensor_shape: Original input tensor shape for `fractional_avg_pool`
//   - out_backprop: 4-D with shape `[batch, height, width, channels]`.  Gradients
//     w.r.t. the output of `fractional_avg_pool`.
//   - row_pooling_sequence: row pooling sequence, form pooling region with
//     col_pooling_sequence.
//   - col_pooling_sequence: column pooling sequence, form pooling region with
//     row_pooling sequence.
//
// - Attr overlapping: When set to True, it means when pooling, the values at the boundary
//   of adjacent pooling cells are used by both cells. For example:
//
//   `index  0  1  2  3  4`
//
//   `value  20 5  16 3  7`
//
//   If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
//   The result would be [41/3, 26/3] for fractional avg pooling.
//
// - Output output: 4-D.  Gradients w.r.t. the input of `fractional_avg_pool`.
@_inlineable @inline(__always)
public static func fractionalAvgPoolGrad<T: Numeric>(
  origInputTensorShape: Tensor<Int64>,
  outBackprop: Tensor<T>,
  rowPoolingSequence: Tensor<Int64>,
  colPoolingSequence: Tensor<Int64>,
  overlapping: Bool = false
) -> Tensor<T> {
  return #tfop("FractionalAvgPoolGrad",
    origInputTensorShape,
    outBackprop,
    rowPoolingSequence,
    colPoolingSequence,
    T: T.self,
    overlapping: overlapping)
}

// Performs fractional max pooling on the input.
//
// Fractional max pooling is slightly different than regular max pooling.  In
// regular max pooling, you downsize an input set by taking the maximum value of
// smaller N x N subsections of the set (often 2x2), and try to reduce the set by
// a factor of N, where N is an integer.  Fractional max pooling, as you might
// expect from the word "fractional", means that the overall reduction ratio N
// does not have to be an integer.
//
// The sizes of the pooling regions are generated randomly but are fairly uniform.
// For example, let's look at the height dimension, and the constraints on the
// list of rows that will be pool boundaries.
//
// First we define the following:
//
// 1.  input_row_length : the number of rows from the input set
// 2.  output_row_length : which will be smaller than the input
// 3.  alpha = input_row_length / output_row_length : our reduction ratio
// 4.  K = floor(alpha)
// 5.  row_pooling_sequence : this is the result list of pool boundary rows
//
// Then, row_pooling_sequence should satisfy:
//
// 1.  a[0] = 0 : the first value of the sequence is 0
// 2.  a[end] = input_row_length : the last value of the sequence is the size
// 3.  K <= (a[i+1] - a[i]) <= K+1 : all intervals are K or K+1 size
// 4.  length(row_pooling_sequence) = output_row_length+1
//
// For more details on fractional max pooling, see this paper:
// [Benjamin Graham, Fractional Max-Pooling](http://arxiv.org/abs/1412.6071)
//
// - Parameter value: 4-D with shape `[batch, height, width, channels]`.
//
// - Attrs:
//   - pooling_ratio: Pooling ratio for each dimension of `value`, currently only
//     supports row and col dimension and should be >= 1.0. For example, a valid
//     pooling ratio looks like [1.0, 1.44, 1.73, 1.0]. The first and last elements
//     must be 1.0 because we don't allow pooling on batch and channels
//     dimensions. 1.44 and 1.73 are pooling ratio on height and width dimensions
//     respectively.
//   - pseudo_random: When set to True, generates the pooling sequence in a
//     pseudorandom fashion, otherwise, in a random fashion. Check paper [Benjamin
//     Graham, Fractional Max-Pooling](http://arxiv.org/abs/1412.6071) for
//     difference between pseudorandom and random.
//   - overlapping: When set to True, it means when pooling, the values at the boundary
//     of adjacent pooling cells are used by both cells. For example:
//
//     `index  0  1  2  3  4`
//
//     `value  20 5  16 3  7`
//
//     If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
//     The result would be [20, 16] for fractional max pooling.
//   - deterministic: When set to True, a fixed pooling region will be used when
//     iterating over a FractionalMaxPool node in the computation graph. Mainly used
//     in unit test to make FractionalMaxPool deterministic.
//   - seed: If either seed or seed2 are set to be non-zero, the random number
//     generator is seeded by the given seed.  Otherwise, it is seeded by a
//     random seed.
//   - seed2: An second seed to avoid seed collision.
//
// - Outputs:
//   - output: output tensor after fractional max pooling.
//   - row_pooling_sequence: row pooling sequence, needed to calculate gradient.
//   - col_pooling_sequence: column pooling sequence, needed to calculate gradient.
@_inlineable @inline(__always)
public static func fractionalMaxPool<T: Numeric>(
  value: Tensor<T>,
  poolingRatio: [Double],
  pseudoRandom: Bool = false,
  overlapping: Bool = false,
  deterministic: Bool = false,
  seed: Int = 0,
  seed2: Int = 0
) -> (Tensor<T>, Tensor<Int64>, Tensor<Int64>) {
  return #tfop("FractionalMaxPool",
    value,
    T: T.self,
    pooling_ratio: poolingRatio,
    pseudo_random: pseudoRandom,
    overlapping: overlapping,
    deterministic: deterministic,
    seed: seed,
    seed2: seed2)
}

// Computes gradient of the FractionalMaxPool function.
//
// - Parameters:
//   - orig_input: Original input for `fractional_max_pool`
//   - orig_output: Original output for `fractional_max_pool`
//   - out_backprop: 4-D with shape `[batch, height, width, channels]`.  Gradients
//     w.r.t. the output of `fractional_max_pool`.
//   - row_pooling_sequence: row pooling sequence, form pooling region with
//     col_pooling_sequence.
//   - col_pooling_sequence: column pooling sequence, form pooling region with
//     row_pooling sequence.
//
// - Attr overlapping: When set to True, it means when pooling, the values at the boundary
//   of adjacent pooling cells are used by both cells. For example:
//
//   `index  0  1  2  3  4`
//
//   `value  20 5  16 3  7`
//
//   If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
//   The result would be [20, 16] for fractional max pooling.
//
// - Output output: 4-D.  Gradients w.r.t. the input of `fractional_max_pool`.
@_inlineable @inline(__always)
public static func fractionalMaxPoolGrad<T: Numeric>(
  origInput: Tensor<T>,
  origOutput: Tensor<T>,
  outBackprop: Tensor<T>,
  rowPoolingSequence: Tensor<Int64>,
  colPoolingSequence: Tensor<Int64>,
  overlapping: Bool = false
) -> Tensor<T> {
  return #tfop("FractionalMaxPoolGrad",
    origInput,
    origOutput,
    outBackprop,
    rowPoolingSequence,
    colPoolingSequence,
    T: T.self,
    overlapping: overlapping)
}

// Batch normalization.
//
// Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
// The size of 1D Tensors matches the dimension C of the 4D Tensors.
//
// - Parameters:
//   - x: A 4D Tensor for input data.
//   - scale: A 1D Tensor for scaling factor, to scale the normalized x.
//   - offset: A 1D Tensor for offset, to shift to the normalized x.
//   - mean: A 1D Tensor for population mean. Used for inference only;
//     must be empty for training.
//   - variance: A 1D Tensor for population variance. Used for inference only;
//     must be empty for training.
//
// - Attrs:
//   - T: The data type for the elements of input and output Tensors.
//   - epsilon: A small float number added to the variance of x.
//   - data_format: The data format for x and y. Either "NHWC" (default) or "NCHW".
//   - is_training: A bool value to indicate the operation is for training (default)
//     or inference.
//
// - Outputs:
//   - y: A 4D Tensor for output data.
//   - batch_mean: A 1D Tensor for the computed batch mean, to be used by TensorFlow
//     to compute the running mean.
//   - batch_variance: A 1D Tensor for the computed batch variance, to be used by
//     TensorFlow to compute the running variance.
//   - reserve_space_1: A 1D Tensor for the computed batch mean, to be reused
//     in the gradient computation.
//   - reserve_space_2: A 1D Tensor for the computed batch variance (inverted variance
//     in the cuDNN case), to be reused in the gradient computation.
@_inlineable @inline(__always)
public static func fusedBatchNorm<T: BinaryFloatingPoint>(
  x: Tensor<T>,
  scale: Tensor<T>,
  offset: Tensor<T>,
  mean: Tensor<T>,
  variance: Tensor<T>,
  epsilon: Double = 0.0001,
  dataFormat: String = "NHWC",
  isTraining: Bool = true
) -> (Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>) {
  return #tfop("FusedBatchNorm",
    x,
    scale,
    offset,
    mean,
    variance,
    T: T.self,
    epsilon: epsilon,
    data_format: dataFormat,
    is_training: isTraining)
}

// Gradient for batch normalization.
//
// Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
// The size of 1D Tensors matches the dimension C of the 4D Tensors.
//
// - Parameters:
//   - y_backprop: A 4D Tensor for the gradient with respect to y.
//   - x: A 4D Tensor for input data.
//   - scale: A 1D Tensor for scaling factor, to scale the normalized x.
//   - reserve_space_1: When is_training is True, a 1D Tensor for the computed batch
//     mean to be reused in gradient computation. When is_training is
//     False, a 1D Tensor for the population mean to be reused in both
//     1st and 2nd order gradient computation.
//   - reserve_space_2: When is_training is True, a 1D Tensor for the computed batch
//     variance (inverted variance in the cuDNN case) to be reused in
//     gradient computation. When is_training is False, a 1D Tensor
//     for the population variance to be reused in both 1st and 2nd
//     order gradient computation.
//
// - Attrs:
//   - T: The data type for the elements of input and output Tensors.
//   - epsilon: A small float number added to the variance of x.
//   - data_format: The data format for y_backprop, x, x_backprop.
//     Either "NHWC" (default) or "NCHW".
//   - is_training: A bool value to indicate the operation is for training (default)
//     or inference.
//
// - Outputs:
//   - x_backprop: A 4D Tensor for the gradient with respect to x.
//   - scale_backprop: A 1D Tensor for the gradient with respect to scale.
//   - offset_backprop: A 1D Tensor for the gradient with respect to offset.
//   - reserve_space_3: Unused placeholder to match the mean input in FusedBatchNorm.
//   - reserve_space_4: Unused placeholder to match the variance input
//     in FusedBatchNorm.
@_inlineable @inline(__always)
public static func fusedBatchNormGrad<T: BinaryFloatingPoint>(
  yBackprop: Tensor<T>,
  x: Tensor<T>,
  scale: Tensor<T>,
  reserveSpace1: Tensor<T>,
  reserveSpace2: Tensor<T>,
  epsilon: Double = 0.0001,
  dataFormat: String = "NHWC",
  isTraining: Bool = true
) -> (Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>) {
  return #tfop("FusedBatchNormGrad",
    yBackprop,
    x,
    scale,
    reserveSpace1,
    reserveSpace2,
    T: T.self,
    epsilon: epsilon,
    data_format: dataFormat,
    is_training: isTraining)
}

// Gradient for batch normalization.
//
// Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
// The size of 1D Tensors matches the dimension C of the 4D Tensors.
//
// - Parameters:
//   - y_backprop: A 4D Tensor for the gradient with respect to y.
//   - x: A 4D Tensor for input data.
//   - scale: A 1D Tensor for scaling factor, to scale the normalized x.
//   - reserve_space_1: When is_training is True, a 1D Tensor for the computed batch
//     mean to be reused in gradient computation. When is_training is
//     False, a 1D Tensor for the population mean to be reused in both
//     1st and 2nd order gradient computation.
//   - reserve_space_2: When is_training is True, a 1D Tensor for the computed batch
//     variance (inverted variance in the cuDNN case) to be reused in
//     gradient computation. When is_training is False, a 1D Tensor
//     for the population variance to be reused in both 1st and 2nd
//     order gradient computation.
//
// - Attrs:
//   - T: The data type for the elements of input and output Tensors.
//   - U: The data type for the scale, offset, mean, and variance.
//   - epsilon: A small float number added to the variance of x.
//   - data_format: The data format for y_backprop, x, x_backprop.
//     Either "NHWC" (default) or "NCHW".
//   - is_training: A bool value to indicate the operation is for training (default)
//     or inference.
//
// - Outputs:
//   - x_backprop: A 4D Tensor for the gradient with respect to x.
//   - scale_backprop: A 1D Tensor for the gradient with respect to scale.
//   - offset_backprop: A 1D Tensor for the gradient with respect to offset.
//   - reserve_space_3: Unused placeholder to match the mean input in FusedBatchNorm.
//   - reserve_space_4: Unused placeholder to match the variance input
//     in FusedBatchNorm.
@_inlineable @inline(__always)
public static func fusedBatchNormGradV2<T: BinaryFloatingPoint, U: BinaryFloatingPoint>(
  yBackprop: Tensor<T>,
  x: Tensor<T>,
  scale: Tensor<Float>,
  reserveSpace1: Tensor<U>,
  reserveSpace2: Tensor<U>,
  epsilon: Double = 0.0001,
  dataFormat: String = "NHWC",
  isTraining: Bool = true
) -> (Tensor<T>, Tensor<U>, Tensor<U>, Tensor<U>, Tensor<U>) {
  return #tfop("FusedBatchNormGradV2",
    yBackprop,
    x,
    scale,
    reserveSpace1,
    reserveSpace2,
    T: T.self,
    U: U.self,
    epsilon: epsilon,
    data_format: dataFormat,
    is_training: isTraining)
}

// Batch normalization.
//
// Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
// The size of 1D Tensors matches the dimension C of the 4D Tensors.
//
// - Parameters:
//   - x: A 4D Tensor for input data.
//   - scale: A 1D Tensor for scaling factor, to scale the normalized x.
//   - offset: A 1D Tensor for offset, to shift to the normalized x.
//   - mean: A 1D Tensor for population mean. Used for inference only;
//     must be empty for training.
//   - variance: A 1D Tensor for population variance. Used for inference only;
//     must be empty for training.
//
// - Attrs:
//   - T: The data type for the elements of input and output Tensors.
//   - U: The data type for the scale, offset, mean, and variance.
//   - epsilon: A small float number added to the variance of x.
//   - data_format: The data format for x and y. Either "NHWC" (default) or "NCHW".
//   - is_training: A bool value to indicate the operation is for training (default)
//     or inference.
//
// - Outputs:
//   - y: A 4D Tensor for output data.
//   - batch_mean: A 1D Tensor for the computed batch mean, to be used by TensorFlow
//     to compute the running mean.
//   - batch_variance: A 1D Tensor for the computed batch variance, to be used by
//     TensorFlow to compute the running variance.
//   - reserve_space_1: A 1D Tensor for the computed batch mean, to be reused
//     in the gradient computation.
//   - reserve_space_2: A 1D Tensor for the computed batch variance (inverted variance
//     in the cuDNN case), to be reused in the gradient computation.
@_inlineable @inline(__always)
public static func fusedBatchNormV2<T: BinaryFloatingPoint, U: BinaryFloatingPoint>(
  x: Tensor<T>,
  scale: Tensor<U>,
  offset: Tensor<U>,
  mean: Tensor<U>,
  variance: Tensor<U>,
  epsilon: Double = 0.0001,
  dataFormat: String = "NHWC",
  isTraining: Bool = true
) -> (Tensor<T>, Tensor<U>, Tensor<U>, Tensor<U>, Tensor<U>) {
  return #tfop("FusedBatchNormV2",
    x,
    scale,
    offset,
    mean,
    variance,
    T: T.self,
    U: U.self,
    epsilon: epsilon,
    data_format: dataFormat,
    is_training: isTraining)
}

// Performs a padding as a preprocess during a convolution.
//
// Similar to FusedResizeAndPadConv2d, this op allows for an optimized
// implementation where the spatial padding transformation stage is fused with the
// im2col lookup, but in this case without the bilinear filtering required for
// resizing. Fusing the padding prevents the need to write out the intermediate
// results as whole tensors, reducing memory pressure, and we can get some latency
// gains by merging the transformation calculations.
// The data_format attribute for Conv2D isn't supported by this op, and 'NHWC'
// order is used instead.
// Internally this op uses a single per-graph scratch buffer, which means that it
// will block if multiple versions are being run in parallel. This is because this
// operator is primarily an optimization to minimize memory usage.
//
// - Parameters:
//   - input: 4-D with shape `[batch, in_height, in_width, in_channels]`.
//   - paddings: A two-column matrix specifying the padding sizes. The number of
//     rows must be the same as the rank of `input`.
//   - filter: 4-D with shape
//     `[filter_height, filter_width, in_channels, out_channels]`.
//
// - Attrs:
//   - strides: 1-D of length 4.  The stride of the sliding window for each dimension
//     of `input`. Must be in the same order as the dimension specified with format.
//   - padding: The type of padding algorithm to use.
@_inlineable @inline(__always)
public static func fusedPadConv2D<T: BinaryFloatingPoint>(
  input: Tensor<T>,
  paddings: Tensor<Int32>,
  filter: Tensor<T>,
  mode: Mode3,
  strides: [Int],
  padding: Padding
) -> Tensor<T> {
  return #tfop("FusedPadConv2D",
    input,
    paddings,
    filter,
    T: T.self,
    mode: mode.rawValue,
    strides: strides,
    padding: padding.rawValue)
}

// Performs a resize and padding as a preprocess during a convolution.
//
// It's often possible to do spatial transformations more efficiently as part of
// the packing stage of a convolution, so this op allows for an optimized
// implementation where these stages are fused together. This prevents the need to
// write out the intermediate results as whole tensors, reducing memory pressure,
// and we can get some latency gains by merging the transformation calculations.
// The data_format attribute for Conv2D isn't supported by this op, and defaults to
// 'NHWC' order.
// Internally this op uses a single per-graph scratch buffer, which means that it
// will block if multiple versions are being run in parallel. This is because this
// operator is primarily an optimization to minimize memory usage.
//
// - Parameters:
//   - input: 4-D with shape `[batch, in_height, in_width, in_channels]`.
//   - size: A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
//     new size for the images.
//   - paddings: A two-column matrix specifying the padding sizes. The number of
//     rows must be the same as the rank of `input`.
//   - filter: 4-D with shape
//     `[filter_height, filter_width, in_channels, out_channels]`.
//
// - Attrs:
//   - resize_align_corners: If true, the centers of the 4 corner pixels of the input and output tensors are
//     aligned, preserving the values at the corner pixels. Defaults to false.
//   - strides: 1-D of length 4.  The stride of the sliding window for each dimension
//     of `input`. Must be in the same order as the dimension specified with format.
//   - padding: The type of padding algorithm to use.
@_inlineable @inline(__always)
public static func fusedResizeAndPadConv2D<T: BinaryFloatingPoint>(
  input: Tensor<T>,
  size: Tensor<Int32>,
  paddings: Tensor<Int32>,
  filter: Tensor<T>,
  resizeAlignCorners: Bool = false,
  mode: Mode3,
  strides: [Int],
  padding: Padding
) -> Tensor<T> {
  return #tfop("FusedResizeAndPadConv2D",
    input,
    size,
    paddings,
    filter,
    T: T.self,
    resize_align_corners: resizeAlignCorners,
    mode: mode.rawValue,
    strides: strides,
    padding: padding.rawValue)
}

// Computes the GRU cell forward propagation for 1 time step.
//
// Args
//     x: Input to the GRU cell.
//     h_prev: State input from the previous GRU cell.
//     w_ru: Weight matrix for the reset and update gate.
//     w_c: Weight matrix for the cell connection gate.
//     b_ru: Bias vector for the reset and update gate.
//     b_c: Bias vector for the cell connection gate.
//
// Returns
//     r: Output of the reset gate.
//     u: Output of the update gate.
//     c: Output of the cell connection gate.
//     h: Current state of the GRU cell.
//
// Note on notation of the variables:
//
// Concatenation of a and b is represented by a_b
// Element-wise dot product of a and b is represented by ab
// Element-wise dot product is represented by \circ
// Matrix multiplication is represented by *
//
// Biases are initialized with :
// `b_ru` - constant_initializer(1.0)
// `b_c` - constant_initializer(0.0)
//
// This kernel op implements the following mathematical equations:
//
// ```
// x_h_prev = [x, h_prev]
//
// [r_bar u_bar] = x_h_prev * w_ru + b_ru
//
// r = sigmoid(r_bar)
// u = sigmoid(u_bar)
//
// h_prevr = h_prev \circ r
//
// x_h_prevr = [x h_prevr]
//
// c_bar = x_h_prevr * w_c + b_c
// c = tanh(c_bar)
//
// h = (1-u) \circ c + u \circ h_prev
// ```
@_inlineable @inline(__always)
public static func gRUBlockCell<T: BinaryFloatingPoint>(
  x: Tensor<T>,
  hPrev: Tensor<T>,
  wRu: Tensor<T>,
  wC: Tensor<T>,
  bRu: Tensor<T>,
  bC: Tensor<T>
) -> (Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>) {
  return #tfop("GRUBlockCell",
    x,
    hPrev,
    wRu,
    wC,
    bRu,
    bC,
    T: T.self)
}

// Computes the GRU cell back-propagation for 1 time step.
//
// Args
//     x: Input to the GRU cell.
//     h_prev: State input from the previous GRU cell.
//     w_ru: Weight matrix for the reset and update gate.
//     w_c: Weight matrix for the cell connection gate.
//     b_ru: Bias vector for the reset and update gate.
//     b_c: Bias vector for the cell connection gate.
//     r: Output of the reset gate.
//     u: Output of the update gate.
//     c: Output of the cell connection gate.
//     d_h: Gradients of the h_new wrt to objective function.
//
// Returns
//     d_x: Gradients of the x wrt to objective function.
//     d_h_prev: Gradients of the h wrt to objective function.
//     d_c_bar Gradients of the c_bar wrt to objective function.
//     d_r_bar_u_bar Gradients of the r_bar & u_bar wrt to objective function.
//
// This kernel op implements the following mathematical equations:
//
// Note on notation of the variables:
//
// Concatenation of a and b is represented by a_b
// Element-wise dot product of a and b is represented by ab
// Element-wise dot product is represented by \circ
// Matrix multiplication is represented by *
//
// Additional notes for clarity:
//
// `w_ru` can be segmented into 4 different matrices.
// ```
// w_ru = [w_r_x w_u_x
//         w_r_h_prev w_u_h_prev]
// ```
// Similarly, `w_c` can be segmented into 2 different matrices.
// ```
// w_c = [w_c_x w_c_h_prevr]
// ```
// Same goes for biases.
// ```
// b_ru = [b_ru_x b_ru_h]
// b_c = [b_c_x b_c_h]
// ```
// Another note on notation:
// ```
// d_x = d_x_component_1 + d_x_component_2
//
// where d_x_component_1 = d_r_bar * w_r_x^T + d_u_bar * w_r_x^T
// and d_x_component_2 = d_c_bar * w_c_x^T
//
// d_h_prev = d_h_prev_component_1 + d_h_prevr \circ r + d_h \circ u
// where d_h_prev_componenet_1 = d_r_bar * w_r_h_prev^T + d_u_bar * w_r_h_prev^T
// ```
//
// Mathematics behind the Gradients below:
// ```
// d_c_bar = d_h \circ (1-u) \circ (1-c \circ c)
// d_u_bar = d_h \circ (h-c) \circ u \circ (1-u)
//
// d_r_bar_u_bar = [d_r_bar d_u_bar]
//
// [d_x_component_1 d_h_prev_component_1] = d_r_bar_u_bar * w_ru^T
//
// [d_x_component_2 d_h_prevr] = d_c_bar * w_c^T
//
// d_x = d_x_component_1 + d_x_component_2
//
// d_h_prev = d_h_prev_component_1 + d_h_prevr \circ r + u
// ```
// Below calculation is performed in the python wrapper for the Gradients
// (not in the gradient kernel.)
// ```
// d_w_ru = x_h_prevr^T * d_c_bar
//
// d_w_c = x_h_prev^T * d_r_bar_u_bar
//
// d_b_ru = sum of d_r_bar_u_bar along axis = 0
//
// d_b_c = sum of d_c_bar along axis = 0
// ```
@_inlineable @inline(__always)
public static func gRUBlockCellGrad<T: BinaryFloatingPoint>(
  x: Tensor<T>,
  hPrev: Tensor<T>,
  wRu: Tensor<T>,
  wC: Tensor<T>,
  bRu: Tensor<T>,
  bC: Tensor<T>,
  r: Tensor<T>,
  u: Tensor<T>,
  c: Tensor<T>,
  dH: Tensor<T>
) -> (Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>) {
  return #tfop("GRUBlockCellGrad",
    x,
    hPrev,
    wRu,
    wC,
    bRu,
    bC,
    r,
    u,
    c,
    dH,
    T: T.self)
}

// Gather slices from `params` according to `indices`.
//
// `indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
// Produces an output tensor with shape `indices.shape + params.shape[1:]` where:
//
// ```python
//     # Scalar indices
//     output[:, ..., :] = params[indices, :, ... :]
//
//     # Vector indices
//     output[i, :, ..., :] = params[indices[i], :, ... :]
//
//     # Higher rank indices
//     output[i, ..., j, :, ... :] = params[indices[i, ..., j], :, ..., :]
// ```
//
// If `indices` is a permutation and `len(indices) == params.shape[0]` then
// this operation will permute `params` accordingly.
//
// `validate_indices`: DEPRECATED. If this operation is assigned to CPU, values in
// `indices` are always validated to be within range. If assigned to GPU,
// out-of-bound indices result in safe but unspecified behavior, which may include
// raising an error.
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="https://www.tensorflow.org/images/Gather.png" alt>
// </div>
@_inlineable @inline(__always)
public static func gather<Tparams: Numeric, Tindices: BinaryInteger>(
  params: Tensor<Tparams>,
  indices: Tensor<Tindices>,
  validateIndices: Bool = true
) -> Tensor<Tparams> {
  return #tfop("Gather",
    params,
    indices,
    Tparams: Tparams.self,
    Tindices: Tindices.self,
    validate_indices: validateIndices)
}

// Gather slices from `params` into a Tensor with shape specified by `indices`.
//
// `indices` is an K-dimensional integer tensor, best thought of as a
// (K-1)-dimensional tensor of indices into `params`, where each element defines a
// slice of `params`:
//
//     output[i_0, ..., i_{K-2}] = params[indices[i0, ..., i_{K-2}]]
//
// Whereas in @{tf.gather} `indices` defines slices into the first
// dimension of `params`, in `tf.gather_nd`, `indices` defines slices into the
// first `N` dimensions of `params`, where `N = indices.shape[-1]`.
//
// The last dimension of `indices` can be at most the rank of
// `params`:
//
//     indices.shape[-1] <= params.rank
//
// The last dimension of `indices` corresponds to elements
// (if `indices.shape[-1] == params.rank`) or slices
// (if `indices.shape[-1] < params.rank`) along dimension `indices.shape[-1]`
// of `params`.  The output tensor has shape
//
//     indices.shape[:-1] + params.shape[indices.shape[-1]:]
//
// Note that on CPU, if an out of bound index is found, an error is returned.
// On GPU, if an out of bound index is found, a 0 is stored in the
// corresponding output value.
//
// Some examples below.
//
// Simple indexing into a matrix:
//
// ```python
//     indices = [[0, 0], [1, 1]]
//     params = [['a', 'b'], ['c', 'd']]
//     output = ['a', 'd']
// ```
//
// Slice indexing into a matrix:
//
// ```python
//     indices = [[1], [0]]
//     params = [['a', 'b'], ['c', 'd']]
//     output = [['c', 'd'], ['a', 'b']]
// ```
//
// Indexing into a 3-tensor:
//
// ```python
//     indices = [[1]]
//     params = [[['a0', 'b0'], ['c0', 'd0']],
//               [['a1', 'b1'], ['c1', 'd1']]]
//     output = [[['a1', 'b1'], ['c1', 'd1']]]
//
//
//     indices = [[0, 1], [1, 0]]
//     params = [[['a0', 'b0'], ['c0', 'd0']],
//               [['a1', 'b1'], ['c1', 'd1']]]
//     output = [['c0', 'd0'], ['a1', 'b1']]
//
//
//     indices = [[0, 0, 1], [1, 0, 1]]
//     params = [[['a0', 'b0'], ['c0', 'd0']],
//               [['a1', 'b1'], ['c1', 'd1']]]
//     output = ['b0', 'b1']
// ```
//
// Batched indexing into a matrix:
//
// ```python
//     indices = [[[0, 0]], [[0, 1]]]
//     params = [['a', 'b'], ['c', 'd']]
//     output = [['a'], ['b']]
// ```
//
// Batched slice indexing into a matrix:
//
// ```python
//     indices = [[[1]], [[0]]]
//     params = [['a', 'b'], ['c', 'd']]
//     output = [[['c', 'd']], [['a', 'b']]]
// ```
//
// Batched indexing into a 3-tensor:
//
// ```python
//     indices = [[[1]], [[0]]]
//     params = [[['a0', 'b0'], ['c0', 'd0']],
//               [['a1', 'b1'], ['c1', 'd1']]]
//     output = [[[['a1', 'b1'], ['c1', 'd1']]],
//               [[['a0', 'b0'], ['c0', 'd0']]]]
//
//     indices = [[[0, 1], [1, 0]], [[0, 0], [1, 1]]]
//     params = [[['a0', 'b0'], ['c0', 'd0']],
//               [['a1', 'b1'], ['c1', 'd1']]]
//     output = [[['c0', 'd0'], ['a1', 'b1']],
//               [['a0', 'b0'], ['c1', 'd1']]]
//
//
//     indices = [[[0, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 1, 0]]]
//     params = [[['a0', 'b0'], ['c0', 'd0']],
//               [['a1', 'b1'], ['c1', 'd1']]]
//     output = [['b0', 'b1'], ['d0', 'c1']]
// ```
//
// - Parameters:
//   - params: The tensor from which to gather values.
//   - indices: Index tensor.
//
// - Output output: Values from `params` gathered from indices given by `indices`, with
//   shape `indices.shape[:-1] + params.shape[indices.shape[-1]:]`.
@_inlineable @inline(__always)
public static func gatherNd<Tparams: Numeric, Tindices: BinaryInteger>(
  params: Tensor<Tparams>,
  indices: Tensor<Tindices>
) -> Tensor<Tparams> {
  return #tfop("GatherNd",
    params,
    indices,
    Tparams: Tparams.self,
    Tindices: Tindices.self)
}

// Gather slices from `params` axis `axis` according to `indices`.
//
// `indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
// Produces an output tensor with shape `params.shape[:axis] + indices.shape +
// params.shape[axis + 1:]` where:
//
// ```python
//     # Scalar indices (output is rank(params) - 1).
//     output[a_0, ..., a_n, b_0, ..., b_n] =
//       params[a_0, ..., a_n, indices, b_0, ..., b_n]
//
//     # Vector indices (output is rank(params)).
//     output[a_0, ..., a_n, i, b_0, ..., b_n] =
//       params[a_0, ..., a_n, indices[i], b_0, ..., b_n]
//
//     # Higher rank indices (output is rank(params) + rank(indices) - 1).
//     output[a_0, ..., a_n, i, ..., j, b_0, ... b_n] =
//       params[a_0, ..., a_n, indices[i, ..., j], b_0, ..., b_n]
// ```
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="https://www.tensorflow.org/images/Gather.png" alt>
// </div>
//
// Note that on CPU, if an out of bound index is found, an error is returned.
// On GPU, if an out of bound index is found, a 0 is stored in the
// corresponding output value.
//
// - Parameters:
//   - params: The tensor from which to gather values. Must be at least rank
//     `axis + 1`.
//   - indices: Index tensor. Must be in range `[0, params.shape[axis])`.
//   - axis: The axis in `params` to gather `indices` from. Defaults to the first
//     dimension. Supports negative indexes.
//
// - Output output: Values from `params` gathered from indices given by `indices`, with
//   shape `params.shape[:axis] + indices.shape + params.shape[axis + 1:]`.
@_inlineable @inline(__always)
public static func gatherV2<Tparams: Numeric, Tindices: BinaryInteger, Taxis: BinaryInteger>(
  params: Tensor<Tparams>,
  indices: Tensor<Tindices>,
  axis: Tensor<Taxis>
) -> Tensor<Tparams> {
  return #tfop("GatherV2",
    params,
    indices,
    axis,
    Tparams: Tparams.self,
    Tindices: Tindices.self,
    Taxis: Taxis.self)
}

@_inlineable @inline(__always)
public static func graphDefVersion(
) -> Tensor<Int32> {
  return #tfop("GraphDefVersion")
}

// Returns the truth value of (x > y) element-wise.
//
// *NOTE*: `Greater` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@_inlineable @inline(__always)
public static func greater<T: Numeric>(
  x: Tensor<T>,
  y: Tensor<T>
) -> Tensor<Bool> {
  return #tfop("Greater",
    x,
    y,
    T: T.self)
}

// Returns the truth value of (x >= y) element-wise.
//
// *NOTE*: `GreaterEqual` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@_inlineable @inline(__always)
public static func greaterEqual<T: Numeric>(
  x: Tensor<T>,
  y: Tensor<T>
) -> Tensor<Bool> {
  return #tfop("GreaterEqual",
    x,
    y,
    T: T.self)
}

// Gives a guarantee to the TF runtime that the input tensor is a constant.
//
// The runtime is then free to make optimizations based on this.
//
// Only accepts value typed tensors as inputs and rejects resource variable handles
// as input.
//
// Returns the input tensor without modification.
@_inlineable @inline(__always)
public static func guaranteeConst<T: Numeric>(
  input: Tensor<T>
) -> Tensor<T> {
  return #tfop("GuaranteeConst",
    input,
    T: T.self)
}

// Convert one or more images from HSV to RGB.
//
// Outputs a tensor of the same shape as the `images` tensor, containing the RGB
// value of the pixels. The output is only well defined if the value in `images`
// are in `[0,1]`.
//
// See `rgb_to_hsv` for a description of the HSV encoding.
//
// - Parameter images: 1-D or higher rank. HSV data to convert. Last dimension must be size 3.
//
// - Output output: `images` converted to RGB.
@_inlineable @inline(__always)
public static func hSVToRGB<T: BinaryFloatingPoint>(
  images: Tensor<T>
) -> Tensor<T> {
  return #tfop("HSVToRGB",
    images,
    T: T.self)
}

// Return histogram of values.
//
// Given the tensor `values`, this operation returns a rank 1 histogram counting
// the number of entries in `values` that fall into every bin.  The bins are
// equal width and determined by the arguments `value_range` and `nbins`.
//
// ```python
// # Bins will be:  (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
// nbins = 5
// value_range = [0.0, 5.0]
// new_values = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]
//
// with tf.get_default_session() as sess:
//   hist = tf.histogram_fixed_width(new_values, value_range, nbins=5)
//   variables.global_variables_initializer().run()
//   sess.run(hist) => [2, 1, 1, 0, 2]
// ```
//
// - Parameters:
//   - values: Numeric `Tensor`.
//   - value_range: Shape [2] `Tensor` of same `dtype` as `values`.
//     values <= value_range[0] will be mapped to hist[0],
//     values >= value_range[1] will be mapped to hist[-1].
//   - nbins: Scalar `int32 Tensor`.  Number of histogram bins.
//
// - Output out: A 1-D `Tensor` holding histogram of values.
@_inlineable @inline(__always)
public static func histogramFixedWidth<T: Numeric, Dtype: BinaryInteger>(
  values: Tensor<T>,
  valueRange: Tensor<T>,
  nbins: Tensor<Int32>
) -> Tensor<Dtype> {
  return #tfop("HistogramFixedWidth",
    values,
    valueRange,
    nbins,
    T: T.self,
    Dtype: Dtype.self)
}

// Return a tensor with the same shape and contents as the input tensor or value.
@_inlineable @inline(__always)
public static func identity<T: Numeric>(
  input: Tensor<T>
) -> Tensor<T> {
  return #tfop("Identity",
    input,
    T: T.self)
}

// Returns a list of tensors with the same shapes and contents as the input
//
// tensors.
//
// This op can be used to override the gradient for complicated functions. For
// example, suppose y = f(x) and we wish to apply a custom function g for backprop
// such that dx = g(dy). In Python,
//
// ```python
// with tf.get_default_graph().gradient_override_map(
//     {'IdentityN': 'OverrideGradientWithG'}):
//   y, _ = identity_n([f(x), x])
//
// @tf.RegisterGradient('OverrideGradientWithG')
// def ApplyG(op, dy, _):
//   return [None, g(dy)]  # Do not backprop to f(x).
// ```
@_inlineable @inline(__always)
public static func identityN<T: Numeric>(
  input: [Tensor<T>]
) -> [Tensor<T>] {
  return #tfop("IdentityN",
    input)
}

// Compute the lower regularized incomplete Gamma function `Q(a, x)`.
//
// The lower regularized incomplete Gamma function is defined as:
//
//
// \\(P(a, x) = gamma(a, x) / Gamma(a) = 1 - Q(a, x)\\)
//
// where
//
// \\(gamma(a, x) = int_{0}^{x} t^{a-1} exp(-t) dt\\)
//
// is the lower incomplete Gamma function.
//
// Note, above `Q(a, x)` (`Igammac`) is the upper regularized complete
// Gamma function.
@_inlineable @inline(__always)
public static func igamma<T: BinaryFloatingPoint>(
  a: Tensor<T>,
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Igamma",
    a,
    x,
    T: T.self)
}

// Compute the upper regularized incomplete Gamma function `Q(a, x)`.
//
// The upper regularized incomplete Gamma function is defined as:
//
// \\(Q(a, x) = Gamma(a, x) / Gamma(a) = 1 - P(a, x)\\)
//
// where
//
// \\(Gamma(a, x) = int_{x}^{\infty} t^{a-1} exp(-t) dt\\)
//
// is the upper incomplete Gama function.
//
// Note, above `P(a, x)` (`Igamma`) is the lower regularized complete
// Gamma function.
@_inlineable @inline(__always)
public static func igammac<T: BinaryFloatingPoint>(
  a: Tensor<T>,
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Igammac",
    a,
    x,
    T: T.self)
}

// Returns the imaginary part of a complex number.
//
// Given a tensor `input` of complex numbers, this operation returns a tensor of
// type `float` that is the imaginary part of each element in `input`. All
// elements in `input` must be complex numbers of the form \\(a + bj\\), where *a*
// is the real part and *b* is the imaginary part returned by this operation.
//
// For example:
//
// ```
// # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
// tf.imag(input) ==> [4.75, 5.75]
// ```
@_inlineable @inline(__always)
public static func imag<T: Numeric, Tout: BinaryFloatingPoint>(
  input: Tensor<T>
) -> Tensor<Tout> {
  return #tfop("Imag",
    input,
    T: T.self,
    Tout: Tout.self)
}

@_inlineable @inline(__always)
public static func inPolymorphicTwice<T: Numeric>(
  a: [Tensor<T>],
  b: [Tensor<T>]
) {
  return #tfop("InPolymorphicTwice",
    a,
    b,
    T: T.self)
}

// Says whether the targets are in the top `K` predictions.
//
// This outputs a `batch_size` bool array, an entry `out[i]` is `true` if the
// prediction for the target class is among the top `k` predictions among
// all predictions for example `i`. Note that the behavior of `InTopK` differs
// from the `TopK` op in its handling of ties; if multiple classes have the
// same prediction value and straddle the top-`k` boundary, all of those
// classes are considered to be in the top `k`.
//
// More formally, let
//
//   \\(predictions_i\\) be the predictions for all classes for example `i`,
//   \\(targets_i\\) be the target class for example `i`,
//   \\(out_i\\) be the output for example `i`,
//
// $$out_i = predictions_{i, targets_i} \in TopKIncludingTies(predictions_i)$$
//
// - Parameters:
//   - predictions: A `batch_size` x `classes` tensor.
//   - targets: A `batch_size` vector of class ids.
//
// - Attr k: Number of top elements to look at for computing precision.
//
// - Output precision: Computed Precision at `k` as a `bool Tensor`.
@_inlineable @inline(__always)
public static func inTopK<T: BinaryInteger>(
  predictions: Tensor<Float>,
  targets: Tensor<T>,
  k: Int
) -> Tensor<Bool> {
  return #tfop("InTopK",
    predictions,
    targets,
    T: T.self,
    k: k)
}

// Says whether the targets are in the top `K` predictions.
//
// This outputs a `batch_size` bool array, an entry `out[i]` is `true` if the
// prediction for the target class is among the top `k` predictions among
// all predictions for example `i`. Note that the behavior of `InTopK` differs
// from the `TopK` op in its handling of ties; if multiple classes have the
// same prediction value and straddle the top-`k` boundary, all of those
// classes are considered to be in the top `k`.
//
// More formally, let
//
//   \\(predictions_i\\) be the predictions for all classes for example `i`,
//   \\(targets_i\\) be the target class for example `i`,
//   \\(out_i\\) be the output for example `i`,
//
// $$out_i = predictions_{i, targets_i} \in TopKIncludingTies(predictions_i)$$
//
// - Parameters:
//   - predictions: A `batch_size` x `classes` tensor.
//   - targets: A `batch_size` vector of class ids.
//   - k: Number of top elements to look at for computing precision.
//
// - Output precision: Computed precision at `k` as a `bool Tensor`.
@_inlineable @inline(__always)
public static func inTopKV2<T: BinaryInteger>(
  predictions: Tensor<Float>,
  targets: Tensor<T>,
  k: Tensor<T>
) -> Tensor<Bool> {
  return #tfop("InTopKV2",
    predictions,
    targets,
    k,
    T: T.self)
}

//     Adds v into specified rows of x.
//
//     Computes y = x; y[i, :] += v; return y.
//
// - Parameters:
//   - x: A `Tensor` of type T.
//   - i: A vector. Indices into the left-most dimension of `x`.
//   - v: A `Tensor` of type T. Same dimension sizes as x except the first dimension, which must be the same as i's size.
//
// - Output y: A `Tensor` of type T. An alias of `x`. The content of `y` is undefined if there are duplicates in `i`.
@_inlineable @inline(__always)
public static func inplaceAdd<T: Numeric>(
  x: Tensor<T>,
  i: Tensor<Int32>,
  v: Tensor<T>
) -> Tensor<T> {
  return #tfop("InplaceAdd",
    x,
    i,
    v,
    T: T.self)
}

//     Subtracts `v` into specified rows of `x`.
//
//     Computes y = x; y[i, :] -= v; return y.
//
// - Parameters:
//   - x: A `Tensor` of type T.
//   - i: A vector. Indices into the left-most dimension of `x`.
//   - v: A `Tensor` of type T. Same dimension sizes as x except the first dimension, which must be the same as i's size.
//
// - Output y: A `Tensor` of type T. An alias of `x`. The content of `y` is undefined if there are duplicates in `i`.
@_inlineable @inline(__always)
public static func inplaceSub<T: Numeric>(
  x: Tensor<T>,
  i: Tensor<Int32>,
  v: Tensor<T>
) -> Tensor<T> {
  return #tfop("InplaceSub",
    x,
    i,
    v,
    T: T.self)
}

//     Updates specified rows with values in `v`.
//
//     Computes `x[i, :] = v; return x`.
//
// - Parameters:
//   - x: A tensor of type `T`.
//   - i: A vector. Indices into the left-most dimension of `x`.
//   - v: A `Tensor` of type T. Same dimension sizes as x except the first dimension, which must be the same as i's size.
//
// - Output y: A `Tensor` of type T. An alias of `x`. The content of `y` is undefined if there are duplicates in `i`.
@_inlineable @inline(__always)
public static func inplaceUpdate<T: Numeric>(
  x: Tensor<T>,
  i: Tensor<Int32>,
  v: Tensor<T>
) -> Tensor<T> {
  return #tfop("InplaceUpdate",
    x,
    i,
    v,
    T: T.self)
}

@_inlineable @inline(__always)
public static func int64Output(
) -> Tensor<Int64> {
  return #tfop("Int64Output")
}

@_inlineable @inline(__always)
public static func intAttr(
  foo: Int = 1
) -> Tensor<Int64> {
  return #tfop("IntAttr",
    foo: foo)
}

@_inlineable @inline(__always)
public static func intInput(
  a: Tensor<Int32>
) {
  return #tfop("IntInput",
    a)
}

@_inlineable @inline(__always)
public static func intInputFloatInput(
  a: Tensor<Int32>,
  b: Tensor<Float>
) {
  return #tfop("IntInputFloatInput",
    a,
    b)
}

@_inlineable @inline(__always)
public static func intInputIntOutput(
  a: Tensor<Int32>
) -> Tensor<Int32> {
  return #tfop("IntInputIntOutput",
    a)
}

@_inlineable @inline(__always)
public static func intOutput(
) -> Tensor<Int32> {
  return #tfop("IntOutput")
}

@_inlineable @inline(__always)
public static func intOutputFloatOutput(
) -> (Tensor<Int32>, Tensor<Float>) {
  return #tfop("IntOutputFloatOutput")
}

// Computes the reciprocal of x element-wise.
//
// I.e., \\(y = 1 / x\\).
@_inlineable @inline(__always)
public static func inv<T: Numeric>(
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Inv",
    x,
    T: T.self)
}

// Computes the gradient for the inverse of `x` wrt its input.
//
// Specifically, `grad = -dy * y*y`, where `y = 1/x`, and `dy`
// is the corresponding input gradient.
@_inlineable @inline(__always)
public static func invGrad<T: BinaryFloatingPoint>(
  y: Tensor<T>,
  dy: Tensor<T>
) -> Tensor<T> {
  return #tfop("InvGrad",
    y,
    dy,
    T: T.self)
}

// Flips all bits elementwise.
//
// The result will have exactly those bits set, that are not set in `x`. The
// computation is performed on the underlying representation of x.
@_inlineable @inline(__always)
public static func invert<T: BinaryInteger>(
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Invert",
    x,
    T: T.self)
}

// Computes the inverse permutation of a tensor.
//
// This operation computes the inverse of an index permutation. It takes a 1-D
// integer tensor `x`, which represents the indices of a zero-based array, and
// swaps each value with its index position. In other words, for an output tensor
// `y` and an input tensor `x`, this operation computes the following:
//
// `y[x[i]] = i for i in [0, 1, ..., len(x) - 1]`
//
// The values must include 0. There can be no duplicate values or negative values.
//
// For example:
//
// ```
// # tensor `x` is [3, 4, 0, 2, 1]
// invert_permutation(x) ==> [2, 4, 3, 0, 1]
// ```
//
// - Parameter x: 1-D.
//
// - Output y: 1-D.
@_inlineable @inline(__always)
public static func invertPermutation<T: BinaryInteger>(
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("InvertPermutation",
    x,
    T: T.self)
}

// Returns which elements of x are finite.
//
// @compatibility(numpy)
// Equivalent to np.isfinite
// @end_compatibility
@_inlineable @inline(__always)
public static func isFinite<T: BinaryFloatingPoint>(
  x: Tensor<T>
) -> Tensor<Bool> {
  return #tfop("IsFinite",
    x,
    T: T.self)
}

// Returns which elements of x are Inf.
//
// @compatibility(numpy)
// Equivalent to np.isinf
// @end_compatibility
@_inlineable @inline(__always)
public static func isInf<T: BinaryFloatingPoint>(
  x: Tensor<T>
) -> Tensor<Bool> {
  return #tfop("IsInf",
    x,
    T: T.self)
}

// Returns which elements of x are NaN.
//
// @compatibility(numpy)
// Equivalent to np.isnan
// @end_compatibility
@_inlineable @inline(__always)
public static func isNan<T: BinaryFloatingPoint>(
  x: Tensor<T>
) -> Tensor<Bool> {
  return #tfop("IsNan",
    x,
    T: T.self)
}

// Checks whether a tensor has been initialized.
//
// Outputs boolean scalar indicating whether the tensor has been initialized.
//
// - Parameter ref: Should be from a `Variable` node. May be uninitialized.
//
// - Attr dtype: The type of elements in the variable tensor.
@_inlineable @inline(__always)
public static func isVariableInitialized<Dtype: Numeric>(
  ref: Tensor<Dtype>
) -> Tensor<Bool> {
  return #tfop("IsVariableInitialized",
    ref,
    Dtype: Dtype.self)
}

// L2 Loss.
//
// Computes half the L2 norm of a tensor without the `sqrt`:
//
//     output = sum(t ** 2) / 2
//
// - Parameter t: Typically 2-D, but may have any dimensions.
//
// - Output output: 0-D.
@_inlineable @inline(__always)
public static func l2Loss<T: BinaryFloatingPoint>(
  t: Tensor<T>
) -> Tensor<T> {
  return #tfop("L2Loss",
    t,
    T: T.self)
}

// Local Response Normalization.
//
// The 4-D `input` tensor is treated as a 3-D array of 1-D vectors (along the last
// dimension), and each vector is normalized independently.  Within a given vector,
// each component is divided by the weighted, squared sum of inputs within
// `depth_radius`.  In detail,
//
//     sqr_sum[a, b, c, d] =
//         sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
//     output = input / (bias + alpha * sqr_sum) ** beta
//
// For details, see [Krizhevsky et al., ImageNet classification with deep
// convolutional neural networks (NIPS 2012)](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks).
//
// - Parameter input: 4-D.
//
// - Attrs:
//   - depth_radius: 0-D.  Half-width of the 1-D normalization window.
//   - bias: An offset (usually positive to avoid dividing by 0).
//   - alpha: A scale factor, usually positive.
//   - beta: An exponent.
@_inlineable @inline(__always)
public static func lRN<T: BinaryFloatingPoint>(
  input: Tensor<T>,
  depthRadius: Int = 5,
  bias: Double = 1,
  alpha: Double = 1,
  beta: Double = 0.5
) -> Tensor<T> {
  return #tfop("LRN",
    input,
    T: T.self,
    depth_radius: depthRadius,
    bias: bias,
    alpha: alpha,
    beta: beta)
}

// Gradients for Local Response Normalization.
//
// - Parameters:
//   - input_grads: 4-D with shape `[batch, height, width, channels]`.
//   - input_image: 4-D with shape `[batch, height, width, channels]`.
//   - output_image: 4-D with shape `[batch, height, width, channels]`.
//
// - Attrs:
//   - depth_radius: A depth radius.
//   - bias: An offset (usually > 0 to avoid dividing by 0).
//   - alpha: A scale factor, usually positive.
//   - beta: An exponent.
//
// - Output output: The gradients for LRN.
@_inlineable @inline(__always)
public static func lRNGrad<T: BinaryFloatingPoint>(
  inputGrads: Tensor<T>,
  inputImage: Tensor<T>,
  outputImage: Tensor<T>,
  depthRadius: Int = 5,
  bias: Double = 1,
  alpha: Double = 1,
  beta: Double = 0.5
) -> Tensor<T> {
  return #tfop("LRNGrad",
    inputGrads,
    inputImage,
    outputImage,
    T: T.self,
    depth_radius: depthRadius,
    bias: bias,
    alpha: alpha,
    beta: beta)
}

// Computes the LSTM cell forward propagation for 1 time step.
//
// This implementation uses 1 weight matrix and 1 bias vector, and there's an
// optional peephole connection.
//
// This kernel op implements the following mathematical equations:
//
// ```python
// xh = [x, h_prev]
// [i, f, ci, o] = xh * w + b
// f = f + forget_bias
//
// if not use_peephole:
//   wci = wcf = wco = 0
//
// i = sigmoid(cs_prev * wci + i)
// f = sigmoid(cs_prev * wcf + f)
// ci = tanh(ci)
//
// cs = ci .* i + cs_prev .* f
// cs = clip(cs, cell_clip)
//
// o = sigmoid(cs * wco + o)
// co = tanh(cs)
// h = co .* o
// ```
//
// - Parameters:
//   - x: The input to the LSTM cell, shape (batch_size, num_inputs).
//   - cs_prev: Value of the cell state at previous time step.
//   - h_prev: Output of the previous cell at previous time step.
//   - w: The weight matrix.
//   - wci: The weight matrix for input gate peephole connection.
//   - wcf: The weight matrix for forget gate peephole connection.
//   - wco: The weight matrix for output gate peephole connection.
//   - b: The bias vector.
//
// - Attrs:
//   - forget_bias: The forget gate bias.
//   - cell_clip: Value to clip the 'cs' value to.
//   - use_peephole: Whether to use peephole weights.
//
// - Outputs:
//   - i: The input gate.
//   - cs: The cell state before the tanh.
//   - f: The forget gate.
//   - o: The output gate.
//   - ci: The cell input.
//   - co: The cell after the tanh.
//   - h: The output h vector.
@_inlineable @inline(__always)
public static func lSTMBlockCell<T: BinaryFloatingPoint>(
  x: Tensor<T>,
  csPrev: Tensor<T>,
  hPrev: Tensor<T>,
  w: Tensor<T>,
  wci: Tensor<T>,
  wcf: Tensor<T>,
  wco: Tensor<T>,
  b: Tensor<T>,
  forgetBias: Double = 1,
  cellClip: Double = 3,
  usePeephole: Bool = false
) -> (Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>) {
  return #tfop("LSTMBlockCell",
    x,
    csPrev,
    hPrev,
    w,
    wci,
    wcf,
    wco,
    b,
    T: T.self,
    forget_bias: forgetBias,
    cell_clip: cellClip,
    use_peephole: usePeephole)
}

// Computes the LSTM cell backward propagation for 1 timestep.
//
// This implementation is to be used in conjunction of LSTMBlockCell.
//
// - Parameters:
//   - x: The input to the LSTM cell, shape (batch_size, num_inputs).
//   - cs_prev: The previous cell state.
//   - h_prev: The previous h state.
//   - w: The weight matrix.
//   - wci: The weight matrix for input gate peephole connection.
//   - wcf: The weight matrix for forget gate peephole connection.
//   - wco: The weight matrix for output gate peephole connection.
//   - b: The bias vector.
//   - i: The input gate.
//   - cs: The cell state before the tanh.
//   - f: The forget gate.
//   - o: The output gate.
//   - ci: The cell input.
//   - co: The cell after the tanh.
//   - cs_grad: The current gradient of cs.
//   - h_grad: The gradient of h vector.
//
// - Attr use_peephole: Whether the cell uses peephole connections.
//
// - Outputs:
//   - cs_prev_grad: The gradient of cs to be back-propped.
//   - dicfo: The derivative wrt to [i, cs, f, o].
//   - wci_grad: The gradient for wci to be back-propped.
//   - wcf_grad: The gradient for wcf to be back-propped.
//   - wco_grad: The gradient for wco to be back-propped.
@_inlineable @inline(__always)
public static func lSTMBlockCellGrad<T: BinaryFloatingPoint>(
  x: Tensor<T>,
  csPrev: Tensor<T>,
  hPrev: Tensor<T>,
  w: Tensor<T>,
  wci: Tensor<T>,
  wcf: Tensor<T>,
  wco: Tensor<T>,
  b: Tensor<T>,
  i: Tensor<T>,
  cs: Tensor<T>,
  f: Tensor<T>,
  o: Tensor<T>,
  ci: Tensor<T>,
  co: Tensor<T>,
  csGrad: Tensor<T>,
  hGrad: Tensor<T>,
  usePeephole: Bool
) -> (Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>) {
  return #tfop("LSTMBlockCellGrad",
    x,
    csPrev,
    hPrev,
    w,
    wci,
    wcf,
    wco,
    b,
    i,
    cs,
    f,
    o,
    ci,
    co,
    csGrad,
    hGrad,
    T: T.self,
    use_peephole: usePeephole)
}

// Generates labels for candidate sampling with a learned unigram distribution.
//
// See explanations of candidate sampling and the data formats at
// go/candidate-sampling.
//
// For each batch, this op picks a single set of sampled candidate labels.
//
// The advantages of sampling candidates per-batch are simplicity and the
// possibility of efficient dense matrix multiplication. The disadvantage is that
// the sampled candidates must be chosen independently of the context and of the
// true labels.
//
// - Parameter true_classes: A batch_size * num_true matrix, in which each row contains the
//   IDs of the num_true target_classes in the corresponding original label.
//
// - Attrs:
//   - num_true: Number of true labels per context.
//   - num_sampled: Number of candidates to randomly sample.
//   - unique: If unique is true, we sample with rejection, so that all sampled
//     candidates in a batch are unique. This requires some approximation to
//     estimate the post-rejection sampling probabilities.
//   - range_max: The sampler will sample integers from the interval [0, range_max).
//   - seed: If either seed or seed2 are set to be non-zero, the random number
//     generator is seeded by the given seed.  Otherwise, it is seeded by a
//     random seed.
//   - seed2: An second seed to avoid seed collision.
//
// - Outputs:
//   - sampled_candidates: A vector of length num_sampled, in which each element is
//     the ID of a sampled candidate.
//   - true_expected_count: A batch_size * num_true matrix, representing
//     the number of times each candidate is expected to occur in a batch
//     of sampled candidates. If unique=true, then this is a probability.
//   - sampled_expected_count: A vector of length num_sampled, for each sampled
//     candidate representing the number of times the candidate is expected
//     to occur in a batch of sampled candidates.  If unique=true, then this is a
//     probability.
@_inlineable @inline(__always)
public static func learnedUnigramCandidateSampler(
  trueClasses: Tensor<Int64>,
  numTrue: Int,
  numSampled: Int,
  unique: Bool,
  rangeMax: Int,
  seed: Int = 0,
  seed2: Int = 0
) -> (Tensor<Int64>, Tensor<Float>, Tensor<Float>) {
  return #tfop("LearnedUnigramCandidateSampler",
    trueClasses,
    num_true: numTrue,
    num_sampled: numSampled,
    unique: unique,
    range_max: rangeMax,
    seed: seed,
    seed2: seed2)
}

// Elementwise computes the bitwise left-shift of `x` and `y`.
//
// If `y` is negative, or greater than or equal to the width of `x` in bits the
// result is implementation defined.
@_inlineable @inline(__always)
public static func leftShift<T: BinaryInteger>(
  x: Tensor<T>,
  y: Tensor<T>
) -> Tensor<T> {
  return #tfop("LeftShift",
    x,
    y,
    T: T.self)
}

// Returns the truth value of (x < y) element-wise.
//
// *NOTE*: `Less` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@_inlineable @inline(__always)
public static func less<T: Numeric>(
  x: Tensor<T>,
  y: Tensor<T>
) -> Tensor<Bool> {
  return #tfop("Less",
    x,
    y,
    T: T.self)
}

// Returns the truth value of (x <= y) element-wise.
//
// *NOTE*: `LessEqual` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@_inlineable @inline(__always)
public static func lessEqual<T: Numeric>(
  x: Tensor<T>,
  y: Tensor<T>
) -> Tensor<Bool> {
  return #tfop("LessEqual",
    x,
    y,
    T: T.self)
}

// Computes the log of the absolute value of `Gamma(x)` element-wise.
@_inlineable @inline(__always)
public static func lgamma<T: BinaryFloatingPoint>(
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Lgamma",
    x,
    T: T.self)
}

// Generates values in an interval.
//
// A sequence of `num` evenly-spaced values are generated beginning at `start`.
// If `num > 1`, the values in the sequence increase by `stop - start / num - 1`,
// so that the last one is exactly `stop`.
//
// For example:
//
// ```
// tf.linspace(10.0, 12.0, 3, name="linspace") => [ 10.0  11.0  12.0]
// ```
//
// - Parameters:
//   - start: First entry in the range.
//   - stop: Last entry in the range.
//   - num: Number of values to generate.
//
// - Output output: 1-D. The generated values.
@_inlineable @inline(__always)
public static func linSpace<T: BinaryFloatingPoint, Tidx: BinaryInteger>(
  start: Tensor<T>,
  stop: Tensor<T>,
  num: Tensor<Tidx>
) -> Tensor<T> {
  return #tfop("LinSpace",
    start,
    stop,
    num,
    T: T.self,
    Tidx: Tidx.self)
}

// Computes the difference between two lists of numbers or strings.
//
// Given a list `x` and a list `y`, this operation returns a list `out` that
// represents all values that are in `x` but not in `y`. The returned list `out`
// is sorted in the same order that the numbers appear in `x` (duplicates are
// preserved). This operation also returns a list `idx` that represents the
// position of each `out` element in `x`. In other words:
//
// `out[i] = x[idx[i]] for i in [0, 1, ..., len(out) - 1]`
//
// For example, given this input:
//
// ```
// x = [1, 2, 3, 4, 5, 6]
// y = [1, 3, 5]
// ```
//
// This operation would return:
//
// ```
// out ==> [2, 4, 6]
// idx ==> [1, 3, 5]
// ```
//
// - Parameters:
//   - x: 1-D. Values to keep.
//   - y: 1-D. Values to remove.
//
// - Outputs:
//   - out: 1-D. Values present in `x` but not in `y`.
//   - idx: 1-D. Positions of `x` values preserved in `out`.
@_inlineable @inline(__always)
public static func listDiff<T: Numeric, Out_idx: BinaryInteger>(
  x: Tensor<T>,
  y: Tensor<T>
) -> (Tensor<T>, Tensor<Out_idx>) {
  return #tfop("ListDiff",
    x,
    y,
    T: T.self,
    Out_idx: Out_idx.self)
}

@_inlineable @inline(__always)
public static func listInput<T: Numeric>(
  a: [Tensor<T>]
) {
  return #tfop("ListInput",
    a,
    T: T.self)
}

@_inlineable @inline(__always)
public static func listOutput<T: Numeric>(
) -> [Tensor<T>] {
  return #tfop("ListOutput")
}

// Computes natural logarithm of x element-wise.
//
// I.e., \\(y = \log_e x\\).
@_inlineable @inline(__always)
public static func log<T: BinaryFloatingPoint>(
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Log",
    x,
    T: T.self)
}

// Computes natural logarithm of (1 + x) element-wise.
//
// I.e., \\(y = \log_e (1 + x)\\).
@_inlineable @inline(__always)
public static func log1p<T: BinaryFloatingPoint>(
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Log1p",
    x,
    T: T.self)
}

// Computes the sign and the log of the absolute value of the determinant of
//
// one or more square matrices.
//
// The input is a tensor of shape `[N, M, M]` whose inner-most 2 dimensions
// form square matrices. The outputs are two tensors containing the signs and
// absolute values of the log determinants for all N input submatrices
// `[..., :, :]` such that the determinant = sign*exp(log_abs_determinant).
// The log_abs_determinant is computed as det(P)*sum(log(diag(LU))) where LU
// is the LU decomposition of the input and P is the corresponding
// permutation matrix.
//
// - Parameter input: Shape is `[N, M, M]`.
//
// - Outputs:
//   - sign: The signs of the log determinants of the inputs. Shape is `[N]`.
//   - log_abs_determinant: The logs of the absolute values of the determinants
//     of the N input matrices.  Shape is `[N]`.
@_inlineable @inline(__always)
public static func logMatrixDeterminant<T: BinaryFloatingPoint>(
  input: Tensor<T>
) -> (Tensor<T>, Tensor<T>) {
  return #tfop("LogMatrixDeterminant",
    input,
    T: T.self)
}

// Computes log softmax activations.
//
// For each batch `i` and class `j` we have
//
//     logsoftmax[i, j] = logits[i, j] - log(sum(exp(logits[i])))
//
// - Parameter logits: 2-D with shape `[batch_size, num_classes]`.
//
// - Output logsoftmax: Same shape as `logits`.
@_inlineable @inline(__always)
public static func logSoftmax<T: BinaryFloatingPoint>(
  logits: Tensor<T>
) -> Tensor<T> {
  return #tfop("LogSoftmax",
    logits,
    T: T.self)
}

// Generates labels for candidate sampling with a log-uniform distribution.
//
// See explanations of candidate sampling and the data formats at
// go/candidate-sampling.
//
// For each batch, this op picks a single set of sampled candidate labels.
//
// The advantages of sampling candidates per-batch are simplicity and the
// possibility of efficient dense matrix multiplication. The disadvantage is that
// the sampled candidates must be chosen independently of the context and of the
// true labels.
//
// - Parameter true_classes: A batch_size * num_true matrix, in which each row contains the
//   IDs of the num_true target_classes in the corresponding original label.
//
// - Attrs:
//   - num_true: Number of true labels per context.
//   - num_sampled: Number of candidates to randomly sample.
//   - unique: If unique is true, we sample with rejection, so that all sampled
//     candidates in a batch are unique. This requires some approximation to
//     estimate the post-rejection sampling probabilities.
//   - range_max: The sampler will sample integers from the interval [0, range_max).
//   - seed: If either seed or seed2 are set to be non-zero, the random number
//     generator is seeded by the given seed.  Otherwise, it is seeded by a
//     random seed.
//   - seed2: An second seed to avoid seed collision.
//
// - Outputs:
//   - sampled_candidates: A vector of length num_sampled, in which each element is
//     the ID of a sampled candidate.
//   - true_expected_count: A batch_size * num_true matrix, representing
//     the number of times each candidate is expected to occur in a batch
//     of sampled candidates. If unique=true, then this is a probability.
//   - sampled_expected_count: A vector of length num_sampled, for each sampled
//     candidate representing the number of times the candidate is expected
//     to occur in a batch of sampled candidates.  If unique=true, then this is a
//     probability.
@_inlineable @inline(__always)
public static func logUniformCandidateSampler(
  trueClasses: Tensor<Int64>,
  numTrue: Int,
  numSampled: Int,
  unique: Bool,
  rangeMax: Int,
  seed: Int = 0,
  seed2: Int = 0
) -> (Tensor<Int64>, Tensor<Float>, Tensor<Float>) {
  return #tfop("LogUniformCandidateSampler",
    trueClasses,
    num_true: numTrue,
    num_sampled: numSampled,
    unique: unique,
    range_max: rangeMax,
    seed: seed,
    seed2: seed2)
}

// Returns the truth value of x AND y element-wise.
//
// *NOTE*: `LogicalAnd` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@_inlineable @inline(__always)
public static func logicalAnd(
  x: Tensor<Bool>,
  y: Tensor<Bool>
) -> Tensor<Bool> {
  return #tfop("LogicalAnd",
    x,
    y)
}

// Returns the truth value of NOT x element-wise.
@_inlineable @inline(__always)
public static func logicalNot(
  x: Tensor<Bool>
) -> Tensor<Bool> {
  return #tfop("LogicalNot",
    x)
}

// Returns the truth value of x OR y element-wise.
//
// *NOTE*: `LogicalOr` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@_inlineable @inline(__always)
public static func logicalOr(
  x: Tensor<Bool>,
  y: Tensor<Bool>
) -> Tensor<Bool> {
  return #tfop("LogicalOr",
    x,
    y)
}

// Forwards the input to the output.
//
// This operator represents the loop termination condition used by the
// "pivot" switches of a loop.
//
// - Parameter input: A boolean scalar, representing the branch predicate of the Switch op.
//
// - Output output: The same tensor as `input`.
@_inlineable @inline(__always)
public static func loopCond(
  input: Tensor<Bool>
) -> Tensor<Bool> {
  return #tfop("LoopCond",
    input)
}

// Op removes all elements in the underlying container.
@_inlineable @inline(__always)
public static func mapClear<Dtypes: Numeric>(
  capacity: Int = 0,
  memoryLimit: Int = 0,
  container: String,
  sharedName: String,
  typeDtypes: Dtypes.Type
) {
  return #tfop("MapClear",
    capacity: capacity,
    memory_limit: memoryLimit,
    container: container,
    shared_name: sharedName)
}

// Op returns the number of incomplete elements in the underlying container.
@_inlineable @inline(__always)
public static func mapIncompleteSize<Dtypes: Numeric>(
  capacity: Int = 0,
  memoryLimit: Int = 0,
  container: String,
  sharedName: String,
  typeDtypes: Dtypes.Type
) -> Tensor<Int32> {
  return #tfop("MapIncompleteSize",
    capacity: capacity,
    memory_limit: memoryLimit,
    container: container,
    shared_name: sharedName)
}

// Op peeks at the values at the specified key.  If the
//
// underlying container does not contain this key
// this op will block until it does.
@_inlineable @inline(__always)
public static func mapPeek<Dtypes: Numeric>(
  key: Tensor<Int64>,
  indices: Tensor<Int32>,
  capacity: Int = 0,
  memoryLimit: Int = 0,
  container: String,
  sharedName: String
) -> [Tensor<Dtypes>] {
  return #tfop("MapPeek",
    key,
    indices,
    capacity: capacity,
    memory_limit: memoryLimit,
    container: container,
    shared_name: sharedName)
}

// Op returns the number of elements in the underlying container.
@_inlineable @inline(__always)
public static func mapSize<Dtypes: Numeric>(
  capacity: Int = 0,
  memoryLimit: Int = 0,
  container: String,
  sharedName: String,
  typeDtypes: Dtypes.Type
) -> Tensor<Int32> {
  return #tfop("MapSize",
    capacity: capacity,
    memory_limit: memoryLimit,
    container: container,
    shared_name: sharedName)
}

// Stage (key, values) in the underlying container which behaves like a hashtable.
//
// - Parameters:
//   - key: int64
//   - values: a list of tensors
//     dtypes A list of data types that inserted values should adhere to.
//
// - Attrs:
//   - capacity: Maximum number of elements in the Staging Area. If > 0, inserts
//     on the container will block when the capacity is reached.
//   - container: If non-empty, this queue is placed in the given container. Otherwise,
//     a default container is used.
//   - shared_name: It is necessary to match this name to the matching Unstage Op.
@_inlineable @inline(__always)
public static func mapStage<Dtypes: Numeric, Fake_dtypes: Numeric>(
  key: Tensor<Int64>,
  indices: Tensor<Int32>,
  values: [Tensor<Fake_dtypes>],
  capacity: Int = 0,
  memoryLimit: Int = 0,
  container: String,
  sharedName: String,
  typeDtypes: Dtypes.Type
) {
  return #tfop("MapStage",
    key,
    indices,
    values,
    capacity: capacity,
    memory_limit: memoryLimit,
    container: container,
    shared_name: sharedName)
}

// Op removes and returns the values associated with the key
//
// from the underlying container.   If the underlying container
// does not contain this key, the op will block until it does.
@_inlineable @inline(__always)
public static func mapUnstage<Dtypes: Numeric>(
  key: Tensor<Int64>,
  indices: Tensor<Int32>,
  capacity: Int = 0,
  memoryLimit: Int = 0,
  container: String,
  sharedName: String
) -> [Tensor<Dtypes>] {
  return #tfop("MapUnstage",
    key,
    indices,
    capacity: capacity,
    memory_limit: memoryLimit,
    container: container,
    shared_name: sharedName)
}

// Op removes and returns a random (key, value)
//
// from the underlying container.   If the underlying container
// does not contain elements, the op will block until it does.
@_inlineable @inline(__always)
public static func mapUnstageNoKey<Dtypes: Numeric>(
  indices: Tensor<Int32>,
  capacity: Int = 0,
  memoryLimit: Int = 0,
  container: String,
  sharedName: String
) -> (Tensor<Int64>, [Tensor<Dtypes>]) {
  return #tfop("MapUnstageNoKey",
    indices,
    capacity: capacity,
    memory_limit: memoryLimit,
    container: container,
    shared_name: sharedName)
}

// Multiply the matrix "a" by the matrix "b".
//
// The inputs must be two-dimensional matrices and the inner dimension of
// "a" (after being transposed if transpose_a is true) must match the
// outer dimension of "b" (after being transposed if transposed_b is
// true).
//
// *Note*: The default kernel implementation for MatMul on GPUs uses
// cublas.
//
// - Attrs:
//   - transpose_a: If true, "a" is transposed before multiplication.
//   - transpose_b: If true, "b" is transposed before multiplication.
@_inlineable @inline(__always)
public static func matMul<T: Numeric>(
  a: Tensor<T>,
  b: Tensor<T>,
  transposeA: Bool = false,
  transposeB: Bool = false
) -> Tensor<T> {
  return #tfop("MatMul",
    a,
    b,
    T: T.self,
    transpose_a: transposeA,
    transpose_b: transposeB)
}

// Copy a tensor setting everything outside a central band in each innermost matrix
//
// to zero.
//
// The `band` part is computed as follows:
// Assume `input` has `k` dimensions `[I, J, K, ..., M, N]`, then the output is a
// tensor with the same shape where
//
// `band[i, j, k, ..., m, n] = in_band(m, n) * input[i, j, k, ..., m, n]`.
//
// The indicator function
//
// `in_band(m, n) = (num_lower < 0 || (m-n) <= num_lower)) &&
//                  (num_upper < 0 || (n-m) <= num_upper)`.
//
// For example:
//
// ```
// # if 'input' is [[ 0,  1,  2, 3]
//                  [-1,  0,  1, 2]
//                  [-2, -1,  0, 1]
//                  [-3, -2, -1, 0]],
//
// tf.matrix_band_part(input, 1, -1) ==> [[ 0,  1,  2, 3]
//                                        [-1,  0,  1, 2]
//                                        [ 0, -1,  0, 1]
//                                        [ 0,  0, -1, 0]],
//
// tf.matrix_band_part(input, 2, 1) ==> [[ 0,  1,  0, 0]
//                                       [-1,  0,  1, 0]
//                                       [-2, -1,  0, 1]
//                                       [ 0, -2, -1, 0]]
// ```
//
// Useful special cases:
//
// ```
//  tf.matrix_band_part(input, 0, -1) ==> Upper triangular part.
//  tf.matrix_band_part(input, -1, 0) ==> Lower triangular part.
//  tf.matrix_band_part(input, 0, 0) ==> Diagonal.
// ```
//
// - Parameters:
//   - input: Rank `k` tensor.
//   - num_lower: 0-D tensor. Number of subdiagonals to keep. If negative, keep entire
//     lower triangle.
//   - num_upper: 0-D tensor. Number of superdiagonals to keep. If negative, keep
//     entire upper triangle.
//
// - Output band: Rank `k` tensor of the same shape as input. The extracted banded tensor.
@_inlineable @inline(__always)
public static func matrixBandPart<T: Numeric, Tindex: BinaryInteger>(
  input: Tensor<T>,
  numLower: Tensor<Tindex>,
  numUpper: Tensor<Tindex>
) -> Tensor<T> {
  return #tfop("MatrixBandPart",
    input,
    numLower,
    numUpper,
    T: T.self,
    Tindex: Tindex.self)
}

// Computes the determinant of one or more square matrices.
//
// The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
// form square matrices. The output is a tensor containing the determinants
// for all input submatrices `[..., :, :]`.
//
// - Parameter input: Shape is `[..., M, M]`.
//
// - Output output: Shape is `[...]`.
@_inlineable @inline(__always)
public static func matrixDeterminant<T: BinaryFloatingPoint>(
  input: Tensor<T>
) -> Tensor<T> {
  return #tfop("MatrixDeterminant",
    input,
    T: T.self)
}

// Returns a batched diagonal tensor with a given batched diagonal values.
//
// Given a `diagonal`, this operation returns a tensor with the `diagonal` and
// everything else padded with zeros. The diagonal is computed as follows:
//
// Assume `diagonal` has `k` dimensions `[I, J, K, ..., N]`, then the output is a
// tensor of rank `k+1` with dimensions [I, J, K, ..., N, N]` where:
//
// `output[i, j, k, ..., m, n] = 1{m=n} * diagonal[i, j, k, ..., n]`.
//
// For example:
//
// ```
// # 'diagonal' is [[1, 2, 3, 4], [5, 6, 7, 8]]
//
// and diagonal.shape = (2, 4)
//
// tf.matrix_diag(diagonal) ==> [[[1, 0, 0, 0]
//                                      [0, 2, 0, 0]
//                                      [0, 0, 3, 0]
//                                      [0, 0, 0, 4]],
//                                     [[5, 0, 0, 0]
//                                      [0, 6, 0, 0]
//                                      [0, 0, 7, 0]
//                                      [0, 0, 0, 8]]]
//
// which has shape (2, 4, 4)
// ```
//
// - Parameter diagonal: Rank `k`, where `k >= 1`.
//
// - Output output: Rank `k+1`, with `output.shape = diagonal.shape + [diagonal.shape[-1]]`.
@_inlineable @inline(__always)
public static func matrixDiag<T: Numeric>(
  diagonal: Tensor<T>
) -> Tensor<T> {
  return #tfop("MatrixDiag",
    diagonal,
    T: T.self)
}

// Returns the batched diagonal part of a batched tensor.
//
// This operation returns a tensor with the `diagonal` part
// of the batched `input`. The `diagonal` part is computed as follows:
//
// Assume `input` has `k` dimensions `[I, J, K, ..., M, N]`, then the output is a
// tensor of rank `k - 1` with dimensions `[I, J, K, ..., min(M, N)]` where:
//
// `diagonal[i, j, k, ..., n] = input[i, j, k, ..., n, n]`.
//
// The input must be at least a matrix.
//
// For example:
//
// ```
// # 'input' is [[[1, 0, 0, 0]
//                [0, 2, 0, 0]
//                [0, 0, 3, 0]
//                [0, 0, 0, 4]],
//               [[5, 0, 0, 0]
//                [0, 6, 0, 0]
//                [0, 0, 7, 0]
//                [0, 0, 0, 8]]]
//
// and input.shape = (2, 4, 4)
//
// tf.matrix_diag_part(input) ==> [[1, 2, 3, 4], [5, 6, 7, 8]]
//
// which has shape (2, 4)
// ```
//
// - Parameter input: Rank `k` tensor where `k >= 2`.
//
// - Output diagonal: The extracted diagonal(s) having shape
//   `diagonal.shape = input.shape[:-2] + [min(input.shape[-2:])]`.
@_inlineable @inline(__always)
public static func matrixDiagPart<T: Numeric>(
  input: Tensor<T>
) -> Tensor<T> {
  return #tfop("MatrixDiagPart",
    input,
    T: T.self)
}

// Computes the matrix exponential of one or more square matrices:
//
// exp(A) = \sum_{n=0}^\infty A^n/n!
//
// The exponential is computed using a combination of the scaling and squaring
// method and the Pade approximation. Details can be founds in:
// Nicholas J. Higham, "The scaling and squaring method for the matrix exponential
// revisited," SIAM J. Matrix Anal. Applic., 26:1179-1193, 2005.
//
// The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
// form square matrices. The output is a tensor of the same shape as the input
// containing the exponential for all input submatrices `[..., :, :]`.
//
// - Parameter input: Shape is `[..., M, M]`.
//
// - Output output: Shape is `[..., M, M]`.
//
//   @compatibility(scipy)
//   Equivalent to scipy.linalg.expm
//   @end_compatibility
@_inlineable @inline(__always)
public static func matrixExponential<T: BinaryFloatingPoint>(
  input: Tensor<T>
) -> Tensor<T> {
  return #tfop("MatrixExponential",
    input,
    T: T.self)
}

// Computes the inverse of one or more square invertible matrices or their
//
// adjoints (conjugate transposes).
//
// The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
// form square matrices. The output is a tensor of the same shape as the input
// containing the inverse for all input submatrices `[..., :, :]`.
//
// The op uses LU decomposition with partial pivoting to compute the inverses.
//
// If a matrix is not invertible there is no guarantee what the op does. It
// may detect the condition and raise an exception or it may simply return a
// garbage result.
//
// - Parameter input: Shape is `[..., M, M]`.
//
// - Output output: Shape is `[..., M, M]`.
//
//   @compatibility(numpy)
//   Equivalent to np.linalg.inv
//   @end_compatibility
@_inlineable @inline(__always)
public static func matrixInverse<T: BinaryFloatingPoint>(
  input: Tensor<T>,
  adjoint: Bool = false
) -> Tensor<T> {
  return #tfop("MatrixInverse",
    input,
    T: T.self,
    adjoint: adjoint)
}

// Computes the matrix logarithm of one or more square matrices:
//
//
// log(exp(A)) = A
//
// This op is only defined for complex matrices. If A is positive-definite and
// real, then casting to a complex matrix, taking the logarithm and casting back
// to a real matrix will give the correct result.
//
// This function computes the matrix logarithm using the Schur-Parlett algorithm.
// Details of the algorithm can be found in Section 11.6.2 of:
// Nicholas J. Higham, Functions of Matrices: Theory and Computation, SIAM 2008.
// ISBN 978-0-898716-46-7.
//
// The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
// form square matrices. The output is a tensor of the same shape as the input
// containing the exponential for all input submatrices `[..., :, :]`.
//
// - Parameter input: Shape is `[..., M, M]`.
//
// - Output output: Shape is `[..., M, M]`.
//
//   @compatibility(scipy)
//   Equivalent to scipy.linalg.logm
//   @end_compatibility
@_inlineable @inline(__always)
public static func matrixLogarithm<T: Numeric>(
  input: Tensor<T>
) -> Tensor<T> {
  return #tfop("MatrixLogarithm",
    input,
    T: T.self)
}

// Returns a batched matrix tensor with new batched diagonal values.
//
// Given `input` and `diagonal`, this operation returns a tensor with the
// same shape and values as `input`, except for the main diagonal of the
// innermost matrices.  These will be overwritten by the values in `diagonal`.
//
// The output is computed as follows:
//
// Assume `input` has `k+1` dimensions `[I, J, K, ..., M, N]` and `diagonal` has
// `k` dimensions `[I, J, K, ..., min(M, N)]`.  Then the output is a
// tensor of rank `k+1` with dimensions `[I, J, K, ..., M, N]` where:
//
//   * `output[i, j, k, ..., m, n] = diagonal[i, j, k, ..., n]` for `m == n`.
//   * `output[i, j, k, ..., m, n] = input[i, j, k, ..., m, n]` for `m != n`.
//
// - Parameters:
//   - input: Rank `k+1`, where `k >= 1`.
//   - diagonal: Rank `k`, where `k >= 1`.
//
// - Output output: Rank `k+1`, with `output.shape = input.shape`.
@_inlineable @inline(__always)
public static func matrixSetDiag<T: Numeric>(
  input: Tensor<T>,
  diagonal: Tensor<T>
) -> Tensor<T> {
  return #tfop("MatrixSetDiag",
    input,
    diagonal,
    T: T.self)
}

// Solves systems of linear equations.
//
// `Matrix` is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
// form square matrices. `Rhs` is a tensor of shape `[..., M, K]`. The `output` is
// a tensor shape `[..., M, K]`.  If `adjoint` is `False` then each output matrix
// satisfies `matrix[..., :, :] * output[..., :, :] = rhs[..., :, :]`.
// If `adjoint` is `True` then each output matrix satisfies
// `adjoint(matrix[..., :, :]) * output[..., :, :] = rhs[..., :, :]`.
//
// - Parameters:
//   - matrix: Shape is `[..., M, M]`.
//   - rhs: Shape is `[..., M, K]`.
//
// - Attr adjoint: Boolean indicating whether to solve with `matrix` or its (block-wise)
//   adjoint.
//
// - Output output: Shape is `[..., M, K]`.
@_inlineable @inline(__always)
public static func matrixSolve<T: BinaryFloatingPoint>(
  matrix: Tensor<T>,
  rhs: Tensor<T>,
  adjoint: Bool = false
) -> Tensor<T> {
  return #tfop("MatrixSolve",
    matrix,
    rhs,
    T: T.self,
    adjoint: adjoint)
}

// Solves one or more linear least-squares problems.
//
// `matrix` is a tensor of shape `[..., M, N]` whose inner-most 2 dimensions
// form real or complex matrices of size `[M, N]`. `Rhs` is a tensor of the same
// type as `matrix` and shape `[..., M, K]`.
// The output is a tensor shape `[..., N, K]` where each output matrix solves
// each of the equations
// `matrix[..., :, :]` * `output[..., :, :]` = `rhs[..., :, :]`
// in the least squares sense.
//
// We use the following notation for (complex) matrix and right-hand sides
// in the batch:
//
// `matrix`=\\(A \in \mathbb{C}^{m \times n}\\),
// `rhs`=\\(B  \in \mathbb{C}^{m \times k}\\),
// `output`=\\(X  \in \mathbb{C}^{n \times k}\\),
// `l2_regularizer`=\\(\lambda \in \mathbb{R}\\).
//
// If `fast` is `True`, then the solution is computed by solving the normal
// equations using Cholesky decomposition. Specifically, if \\(m \ge n\\) then
// \\(X = (A^H A + \lambda I)^{-1} A^H B\\), which solves the least-squares
// problem \\(X = \mathrm{argmin}_{Z \in \Re^{n \times k} } ||A Z - B||_F^2 + \lambda ||Z||_F^2\\). 
// If \\(m \lt n\\) then `output` is computed as
// \\(X = A^H (A A^H + \lambda I)^{-1} B\\), which (for \\(\lambda = 0\\)) is the
// minimum-norm solution to the under-determined linear system, i.e.
// \\(X = \mathrm{argmin}_{Z \in \mathbb{C}^{n \times k} } ||Z||_F^2 \\),
// subject to \\(A Z = B\\). Notice that the fast path is only numerically stable
// when \\(A\\) is numerically full rank and has a condition number
// \\(\mathrm{cond}(A) \lt \frac{1}{\sqrt{\epsilon_{mach} } }\\) or \\(\lambda\\) is
// sufficiently large.
//
// If `fast` is `False` an algorithm based on the numerically robust complete
// orthogonal decomposition is used. This computes the minimum-norm
// least-squares solution, even when \\(A\\) is rank deficient. This path is
// typically 6-7 times slower than the fast path. If `fast` is `False` then
// `l2_regularizer` is ignored.
//
// - Parameters:
//   - matrix: Shape is `[..., M, N]`.
//   - rhs: Shape is `[..., M, K]`.
//   - l2_regularizer: Scalar tensor.
//
//     @compatibility(numpy)
//     Equivalent to np.linalg.lstsq
//     @end_compatibility
//
// - Output output: Shape is `[..., N, K]`.
@_inlineable @inline(__always)
public static func matrixSolveLs<T: BinaryFloatingPoint>(
  matrix: Tensor<T>,
  rhs: Tensor<T>,
  l2Regularizer: Tensor<Double>,
  fast: Bool = true
) -> Tensor<T> {
  return #tfop("MatrixSolveLs",
    matrix,
    rhs,
    l2Regularizer,
    T: T.self,
    fast: fast)
}

// Solves systems of linear equations with upper or lower triangular matrices by
//
// backsubstitution.
//
// `matrix` is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions form
// square matrices. If `lower` is `True` then the strictly upper triangular part
// of each inner-most matrix is assumed to be zero and not accessed.
// If `lower` is False then the strictly lower triangular part of each inner-most
// matrix is assumed to be zero and not accessed.
// `rhs` is a tensor of shape `[..., M, K]`.
//
// The output is a tensor of shape `[..., M, K]`. If `adjoint` is
// `True` then the innermost matrices in `output` satisfy matrix equations
// `matrix[..., :, :] * output[..., :, :] = rhs[..., :, :]`.
// If `adjoint` is `False` then the strictly then the  innermost matrices in
// `output` satisfy matrix equations
// `adjoint(matrix[..., i, k]) * output[..., k, j] = rhs[..., i, j]`.
//
// - Parameters:
//   - matrix: Shape is `[..., M, M]`.
//   - rhs: Shape is `[..., M, K]`.
//
// - Attrs:
//   - lower: Boolean indicating whether the innermost matrices in `matrix` are
//     lower or upper triangular.
//   - adjoint: Boolean indicating whether to solve with `matrix` or its (block-wise)
//              adjoint.
//
//     @compatibility(numpy)
//     Equivalent to scipy.linalg.solve_triangular
//     @end_compatibility
//
// - Output output: Shape is `[..., M, K]`.
@_inlineable @inline(__always)
public static func matrixTriangularSolve<T: BinaryFloatingPoint>(
  matrix: Tensor<T>,
  rhs: Tensor<T>,
  lower: Bool = true,
  adjoint: Bool = false
) -> Tensor<T> {
  return #tfop("MatrixTriangularSolve",
    matrix,
    rhs,
    T: T.self,
    lower: lower,
    adjoint: adjoint)
}

// Computes the maximum of elements across dimensions of a tensor.
//
// Reduces `input` along the dimensions given in `axis`. Unless
// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
// `axis`. If `keep_dims` is true, the reduced dimensions are
// retained with length 1.
//
// - Parameters:
//   - input: The tensor to reduce.
//   - reduction_indices: The dimensions to reduce. Must be in the range
//     `[-rank(input), rank(input))`.
//
// - Attr keep_dims: If true, retain reduced dimensions with length 1.
//
// - Output output: The reduced tensor.
@_inlineable @inline(__always)
public static func max<T: Numeric, Tidx: BinaryInteger>(
  input: Tensor<T>,
  reductionIndices: Tensor<Tidx>,
  keepDims: Bool = false
) -> Tensor<T> {
  return #tfop("Max",
    input,
    reductionIndices,
    T: T.self,
    Tidx: Tidx.self,
    keep_dims: keepDims)
}

// Performs max pooling on the input.
//
// - Parameter input: 4-D input to pool over.
//
// - Attrs:
//   - ksize: The size of the window for each dimension of the input tensor.
//   - strides: The stride of the sliding window for each dimension of the
//     input tensor.
//   - padding: The type of padding algorithm to use.
//   - data_format: Specify the data format of the input and output data. With the
//     default format "NHWC", the data is stored in the order of:
//         [batch, in_height, in_width, in_channels].
//     Alternatively, the format could be "NCHW", the data storage order of:
//         [batch, in_channels, in_height, in_width].
//
// - Output output: The max pooled output tensor.
@_inlineable @inline(__always)
public static func maxPool<T: Numeric>(
  input: Tensor<T>,
  ksize: [Int],
  strides: [Int],
  padding: Padding,
  dataFormat: DataFormat2 = .nhwc
) -> Tensor<T> {
  return #tfop("MaxPool",
    input,
    T: T.self,
    ksize: ksize,
    strides: strides,
    padding: padding.rawValue,
    data_format: dataFormat.rawValue)
}

// Performs 3D max pooling on the input.
//
// - Parameter input: Shape `[batch, depth, rows, cols, channels]` tensor to pool over.
//
// - Attrs:
//   - ksize: 1-D tensor of length 5. The size of the window for each dimension of
//     the input tensor. Must have `ksize[0] = ksize[4] = 1`.
//   - strides: 1-D tensor of length 5. The stride of the sliding window for each
//     dimension of `input`. Must have `strides[0] = strides[4] = 1`.
//   - padding: The type of padding algorithm to use.
//   - data_format: The data format of the input and output data. With the
//     default format "NDHWC", the data is stored in the order of:
//         [batch, in_depth, in_height, in_width, in_channels].
//     Alternatively, the format could be "NCDHW", the data storage order is:
//         [batch, in_channels, in_depth, in_height, in_width].
//
// - Output output: The max pooled output tensor.
@_inlineable @inline(__always)
public static func maxPool3D<T: BinaryFloatingPoint>(
  input: Tensor<T>,
  ksize: [Int],
  strides: [Int],
  padding: Padding,
  dataFormat: DataFormat1 = .ndhwc
) -> Tensor<T> {
  return #tfop("MaxPool3D",
    input,
    T: T.self,
    ksize: ksize,
    strides: strides,
    padding: padding.rawValue,
    data_format: dataFormat.rawValue)
}

// Computes gradients of max pooling function.
//
// - Parameters:
//   - orig_input: The original input tensor.
//   - orig_output: The original output tensor.
//   - grad: Output backprop of shape `[batch, depth, rows, cols, channels]`.
//
// - Attrs:
//   - ksize: 1-D tensor of length 5. The size of the window for each dimension of
//     the input tensor. Must have `ksize[0] = ksize[4] = 1`.
//   - strides: 1-D tensor of length 5. The stride of the sliding window for each
//     dimension of `input`. Must have `strides[0] = strides[4] = 1`.
//   - padding: The type of padding algorithm to use.
//   - data_format: The data format of the input and output data. With the
//     default format "NDHWC", the data is stored in the order of:
//         [batch, in_depth, in_height, in_width, in_channels].
//     Alternatively, the format could be "NCDHW", the data storage order is:
//         [batch, in_channels, in_depth, in_height, in_width].
@_inlineable @inline(__always)
public static func maxPool3DGrad<T: BinaryFloatingPoint, Tinput: BinaryFloatingPoint>(
  origInput: Tensor<Tinput>,
  origOutput: Tensor<Tinput>,
  grad: Tensor<T>,
  ksize: [Int],
  strides: [Int],
  padding: Padding,
  dataFormat: DataFormat1 = .ndhwc
) -> Tensor<T> {
  return #tfop("MaxPool3DGrad",
    origInput,
    origOutput,
    grad,
    T: T.self,
    Tinput: Tinput.self,
    ksize: ksize,
    strides: strides,
    padding: padding.rawValue,
    data_format: dataFormat.rawValue)
}

// Computes second-order gradients of the maxpooling function.
//
// - Parameters:
//   - orig_input: The original input tensor.
//   - orig_output: The original output tensor.
//   - grad: Output backprop of shape `[batch, depth, rows, cols, channels]`.
//
// - Attrs:
//   - ksize: 1-D tensor of length 5. The size of the window for each dimension of
//     the input tensor. Must have `ksize[0] = ksize[4] = 1`.
//   - strides: 1-D tensor of length 5. The stride of the sliding window for each
//     dimension of `input`. Must have `strides[0] = strides[4] = 1`.
//   - padding: The type of padding algorithm to use.
//   - data_format: The data format of the input and output data. With the
//     default format "NDHWC", the data is stored in the order of:
//         [batch, in_depth, in_height, in_width, in_channels].
//     Alternatively, the format could be "NCDHW", the data storage order is:
//         [batch, in_channels, in_depth, in_height, in_width].
//
// - Output output: Gradients of gradients w.r.t. the input to `max_pool`.
@_inlineable @inline(__always)
public static func maxPool3DGradGrad<T: BinaryFloatingPoint>(
  origInput: Tensor<T>,
  origOutput: Tensor<T>,
  grad: Tensor<T>,
  ksize: [Int],
  strides: [Int],
  padding: Padding,
  dataFormat: DataFormat1 = .ndhwc
) -> Tensor<T> {
  return #tfop("MaxPool3DGradGrad",
    origInput,
    origOutput,
    grad,
    T: T.self,
    ksize: ksize,
    strides: strides,
    padding: padding.rawValue,
    data_format: dataFormat.rawValue)
}

// Computes gradients of the maxpooling function.
//
// - Parameters:
//   - orig_input: The original input tensor.
//   - orig_output: The original output tensor.
//   - grad: 4-D.  Gradients w.r.t. the output of `max_pool`.
//
// - Attrs:
//   - ksize: The size of the window for each dimension of the input tensor.
//   - strides: The stride of the sliding window for each dimension of the
//     input tensor.
//   - padding: The type of padding algorithm to use.
//   - data_format: Specify the data format of the input and output data. With the
//     default format "NHWC", the data is stored in the order of:
//         [batch, in_height, in_width, in_channels].
//     Alternatively, the format could be "NCHW", the data storage order of:
//         [batch, in_channels, in_height, in_width].
//
// - Output output: Gradients w.r.t. the input to `max_pool`.
@_inlineable @inline(__always)
public static func maxPoolGrad<T: Numeric>(
  origInput: Tensor<T>,
  origOutput: Tensor<T>,
  grad: Tensor<T>,
  ksize: [Int],
  strides: [Int],
  padding: Padding,
  dataFormat: DataFormat = .nhwc
) -> Tensor<T> {
  return #tfop("MaxPoolGrad",
    origInput,
    origOutput,
    grad,
    T: T.self,
    ksize: ksize,
    strides: strides,
    padding: padding.rawValue,
    data_format: dataFormat.rawValue)
}

// Computes second-order gradients of the maxpooling function.
//
// - Parameters:
//   - orig_input: The original input tensor.
//   - orig_output: The original output tensor.
//   - grad: 4-D.  Gradients of gradients w.r.t. the input of `max_pool`.
//
// - Attrs:
//   - ksize: The size of the window for each dimension of the input tensor.
//   - strides: The stride of the sliding window for each dimension of the
//     input tensor.
//   - padding: The type of padding algorithm to use.
//   - data_format: Specify the data format of the input and output data. With the
//     default format "NHWC", the data is stored in the order of:
//         [batch, in_height, in_width, in_channels].
//     Alternatively, the format could be "NCHW", the data storage order of:
//         [batch, in_channels, in_height, in_width].
//
// - Output output: Gradients of gradients w.r.t. the input to `max_pool`.
@_inlineable @inline(__always)
public static func maxPoolGradGrad<T: Numeric>(
  origInput: Tensor<T>,
  origOutput: Tensor<T>,
  grad: Tensor<T>,
  ksize: [Int],
  strides: [Int],
  padding: Padding,
  dataFormat: DataFormat = .nhwc
) -> Tensor<T> {
  return #tfop("MaxPoolGradGrad",
    origInput,
    origOutput,
    grad,
    T: T.self,
    ksize: ksize,
    strides: strides,
    padding: padding.rawValue,
    data_format: dataFormat.rawValue)
}

// Computes second-order gradients of the maxpooling function.
//
// - Parameters:
//   - orig_input: The original input tensor.
//   - orig_output: The original output tensor.
//   - grad: 4-D.  Gradients of gradients w.r.t. the input of `max_pool`.
//   - ksize: The size of the window for each dimension of the input tensor.
//   - strides: The stride of the sliding window for each dimension of the
//     input tensor.
//
// - Attrs:
//   - padding: The type of padding algorithm to use.
//   - data_format: Specify the data format of the input and output data. With the
//     default format "NHWC", the data is stored in the order of:
//         [batch, in_height, in_width, in_channels].
//     Alternatively, the format could be "NCHW", the data storage order of:
//         [batch, in_channels, in_height, in_width].
//
// - Output output: Gradients of gradients w.r.t. the input to `max_pool`.
@_inlineable @inline(__always)
public static func maxPoolGradGradV2<T: Numeric>(
  origInput: Tensor<T>,
  origOutput: Tensor<T>,
  grad: Tensor<T>,
  ksize: Tensor<Int32>,
  strides: Tensor<Int32>,
  padding: Padding,
  dataFormat: DataFormat = .nhwc
) -> Tensor<T> {
  return #tfop("MaxPoolGradGradV2",
    origInput,
    origOutput,
    grad,
    ksize,
    strides,
    T: T.self,
    padding: padding.rawValue,
    data_format: dataFormat.rawValue)
}

// Computes second-order gradients of the maxpooling function.
//
// - Parameters:
//   - input: The original input.
//   - grad: 4-D with shape `[batch, height, width, channels]`.  Gradients w.r.t. the
//     input of `max_pool`.
//   - argmax: The indices of the maximum values chosen for each output of `max_pool`.
//
// - Attrs:
//   - ksize: The size of the window for each dimension of the input tensor.
//   - strides: The stride of the sliding window for each dimension of the
//     input tensor.
//   - padding: The type of padding algorithm to use.
//
// - Output output: Gradients of gradients w.r.t. the input of `max_pool`.
@_inlineable @inline(__always)
public static func maxPoolGradGradWithArgmax<Targmax: BinaryInteger, T: Numeric>(
  input: Tensor<T>,
  grad: Tensor<T>,
  argmax: Tensor<Targmax>,
  ksize: [Int],
  strides: [Int],
  padding: Padding
) -> Tensor<T> {
  return #tfop("MaxPoolGradGradWithArgmax",
    input,
    grad,
    argmax,
    Targmax: Targmax.self,
    T: T.self,
    ksize: ksize,
    strides: strides,
    padding: padding.rawValue)
}

// Computes gradients of the maxpooling function.
//
// - Parameters:
//   - orig_input: The original input tensor.
//   - orig_output: The original output tensor.
//   - grad: 4-D.  Gradients w.r.t. the output of `max_pool`.
//   - ksize: The size of the window for each dimension of the input tensor.
//   - strides: The stride of the sliding window for each dimension of the
//     input tensor.
//
// - Attrs:
//   - padding: The type of padding algorithm to use.
//   - data_format: Specify the data format of the input and output data. With the
//     default format "NHWC", the data is stored in the order of:
//         [batch, in_height, in_width, in_channels].
//     Alternatively, the format could be "NCHW", the data storage order of:
//         [batch, in_channels, in_height, in_width].
//
// - Output output: Gradients w.r.t. the input to `max_pool`.
@_inlineable @inline(__always)
public static func maxPoolGradV2<T: Numeric>(
  origInput: Tensor<T>,
  origOutput: Tensor<T>,
  grad: Tensor<T>,
  ksize: Tensor<Int32>,
  strides: Tensor<Int32>,
  padding: Padding,
  dataFormat: DataFormat = .nhwc
) -> Tensor<T> {
  return #tfop("MaxPoolGradV2",
    origInput,
    origOutput,
    grad,
    ksize,
    strides,
    T: T.self,
    padding: padding.rawValue,
    data_format: dataFormat.rawValue)
}

// Computes gradients of the maxpooling function.
//
// - Parameters:
//   - input: The original input.
//   - grad: 4-D with shape `[batch, height, width, channels]`.  Gradients w.r.t. the
//     output of `max_pool`.
//   - argmax: The indices of the maximum values chosen for each output of `max_pool`.
//
// - Attrs:
//   - ksize: The size of the window for each dimension of the input tensor.
//   - strides: The stride of the sliding window for each dimension of the
//     input tensor.
//   - padding: The type of padding algorithm to use.
//
// - Output output: Gradients w.r.t. the input of `max_pool`.
@_inlineable @inline(__always)
public static func maxPoolGradWithArgmax<Targmax: BinaryInteger, T: Numeric>(
  input: Tensor<T>,
  grad: Tensor<T>,
  argmax: Tensor<Targmax>,
  ksize: [Int],
  strides: [Int],
  padding: Padding
) -> Tensor<T> {
  return #tfop("MaxPoolGradWithArgmax",
    input,
    grad,
    argmax,
    Targmax: Targmax.self,
    T: T.self,
    ksize: ksize,
    strides: strides,
    padding: padding.rawValue)
}

// Performs max pooling on the input.
//
// - Parameters:
//   - input: 4-D input to pool over.
//   - ksize: The size of the window for each dimension of the input tensor.
//   - strides: The stride of the sliding window for each dimension of the
//     input tensor.
//
// - Attrs:
//   - padding: The type of padding algorithm to use.
//   - data_format: Specify the data format of the input and output data. With the
//     default format "NHWC", the data is stored in the order of:
//         [batch, in_height, in_width, in_channels].
//     Alternatively, the format could be "NCHW", the data storage order of:
//         [batch, in_channels, in_height, in_width].
//
// - Output output: The max pooled output tensor.
@_inlineable @inline(__always)
public static func maxPoolV2<T: Numeric>(
  input: Tensor<T>,
  ksize: Tensor<Int32>,
  strides: Tensor<Int32>,
  padding: Padding,
  dataFormat: DataFormat2 = .nhwc
) -> Tensor<T> {
  return #tfop("MaxPoolV2",
    input,
    ksize,
    strides,
    T: T.self,
    padding: padding.rawValue,
    data_format: dataFormat.rawValue)
}

// Performs max pooling on the input and outputs both max values and indices.
//
// The indices in `argmax` are flattened, so that a maximum value at position
// `[b, y, x, c]` becomes flattened index
// `((b * height + y) * width + x) * channels + c`.
//
// The indices returned are always in `[0, height) x [0, width)` before flattening,
// even if padding is involved and the mathematically correct answer is outside
// (either negative or too large).  This is a bug, but fixing it is difficult to do
// in a safe backwards compatible way, especially due to flattening.
//
// - Parameter input: 4-D with shape `[batch, height, width, channels]`.  Input to pool over.
//
// - Attrs:
//   - ksize: The size of the window for each dimension of the input tensor.
//   - strides: The stride of the sliding window for each dimension of the
//     input tensor.
//   - padding: The type of padding algorithm to use.
//
// - Outputs:
//   - output: The max pooled output tensor.
//   - argmax: 4-D.  The flattened indices of the max values chosen for each output.
@_inlineable @inline(__always)
public static func maxPoolWithArgmax<Targmax: BinaryInteger, T: Numeric>(
  input: Tensor<T>,
  ksize: [Int],
  strides: [Int],
  padding: Padding
) -> (Tensor<T>, Tensor<Targmax>) {
  return #tfop("MaxPoolWithArgmax",
    input,
    Targmax: Targmax.self,
    T: T.self,
    ksize: ksize,
    strides: strides,
    padding: padding.rawValue)
}

// Returns the max of x and y (i.e. x > y ? x : y) element-wise.
//
// *NOTE*: `Maximum` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@_inlineable @inline(__always)
public static func maximum<T: Numeric>(
  x: Tensor<T>,
  y: Tensor<T>
) -> Tensor<T> {
  return #tfop("Maximum",
    x,
    y,
    T: T.self)
}

// Computes the mean of elements across dimensions of a tensor.
//
// Reduces `input` along the dimensions given in `axis`. Unless
// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
// `axis`. If `keep_dims` is true, the reduced dimensions are
// retained with length 1.
//
// - Parameters:
//   - input: The tensor to reduce.
//   - reduction_indices: The dimensions to reduce. Must be in the range
//     `[-rank(input), rank(input))`.
//
// - Attr keep_dims: If true, retain reduced dimensions with length 1.
//
// - Output output: The reduced tensor.
@_inlineable @inline(__always)
public static func mean<T: Numeric, Tidx: BinaryInteger>(
  input: Tensor<T>,
  reductionIndices: Tensor<Tidx>,
  keepDims: Bool = false
) -> Tensor<T> {
  return #tfop("Mean",
    input,
    reductionIndices,
    T: T.self,
    Tidx: Tidx.self,
    keep_dims: keepDims)
}

// Forwards the value of an available tensor from `inputs` to `output`.
//
// `Merge` waits for at least one of the tensors in `inputs` to become available.
// It is usually combined with `Switch` to implement branching.
//
// `Merge` forwards the first tensor to become available to `output`, and sets
// `value_index` to its index in `inputs`.
//
// - Parameter inputs: The input tensors, exactly one of which will become available.
//
// - Outputs:
//   - output: Will be set to the available input tensor.
//   - value_index: The index of the chosen input tensor in `inputs`.
@_inlineable @inline(__always)
public static func merge<T: Numeric>(
  inputs: [Tensor<T>]
) -> (Tensor<T>, Tensor<Int32>) {
  return #tfop("Merge",
    inputs,
    T: T.self)
}

// Transforms a spectrogram into a form that's useful for speech recognition.
//
// Mel Frequency Cepstral Coefficients are a way of representing audio data that's
// been effective as an input feature for machine learning. They are created by
// taking the spectrum of a spectrogram (a 'cepstrum'), and discarding some of the
// higher frequencies that are less significant to the human ear. They have a long
// history in the speech recognition world, and https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
// is a good resource to learn more.
//
// - Parameters:
//   - spectrogram: Typically produced by the Spectrogram op, with magnitude_squared
//     set to true.
//   - sample_rate: How many samples per second the source audio used.
//
// - Attrs:
//   - upper_frequency_limit: The highest frequency to use when calculating the
//     ceptstrum.
//   - lower_frequency_limit: The lowest frequency to use when calculating the
//     ceptstrum.
//   - filterbank_channel_count: Resolution of the Mel bank used internally.
//   - dct_coefficient_count: How many output channels to produce per time slice.
@_inlineable @inline(__always)
public static func mfcc(
  spectrogram: Tensor<Float>,
  sampleRate: Tensor<Int32>,
  upperFrequencyLimit: Double = 4000,
  lowerFrequencyLimit: Double = 20,
  filterbankChannelCount: Int = 40,
  dctCoefficientCount: Int = 13
) -> Tensor<Float> {
  return #tfop("Mfcc",
    spectrogram,
    sampleRate,
    upper_frequency_limit: upperFrequencyLimit,
    lower_frequency_limit: lowerFrequencyLimit,
    filterbank_channel_count: filterbankChannelCount,
    dct_coefficient_count: dctCoefficientCount)
}

// Computes the minimum of elements across dimensions of a tensor.
//
// Reduces `input` along the dimensions given in `axis`. Unless
// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
// `axis`. If `keep_dims` is true, the reduced dimensions are
// retained with length 1.
//
// - Parameters:
//   - input: The tensor to reduce.
//   - reduction_indices: The dimensions to reduce. Must be in the range
//     `[-rank(input), rank(input))`.
//
// - Attr keep_dims: If true, retain reduced dimensions with length 1.
//
// - Output output: The reduced tensor.
@_inlineable @inline(__always)
public static func min<T: Numeric, Tidx: BinaryInteger>(
  input: Tensor<T>,
  reductionIndices: Tensor<Tidx>,
  keepDims: Bool = false
) -> Tensor<T> {
  return #tfop("Min",
    input,
    reductionIndices,
    T: T.self,
    Tidx: Tidx.self,
    keep_dims: keepDims)
}

// Returns the min of x and y (i.e. x < y ? x : y) element-wise.
//
// *NOTE*: `Minimum` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@_inlineable @inline(__always)
public static func minimum<T: Numeric>(
  x: Tensor<T>,
  y: Tensor<T>
) -> Tensor<T> {
  return #tfop("Minimum",
    x,
    y,
    T: T.self)
}

// Pads a tensor with mirrored values.
//
// This operation pads a `input` with mirrored values according to the `paddings`
// you specify. `paddings` is an integer tensor with shape `[n, 2]`, where n is
// the rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
// how many values to add before the contents of `input` in that dimension, and
// `paddings[D, 1]` indicates how many values to add after the contents of `input`
// in that dimension. Both `paddings[D, 0]` and `paddings[D, 1]` must be no greater
// than `input.dim_size(D)` (or `input.dim_size(D) - 1`) if `copy_border` is true
// (if false, respectively).
//
// The padded size of each dimension D of the output is:
//
// `paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`
//
// For example:
//
// ```
// # 't' is [[1, 2, 3], [4, 5, 6]].
// # 'paddings' is [[1, 1]], [2, 2]].
// # 'mode' is SYMMETRIC.
// # rank of 't' is 2.
// pad(t, paddings) ==> [[2, 1, 1, 2, 3, 3, 2]
//                       [2, 1, 1, 2, 3, 3, 2]
//                       [5, 4, 4, 5, 6, 6, 5]
//                       [5, 4, 4, 5, 6, 6, 5]]
// ```
//
// - Parameters:
//   - input: The input tensor to be padded.
//   - paddings: A two-column matrix specifying the padding sizes. The number of
//     rows must be the same as the rank of `input`.
//
// - Attr mode: Either `REFLECT` or `SYMMETRIC`. In reflect mode the padded regions
//   do not include the borders, while in symmetric mode the padded regions
//   do include the borders. For example, if `input` is `[1, 2, 3]` and `paddings`
//   is `[0, 2]`, then the output is `[1, 2, 3, 2, 1]` in reflect mode, and
//   it is `[1, 2, 3, 3, 2]` in symmetric mode.
//
// - Output output: The padded tensor.
@_inlineable @inline(__always)
public static func mirrorPad<T: Numeric, Tpaddings: BinaryInteger>(
  input: Tensor<T>,
  paddings: Tensor<Tpaddings>,
  mode: Mode3
) -> Tensor<T> {
  return #tfop("MirrorPad",
    input,
    paddings,
    T: T.self,
    Tpaddings: Tpaddings.self,
    mode: mode.rawValue)
}

// Gradient op for `MirrorPad` op. This op folds a mirror-padded tensor.
//
// This operation folds the padded areas of `input` by `MirrorPad` according to the
// `paddings` you specify. `paddings` must be the same as `paddings` argument
// given to the corresponding `MirrorPad` op.
//
// The folded size of each dimension D of the output is:
//
// `input.dim_size(D) - paddings(D, 0) - paddings(D, 1)`
//
// For example:
//
// ```
// # 't' is [[1, 2, 3], [4, 5, 6], [7, 8, 9]].
// # 'paddings' is [[0, 1]], [0, 1]].
// # 'mode' is SYMMETRIC.
// # rank of 't' is 2.
// pad(t, paddings) ==> [[ 1,  5]
//                       [11, 28]]
// ```
//
// - Parameters:
//   - input: The input tensor to be folded.
//   - paddings: A two-column matrix specifying the padding sizes. The number of
//     rows must be the same as the rank of `input`.
//
// - Attr mode: The mode used in the `MirrorPad` op.
//
// - Output output: The folded tensor.
@_inlineable @inline(__always)
public static func mirrorPadGrad<T: Numeric, Tpaddings: BinaryInteger>(
  input: Tensor<T>,
  paddings: Tensor<Tpaddings>,
  mode: Mode3
) -> Tensor<T> {
  return #tfop("MirrorPadGrad",
    input,
    paddings,
    T: T.self,
    Tpaddings: Tpaddings.self,
    mode: mode.rawValue)
}

@_inlineable @inline(__always)
public static func mixedStruct(
  nA: Int
) -> ([Tensor<Int32>], Tensor<Float>) {
  return #tfop("MixedStruct",
    n_a: nA)
}

// Returns element-wise remainder of division. This emulates C semantics in that
//
// the result here is consistent with a truncating divide. E.g.
// `tf.truncatediv(x, y) * y + truncate_mod(x, y) = x`.
//
// *NOTE*: `Mod` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@_inlineable @inline(__always)
public static func mod<T: Numeric>(
  x: Tensor<T>,
  y: Tensor<T>
) -> Tensor<T> {
  return #tfop("Mod",
    x,
    y,
    T: T.self)
}

// Returns x * y element-wise.
//
// *NOTE*: `Multiply` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@_inlineable @inline(__always)
public static func mul<T: Numeric>(
  x: Tensor<T>,
  y: Tensor<T>
) -> Tensor<T> {
  return #tfop("Mul",
    x,
    y,
    T: T.self)
}

// Draws samples from a multinomial distribution.
//
// - Parameters:
//   - logits: 2-D Tensor with shape `[batch_size, num_classes]`.  Each slice `[i, :]`
//     represents the unnormalized log probabilities for all classes.
//   - num_samples: 0-D.  Number of independent samples to draw for each row slice.
//
// - Attrs:
//   - seed: If either seed or seed2 is set to be non-zero, the internal random number
//     generator is seeded by the given seed.  Otherwise, a random seed is used.
//   - seed2: A second seed to avoid seed collision.
//
// - Output output: 2-D Tensor with shape `[batch_size, num_samples]`.  Each slice `[i, :]`
//   contains the drawn class labels with range `[0, num_classes)`.
@_inlineable @inline(__always)
public static func multinomial<T: Numeric, Output_dtype: BinaryInteger>(
  logits: Tensor<T>,
  numSamples: Tensor<Int32>,
  seed: Int = 0,
  seed2: Int = 0
) -> Tensor<Output_dtype> {
  return #tfop("Multinomial",
    logits,
    numSamples,
    T: T.self,
    Output_dtype: Output_dtype.self,
    seed: seed,
    seed2: seed2)
}

@_inlineable @inline(__always)
public static func nInPolymorphicTwice<T: Numeric>(
  a: [Tensor<T>],
  b: [Tensor<T>]
) {
  return #tfop("NInPolymorphicTwice",
    a,
    b,
    T: T.self)
}

@_inlineable @inline(__always)
public static func nInTwoTypeVariables<S: Numeric, T: Numeric>(
  a: [Tensor<S>],
  b: [Tensor<T>]
) {
  return #tfop("NInTwoTypeVariables",
    a,
    b,
    S: S.self,
    T: T.self)
}

@_inlineable @inline(__always)
public static func nIntsIn(
  a: [Tensor<Int32>]
) {
  return #tfop("NIntsIn",
    a)
}

@_inlineable @inline(__always)
public static func nIntsOut(
  n: Int
) -> [Tensor<Int32>] {
  return #tfop("NIntsOut",
    N: n)
}

@_inlineable @inline(__always)
public static func nIntsOutDefault(
  n: Int = 3
) -> [Tensor<Int32>] {
  return #tfop("NIntsOutDefault",
    N: n)
}

@_inlineable @inline(__always)
public static func nPolymorphicIn<T: Numeric>(
  a: [Tensor<T>]
) {
  return #tfop("NPolymorphicIn",
    a,
    T: T.self)
}

@_inlineable @inline(__always)
public static func nPolymorphicOut<T: Numeric>(
  n: Int
) -> [Tensor<T>] {
  return #tfop("NPolymorphicOut",
    T: T.self,
    N: n)
}

@_inlineable @inline(__always)
public static func nPolymorphicOutDefault<T: Numeric>(
  n: Int = 2
) -> [Tensor<T>] {
  return #tfop("NPolymorphicOutDefault",
    T: T.self,
    N: n)
}

@_inlineable @inline(__always)
public static func nPolymorphicRestrictIn<T: Numeric>(
  a: [Tensor<T>]
) {
  return #tfop("NPolymorphicRestrictIn",
    a,
    T: T.self)
}

@_inlineable @inline(__always)
public static func nPolymorphicRestrictOut<T: Numeric>(
  n: Int
) -> [Tensor<T>] {
  return #tfop("NPolymorphicRestrictOut",
    T: T.self,
    N: n)
}

// Computes numerical negative value element-wise.
//
// I.e., \\(y = -x\\).
@_inlineable @inline(__always)
public static func neg<T: Numeric>(
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Neg",
    x,
    T: T.self)
}

// Training via negative sampling.
//
// - Parameters:
//   - w_in: input word embedding.
//   - w_out: output word embedding.
//   - examples: A vector of word ids.
//   - labels: A vector of word ids.
//
// - Attrs:
//   - vocab_count: Count of words in the vocabulary.
//   - num_negative_samples: Number of negative samples per example.
@_inlineable @inline(__always)
public static func negTrain(
  wIn: Tensor<Float>,
  wOut: Tensor<Float>,
  examples: Tensor<Int32>,
  labels: Tensor<Int32>,
  lr: Tensor<Float>,
  vocabCount: [Int],
  numNegativeSamples: Int
) {
  return #tfop("NegTrain",
    wIn,
    wOut,
    examples,
    labels,
    lr,
    vocab_count: vocabCount,
    num_negative_samples: numNegativeSamples)
}

// Makes its input available to the next iteration.
//
// - Parameter data: The tensor to be made available to the next iteration.
//
// - Output output: The same tensor as `data`.
@_inlineable @inline(__always)
public static func nextIteration<T: Numeric>(
  data: Tensor<T>
) -> Tensor<T> {
  return #tfop("NextIteration",
    data,
    T: T.self)
}

// Does nothing. Only useful as a placeholder for control edges.
@_inlineable @inline(__always)
public static func noOp(
) {
  return #tfop("NoOp")
}

// Greedily selects a subset of bounding boxes in descending order of score,
//
// pruning away boxes that have high intersection-over-union (IOU) overlap
// with previously selected boxes.  Bounding boxes are supplied as
// [y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any
// diagonal pair of box corners and the coordinates can be provided as normalized
// (i.e., lying in the interval [0, 1]) or absolute.  Note that this algorithm
// is agnostic to where the origin is in the coordinate system.  Note that this
// algorithm is invariant to orthogonal transformations and translations
// of the coordinate system; thus translating or reflections of the coordinate
// system result in the same boxes being selected by the algorithm.
// The output of this operation is a set of integers indexing into the input
// collection of bounding boxes representing the selected boxes.  The bounding
// box coordinates corresponding to the selected indices can then be obtained
// using the `tf.gather operation`.  For example:
//   selected_indices = tf.image.non_max_suppression(
//       boxes, scores, max_output_size, iou_threshold)
//   selected_boxes = tf.gather(boxes, selected_indices)
//
// - Parameters:
//   - boxes: A 2-D float tensor of shape `[num_boxes, 4]`.
//   - scores: A 1-D float tensor of shape `[num_boxes]` representing a single
//     score corresponding to each box (each row of boxes).
//   - max_output_size: A scalar integer tensor representing the maximum number of
//     boxes to be selected by non max suppression.
//
// - Attr iou_threshold: A float representing the threshold for deciding whether boxes
//   overlap too much with respect to IOU.
//
// - Output selected_indices: A 1-D integer tensor of shape `[M]` representing the selected
//   indices from the boxes tensor, where `M <= max_output_size`.
@_inlineable @inline(__always)
public static func nonMaxSuppression(
  boxes: Tensor<Float>,
  scores: Tensor<Float>,
  maxOutputSize: Tensor<Int32>,
  iouThreshold: Double = 0.5
) -> Tensor<Int32> {
  return #tfop("NonMaxSuppression",
    boxes,
    scores,
    maxOutputSize,
    iou_threshold: iouThreshold)
}

// Greedily selects a subset of bounding boxes in descending order of score,
//
// pruning away boxes that have high intersection-over-union (IOU) overlap
// with previously selected boxes.  Bounding boxes are supplied as
// [y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any
// diagonal pair of box corners and the coordinates can be provided as normalized
// (i.e., lying in the interval [0, 1]) or absolute.  Note that this algorithm
// is agnostic to where the origin is in the coordinate system.  Note that this
// algorithm is invariant to orthogonal transformations and translations
// of the coordinate system; thus translating or reflections of the coordinate
// system result in the same boxes being selected by the algorithm.
//
// The output of this operation is a set of integers indexing into the input
// collection of bounding boxes representing the selected boxes.  The bounding
// box coordinates corresponding to the selected indices can then be obtained
// using the `tf.gather operation`.  For example:
//
//   selected_indices = tf.image.non_max_suppression_v2(
//       boxes, scores, max_output_size, iou_threshold)
//   selected_boxes = tf.gather(boxes, selected_indices)
//
// - Parameters:
//   - boxes: A 2-D float tensor of shape `[num_boxes, 4]`.
//   - scores: A 1-D float tensor of shape `[num_boxes]` representing a single
//     score corresponding to each box (each row of boxes).
//   - max_output_size: A scalar integer tensor representing the maximum number of
//     boxes to be selected by non max suppression.
//   - iou_threshold: A 0-D float tensor representing the threshold for deciding whether
//     boxes overlap too much with respect to IOU.
//
// - Output selected_indices: A 1-D integer tensor of shape `[M]` representing the selected
//   indices from the boxes tensor, where `M <= max_output_size`.
@_inlineable @inline(__always)
public static func nonMaxSuppressionV2(
  boxes: Tensor<Float>,
  scores: Tensor<Float>,
  maxOutputSize: Tensor<Int32>,
  iouThreshold: Tensor<Float>
) -> Tensor<Int32> {
  return #tfop("NonMaxSuppressionV2",
    boxes,
    scores,
    maxOutputSize,
    iouThreshold)
}

@_inlineable @inline(__always)
public static func none(
) {
  return #tfop("None")
}

// Returns the truth value of (x != y) element-wise.
//
// *NOTE*: `NotEqual` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@_inlineable @inline(__always)
public static func notEqual<T: Numeric>(
  x: Tensor<T>,
  y: Tensor<T>
) -> Tensor<Bool> {
  return #tfop("NotEqual",
    x,
    y,
    T: T.self)
}

// Finds values of the `n`-th order statistic for the last dimension.
//
// If the input is a vector (rank-1), finds the entries which is the nth-smallest
// value in the vector and outputs their values as scalar tensor.
//
// For matrices (resp. higher rank input), computes the entries which is the
// nth-smallest value in each row (resp. vector along the last dimension). Thus,
//
//     values.shape = input.shape[:-1]
//
// - Parameters:
//   - input: 1-D or higher with last dimension at least `n+1`.
//   - n: 0-D. Position of sorted vector to select along the last dimension (along
//     each row for matrices). Valid range of n is `[0, input.shape[:-1])`
//
// - Attr reverse: When set to True, find the nth-largest value in the vector and vice
//   versa.
//
// - Output values: The `n`-th order statistic along each last dimensional slice.
@_inlineable @inline(__always)
public static func nthElement<T: Numeric>(
  input: Tensor<T>,
  n: Tensor<Int32>,
  reverse: Bool = false
) -> Tensor<T> {
  return #tfop("NthElement",
    input,
    n,
    T: T.self,
    reverse: reverse)
}

@_inlineable @inline(__always)
public static func old(
) {
  return #tfop("Old")
}

// Returns a one-hot tensor.
//
// The locations represented by indices in `indices` take value `on_value`,
// while all other locations take value `off_value`.
//
// If the input `indices` is rank `N`, the output will have rank `N+1`,
// The new axis is created at dimension `axis` (default: the new axis is
// appended at the end).
//
// If `indices` is a scalar the output shape will be a vector of length `depth`.
//
// If `indices` is a vector of length `features`, the output shape will be:
// ```
//   features x depth if axis == -1
//   depth x features if axis == 0
// ```
//
// If `indices` is a matrix (batch) with shape `[batch, features]`,
// the output shape will be:
// ```
//   batch x features x depth if axis == -1
//   batch x depth x features if axis == 1
//   depth x batch x features if axis == 0
// ```
//
//
// Examples
// =========
//
// Suppose that
//
// ```
//   indices = [0, 2, -1, 1]
//   depth = 3
//   on_value = 5.0
//   off_value = 0.0
//   axis = -1
// ```
//
// Then output is `[4 x 3]`:
//
//     ```output =
//       [5.0 0.0 0.0]  // one_hot(0)
//       [0.0 0.0 5.0]  // one_hot(2)
//       [0.0 0.0 0.0]  // one_hot(-1)
//       [0.0 5.0 0.0]  // one_hot(1)
//     ```
//
// Suppose that
//
// ```
//   indices = [0, 2, -1, 1]
//   depth = 3
//   on_value = 0.0
//   off_value = 3.0
//   axis = 0
// ```
//
// Then output is `[3 x 4]`:
//
//     ```output =
//       [0.0 3.0 3.0 3.0]
//       [3.0 3.0 3.0 0.0]
//       [3.0 3.0 3.0 3.0]
//       [3.0 0.0 3.0 3.0]
//     //  ^                one_hot(0)
//     //      ^            one_hot(2)
//     //          ^        one_hot(-1)
//     //              ^    one_hot(1)
//     ```
// Suppose that
//
// ```
//   indices = [[0, 2], [1, -1]]
//   depth = 3
//   on_value = 1.0
//   off_value = 0.0
//   axis = -1
// ```
//
// Then output is `[2 x 2 x 3]`:
//
//     ```output =
//       [
//         [1.0, 0.0, 0.0]  // one_hot(0)
//         [0.0, 0.0, 1.0]  // one_hot(2)
//       ][
//         [0.0, 1.0, 0.0]  // one_hot(1)
//         [0.0, 0.0, 0.0]  // one_hot(-1)
//       ]```
//
// - Parameters:
//   - indices: A tensor of indices.
//   - depth: A scalar defining the depth of the one hot dimension.
//   - on_value: A scalar defining the value to fill in output when `indices[j] = i`.
//   - off_value: A scalar defining the value to fill in output when `indices[j] != i`.
//
// - Attr axis: The axis to fill (default: -1, a new inner-most axis).
//
// - Output output: The one-hot tensor.
@_inlineable @inline(__always)
public static func oneHot<T: Numeric, Ti: BinaryInteger>(
  indices: Tensor<Ti>,
  depth: Tensor<Int32>,
  onValue: Tensor<T>,
  offValue: Tensor<T>,
  axis: Int = -1
) -> Tensor<T> {
  return #tfop("OneHot",
    indices,
    depth,
    onValue,
    offValue,
    T: T.self,
    Ti: Ti.self,
    axis: axis)
}

// Returns a tensor of ones with the same shape and type as x.
//
// - Parameter x: a tensor of type T.
//
// - Output y: a tensor of the same shape and type as x but filled with ones.
@_inlineable @inline(__always)
public static func onesLike<T: Numeric>(
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("OnesLike",
    x,
    T: T.self)
}

@_inlineable @inline(__always)
public static func opWithDefaultAttr(
  defaultFloat: Double = 123
) -> Tensor<Int32> {
  return #tfop("OpWithDefaultAttr",
    default_float: defaultFloat)
}

@_inlineable @inline(__always)
public static func opWithFutureDefaultAttr(
) {
  return #tfop("OpWithFutureDefaultAttr")
}

// Op removes all elements in the underlying container.
@_inlineable @inline(__always)
public static func orderedMapClear<Dtypes: Numeric>(
  capacity: Int = 0,
  memoryLimit: Int = 0,
  container: String,
  sharedName: String,
  typeDtypes: Dtypes.Type
) {
  return #tfop("OrderedMapClear",
    capacity: capacity,
    memory_limit: memoryLimit,
    container: container,
    shared_name: sharedName)
}

// Op returns the number of incomplete elements in the underlying container.
@_inlineable @inline(__always)
public static func orderedMapIncompleteSize<Dtypes: Numeric>(
  capacity: Int = 0,
  memoryLimit: Int = 0,
  container: String,
  sharedName: String,
  typeDtypes: Dtypes.Type
) -> Tensor<Int32> {
  return #tfop("OrderedMapIncompleteSize",
    capacity: capacity,
    memory_limit: memoryLimit,
    container: container,
    shared_name: sharedName)
}

// Op peeks at the values at the specified key.  If the
//
// underlying container does not contain this key
// this op will block until it does.   This Op is optimized for
// performance.
@_inlineable @inline(__always)
public static func orderedMapPeek<Dtypes: Numeric>(
  key: Tensor<Int64>,
  indices: Tensor<Int32>,
  capacity: Int = 0,
  memoryLimit: Int = 0,
  container: String,
  sharedName: String
) -> [Tensor<Dtypes>] {
  return #tfop("OrderedMapPeek",
    key,
    indices,
    capacity: capacity,
    memory_limit: memoryLimit,
    container: container,
    shared_name: sharedName)
}

// Op returns the number of elements in the underlying container.
@_inlineable @inline(__always)
public static func orderedMapSize<Dtypes: Numeric>(
  capacity: Int = 0,
  memoryLimit: Int = 0,
  container: String,
  sharedName: String,
  typeDtypes: Dtypes.Type
) -> Tensor<Int32> {
  return #tfop("OrderedMapSize",
    capacity: capacity,
    memory_limit: memoryLimit,
    container: container,
    shared_name: sharedName)
}

// Stage (key, values) in the underlying container which behaves like a ordered
//
// associative container.   Elements are ordered by key.
//
// - Parameters:
//   - key: int64
//   - values: a list of tensors
//     dtypes A list of data types that inserted values should adhere to.
//
// - Attrs:
//   - capacity: Maximum number of elements in the Staging Area. If > 0, inserts
//     on the container will block when the capacity is reached.
//   - container: If non-empty, this queue is placed in the given container. Otherwise,
//     a default container is used.
//   - shared_name: It is necessary to match this name to the matching Unstage Op.
@_inlineable @inline(__always)
public static func orderedMapStage<Dtypes: Numeric, Fake_dtypes: Numeric>(
  key: Tensor<Int64>,
  indices: Tensor<Int32>,
  values: [Tensor<Fake_dtypes>],
  capacity: Int = 0,
  memoryLimit: Int = 0,
  container: String,
  sharedName: String,
  typeDtypes: Dtypes.Type
) {
  return #tfop("OrderedMapStage",
    key,
    indices,
    values,
    capacity: capacity,
    memory_limit: memoryLimit,
    container: container,
    shared_name: sharedName)
}

// Op removes and returns the values associated with the key
//
// from the underlying container.   If the underlying container
// does not contain this key, the op will block until it does.
@_inlineable @inline(__always)
public static func orderedMapUnstage<Dtypes: Numeric>(
  key: Tensor<Int64>,
  indices: Tensor<Int32>,
  capacity: Int = 0,
  memoryLimit: Int = 0,
  container: String,
  sharedName: String
) -> [Tensor<Dtypes>] {
  return #tfop("OrderedMapUnstage",
    key,
    indices,
    capacity: capacity,
    memory_limit: memoryLimit,
    container: container,
    shared_name: sharedName)
}

// Op removes and returns the (key, value) element with the smallest
//
// key from the underlying container.   If the underlying container
// does not contain elements, the op will block until it does.
@_inlineable @inline(__always)
public static func orderedMapUnstageNoKey<Dtypes: Numeric>(
  indices: Tensor<Int32>,
  capacity: Int = 0,
  memoryLimit: Int = 0,
  container: String,
  sharedName: String
) -> (Tensor<Int64>, [Tensor<Dtypes>]) {
  return #tfop("OrderedMapUnstageNoKey",
    indices,
    capacity: capacity,
    memory_limit: memoryLimit,
    container: container,
    shared_name: sharedName)
}

@_inlineable @inline(__always)
public static func outT<T: Numeric>(
) -> Tensor<T> {
  return #tfop("OutT",
    T: T.self)
}

@_inlineable @inline(__always)
public static func outTypeList<T: Numeric>(
) -> [Tensor<T>] {
  return #tfop("OutTypeList")
}

@_inlineable @inline(__always)
public static func outTypeListRestrict<T: Numeric>(
) -> [Tensor<T>] {
  return #tfop("OutTypeListRestrict")
}

// Packs a list of `N` rank-`R` tensors into one rank-`(R+1)` tensor.
//
// Packs the `N` tensors in `values` into a tensor with rank one higher than each
// tensor in `values`, by packing them along the `axis` dimension.
// Given a list of tensors of shape `(A, B, C)`;
//
// if `axis == 0` then the `output` tensor will have the shape `(N, A, B, C)`.
// if `axis == 1` then the `output` tensor will have the shape `(A, N, B, C)`.
// Etc.
//
// For example:
//
// ```
// # 'x' is [1, 4]
// # 'y' is [2, 5]
// # 'z' is [3, 6]
// pack([x, y, z]) => [[1, 4], [2, 5], [3, 6]]  # Pack along first dim.
// pack([x, y, z], axis=1) => [[1, 2, 3], [4, 5, 6]]
// ```
//
// This is the opposite of `unpack`.
//
// - Parameter values: Must be of same shape and type.
//
// - Attr axis: Dimension along which to pack.  Negative values wrap around, so the
//   valid range is `[-(R+1), R+1)`.
//
// - Output output: The packed tensor.
@_inlineable @inline(__always)
public static func pack<T: Numeric>(
  values: [Tensor<T>],
  axis: Int = 0
) -> Tensor<T> {
  return #tfop("Pack",
    values,
    T: T.self,
    axis: axis)
}

// Pads a tensor with zeros.
//
// This operation pads a `input` with zeros according to the `paddings` you
// specify. `paddings` is an integer tensor with shape `[Dn, 2]`, where n is the
// rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
// how many zeros to add before the contents of `input` in that dimension, and
// `paddings[D, 1]` indicates how many zeros to add after the contents of `input`
// in that dimension.
//
// The padded size of each dimension D of the output is:
//
// `paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`
//
// For example:
//
// ```
// # 't' is [[1, 1], [2, 2]]
// # 'paddings' is [[1, 1], [2, 2]]
// # rank of 't' is 2
// pad(t, paddings) ==> [[0, 0, 0, 0, 0, 0]
//                       [0, 0, 1, 1, 0, 0]
//                       [0, 0, 2, 2, 0, 0]
//                       [0, 0, 0, 0, 0, 0]]
// ```
//
@_inlineable @inline(__always)
public static func pad<T: Numeric, Tpaddings: BinaryInteger>(
  input: Tensor<T>,
  paddings: Tensor<Tpaddings>
) -> Tensor<T> {
  return #tfop("Pad",
    input,
    paddings,
    T: T.self,
    Tpaddings: Tpaddings.self)
}

// Pads a tensor.
//
// This operation pads `input` according to the `paddings` and `constant_values`
// you specify. `paddings` is an integer tensor with shape `[Dn, 2]`, where n is
// the rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
// how many padding values to add before the contents of `input` in that dimension,
// and `paddings[D, 1]` indicates how many padding values to add after the contents
// of `input` in that dimension. `constant_values` is a scalar tensor of the same
// type as `input` that indicates the value to use for padding `input`.
//
// The padded size of each dimension D of the output is:
//
// `paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`
//
// For example:
//
// ```
// # 't' is [[1, 1], [2, 2]]
// # 'paddings' is [[1, 1], [2, 2]]
// # 'constant_values' is 0
// # rank of 't' is 2
// pad(t, paddings) ==> [[0, 0, 0, 0, 0, 0]
//                       [0, 0, 1, 1, 0, 0]
//                       [0, 0, 2, 2, 0, 0]
//                       [0, 0, 0, 0, 0, 0]]
// ```
@_inlineable @inline(__always)
public static func padV2<T: Numeric, Tpaddings: BinaryInteger>(
  input: Tensor<T>,
  paddings: Tensor<Tpaddings>,
  constantValues: Tensor<T>
) -> Tensor<T> {
  return #tfop("PadV2",
    input,
    paddings,
    constantValues,
    T: T.self,
    Tpaddings: Tpaddings.self)
}

// Interleave the values from the `data` tensors into a single tensor.
//
// Builds a merged tensor such that
//
// ```python
//     merged[indices[m][i, ..., j], ...] = data[m][i, ..., j, ...]
// ```
//
// For example, if each `indices[m]` is scalar or vector, we have
//
// ```python
//     # Scalar indices:
//     merged[indices[m], ...] = data[m][...]
//
//     # Vector indices:
//     merged[indices[m][i], ...] = data[m][i, ...]
// ```
//
// Each `data[i].shape` must start with the corresponding `indices[i].shape`,
// and the rest of `data[i].shape` must be constant w.r.t. `i`.  That is, we
// must have `data[i].shape = indices[i].shape + constant`.  In terms of this
// `constant`, the output shape is
//
//     merged.shape = [max(indices)] + constant
//
// Values may be merged in parallel, so if an index appears in both `indices[m][i]`
// and `indices[n][j]`, the result may be invalid. This differs from the normal
// DynamicStitch operator that defines the behavior in that case.
//
// For example:
//
// ```python
//     indices[0] = 6
//     indices[1] = [4, 1]
//     indices[2] = [[5, 2], [0, 3]]
//     data[0] = [61, 62]
//     data[1] = [[41, 42], [11, 12]]
//     data[2] = [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]
//     merged = [[1, 2], [11, 12], [21, 22], [31, 32], [41, 42],
//               [51, 52], [61, 62]]
// ```
//
// This method can be used to merge partitions created by `dynamic_partition`
// as illustrated on the following example:
//
// ```python
//     # Apply function (increments x_i) on elements for which a certain condition
//     # apply (x_i != -1 in this example).
//     x=tf.constant([0.1, -1., 5.2, 4.3, -1., 7.4])
//     condition_mask=tf.not_equal(x,tf.constant(-1.))
//     partitioned_data = tf.dynamic_partition(
//         x, tf.cast(condition_mask, tf.int32) , 2)
//     partitioned_data[1] = partitioned_data[1] + 1.0
//     condition_indices = tf.dynamic_partition(
//         tf.range(tf.shape(x)[0]), tf.cast(condition_mask, tf.int32) , 2)
//     x = tf.dynamic_stitch(condition_indices, partitioned_data)
//     # Here x=[1.1, -1., 6.2, 5.3, -1, 8.4], the -1. values remain
//     # unchanged.
// ```
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="https://www.tensorflow.org/images/DynamicStitch.png" alt>
// </div>
@_inlineable @inline(__always)
public static func parallelDynamicStitch<T: Numeric>(
  indices: [Tensor<Int32>],
  data: [Tensor<T>]
) -> Tensor<T> {
  return #tfop("ParallelDynamicStitch",
    indices,
    data,
    T: T.self)
}

// Outputs random values from a normal distribution. The parameters may each be a
//
// scalar which applies to the entire output, or a vector of length shape[0] which
// stores the parameters for each batch.
//
// - Parameters:
//   - shape: The shape of the output tensor. Batches are indexed by the 0th dimension.
//   - means: The mean parameter of each batch.
//   - stdevs: The standard deviation parameter of each batch. Must be greater than 0.
//   - minvals: The minimum cutoff. May be -infinity.
//   - maxvals: The maximum cutoff. May be +infinity, and must be more than the minval
//     for each batch.
//
// - Attrs:
//   - seed: If either `seed` or `seed2` are set to be non-zero, the random number
//     generator is seeded by the given seed.  Otherwise, it is seeded by a
//     random seed.
//   - seed2: A second seed to avoid seed collision.
//   - dtype: The type of the output.
//
// - Output output: A matrix of shape num_batches x samples_per_batch, filled with random
//   truncated normal values using the parameters for each row.
@_inlineable @inline(__always)
public static func parameterizedTruncatedNormal<Dtype: BinaryFloatingPoint, T: BinaryInteger>(
  shape: Tensor<T>,
  means: Tensor<Dtype>,
  stdevs: Tensor<Dtype>,
  minvals: Tensor<Dtype>,
  maxvals: Tensor<Dtype>,
  seed: Int = 0,
  seed2: Int = 0
) -> Tensor<Dtype> {
  return #tfop("ParameterizedTruncatedNormal",
    shape,
    means,
    stdevs,
    minvals,
    maxvals,
    Dtype: Dtype.self,
    T: T.self,
    seed: seed,
    seed2: seed2)
}

// Compute the polygamma function \\(\psi^{(n)}(x)\\).
//
// The polygamma function is defined as:
//
//
// \\(\psi^{(n)}(x) = \frac{d^n}{dx^n} \psi(x)\\)
//
// where \\(\psi(x)\\) is the digamma function.
@_inlineable @inline(__always)
public static func polygamma<T: BinaryFloatingPoint>(
  a: Tensor<T>,
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Polygamma",
    a,
    x,
    T: T.self)
}

@_inlineable @inline(__always)
public static func polymorphic<T: Numeric>(
  a: Tensor<T>
) -> Tensor<T> {
  return #tfop("Polymorphic",
    a,
    T: T.self)
}

@_inlineable @inline(__always)
public static func polymorphicDefaultOut<T: Numeric>(
) -> Tensor<T> {
  return #tfop("PolymorphicDefaultOut",
    T: T.self)
}

@_inlineable @inline(__always)
public static func polymorphicOut<T: Numeric>(
) -> Tensor<T> {
  return #tfop("PolymorphicOut",
    T: T.self)
}

// Computes element-wise population count (a.k.a. popcount, bitsum, bitcount).
//
// For each entry in `x`, calculates the number of `1` (on) bits in the binary
// representation of that entry.
//
// **NOTE**: It is more efficient to first `tf.bitcast` your tensors into
// `int32` or `int64` and perform the bitcount on the result, than to feed in
// 8- or 16-bit inputs and then aggregate the resulting counts.
@_inlineable @inline(__always)
public static func populationCount<T: BinaryInteger>(
  x: Tensor<T>
) -> Tensor<UInt8> {
  return #tfop("PopulationCount",
    x,
    T: T.self)
}

// Computes the power of one value to another.
//
// Given a tensor `x` and a tensor `y`, this operation computes \\(x^y\\) for
// corresponding elements in `x` and `y`. For example:
//
// ```
// # tensor 'x' is [[2, 2]], [3, 3]]
// # tensor 'y' is [[8, 16], [2, 3]]
// tf.pow(x, y) ==> [[256, 65536], [9, 27]]
// ```
@_inlineable @inline(__always)
public static func pow<T: Numeric>(
  x: Tensor<T>,
  y: Tensor<T>
) -> Tensor<T> {
  return #tfop("Pow",
    x,
    y,
    T: T.self)
}

// An identity op that triggers an error if a gradient is requested.
//
// When executed in a graph, this op outputs its input tensor as-is.
//
// When building ops to compute gradients, the TensorFlow gradient system
// will return an error when trying to lookup the gradient of this op,
// because no gradient must ever be registered for this function.  This
// op exists to prevent subtle bugs from silently returning unimplemented
// gradients in some corner cases.
//
// - Parameter input: any tensor.
//
// - Attr message: Will be printed in the error when anyone tries to differentiate
//   this operation.
//
// - Output output: the same input tensor.
@_inlineable @inline(__always)
public static func preventGradient<T: Numeric>(
  input: Tensor<T>,
  message: String
) -> Tensor<T> {
  return #tfop("PreventGradient",
    input,
    T: T.self,
    message: message)
}

// Prints a list of tensors.
//
// Passes `input` through to `output` and prints `data` when evaluating.
//
// - Parameters:
//   - input: The tensor passed to `output`
//   - data: A list of tensors to print out when op is evaluated.
//
// - Attrs:
//   - message: A string, prefix of the error message.
//   - first_n: Only log `first_n` number of times. -1 disables logging.
//   - summarize: Only print this many entries of each tensor.
//
// - Output output: = The unmodified `input` tensor
@_inlineable @inline(__always)
public static func print<T: Numeric, U: Numeric>(
  input: Tensor<T>,
  data: [Tensor<U>],
  message: String,
  firstN: Int = -1,
  summarize: Int = 3
) -> Tensor<T> {
  return #tfop("Print",
    input,
    data,
    T: T.self,
    message: message,
    first_n: firstN,
    summarize: summarize)
}

// Computes the product of elements across dimensions of a tensor.
//
// Reduces `input` along the dimensions given in `axis`. Unless
// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
// `axis`. If `keep_dims` is true, the reduced dimensions are
// retained with length 1.
//
// - Parameters:
//   - input: The tensor to reduce.
//   - reduction_indices: The dimensions to reduce. Must be in the range
//     `[-rank(input), rank(input))`.
//
// - Attr keep_dims: If true, retain reduced dimensions with length 1.
//
// - Output output: The reduced tensor.
@_inlineable @inline(__always)
public static func prod<T: Numeric, Tidx: BinaryInteger>(
  input: Tensor<T>,
  reductionIndices: Tensor<Tidx>,
  keepDims: Bool = false
) -> Tensor<T> {
  return #tfop("Prod",
    input,
    reductionIndices,
    T: T.self,
    Tidx: Tidx.self,
    keep_dims: keepDims)
}

// Invokes a python function to compute func(input)->output.
//
// This operation is considered stateful. For a stateless version, see
// PyFuncStateless.
//
// - Parameter input: List of Tensors that will provide input to the Op.
//
// - Attrs:
//   - token: A token representing a registered python function in this address space.
//   - Tin: Data types of the inputs to the op.
//   - Tout: Data types of the outputs from the op.
//     The length of the list specifies the number of outputs.
//
// - Output output: The outputs from the Op.
@_inlineable @inline(__always)
public static func pyFunc<Tin: Numeric, Tout: Numeric>(
  input: [Tensor<Tin>],
  token: String
) -> [Tensor<Tout>] {
  return #tfop("PyFunc",
    input,
    token: token)
}

// A stateless version of PyFunc.
@_inlineable @inline(__always)
public static func pyFuncStateless<Tin: Numeric, Tout: Numeric>(
  input: [Tensor<Tin>],
  token: String
) -> [Tensor<Tout>] {
  return #tfop("PyFuncStateless",
    input,
    token: token)
}

// Computes the QR decompositions of one or more matrices.
//
// Computes the QR decomposition of each inner matrix in `tensor` such that
// `tensor[..., :, :] = q[..., :, :] * r[..., :,:])`
//
// ```python
// # a is a tensor.
// # q is a tensor of orthonormal matrices.
// # r is a tensor of upper triangular matrices.
// q, r = qr(a)
// q_full, r_full = qr(a, full_matrices=True)
// ```
//
// - Parameter input: A tensor of shape `[..., M, N]` whose inner-most 2 dimensions
//   form matrices of size `[M, N]`. Let `P` be the minimum of `M` and `N`.
//
// - Attr full_matrices: If true, compute full-sized `q` and `r`. If false
//   (the default), compute only the leading `P` columns of `q`.
//
// - Outputs:
//   - q: Orthonormal basis for range of `a`. If `full_matrices` is `False` then
//     shape is `[..., M, P]`; if `full_matrices` is `True` then shape is
//     `[..., M, M]`.
//   - r: Triangular factor. If `full_matrices` is `False` then shape is
//     `[..., P, N]`. If `full_matrices` is `True` then shape is `[..., M, N]`.
@_inlineable @inline(__always)
public static func qr<T: BinaryFloatingPoint>(
  input: Tensor<T>,
  fullMatrices: Bool = false
) -> (Tensor<T>, Tensor<T>) {
  return #tfop("Qr",
    input,
    T: T.self,
    full_matrices: fullMatrices)
}

// Use QuantizeAndDequantizeV2 instead.
@_inlineable @inline(__always)
public static func quantizeAndDequantize<T: BinaryFloatingPoint>(
  input: Tensor<T>,
  signedInput: Bool = true,
  numBits: Int = 8,
  rangeGiven: Bool = false,
  inputMin: Double = 0,
  inputMax: Double = 0
) -> Tensor<T> {
  return #tfop("QuantizeAndDequantize",
    input,
    T: T.self,
    signed_input: signedInput,
    num_bits: numBits,
    range_given: rangeGiven,
    input_min: inputMin,
    input_max: inputMax)
}

// Quantizes then dequantizes a tensor.
//
// This op simulates the precision loss from the quantized forward pass by:
// 1. Quantizing the tensor to fixed point numbers, which should match the target
//    quantization method when it is used in inference.
// 2. Dequantizing it back to floating point numbers for the following ops, most
//    likely matmul.
//
// There are different ways to quantize. This version uses only scaling, so 0.0
// maps to 0.
//
// From the specified 'num_bits' in the quantized output type, it determines
// minimum and maximum representable quantized values.
//
// e.g.
//
// *   [-128, 127] for signed, num_bits = 8, or
// *   [0, 255] for unsigned, num_bits = 8.
//
// If range_given == False, the initial input_min, input_max will be determined
// automatically as the minimum and maximum values in the input tensor, otherwise
// the specified values of input_min, input_max are used.
//
// Note: If the input_min, input_max are specified, they do not need to equal the
// actual minimum and maximum values in the tensor. e.g. in some cases it may be
// beneficial to specify these values such that the low probability extremes of the
// input distribution are clipped.
//
// This op determines the maximum scale_factor that would map the initial
// [input_min, input_max] range to a range that lies within the representable
// quantized range.
//
// It determines the scale from one of input_min and input_max, then updates the
// other one to maximize the respresentable range.
//
// e.g.
//
// *   if the output is signed, num_bits = 8, [input_min, input_max] = [-10.0,
//     5.0]: it would use a scale_factor of -128 / -10.0 = 12.8 In this case, it
//     would update input_max to be 127 / 12.8 = 9.921875
// *   if the output is signed, num_bits = 8, [input_min, input_max] = [-10.0,
//     10.0]: it would use a scale_factor of 127 / 10.0 = 12.7 In this case, it
//     would update input_min to be 128.0 / 12.7 = -10.07874
// *   if the output is unsigned, input_min is forced to be 0, and only the
//     specifide input_max is used.
//
// After determining the scale_factor and updating the input tange, it applies the
// following to each value in the 'input' tensor.
//
// output = round(clamp(value, input_min, input_max) * scale_factor) / scale_factor.
//
//
// - Parameters:
//   - input: Tensor to quantize and then dequantize.
//   - input_min: If `range_given == True`, this specifies the minimum input value that needs to
//     be represented, otherwise it is determined from the min value of the `input`
//     tensor.
//   - input_max: If `range_given == True`, this specifies the maximum input value that needs to
//     be represented, otherwise it is determined from the max value of the `input`
//     tensor.
//
// - Attrs:
//   - signed_input: Whether the quantization is signed or unsigned. (actually this parameter should
//     have been called <b>`signed_output`</b>)
//   - num_bits: The bitwidth of the quantization.
//   - range_given: Whether the range is given or should be determined from the `input` tensor.
@_inlineable @inline(__always)
public static func quantizeAndDequantizeV2<T: BinaryFloatingPoint>(
  input: Tensor<T>,
  inputMin: Tensor<T>,
  inputMax: Tensor<T>,
  signedInput: Bool = true,
  numBits: Int = 8,
  rangeGiven: Bool = false
) -> Tensor<T> {
  return #tfop("QuantizeAndDequantizeV2",
    input,
    inputMin,
    inputMax,
    T: T.self,
    signed_input: signedInput,
    num_bits: numBits,
    range_given: rangeGiven)
}

// Quantizes then dequantizes a tensor.
//
// This is almost identical to QuantizeAndDequantizeV2, except that num_bits is a
// tensor, so its value can change during training.
@_inlineable @inline(__always)
public static func quantizeAndDequantizeV3<T: BinaryFloatingPoint>(
  input: Tensor<T>,
  inputMin: Tensor<T>,
  inputMax: Tensor<T>,
  numBits: Tensor<Int32>,
  signedInput: Bool = true,
  rangeGiven: Bool = true
) -> Tensor<T> {
  return #tfop("QuantizeAndDequantizeV3",
    input,
    inputMin,
    inputMax,
    numBits,
    T: T.self,
    signed_input: signedInput,
    range_given: rangeGiven)
}

// Convert the quantized 'input' tensor into a lower-precision 'output', using the
//
// actual distribution of the values to maximize the usage of the lower bit depth
// and adjusting the output min and max ranges accordingly.
//
// [input_min, input_max] are scalar floats that specify the range for the float
// interpretation of the 'input' data. For example, if input_min is -1.0f and
// input_max is 1.0f, and we are dealing with quint16 quantized data, then a 0
// value in the 16-bit data should be interpreted as -1.0f, and a 65535 means 1.0f.
//
// This operator tries to squeeze as much precision as possible into an output with
// a lower bit depth by calculating the actual min and max values found in the
// data. For example, maybe that quint16 input has no values lower than 16,384 and
// none higher than 49,152. That means only half the range is actually needed, all
// the float interpretations are between -0.5f and 0.5f, so if we want to compress
// the data into a quint8 output, we can use that range rather than the theoretical
// -1.0f to 1.0f that is suggested by the input min and max.
//
// In practice, this is most useful for taking output from operations like
// QuantizedMatMul that can produce higher bit-depth outputs than their inputs and
// may have large potential output ranges, but in practice have a distribution of
// input values that only uses a small fraction of the possible range. By feeding
// that output into this operator, we can reduce it from 32 bits down to 8 with
// minimal loss of accuracy.
//
// - Parameters:
//   - input_min: The float value that the minimum quantized input value represents.
//   - input_max: The float value that the maximum quantized input value represents.
//
// - Attrs:
//   - Tinput: The type of the input.
//   - out_type: The type of the output. Should be a lower bit depth than Tinput.
//
// - Outputs:
//   - output_min: The float value that the minimum quantized output value represents.
//   - output_max: The float value that the maximum quantized output value represents.
@_inlineable @inline(__always)
public static func quantizeDownAndShrinkRange<Tinput: Numeric, Out_type: Numeric>(
  input: Tensor<Tinput>,
  inputMin: Tensor<Float>,
  inputMax: Tensor<Float>
) -> (Tensor<Out_type>, Tensor<Float>, Tensor<Float>) {
  return #tfop("QuantizeDownAndShrinkRange",
    input,
    inputMin,
    inputMax,
    Tinput: Tinput.self,
    Out_type: Out_type.self)
}

// Quantize the 'input' tensor of type float to 'output' tensor of type 'T'.
//
// [min_range, max_range] are scalar floats that specify the range for
// the 'input' data. The 'mode' attribute controls exactly which calculations are
// used to convert the float values to their quantized equivalents.  The
// 'round_mode' attribute controls which rounding tie-breaking algorithm is used
// when rounding float values to their quantized equivalents.
//
// In 'MIN_COMBINED' mode, each value of the tensor will undergo the following:
//
// ```
// out[i] = (in[i] - min_range) * range(T) / (max_range - min_range)
// if T == qint8, out[i] -= (range(T) + 1) / 2.0
// ```
//
// here `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`
//
// *MIN_COMBINED Mode Example*
//
// Assume the input is type float and has a possible range of [0.0, 6.0] and the
// output type is quint8 ([0, 255]). The min_range and max_range values should be
// specified as 0.0 and 6.0. Quantizing from float to quint8 will multiply each
// value of the input by 255/6 and cast to quint8.
//
// If the output type was qint8 ([-128, 127]), the operation will additionally
// subtract each value by 128 prior to casting, so that the range of values aligns
// with the range of qint8.
//
// If the mode is 'MIN_FIRST', then this approach is used:
//
// ```
// num_discrete_values = 1 << (# of bits in T)
// range_adjust = num_discrete_values / (num_discrete_values - 1)
// range = (range_max - range_min) * range_adjust
// range_scale = num_discrete_values / range
// quantized = round(input * range_scale) - round(range_min * range_scale) +
//   numeric_limits<T>::min()
// quantized = max(quantized, numeric_limits<T>::min())
// quantized = min(quantized, numeric_limits<T>::max())
// ```
//
// The biggest difference between this and MIN_COMBINED is that the minimum range
// is rounded first, before it's subtracted from the rounded value. With
// MIN_COMBINED, a small bias is introduced where repeated iterations of quantizing
// and dequantizing will introduce a larger and larger error.
//
// *SCALED mode Example*
//
// `SCALED` mode matches the quantization approach used in
// `QuantizeAndDequantize{V2|V3}`.
//
// If the mode is `SCALED`, we do not use the full range of the output type,
// choosing to elide the lowest possible value for symmetry (e.g., output range is
// -127 to 127, not -128 to 127 for signed 8 bit quantization), so that 0.0 maps to
// 0.
//
// We first find the range of values in our tensor. The
// range we use is always centered on 0, so we find m such that
//
// ```c++
//   m = max(abs(input_min), abs(input_max))
// ```
//
// Our input tensor range is then `[-m, m]`.
//
// Next, we choose our fixed-point quantization buckets, `[min_fixed, max_fixed]`.
// If T is signed, this is
//
// ```
//   num_bits = sizeof(T) * 8
//   [min_fixed, max_fixed] =
//       [-(1 << (num_bits - 1) - 1), (1 << (num_bits - 1)) - 1]
// ```
//
// Otherwise, if T is unsigned, the fixed-point range is
//
// ```
//   [min_fixed, max_fixed] = [0, (1 << num_bits) - 1]
// ```
//
// From this we compute our scaling factor, s:
//
// ```c++
//   s = (max_fixed - min_fixed) / (2 * m)
// ```
//
// Now we can quantize the elements of our tensor:
//
// ```c++
// result = round(input * s)
// ```
//
// One thing to watch out for is that the operator may choose to adjust the
// requested minimum and maximum values slightly during the quantization process,
// so you should always use the output ports as the range for further calculations.
// For example, if the requested minimum and maximum values are close to equal,
// they will be separated by a small epsilon value to prevent ill-formed quantized
// buffers from being created. Otherwise, you can end up with buffers where all the
// quantized values map to the same float value, which causes problems for
// operations that have to perform further calculations on them.
//
// - Parameters:
//   - min_range: The minimum scalar value possibly produced for the input.
//   - max_range: The maximum scalar value possibly produced for the input.
//
// - Outputs:
//   - output: The quantized data produced from the float input.
//   - output_min: The actual minimum scalar value used for the output.
//   - output_max: The actual maximum scalar value used for the output.
@_inlineable @inline(__always)
public static func quantizeV2<T: Numeric>(
  input: Tensor<Float>,
  minRange: Tensor<Float>,
  maxRange: Tensor<Float>,
  mode: Mode = .minCombined,
  roundMode: RoundMode = .halfAwayFromZero
) -> (Tensor<T>, Tensor<Float>, Tensor<Float>) {
  return #tfop("QuantizeV2",
    input,
    minRange,
    maxRange,
    T: T.self,
    mode: mode.rawValue,
    round_mode: roundMode.rawValue)
}

// Returns x + y element-wise, working on quantized buffers.
//
// - Parameters:
//   - min_x: The float value that the lowest quantized `x` value represents.
//   - max_x: The float value that the highest quantized `x` value represents.
//   - min_y: The float value that the lowest quantized `y` value represents.
//   - max_y: The float value that the highest quantized `y` value represents.
//
// - Outputs:
//   - min_z: The float value that the lowest quantized output value represents.
//   - max_z: The float value that the highest quantized output value represents.
//
//     *NOTE*: `QuantizedAdd` supports limited forms of broadcasting. More about
//     broadcasting [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@_inlineable @inline(__always)
public static func quantizedAdd<T1: Numeric, T2: Numeric, Toutput: Numeric>(
  x: Tensor<T1>,
  y: Tensor<T2>,
  minX: Tensor<Float>,
  maxX: Tensor<Float>,
  minY: Tensor<Float>,
  maxY: Tensor<Float>
) -> (Tensor<Toutput>, Tensor<Float>, Tensor<Float>) {
  return #tfop("QuantizedAdd",
    x,
    y,
    minX,
    maxX,
    minY,
    maxY,
    T1: T1.self,
    T2: T2.self,
    Toutput: Toutput.self)
}

// Produces the average pool of the input tensor for quantized types.
//
// - Parameters:
//   - input: 4-D with shape `[batch, height, width, channels]`.
//   - min_input: The float value that the lowest quantized input value represents.
//   - max_input: The float value that the highest quantized input value represents.
//
// - Attrs:
//   - ksize: The size of the window for each dimension of the input tensor.
//     The length must be 4 to match the number of dimensions of the input.
//   - strides: The stride of the sliding window for each dimension of the input
//     tensor.  The length must be 4 to match the number of dimensions of the input.
//   - padding: The type of padding algorithm to use.
//
// - Outputs:
//   - min_output: The float value that the lowest quantized output value represents.
//   - max_output: The float value that the highest quantized output value represents.
@_inlineable @inline(__always)
public static func quantizedAvgPool<T: Numeric>(
  input: Tensor<T>,
  minInput: Tensor<Float>,
  maxInput: Tensor<Float>,
  ksize: [Int],
  strides: [Int],
  padding: Padding
) -> (Tensor<T>, Tensor<Float>, Tensor<Float>) {
  return #tfop("QuantizedAvgPool",
    input,
    minInput,
    maxInput,
    T: T.self,
    ksize: ksize,
    strides: strides,
    padding: padding.rawValue)
}

// Quantized Batch normalization.
//
// This op is deprecated and will be removed in the future. Prefer
// `tf.nn.batch_normalization`.
//
// - Parameters:
//   - t: A 4D input Tensor.
//   - t_min: The value represented by the lowest quantized input.
//   - t_max: The value represented by the highest quantized input.
//   - m: A 1D mean Tensor with size matching the last dimension of t.
//     This is the first output from tf.nn.moments,
//     or a saved moving average thereof.
//   - m_min: The value represented by the lowest quantized mean.
//   - m_max: The value represented by the highest quantized mean.
//   - v: A 1D variance Tensor with size matching the last dimension of t.
//     This is the second output from tf.nn.moments,
//     or a saved moving average thereof.
//   - v_min: The value represented by the lowest quantized variance.
//   - v_max: The value represented by the highest quantized variance.
//   - beta: A 1D beta Tensor with size matching the last dimension of t.
//     An offset to be added to the normalized tensor.
//   - beta_min: The value represented by the lowest quantized offset.
//   - beta_max: The value represented by the highest quantized offset.
//   - gamma: A 1D gamma Tensor with size matching the last dimension of t.
//     If "scale_after_normalization" is true, this tensor will be multiplied
//     with the normalized tensor.
//   - gamma_min: The value represented by the lowest quantized gamma.
//   - gamma_max: The value represented by the highest quantized gamma.
//
// - Attrs:
//   - variance_epsilon: A small float number to avoid dividing by 0.
//   - scale_after_normalization: A bool indicating whether the resulted tensor
//     needs to be multiplied with gamma.
@_inlineable @inline(__always)
public static func quantizedBatchNormWithGlobalNormalization<Tinput: Numeric, Out_type: Numeric>(
  t: Tensor<Tinput>,
  tMin: Tensor<Float>,
  tMax: Tensor<Float>,
  m: Tensor<Tinput>,
  mMin: Tensor<Float>,
  mMax: Tensor<Float>,
  v: Tensor<Tinput>,
  vMin: Tensor<Float>,
  vMax: Tensor<Float>,
  beta: Tensor<Tinput>,
  betaMin: Tensor<Float>,
  betaMax: Tensor<Float>,
  gamma: Tensor<Tinput>,
  gammaMin: Tensor<Float>,
  gammaMax: Tensor<Float>,
  varianceEpsilon: Double,
  scaleAfterNormalization: Bool
) -> (Tensor<Out_type>, Tensor<Float>, Tensor<Float>) {
  return #tfop("QuantizedBatchNormWithGlobalNormalization",
    t,
    tMin,
    tMax,
    m,
    mMin,
    mMax,
    v,
    vMin,
    vMax,
    beta,
    betaMin,
    betaMax,
    gamma,
    gammaMin,
    gammaMax,
    Tinput: Tinput.self,
    Out_type: Out_type.self,
    variance_epsilon: varianceEpsilon,
    scale_after_normalization: scaleAfterNormalization)
}

// Adds Tensor 'bias' to Tensor 'input' for Quantized types.
//
// Broadcasts the values of bias on dimensions 0..N-2 of 'input'.
//
// - Parameters:
//   - bias: A 1D bias Tensor with size matching the last dimension of 'input'.
//   - min_input: The float value that the lowest quantized input value represents.
//   - max_input: The float value that the highest quantized input value represents.
//   - min_bias: The float value that the lowest quantized bias value represents.
//   - max_bias: The float value that the highest quantized bias value represents.
//
// - Outputs:
//   - min_out: The float value that the lowest quantized output value represents.
//   - max_out: The float value that the highest quantized output value represents.
@_inlineable @inline(__always)
public static func quantizedBiasAdd<T1: Numeric, T2: Numeric, Out_type: Numeric>(
  input: Tensor<T1>,
  bias: Tensor<T2>,
  minInput: Tensor<Float>,
  maxInput: Tensor<Float>,
  minBias: Tensor<Float>,
  maxBias: Tensor<Float>
) -> (Tensor<Out_type>, Tensor<Float>, Tensor<Float>) {
  return #tfop("QuantizedBiasAdd",
    input,
    bias,
    minInput,
    maxInput,
    minBias,
    maxBias,
    T1: T1.self,
    T2: T2.self,
    Out_type: Out_type.self)
}

// Concatenates quantized tensors along one dimension.
//
// - Parameters:
//   - concat_dim: 0-D.  The dimension along which to concatenate.  Must be in the
//     range [0, rank(values)).
//   - values: The `N` Tensors to concatenate. Their ranks and types must match,
//     and their sizes must match in all dimensions except `concat_dim`.
//   - input_mins: The minimum scalar values for each of the input tensors.
//   - input_maxes: The maximum scalar values for each of the input tensors.
//
// - Outputs:
//   - output: A `Tensor` with the concatenation of values stacked along the
//     `concat_dim` dimension.  This tensor's shape matches that of `values` except
//     in `concat_dim` where it has the sum of the sizes.
//   - output_min: The float value that the minimum quantized output value represents.
//   - output_max: The float value that the maximum quantized output value represents.
@_inlineable @inline(__always)
public static func quantizedConcat<T: Numeric>(
  concatDim: Tensor<Int32>,
  values: [Tensor<T>],
  inputMins: [Tensor<Float>],
  inputMaxes: [Tensor<Float>]
) -> (Tensor<T>, Tensor<Float>, Tensor<Float>) {
  return #tfop("QuantizedConcat",
    concatDim,
    values,
    inputMins,
    inputMaxes,
    T: T.self)
}

// Computes a 2D convolution given quantized 4D input and filter tensors.
//
// The inputs are quantized tensors where the lowest value represents the real
// number of the associated minimum, and the highest represents the maximum.
// This means that you can only interpret the quantized output in the same way, by
// taking the returned minimum and maximum values into account.
//
// - Parameters:
//   - filter: filter's input_depth dimension must match input's depth dimensions.
//   - min_input: The float value that the lowest quantized input value represents.
//   - max_input: The float value that the highest quantized input value represents.
//   - min_filter: The float value that the lowest quantized filter value represents.
//   - max_filter: The float value that the highest quantized filter value represents.
//
// - Attrs:
//   - strides: The stride of the sliding window for each dimension of the input
//     tensor.
//   - padding: The type of padding algorithm to use.
//   - dilations: 1-D tensor of length 4.  The dilation factor for each dimension of
//     `input`. If set to k > 1, there will be k-1 skipped cells between each
//     filter element on that dimension. The dimension order is determined by the
//     value of `data_format`, see above for details. Dilations in the batch and
//     depth dimensions must be 1.
//
// - Outputs:
//   - min_output: The float value that the lowest quantized output value represents.
//   - max_output: The float value that the highest quantized output value represents.
@_inlineable @inline(__always)
public static func quantizedConv2D<Tinput: Numeric, Tfilter: Numeric, Out_type: Numeric>(
  input: Tensor<Tinput>,
  filter: Tensor<Tfilter>,
  minInput: Tensor<Float>,
  maxInput: Tensor<Float>,
  minFilter: Tensor<Float>,
  maxFilter: Tensor<Float>,
  strides: [Int],
  padding: Padding,
  dilations: [Int]
) -> (Tensor<Out_type>, Tensor<Float>, Tensor<Float>) {
  return #tfop("QuantizedConv2D",
    input,
    filter,
    minInput,
    maxInput,
    minFilter,
    maxFilter,
    Tinput: Tinput.self,
    Tfilter: Tfilter.self,
    Out_type: Out_type.self,
    strides: strides,
    padding: padding.rawValue,
    dilations: dilations)
}

// Quantized Instance normalization.
//
// - Parameters:
//   - x: A 4D input Tensor.
//   - x_min: The value represented by the lowest quantized input.
//   - x_max: The value represented by the highest quantized input.
//
// - Attrs:
//   - output_range_given: If True, `given_y_min` and `given_y_min`
//     and `given_y_max` are used as the output range. Otherwise,
//     the implementation computes the output range.
//   - given_y_min: Output in `y_min` if `output_range_given` is True.
//   - given_y_max: Output in `y_max` if `output_range_given` is True.
//   - variance_epsilon: A small float number to avoid dividing by 0.
//   - min_separation: Minimum value of `y_max - y_min`
//
// - Outputs:
//   - y: A 4D Tensor.
//   - y_min: The value represented by the lowest quantized output.
//   - y_max: The value represented by the highest quantized output.
@_inlineable @inline(__always)
public static func quantizedInstanceNorm<T: Numeric>(
  x: Tensor<T>,
  xMin: Tensor<Float>,
  xMax: Tensor<Float>,
  outputRangeGiven: Bool = false,
  givenYMin: Double = 0,
  givenYMax: Double = 0,
  varianceEpsilon: Double = 1e-05,
  minSeparation: Double = 0.001
) -> (Tensor<T>, Tensor<Float>, Tensor<Float>) {
  return #tfop("QuantizedInstanceNorm",
    x,
    xMin,
    xMax,
    T: T.self,
    output_range_given: outputRangeGiven,
    given_y_min: givenYMin,
    given_y_max: givenYMax,
    variance_epsilon: varianceEpsilon,
    min_separation: minSeparation)
}

// Perform a quantized matrix multiplication of  `a` by the matrix `b`.
//
// The inputs must be two-dimensional matrices and the inner dimension of
// `a` (after being transposed if `transpose_a` is non-zero) must match the
// outer dimension of `b` (after being transposed if `transposed_b` is
// non-zero).
//
// - Parameters:
//   - a: Must be a two-dimensional tensor.
//   - b: Must be a two-dimensional tensor.
//   - min_a: The float value that the lowest quantized `a` value represents.
//   - max_a: The float value that the highest quantized `a` value represents.
//   - min_b: The float value that the lowest quantized `b` value represents.
//   - max_b: The float value that the highest quantized `b` value represents.
//
// - Attrs:
//   - transpose_a: If true, `a` is transposed before multiplication.
//   - transpose_b: If true, `b` is transposed before multiplication.
//   - Tactivation: The type of output produced by activation function
//     following this operation.
//
// - Outputs:
//   - min_out: The float value that the lowest quantized output value represents.
//   - max_out: The float value that the highest quantized output value represents.
@_inlineable @inline(__always)
public static func quantizedMatMul<T1: Numeric, T2: Numeric, Toutput: Numeric, Tactivation: Numeric>(
  a: Tensor<T1>,
  b: Tensor<T2>,
  minA: Tensor<Float>,
  maxA: Tensor<Float>,
  minB: Tensor<Float>,
  maxB: Tensor<Float>,
  transposeA: Bool = false,
  transposeB: Bool = false,
  typeTactivation: Tactivation.Type
) -> (Tensor<Toutput>, Tensor<Float>, Tensor<Float>) {
  return #tfop("QuantizedMatMul",
    a,
    b,
    minA,
    maxA,
    minB,
    maxB,
    T1: T1.self,
    T2: T2.self,
    Toutput: Toutput.self,
    Tactivation: Tactivation.self,
    transpose_a: transposeA,
    transpose_b: transposeB)
}

// Produces the max pool of the input tensor for quantized types.
//
// - Parameters:
//   - input: The 4D (batch x rows x cols x depth) Tensor to MaxReduce over.
//   - min_input: The float value that the lowest quantized input value represents.
//   - max_input: The float value that the highest quantized input value represents.
//
// - Attrs:
//   - ksize: The size of the window for each dimension of the input tensor.
//     The length must be 4 to match the number of dimensions of the input.
//   - strides: The stride of the sliding window for each dimension of the input
//     tensor. The length must be 4 to match the number of dimensions of the input.
//   - padding: The type of padding algorithm to use.
//
// - Outputs:
//   - min_output: The float value that the lowest quantized output value represents.
//   - max_output: The float value that the highest quantized output value represents.
@_inlineable @inline(__always)
public static func quantizedMaxPool<T: Numeric>(
  input: Tensor<T>,
  minInput: Tensor<Float>,
  maxInput: Tensor<Float>,
  ksize: [Int],
  strides: [Int],
  padding: Padding
) -> (Tensor<T>, Tensor<Float>, Tensor<Float>) {
  return #tfop("QuantizedMaxPool",
    input,
    minInput,
    maxInput,
    T: T.self,
    ksize: ksize,
    strides: strides,
    padding: padding.rawValue)
}

// Returns x * y element-wise, working on quantized buffers.
//
// - Parameters:
//   - min_x: The float value that the lowest quantized `x` value represents.
//   - max_x: The float value that the highest quantized `x` value represents.
//   - min_y: The float value that the lowest quantized `y` value represents.
//   - max_y: The float value that the highest quantized `y` value represents.
//
// - Outputs:
//   - min_z: The float value that the lowest quantized output value represents.
//   - max_z: The float value that the highest quantized output value represents.
//
//     *NOTE*: `QuantizedMul` supports limited forms of broadcasting. More about
//     broadcasting [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@_inlineable @inline(__always)
public static func quantizedMul<T1: Numeric, T2: Numeric, Toutput: Numeric>(
  x: Tensor<T1>,
  y: Tensor<T2>,
  minX: Tensor<Float>,
  maxX: Tensor<Float>,
  minY: Tensor<Float>,
  maxY: Tensor<Float>
) -> (Tensor<Toutput>, Tensor<Float>, Tensor<Float>) {
  return #tfop("QuantizedMul",
    x,
    y,
    minX,
    maxX,
    minY,
    maxY,
    T1: T1.self,
    T2: T2.self,
    Toutput: Toutput.self)
}

// Computes Quantized Rectified Linear: `max(features, 0)`
//
// - Parameters:
//   - min_features: The float value that the lowest quantized value represents.
//   - max_features: The float value that the highest quantized value represents.
//
// - Outputs:
//   - activations: Has the same output shape as "features".
//   - min_activations: The float value that the lowest quantized value represents.
//   - max_activations: The float value that the highest quantized value represents.
@_inlineable @inline(__always)
public static func quantizedRelu<Tinput: Numeric, Out_type: Numeric>(
  features: Tensor<Tinput>,
  minFeatures: Tensor<Float>,
  maxFeatures: Tensor<Float>
) -> (Tensor<Out_type>, Tensor<Float>, Tensor<Float>) {
  return #tfop("QuantizedRelu",
    features,
    minFeatures,
    maxFeatures,
    Tinput: Tinput.self,
    Out_type: Out_type.self)
}

// Computes Quantized Rectified Linear 6: `min(max(features, 0), 6)`
//
// - Parameters:
//   - min_features: The float value that the lowest quantized value represents.
//   - max_features: The float value that the highest quantized value represents.
//
// - Outputs:
//   - activations: Has the same output shape as "features".
//   - min_activations: The float value that the lowest quantized value represents.
//   - max_activations: The float value that the highest quantized value represents.
@_inlineable @inline(__always)
public static func quantizedRelu6<Tinput: Numeric, Out_type: Numeric>(
  features: Tensor<Tinput>,
  minFeatures: Tensor<Float>,
  maxFeatures: Tensor<Float>
) -> (Tensor<Out_type>, Tensor<Float>, Tensor<Float>) {
  return #tfop("QuantizedRelu6",
    features,
    minFeatures,
    maxFeatures,
    Tinput: Tinput.self,
    Out_type: Out_type.self)
}

// Computes Quantized Rectified Linear X: `min(max(features, 0), max_value)`
//
// - Parameters:
//   - min_features: The float value that the lowest quantized value represents.
//   - max_features: The float value that the highest quantized value represents.
//
// - Outputs:
//   - activations: Has the same output shape as "features".
//   - min_activations: The float value that the lowest quantized value represents.
//   - max_activations: The float value that the highest quantized value represents.
@_inlineable @inline(__always)
public static func quantizedReluX<Tinput: Numeric, Out_type: Numeric>(
  features: Tensor<Tinput>,
  maxValue: Tensor<Float>,
  minFeatures: Tensor<Float>,
  maxFeatures: Tensor<Float>
) -> (Tensor<Out_type>, Tensor<Float>, Tensor<Float>) {
  return #tfop("QuantizedReluX",
    features,
    maxValue,
    minFeatures,
    maxFeatures,
    Tinput: Tinput.self,
    Out_type: Out_type.self)
}

// Reshapes a quantized tensor as per the Reshape op.
//
// ```
//
// - Parameters:
//   - shape: Defines the shape of the output tensor.
//   - input_min: The minimum value of the input.
//   - input_max: The maximum value of the input.
//
// - Outputs:
//   - output_min: This value is copied from input_min.
//   - output_max: This value is copied from input_max.
@_inlineable @inline(__always)
public static func quantizedReshape<T: Numeric, Tshape: BinaryInteger>(
  tensor: Tensor<T>,
  shape: Tensor<Tshape>,
  inputMin: Tensor<Float>,
  inputMax: Tensor<Float>
) -> (Tensor<T>, Tensor<Float>, Tensor<Float>) {
  return #tfop("QuantizedReshape",
    tensor,
    shape,
    inputMin,
    inputMax,
    T: T.self,
    Tshape: Tshape.self)
}

// Resize quantized `images` to `size` using quantized bilinear interpolation.
//
// Input images and output images must be quantized types.
//
// - Parameters:
//   - images: 4-D with shape `[batch, height, width, channels]`.
//   - size: = A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
//     new size for the images.
//
// - Attr align_corners: If true, the centers of the 4 corner pixels of the input and output tensors are
//   aligned, preserving the values at the corner pixels. Defaults to false.
//
// - Output resized_images: 4-D with shape
//   `[batch, new_height, new_width, channels]`.
@_inlineable @inline(__always)
public static func quantizedResizeBilinear<T: BinaryFloatingPoint>(
  images: Tensor<T>,
  size: Tensor<Int32>,
  min: Tensor<Float>,
  max: Tensor<Float>,
  alignCorners: Bool = false
) -> (Tensor<T>, Tensor<Float>, Tensor<Float>) {
  return #tfop("QuantizedResizeBilinear",
    images,
    size,
    min,
    max,
    T: T.self,
    align_corners: alignCorners)
}

// Converts one or more images from RGB to HSV.
//
// Outputs a tensor of the same shape as the `images` tensor, containing the HSV
// value of the pixels. The output is only well defined if the value in `images`
// are in `[0,1]`.
//
// `output[..., 0]` contains hue, `output[..., 1]` contains saturation, and
// `output[..., 2]` contains value. All HSV values are in `[0,1]`. A hue of 0
// corresponds to pure red, hue 1/3 is pure green, and 2/3 is pure blue.
//
// - Parameter images: 1-D or higher rank. RGB data to convert. Last dimension must be size 3.
//
// - Output output: `images` converted to HSV.
@_inlineable @inline(__always)
public static func rGBToHSV<T: BinaryFloatingPoint>(
  images: Tensor<T>
) -> Tensor<T> {
  return #tfop("RGBToHSV",
    images,
    T: T.self)
}

// Randomly crop `image`.
//
// `size` is a 1-D int64 tensor with 2 elements representing the crop height and
// width.  The values must be non negative.
//
// This Op picks a random location in `image` and crops a `height` by `width`
// rectangle from that location.  The random location is picked so the cropped
// area will fit inside the original image.
//
// - Parameters:
//   - image: 3-D of shape `[height, width, channels]`.
//   - size: 1-D of length 2 containing: `crop_height`, `crop_width`..
//
// - Attrs:
//   - seed: If either seed or seed2 are set to be non-zero, the random number
//     generator is seeded by the given seed.  Otherwise, it is seeded by a
//     random seed.
//   - seed2: An second seed to avoid seed collision.
//
// - Output output: 3-D of shape `[crop_height, crop_width, channels].`
@_inlineable @inline(__always)
public static func randomCrop<T: Numeric>(
  image: Tensor<T>,
  size: Tensor<Int64>,
  seed: Int = 0,
  seed2: Int = 0
) -> Tensor<T> {
  return #tfop("RandomCrop",
    image,
    size,
    T: T.self,
    seed: seed,
    seed2: seed2)
}

// Outputs random values from the Gamma distribution(s) described by alpha.
//
// This op uses the algorithm by Marsaglia et al. to acquire samples via
// transformation-rejection from pairs of uniform and normal random variables.
// See http://dl.acm.org/citation.cfm?id=358414
//
// - Parameters:
//   - shape: 1-D integer tensor. Shape of independent samples to draw from each
//     distribution described by the shape parameters given in alpha.
//   - alpha: A tensor in which each scalar is a "shape" parameter describing the
//     associated gamma distribution.
//
// - Attrs:
//   - seed: If either `seed` or `seed2` are set to be non-zero, the random number
//     generator is seeded by the given seed.  Otherwise, it is seeded by a
//     random seed.
//   - seed2: A second seed to avoid seed collision.
//
// - Output output: A tensor with shape `shape + shape(alpha)`. Each slice
//   `[:, ..., :, i0, i1, ...iN]` contains the samples drawn for
//   `alpha[i0, i1, ...iN]`. The dtype of the output matches the dtype of alpha.
@_inlineable @inline(__always)
public static func randomGamma<S: BinaryInteger, T: BinaryFloatingPoint>(
  shape: Tensor<S>,
  alpha: Tensor<T>,
  seed: Int = 0,
  seed2: Int = 0
) -> Tensor<T> {
  return #tfop("RandomGamma",
    shape,
    alpha,
    S: S.self,
    T: T.self,
    seed: seed,
    seed2: seed2)
}

// Use RandomPoissonV2 instead.
@_inlineable @inline(__always)
public static func randomPoisson<S: BinaryInteger, Dtype: BinaryFloatingPoint>(
  shape: Tensor<S>,
  rate: Tensor<Dtype>,
  seed: Int = 0,
  seed2: Int = 0
) -> Tensor<Dtype> {
  return #tfop("RandomPoisson",
    shape,
    rate,
    S: S.self,
    Dtype: Dtype.self,
    seed: seed,
    seed2: seed2)
}

// Outputs random values from the Poisson distribution(s) described by rate.
//
// This op uses two algorithms, depending on rate. If rate >= 10, then
// the algorithm by Hormann is used to acquire samples via
// transformation-rejection.
// See http://www.sciencedirect.com/science/article/pii/0167668793909974.
//
// Otherwise, Knuth's algorithm is used to acquire samples via multiplying uniform
// random variables.
// See Donald E. Knuth (1969). Seminumerical Algorithms. The Art of Computer
// Programming, Volume 2. Addison Wesley
//
// - Parameters:
//   - shape: 1-D integer tensor. Shape of independent samples to draw from each
//     distribution described by the shape parameters given in rate.
//   - rate: A tensor in which each scalar is a "rate" parameter describing the
//     associated poisson distribution.
//
// - Attrs:
//   - seed: If either `seed` or `seed2` are set to be non-zero, the random number
//     generator is seeded by the given seed.  Otherwise, it is seeded by a
//     random seed.
//   - seed2: A second seed to avoid seed collision.
//
// - Output output: A tensor with shape `shape + shape(rate)`. Each slice
//   `[:, ..., :, i0, i1, ...iN]` contains the samples drawn for
//   `rate[i0, i1, ...iN]`.
@_inlineable @inline(__always)
public static func randomPoissonV2<S: BinaryInteger, R: Numeric, Dtype: Numeric>(
  shape: Tensor<S>,
  rate: Tensor<R>,
  seed: Int = 0,
  seed2: Int = 0
) -> Tensor<Dtype> {
  return #tfop("RandomPoissonV2",
    shape,
    rate,
    S: S.self,
    R: R.self,
    Dtype: Dtype.self,
    seed: seed,
    seed2: seed2)
}

// Randomly shuffles a tensor along its first dimension.
//
//   The tensor is shuffled along dimension 0, such that each `value[j]` is mapped
//   to one and only one `output[i]`. For example, a mapping that might occur for a
//   3x2 tensor is:
//
// ```
// [[1, 2],       [[5, 6],
//  [3, 4],  ==>   [1, 2],
//  [5, 6]]        [3, 4]]
// ```
//
// - Parameter value: The tensor to be shuffled.
//
// - Attrs:
//   - seed: If either `seed` or `seed2` are set to be non-zero, the random number
//     generator is seeded by the given seed.  Otherwise, it is seeded by a
//     random seed.
//   - seed2: A second seed to avoid seed collision.
//
// - Output output: A tensor of same shape and type as `value`, shuffled along its first
//   dimension.
@_inlineable @inline(__always)
public static func randomShuffle<T: Numeric>(
  value: Tensor<T>,
  seed: Int = 0,
  seed2: Int = 0
) -> Tensor<T> {
  return #tfop("RandomShuffle",
    value,
    T: T.self,
    seed: seed,
    seed2: seed2)
}

// Outputs random values from a normal distribution.
//
// The generated values will have mean 0 and standard deviation 1.
//
// - Parameter shape: The shape of the output tensor.
//
// - Attrs:
//   - seed: If either `seed` or `seed2` are set to be non-zero, the random number
//     generator is seeded by the given seed.  Otherwise, it is seeded by a
//     random seed.
//   - seed2: A second seed to avoid seed collision.
//   - dtype: The type of the output.
//
// - Output output: A tensor of the specified shape filled with random normal values.
@_inlineable @inline(__always)
public static func randomStandardNormal<Dtype: BinaryFloatingPoint, T: BinaryInteger>(
  shape: Tensor<T>,
  seed: Int = 0,
  seed2: Int = 0
) -> Tensor<Dtype> {
  return #tfop("RandomStandardNormal",
    shape,
    Dtype: Dtype.self,
    T: T.self,
    seed: seed,
    seed2: seed2)
}

// Outputs random values from a uniform distribution.
//
// The generated values follow a uniform distribution in the range `[0, 1)`. The
// lower bound 0 is included in the range, while the upper bound 1 is excluded.
//
// - Parameter shape: The shape of the output tensor.
//
// - Attrs:
//   - seed: If either `seed` or `seed2` are set to be non-zero, the random number
//     generator is seeded by the given seed.  Otherwise, it is seeded by a
//     random seed.
//   - seed2: A second seed to avoid seed collision.
//   - dtype: The type of the output.
//
// - Output output: A tensor of the specified shape filled with uniform random values.
@_inlineable @inline(__always)
public static func randomUniform<Dtype: BinaryFloatingPoint, T: BinaryInteger>(
  shape: Tensor<T>,
  seed: Int = 0,
  seed2: Int = 0
) -> Tensor<Dtype> {
  return #tfop("RandomUniform",
    shape,
    Dtype: Dtype.self,
    T: T.self,
    seed: seed,
    seed2: seed2)
}

// Outputs random integers from a uniform distribution.
//
// The generated values are uniform integers in the range `[minval, maxval)`.
// The lower bound `minval` is included in the range, while the upper bound
// `maxval` is excluded.
//
// The random integers are slightly biased unless `maxval - minval` is an exact
// power of two.  The bias is small for values of `maxval - minval` significantly
// smaller than the range of the output (either `2^32` or `2^64`).
//
// - Parameters:
//   - shape: The shape of the output tensor.
//   - minval: 0-D.  Inclusive lower bound on the generated integers.
//   - maxval: 0-D.  Exclusive upper bound on the generated integers.
//
// - Attrs:
//   - seed: If either `seed` or `seed2` are set to be non-zero, the random number
//     generator is seeded by the given seed.  Otherwise, it is seeded by a
//     random seed.
//   - seed2: A second seed to avoid seed collision.
//
// - Output output: A tensor of the specified shape filled with uniform random integers.
@_inlineable @inline(__always)
public static func randomUniformInt<Tout: BinaryInteger, T: BinaryInteger>(
  shape: Tensor<T>,
  minval: Tensor<Tout>,
  maxval: Tensor<Tout>,
  seed: Int = 0,
  seed2: Int = 0
) -> Tensor<Tout> {
  return #tfop("RandomUniformInt",
    shape,
    minval,
    maxval,
    Tout: Tout.self,
    T: T.self,
    seed: seed,
    seed2: seed2)
}

// Creates a sequence of numbers.
//
// This operation creates a sequence of numbers that begins at `start` and
// extends by increments of `delta` up to but not including `limit`.
//
// For example:
//
// ```
// # 'start' is 3
// # 'limit' is 18
// # 'delta' is 3
// tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15]
// ```
//
// - Parameters:
//   - start: 0-D (scalar). First entry in the sequence.
//   - limit: 0-D (scalar). Upper limit of sequence, exclusive.
//   - delta: 0-D (scalar). Optional. Default is 1. Number that increments `start`.
//
// - Output output: 1-D.
@_inlineable @inline(__always)
public static func range<Tidx: Numeric>(
  start: Tensor<Tidx>,
  limit: Tensor<Tidx>,
  delta: Tensor<Tidx>
) -> Tensor<Tidx> {
  return #tfop("Range",
    start,
    limit,
    delta,
    Tidx: Tidx.self)
}

// Returns the rank of a tensor.
//
// This operation returns an integer representing the rank of `input`.
//
// For example:
//
// ```
// # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
// # shape of tensor 't' is [2, 2, 3]
// rank(t) ==> 3
// ```
//
// **Note**: The rank of a tensor is not the same as the rank of a matrix. The rank
// of a tensor is the number of indices required to uniquely select each element
// of the tensor. Rank is also known as "order", "degree", or "ndims."
@_inlineable @inline(__always)
public static func rank<T: Numeric>(
  input: Tensor<T>
) -> Tensor<Int32> {
  return #tfop("Rank",
    input,
    T: T.self)
}

// Returns the real part of a complex number.
//
// Given a tensor `input` of complex numbers, this operation returns a tensor of
// type `float` that is the real part of each element in `input`. All elements in
// `input` must be complex numbers of the form \\(a + bj\\), where *a* is the real
//  part returned by this operation and *b* is the imaginary part.
//
// For example:
//
// ```
// # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
// tf.real(input) ==> [-2.25, 3.25]
// ```
@_inlineable @inline(__always)
public static func real<T: Numeric, Tout: BinaryFloatingPoint>(
  input: Tensor<T>
) -> Tensor<Tout> {
  return #tfop("Real",
    input,
    T: T.self,
    Tout: Tout.self)
}

// Returns x / y element-wise for real types.
//
// If `x` and `y` are reals, this will return the floating-point division.
//
// *NOTE*: `Div` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@_inlineable @inline(__always)
public static func realDiv<T: Numeric>(
  x: Tensor<T>,
  y: Tensor<T>
) -> Tensor<T> {
  return #tfop("RealDiv",
    x,
    y,
    T: T.self)
}

// Computes the reciprocal of x element-wise.
//
// I.e., \\(y = 1 / x\\).
@_inlineable @inline(__always)
public static func reciprocal<T: Numeric>(
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Reciprocal",
    x,
    T: T.self)
}

// Computes the gradient for the inverse of `x` wrt its input.
//
// Specifically, `grad = -dy * y*y`, where `y = 1/x`, and `dy`
// is the corresponding input gradient.
@_inlineable @inline(__always)
public static func reciprocalGrad<T: BinaryFloatingPoint>(
  y: Tensor<T>,
  dy: Tensor<T>
) -> Tensor<T> {
  return #tfop("ReciprocalGrad",
    y,
    dy,
    T: T.self)
}

// Creates or finds a child frame, and makes `data` available to the child frame.
//
// The unique `frame_name` is used by the `Executor` to identify frames. If
// `is_constant` is true, `output` is a constant in the child frame; otherwise
// it may be changed in the child frame. At most `parallel_iterations` iterations
// are run in parallel in the child frame.
//
// - Parameter data: The tensor to be made available to the child frame.
//
// - Attrs:
//   - frame_name: The name of the child frame.
//   - is_constant: If true, the output is constant within the child frame.
//   - parallel_iterations: The number of iterations allowed to run in parallel.
//
// - Output output: The same tensor as `data`.
@_inlineable @inline(__always)
public static func refEnter<T: Numeric>(
  data: Tensor<T>,
  frameName: String,
  isConstant: Bool = false,
  parallelIterations: Int = 10
) -> Tensor<T> {
  return #tfop("RefEnter",
    data,
    T: T.self,
    frame_name: frameName,
    is_constant: isConstant,
    parallel_iterations: parallelIterations)
}

// Exits the current frame to its parent frame.
//
// Exit makes its input `data` available to the parent frame.
//
// - Parameter data: The tensor to be made available to the parent frame.
//
// - Output output: The same tensor as `data`.
@_inlineable @inline(__always)
public static func refExit<T: Numeric>(
  data: Tensor<T>
) -> Tensor<T> {
  return #tfop("RefExit",
    data,
    T: T.self)
}

// Return the same ref tensor as the input ref tensor.
@_inlineable @inline(__always)
public static func refIdentity<T: Numeric>(
  input: Tensor<T>
) -> Tensor<T> {
  return #tfop("RefIdentity",
    input,
    T: T.self)
}

@_inlineable @inline(__always)
public static func refIn<T: Numeric>(
  a: Tensor<T>
) {
  return #tfop("RefIn",
    a,
    T: T.self)
}

@_inlineable @inline(__always)
public static func refInputFloatInput(
  a: Tensor<Float>,
  b: Tensor<Float>
) {
  return #tfop("RefInputFloatInput",
    a,
    b)
}

@_inlineable @inline(__always)
public static func refInputFloatInputIntOutput(
  a: Tensor<Float>,
  b: Tensor<Float>
) -> Tensor<Int32> {
  return #tfop("RefInputFloatInputIntOutput",
    a,
    b)
}

@_inlineable @inline(__always)
public static func refInputIntInput(
  a: Tensor<Int32>,
  b: Tensor<Int32>
) {
  return #tfop("RefInputIntInput",
    a,
    b)
}

// Forwards the value of an available tensor from `inputs` to `output`.
//
// `Merge` waits for at least one of the tensors in `inputs` to become available.
// It is usually combined with `Switch` to implement branching.
//
// `Merge` forwards the first tensor for become available to `output`, and sets
// `value_index` to its index in `inputs`.
//
// - Parameter inputs: The input tensors, exactly one of which will become available.
//
// - Outputs:
//   - output: Will be set to the available input tensor.
//   - value_index: The index of the chosen input tensor in `inputs`.
@_inlineable @inline(__always)
public static func refMerge<T: Numeric>(
  inputs: [Tensor<T>]
) -> (Tensor<T>, Tensor<Int32>) {
  return #tfop("RefMerge",
    inputs,
    T: T.self)
}

// Makes its input available to the next iteration.
//
// - Parameter data: The tensor to be made available to the next iteration.
//
// - Output output: The same tensor as `data`.
@_inlineable @inline(__always)
public static func refNextIteration<T: Numeric>(
  data: Tensor<T>
) -> Tensor<T> {
  return #tfop("RefNextIteration",
    data,
    T: T.self)
}

@_inlineable @inline(__always)
public static func refOut<T: Numeric>(
) -> Tensor<T> {
  return #tfop("RefOut",
    T: T.self)
}

@_inlineable @inline(__always)
public static func refOutput(
) -> Tensor<Int32> {
  return #tfop("RefOutput")
}

@_inlineable @inline(__always)
public static func refOutputFloatOutput(
) -> (Tensor<Float>, Tensor<Float>) {
  return #tfop("RefOutputFloatOutput")
}

// Forwards the `index`th element of `inputs` to `output`.
//
// - Parameters:
//   - index: A scalar that determines the input that gets selected.
//   - inputs: A list of ref tensors, one of which will be forwarded to `output`.
//
// - Output output: The forwarded tensor.
@_inlineable @inline(__always)
public static func refSelect<T: Numeric>(
  index: Tensor<Int32>,
  inputs: [Tensor<T>]
) -> Tensor<T> {
  return #tfop("RefSelect",
    index,
    inputs,
    T: T.self)
}

// Forwards the ref tensor `data` to the output port determined by `pred`.
//
// If `pred` is true, the `data` input is forwarded to `output_true`. Otherwise,
// the data goes to `output_false`.
//
// See also `Switch` and `Merge`.
//
// - Parameters:
//   - data: The ref tensor to be forwarded to the appropriate output.
//   - pred: A scalar that specifies which output port will receive data.
//
// - Outputs:
//   - output_false: If `pred` is false, data will be forwarded to this output.
//   - output_true: If `pred` is true, data will be forwarded to this output.
@_inlineable @inline(__always)
public static func refSwitch<T: Numeric>(
  data: Tensor<T>,
  pred: Tensor<Bool>
) -> (Tensor<T>, Tensor<T>) {
  return #tfop("RefSwitch",
    data,
    pred,
    T: T.self)
}

// Computes rectified linear: `max(features, 0)`.
@_inlineable @inline(__always)
public static func relu<T: Numeric>(
  features: Tensor<T>
) -> Tensor<T> {
  return #tfop("Relu",
    features,
    T: T.self)
}

// Computes rectified linear 6: `min(max(features, 0), 6)`.
@_inlineable @inline(__always)
public static func relu6<T: Numeric>(
  features: Tensor<T>
) -> Tensor<T> {
  return #tfop("Relu6",
    features,
    T: T.self)
}

// Computes rectified linear 6 gradients for a Relu6 operation.
//
// - Parameters:
//   - gradients: The backpropagated gradients to the corresponding Relu6 operation.
//   - features: The features passed as input to the corresponding Relu6 operation, or
//     its output; using either one produces the same result.
//
// - Output backprops: The gradients:
//   `gradients * (features > 0) * (features < 6)`.
@_inlineable @inline(__always)
public static func relu6Grad<T: Numeric>(
  gradients: Tensor<T>,
  features: Tensor<T>
) -> Tensor<T> {
  return #tfop("Relu6Grad",
    gradients,
    features,
    T: T.self)
}

// Computes rectified linear gradients for a Relu operation.
//
// - Parameters:
//   - gradients: The backpropagated gradients to the corresponding Relu operation.
//   - features: The features passed as input to the corresponding Relu operation, OR
//     the outputs of that operation (both work equivalently).
//
// - Output backprops: `gradients * (features > 0)`.
@_inlineable @inline(__always)
public static func reluGrad<T: Numeric>(
  gradients: Tensor<T>,
  features: Tensor<T>
) -> Tensor<T> {
  return #tfop("ReluGrad",
    gradients,
    features,
    T: T.self)
}

// Execute a sub graph on a remote processor.
//
// The graph specifications(such as graph itself, input tensors and output names)
// are stored as a serialized protocol buffer of RemoteFusedGraphExecuteInfo
// as serialized_remote_fused_graph_execute_info.
// The specifications will be passed to a dedicated registered
// remote fused graph executor.  The executor will send the graph specifications
// to a remote processor and execute that graph.  The execution results
// will be passed to consumer nodes as outputs of this node.
//
// - Parameter inputs: Arbitrary number of tensors with arbitrary data types
//
// - Attr serialized_remote_fused_graph_execute_info: Serialized protocol buffer
//   of RemoteFusedGraphExecuteInfo which contains graph specifications.
//
// - Output outputs: Arbitrary number of tensors with arbitrary data types
@_inlineable @inline(__always)
public static func remoteFusedGraphExecute<Tinputs: Numeric, Toutputs: Numeric>(
  inputs: [Tensor<Tinputs>],
  serializedRemoteFusedGraphExecuteInfo: String
) -> [Tensor<Toutputs>] {
  return #tfop("RemoteFusedGraphExecute",
    inputs,
    serialized_remote_fused_graph_execute_info: serializedRemoteFusedGraphExecuteInfo)
}

// Given a quantized tensor described by (input, input_min, input_max), outputs a
//
// range that covers the actual values present in that tensor.  This op is
// typically used to produce the requested_output_min and requested_output_max for
// Requantize.
//
// - Parameters:
//   - input_min: The float value that the minimum quantized input value represents.
//   - input_max: The float value that the maximum quantized input value represents.
//
// - Attr Tinput: The type of the input.
//
// - Outputs:
//   - output_min: The computed min output.
//   - output_max: the computed max output.
@_inlineable @inline(__always)
public static func requantizationRange<Tinput: Numeric>(
  input: Tensor<Tinput>,
  inputMin: Tensor<Float>,
  inputMax: Tensor<Float>
) -> (Tensor<Float>, Tensor<Float>) {
  return #tfop("RequantizationRange",
    input,
    inputMin,
    inputMax,
    Tinput: Tinput.self)
}

// Convert the quantized 'input' tensor into a lower-precision 'output', using the
//
// output range specified with 'requested_output_min' and 'requested_output_max'.
//
// [input_min, input_max] are scalar floats that specify the range for the float
// interpretation of the 'input' data. For example, if input_min is -1.0f and
// input_max is 1.0f, and we are dealing with quint16 quantized data, then a 0
// value in the 16-bit data should be interpreted as -1.0f, and a 65535 means 1.0f.
//
// - Parameters:
//   - input_min: The float value that the minimum quantized input value represents.
//   - input_max: The float value that the maximum quantized input value represents.
//   - requested_output_min: The float value that the minimum quantized output value represents.
//   - requested_output_max: The float value that the maximum quantized output value represents.
//
// - Attrs:
//   - Tinput: The type of the input.
//   - out_type: The type of the output. Should be a lower bit depth than Tinput.
//
// - Outputs:
//   - output_min: The requested_output_min value is copied into this output.
//   - output_max: The requested_output_max value is copied into this output.
@_inlineable @inline(__always)
public static func requantize<Tinput: Numeric, Out_type: Numeric>(
  input: Tensor<Tinput>,
  inputMin: Tensor<Float>,
  inputMax: Tensor<Float>,
  requestedOutputMin: Tensor<Float>,
  requestedOutputMax: Tensor<Float>
) -> (Tensor<Out_type>, Tensor<Float>, Tensor<Float>) {
  return #tfop("Requantize",
    input,
    inputMin,
    inputMax,
    requestedOutputMin,
    requestedOutputMax,
    Tinput: Tinput.self,
    Out_type: Out_type.self)
}

@_inlineable @inline(__always)
public static func requiresOlderGraphVersion(
) -> Tensor<Int32> {
  return #tfop("RequiresOlderGraphVersion")
}

@_inlineable @inline(__always)
public static func reservedAttr(
  range: Int
) {
  return #tfop("ReservedAttr",
    range: range)
}

@_inlineable @inline(__always)
public static func reservedInput(
  input: Tensor<Int32>
) {
  return #tfop("ReservedInput",
    input)
}

// Reshapes a tensor.
//
// Given `tensor`, this operation returns a tensor that has the same values
// as `tensor` with shape `shape`.
//
// If one component of `shape` is the special value -1, the size of that dimension
// is computed so that the total size remains constant.  In particular, a `shape`
// of `[-1]` flattens into 1-D.  At most one component of `shape` can be -1.
//
// If `shape` is 1-D or higher, then the operation returns a tensor with shape
// `shape` filled with the values of `tensor`. In this case, the number of elements
// implied by `shape` must be the same as the number of elements in `tensor`.
//
// For example:
//
// ```
// # tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
// # tensor 't' has shape [9]
// reshape(t, [3, 3]) ==> [[1, 2, 3],
//                         [4, 5, 6],
//                         [7, 8, 9]]
//
// # tensor 't' is [[[1, 1], [2, 2]],
// #                [[3, 3], [4, 4]]]
// # tensor 't' has shape [2, 2, 2]
// reshape(t, [2, 4]) ==> [[1, 1, 2, 2],
//                         [3, 3, 4, 4]]
//
// # tensor 't' is [[[1, 1, 1],
// #                 [2, 2, 2]],
// #                [[3, 3, 3],
// #                 [4, 4, 4]],
// #                [[5, 5, 5],
// #                 [6, 6, 6]]]
// # tensor 't' has shape [3, 2, 3]
// # pass '[-1]' to flatten 't'
// reshape(t, [-1]) ==> [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]
//
// # -1 can also be used to infer the shape
//
// # -1 is inferred to be 9:
// reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
//                          [4, 4, 4, 5, 5, 5, 6, 6, 6]]
// # -1 is inferred to be 2:
// reshape(t, [-1, 9]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
//                          [4, 4, 4, 5, 5, 5, 6, 6, 6]]
// # -1 is inferred to be 3:
// reshape(t, [ 2, -1, 3]) ==> [[[1, 1, 1],
//                               [2, 2, 2],
//                               [3, 3, 3]],
//                              [[4, 4, 4],
//                               [5, 5, 5],
//                               [6, 6, 6]]]
//
// # tensor 't' is [7]
// # shape `[]` reshapes to a scalar
// reshape(t, []) ==> 7
// ```
//
// - Parameter shape: Defines the shape of the output tensor.
@_inlineable @inline(__always)
public static func reshape<T: Numeric, Tshape: BinaryInteger>(
  tensor: Tensor<T>,
  shape: Tensor<Tshape>
) -> Tensor<T> {
  return #tfop("Reshape",
    tensor,
    shape,
    T: T.self,
    Tshape: Tshape.self)
}

// Resize `images` to `size` using area interpolation.
//
// Input images can be of different types but output images are always float.
//
// The range of pixel values for the output image might be slightly different
// from the range for the input image because of limited numerical precision.
// To guarantee an output range, for example `[0.0, 1.0]`, apply
// `tf.clip_by_value` to the output.
//
// Each output pixel is computed by first transforming the pixel's footprint into
// the input tensor and then averaging the pixels that intersect the footprint. An
// input pixel's contribution to the average is weighted by the fraction of its
// area that intersects the footprint.  This is the same as OpenCV's INTER_AREA.
//
// - Parameters:
//   - images: 4-D with shape `[batch, height, width, channels]`.
//   - size: = A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
//     new size for the images.
//
// - Attr align_corners: If true, the centers of the 4 corner pixels of the input and output tensors are
//   aligned, preserving the values at the corner pixels. Defaults to false.
//
// - Output resized_images: 4-D with shape
//   `[batch, new_height, new_width, channels]`.
@_inlineable @inline(__always)
public static func resizeArea<T: Numeric>(
  images: Tensor<T>,
  size: Tensor<Int32>,
  alignCorners: Bool = false
) -> Tensor<Float> {
  return #tfop("ResizeArea",
    images,
    size,
    T: T.self,
    align_corners: alignCorners)
}

// Resize `images` to `size` using bicubic interpolation.
//
// Input images can be of different types but output images are always float.
//
// - Parameters:
//   - images: 4-D with shape `[batch, height, width, channels]`.
//   - size: = A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
//     new size for the images.
//
// - Attr align_corners: If true, the centers of the 4 corner pixels of the input and output tensors are
//   aligned, preserving the values at the corner pixels. Defaults to false.
//
// - Output resized_images: 4-D with shape
//   `[batch, new_height, new_width, channels]`.
@_inlineable @inline(__always)
public static func resizeBicubic<T: Numeric>(
  images: Tensor<T>,
  size: Tensor<Int32>,
  alignCorners: Bool = false
) -> Tensor<Float> {
  return #tfop("ResizeBicubic",
    images,
    size,
    T: T.self,
    align_corners: alignCorners)
}

// Computes the gradient of bicubic interpolation.
//
// - Parameters:
//   - grads: 4-D with shape `[batch, height, width, channels]`.
//   - original_image: 4-D with shape `[batch, orig_height, orig_width, channels]`,
//     The image tensor that was resized.
//
// - Attr align_corners: If true, the centers of the 4 corner pixels of the input and grad tensors are
//   aligned. Defaults to false.
//
// - Output output: 4-D with shape `[batch, orig_height, orig_width, channels]`.
//   Gradients with respect to the input image. Input image must have been
//   float or double.
@_inlineable @inline(__always)
public static func resizeBicubicGrad<T: BinaryFloatingPoint>(
  grads: Tensor<Float>,
  originalImage: Tensor<T>,
  alignCorners: Bool = false
) -> Tensor<T> {
  return #tfop("ResizeBicubicGrad",
    grads,
    originalImage,
    T: T.self,
    align_corners: alignCorners)
}

// Resize `images` to `size` using bilinear interpolation.
//
// Input images can be of different types but output images are always float.
//
// - Parameters:
//   - images: 4-D with shape `[batch, height, width, channels]`.
//   - size: = A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
//     new size for the images.
//
// - Attr align_corners: If true, the centers of the 4 corner pixels of the input and output tensors are
//   aligned, preserving the values at the corner pixels. Defaults to false.
//
// - Output resized_images: 4-D with shape
//   `[batch, new_height, new_width, channels]`.
@_inlineable @inline(__always)
public static func resizeBilinear<T: Numeric>(
  images: Tensor<T>,
  size: Tensor<Int32>,
  alignCorners: Bool = false
) -> Tensor<Float> {
  return #tfop("ResizeBilinear",
    images,
    size,
    T: T.self,
    align_corners: alignCorners)
}

// Computes the gradient of bilinear interpolation.
//
// - Parameters:
//   - grads: 4-D with shape `[batch, height, width, channels]`.
//   - original_image: 4-D with shape `[batch, orig_height, orig_width, channels]`,
//     The image tensor that was resized.
//
// - Attr align_corners: If true, the centers of the 4 corner pixels of the input and grad tensors are
//   aligned. Defaults to false.
//
// - Output output: 4-D with shape `[batch, orig_height, orig_width, channels]`.
//   Gradients with respect to the input image. Input image must have been
//   float or double.
@_inlineable @inline(__always)
public static func resizeBilinearGrad<T: BinaryFloatingPoint>(
  grads: Tensor<Float>,
  originalImage: Tensor<T>,
  alignCorners: Bool = false
) -> Tensor<T> {
  return #tfop("ResizeBilinearGrad",
    grads,
    originalImage,
    T: T.self,
    align_corners: alignCorners)
}

// Resize `images` to `size` using nearest neighbor interpolation.
//
// - Parameters:
//   - images: 4-D with shape `[batch, height, width, channels]`.
//   - size: = A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
//     new size for the images.
//
// - Attr align_corners: If true, the centers of the 4 corner pixels of the input and output tensors are
//   aligned, preserving the values at the corner pixels. Defaults to false.
//
// - Output resized_images: 4-D with shape
//   `[batch, new_height, new_width, channels]`.
@_inlineable @inline(__always)
public static func resizeNearestNeighbor<T: Numeric>(
  images: Tensor<T>,
  size: Tensor<Int32>,
  alignCorners: Bool = false
) -> Tensor<T> {
  return #tfop("ResizeNearestNeighbor",
    images,
    size,
    T: T.self,
    align_corners: alignCorners)
}

// Computes the gradient of nearest neighbor interpolation.
//
// - Parameters:
//   - grads: 4-D with shape `[batch, height, width, channels]`.
//   - size: = A 1-D int32 Tensor of 2 elements: `orig_height, orig_width`. The
//     original input size.
//
// - Attr align_corners: If true, the centers of the 4 corner pixels of the input and grad tensors are
//   aligned. Defaults to false.
//
// - Output output: 4-D with shape `[batch, orig_height, orig_width, channels]`. Gradients
//   with respect to the input image.
@_inlineable @inline(__always)
public static func resizeNearestNeighborGrad<T: Numeric>(
  grads: Tensor<T>,
  size: Tensor<Int32>,
  alignCorners: Bool = false
) -> Tensor<T> {
  return #tfop("ResizeNearestNeighborGrad",
    grads,
    size,
    T: T.self,
    align_corners: alignCorners)
}

@_inlineable @inline(__always)
public static func restrict<T: Numeric>(
  a: Tensor<T>
) -> Tensor<T> {
  return #tfop("Restrict",
    a,
    T: T.self)
}

// Reverses specific dimensions of a tensor.
//
// Given a `tensor`, and a `bool` tensor `dims` representing the dimensions
// of `tensor`, this operation reverses each dimension i of `tensor` where
// `dims[i]` is `True`.
//
// `tensor` can have up to 8 dimensions. The number of dimensions
// of `tensor` must equal the number of elements in `dims`. In other words:
//
// `rank(tensor) = size(dims)`
//
// For example:
//
// ```
// # tensor 't' is [[[[ 0,  1,  2,  3],
// #                  [ 4,  5,  6,  7],
// #                  [ 8,  9, 10, 11]],
// #                 [[12, 13, 14, 15],
// #                  [16, 17, 18, 19],
// #                  [20, 21, 22, 23]]]]
// # tensor 't' shape is [1, 2, 3, 4]
//
// # 'dims' is [False, False, False, True]
// reverse(t, dims) ==> [[[[ 3,  2,  1,  0],
//                         [ 7,  6,  5,  4],
//                         [ 11, 10, 9, 8]],
//                        [[15, 14, 13, 12],
//                         [19, 18, 17, 16],
//                         [23, 22, 21, 20]]]]
//
// # 'dims' is [False, True, False, False]
// reverse(t, dims) ==> [[[[12, 13, 14, 15],
//                         [16, 17, 18, 19],
//                         [20, 21, 22, 23]
//                        [[ 0,  1,  2,  3],
//                         [ 4,  5,  6,  7],
//                         [ 8,  9, 10, 11]]]]
//
// # 'dims' is [False, False, True, False]
// reverse(t, dims) ==> [[[[8, 9, 10, 11],
//                         [4, 5, 6, 7],
//                         [0, 1, 2, 3]]
//                        [[20, 21, 22, 23],
//                         [16, 17, 18, 19],
//                         [12, 13, 14, 15]]]]
// ```
//
// - Parameters:
//   - tensor: Up to 8-D.
//   - dims: 1-D. The dimensions to reverse.
//
// - Output output: The same shape as `tensor`.
@_inlineable @inline(__always)
public static func reverse<T: Numeric>(
  tensor: Tensor<T>,
  dims: Tensor<Bool>
) -> Tensor<T> {
  return #tfop("Reverse",
    tensor,
    dims,
    T: T.self)
}

// Reverses variable length slices.
//
// This op first slices `input` along the dimension `batch_dim`, and for each
// slice `i`, reverses the first `seq_lengths[i]` elements along
// the dimension `seq_dim`.
//
// The elements of `seq_lengths` must obey `seq_lengths[i] <= input.dims[seq_dim]`,
// and `seq_lengths` must be a vector of length `input.dims[batch_dim]`.
//
// The output slice `i` along dimension `batch_dim` is then given by input
// slice `i`, with the first `seq_lengths[i]` slices along dimension
// `seq_dim` reversed.
//
// For example:
//
// ```
// # Given this:
// batch_dim = 0
// seq_dim = 1
// input.dims = (4, 8, ...)
// seq_lengths = [7, 2, 3, 5]
//
// # then slices of input are reversed on seq_dim, but only up to seq_lengths:
// output[0, 0:7, :, ...] = input[0, 7:0:-1, :, ...]
// output[1, 0:2, :, ...] = input[1, 2:0:-1, :, ...]
// output[2, 0:3, :, ...] = input[2, 3:0:-1, :, ...]
// output[3, 0:5, :, ...] = input[3, 5:0:-1, :, ...]
//
// # while entries past seq_lens are copied through:
// output[0, 7:, :, ...] = input[0, 7:, :, ...]
// output[1, 2:, :, ...] = input[1, 2:, :, ...]
// output[2, 3:, :, ...] = input[2, 3:, :, ...]
// output[3, 2:, :, ...] = input[3, 2:, :, ...]
// ```
//
// In contrast, if:
//
// ```
// # Given this:
// batch_dim = 2
// seq_dim = 0
// input.dims = (8, ?, 4, ...)
// seq_lengths = [7, 2, 3, 5]
//
// # then slices of input are reversed on seq_dim, but only up to seq_lengths:
// output[0:7, :, 0, :, ...] = input[7:0:-1, :, 0, :, ...]
// output[0:2, :, 1, :, ...] = input[2:0:-1, :, 1, :, ...]
// output[0:3, :, 2, :, ...] = input[3:0:-1, :, 2, :, ...]
// output[0:5, :, 3, :, ...] = input[5:0:-1, :, 3, :, ...]
//
// # while entries past seq_lens are copied through:
// output[7:, :, 0, :, ...] = input[7:, :, 0, :, ...]
// output[2:, :, 1, :, ...] = input[2:, :, 1, :, ...]
// output[3:, :, 2, :, ...] = input[3:, :, 2, :, ...]
// output[2:, :, 3, :, ...] = input[2:, :, 3, :, ...]
// ```
//
// - Parameters:
//   - input: The input to reverse.
//   - seq_lengths: 1-D with length `input.dims(batch_dim)` and
//     `max(seq_lengths) <= input.dims(seq_dim)`
//
// - Attrs:
//   - seq_dim: The dimension which is partially reversed.
//   - batch_dim: The dimension along which reversal is performed.
//
// - Output output: The partially reversed input. It has the same shape as `input`.
@_inlineable @inline(__always)
public static func reverseSequence<T: Numeric, Tlen: BinaryInteger>(
  input: Tensor<T>,
  seqLengths: Tensor<Tlen>,
  seqDim: Int,
  batchDim: Int = 0
) -> Tensor<T> {
  return #tfop("ReverseSequence",
    input,
    seqLengths,
    T: T.self,
    Tlen: Tlen.self,
    seq_dim: seqDim,
    batch_dim: batchDim)
}

// Reverses specific dimensions of a tensor.
//
// NOTE `tf.reverse` has now changed behavior in preparation for 1.0.
// `tf.reverse_v2` is currently an alias that will be deprecated before TF 1.0.
//
// Given a `tensor`, and a `int32` tensor `axis` representing the set of
// dimensions of `tensor` to reverse. This operation reverses each dimension
// `i` for which there exists `j` s.t. `axis[j] == i`.
//
// `tensor` can have up to 8 dimensions. The number of dimensions specified
// in `axis` may be 0 or more entries. If an index is specified more than
// once, a InvalidArgument error is raised.
//
// For example:
//
// ```
// # tensor 't' is [[[[ 0,  1,  2,  3],
// #                  [ 4,  5,  6,  7],
// #                  [ 8,  9, 10, 11]],
// #                 [[12, 13, 14, 15],
// #                  [16, 17, 18, 19],
// #                  [20, 21, 22, 23]]]]
// # tensor 't' shape is [1, 2, 3, 4]
//
// # 'dims' is [3] or 'dims' is [-1]
// reverse(t, dims) ==> [[[[ 3,  2,  1,  0],
//                         [ 7,  6,  5,  4],
//                         [ 11, 10, 9, 8]],
//                        [[15, 14, 13, 12],
//                         [19, 18, 17, 16],
//                         [23, 22, 21, 20]]]]
//
// # 'dims' is '[1]' (or 'dims' is '[-3]')
// reverse(t, dims) ==> [[[[12, 13, 14, 15],
//                         [16, 17, 18, 19],
//                         [20, 21, 22, 23]
//                        [[ 0,  1,  2,  3],
//                         [ 4,  5,  6,  7],
//                         [ 8,  9, 10, 11]]]]
//
// # 'dims' is '[2]' (or 'dims' is '[-2]')
// reverse(t, dims) ==> [[[[8, 9, 10, 11],
//                         [4, 5, 6, 7],
//                         [0, 1, 2, 3]]
//                        [[20, 21, 22, 23],
//                         [16, 17, 18, 19],
//                         [12, 13, 14, 15]]]]
// ```
//
// - Parameters:
//   - tensor: Up to 8-D.
//   - axis: 1-D. The indices of the dimensions to reverse. Must be in the range
//     `[-rank(tensor), rank(tensor))`.
//
// - Output output: The same shape as `tensor`.
@_inlineable @inline(__always)
public static func reverseV2<Tidx: BinaryInteger, T: Numeric>(
  tensor: Tensor<T>,
  axis: Tensor<Tidx>
) -> Tensor<T> {
  return #tfop("ReverseV2",
    tensor,
    axis,
    Tidx: Tidx.self,
    T: T.self)
}

// Elementwise computes the bitwise right-shift of `x` and `y`.
//
// Performs a logical shift for unsigned integer types, and an arithmetic shift
// for signed integer types.
//
// If `y` is negative, or greater than or equal to than the width of `x` in bits
// the result is implementation defined.
@_inlineable @inline(__always)
public static func rightShift<T: BinaryInteger>(
  x: Tensor<T>,
  y: Tensor<T>
) -> Tensor<T> {
  return #tfop("RightShift",
    x,
    y,
    T: T.self)
}

// Returns element-wise integer closest to x.
//
// If the result is midway between two representable values,
// the even representable is chosen.
// For example:
//
// ```
// rint(-1.5) ==> -2.0
// rint(0.5000001) ==> 1.0
// rint([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]) ==> [-2., -2., -0., 0., 2., 2., 2.]
// ```
@_inlineable @inline(__always)
public static func rint<T: BinaryFloatingPoint>(
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Rint",
    x,
    T: T.self)
}

// Rolls the elements of a tensor along an axis.
//
// The elements are shifted positively (towards larger indices) by the offset of
// `shift` along the dimension of `axis`. Negative `shift` values will shift
// elements in the opposite direction. Elements that roll passed the last position
// will wrap around to the first and vice versa. Multiple shifts along multiple
// axes may be specified.
//
// For example:
//
// ```
// # 't' is [0, 1, 2, 3, 4]
// roll(t, shift=2, axis=0) ==> [3, 4, 0, 1, 2]
//
// # shifting along multiple dimensions
// # 't' is [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
// roll(t, shift=[1, -2], axis=[0, 1]) ==> [[7, 8, 9, 5, 6], [2, 3, 4, 0, 1]]
//
// # shifting along the same axis multiple times
// # 't' is [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
// roll(t, shift=[2, -3], axis=[1, 1]) ==> [[1, 2, 3, 4, 0], [6, 7, 8, 9, 5]]
// ```
//
// - Parameters:
//   - shift: Dimension must be 0-D or 1-D. `shift[i]` specifies the number of places by which
//     elements are shifted positively (towards larger indices) along the dimension
//     specified by `axis[i]`. Negative shifts will roll the elements in the opposite
//     direction.
//   - axis: Dimension must be 0-D or 1-D. `axis[i]` specifies the dimension that the shift
//     `shift[i]` should occur. If the same axis is referenced more than once, the
//     total shift for that axis will be the sum of all the shifts that belong to that
//     axis.
//
// - Output output: Has the same shape and size as the input. The elements are shifted
//   positively (towards larger indices) by the offsets of `shift` along the
//   dimensions of `axis`.
@_inlineable @inline(__always)
public static func roll<T: Numeric, Tshift: BinaryInteger, Taxis: BinaryInteger>(
  input: Tensor<T>,
  shift: Tensor<Tshift>,
  axis: Tensor<Taxis>
) -> Tensor<T> {
  return #tfop("Roll",
    input,
    shift,
    axis,
    T: T.self,
    Tshift: Tshift.self,
    Taxis: Taxis.self)
}

// Rounds the values of a tensor to the nearest integer, element-wise.
//
// Rounds half to even.  Also known as bankers rounding. If you want to round
// according to the current system rounding mode use std::cint.
@_inlineable @inline(__always)
public static func round<T: Numeric>(
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Round",
    x,
    T: T.self)
}

// Computes reciprocal of square root of x element-wise.
//
// I.e., \\(y = 1 / \sqrt{x}\\).
@_inlineable @inline(__always)
public static func rsqrt<T: BinaryFloatingPoint>(
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Rsqrt",
    x,
    T: T.self)
}

// Computes the gradient for the rsqrt of `x` wrt its input.
//
// Specifically, `grad = dy * -0.5 * y^3`, where `y = rsqrt(x)`, and `dy`
// is the corresponding input gradient.
@_inlineable @inline(__always)
public static func rsqrtGrad<T: BinaryFloatingPoint>(
  y: Tensor<T>,
  dy: Tensor<T>
) -> Tensor<T> {
  return #tfop("RsqrtGrad",
    y,
    dy,
    T: T.self)
}

// Generate a single randomly distorted bounding box for an image.
//
// Bounding box annotations are often supplied in addition to ground-truth labels
// in image recognition or object localization tasks. A common technique for
// training such a system is to randomly distort an image while preserving
// its content, i.e. *data augmentation*. This Op outputs a randomly distorted
// localization of an object, i.e. bounding box, given an `image_size`,
// `bounding_boxes` and a series of constraints.
//
// The output of this Op is a single bounding box that may be used to crop the
// original image. The output is returned as 3 tensors: `begin`, `size` and
// `bboxes`. The first 2 tensors can be fed directly into `tf.slice` to crop the
// image. The latter may be supplied to `tf.image.draw_bounding_boxes` to visualize
// what the bounding box looks like.
//
// Bounding boxes are supplied and returned as `[y_min, x_min, y_max, x_max]`. The
// bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
// height of the underlying image.
//
// For example,
//
// ```python
//     # Generate a single distorted bounding box.
//     begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
//         tf.shape(image),
//         bounding_boxes=bounding_boxes)
//
//     # Draw the bounding box in an image summary.
//     image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
//                                                   bbox_for_draw)
//     tf.summary.image('images_with_box', image_with_box)
//
//     # Employ the bounding box to distort the image.
//     distorted_image = tf.slice(image, begin, size)
// ```
//
// Note that if no bounding box information is available, setting
// `use_image_if_no_bounding_boxes = true` will assume there is a single implicit
// bounding box covering the whole image. If `use_image_if_no_bounding_boxes` is
// false and no bounding boxes are supplied, an error is raised.
//
// - Parameters:
//   - image_size: 1-D, containing `[height, width, channels]`.
//   - bounding_boxes: 3-D with shape `[batch, N, 4]` describing the N bounding boxes
//     associated with the image.
//
// - Attrs:
//   - seed: If either `seed` or `seed2` are set to non-zero, the random number
//     generator is seeded by the given `seed`.  Otherwise, it is seeded by a random
//     seed.
//   - seed2: A second seed to avoid seed collision.
//   - min_object_covered: The cropped area of the image must contain at least this
//     fraction of any bounding box supplied. The value of this parameter should be
//     non-negative. In the case of 0, the cropped area does not need to overlap
//     any of the bounding boxes supplied.
//   - aspect_ratio_range: The cropped area of the image must have an aspect ratio =
//     width / height within this range.
//   - area_range: The cropped area of the image must contain a fraction of the
//     supplied image within in this range.
//   - max_attempts: Number of attempts at generating a cropped region of the image
//     of the specified constraints. After `max_attempts` failures, return the entire
//     image.
//   - use_image_if_no_bounding_boxes: Controls behavior if no bounding boxes supplied.
//     If true, assume an implicit bounding box covering the whole input. If false,
//     raise an error.
//
// - Outputs:
//   - begin: 1-D, containing `[offset_height, offset_width, 0]`. Provide as input to
//     `tf.slice`.
//   - size: 1-D, containing `[target_height, target_width, -1]`. Provide as input to
//     `tf.slice`.
//   - bboxes: 3-D with shape `[1, 1, 4]` containing the distorted bounding box.
//     Provide as input to `tf.image.draw_bounding_boxes`.
@_inlineable @inline(__always)
public static func sampleDistortedBoundingBox<T: BinaryInteger>(
  imageSize: Tensor<T>,
  boundingBoxes: Tensor<Float>,
  seed: Int = 0,
  seed2: Int = 0,
  minObjectCovered: Double = 0.1,
  aspectRatioRange: [Double],
  areaRange: [Double],
  maxAttempts: Int = 100,
  useImageIfNoBoundingBoxes: Bool = false
) -> (Tensor<T>, Tensor<T>, Tensor<Float>) {
  return #tfop("SampleDistortedBoundingBox",
    imageSize,
    boundingBoxes,
    T: T.self,
    seed: seed,
    seed2: seed2,
    min_object_covered: minObjectCovered,
    aspect_ratio_range: aspectRatioRange,
    area_range: areaRange,
    max_attempts: maxAttempts,
    use_image_if_no_bounding_boxes: useImageIfNoBoundingBoxes)
}

// Generate a single randomly distorted bounding box for an image.
//
// Bounding box annotations are often supplied in addition to ground-truth labels
// in image recognition or object localization tasks. A common technique for
// training such a system is to randomly distort an image while preserving
// its content, i.e. *data augmentation*. This Op outputs a randomly distorted
// localization of an object, i.e. bounding box, given an `image_size`,
// `bounding_boxes` and a series of constraints.
//
// The output of this Op is a single bounding box that may be used to crop the
// original image. The output is returned as 3 tensors: `begin`, `size` and
// `bboxes`. The first 2 tensors can be fed directly into `tf.slice` to crop the
// image. The latter may be supplied to `tf.image.draw_bounding_boxes` to visualize
// what the bounding box looks like.
//
// Bounding boxes are supplied and returned as `[y_min, x_min, y_max, x_max]`. The
// bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
// height of the underlying image.
//
// For example,
//
// ```python
//     # Generate a single distorted bounding box.
//     begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
//         tf.shape(image),
//         bounding_boxes=bounding_boxes)
//
//     # Draw the bounding box in an image summary.
//     image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
//                                                   bbox_for_draw)
//     tf.summary.image('images_with_box', image_with_box)
//
//     # Employ the bounding box to distort the image.
//     distorted_image = tf.slice(image, begin, size)
// ```
//
// Note that if no bounding box information is available, setting
// `use_image_if_no_bounding_boxes = true` will assume there is a single implicit
// bounding box covering the whole image. If `use_image_if_no_bounding_boxes` is
// false and no bounding boxes are supplied, an error is raised.
//
// - Parameters:
//   - image_size: 1-D, containing `[height, width, channels]`.
//   - bounding_boxes: 3-D with shape `[batch, N, 4]` describing the N bounding boxes
//     associated with the image.
//   - min_object_covered: The cropped area of the image must contain at least this
//     fraction of any bounding box supplied. The value of this parameter should be
//     non-negative. In the case of 0, the cropped area does not need to overlap
//     any of the bounding boxes supplied.
//
// - Attrs:
//   - seed: If either `seed` or `seed2` are set to non-zero, the random number
//     generator is seeded by the given `seed`.  Otherwise, it is seeded by a random
//     seed.
//   - seed2: A second seed to avoid seed collision.
//   - aspect_ratio_range: The cropped area of the image must have an aspect ratio =
//     width / height within this range.
//   - area_range: The cropped area of the image must contain a fraction of the
//     supplied image within in this range.
//   - max_attempts: Number of attempts at generating a cropped region of the image
//     of the specified constraints. After `max_attempts` failures, return the entire
//     image.
//   - use_image_if_no_bounding_boxes: Controls behavior if no bounding boxes supplied.
//     If true, assume an implicit bounding box covering the whole input. If false,
//     raise an error.
//
// - Outputs:
//   - begin: 1-D, containing `[offset_height, offset_width, 0]`. Provide as input to
//     `tf.slice`.
//   - size: 1-D, containing `[target_height, target_width, -1]`. Provide as input to
//     `tf.slice`.
//   - bboxes: 3-D with shape `[1, 1, 4]` containing the distorted bounding box.
//     Provide as input to `tf.image.draw_bounding_boxes`.
@_inlineable @inline(__always)
public static func sampleDistortedBoundingBoxV2<T: BinaryInteger>(
  imageSize: Tensor<T>,
  boundingBoxes: Tensor<Float>,
  minObjectCovered: Tensor<Float>,
  seed: Int = 0,
  seed2: Int = 0,
  aspectRatioRange: [Double],
  areaRange: [Double],
  maxAttempts: Int = 100,
  useImageIfNoBoundingBoxes: Bool = false
) -> (Tensor<T>, Tensor<T>, Tensor<Float>) {
  return #tfop("SampleDistortedBoundingBoxV2",
    imageSize,
    boundingBoxes,
    minObjectCovered,
    T: T.self,
    seed: seed,
    seed2: seed2,
    aspect_ratio_range: aspectRatioRange,
    area_range: areaRange,
    max_attempts: maxAttempts,
    use_image_if_no_bounding_boxes: useImageIfNoBoundingBoxes)
}

// Adds sparse updates to a variable reference.
//
// This operation computes
//
//     # Scalar indices
//     ref[indices, ...] += updates[...]
//
//     # Vector indices (for each i)
//     ref[indices[i], ...] += updates[i, ...]
//
//     # High rank indices (for each i, ..., j)
//     ref[indices[i, ..., j], ...] += updates[i, ..., j, ...]
//
// This operation outputs `ref` after the update is done.
// This makes it easier to chain operations that need to use the reset value.
//
// Duplicate entries are handled correctly: if multiple `indices` reference
// the same location, their contributions add.
//
// Requires `updates.shape = indices.shape + ref.shape[1:]` or `updates.shape = []`.
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="https://www.tensorflow.org/images/ScatterAdd.png" alt>
// </div>
//
// - Parameters:
//   - ref: Should be from a `Variable` node.
//   - indices: A tensor of indices into the first dimension of `ref`.
//   - updates: A tensor of updated values to add to `ref`.
//
// - Attr use_locking: If True, the addition will be protected by a lock;
//   otherwise the behavior is undefined, but may exhibit less contention.
//
// - Output output_ref: = Same as `ref`.  Returned as a convenience for operations that want
//   to use the updated values after the update is done.
@_inlineable @inline(__always)
public static func scatterAdd<T: Numeric, Tindices: BinaryInteger>(
  ref: Tensor<T>,
  indices: Tensor<Tindices>,
  updates: Tensor<T>,
  useLocking: Bool = false
) -> Tensor<T> {
  return #tfop("ScatterAdd",
    ref,
    indices,
    updates,
    T: T.self,
    Tindices: Tindices.self,
    use_locking: useLocking)
}

// Divides a variable reference by sparse updates.
//
// This operation computes
//
// ```python
//     # Scalar indices
//     ref[indices, ...] /= updates[...]
//
//     # Vector indices (for each i)
//     ref[indices[i], ...] /= updates[i, ...]
//
//     # High rank indices (for each i, ..., j)
//     ref[indices[i, ..., j], ...] /= updates[i, ..., j, ...]
// ```
//
// This operation outputs `ref` after the update is done.
// This makes it easier to chain operations that need to use the reset value.
//
// Duplicate entries are handled correctly: if multiple `indices` reference
// the same location, their contributions divide.
//
// Requires `updates.shape = indices.shape + ref.shape[1:]` or `updates.shape = []`.
//
// - Parameters:
//   - ref: Should be from a `Variable` node.
//   - indices: A tensor of indices into the first dimension of `ref`.
//   - updates: A tensor of values that `ref` is divided by.
//
// - Attr use_locking: If True, the operation will be protected by a lock;
//   otherwise the behavior is undefined, but may exhibit less contention.
//
// - Output output_ref: = Same as `ref`.  Returned as a convenience for operations that want
//   to use the updated values after the update is done.
@_inlineable @inline(__always)
public static func scatterDiv<T: Numeric, Tindices: BinaryInteger>(
  ref: Tensor<T>,
  indices: Tensor<Tindices>,
  updates: Tensor<T>,
  useLocking: Bool = false
) -> Tensor<T> {
  return #tfop("ScatterDiv",
    ref,
    indices,
    updates,
    T: T.self,
    Tindices: Tindices.self,
    use_locking: useLocking)
}

// Reduces sparse updates into a variable reference using the `max` operation.
//
// This operation computes
//
//     # Scalar indices
//     ref[indices, ...] = max(ref[indices, ...], updates[...])
//
//     # Vector indices (for each i)
//     ref[indices[i], ...] = max(ref[indices[i], ...], updates[i, ...])
//
//     # High rank indices (for each i, ..., j)
//     ref[indices[i, ..., j], ...] = max(ref[indices[i, ..., j], ...], updates[i, ..., j, ...])
//
// This operation outputs `ref` after the update is done.
// This makes it easier to chain operations that need to use the reset value.
//
// Duplicate entries are handled correctly: if multiple `indices` reference
// the same location, their contributions combine.
//
// Requires `updates.shape = indices.shape + ref.shape[1:]` or `updates.shape = []`.
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="https://www.tensorflow.org/images/ScatterAdd.png" alt>
// </div>
//
// - Parameters:
//   - ref: Should be from a `Variable` node.
//   - indices: A tensor of indices into the first dimension of `ref`.
//   - updates: A tensor of updated values to reduce into `ref`.
//
// - Attr use_locking: If True, the update will be protected by a lock;
//   otherwise the behavior is undefined, but may exhibit less contention.
//
// - Output output_ref: = Same as `ref`.  Returned as a convenience for operations that want
//   to use the updated values after the update is done.
@_inlineable @inline(__always)
public static func scatterMax<T: Numeric, Tindices: BinaryInteger>(
  ref: Tensor<T>,
  indices: Tensor<Tindices>,
  updates: Tensor<T>,
  useLocking: Bool = false
) -> Tensor<T> {
  return #tfop("ScatterMax",
    ref,
    indices,
    updates,
    T: T.self,
    Tindices: Tindices.self,
    use_locking: useLocking)
}

// Reduces sparse updates into a variable reference using the `min` operation.
//
// This operation computes
//
//     # Scalar indices
//     ref[indices, ...] = min(ref[indices, ...], updates[...])
//
//     # Vector indices (for each i)
//     ref[indices[i], ...] = min(ref[indices[i], ...], updates[i, ...])
//
//     # High rank indices (for each i, ..., j)
//     ref[indices[i, ..., j], ...] = min(ref[indices[i, ..., j], ...], updates[i, ..., j, ...])
//
// This operation outputs `ref` after the update is done.
// This makes it easier to chain operations that need to use the reset value.
//
// Duplicate entries are handled correctly: if multiple `indices` reference
// the same location, their contributions combine.
//
// Requires `updates.shape = indices.shape + ref.shape[1:]` or `updates.shape = []`.
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="https://www.tensorflow.org/images/ScatterAdd.png" alt>
// </div>
//
// - Parameters:
//   - ref: Should be from a `Variable` node.
//   - indices: A tensor of indices into the first dimension of `ref`.
//   - updates: A tensor of updated values to reduce into `ref`.
//
// - Attr use_locking: If True, the update will be protected by a lock;
//   otherwise the behavior is undefined, but may exhibit less contention.
//
// - Output output_ref: = Same as `ref`.  Returned as a convenience for operations that want
//   to use the updated values after the update is done.
@_inlineable @inline(__always)
public static func scatterMin<T: Numeric, Tindices: BinaryInteger>(
  ref: Tensor<T>,
  indices: Tensor<Tindices>,
  updates: Tensor<T>,
  useLocking: Bool = false
) -> Tensor<T> {
  return #tfop("ScatterMin",
    ref,
    indices,
    updates,
    T: T.self,
    Tindices: Tindices.self,
    use_locking: useLocking)
}

// Multiplies sparse updates into a variable reference.
//
// This operation computes
//
// ```python
//     # Scalar indices
//     ref[indices, ...] *= updates[...]
//
//     # Vector indices (for each i)
//     ref[indices[i], ...] *= updates[i, ...]
//
//     # High rank indices (for each i, ..., j)
//     ref[indices[i, ..., j], ...] *= updates[i, ..., j, ...]
// ```
//
// This operation outputs `ref` after the update is done.
// This makes it easier to chain operations that need to use the reset value.
//
// Duplicate entries are handled correctly: if multiple `indices` reference
// the same location, their contributions multiply.
//
// Requires `updates.shape = indices.shape + ref.shape[1:]` or `updates.shape = []`.
//
// - Parameters:
//   - ref: Should be from a `Variable` node.
//   - indices: A tensor of indices into the first dimension of `ref`.
//   - updates: A tensor of updated values to multiply to `ref`.
//
// - Attr use_locking: If True, the operation will be protected by a lock;
//   otherwise the behavior is undefined, but may exhibit less contention.
//
// - Output output_ref: = Same as `ref`.  Returned as a convenience for operations that want
//   to use the updated values after the update is done.
@_inlineable @inline(__always)
public static func scatterMul<T: Numeric, Tindices: BinaryInteger>(
  ref: Tensor<T>,
  indices: Tensor<Tindices>,
  updates: Tensor<T>,
  useLocking: Bool = false
) -> Tensor<T> {
  return #tfop("ScatterMul",
    ref,
    indices,
    updates,
    T: T.self,
    Tindices: Tindices.self,
    use_locking: useLocking)
}

// Scatter `updates` into a new tensor according to `indices`.
//
// Creates a new tensor by applying sparse `updates` to individual values or
// slices within a tensor (initially zero for numeric, empty for string) of
// the given `shape` according to indices.  This operator is the inverse of the
// @{tf.gather_nd} operator which extracts values or slices from a given tensor.
//
// **WARNING**: The order in which updates are applied is nondeterministic, so the
// output will be nondeterministic if `indices` contains duplicates.
//
// `indices` is an integer tensor containing indices into a new tensor of shape
// `shape`.  The last dimension of `indices` can be at most the rank of `shape`:
//
//     indices.shape[-1] <= shape.rank
//
// The last dimension of `indices` corresponds to indices into elements
// (if `indices.shape[-1] = shape.rank`) or slices
// (if `indices.shape[-1] < shape.rank`) along dimension `indices.shape[-1]` of
// `shape`.  `updates` is a tensor with shape
//
//     indices.shape[:-1] + shape[indices.shape[-1]:]
//
// The simplest form of scatter is to insert individual elements in a tensor by
// index. For example, say we want to insert 4 scattered elements in a rank-1
// tensor with 8 elements.
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="https://www.tensorflow.org/images/ScatterNd1.png" alt>
// </div>
//
// In Python, this scatter operation would look like this:
//
// ```python
//     indices = tf.constant([[4], [3], [1], [7]])
//     updates = tf.constant([9, 10, 11, 12])
//     shape = tf.constant([8])
//     scatter = tf.scatter_nd(indices, updates, shape)
//     with tf.Session() as sess:
//       print(sess.run(scatter))
// ```
//
// The resulting tensor would look like this:
//
//     [0, 11, 0, 10, 9, 0, 0, 12]
//
// We can also, insert entire slices of a higher rank tensor all at once. For
// example, if we wanted to insert two slices in the first dimension of a
// rank-3 tensor with two matrices of new values.
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="https://www.tensorflow.org/images/ScatterNd2.png" alt>
// </div>
//
// In Python, this scatter operation would look like this:
//
// ```python
//     indices = tf.constant([[0], [2]])
//     updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
//                             [7, 7, 7, 7], [8, 8, 8, 8]],
//                            [[5, 5, 5, 5], [6, 6, 6, 6],
//                             [7, 7, 7, 7], [8, 8, 8, 8]]])
//     shape = tf.constant([4, 4, 4])
//     scatter = tf.scatter_nd(indices, updates, shape)
//     with tf.Session() as sess:
//       print(sess.run(scatter))
// ```
//
// The resulting tensor would look like this:
//
//     [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
//      [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
//      [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
//      [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
//
// Note that on CPU, if an out of bound index is found, an error is returned.
// On GPU, if an out of bound index is found, the index is ignored.
//
// - Parameters:
//   - indices: Index tensor.
//   - updates: Updates to scatter into output.
//   - shape: 1-D. The shape of the resulting tensor.
//
// - Output output: A new tensor with the given shape and updates applied according
//   to the indices.
@_inlineable @inline(__always)
public static func scatterNd<T: Numeric, Tindices: BinaryInteger>(
  indices: Tensor<Tindices>,
  updates: Tensor<T>,
  shape: Tensor<Tindices>
) -> Tensor<T> {
  return #tfop("ScatterNd",
    indices,
    updates,
    shape,
    T: T.self,
    Tindices: Tindices.self)
}

// Applies sparse addition between `updates` and individual values or slices
//
// within a given variable according to `indices`.
//
// `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.
//
// `indices` must be integer tensor, containing indices into `ref`.
// It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.
//
// The innermost dimension of `indices` (with length `K`) corresponds to
// indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
// dimension of `ref`.
//
// `updates` is `Tensor` of rank `Q-1+P-K` with shape:
//
// ```
// [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
// ```
//
// For example, say we want to add 4 scattered elements to a rank-1 tensor to 8
// elements. In Python, that addition would look like this:
//
//     ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
//     indices = tf.constant([[4], [3], [1], [7]])
//     updates = tf.constant([9, 10, 11, 12])
//     add = tf.scatter_nd_add(ref, indices, updates)
//     with tf.Session() as sess:
//       print sess.run(add)
//
// The resulting update to ref would look like this:
//
//     [1, 13, 3, 14, 14, 6, 7, 20]
//
// See @{tf.scatter_nd} for more details about how to make updates to
// slices.
//
// - Parameters:
//   - ref: A mutable Tensor. Should be from a Variable node.
//   - indices: A Tensor. Must be one of the following types: int32, int64.
//     A tensor of indices into ref.
//   - updates: A Tensor. Must have the same type as ref. A tensor of updated values
//     to add to ref.
//
// - Attr use_locking: An optional bool. Defaults to True. If True, the assignment will
//   be protected by a lock; otherwise the behavior is undefined,
//   but may exhibit less contention.
//
// - Output output_ref: Same as ref. Returned as a convenience for operations that want
//   to use the updated values after the update is done.
@_inlineable @inline(__always)
public static func scatterNdAdd<T: Numeric, Tindices: BinaryInteger>(
  ref: Tensor<T>,
  indices: Tensor<Tindices>,
  updates: Tensor<T>,
  useLocking: Bool = false
) -> Tensor<T> {
  return #tfop("ScatterNdAdd",
    ref,
    indices,
    updates,
    T: T.self,
    Tindices: Tindices.self,
    use_locking: useLocking)
}

// Applies sparse addition to `input` using individual values or slices
//
// from `updates` according to indices `indices`.  The updates are non-aliasing:
// `input` is only modified in-place if no other operations will use it.
// Otherwise, a copy of `input` is made.  This operation has a gradient with
// respect to both `input` and `updates`.
//
// `input` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.
//
// `indices` must be integer tensor, containing indices into `input`.
// It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.
//
// The innermost dimension of `indices` (with length `K`) corresponds to
// indices into elements (if `K = P`) or `(P-K)`-dimensional slices
// (if `K < P`) along the `K`th dimension of `input`.
//
// `updates` is `Tensor` of rank `Q-1+P-K` with shape:
//
// ```
// [d_0, ..., d_{Q-2}, input.shape[K], ..., input.shape[P-1]].
// ```
//
// For example, say we want to add 4 scattered elements to a rank-1 tensor to 8
// elements. In Python, that addition would look like this:
//
//     input = tf.constant([1, 2, 3, 4, 5, 6, 7, 8])
//     indices = tf.constant([[4], [3], [1], [7]])
//     updates = tf.constant([9, 10, 11, 12])
//     output = tf.scatter_nd_non_aliasing_add(input, indices, updates)
//     with tf.Session() as sess:
//       print(sess.run(output))
//
// The resulting value `output` would look like this:
//
//     [1, 13, 3, 14, 14, 6, 7, 20]
//
// See @{tf.scatter_nd} for more details about how to make updates to slices.
//
// - Parameters:
//   - input: A Tensor.
//   - indices: A Tensor. Must be one of the following types: `int32`, `int64`.
//     A tensor of indices into `input`.
//   - updates: A Tensor. Must have the same type as ref. A tensor of updated values
//     to add to `input`.
//
// - Output output: A `Tensor` with the same shape as `input`, containing values of `input`
//   updated with `updates`.
@_inlineable @inline(__always)
public static func scatterNdNonAliasingAdd<T: Numeric, Tindices: BinaryInteger>(
  input: Tensor<T>,
  indices: Tensor<Tindices>,
  updates: Tensor<T>
) -> Tensor<T> {
  return #tfop("ScatterNdNonAliasingAdd",
    input,
    indices,
    updates,
    T: T.self,
    Tindices: Tindices.self)
}

// Applies sparse subtraction between `updates` and individual values or slices
//
// within a given variable according to `indices`.
//
// `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.
//
// `indices` must be integer tensor, containing indices into `ref`.
// It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.
//
// The innermost dimension of `indices` (with length `K`) corresponds to
// indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
// dimension of `ref`.
//
// `updates` is `Tensor` of rank `Q-1+P-K` with shape:
//
// ```
// [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
// ```
//
// For example, say we want to subtract 4 scattered elements from a rank-1 tensor
// with 8 elements. In Python, that subtraction would look like this:
//
//     ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
//     indices = tf.constant([[4], [3], [1], [7]])
//     updates = tf.constant([9, 10, 11, 12])
//     sub = tf.scatter_nd_sub(ref, indices, updates)
//     with tf.Session() as sess:
//       print sess.run(sub)
//
// The resulting update to ref would look like this:
//
//     [1, -9, 3, -6, -4, 6, 7, -4]
//
// See @{tf.scatter_nd} for more details about how to make updates to
// slices.
//
// - Parameters:
//   - ref: A mutable Tensor. Should be from a Variable node.
//   - indices: A Tensor. Must be one of the following types: int32, int64.
//     A tensor of indices into ref.
//   - updates: A Tensor. Must have the same type as ref. A tensor of updated values
//     to subtract from ref.
//
// - Attr use_locking: An optional bool. Defaults to True. If True, the assignment will
//   be protected by a lock; otherwise the behavior is undefined,
//   but may exhibit less contention.
//
// - Output output_ref: Same as ref. Returned as a convenience for operations that want
//   to use the updated values after the update is done.
@_inlineable @inline(__always)
public static func scatterNdSub<T: Numeric, Tindices: BinaryInteger>(
  ref: Tensor<T>,
  indices: Tensor<Tindices>,
  updates: Tensor<T>,
  useLocking: Bool = false
) -> Tensor<T> {
  return #tfop("ScatterNdSub",
    ref,
    indices,
    updates,
    T: T.self,
    Tindices: Tindices.self,
    use_locking: useLocking)
}

// Applies sparse `updates` to individual values or slices within a given
//
// variable according to `indices`.
//
// `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.
//
// `indices` must be integer tensor, containing indices into `ref`.
// It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.
//
// The innermost dimension of `indices` (with length `K`) corresponds to
// indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
// dimension of `ref`.
//
// `updates` is `Tensor` of rank `Q-1+P-K` with shape:
//
// ```
// [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
// ```
//
// For example, say we want to update 4 scattered elements to a rank-1 tensor to
// 8 elements. In Python, that update would look like this:
//
// ```python
//     ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
//     indices = tf.constant([[4], [3], [1] ,[7]])
//     updates = tf.constant([9, 10, 11, 12])
//     update = tf.scatter_nd_update(ref, indices, updates)
//     with tf.Session() as sess:
//       print sess.run(update)
// ```
//
// The resulting update to ref would look like this:
//
//     [1, 11, 3, 10, 9, 6, 7, 12]
//
// See @{tf.scatter_nd} for more details about how to make updates to
// slices.
//
// - Parameters:
//   - ref: A mutable Tensor. Should be from a Variable node.
//   - indices: A Tensor. Must be one of the following types: int32, int64.
//     A tensor of indices into ref.
//   - updates: A Tensor. Must have the same type as ref. A tensor of updated
//     values to add to ref.
//
// - Attr use_locking: An optional bool. Defaults to True. If True, the assignment will
//   be protected by a lock; otherwise the behavior is undefined,
//   but may exhibit less contention.
//
// - Output output_ref: Same as ref. Returned as a convenience for operations that want to
//   use the updated values after the update is done.
@_inlineable @inline(__always)
public static func scatterNdUpdate<T: Numeric, Tindices: BinaryInteger>(
  ref: Tensor<T>,
  indices: Tensor<Tindices>,
  updates: Tensor<T>,
  useLocking: Bool = true
) -> Tensor<T> {
  return #tfop("ScatterNdUpdate",
    ref,
    indices,
    updates,
    T: T.self,
    Tindices: Tindices.self,
    use_locking: useLocking)
}

// Subtracts sparse updates to a variable reference.
//
// ```python
//     # Scalar indices
//     ref[indices, ...] -= updates[...]
//
//     # Vector indices (for each i)
//     ref[indices[i], ...] -= updates[i, ...]
//
//     # High rank indices (for each i, ..., j)
//     ref[indices[i, ..., j], ...] -= updates[i, ..., j, ...]
// ```
//
// This operation outputs `ref` after the update is done.
// This makes it easier to chain operations that need to use the reset value.
//
// Duplicate entries are handled correctly: if multiple `indices` reference
// the same location, their (negated) contributions add.
//
// Requires `updates.shape = indices.shape + ref.shape[1:]` or `updates.shape = []`.
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="https://www.tensorflow.org/images/ScatterSub.png" alt>
// </div>
//
// - Parameters:
//   - ref: Should be from a `Variable` node.
//   - indices: A tensor of indices into the first dimension of `ref`.
//   - updates: A tensor of updated values to subtract from `ref`.
//
// - Attr use_locking: If True, the subtraction will be protected by a lock;
//   otherwise the behavior is undefined, but may exhibit less contention.
//
// - Output output_ref: = Same as `ref`.  Returned as a convenience for operations that want
//   to use the updated values after the update is done.
@_inlineable @inline(__always)
public static func scatterSub<T: Numeric, Tindices: BinaryInteger>(
  ref: Tensor<T>,
  indices: Tensor<Tindices>,
  updates: Tensor<T>,
  useLocking: Bool = false
) -> Tensor<T> {
  return #tfop("ScatterSub",
    ref,
    indices,
    updates,
    T: T.self,
    Tindices: Tindices.self,
    use_locking: useLocking)
}

// Applies sparse updates to a variable reference.
//
// This operation computes
//
// ```python
//     # Scalar indices
//     ref[indices, ...] = updates[...]
//
//     # Vector indices (for each i)
//     ref[indices[i], ...] = updates[i, ...]
//
//     # High rank indices (for each i, ..., j)
//     ref[indices[i, ..., j], ...] = updates[i, ..., j, ...]
// ```
//
// This operation outputs `ref` after the update is done.
// This makes it easier to chain operations that need to use the reset value.
//
// If values in `ref` is to be updated more than once, because there are
// duplicate entries in `indices`, the order at which the updates happen
// for each value is undefined.
//
// Requires `updates.shape = indices.shape + ref.shape[1:]` or `updates.shape = []`.
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="https://www.tensorflow.org/images/ScatterUpdate.png" alt>
// </div>
//
// - Parameters:
//   - ref: Should be from a `Variable` node.
//   - indices: A tensor of indices into the first dimension of `ref`.
//   - updates: A tensor of updated values to store in `ref`.
//
// - Attr use_locking: If True, the assignment will be protected by a lock;
//   otherwise the behavior is undefined, but may exhibit less contention.
//
// - Output output_ref: = Same as `ref`.  Returned as a convenience for operations that want
//   to use the updated values after the update is done.
@_inlineable @inline(__always)
public static func scatterUpdate<T: Numeric, Tindices: BinaryInteger>(
  ref: Tensor<T>,
  indices: Tensor<Tindices>,
  updates: Tensor<T>,
  useLocking: Bool = true
) -> Tensor<T> {
  return #tfop("ScatterUpdate",
    ref,
    indices,
    updates,
    T: T.self,
    Tindices: Tindices.self,
    use_locking: useLocking)
}

// Distributed version of Stochastic Dual Coordinate Ascent (SDCA) optimizer for
//
// linear models with L1 + L2 regularization. As global optimization objective is
// strongly-convex, the optimizer optimizes the dual objective at each step. The
// optimizer applies each update one example at a time. Examples are sampled
// uniformly, and the optimizer is learning rate free and enjoys linear convergence
// rate.
//
// [Proximal Stochastic Dual Coordinate Ascent](http://arxiv.org/pdf/1211.2717v1.pdf).<br>
// Shai Shalev-Shwartz, Tong Zhang. 2012
//
// $$Loss Objective = \sum f_{i} (wx_{i}) + (l2 / 2) * |w|^2 + l1 * |w|$$
//
// [Adding vs. Averaging in Distributed Primal-Dual Optimization](http://arxiv.org/abs/1502.03508).<br>
// Chenxin Ma, Virginia Smith, Martin Jaggi, Michael I. Jordan,
// Peter Richtarik, Martin Takac. 2015
//
// [Stochastic Dual Coordinate Ascent with Adaptive Probabilities](https://arxiv.org/abs/1502.08053).<br>
// Dominik Csiba, Zheng Qu, Peter Richtarik. 2015
//
// - Parameters:
//   - sparse_example_indices: a list of vectors which contain example indices.
//   - sparse_feature_indices: a list of vectors which contain feature indices.
//   - sparse_feature_values: a list of vectors which contains feature value
//     associated with each feature group.
//   - dense_features: a list of matrices which contains the dense feature values.
//   - example_weights: a vector which contains the weight associated with each
//     example.
//   - example_labels: a vector which contains the label/target associated with each
//     example.
//   - sparse_indices: a list of vectors where each value is the indices which has
//     corresponding weights in sparse_weights. This field maybe omitted for the
//     dense approach.
//   - sparse_weights: a list of vectors where each value is the weight associated with
//     a sparse feature group.
//   - dense_weights: a list of vectors where the values are the weights associated
//     with a dense feature group.
//   - example_state_data: a list of vectors containing the example state data.
//
// - Attrs:
//   - loss_type: Type of the primal loss. Currently SdcaSolver supports logistic,
//     squared and hinge losses.
//   - adaptative: Whether to use Adaptive SDCA for the inner loop.
//   - num_sparse_features: Number of sparse feature groups to train on.
//   - num_sparse_features_with_values: Number of sparse feature groups with values
//     associated with it, otherwise implicitly treats values as 1.0.
//   - num_dense_features: Number of dense feature groups to train on.
//   - l1: Symmetric l1 regularization strength.
//   - l2: Symmetric l2 regularization strength.
//   - num_loss_partitions: Number of partitions of the global loss function.
//   - num_inner_iterations: Number of iterations per mini-batch.
//
// - Outputs:
//   - out_example_state_data: a list of vectors containing the updated example state
//     data.
//   - out_delta_sparse_weights: a list of vectors where each value is the delta
//     weights associated with a sparse feature group.
//   - out_delta_dense_weights: a list of vectors where the values are the delta
//     weights associated with a dense feature group.
@_inlineable @inline(__always)
public static func sdcaOptimizer(
  sparseExampleIndices: [Tensor<Int64>],
  sparseFeatureIndices: [Tensor<Int64>],
  sparseFeatureValues: [Tensor<Float>],
  denseFeatures: [Tensor<Float>],
  exampleWeights: Tensor<Float>,
  exampleLabels: Tensor<Float>,
  sparseIndices: [Tensor<Int64>],
  sparseWeights: [Tensor<Float>],
  denseWeights: [Tensor<Float>],
  exampleStateData: Tensor<Float>,
  lossType: LossType,
  adaptative: Bool = false,
  l1: Double,
  l2: Double,
  numLossPartitions: Int,
  numInnerIterations: Int
) -> (Tensor<Float>, [Tensor<Float>], [Tensor<Float>]) {
  return #tfop("SdcaOptimizer",
    sparseExampleIndices,
    sparseFeatureIndices,
    sparseFeatureValues,
    denseFeatures,
    exampleWeights,
    exampleLabels,
    sparseIndices,
    sparseWeights,
    denseWeights,
    exampleStateData,
    loss_type: lossType.rawValue,
    adaptative: adaptative,
    l1: l1,
    l2: l2,
    num_loss_partitions: numLossPartitions,
    num_inner_iterations: numInnerIterations)
}

// Applies L1 regularization shrink step on the parameters.
//
// - Parameter weights: a list of vectors where each value is the weight associated with a
//   feature group.
//
// - Attrs:
//   - num_features: Number of feature groups to apply shrinking step.
//   - l1: Symmetric l1 regularization strength.
//   - l2: Symmetric l2 regularization strength. Should be a positive float.
@_inlineable @inline(__always)
public static func sdcaShrinkL1(
  weights: [Tensor<Float>],
  l1: Double,
  l2: Double
) {
  return #tfop("SdcaShrinkL1",
    weights,
    l1: l1,
    l2: l2)
}

// Computes the maximum along segments of a tensor.
//
// Read @{$math_ops#Segmentation$the section on segmentation} for an explanation of
// segments.
//
// Computes a tensor such that
// \\(output_i = \max_j(data_j)\\) where `max` is over `j` such
// that `segment_ids[j] == i`.
//
// If the max is empty for a given segment ID `i`, `output[i] = 0`.
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="https://www.tensorflow.org/images/SegmentMax.png" alt>
// </div>
//
// - Parameter segment_ids: A 1-D tensor whose rank is equal to the rank of `data`'s
//   first dimension.  Values should be sorted and can be repeated.
//
// - Output output: Has same shape as data, except for dimension 0 which
//   has size `k`, the number of segments.
@_inlineable @inline(__always)
public static func segmentMax<T: Numeric, Tindices: BinaryInteger>(
  data: Tensor<T>,
  segmentIds: Tensor<Tindices>
) -> Tensor<T> {
  return #tfop("SegmentMax",
    data,
    segmentIds,
    T: T.self,
    Tindices: Tindices.self)
}

// Computes the mean along segments of a tensor.
//
// Read @{$math_ops#Segmentation$the section on segmentation} for an explanation of
// segments.
//
// Computes a tensor such that
// \\(output_i = \frac{\sum_j data_j}{N}\\) where `mean` is
// over `j` such that `segment_ids[j] == i` and `N` is the total number of
// values summed.
//
// If the mean is empty for a given segment ID `i`, `output[i] = 0`.
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="https://www.tensorflow.org/images/SegmentMean.png" alt>
// </div>
//
// - Parameter segment_ids: A 1-D tensor whose rank is equal to the rank of `data`'s
//   first dimension.  Values should be sorted and can be repeated.
//
// - Output output: Has same shape as data, except for dimension 0 which
//   has size `k`, the number of segments.
@_inlineable @inline(__always)
public static func segmentMean<T: Numeric, Tindices: BinaryInteger>(
  data: Tensor<T>,
  segmentIds: Tensor<Tindices>
) -> Tensor<T> {
  return #tfop("SegmentMean",
    data,
    segmentIds,
    T: T.self,
    Tindices: Tindices.self)
}

// Computes the minimum along segments of a tensor.
//
// Read @{$math_ops#Segmentation$the section on segmentation} for an explanation of
// segments.
//
// Computes a tensor such that
// \\(output_i = \min_j(data_j)\\) where `min` is over `j` such
// that `segment_ids[j] == i`.
//
// If the min is empty for a given segment ID `i`, `output[i] = 0`.
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="https://www.tensorflow.org/images/SegmentMin.png" alt>
// </div>
//
// - Parameter segment_ids: A 1-D tensor whose rank is equal to the rank of `data`'s
//   first dimension.  Values should be sorted and can be repeated.
//
// - Output output: Has same shape as data, except for dimension 0 which
//   has size `k`, the number of segments.
@_inlineable @inline(__always)
public static func segmentMin<T: Numeric, Tindices: BinaryInteger>(
  data: Tensor<T>,
  segmentIds: Tensor<Tindices>
) -> Tensor<T> {
  return #tfop("SegmentMin",
    data,
    segmentIds,
    T: T.self,
    Tindices: Tindices.self)
}

// Computes the product along segments of a tensor.
//
// Read @{$math_ops#Segmentation$the section on segmentation} for an explanation of
// segments.
//
// Computes a tensor such that
// \\(output_i = \prod_j data_j\\) where the product is over `j` such
// that `segment_ids[j] == i`.
//
// If the product is empty for a given segment ID `i`, `output[i] = 1`.
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="https://www.tensorflow.org/images/SegmentProd.png" alt>
// </div>
//
// - Parameter segment_ids: A 1-D tensor whose rank is equal to the rank of `data`'s
//   first dimension.  Values should be sorted and can be repeated.
//
// - Output output: Has same shape as data, except for dimension 0 which
//   has size `k`, the number of segments.
@_inlineable @inline(__always)
public static func segmentProd<T: Numeric, Tindices: BinaryInteger>(
  data: Tensor<T>,
  segmentIds: Tensor<Tindices>
) -> Tensor<T> {
  return #tfop("SegmentProd",
    data,
    segmentIds,
    T: T.self,
    Tindices: Tindices.self)
}

// Computes the sum along segments of a tensor.
//
// Read @{$math_ops#Segmentation$the section on segmentation} for an explanation of
// segments.
//
// Computes a tensor such that
// \\(output_i = \sum_j data_j\\) where sum is over `j` such
// that `segment_ids[j] == i`.
//
// If the sum is empty for a given segment ID `i`, `output[i] = 0`.
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="https://www.tensorflow.org/images/SegmentSum.png" alt>
// </div>
//
// - Parameter segment_ids: A 1-D tensor whose rank is equal to the rank of `data`'s
//   first dimension.  Values should be sorted and can be repeated.
//
// - Output output: Has same shape as data, except for dimension 0 which
//   has size `k`, the number of segments.
@_inlineable @inline(__always)
public static func segmentSum<T: Numeric, Tindices: BinaryInteger>(
  data: Tensor<T>,
  segmentIds: Tensor<Tindices>
) -> Tensor<T> {
  return #tfop("SegmentSum",
    data,
    segmentIds,
    T: T.self,
    Tindices: Tindices.self)
}

// Selects elements from `x` or `y`, depending on `condition`.
//
// The `x`, and `y` tensors must all have the same shape, and the
// output will also have that shape.
//
// The `condition` tensor must be a scalar if `x` and `y` are scalars.
// If `x` and `y` are vectors or higher rank, then `condition` must be either a
// scalar, a vector with size matching the first dimension of `x`, or must have
// the same shape as `x`.
//
// The `condition` tensor acts as a mask that chooses, based on the value at each
// element, whether the corresponding element / row in the output should be
// taken from `x` (if true) or `y` (if false).
//
// If `condition` is a vector and `x` and `y` are higher rank matrices, then
// it chooses which row (outer dimension) to copy from `x` and `y`.
// If `condition` has the same shape as `x` and `y`, then it chooses which
// element to copy from `x` and `y`.
//
// For example:
//
// ```python
// # 'condition' tensor is [[True,  False]
// #                        [False, True]]
// # 't' is [[1, 2],
// #         [3, 4]]
// # 'e' is [[5, 6],
// #         [7, 8]]
// select(condition, t, e)  # => [[1, 6], [7, 4]]
//
//
// # 'condition' tensor is [True, False]
// # 't' is [[1, 2],
// #         [3, 4]]
// # 'e' is [[5, 6],
// #         [7, 8]]
// select(condition, t, e) ==> [[1, 2],
//                              [7, 8]]
//
// ```
//
// - Parameters:
//   - t: = A `Tensor` which may have the same shape as `condition`.
//     If `condition` is rank 1, `x` may have higher rank,
//     but its first dimension must match the size of `condition`.
//   - e: = A `Tensor` with the same type and shape as `x`.
//
// - Output output: = A `Tensor` with the same type and shape as `x` and `y`.
@_inlineable @inline(__always)
public static func select<T: Numeric>(
  condition: Tensor<Bool>,
  t: Tensor<T>,
  e: Tensor<T>
) -> Tensor<T> {
  return #tfop("Select",
    condition,
    t,
    e,
    T: T.self)
}

// Computes the Eigen Decomposition of a batch of square self-adjoint matrices.
//
// The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
// form square matrices, with the same constraints as the single matrix
// SelfAdjointEig.
//
// The result is a [..., M+1, M] matrix with [..., 0,:] containing the
// eigenvalues, and subsequent [...,1:, :] containing the eigenvectors. The eigenvalues
// are sorted in non-decreasing order.
//
// - Parameter input: Shape is `[..., M, M]`.
//
// - Output output: Shape is `[..., M+1, M]`.
@_inlineable @inline(__always)
public static func selfAdjointEig<T: BinaryFloatingPoint>(
  input: Tensor<T>
) -> Tensor<T> {
  return #tfop("SelfAdjointEig",
    input,
    T: T.self)
}

// Computes the eigen decomposition of one or more square self-adjoint matrices.
//
// Computes the eigenvalues and (optionally) eigenvectors of each inner matrix in
// `input` such that `input[..., :, :] = v[..., :, :] * diag(e[..., :])`. The eigenvalues
// are sorted in non-decreasing order.
//
// ```python
// # a is a tensor.
// # e is a tensor of eigenvalues.
// # v is a tensor of eigenvectors.
// e, v = self_adjoint_eig(a)
// e = self_adjoint_eig(a, compute_v=False)
// ```
//
// - Parameter input: `Tensor` input of shape `[N, N]`.
//
// - Attr compute_v: If `True` then eigenvectors will be computed and returned in `v`.
//   Otherwise, only the eigenvalues will be computed.
//
// - Outputs:
//   - e: Eigenvalues. Shape is `[N]`.
//   - v: Eigenvectors. Shape is `[N, N]`.
@_inlineable @inline(__always)
public static func selfAdjointEigV2<T: BinaryFloatingPoint>(
  input: Tensor<T>,
  computeV: Bool = true
) -> (Tensor<T>, Tensor<T>) {
  return #tfop("SelfAdjointEigV2",
    input,
    T: T.self,
    compute_v: computeV)
}

// Computes scaled exponential linear: `scale * alpha * (exp(features) - 1)`
//
// if < 0, `scale * features` otherwise.
//
// See [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
@_inlineable @inline(__always)
public static func selu<T: BinaryFloatingPoint>(
  features: Tensor<T>
) -> Tensor<T> {
  return #tfop("Selu",
    features,
    T: T.self)
}

// Computes gradients for the scaled exponential linear (Selu) operation.
//
// - Parameters:
//   - gradients: The backpropagated gradients to the corresponding Selu operation.
//   - outputs: The outputs of the corresponding Selu operation.
//
// - Output backprops: The gradients: `gradients * (outputs + scale * alpha)`
//   if outputs < 0, `scale * gradients` otherwise.
@_inlineable @inline(__always)
public static func seluGrad<T: BinaryFloatingPoint>(
  gradients: Tensor<T>,
  outputs: Tensor<T>
) -> Tensor<T> {
  return #tfop("SeluGrad",
    gradients,
    outputs,
    T: T.self)
}

// Serialize an `N`-minibatch `SparseTensor` into an `[N, 3]` `Tensor` object.
//
// The `SparseTensor` must have rank `R` greater than 1, and the first dimension
// is treated as the minibatch dimension.  Elements of the `SparseTensor`
// must be sorted in increasing order of this first dimension.  The serialized
// `SparseTensor` objects going into each row of `serialized_sparse` will have
// rank `R-1`.
//
// The minibatch size `N` is extracted from `sparse_shape[0]`.
//
// - Parameters:
//   - sparse_indices: 2-D.  The `indices` of the minibatch `SparseTensor`.
//   - sparse_values: 1-D.  The `values` of the minibatch `SparseTensor`.
//   - sparse_shape: 1-D.  The `shape` of the minibatch `SparseTensor`.
//
// - Attr out_type: The `dtype` to use for serialization; the supported types are `string`
//   (default) and `variant`.
@_inlineable @inline(__always)
public static func serializeManySparse<T: Numeric, Out_type: Numeric>(
  sparseIndices: Tensor<Int64>,
  sparseValues: Tensor<T>,
  sparseShape: Tensor<Int64>
) -> Tensor<Out_type> {
  return #tfop("SerializeManySparse",
    sparseIndices,
    sparseValues,
    sparseShape,
    T: T.self,
    Out_type: Out_type.self)
}

// Serialize a `SparseTensor` into a `[3]` `Tensor` object.
//
// - Parameters:
//   - sparse_indices: 2-D.  The `indices` of the `SparseTensor`.
//   - sparse_values: 1-D.  The `values` of the `SparseTensor`.
//   - sparse_shape: 1-D.  The `shape` of the `SparseTensor`.
//
// - Attr out_type: The `dtype` to use for serialization; the supported types are `string`
//   (default) and `variant`.
@_inlineable @inline(__always)
public static func serializeSparse<T: Numeric, Out_type: Numeric>(
  sparseIndices: Tensor<Int64>,
  sparseValues: Tensor<T>,
  sparseShape: Tensor<Int64>
) -> Tensor<Out_type> {
  return #tfop("SerializeSparse",
    sparseIndices,
    sparseValues,
    sparseShape,
    T: T.self,
    Out_type: Out_type.self)
}

// Number of unique elements along last dimension of input `set`.
//
// Input `set` is a `SparseTensor` represented by `set_indices`, `set_values`,
// and `set_shape`. The last dimension contains values in a set, duplicates are
// allowed but ignored.
//
// If `validate_indices` is `True`, this op validates the order and range of `set`
// indices.
//
// - Parameters:
//   - set_indices: 2D `Tensor`, indices of a `SparseTensor`.
//   - set_values: 1D `Tensor`, values of a `SparseTensor`.
//   - set_shape: 1D `Tensor`, shape of a `SparseTensor`.
//
// - Output size: For `set` ranked `n`, this is a `Tensor` with rank `n-1`, and the same 1st
//   `n-1` dimensions as `set`. Each value is the number of unique elements in
//   the corresponding `[0...n-1]` dimension of `set`.
@_inlineable @inline(__always)
public static func setSize<T: BinaryInteger>(
  setIndices: Tensor<Int64>,
  setValues: Tensor<T>,
  setShape: Tensor<Int64>,
  validateIndices: Bool = true
) -> Tensor<Int32> {
  return #tfop("SetSize",
    setIndices,
    setValues,
    setShape,
    T: T.self,
    validate_indices: validateIndices)
}

// Returns the shape of a tensor.
//
// This operation returns a 1-D integer tensor representing the shape of `input`.
//
// For example:
//
// ```
// # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
// shape(t) ==> [2, 2, 3]
// ```
@_inlineable @inline(__always)
public static func shape<T: Numeric, Out_type: BinaryInteger>(
  input: Tensor<T>
) -> Tensor<Out_type> {
  return #tfop("Shape",
    input,
    T: T.self,
    Out_type: Out_type.self)
}

// Returns shape of tensors.
//
// This operation returns N 1-D integer tensors representing shape of `input[i]s`.
@_inlineable @inline(__always)
public static func shapeN<T: Numeric, Out_type: BinaryInteger>(
  input: [Tensor<T>]
) -> [Tensor<Out_type>] {
  return #tfop("ShapeN",
    input,
    T: T.self,
    Out_type: Out_type.self)
}

// Computes sigmoid of `x` element-wise.
//
// Specifically, `y = 1 / (1 + exp(-x))`.
@_inlineable @inline(__always)
public static func sigmoid<T: BinaryFloatingPoint>(
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Sigmoid",
    x,
    T: T.self)
}

// Computes the gradient of the sigmoid of `x` wrt its input.
//
// Specifically, `grad = dy * y * (1 - y)`, where `y = sigmoid(x)`, and
// `dy` is the corresponding input gradient.
@_inlineable @inline(__always)
public static func sigmoidGrad<T: BinaryFloatingPoint>(
  y: Tensor<T>,
  dy: Tensor<T>
) -> Tensor<T> {
  return #tfop("SigmoidGrad",
    y,
    dy,
    T: T.self)
}

// Returns an element-wise indication of the sign of a number.
//
// `y = sign(x) = -1` if `x < 0`; 0 if `x == 0`; 1 if `x > 0`.
//
// For complex numbers, `y = sign(x) = x / |x|` if `x != 0`, otherwise `y = 0`.
@_inlineable @inline(__always)
public static func sign<T: Numeric>(
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Sign",
    x,
    T: T.self)
}

@_inlineable @inline(__always)
public static func simple(
  a: Tensor<Int32>
) -> Tensor<Float> {
  return #tfop("Simple",
    a)
}

@_inlineable @inline(__always)
public static func simpleStruct(
  nA: Int
) -> [Tensor<Int32>] {
  return #tfop("SimpleStruct",
    n_a: nA)
}

// Computes sin of x element-wise.
@_inlineable @inline(__always)
public static func sin<T: BinaryFloatingPoint>(
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Sin",
    x,
    T: T.self)
}

// Computes hyperbolic sine of x element-wise.
@_inlineable @inline(__always)
public static func sinh<T: BinaryFloatingPoint>(
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Sinh",
    x,
    T: T.self)
}

// Returns the size of a tensor.
//
// This operation returns an integer representing the number of elements in
// `input`.
//
// For example:
//
// ```
// # 't' is [[[1, 1,, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]]
// size(t) ==> 12
// ```
@_inlineable @inline(__always)
public static func size<T: Numeric, Out_type: BinaryInteger>(
  input: Tensor<T>
) -> Tensor<Out_type> {
  return #tfop("Size",
    input,
    T: T.self,
    Out_type: Out_type.self)
}

// Return a slice from 'input'.
//
// The output tensor is a tensor with dimensions described by 'size'
// whose values are extracted from 'input' starting at the offsets in
// 'begin'.
//
// *Requirements*:
//   0 <= begin[i] <= begin[i] + size[i] <= Di  for i in [0, n)
//
// - Parameters:
//   - begin: begin[i] specifies the offset into the 'i'th dimension of
//     'input' to slice from.
//   - size: size[i] specifies the number of elements of the 'i'th dimension
//     of 'input' to slice. If size[i] is -1, all remaining elements in dimension
//     i are included in the slice (i.e. this is equivalent to setting
//     size[i] = input.dim_size(i) - begin[i]).
@_inlineable @inline(__always)
public static func slice<T: Numeric, Index: BinaryInteger>(
  input: Tensor<T>,
  begin: Tensor<Index>,
  size: Tensor<Index>
) -> Tensor<T> {
  return #tfop("Slice",
    input,
    begin,
    size,
    T: T.self,
    Index: Index.self)
}

// Returns a copy of the input tensor.
@_inlineable @inline(__always)
public static func snapshot<T: Numeric>(
  input: Tensor<T>
) -> Tensor<T> {
  return #tfop("Snapshot",
    input,
    T: T.self)
}

// Computes softmax activations.
//
// For each batch `i` and class `j` we have
//
//     softmax[i, j] = exp(logits[i, j]) / sum_j(exp(logits[i, j]))
//
// - Parameter logits: 2-D with shape `[batch_size, num_classes]`.
//
// - Output softmax: Same shape as `logits`.
@_inlineable @inline(__always)
public static func softmax<T: BinaryFloatingPoint>(
  logits: Tensor<T>
) -> Tensor<T> {
  return #tfop("Softmax",
    logits,
    T: T.self)
}

// Computes softmax cross entropy cost and gradients to backpropagate.
//
// Inputs are the logits, not probabilities.
//
// - Parameters:
//   - features: batch_size x num_classes matrix
//   - labels: batch_size x num_classes matrix
//     The caller must ensure that each batch of labels represents a valid
//     probability distribution.
//
// - Outputs:
//   - loss: Per example loss (batch_size vector).
//   - backprop: backpropagated gradients (batch_size x num_classes matrix).
@_inlineable @inline(__always)
public static func softmaxCrossEntropyWithLogits<T: BinaryFloatingPoint>(
  features: Tensor<T>,
  labels: Tensor<T>
) -> (Tensor<T>, Tensor<T>) {
  return #tfop("SoftmaxCrossEntropyWithLogits",
    features,
    labels,
    T: T.self)
}

// Computes softplus: `log(exp(features) + 1)`.
@_inlineable @inline(__always)
public static func softplus<T: Numeric>(
  features: Tensor<T>
) -> Tensor<T> {
  return #tfop("Softplus",
    features,
    T: T.self)
}

// Computes softplus gradients for a softplus operation.
//
// - Parameters:
//   - gradients: The backpropagated gradients to the corresponding softplus operation.
//   - features: The features passed as input to the corresponding softplus operation.
//
// - Output backprops: The gradients: `gradients / (1 + exp(-features))`.
@_inlineable @inline(__always)
public static func softplusGrad<T: Numeric>(
  gradients: Tensor<T>,
  features: Tensor<T>
) -> Tensor<T> {
  return #tfop("SoftplusGrad",
    gradients,
    features,
    T: T.self)
}

// Computes softsign: `features / (abs(features) + 1)`.
@_inlineable @inline(__always)
public static func softsign<T: Numeric>(
  features: Tensor<T>
) -> Tensor<T> {
  return #tfop("Softsign",
    features,
    T: T.self)
}

// Computes softsign gradients for a softsign operation.
//
// - Parameters:
//   - gradients: The backpropagated gradients to the corresponding softsign operation.
//   - features: The features passed as input to the corresponding softsign operation.
//
// - Output backprops: The gradients: `gradients / (1 + abs(features)) ** 2`.
@_inlineable @inline(__always)
public static func softsignGrad<T: Numeric>(
  gradients: Tensor<T>,
  features: Tensor<T>
) -> Tensor<T> {
  return #tfop("SoftsignGrad",
    gradients,
    features,
    T: T.self)
}

// SpaceToBatch for 4-D tensors of type T.
//
// This is a legacy version of the more general SpaceToBatchND.
//
// Zero-pads and then rearranges (permutes) blocks of spatial data into batch.
// More specifically, this op outputs a copy of the input tensor where values from
// the `height` and `width` dimensions are moved to the `batch` dimension. After
// the zero-padding, both `height` and `width` of the input must be divisible by the
// block size.
//
// - Parameters:
//   - input: 4-D with shape `[batch, height, width, depth]`.
//   - paddings: 2-D tensor of non-negative integers with shape `[2, 2]`. It specifies
//       the padding of the input with zeros across the spatial dimensions as follows:
//
//           paddings = [[pad_top, pad_bottom], [pad_left, pad_right]]
//
//       The effective spatial dimensions of the zero-padded input tensor will be:
//
//           height_pad = pad_top + height + pad_bottom
//           width_pad = pad_left + width + pad_right
//
//     The attr `block_size` must be greater than one. It indicates the block size.
//
//       * Non-overlapping blocks of size `block_size x block size` in the height and
//         width dimensions are rearranged into the batch dimension at each location.
//       * The batch of the output tensor is `batch * block_size * block_size`.
//       * Both height_pad and width_pad must be divisible by block_size.
//
//     The shape of the output will be:
//
//         [batch*block_size*block_size, height_pad/block_size, width_pad/block_size,
//          depth]
//
//     Some examples:
//
//     (1) For the following input of shape `[1, 2, 2, 1]` and block_size of 2:
//
//     ```
//     x = [[[[1], [2]], [[3], [4]]]]
//     ```
//
//     The output tensor has shape `[4, 1, 1, 1]` and value:
//
//     ```
//     [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
//     ```
//
//     (2) For the following input of shape `[1, 2, 2, 3]` and block_size of 2:
//
//     ```
//     x = [[[[1, 2, 3], [4, 5, 6]],
//           [[7, 8, 9], [10, 11, 12]]]]
//     ```
//
//     The output tensor has shape `[4, 1, 1, 3]` and value:
//
//     ```
//     [[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]]
//     ```
//
//     (3) For the following input of shape `[1, 4, 4, 1]` and block_size of 2:
//
//     ```
//     x = [[[[1],   [2],  [3],  [4]],
//           [[5],   [6],  [7],  [8]],
//           [[9],  [10], [11],  [12]],
//           [[13], [14], [15],  [16]]]]
//     ```
//
//     The output tensor has shape `[4, 2, 2, 1]` and value:
//
//     ```
//     x = [[[[1], [3]], [[9], [11]]],
//          [[[2], [4]], [[10], [12]]],
//          [[[5], [7]], [[13], [15]]],
//          [[[6], [8]], [[14], [16]]]]
//     ```
//
//     (4) For the following input of shape `[2, 2, 4, 1]` and block_size of 2:
//
//     ```
//     x = [[[[1],   [2],  [3],  [4]],
//           [[5],   [6],  [7],  [8]]],
//          [[[9],  [10], [11],  [12]],
//           [[13], [14], [15],  [16]]]]
//     ```
//
//     The output tensor has shape `[8, 1, 2, 1]` and value:
//
//     ```
//     x = [[[[1], [3]]], [[[9], [11]]], [[[2], [4]]], [[[10], [12]]],
//          [[[5], [7]]], [[[13], [15]]], [[[6], [8]]], [[[14], [16]]]]
//     ```
//
//     Among others, this operation is useful for reducing atrous convolution into
//     regular convolution.
@_inlineable @inline(__always)
public static func spaceToBatch<T: Numeric, Tpaddings: BinaryInteger>(
  input: Tensor<T>,
  paddings: Tensor<Tpaddings>,
  blockSize: Int
) -> Tensor<T> {
  return #tfop("SpaceToBatch",
    input,
    paddings,
    T: T.self,
    Tpaddings: Tpaddings.self,
    block_size: blockSize)
}

// SpaceToBatch for N-D tensors of type T.
//
// This operation divides "spatial" dimensions `[1, ..., M]` of the input into a
// grid of blocks of shape `block_shape`, and interleaves these blocks with the
// "batch" dimension (0) such that in the output, the spatial dimensions
// `[1, ..., M]` correspond to the position within the grid, and the batch
// dimension combines both the position within a spatial block and the original
// batch position.  Prior to division into blocks, the spatial dimensions of the
// input are optionally zero padded according to `paddings`.  See below for a
// precise description.
//
// - Parameters:
//   - input: N-D with shape `input_shape = [batch] + spatial_shape + remaining_shape`,
//     where spatial_shape has `M` dimensions.
//   - block_shape: 1-D with shape `[M]`, all values must be >= 1.
//   - paddings: 2-D with shape `[M, 2]`, all values must be >= 0.
//       `paddings[i] = [pad_start, pad_end]` specifies the padding for input dimension
//       `i + 1`, which corresponds to spatial dimension `i`.  It is required that
//       `block_shape[i]` divides `input_shape[i + 1] + pad_start + pad_end`.
//
//     This operation is equivalent to the following steps:
//
//     1. Zero-pad the start and end of dimensions `[1, ..., M]` of the
//        input according to `paddings` to produce `padded` of shape `padded_shape`.
//
//     2. Reshape `padded` to `reshaped_padded` of shape:
//
//          [batch] +
//          [padded_shape[1] / block_shape[0],
//            block_shape[0],
//           ...,
//           padded_shape[M] / block_shape[M-1],
//           block_shape[M-1]] +
//          remaining_shape
//
//     3. Permute dimensions of `reshaped_padded` to produce
//        `permuted_reshaped_padded` of shape:
//
//          block_shape +
//          [batch] +
//          [padded_shape[1] / block_shape[0],
//           ...,
//           padded_shape[M] / block_shape[M-1]] +
//          remaining_shape
//
//     4. Reshape `permuted_reshaped_padded` to flatten `block_shape` into the batch
//        dimension, producing an output tensor of shape:
//
//          [batch * prod(block_shape)] +
//          [padded_shape[1] / block_shape[0],
//           ...,
//           padded_shape[M] / block_shape[M-1]] +
//          remaining_shape
//
//     Some examples:
//
//     (1) For the following input of shape `[1, 2, 2, 1]`, `block_shape = [2, 2]`, and
//         `paddings = [[0, 0], [0, 0]]`:
//
//     ```
//     x = [[[[1], [2]], [[3], [4]]]]
//     ```
//
//     The output tensor has shape `[4, 1, 1, 1]` and value:
//
//     ```
//     [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
//     ```
//
//     (2) For the following input of shape `[1, 2, 2, 3]`, `block_shape = [2, 2]`, and
//         `paddings = [[0, 0], [0, 0]]`:
//
//     ```
//     x = [[[[1, 2, 3], [4, 5, 6]],
//           [[7, 8, 9], [10, 11, 12]]]]
//     ```
//
//     The output tensor has shape `[4, 1, 1, 3]` and value:
//
//     ```
//     [[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]]
//     ```
//
//     (3) For the following input of shape `[1, 4, 4, 1]`, `block_shape = [2, 2]`, and
//         `paddings = [[0, 0], [0, 0]]`:
//
//     ```
//     x = [[[[1],   [2],  [3],  [4]],
//           [[5],   [6],  [7],  [8]],
//           [[9],  [10], [11],  [12]],
//           [[13], [14], [15],  [16]]]]
//     ```
//
//     The output tensor has shape `[4, 2, 2, 1]` and value:
//
//     ```
//     x = [[[[1], [3]], [[9], [11]]],
//          [[[2], [4]], [[10], [12]]],
//          [[[5], [7]], [[13], [15]]],
//          [[[6], [8]], [[14], [16]]]]
//     ```
//
//     (4) For the following input of shape `[2, 2, 4, 1]`, block_shape = `[2, 2]`, and
//         paddings = `[[0, 0], [2, 0]]`:
//
//     ```
//     x = [[[[1],   [2],  [3],  [4]],
//           [[5],   [6],  [7],  [8]]],
//          [[[9],  [10], [11],  [12]],
//           [[13], [14], [15],  [16]]]]
//     ```
//
//     The output tensor has shape `[8, 1, 3, 1]` and value:
//
//     ```
//     x = [[[[0], [1], [3]]], [[[0], [9], [11]]],
//          [[[0], [2], [4]]], [[[0], [10], [12]]],
//          [[[0], [5], [7]]], [[[0], [13], [15]]],
//          [[[0], [6], [8]]], [[[0], [14], [16]]]]
//     ```
//
//     Among others, this operation is useful for reducing atrous convolution into
//     regular convolution.
@_inlineable @inline(__always)
public static func spaceToBatchND<T: Numeric, Tblock_shape: BinaryInteger, Tpaddings: BinaryInteger>(
  input: Tensor<T>,
  blockShape: Tensor<Tblock_shape>,
  paddings: Tensor<Tpaddings>
) -> Tensor<T> {
  return #tfop("SpaceToBatchND",
    input,
    blockShape,
    paddings,
    T: T.self,
    Tblock_shape: Tblock_shape.self,
    Tpaddings: Tpaddings.self)
}

// SpaceToDepth for tensors of type T.
//
// Rearranges blocks of spatial data, into depth. More specifically,
// this op outputs a copy of the input tensor where values from the `height`
// and `width` dimensions are moved to the `depth` dimension.
// The attr `block_size` indicates the input block size.
//
//   * Non-overlapping blocks of size `block_size x block size` are rearranged
//     into depth at each location.
//   * The depth of the output tensor is `block_size * block_size * input_depth`.
//   * The Y, X coordinates within each block of the input become the high order
//     component of the output channel index.
//   * The input tensor's height and width must be divisible by block_size.
//
// The `data_format` attr specifies the layout of the input and output tensors
// with the following options:
//   "NHWC": `[ batch, height, width, channels ]`
//   "NCHW": `[ batch, channels, height, width ]`
//   "NCHW_VECT_C":
//       `qint8 [ batch, channels / 4, height, width, 4 ]`
//
// It is useful to consider the operation as transforming a 6-D Tensor.
// e.g. for data_format = NHWC,
//      Each element in the input tensor can be specified via 6 coordinates,
//      ordered by decreasing memory layout significance as:
//      n,oY,bY,oX,bX,iC  (where n=batch index, oX, oY means X or Y coordinates
//                         within the output image, bX, bY means coordinates
//                         within the input block, iC means input channels).
//      The output would be a transpose to the following layout:
//      n,oY,oX,bY,bX,iC
//
// This operation is useful for resizing the activations between convolutions
// (but keeping all data), e.g. instead of pooling. It is also useful for training
// purely convolutional models.
//
// For example, given an input of shape `[1, 2, 2, 1]`, data_format = "NHWC" and
// block_size = 2:
//
// ```
// x = [[[[1], [2]],
//       [[3], [4]]]]
// ```
//
// This operation will output a tensor of shape `[1, 1, 1, 4]`:
//
// ```
// [[[[1, 2, 3, 4]]]]
// ```
//
// Here, the input has a batch of 1 and each batch element has shape `[2, 2, 1]`,
// the corresponding output will have a single element (i.e. width and height are
// both 1) and will have a depth of 4 channels (1 * block_size * block_size).
// The output element shape is `[1, 1, 4]`.
//
// For an input tensor with larger depth, here of shape `[1, 2, 2, 3]`, e.g.
//
// ```
// x = [[[[1, 2, 3], [4, 5, 6]],
//       [[7, 8, 9], [10, 11, 12]]]]
// ```
//
// This operation, for block_size of 2, will return the following tensor of shape
// `[1, 1, 1, 12]`
//
// ```
// [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
// ```
//
// Similarly, for the following input of shape `[1 4 4 1]`, and a block size of 2:
//
// ```
// x = [[[[1],   [2],  [5],  [6]],
//       [[3],   [4],  [7],  [8]],
//       [[9],  [10], [13],  [14]],
//       [[11], [12], [15],  [16]]]]
// ```
//
// the operator will return the following tensor of shape `[1 2 2 4]`:
//
// ```
// x = [[[[1, 2, 3, 4],
//        [5, 6, 7, 8]],
//       [[9, 10, 11, 12],
//        [13, 14, 15, 16]]]]
// ```
//
// - Attr block_size: The size of the spatial block.
@_inlineable @inline(__always)
public static func spaceToDepth<T: Numeric>(
  input: Tensor<T>,
  blockSize: Int,
  dataFormat: DataFormat2 = .nhwc
) -> Tensor<T> {
  return #tfop("SpaceToDepth",
    input,
    T: T.self,
    block_size: blockSize,
    data_format: dataFormat.rawValue)
}

// Adds two `SparseTensor` objects to produce another `SparseTensor`.
//
// The input `SparseTensor` objects' indices are assumed ordered in standard
// lexicographic order.  If this is not the case, before this step run
// `SparseReorder` to restore index ordering.
//
// By default, if two values sum to zero at some index, the output `SparseTensor`
// would still include that particular location in its index, storing a zero in the
// corresponding value slot.  To override this, callers can specify `thresh`,
// indicating that if the sum has a magnitude strictly smaller than `thresh`, its
// corresponding value and index would then not be included.  In particular,
// `thresh == 0` (default) means everything is kept and actual thresholding happens
// only for a positive value.
//
// In the following shapes, `nnz` is the count after taking `thresh` into account.
//
// - Parameters:
//   - a_indices: 2-D.  The `indices` of the first `SparseTensor`, size `[nnz, ndims]` Matrix.
//   - a_values: 1-D.  The `values` of the first `SparseTensor`, size `[nnz]` Vector.
//   - a_shape: 1-D.  The `shape` of the first `SparseTensor`, size `[ndims]` Vector.
//   - b_indices: 2-D.  The `indices` of the second `SparseTensor`, size `[nnz, ndims]` Matrix.
//   - b_values: 1-D.  The `values` of the second `SparseTensor`, size `[nnz]` Vector.
//   - b_shape: 1-D.  The `shape` of the second `SparseTensor`, size `[ndims]` Vector.
//   - thresh: 0-D.  The magnitude threshold that determines if an output value/index
//     pair takes space.
@_inlineable @inline(__always)
public static func sparseAdd<T: Numeric, Treal: Numeric>(
  aIndices: Tensor<Int64>,
  aValues: Tensor<T>,
  aShape: Tensor<Int64>,
  bIndices: Tensor<Int64>,
  bValues: Tensor<T>,
  bShape: Tensor<Int64>,
  thresh: Tensor<Treal>
) -> (Tensor<Int64>, Tensor<T>, Tensor<Int64>) {
  return #tfop("SparseAdd",
    aIndices,
    aValues,
    aShape,
    bIndices,
    bValues,
    bShape,
    thresh,
    T: T.self,
    Treal: Treal.self)
}

// The gradient operator for the SparseAdd op.
//
// The SparseAdd op calculates A + B, where A, B, and the sum are all represented
// as `SparseTensor` objects.  This op takes in the upstream gradient w.r.t.
// non-empty values of the sum, and outputs the gradients w.r.t. the non-empty
// values of A and B.
//
// - Parameters:
//   - backprop_val_grad: 1-D with shape `[nnz(sum)]`.  The gradient with respect to
//     the non-empty values of the sum.
//   - a_indices: 2-D.  The `indices` of the `SparseTensor` A, size `[nnz(A), ndims]`.
//   - b_indices: 2-D.  The `indices` of the `SparseTensor` B, size `[nnz(B), ndims]`.
//   - sum_indices: 2-D.  The `indices` of the sum `SparseTensor`, size
//     `[nnz(sum), ndims]`.
//
// - Outputs:
//   - a_val_grad: 1-D with shape `[nnz(A)]`. The gradient with respect to the
//     non-empty values of A.
//   - b_val_grad: 1-D with shape `[nnz(B)]`. The gradient with respect to the
//     non-empty values of B.
@_inlineable @inline(__always)
public static func sparseAddGrad<T: Numeric>(
  backpropValGrad: Tensor<T>,
  aIndices: Tensor<Int64>,
  bIndices: Tensor<Int64>,
  sumIndices: Tensor<Int64>
) -> (Tensor<T>, Tensor<T>) {
  return #tfop("SparseAddGrad",
    backpropValGrad,
    aIndices,
    bIndices,
    sumIndices,
    T: T.self)
}

// var: Should be from a Variable().
//
// - Parameters:
//   - accum: Should be from a Variable().
//   - accum_update: : Should be from a Variable().
//   - lr: Learning rate. Must be a scalar.
//   - rho: Decay factor. Must be a scalar.
//   - epsilon: Constant factor. Must be a scalar.
//   - grad: The gradient.
//   - indices: A vector of indices into the first dimension of var and accum.
//
// - Attr use_locking: If True, updating of the var and accum tensors will be protected by
//   a lock; otherwise the behavior is undefined, but may exhibit less contention.
//
// - Output out: Same as "var".
@_inlineable @inline(__always)
public static func sparseApplyAdadelta<T: Numeric, Tindices: BinaryInteger>(
  var_: Tensor<T>,
  accum: Tensor<T>,
  accumUpdate: Tensor<T>,
  lr: Tensor<T>,
  rho: Tensor<T>,
  epsilon: Tensor<T>,
  grad: Tensor<T>,
  indices: Tensor<Tindices>,
  useLocking: Bool = false
) -> Tensor<T> {
  return #tfop("SparseApplyAdadelta",
    var_,
    accum,
    accumUpdate,
    lr,
    rho,
    epsilon,
    grad,
    indices,
    T: T.self,
    Tindices: Tindices.self,
    use_locking: useLocking)
}

// Update relevant entries in '*var' and '*accum' according to the adagrad scheme.
//
// That is for rows we have grad for, we update var and accum as follows:
// accum += grad * grad
// var -= lr * grad * (1 / sqrt(accum))
//
// - Parameters:
//   - var: Should be from a Variable().
//   - accum: Should be from a Variable().
//   - lr: Learning rate. Must be a scalar.
//   - grad: The gradient.
//   - indices: A vector of indices into the first dimension of var and accum.
//
// - Attr use_locking: If `True`, updating of the var and accum tensors will be protected
//   by a lock; otherwise the behavior is undefined, but may exhibit less
//   contention.
//
// - Output out: Same as "var".
@_inlineable @inline(__always)
public static func sparseApplyAdagrad<T: Numeric, Tindices: BinaryInteger>(
  var_: Tensor<T>,
  accum: Tensor<T>,
  lr: Tensor<T>,
  grad: Tensor<T>,
  indices: Tensor<Tindices>,
  useLocking: Bool = false
) -> Tensor<T> {
  return #tfop("SparseApplyAdagrad",
    var_,
    accum,
    lr,
    grad,
    indices,
    T: T.self,
    Tindices: Tindices.self,
    use_locking: useLocking)
}

// Update entries in '*var' and '*accum' according to the proximal adagrad scheme.
//
// - Parameters:
//   - var: Should be from a Variable().
//   - gradient_accumulator: Should be from a Variable().
//   - gradient_squared_accumulator: Should be from a Variable().
//   - grad: The gradient.
//   - indices: A vector of indices into the first dimension of var and accum.
//   - lr: Learning rate. Must be a scalar.
//   - l1: L1 regularization. Must be a scalar.
//   - l2: L2 regularization. Must be a scalar.
//   - global_step: Training step number. Must be a scalar.
//
// - Attr use_locking: If True, updating of the var and accum tensors will be protected by
//   a lock; otherwise the behavior is undefined, but may exhibit less contention.
//
// - Output out: Same as "var".
@_inlineable @inline(__always)
public static func sparseApplyAdagradDA<T: Numeric, Tindices: BinaryInteger>(
  var_: Tensor<T>,
  gradientAccumulator: Tensor<T>,
  gradientSquaredAccumulator: Tensor<T>,
  grad: Tensor<T>,
  indices: Tensor<Tindices>,
  lr: Tensor<T>,
  l1: Tensor<T>,
  l2: Tensor<T>,
  globalStep: Tensor<Int64>,
  useLocking: Bool = false
) -> Tensor<T> {
  return #tfop("SparseApplyAdagradDA",
    var_,
    gradientAccumulator,
    gradientSquaredAccumulator,
    grad,
    indices,
    lr,
    l1,
    l2,
    globalStep,
    T: T.self,
    Tindices: Tindices.self,
    use_locking: useLocking)
}

// Update '*var' according to the centered RMSProp algorithm.
//
// The centered RMSProp algorithm uses an estimate of the centered second moment
// (i.e., the variance) for normalization, as opposed to regular RMSProp, which
// uses the (uncentered) second moment. This often helps with training, but is
// slightly more expensive in terms of computation and memory.
//
// Note that in dense implementation of this algorithm, mg, ms, and mom will
// update even if the grad is zero, but in this sparse implementation, mg, ms,
// and mom will not update in iterations during which the grad is zero.
//
// mean_square = decay * mean_square + (1-decay) * gradient ** 2
// mean_grad = decay * mean_grad + (1-decay) * gradient
// Delta = learning_rate * gradient / sqrt(mean_square + epsilon - mean_grad ** 2)
//
// ms <- rho * ms_{t-1} + (1-rho) * grad * grad
// mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
// var <- var - mom
//
// - Parameters:
//   - var: Should be from a Variable().
//   - mg: Should be from a Variable().
//   - ms: Should be from a Variable().
//   - mom: Should be from a Variable().
//   - lr: Scaling factor. Must be a scalar.
//   - rho: Decay rate. Must be a scalar.
//   - epsilon: Ridge term. Must be a scalar.
//   - grad: The gradient.
//   - indices: A vector of indices into the first dimension of var, ms and mom.
//
// - Attr use_locking: If `True`, updating of the var, mg, ms, and mom tensors is
//   protected by a lock; otherwise the behavior is undefined, but may exhibit less
//   contention.
//
// - Output out: Same as "var".
@_inlineable @inline(__always)
public static func sparseApplyCenteredRMSProp<T: Numeric, Tindices: BinaryInteger>(
  var_: Tensor<T>,
  mg: Tensor<T>,
  ms: Tensor<T>,
  mom: Tensor<T>,
  lr: Tensor<T>,
  rho: Tensor<T>,
  momentum: Tensor<T>,
  epsilon: Tensor<T>,
  grad: Tensor<T>,
  indices: Tensor<Tindices>,
  useLocking: Bool = false
) -> Tensor<T> {
  return #tfop("SparseApplyCenteredRMSProp",
    var_,
    mg,
    ms,
    mom,
    lr,
    rho,
    momentum,
    epsilon,
    grad,
    indices,
    T: T.self,
    Tindices: Tindices.self,
    use_locking: useLocking)
}

// Update relevant entries in '*var' according to the Ftrl-proximal scheme.
//
// That is for rows we have grad for, we update var, accum and linear as follows:
// accum_new = accum + grad * grad
// linear += grad + (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
// quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
// var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
// accum = accum_new
//
// - Parameters:
//   - var: Should be from a Variable().
//   - accum: Should be from a Variable().
//   - linear: Should be from a Variable().
//   - grad: The gradient.
//   - indices: A vector of indices into the first dimension of var and accum.
//   - lr: Scaling factor. Must be a scalar.
//   - l1: L1 regularization. Must be a scalar.
//   - l2: L2 regularization. Must be a scalar.
//   - lr_power: Scaling factor. Must be a scalar.
//
// - Attr use_locking: If `True`, updating of the var and accum tensors will be protected
//   by a lock; otherwise the behavior is undefined, but may exhibit less
//   contention.
//
// - Output out: Same as "var".
@_inlineable @inline(__always)
public static func sparseApplyFtrl<T: Numeric, Tindices: BinaryInteger>(
  var_: Tensor<T>,
  accum: Tensor<T>,
  linear: Tensor<T>,
  grad: Tensor<T>,
  indices: Tensor<Tindices>,
  lr: Tensor<T>,
  l1: Tensor<T>,
  l2: Tensor<T>,
  lrPower: Tensor<T>,
  useLocking: Bool = false
) -> Tensor<T> {
  return #tfop("SparseApplyFtrl",
    var_,
    accum,
    linear,
    grad,
    indices,
    lr,
    l1,
    l2,
    lrPower,
    T: T.self,
    Tindices: Tindices.self,
    use_locking: useLocking)
}

// Update relevant entries in '*var' according to the Ftrl-proximal scheme.
//
// That is for rows we have grad for, we update var, accum and linear as follows:
// grad_with_shrinkage = grad + 2 * l2_shrinkage * var
// accum_new = accum + grad_with_shrinkage * grad_with_shrinkage
// linear += grad_with_shrinkage +
//     (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
// quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
// var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
// accum = accum_new
//
// - Parameters:
//   - var: Should be from a Variable().
//   - accum: Should be from a Variable().
//   - linear: Should be from a Variable().
//   - grad: The gradient.
//   - indices: A vector of indices into the first dimension of var and accum.
//   - lr: Scaling factor. Must be a scalar.
//   - l1: L1 regularization. Must be a scalar.
//   - l2: L2 shrinkage regulariation. Must be a scalar.
//   - lr_power: Scaling factor. Must be a scalar.
//
// - Attr use_locking: If `True`, updating of the var and accum tensors will be protected
//   by a lock; otherwise the behavior is undefined, but may exhibit less
//   contention.
//
// - Output out: Same as "var".
@_inlineable @inline(__always)
public static func sparseApplyFtrlV2<T: Numeric, Tindices: BinaryInteger>(
  var_: Tensor<T>,
  accum: Tensor<T>,
  linear: Tensor<T>,
  grad: Tensor<T>,
  indices: Tensor<Tindices>,
  lr: Tensor<T>,
  l1: Tensor<T>,
  l2: Tensor<T>,
  l2Shrinkage: Tensor<T>,
  lrPower: Tensor<T>,
  useLocking: Bool = false
) -> Tensor<T> {
  return #tfop("SparseApplyFtrlV2",
    var_,
    accum,
    linear,
    grad,
    indices,
    lr,
    l1,
    l2,
    l2Shrinkage,
    lrPower,
    T: T.self,
    Tindices: Tindices.self,
    use_locking: useLocking)
}

// Update relevant entries in '*var' and '*accum' according to the momentum scheme.
//
// Set use_nesterov = True if you want to use Nesterov momentum.
//
// That is for rows we have grad for, we update var and accum as follows:
//
// accum = accum * momentum + grad
// var -= lr * accum
//
// - Parameters:
//   - var: Should be from a Variable().
//   - accum: Should be from a Variable().
//   - lr: Learning rate. Must be a scalar.
//   - grad: The gradient.
//   - indices: A vector of indices into the first dimension of var and accum.
//   - momentum: Momentum. Must be a scalar.
//
// - Attrs:
//   - use_locking: If `True`, updating of the var and accum tensors will be protected
//     by a lock; otherwise the behavior is undefined, but may exhibit less
//     contention.
//   - use_nesterov: If `True`, the tensor passed to compute grad will be
//     var - lr * momentum * accum, so in the end, the var you get is actually
//     var - lr * momentum * accum.
//
// - Output out: Same as "var".
@_inlineable @inline(__always)
public static func sparseApplyMomentum<T: Numeric, Tindices: BinaryInteger>(
  var_: Tensor<T>,
  accum: Tensor<T>,
  lr: Tensor<T>,
  grad: Tensor<T>,
  indices: Tensor<Tindices>,
  momentum: Tensor<T>,
  useLocking: Bool = false,
  useNesterov: Bool = false
) -> Tensor<T> {
  return #tfop("SparseApplyMomentum",
    var_,
    accum,
    lr,
    grad,
    indices,
    momentum,
    T: T.self,
    Tindices: Tindices.self,
    use_locking: useLocking,
    use_nesterov: useNesterov)
}

// Sparse update entries in '*var' and '*accum' according to FOBOS algorithm.
//
// That is for rows we have grad for, we update var and accum as follows:
// accum += grad * grad
// prox_v = var
// prox_v -= lr * grad * (1 / sqrt(accum))
// var = sign(prox_v)/(1+lr*l2) * max{|prox_v|-lr*l1,0}
//
// - Parameters:
//   - var: Should be from a Variable().
//   - accum: Should be from a Variable().
//   - lr: Learning rate. Must be a scalar.
//   - l1: L1 regularization. Must be a scalar.
//   - l2: L2 regularization. Must be a scalar.
//   - grad: The gradient.
//   - indices: A vector of indices into the first dimension of var and accum.
//
// - Attr use_locking: If True, updating of the var and accum tensors will be protected by
//   a lock; otherwise the behavior is undefined, but may exhibit less contention.
//
// - Output out: Same as "var".
@_inlineable @inline(__always)
public static func sparseApplyProximalAdagrad<T: Numeric, Tindices: BinaryInteger>(
  var_: Tensor<T>,
  accum: Tensor<T>,
  lr: Tensor<T>,
  l1: Tensor<T>,
  l2: Tensor<T>,
  grad: Tensor<T>,
  indices: Tensor<Tindices>,
  useLocking: Bool = false
) -> Tensor<T> {
  return #tfop("SparseApplyProximalAdagrad",
    var_,
    accum,
    lr,
    l1,
    l2,
    grad,
    indices,
    T: T.self,
    Tindices: Tindices.self,
    use_locking: useLocking)
}

// Sparse update '*var' as FOBOS algorithm with fixed learning rate.
//
// That is for rows we have grad for, we update var as follows:
// prox_v = var - alpha * grad
// var = sign(prox_v)/(1+alpha*l2) * max{|prox_v|-alpha*l1,0}
//
// - Parameters:
//   - var: Should be from a Variable().
//   - alpha: Scaling factor. Must be a scalar.
//   - l1: L1 regularization. Must be a scalar.
//   - l2: L2 regularization. Must be a scalar.
//   - grad: The gradient.
//   - indices: A vector of indices into the first dimension of var and accum.
//
// - Attr use_locking: If True, the subtraction will be protected by a lock;
//   otherwise the behavior is undefined, but may exhibit less contention.
//
// - Output out: Same as "var".
@_inlineable @inline(__always)
public static func sparseApplyProximalGradientDescent<T: Numeric, Tindices: BinaryInteger>(
  var_: Tensor<T>,
  alpha: Tensor<T>,
  l1: Tensor<T>,
  l2: Tensor<T>,
  grad: Tensor<T>,
  indices: Tensor<Tindices>,
  useLocking: Bool = false
) -> Tensor<T> {
  return #tfop("SparseApplyProximalGradientDescent",
    var_,
    alpha,
    l1,
    l2,
    grad,
    indices,
    T: T.self,
    Tindices: Tindices.self,
    use_locking: useLocking)
}

// Update '*var' according to the RMSProp algorithm.
//
// Note that in dense implementation of this algorithm, ms and mom will
// update even if the grad is zero, but in this sparse implementation, ms
// and mom will not update in iterations during which the grad is zero.
//
// mean_square = decay * mean_square + (1-decay) * gradient ** 2
// Delta = learning_rate * gradient / sqrt(mean_square + epsilon)
//
// ms <- rho * ms_{t-1} + (1-rho) * grad * grad
// mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
// var <- var - mom
//
// - Parameters:
//   - var: Should be from a Variable().
//   - ms: Should be from a Variable().
//   - mom: Should be from a Variable().
//   - lr: Scaling factor. Must be a scalar.
//   - rho: Decay rate. Must be a scalar.
//   - epsilon: Ridge term. Must be a scalar.
//   - grad: The gradient.
//   - indices: A vector of indices into the first dimension of var, ms and mom.
//
// - Attr use_locking: If `True`, updating of the var, ms, and mom tensors is protected
//   by a lock; otherwise the behavior is undefined, but may exhibit less
//   contention.
//
// - Output out: Same as "var".
@_inlineable @inline(__always)
public static func sparseApplyRMSProp<T: Numeric, Tindices: BinaryInteger>(
  var_: Tensor<T>,
  ms: Tensor<T>,
  mom: Tensor<T>,
  lr: Tensor<T>,
  rho: Tensor<T>,
  momentum: Tensor<T>,
  epsilon: Tensor<T>,
  grad: Tensor<T>,
  indices: Tensor<Tindices>,
  useLocking: Bool = false
) -> Tensor<T> {
  return #tfop("SparseApplyRMSProp",
    var_,
    ms,
    mom,
    lr,
    rho,
    momentum,
    epsilon,
    grad,
    indices,
    T: T.self,
    Tindices: Tindices.self,
    use_locking: useLocking)
}

// Concatenates a list of `SparseTensor` along the specified dimension.
//
// Concatenation is with respect to the dense versions of these sparse tensors.
// It is assumed that each input is a `SparseTensor` whose elements are ordered
// along increasing dimension number.
//
// All inputs' shapes must match, except for the concat dimension.  The
// `indices`, `values`, and `shapes` lists must have the same length.
//
// The output shape is identical to the inputs', except along the concat
// dimension, where it is the sum of the inputs' sizes along that dimension.
//
// The output elements will be resorted to preserve the sort order along
// increasing dimension number.
//
// This op runs in `O(M log M)` time, where `M` is the total number of non-empty
// values across all inputs. This is due to the need for an internal sort in
// order to concatenate efficiently across an arbitrary dimension.
//
// For example, if `concat_dim = 1` and the inputs are
//
//     sp_inputs[0]: shape = [2, 3]
//     [0, 2]: "a"
//     [1, 0]: "b"
//     [1, 1]: "c"
//
//     sp_inputs[1]: shape = [2, 4]
//     [0, 1]: "d"
//     [0, 2]: "e"
//
// then the output will be
//
//     shape = [2, 7]
//     [0, 2]: "a"
//     [0, 4]: "d"
//     [0, 5]: "e"
//     [1, 0]: "b"
//     [1, 1]: "c"
//
// Graphically this is equivalent to doing
//
//     [    a] concat [  d e  ] = [    a   d e  ]
//     [b c  ]        [       ]   [b c          ]
//
// - Parameters:
//   - indices: 2-D.  Indices of each input `SparseTensor`.
//   - values: 1-D.  Non-empty values of each `SparseTensor`.
//   - shapes: 1-D.  Shapes of each `SparseTensor`.
//
// - Attr concat_dim: Dimension to concatenate along. Must be in range [-rank, rank),
//   where rank is the number of dimensions in each input `SparseTensor`.
//
// - Outputs:
//   - output_indices: 2-D.  Indices of the concatenated `SparseTensor`.
//   - output_values: 1-D.  Non-empty values of the concatenated `SparseTensor`.
//   - output_shape: 1-D.  Shape of the concatenated `SparseTensor`.
@_inlineable @inline(__always)
public static func sparseConcat<T: Numeric>(
  indices: [Tensor<Int64>],
  values: [Tensor<T>],
  shapes: [Tensor<Int64>],
  concatDim: Int
) -> (Tensor<Int64>, Tensor<T>, Tensor<Int64>) {
  return #tfop("SparseConcat",
    indices,
    values,
    shapes,
    T: T.self,
    concat_dim: concatDim)
}

// Generates sparse cross from a list of sparse and dense tensors.
//
// The op takes two lists, one of 2D `SparseTensor` and one of 2D `Tensor`, each
// representing features of one feature column. It outputs a 2D `SparseTensor` with
// the batchwise crosses of these features.
//
// For example, if the inputs are
//
//     inputs[0]: SparseTensor with shape = [2, 2]
//     [0, 0]: "a"
//     [1, 0]: "b"
//     [1, 1]: "c"
//
//     inputs[1]: SparseTensor with shape = [2, 1]
//     [0, 0]: "d"
//     [1, 0]: "e"
//
//     inputs[2]: Tensor [["f"], ["g"]]
//
// then the output will be
//
//     shape = [2, 2]
//     [0, 0]: "a_X_d_X_f"
//     [1, 0]: "b_X_e_X_g"
//     [1, 1]: "c_X_e_X_g"
//
// if hashed_output=true then the output will be
//
//     shape = [2, 2]
//     [0, 0]: FingerprintCat64(
//                 Fingerprint64("f"), FingerprintCat64(
//                     Fingerprint64("d"), Fingerprint64("a")))
//     [1, 0]: FingerprintCat64(
//                 Fingerprint64("g"), FingerprintCat64(
//                     Fingerprint64("e"), Fingerprint64("b")))
//     [1, 1]: FingerprintCat64(
//                 Fingerprint64("g"), FingerprintCat64(
//                     Fingerprint64("e"), Fingerprint64("c")))
//
// - Parameters:
//   - indices: 2-D.  Indices of each input `SparseTensor`.
//   - values: 1-D.   values of each `SparseTensor`.
//   - shapes: 1-D.   Shapes of each `SparseTensor`.
//   - dense_inputs: 2-D.    Columns represented by dense `Tensor`.
//
// - Attrs:
//   - hashed_output: If true, returns the hash of the cross instead of the string.
//     This will allow us avoiding string manipulations.
//   - num_buckets: It is used if hashed_output is true.
//     output = hashed_value%num_buckets if num_buckets > 0 else hashed_value.
//   - hash_key: Specify the hash_key that will be used by the `FingerprintCat64`
//     function to combine the crosses fingerprints.
//
// - Outputs:
//   - output_indices: 2-D.  Indices of the concatenated `SparseTensor`.
//   - output_values: 1-D.  Non-empty values of the concatenated or hashed
//     `SparseTensor`.
//   - output_shape: 1-D.  Shape of the concatenated `SparseTensor`.
@_inlineable @inline(__always)
public static func sparseCross<Sparse_types: BinaryInteger, Dense_types: BinaryInteger, Out_type: BinaryInteger, Internal_type: BinaryInteger>(
  indices: [Tensor<Int64>],
  values: [Tensor<Sparse_types>],
  shapes: [Tensor<Int64>],
  denseInputs: [Tensor<Dense_types>],
  hashedOutput: Bool,
  numBuckets: Int,
  hashKey: Int,
  typeInternal_type: Internal_type.Type
) -> (Tensor<Int64>, Tensor<Out_type>, Tensor<Int64>) {
  return #tfop("SparseCross",
    indices,
    values,
    shapes,
    denseInputs,
    Out_type: Out_type.self,
    Internal_type: Internal_type.self,
    hashed_output: hashedOutput,
    num_buckets: numBuckets,
    hash_key: hashKey)
}

// Adds up a SparseTensor and a dense Tensor, using these special rules:
//
// (1) Broadcasts the dense side to have the same shape as the sparse side, if
//     eligible;
// (2) Then, only the dense values pointed to by the indices of the SparseTensor
//     participate in the cwise addition.
//
// By these rules, the result is a logical SparseTensor with exactly the same
// indices and shape, but possibly with different non-zero values.  The output of
// this Op is the resultant non-zero values.
//
// - Parameters:
//   - sp_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
//     SparseTensor, possibly not in canonical ordering.
//   - sp_values: 1-D.  `N` non-empty values corresponding to `sp_indices`.
//   - sp_shape: 1-D.  Shape of the input SparseTensor.
//   - dense: `R`-D.  The dense Tensor operand.
//
// - Output output: 1-D.  The `N` values that are operated on.
@_inlineable @inline(__always)
public static func sparseDenseCwiseAdd<T: Numeric>(
  spIndices: Tensor<Int64>,
  spValues: Tensor<T>,
  spShape: Tensor<Int64>,
  dense: Tensor<T>
) -> Tensor<T> {
  return #tfop("SparseDenseCwiseAdd",
    spIndices,
    spValues,
    spShape,
    dense,
    T: T.self)
}

// Component-wise divides a SparseTensor by a dense Tensor.
//
// *Limitation*: this Op only broadcasts the dense side to the sparse side, but not
// the other direction.
//
// - Parameters:
//   - sp_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
//     SparseTensor, possibly not in canonical ordering.
//   - sp_values: 1-D.  `N` non-empty values corresponding to `sp_indices`.
//   - sp_shape: 1-D.  Shape of the input SparseTensor.
//   - dense: `R`-D.  The dense Tensor operand.
//
// - Output output: 1-D.  The `N` values that are operated on.
@_inlineable @inline(__always)
public static func sparseDenseCwiseDiv<T: Numeric>(
  spIndices: Tensor<Int64>,
  spValues: Tensor<T>,
  spShape: Tensor<Int64>,
  dense: Tensor<T>
) -> Tensor<T> {
  return #tfop("SparseDenseCwiseDiv",
    spIndices,
    spValues,
    spShape,
    dense,
    T: T.self)
}

// Component-wise multiplies a SparseTensor by a dense Tensor.
//
// The output locations corresponding to the implicitly zero elements in the sparse
// tensor will be zero (i.e., will not take up storage space), regardless of the
// contents of the dense tensor (even if it's +/-INF and that INF*0 == NaN).
//
// *Limitation*: this Op only broadcasts the dense side to the sparse side, but not
// the other direction.
//
// - Parameters:
//   - sp_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
//     SparseTensor, possibly not in canonical ordering.
//   - sp_values: 1-D.  `N` non-empty values corresponding to `sp_indices`.
//   - sp_shape: 1-D.  Shape of the input SparseTensor.
//   - dense: `R`-D.  The dense Tensor operand.
//
// - Output output: 1-D.  The `N` values that are operated on.
@_inlineable @inline(__always)
public static func sparseDenseCwiseMul<T: Numeric>(
  spIndices: Tensor<Int64>,
  spValues: Tensor<T>,
  spShape: Tensor<Int64>,
  dense: Tensor<T>
) -> Tensor<T> {
  return #tfop("SparseDenseCwiseMul",
    spIndices,
    spValues,
    spShape,
    dense,
    T: T.self)
}

// Fills empty rows in the input 2-D `SparseTensor` with a default value.
//
// The input `SparseTensor` is represented via the tuple of inputs
// (`indices`, `values`, `dense_shape`).  The output `SparseTensor` has the
// same `dense_shape` but with indices `output_indices` and values
// `output_values`.
//
// This op inserts a single entry for every row that doesn't have any values.
// The index is created as `[row, 0, ..., 0]` and the inserted value
// is `default_value`.
//
// For example, suppose `sp_input` has shape `[5, 6]` and non-empty values:
//
//     [0, 1]: a
//     [0, 3]: b
//     [2, 0]: c
//     [3, 1]: d
//
// Rows 1 and 4 are empty, so the output will be of shape `[5, 6]` with values:
//
//     [0, 1]: a
//     [0, 3]: b
//     [1, 0]: default_value
//     [2, 0]: c
//     [3, 1]: d
//     [4, 0]: default_value
//
// The output `SparseTensor` will be in row-major order and will have the
// same shape as the input.
//
// This op also returns an indicator vector shaped `[dense_shape[0]]` such that
//
//     empty_row_indicator[i] = True iff row i was an empty row.
//
// And a reverse index map vector shaped `[indices.shape[0]]` that is used during
// backpropagation,
//
//     reverse_index_map[j] = out_j s.t. indices[j, :] == output_indices[out_j, :]
//
// - Parameters:
//   - indices: 2-D. the indices of the sparse tensor.
//   - values: 1-D. the values of the sparse tensor.
//   - dense_shape: 1-D. the shape of the sparse tensor.
//   - default_value: 0-D. default value to insert into location `[row, 0, ..., 0]`
//       for rows missing from the input sparse tensor.
//     output indices: 2-D. the indices of the filled sparse tensor.
//
// - Outputs:
//   - output_values: 1-D. the values of the filled sparse tensor.
//   - empty_row_indicator: 1-D. whether the dense row was missing in the
//     input sparse tensor.
//   - reverse_index_map: 1-D. a map from the input indices to the output indices.
@_inlineable @inline(__always)
public static func sparseFillEmptyRows<T: Numeric>(
  indices: Tensor<Int64>,
  values: Tensor<T>,
  denseShape: Tensor<Int64>,
  defaultValue: Tensor<T>
) -> (Tensor<Int64>, Tensor<T>, Tensor<Bool>, Tensor<Int64>) {
  return #tfop("SparseFillEmptyRows",
    indices,
    values,
    denseShape,
    defaultValue,
    T: T.self)
}

// The gradient of SparseFillEmptyRows.
//
// Takes vectors reverse_index_map, shaped `[N]`, and grad_values,
// shaped `[N_full]`, where `N_full >= N` and copies data into either
// `d_values` or `d_default_value`.  Here `d_values` is shaped `[N]` and
// `d_default_value` is a scalar.
//
//   d_values[j] = grad_values[reverse_index_map[j]]
//   d_default_value = sum_{k : 0 .. N_full - 1} (
//      grad_values[k] * 1{k not in reverse_index_map})
//
// - Parameters:
//   - reverse_index_map: 1-D.  The reverse index map from SparseFillEmptyRows.
//   - grad_values: 1-D.  The gradients from backprop.
//
// - Outputs:
//   - d_values: 1-D.  The backprop into values.
//   - d_default_value: 0-D.  The backprop into default_value.
@_inlineable @inline(__always)
public static func sparseFillEmptyRowsGrad<T: Numeric>(
  reverseIndexMap: Tensor<Int64>,
  gradValues: Tensor<T>
) -> (Tensor<T>, Tensor<T>) {
  return #tfop("SparseFillEmptyRowsGrad",
    reverseIndexMap,
    gradValues,
    T: T.self)
}

// Multiply matrix "a" by matrix "b".
//
// The inputs must be two-dimensional matrices and the inner dimension of "a" must
// match the outer dimension of "b". This op is optimized for the case where at
// least one of "a" or "b" is sparse. The breakeven for using this versus a dense
// matrix multiply on one platform was 30% zero values in the sparse matrix.
//
// The gradient computation of this operation will only take advantage of sparsity
// in the input gradient when that gradient comes from a Relu.
@_inlineable @inline(__always)
public static func sparseMatMul<Ta: BinaryFloatingPoint, Tb: BinaryFloatingPoint>(
  a: Tensor<Ta>,
  b: Tensor<Tb>,
  transposeA: Bool = false,
  transposeB: Bool = false,
  aIsSparse: Bool = false,
  bIsSparse: Bool = false
) -> Tensor<Float> {
  return #tfop("SparseMatMul",
    a,
    b,
    Ta: Ta.self,
    Tb: Tb.self,
    transpose_a: transposeA,
    transpose_b: transposeB,
    a_is_sparse: aIsSparse,
    b_is_sparse: bIsSparse)
}

// Computes the max of elements across dimensions of a SparseTensor.
//
// This Op takes a SparseTensor and is the sparse counterpart to
// `tf.reduce_max()`.  In particular, this Op also returns a dense `Tensor`
// instead of a sparse one.
//
// Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
// `reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
// with length 1.
//
// If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
// with a single element is returned.  Additionally, the axes can be negative,
// which are interpreted according to the indexing rules in Python.
//
// - Parameters:
//   - input_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
//     SparseTensor, possibly not in canonical ordering.
//   - input_values: 1-D.  `N` non-empty values corresponding to `input_indices`.
//   - input_shape: 1-D.  Shape of the input SparseTensor.
//   - reduction_axes: 1-D.  Length-`K` vector containing the reduction axes.
//
// - Attr keep_dims: If true, retain reduced dimensions with length 1.
//
// - Output output: `R-K`-D.  The reduced Tensor.
@_inlineable @inline(__always)
public static func sparseReduceMax<T: Numeric>(
  inputIndices: Tensor<Int64>,
  inputValues: Tensor<T>,
  inputShape: Tensor<Int64>,
  reductionAxes: Tensor<Int32>,
  keepDims: Bool = false
) -> Tensor<T> {
  return #tfop("SparseReduceMax",
    inputIndices,
    inputValues,
    inputShape,
    reductionAxes,
    T: T.self,
    keep_dims: keepDims)
}

// Computes the max of elements across dimensions of a SparseTensor.
//
// This Op takes a SparseTensor and is the sparse counterpart to
// `tf.reduce_max()`.  In contrast to SparseReduceMax, this Op returns a
// SparseTensor.
//
// Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
// `reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
// with length 1.
//
// If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
// with a single element is returned.  Additionally, the axes can be negative,
// which are interpreted according to the indexing rules in Python.
//
// - Parameters:
//   - input_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
//     SparseTensor, possibly not in canonical ordering.
//   - input_values: 1-D.  `N` non-empty values corresponding to `input_indices`.
//   - input_shape: 1-D.  Shape of the input SparseTensor.
//   - reduction_axes: 1-D.  Length-`K` vector containing the reduction axes.
//
// - Attr keep_dims: If true, retain reduced dimensions with length 1.
@_inlineable @inline(__always)
public static func sparseReduceMaxSparse<T: Numeric>(
  inputIndices: Tensor<Int64>,
  inputValues: Tensor<T>,
  inputShape: Tensor<Int64>,
  reductionAxes: Tensor<Int32>,
  keepDims: Bool = false
) -> (Tensor<Int64>, Tensor<T>, Tensor<Int64>) {
  return #tfop("SparseReduceMaxSparse",
    inputIndices,
    inputValues,
    inputShape,
    reductionAxes,
    T: T.self,
    keep_dims: keepDims)
}

// Computes the sum of elements across dimensions of a SparseTensor.
//
// This Op takes a SparseTensor and is the sparse counterpart to
// `tf.reduce_sum()`.  In particular, this Op also returns a dense `Tensor`
// instead of a sparse one.
//
// Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
// `reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
// with length 1.
//
// If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
// with a single element is returned.  Additionally, the axes can be negative,
// which are interpreted according to the indexing rules in Python.
//
// - Parameters:
//   - input_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
//     SparseTensor, possibly not in canonical ordering.
//   - input_values: 1-D.  `N` non-empty values corresponding to `input_indices`.
//   - input_shape: 1-D.  Shape of the input SparseTensor.
//   - reduction_axes: 1-D.  Length-`K` vector containing the reduction axes.
//
// - Attr keep_dims: If true, retain reduced dimensions with length 1.
//
// - Output output: `R-K`-D.  The reduced Tensor.
@_inlineable @inline(__always)
public static func sparseReduceSum<T: Numeric>(
  inputIndices: Tensor<Int64>,
  inputValues: Tensor<T>,
  inputShape: Tensor<Int64>,
  reductionAxes: Tensor<Int32>,
  keepDims: Bool = false
) -> Tensor<T> {
  return #tfop("SparseReduceSum",
    inputIndices,
    inputValues,
    inputShape,
    reductionAxes,
    T: T.self,
    keep_dims: keepDims)
}

// Computes the sum of elements across dimensions of a SparseTensor.
//
// This Op takes a SparseTensor and is the sparse counterpart to
// `tf.reduce_sum()`.  In contrast to SparseReduceSum, this Op returns a
// SparseTensor.
//
// Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
// `reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
// with length 1.
//
// If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
// with a single element is returned.  Additionally, the axes can be negative,
// which are interpreted according to the indexing rules in Python.
//
// - Parameters:
//   - input_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
//     SparseTensor, possibly not in canonical ordering.
//   - input_values: 1-D.  `N` non-empty values corresponding to `input_indices`.
//   - input_shape: 1-D.  Shape of the input SparseTensor.
//   - reduction_axes: 1-D.  Length-`K` vector containing the reduction axes.
//
// - Attr keep_dims: If true, retain reduced dimensions with length 1.
@_inlineable @inline(__always)
public static func sparseReduceSumSparse<T: Numeric>(
  inputIndices: Tensor<Int64>,
  inputValues: Tensor<T>,
  inputShape: Tensor<Int64>,
  reductionAxes: Tensor<Int32>,
  keepDims: Bool = false
) -> (Tensor<Int64>, Tensor<T>, Tensor<Int64>) {
  return #tfop("SparseReduceSumSparse",
    inputIndices,
    inputValues,
    inputShape,
    reductionAxes,
    T: T.self,
    keep_dims: keepDims)
}

// Reorders a SparseTensor into the canonical, row-major ordering.
//
// Note that by convention, all sparse ops preserve the canonical ordering along
// increasing dimension number. The only time ordering can be violated is during
// manual manipulation of the indices and values vectors to add entries.
//
// Reordering does not affect the shape of the SparseTensor.
//
// If the tensor has rank `R` and `N` non-empty values, `input_indices` has
// shape `[N, R]`, input_values has length `N`, and input_shape has length `R`.
//
// - Parameters:
//   - input_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
//     SparseTensor, possibly not in canonical ordering.
//   - input_values: 1-D.  `N` non-empty values corresponding to `input_indices`.
//   - input_shape: 1-D.  Shape of the input SparseTensor.
//
// - Outputs:
//   - output_indices: 2-D.  `N x R` matrix with the same indices as input_indices, but
//     in canonical row-major ordering.
//   - output_values: 1-D.  `N` non-empty values corresponding to `output_indices`.
@_inlineable @inline(__always)
public static func sparseReorder<T: Numeric>(
  inputIndices: Tensor<Int64>,
  inputValues: Tensor<T>,
  inputShape: Tensor<Int64>
) -> (Tensor<Int64>, Tensor<T>) {
  return #tfop("SparseReorder",
    inputIndices,
    inputValues,
    inputShape,
    T: T.self)
}

// Reshapes a SparseTensor to represent values in a new dense shape.
//
// This operation has the same semantics as reshape on the represented dense
// tensor.  The `input_indices` are recomputed based on the requested `new_shape`.
//
// If one component of `new_shape` is the special value -1, the size of that
// dimension is computed so that the total dense size remains constant.  At
// most one component of `new_shape` can be -1.  The number of dense elements
// implied by `new_shape` must be the same as the number of dense elements
// originally implied by `input_shape`.
//
// Reshaping does not affect the order of values in the SparseTensor.
//
// If the input tensor has rank `R_in` and `N` non-empty values, and `new_shape`
// has length `R_out`, then `input_indices` has shape `[N, R_in]`,
// `input_shape` has length `R_in`, `output_indices` has shape `[N, R_out]`, and
// `output_shape` has length `R_out`.
//
// - Parameters:
//   - input_indices: 2-D.  `N x R_in` matrix with the indices of non-empty values in a
//     SparseTensor.
//   - input_shape: 1-D.  `R_in` vector with the input SparseTensor's dense shape.
//   - new_shape: 1-D.  `R_out` vector with the requested new dense shape.
//
// - Outputs:
//   - output_indices: 2-D.  `N x R_out` matrix with the updated indices of non-empty
//     values in the output SparseTensor.
//   - output_shape: 1-D.  `R_out` vector with the full dense shape of the output
//     SparseTensor.  This is the same as `new_shape` but with any -1 dimensions
//     filled in.
@_inlineable @inline(__always)
public static func sparseReshape(
  inputIndices: Tensor<Int64>,
  inputShape: Tensor<Int64>,
  newShape: Tensor<Int64>
) -> (Tensor<Int64>, Tensor<Int64>) {
  return #tfop("SparseReshape",
    inputIndices,
    inputShape,
    newShape)
}

// Computes the mean along sparse segments of a tensor.
//
// Read @{$math_ops#Segmentation$the section on segmentation} for an explanation of
// segments.
//
// Like `SegmentMean`, but `segment_ids` can have rank less than `data`'s first
// dimension, selecting a subset of dimension 0, specified by `indices`.
//
// - Parameters:
//   - indices: A 1-D tensor. Has same rank as `segment_ids`.
//   - segment_ids: A 1-D tensor. Values should be sorted and can be repeated.
//
// - Output output: Has same shape as data, except for dimension 0 which
//   has size `k`, the number of segments.
@_inlineable @inline(__always)
public static func sparseSegmentMean<T: BinaryFloatingPoint, Tidx: BinaryInteger>(
  data: Tensor<T>,
  indices: Tensor<Tidx>,
  segmentIds: Tensor<Int32>
) -> Tensor<T> {
  return #tfop("SparseSegmentMean",
    data,
    indices,
    segmentIds,
    T: T.self,
    Tidx: Tidx.self)
}

// Computes gradients for SparseSegmentMean.
//
// Returns tensor "output" with same shape as grad, except for dimension 0 whose
// value is output_dim0.
//
// - Parameters:
//   - grad: gradient propagated to the SparseSegmentMean op.
//   - indices: indices passed to the corresponding SparseSegmentMean op.
//   - segment_ids: segment_ids passed to the corresponding SparseSegmentMean op.
//   - output_dim0: dimension 0 of "data" passed to SparseSegmentMean op.
@_inlineable @inline(__always)
public static func sparseSegmentMeanGrad<T: BinaryFloatingPoint, Tidx: BinaryInteger>(
  grad: Tensor<T>,
  indices: Tensor<Tidx>,
  segmentIds: Tensor<Int32>,
  outputDim0: Tensor<Int32>
) -> Tensor<T> {
  return #tfop("SparseSegmentMeanGrad",
    grad,
    indices,
    segmentIds,
    outputDim0,
    T: T.self,
    Tidx: Tidx.self)
}

// Computes the mean along sparse segments of a tensor.
//
// Like `SparseSegmentMean`, but allows missing ids in `segment_ids`. If an id is
// misisng, the `output` tensor at that position will be zeroed.
//
// Read @{$math_ops#Segmentation$the section on segmentation} for an explanation of
// segments.
//
// - Parameters:
//   - indices: A 1-D tensor. Has same rank as `segment_ids`.
//   - segment_ids: A 1-D tensor. Values should be sorted and can be repeated.
//   - num_segments: Should equal the number of distinct segment IDs.
//
// - Output output: Has same shape as data, except for dimension 0 which has size
//   `num_segments`.
@_inlineable @inline(__always)
public static func sparseSegmentMeanWithNumSegments<T: BinaryFloatingPoint, Tidx: BinaryInteger, Tnumsegments: BinaryInteger>(
  data: Tensor<T>,
  indices: Tensor<Tidx>,
  segmentIds: Tensor<Int32>,
  numSegments: Tensor<Tnumsegments>
) -> Tensor<T> {
  return #tfop("SparseSegmentMeanWithNumSegments",
    data,
    indices,
    segmentIds,
    numSegments,
    T: T.self,
    Tidx: Tidx.self,
    Tnumsegments: Tnumsegments.self)
}

// Computes the sum along sparse segments of a tensor divided by the sqrt of N.
//
// N is the size of the segment being reduced.
//
// Read @{$math_ops#Segmentation$the section on segmentation} for an explanation of
// segments.
//
// - Parameters:
//   - indices: A 1-D tensor. Has same rank as `segment_ids`.
//   - segment_ids: A 1-D tensor. Values should be sorted and can be repeated.
//
// - Output output: Has same shape as data, except for dimension 0 which
//   has size `k`, the number of segments.
@_inlineable @inline(__always)
public static func sparseSegmentSqrtN<T: BinaryFloatingPoint, Tidx: BinaryInteger>(
  data: Tensor<T>,
  indices: Tensor<Tidx>,
  segmentIds: Tensor<Int32>
) -> Tensor<T> {
  return #tfop("SparseSegmentSqrtN",
    data,
    indices,
    segmentIds,
    T: T.self,
    Tidx: Tidx.self)
}

// Computes gradients for SparseSegmentSqrtN.
//
// Returns tensor "output" with same shape as grad, except for dimension 0 whose
// value is output_dim0.
//
// - Parameters:
//   - grad: gradient propagated to the SparseSegmentSqrtN op.
//   - indices: indices passed to the corresponding SparseSegmentSqrtN op.
//   - segment_ids: segment_ids passed to the corresponding SparseSegmentSqrtN op.
//   - output_dim0: dimension 0 of "data" passed to SparseSegmentSqrtN op.
@_inlineable @inline(__always)
public static func sparseSegmentSqrtNGrad<T: BinaryFloatingPoint, Tidx: BinaryInteger>(
  grad: Tensor<T>,
  indices: Tensor<Tidx>,
  segmentIds: Tensor<Int32>,
  outputDim0: Tensor<Int32>
) -> Tensor<T> {
  return #tfop("SparseSegmentSqrtNGrad",
    grad,
    indices,
    segmentIds,
    outputDim0,
    T: T.self,
    Tidx: Tidx.self)
}

// Computes the sum along sparse segments of a tensor divided by the sqrt of N.
//
// N is the size of the segment being reduced.
//
// Like `SparseSegmentSqrtN`, but allows missing ids in `segment_ids`. If an id is
// misisng, the `output` tensor at that position will be zeroed.
//
// Read @{$math_ops#Segmentation$the section on segmentation} for an explanation of
// segments.
//
// - Parameters:
//   - indices: A 1-D tensor. Has same rank as `segment_ids`.
//   - segment_ids: A 1-D tensor. Values should be sorted and can be repeated.
//   - num_segments: Should equal the number of distinct segment IDs.
//
// - Output output: Has same shape as data, except for dimension 0 which
//   has size `k`, the number of segments.
@_inlineable @inline(__always)
public static func sparseSegmentSqrtNWithNumSegments<T: BinaryFloatingPoint, Tidx: BinaryInteger, Tnumsegments: BinaryInteger>(
  data: Tensor<T>,
  indices: Tensor<Tidx>,
  segmentIds: Tensor<Int32>,
  numSegments: Tensor<Tnumsegments>
) -> Tensor<T> {
  return #tfop("SparseSegmentSqrtNWithNumSegments",
    data,
    indices,
    segmentIds,
    numSegments,
    T: T.self,
    Tidx: Tidx.self,
    Tnumsegments: Tnumsegments.self)
}

// Computes the sum along sparse segments of a tensor.
//
// Read @{$math_ops#Segmentation$the section on segmentation} for an explanation of
// segments.
//
// Like `SegmentSum`, but `segment_ids` can have rank less than `data`'s first
// dimension, selecting a subset of dimension 0, specified by `indices`.
//
// For example:
//
// ```python
// c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])
//
// # Select two rows, one segment.
// tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 0]))
// # => [[0 0 0 0]]
//
// # Select two rows, two segment.
// tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 1]))
// # => [[ 1  2  3  4]
// #     [-1 -2 -3 -4]]
//
// # Select all rows, two segments.
// tf.sparse_segment_sum(c, tf.constant([0, 1, 2]), tf.constant([0, 0, 1]))
// # => [[0 0 0 0]
// #     [5 6 7 8]]
//
// # Which is equivalent to:
// tf.segment_sum(c, tf.constant([0, 0, 1]))
// ```
//
// - Parameters:
//   - indices: A 1-D tensor. Has same rank as `segment_ids`.
//   - segment_ids: A 1-D tensor. Values should be sorted and can be repeated.
//
// - Output output: Has same shape as data, except for dimension 0 which
//   has size `k`, the number of segments.
@_inlineable @inline(__always)
public static func sparseSegmentSum<T: Numeric, Tidx: BinaryInteger>(
  data: Tensor<T>,
  indices: Tensor<Tidx>,
  segmentIds: Tensor<Int32>
) -> Tensor<T> {
  return #tfop("SparseSegmentSum",
    data,
    indices,
    segmentIds,
    T: T.self,
    Tidx: Tidx.self)
}

// Computes the sum along sparse segments of a tensor.
//
// Like `SparseSegmentSum`, but allows missing ids in `segment_ids`. If an id is
// misisng, the `output` tensor at that position will be zeroed.
//
// Read @{$math_ops#Segmentation$the section on segmentation} for an explanation of
// segments.
//
// For example:
//
// ```python
// c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])
//
// tf.sparse_segment_sum_with_num_segments(
//     c, tf.constant([0, 1]), tf.constant([0, 0]), num_segments=3)
// # => [[0 0 0 0]
// #     [0 0 0 0]
// #     [0 0 0 0]]
//
// tf.sparse_segment_sum_with_num_segments(c,
//                                         tf.constant([0, 1]),
//                                         tf.constant([0, 2],
//                                         num_segments=4))
// # => [[ 1  2  3  4]
// #     [ 0  0  0  0]
// #     [-1 -2 -3 -4]
// #     [ 0  0  0  0]]
// ```
//
// - Parameters:
//   - indices: A 1-D tensor. Has same rank as `segment_ids`.
//   - segment_ids: A 1-D tensor. Values should be sorted and can be repeated.
//   - num_segments: Should equal the number of distinct segment IDs.
//
// - Output output: Has same shape as data, except for dimension 0 which
//   has size `num_segments`.
@_inlineable @inline(__always)
public static func sparseSegmentSumWithNumSegments<T: Numeric, Tidx: BinaryInteger, Tnumsegments: BinaryInteger>(
  data: Tensor<T>,
  indices: Tensor<Tidx>,
  segmentIds: Tensor<Int32>,
  numSegments: Tensor<Tnumsegments>
) -> Tensor<T> {
  return #tfop("SparseSegmentSumWithNumSegments",
    data,
    indices,
    segmentIds,
    numSegments,
    T: T.self,
    Tidx: Tidx.self,
    Tnumsegments: Tnumsegments.self)
}

// Slice a `SparseTensor` based on the `start` and `size`.
//
// For example, if the input is
//
//     input_tensor = shape = [2, 7]
//     [    a   d e  ]
//     [b c          ]
//
// Graphically the output tensors are:
//
//     sparse_slice([0, 0], [2, 4]) = shape = [2, 4]
//     [    a  ]
//     [b c    ]
//
//     sparse_slice([0, 4], [2, 3]) = shape = [2, 3]
//     [ d e  ]
//     [      ]
//
// - Parameters:
//   - indices: 2-D tensor represents the indices of the sparse tensor.
//   - values: 1-D tensor represents the values of the sparse tensor.
//   - shape: 1-D. tensor represents the shape of the sparse tensor.
//   - start: 1-D. tensor represents the start of the slice.
//   - size: 1-D. tensor represents the size of the slice.
//     output indices: A list of 1-D tensors represents the indices of the output
//     sparse tensors.
//
// - Outputs:
//   - output_values: A list of 1-D tensors represents the values of the output sparse
//     tensors.
//   - output_shape: A list of 1-D tensors represents the shape of the output sparse
//     tensors.
@_inlineable @inline(__always)
public static func sparseSlice<T: Numeric>(
  indices: Tensor<Int64>,
  values: Tensor<T>,
  shape: Tensor<Int64>,
  start: Tensor<Int64>,
  size: Tensor<Int64>
) -> (Tensor<Int64>, Tensor<T>, Tensor<Int64>) {
  return #tfop("SparseSlice",
    indices,
    values,
    shape,
    start,
    size,
    T: T.self)
}

// Applies softmax to a batched N-D `SparseTensor`.
//
// The inputs represent an N-D SparseTensor  with logical shape `[..., B, C]`
// (where `N >= 2`), and with indices sorted in the canonical lexicographic order.
//
// This op is equivalent to applying the normal `tf.nn.softmax()` to each innermost
// logical submatrix with shape `[B, C]`, but with the catch that *the implicitly
// zero elements do not participate*.  Specifically, the algorithm is equivalent
// to the following:
//
//   (1) Applies `tf.nn.softmax()` to a densified view of each innermost submatrix
//       with shape `[B, C]`, along the size-C dimension;
//   (2) Masks out the original implicitly-zero locations;
//   (3) Renormalizes the remaining elements.
//
// Hence, the `SparseTensor` result has exactly the same non-zero indices and
// shape.
//
// - Parameters:
//   - sp_indices: 2-D.  `NNZ x R` matrix with the indices of non-empty values in a
//     SparseTensor, in canonical ordering.
//   - sp_values: 1-D.  `NNZ` non-empty values corresponding to `sp_indices`.
//   - sp_shape: 1-D.  Shape of the input SparseTensor.
//
// - Output output: 1-D.  The `NNZ` values for the result `SparseTensor`.
@_inlineable @inline(__always)
public static func sparseSoftmax<T: BinaryFloatingPoint>(
  spIndices: Tensor<Int64>,
  spValues: Tensor<T>,
  spShape: Tensor<Int64>
) -> Tensor<T> {
  return #tfop("SparseSoftmax",
    spIndices,
    spValues,
    spShape,
    T: T.self)
}

// Computes softmax cross entropy cost and gradients to backpropagate.
//
// Unlike `SoftmaxCrossEntropyWithLogits`, this operation does not accept
// a matrix of label probabilities, but rather a single label per row
// of features.  This label is considered to have probability 1.0 for the
// given row.
//
// Inputs are the logits, not probabilities.
//
// - Parameters:
//   - features: batch_size x num_classes matrix
//   - labels: batch_size vector with values in [0, num_classes).
//     This is the label for the given minibatch entry.
//
// - Outputs:
//   - loss: Per example loss (batch_size vector).
//   - backprop: backpropagated gradients (batch_size x num_classes matrix).
@_inlineable @inline(__always)
public static func sparseSoftmaxCrossEntropyWithLogits<T: BinaryFloatingPoint, Tlabels: BinaryInteger>(
  features: Tensor<T>,
  labels: Tensor<Tlabels>
) -> (Tensor<T>, Tensor<T>) {
  return #tfop("SparseSoftmaxCrossEntropyWithLogits",
    features,
    labels,
    T: T.self,
    Tlabels: Tlabels.self)
}

// Returns the element-wise max of two SparseTensors.
//
// Assumes the two SparseTensors have the same shape, i.e., no broadcasting.
//
// - Parameters:
//   - a_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
//     SparseTensor, in the canonical lexicographic ordering.
//   - a_values: 1-D.  `N` non-empty values corresponding to `a_indices`.
//   - a_shape: 1-D.  Shape of the input SparseTensor.
//   - b_indices: counterpart to `a_indices` for the other operand.
//   - b_values: counterpart to `a_values` for the other operand; must be of the same dtype.
//   - b_shape: counterpart to `a_shape` for the other operand; the two shapes must be equal.
//
// - Outputs:
//   - output_indices: 2-D.  The indices of the output SparseTensor.
//   - output_values: 1-D.  The values of the output SparseTensor.
@_inlineable @inline(__always)
public static func sparseSparseMaximum<T: Numeric>(
  aIndices: Tensor<Int64>,
  aValues: Tensor<T>,
  aShape: Tensor<Int64>,
  bIndices: Tensor<Int64>,
  bValues: Tensor<T>,
  bShape: Tensor<Int64>
) -> (Tensor<Int64>, Tensor<T>) {
  return #tfop("SparseSparseMaximum",
    aIndices,
    aValues,
    aShape,
    bIndices,
    bValues,
    bShape,
    T: T.self)
}

// Returns the element-wise min of two SparseTensors.
//
// Assumes the two SparseTensors have the same shape, i.e., no broadcasting.
//
// - Parameters:
//   - a_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
//     SparseTensor, in the canonical lexicographic ordering.
//   - a_values: 1-D.  `N` non-empty values corresponding to `a_indices`.
//   - a_shape: 1-D.  Shape of the input SparseTensor.
//   - b_indices: counterpart to `a_indices` for the other operand.
//   - b_values: counterpart to `a_values` for the other operand; must be of the same dtype.
//   - b_shape: counterpart to `a_shape` for the other operand; the two shapes must be equal.
//
// - Outputs:
//   - output_indices: 2-D.  The indices of the output SparseTensor.
//   - output_values: 1-D.  The values of the output SparseTensor.
@_inlineable @inline(__always)
public static func sparseSparseMinimum<T: Numeric>(
  aIndices: Tensor<Int64>,
  aValues: Tensor<T>,
  aShape: Tensor<Int64>,
  bIndices: Tensor<Int64>,
  bValues: Tensor<T>,
  bShape: Tensor<Int64>
) -> (Tensor<Int64>, Tensor<T>) {
  return #tfop("SparseSparseMinimum",
    aIndices,
    aValues,
    aShape,
    bIndices,
    bValues,
    bShape,
    T: T.self)
}

// Split a `SparseTensor` into `num_split` tensors along one dimension.
//
// If the `shape[split_dim]` is not an integer multiple of `num_split`. Slices
// `[0 : shape[split_dim] % num_split]` gets one extra dimension.
// For example, if `split_dim = 1` and `num_split = 2` and the input is
//
//     input_tensor = shape = [2, 7]
//     [    a   d e  ]
//     [b c          ]
//
// Graphically the output tensors are:
//
//     output_tensor[0] = shape = [2, 4]
//     [    a  ]
//     [b c    ]
//
//     output_tensor[1] = shape = [2, 3]
//     [ d e  ]
//     [      ]
//
// - Parameters:
//   - split_dim: 0-D.  The dimension along which to split.  Must be in the range
//     `[0, rank(shape))`.
//   - indices: 2-D tensor represents the indices of the sparse tensor.
//   - values: 1-D tensor represents the values of the sparse tensor.
//   - shape: 1-D. tensor represents the shape of the sparse tensor.
//     output indices: A list of 1-D tensors represents the indices of the output
//     sparse tensors.
//
// - Attr num_split: The number of ways to split.
//
// - Outputs:
//   - output_values: A list of 1-D tensors represents the values of the output sparse
//     tensors.
//   - output_shape: A list of 1-D tensors represents the shape of the output sparse
//     tensors.
@_inlineable @inline(__always)
public static func sparseSplit<T: Numeric>(
  splitDim: Tensor<Int64>,
  indices: Tensor<Int64>,
  values: Tensor<T>,
  shape: Tensor<Int64>,
  numSplit: Int
) -> ([Tensor<Int64>], [Tensor<T>], [Tensor<Int64>]) {
  return #tfop("SparseSplit",
    splitDim,
    indices,
    values,
    shape,
    T: T.self,
    num_split: numSplit)
}

// Adds up a `SparseTensor` and a dense `Tensor`, producing a dense `Tensor`.
//
// This Op does not require `a_indices` be sorted in standard lexicographic order.
//
// - Parameters:
//   - a_indices: 2-D.  The `indices` of the `SparseTensor`, with shape `[nnz, ndims]`.
//   - a_values: 1-D.  The `values` of the `SparseTensor`, with shape `[nnz]`.
//   - a_shape: 1-D.  The `shape` of the `SparseTensor`, with shape `[ndims]`.
//   - b: `ndims`-D Tensor.  With shape `a_shape`.
@_inlineable @inline(__always)
public static func sparseTensorDenseAdd<T: Numeric, Tindices: BinaryInteger>(
  aIndices: Tensor<Tindices>,
  aValues: Tensor<T>,
  aShape: Tensor<Tindices>,
  b: Tensor<T>
) -> Tensor<T> {
  return #tfop("SparseTensorDenseAdd",
    aIndices,
    aValues,
    aShape,
    b,
    T: T.self,
    Tindices: Tindices.self)
}

// Multiply SparseTensor (of rank 2) "A" by dense matrix "B".
//
// No validity checking is performed on the indices of A.  However, the following
// input format is recommended for optimal behavior:
//
// if adjoint_a == false:
//   A should be sorted in lexicographically increasing order.  Use SparseReorder
//   if you're not sure.
// if adjoint_a == true:
//   A should be sorted in order of increasing dimension 1 (i.e., "column major"
//   order instead of "row major" order).
//
// - Parameters:
//   - a_indices: 2-D.  The `indices` of the `SparseTensor`, size `[nnz, 2]` Matrix.
//   - a_values: 1-D.  The `values` of the `SparseTensor`, size `[nnz]` Vector.
//   - a_shape: 1-D.  The `shape` of the `SparseTensor`, size `[2]` Vector.
//   - b: 2-D.  A dense Matrix.
//
// - Attrs:
//   - adjoint_a: Use the adjoint of A in the matrix multiply.  If A is complex, this
//     is transpose(conj(A)).  Otherwise it's transpose(A).
//   - adjoint_b: Use the adjoint of B in the matrix multiply.  If B is complex, this
//     is transpose(conj(B)).  Otherwise it's transpose(B).
@_inlineable @inline(__always)
public static func sparseTensorDenseMatMul<T: Numeric, Tindices: BinaryInteger>(
  aIndices: Tensor<Tindices>,
  aValues: Tensor<T>,
  aShape: Tensor<Int64>,
  b: Tensor<T>,
  adjointA: Bool = false,
  adjointB: Bool = false
) -> Tensor<T> {
  return #tfop("SparseTensorDenseMatMul",
    aIndices,
    aValues,
    aShape,
    b,
    T: T.self,
    Tindices: Tindices.self,
    adjoint_a: adjointA,
    adjoint_b: adjointB)
}

// Converts a sparse representation into a dense tensor.
//
// Builds an array `dense` with shape `output_shape` such that
//
// ```
// # If sparse_indices is scalar
// dense[i] = (i == sparse_indices ? sparse_values : default_value)
//
// # If sparse_indices is a vector, then for each i
// dense[sparse_indices[i]] = sparse_values[i]
//
// # If sparse_indices is an n by d matrix, then for each i in [0, n)
// dense[sparse_indices[i][0], ..., sparse_indices[i][d-1]] = sparse_values[i]
// ```
//
// All other values in `dense` are set to `default_value`.  If `sparse_values` is a
// scalar, all sparse indices are set to this single value.
//
// Indices should be sorted in lexicographic order, and indices must not
// contain any repeats. If `validate_indices` is true, these properties
// are checked during execution.
//
// - Parameters:
//   - sparse_indices: 0-D, 1-D, or 2-D.  `sparse_indices[i]` contains the complete
//     index where `sparse_values[i]` will be placed.
//   - output_shape: 1-D.  Shape of the dense output tensor.
//   - sparse_values: 1-D.  Values corresponding to each row of `sparse_indices`,
//     or a scalar value to be used for all sparse indices.
//   - default_value: Scalar value to set for indices not specified in
//     `sparse_indices`.
//
// - Attr validate_indices: If true, indices are checked to make sure they are sorted in
//   lexicographic order and that there are no repeats.
//
// - Output dense: Dense output tensor of shape `output_shape`.
@_inlineable @inline(__always)
public static func sparseToDense<T: Numeric, Tindices: BinaryInteger>(
  sparseIndices: Tensor<Tindices>,
  outputShape: Tensor<Tindices>,
  sparseValues: Tensor<T>,
  defaultValue: Tensor<T>,
  validateIndices: Bool = true
) -> Tensor<T> {
  return #tfop("SparseToDense",
    sparseIndices,
    outputShape,
    sparseValues,
    defaultValue,
    T: T.self,
    Tindices: Tindices.self,
    validate_indices: validateIndices)
}

// Applies set operation along last dimension of 2 `SparseTensor` inputs.
//
// See SetOperationOp::SetOperationFromContext for values of `set_operation`.
//
// If `validate_indices` is `True`, `SparseToSparseSetOperation` validates the
// order and range of `set1` and `set2` indices.
//
// Input `set1` is a `SparseTensor` represented by `set1_indices`, `set1_values`,
// and `set1_shape`. For `set1` ranked `n`, 1st `n-1` dimensions must be the same
// as `set2`. Dimension `n` contains values in a set, duplicates are allowed but
// ignored.
//
// Input `set2` is a `SparseTensor` represented by `set2_indices`, `set2_values`,
// and `set2_shape`. For `set2` ranked `n`, 1st `n-1` dimensions must be the same
// as `set1`. Dimension `n` contains values in a set, duplicates are allowed but
// ignored.
//
// If `validate_indices` is `True`, this op validates the order and range of `set1`
// and `set2` indices.
//
// Output `result` is a `SparseTensor` represented by `result_indices`,
// `result_values`, and `result_shape`. For `set1` and `set2` ranked `n`, this
// has rank `n` and the same 1st `n-1` dimensions as `set1` and `set2`. The `nth`
// dimension contains the result of `set_operation` applied to the corresponding
// `[0...n-1]` dimension of `set`.
//
// - Parameters:
//   - set1_indices: 2D `Tensor`, indices of a `SparseTensor`. Must be in row-major
//     order.
//   - set1_values: 1D `Tensor`, values of a `SparseTensor`. Must be in row-major
//     order.
//   - set1_shape: 1D `Tensor`, shape of a `SparseTensor`. `set1_shape[0...n-1]` must
//     be the same as `set2_shape[0...n-1]`, `set1_shape[n]` is the
//     max set size across `0...n-1` dimensions.
//   - set2_indices: 2D `Tensor`, indices of a `SparseTensor`. Must be in row-major
//     order.
//   - set2_values: 1D `Tensor`, values of a `SparseTensor`. Must be in row-major
//     order.
//   - set2_shape: 1D `Tensor`, shape of a `SparseTensor`. `set2_shape[0...n-1]` must
//     be the same as `set1_shape[0...n-1]`, `set2_shape[n]` is the
//     max set size across `0...n-1` dimensions.
//
// - Outputs:
//   - result_indices: 2D indices of a `SparseTensor`.
//   - result_values: 1D values of a `SparseTensor`.
//   - result_shape: 1D `Tensor` shape of a `SparseTensor`. `result_shape[0...n-1]` is
//     the same as the 1st `n-1` dimensions of `set1` and `set2`, `result_shape[n]`
//     is the max result set size across all `0...n-1` dimensions.
@_inlineable @inline(__always)
public static func sparseToSparseSetOperation<T: BinaryInteger>(
  set1Indices: Tensor<Int64>,
  set1Values: Tensor<T>,
  set1Shape: Tensor<Int64>,
  set2Indices: Tensor<Int64>,
  set2Values: Tensor<T>,
  set2Shape: Tensor<Int64>,
  setOperation: String,
  validateIndices: Bool = true
) -> (Tensor<Int64>, Tensor<T>, Tensor<Int64>) {
  return #tfop("SparseToSparseSetOperation",
    set1Indices,
    set1Values,
    set1Shape,
    set2Indices,
    set2Values,
    set2Shape,
    T: T.self,
    set_operation: setOperation,
    validate_indices: validateIndices)
}

// Splits a tensor into `num_split` tensors along one dimension.
//
// - Parameters:
//   - split_dim: 0-D.  The dimension along which to split.  Must be in the range
//     `[-rank(value), rank(value))`.
//   - value: The tensor to split.
//
// - Attr num_split: The number of ways to split.  Must evenly divide
//   `value.shape[split_dim]`.
//
// - Output output: They are identically shaped tensors, whose shape matches that of `value`
//   except along `axis`, where their sizes are
//   `values.shape[split_dim] / num_split`.
@_inlineable @inline(__always)
public static func split<T: Numeric>(
  splitDim: Tensor<Int32>,
  value: Tensor<T>,
  numSplit: Int
) -> [Tensor<T>] {
  return #tfop("Split",
    splitDim,
    value,
    T: T.self,
    num_split: numSplit)
}

// Splits a tensor into `num_split` tensors along one dimension.
//
// - Parameters:
//   - value: The tensor to split.
//   - size_splits: list containing the sizes of each output tensor along the split
//     dimension. Must sum to the dimension of value along split_dim.
//     Can contain one -1 indicating that dimension is to be inferred.
//   - split_dim: 0-D.  The dimension along which to split.  Must be in the range
//     `[-rank(value), rank(value))`.
//
// - Output output: Tensors whose shape matches that of `value`
//   except along `axis`, where their sizes are
//   `size_splits[i]`.
@_inlineable @inline(__always)
public static func splitV<T: Numeric, Tlen: BinaryInteger>(
  value: Tensor<T>,
  sizeSplits: Tensor<Tlen>,
  splitDim: Tensor<Int32>,
  numSplit: Int
) -> [Tensor<T>] {
  return #tfop("SplitV",
    value,
    sizeSplits,
    splitDim,
    T: T.self,
    Tlen: Tlen.self,
    num_split: numSplit)
}

// Computes square root of x element-wise.
//
// I.e., \\(y = \sqrt{x} = x^{1/2}\\).
@_inlineable @inline(__always)
public static func sqrt<T: BinaryFloatingPoint>(
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Sqrt",
    x,
    T: T.self)
}

// Computes the gradient for the sqrt of `x` wrt its input.
//
// Specifically, `grad = dy * 0.5 / y`, where `y = sqrt(x)`, and `dy`
// is the corresponding input gradient.
@_inlineable @inline(__always)
public static func sqrtGrad<T: BinaryFloatingPoint>(
  y: Tensor<T>,
  dy: Tensor<T>
) -> Tensor<T> {
  return #tfop("SqrtGrad",
    y,
    dy,
    T: T.self)
}

// Computes square of x element-wise.
//
// I.e., \\(y = x * x = x^2\\).
@_inlineable @inline(__always)
public static func square<T: Numeric>(
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Square",
    x,
    T: T.self)
}

// Returns (x - y)(x - y) element-wise.
//
// *NOTE*: `SquaredDifference` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@_inlineable @inline(__always)
public static func squaredDifference<T: Numeric>(
  x: Tensor<T>,
  y: Tensor<T>
) -> Tensor<T> {
  return #tfop("SquaredDifference",
    x,
    y,
    T: T.self)
}

// Removes dimensions of size 1 from the shape of a tensor.
//
// Given a tensor `input`, this operation returns a tensor of the same type with
// all dimensions of size 1 removed. If you don't want to remove all size 1
// dimensions, you can remove specific size 1 dimensions by specifying
// `axis`.
//
// For example:
//
// ```
// # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
// shape(squeeze(t)) ==> [2, 3]
// ```
//
// Or, to remove specific size 1 dimensions:
//
// ```
// # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
// shape(squeeze(t, [2, 4])) ==> [1, 2, 3, 1]
// ```
//
// - Parameter input: The `input` to squeeze.
//
// - Attr squeeze_dims: If specified, only squeezes the dimensions listed. The dimension
//   index starts at 0. It is an error to squeeze a dimension that is not 1. Must
//   be in the range `[-rank(input), rank(input))`.
//
// - Output output: Contains the same data as `input`, but has one or more dimensions of
//   size 1 removed.
@_inlineable @inline(__always)
public static func squeeze<T: Numeric>(
  input: Tensor<T>,
  squeezeDims: [Int]
) -> Tensor<T> {
  return #tfop("Squeeze",
    input,
    T: T.self,
    squeeze_dims: squeezeDims)
}

// Stage values similar to a lightweight Enqueue.
//
// The basic functionality of this Op is similar to a queue with many
// fewer capabilities and options.  This Op is optimized for performance.
//
// - Parameter values: a list of tensors
//   dtypes A list of data types that inserted values should adhere to.
//
// - Attrs:
//   - capacity: Maximum number of elements in the Staging Area. If > 0, inserts
//     on the container will block when the capacity is reached.
//   - memory_limit: The maximum number of bytes allowed for Tensors in the Staging Area.
//     If > 0, inserts will block until sufficient space is available.
//   - container: If non-empty, this queue is placed in the given container. Otherwise,
//     a default container is used.
//   - shared_name: It is necessary to match this name to the matching Unstage Op.
@_inlineable @inline(__always)
public static func stage<Dtypes: Numeric>(
  values: [Tensor<Dtypes>],
  capacity: Int = 0,
  memoryLimit: Int = 0,
  container: String,
  sharedName: String
) {
  return #tfop("Stage",
    values,
    capacity: capacity,
    memory_limit: memoryLimit,
    container: container,
    shared_name: sharedName)
}

// Op removes all elements in the underlying container.
@_inlineable @inline(__always)
public static func stageClear<Dtypes: Numeric>(
  capacity: Int = 0,
  memoryLimit: Int = 0,
  container: String,
  sharedName: String,
  typeDtypes: Dtypes.Type
) {
  return #tfop("StageClear",
    capacity: capacity,
    memory_limit: memoryLimit,
    container: container,
    shared_name: sharedName)
}

// Op peeks at the values at the specified index.  If the
//
// underlying container does not contain sufficient elements
// this op will block until it does.   This Op is optimized for
// performance.
@_inlineable @inline(__always)
public static func stagePeek<Dtypes: Numeric>(
  index: Tensor<Int32>,
  capacity: Int = 0,
  memoryLimit: Int = 0,
  container: String,
  sharedName: String
) -> [Tensor<Dtypes>] {
  return #tfop("StagePeek",
    index,
    capacity: capacity,
    memory_limit: memoryLimit,
    container: container,
    shared_name: sharedName)
}

// Op returns the number of elements in the underlying container.
@_inlineable @inline(__always)
public static func stageSize<Dtypes: Numeric>(
  capacity: Int = 0,
  memoryLimit: Int = 0,
  container: String,
  sharedName: String,
  typeDtypes: Dtypes.Type
) -> Tensor<Int32> {
  return #tfop("StageSize",
    capacity: capacity,
    memory_limit: memoryLimit,
    container: container,
    shared_name: sharedName)
}

// Outputs deterministic pseudorandom values from a normal distribution.
//
// The generated values will have mean 0 and standard deviation 1.
//
// The outputs are a deterministic function of `shape` and `seed`.
//
// - Parameters:
//   - shape: The shape of the output tensor.
//   - seed: 2 seeds (shape [2]).
//
// - Attr dtype: The type of the output.
//
// - Output output: Random values with specified shape.
@_inlineable @inline(__always)
public static func statelessRandomNormal<Dtype: BinaryFloatingPoint, T: BinaryInteger, Tseed: BinaryInteger>(
  shape: Tensor<T>,
  seed: Tensor<Tseed>
) -> Tensor<Dtype> {
  return #tfop("StatelessRandomNormal",
    shape,
    seed,
    Dtype: Dtype.self,
    T: T.self,
    Tseed: Tseed.self)
}

// Outputs deterministic pseudorandom random values from a uniform distribution.
//
// The generated values follow a uniform distribution in the range `[0, 1)`. The
// lower bound 0 is included in the range, while the upper bound 1 is excluded.
//
// The outputs are a deterministic function of `shape` and `seed`.
//
// - Parameters:
//   - shape: The shape of the output tensor.
//   - seed: 2 seeds (shape [2]).
//
// - Attr dtype: The type of the output.
//
// - Output output: Random values with specified shape.
@_inlineable @inline(__always)
public static func statelessRandomUniform<Dtype: BinaryFloatingPoint, T: BinaryInteger, Tseed: BinaryInteger>(
  shape: Tensor<T>,
  seed: Tensor<Tseed>
) -> Tensor<Dtype> {
  return #tfop("StatelessRandomUniform",
    shape,
    seed,
    Dtype: Dtype.self,
    T: T.self,
    Tseed: Tseed.self)
}

// Outputs deterministic pseudorandom values from a truncated normal distribution.
//
// The generated values follow a normal distribution with mean 0 and standard
// deviation 1, except that values whose magnitude is more than 2 standard
// deviations from the mean are dropped and re-picked.
//
// The outputs are a deterministic function of `shape` and `seed`.
//
// - Parameters:
//   - shape: The shape of the output tensor.
//   - seed: 2 seeds (shape [2]).
//
// - Attr dtype: The type of the output.
//
// - Output output: Random values with specified shape.
@_inlineable @inline(__always)
public static func statelessTruncatedNormal<Dtype: BinaryFloatingPoint, T: BinaryInteger, Tseed: BinaryInteger>(
  shape: Tensor<T>,
  seed: Tensor<Tseed>
) -> Tensor<Dtype> {
  return #tfop("StatelessTruncatedNormal",
    shape,
    seed,
    Dtype: Dtype.self,
    T: T.self,
    Tseed: Tseed.self)
}

// Stops gradient computation.
//
// When executed in a graph, this op outputs its input tensor as-is.
//
// When building ops to compute gradients, this op prevents the contribution of
// its inputs to be taken into account.  Normally, the gradient generator adds ops
// to a graph to compute the derivatives of a specified 'loss' by recursively
// finding out inputs that contributed to its computation.  If you insert this op
// in the graph it inputs are masked from the gradient generator.  They are not
// taken into account for computing gradients.
//
// This is useful any time you want to compute a value with TensorFlow but need
// to pretend that the value was a constant. Some examples include:
//
// *  The *EM* algorithm where the *M-step* should not involve backpropagation
//    through the output of the *E-step*.
// *  Contrastive divergence training of Boltzmann machines where, when
//    differentiating the energy function, the training must not backpropagate
//    through the graph that generated the samples from the model.
// *  Adversarial training, where no backprop should happen through the adversarial
//    example generation process.
@_inlineable @inline(__always)
public static func stopGradient<T: Numeric>(
  input: Tensor<T>
) -> Tensor<T> {
  return #tfop("StopGradient",
    input,
    T: T.self)
}

// Return a strided slice from `input`.
//
// Note, most python users will want to use the Python `Tensor.__getitem__`
// or `Variable.__getitem__` rather than this op directly.
//
// The goal of this op is to produce a new tensor with a subset of
// the elements from the `n` dimensional `input` tensor. The subset is chosen using
// a sequence of `m` sparse range specifications encoded into the arguments
// of this function. Note, in some cases
// `m` could be equal to `n`, but this need not be the case. Each
// range specification entry can be one of the following:
//
// - An ellipsis (...). Ellipses are used to imply zero or more
//   dimensions of full-dimension selection and are produced using
//   `ellipsis_mask`. For example, `foo[...]` is the identity slice.
//
// - A new axis. This is used to insert a new shape=1 dimension and is
//   produced using `new_axis_mask`. For example, `foo[:, ...]` where
//   `foo` is shape `(3, 4)` produces a `(1, 3, 4)` tensor.
//
//
// - A range `begin:end:stride`. This is used to specify how much to choose from
//   a given dimension. `stride` can be any integer but 0.  `begin` is an integer
//   which represents the index of the first value to select while `end` represents
//   the index of the last value to select. The number of values selected in each
//   dimension is `end - begin` if `stride > 0` and `begin - end` if `stride < 0`.
//   `begin` and `end` can be negative where `-1` is the last element, `-2` is
//   the second to last. `begin_mask` controls whether to replace the explicitly
//   given `begin` with an implicit effective value of `0` if `stride > 0` and
//   `-1` if `stride < 0`. `end_mask` is analogous but produces the number
//   required to create the largest open interval. For example, given a shape
//   `(3,)` tensor `foo[:]`, the effective `begin` and `end` are `0` and `3`. Do
//   not assume this is equivalent to `foo[0:-1]` which has an effective `begin`
//   and `end` of `0` and `2`. Another example is `foo[-2::-1]` which reverses the
//   first dimension of a tensor while dropping the last two (in the original
//   order elements). For example `foo = [1,2,3,4]; foo[-2::-1]` is `[4,3]`.
//
// - A single index. This is used to keep only elements that have a given
//   index. For example (`foo[2, :]` on a shape `(5,6)` tensor produces a
//   shape `(6,)` tensor. This is encoded in `begin` and `end` and
//   `shrink_axis_mask`.
//
// Each conceptual range specification is encoded in the op's argument. This
// encoding is best understand by considering a non-trivial example. In
// particular,
// `foo[1, 2:4, None, ..., :-3:-1, :]` will be encoded as
//
// ```
// begin = [1, 2, x, x, 0, x] # x denotes don't care (usually 0)
// end = [2, 4, x, x, -3, x]
// strides = [1, 1, x, x, -1, 1]
// begin_mask = 1<<4 | 1 << 5 = 48
// end_mask = 1<<5 = 32
// ellipsis_mask = 1<<3 = 8
// new_axis_mask = 1<<2 4
// shrink_axis_mask = 1<<0
// ```
//
// In this case if `foo.shape` is (5, 5, 5, 5, 5, 5) the final shape of
// the slice becomes (2, 1, 5, 5, 2, 5).
// Let us walk step by step through each argument specification.
//
// 1.  The first argument in the example slice is turned into `begin = 1` and
// `end = begin + 1 = 2`. To disambiguate from the original spec `2:4` we
// also set the appropriate bit in `shrink_axis_mask`.
//
// 2. `2:4` is contributes 2, 4, 1 to begin, end, and stride. All masks have
// zero bits contributed.
//
// 3. None is a synonym for `tf.newaxis`. This means insert a dimension of size 1
// dimension in the final shape. Dummy values are contributed to begin,
// end and stride, while the new_axis_mask bit is set.
//
// 4. `...` grab the full ranges from as many dimensions as needed to
// fully specify a slice for every dimension of the input shape.
//
// 5. `:-3:-1` shows the use of negative indices. A negative index `i` associated
// with a dimension that has shape `s` is converted to a positive index
// `s + i`. So `-1` becomes `s-1` (i.e. the last element). This conversion
// is done internally so begin, end and strides receive x, -3, and -1.
// The appropriate begin_mask bit is set to indicate the start range is the
// full range (ignoring the x).
//
// 6. `:` indicates that the entire contents of the corresponding dimension
// is selected. This is equivalent to `::` or `0::1`. begin, end, and strides
// receive 0, 0, and 1, respectively. The appropriate bits in `begin_mask` and
// `end_mask` are also set.
//
// *Requirements*:
//   `0 != strides[i] for i in [0, m)`
//   `ellipsis_mask must be a power of two (only one ellipsis)`
//
// - Parameters:
//   - begin: `begin[k]` specifies the offset into the `k`th range specification.
//     The exact dimension this corresponds to will be determined by context.
//     Out-of-bounds values will be silently clamped. If the `k`th bit of
//     `begin_mask` then `begin[k]` is ignored and the full range of the
//     appropriate dimension is used instead. Negative values causes indexing
//     to start from the highest element e.g. If `foo==[1,2,3]` then `foo[-1]==3`.
//   - end: `end[i]` is like `begin` with the exception that `end_mask` is
//     used to determine full ranges.
//   - strides: `strides[i]` specifies the increment in the `i`th specification
//     after extracting a given element. Negative indices will reverse
//     the original order. Out or range values are
//     clamped to `[0,dim[i]) if slice[i]>0` or `[-1,dim[i]-1] if slice[i] < 0`
//
// - Attrs:
//   - begin_mask: a bitmask where a bit i being 1 means to ignore the begin
//     value and instead use the largest interval possible. At runtime
//     begin[i] will be replaced with `[0, n-1) if `stride[i] > 0` or
//     `[-1, n-1]` if `stride[i] < 0`
//   - end_mask: analogous to `begin_mask`
//   - ellipsis_mask: a bitmask where bit `i` being 1 means the `i`th
//     position is actually an ellipsis. One bit at most can be 1.
//     If `ellipsis_mask == 0`, then an implicit ellipsis mask of `1 << (m+1)`
//     is provided. This means that `foo[3:5] == foo[3:5, ...]`. An ellipsis
//     implicitly creates as many range specifications as necessary to fully
//     specify the sliced range for every dimension. For example for a 4-dimensional
//     tensor `foo` the slice `foo[2, ..., 5:8]` implies `foo[2, :, :, 5:8]`.
//   - new_axis_mask: a bitmask where bit `i` being 1 means the `i`th
//     specification creates a new shape 1 dimension. For example
//     `foo[:4, tf.newaxis, :2]` would produce a shape `(4, 1, 2)` tensor.
//   - shrink_axis_mask: a bitmask where bit `i` implies that the `i`th
//     specification should shrink the dimensionality. begin and end
//     must imply a slice of size 1 in the dimension. For example in
//     python one might do `foo[:, 3, :]` which would result in
//     `shrink_axis_mask` being 2.
@_inlineable @inline(__always)
public static func stridedSlice<T: Numeric, Index: BinaryInteger>(
  input: Tensor<T>,
  begin: Tensor<Index>,
  end: Tensor<Index>,
  strides: Tensor<Index>,
  beginMask: Int = 0,
  endMask: Int = 0,
  ellipsisMask: Int = 0,
  newAxisMask: Int = 0,
  shrinkAxisMask: Int = 0
) -> Tensor<T> {
  return #tfop("StridedSlice",
    input,
    begin,
    end,
    strides,
    T: T.self,
    Index: Index.self,
    begin_mask: beginMask,
    end_mask: endMask,
    ellipsis_mask: ellipsisMask,
    new_axis_mask: newAxisMask,
    shrink_axis_mask: shrinkAxisMask)
}

// Assign `value` to the sliced l-value reference of `ref`.
//
// The values of `value` are assigned to the positions in the variable
// `ref` that are selected by the slice parameters. The slice parameters
// `begin, `end`, `strides`, etc. work exactly as in `StridedSlice`.
//
// NOTE this op currently does not support broadcasting and so `value`'s
// shape must be exactly the shape produced by the slice of `ref`.
@_inlineable @inline(__always)
public static func stridedSliceAssign<T: Numeric, Index: BinaryInteger>(
  ref: Tensor<T>,
  begin: Tensor<Index>,
  end: Tensor<Index>,
  strides: Tensor<Index>,
  value: Tensor<T>,
  beginMask: Int = 0,
  endMask: Int = 0,
  ellipsisMask: Int = 0,
  newAxisMask: Int = 0,
  shrinkAxisMask: Int = 0
) -> Tensor<T> {
  return #tfop("StridedSliceAssign",
    ref,
    begin,
    end,
    strides,
    value,
    T: T.self,
    Index: Index.self,
    begin_mask: beginMask,
    end_mask: endMask,
    ellipsis_mask: ellipsisMask,
    new_axis_mask: newAxisMask,
    shrink_axis_mask: shrinkAxisMask)
}

// Returns the gradient of `StridedSlice`.
//
// Since `StridedSlice` cuts out pieces of its `input` which is size
// `shape`, its gradient will have the same shape (which is passed here
// as `shape`). The gradient will be zero in any element that the slice
// does not select.
//
// Arguments are the same as StridedSliceGrad with the exception that
// `dy` is the input gradient to be propagated and `shape` is the
// shape of `StridedSlice`'s `input`.
@_inlineable @inline(__always)
public static func stridedSliceGrad<T: Numeric, Index: BinaryInteger>(
  shape: Tensor<Index>,
  begin: Tensor<Index>,
  end: Tensor<Index>,
  strides: Tensor<Index>,
  dy: Tensor<T>,
  beginMask: Int = 0,
  endMask: Int = 0,
  ellipsisMask: Int = 0,
  newAxisMask: Int = 0,
  shrinkAxisMask: Int = 0
) -> Tensor<T> {
  return #tfop("StridedSliceGrad",
    shape,
    begin,
    end,
    strides,
    dy,
    T: T.self,
    Index: Index.self,
    begin_mask: beginMask,
    end_mask: endMask,
    ellipsis_mask: ellipsisMask,
    new_axis_mask: newAxisMask,
    shrink_axis_mask: shrinkAxisMask)
}

@_inlineable @inline(__always)
public static func stringListAttr(
  a: [String],
  b: String
) {
  return #tfop("StringListAttr",
    a: a,
    b: b)
}

// Returns x - y element-wise.
//
// *NOTE*: `Subtract` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@_inlineable @inline(__always)
public static func sub<T: Numeric>(
  x: Tensor<T>,
  y: Tensor<T>
) -> Tensor<T> {
  return #tfop("Sub",
    x,
    y,
    T: T.self)
}

// Computes the sum of elements across dimensions of a tensor.
//
// Reduces `input` along the dimensions given in `axis`. Unless
// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
// `axis`. If `keep_dims` is true, the reduced dimensions are
// retained with length 1.
//
// - Parameters:
//   - input: The tensor to reduce.
//   - reduction_indices: The dimensions to reduce. Must be in the range
//     `[-rank(input), rank(input))`.
//
// - Attr keep_dims: If true, retain reduced dimensions with length 1.
//
// - Output output: The reduced tensor.
@_inlineable @inline(__always)
public static func sum<T: Numeric, Tidx: BinaryInteger>(
  input: Tensor<T>,
  reductionIndices: Tensor<Tidx>,
  keepDims: Bool = false
) -> Tensor<T> {
  return #tfop("Sum",
    input,
    reductionIndices,
    T: T.self,
    Tidx: Tidx.self,
    keep_dims: keepDims)
}

// Computes the singular value decompositions of one or more matrices.
//
// Computes the SVD of each inner matrix in `input` such that
// `input[..., :, :] = u[..., :, :] * diag(s[..., :, :]) * transpose(v[..., :, :])`
//
// ```python
// # a is a tensor containing a batch of matrices.
// # s is a tensor of singular values for each matrix.
// # u is the tensor containing of left singular vectors for each matrix.
// # v is the tensor containing of right singular vectors for each matrix.
// s, u, v = svd(a)
// s, _, _ = svd(a, compute_uv=False)
// ```
//
// - Parameter input: A tensor of shape `[..., M, N]` whose inner-most 2 dimensions
//   form matrices of size `[M, N]`. Let `P` be the minimum of `M` and `N`.
//
// - Attrs:
//   - compute_uv: If true, left and right singular vectors will be
//     computed and returned in `u` and `v`, respectively.
//     If false, `u` and `v` are not set and should never referenced.
//   - full_matrices: If true, compute full-sized `u` and `v`. If false
//     (the default), compute only the leading `P` singular vectors.
//     Ignored if `compute_uv` is `False`.
//
// - Outputs:
//   - s: Singular values. Shape is `[..., P]`.
//   - u: Left singular vectors. If `full_matrices` is `False` then shape is
//     `[..., M, P]`; if `full_matrices` is `True` then shape is
//     `[..., M, M]`. Undefined if `compute_uv` is `False`.
//   - v: Left singular vectors. If `full_matrices` is `False` then shape is
//     `[..., N, P]`. If `full_matrices` is `True` then shape is `[..., N, N]`.
//     Undefined if `compute_uv` is false.
@_inlineable @inline(__always)
public static func svd<T: BinaryFloatingPoint>(
  input: Tensor<T>,
  computeUv: Bool = true,
  fullMatrices: Bool = false
) -> (Tensor<T>, Tensor<T>, Tensor<T>) {
  return #tfop("Svd",
    input,
    T: T.self,
    compute_uv: computeUv,
    full_matrices: fullMatrices)
}

// Forwards `data` to the output port determined by `pred`.
//
// If `pred` is true, the `data` input is forwarded to `output_true`. Otherwise,
// the data goes to `output_false`.
//
// See also `RefSwitch` and `Merge`.
//
// - Parameters:
//   - data: The tensor to be forwarded to the appropriate output.
//   - pred: A scalar that specifies which output port will receive data.
//
// - Outputs:
//   - output_false: If `pred` is false, data will be forwarded to this output.
//   - output_true: If `pred` is true, data will be forwarded to this output.
@_inlineable @inline(__always)
public static func switch_<T: Numeric>(
  data: Tensor<T>,
  pred: Tensor<Bool>
) -> (Tensor<T>, Tensor<T>) {
  return #tfop("Switch",
    data,
    pred,
    T: T.self)
}

// Read `SparseTensors` from a `SparseTensorsMap` and concatenate them.
//
// The input `sparse_handles` must be an `int64` matrix of shape `[N, 1]` where
// `N` is the minibatch size and the rows correspond to the output handles of
// `AddSparseToTensorsMap` or `AddManySparseToTensorsMap`.  The ranks of the
// original `SparseTensor` objects that went into the given input ops must all
// match.  When the final `SparseTensor` is created, it has rank one
// higher than the ranks of the incoming `SparseTensor` objects
// (they have been concatenated along a new row dimension on the left).
//
// The output `SparseTensor` object's shape values for all dimensions but the
// first are the max across the input `SparseTensor` objects' shape values
// for the corresponding dimensions.  Its first shape value is `N`, the minibatch
// size.
//
// The input `SparseTensor` objects' indices are assumed ordered in
// standard lexicographic order.  If this is not the case, after this
// step run `SparseReorder` to restore index ordering.
//
// For example, if the handles represent an input, which is a `[2, 3]` matrix
// representing two original `SparseTensor` objects:
//
// ```
//     index = [ 0]
//             [10]
//             [20]
//     values = [1, 2, 3]
//     shape = [50]
// ```
//
// and
//
// ```
//     index = [ 2]
//             [10]
//     values = [4, 5]
//     shape = [30]
// ```
//
// then the final `SparseTensor` will be:
//
// ```
//     index = [0  0]
//             [0 10]
//             [0 20]
//             [1  2]
//             [1 10]
//     values = [1, 2, 3, 4, 5]
//     shape = [2 50]
// ```
//
// - Parameter sparse_handles: 1-D, The `N` serialized `SparseTensor` objects.
//   Shape: `[N]`.
//
// - Attrs:
//   - dtype: The `dtype` of the `SparseTensor` objects stored in the
//     `SparseTensorsMap`.
//   - container: The container name for the `SparseTensorsMap` read by this op.
//   - shared_name: The shared name for the `SparseTensorsMap` read by this op.
//     It should not be blank; rather the `shared_name` or unique Operation name
//     of the Op that created the original `SparseTensorsMap` should be used.
//
// - Outputs:
//   - sparse_indices: 2-D.  The `indices` of the minibatch `SparseTensor`.
//   - sparse_values: 1-D.  The `values` of the minibatch `SparseTensor`.
//   - sparse_shape: 1-D.  The `shape` of the minibatch `SparseTensor`.
@_inlineable @inline(__always)
public static func takeManySparseFromTensorsMap<Dtype: Numeric>(
  sparseHandles: Tensor<Int64>,
  container: String,
  sharedName: String
) -> (Tensor<Int64>, Tensor<Dtype>, Tensor<Int64>) {
  return #tfop("TakeManySparseFromTensorsMap",
    sparseHandles,
    Dtype: Dtype.self,
    container: container,
    shared_name: sharedName)
}

// Computes tan of x element-wise.
@_inlineable @inline(__always)
public static func tan<T: Numeric>(
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Tan",
    x,
    T: T.self)
}

// Computes hyperbolic tangent of `x` element-wise.
@_inlineable @inline(__always)
public static func tanh<T: BinaryFloatingPoint>(
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Tanh",
    x,
    T: T.self)
}

// Computes the gradient for the tanh of `x` wrt its input.
//
// Specifically, `grad = dy * (1 - y*y)`, where `y = tanh(x)`, and `dy`
// is the corresponding input gradient.
@_inlineable @inline(__always)
public static func tanhGrad<T: BinaryFloatingPoint>(
  y: Tensor<T>,
  dy: Tensor<T>
) -> Tensor<T> {
  return #tfop("TanhGrad",
    y,
    dy,
    T: T.self)
}

@_inlineable @inline(__always)
public static func testAttr<T: BinaryFloatingPoint>(
) -> Tensor<T> {
  return #tfop("TestAttr",
    T: T.self)
}

// Generates labels for candidate sampling with a learned unigram distribution.
//
// See explanations of candidate sampling and the data formats at
// go/candidate-sampling.
//
// For each batch, this op picks a single set of sampled candidate labels.
//
// The advantages of sampling candidates per-batch are simplicity and the
// possibility of efficient dense matrix multiplication. The disadvantage is that
// the sampled candidates must be chosen independently of the context and of the
// true labels.
//
// - Parameter true_classes: A batch_size * num_true matrix, in which each row contains the
//   IDs of the num_true target_classes in the corresponding original label.
//
// - Attrs:
//   - num_true: Number of true labels per context.
//   - num_sampled: Number of candidates to randomly sample.
//   - unique: If unique is true, we sample with rejection, so that all sampled
//     candidates in a batch are unique. This requires some approximation to
//     estimate the post-rejection sampling probabilities.
//   - range_max: The sampler will sample integers from the interval [0, range_max).
//   - seed: If either seed or seed2 are set to be non-zero, the random number
//     generator is seeded by the given seed.  Otherwise, it is seeded by a
//     random seed.
//   - seed2: An second seed to avoid seed collision.
//
// - Outputs:
//   - sampled_candidates: A vector of length num_sampled, in which each element is
//     the ID of a sampled candidate.
//   - true_expected_count: A batch_size * num_true matrix, representing
//     the number of times each candidate is expected to occur in a batch
//     of sampled candidates. If unique=true, then this is a probability.
//   - sampled_expected_count: A vector of length num_sampled, for each sampled
//     candidate representing the number of times the candidate is expected
//     to occur in a batch of sampled candidates.  If unique=true, then this is a
//     probability.
@_inlineable @inline(__always)
public static func threadUnsafeUnigramCandidateSampler(
  trueClasses: Tensor<Int64>,
  numTrue: Int,
  numSampled: Int,
  unique: Bool,
  rangeMax: Int,
  seed: Int = 0,
  seed2: Int = 0
) -> (Tensor<Int64>, Tensor<Float>, Tensor<Float>) {
  return #tfop("ThreadUnsafeUnigramCandidateSampler",
    trueClasses,
    num_true: numTrue,
    num_sampled: numSampled,
    unique: unique,
    range_max: rangeMax,
    seed: seed,
    seed2: seed2)
}

// Constructs a tensor by tiling a given tensor.
//
// This operation creates a new tensor by replicating `input` `multiples` times.
// The output tensor's i'th dimension has `input.dims(i) * multiples[i]` elements,
// and the values of `input` are replicated `multiples[i]` times along the 'i'th
// dimension. For example, tiling `[a b c d]` by `[2]` produces
// `[a b c d a b c d]`.
//
// - Parameters:
//   - input: 1-D or higher.
//   - multiples: 1-D. Length must be the same as the number of dimensions in `input`
@_inlineable @inline(__always)
public static func tile<T: Numeric, Tmultiples: BinaryInteger>(
  input: Tensor<T>,
  multiples: Tensor<Tmultiples>
) -> Tensor<T> {
  return #tfop("Tile",
    input,
    multiples,
    T: T.self,
    Tmultiples: Tmultiples.self)
}

// Returns the gradient of `Tile`.
//
// Since `Tile` takes an input and repeats the input `multiples` times
// along each dimension, `TileGrad` takes in `multiples` and aggregates
// each repeated tile of `input` into `output`.
@_inlineable @inline(__always)
public static func tileGrad<T: Numeric>(
  input: Tensor<T>,
  multiples: Tensor<Int32>
) -> Tensor<T> {
  return #tfop("TileGrad",
    input,
    multiples,
    T: T.self)
}

// Provides the time since epoch in seconds.
//
// Returns the timestamp as a `float64` for seconds since the Unix epoch.
//
// Note: the timestamp is computed when the op is executed, not when it is added
// to the graph.
@_inlineable @inline(__always)
public static func timestamp(
) -> Tensor<Double> {
  return #tfop("Timestamp")
}

// Finds values and indices of the `k` largest elements for the last dimension.
//
// If the input is a vector (rank-1), finds the `k` largest entries in the vector
// and outputs their values and indices as vectors.  Thus `values[j]` is the
// `j`-th largest entry in `input`, and its index is `indices[j]`.
//
// For matrices (resp. higher rank input), computes the top `k` entries in each
// row (resp. vector along the last dimension).  Thus,
//
//     values.shape = indices.shape = input.shape[:-1] + [k]
//
// If two elements are equal, the lower-index element appears first.
//
// If `k` varies dynamically, use `TopKV2` below.
//
// - Parameter input: 1-D or higher with last dimension at least `k`.
//
// - Attrs:
//   - k: Number of top elements to look for along the last dimension (along each
//     row for matrices).
//   - sorted: If true the resulting `k` elements will be sorted by the values in
//     descending order.
//
// - Outputs:
//   - values: The `k` largest elements along each last dimensional slice.
//   - indices: The indices of `values` within the last dimension of `input`.
@_inlineable @inline(__always)
public static func topK<T: Numeric>(
  input: Tensor<T>,
  k: Int,
  sorted: Bool = true
) -> (Tensor<T>, Tensor<Int32>) {
  return #tfop("TopK",
    input,
    T: T.self,
    k: k,
    sorted: sorted)
}

// Finds values and indices of the `k` largest elements for the last dimension.
//
// If the input is a vector (rank-1), finds the `k` largest entries in the vector
// and outputs their values and indices as vectors.  Thus `values[j]` is the
// `j`-th largest entry in `input`, and its index is `indices[j]`.
//
// For matrices (resp. higher rank input), computes the top `k` entries in each
// row (resp. vector along the last dimension).  Thus,
//
//     values.shape = indices.shape = input.shape[:-1] + [k]
//
// If two elements are equal, the lower-index element appears first.
//
// - Parameters:
//   - input: 1-D or higher with last dimension at least `k`.
//   - k: 0-D.  Number of top elements to look for along the last dimension (along each
//     row for matrices).
//
// - Attr sorted: If true the resulting `k` elements will be sorted by the values in
//   descending order.
//
// - Outputs:
//   - values: The `k` largest elements along each last dimensional slice.
//   - indices: The indices of `values` within the last dimension of `input`.
@_inlineable @inline(__always)
public static func topKV2<T: Numeric>(
  input: Tensor<T>,
  k: Tensor<Int32>,
  sorted: Bool = true
) -> (Tensor<T>, Tensor<Int32>) {
  return #tfop("TopKV2",
    input,
    k,
    T: T.self,
    sorted: sorted)
}

// Shuffle dimensions of x according to a permutation.
//
// The output `y` has the same rank as `x`. The shapes of `x` and `y` satisfy:
//   `y.shape[i] == x.shape[perm[i]] for i in [0, 1, ..., rank(x) - 1]`
@_inlineable @inline(__always)
public static func transpose<T: Numeric, Tperm: BinaryInteger>(
  x: Tensor<T>,
  perm: Tensor<Tperm>
) -> Tensor<T> {
  return #tfop("Transpose",
    x,
    perm,
    T: T.self,
    Tperm: Tperm.self)
}

// Returns x / y element-wise for integer types.
//
// Truncation designates that negative numbers will round fractional quantities
// toward zero. I.e. -7 / 5 = -1. This matches C semantics but it is different
// than Python semantics. See `FloorDiv` for a division function that matches
// Python Semantics.
//
// *NOTE*: `TruncateDiv` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@_inlineable @inline(__always)
public static func truncateDiv<T: Numeric>(
  x: Tensor<T>,
  y: Tensor<T>
) -> Tensor<T> {
  return #tfop("TruncateDiv",
    x,
    y,
    T: T.self)
}

// Returns element-wise remainder of division. This emulates C semantics in that
//
// the result here is consistent with a truncating divide. E.g. `truncate(x / y) *
// y + truncate_mod(x, y) = x`.
//
// *NOTE*: `TruncateMod` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@_inlineable @inline(__always)
public static func truncateMod<T: Numeric>(
  x: Tensor<T>,
  y: Tensor<T>
) -> Tensor<T> {
  return #tfop("TruncateMod",
    x,
    y,
    T: T.self)
}

// Outputs random values from a truncated normal distribution.
//
// The generated values follow a normal distribution with mean 0 and standard
// deviation 1, except that values whose magnitude is more than 2 standard
// deviations from the mean are dropped and re-picked.
//
// - Parameter shape: The shape of the output tensor.
//
// - Attrs:
//   - seed: If either `seed` or `seed2` are set to be non-zero, the random number
//     generator is seeded by the given seed.  Otherwise, it is seeded by a
//     random seed.
//   - seed2: A second seed to avoid seed collision.
//   - dtype: The type of the output.
//
// - Output output: A tensor of the specified shape filled with random truncated normal
//   values.
@_inlineable @inline(__always)
public static func truncatedNormal<Dtype: BinaryFloatingPoint, T: BinaryInteger>(
  shape: Tensor<T>,
  seed: Int = 0,
  seed2: Int = 0
) -> Tensor<Dtype> {
  return #tfop("TruncatedNormal",
    shape,
    Dtype: Dtype.self,
    T: T.self,
    seed: seed,
    seed2: seed2)
}

@_inlineable @inline(__always)
public static func twoFloatInputs(
  a: Tensor<Float>,
  b: Tensor<Float>
) {
  return #tfop("TwoFloatInputs",
    a,
    b)
}

@_inlineable @inline(__always)
public static func twoFloatInputsFloatOutput(
  a: Tensor<Float>,
  b: Tensor<Float>
) -> Tensor<Float> {
  return #tfop("TwoFloatInputsFloatOutput",
    a,
    b)
}

@_inlineable @inline(__always)
public static func twoFloatInputsIntOutput(
  a: Tensor<Float>,
  b: Tensor<Float>
) -> Tensor<Int32> {
  return #tfop("TwoFloatInputsIntOutput",
    a,
    b)
}

@_inlineable @inline(__always)
public static func twoFloatOutputs(
) -> (Tensor<Float>, Tensor<Float>) {
  return #tfop("TwoFloatOutputs")
}

@_inlineable @inline(__always)
public static func twoIntInputs(
  a: Tensor<Int32>,
  b: Tensor<Int32>
) {
  return #tfop("TwoIntInputs",
    a,
    b)
}

@_inlineable @inline(__always)
public static func twoIntOutputs(
) -> (Tensor<Int32>, Tensor<Int32>) {
  return #tfop("TwoIntOutputs")
}

@_inlineable @inline(__always)
public static func twoRefsIn<T: Numeric>(
  a: Tensor<T>,
  b: Tensor<T>
) {
  return #tfop("TwoRefsIn",
    a,
    b,
    T: T.self)
}

@_inlineable @inline(__always)
public static func typeList<T: Numeric>(
  a: [Tensor<T>]
) {
  return #tfop("TypeList",
    a)
}

@_inlineable @inline(__always)
public static func typeListRestrict<T: Numeric>(
  a: [Tensor<T>]
) {
  return #tfop("TypeListRestrict",
    a)
}

@_inlineable @inline(__always)
public static func typeListTwice<T: Numeric>(
  a: [Tensor<T>],
  b: [Tensor<T>]
) {
  return #tfop("TypeListTwice",
    a,
    b)
}

@_inlineable @inline(__always)
public static func unary<T: Numeric>(
  a: Tensor<T>
) -> Tensor<T> {
  return #tfop("Unary",
    a,
    T: T.self)
}

// Reverses the operation of Batch for a single output Tensor.
//
// An instance of Unbatch either receives an empty batched_tensor, in which case it
// asynchronously waits until the values become available from a concurrently
// running instance of Unbatch with the same container and shared_name, or receives
// a non-empty batched_tensor in which case it finalizes all other concurrently
// running instances and outputs its own element from the batch.
//
// batched_tensor: The possibly transformed output of Batch. The size of the first
//  dimension should remain unchanged by the transformations for the operation to
//  work.
// batch_index: The matching batch_index obtained from Batch.
// id: The id scalar emitted by Batch.
// unbatched_tensor: The Tensor corresponding to this execution.
// timeout_micros: Maximum amount of time (in microseconds) to wait to receive the
//  batched input tensor associated with a given invocation of the op.
// container: Container to control resource sharing.
// shared_name: Instances of Unbatch with the same container and shared_name are
//  assumed to possibly belong to the same batch. If left empty, the op name will
//  be used as the shared name.
@_inlineable @inline(__always)
public static func unbatch<T: Numeric>(
  batchedTensor: Tensor<T>,
  batchIndex: Tensor<Int64>,
  id: Tensor<Int64>,
  timeoutMicros: Int,
  container: String,
  sharedName: String
) -> Tensor<T> {
  return #tfop("Unbatch",
    batchedTensor,
    batchIndex,
    id,
    T: T.self,
    timeout_micros: timeoutMicros,
    container: container,
    shared_name: sharedName)
}

// Gradient of Unbatch.
//
// Acts like Batch but using the given batch_index index of batching things as they
// become available. This ensures that the gradients are propagated back in the
// same session which did the forward pass.
//
// original_input: The input to the Unbatch operation this is the gradient of.
// batch_index: The batch_index given to the Unbatch operation this is the gradient
// of.
// grad: The downstream gradient.
// id: The id scalar emitted by Batch.
// batched_grad: The return value, either an empty tensor or the batched gradient.
// container: Container to control resource sharing.
// shared_name: Instances of UnbatchGrad with the same container and shared_name
//  are assumed to possibly belong to the same batch. If left empty, the op name
//  will be used as the shared name.
@_inlineable @inline(__always)
public static func unbatchGrad<T: Numeric>(
  originalInput: Tensor<T>,
  batchIndex: Tensor<Int64>,
  grad: Tensor<T>,
  id: Tensor<Int64>,
  container: String,
  sharedName: String
) -> Tensor<T> {
  return #tfop("UnbatchGrad",
    originalInput,
    batchIndex,
    grad,
    id,
    T: T.self,
    container: container,
    shared_name: sharedName)
}

// Generates labels for candidate sampling with a uniform distribution.
//
// See explanations of candidate sampling and the data formats at
// go/candidate-sampling.
//
// For each batch, this op picks a single set of sampled candidate labels.
//
// The advantages of sampling candidates per-batch are simplicity and the
// possibility of efficient dense matrix multiplication. The disadvantage is that
// the sampled candidates must be chosen independently of the context and of the
// true labels.
//
// - Parameter true_classes: A batch_size * num_true matrix, in which each row contains the
//   IDs of the num_true target_classes in the corresponding original label.
//
// - Attrs:
//   - num_true: Number of true labels per context.
//   - num_sampled: Number of candidates to randomly sample.
//   - unique: If unique is true, we sample with rejection, so that all sampled
//     candidates in a batch are unique. This requires some approximation to
//     estimate the post-rejection sampling probabilities.
//   - range_max: The sampler will sample integers from the interval [0, range_max).
//   - seed: If either seed or seed2 are set to be non-zero, the random number
//     generator is seeded by the given seed.  Otherwise, it is seeded by a
//     random seed.
//   - seed2: An second seed to avoid seed collision.
//
// - Outputs:
//   - sampled_candidates: A vector of length num_sampled, in which each element is
//     the ID of a sampled candidate.
//   - true_expected_count: A batch_size * num_true matrix, representing
//     the number of times each candidate is expected to occur in a batch
//     of sampled candidates. If unique=true, then this is a probability.
//   - sampled_expected_count: A vector of length num_sampled, for each sampled
//     candidate representing the number of times the candidate is expected
//     to occur in a batch of sampled candidates.  If unique=true, then this is a
//     probability.
@_inlineable @inline(__always)
public static func uniformCandidateSampler(
  trueClasses: Tensor<Int64>,
  numTrue: Int,
  numSampled: Int,
  unique: Bool,
  rangeMax: Int,
  seed: Int = 0,
  seed2: Int = 0
) -> (Tensor<Int64>, Tensor<Float>, Tensor<Float>) {
  return #tfop("UniformCandidateSampler",
    trueClasses,
    num_true: numTrue,
    num_sampled: numSampled,
    unique: unique,
    range_max: rangeMax,
    seed: seed,
    seed2: seed2)
}

// Finds unique elements in a 1-D tensor.
//
// This operation returns a tensor `y` containing all of the unique elements of `x`
// sorted in the same order that they occur in `x`. This operation also returns a
// tensor `idx` the same size as `x` that contains the index of each value of `x`
// in the unique output `y`. In other words:
//
// `y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`
//
// For example:
//
// ```
// # tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
// y, idx = unique(x)
// y ==> [1, 2, 4, 7, 8]
// idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
// ```
//
// - Parameter x: 1-D.
//
// - Outputs:
//   - y: 1-D.
//   - idx: 1-D.
@_inlineable @inline(__always)
public static func unique<T: Numeric, Out_idx: BinaryInteger>(
  x: Tensor<T>
) -> (Tensor<T>, Tensor<Out_idx>) {
  return #tfop("Unique",
    x,
    T: T.self,
    Out_idx: Out_idx.self)
}

// Finds unique elements along an axis of a tensor.
//
// This operation either returns a tensor `y` containing unique elements
// along the `axis` of a tensor. The returned unique elements is sorted
// in the same order as they occur along `axis` in `x`.
// This operation also returns a tensor `idx` that is the same size as
// the number of the elements in `x` along the `axis` dimension. It
// contains the index in the unique output `y`.
// In other words, for an `1-D` tensor `x` with `axis = None:
//
// `y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`
//
// For example:
//
// ```
// # tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
// y, idx = unique(x)
// y ==> [1, 2, 4, 7, 8]
// idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
// ```
//
// For an `2-D` tensor `x` with `axis = 0`:
//
// ```
// # tensor 'x' is [[1, 0, 0],
// #                [1, 0, 0],
// #                [2, 0, 0]]
// y, idx = unique(x, axis=0)
// y ==> [[1, 0, 0],
//        [2, 0, 0]]
// idx ==> [0, 0, 1]
// ```
//
// For an `2-D` tensor `x` with `axis = 1`:
//
// ```
// # tensor 'x' is [[1, 0, 0],
// #                [1, 0, 0],
// #                [2, 0, 0]]
// y, idx = unique(x, axis=1)
// y ==> [[1, 0],
//        [1, 0],
//        [2, 0]]
// idx ==> [0, 1, 1]
// ```
//
// - Parameters:
//   - x: A `Tensor`.
//   - axis: A `Tensor` of type `int32` (default: None). The axis of the Tensor to
//     find the unique elements.
//
// - Outputs:
//   - y: A `Tensor`. Unique elements along the `axis` of `Tensor` x.
//   - idx: A 1-D Tensor. Has the same type as x that contains the index of each
//     value of x in the output y.
@_inlineable @inline(__always)
public static func uniqueV2<T: Numeric, Taxis: BinaryInteger, Out_idx: BinaryInteger>(
  x: Tensor<T>,
  axis: Tensor<Taxis>
) -> (Tensor<T>, Tensor<Out_idx>) {
  return #tfop("UniqueV2",
    x,
    axis,
    T: T.self,
    Taxis: Taxis.self,
    Out_idx: Out_idx.self)
}

// Finds unique elements in a 1-D tensor.
//
// This operation returns a tensor `y` containing all of the unique elements of `x`
// sorted in the same order that they occur in `x`. This operation also returns a
// tensor `idx` the same size as `x` that contains the index of each value of `x`
// in the unique output `y`. Finally, it returns a third tensor `count` that
// contains the count of each element of `y` in `x`. In other words:
//
// `y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`
//
// For example:
//
// ```
// # tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
// y, idx, count = unique_with_counts(x)
// y ==> [1, 2, 4, 7, 8]
// idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
// count ==> [2, 1, 3, 1, 2]
// ```
//
// - Parameter x: 1-D.
//
// - Outputs:
//   - y: 1-D.
//   - idx: 1-D.
//   - count: 1-D.
@_inlineable @inline(__always)
public static func uniqueWithCounts<T: Numeric, Out_idx: BinaryInteger>(
  x: Tensor<T>
) -> (Tensor<T>, Tensor<Out_idx>, Tensor<Out_idx>) {
  return #tfop("UniqueWithCounts",
    x,
    T: T.self,
    Out_idx: Out_idx.self)
}

// Finds unique elements along an axis of a tensor.
//
// This operation either returns a tensor `y` containing unique elements
// along the `axis` of a tensor. The returned unique elements is sorted
// in the same order as they occur along `axis` in `x`.
// This operation also returns a tensor `idx` and a tensor `count`
// that are the same size as the number of the elements in `x` along the
// `axis` dimension. The `idx` contains the index in the unique output `y`
// and the `count` contains the count in the unique output `y`.
// In other words, for an `1-D` tensor `x` with `axis = None:
//
// `y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`
//
// For example:
//
// ```
// # tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
// y, idx, count = unique_with_counts(x)
// y ==> [1, 2, 4, 7, 8]
// idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
// count ==> [2, 1, 3, 1, 2]
// ```
//
// For an `2-D` tensor `x` with `axis = 0`:
//
// ```
// # tensor 'x' is [[1, 0, 0],
// #                [1, 0, 0],
// #                [2, 0, 0]]
// y, idx, count = unique_with_counts(x, axis=0)
// y ==> [[1, 0, 0],
//        [2, 0, 0]]
// idx ==> [0, 0, 1]
// count ==> [2, 1]
// ```
//
// For an `2-D` tensor `x` with `axis = 1`:
//
// ```
// # tensor 'x' is [[1, 0, 0],
// #                [1, 0, 0],
// #                [2, 0, 0]]
// y, idx, count = unique_with_counts(x, axis=1)
// y ==> [[1, 0],
//        [1, 0],
//        [2, 0]]
// idx ==> [0, 1, 1]
// count ==> [1, 2]
// ```
//
// - Parameters:
//   - x: A `Tensor`.
//   - axis: A `Tensor` of type `int32` (default: None). The axis of the Tensor to
//     find the unique elements.
//
// - Outputs:
//   - y: A `Tensor`. Unique elements along the `axis` of `Tensor` x.
//   - idx: A 1-D Tensor. Has the same type as x that contains the index of each
//     value of x in the output y.
//   - count: A 1-D Tensor. The count of each value of x in the output y.
@_inlineable @inline(__always)
public static func uniqueWithCountsV2<T: Numeric, Taxis: BinaryInteger, Out_idx: BinaryInteger>(
  x: Tensor<T>,
  axis: Tensor<Taxis>
) -> (Tensor<T>, Tensor<Out_idx>, Tensor<Out_idx>) {
  return #tfop("UniqueWithCountsV2",
    x,
    axis,
    T: T.self,
    Taxis: Taxis.self,
    Out_idx: Out_idx.self)
}

// Unpacks a given dimension of a rank-`R` tensor into `num` rank-`(R-1)` tensors.
//
// Unpacks `num` tensors from `value` by chipping it along the `axis` dimension.
// For example, given a tensor of shape `(A, B, C, D)`;
//
// If `axis == 0` then the i'th tensor in `output` is the slice `value[i, :, :, :]`
//   and each tensor in `output` will have shape `(B, C, D)`. (Note that the
//   dimension unpacked along is gone, unlike `split`).
//
// If `axis == 1` then the i'th tensor in `output` is the slice `value[:, i, :, :]`
//   and each tensor in `output` will have shape `(A, C, D)`.
// Etc.
//
// This is the opposite of `pack`.
//
// - Parameter value: 1-D or higher, with `axis` dimension size equal to `num`.
//
// - Attr axis: Dimension along which to unpack.  Negative values wrap around, so the
//   valid range is `[-R, R)`.
//
// - Output output: The list of tensors unpacked from `value`.
@_inlineable @inline(__always)
public static func unpack<T: Numeric>(
  value: Tensor<T>,
  num: Int,
  axis: Int = 0
) -> [Tensor<T>] {
  return #tfop("Unpack",
    value,
    T: T.self,
    num: num,
    axis: axis)
}

// Converts a flat index or array of flat indices into a tuple of
//
// coordinate arrays.
//
// @compatibility(numpy)
// Equivalent to np.unravel_index
// @end_compatibility
//
// - Parameters:
//   - indices: An 0-D or 1-D `int` Tensor whose elements are indices into the
//     flattened version of an array of dimensions dims.
//   - dims: An 1-D `int` Tensor. The shape of the array to use for unraveling
//     indices.
//
// - Output output: An 2-D (or 1-D if indices is 0-D) tensor where each row has the
//   same shape as the indices array.
@_inlineable @inline(__always)
public static func unravelIndex<Tidx: BinaryInteger>(
  indices: Tensor<Tidx>,
  dims: Tensor<Tidx>
) -> Tensor<Tidx> {
  return #tfop("UnravelIndex",
    indices,
    dims,
    Tidx: Tidx.self)
}

// Computes the maximum along segments of a tensor.
//
// Read @{$math_ops#Segmentation$the section on segmentation} for an explanation of
// segments.
//
// This operator is similar to the unsorted segment sum operator found
// [(here)](../../../api_docs/python/math_ops.md#UnsortedSegmentSum).
// Instead of computing the sum over segments, it computes the maximum such that:
//
// \\(output_i = \max_j data_j\\) where max is over `j` such
// that `segment_ids[j] == i`.
//
// If the maximum is empty for a given segment ID `i`, it outputs the smallest
// possible value for the specific numeric type,
// `output[i] = numeric_limits<T>::lowest()`.
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="https://www.tensorflow.org/images/UnsortedSegmentMax.png" alt>
// </div>
//
// - Parameter segment_ids: A 1-D tensor whose rank is equal to the rank of `data`'s
//   first dimension.
//
// - Output output: Has same shape as data, except for dimension 0 which
//   has size `num_segments`.
@_inlineable @inline(__always)
public static func unsortedSegmentMax<T: Numeric, Tindices: BinaryInteger, Tnumsegments: BinaryInteger>(
  data: Tensor<T>,
  segmentIds: Tensor<Tindices>,
  numSegments: Tensor<Tnumsegments>
) -> Tensor<T> {
  return #tfop("UnsortedSegmentMax",
    data,
    segmentIds,
    numSegments,
    T: T.self,
    Tindices: Tindices.self,
    Tnumsegments: Tnumsegments.self)
}

// Computes the minimum along segments of a tensor.
//
// Read @{$math_ops#segmentation$the section on segmentation} for an explanation of
// segments.
//
// This operator is similar to the unsorted segment sum operator found
// [(here)](../../../api_docs/python/math_ops.md#UnsortedSegmentSum).
// Instead of computing the sum over segments, it computes the minimum such that:
//
// \\(output_i = \min_j data_j\\) where min is over `j` such
// that `segment_ids[j] == i`.
//
// If the minimum is empty for a given segment ID `i`, it outputs the largest
// possible value for the specific numeric type,
// `output[i] = numeric_limits<T>::max()`.
//
// - Parameter segment_ids: A 1-D tensor whose rank is equal to the rank of `data`'s
//   first dimension.
//
// - Output output: Has same shape as data, except for dimension 0 which
//   has size `num_segments`.
@_inlineable @inline(__always)
public static func unsortedSegmentMin<T: Numeric, Tindices: BinaryInteger, Tnumsegments: BinaryInteger>(
  data: Tensor<T>,
  segmentIds: Tensor<Tindices>,
  numSegments: Tensor<Tnumsegments>
) -> Tensor<T> {
  return #tfop("UnsortedSegmentMin",
    data,
    segmentIds,
    numSegments,
    T: T.self,
    Tindices: Tindices.self,
    Tnumsegments: Tnumsegments.self)
}

// Computes the product along segments of a tensor.
//
// Read @{$math_ops#segmentation$the section on segmentation} for an explanation of
// segments.
//
// This operator is similar to the unsorted segment sum operator found
// [(here)](../../../api_docs/python/math_ops.md#UnsortedSegmentSum).
// Instead of computing the sum over segments, it computes the product of all
// entries belonging to a segment such that:
//
// \\(output_i = \prod_j data_j\\) where the product is over `j` such
// that `segment_ids[j] == i`.
//
// If there is no entry for a given segment ID `i`, it outputs 1.
//
// - Parameter segment_ids: A 1-D tensor whose rank is equal to the rank of `data`'s
//   first dimension.
//
// - Output output: Has same shape as data, except for dimension 0 which
//   has size `num_segments`.
@_inlineable @inline(__always)
public static func unsortedSegmentProd<T: Numeric, Tindices: BinaryInteger, Tnumsegments: BinaryInteger>(
  data: Tensor<T>,
  segmentIds: Tensor<Tindices>,
  numSegments: Tensor<Tnumsegments>
) -> Tensor<T> {
  return #tfop("UnsortedSegmentProd",
    data,
    segmentIds,
    numSegments,
    T: T.self,
    Tindices: Tindices.self,
    Tnumsegments: Tnumsegments.self)
}

// Computes the sum along segments of a tensor.
//
// Read @{$math_ops#Segmentation$the section on segmentation} for an explanation of
// segments.
//
// Computes a tensor such that
// `(output[i] = sum_{j...} data[j...]` where the sum is over tuples `j...` such
// that `segment_ids[j...] == i`.  Unlike `SegmentSum`, `segment_ids`
// need not be sorted and need not cover all values in the full
// range of valid values.
//
// If the sum is empty for a given segment ID `i`, `output[i] = 0`.
// If the given segment ID `i` is negative, the value is dropped and will not be
// added to the sum of the segment.
//
// `num_segments` should equal the number of distinct segment IDs.
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="https://www.tensorflow.org/images/UnsortedSegmentSum.png" alt>
// </div>
//
// - Parameter segment_ids: A tensor whose shape is a prefix of `data.shape`.
//
// - Output output: Has same shape as data, except for the first `segment_ids.rank`
//   dimensions, which are replaced with a single dimension which has size
//   `num_segments`.
@_inlineable @inline(__always)
public static func unsortedSegmentSum<T: Numeric, Tindices: BinaryInteger, Tnumsegments: BinaryInteger>(
  data: Tensor<T>,
  segmentIds: Tensor<Tindices>,
  numSegments: Tensor<Tnumsegments>
) -> Tensor<T> {
  return #tfop("UnsortedSegmentSum",
    data,
    segmentIds,
    numSegments,
    T: T.self,
    Tindices: Tindices.self,
    Tnumsegments: Tnumsegments.self)
}

// Op is similar to a lightweight Dequeue.
//
// The basic functionality is similar to dequeue with many fewer
// capabilities and options.  This Op is optimized for performance.
@_inlineable @inline(__always)
public static func unstage<Dtypes: Numeric>(
  capacity: Int = 0,
  memoryLimit: Int = 0,
  container: String,
  sharedName: String
) -> [Tensor<Dtypes>] {
  return #tfop("Unstage",
    capacity: capacity,
    memory_limit: memoryLimit,
    container: container,
    shared_name: sharedName)
}

// Returns locations of nonzero / true values in a tensor.
//
// This operation returns the coordinates of true elements in `condition`. The
// coordinates are returned in a 2-D tensor where the first dimension (rows)
// represents the number of true elements, and the second dimension (columns)
// represents the coordinates of the true elements. Keep in mind, the shape of
// the output tensor can vary depending on how many true values there are in
// `condition`. Indices are output in row-major order.
//
// For example:
//
// ```
// # 'input' tensor is [[True, False]
// #                    [True, False]]
// # 'input' has two true values, so output has two coordinates.
// # 'input' has rank of 2, so coordinates have two indices.
// where(input) ==> [[0, 0],
//                   [1, 0]]
//
// # `condition` tensor is [[[True, False]
// #                     [True, False]]
// #                    [[False, True]
// #                     [False, True]]
// #                    [[False, False]
// #                     [False, True]]]
// # 'input' has 5 true values, so output has 5 coordinates.
// # 'input' has rank of 3, so coordinates have three indices.
// where(input) ==> [[0, 0, 0],
//                   [0, 1, 0],
//                   [1, 0, 1],
//                   [1, 1, 1],
//                   [2, 1, 1]]
//
// # `condition` tensor is [[[1.5,  0.0]
// #                     [-0.5, 0.0]]
// #                    [[0.0,  0.25]
// #                     [0.0,  0.75]]
// #                    [[0.0,  0.0]
// #                     [0.0,  0.01]]]
// # 'input' has 5 nonzero values, so output has 5 coordinates.
// # 'input' has rank of 3, so coordinates have three indices.
// where(input) ==> [[0, 0, 0],
//                   [0, 1, 0],
//                   [1, 0, 1],
//                   [1, 1, 1],
//                   [2, 1, 1]]
//
// # `condition` tensor is [[[1.5 + 0.0j, 0.0  + 0.0j]
// #                     [0.0 + 0.5j, 0.0  + 0.0j]]
// #                    [[0.0 + 0.0j, 0.25 + 1.5j]
// #                     [0.0 + 0.0j, 0.75 + 0.0j]]
// #                    [[0.0 + 0.0j, 0.0  + 0.0j]
// #                     [0.0 + 0.0j, 0.01 + 0.0j]]]
// # 'input' has 5 nonzero magnitude values, so output has 5 coordinates.
// # 'input' has rank of 3, so coordinates have three indices.
// where(input) ==> [[0, 0, 0],
//                   [0, 1, 0],
//                   [1, 0, 1],
//                   [1, 1, 1],
//                   [2, 1, 1]]
// ```
@_inlineable @inline(__always)
public static func where_<T: Numeric>(
  input: Tensor<T>
) -> Tensor<Int64> {
  return #tfop("Where",
    input,
    T: T.self)
}

// Returns a tensor of zeros with the same shape and type as x.
//
// - Parameter x: a tensor of type T.
//
// - Output y: a tensor of the same shape and type as x but filled with zeros.
@_inlineable @inline(__always)
public static func zerosLike<T: Numeric>(
  x: Tensor<T>
) -> Tensor<T> {
  return #tfop("ZerosLike",
    x,
    T: T.self)
}

// Compute the Hurwitz zeta function \\(\zeta(x, q)\\).
//
// The Hurwitz zeta function is defined as:
//
//
// \\(\zeta(x, q) = \sum_{n=0}^{\infty} (q + n)^{-x}\\)
@_inlineable @inline(__always)
public static func zeta<T: BinaryFloatingPoint>(
  x: Tensor<T>,
  q: Tensor<T>
) -> Tensor<T> {
  return #tfop("Zeta",
    x,
    q,
    T: T.self)
}

}