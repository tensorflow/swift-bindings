// !!! THIS CODE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND !!!
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

static let generatedTensorFlowVersion = "1.13.1"
static let generatedTensorFlowGitVersion = "v1.12.0-11444-g8b2884c9cb"

// @_frozen // SR-9739
public enum A {
  case apples
  case oranges

  @inlinable
  var cName: String {
    @inline(__always)
    get {
      switch self {
      case .apples: return "apples"
      case .oranges: return "oranges"
      }
    }
  }
}

// @_frozen // SR-9739
public enum DataFormat {
  case nchw
  case nhwc

  @inlinable
  var cName: String {
    @inline(__always)
    get {
      switch self {
      case .nchw: return "NCHW"
      case .nhwc: return "NHWC"
      }
    }
  }
}

// @_frozen // SR-9739
public enum DataFormat1 {
  case ncdhw
  case ndhwc

  @inlinable
  var cName: String {
    @inline(__always)
    get {
      switch self {
      case .ncdhw: return "NCDHW"
      case .ndhwc: return "NDHWC"
      }
    }
  }
}

// @_frozen // SR-9739
public enum DataFormat4 {
  case nchw
  case nchwVectC
  case nhwc

  @inlinable
  var cName: String {
    @inline(__always)
    get {
      switch self {
      case .nchw: return "NCHW"
      case .nchwVectC: return "NCHW_VECT_C"
      case .nhwc: return "NHWC"
      }
    }
  }
}

// @_frozen // SR-9739
public enum DensityUnit {
  case cm
  case in_

  @inlinable
  var cName: String {
    @inline(__always)
    get {
      switch self {
      case .cm: return "cm"
      case .in_: return "in"
      }
    }
  }
}

// @_frozen // SR-9739
public enum Direction {
  case bidirectional
  case unidirectional

  @inlinable
  var cName: String {
    @inline(__always)
    get {
      switch self {
      case .bidirectional: return "bidirectional"
      case .unidirectional: return "unidirectional"
      }
    }
  }
}

// @_frozen // SR-9739
public enum Errors {
  case ignore
  case replace
  case strict

  @inlinable
  var cName: String {
    @inline(__always)
    get {
      switch self {
      case .ignore: return "ignore"
      case .replace: return "replace"
      case .strict: return "strict"
      }
    }
  }
}

// @_frozen // SR-9739
public enum FinalOp {
  case div
  case id

  @inlinable
  var cName: String {
    @inline(__always)
    get {
      switch self {
      case .div: return "Div"
      case .id: return "Id"
      }
    }
  }
}

// @_frozen // SR-9739
public enum Format {
  case empty
  case grayscale
  case rgb

  @inlinable
  var cName: String {
    @inline(__always)
    get {
      switch self {
      case .empty: return ""
      case .grayscale: return "grayscale"
      case .rgb: return "rgb"
      }
    }
  }
}

// @_frozen // SR-9739
public enum InputMode {
  case autoSelect
  case linearInput
  case skipInput

  @inlinable
  var cName: String {
    @inline(__always)
    get {
      switch self {
      case .autoSelect: return "auto_select"
      case .linearInput: return "linear_input"
      case .skipInput: return "skip_input"
      }
    }
  }
}

// @_frozen // SR-9739
public enum LossType {
  case hingeLoss
  case logisticLoss
  case poissonLoss
  case smoothHingeLoss
  case squaredLoss

  @inlinable
  var cName: String {
    @inline(__always)
    get {
      switch self {
      case .hingeLoss: return "hinge_loss"
      case .logisticLoss: return "logistic_loss"
      case .poissonLoss: return "poisson_loss"
      case .smoothHingeLoss: return "smooth_hinge_loss"
      case .squaredLoss: return "squared_loss"
      }
    }
  }
}

// @_frozen // SR-9739
public enum MergeOp {
  case add
  case max
  case min
  case mul

  @inlinable
  var cName: String {
    @inline(__always)
    get {
      switch self {
      case .add: return "Add"
      case .max: return "Max"
      case .min: return "Min"
      case .mul: return "Mul"
      }
    }
  }
}

// @_frozen // SR-9739
public enum Method {
  case bilinear
  case nearest

  @inlinable
  var cName: String {
    @inline(__always)
    get {
      switch self {
      case .bilinear: return "bilinear"
      case .nearest: return "nearest"
      }
    }
  }
}

// @_frozen // SR-9739
public enum Method3 {
  case bilinear

  @inlinable
  var cName: String {
    @inline(__always)
    get {
      switch self {
      case .bilinear: return "bilinear"
      }
    }
  }
}

// @_frozen // SR-9739
public enum Mode {
  case minCombined
  case minFirst
  case scaled

  @inlinable
  var cName: String {
    @inline(__always)
    get {
      switch self {
      case .minCombined: return "MIN_COMBINED"
      case .minFirst: return "MIN_FIRST"
      case .scaled: return "SCALED"
      }
    }
  }
}

// @_frozen // SR-9739
public enum Mode5 {
  case reflect
  case symmetric

  @inlinable
  var cName: String {
    @inline(__always)
    get {
      switch self {
      case .reflect: return "REFLECT"
      case .symmetric: return "SYMMETRIC"
      }
    }
  }
}

// @_frozen // SR-9739
public enum OutputEncoding {
  case utf16Be
  case utf32Be
  case utf8

  @inlinable
  var cName: String {
    @inline(__always)
    get {
      switch self {
      case .utf16Be: return "UTF-16-BE"
      case .utf32Be: return "UTF-32-BE"
      case .utf8: return "UTF-8"
      }
    }
  }
}

// @_frozen // SR-9739
public enum Padding {
  case same
  case valid

  @inlinable
  var cName: String {
    @inline(__always)
    get {
      switch self {
      case .same: return "SAME"
      case .valid: return "VALID"
      }
    }
  }
}

// @_frozen // SR-9739
public enum Padding2 {
  case explicit
  case same
  case valid

  @inlinable
  var cName: String {
    @inline(__always)
    get {
      switch self {
      case .explicit: return "EXPLICIT"
      case .same: return "SAME"
      case .valid: return "VALID"
      }
    }
  }
}

// @_frozen // SR-9739
public enum Reduction {
  case max
  case min
  case prod
  case sum

  @inlinable
  var cName: String {
    @inline(__always)
    get {
      switch self {
      case .max: return "max"
      case .min: return "min"
      case .prod: return "prod"
      case .sum: return "sum"
      }
    }
  }
}

// @_frozen // SR-9739
public enum RnnMode {
  case gru
  case lstm
  case rnnRelu
  case rnnTanh

  @inlinable
  var cName: String {
    @inline(__always)
    get {
      switch self {
      case .gru: return "gru"
      case .lstm: return "lstm"
      case .rnnRelu: return "rnn_relu"
      case .rnnTanh: return "rnn_tanh"
      }
    }
  }
}

// @_frozen // SR-9739
public enum RoundMode {
  case halfToEven
  case halfUp

  @inlinable
  var cName: String {
    @inline(__always)
    get {
      switch self {
      case .halfToEven: return "HALF_TO_EVEN"
      case .halfUp: return "HALF_UP"
      }
    }
  }
}

// @_frozen // SR-9739
public enum RoundMode6 {
  case halfAwayFromZero
  case halfToEven

  @inlinable
  var cName: String {
    @inline(__always)
    get {
      switch self {
      case .halfAwayFromZero: return "HALF_AWAY_FROM_ZERO"
      case .halfToEven: return "HALF_TO_EVEN"
      }
    }
  }
}

// @_frozen // SR-9739
public enum Unit {
  case byte
  case utf8Char

  @inlinable
  var cName: String {
    @inline(__always)
    get {
      switch self {
      case .byte: return "BYTE"
      case .utf8Char: return "UTF8_CHAR"
      }
    }
  }
}

@inlinable @inline(__always)
public static func a(
) -> Tensor<Float> {
  let ret: TensorHandle<Float> = #tfop("A")
  return Tensor(handle: ret)
}

/// Raise a exception to abort the process when called.
///
/// If exit_without_error is true, the process will exit normally,
/// otherwise it will exit with a SIGABORT signal.
///
/// Returns nothing but an exception.
///
/// - Attr error_msg: A string which is the message associated with the exception.
@inlinable @inline(__always)
public static func abort(
  errorMsg: String,
  exitWithoutError: Bool = false
) {
  return #tfop("Abort",
    error_msg: errorMsg,
    exit_without_error: exitWithoutError)
}

/// Computes the absolute value of a tensor.
///
/// Given a tensor `x`, this operation returns a tensor containing the absolute
/// value of each element in `x`. For example, if x is an input element and y is
/// an output element, this operation computes \\(y = |x|\\).
@inlinable @inline(__always)
public static func abs<T: Numeric & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Abs",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes acos of x element-wise.
@inlinable @inline(__always)
public static func acos<T: Numeric & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Acos",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes inverse hyperbolic cosine of x element-wise.
@inlinable @inline(__always)
public static func acosh<T: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Acosh",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Returns x + y element-wise.
///
/// *NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@inlinable @inline(__always)
public static func add<T: Numeric & TensorFlowScalar>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Add",
    x,
    y,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Add an `N`-minibatch `SparseTensor` to a `SparseTensorsMap`, return `N` handles.
///
/// A `SparseTensor` of rank `R` is represented by three tensors: `sparse_indices`,
/// `sparse_values`, and `sparse_shape`, where
///
/// ```sparse_indices.shape[1] == sparse_shape.shape[0] == R```
///
/// An `N`-minibatch of `SparseTensor` objects is represented as a `SparseTensor`
/// having a first `sparse_indices` column taking values between `[0, N)`, where
/// the minibatch size `N == sparse_shape[0]`.
///
/// The input `SparseTensor` must have rank `R` greater than 1, and the first
/// dimension is treated as the minibatch dimension.  Elements of the `SparseTensor`
/// must be sorted in increasing order of this first dimension.  The stored
/// `SparseTensor` objects pointed to by each row of the output `sparse_handles`
/// will have rank `R-1`.
///
/// The `SparseTensor` values can then be read out as part of a minibatch by passing
/// the given keys as vector elements to `TakeManySparseFromTensorsMap`.  To ensure
/// the correct `SparseTensorsMap` is accessed, ensure that the same
/// `container` and `shared_name` are passed to that Op.  If no `shared_name`
/// is provided here, instead use the *name* of the Operation created by calling
/// `AddManySparseToTensorsMap` as the `shared_name` passed to
/// `TakeManySparseFromTensorsMap`.  Ensure the Operations are colocated.
///
/// - Parameters:
///   - sparse_indices: 2-D.  The `indices` of the minibatch `SparseTensor`.
///     `sparse_indices[:, 0]` must be ordered values in `[0, N)`.
///   - sparse_values: 1-D.  The `values` of the minibatch `SparseTensor`.
///   - sparse_shape: 1-D.  The `shape` of the minibatch `SparseTensor`.
///     The minibatch size `N == sparse_shape[0]`.
///
/// - Attrs:
///   - container: The container name for the `SparseTensorsMap` created by this op.
///   - shared_name: The shared name for the `SparseTensorsMap` created by this op.
///     If blank, the new Operation's unique name is used.
///
/// - Output sparse_handles: 1-D.  The handles of the `SparseTensor` now stored in the
///   `SparseTensorsMap`.  Shape: `[N]`.
@inlinable @inline(__always)
public static func addManySparseToTensorsMap<T: TensorFlowScalar>(
  sparseIndices: Tensor<Int64>,
  sparseValues: Tensor<T>,
  sparseShape: Tensor<Int64>,
  container: String,
  sharedName: String
) -> Tensor<Int64> {
  let ret: TensorHandle<Int64> = #tfop("AddManySparseToTensorsMap",
    sparseIndices,
    sparseValues,
    sparseShape,
    T$dtype: T.tensorFlowDataType,
    container: container,
    shared_name: sharedName)
  return Tensor(handle: ret)
}

/// Add all input tensors element wise.
///
/// - Parameter inputs: Must all be the same size and shape.
@inlinable @inline(__always)
public static func addN<T: Numeric & TensorFlowScalar>(
  inputs: [Tensor<T>]
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("AddN",
    inputs,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Add a `SparseTensor` to a `SparseTensorsMap` return its handle.
///
/// A `SparseTensor` is represented by three tensors: `sparse_indices`,
/// `sparse_values`, and `sparse_shape`.
///
/// This operator takes the given `SparseTensor` and adds it to a container
/// object (a `SparseTensorsMap`).  A unique key within this container is generated
/// in the form of an `int64`, and this is the value that is returned.
///
/// The `SparseTensor` can then be read out as part of a minibatch by passing
/// the key as a vector element to `TakeManySparseFromTensorsMap`.  To ensure
/// the correct `SparseTensorsMap` is accessed, ensure that the same
/// `container` and `shared_name` are passed to that Op.  If no `shared_name`
/// is provided here, instead use the *name* of the Operation created by calling
/// `AddSparseToTensorsMap` as the `shared_name` passed to
/// `TakeManySparseFromTensorsMap`.  Ensure the Operations are colocated.
///
/// - Parameters:
///   - sparse_indices: 2-D.  The `indices` of the `SparseTensor`.
///   - sparse_values: 1-D.  The `values` of the `SparseTensor`.
///   - sparse_shape: 1-D.  The `shape` of the `SparseTensor`.
///
/// - Attrs:
///   - container: The container name for the `SparseTensorsMap` created by this op.
///   - shared_name: The shared name for the `SparseTensorsMap` created by this op.
///     If blank, the new Operation's unique name is used.
///
/// - Output sparse_handle: 0-D.  The handle of the `SparseTensor` now stored in the
///   `SparseTensorsMap`.
@inlinable @inline(__always)
public static func addSparseToTensorsMap<T: TensorFlowScalar>(
  sparseIndices: Tensor<Int64>,
  sparseValues: Tensor<T>,
  sparseShape: Tensor<Int64>,
  container: String,
  sharedName: String
) -> Tensor<Int64> {
  let ret: TensorHandle<Int64> = #tfop("AddSparseToTensorsMap",
    sparseIndices,
    sparseValues,
    sparseShape,
    T$dtype: T.tensorFlowDataType,
    container: container,
    shared_name: sharedName)
  return Tensor(handle: ret)
}

/// Returns x + y element-wise.
///
/// *NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@inlinable @inline(__always)
public static func addV2<T: Numeric & TensorFlowScalar>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("AddV2",
    x,
    y,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Deprecated. Disallowed in GraphDef version >= 2.
@inlinable @inline(__always)
public static func adjustContrast<T: Numeric & TensorFlowScalar>(
  images: Tensor<T>,
  contrastFactor: Tensor<Float>,
  minValue: Tensor<Float>,
  maxValue: Tensor<Float>
) -> Tensor<Float> {
  let ret: TensorHandle<Float> = #tfop("AdjustContrast",
    images,
    contrastFactor,
    minValue,
    maxValue,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Adjust the contrast of one or more images.
///
/// `images` is a tensor of at least 3 dimensions.  The last 3 dimensions are
/// interpreted as `[height, width, channels]`.  The other dimensions only
/// represent a collection of images, such as `[batch, height, width, channels].`
///
/// Contrast is adjusted independently for each channel of each image.
///
/// For each channel, the Op first computes the mean of the image pixels in the
/// channel and then adjusts each component of each pixel to
/// `(x - mean) * contrast_factor + mean`.
///
/// - Parameters:
///   - images: Images to adjust.  At least 3-D.
///   - contrast_factor: A float multiplier for adjusting contrast.
///
/// - Output output: The contrast-adjusted image or images.
@inlinable @inline(__always)
public static func adjustContrastv2<T: FloatingPoint & TensorFlowScalar>(
  images: Tensor<T>,
  contrastFactor: Tensor<Float>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("AdjustContrastv2",
    images,
    contrastFactor,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Adjust the hue of one or more images.
///
/// `images` is a tensor of at least 3 dimensions.  The last dimension is
/// interpretted as channels, and must be three.
///
/// The input image is considered in the RGB colorspace. Conceptually, the RGB
/// colors are first mapped into HSV. A delta is then applied all the hue values,
/// and then remapped back to RGB colorspace.
///
/// - Parameters:
///   - images: Images to adjust.  At least 3-D.
///   - delta: A float delta to add to the hue.
///
/// - Output output: The hue-adjusted image or images.
@inlinable @inline(__always)
public static func adjustHue<T: FloatingPoint & TensorFlowScalar>(
  images: Tensor<T>,
  delta: Tensor<Float>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("AdjustHue",
    images,
    delta,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Adjust the saturation of one or more images.
///
/// `images` is a tensor of at least 3 dimensions.  The last dimension is
/// interpretted as channels, and must be three.
///
/// The input image is considered in the RGB colorspace. Conceptually, the RGB
/// colors are first mapped into HSV. A scale is then applied all the saturation
/// values, and then remapped back to RGB colorspace.
///
/// - Parameters:
///   - images: Images to adjust.  At least 3-D.
///   - scale: A float scale to add to the saturation.
///
/// - Output output: The hue-adjusted image or images.
@inlinable @inline(__always)
public static func adjustSaturation<T: FloatingPoint & TensorFlowScalar>(
  images: Tensor<T>,
  scale: Tensor<Float>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("AdjustSaturation",
    images,
    scale,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the "logical and" of elements across dimensions of a tensor.
///
/// Reduces `input` along the dimensions given in `axis`. Unless
/// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
/// `axis`. If `keep_dims` is true, the reduced dimensions are
/// retained with length 1.
///
/// - Parameters:
///   - input: The tensor to reduce.
///   - reduction_indices: The dimensions to reduce. Must be in the range
///     `[-rank(input), rank(input))`.
///
/// - Attr keep_dims: If true, retain reduced dimensions with length 1.
///
/// - Output output: The reduced tensor.
@inlinable @inline(__always)
public static func all<Tidx: BinaryInteger & TensorFlowScalar>(
  _ input: Tensor<Bool>,
  reductionIndices: Tensor<Tidx>,
  keepDims: Bool = false
) -> Tensor<Bool> {
  let ret: TensorHandle<Bool> = #tfop("All",
    input,
    reductionIndices,
    Tidx$dtype: Tidx.tensorFlowDataType,
    keep_dims: keepDims)
  return Tensor(handle: ret)
}

/// Generates labels for candidate sampling with a learned unigram distribution.
///
/// See explanations of candidate sampling and the data formats at
/// go/candidate-sampling.
///
/// For each batch, this op picks a single set of sampled candidate labels.
///
/// The advantages of sampling candidates per-batch are simplicity and the
/// possibility of efficient dense matrix multiplication. The disadvantage is that
/// the sampled candidates must be chosen independently of the context and of the
/// true labels.
///
/// - Parameter true_classes: A batch_size * num_true matrix, in which each row contains the
///   IDs of the num_true target_classes in the corresponding original label.
///
/// - Attrs:
///   - num_true: Number of true labels per context.
///   - num_sampled: Number of candidates to produce.
///   - unique: If unique is true, we sample with rejection, so that all sampled
///     candidates in a batch are unique. This requires some approximation to
///     estimate the post-rejection sampling probabilities.
///   - seed: If either seed or seed2 are set to be non-zero, the random number
///     generator is seeded by the given seed.  Otherwise, it is seeded by a
///     random seed.
///   - seed2: An second seed to avoid seed collision.
///
/// - Outputs:
///   - sampled_candidates: A vector of length num_sampled, in which each element is
///     the ID of a sampled candidate.
///   - true_expected_count: A batch_size * num_true matrix, representing
///     the number of times each candidate is expected to occur in a batch
///     of sampled candidates. If unique=true, then this is a probability.
///   - sampled_expected_count: A vector of length num_sampled, for each sampled
///     candidate representing the number of times the candidate is expected
///     to occur in a batch of sampled candidates.  If unique=true, then this is a
///     probability.
@inlinable @inline(__always)
public static func allCandidateSampler(
  trueClasses: Tensor<Int64>,
  numTrue: Int64,
  numSampled: Int64,
  unique: Bool,
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> (sampledCandidates: Tensor<Int64>, trueExpectedCount: Tensor<Float>, sampledExpectedCount: Tensor<Float>) {
  let ret: (TensorHandle<Int64>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("AllCandidateSampler",
    trueClasses,
    num_true: numTrue,
    num_sampled: numSampled,
    unique: unique,
    seed: seed,
    seed2: seed2)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// An Op to exchange data across TPU replicas.
///
/// On each replica, the input is split into `split_count` blocks along
/// `split_dimension` and send to the other replicas given group_assignment. After
/// receiving `split_count` - 1 blocks from other replicas, we concatenate the
/// blocks along `concat_dimension` as the output.
///
/// For example, suppose there are 2 TPU replicas:
/// replica 0 receives input: `[[A, B]]`
/// replica 1 receives input: `[[C, D]]`
///
/// group_assignment=`[[0, 1]]`
/// concat_dimension=0
/// split_dimension=1
/// split_count=2
///
/// replica 0's output: `[[A], [C]]`
/// replica 1's output: `[[B], [D]]`
///
/// - Parameters:
///   - input: The local input to the sum.
///   - group_assignment: An int32 tensor with shape
///     [num_groups, num_replicas_per_group]. `group_assignment[i]` represents the
///     replica ids in the ith subgroup.
///
/// - Attrs:
///   - T: The type of elements to be exchanged.
///   - concat_dimension: The dimension number to concatenate.
///   - split_dimension: The dimension number to split.
///   - split_count: The number of splits, this number must equal to the sub-group
///     size(group_assignment.get_shape()[1])
///
/// - Output output: The exchanged result.
@inlinable @inline(__always)
public static func allToAll<T: TensorFlowScalar>(
  _ input: Tensor<T>,
  groupAssignment: Tensor<Int32>,
  concatDimension: Int64,
  splitDimension: Int64,
  splitCount: Int64
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("AllToAll",
    input,
    groupAssignment,
    T$dtype: T.tensorFlowDataType,
    concat_dimension: concatDimension,
    split_dimension: splitDimension,
    split_count: splitCount)
  return Tensor(handle: ret)
}

/// Returns the argument of a complex number.
///
/// Given a tensor `input` of complex numbers, this operation returns a tensor of
/// type `float` that is the argument of each element in `input`. All elements in
/// `input` must be complex numbers of the form \\(a + bj\\), where *a*
/// is the real part and *b* is the imaginary part.
///
/// The argument returned by this operation is of the form \\(atan2(b, a)\\).
///
/// For example:
///
/// ```
/// # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
/// tf.angle(input) ==> [2.0132, 1.056]
/// ```
///
/// @compatibility(numpy)
/// Equivalent to np.angle.
/// @end_compatibility
@inlinable @inline(__always)
public static func angle<T: TensorFlowScalar, Tout: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>
) -> Tensor<Tout> {
  let ret: TensorHandle<Tout> = #tfop("Angle",
    input,
    T$dtype: T.tensorFlowDataType,
    Tout$dtype: Tout.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the "logical or" of elements across dimensions of a tensor.
///
/// Reduces `input` along the dimensions given in `axis`. Unless
/// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
/// `axis`. If `keep_dims` is true, the reduced dimensions are
/// retained with length 1.
///
/// - Parameters:
///   - input: The tensor to reduce.
///   - reduction_indices: The dimensions to reduce. Must be in the range
///     `[-rank(input), rank(input))`.
///
/// - Attr keep_dims: If true, retain reduced dimensions with length 1.
///
/// - Output output: The reduced tensor.
@inlinable @inline(__always)
public static func any<Tidx: BinaryInteger & TensorFlowScalar>(
  _ input: Tensor<Bool>,
  reductionIndices: Tensor<Tidx>,
  keepDims: Bool = false
) -> Tensor<Bool> {
  let ret: TensorHandle<Bool> = #tfop("Any",
    input,
    reductionIndices,
    Tidx$dtype: Tidx.tensorFlowDataType,
    keep_dims: keepDims)
  return Tensor(handle: ret)
}

/// Returns the truth value of abs(x-y) < tolerance element-wise.
@inlinable @inline(__always)
public static func approximateEqual<T: Numeric & TensorFlowScalar>(
  _ x: Tensor<T>,
  _ y: Tensor<T>,
  tolerance: Double = 1e-05
) -> Tensor<Bool> {
  let ret: TensorHandle<Bool> = #tfop("ApproximateEqual",
    x,
    y,
    T$dtype: T.tensorFlowDataType,
    tolerance: tolerance)
  return Tensor(handle: ret)
}

/// Returns the index with the largest value across dimensions of a tensor.
///
/// Note that in case of ties the identity of the return value is not guaranteed.
///
/// Usage:
///   ```python
///   import tensorflow as tf
///   a = [1, 10, 26.9, 2.8, 166.32, 62.3]
///   b = tf.math.argmax(input = a)
///   c = tf.keras.backend.eval(b)  
///   # c = 4
///   # here a[4] = 166.32 which is the largest element of a across axis 0
///   ```
///
/// - Parameter dimension: int32 or int64, must be in the range `[-rank(input), rank(input))`.
///   Describes which dimension of the input Tensor to reduce across. For vectors,
///   use dimension = 0.
@inlinable @inline(__always)
public static func argMax<T: Numeric & TensorFlowScalar, Tidx: BinaryInteger & TensorFlowScalar, OutputType: BinaryInteger & TensorFlowScalar>(
  _ input: Tensor<T>,
  dimension: Tensor<Tidx>
) -> Tensor<OutputType> {
  let ret: TensorHandle<OutputType> = #tfop("ArgMax",
    input,
    dimension,
    T$dtype: T.tensorFlowDataType,
    Tidx$dtype: Tidx.tensorFlowDataType,
    output_type$dtype: OutputType.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Returns the index with the smallest value across dimensions of a tensor.
///
/// Note that in case of ties the identity of the return value is not guaranteed.
///
/// Usage:
///   ```python
///   import tensorflow as tf
///   a = [1, 10, 26.9, 2.8, 166.32, 62.3]
///   b = tf.math.argmin(input = a)
///   c = tf.keras.backend.eval(b)  
///   # c = 0
///   # here a[0] = 1 which is the smallest element of a across axis 0
///   ```
///
/// - Parameter dimension: int32 or int64, must be in the range `[-rank(input), rank(input))`.
///   Describes which dimension of the input Tensor to reduce across. For vectors,
///   use dimension = 0.
@inlinable @inline(__always)
public static func argMin<T: Numeric & TensorFlowScalar, Tidx: BinaryInteger & TensorFlowScalar, OutputType: BinaryInteger & TensorFlowScalar>(
  _ input: Tensor<T>,
  dimension: Tensor<Tidx>
) -> Tensor<OutputType> {
  let ret: TensorHandle<OutputType> = #tfop("ArgMin",
    input,
    dimension,
    T$dtype: T.tensorFlowDataType,
    Tidx$dtype: Tidx.tensorFlowDataType,
    output_type$dtype: OutputType.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Converts each entry in the given tensor to strings.  Supports many numeric
///
/// types and boolean.
///
/// - Attrs:
///   - precision: The post-decimal precision to use for floating point numbers.
///     Only used if precision > -1.
///   - scientific: Use scientific notation for floating point numbers.
///   - shortest: Use shortest representation (either scientific or standard) for
///     floating point numbers.
///   - width: Pad pre-decimal numbers to this width.
///     Applies to both floating point and integer numbers.
///     Only used if width > -1.
///   - fill: The value to pad if width > -1.  If empty, pads with spaces.
///     Another typical value is '0'.  String cannot be longer than 1 character.
@inlinable @inline(__always)
public static func asString<T: TensorFlowScalar>(
  _ input: Tensor<T>,
  precision: Int64 = -1,
  scientific: Bool = false,
  shortest: Bool = false,
  width: Int64 = -1,
  fill: String
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("AsString",
    input,
    T$dtype: T.tensorFlowDataType,
    precision: precision,
    scientific: scientific,
    shortest: shortest,
    width: width,
    fill: fill)
  return StringTensor(handle: ret)
}

/// Computes the trignometric inverse sine of x element-wise.
///
/// The `tf.math.asin` operation returns the inverse of `tf.math.sin`, such that
/// if `y = tf.math.sin(x)` then, `x = tf.math.asin(y)`.
///
/// **Note**: The output of `tf.math.asin` will lie within the invertible range 
/// of sine, i.e [-pi/2, pi/2].
///
/// For example:
///
/// ```python
/// # Note: [1.047, 0.785] ~= [(pi/3), (pi/4)]
/// x = tf.constant([1.047, 0.785])
/// y = tf.math.sin(x) # [0.8659266, 0.7068252]
///
/// tf.math.asin(y) # [1.047, 0.785] = x
/// ```
///
@inlinable @inline(__always)
public static func asin<T: Numeric & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Asin",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes inverse hyperbolic sine of x element-wise.
@inlinable @inline(__always)
public static func asinh<T: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Asinh",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Asserts that the given condition is true.
///
/// If `condition` evaluates to false, print the list of tensors in `data`.
/// `summarize` determines how many entries of the tensors to print.
///
/// - Parameters:
///   - condition: The condition to evaluate.
///   - data: The tensors to print out when condition is false.
///
/// - Attr summarize: Print this many entries of each tensor.
@inlinable @inline(__always)
public static func assert<T: TensorFlowScalar>(
  condition: Tensor<Bool>,
  data: [Tensor<T>],
  summarize: Int64 = 3
) {
  return #tfop("Assert",
    condition,
    data,
    summarize: summarize)
}

/// Computes the trignometric inverse tangent of x element-wise.
///
/// The `tf.math.atan` operation returns the inverse of `tf.math.tan`, such that
/// if `y = tf.math.tan(x)` then, `x = tf.math.atan(y)`.
///
/// **Note**: The output of `tf.math.atan` will lie within the invertible range 
/// of tan, i.e (-pi/2, pi/2).
///
/// For example:
///
/// ```python
/// # Note: [1.047, 0.785] ~= [(pi/3), (pi/4)]
/// x = tf.constant([1.047, 0.785])
/// y = tf.math.tan(x) # [1.731261, 0.99920404]
///
/// tf.math.atan(y) # [1.047, 0.785] = x
/// ```
///
@inlinable @inline(__always)
public static func atan<T: Numeric & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Atan",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes arctangent of `y/x` element-wise, respecting signs of the arguments.
///
/// This is the angle \( \theta \in [-\pi, \pi] \) such that
/// \[ x = r \cos(\theta) \]
/// and
/// \[ y = r \sin(\theta) \]
/// where \(r = \sqrt(x^2 + y^2) \).
@inlinable @inline(__always)
public static func atan2<T: FloatingPoint & TensorFlowScalar>(
  _ y: Tensor<T>,
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Atan2",
    y,
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes inverse hyperbolic tangent of x element-wise.
@inlinable @inline(__always)
public static func atanh<T: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Atanh",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func attr(
  _ a: Int64
) {
  return #tfop("Attr",
    a: a)
}

@inlinable @inline(__always)
public static func attrBool(
  _ a: Bool
) {
  return #tfop("AttrBool",
    a: a)
}

@inlinable @inline(__always)
public static func attrBoolList(
  _ a: [Bool]
) {
  return #tfop("AttrBoolList",
    a: a)
}

@inlinable @inline(__always)
public static func attrDefault(
  _ a: String = "banana"
) {
  return #tfop("AttrDefault",
    a: a)
}

@inlinable @inline(__always)
public static func attrEmptyListDefault(
  _ a: [Double]
) {
  return #tfop("AttrEmptyListDefault",
    a: a)
}

@inlinable @inline(__always)
public static func attrEnum(
  _ a: A
) {
  return #tfop("AttrEnum",
    a: a.cName)
}

@inlinable @inline(__always)
public static func attrEnumList(
  _ a: [String]
) {
  return #tfop("AttrEnumList",
    a: a)
}

@inlinable @inline(__always)
public static func attrFloat(
  _ a: Double
) {
  return #tfop("AttrFloat",
    a: a)
}

@inlinable @inline(__always)
public static func attrListDefault(
  _ a: [Int32] = [5, 15]
) {
  return #tfop("AttrListDefault",
    a: a)
}

@inlinable @inline(__always)
public static func attrListMin(
  _ a: [Int32]
) {
  return #tfop("AttrListMin",
    a: a)
}

@inlinable @inline(__always)
public static func attrListTypeDefault<T: TensorFlowScalar>(
  _ a: [Tensor<T>],
  _ b: [Tensor<T>]
) {
  return #tfop("AttrListTypeDefault",
    a,
    b,
    T$dtype: T.tensorFlowDataType)
}

@inlinable @inline(__always)
public static func attrMin(
  _ a: Int64
) {
  return #tfop("AttrMin",
    a: a)
}

@inlinable @inline(__always)
public static func attrTypeDefault<T: TensorFlowScalar>(
  _ a: Tensor<T>
) {
  return #tfop("AttrTypeDefault",
    a,
    T$dtype: T.tensorFlowDataType)
}

/// Produces a visualization of audio data over time.
///
/// Spectrograms are a standard way of representing audio information as a series of
/// slices of frequency information, one slice for each window of time. By joining
/// these together into a sequence, they form a distinctive fingerprint of the sound
/// over time.
///
/// This op expects to receive audio data as an input, stored as floats in the range
/// -1 to 1, together with a window width in samples, and a stride specifying how
/// far to move the window between slices. From this it generates a three
/// dimensional output. The lowest dimension has an amplitude value for each
/// frequency during that time slice. The next dimension is time, with successive
/// frequency slices. The final dimension is for the channels in the input, so a
/// stereo audio input would have two here for example.
///
/// This means the layout when converted and saved as an image is rotated 90 degrees
/// clockwise from a typical spectrogram. Time is descending down the Y axis, and
/// the frequency decreases from left to right.
///
/// Each value in the result represents the square root of the sum of the real and
/// imaginary parts of an FFT on the current window of samples. In this way, the
/// lowest dimension represents the power of each frequency in the current window,
/// and adjacent windows are concatenated in the next dimension.
///
/// To get a more intuitive and visual look at what this operation does, you can run
/// tensorflow/examples/wav_to_spectrogram to read in an audio file and save out the
/// resulting spectrogram as a PNG image.
///
/// - Parameter input: Float representation of audio data.
///
/// - Attrs:
///   - window_size: How wide the input window is in samples. For the highest efficiency
///     this should be a power of two, but other values are accepted.
///   - stride: How widely apart the center of adjacent sample windows should be.
///   - magnitude_squared: Whether to return the squared magnitude or just the
///     magnitude. Using squared magnitude can avoid extra calculations.
///
/// - Output spectrogram: 3D representation of the audio frequencies as an image.
@inlinable @inline(__always)
public static func audioSpectrogram(
  _ input: Tensor<Float>,
  windowSize: Int64,
  stride: Int64,
  magnitudeSquared: Bool = false
) -> Tensor<Float> {
  let ret: TensorHandle<Float> = #tfop("AudioSpectrogram",
    input,
    window_size: windowSize,
    stride: stride,
    magnitude_squared: magnitudeSquared)
  return Tensor(handle: ret)
}

/// Outputs a `Summary` protocol buffer with audio.
///
/// The summary has up to `max_outputs` summary values containing audio. The
/// audio is built from `tensor` which must be 3-D with shape `[batch_size,
/// frames, channels]` or 2-D with shape `[batch_size, frames]`. The values are
/// assumed to be in the range of `[-1.0, 1.0]` with a sample rate of `sample_rate`.
///
/// The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
/// build the `tag` of the summary values:
///
/// *  If `max_outputs` is 1, the summary value tag is '*tag*/audio'.
/// *  If `max_outputs` is greater than 1, the summary value tags are
///    generated sequentially as '*tag*/audio/0', '*tag*/audio/1', etc.
///
/// - Parameters:
///   - tag: Scalar. Used to build the `tag` attribute of the summary values.
///   - tensor: 2-D of shape `[batch_size, frames]`.
///
/// - Attrs:
///   - sample_rate: The sample rate of the signal in hertz.
///   - max_outputs: Max number of batch elements to generate audio for.
///
/// - Output summary: Scalar. Serialized `Summary` protocol buffer.
@inlinable @inline(__always)
public static func audioSummary(
  tag: StringTensor,
  _ tensor: Tensor<Float>,
  sampleRate: Double,
  maxOutputs: Int64 = 3
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("AudioSummary",
    tag,
    tensor,
    sample_rate: sampleRate,
    max_outputs: maxOutputs)
  return StringTensor(handle: ret)
}

/// Outputs a `Summary` protocol buffer with audio.
///
/// The summary has up to `max_outputs` summary values containing audio. The
/// audio is built from `tensor` which must be 3-D with shape `[batch_size,
/// frames, channels]` or 2-D with shape `[batch_size, frames]`. The values are
/// assumed to be in the range of `[-1.0, 1.0]` with a sample rate of `sample_rate`.
///
/// The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
/// build the `tag` of the summary values:
///
/// *  If `max_outputs` is 1, the summary value tag is '*tag*/audio'.
/// *  If `max_outputs` is greater than 1, the summary value tags are
///    generated sequentially as '*tag*/audio/0', '*tag*/audio/1', etc.
///
/// - Parameters:
///   - tag: Scalar. Used to build the `tag` attribute of the summary values.
///   - tensor: 2-D of shape `[batch_size, frames]`.
///   - sample_rate: The sample rate of the signal in hertz.
///
/// - Attr max_outputs: Max number of batch elements to generate audio for.
///
/// - Output summary: Scalar. Serialized `Summary` protocol buffer.
@inlinable @inline(__always)
public static func audioSummaryV2(
  tag: StringTensor,
  _ tensor: Tensor<Float>,
  sampleRate: Tensor<Float>,
  maxOutputs: Int64 = 3
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("AudioSummaryV2",
    tag,
    tensor,
    sampleRate,
    max_outputs: maxOutputs)
  return StringTensor(handle: ret)
}

/// Performs average pooling on the input.
///
/// Each entry in `output` is the mean of the corresponding size `ksize`
/// window in `value`.
///
/// - Parameter value: 4-D with shape `[batch, height, width, channels]`.
///
/// - Attrs:
///   - ksize: The size of the sliding window for each dimension of `value`.
///   - strides: The stride of the sliding window for each dimension of `value`.
///   - padding: The type of padding algorithm to use.
///   - data_format: Specify the data format of the input and output data. With the
///     default format "NHWC", the data is stored in the order of:
///         [batch, in_height, in_width, in_channels].
///     Alternatively, the format could be "NCHW", the data storage order of:
///         [batch, in_channels, in_height, in_width].
///
/// - Output output: The average pooled output tensor.
@inlinable @inline(__always)
public static func avgPool<T: FloatingPoint & TensorFlowScalar>(
  value: Tensor<T>,
  ksize: [Int32],
  strides: [Int32],
  padding: Padding,
  dataFormat: DataFormat = .nhwc
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("AvgPool",
    value,
    T$dtype: T.tensorFlowDataType,
    ksize: ksize,
    strides: strides,
    padding: padding.cName,
    data_format: dataFormat.cName)
  return Tensor(handle: ret)
}

/// Performs 3D average pooling on the input.
///
/// - Parameter input: Shape `[batch, depth, rows, cols, channels]` tensor to pool over.
///
/// - Attrs:
///   - ksize: 1-D tensor of length 5. The size of the window for each dimension of
///     the input tensor. Must have `ksize[0] = ksize[4] = 1`.
///   - strides: 1-D tensor of length 5. The stride of the sliding window for each
///     dimension of `input`. Must have `strides[0] = strides[4] = 1`.
///   - padding: The type of padding algorithm to use.
///   - data_format: The data format of the input and output data. With the
///     default format "NDHWC", the data is stored in the order of:
///         [batch, in_depth, in_height, in_width, in_channels].
///     Alternatively, the format could be "NCDHW", the data storage order is:
///         [batch, in_channels, in_depth, in_height, in_width].
///
/// - Output output: The average pooled output tensor.
@inlinable @inline(__always)
public static func avgPool3D<T: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>,
  ksize: [Int32],
  strides: [Int32],
  padding: Padding,
  dataFormat: DataFormat1 = .ndhwc
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("AvgPool3D",
    input,
    T$dtype: T.tensorFlowDataType,
    ksize: ksize,
    strides: strides,
    padding: padding.cName,
    data_format: dataFormat.cName)
  return Tensor(handle: ret)
}

/// Computes gradients of average pooling function.
///
/// - Parameters:
///   - orig_input_shape: The original input dimensions.
///   - grad: Output backprop of shape `[batch, depth, rows, cols, channels]`.
///
/// - Attrs:
///   - ksize: 1-D tensor of length 5. The size of the window for each dimension of
///     the input tensor. Must have `ksize[0] = ksize[4] = 1`.
///   - strides: 1-D tensor of length 5. The stride of the sliding window for each
///     dimension of `input`. Must have `strides[0] = strides[4] = 1`.
///   - padding: The type of padding algorithm to use.
///   - data_format: The data format of the input and output data. With the
///     default format "NDHWC", the data is stored in the order of:
///         [batch, in_depth, in_height, in_width, in_channels].
///     Alternatively, the format could be "NCDHW", the data storage order is:
///         [batch, in_channels, in_depth, in_height, in_width].
///
/// - Output output: The backprop for input.
@inlinable @inline(__always)
public static func avgPool3DGrad<T: FloatingPoint & TensorFlowScalar>(
  origInputShape: Tensor<Int32>,
  grad: Tensor<T>,
  ksize: [Int32],
  strides: [Int32],
  padding: Padding,
  dataFormat: DataFormat1 = .ndhwc
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("AvgPool3DGrad",
    origInputShape,
    grad,
    T$dtype: T.tensorFlowDataType,
    ksize: ksize,
    strides: strides,
    padding: padding.cName,
    data_format: dataFormat.cName)
  return Tensor(handle: ret)
}

/// Computes gradients of the average pooling function.
///
/// - Parameters:
///   - orig_input_shape: 1-D.  Shape of the original input to `avg_pool`.
///   - grad: 4-D with shape `[batch, height, width, channels]`.  Gradients w.r.t.
///     the output of `avg_pool`.
///
/// - Attrs:
///   - ksize: The size of the sliding window for each dimension of the input.
///   - strides: The stride of the sliding window for each dimension of the input.
///   - padding: The type of padding algorithm to use.
///   - data_format: Specify the data format of the input and output data. With the
///     default format "NHWC", the data is stored in the order of:
///         [batch, in_height, in_width, in_channels].
///     Alternatively, the format could be "NCHW", the data storage order of:
///         [batch, in_channels, in_height, in_width].
///
/// - Output output: 4-D.  Gradients w.r.t. the input of `avg_pool`.
@inlinable @inline(__always)
public static func avgPoolGrad<T: FloatingPoint & TensorFlowScalar>(
  origInputShape: Tensor<Int32>,
  grad: Tensor<T>,
  ksize: [Int32],
  strides: [Int32],
  padding: Padding,
  dataFormat: DataFormat = .nhwc
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("AvgPoolGrad",
    origInputShape,
    grad,
    T$dtype: T.tensorFlowDataType,
    ksize: ksize,
    strides: strides,
    padding: padding.cName,
    data_format: dataFormat.cName)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func b(
) -> Tensor<Float> {
  let ret: TensorHandle<Float> = #tfop("B")
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func batchCholesky<T: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("BatchCholesky",
    input,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func batchCholeskyGrad<T: FloatingPoint & TensorFlowScalar>(
  l: Tensor<T>,
  grad: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("BatchCholeskyGrad",
    l,
    grad,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Multiplies slices of two tensors in batches.
///
/// Multiplies all slices of `Tensor` `x` and `y` (each slice can be
/// viewed as an element of a batch), and arranges the individual results
/// in a single output tensor of the same batch size. Each of the
/// individual slices can optionally be adjointed (to adjoint a matrix
/// means to transpose and conjugate it) before multiplication by setting
/// the `adj_x` or `adj_y` flag to `True`, which are by default `False`.
///
/// The input tensors `x` and `y` are 2-D or higher with shape `[..., r_x, c_x]`
/// and `[..., r_y, c_y]`.
///
/// The output tensor is 2-D or higher with shape `[..., r_o, c_o]`, where:
///
///     r_o = c_x if adj_x else r_x
///     c_o = r_y if adj_y else c_y
///
/// It is computed as:
///
///     output[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :])
///
/// - Parameters:
///   - x: 2-D or higher with shape `[..., r_x, c_x]`.
///   - y: 2-D or higher with shape `[..., r_y, c_y]`.
///
/// - Attrs:
///   - adj_x: If `True`, adjoint the slices of `x`. Defaults to `False`.
///   - adj_y: If `True`, adjoint the slices of `y`. Defaults to `False`.
///
/// - Output output: 3-D or higher with shape `[..., r_o, c_o]`
@inlinable @inline(__always)
public static func batchMatMul<T: Numeric & TensorFlowScalar>(
  _ x: Tensor<T>,
  _ y: Tensor<T>,
  adjX: Bool = false,
  adjY: Bool = false
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("BatchMatMul",
    x,
    y,
    T$dtype: T.tensorFlowDataType,
    adj_x: adjX,
    adj_y: adjY)
  return Tensor(handle: ret)
}

/// Multiplies slices of two tensors in batches.
///
/// Multiplies all slices of `Tensor` `x` and `y` (each slice can be
/// viewed as an element of a batch), and arranges the individual results
/// in a single output tensor of the same batch size. Each of the
/// individual slices can optionally be adjointed (to adjoint a matrix
/// means to transpose and conjugate it) before multiplication by setting
/// the `adj_x` or `adj_y` flag to `True`, which are by default `False`.
///
/// The input tensors `x` and `y` are 2-D or higher with shape `[..., r_x, c_x]`
/// and `[..., r_y, c_y]`.
///
/// The output tensor is 2-D or higher with shape `[..., r_o, c_o]`, where:
///
///     r_o = c_x if adj_x else r_x
///     c_o = r_y if adj_y else c_y
///
/// It is computed as:
///
///     output[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :])
///
/// *NOTE*: `BatchMatMulV2` supports broadcasting in the batch dimensions. More
/// about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
///
///
/// - Parameters:
///   - x: 2-D or higher with shape `[..., r_x, c_x]`.
///   - y: 2-D or higher with shape `[..., r_y, c_y]`.
///
/// - Attrs:
///   - adj_x: If `True`, adjoint the slices of `x`. Defaults to `False`.
///   - adj_y: If `True`, adjoint the slices of `y`. Defaults to `False`.
///
/// - Output output: 3-D or higher with shape `[..., r_o, c_o]`
@inlinable @inline(__always)
public static func batchMatMulV2<T: Numeric & TensorFlowScalar>(
  _ x: Tensor<T>,
  _ y: Tensor<T>,
  adjX: Bool = false,
  adjY: Bool = false
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("BatchMatMulV2",
    x,
    y,
    T$dtype: T.tensorFlowDataType,
    adj_x: adjX,
    adj_y: adjY)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func batchMatrixBandPart<T: TensorFlowScalar>(
  _ input: Tensor<T>,
  numLower: Tensor<Int64>,
  numUpper: Tensor<Int64>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("BatchMatrixBandPart",
    input,
    numLower,
    numUpper,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func batchMatrixDeterminant<T: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("BatchMatrixDeterminant",
    input,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func batchMatrixDiag<T: TensorFlowScalar>(
  diagonal: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("BatchMatrixDiag",
    diagonal,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func batchMatrixDiagPart<T: TensorFlowScalar>(
  _ input: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("BatchMatrixDiagPart",
    input,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func batchMatrixInverse<T: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>,
  adjoint: Bool = false
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("BatchMatrixInverse",
    input,
    T$dtype: T.tensorFlowDataType,
    adjoint: adjoint)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func batchMatrixSetDiag<T: TensorFlowScalar>(
  _ input: Tensor<T>,
  diagonal: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("BatchMatrixSetDiag",
    input,
    diagonal,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func batchMatrixSolve<T: FloatingPoint & TensorFlowScalar>(
  matrix: Tensor<T>,
  rhs: Tensor<T>,
  adjoint: Bool = false
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("BatchMatrixSolve",
    matrix,
    rhs,
    T$dtype: T.tensorFlowDataType,
    adjoint: adjoint)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func batchMatrixSolveLs<T: FloatingPoint & TensorFlowScalar>(
  matrix: Tensor<T>,
  rhs: Tensor<T>,
  l2Regularizer: Tensor<Double>,
  fast: Bool = true
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("BatchMatrixSolveLs",
    matrix,
    rhs,
    l2Regularizer,
    T$dtype: T.tensorFlowDataType,
    fast: fast)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func batchMatrixTriangularSolve<T: FloatingPoint & TensorFlowScalar>(
  matrix: Tensor<T>,
  rhs: Tensor<T>,
  lower: Bool = true,
  adjoint: Bool = false
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("BatchMatrixTriangularSolve",
    matrix,
    rhs,
    T$dtype: T.tensorFlowDataType,
    lower: lower,
    adjoint: adjoint)
  return Tensor(handle: ret)
}

/// Batch normalization.
///
/// This op is deprecated. Prefer `tf.nn.batch_normalization`.
///
/// - Parameters:
///   - t: A 4D input Tensor.
///   - m: A 1D mean Tensor with size matching the last dimension of t.
///     This is the first output from tf.nn.moments,
///     or a saved moving average thereof.
///   - v: A 1D variance Tensor with size matching the last dimension of t.
///     This is the second output from tf.nn.moments,
///     or a saved moving average thereof.
///   - beta: A 1D beta Tensor with size matching the last dimension of t.
///     An offset to be added to the normalized tensor.
///   - gamma: A 1D gamma Tensor with size matching the last dimension of t.
///     If "scale_after_normalization" is true, this tensor will be multiplied
///     with the normalized tensor.
///
/// - Attrs:
///   - variance_epsilon: A small float number to avoid dividing by 0.
///   - scale_after_normalization: A bool indicating whether the resulted tensor
///     needs to be multiplied with gamma.
@inlinable @inline(__always)
public static func batchNormWithGlobalNormalization<T: Numeric & TensorFlowScalar>(
  t: Tensor<T>,
  m: Tensor<T>,
  v: Tensor<T>,
  beta: Tensor<T>,
  gamma: Tensor<T>,
  varianceEpsilon: Double,
  scaleAfterNormalization: Bool
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("BatchNormWithGlobalNormalization",
    t,
    m,
    v,
    beta,
    gamma,
    T$dtype: T.tensorFlowDataType,
    variance_epsilon: varianceEpsilon,
    scale_after_normalization: scaleAfterNormalization)
  return Tensor(handle: ret)
}

/// Gradients for batch normalization.
///
/// This op is deprecated. See `tf.nn.batch_normalization`.
///
/// - Parameters:
///   - t: A 4D input Tensor.
///   - m: A 1D mean Tensor with size matching the last dimension of t.
///     This is the first output from tf.nn.moments,
///     or a saved moving average thereof.
///   - v: A 1D variance Tensor with size matching the last dimension of t.
///     This is the second output from tf.nn.moments,
///     or a saved moving average thereof.
///   - gamma: A 1D gamma Tensor with size matching the last dimension of t.
///     If "scale_after_normalization" is true, this Tensor will be multiplied
///     with the normalized Tensor.
///   - backprop: 4D backprop Tensor.
///
/// - Attrs:
///   - variance_epsilon: A small float number to avoid dividing by 0.
///   - scale_after_normalization: A bool indicating whether the resulted tensor
///     needs to be multiplied with gamma.
///
/// - Outputs:
///   - dx: 4D backprop tensor for input.
///   - dm: 1D backprop tensor for mean.
///   - dv: 1D backprop tensor for variance.
///   - db: 1D backprop tensor for beta.
///   - dg: 1D backprop tensor for gamma.
@inlinable @inline(__always)
public static func batchNormWithGlobalNormalizationGrad<T: Numeric & TensorFlowScalar>(
  t: Tensor<T>,
  m: Tensor<T>,
  v: Tensor<T>,
  gamma: Tensor<T>,
  backprop: Tensor<T>,
  varianceEpsilon: Double,
  scaleAfterNormalization: Bool
) -> (dx: Tensor<T>, dm: Tensor<T>, dv: Tensor<T>, db: Tensor<T>, dg: Tensor<T>) {
  let ret: (TensorHandle<T>, TensorHandle<T>, TensorHandle<T>, TensorHandle<T>, TensorHandle<T>) = #tfop("BatchNormWithGlobalNormalizationGrad",
    t,
    m,
    v,
    gamma,
    backprop,
    T$dtype: T.tensorFlowDataType,
    variance_epsilon: varianceEpsilon,
    scale_after_normalization: scaleAfterNormalization)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2), Tensor(handle: ret.3), Tensor(handle: ret.4))
}

@inlinable @inline(__always)
public static func batchSelfAdjointEig<T: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("BatchSelfAdjointEig",
    input,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func batchSelfAdjointEigV2<T: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>,
  computeV: Bool = true
) -> (e: Tensor<T>, v: Tensor<T>) {
  let ret: (TensorHandle<T>, TensorHandle<T>) = #tfop("BatchSelfAdjointEigV2",
    input,
    T$dtype: T.tensorFlowDataType,
    compute_v: computeV)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

@inlinable @inline(__always)
public static func batchSvd<T: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>,
  computeUv: Bool = true,
  fullMatrices: Bool = false
) -> (s: Tensor<T>, u: Tensor<T>, v: Tensor<T>) {
  let ret: (TensorHandle<T>, TensorHandle<T>, TensorHandle<T>) = #tfop("BatchSvd",
    input,
    T$dtype: T.tensorFlowDataType,
    compute_uv: computeUv,
    full_matrices: fullMatrices)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// BatchToSpace for 4-D tensors of type T.
///
/// This is a legacy version of the more general BatchToSpaceND.
///
/// Rearranges (permutes) data from batch into blocks of spatial data, followed by
/// cropping. This is the reverse transformation of SpaceToBatch. More specifically,
/// this op outputs a copy of the input tensor where values from the `batch`
/// dimension are moved in spatial blocks to the `height` and `width` dimensions,
/// followed by cropping along the `height` and `width` dimensions.
///
/// - Parameters:
///   - input: 4-D tensor with shape
///     `[batch*block_size*block_size, height_pad/block_size, width_pad/block_size,
///       depth]`. Note that the batch size of the input tensor must be divisible by
///     `block_size * block_size`.
///   - crops: 2-D tensor of non-negative integers with shape `[2, 2]`. It specifies
///     how many elements to crop from the intermediate result across the spatial
///     dimensions as follows:
///
///         crops = [[crop_top, crop_bottom], [crop_left, crop_right]]
///
/// - Output output: 4-D with shape `[batch, height, width, depth]`, where:
///
///         height = height_pad - crop_top - crop_bottom
///         width = width_pad - crop_left - crop_right
///
///   The attr `block_size` must be greater than one. It indicates the block size.
///
///   Some examples:
///
///   (1) For the following input of shape `[4, 1, 1, 1]` and block_size of 2:
///
///   ```
///   [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
///   ```
///
///   The output tensor has shape `[1, 2, 2, 1]` and value:
///
///   ```
///   x = [[[[1], [2]], [[3], [4]]]]
///   ```
///
///   (2) For the following input of shape `[4, 1, 1, 3]` and block_size of 2:
///
///   ```
///   [[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]]
///   ```
///
///   The output tensor has shape `[1, 2, 2, 3]` and value:
///
///   ```
///   x = [[[[1, 2, 3], [4, 5, 6]],
///         [[7, 8, 9], [10, 11, 12]]]]
///   ```
///
///   (3) For the following input of shape `[4, 2, 2, 1]` and block_size of 2:
///
///   ```
///   x = [[[[1], [3]], [[9], [11]]],
///        [[[2], [4]], [[10], [12]]],
///        [[[5], [7]], [[13], [15]]],
///        [[[6], [8]], [[14], [16]]]]
///   ```
///
///   The output tensor has shape `[1, 4, 4, 1]` and value:
///
///   ```
///   x = [[[1],   [2],  [3],  [4]],
///        [[5],   [6],  [7],  [8]],
///        [[9],  [10], [11],  [12]],
///        [[13], [14], [15],  [16]]]
///   ```
///
///   (4) For the following input of shape `[8, 1, 2, 1]` and block_size of 2:
///
///   ```
///   x = [[[[1], [3]]], [[[9], [11]]], [[[2], [4]]], [[[10], [12]]],
///        [[[5], [7]]], [[[13], [15]]], [[[6], [8]]], [[[14], [16]]]]
///   ```
///
///   The output tensor has shape `[2, 2, 4, 1]` and value:
///
///   ```
///   x = [[[[1], [3]], [[5], [7]]],
///        [[[2], [4]], [[10], [12]]],
///        [[[5], [7]], [[13], [15]]],
///        [[[6], [8]], [[14], [16]]]]
///   ```
@inlinable @inline(__always)
public static func batchToSpace<T: TensorFlowScalar, Tidx: BinaryInteger & TensorFlowScalar>(
  _ input: Tensor<T>,
  crops: Tensor<Tidx>,
  blockSize: Int64
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("BatchToSpace",
    input,
    crops,
    T$dtype: T.tensorFlowDataType,
    Tidx$dtype: Tidx.tensorFlowDataType,
    block_size: blockSize)
  return Tensor(handle: ret)
}

/// BatchToSpace for N-D tensors of type T.
///
/// This operation reshapes the "batch" dimension 0 into `M + 1` dimensions of shape
/// `block_shape + [batch]`, interleaves these blocks back into the grid defined by
/// the spatial dimensions `[1, ..., M]`, to obtain a result with the same rank as
/// the input.  The spatial dimensions of this intermediate result are then
/// optionally cropped according to `crops` to produce the output.  This is the
/// reverse of SpaceToBatch.  See below for a precise description.
///
/// - Parameters:
///   - input: N-D with shape `input_shape = [batch] + spatial_shape + remaining_shape`,
///     where spatial_shape has M dimensions.
///   - block_shape: 1-D with shape `[M]`, all values must be >= 1.
///   - crops: 2-D with shape `[M, 2]`, all values must be >= 0.
///       `crops[i] = [crop_start, crop_end]` specifies the amount to crop from input
///       dimension `i + 1`, which corresponds to spatial dimension `i`.  It is
///       required that
///       `crop_start[i] + crop_end[i] <= block_shape[i] * input_shape[i + 1]`.
///
///     This operation is equivalent to the following steps:
///
///     1. Reshape `input` to `reshaped` of shape:
///          [block_shape[0], ..., block_shape[M-1],
///           batch / prod(block_shape),
///           input_shape[1], ..., input_shape[N-1]]
///
///     2. Permute dimensions of `reshaped` to produce `permuted` of shape
///          [batch / prod(block_shape),
///
///           input_shape[1], block_shape[0],
///           ...,
///           input_shape[M], block_shape[M-1],
///
///           input_shape[M+1], ..., input_shape[N-1]]
///
///     3. Reshape `permuted` to produce `reshaped_permuted` of shape
///          [batch / prod(block_shape),
///
///           input_shape[1] * block_shape[0],
///           ...,
///           input_shape[M] * block_shape[M-1],
///
///           input_shape[M+1],
///           ...,
///           input_shape[N-1]]
///
///     4. Crop the start and end of dimensions `[1, ..., M]` of
///        `reshaped_permuted` according to `crops` to produce the output of shape:
///          [batch / prod(block_shape),
///
///           input_shape[1] * block_shape[0] - crops[0,0] - crops[0,1],
///           ...,
///           input_shape[M] * block_shape[M-1] - crops[M-1,0] - crops[M-1,1],
///
///           input_shape[M+1], ..., input_shape[N-1]]
///
///     Some examples:
///
///     (1) For the following input of shape `[4, 1, 1, 1]`, `block_shape = [2, 2]`, and
///         `crops = [[0, 0], [0, 0]]`:
///
///     ```
///     [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
///     ```
///
///     The output tensor has shape `[1, 2, 2, 1]` and value:
///
///     ```
///     x = [[[[1], [2]], [[3], [4]]]]
///     ```
///
///     (2) For the following input of shape `[4, 1, 1, 3]`, `block_shape = [2, 2]`, and
///         `crops = [[0, 0], [0, 0]]`:
///
///     ```
///     [[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]]
///     ```
///
///     The output tensor has shape `[1, 2, 2, 3]` and value:
///
///     ```
///     x = [[[[1, 2, 3], [4, 5, 6]],
///           [[7, 8, 9], [10, 11, 12]]]]
///     ```
///
///     (3) For the following input of shape `[4, 2, 2, 1]`, `block_shape = [2, 2]`, and
///         `crops = [[0, 0], [0, 0]]`:
///
///     ```
///     x = [[[[1], [3]], [[9], [11]]],
///          [[[2], [4]], [[10], [12]]],
///          [[[5], [7]], [[13], [15]]],
///          [[[6], [8]], [[14], [16]]]]
///     ```
///
///     The output tensor has shape `[1, 4, 4, 1]` and value:
///
///     ```
///     x = [[[1],   [2],  [3],  [4]],
///          [[5],   [6],  [7],  [8]],
///          [[9],  [10], [11],  [12]],
///          [[13], [14], [15],  [16]]]
///     ```
///
///     (4) For the following input of shape `[8, 1, 3, 1]`, `block_shape = [2, 2]`, and
///         `crops = [[0, 0], [2, 0]]`:
///
///     ```
///     x = [[[[0], [1], [3]]], [[[0], [9], [11]]],
///          [[[0], [2], [4]]], [[[0], [10], [12]]],
///          [[[0], [5], [7]]], [[[0], [13], [15]]],
///          [[[0], [6], [8]]], [[[0], [14], [16]]]]
///     ```
///
///     The output tensor has shape `[2, 2, 4, 1]` and value:
///
///     ```
///     x = [[[[1],   [2],  [3],  [4]],
///           [[5],   [6],  [7],  [8]]],
///          [[[9],  [10], [11],  [12]],
///           [[13], [14], [15],  [16]]]]
///     ```
@inlinable @inline(__always)
public static func batchToSpaceND<T: TensorFlowScalar, TblockShape: BinaryInteger & TensorFlowScalar, Tcrops: BinaryInteger & TensorFlowScalar>(
  _ input: Tensor<T>,
  blockShape: Tensor<TblockShape>,
  crops: Tensor<Tcrops>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("BatchToSpaceND",
    input,
    blockShape,
    crops,
    T$dtype: T.tensorFlowDataType,
    Tblock_shape$dtype: TblockShape.tensorFlowDataType,
    Tcrops$dtype: Tcrops.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the Bessel i0e function of `x` element-wise.
///
/// Exponentially scaled modified Bessel function of order 0 defined as
/// `bessel_i0e(x) = exp(-abs(x)) bessel_i0(x)`.
///
/// This function is faster and numerically stabler than `bessel_i0(x)`.
@inlinable @inline(__always)
public static func besselI0e<T: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("BesselI0e",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the Bessel i1e function of `x` element-wise.
///
/// Exponentially scaled modified Bessel function of order 0 defined as
/// `bessel_i1e(x) = exp(-abs(x)) bessel_i1(x)`.
///
/// This function is faster and numerically stabler than `bessel_i1(x)`.
@inlinable @inline(__always)
public static func besselI1e<T: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("BesselI1e",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Compute the regularized incomplete beta integral \\(I_x(a, b)\\).
///
/// The regularized incomplete beta integral is defined as:
///
///
/// \\(I_x(a, b) = \frac{B(x; a, b)}{B(a, b)}\\)
///
/// where
///
///
/// \\(B(x; a, b) = \int_0^x t^{a-1} (1 - t)^{b-1} dt\\)
///
///
/// is the incomplete beta function and \\(B(a, b)\\) is the *complete*
/// beta function.
@inlinable @inline(__always)
public static func betainc<T: FloatingPoint & TensorFlowScalar>(
  _ a: Tensor<T>,
  _ b: Tensor<T>,
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Betainc",
    a,
    b,
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Adds `bias` to `value`.
///
/// This is a special case of `tf.add` where `bias` is restricted to be 1-D.
/// Broadcasting is supported, so `value` may have any number of dimensions.
///
/// - Parameters:
///   - value: Any number of dimensions.
///   - bias: 1-D with size the last dimension of `value`.
///
/// - Attr data_format: Specify the data format of the input and output data. With the
///   default format "NHWC", the bias tensor will be added to the last dimension
///   of the value tensor.
///   Alternatively, the format could be "NCHW", the data storage order of:
///       [batch, in_channels, in_height, in_width].
///   The tensor will be added to "in_channels", the third-to-the-last
///       dimension.
///
/// - Output output: Broadcasted sum of `value` and `bias`.
@inlinable @inline(__always)
public static func biasAdd<T: Numeric & TensorFlowScalar>(
  value: Tensor<T>,
  bias: Tensor<T>,
  dataFormat: DataFormat = .nhwc
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("BiasAdd",
    value,
    bias,
    T$dtype: T.tensorFlowDataType,
    data_format: dataFormat.cName)
  return Tensor(handle: ret)
}

/// The backward operation for "BiasAdd" on the "bias" tensor.
///
/// It accumulates all the values from out_backprop into the feature dimension.
/// For NHWC data format, the feature dimension is the last. For NCHW data format,
/// the feature dimension is the third-to-last.
///
/// - Parameter out_backprop: Any number of dimensions.
///
/// - Attr data_format: Specify the data format of the input and output data. With the
///   default format "NHWC", the bias tensor will be added to the last dimension
///   of the value tensor.
///   Alternatively, the format could be "NCHW", the data storage order of:
///       [batch, in_channels, in_height, in_width].
///   The tensor will be added to "in_channels", the third-to-the-last
///       dimension.
///
/// - Output output: 1-D with size the feature dimension of `out_backprop`.
@inlinable @inline(__always)
public static func biasAddGrad<T: Numeric & TensorFlowScalar>(
  outBackprop: Tensor<T>,
  dataFormat: DataFormat = .nhwc
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("BiasAddGrad",
    outBackprop,
    T$dtype: T.tensorFlowDataType,
    data_format: dataFormat.cName)
  return Tensor(handle: ret)
}

/// Adds `bias` to `value`.
///
/// This is a deprecated version of BiasAdd and will be soon removed.
///
/// This is a special case of `tf.add` where `bias` is restricted to be 1-D.
/// Broadcasting is supported, so `value` may have any number of dimensions.
///
/// - Parameters:
///   - value: Any number of dimensions.
///   - bias: 1-D with size the last dimension of `value`.
///
/// - Output output: Broadcasted sum of `value` and `bias`.
@inlinable @inline(__always)
public static func biasAddV1<T: Numeric & TensorFlowScalar>(
  value: Tensor<T>,
  bias: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("BiasAddV1",
    value,
    bias,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func binary<T: TensorFlowScalar>(
  _ a: Tensor<T>,
  _ b: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Binary",
    a,
    b,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Counts the number of occurrences of each value in an integer array.
///
/// Outputs a vector with length `size` and the same dtype as `weights`. If
/// `weights` are empty, then index `i` stores the number of times the value `i` is
/// counted in `arr`. If `weights` are non-empty, then index `i` stores the sum of
/// the value in `weights` at each index where the corresponding value in `arr` is
/// `i`.
///
/// Values in `arr` outside of the range [0, size) are ignored.
///
/// - Parameters:
///   - arr: int32 `Tensor`.
///   - size: non-negative int32 scalar `Tensor`.
///   - weights: is an int32, int64, float32, or float64 `Tensor` with the same
///     shape as `arr`, or a length-0 `Tensor`, in which case it acts as all weights
///     equal to 1.
///
/// - Output bins: 1D `Tensor` with length equal to `size`. The counts or summed weights for
///   each value in the range [0, size).
@inlinable @inline(__always)
public static func bincount<T: Numeric & TensorFlowScalar>(
  arr: Tensor<Int32>,
  size: Tensor<Int32>,
  weights: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Bincount",
    arr,
    size,
    weights,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Bitcasts a tensor from one type to another without copying data.
///
/// Given a tensor `input`, this operation returns a tensor that has the same buffer
/// data as `input` with datatype `type`.
///
/// If the input datatype `T` is larger than the output datatype `type` then the
/// shape changes from [...] to [..., sizeof(`T`)/sizeof(`type`)].
///
/// If `T` is smaller than `type`, the operator requires that the rightmost
/// dimension be equal to sizeof(`type`)/sizeof(`T`). The shape then goes from
/// [..., sizeof(`type`)/sizeof(`T`)] to [...].
///
/// *NOTE*: Bitcast is implemented as a low-level cast, so machines with different
/// endian orderings will give different results.
@inlinable @inline(__always)
public static func bitcast<T: Numeric & TensorFlowScalar, Type: Numeric & TensorFlowScalar>(
  _ input: Tensor<T>
) -> Tensor<Type> {
  let ret: TensorHandle<Type> = #tfop("Bitcast",
    input,
    T$dtype: T.tensorFlowDataType,
    type$dtype: Type.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Elementwise computes the bitwise AND of `x` and `y`.
///
/// The result will have those bits set, that are set in both `x` and `y`. The
/// computation is performed on the underlying representations of `x` and `y`.
@inlinable @inline(__always)
public static func bitwiseAnd<T: BinaryInteger & TensorFlowScalar>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("BitwiseAnd",
    x,
    y,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Elementwise computes the bitwise OR of `x` and `y`.
///
/// The result will have those bits set, that are set in `x`, `y` or both. The
/// computation is performed on the underlying representations of `x` and `y`.
@inlinable @inline(__always)
public static func bitwiseOr<T: BinaryInteger & TensorFlowScalar>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("BitwiseOr",
    x,
    y,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Elementwise computes the bitwise XOR of `x` and `y`.
///
/// The result will have those bits set, that are different in `x` and `y`. The
/// computation is performed on the underlying representations of `x` and `y`.
@inlinable @inline(__always)
public static func bitwiseXor<T: BinaryInteger & TensorFlowScalar>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("BitwiseXor",
    x,
    y,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the LSTM cell forward propagation for all the time steps.
///
/// This is equivalent to applying LSTMBlockCell in a loop, like so:
///
/// ```python
/// for x1 in unpack(x):
///   i1, cs1, f1, o1, ci1, co1, h1 = LSTMBlock(
///     x1, cs_prev, h_prev, w, wci, wcf, wco, b)
///   cs_prev = cs1
///   h_prev = h1
///   i.append(i1)
///   cs.append(cs1)
///   f.append(f1)
///   o.append(o1)
///   ci.append(ci1)
///   co.append(co1)
///   h.append(h1)
/// return pack(i), pack(cs), pack(f), pack(o), pack(ci), pack(ch), pack(h)
/// ```
///
/// - Parameters:
///   - seq_len_max: Maximum time length actually used by this input. Outputs are padded
///     with zeros beyond this length.
///   - x: The sequence input to the LSTM, shape (timelen, batch_size, num_inputs).
///   - cs_prev: Value of the initial cell state.
///   - h_prev: Initial output of cell (to be used for peephole).
///   - w: The weight matrix.
///   - wci: The weight matrix for input gate peephole connection.
///   - wcf: The weight matrix for forget gate peephole connection.
///   - wco: The weight matrix for output gate peephole connection.
///   - b: The bias vector.
///
/// - Attrs:
///   - forget_bias: The forget gate bias.
///   - cell_clip: Value to clip the 'cs' value to.
///   - use_peephole: Whether to use peephole weights.
///
/// - Outputs:
///   - i: The input gate over the whole time sequence.
///   - cs: The cell state before the tanh over the whole time sequence.
///   - f: The forget gate over the whole time sequence.
///   - o: The output gate over the whole time sequence.
///   - ci: The cell input over the whole time sequence.
///   - co: The cell after the tanh over the whole time sequence.
///   - h: The output h vector over the whole time sequence.
@inlinable @inline(__always)
public static func blockLSTM<T: FloatingPoint & TensorFlowScalar>(
  seqLenMax: Tensor<Int64>,
  _ x: Tensor<T>,
  csPrev: Tensor<T>,
  hPrev: Tensor<T>,
  w: Tensor<T>,
  wci: Tensor<T>,
  wcf: Tensor<T>,
  wco: Tensor<T>,
  _ b: Tensor<T>,
  forgetBias: Double = 1,
  cellClip: Double = 3,
  usePeephole: Bool = false
) -> (i: Tensor<T>, cs: Tensor<T>, f: Tensor<T>, o: Tensor<T>, ci: Tensor<T>, co: Tensor<T>, h: Tensor<T>) {
  let ret: (TensorHandle<T>, TensorHandle<T>, TensorHandle<T>, TensorHandle<T>, TensorHandle<T>, TensorHandle<T>, TensorHandle<T>) = #tfop("BlockLSTM",
    seqLenMax,
    x,
    csPrev,
    hPrev,
    w,
    wci,
    wcf,
    wco,
    b,
    T$dtype: T.tensorFlowDataType,
    forget_bias: forgetBias,
    cell_clip: cellClip,
    use_peephole: usePeephole)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2), Tensor(handle: ret.3), Tensor(handle: ret.4), Tensor(handle: ret.5), Tensor(handle: ret.6))
}

/// Computes the LSTM cell backward propagation for the entire time sequence.
///
/// This implementation is to be used in conjunction of LSTMBlock.
///
/// - Parameters:
///   - seq_len_max: Maximum time length actually used by this input. Outputs are padded
///     with zeros beyond this length.
///   - x: The sequence input to the LSTM, shape (timelen, batch_size, num_inputs).
///   - cs_prev: Value of the initial cell state.
///   - h_prev: Initial output of cell (to be used for peephole).
///   - w: The weight matrix.
///   - wci: The weight matrix for input gate peephole connection.
///   - wcf: The weight matrix for forget gate peephole connection.
///   - wco: The weight matrix for output gate peephole connection.
///   - b: The bias vector.
///   - i: The input gate over the whole time sequence.
///   - cs: The cell state before the tanh over the whole time sequence.
///   - f: The forget gate over the whole time sequence.
///   - o: The output gate over the whole time sequence.
///   - ci: The cell input over the whole time sequence.
///   - co: The cell after the tanh over the whole time sequence.
///   - h: The output h vector over the whole time sequence.
///   - cs_grad: The current gradient of cs.
///   - h_grad: The gradient of h vector.
///
/// - Attr use_peephole: Whether to use peephole weights.
///
/// - Outputs:
///   - x_grad: The gradient of x to be back-propped.
///   - cs_prev_grad: The gradient of cs_prev to be back-propped.
///   - h_prev_grad: The gradient of h_prev to be back-propped.
///   - w_grad: The gradient for w to be back-propped.
///   - wci_grad: The gradient for wci to be back-propped.
///   - wcf_grad: The gradient for wcf to be back-propped.
///   - wco_grad: The gradient for wco to be back-propped.
///   - b_grad: The gradient for w to be back-propped.
@inlinable @inline(__always)
public static func blockLSTMGrad<T: FloatingPoint & TensorFlowScalar>(
  seqLenMax: Tensor<Int64>,
  _ x: Tensor<T>,
  csPrev: Tensor<T>,
  hPrev: Tensor<T>,
  w: Tensor<T>,
  wci: Tensor<T>,
  wcf: Tensor<T>,
  wco: Tensor<T>,
  _ b: Tensor<T>,
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
) -> (xGrad: Tensor<T>, csPrevGrad: Tensor<T>, hPrevGrad: Tensor<T>, wGrad: Tensor<T>, wciGrad: Tensor<T>, wcfGrad: Tensor<T>, wcoGrad: Tensor<T>, bGrad: Tensor<T>) {
  let ret: (TensorHandle<T>, TensorHandle<T>, TensorHandle<T>, TensorHandle<T>, TensorHandle<T>, TensorHandle<T>, TensorHandle<T>, TensorHandle<T>) = #tfop("BlockLSTMGrad",
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
    T$dtype: T.tensorFlowDataType,
    use_peephole: usePeephole)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2), Tensor(handle: ret.3), Tensor(handle: ret.4), Tensor(handle: ret.5), Tensor(handle: ret.6), Tensor(handle: ret.7))
}

/// Makes the summary of accumulated stats for the batch.
///
/// The summary stats contains gradients and hessians accumulated into the corresponding node and bucket for each example.
///
/// - Parameters:
///   - node_ids: int32 Rank 1 Tensor containing node ids, which each example falls into for the requested layer.
///   - gradients: float32; Rank 2 Tensor (shape=[#examples, 1]) for gradients.
///   - hessians: float32; Rank 2 Tensor (shape=[#examples, 1]) for hessians.
///   - bucketized_features_list: int32 list of Rank 1 Tensors, each containing the bucketized feature (for each feature column).
///
/// - Attrs:
///   - max_splits: int; the maximum number of splits possible in the whole tree.
///   - num_buckets: int; equals to the maximum possible value of bucketized feature.
///   - num_features: int; inferred from the size of bucketized_features_list; the number of features.
///
/// - Output stats_summary: output Rank 4 Tensor (shape=[#features, #splits, #buckets, 2]) containing accumulated stats put into the corresponding node and bucket. The first index of 4th dimension refers to gradients, and the second to hessians.
@inlinable @inline(__always)
public static func boostedTreesMakeStatsSummary(
  nodeIds: Tensor<Int32>,
  gradients: Tensor<Float>,
  hessians: Tensor<Float>,
  bucketizedFeaturesList: [Tensor<Int32>],
  maxSplits: Int64,
  numBuckets: Int64
) -> Tensor<Float> {
  let ret: TensorHandle<Float> = #tfop("BoostedTreesMakeStatsSummary",
    nodeIds,
    gradients,
    hessians,
    bucketizedFeaturesList,
    max_splits: maxSplits,
    num_buckets: numBuckets)
  return Tensor(handle: ret)
}

/// Return the shape of s0 op s1 with broadcast.
///
/// Given `s0` and `s1`, tensors that represent shapes, compute `r0`, the
/// broadcasted shape. `s0`, `s1` and `r0` are all integer vectors.
@inlinable @inline(__always)
public static func broadcastArgs<T: BinaryInteger & TensorFlowScalar>(
  s0: Tensor<T>,
  s1: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("BroadcastArgs",
    s0,
    s1,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Return the reduction indices for computing gradients of s0 op s1 with broadcast.
///
/// This is typically used by gradient computations for a broadcasting operation.
@inlinable @inline(__always)
public static func broadcastGradientArgs<T: BinaryInteger & TensorFlowScalar>(
  s0: Tensor<T>,
  s1: Tensor<T>
) -> (r0: Tensor<T>, r1: Tensor<T>) {
  let ret: (TensorHandle<T>, TensorHandle<T>) = #tfop("BroadcastGradientArgs",
    s0,
    s1,
    T$dtype: T.tensorFlowDataType)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

/// Broadcast an array for a compatible shape.
///
/// Broadcasting is the process of making arrays to have compatible shapes
/// for arithmetic operations. Two shapes are compatible if for each
/// dimension pair they are either equal or one of them is one. When trying
/// to broadcast a Tensor to a shape, it starts with the trailing dimensions,
/// and works its way forward.
///
/// For example,
/// ```
/// >>> x = tf.constant([1, 2, 3])
/// >>> y = tf.broadcast_to(x, [3, 3])
/// >>> sess.run(y)
/// array([[1, 2, 3],
///        [1, 2, 3],
///        [1, 2, 3]], dtype=int32)
/// ```
/// In the above example, the input Tensor with the shape of `[1, 3]`
/// is broadcasted to output Tensor with shape of `[3, 3]`.
///
/// - Parameters:
///   - input: A Tensor to broadcast.
///   - shape: An 1-D `int` Tensor. The shape of the desired output.
///
/// - Output output: A Tensor.
@inlinable @inline(__always)
public static func broadcastTo<T: TensorFlowScalar, Tidx: BinaryInteger & TensorFlowScalar>(
  _ input: Tensor<T>,
  shape: Tensor<Tidx>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("BroadcastTo",
    input,
    shape,
    T$dtype: T.tensorFlowDataType,
    Tidx$dtype: Tidx.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Bucketizes 'input' based on 'boundaries'.
///
/// For example, if the inputs are
///     boundaries = [0, 10, 100]
///     input = [[-5, 10000]
///              [150,   10]
///              [5,    100]]
///
/// then the output will be
///     output = [[0, 3]
///               [3, 2]
///               [1, 3]]
///
/// - Parameter input: Any shape of Tensor contains with int or float type.
///
/// - Attr boundaries: A sorted list of floats gives the boundary of the buckets.
///
/// - Output output: Same shape with 'input', each value of input replaced with bucket index.
///
///   @compatibility(numpy)
///   Equivalent to np.digitize.
///   @end_compatibility
@inlinable @inline(__always)
public static func bucketize<T: Numeric & TensorFlowScalar>(
  _ input: Tensor<T>,
  boundaries: [Double]
) -> Tensor<Int32> {
  let ret: TensorHandle<Int32> = #tfop("Bucketize",
    input,
    T$dtype: T.tensorFlowDataType,
    boundaries: boundaries)
  return Tensor(handle: ret)
}

/// Performs greedy decoding on the logits given in inputs.
///
/// A note about the attribute merge_repeated: if enabled, when
/// consecutive logits' maximum indices are the same, only the first of
/// these is emitted.  Labeling the blank '*', the sequence "A B B * B B"
/// becomes "A B B" if merge_repeated = True and "A B B B B" if
/// merge_repeated = False.
///
/// Regardless of the value of merge_repeated, if the maximum index of a given
/// time and batch corresponds to the blank, index `(num_classes - 1)`, no new
/// element is emitted.
///
/// - Parameters:
///   - inputs: 3-D, shape: `(max_time x batch_size x num_classes)`, the logits.
///   - sequence_length: A vector containing sequence lengths, size `(batch_size)`.
///
/// - Attr merge_repeated: If True, merge repeated classes in output.
///
/// - Outputs:
///   - decoded_indices: Indices matrix, size `(total_decoded_outputs x 2)`,
///     of a `SparseTensor<int64, 2>`.  The rows store: [batch, time].
///   - decoded_values: Values vector, size: `(total_decoded_outputs)`,
///     of a `SparseTensor<int64, 2>`.  The vector stores the decoded classes.
///   - decoded_shape: Shape vector, size `(2)`, of the decoded SparseTensor.
///     Values are: `[batch_size, max_decoded_length]`.
///   - log_probability: Matrix, size `(batch_size x 1)`, containing sequence
///     log-probabilities.
@inlinable @inline(__always)
public static func cTCGreedyDecoder(
  inputs: Tensor<Float>,
  sequenceLength: Tensor<Int32>,
  mergeRepeated: Bool = false
) -> (decodedIndices: Tensor<Int64>, decodedValues: Tensor<Int64>, decodedShape: Tensor<Int64>, logProbability: Tensor<Float>) {
  let ret: (TensorHandle<Int64>, TensorHandle<Int64>, TensorHandle<Int64>, TensorHandle<Float>) = #tfop("CTCGreedyDecoder",
    inputs,
    sequenceLength,
    merge_repeated: mergeRepeated)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2), Tensor(handle: ret.3))
}

/// Calculates the CTC Loss (log probability) for each batch entry.  Also calculates
///
/// the gradient.  This class performs the softmax operation for you, so inputs
/// should be e.g. linear projections of outputs by an LSTM.
///
/// - Parameters:
///   - inputs: 3-D, shape: `(max_time x batch_size x num_classes)`, the logits.
///   - labels_indices: The indices of a `SparseTensor<int32, 2>`.
///     `labels_indices(i, :) == [b, t]` means `labels_values(i)` stores the id for
///     `(batch b, time t)`.
///   - labels_values: The values (labels) associated with the given batch and time.
///   - sequence_length: A vector containing sequence lengths (batch).
///
/// - Attrs:
///   - preprocess_collapse_repeated: Scalar, if true then repeated labels are
///     collapsed prior to the CTC calculation.
///   - ctc_merge_repeated: Scalar.  If set to false, *during* CTC calculation
///     repeated non-blank labels will not be merged and are interpreted as
///     individual labels.  This is a simplified version of CTC.
///   - ignore_longer_outputs_than_inputs: Scalar. If set to true, during CTC
///     calculation, items that have longer output sequences than input sequences
///     are skipped: they don't contribute to the loss term and have zero-gradient.
///
/// - Outputs:
///   - loss: A vector (batch) containing log-probabilities.
///   - gradient: The gradient of `loss`.  3-D, shape:
///     `(max_time x batch_size x num_classes)`.
@inlinable @inline(__always)
public static func cTCLoss(
  inputs: Tensor<Float>,
  labelsIndices: Tensor<Int64>,
  labelsValues: Tensor<Int32>,
  sequenceLength: Tensor<Int32>,
  preprocessCollapseRepeated: Bool = false,
  ctcMergeRepeated: Bool = true,
  ignoreLongerOutputsThanInputs: Bool = false
) -> (loss: Tensor<Float>, gradient: Tensor<Float>) {
  let ret: (TensorHandle<Float>, TensorHandle<Float>) = #tfop("CTCLoss",
    inputs,
    labelsIndices,
    labelsValues,
    sequenceLength,
    preprocess_collapse_repeated: preprocessCollapseRepeated,
    ctc_merge_repeated: ctcMergeRepeated,
    ignore_longer_outputs_than_inputs: ignoreLongerOutputsThanInputs)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

/// Cast x of type SrcT to y of DstT.
@inlinable @inline(__always)
public static func cast<Srct: TensorFlowScalar, Dstt: TensorFlowScalar>(
  _ x: Tensor<Srct>,
  truncate: Bool = false
) -> Tensor<Dstt> {
  let ret: TensorHandle<Dstt> = #tfop("Cast",
    x,
    SrcT$dtype: Srct.tensorFlowDataType,
    DstT$dtype: Dstt.tensorFlowDataType,
    Truncate: truncate)
  return Tensor(handle: ret)
}

/// Returns element-wise smallest integer not less than x.
@inlinable @inline(__always)
public static func ceil<T: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Ceil",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Checks a tensor for NaN and Inf values.
///
/// When run, reports an `InvalidArgument` error if `tensor` has any values
/// that are not a number (NaN) or infinity (Inf). Otherwise, passes `tensor` as-is.
///
/// - Attr message: Prefix of the error message.
@inlinable @inline(__always)
public static func checkNumerics<T: FloatingPoint & TensorFlowScalar>(
  _ tensor: Tensor<T>,
  message: String
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("CheckNumerics",
    tensor,
    T$dtype: T.tensorFlowDataType,
    message: message)
  return Tensor(handle: ret)
}

/// Computes the Cholesky decomposition of one or more square matrices.
///
/// The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
/// form square matrices.
///
/// The input has to be symmetric and positive definite. Only the lower-triangular
/// part of the input will be used for this operation. The upper-triangular part
/// will not be read.
///
/// The output is a tensor of the same shape as the input
/// containing the Cholesky decompositions for all input submatrices `[..., :, :]`.
///
/// **Note**: The gradient computation on GPU is faster for large matrices but
/// not for large batch dimensions when the submatrices are small. In this
/// case it might be faster to use the CPU.
///
/// - Parameter input: Shape is `[..., M, M]`.
///
/// - Output output: Shape is `[..., M, M]`.
@inlinable @inline(__always)
public static func cholesky<T: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Cholesky",
    input,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the reverse mode backpropagated gradient of the Cholesky algorithm.
///
/// For an explanation see "Differentiation of the Cholesky algorithm" by
/// Iain Murray http://arxiv.org/abs/1602.07527.
///
/// - Parameters:
///   - l: Output of batch Cholesky algorithm l = cholesky(A). Shape is `[..., M, M]`.
///     Algorithm depends only on lower triangular part of the innermost matrices of
///     this tensor.
///   - grad: df/dl where f is some scalar function. Shape is `[..., M, M]`.
///     Algorithm depends only on lower triangular part of the innermost matrices of
///     this tensor.
///
/// - Output output: Symmetrized version of df/dA . Shape is `[..., M, M]`
@inlinable @inline(__always)
public static func choleskyGrad<T: FloatingPoint & TensorFlowScalar>(
  l: Tensor<T>,
  grad: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("CholeskyGrad",
    l,
    grad,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Clips tensor values to a specified min and max.
///
/// Given a tensor `t`, this operation returns a tensor of the same type and
/// shape as `t` with its values clipped to `clip_value_min` and `clip_value_max`.
/// Any values less than `clip_value_min` are set to `clip_value_min`. Any values
/// greater than `clip_value_max` are set to `clip_value_max`.
///
/// - Parameters:
///   - t: A `Tensor`.
///   - clip_value_min: A 0-D (scalar) `Tensor`, or a `Tensor` with the same shape
///     as `t`. The minimum value to clip by.
///   - clip_value_max: A 0-D (scalar) `Tensor`, or a `Tensor` with the same shape
///     as `t`. The maximum value to clip by.
///
/// - Output output: A clipped `Tensor` with the same shape as input 't'.
@inlinable @inline(__always)
public static func clipByValue<T: Numeric & TensorFlowScalar>(
  t: Tensor<T>,
  clipValueMin: Tensor<T>,
  clipValueMax: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("ClipByValue",
    t,
    clipValueMin,
    clipValueMax,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// An Op to permute tensors across replicated TPU instances.
///
/// Each instance supplies its own input.
///
/// For example, suppose there are 4 TPU instances: `[A, B, C, D]`. Passing
/// source_target_pairs=`[[0,1],[1,2],[2,3],[3,0]]` gets the outputs:
/// `[D, A, B, C]`.
///
/// - Parameters:
///   - input: The local input to be permuted. Currently only supports float and
///     bfloat16.
///   - source_target_pairs: A tensor with shape [num_pairs, 2].
///
/// - Attr T: The type of elements to be exchanged.
///
/// - Output output: The permuted input.
@inlinable @inline(__always)
public static func collectivePermute<T: Numeric & TensorFlowScalar>(
  _ input: Tensor<T>,
  sourceTargetPairs: Tensor<Int32>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("CollectivePermute",
    input,
    sourceTargetPairs,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Mutually reduces multiple tensors of identical type and shape.
@inlinable @inline(__always)
public static func collectiveReduce<T: Numeric & TensorFlowScalar>(
  _ input: Tensor<T>,
  groupSize: Int64,
  groupKey: Int64,
  instanceKey: Int64,
  mergeOp: MergeOp,
  finalOp: FinalOp,
  subdivOffsets: [Int32],
  waitFor: [Int32]
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("CollectiveReduce",
    input,
    T$dtype: T.tensorFlowDataType,
    group_size: groupSize,
    group_key: groupKey,
    instance_key: instanceKey,
    merge_op: mergeOp.cName,
    final_op: finalOp.cName,
    subdiv_offsets: subdivOffsets,
    wait_for: waitFor)
  return Tensor(handle: ret)
}

/// Greedily selects a subset of bounding boxes in descending order of score,
///
/// This operation performs non_max_suppression on the inputs per batch, across
/// all classes.
/// Prunes away boxes that have high intersection-over-union (IOU) overlap
/// with previously selected boxes.  Bounding boxes are supplied as
/// [y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any
/// diagonal pair of box corners and the coordinates can be provided as normalized
/// (i.e., lying in the interval [0, 1]) or absolute.  Note that this algorithm
/// is agnostic to where the origin is in the coordinate system. Also note that
/// this algorithm is invariant to orthogonal transformations and translations
/// of the coordinate system; thus translating or reflections of the coordinate
/// system result in the same boxes being selected by the algorithm.
/// The output of this operation is the final boxes, scores and classes tensor
/// returned after performing non_max_suppression.
///
/// - Parameters:
///   - boxes: A 4-D float tensor of shape `[batch_size, num_boxes, q, 4]`. If `q` is 1 then 
///     same boxes are used for all classes otherwise, if `q` is equal to number of 
///     classes, class-specific boxes are used.
///   - scores: A 3-D float tensor of shape `[batch_size, num_boxes, num_classes]`
///     representing a single score corresponding to each box (each row of boxes).
///   - max_output_size_per_class: A scalar integer tensor representing the maximum number of 
///     boxes to be selected by non max suppression per class
///   - max_total_size: A scalar representing maximum number of boxes retained over all classes.
///   - iou_threshold: A 0-D float tensor representing the threshold for deciding whether
///     boxes overlap too much with respect to IOU.
///   - score_threshold: A 0-D float tensor representing the threshold for deciding when to remove
///     boxes based on score.
///
/// - Attr pad_per_class: If false, the output nmsed boxes, scores and classes
///   are padded/clipped to `max_total_size`. If true, the
///   output nmsed boxes, scores and classes are padded to be of length
///   `max_size_per_class`*`num_classes`, unless it exceeds `max_total_size` in
///   which case it is clipped to `max_total_size`. Defaults to false.
///
/// - Outputs:
///   - nmsed_boxes: A [batch_size, max_detections, 4] float32 tensor 
///     containing the non-max suppressed boxes.
///   - nmsed_scores: A [batch_size, max_detections] float32 tensor 
///     containing the scores for the boxes.
///   - nmsed_classes: A [batch_size, max_detections] float32 tensor 
///     containing the classes for the boxes.
///   - valid_detections: A [batch_size] int32 tensor indicating the number of
///     valid detections per batch item. Only the top num_detections[i] entries in
///     nms_boxes[i], nms_scores[i] and nms_class[i] are valid. The rest of the
///     entries are zero paddings.
@inlinable @inline(__always)
public static func combinedNonMaxSuppression(
  boxes: Tensor<Float>,
  scores: Tensor<Float>,
  maxOutputSizePerClass: Tensor<Int32>,
  maxTotalSize: Tensor<Int32>,
  iouThreshold: Tensor<Float>,
  scoreThreshold: Tensor<Float>,
  padPerClass: Bool = false
) -> (nmsedBoxes: Tensor<Float>, nmsedScores: Tensor<Float>, nmsedClasses: Tensor<Float>, validDetections: Tensor<Int32>) {
  let ret: (TensorHandle<Float>, TensorHandle<Float>, TensorHandle<Float>, TensorHandle<Int32>) = #tfop("CombinedNonMaxSuppression",
    boxes,
    scores,
    maxOutputSizePerClass,
    maxTotalSize,
    iouThreshold,
    scoreThreshold,
    pad_per_class: padPerClass)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2), Tensor(handle: ret.3))
}

/// Compare values of `input` to `threshold` and pack resulting bits into a `uint8`.
///
/// Each comparison returns a boolean `true` (if `input_value > threshold`)
/// or and `false` otherwise.
///
/// This operation is useful for Locality-Sensitive-Hashing (LSH) and other
/// algorithms that use hashing approximations of cosine and `L2` distances;
/// codes can be generated from an input via:
///
/// ```python
/// codebook_size = 50
/// codebook_bits = codebook_size * 32
/// codebook = tf.get_variable('codebook', [x.shape[-1].value, codebook_bits],
///                            dtype=x.dtype,
///                            initializer=tf.orthogonal_initializer())
/// codes = compare_and_threshold(tf.matmul(x, codebook), threshold=0.)
/// codes = tf.bitcast(codes, tf.int32)  # go from uint8 to int32
/// # now codes has shape x.shape[:-1] + [codebook_size]
/// ```
///
/// **NOTE**: Currently, the innermost dimension of the tensor must be divisible
/// by 8.
///
/// Given an `input` shaped `[s0, s1, ..., s_n]`, the output is
/// a `uint8` tensor shaped `[s0, s1, ..., s_n / 8]`.
///
/// - Parameters:
///   - input: Values to compare against `threshold` and bitpack.
///   - threshold: Threshold to compare against.
///
/// - Attr T: The type of the input and threshold.
///
/// - Output output: The bitpacked comparisons.
@inlinable @inline(__always)
public static func compareAndBitpack<T: TensorFlowScalar>(
  _ input: Tensor<T>,
  threshold: Tensor<T>
) -> Tensor<UInt8> {
  let ret: TensorHandle<UInt8> = #tfop("CompareAndBitpack",
    input,
    threshold,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Converts two real numbers to a complex number.
///
/// Given a tensor `real` representing the real part of a complex number, and a
/// tensor `imag` representing the imaginary part of a complex number, this
/// operation returns complex numbers elementwise of the form \\(a + bj\\), where
/// *a* represents the `real` part and *b* represents the `imag` part.
///
/// The input tensors `real` and `imag` must have the same shape.
///
/// For example:
///
/// ```
/// # tensor 'real' is [2.25, 3.25]
/// # tensor `imag` is [4.75, 5.75]
/// tf.complex(real, imag) ==> [[2.25 + 4.75j], [3.25 + 5.75j]]
/// ```
@inlinable @inline(__always)
public static func complex<T: FloatingPoint & TensorFlowScalar, Tout: TensorFlowScalar>(
  real: Tensor<T>,
  imag: Tensor<T>
) -> Tensor<Tout> {
  let ret: TensorHandle<Tout> = #tfop("Complex",
    real,
    imag,
    T$dtype: T.tensorFlowDataType,
    Tout$dtype: Tout.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the complex absolute value of a tensor.
///
/// Given a tensor `x` of complex numbers, this operation returns a tensor of type
/// `float` or `double` that is the absolute value of each element in `x`. All
/// elements in `x` must be complex numbers of the form \\(a + bj\\). The absolute
/// value is computed as \\( \sqrt{a^2 + b^2}\\).
@inlinable @inline(__always)
public static func complexAbs<T: TensorFlowScalar, Tout: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<Tout> {
  let ret: TensorHandle<Tout> = #tfop("ComplexAbs",
    x,
    T$dtype: T.tensorFlowDataType,
    Tout$dtype: Tout.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the ids of the positions in sampled_candidates that match true_labels.
///
/// When doing log-odds NCE, the result of this op should be passed through a
/// SparseToDense op, then added to the logits of the sampled candidates. This has
/// the effect of 'removing' the sampled labels that match the true labels by
/// making the classifier sure that they are sampled labels.
///
/// - Parameters:
///   - true_classes: The true_classes output of UnpackSparseLabels.
///   - sampled_candidates: The sampled_candidates output of CandidateSampler.
///
/// - Attrs:
///   - num_true: Number of true labels per context.
///   - seed: If either seed or seed2 are set to be non-zero, the random number
///     generator is seeded by the given seed.  Otherwise, it is seeded by a
///     random seed.
///   - seed2: An second seed to avoid seed collision.
///
/// - Outputs:
///   - indices: A vector of indices corresponding to rows of true_candidates.
///   - ids: A vector of IDs of positions in sampled_candidates that match a true_label
///     for the row with the corresponding index in indices.
///   - weights: A vector of the same length as indices and ids, in which each element
///     is -FLOAT_MAX.
@inlinable @inline(__always)
public static func computeAccidentalHits(
  trueClasses: Tensor<Int64>,
  sampledCandidates: Tensor<Int64>,
  numTrue: Int64,
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> (indices: Tensor<Int32>, ids: Tensor<Int64>, weights: Tensor<Float>) {
  let ret: (TensorHandle<Int32>, TensorHandle<Int64>, TensorHandle<Float>) = #tfop("ComputeAccidentalHits",
    trueClasses,
    sampledCandidates,
    num_true: numTrue,
    seed: seed,
    seed2: seed2)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Concatenates tensors along one dimension.
///
/// - Parameters:
///   - concat_dim: 0-D.  The dimension along which to concatenate.  Must be in the
///     range [0, rank(values)).
///   - values: The `N` Tensors to concatenate. Their ranks and types must match,
///     and their sizes must match in all dimensions except `concat_dim`.
///
/// - Output output: A `Tensor` with the concatenation of values stacked along the
///   `concat_dim` dimension.  This tensor's shape matches that of `values` except
///   in `concat_dim` where it has the sum of the sizes.
@inlinable @inline(__always)
public static func concat<T: TensorFlowScalar>(
  concatDim: Tensor<Int32>,
  _ values: [Tensor<T>]
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Concat",
    concatDim,
    values,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Concatenates tensors along one dimension.
///
/// - Parameters:
///   - values: List of `N` Tensors to concatenate. Their ranks and types must match,
///     and their sizes must match in all dimensions except `concat_dim`.
///   - axis: 0-D.  The dimension along which to concatenate.  Must be in the
///     range [-rank(values), rank(values)).
///
/// - Output output: A `Tensor` with the concatenation of values stacked along the
///   `concat_dim` dimension.  This tensor's shape matches that of `values` except
///   in `concat_dim` where it has the sum of the sizes.
@inlinable @inline(__always)
public static func concatV2<T: TensorFlowScalar, Tidx: BinaryInteger & TensorFlowScalar>(
  _ values: [Tensor<T>],
  axis: Tensor<Tidx>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("ConcatV2",
    values,
    axis,
    T$dtype: T.tensorFlowDataType,
    Tidx$dtype: Tidx.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Sets up the centralized structures for a distributed TPU system.
///
/// - Attrs:
///   - embedding_config: Reserved. Do not use.
///   - tpu_embedding_config: Serialized tensorflow.tpu.TPUEmbeddingConfiguration that
///     describes the embedding lookups of the program.
///   - is_global_init: Reserved. Do not use.
///
/// - Output topology: A serialized tensorflow.tpu.TopologyProto that describes the TPU
///   topology.
@inlinable @inline(__always)
public static func configureDistributedTPU(
  embeddingConfig: String,
  tpuEmbeddingConfig: String,
  isGlobalInit: Bool = false
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("ConfigureDistributedTPU",
    embedding_config: embeddingConfig,
    tpu_embedding_config: tpuEmbeddingConfig,
    is_global_init: isGlobalInit)
  return StringTensor(handle: ret)
}

/// Returns the complex conjugate of a complex number.
///
/// Given a tensor `input` of complex numbers, this operation returns a tensor of
/// complex numbers that are the complex conjugate of each element in `input`. The
/// complex numbers in `input` must be of the form \\(a + bj\\), where *a* is the
/// real part and *b* is the imaginary part.
///
/// The complex conjugate returned by this operation is of the form \\(a - bj\\).
///
/// For example:
///
/// ```
/// # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
/// tf.conj(input) ==> [-2.25 - 4.75j, 3.25 - 5.75j]
/// ```
@inlinable @inline(__always)
public static func conj<T: TensorFlowScalar>(
  _ input: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Conj",
    input,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Shuffle dimensions of x according to a permutation and conjugate the result.
///
/// The output `y` has the same rank as `x`. The shapes of `x` and `y` satisfy:
///   `y.shape[i] == x.shape[perm[i]] for i in [0, 1, ..., rank(x) - 1]`
///   `y[i,j,k,...,s,t,u] == conj(x[perm[i], perm[j], perm[k],...,perm[s], perm[t], perm[u]])`
@inlinable @inline(__always)
public static func conjugateTranspose<T: TensorFlowScalar, Tperm: BinaryInteger & TensorFlowScalar>(
  _ x: Tensor<T>,
  perm: Tensor<Tperm>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("ConjugateTranspose",
    x,
    perm,
    T$dtype: T.tensorFlowDataType,
    Tperm$dtype: Tperm.tensorFlowDataType)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func constructionFails(
) {
  return #tfop("ConstructionFails")
}

/// Does nothing. Serves as a control trigger for scheduling.
///
/// Only useful as a placeholder for control edges.
@inlinable @inline(__always)
public static func controlTrigger(
) {
  return #tfop("ControlTrigger")
}

/// Computes a 2-D convolution given 4-D `input` and `filter` tensors.
///
/// Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
/// and a filter / kernel tensor of shape
/// `[filter_height, filter_width, in_channels, out_channels]`, this op
/// performs the following:
///
/// 1. Flattens the filter to a 2-D matrix with shape
///    `[filter_height * filter_width * in_channels, output_channels]`.
/// 2. Extracts image patches from the input tensor to form a *virtual*
///    tensor of shape `[batch, out_height, out_width,
///    filter_height * filter_width * in_channels]`.
/// 3. For each patch, right-multiplies the filter matrix and the image patch
///    vector.
///
/// In detail, with the default NHWC format,
///
///     output[b, i, j, k] =
///         sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
///                         filter[di, dj, q, k]
///
/// Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
/// horizontal and vertices strides, `strides = [1, stride, stride, 1]`.
///
/// - Parameters:
///   - input: A 4-D tensor. The dimension order is interpreted according to the value
///     of `data_format`, see below for details.
///   - filter: A 4-D tensor of shape
///     `[filter_height, filter_width, in_channels, out_channels]`
///
/// - Attrs:
///   - strides: 1-D tensor of length 4.  The stride of the sliding window for each
///     dimension of `input`. The dimension order is determined by the value of
///     `data_format`, see below for details.
///   - padding: The type of padding algorithm to use.
///   - explicit_paddings: If `padding` is `"EXPLICIT"`, the list of explicit padding amounts. For the ith
///     dimension, the amount of padding inserted before and after the dimension is
///     `explicit_paddings[2 * i]` and `explicit_paddings[2 * i + 1]`, respectively. If
///     `padding` is not `"EXPLICIT"`, `explicit_paddings` must be empty.
///   - data_format: Specify the data format of the input and output data. With the
///     default format "NHWC", the data is stored in the order of:
///         [batch, height, width, channels].
///     Alternatively, the format could be "NCHW", the data storage order of:
///         [batch, channels, height, width].
///   - dilations: 1-D tensor of length 4.  The dilation factor for each dimension of
///     `input`. If set to k > 1, there will be k-1 skipped cells between each
///     filter element on that dimension. The dimension order is determined by the
///     value of `data_format`, see above for details. Dilations in the batch and
///     depth dimensions must be 1.
///
/// - Output output: A 4-D tensor. The dimension order is determined by the value of
///   `data_format`, see below for details.
@inlinable @inline(__always)
public static func conv2D<T: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>,
  filter: Tensor<T>,
  strides: [Int32],
  useCudnnOnGpu: Bool = true,
  padding: Padding2,
  explicitPaddings: [Int32],
  dataFormat: DataFormat = .nhwc,
  dilations: [Int32] = [1, 1, 1, 1]
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Conv2D",
    input,
    filter,
    T$dtype: T.tensorFlowDataType,
    strides: strides,
    use_cudnn_on_gpu: useCudnnOnGpu,
    padding: padding.cName,
    explicit_paddings: explicitPaddings,
    data_format: dataFormat.cName,
    dilations: dilations)
  return Tensor(handle: ret)
}

/// Computes the gradients of convolution with respect to the filter.
///
/// - Parameters:
///   - input: 4-D with shape `[batch, in_height, in_width, in_channels]`.
///   - filter_sizes: An integer vector representing the tensor shape of `filter`,
///     where `filter` is a 4-D
///     `[filter_height, filter_width, in_channels, out_channels]` tensor.
///   - out_backprop: 4-D with shape `[batch, out_height, out_width, out_channels]`.
///     Gradients w.r.t. the output of the convolution.
///
/// - Attrs:
///   - strides: The stride of the sliding window for each dimension of the input
///     of the convolution. Must be in the same order as the dimension specified with
///     format.
///   - padding: The type of padding algorithm to use.
///   - explicit_paddings: If `padding` is `"EXPLICIT"`, the list of explicit padding amounts. For the ith
///     dimension, the amount of padding inserted before and after the dimension is
///     `explicit_paddings[2 * i]` and `explicit_paddings[2 * i + 1]`, respectively. If
///     `padding` is not `"EXPLICIT"`, `explicit_paddings` must be empty.
///   - data_format: Specify the data format of the input and output data. With the
///     default format "NHWC", the data is stored in the order of:
///         [batch, in_height, in_width, in_channels].
///     Alternatively, the format could be "NCHW", the data storage order of:
///         [batch, in_channels, in_height, in_width].
///   - dilations: 1-D tensor of length 4.  The dilation factor for each dimension of
///     `input`. If set to k > 1, there will be k-1 skipped cells between each filter
///     element on that dimension. The dimension order is determined by the value of
///     `data_format`, see above for details. Dilations in the batch and depth
///     dimensions must be 1.
///
/// - Output output: 4-D with shape
///   `[filter_height, filter_width, in_channels, out_channels]`.  Gradient w.r.t.
///   the `filter` input of the convolution.
@inlinable @inline(__always)
public static func conv2DBackpropFilter<T: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>,
  filterSizes: Tensor<Int32>,
  outBackprop: Tensor<T>,
  strides: [Int32],
  useCudnnOnGpu: Bool = true,
  padding: Padding2,
  explicitPaddings: [Int32],
  dataFormat: DataFormat = .nhwc,
  dilations: [Int32] = [1, 1, 1, 1]
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Conv2DBackpropFilter",
    input,
    filterSizes,
    outBackprop,
    T$dtype: T.tensorFlowDataType,
    strides: strides,
    use_cudnn_on_gpu: useCudnnOnGpu,
    padding: padding.cName,
    explicit_paddings: explicitPaddings,
    data_format: dataFormat.cName,
    dilations: dilations)
  return Tensor(handle: ret)
}

/// Computes the gradients of convolution with respect to the input.
///
/// - Parameters:
///   - input_sizes: An integer vector representing the shape of `input`,
///     where `input` is a 4-D `[batch, height, width, channels]` tensor.
///   - filter: 4-D with shape
///     `[filter_height, filter_width, in_channels, out_channels]`.
///   - out_backprop: 4-D with shape `[batch, out_height, out_width, out_channels]`.
///     Gradients w.r.t. the output of the convolution.
///
/// - Attrs:
///   - strides: The stride of the sliding window for each dimension of the input
///     of the convolution. Must be in the same order as the dimension specified with
///     format.
///   - padding: The type of padding algorithm to use.
///   - explicit_paddings: If `padding` is `"EXPLICIT"`, the list of explicit padding amounts. For the ith
///     dimension, the amount of padding inserted before and after the dimension is
///     `explicit_paddings[2 * i]` and `explicit_paddings[2 * i + 1]`, respectively. If
///     `padding` is not `"EXPLICIT"`, `explicit_paddings` must be empty.
///   - data_format: Specify the data format of the input and output data. With the
///     default format "NHWC", the data is stored in the order of:
///         [batch, in_height, in_width, in_channels].
///     Alternatively, the format could be "NCHW", the data storage order of:
///         [batch, in_channels, in_height, in_width].
///   - dilations: 1-D tensor of length 4.  The dilation factor for each dimension of
///     `input`. If set to k > 1, there will be k-1 skipped cells between each filter
///     element on that dimension. The dimension order is determined by the value of
///     `data_format`, see above for details. Dilations in the batch and depth
///     dimensions must be 1.
///
/// - Output output: 4-D with shape `[batch, in_height, in_width, in_channels]`.  Gradient
///   w.r.t. the input of the convolution.
@inlinable @inline(__always)
public static func conv2DBackpropInput<T: FloatingPoint & TensorFlowScalar>(
  inputSizes: Tensor<Int32>,
  filter: Tensor<T>,
  outBackprop: Tensor<T>,
  strides: [Int32],
  useCudnnOnGpu: Bool = true,
  padding: Padding2,
  explicitPaddings: [Int32],
  dataFormat: DataFormat = .nhwc,
  dilations: [Int32] = [1, 1, 1, 1]
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Conv2DBackpropInput",
    inputSizes,
    filter,
    outBackprop,
    T$dtype: T.tensorFlowDataType,
    strides: strides,
    use_cudnn_on_gpu: useCudnnOnGpu,
    padding: padding.cName,
    explicit_paddings: explicitPaddings,
    data_format: dataFormat.cName,
    dilations: dilations)
  return Tensor(handle: ret)
}

/// Computes a 3-D convolution given 5-D `input` and `filter` tensors.
///
/// In signal processing, cross-correlation is a measure of similarity of
/// two waveforms as a function of a time-lag applied to one of them. This
/// is also known as a sliding dot product or sliding inner-product.
///
/// Our Conv3D implements a form of cross-correlation.
///
/// - Parameters:
///   - input: Shape `[batch, in_depth, in_height, in_width, in_channels]`.
///   - filter: Shape `[filter_depth, filter_height, filter_width, in_channels,
///     out_channels]`. `in_channels` must match between `input` and `filter`.
///
/// - Attrs:
///   - strides: 1-D tensor of length 5. The stride of the sliding window for each
///     dimension of `input`. Must have `strides[0] = strides[4] = 1`.
///   - padding: The type of padding algorithm to use.
///   - data_format: The data format of the input and output data. With the
///     default format "NDHWC", the data is stored in the order of:
///         [batch, in_depth, in_height, in_width, in_channels].
///     Alternatively, the format could be "NCDHW", the data storage order is:
///         [batch, in_channels, in_depth, in_height, in_width].
///   - dilations: 1-D tensor of length 5.  The dilation factor for each dimension of
///     `input`. If set to k > 1, there will be k-1 skipped cells between each
///     filter element on that dimension. The dimension order is determined by the
///     value of `data_format`, see above for details. Dilations in the batch and
///     depth dimensions must be 1.
@inlinable @inline(__always)
public static func conv3D<T: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>,
  filter: Tensor<T>,
  strides: [Int32],
  padding: Padding,
  dataFormat: DataFormat1 = .ndhwc,
  dilations: [Int32] = [1, 1, 1, 1, 1]
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Conv3D",
    input,
    filter,
    T$dtype: T.tensorFlowDataType,
    strides: strides,
    padding: padding.cName,
    data_format: dataFormat.cName,
    dilations: dilations)
  return Tensor(handle: ret)
}

/// Computes the gradients of 3-D convolution with respect to the filter.
///
/// - Parameters:
///   - input: Shape `[batch, depth, rows, cols, in_channels]`.
///   - filter: Shape `[depth, rows, cols, in_channels, out_channels]`.
///     `in_channels` must match between `input` and `filter`.
///   - out_backprop: Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
///     out_channels]`.
///
/// - Attrs:
///   - strides: 1-D tensor of length 5. The stride of the sliding window for each
///     dimension of `input`. Must have `strides[0] = strides[4] = 1`.
///   - padding: The type of padding algorithm to use.
@inlinable @inline(__always)
public static func conv3DBackpropFilter<T: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>,
  filter: Tensor<T>,
  outBackprop: Tensor<T>,
  strides: [Int32],
  padding: Padding,
  dilations: [Int32] = [1, 1, 1, 1, 1]
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Conv3DBackpropFilter",
    input,
    filter,
    outBackprop,
    T$dtype: T.tensorFlowDataType,
    strides: strides,
    padding: padding.cName,
    dilations: dilations)
  return Tensor(handle: ret)
}

/// Computes the gradients of 3-D convolution with respect to the filter.
///
/// - Parameters:
///   - input: Shape `[batch, depth, rows, cols, in_channels]`.
///   - filter_sizes: An integer vector representing the tensor shape of `filter`,
///     where `filter` is a 5-D
///     `[filter_depth, filter_height, filter_width, in_channels, out_channels]`
///     tensor.
///   - out_backprop: Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
///     out_channels]`.
///
/// - Attrs:
///   - strides: 1-D tensor of length 5. The stride of the sliding window for each
///     dimension of `input`. Must have `strides[0] = strides[4] = 1`.
///   - padding: The type of padding algorithm to use.
///   - data_format: The data format of the input and output data. With the
///     default format "NDHWC", the data is stored in the order of:
///         [batch, in_depth, in_height, in_width, in_channels].
///     Alternatively, the format could be "NCDHW", the data storage order is:
///         [batch, in_channels, in_depth, in_height, in_width].
///   - dilations: 1-D tensor of length 5.  The dilation factor for each dimension of
///     `input`. If set to k > 1, there will be k-1 skipped cells between each
///     filter element on that dimension. The dimension order is determined by the
///     value of `data_format`, see above for details. Dilations in the batch and
///     depth dimensions must be 1.
@inlinable @inline(__always)
public static func conv3DBackpropFilterV2<T: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>,
  filterSizes: Tensor<Int32>,
  outBackprop: Tensor<T>,
  strides: [Int32],
  padding: Padding,
  dataFormat: DataFormat1 = .ndhwc,
  dilations: [Int32] = [1, 1, 1, 1, 1]
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Conv3DBackpropFilterV2",
    input,
    filterSizes,
    outBackprop,
    T$dtype: T.tensorFlowDataType,
    strides: strides,
    padding: padding.cName,
    data_format: dataFormat.cName,
    dilations: dilations)
  return Tensor(handle: ret)
}

/// Computes the gradients of 3-D convolution with respect to the input.
///
/// - Parameters:
///   - input: Shape `[batch, depth, rows, cols, in_channels]`.
///   - filter: Shape `[depth, rows, cols, in_channels, out_channels]`.
///     `in_channels` must match between `input` and `filter`.
///   - out_backprop: Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
///     out_channels]`.
///
/// - Attrs:
///   - strides: 1-D tensor of length 5. The stride of the sliding window for each
///     dimension of `input`. Must have `strides[0] = strides[4] = 1`.
///   - padding: The type of padding algorithm to use.
@inlinable @inline(__always)
public static func conv3DBackpropInput<T: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>,
  filter: Tensor<T>,
  outBackprop: Tensor<T>,
  strides: [Int32],
  padding: Padding,
  dilations: [Int32] = [1, 1, 1, 1, 1]
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Conv3DBackpropInput",
    input,
    filter,
    outBackprop,
    T$dtype: T.tensorFlowDataType,
    strides: strides,
    padding: padding.cName,
    dilations: dilations)
  return Tensor(handle: ret)
}

/// Computes the gradients of 3-D convolution with respect to the input.
///
/// - Parameters:
///   - input_sizes: An integer vector representing the tensor shape of `input`,
///     where `input` is a 5-D
///     `[batch, depth, rows, cols, in_channels]` tensor.
///   - filter: Shape `[depth, rows, cols, in_channels, out_channels]`.
///     `in_channels` must match between `input` and `filter`.
///   - out_backprop: Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
///     out_channels]`.
///
/// - Attrs:
///   - strides: 1-D tensor of length 5. The stride of the sliding window for each
///     dimension of `input`. Must have `strides[0] = strides[4] = 1`.
///   - padding: The type of padding algorithm to use.
///   - data_format: The data format of the input and output data. With the
///     default format "NDHWC", the data is stored in the order of:
///         [batch, in_depth, in_height, in_width, in_channels].
///     Alternatively, the format could be "NCDHW", the data storage order is:
///         [batch, in_channels, in_depth, in_height, in_width].
///   - dilations: 1-D tensor of length 5.  The dilation factor for each dimension of
///     `input`. If set to k > 1, there will be k-1 skipped cells between each
///     filter element on that dimension. The dimension order is determined by the
///     value of `data_format`, see above for details. Dilations in the batch and
///     depth dimensions must be 1.
@inlinable @inline(__always)
public static func conv3DBackpropInputV2<T: FloatingPoint & TensorFlowScalar, Tshape: BinaryInteger & TensorFlowScalar>(
  inputSizes: Tensor<Tshape>,
  filter: Tensor<T>,
  outBackprop: Tensor<T>,
  strides: [Int32],
  padding: Padding,
  dataFormat: DataFormat1 = .ndhwc,
  dilations: [Int32] = [1, 1, 1, 1, 1]
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Conv3DBackpropInputV2",
    inputSizes,
    filter,
    outBackprop,
    T$dtype: T.tensorFlowDataType,
    Tshape$dtype: Tshape.tensorFlowDataType,
    strides: strides,
    padding: padding.cName,
    data_format: dataFormat.cName,
    dilations: dilations)
  return Tensor(handle: ret)
}

/// Copy Op.
///
/// Performs CPU-to-CPU or GPU-to-GPU deep-copying of tensor, depending on the
/// device on which the tensor is allocated.
/// N.B.: If the all downstream attached debug ops are disabled given the current
/// gRPC gating status, the output will simply forward the input tensor without
/// deep-copying. See the documentation of Debug* ops for more details.
///
/// Unlike the CopyHost Op, this op does not have HostMemory constraint on its
/// input or output.
///
/// - Parameter input: Input tensor.
///
/// - Attrs:
///   - tensor_name: The name of the input tensor.
///   - debug_ops_spec: A list of debug op spec (op, url, gated_grpc) for attached debug
///     ops. Each element of the list has the format
///     <debug_op>;<grpc_url>;<gated_grpc>, wherein gated_grpc is boolean represented
///     as 0/1. E.g., "DebugIdentity;grpc://foo:3333;1",
///     "DebugIdentity;file:///tmp/tfdbg_1;0".
///
/// - Output output: Output tensor, deep-copied from input.
@inlinable @inline(__always)
public static func copy<T: TensorFlowScalar>(
  _ input: Tensor<T>,
  tensorName: String,
  debugOpsSpec: [String]
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Copy",
    input,
    T$dtype: T.tensorFlowDataType,
    tensor_name: tensorName,
    debug_ops_spec: debugOpsSpec)
  return Tensor(handle: ret)
}

/// Copy Host Op.
///
/// Performs CPU-to-CPU deep-copying of tensor.
/// N.B.: If the all downstream attached debug ops are disabled given the current
/// gRPC gating status, the output will simply forward the input tensor without
/// deep-copying. See the documentation of Debug* ops for more details.
///
/// Unlike the Copy Op, this op has HostMemory constraint on its input or output.
///
/// - Parameter input: Input tensor.
///
/// - Attrs:
///   - tensor_name: The name of the input tensor.
///   - debug_ops_spec: A list of debug op spec (op, url, gated_grpc) for attached debug
///     ops. Each element of the list has the format
///     <debug_op>;<grpc_url>;<gated_grpc>, wherein gated_grpc is boolean represented
///     as 0/1. E.g., "DebugIdentity;grpc://foo:3333;1",
///     "DebugIdentity;file:///tmp/tfdbg_1;0".
///
/// - Output output: Output tensor, deep-copied from input.
@inlinable @inline(__always)
public static func copyHost<T: TensorFlowScalar>(
  _ input: Tensor<T>,
  tensorName: String,
  debugOpsSpec: [String]
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("CopyHost",
    input,
    T$dtype: T.tensorFlowDataType,
    tensor_name: tensorName,
    debug_ops_spec: debugOpsSpec)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func copyOp<T: TensorFlowScalar>(
  _ a: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("CopyOp",
    a,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes cos of x element-wise.
@inlinable @inline(__always)
public static func cos<T: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Cos",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes hyperbolic cosine of x element-wise.
@inlinable @inline(__always)
public static func cosh<T: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Cosh",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Extracts crops from the input image tensor and resizes them.
///
/// Extracts crops from the input image tensor and resizes them using bilinear
/// sampling or nearest neighbor sampling (possibly with aspect ratio change) to a
/// common output size specified by `crop_size`. This is more general than the
/// `crop_to_bounding_box` op which extracts a fixed size slice from the input image
/// and does not allow resizing or aspect ratio change.
///
/// Returns a tensor with `crops` from the input `image` at positions defined at the
/// bounding box locations in `boxes`. The cropped boxes are all resized (with
/// bilinear or nearest neighbor interpolation) to a fixed
/// `size = [crop_height, crop_width]`. The result is a 4-D tensor
/// `[num_boxes, crop_height, crop_width, depth]`. The resizing is corner aligned.
/// In particular, if `boxes = [[0, 0, 1, 1]]`, the method will give identical
/// results to using `tf.image.resize_bilinear()` or
/// `tf.image.resize_nearest_neighbor()`(depends on the `method` argument) with
/// `align_corners=True`.
///
/// - Parameters:
///   - image: A 4-D tensor of shape `[batch, image_height, image_width, depth]`.
///     Both `image_height` and `image_width` need to be positive.
///   - boxes: A 2-D tensor of shape `[num_boxes, 4]`. The `i`-th row of the tensor
///     specifies the coordinates of a box in the `box_ind[i]` image and is specified
///     in normalized coordinates `[y1, x1, y2, x2]`. A normalized coordinate value of
///     `y` is mapped to the image coordinate at `y * (image_height - 1)`, so as the
///     `[0, 1]` interval of normalized image height is mapped to
///     `[0, image_height - 1]` in image height coordinates. We do allow `y1` > `y2`, in
///     which case the sampled crop is an up-down flipped version of the original
///     image. The width dimension is treated similarly. Normalized coordinates
///     outside the `[0, 1]` range are allowed, in which case we use
///     `extrapolation_value` to extrapolate the input image values.
///   - box_ind: A 1-D tensor of shape `[num_boxes]` with int32 values in `[0, batch)`.
///     The value of `box_ind[i]` specifies the image that the `i`-th box refers to.
///   - crop_size: A 1-D tensor of 2 elements, `size = [crop_height, crop_width]`. All
///     cropped image patches are resized to this size. The aspect ratio of the image
///     content is not preserved. Both `crop_height` and `crop_width` need to be
///     positive.
///
/// - Attrs:
///   - method: A string specifying the sampling method for resizing. It can be either
///     `"bilinear"` or `"nearest"` and default to `"bilinear"`. Currently two sampling
///     methods are supported: Bilinear and Nearest Neighbor.
///   - extrapolation_value: Value used for extrapolation, when applicable.
///
/// - Output crops: A 4-D tensor of shape `[num_boxes, crop_height, crop_width, depth]`.
@inlinable @inline(__always)
public static func cropAndResize<T: Numeric & TensorFlowScalar>(
  image: Tensor<T>,
  boxes: Tensor<Float>,
  boxInd: Tensor<Int32>,
  cropSize: Tensor<Int32>,
  method: Method = .bilinear,
  extrapolationValue: Double = 0
) -> Tensor<Float> {
  let ret: TensorHandle<Float> = #tfop("CropAndResize",
    image,
    boxes,
    boxInd,
    cropSize,
    T$dtype: T.tensorFlowDataType,
    method: method.cName,
    extrapolation_value: extrapolationValue)
  return Tensor(handle: ret)
}

/// Computes the gradient of the crop_and_resize op wrt the input boxes tensor.
///
/// - Parameters:
///   - grads: A 4-D tensor of shape `[num_boxes, crop_height, crop_width, depth]`.
///   - image: A 4-D tensor of shape `[batch, image_height, image_width, depth]`.
///     Both `image_height` and `image_width` need to be positive.
///   - boxes: A 2-D tensor of shape `[num_boxes, 4]`. The `i`-th row of the tensor
///     specifies the coordinates of a box in the `box_ind[i]` image and is specified
///     in normalized coordinates `[y1, x1, y2, x2]`. A normalized coordinate value of
///     `y` is mapped to the image coordinate at `y * (image_height - 1)`, so as the
///     `[0, 1]` interval of normalized image height is mapped to
///     `[0, image_height - 1] in image height coordinates. We do allow y1 > y2, in
///     which case the sampled crop is an up-down flipped version of the original
///     image. The width dimension is treated similarly. Normalized coordinates
///     outside the `[0, 1]` range are allowed, in which case we use
///     `extrapolation_value` to extrapolate the input image values.
///   - box_ind: A 1-D tensor of shape `[num_boxes]` with int32 values in `[0, batch)`.
///     The value of `box_ind[i]` specifies the image that the `i`-th box refers to.
///
/// - Attr method: A string specifying the interpolation method. Only 'bilinear' is
///   supported for now.
///
/// - Output output: A 2-D tensor of shape `[num_boxes, 4]`.
@inlinable @inline(__always)
public static func cropAndResizeGradBoxes<T: Numeric & TensorFlowScalar>(
  grads: Tensor<Float>,
  image: Tensor<T>,
  boxes: Tensor<Float>,
  boxInd: Tensor<Int32>,
  method: Method3 = .bilinear
) -> Tensor<Float> {
  let ret: TensorHandle<Float> = #tfop("CropAndResizeGradBoxes",
    grads,
    image,
    boxes,
    boxInd,
    T$dtype: T.tensorFlowDataType,
    method: method.cName)
  return Tensor(handle: ret)
}

/// Computes the gradient of the crop_and_resize op wrt the input image tensor.
///
/// - Parameters:
///   - grads: A 4-D tensor of shape `[num_boxes, crop_height, crop_width, depth]`.
///   - boxes: A 2-D tensor of shape `[num_boxes, 4]`. The `i`-th row of the tensor
///     specifies the coordinates of a box in the `box_ind[i]` image and is specified
///     in normalized coordinates `[y1, x1, y2, x2]`. A normalized coordinate value of
///     `y` is mapped to the image coordinate at `y * (image_height - 1)`, so as the
///     `[0, 1]` interval of normalized image height is mapped to
///     `[0, image_height - 1] in image height coordinates. We do allow y1 > y2, in
///     which case the sampled crop is an up-down flipped version of the original
///     image. The width dimension is treated similarly. Normalized coordinates
///     outside the `[0, 1]` range are allowed, in which case we use
///     `extrapolation_value` to extrapolate the input image values.
///   - box_ind: A 1-D tensor of shape `[num_boxes]` with int32 values in `[0, batch)`.
///     The value of `box_ind[i]` specifies the image that the `i`-th box refers to.
///   - image_size: A 1-D tensor with value `[batch, image_height, image_width, depth]`
///     containing the original image size. Both `image_height` and `image_width` need
///     to be positive.
///
/// - Attr method: A string specifying the interpolation method. Only 'bilinear' is
///   supported for now.
///
/// - Output output: A 4-D tensor of shape `[batch, image_height, image_width, depth]`.
@inlinable @inline(__always)
public static func cropAndResizeGradImage<T: FloatingPoint & TensorFlowScalar>(
  grads: Tensor<Float>,
  boxes: Tensor<Float>,
  boxInd: Tensor<Int32>,
  imageSize: Tensor<Int32>,
  method: Method = .bilinear
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("CropAndResizeGradImage",
    grads,
    boxes,
    boxInd,
    imageSize,
    T$dtype: T.tensorFlowDataType,
    method: method.cName)
  return Tensor(handle: ret)
}

/// Compute the pairwise cross product.
///
/// `a` and `b` must be the same shape; they can either be simple 3-element vectors,
/// or any shape where the innermost dimension is 3. In the latter case, each pair
/// of corresponding 3-element vectors is cross-multiplied independently.
///
/// - Parameters:
///   - a: A tensor containing 3-element vectors.
///   - b: Another tensor, of same type and shape as `a`.
///
/// - Output product: Pairwise cross product of the vectors in `a` and `b`.
@inlinable @inline(__always)
public static func cross<T: Numeric & TensorFlowScalar>(
  _ a: Tensor<T>,
  _ b: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Cross",
    a,
    b,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// An Op to sum inputs across replicated TPU instances.
///
/// Each instance supplies its own input.
///
/// For example, suppose there are 8 TPU instances: `[A, B, C, D, E, F, G, H]`.
/// Passing group_assignment=`[[0,2,4,6],[1,3,5,7]]` sets `A, C, E, G` as group 0,
/// and `B, D, F, H` as group 1. Thus we get the outputs:
/// `[A+C+E+G, B+D+F+H, A+C+E+G, B+D+F+H, A+C+E+G, B+D+F+H, A+C+E+G, B+D+F+H]`.
///
/// - Parameters:
///   - input: The local input to the sum.
///   - group_assignment: An int32 tensor with shape
///     [num_groups, num_replicas_per_group]. `group_assignment[i]` represents the
///     replica ids in the ith subgroup.
///
/// - Attr T: The type of elements to be summed.
///
/// - Output output: The sum of all the distributed inputs.
@inlinable @inline(__always)
public static func crossReplicaSum<T: Numeric & TensorFlowScalar>(
  _ input: Tensor<T>,
  groupAssignment: Tensor<Int32>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("CrossReplicaSum",
    input,
    groupAssignment,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// A RNN backed by cuDNN.
///
/// Computes the RNN from the input and initial states, with respect to the params
/// buffer.
///
/// rnn_mode: Indicates the type of the RNN model.
/// input_mode: Indicate whether there is a linear projection between the input and
///   the actual computation before the first layer. 'skip_input' is only allowed
///   when input_size == num_units; 'auto_select' implies 'skip_input' when
///   input_size == num_units; otherwise, it implies 'linear_input'.
/// direction: Indicates whether a bidirectional model will be used. Should be
///   "unidirectional" or "bidirectional".
/// dropout: Dropout probability. When set to 0., dropout is disabled.
/// seed: The 1st part of a seed to initialize dropout.
/// seed2: The 2nd part of a seed to initialize dropout.
/// input: A 3-D tensor with the shape of [seq_length, batch_size, input_size].
/// input_h: A 3-D tensor with the shape of [num_layer * dir, batch_size,
///     num_units].
/// input_c: For LSTM, a 3-D tensor with the shape of
///     [num_layer * dir, batch, num_units]. For other models, it is ignored.
/// params: A 1-D tensor that contains the weights and biases in an opaque layout.
///     The size must be created through CudnnRNNParamsSize, and initialized
///     separately. Note that they might not be compatible across different
///     generations. So it is a good idea to save and restore
/// output: A 3-D tensor with the shape of [seq_length, batch_size,
///     dir * num_units].
/// output_h: The same shape has input_h.
/// output_c: The same shape as input_c for LSTM. An empty tensor for other models.
/// is_training: Indicates whether this operation is used for inferenece or
///   training.
/// reserve_space: An opaque tensor that can be used in backprop calculation. It
///   is only produced if is_training is false.
@inlinable @inline(__always)
public static func cudnnRNN<T: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>,
  inputH: Tensor<T>,
  inputC: Tensor<T>,
  params: Tensor<T>,
  rnnMode: RnnMode = .lstm,
  inputMode: InputMode = .linearInput,
  direction: Direction = .unidirectional,
  dropout: Double = 0,
  seed: Int64 = 0,
  seed2: Int64 = 0,
  isTraining: Bool = true
) -> (output: Tensor<T>, outputH: Tensor<T>, outputC: Tensor<T>, reserveSpace: Tensor<T>) {
  let ret: (TensorHandle<T>, TensorHandle<T>, TensorHandle<T>, TensorHandle<T>) = #tfop("CudnnRNN",
    input,
    inputH,
    inputC,
    params,
    T$dtype: T.tensorFlowDataType,
    rnn_mode: rnnMode.cName,
    input_mode: inputMode.cName,
    direction: direction.cName,
    dropout: dropout,
    seed: seed,
    seed2: seed2,
    is_training: isTraining)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2), Tensor(handle: ret.3))
}

/// Backprop step of CudnnRNN.
///
/// Compute the backprop of both data and weights in a RNN.
///
/// rnn_mode: Indicates the type of the RNN model.
/// input_mode: Indicate whether there is a linear projection between the input and
///     the actual computation before the first layer. 'skip_input' is only allowed
///     when input_size == num_units; 'auto_select' implies 'skip_input' when
///     input_size == num_units; otherwise, it implies 'linear_input'.
/// direction: Indicates whether a bidirectional model will be used. Should be
///   "unidirectional" or "bidirectional".
/// dropout: Dropout probability. When set to 0., dropout is disabled.
/// seed: The 1st part of a seed to initialize dropout.
/// seed2: The 2nd part of a seed to initialize dropout.
/// input: A 3-D tensor with the shape of [seq_length, batch_size, input_size].
/// input_h: A 3-D tensor with the shape of [num_layer * dir, batch_size,
///     num_units].
/// input_c: For LSTM, a 3-D tensor with the shape of
///     [num_layer * dir, batch, num_units]. For other models, it is ignored.
/// params: A 1-D tensor that contains the weights and biases in an opaque layout.
///     The size must be created through CudnnRNNParamsSize, and initialized
///     separately. Note that they might not be compatible across different
///     generations. So it is a good idea to save and restore
/// output: A 3-D tensor with the shape of [seq_length, batch_size,
///     dir * num_units].
/// output_h: The same shape has input_h.
/// output_c: The same shape as input_c for LSTM. An empty tensor for other models.
/// output_backprop: A 3-D tensor with the same shape as output in the forward pass.
/// output_h_backprop: A 3-D tensor with the same shape as output_h in the forward
///     pass.
/// output_c_backprop: A 3-D tensor with the same shape as output_c in the forward
///     pass.
/// reserve_space: The same reserve_space produced in for forward operation.
/// input_backprop: The backprop to input in the forward pass. Has the same shape
///     as input.
/// input_h_backprop: The backprop to input_h in the forward pass. Has the same
///     shape as input_h.
/// input_c_backprop: The backprop to input_c in the forward pass. Has the same
///     shape as input_c.
/// params_backprop: The backprop to the params buffer in the forward pass. Has the
///     same shape as params.
@inlinable @inline(__always)
public static func cudnnRNNBackprop<T: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>,
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
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> (inputBackprop: Tensor<T>, inputHBackprop: Tensor<T>, inputCBackprop: Tensor<T>, paramsBackprop: Tensor<T>) {
  let ret: (TensorHandle<T>, TensorHandle<T>, TensorHandle<T>, TensorHandle<T>) = #tfop("CudnnRNNBackprop",
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
    T$dtype: T.tensorFlowDataType,
    rnn_mode: rnnMode.cName,
    input_mode: inputMode.cName,
    direction: direction.cName,
    dropout: dropout,
    seed: seed,
    seed2: seed2)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2), Tensor(handle: ret.3))
}

/// Backprop step of CudnnRNN.
///
/// Compute the backprop of both data and weights in a RNN. Takes an extra
///     "host_reserved" inupt than CudnnRNNBackprop, which is used to determine RNN
///     cudnnRNNAlgo_t and cudnnMathType_t.
///
/// rnn_mode: Indicates the type of the RNN model.
/// input_mode: Indicates whether there is a linear projection between the input and
///     the actual computation before the first layer. 'skip_input' is only allowed
///     when input_size == num_units; 'auto_select' implies 'skip_input' when
///     input_size == num_units; otherwise, it implies 'linear_input'.
/// direction: Indicates whether a bidirectional model will be used. Should be
///   "unidirectional" or "bidirectional".
/// dropout: Dropout probability. When set to 0., dropout is disabled.
/// seed: The 1st part of a seed to initialize dropout.
/// seed2: The 2nd part of a seed to initialize dropout.
/// input: A 3-D tensor with the shape of [seq_length, batch_size, input_size].
/// input_h: A 3-D tensor with the shape of [num_layer * dir, batch_size,
///     num_units].
/// input_c: For LSTM, a 3-D tensor with the shape of
///     [num_layer * dir, batch, num_units]. For other models, it is ignored.
/// params: A 1-D tensor that contains the weights and biases in an opaque layout.
///     The size must be created through CudnnRNNParamsSize, and initialized
///     separately. Note that they might not be compatible across different
///     generations. So it is a good idea to save and restore
/// output: A 3-D tensor with the shape of [seq_length, batch_size,
///     dir * num_units].
/// output_h: The same shape has input_h.
/// output_c: The same shape as input_c for LSTM. An empty tensor for other models.
/// output_backprop: A 3-D tensor with the same shape as output in the forward pass.
/// output_h_backprop: A 3-D tensor with the same shape as output_h in the forward
///     pass.
/// output_c_backprop: A 3-D tensor with the same shape as output_c in the forward
///     pass.
/// reserve_space: The same reserve_space produced in the forward operation.
/// host_reserved: The same host_reserved produced in the forward operation.
/// input_backprop: The backprop to input in the forward pass. Has the same shape
///     as input.
/// input_h_backprop: The backprop to input_h in the forward pass. Has the same
///     shape as input_h.
/// input_c_backprop: The backprop to input_c in the forward pass. Has the same
///     shape as input_c.
/// params_backprop: The backprop to the params buffer in the forward pass. Has the
///     same shape as params.
@inlinable @inline(__always)
public static func cudnnRNNBackpropV2<T: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>,
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
  hostReserved: Tensor<Int8>,
  rnnMode: RnnMode = .lstm,
  inputMode: InputMode = .linearInput,
  direction: Direction = .unidirectional,
  dropout: Double = 0,
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> (inputBackprop: Tensor<T>, inputHBackprop: Tensor<T>, inputCBackprop: Tensor<T>, paramsBackprop: Tensor<T>) {
  let ret: (TensorHandle<T>, TensorHandle<T>, TensorHandle<T>, TensorHandle<T>) = #tfop("CudnnRNNBackpropV2",
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
    hostReserved,
    T$dtype: T.tensorFlowDataType,
    rnn_mode: rnnMode.cName,
    input_mode: inputMode.cName,
    direction: direction.cName,
    dropout: dropout,
    seed: seed,
    seed2: seed2)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2), Tensor(handle: ret.3))
}

/// Backprop step of CudnnRNNV3.
///
/// Compute the backprop of both data and weights in a RNN. Takes an extra
///     "sequence_lengths" input than CudnnRNNBackprop.
///
/// rnn_mode: Indicates the type of the RNN model.
/// input_mode: Indicates whether there is a linear projection between the input and
///     the actual computation before the first layer. 'skip_input' is only allowed
///     when input_size == num_units; 'auto_select' implies 'skip_input' when
///     input_size == num_units; otherwise, it implies 'linear_input'.
/// direction: Indicates whether a bidirectional model will be used. Should be
///   "unidirectional" or "bidirectional".
/// dropout: Dropout probability. When set to 0., dropout is disabled.
/// seed: The 1st part of a seed to initialize dropout.
/// seed2: The 2nd part of a seed to initialize dropout.
/// input: If time_major is true, this is a 3-D tensor with the shape of
///     [seq_length, batch_size, input_size]. If time_major is false, the shape is
///     [batch_size, seq_length, input_size].
/// input_h: If time_major is true, this is a 3-D tensor with the shape of
///     [num_layer * dir, batch_size, num_units]. If time_major is false, the shape
///     is [batch_size, num_layer * dir, num_units].
/// input_c: For LSTM, a 3-D tensor with the shape of
///     [num_layer * dir, batch, num_units]. For other models, it is ignored.
/// params: A 1-D tensor that contains the weights and biases in an opaque layout.
///     The size must be created through CudnnRNNParamsSize, and initialized
///     separately. Note that they might not be compatible across different
///     generations. So it is a good idea to save and restore
/// sequence_lengths: a vector of lengths of each input sequence.
/// output: If time_major is true, this is a 3-D tensor with the shape of
///     [seq_length, batch_size, dir * num_units]. If time_major is false, the
///     shape is [batch_size, seq_length, dir * num_units].
/// output_h: The same shape has input_h.
/// output_c: The same shape as input_c for LSTM. An empty tensor for other models.
/// output_backprop: A 3-D tensor with the same shape as output in the forward pass.
/// output_h_backprop: A 3-D tensor with the same shape as output_h in the forward
///     pass.
/// output_c_backprop: A 3-D tensor with the same shape as output_c in the forward
///     pass.
/// time_major: Indicates whether the input/output format is time major or batch
///     major.
/// reserve_space: The same reserve_space produced in the forward operation.
/// input_backprop: The backprop to input in the forward pass. Has the same shape
///     as input.
/// input_h_backprop: The backprop to input_h in the forward pass. Has the same
///     shape as input_h.
/// input_c_backprop: The backprop to input_c in the forward pass. Has the same
///     shape as input_c.
/// params_backprop: The backprop to the params buffer in the forward pass. Has the
///     same shape as params.
@inlinable @inline(__always)
public static func cudnnRNNBackpropV3<T: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>,
  inputH: Tensor<T>,
  inputC: Tensor<T>,
  params: Tensor<T>,
  sequenceLengths: Tensor<Int32>,
  output: Tensor<T>,
  outputH: Tensor<T>,
  outputC: Tensor<T>,
  outputBackprop: Tensor<T>,
  outputHBackprop: Tensor<T>,
  outputCBackprop: Tensor<T>,
  reserveSpace: Tensor<T>,
  hostReserved: Tensor<Int8>,
  rnnMode: RnnMode = .lstm,
  inputMode: InputMode = .linearInput,
  direction: Direction = .unidirectional,
  dropout: Double = 0,
  seed: Int64 = 0,
  seed2: Int64 = 0,
  timeMajor: Bool = true
) -> (inputBackprop: Tensor<T>, inputHBackprop: Tensor<T>, inputCBackprop: Tensor<T>, paramsBackprop: Tensor<T>) {
  let ret: (TensorHandle<T>, TensorHandle<T>, TensorHandle<T>, TensorHandle<T>) = #tfop("CudnnRNNBackpropV3",
    input,
    inputH,
    inputC,
    params,
    sequenceLengths,
    output,
    outputH,
    outputC,
    outputBackprop,
    outputHBackprop,
    outputCBackprop,
    reserveSpace,
    hostReserved,
    T$dtype: T.tensorFlowDataType,
    rnn_mode: rnnMode.cName,
    input_mode: inputMode.cName,
    direction: direction.cName,
    dropout: dropout,
    seed: seed,
    seed2: seed2,
    time_major: timeMajor)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2), Tensor(handle: ret.3))
}

/// Converts CudnnRNN params from canonical form to usable form.
///
/// Writes a set of weights into the opaque params buffer so they can be used in
/// upcoming training or inferences.
///
/// Note that the params buffer may not be compatible across different GPUs. So any
/// save and restoration should be converted to and from the canonical weights and
/// biases.
///
/// num_layers: Specifies the number of layers in the RNN model.
/// num_units: Specifies the size of the hidden state.
/// input_size: Specifies the size of the input state.
/// weights: the canonical form of weights that can be used for saving
///     and restoration. They are more likely to be compatible across different
///     generations.
/// biases: the canonical form of biases that can be used for saving
///     and restoration. They are more likely to be compatible across different
///     generations.
/// num_params: number of parameter sets for all layers.
///     Each layer may contain multiple parameter sets, with each set consisting of
///     a weight matrix and a bias vector.
/// rnn_mode: Indicates the type of the RNN model.
/// input_mode: Indicate whether there is a linear projection between the input and
///     The actual computation before the first layer. 'skip_input' is only allowed
///     when input_size == num_units; 'auto_select' implies 'skip_input' when
///     input_size == num_units; otherwise, it implies 'linear_input'.
/// direction: Indicates whether a bidirectional model will be used.
///     dir = (direction == bidirectional) ? 2 : 1
/// dropout: dropout probability. When set to 0., dropout is disabled.
/// seed: the 1st part of a seed to initialize dropout.
/// seed2: the 2nd part of a seed to initialize dropout.
@inlinable @inline(__always)
public static func cudnnRNNCanonicalToParams<T: FloatingPoint & TensorFlowScalar>(
  numLayers: Tensor<Int32>,
  numUnits: Tensor<Int32>,
  inputSize: Tensor<Int32>,
  weights: [Tensor<T>],
  biases: [Tensor<T>],
  rnnMode: RnnMode = .lstm,
  inputMode: InputMode = .linearInput,
  direction: Direction = .unidirectional,
  dropout: Double = 0,
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("CudnnRNNCanonicalToParams",
    numLayers,
    numUnits,
    inputSize,
    weights,
    biases,
    T$dtype: T.tensorFlowDataType,
    rnn_mode: rnnMode.cName,
    input_mode: inputMode.cName,
    direction: direction.cName,
    dropout: dropout,
    seed: seed,
    seed2: seed2)
  return Tensor(handle: ret)
}

/// Computes size of weights that can be used by a Cudnn RNN model.
///
/// Return the params size that can be used by the Cudnn RNN model. Subsequent
/// weight allocation and initialization should use this size.
///
/// num_layers: Specifies the number of layers in the RNN model.
/// num_units: Specifies the size of the hidden state.
/// input_size: Specifies the size of the input state.
/// rnn_mode: Indicates the type of the RNN model.
/// input_mode: Indicate whether there is a linear projection between the input and
///   The actual computation before the first layer. 'skip_input' is only allowed
///   when input_size == num_units; 'auto_select' implies 'skip_input' when
///   input_size == num_units; otherwise, it implies 'linear_input'.
/// direction: Indicates whether a bidirectional model will be used.
///   dir = (direction == bidirectional) ? 2 : 1
/// dropout: dropout probability. When set to 0., dropout is disabled.
/// seed: the 1st part of a seed to initialize dropout.
/// seed2: the 2nd part of a seed to initialize dropout.
/// params_size: The size of the params buffer that should be allocated and
///   initialized for this RNN model. Note that this params buffer may not be
///   compatible across GPUs. Please use CudnnRNNParamsWeights and
///   CudnnRNNParamsBiases to save and restore them in a way that is compatible
///   across different runs.
@inlinable @inline(__always)
public static func cudnnRNNParamsSize<T: FloatingPoint & TensorFlowScalar, S: BinaryInteger & TensorFlowScalar>(
  numLayers: Tensor<Int32>,
  numUnits: Tensor<Int32>,
  inputSize: Tensor<Int32>,
  rnnMode: RnnMode = .lstm,
  inputMode: InputMode = .linearInput,
  direction: Direction = .unidirectional,
  dropout: Double = 0,
  seed: Int64 = 0,
  seed2: Int64 = 0,
  typeT: T.Type
) -> Tensor<S> {
  let ret: TensorHandle<S> = #tfop("CudnnRNNParamsSize",
    numLayers,
    numUnits,
    inputSize,
    T$dtype: T.tensorFlowDataType,
    S$dtype: S.tensorFlowDataType,
    rnn_mode: rnnMode.cName,
    input_mode: inputMode.cName,
    direction: direction.cName,
    dropout: dropout,
    seed: seed,
    seed2: seed2)
  return Tensor(handle: ret)
}

/// A RNN backed by cuDNN.
///
/// Computes the RNN from the input and initial states, with respect to the params
/// buffer. Produces one extra output "host_reserved" than CudnnRNN.
///
/// rnn_mode: Indicates the type of the RNN model.
/// input_mode: Indicates whether there is a linear projection between the input and
///   the actual computation before the first layer. 'skip_input' is only allowed
///   when input_size == num_units; 'auto_select' implies 'skip_input' when
///   input_size == num_units; otherwise, it implies 'linear_input'.
/// direction: Indicates whether a bidirectional model will be used. Should be
///   "unidirectional" or "bidirectional".
/// dropout: Dropout probability. When set to 0., dropout is disabled.
/// seed: The 1st part of a seed to initialize dropout.
/// seed2: The 2nd part of a seed to initialize dropout.
/// input: A 3-D tensor with the shape of [seq_length, batch_size, input_size].
/// input_h: A 3-D tensor with the shape of [num_layer * dir, batch_size,
///     num_units].
/// input_c: For LSTM, a 3-D tensor with the shape of
///     [num_layer * dir, batch, num_units]. For other models, it is ignored.
/// params: A 1-D tensor that contains the weights and biases in an opaque layout.
///     The size must be created through CudnnRNNParamsSize, and initialized
///     separately. Note that they might not be compatible across different
///     generations. So it is a good idea to save and restore
/// output: A 3-D tensor with the shape of [seq_length, batch_size,
///     dir * num_units].
/// output_h: The same shape has input_h.
/// output_c: The same shape as input_c for LSTM. An empty tensor for other models.
/// is_training: Indicates whether this operation is used for inferenece or
///   training.
/// reserve_space: An opaque tensor that can be used in backprop calculation. It
///   is only produced if is_training is true.
/// host_reserved: An opaque tensor that can be used in backprop calculation. It is
///   only produced if is_training is true. It is output on host memory rather than
///   device memory.
@inlinable @inline(__always)
public static func cudnnRNNV2<T: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>,
  inputH: Tensor<T>,
  inputC: Tensor<T>,
  params: Tensor<T>,
  rnnMode: RnnMode = .lstm,
  inputMode: InputMode = .linearInput,
  direction: Direction = .unidirectional,
  dropout: Double = 0,
  seed: Int64 = 0,
  seed2: Int64 = 0,
  isTraining: Bool = true
) -> (output: Tensor<T>, outputH: Tensor<T>, outputC: Tensor<T>, reserveSpace: Tensor<T>, hostReserved: Tensor<Int8>) {
  let ret: (TensorHandle<T>, TensorHandle<T>, TensorHandle<T>, TensorHandle<T>, TensorHandle<Int8>) = #tfop("CudnnRNNV2",
    input,
    inputH,
    inputC,
    params,
    T$dtype: T.tensorFlowDataType,
    rnn_mode: rnnMode.cName,
    input_mode: inputMode.cName,
    direction: direction.cName,
    dropout: dropout,
    seed: seed,
    seed2: seed2,
    is_training: isTraining)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2), Tensor(handle: ret.3), Tensor(handle: ret.4))
}

/// A RNN backed by cuDNN.
///
/// Computes the RNN from the input and initial states, with respect to the params
/// buffer. Accepts one extra input "sequence_lengths" than CudnnRNN.
///
/// rnn_mode: Indicates the type of the RNN model.
/// input_mode: Indicates whether there is a linear projection between the input and
///   the actual computation before the first layer. 'skip_input' is only allowed
///   when input_size == num_units; 'auto_select' implies 'skip_input' when
///   input_size == num_units; otherwise, it implies 'linear_input'.
/// direction: Indicates whether a bidirectional model will be used. Should be
///   "unidirectional" or "bidirectional".
/// dropout: Dropout probability. When set to 0., dropout is disabled.
/// seed: The 1st part of a seed to initialize dropout.
/// seed2: The 2nd part of a seed to initialize dropout.
/// input: If time_major is true, this is a 3-D tensor with the shape of
///     [seq_length, batch_size, input_size]. If time_major is false, the shape is
///     [batch_size, seq_length, input_size].
/// input_h: If time_major is true, this is a 3-D tensor with the shape of
///     [num_layer * dir, batch_size, num_units]. If time_major is false, the shape
///     is [batch_size, num_layer * dir, num_units].
/// input_c: For LSTM, a 3-D tensor with the shape of
///     [num_layer * dir, batch, num_units]. For other models, it is ignored.
/// params: A 1-D tensor that contains the weights and biases in an opaque layout.
///     The size must be created through CudnnRNNParamsSize, and initialized
///     separately. Note that they might not be compatible across different
///     generations. So it is a good idea to save and restore
/// sequence_lengths: a vector of lengths of each input sequence.
/// output: If time_major is true, this is a 3-D tensor with the shape of
///     [seq_length, batch_size, dir * num_units]. If time_major is false, the
///     shape is [batch_size, seq_length, dir * num_units].
/// output_h: The same shape has input_h.
/// output_c: The same shape as input_c for LSTM. An empty tensor for other models.
/// is_training: Indicates whether this operation is used for inferenece or
///   training.
/// time_major: Indicates whether the input/output format is time major or batch
///     major.
/// reserve_space: An opaque tensor that can be used in backprop calculation. It
///   is only produced if is_training is true.
@inlinable @inline(__always)
public static func cudnnRNNV3<T: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>,
  inputH: Tensor<T>,
  inputC: Tensor<T>,
  params: Tensor<T>,
  sequenceLengths: Tensor<Int32>,
  rnnMode: RnnMode = .lstm,
  inputMode: InputMode = .linearInput,
  direction: Direction = .unidirectional,
  dropout: Double = 0,
  seed: Int64 = 0,
  seed2: Int64 = 0,
  isTraining: Bool = true,
  timeMajor: Bool = true
) -> (output: Tensor<T>, outputH: Tensor<T>, outputC: Tensor<T>, reserveSpace: Tensor<T>, hostReserved: Tensor<Int8>) {
  let ret: (TensorHandle<T>, TensorHandle<T>, TensorHandle<T>, TensorHandle<T>, TensorHandle<Int8>) = #tfop("CudnnRNNV3",
    input,
    inputH,
    inputC,
    params,
    sequenceLengths,
    T$dtype: T.tensorFlowDataType,
    rnn_mode: rnnMode.cName,
    input_mode: inputMode.cName,
    direction: direction.cName,
    dropout: dropout,
    seed: seed,
    seed2: seed2,
    is_training: isTraining,
    time_major: timeMajor)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2), Tensor(handle: ret.3), Tensor(handle: ret.4))
}

/// Compute the cumulative product of the tensor `x` along `axis`.
///
/// By default, this op performs an inclusive cumprod, which means that the first
/// element of the input is identical to the first element of the output:
///
/// ```python
/// tf.cumprod([a, b, c])  # => [a, a * b, a * b * c]
/// ```
///
/// By setting the `exclusive` kwarg to `True`, an exclusive cumprod is
/// performed instead:
///
/// ```python
/// tf.cumprod([a, b, c], exclusive=True)  # => [1, a, a * b]
/// ```
///
/// By setting the `reverse` kwarg to `True`, the cumprod is performed in the
/// opposite direction:
///
/// ```python
/// tf.cumprod([a, b, c], reverse=True)  # => [a * b * c, b * c, c]
/// ```
///
/// This is more efficient than using separate `tf.reverse` ops.
///
/// The `reverse` and `exclusive` kwargs can also be combined:
///
/// ```python
/// tf.cumprod([a, b, c], exclusive=True, reverse=True)  # => [b * c, c, 1]
/// ```
///
/// - Parameters:
///   - x: A `Tensor`. Must be one of the following types: `float32`, `float64`,
///     `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`,
///     `complex128`, `qint8`, `quint8`, `qint32`, `half`.
///   - axis: A `Tensor` of type `int32` (default: 0). Must be in the range
///     `[-rank(x), rank(x))`.
///
/// - Attrs:
///   - exclusive: If `True`, perform exclusive cumprod.
///   - reverse: A `bool` (default: False).
@inlinable @inline(__always)
public static func cumprod<T: Numeric & TensorFlowScalar, Tidx: BinaryInteger & TensorFlowScalar>(
  _ x: Tensor<T>,
  axis: Tensor<Tidx>,
  exclusive: Bool = false,
  reverse: Bool = false
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Cumprod",
    x,
    axis,
    T$dtype: T.tensorFlowDataType,
    Tidx$dtype: Tidx.tensorFlowDataType,
    exclusive: exclusive,
    reverse: reverse)
  return Tensor(handle: ret)
}

/// Compute the cumulative sum of the tensor `x` along `axis`.
///
/// By default, this op performs an inclusive cumsum, which means that the first
/// element of the input is identical to the first element of the output:
///
/// ```python
/// tf.cumsum([a, b, c])  # => [a, a + b, a + b + c]
/// ```
///
/// By setting the `exclusive` kwarg to `True`, an exclusive cumsum is
/// performed instead:
///
/// ```python
/// tf.cumsum([a, b, c], exclusive=True)  # => [0, a, a + b]
/// ```
///
/// By setting the `reverse` kwarg to `True`, the cumsum is performed in the
/// opposite direction:
///
/// ```python
/// tf.cumsum([a, b, c], reverse=True)  # => [a + b + c, b + c, c]
/// ```
///
/// This is more efficient than using separate `tf.reverse` ops.
///
/// The `reverse` and `exclusive` kwargs can also be combined:
///
/// ```python
/// tf.cumsum([a, b, c], exclusive=True, reverse=True)  # => [b + c, c, 0]
/// ```
///
/// - Parameters:
///   - x: A `Tensor`. Must be one of the following types: `float32`, `float64`,
///     `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`,
///     `complex128`, `qint8`, `quint8`, `qint32`, `half`.
///   - axis: A `Tensor` of type `int32` (default: 0). Must be in the range
///     `[-rank(x), rank(x))`.
///
/// - Attrs:
///   - exclusive: If `True`, perform exclusive cumsum.
///   - reverse: A `bool` (default: False).
@inlinable @inline(__always)
public static func cumsum<T: Numeric & TensorFlowScalar, Tidx: BinaryInteger & TensorFlowScalar>(
  _ x: Tensor<T>,
  axis: Tensor<Tidx>,
  exclusive: Bool = false,
  reverse: Bool = false
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Cumsum",
    x,
    axis,
    T$dtype: T.tensorFlowDataType,
    Tidx$dtype: Tidx.tensorFlowDataType,
    exclusive: exclusive,
    reverse: reverse)
  return Tensor(handle: ret)
}

/// Returns the dimension index in the destination data format given the one in
///
/// the source data format.
///
/// - Parameter x: A Tensor with each element as a dimension index in source data format.
///   Must be in the range [-4, 4).
///
/// - Attrs:
///   - src_format: source data format.
///   - dst_format: destination data format.
///
/// - Output y: A Tensor with each element as a dimension index in destination data format.
@inlinable @inline(__always)
public static func dataFormatDimMap<T: BinaryInteger & TensorFlowScalar>(
  _ x: Tensor<T>,
  srcFormat: String = "NHWC",
  dstFormat: String = "NCHW"
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("DataFormatDimMap",
    x,
    T$dtype: T.tensorFlowDataType,
    src_format: srcFormat,
    dst_format: dstFormat)
  return Tensor(handle: ret)
}

/// Returns the permuted vector/tensor in the destination data format given the
///
/// one in the source data format.
///
/// - Parameter x: Vector of size 4 or Tensor of shape (4, 2) in source data format.
///
/// - Attrs:
///   - src_format: source data format.
///   - dst_format: destination data format.
///
/// - Output y: Vector of size 4 or Tensor of shape (4, 2) in destination data format.
@inlinable @inline(__always)
public static func dataFormatVecPermute<T: BinaryInteger & TensorFlowScalar>(
  _ x: Tensor<T>,
  srcFormat: String = "NHWC",
  dstFormat: String = "NCHW"
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("DataFormatVecPermute",
    x,
    T$dtype: T.tensorFlowDataType,
    src_format: srcFormat,
    dst_format: dstFormat)
  return Tensor(handle: ret)
}

/// Identity op for gradient debugging.
///
/// This op is hidden from public in Python. It is used by TensorFlow Debugger to
/// register gradient tensors for gradient debugging.
/// This op operates on non-reference-type tensors.
@inlinable @inline(__always)
public static func debugGradientIdentity<T: TensorFlowScalar>(
  _ input: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("DebugGradientIdentity",
    input,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Debug Identity Op.
///
/// Provides an identity mapping of the non-Ref type input tensor for debugging.
///
/// - Parameter input: Input tensor, non-Reference type.
///
/// - Attrs:
///   - tensor_name: Name of the input tensor.
///   - debug_urls: List of URLs to debug targets, e.g.,
///     file:///foo/tfdbg_dump, grpc:://localhost:11011
///   - gated_grpc: Whether this op will be gated. If any of the debug_urls of this
///     debug node is of the grpc:// scheme, when the value of this attribute is set
///     to True, the data will not actually be sent via the grpc stream unless this
///     debug op has been enabled at the debug_url. If all of the debug_urls of this
///     debug node are of the grpc:// scheme and the debug op is enabled at none of
///     them, the output will be an empty Tensor.
///
/// - Output output: Output tensor that equals the input tensor.
@inlinable @inline(__always)
public static func debugIdentity<T: TensorFlowScalar>(
  _ input: Tensor<T>,
  deviceName: String,
  tensorName: String,
  debugUrls: [String],
  gatedGrpc: Bool = false
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("DebugIdentity",
    input,
    T$dtype: T.tensorFlowDataType,
    device_name: deviceName,
    tensor_name: tensorName,
    debug_urls: debugUrls,
    gated_grpc: gatedGrpc)
  return Tensor(handle: ret)
}

/// Debug NaN Value Counter Op
///
/// Counts number of NaNs in the input tensor, for debugging.
///
/// - Parameter input: Input tensor, non-Reference type.
///
/// - Attrs:
///   - tensor_name: Name of the input tensor.
///   - debug_urls: List of URLs to debug targets, e.g.,
///     file:///foo/tfdbg_dump, grpc:://localhost:11011.
///   - gated_grpc: Whether this op will be gated. If any of the debug_urls of this
///     debug node is of the grpc:// scheme, when the value of this attribute is set
///     to True, the data will not actually be sent via the grpc stream unless this
///     debug op has been enabled at the debug_url. If all of the debug_urls of this
///     debug node are of the grpc:// scheme and the debug op is enabled at none of
///     them, the output will be an empty Tensor.
///
/// - Output output: An integer output tensor that is the number of NaNs in the input.
@inlinable @inline(__always)
public static func debugNanCount<T: TensorFlowScalar>(
  _ input: Tensor<T>,
  deviceName: String,
  tensorName: String,
  debugUrls: [String],
  gatedGrpc: Bool = false
) -> Tensor<Int64> {
  let ret: TensorHandle<Int64> = #tfop("DebugNanCount",
    input,
    T$dtype: T.tensorFlowDataType,
    device_name: deviceName,
    tensor_name: tensorName,
    debug_urls: debugUrls,
    gated_grpc: gatedGrpc)
  return Tensor(handle: ret)
}

/// Debug Numeric Summary Op.
///
/// Provide a basic summary of numeric value types, range and distribution.
///
/// - Parameter input: Input tensor, non-Reference type, float or double.
///
/// - Attrs:
///   - tensor_name: Name of the input tensor.
///   - debug_urls: List of URLs to debug targets, e.g.,
///     file:///foo/tfdbg_dump, grpc:://localhost:11011
///   - lower_bound: (float) The lower bound <= which values will be included in the
///     generalized -inf count. Default: -inf.
///   - upper_bound: (float) The upper bound >= which values will be included in the
///     generalized +inf count. Default: +inf.
///   - mute_if_healthy: (bool) Do not send data to the debug URLs unless at least one
///     of elements [2], [3] and [7] (i.e., the nan count and the generalized -inf and
///     inf counts) is non-zero.
///   - gated_grpc: Whether this op will be gated. If any of the debug_urls of this
///     debug node is of the grpc:// scheme, when the value of this attribute is set
///     to True, the data will not actually be sent via the grpc stream unless this
///     debug op has been enabled at the debug_url. If all of the debug_urls of this
///     debug node are of the grpc:// scheme and the debug op is enabled at none of
///     them, the output will be an empty Tensor.
///
/// - Output output: A double tensor of shape [14 + nDimensions], where nDimensions is the
///     the number of dimensions of the tensor's shape. The elements of output are:
///     [0]: is initialized (1.0) or not (0.0).
///     [1]: total number of elements
///     [2]: NaN element count
///     [3]: generalized -inf count: elements <= lower_bound. lower_bound is -inf by
///       default.
///     [4]: negative element count (excluding -inf), if lower_bound is the default
///       -inf. Otherwise, this is the count of elements > lower_bound and < 0.
///     [5]: zero element count
///     [6]: positive element count (excluding +inf), if upper_bound is the default
///       -inf. Otherwise, this is the count of elements < upper_bound and > 0.
///     [7]: generalized +inf count, elements >= upper_bound. upper_bound is +inf by
///       default.
///   Output elements [1:8] are all zero, if the tensor is uninitialized.
///     [8]: minimum of all non-inf and non-NaN elements.
///          If uninitialized or no such element exists: +inf.
///     [9]: maximum of all non-inf and non-NaN elements.
///          If uninitialized or no such element exists: -inf.
///     [10]: mean of all non-inf and non-NaN elements.
///           If uninitialized or no such element exists: NaN.
///     [11]: variance of all non-inf and non-NaN elements.
///           If uninitialized or no such element exists: NaN.
///     [12]: Data type of the tensor encoded as an enum integer. See the DataType
///           proto for more details.
///     [13]: Number of dimensions of the tensor (ndims).
///     [14+]: Sizes of the dimensions.
@inlinable @inline(__always)
public static func debugNumericSummary<T: TensorFlowScalar>(
  _ input: Tensor<T>,
  deviceName: String,
  tensorName: String,
  debugUrls: [String],
  lowerBound: Double = -Double.infinity,
  upperBound: Double = Double.infinity,
  muteIfHealthy: Bool = false,
  gatedGrpc: Bool = false
) -> Tensor<Double> {
  let ret: TensorHandle<Double> = #tfop("DebugNumericSummary",
    input,
    T$dtype: T.tensorFlowDataType,
    device_name: deviceName,
    tensor_name: tensorName,
    debug_urls: debugUrls,
    lower_bound: lowerBound,
    upper_bound: upperBound,
    mute_if_healthy: muteIfHealthy,
    gated_grpc: gatedGrpc)
  return Tensor(handle: ret)
}

/// Decode and Crop a JPEG-encoded image to a uint8 tensor.
///
/// The attr `channels` indicates the desired number of color channels for the
/// decoded image.
///
/// Accepted values are:
///
/// *   0: Use the number of channels in the JPEG-encoded image.
/// *   1: output a grayscale image.
/// *   3: output an RGB image.
///
/// If needed, the JPEG-encoded image is transformed to match the requested number
/// of color channels.
///
/// The attr `ratio` allows downscaling the image by an integer factor during
/// decoding.  Allowed values are: 1, 2, 4, and 8.  This is much faster than
/// downscaling the image later.
///
///
/// It is equivalent to a combination of decode and crop, but much faster by only
/// decoding partial jpeg image.
///
/// - Parameters:
///   - contents: 0-D.  The JPEG-encoded image.
///   - crop_window: 1-D.  The crop window: [crop_y, crop_x, crop_height, crop_width].
///
/// - Attrs:
///   - channels: Number of color channels for the decoded image.
///   - ratio: Downscaling ratio.
///   - fancy_upscaling: If true use a slower but nicer upscaling of the
///     chroma planes (yuv420/422 only).
///   - try_recover_truncated: If true try to recover an image from truncated input.
///   - acceptable_fraction: The minimum required fraction of lines before a truncated
///     input is accepted.
///   - dct_method: string specifying a hint about the algorithm used for
///     decompression.  Defaults to "" which maps to a system-specific
///     default.  Currently valid values are ["INTEGER_FAST",
///     "INTEGER_ACCURATE"].  The hint may be ignored (e.g., the internal
///     jpeg library changes to a version that does not have that specific
///     option.)
///
/// - Output image: 3-D with shape `[height, width, channels]`..
@inlinable @inline(__always)
public static func decodeAndCropJpeg(
  contents: StringTensor,
  cropWindow: Tensor<Int32>,
  channels: Int64 = 0,
  ratio: Int64 = 1,
  fancyUpscaling: Bool = true,
  tryRecoverTruncated: Bool = false,
  acceptableFraction: Double = 1,
  dctMethod: String
) -> Tensor<UInt8> {
  let ret: TensorHandle<UInt8> = #tfop("DecodeAndCropJpeg",
    contents,
    cropWindow,
    channels: channels,
    ratio: ratio,
    fancy_upscaling: fancyUpscaling,
    try_recover_truncated: tryRecoverTruncated,
    acceptable_fraction: acceptableFraction,
    dct_method: dctMethod)
  return Tensor(handle: ret)
}

/// Decode web-safe base64-encoded strings.
///
/// Input may or may not have padding at the end. See EncodeBase64 for padding.
/// Web-safe means that input must use - and _ instead of + and /.
///
/// - Parameter input: Base64 strings to decode.
///
/// - Output output: Decoded strings.
@inlinable @inline(__always)
public static func decodeBase64(
  _ input: StringTensor
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("DecodeBase64",
    input)
  return StringTensor(handle: ret)
}

/// Decode the first frame of a BMP-encoded image to a uint8 tensor.
///
/// The attr `channels` indicates the desired number of color channels for the
/// decoded image.
///
/// Accepted values are:
///
/// *   0: Use the number of channels in the BMP-encoded image.
/// *   3: output an RGB image.
/// *   4: output an RGBA image.
///
/// - Parameter contents: 0-D.  The BMP-encoded image.
///
/// - Output image: 3-D with shape `[height, width, channels]`. RGB order
@inlinable @inline(__always)
public static func decodeBmp(
  contents: StringTensor,
  channels: Int64 = 0
) -> Tensor<UInt8> {
  let ret: TensorHandle<UInt8> = #tfop("DecodeBmp",
    contents,
    channels: channels)
  return Tensor(handle: ret)
}

/// Decompress strings.
///
/// This op decompresses each element of the `bytes` input `Tensor`, which
/// is assumed to be compressed using the given `compression_type`.
///
/// The `output` is a string `Tensor` of the same shape as `bytes`,
/// each element containing the decompressed data from the corresponding
/// element in `bytes`.
///
/// - Parameter bytes: A Tensor of string which is compressed.
///
/// - Attr compression_type: A scalar containing either (i) the empty string (no
///   compression), (ii) "ZLIB", or (iii) "GZIP".
///
/// - Output output: A Tensor with the same shape as input `bytes`, uncompressed
///   from bytes.
@inlinable @inline(__always)
public static func decodeCompressed(
  bytes: StringTensor,
  compressionType: String
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("DecodeCompressed",
    bytes,
    compression_type: compressionType)
  return StringTensor(handle: ret)
}

/// Decode the frame(s) of a GIF-encoded image to a uint8 tensor.
///
/// GIF images with frame or transparency compression are not supported.
/// On Linux and MacOS systems, convert animated GIFs from compressed to
/// uncompressed by running:
///
///     convert $src.gif -coalesce $dst.gif
///
/// This op also supports decoding JPEGs and PNGs, though it is cleaner to use
/// `tf.image.decode_image`.
///
/// - Parameter contents: 0-D.  The GIF-encoded image.
///
/// - Output image: 4-D with shape `[num_frames, height, width, 3]`. RGB channel order.
@inlinable @inline(__always)
public static func decodeGif(
  contents: StringTensor
) -> Tensor<UInt8> {
  let ret: TensorHandle<UInt8> = #tfop("DecodeGif",
    contents)
  return Tensor(handle: ret)
}

/// Convert JSON-encoded Example records to binary protocol buffer strings.
///
/// This op translates a tensor containing Example records, encoded using
/// the [standard JSON
/// mapping](https://developers.google.com/protocol-buffers/docs/proto3#json),
/// into a tensor containing the same records encoded as binary protocol
/// buffers. The resulting tensor can then be fed to any of the other
/// Example-parsing ops.
///
/// - Parameter json_examples: Each string is a JSON object serialized according to the JSON
///   mapping of the Example proto.
///
/// - Output binary_examples: Each string is a binary Example protocol buffer corresponding
///   to the respective element of `json_examples`.
@inlinable @inline(__always)
public static func decodeJSONExample(
  jsonExamples: StringTensor
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("DecodeJSONExample",
    jsonExamples)
  return StringTensor(handle: ret)
}

/// Decode a JPEG-encoded image to a uint8 tensor.
///
/// The attr `channels` indicates the desired number of color channels for the
/// decoded image.
///
/// Accepted values are:
///
/// *   0: Use the number of channels in the JPEG-encoded image.
/// *   1: output a grayscale image.
/// *   3: output an RGB image.
///
/// If needed, the JPEG-encoded image is transformed to match the requested number
/// of color channels.
///
/// The attr `ratio` allows downscaling the image by an integer factor during
/// decoding.  Allowed values are: 1, 2, 4, and 8.  This is much faster than
/// downscaling the image later.
///
///
/// This op also supports decoding PNGs and non-animated GIFs since the interface is
/// the same, though it is cleaner to use `tf.image.decode_image`.
///
/// - Parameter contents: 0-D.  The JPEG-encoded image.
///
/// - Attrs:
///   - channels: Number of color channels for the decoded image.
///   - ratio: Downscaling ratio.
///   - fancy_upscaling: If true use a slower but nicer upscaling of the
///     chroma planes (yuv420/422 only).
///   - try_recover_truncated: If true try to recover an image from truncated input.
///   - acceptable_fraction: The minimum required fraction of lines before a truncated
///     input is accepted.
///   - dct_method: string specifying a hint about the algorithm used for
///     decompression.  Defaults to "" which maps to a system-specific
///     default.  Currently valid values are ["INTEGER_FAST",
///     "INTEGER_ACCURATE"].  The hint may be ignored (e.g., the internal
///     jpeg library changes to a version that does not have that specific
///     option.)
///
/// - Output image: 3-D with shape `[height, width, channels]`..
@inlinable @inline(__always)
public static func decodeJpeg(
  contents: StringTensor,
  channels: Int64 = 0,
  ratio: Int64 = 1,
  fancyUpscaling: Bool = true,
  tryRecoverTruncated: Bool = false,
  acceptableFraction: Double = 1,
  dctMethod: String
) -> Tensor<UInt8> {
  let ret: TensorHandle<UInt8> = #tfop("DecodeJpeg",
    contents,
    channels: channels,
    ratio: ratio,
    fancy_upscaling: fancyUpscaling,
    try_recover_truncated: tryRecoverTruncated,
    acceptable_fraction: acceptableFraction,
    dct_method: dctMethod)
  return Tensor(handle: ret)
}

/// Decode a PNG-encoded image to a uint8 or uint16 tensor.
///
/// The attr `channels` indicates the desired number of color channels for the
/// decoded image.
///
/// Accepted values are:
///
/// *   0: Use the number of channels in the PNG-encoded image.
/// *   1: output a grayscale image.
/// *   3: output an RGB image.
/// *   4: output an RGBA image.
///
/// If needed, the PNG-encoded image is transformed to match the requested number
/// of color channels.
///
/// This op also supports decoding JPEGs and non-animated GIFs since the interface
/// is the same, though it is cleaner to use `tf.image.decode_image`.
///
/// - Parameter contents: 0-D.  The PNG-encoded image.
///
/// - Attr channels: Number of color channels for the decoded image.
///
/// - Output image: 3-D with shape `[height, width, channels]`.
@inlinable @inline(__always)
public static func decodePng<Dtype: UnsignedInteger & TensorFlowScalar>(
  contents: StringTensor,
  channels: Int64 = 0
) -> Tensor<Dtype> {
  let ret: TensorHandle<Dtype> = #tfop("DecodePng",
    contents,
    dtype$dtype: Dtype.tensorFlowDataType,
    channels: channels)
  return Tensor(handle: ret)
}

/// Reinterpret the bytes of a string as a vector of numbers.
///
/// - Parameter bytes: All the elements must have the same length.
///
/// - Attr little_endian: Whether the input `bytes` are in little-endian order.
///   Ignored for `out_type` values that are stored in a single byte like
///   `uint8`.
///
/// - Output output: A Tensor with one more dimension than the input `bytes`.  The
///   added dimension will have size equal to the length of the elements
///   of `bytes` divided by the number of bytes to represent `out_type`.
@inlinable @inline(__always)
public static func decodeRaw<OutType: TensorFlowScalar>(
  bytes: StringTensor,
  littleEndian: Bool = true
) -> Tensor<OutType> {
  let ret: TensorHandle<OutType> = #tfop("DecodeRaw",
    bytes,
    out_type$dtype: OutType.tensorFlowDataType,
    little_endian: littleEndian)
  return Tensor(handle: ret)
}

/// Decode a 16-bit PCM WAV file to a float tensor.
///
/// The -32768 to 32767 signed 16-bit values will be scaled to -1.0 to 1.0 in float.
///
/// When desired_channels is set, if the input contains fewer channels than this
/// then the last channel will be duplicated to give the requested number, else if
/// the input has more channels than requested then the additional channels will be
/// ignored.
///
/// If desired_samples is set, then the audio will be cropped or padded with zeroes
/// to the requested length.
///
/// The first output contains a Tensor with the content of the audio samples. The
/// lowest dimension will be the number of channels, and the second will be the
/// number of samples. For example, a ten-sample-long stereo WAV file should give an
/// output shape of [10, 2].
///
/// - Parameter contents: The WAV-encoded audio, usually from a file.
///
/// - Attrs:
///   - desired_channels: Number of sample channels wanted.
///   - desired_samples: Length of audio requested.
///
/// - Outputs:
///   - audio: 2-D with shape `[length, channels]`.
///   - sample_rate: Scalar holding the sample rate found in the WAV header.
@inlinable @inline(__always)
public static func decodeWav(
  contents: StringTensor,
  desiredChannels: Int64 = -1,
  desiredSamples: Int64 = -1
) -> (audio: Tensor<Float>, sampleRate: Tensor<Int32>) {
  let ret: (TensorHandle<Float>, TensorHandle<Int32>) = #tfop("DecodeWav",
    contents,
    desired_channels: desiredChannels,
    desired_samples: desiredSamples)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

/// Makes a copy of `x`.
///
/// - Parameter x: The source tensor of type `T`.
///
/// - Output y:     y: A `Tensor` of type `T`. A copy of `x`. Guaranteed that `y`
///         is not an alias of `x`.
@inlinable @inline(__always)
public static func deepCopy<T: TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("DeepCopy",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Delete the tensor specified by its handle in the session.
///
/// - Parameter handle: The handle for a tensor stored in the session state.
@inlinable @inline(__always)
public static func deleteSessionTensor(
  handle: StringTensor
) {
  return #tfop("DeleteSessionTensor",
    handle)
}

/// Applies set operation along last dimension of 2 `Tensor` inputs.
///
/// See SetOperationOp::SetOperationFromContext for values of `set_operation`.
///
/// Output `result` is a `SparseTensor` represented by `result_indices`,
/// `result_values`, and `result_shape`. For `set1` and `set2` ranked `n`, this
/// has rank `n` and the same 1st `n-1` dimensions as `set1` and `set2`. The `nth`
/// dimension contains the result of `set_operation` applied to the corresponding
/// `[0...n-1]` dimension of `set`.
///
/// - Parameters:
///   - set1: `Tensor` with rank `n`. 1st `n-1` dimensions must be the same as `set2`.
///     Dimension `n` contains values in a set, duplicates are allowed but ignored.
///   - set2: `Tensor` with rank `n`. 1st `n-1` dimensions must be the same as `set1`.
///     Dimension `n` contains values in a set, duplicates are allowed but ignored.
///
/// - Outputs:
///   - result_indices: 2D indices of a `SparseTensor`.
///   - result_values: 1D values of a `SparseTensor`.
///   - result_shape: 1D `Tensor` shape of a `SparseTensor`. `result_shape[0...n-1]` is
///     the same as the 1st `n-1` dimensions of `set1` and `set2`, `result_shape[n]`
///     is the max result set size across all `0...n-1` dimensions.
@inlinable @inline(__always)
public static func denseToDenseSetOperation<T: BinaryInteger & TensorFlowScalar>(
  set1: Tensor<T>,
  set2: Tensor<T>,
  setOperation: String,
  validateIndices: Bool = true
) -> (resultIndices: Tensor<Int64>, resultValues: Tensor<T>, resultShape: Tensor<Int64>) {
  let ret: (TensorHandle<Int64>, TensorHandle<T>, TensorHandle<Int64>) = #tfop("DenseToDenseSetOperation",
    set1,
    set2,
    T$dtype: T.tensorFlowDataType,
    set_operation: setOperation,
    validate_indices: validateIndices)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Applies set operation along last dimension of `Tensor` and `SparseTensor`.
///
/// See SetOperationOp::SetOperationFromContext for values of `set_operation`.
///
/// Input `set2` is a `SparseTensor` represented by `set2_indices`, `set2_values`,
/// and `set2_shape`. For `set2` ranked `n`, 1st `n-1` dimensions must be the same
/// as `set1`. Dimension `n` contains values in a set, duplicates are allowed but
/// ignored.
///
/// If `validate_indices` is `True`, this op validates the order and range of `set2`
/// indices.
///
/// Output `result` is a `SparseTensor` represented by `result_indices`,
/// `result_values`, and `result_shape`. For `set1` and `set2` ranked `n`, this
/// has rank `n` and the same 1st `n-1` dimensions as `set1` and `set2`. The `nth`
/// dimension contains the result of `set_operation` applied to the corresponding
/// `[0...n-1]` dimension of `set`.
///
/// - Parameters:
///   - set1: `Tensor` with rank `n`. 1st `n-1` dimensions must be the same as `set2`.
///     Dimension `n` contains values in a set, duplicates are allowed but ignored.
///   - set2_indices: 2D `Tensor`, indices of a `SparseTensor`. Must be in row-major
///     order.
///   - set2_values: 1D `Tensor`, values of a `SparseTensor`. Must be in row-major
///     order.
///   - set2_shape: 1D `Tensor`, shape of a `SparseTensor`. `set2_shape[0...n-1]` must
///     be the same as the 1st `n-1` dimensions of `set1`, `result_shape[n]` is the
///     max set size across `n-1` dimensions.
///
/// - Outputs:
///   - result_indices: 2D indices of a `SparseTensor`.
///   - result_values: 1D values of a `SparseTensor`.
///   - result_shape: 1D `Tensor` shape of a `SparseTensor`. `result_shape[0...n-1]` is
///     the same as the 1st `n-1` dimensions of `set1` and `set2`, `result_shape[n]`
///     is the max result set size across all `0...n-1` dimensions.
@inlinable @inline(__always)
public static func denseToSparseSetOperation<T: BinaryInteger & TensorFlowScalar>(
  set1: Tensor<T>,
  set2Indices: Tensor<Int64>,
  set2Values: Tensor<T>,
  set2Shape: Tensor<Int64>,
  setOperation: String,
  validateIndices: Bool = true
) -> (resultIndices: Tensor<Int64>, resultValues: Tensor<T>, resultShape: Tensor<Int64>) {
  let ret: (TensorHandle<Int64>, TensorHandle<T>, TensorHandle<Int64>) = #tfop("DenseToSparseSetOperation",
    set1,
    set2Indices,
    set2Values,
    set2Shape,
    T$dtype: T.tensorFlowDataType,
    set_operation: setOperation,
    validate_indices: validateIndices)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// DepthToSpace for tensors of type T.
///
/// Rearranges data from depth into blocks of spatial data.
/// This is the reverse transformation of SpaceToDepth. More specifically,
/// this op outputs a copy of the input tensor where values from the `depth`
/// dimension are moved in spatial blocks to the `height` and `width` dimensions.
/// The attr `block_size` indicates the input block size and how the data is moved.
///
///   * Chunks of data of size `block_size * block_size` from depth are rearranged
///     into non-overlapping blocks of size `block_size x block_size`
///   * The width the output tensor is `input_depth * block_size`, whereas the
///     height is `input_height * block_size`.
///   * The Y, X coordinates within each block of the output image are determined
///     by the high order component of the input channel index.
///   * The depth of the input tensor must be divisible by
///     `block_size * block_size`.
///
/// The `data_format` attr specifies the layout of the input and output tensors
/// with the following options:
///   "NHWC": `[ batch, height, width, channels ]`
///   "NCHW": `[ batch, channels, height, width ]`
///   "NCHW_VECT_C":
///       `qint8 [ batch, channels / 4, height, width, 4 ]`
///
/// It is useful to consider the operation as transforming a 6-D Tensor.
/// e.g. for data_format = NHWC,
///      Each element in the input tensor can be specified via 6 coordinates,
///      ordered by decreasing memory layout significance as:
///      n,iY,iX,bY,bX,oC  (where n=batch index, iX, iY means X or Y coordinates
///                         within the input image, bX, bY means coordinates
///                         within the output block, oC means output channels).
///      The output would be the input transposed to the following layout:
///      n,iY,bY,iX,bX,oC
///
/// This operation is useful for resizing the activations between convolutions
/// (but keeping all data), e.g. instead of pooling. It is also useful for training
/// purely convolutional models.
///
/// For example, given an input of shape `[1, 1, 1, 4]`, data_format = "NHWC" and
/// block_size = 2:
///
/// ```
/// x = [[[[1, 2, 3, 4]]]]
///
/// ```
///
/// This operation will output a tensor of shape `[1, 2, 2, 1]`:
///
/// ```
///    [[[[1], [2]],
///      [[3], [4]]]]
/// ```
///
/// Here, the input has a batch of 1 and each batch element has shape `[1, 1, 4]`,
/// the corresponding output will have 2x2 elements and will have a depth of
/// 1 channel (1 = `4 / (block_size * block_size)`).
/// The output element shape is `[2, 2, 1]`.
///
/// For an input tensor with larger depth, here of shape `[1, 1, 1, 12]`, e.g.
///
/// ```
/// x = [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
/// ```
///
/// This operation, for block size of 2, will return the following tensor of shape
/// `[1, 2, 2, 3]`
///
/// ```
///    [[[[1, 2, 3], [4, 5, 6]],
///      [[7, 8, 9], [10, 11, 12]]]]
///
/// ```
///
/// Similarly, for the following input of shape `[1 2 2 4]`, and a block size of 2:
///
/// ```
/// x =  [[[[1, 2, 3, 4],
///        [5, 6, 7, 8]],
///       [[9, 10, 11, 12],
///        [13, 14, 15, 16]]]]
/// ```
///
/// the operator will return the following tensor of shape `[1 4 4 1]`:
///
/// ```
/// x = [[[ [1],   [2],  [5],  [6]],
///       [ [3],   [4],  [7],  [8]],
///       [ [9],  [10], [13],  [14]],
///       [ [11], [12], [15],  [16]]]]
///
/// ```
///
/// - Attr block_size: The size of the spatial block, same as in Space2Depth.
@inlinable @inline(__always)
public static func depthToSpace<T: TensorFlowScalar>(
  _ input: Tensor<T>,
  blockSize: Int64,
  dataFormat: DataFormat4 = .nhwc
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("DepthToSpace",
    input,
    T$dtype: T.tensorFlowDataType,
    block_size: blockSize,
    data_format: dataFormat.cName)
  return Tensor(handle: ret)
}

/// Computes a 2-D depthwise convolution given 4-D `input` and `filter` tensors.
///
/// Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
/// and a filter / kernel tensor of shape
/// `[filter_height, filter_width, in_channels, channel_multiplier]`, containing
/// `in_channels` convolutional filters of depth 1, `depthwise_conv2d` applies
/// a different filter to each input channel (expanding from 1 channel to
/// `channel_multiplier` channels for each), then concatenates the results
/// together. Thus, the output has `in_channels * channel_multiplier` channels.
///
/// ```
/// for k in 0..in_channels-1
///   for q in 0..channel_multiplier-1
///     output[b, i, j, k * channel_multiplier + q] =
///       sum_{di, dj} input[b, strides[1] * i + di, strides[2] * j + dj, k] *
///                         filter[di, dj, k, q]
/// ```
///
/// Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
/// horizontal and vertices strides, `strides = [1, stride, stride, 1]`.
///
/// - Attrs:
///   - strides: 1-D of length 4.  The stride of the sliding window for each dimension
///     of `input`.
///   - padding: The type of padding algorithm to use.
///   - data_format: Specify the data format of the input and output data. With the
///     default format "NHWC", the data is stored in the order of:
///         [batch, height, width, channels].
///     Alternatively, the format could be "NCHW", the data storage order of:
///         [batch, channels, height, width].
///   - dilations: 1-D tensor of length 4.  The dilation factor for each dimension of
///     `input`. If set to k > 1, there will be k-1 skipped cells between each filter
///     element on that dimension. The dimension order is determined by the value of
///     `data_format`, see above for details. Dilations in the batch and depth
///     dimensions must be 1.
@inlinable @inline(__always)
public static func depthwiseConv2dNative<T: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>,
  filter: Tensor<T>,
  strides: [Int32],
  padding: Padding,
  dataFormat: DataFormat = .nhwc,
  dilations: [Int32] = [1, 1, 1, 1]
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("DepthwiseConv2dNative",
    input,
    filter,
    T$dtype: T.tensorFlowDataType,
    strides: strides,
    padding: padding.cName,
    data_format: dataFormat.cName,
    dilations: dilations)
  return Tensor(handle: ret)
}

/// Computes the gradients of depthwise convolution with respect to the filter.
///
/// - Parameters:
///   - input: 4-D with shape based on `data_format`.  For example, if
///     `data_format` is 'NHWC' then `input` is a 4-D `[batch, in_height,
///     in_width, in_channels]` tensor.
///   - filter_sizes: An integer vector representing the tensor shape of `filter`,
///     where `filter` is a 4-D
///     `[filter_height, filter_width, in_channels, depthwise_multiplier]` tensor.
///   - out_backprop: 4-D with shape  based on `data_format`.
///     For example, if `data_format` is 'NHWC' then
///     out_backprop shape is `[batch, out_height, out_width, out_channels]`.
///     Gradients w.r.t. the output of the convolution.
///
/// - Attrs:
///   - strides: The stride of the sliding window for each dimension of the input
///     of the convolution.
///   - padding: The type of padding algorithm to use.
///   - data_format: Specify the data format of the input and output data. With the
///     default format "NHWC", the data is stored in the order of:
///         [batch, height, width, channels].
///     Alternatively, the format could be "NCHW", the data storage order of:
///         [batch, channels, height, width].
///   - dilations: 1-D tensor of length 4.  The dilation factor for each dimension of
///     `input`. If set to k > 1, there will be k-1 skipped cells between each filter
///     element on that dimension. The dimension order is determined by the value of
///     `data_format`, see above for details. Dilations in the batch and depth
///     dimensions must be 1.
///
/// - Output output: 4-D with shape
///   `[filter_height, filter_width, in_channels, out_channels]`.  Gradient w.r.t.
///   the `filter` input of the convolution.
@inlinable @inline(__always)
public static func depthwiseConv2dNativeBackpropFilter<T: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>,
  filterSizes: Tensor<Int32>,
  outBackprop: Tensor<T>,
  strides: [Int32],
  padding: Padding,
  dataFormat: DataFormat = .nhwc,
  dilations: [Int32] = [1, 1, 1, 1]
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("DepthwiseConv2dNativeBackpropFilter",
    input,
    filterSizes,
    outBackprop,
    T$dtype: T.tensorFlowDataType,
    strides: strides,
    padding: padding.cName,
    data_format: dataFormat.cName,
    dilations: dilations)
  return Tensor(handle: ret)
}

/// Computes the gradients of depthwise convolution with respect to the input.
///
/// - Parameters:
///   - input_sizes: An integer vector representing the shape of `input`, based
///     on `data_format`.  For example, if `data_format` is 'NHWC' then
///      `input` is a 4-D `[batch, height, width, channels]` tensor.
///   - filter: 4-D with shape
///     `[filter_height, filter_width, in_channels, depthwise_multiplier]`.
///   - out_backprop: 4-D with shape  based on `data_format`.
///     For example, if `data_format` is 'NHWC' then
///     out_backprop shape is `[batch, out_height, out_width, out_channels]`.
///     Gradients w.r.t. the output of the convolution.
///
/// - Attrs:
///   - strides: The stride of the sliding window for each dimension of the input
///     of the convolution.
///   - padding: The type of padding algorithm to use.
///   - data_format: Specify the data format of the input and output data. With the
///     default format "NHWC", the data is stored in the order of:
///         [batch, height, width, channels].
///     Alternatively, the format could be "NCHW", the data storage order of:
///         [batch, channels, height, width].
///   - dilations: 1-D tensor of length 4.  The dilation factor for each dimension of
///     `input`. If set to k > 1, there will be k-1 skipped cells between each filter
///     element on that dimension. The dimension order is determined by the value of
///     `data_format`, see above for details. Dilations in the batch and depth
///     dimensions must be 1.
///
/// - Output output: 4-D with shape according to `data_format`.  For example, if
///   `data_format` is 'NHWC', output shape is `[batch, in_height,
///   in_width, in_channels]`.  Gradient w.r.t. the input of the
///   convolution.
@inlinable @inline(__always)
public static func depthwiseConv2dNativeBackpropInput<T: FloatingPoint & TensorFlowScalar>(
  inputSizes: Tensor<Int32>,
  filter: Tensor<T>,
  outBackprop: Tensor<T>,
  strides: [Int32],
  padding: Padding,
  dataFormat: DataFormat = .nhwc,
  dilations: [Int32] = [1, 1, 1, 1]
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("DepthwiseConv2dNativeBackpropInput",
    inputSizes,
    filter,
    outBackprop,
    T$dtype: T.tensorFlowDataType,
    strides: strides,
    padding: padding.cName,
    data_format: dataFormat.cName,
    dilations: dilations)
  return Tensor(handle: ret)
}

/// Dequantize the 'input' tensor into a float Tensor.
///
/// [min_range, max_range] are scalar floats that specify the range for
/// the 'input' data. The 'mode' attribute controls exactly which calculations are
/// used to convert the float values to their quantized equivalents.
///
/// In 'MIN_COMBINED' mode, each value of the tensor will undergo the following:
///
/// ```
/// if T == qint8: in[i] += (range(T) + 1)/ 2.0
/// out[i] = min_range + (in[i]* (max_range - min_range) / range(T))
/// ```
/// here `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`
///
/// *MIN_COMBINED Mode Example*
///
/// If the input comes from a QuantizedRelu6, the output type is
/// quint8 (range of 0-255) but the possible range of QuantizedRelu6 is
/// 0-6.  The min_range and max_range values are therefore 0.0 and 6.0.
/// Dequantize on quint8 will take each value, cast to float, and multiply
/// by 6 / 255.
/// Note that if quantizedtype is qint8, the operation will additionally add
/// each value by 128 prior to casting.
///
/// If the mode is 'MIN_FIRST', then this approach is used:
///
/// ```c++
/// num_discrete_values = 1 << (# of bits in T)
/// range_adjust = num_discrete_values / (num_discrete_values - 1)
/// range = (range_max - range_min) * range_adjust
/// range_scale = range / num_discrete_values
/// const double offset_input = static_cast<double>(input) - lowest_quantized;
/// result = range_min + ((input - numeric_limits<T>::min()) * range_scale)
/// ```
///
/// *SCALED mode Example*
///
/// `SCALED` mode matches the quantization approach used in
/// `QuantizeAndDequantize{V2|V3}`.
///
/// If the mode is `SCALED`, we do not use the full range of the output type,
/// choosing to elide the lowest possible value for symmetry (e.g., output range is
/// -127 to 127, not -128 to 127 for signed 8 bit quantization), so that 0.0 maps to
/// 0.
///
/// We first find the range of values in our tensor. The
/// range we use is always centered on 0, so we find m such that
/// ```c++
///   m = max(abs(input_min), abs(input_max))
/// ```
///
/// Our input tensor range is then `[-m, m]`.
///
/// Next, we choose our fixed-point quantization buckets, `[min_fixed, max_fixed]`.
/// If T is signed, this is
/// ```
///   num_bits = sizeof(T) * 8
///   [min_fixed, max_fixed] =
///       [-(1 << (num_bits - 1) - 1), (1 << (num_bits - 1)) - 1]
/// ```
///
/// Otherwise, if T is unsigned, the fixed-point range is
/// ```
///   [min_fixed, max_fixed] = [0, (1 << num_bits) - 1]
/// ```
///
/// From this we compute our scaling factor, s:
/// ```c++
///   s = (2 * m) / (max_fixed - min_fixed)
/// ```
///
/// Now we can dequantize the elements of our tensor:
/// ```c++
/// result = input * s
/// ```
///
/// - Parameters:
///   - min_range: The minimum scalar value possibly produced for the input.
///   - max_range: The maximum scalar value possibly produced for the input.
@inlinable @inline(__always)
public static func dequantize<T: TensorFlowScalar>(
  _ input: Tensor<T>,
  minRange: Tensor<Float>,
  maxRange: Tensor<Float>,
  mode: Mode = .minCombined
) -> Tensor<Float> {
  let ret: TensorHandle<Float> = #tfop("Dequantize",
    input,
    minRange,
    maxRange,
    T$dtype: T.tensorFlowDataType,
    mode: mode.cName)
  return Tensor(handle: ret)
}

/// Deserialize and concatenate `SparseTensors` from a serialized minibatch.
///
/// The input `serialized_sparse` must be a string matrix of shape `[N x 3]` where
/// `N` is the minibatch size and the rows correspond to packed outputs of
/// `SerializeSparse`.  The ranks of the original `SparseTensor` objects
/// must all match.  When the final `SparseTensor` is created, it has rank one
/// higher than the ranks of the incoming `SparseTensor` objects
/// (they have been concatenated along a new row dimension).
///
/// The output `SparseTensor` object's shape values for all dimensions but the
/// first are the max across the input `SparseTensor` objects' shape values
/// for the corresponding dimensions.  Its first shape value is `N`, the minibatch
/// size.
///
/// The input `SparseTensor` objects' indices are assumed ordered in
/// standard lexicographic order.  If this is not the case, after this
/// step run `SparseReorder` to restore index ordering.
///
/// For example, if the serialized input is a `[2 x 3]` matrix representing two
/// original `SparseTensor` objects:
///
///     index = [ 0]
///             [10]
///             [20]
///     values = [1, 2, 3]
///     shape = [50]
///
/// and
///
///     index = [ 2]
///             [10]
///     values = [4, 5]
///     shape = [30]
///
/// then the final deserialized `SparseTensor` will be:
///
///     index = [0  0]
///             [0 10]
///             [0 20]
///             [1  2]
///             [1 10]
///     values = [1, 2, 3, 4, 5]
///     shape = [2 50]
///
/// - Parameter serialized_sparse: 2-D, The `N` serialized `SparseTensor` objects.
///   Must have 3 columns.
///
/// - Attr dtype: The `dtype` of the serialized `SparseTensor` objects.
@inlinable @inline(__always)
public static func deserializeManySparse<Dtype: TensorFlowScalar>(
  serializedSparse: StringTensor
) -> (sparseIndices: Tensor<Int64>, sparseValues: Tensor<Dtype>, sparseShape: Tensor<Int64>) {
  let ret: (TensorHandle<Int64>, TensorHandle<Dtype>, TensorHandle<Int64>) = #tfop("DeserializeManySparse",
    serializedSparse,
    dtype$dtype: Dtype.tensorFlowDataType)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Deserialize `SparseTensor` objects.
///
/// The input `serialized_sparse` must have the shape `[?, ?, ..., ?, 3]` where
/// the last dimension stores serialized `SparseTensor` objects and the other N
/// dimensions (N >= 0) correspond to a batch. The ranks of the original
/// `SparseTensor` objects must all match. When the final `SparseTensor` is
/// created, its rank is the rank of the incoming `SparseTensor` objects plus N;
/// the sparse tensors have been concatenated along new dimensions, one for each
/// batch.
///
/// The output `SparseTensor` object's shape values for the original dimensions
/// are the max across the input `SparseTensor` objects' shape values for the
/// corresponding dimensions. The new dimensions match the size of the batch.
///
/// The input `SparseTensor` objects' indices are assumed ordered in
/// standard lexicographic order.  If this is not the case, after this
/// step run `SparseReorder` to restore index ordering.
///
/// For example, if the serialized input is a `[2 x 3]` matrix representing two
/// original `SparseTensor` objects:
///
///     index = [ 0]
///             [10]
///             [20]
///     values = [1, 2, 3]
///     shape = [50]
///
/// and
///
///     index = [ 2]
///             [10]
///     values = [4, 5]
///     shape = [30]
///
/// then the final deserialized `SparseTensor` will be:
///
///     index = [0  0]
///             [0 10]
///             [0 20]
///             [1  2]
///             [1 10]
///     values = [1, 2, 3, 4, 5]
///     shape = [2 50]
///
/// - Parameter serialized_sparse: The serialized `SparseTensor` objects. The last dimension
///   must have 3 columns.
///
/// - Attr dtype: The `dtype` of the serialized `SparseTensor` objects.
@inlinable @inline(__always)
public static func deserializeSparse<Dtype: TensorFlowScalar, Tserialized: TensorFlowScalar>(
  serializedSparse: Tensor<Tserialized>
) -> (sparseIndices: Tensor<Int64>, sparseValues: Tensor<Dtype>, sparseShape: Tensor<Int64>) {
  let ret: (TensorHandle<Int64>, TensorHandle<Dtype>, TensorHandle<Int64>) = #tfop("DeserializeSparse",
    serializedSparse,
    dtype$dtype: Dtype.tensorFlowDataType,
    Tserialized$dtype: Tserialized.tensorFlowDataType)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

@inlinable @inline(__always)
public static func devicePlacementOp(
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("DevicePlacementOp")
  return StringTensor(handle: ret)
}

/// Returns a diagonal tensor with a given diagonal values.
///
/// Given a `diagonal`, this operation returns a tensor with the `diagonal` and
/// everything else padded with zeros. The diagonal is computed as follows:
///
/// Assume `diagonal` has dimensions [D1,..., Dk], then the output is a tensor of
/// rank 2k with dimensions [D1,..., Dk, D1,..., Dk] where:
///
/// `output[i1,..., ik, i1,..., ik] = diagonal[i1, ..., ik]` and 0 everywhere else.
///
/// For example:
///
/// ```
/// # 'diagonal' is [1, 2, 3, 4]
/// tf.diag(diagonal) ==> [[1, 0, 0, 0]
///                        [0, 2, 0, 0]
///                        [0, 0, 3, 0]
///                        [0, 0, 0, 4]]
/// ```
///
/// - Parameter diagonal: Rank k tensor where k is at most 1.
@inlinable @inline(__always)
public static func diag<T: Numeric & TensorFlowScalar>(
  diagonal: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Diag",
    diagonal,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Returns the diagonal part of the tensor.
///
/// This operation returns a tensor with the `diagonal` part
/// of the `input`. The `diagonal` part is computed as follows:
///
/// Assume `input` has dimensions `[D1,..., Dk, D1,..., Dk]`, then the output is a
/// tensor of rank `k` with dimensions `[D1,..., Dk]` where:
///
/// `diagonal[i1,..., ik] = input[i1, ..., ik, i1,..., ik]`.
///
/// For example:
///
/// ```
/// # 'input' is [[1, 0, 0, 0]
///               [0, 2, 0, 0]
///               [0, 0, 3, 0]
///               [0, 0, 0, 4]]
///
/// tf.diag_part(input) ==> [1, 2, 3, 4]
/// ```
///
/// - Parameter input: Rank k tensor where k is even and not zero.
///
/// - Output diagonal: The extracted diagonal.
@inlinable @inline(__always)
public static func diagPart<T: Numeric & TensorFlowScalar>(
  _ input: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("DiagPart",
    input,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes Psi, the derivative of Lgamma (the log of the absolute value of
///
/// `Gamma(x)`), element-wise.
@inlinable @inline(__always)
public static func digamma<T: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Digamma",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the grayscale dilation of 4-D `input` and 3-D `filter` tensors.
///
/// The `input` tensor has shape `[batch, in_height, in_width, depth]` and the
/// `filter` tensor has shape `[filter_height, filter_width, depth]`, i.e., each
/// input channel is processed independently of the others with its own structuring
/// function. The `output` tensor has shape
/// `[batch, out_height, out_width, depth]`. The spatial dimensions of the output
/// tensor depend on the `padding` algorithm. We currently only support the default
/// "NHWC" `data_format`.
///
/// In detail, the grayscale morphological 2-D dilation is the max-sum correlation
/// (for consistency with `conv2d`, we use unmirrored filters):
///
///     output[b, y, x, c] =
///        max_{dy, dx} input[b,
///                           strides[1] * y + rates[1] * dy,
///                           strides[2] * x + rates[2] * dx,
///                           c] +
///                     filter[dy, dx, c]
///
/// Max-pooling is a special case when the filter has size equal to the pooling
/// kernel size and contains all zeros.
///
/// Note on duality: The dilation of `input` by the `filter` is equal to the
/// negation of the erosion of `-input` by the reflected `filter`.
///
/// - Parameters:
///   - input: 4-D with shape `[batch, in_height, in_width, depth]`.
///   - filter: 3-D with shape `[filter_height, filter_width, depth]`.
///
/// - Attrs:
///   - strides: The stride of the sliding window for each dimension of the input
///     tensor. Must be: `[1, stride_height, stride_width, 1]`.
///   - rates: The input stride for atrous morphological dilation. Must be:
///     `[1, rate_height, rate_width, 1]`.
///   - padding: The type of padding algorithm to use.
///
/// - Output output: 4-D with shape `[batch, out_height, out_width, depth]`.
@inlinable @inline(__always)
public static func dilation2D<T: Numeric & TensorFlowScalar>(
  _ input: Tensor<T>,
  filter: Tensor<T>,
  strides: [Int32],
  rates: [Int32],
  padding: Padding
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Dilation2D",
    input,
    filter,
    T$dtype: T.tensorFlowDataType,
    strides: strides,
    rates: rates,
    padding: padding.cName)
  return Tensor(handle: ret)
}

/// Computes the gradient of morphological 2-D dilation with respect to the filter.
///
/// - Parameters:
///   - input: 4-D with shape `[batch, in_height, in_width, depth]`.
///   - filter: 3-D with shape `[filter_height, filter_width, depth]`.
///   - out_backprop: 4-D with shape `[batch, out_height, out_width, depth]`.
///
/// - Attrs:
///   - strides: 1-D of length 4. The stride of the sliding window for each dimension of
///     the input tensor. Must be: `[1, stride_height, stride_width, 1]`.
///   - rates: 1-D of length 4. The input stride for atrous morphological dilation.
///     Must be: `[1, rate_height, rate_width, 1]`.
///   - padding: The type of padding algorithm to use.
///
/// - Output filter_backprop: 3-D with shape `[filter_height, filter_width, depth]`.
@inlinable @inline(__always)
public static func dilation2DBackpropFilter<T: Numeric & TensorFlowScalar>(
  _ input: Tensor<T>,
  filter: Tensor<T>,
  outBackprop: Tensor<T>,
  strides: [Int32],
  rates: [Int32],
  padding: Padding
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Dilation2DBackpropFilter",
    input,
    filter,
    outBackprop,
    T$dtype: T.tensorFlowDataType,
    strides: strides,
    rates: rates,
    padding: padding.cName)
  return Tensor(handle: ret)
}

/// Computes the gradient of morphological 2-D dilation with respect to the input.
///
/// - Parameters:
///   - input: 4-D with shape `[batch, in_height, in_width, depth]`.
///   - filter: 3-D with shape `[filter_height, filter_width, depth]`.
///   - out_backprop: 4-D with shape `[batch, out_height, out_width, depth]`.
///
/// - Attrs:
///   - strides: 1-D of length 4. The stride of the sliding window for each dimension of
///     the input tensor. Must be: `[1, stride_height, stride_width, 1]`.
///   - rates: 1-D of length 4. The input stride for atrous morphological dilation.
///     Must be: `[1, rate_height, rate_width, 1]`.
///   - padding: The type of padding algorithm to use.
///
/// - Output in_backprop: 4-D with shape `[batch, in_height, in_width, depth]`.
@inlinable @inline(__always)
public static func dilation2DBackpropInput<T: Numeric & TensorFlowScalar>(
  _ input: Tensor<T>,
  filter: Tensor<T>,
  outBackprop: Tensor<T>,
  strides: [Int32],
  rates: [Int32],
  padding: Padding
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Dilation2DBackpropInput",
    input,
    filter,
    outBackprop,
    T$dtype: T.tensorFlowDataType,
    strides: strides,
    rates: rates,
    padding: padding.cName)
  return Tensor(handle: ret)
}

/// Returns x / y element-wise.
///
/// *NOTE*: `Div` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@inlinable @inline(__always)
public static func div<T: Numeric & TensorFlowScalar>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Div",
    x,
    y,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Returns 0 if the denominator is zero.
///
///
/// *NOTE*: `DivNoNan` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@inlinable @inline(__always)
public static func divNoNan<T: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("DivNoNan",
    x,
    y,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Draw bounding boxes on a batch of images.
///
/// Outputs a copy of `images` but draws on top of the pixels zero or more bounding
/// boxes specified by the locations in `boxes`. The coordinates of the each
/// bounding box in `boxes` are encoded as `[y_min, x_min, y_max, x_max]`. The
/// bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
/// height of the underlying image.
///
/// For example, if an image is 100 x 200 pixels (height x width) and the bounding
/// box is `[0.1, 0.2, 0.5, 0.9]`, the upper-left and bottom-right coordinates of
/// the bounding box will be `(40, 10)` to `(180, 50)` (in (x,y) coordinates).
///
/// Parts of the bounding box may fall outside the image.
///
/// - Parameters:
///   - images: 4-D with shape `[batch, height, width, depth]`. A batch of images.
///   - boxes: 3-D with shape `[batch, num_bounding_boxes, 4]` containing bounding
///     boxes.
///
/// - Output output: 4-D with the same shape as `images`. The batch of input images with
///   bounding boxes drawn on the images.
@inlinable @inline(__always)
public static func drawBoundingBoxes<T: FloatingPoint & TensorFlowScalar>(
  images: Tensor<T>,
  boxes: Tensor<Float>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("DrawBoundingBoxes",
    images,
    boxes,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Interleave the values from the `data` tensors into a single tensor.
///
/// Builds a merged tensor such that
///
/// ```python
///     merged[indices[m][i, ..., j], ...] = data[m][i, ..., j, ...]
/// ```
///
/// For example, if each `indices[m]` is scalar or vector, we have
///
/// ```python
///     # Scalar indices:
///     merged[indices[m], ...] = data[m][...]
///
///     # Vector indices:
///     merged[indices[m][i], ...] = data[m][i, ...]
/// ```
///
/// Each `data[i].shape` must start with the corresponding `indices[i].shape`,
/// and the rest of `data[i].shape` must be constant w.r.t. `i`.  That is, we
/// must have `data[i].shape = indices[i].shape + constant`.  In terms of this
/// `constant`, the output shape is
///
///     merged.shape = [max(indices)] + constant
///
/// Values are merged in order, so if an index appears in both `indices[m][i]` and
/// `indices[n][j]` for `(m,i) < (n,j)` the slice `data[n][j]` will appear in the
/// merged result. If you do not need this guarantee, ParallelDynamicStitch might
/// perform better on some devices.
///
/// For example:
///
/// ```python
///     indices[0] = 6
///     indices[1] = [4, 1]
///     indices[2] = [[5, 2], [0, 3]]
///     data[0] = [61, 62]
///     data[1] = [[41, 42], [11, 12]]
///     data[2] = [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]
///     merged = [[1, 2], [11, 12], [21, 22], [31, 32], [41, 42],
///               [51, 52], [61, 62]]
/// ```
///
/// This method can be used to merge partitions created by `dynamic_partition`
/// as illustrated on the following example:
///
/// ```python
///     # Apply function (increments x_i) on elements for which a certain condition
///     # apply (x_i != -1 in this example).
///     x=tf.constant([0.1, -1., 5.2, 4.3, -1., 7.4])
///     condition_mask=tf.not_equal(x,tf.constant(-1.))
///     partitioned_data = tf.dynamic_partition(
///         x, tf.cast(condition_mask, tf.int32) , 2)
///     partitioned_data[1] = partitioned_data[1] + 1.0
///     condition_indices = tf.dynamic_partition(
///         tf.range(tf.shape(x)[0]), tf.cast(condition_mask, tf.int32) , 2)
///     x = tf.dynamic_stitch(condition_indices, partitioned_data)
///     # Here x=[1.1, -1., 6.2, 5.3, -1, 8.4], the -1. values remain
///     # unchanged.
/// ```
///
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/DynamicStitch.png" alt>
/// </div>
@inlinable @inline(__always)
public static func dynamicStitch<T: TensorFlowScalar>(
  indices: [Tensor<Int32>],
  data: [Tensor<T>]
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("DynamicStitch",
    indices,
    data,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the (possibly normalized) Levenshtein Edit Distance.
///
/// The inputs are variable-length sequences provided by SparseTensors
///   (hypothesis_indices, hypothesis_values, hypothesis_shape)
/// and
///   (truth_indices, truth_values, truth_shape).
///
/// The inputs are:
///
/// - Parameters:
///   - hypothesis_indices: The indices of the hypothesis list SparseTensor.
///     This is an N x R int64 matrix.
///   - hypothesis_values: The values of the hypothesis list SparseTensor.
///     This is an N-length vector.
///   - hypothesis_shape: The shape of the hypothesis list SparseTensor.
///     This is an R-length vector.
///   - truth_indices: The indices of the truth list SparseTensor.
///     This is an M x R int64 matrix.
///   - truth_values: The values of the truth list SparseTensor.
///     This is an M-length vector.
///   - truth_shape: truth indices, vector.
///
/// - Attr normalize: boolean (if true, edit distances are normalized by length of truth).
///
///   The output is:
///
/// - Output output: A dense float tensor with rank R - 1.
///
///   For the example input:
///
///       // hypothesis represents a 2x1 matrix with variable-length values:
///       //   (0,0) = ["a"]
///       //   (1,0) = ["b"]
///       hypothesis_indices = [[0, 0, 0],
///                             [1, 0, 0]]
///       hypothesis_values = ["a", "b"]
///       hypothesis_shape = [2, 1, 1]
///
///       // truth represents a 2x2 matrix with variable-length values:
///       //   (0,0) = []
///       //   (0,1) = ["a"]
///       //   (1,0) = ["b", "c"]
///       //   (1,1) = ["a"]
///       truth_indices = [[0, 1, 0],
///                        [1, 0, 0],
///                        [1, 0, 1],
///                        [1, 1, 0]]
///       truth_values = ["a", "b", "c", "a"]
///       truth_shape = [2, 2, 2]
///       normalize = true
///
///   The output will be:
///
///       // output is a 2x2 matrix with edit distances normalized by truth lengths.
///       output = [[inf, 1.0],  // (0,0): no truth, (0,1): no hypothesis
///                 [0.5, 1.0]]  // (1,0): addition, (1,1): no hypothesis
@inlinable @inline(__always)
public static func editDistance<T: TensorFlowScalar>(
  hypothesisIndices: Tensor<Int64>,
  hypothesisValues: Tensor<T>,
  hypothesisShape: Tensor<Int64>,
  truthIndices: Tensor<Int64>,
  truthValues: Tensor<T>,
  truthShape: Tensor<Int64>,
  normalize: Bool = true
) -> Tensor<Float> {
  let ret: TensorHandle<Float> = #tfop("EditDistance",
    hypothesisIndices,
    hypothesisValues,
    hypothesisShape,
    truthIndices,
    truthValues,
    truthShape,
    T$dtype: T.tensorFlowDataType,
    normalize: normalize)
  return Tensor(handle: ret)
}

/// Computes exponential linear: `exp(features) - 1` if < 0, `features` otherwise.
///
/// See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
/// ](http://arxiv.org/abs/1511.07289)
@inlinable @inline(__always)
public static func elu<T: FloatingPoint & TensorFlowScalar>(
  features: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Elu",
    features,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes gradients for the exponential linear (Elu) operation.
///
/// - Parameters:
///   - gradients: The backpropagated gradients to the corresponding Elu operation.
///   - outputs: The outputs of the corresponding Elu operation.
///
/// - Output backprops: The gradients: `gradients * (outputs + 1)` if outputs < 0,
///   `gradients` otherwise.
@inlinable @inline(__always)
public static func eluGrad<T: FloatingPoint & TensorFlowScalar>(
  gradients: Tensor<T>,
  outputs: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("EluGrad",
    gradients,
    outputs,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Creates a tensor with the given shape.
///
/// This operation creates a tensor of `shape` and `dtype`.
///
/// - Parameter shape: 1-D. Represents the shape of the output tensor.
///
/// - Attr init: If True, initialize the returned tensor with the default value of dtype.  Otherwise, the implementation is free not to initializethe tensor's content.
///
/// - Output output: A `Tensor` of type `T`.
@inlinable @inline(__always)
public static func empty<Dtype: TensorFlowScalar>(
  shape: Tensor<Int32>,
  init_: Bool = false
) -> Tensor<Dtype> {
  let ret: TensorHandle<Dtype> = #tfop("Empty",
    shape,
    dtype$dtype: Dtype.tensorFlowDataType,
    init: init_)
  return Tensor(handle: ret)
}

/// Encode strings into web-safe base64 format.
///
/// Refer to the following article for more information on base64 format:
/// en.wikipedia.org/wiki/Base64. Base64 strings may have padding with '=' at the
/// end so that the encoded has length multiple of 4. See Padding section of the
/// link above.
///
/// Web-safe means that the encoder uses - and _ instead of + and /.
///
/// - Parameter input: Strings to be encoded.
///
/// - Attr pad: Bool whether padding is applied at the ends.
///
/// - Output output: Input strings encoded in base64.
@inlinable @inline(__always)
public static func encodeBase64(
  _ input: StringTensor,
  pad: Bool = false
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("EncodeBase64",
    input,
    pad: pad)
  return StringTensor(handle: ret)
}

/// JPEG-encode an image.
///
/// `image` is a 3-D uint8 Tensor of shape `[height, width, channels]`.
///
/// The attr `format` can be used to override the color format of the encoded
/// output.  Values can be:
///
/// *   `''`: Use a default format based on the number of channels in the image.
/// *   `grayscale`: Output a grayscale JPEG image.  The `channels` dimension
///     of `image` must be 1.
/// *   `rgb`: Output an RGB JPEG image. The `channels` dimension
///     of `image` must be 3.
///
/// If `format` is not specified or is the empty string, a default format is picked
/// in function of the number of channels in `image`:
///
/// *   1: Output a grayscale image.
/// *   3: Output an RGB image.
///
/// - Parameter image: 3-D with shape `[height, width, channels]`.
///
/// - Attrs:
///   - format: Per pixel image format.
///   - quality: Quality of the compression from 0 to 100 (higher is better and slower).
///   - progressive: If True, create a JPEG that loads progressively (coarse to fine).
///   - optimize_size: If True, spend CPU/RAM to reduce size with no quality change.
///   - chroma_downsampling: See http://en.wikipedia.org/wiki/Chroma_subsampling.
///   - density_unit: Unit used to specify `x_density` and `y_density`:
///     pixels per inch (`'in'`) or centimeter (`'cm'`).
///   - x_density: Horizontal pixels per density unit.
///   - y_density: Vertical pixels per density unit.
///   - xmp_metadata: If not empty, embed this XMP metadata in the image header.
///
/// - Output contents: 0-D. JPEG-encoded image.
@inlinable @inline(__always)
public static func encodeJpeg(
  image: Tensor<UInt8>,
  format: Format,
  quality: Int64 = 95,
  progressive: Bool = false,
  optimizeSize: Bool = false,
  chromaDownsampling: Bool = true,
  densityUnit: DensityUnit = .in_,
  xDensity: Int64 = 300,
  yDensity: Int64 = 300,
  xmpMetadata: String
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("EncodeJpeg",
    image,
    format: format.cName,
    quality: quality,
    progressive: progressive,
    optimize_size: optimizeSize,
    chroma_downsampling: chromaDownsampling,
    density_unit: densityUnit.cName,
    x_density: xDensity,
    y_density: yDensity,
    xmp_metadata: xmpMetadata)
  return StringTensor(handle: ret)
}

/// JPEG encode input image with provided compression quality.
///
/// `image` is a 3-D uint8 Tensor of shape `[height, width, channels]`.
/// `quality` is an int32 jpeg compression quality value between 0 and 100.
///
///
/// - Parameters:
///   - images: Images to adjust.  At least 3-D.
///   - quality: An int quality to encode to.
///
/// - Output contents: 0-D. JPEG-encoded image.
@inlinable @inline(__always)
public static func encodeJpegVariableQuality(
  images: Tensor<UInt8>,
  quality: Tensor<Int32>
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("EncodeJpegVariableQuality",
    images,
    quality)
  return StringTensor(handle: ret)
}

/// PNG-encode an image.
///
/// `image` is a 3-D uint8 or uint16 Tensor of shape `[height, width, channels]`
/// where `channels` is:
///
/// *   1: for grayscale.
/// *   2: for grayscale + alpha.
/// *   3: for RGB.
/// *   4: for RGBA.
///
/// The ZLIB compression level, `compression`, can be -1 for the PNG-encoder
/// default or a value from 0 to 9.  9 is the highest compression level, generating
/// the smallest output, but is slower.
///
/// - Parameter image: 3-D with shape `[height, width, channels]`.
///
/// - Attr compression: Compression level.
///
/// - Output contents: 0-D. PNG-encoded image.
@inlinable @inline(__always)
public static func encodePng<T: UnsignedInteger & TensorFlowScalar>(
  image: Tensor<T>,
  compression: Int64 = -1
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("EncodePng",
    image,
    T$dtype: T.tensorFlowDataType,
    compression: compression)
  return StringTensor(handle: ret)
}

/// The op serializes protobuf messages provided in the input tensors.
///
/// The types of the tensors in `values` must match the schema for the
/// fields specified in `field_names`. All the tensors in `values` must
/// have a common shape prefix, *batch_shape*.
///
/// The `sizes` tensor specifies repeat counts for each field.  The repeat
/// count (last dimension) of a each tensor in `values` must be greater
/// than or equal to corresponding repeat count in `sizes`.
///
/// A `message_type` name must be provided to give context for the field
/// names. The actual message descriptor can be looked up either in the
/// linked-in descriptor pool or a filename provided by the caller using
/// the `descriptor_source` attribute.
///
/// The `descriptor_source` attribute selects a source of protocol
/// descriptors to consult when looking up `message_type`. This may be a
/// filename containing a serialized `FileDescriptorSet` message,
/// or the special value `local://`, in which case only descriptors linked
/// into the code will be searched; the filename can be on any filesystem
/// accessible to TensorFlow.
///
/// You can build a `descriptor_source` file using the `--descriptor_set_out`
/// and `--include_imports` options to the protocol compiler `protoc`.
///
/// The `local://` database only covers descriptors linked into the
/// code via C++ libraries, not Python imports. You can link in a proto descriptor
/// by creating a cc_library target with alwayslink=1.
///
/// There are a few special cases in the value mapping:
///
/// Submessage and group fields must be pre-serialized as TensorFlow strings.
///
/// TensorFlow lacks support for unsigned int64s, so they must be
/// represented as `tf.int64` with the same twos-complement bit pattern
/// (the obvious way).
///
/// Unsigned int32 values can be represented exactly with `tf.int64`, or
/// with sign wrapping if the input is of type `tf.int32`.
///
/// - Parameters:
///   - sizes: Tensor of int32 with shape `[batch_shape, len(field_names)]`.
///   - values: List of tensors containing values for the corresponding field.
///
/// - Attrs:
///   - field_names: List of strings containing proto field names.
///   - message_type: Name of the proto message type to decode.
///   - Tinput_types: The input types.
///
/// - Output bytes: Tensor of serialized protos with shape `batch_shape`.
@inlinable @inline(__always)
public static func encodeProto<TinputTypes: TensorFlowScalar>(
  sizes: Tensor<Int32>,
  _ values: [Tensor<TinputTypes>],
  fieldNames: [String],
  messageType: String,
  descriptorSource: String = "local://"
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("EncodeProto",
    sizes,
    values,
    field_names: fieldNames,
    message_type: messageType,
    descriptor_source: descriptorSource)
  return StringTensor(handle: ret)
}

/// Encode audio data using the WAV file format.
///
/// This operation will generate a string suitable to be saved out to create a .wav
/// audio file. It will be encoded in the 16-bit PCM format. It takes in float
/// values in the range -1.0f to 1.0f, and any outside that value will be clamped to
/// that range.
///
/// `audio` is a 2-D float Tensor of shape `[length, channels]`.
/// `sample_rate` is a scalar Tensor holding the rate to use (e.g. 44100).
///
/// - Parameters:
///   - audio: 2-D with shape `[length, channels]`.
///   - sample_rate: Scalar containing the sample frequency.
///
/// - Output contents: 0-D. WAV-encoded file contents.
@inlinable @inline(__always)
public static func encodeWav(
  audio: Tensor<Float>,
  sampleRate: Tensor<Int32>
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("EncodeWav",
    audio,
    sampleRate)
  return StringTensor(handle: ret)
}

/// An op that enqueues a list of input batch tensors to TPUEmbedding.
///
/// - Parameters:
///   - batch: A list of 1D tensors, one for each embedding table, containing the
///     indices into the tables.
///   - mode_override: A string input that overrides the mode specified in the
///     TPUEmbeddingConfiguration. Supported values are {'unspecified', 'inference',
///     'training', 'backward_pass_only'}. When set to 'unspecified', the mode set
///     in TPUEmbeddingConfiguration is used, otherwise mode_override is used.
///
/// - Attr device_ordinal: The TPU device to use. Should be >= 0 and less than the number
///   of TPU cores in the task on which the node is placed.
@inlinable @inline(__always)
public static func enqueueTPUEmbeddingIntegerBatch(
  batch: [Tensor<Int32>],
  modeOverride: StringTensor,
  deviceOrdinal: Int64 = -1
) {
  return #tfop("EnqueueTPUEmbeddingIntegerBatch",
    batch,
    modeOverride,
    device_ordinal: deviceOrdinal)
}

/// An op that enqueues TPUEmbedding input indices from a SparseTensor.
///
/// This Op eases the porting of code that uses embedding_lookup_sparse(),
/// although some Python preprocessing of the SparseTensor arguments to
/// embedding_lookup_sparse() is required to produce the arguments to this Op,
/// since only a single EnqueueTPUEmbeddingSparseBatch Op is allowed per training
/// step.
///
/// The tensors at corresponding positions in the three input lists
/// must have the same shape, i.e. rank 1 with dim_size() equal to the total
/// number of lookups into the table described by the corresponding table_id.
///
/// - Parameters:
///   - sample_indices: A list of rank 1 Tensors specifying the training example and
///     feature to which the corresponding embedding_indices and aggregation_weights
///     values belong. sample_indices[i] must equal b * nf + f, where nf is the
///     number of features from the corresponding table, f is in [0, nf), and
///     b is in [0, batch size).
///   - embedding_indices: A list of rank 1 Tensors, indices into the embedding tables.
///   - aggregation_weights: A list of rank 1 Tensors containing per sample -- i.e. per
///     (training example, feature) -- aggregation weights.
///   - mode_override: A string input that overrides the mode specified in the
///     TPUEmbeddingConfiguration. Supported values are {'unspecified', 'inference',
///     'training', 'backward_pass_only'}. When set to 'unspecified', the mode set
///     in TPUEmbeddingConfiguration is used, otherwise mode_override is used.
///
/// - Attrs:
///   - device_ordinal: The TPU device to use. Should be >= 0 and less than the number
///     of TPU cores in the task on which the node is placed.
///   - combiners: A list of string scalars, one for each embedding table that specify
///     how to normalize the embedding activations after weighted summation.
///     Supported combiners are 'mean', 'sum', or 'sqrtn'. It is invalid to have
///     the sum of the weights be 0 for 'mean' or the sum of the squared weights be
///     0 for 'sqrtn'. If combiners isn't passed, the default is to use 'sum' for
///     all tables.
@inlinable @inline(__always)
public static func enqueueTPUEmbeddingSparseBatch<T1: BinaryInteger & TensorFlowScalar, T2: BinaryInteger & TensorFlowScalar, T3: FloatingPoint & TensorFlowScalar>(
  sampleIndices: [Tensor<T1>],
  embeddingIndices: [Tensor<T2>],
  aggregationWeights: [Tensor<T3>],
  modeOverride: StringTensor,
  deviceOrdinal: Int64 = -1,
  combiners: [String]
) {
  return #tfop("EnqueueTPUEmbeddingSparseBatch",
    sampleIndices,
    embeddingIndices,
    aggregationWeights,
    modeOverride,
    T1$dtype: T1.tensorFlowDataType,
    T2$dtype: T2.tensorFlowDataType,
    T3$dtype: T3.tensorFlowDataType,
    device_ordinal: deviceOrdinal,
    combiners: combiners)
}

/// Eases the porting of code that uses tf.nn.embedding_lookup_sparse().
///
/// sample_indices[i], embedding_indices[i] and aggregation_weights[i] correspond
/// to the ith feature. table_ids[i] indicates which embedding table to look up ith
/// feature.
///
/// The tensors at corresponding positions in the three input lists (sample_indices,
/// embedding_indices and aggregation_weights) must have the same shape, i.e. rank 1
/// with dim_size() equal to the total number of lookups into the table described by
/// the corresponding feature.
///
/// - Parameters:
///   - sample_indices: A list of rank 1 Tensors specifying the training example to
///     which the corresponding embedding_indices and aggregation_weights values
///     belong. It corresponds to sp_ids.indices[:,0] in  embedding_lookup_sparse().
///   - embedding_indices: A list of rank 1 Tensors, indices into the embedding tables.
///     It corresponds to sp_ids.values in embedding_lookup_sparse().
///   - aggregation_weights: A list of rank 1 Tensors containing per training example
///     aggregation weights. It corresponds to sp_weights.values in
///     embedding_lookup_sparse().
///   - mode_override: A string input that overrides the mode specified in the
///     TPUEmbeddingConfiguration. Supported values are {'unspecified', 'inference',
///     'training', 'backward_pass_only'}. When set to 'unspecified', the mode set
///     in TPUEmbeddingConfiguration is used, otherwise mode_override is used.
///
/// - Attrs:
///   - device_ordinal: The TPU device to use. Should be >= 0 and less than the number
///     of TPU cores in the task on which the node is placed.
///   - combiners: A list of string scalars, one for each embedding table that specify
///     how to normalize the embedding activations after weighted summation.
///     Supported combiners are 'mean', 'sum', or 'sqrtn'. It is invalid to have
///     the sum of the weights be 0 for 'mean' or the sum of the squared weights be
///     0 for 'sqrtn'. If combiners isn't passed, the default is to use 'sum' for
///     all tables.
///   - table_ids: A list of integers specifying the identifier of the embedding table
///     (offset of TableDescriptor in the TPUEmbeddingConfiguration) to lookup the
///     corresponding input. The ith input is looked up using table_ids[i]. The size
///     of the table_ids list must be equal to that of sample_indices,
///     embedding_indices and aggregation_weights.
@inlinable @inline(__always)
public static func enqueueTPUEmbeddingSparseTensorBatch<T1: BinaryInteger & TensorFlowScalar, T2: BinaryInteger & TensorFlowScalar, T3: FloatingPoint & TensorFlowScalar>(
  sampleIndices: [Tensor<T1>],
  embeddingIndices: [Tensor<T2>],
  aggregationWeights: [Tensor<T3>],
  modeOverride: StringTensor,
  deviceOrdinal: Int64 = -1,
  combiners: [String],
  tableIds: [Int32]
) {
  return #tfop("EnqueueTPUEmbeddingSparseTensorBatch",
    sampleIndices,
    embeddingIndices,
    aggregationWeights,
    modeOverride,
    T1$dtype: T1.tensorFlowDataType,
    T2$dtype: T2.tensorFlowDataType,
    T3$dtype: T3.tensorFlowDataType,
    device_ordinal: deviceOrdinal,
    combiners: combiners,
    table_ids: tableIds)
}

/// Creates or finds a child frame, and makes `data` available to the child frame.
///
/// This op is used together with `Exit` to create loops in the graph.
/// The unique `frame_name` is used by the `Executor` to identify frames. If
/// `is_constant` is true, `output` is a constant in the child frame; otherwise
/// it may be changed in the child frame. At most `parallel_iterations` iterations
/// are run in parallel in the child frame.
///
/// - Parameter data: The tensor to be made available to the child frame.
///
/// - Attrs:
///   - frame_name: The name of the child frame.
///   - is_constant: If true, the output is constant within the child frame.
///   - parallel_iterations: The number of iterations allowed to run in parallel.
///
/// - Output output: The same tensor as `data`.
@inlinable @inline(__always)
public static func enter<T: TensorFlowScalar>(
  data: Tensor<T>,
  frameName: String,
  isConstant: Bool = false,
  parallelIterations: Int64 = 10
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Enter",
    data,
    T$dtype: T.tensorFlowDataType,
    frame_name: frameName,
    is_constant: isConstant,
    parallel_iterations: parallelIterations)
  return Tensor(handle: ret)
}

/// Returns the truth value of (x == y) element-wise.
///
/// *NOTE*: `Equal` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@inlinable @inline(__always)
public static func equal<T: TensorFlowScalar>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<Bool> {
  let ret: TensorHandle<Bool> = #tfop("Equal",
    x,
    y,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the Gauss error function of `x` element-wise.
@inlinable @inline(__always)
public static func erf<T: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Erf",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the complementary error function of `x` element-wise.
@inlinable @inline(__always)
public static func erfc<T: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Erfc",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the euclidean norm of elements across dimensions of a tensor.
///
/// Reduces `input` along the dimensions given in `axis`. Unless
/// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
/// `axis`. If `keep_dims` is true, the reduced dimensions are
/// retained with length 1.
///
/// - Parameters:
///   - input: The tensor to reduce.
///   - reduction_indices: The dimensions to reduce. Must be in the range
///     `[-rank(input), rank(input))`.
///
/// - Attr keep_dims: If true, retain reduced dimensions with length 1.
///
/// - Output output: The reduced tensor.
@inlinable @inline(__always)
public static func euclideanNorm<T: Numeric & TensorFlowScalar, Tidx: BinaryInteger & TensorFlowScalar>(
  _ input: Tensor<T>,
  reductionIndices: Tensor<Tidx>,
  keepDims: Bool = false
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("EuclideanNorm",
    input,
    reductionIndices,
    T$dtype: T.tensorFlowDataType,
    Tidx$dtype: Tidx.tensorFlowDataType,
    keep_dims: keepDims)
  return Tensor(handle: ret)
}

/// Exits the current frame to its parent frame.
///
/// Exit makes its input `data` available to the parent frame.
///
/// - Parameter data: The tensor to be made available to the parent frame.
///
/// - Output output: The same tensor as `data`.
@inlinable @inline(__always)
public static func exit<T: TensorFlowScalar>(
  data: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Exit",
    data,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes exponential of x element-wise.  \\(y = e^x\\).
@inlinable @inline(__always)
public static func exp<T: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Exp",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Inserts a dimension of 1 into a tensor's shape.
///
/// Given a tensor `input`, this operation inserts a dimension of 1 at the
/// dimension index `axis` of `input`'s shape. The dimension index `axis` starts at
/// zero; if you specify a negative number for `axis` it is counted backward from
/// the end.
///
/// This operation is useful if you want to add a batch dimension to a single
/// element. For example, if you have a single image of shape `[height, width,
/// channels]`, you can make it a batch of 1 image with `expand_dims(image, 0)`,
/// which will make the shape `[1, height, width, channels]`.
///
/// Other examples:
///
/// ```
/// # 't' is a tensor of shape [2]
/// shape(expand_dims(t, 0)) ==> [1, 2]
/// shape(expand_dims(t, 1)) ==> [2, 1]
/// shape(expand_dims(t, -1)) ==> [2, 1]
///
/// # 't2' is a tensor of shape [2, 3, 5]
/// shape(expand_dims(t2, 0)) ==> [1, 2, 3, 5]
/// shape(expand_dims(t2, 2)) ==> [2, 3, 1, 5]
/// shape(expand_dims(t2, 3)) ==> [2, 3, 5, 1]
/// ```
///
/// This operation requires that:
///
/// `-1-input.dims() <= dim <= input.dims()`
///
/// This operation is related to `squeeze()`, which removes dimensions of
/// size 1.
///
/// - Parameter dim: 0-D (scalar). Specifies the dimension index at which to
///   expand the shape of `input`. Must be in the range
///   `[-rank(input) - 1, rank(input)]`.
///
/// - Output output: Contains the same data as `input`, but its shape has an additional
///   dimension of size 1 added.
@inlinable @inline(__always)
public static func expandDims<T: TensorFlowScalar, Tdim: BinaryInteger & TensorFlowScalar>(
  _ input: Tensor<T>,
  dim: Tensor<Tdim>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("ExpandDims",
    input,
    dim,
    T$dtype: T.tensorFlowDataType,
    Tdim$dtype: Tdim.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes exponential of x - 1 element-wise.
///
/// I.e., \\(y = (\exp x) - 1\\).
@inlinable @inline(__always)
public static func expm1<T: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Expm1",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Extracts a glimpse from the input tensor.
///
/// Returns a set of windows called glimpses extracted at location
/// `offsets` from the input tensor. If the windows only partially
/// overlaps the inputs, the non overlapping areas will be filled with
/// random noise.
///
/// The result is a 4-D tensor of shape `[batch_size, glimpse_height,
/// glimpse_width, channels]`. The channels and batch dimensions are the
/// same as that of the input tensor. The height and width of the output
/// windows are specified in the `size` parameter.
///
/// The argument `normalized` and `centered` controls how the windows are built:
///
/// * If the coordinates are normalized but not centered, 0.0 and 1.0
///   correspond to the minimum and maximum of each height and width
///   dimension.
/// * If the coordinates are both normalized and centered, they range from
///   -1.0 to 1.0. The coordinates (-1.0, -1.0) correspond to the upper
///   left corner, the lower right corner is located at (1.0, 1.0) and the
///   center is at (0, 0).
/// * If the coordinates are not normalized they are interpreted as
///   numbers of pixels.
///
/// - Parameters:
///   - input: A 4-D float tensor of shape `[batch_size, height, width, channels]`.
///   - size: A 1-D tensor of 2 elements containing the size of the glimpses
///     to extract.  The glimpse height must be specified first, following
///     by the glimpse width.
///   - offsets: A 2-D integer tensor of shape `[batch_size, 2]` containing
///     the y, x locations of the center of each window.
///
/// - Attrs:
///   - centered: indicates if the offset coordinates are centered relative to
///     the image, in which case the (0, 0) offset is relative to the center
///     of the input images. If false, the (0,0) offset corresponds to the
///     upper left corner of the input images.
///   - normalized: indicates if the offset coordinates are normalized.
///   - uniform_noise: indicates if the noise should be generated using a
///     uniform distribution or a Gaussian distribution.
///   - noise: indicates if the noise should `uniform`, `gaussian`, or
///     `zero`. The default is `uniform` which means the the noise type
///     will be decided by `uniform_noise`.
///
/// - Output glimpse: A tensor representing the glimpses `[batch_size,
///   glimpse_height, glimpse_width, channels]`.
@inlinable @inline(__always)
public static func extractGlimpse(
  _ input: Tensor<Float>,
  size: Tensor<Int32>,
  offsets: Tensor<Float>,
  centered: Bool = true,
  normalized: Bool = true,
  uniformNoise: Bool = true,
  noise: String = "uniform"
) -> Tensor<Float> {
  let ret: TensorHandle<Float> = #tfop("ExtractGlimpse",
    input,
    size,
    offsets,
    centered: centered,
    normalized: normalized,
    uniform_noise: uniformNoise,
    noise: noise)
  return Tensor(handle: ret)
}

/// Extract `patches` from `images` and put them in the "depth" output dimension.
///
/// - Parameter images: 4-D Tensor with shape `[batch, in_rows, in_cols, depth]`.
///
/// - Attrs:
///   - ksizes: The size of the sliding window for each dimension of `images`.
///   - strides: 1-D of length 4. How far the centers of two consecutive patches are in
///     the images. Must be: `[1, stride_rows, stride_cols, 1]`.
///   - rates: 1-D of length 4. Must be: `[1, rate_rows, rate_cols, 1]`. This is the
///     input stride, specifying how far two consecutive patch samples are in the
///     input. Equivalent to extracting patches with
///     `patch_sizes_eff = patch_sizes + (patch_sizes - 1) * (rates - 1)`, followed by
///     subsampling them spatially by a factor of `rates`. This is equivalent to
///     `rate` in dilated (a.k.a. Atrous) convolutions.
///   - padding: The type of padding algorithm to use.
///
///     We specify the size-related attributes as:
///
///     ```python
///           ksizes = [1, ksize_rows, ksize_cols, 1]
///           strides = [1, strides_rows, strides_cols, 1]
///           rates = [1, rates_rows, rates_cols, 1]
///     ```
///
/// - Output patches: 4-D Tensor with shape `[batch, out_rows, out_cols, ksize_rows *
///   ksize_cols * depth]` containing image patches with size
///   `ksize_rows x ksize_cols x depth` vectorized in the "depth" dimension. Note
///   `out_rows` and `out_cols` are the dimensions of the output patches.
@inlinable @inline(__always)
public static func extractImagePatches<T: Numeric & TensorFlowScalar>(
  images: Tensor<T>,
  ksizes: [Int32],
  strides: [Int32],
  rates: [Int32],
  padding: Padding
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("ExtractImagePatches",
    images,
    T$dtype: T.tensorFlowDataType,
    ksizes: ksizes,
    strides: strides,
    rates: rates,
    padding: padding.cName)
  return Tensor(handle: ret)
}

/// Extract the shape information of a JPEG-encoded image.
///
/// This op only parses the image header, so it is much faster than DecodeJpeg.
///
/// - Parameter contents: 0-D. The JPEG-encoded image.
///
/// - Attr output_type: (Optional) The output type of the operation (int32 or int64).
///   Defaults to int32.
///
/// - Output image_shape: 1-D. The image shape with format [height, width, channels].
@inlinable @inline(__always)
public static func extractJpegShape<OutputType: BinaryInteger & TensorFlowScalar>(
  contents: StringTensor
) -> Tensor<OutputType> {
  let ret: TensorHandle<OutputType> = #tfop("ExtractJpegShape",
    contents,
    output_type$dtype: OutputType.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Extract `patches` from `input` and put them in the "depth" output dimension. 3D extension of `extract_image_patches`.
///
/// - Parameter input: 5-D Tensor with shape `[batch, in_planes, in_rows, in_cols, depth]`.
///
/// - Attrs:
///   - ksizes: The size of the sliding window for each dimension of `input`.
///   - strides: 1-D of length 5. How far the centers of two consecutive patches are in
///     `input`. Must be: `[1, stride_planes, stride_rows, stride_cols, 1]`.
///   - padding: The type of padding algorithm to use.
///
///     We specify the size-related attributes as:
///
///     ```python
///           ksizes = [1, ksize_planes, ksize_rows, ksize_cols, 1]
///           strides = [1, stride_planes, strides_rows, strides_cols, 1]
///     ```
///
/// - Output patches: 5-D Tensor with shape `[batch, out_planes, out_rows, out_cols,
///   ksize_planes * ksize_rows * ksize_cols * depth]` containing patches
///   with size `ksize_planes x ksize_rows x ksize_cols x depth` vectorized
///   in the "depth" dimension. Note `out_planes`, `out_rows` and `out_cols`
///   are the dimensions of the output patches.
@inlinable @inline(__always)
public static func extractVolumePatches<T: Numeric & TensorFlowScalar>(
  _ input: Tensor<T>,
  ksizes: [Int32],
  strides: [Int32],
  padding: Padding
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("ExtractVolumePatches",
    input,
    T$dtype: T.tensorFlowDataType,
    ksizes: ksizes,
    strides: strides,
    padding: padding.cName)
  return Tensor(handle: ret)
}

/// Fast Fourier transform.
///
/// Computes the 1-dimensional discrete Fourier transform over the inner-most
/// dimension of `input`.
///
/// - Parameter input: A complex tensor.
///
/// - Output output: A complex tensor of the same shape as `input`. The inner-most
///     dimension of `input` is replaced with its 1D Fourier transform.
///
///   @compatibility(numpy)
///   Equivalent to np.fft.fft
///   @end_compatibility
@inlinable @inline(__always)
public static func fFT<Tcomplex: TensorFlowScalar>(
  _ input: Tensor<Tcomplex>
) -> Tensor<Tcomplex> {
  let ret: TensorHandle<Tcomplex> = #tfop("FFT",
    input,
    Tcomplex$dtype: Tcomplex.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// 2D fast Fourier transform.
///
/// Computes the 2-dimensional discrete Fourier transform over the inner-most
/// 2 dimensions of `input`.
///
/// - Parameter input: A complex tensor.
///
/// - Output output: A complex tensor of the same shape as `input`. The inner-most 2
///     dimensions of `input` are replaced with their 2D Fourier transform.
///
///   @compatibility(numpy)
///   Equivalent to np.fft.fft2
///   @end_compatibility
@inlinable @inline(__always)
public static func fFT2D<Tcomplex: TensorFlowScalar>(
  _ input: Tensor<Tcomplex>
) -> Tensor<Tcomplex> {
  let ret: TensorHandle<Tcomplex> = #tfop("FFT2D",
    input,
    Tcomplex$dtype: Tcomplex.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// 3D fast Fourier transform.
///
/// Computes the 3-dimensional discrete Fourier transform over the inner-most 3
/// dimensions of `input`.
///
/// - Parameter input: A complex64 tensor.
///
/// - Output output: A complex64 tensor of the same shape as `input`. The inner-most 3
///     dimensions of `input` are replaced with their 3D Fourier transform.
///
///   @compatibility(numpy)
///   Equivalent to np.fft.fftn with 3 dimensions.
///   @end_compatibility
@inlinable @inline(__always)
public static func fFT3D<Tcomplex: TensorFlowScalar>(
  _ input: Tensor<Tcomplex>
) -> Tensor<Tcomplex> {
  let ret: TensorHandle<Tcomplex> = #tfop("FFT3D",
    input,
    Tcomplex$dtype: Tcomplex.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Output a fact about factorials.
@inlinable @inline(__always)
public static func fact(
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("Fact")
  return StringTensor(handle: ret)
}

/// Fake-quantize the 'inputs' tensor, type float to 'outputs' tensor of same type.
///
/// Attributes `[min; max]` define the clamping range for the `inputs` data.
/// `inputs` values are quantized into the quantization range (`[0; 2^num_bits - 1]`
/// when `narrow_range` is false and `[1; 2^num_bits - 1]` when it is true) and
/// then de-quantized and output as floats in `[min; max]` interval.
/// `num_bits` is the bitwidth of the quantization; between 2 and 16, inclusive.
///
/// Before quantization, `min` and `max` values are adjusted with the following
/// logic.
/// It is suggested to have `min <= 0 <= max`. If `0` is not in the range of values,
/// the behavior can be unexpected:
/// If `0 < min < max`: `min_adj = 0` and `max_adj = max - min`.
/// If `min < max < 0`: `min_adj = min - max` and `max_adj = 0`.
/// If `min <= 0 <= max`: `scale = (max - min) / (2^num_bits - 1) `,
/// `min_adj = scale * round(min / scale)` and `max_adj = max + min_adj - min`.
///
/// Quantization is called fake since the output is still in floating point.
@inlinable @inline(__always)
public static func fakeQuantWithMinMaxArgs(
  inputs: Tensor<Float>,
  min: Double = -6,
  max: Double = 6,
  numBits: Int64 = 8,
  narrowRange: Bool = false
) -> Tensor<Float> {
  let ret: TensorHandle<Float> = #tfop("FakeQuantWithMinMaxArgs",
    inputs,
    min: min,
    max: max,
    num_bits: numBits,
    narrow_range: narrowRange)
  return Tensor(handle: ret)
}

/// Compute gradients for a FakeQuantWithMinMaxArgs operation.
///
/// - Parameters:
///   - gradients: Backpropagated gradients above the FakeQuantWithMinMaxArgs operation.
///   - inputs: Values passed as inputs to the FakeQuantWithMinMaxArgs operation.
///
/// - Output backprops: Backpropagated gradients below the FakeQuantWithMinMaxArgs operation:
///   `gradients * (inputs >= min && inputs <= max)`.
@inlinable @inline(__always)
public static func fakeQuantWithMinMaxArgsGradient(
  gradients: Tensor<Float>,
  inputs: Tensor<Float>,
  min: Double = -6,
  max: Double = 6,
  numBits: Int64 = 8,
  narrowRange: Bool = false
) -> Tensor<Float> {
  let ret: TensorHandle<Float> = #tfop("FakeQuantWithMinMaxArgsGradient",
    gradients,
    inputs,
    min: min,
    max: max,
    num_bits: numBits,
    narrow_range: narrowRange)
  return Tensor(handle: ret)
}

/// Fake-quantize the 'inputs' tensor of type float via global float scalars `min`
///
/// and `max` to 'outputs' tensor of same shape as `inputs`.
///
/// `[min; max]` define the clamping range for the `inputs` data.
/// `inputs` values are quantized into the quantization range (`[0; 2^num_bits - 1]`
/// when `narrow_range` is false and `[1; 2^num_bits - 1]` when it is true) and
/// then de-quantized and output as floats in `[min; max]` interval.
/// `num_bits` is the bitwidth of the quantization; between 2 and 16, inclusive.
///
/// Before quantization, `min` and `max` values are adjusted with the following
/// logic.
/// It is suggested to have `min <= 0 <= max`. If `0` is not in the range of values,
/// the behavior can be unexpected:
/// If `0 < min < max`: `min_adj = 0` and `max_adj = max - min`.
/// If `min < max < 0`: `min_adj = min - max` and `max_adj = 0`.
/// If `min <= 0 <= max`: `scale = (max - min) / (2^num_bits - 1) `,
/// `min_adj = scale * round(min / scale)` and `max_adj = max + min_adj - min`.
///
/// This operation has a gradient and thus allows for training `min` and `max`
/// values.
@inlinable @inline(__always)
public static func fakeQuantWithMinMaxVars(
  inputs: Tensor<Float>,
  min: Tensor<Float>,
  max: Tensor<Float>,
  numBits: Int64 = 8,
  narrowRange: Bool = false
) -> Tensor<Float> {
  let ret: TensorHandle<Float> = #tfop("FakeQuantWithMinMaxVars",
    inputs,
    min,
    max,
    num_bits: numBits,
    narrow_range: narrowRange)
  return Tensor(handle: ret)
}

/// Compute gradients for a FakeQuantWithMinMaxVars operation.
///
/// - Parameters:
///   - gradients: Backpropagated gradients above the FakeQuantWithMinMaxVars operation.
///   - inputs: Values passed as inputs to the FakeQuantWithMinMaxVars operation.
///     min, max: Quantization interval, scalar floats.
///
/// - Attrs:
///   - num_bits: The bitwidth of the quantization; between 2 and 8, inclusive.
///   - narrow_range: Whether to quantize into 2^num_bits - 1 distinct values.
///
/// - Outputs:
///   - backprops_wrt_input: Backpropagated gradients w.r.t. inputs:
///     `gradients * (inputs >= min && inputs <= max)`.
///   - backprop_wrt_min: Backpropagated gradients w.r.t. min parameter:
///     `sum(gradients * (inputs < min))`.
///   - backprop_wrt_max: Backpropagated gradients w.r.t. max parameter:
///     `sum(gradients * (inputs > max))`.
@inlinable @inline(__always)
public static func fakeQuantWithMinMaxVarsGradient(
  gradients: Tensor<Float>,
  inputs: Tensor<Float>,
  min: Tensor<Float>,
  max: Tensor<Float>,
  numBits: Int64 = 8,
  narrowRange: Bool = false
) -> (backpropsWrtInput: Tensor<Float>, backpropWrtMin: Tensor<Float>, backpropWrtMax: Tensor<Float>) {
  let ret: (TensorHandle<Float>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("FakeQuantWithMinMaxVarsGradient",
    gradients,
    inputs,
    min,
    max,
    num_bits: numBits,
    narrow_range: narrowRange)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Fake-quantize the 'inputs' tensor of type float and one of the shapes: `[d]`,
///
/// `[b, d]` `[b, h, w, d]` via per-channel floats `min` and `max` of shape `[d]`
/// to 'outputs' tensor of same shape as `inputs`.
///
/// `[min; max]` define the clamping range for the `inputs` data.
/// `inputs` values are quantized into the quantization range (`[0; 2^num_bits - 1]`
/// when `narrow_range` is false and `[1; 2^num_bits - 1]` when it is true) and
/// then de-quantized and output as floats in `[min; max]` interval.
/// `num_bits` is the bitwidth of the quantization; between 2 and 16, inclusive.
///
/// Before quantization, `min` and `max` values are adjusted with the following
/// logic.
/// It is suggested to have `min <= 0 <= max`. If `0` is not in the range of values,
/// the behavior can be unexpected:
/// If `0 < min < max`: `min_adj = 0` and `max_adj = max - min`.
/// If `min < max < 0`: `min_adj = min - max` and `max_adj = 0`.
/// If `min <= 0 <= max`: `scale = (max - min) / (2^num_bits - 1) `,
/// `min_adj = scale * round(min / scale)` and `max_adj = max + min_adj - min`.
///
/// This operation has a gradient and thus allows for training `min` and `max`
/// values.
@inlinable @inline(__always)
public static func fakeQuantWithMinMaxVarsPerChannel(
  inputs: Tensor<Float>,
  min: Tensor<Float>,
  max: Tensor<Float>,
  numBits: Int64 = 8,
  narrowRange: Bool = false
) -> Tensor<Float> {
  let ret: TensorHandle<Float> = #tfop("FakeQuantWithMinMaxVarsPerChannel",
    inputs,
    min,
    max,
    num_bits: numBits,
    narrow_range: narrowRange)
  return Tensor(handle: ret)
}

/// Compute gradients for a FakeQuantWithMinMaxVarsPerChannel operation.
///
/// - Parameters:
///   - gradients: Backpropagated gradients above the FakeQuantWithMinMaxVars operation,
///     shape one of: `[d]`, `[b, d]`,  `[b, h, w, d]`.
///   - inputs: Values passed as inputs to the FakeQuantWithMinMaxVars operation, shape
///       same as `gradients`.
///     min, max: Quantization interval, floats of shape `[d]`.
///
/// - Attrs:
///   - num_bits: The bitwidth of the quantization; between 2 and 16, inclusive.
///   - narrow_range: Whether to quantize into 2^num_bits - 1 distinct values.
///
/// - Outputs:
///   - backprops_wrt_input: Backpropagated gradients w.r.t. inputs, shape same as
///     `inputs`:
///       `gradients * (inputs >= min && inputs <= max)`.
///   - backprop_wrt_min: Backpropagated gradients w.r.t. min parameter, shape `[d]`:
///     `sum_per_d(gradients * (inputs < min))`.
///   - backprop_wrt_max: Backpropagated gradients w.r.t. max parameter, shape `[d]`:
///     `sum_per_d(gradients * (inputs > max))`.
@inlinable @inline(__always)
public static func fakeQuantWithMinMaxVarsPerChannelGradient(
  gradients: Tensor<Float>,
  inputs: Tensor<Float>,
  min: Tensor<Float>,
  max: Tensor<Float>,
  numBits: Int64 = 8,
  narrowRange: Bool = false
) -> (backpropsWrtInput: Tensor<Float>, backpropWrtMin: Tensor<Float>, backpropWrtMax: Tensor<Float>) {
  let ret: (TensorHandle<Float>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("FakeQuantWithMinMaxVarsPerChannelGradient",
    gradients,
    inputs,
    min,
    max,
    num_bits: numBits,
    narrow_range: narrowRange)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Creates a tensor filled with a scalar value.
///
/// This operation creates a tensor of shape `dims` and fills it with `value`.
///
/// For example:
///
/// ```
/// # Output tensor has shape [2, 3].
/// fill([2, 3], 9) ==> [[9, 9, 9]
///                      [9, 9, 9]]
/// ```
///
/// `tf.fill` differs from `tf.constant` in a few ways:
///
/// *   `tf.fill` only supports scalar contents, whereas `tf.constant` supports
///     Tensor values.
/// *   `tf.fill` creates an Op in the computation graph that constructs the actual
///     Tensor value at runtime. This is in contrast to `tf.constant` which embeds
///     the entire Tensor into the graph with a `Const` node.
/// *   Because `tf.fill` evaluates at graph runtime, it supports dynamic shapes
///     based on other runtime Tensors, unlike `tf.constant`.
///
/// - Parameters:
///   - dims: 1-D. Represents the shape of the output tensor.
///   - value: 0-D (scalar). Value to fill the returned tensor.
///
///     @compatibility(numpy)
///     Equivalent to np.full
///     @end_compatibility
@inlinable @inline(__always)
public static func fill<T: TensorFlowScalar, IndexType: BinaryInteger & TensorFlowScalar>(
  dims: Tensor<IndexType>,
  value: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Fill",
    dims,
    value,
    T$dtype: T.tensorFlowDataType,
    index_type$dtype: IndexType.tensorFlowDataType)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func fiveFloatOutputs(
) -> (a: Tensor<Float>, b: Tensor<Float>, c: Tensor<Float>, d: Tensor<Float>, e: Tensor<Float>) {
  let ret: (TensorHandle<Float>, TensorHandle<Float>, TensorHandle<Float>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("FiveFloatOutputs")
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2), Tensor(handle: ret.3), Tensor(handle: ret.4))
}

/// Generates labels for candidate sampling with a learned unigram distribution.
///
/// A unigram sampler could use a fixed unigram distribution read from a
/// file or passed in as an in-memory array instead of building up the distribution
/// from data on the fly. There is also an option to skew the distribution by
/// applying a distortion power to the weights.
///
/// The vocabulary file should be in CSV-like format, with the last field
/// being the weight associated with the word.
///
/// For each batch, this op picks a single set of sampled candidate labels.
///
/// The advantages of sampling candidates per-batch are simplicity and the
/// possibility of efficient dense matrix multiplication. The disadvantage is that
/// the sampled candidates must be chosen independently of the context and of the
/// true labels.
///
/// - Parameter true_classes: A batch_size * num_true matrix, in which each row contains the
///   IDs of the num_true target_classes in the corresponding original label.
///
/// - Attrs:
///   - num_true: Number of true labels per context.
///   - num_sampled: Number of candidates to randomly sample.
///   - unique: If unique is true, we sample with rejection, so that all sampled
///     candidates in a batch are unique. This requires some approximation to
///     estimate the post-rejection sampling probabilities.
///   - range_max: The sampler will sample integers from the interval [0, range_max).
///   - vocab_file: Each valid line in this file (which should have a CSV-like format)
///     corresponds to a valid word ID. IDs are in sequential order, starting from
///     num_reserved_ids. The last entry in each line is expected to be a value
///     corresponding to the count or relative probability. Exactly one of vocab_file
///     and unigrams needs to be passed to this op.
///   - distortion: The distortion is used to skew the unigram probability distribution.
///     Each weight is first raised to the distortion's power before adding to the
///     internal unigram distribution. As a result, distortion = 1.0 gives regular
///     unigram sampling (as defined by the vocab file), and distortion = 0.0 gives
///     a uniform distribution.
///   - num_reserved_ids: Optionally some reserved IDs can be added in the range [0,
///     ..., num_reserved_ids) by the users. One use case is that a special unknown
///     word token is used as ID 0. These IDs will have a sampling probability of 0.
///   - num_shards: A sampler can be used to sample from a subset of the original range
///     in order to speed up the whole computation through parallelism. This parameter
///     (together with 'shard') indicates the number of partitions that are being
///     used in the overall computation.
///   - shard: A sampler can be used to sample from a subset of the original range
///     in order to speed up the whole computation through parallelism. This parameter
///     (together with 'num_shards') indicates the particular partition number of a
///     sampler op, when partitioning is being used.
///   - unigrams: A list of unigram counts or probabilities, one per ID in sequential
///     order. Exactly one of vocab_file and unigrams should be passed to this op.
///   - seed: If either seed or seed2 are set to be non-zero, the random number
///     generator is seeded by the given seed.  Otherwise, it is seeded by a
///     random seed.
///   - seed2: An second seed to avoid seed collision.
///
/// - Outputs:
///   - sampled_candidates: A vector of length num_sampled, in which each element is
///     the ID of a sampled candidate.
///   - true_expected_count: A batch_size * num_true matrix, representing
///     the number of times each candidate is expected to occur in a batch
///     of sampled candidates. If unique=true, then this is a probability.
///   - sampled_expected_count: A vector of length num_sampled, for each sampled
///     candidate representing the number of times the candidate is expected
///     to occur in a batch of sampled candidates.  If unique=true, then this is a
///     probability.
@inlinable @inline(__always)
public static func fixedUnigramCandidateSampler(
  trueClasses: Tensor<Int64>,
  numTrue: Int64,
  numSampled: Int64,
  unique: Bool,
  rangeMax: Int64,
  vocabFile: String,
  distortion: Double = 1,
  numReservedIds: Int64 = 0,
  numShards: Int64 = 1,
  shard: Int64 = 0,
  unigrams: [Double],
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> (sampledCandidates: Tensor<Int64>, trueExpectedCount: Tensor<Float>, sampledExpectedCount: Tensor<Float>) {
  let ret: (TensorHandle<Int64>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("FixedUnigramCandidateSampler",
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
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

@inlinable @inline(__always)
public static func floatInput(
  _ a: Tensor<Float>
) {
  return #tfop("FloatInput",
    a)
}

@inlinable @inline(__always)
public static func floatOutput(
) -> Tensor<Float> {
  let ret: TensorHandle<Float> = #tfop("FloatOutput")
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func floatOutputStringOutput(
) -> (a: Tensor<Float>, b: StringTensor) {
  let ret: (TensorHandle<Float>, TensorHandle<String>) = #tfop("FloatOutputStringOutput")
  return (Tensor(handle: ret.0), StringTensor(handle: ret.1))
}

/// Returns element-wise largest integer not greater than x.
@inlinable @inline(__always)
public static func floor<T: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Floor",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Returns x // y element-wise.
///
/// *NOTE*: `FloorDiv` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@inlinable @inline(__always)
public static func floorDiv<T: Numeric & TensorFlowScalar>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("FloorDiv",
    x,
    y,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Returns element-wise remainder of division. When `x < 0` xor `y < 0` is
///
/// true, this follows Python semantics in that the result here is consistent
/// with a flooring divide. E.g. `floor(x / y) * y + mod(x, y) = x`.
///
/// *NOTE*: `FloorMod` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@inlinable @inline(__always)
public static func floorMod<T: Numeric & TensorFlowScalar>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("FloorMod",
    x,
    y,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func foo1(
  _ a: Tensor<Float>,
  _ b: Tensor<Int32>,
  c: Tensor<Int32>
) -> (d: Tensor<Float>, e: Tensor<Int32>) {
  let ret: (TensorHandle<Float>, TensorHandle<Int32>) = #tfop("Foo1",
    a,
    b,
    c)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

@inlinable @inline(__always)
public static func foo2(
  _ a: Tensor<Float>,
  _ b: StringTensor,
  c: StringTensor
) -> (d: Tensor<Float>, e: Tensor<Int32>) {
  let ret: (TensorHandle<Float>, TensorHandle<Int32>) = #tfop("Foo2",
    a,
    b,
    c)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

@inlinable @inline(__always)
public static func foo3(
  _ a: Tensor<Float>,
  _ b: StringTensor,
  c: Tensor<Float>
) -> (d: Tensor<Float>, e: Tensor<Int32>) {
  let ret: (TensorHandle<Float>, TensorHandle<Int32>) = #tfop("Foo3",
    a,
    b,
    c)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

/// Performs fractional average pooling on the input.
///
/// Fractional average pooling is similar to Fractional max pooling in the pooling
/// region generation step. The only difference is that after pooling regions are
/// generated, a mean operation is performed instead of a max operation in each
/// pooling region.
///
/// - Parameter value: 4-D with shape `[batch, height, width, channels]`.
///
/// - Attrs:
///   - pooling_ratio: Pooling ratio for each dimension of `value`, currently only
///     supports row and col dimension and should be >= 1.0. For example, a valid
///     pooling ratio looks like [1.0, 1.44, 1.73, 1.0]. The first and last elements
///     must be 1.0 because we don't allow pooling on batch and channels
///     dimensions. 1.44 and 1.73 are pooling ratio on height and width dimensions
///     respectively.
///   - pseudo_random: When set to True, generates the pooling sequence in a
///     pseudorandom fashion, otherwise, in a random fashion. Check paper [Benjamin
///     Graham, Fractional Max-Pooling](http://arxiv.org/abs/1412.6071) for
///     difference between pseudorandom and random.
///   - overlapping: When set to True, it means when pooling, the values at the boundary
///     of adjacent pooling cells are used by both cells. For example:
///
///     `index  0  1  2  3  4`
///
///     `value  20 5  16 3  7`
///
///     If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
///     The result would be [41/3, 26/3] for fractional avg pooling.
///   - deterministic: When set to True, a fixed pooling region will be used when
///     iterating over a FractionalAvgPool node in the computation graph. Mainly used
///     in unit test to make FractionalAvgPool deterministic.
///   - seed: If either seed or seed2 are set to be non-zero, the random number
///     generator is seeded by the given seed.  Otherwise, it is seeded by a
///     random seed.
///   - seed2: An second seed to avoid seed collision.
///
/// - Outputs:
///   - output: output tensor after fractional avg pooling.
///   - row_pooling_sequence: row pooling sequence, needed to calculate gradient.
///   - col_pooling_sequence: column pooling sequence, needed to calculate gradient.
@inlinable @inline(__always)
public static func fractionalAvgPool<T: Numeric & TensorFlowScalar>(
  value: Tensor<T>,
  poolingRatio: [Double],
  pseudoRandom: Bool = false,
  overlapping: Bool = false,
  deterministic: Bool = false,
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> (output: Tensor<T>, rowPoolingSequence: Tensor<Int64>, colPoolingSequence: Tensor<Int64>) {
  let ret: (TensorHandle<T>, TensorHandle<Int64>, TensorHandle<Int64>) = #tfop("FractionalAvgPool",
    value,
    T$dtype: T.tensorFlowDataType,
    pooling_ratio: poolingRatio,
    pseudo_random: pseudoRandom,
    overlapping: overlapping,
    deterministic: deterministic,
    seed: seed,
    seed2: seed2)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Computes gradient of the FractionalAvgPool function.
///
/// Unlike FractionalMaxPoolGrad, we don't need to find arg_max for
/// FractionalAvgPoolGrad, we just need to evenly back-propagate each element of
/// out_backprop to those indices that form the same pooling cell. Therefore, we
/// just need to know the shape of original input tensor, instead of the whole
/// tensor.
///
/// - Parameters:
///   - orig_input_tensor_shape: Original input tensor shape for `fractional_avg_pool`
///   - out_backprop: 4-D with shape `[batch, height, width, channels]`.  Gradients
///     w.r.t. the output of `fractional_avg_pool`.
///   - row_pooling_sequence: row pooling sequence, form pooling region with
///     col_pooling_sequence.
///   - col_pooling_sequence: column pooling sequence, form pooling region with
///     row_pooling sequence.
///
/// - Attr overlapping: When set to True, it means when pooling, the values at the boundary
///   of adjacent pooling cells are used by both cells. For example:
///
///   `index  0  1  2  3  4`
///
///   `value  20 5  16 3  7`
///
///   If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
///   The result would be [41/3, 26/3] for fractional avg pooling.
///
/// - Output output: 4-D.  Gradients w.r.t. the input of `fractional_avg_pool`.
@inlinable @inline(__always)
public static func fractionalAvgPoolGrad<T: Numeric & TensorFlowScalar>(
  origInputTensorShape: Tensor<Int64>,
  outBackprop: Tensor<T>,
  rowPoolingSequence: Tensor<Int64>,
  colPoolingSequence: Tensor<Int64>,
  overlapping: Bool = false
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("FractionalAvgPoolGrad",
    origInputTensorShape,
    outBackprop,
    rowPoolingSequence,
    colPoolingSequence,
    T$dtype: T.tensorFlowDataType,
    overlapping: overlapping)
  return Tensor(handle: ret)
}

/// Performs fractional max pooling on the input.
///
/// Fractional max pooling is slightly different than regular max pooling.  In
/// regular max pooling, you downsize an input set by taking the maximum value of
/// smaller N x N subsections of the set (often 2x2), and try to reduce the set by
/// a factor of N, where N is an integer.  Fractional max pooling, as you might
/// expect from the word "fractional", means that the overall reduction ratio N
/// does not have to be an integer.
///
/// The sizes of the pooling regions are generated randomly but are fairly uniform.
/// For example, let's look at the height dimension, and the constraints on the
/// list of rows that will be pool boundaries.
///
/// First we define the following:
///
/// 1.  input_row_length : the number of rows from the input set
/// 2.  output_row_length : which will be smaller than the input
/// 3.  alpha = input_row_length / output_row_length : our reduction ratio
/// 4.  K = floor(alpha)
/// 5.  row_pooling_sequence : this is the result list of pool boundary rows
///
/// Then, row_pooling_sequence should satisfy:
///
/// 1.  a[0] = 0 : the first value of the sequence is 0
/// 2.  a[end] = input_row_length : the last value of the sequence is the size
/// 3.  K <= (a[i+1] - a[i]) <= K+1 : all intervals are K or K+1 size
/// 4.  length(row_pooling_sequence) = output_row_length+1
///
/// For more details on fractional max pooling, see this paper:
/// [Benjamin Graham, Fractional Max-Pooling](http://arxiv.org/abs/1412.6071)
///
/// - Parameter value: 4-D with shape `[batch, height, width, channels]`.
///
/// - Attrs:
///   - pooling_ratio: Pooling ratio for each dimension of `value`, currently only
///     supports row and col dimension and should be >= 1.0. For example, a valid
///     pooling ratio looks like [1.0, 1.44, 1.73, 1.0]. The first and last elements
///     must be 1.0 because we don't allow pooling on batch and channels
///     dimensions. 1.44 and 1.73 are pooling ratio on height and width dimensions
///     respectively.
///   - pseudo_random: When set to True, generates the pooling sequence in a
///     pseudorandom fashion, otherwise, in a random fashion. Check paper [Benjamin
///     Graham, Fractional Max-Pooling](http://arxiv.org/abs/1412.6071) for
///     difference between pseudorandom and random.
///   - overlapping: When set to True, it means when pooling, the values at the boundary
///     of adjacent pooling cells are used by both cells. For example:
///
///     `index  0  1  2  3  4`
///
///     `value  20 5  16 3  7`
///
///     If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
///     The result would be [20, 16] for fractional max pooling.
///   - deterministic: When set to True, a fixed pooling region will be used when
///     iterating over a FractionalMaxPool node in the computation graph. Mainly used
///     in unit test to make FractionalMaxPool deterministic.
///   - seed: If either seed or seed2 are set to be non-zero, the random number
///     generator is seeded by the given seed.  Otherwise, it is seeded by a
///     random seed.
///   - seed2: An second seed to avoid seed collision.
///
/// - Outputs:
///   - output: output tensor after fractional max pooling.
///   - row_pooling_sequence: row pooling sequence, needed to calculate gradient.
///   - col_pooling_sequence: column pooling sequence, needed to calculate gradient.
@inlinable @inline(__always)
public static func fractionalMaxPool<T: Numeric & TensorFlowScalar>(
  value: Tensor<T>,
  poolingRatio: [Double],
  pseudoRandom: Bool = false,
  overlapping: Bool = false,
  deterministic: Bool = false,
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> (output: Tensor<T>, rowPoolingSequence: Tensor<Int64>, colPoolingSequence: Tensor<Int64>) {
  let ret: (TensorHandle<T>, TensorHandle<Int64>, TensorHandle<Int64>) = #tfop("FractionalMaxPool",
    value,
    T$dtype: T.tensorFlowDataType,
    pooling_ratio: poolingRatio,
    pseudo_random: pseudoRandom,
    overlapping: overlapping,
    deterministic: deterministic,
    seed: seed,
    seed2: seed2)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Computes gradient of the FractionalMaxPool function.
///
/// - Parameters:
///   - orig_input: Original input for `fractional_max_pool`
///   - orig_output: Original output for `fractional_max_pool`
///   - out_backprop: 4-D with shape `[batch, height, width, channels]`.  Gradients
///     w.r.t. the output of `fractional_max_pool`.
///   - row_pooling_sequence: row pooling sequence, form pooling region with
///     col_pooling_sequence.
///   - col_pooling_sequence: column pooling sequence, form pooling region with
///     row_pooling sequence.
///
/// - Attr overlapping: When set to True, it means when pooling, the values at the boundary
///   of adjacent pooling cells are used by both cells. For example:
///
///   `index  0  1  2  3  4`
///
///   `value  20 5  16 3  7`
///
///   If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
///   The result would be [20, 16] for fractional max pooling.
///
/// - Output output: 4-D.  Gradients w.r.t. the input of `fractional_max_pool`.
@inlinable @inline(__always)
public static func fractionalMaxPoolGrad<T: Numeric & TensorFlowScalar>(
  origInput: Tensor<T>,
  origOutput: Tensor<T>,
  outBackprop: Tensor<T>,
  rowPoolingSequence: Tensor<Int64>,
  colPoolingSequence: Tensor<Int64>,
  overlapping: Bool = false
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("FractionalMaxPoolGrad",
    origInput,
    origOutput,
    outBackprop,
    rowPoolingSequence,
    colPoolingSequence,
    T$dtype: T.tensorFlowDataType,
    overlapping: overlapping)
  return Tensor(handle: ret)
}

/// Batch normalization.
///
/// Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
/// The size of 1D Tensors matches the dimension C of the 4D Tensors.
///
/// - Parameters:
///   - x: A 4D Tensor for input data.
///   - scale: A 1D Tensor for scaling factor, to scale the normalized x.
///   - offset: A 1D Tensor for offset, to shift to the normalized x.
///   - mean: A 1D Tensor for population mean. Used for inference only;
///     must be empty for training.
///   - variance: A 1D Tensor for population variance. Used for inference only;
///     must be empty for training.
///
/// - Attrs:
///   - T: The data type for the elements of input and output Tensors.
///   - epsilon: A small float number added to the variance of x.
///   - data_format: The data format for x and y. Either "NHWC" (default) or "NCHW".
///   - is_training: A bool value to indicate the operation is for training (default)
///     or inference.
///
/// - Outputs:
///   - y: A 4D Tensor for output data.
///   - batch_mean: A 1D Tensor for the computed batch mean, to be used by TensorFlow
///     to compute the running mean.
///   - batch_variance: A 1D Tensor for the computed batch variance, to be used by
///     TensorFlow to compute the running variance.
///   - reserve_space_1: A 1D Tensor for the computed batch mean, to be reused
///     in the gradient computation.
///   - reserve_space_2: A 1D Tensor for the computed batch variance (inverted variance
///     in the cuDNN case), to be reused in the gradient computation.
@inlinable @inline(__always)
public static func fusedBatchNorm<T: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>,
  scale: Tensor<T>,
  offset: Tensor<T>,
  mean: Tensor<T>,
  variance: Tensor<T>,
  epsilon: Double = 0.0001,
  dataFormat: DataFormat = .nhwc,
  isTraining: Bool = true
) -> (y: Tensor<T>, batchMean: Tensor<T>, batchVariance: Tensor<T>, reserveSpace1: Tensor<T>, reserveSpace2: Tensor<T>) {
  let ret: (TensorHandle<T>, TensorHandle<T>, TensorHandle<T>, TensorHandle<T>, TensorHandle<T>) = #tfop("FusedBatchNorm",
    x,
    scale,
    offset,
    mean,
    variance,
    T$dtype: T.tensorFlowDataType,
    epsilon: epsilon,
    data_format: dataFormat.cName,
    is_training: isTraining)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2), Tensor(handle: ret.3), Tensor(handle: ret.4))
}

/// Gradient for batch normalization.
///
/// Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
/// The size of 1D Tensors matches the dimension C of the 4D Tensors.
///
/// - Parameters:
///   - y_backprop: A 4D Tensor for the gradient with respect to y.
///   - x: A 4D Tensor for input data.
///   - scale: A 1D Tensor for scaling factor, to scale the normalized x.
///   - reserve_space_1: When is_training is True, a 1D Tensor for the computed batch
///     mean to be reused in gradient computation. When is_training is
///     False, a 1D Tensor for the population mean to be reused in both
///     1st and 2nd order gradient computation.
///   - reserve_space_2: When is_training is True, a 1D Tensor for the computed batch
///     variance (inverted variance in the cuDNN case) to be reused in
///     gradient computation. When is_training is False, a 1D Tensor
///     for the population variance to be reused in both 1st and 2nd
///     order gradient computation.
///
/// - Attrs:
///   - T: The data type for the elements of input and output Tensors.
///   - epsilon: A small float number added to the variance of x.
///   - data_format: The data format for y_backprop, x, x_backprop.
///     Either "NHWC" (default) or "NCHW".
///   - is_training: A bool value to indicate the operation is for training (default)
///     or inference.
///
/// - Outputs:
///   - x_backprop: A 4D Tensor for the gradient with respect to x.
///   - scale_backprop: A 1D Tensor for the gradient with respect to scale.
///   - offset_backprop: A 1D Tensor for the gradient with respect to offset.
///   - reserve_space_3: Unused placeholder to match the mean input in FusedBatchNorm.
///   - reserve_space_4: Unused placeholder to match the variance input
///     in FusedBatchNorm.
@inlinable @inline(__always)
public static func fusedBatchNormGrad<T: FloatingPoint & TensorFlowScalar>(
  yBackprop: Tensor<T>,
  _ x: Tensor<T>,
  scale: Tensor<T>,
  reserveSpace1: Tensor<T>,
  reserveSpace2: Tensor<T>,
  epsilon: Double = 0.0001,
  dataFormat: DataFormat = .nhwc,
  isTraining: Bool = true
) -> (xBackprop: Tensor<T>, scaleBackprop: Tensor<T>, offsetBackprop: Tensor<T>, reserveSpace3: Tensor<T>, reserveSpace4: Tensor<T>) {
  let ret: (TensorHandle<T>, TensorHandle<T>, TensorHandle<T>, TensorHandle<T>, TensorHandle<T>) = #tfop("FusedBatchNormGrad",
    yBackprop,
    x,
    scale,
    reserveSpace1,
    reserveSpace2,
    T$dtype: T.tensorFlowDataType,
    epsilon: epsilon,
    data_format: dataFormat.cName,
    is_training: isTraining)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2), Tensor(handle: ret.3), Tensor(handle: ret.4))
}

/// Gradient for batch normalization.
///
/// Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
/// The size of 1D Tensors matches the dimension C of the 4D Tensors.
///
/// - Parameters:
///   - y_backprop: A 4D Tensor for the gradient with respect to y.
///   - x: A 4D Tensor for input data.
///   - scale: A 1D Tensor for scaling factor, to scale the normalized x.
///   - reserve_space_1: When is_training is True, a 1D Tensor for the computed batch
///     mean to be reused in gradient computation. When is_training is
///     False, a 1D Tensor for the population mean to be reused in both
///     1st and 2nd order gradient computation.
///   - reserve_space_2: When is_training is True, a 1D Tensor for the computed batch
///     variance (inverted variance in the cuDNN case) to be reused in
///     gradient computation. When is_training is False, a 1D Tensor
///     for the population variance to be reused in both 1st and 2nd
///     order gradient computation.
///
/// - Attrs:
///   - T: The data type for the elements of input and output Tensors.
///   - U: The data type for the scale, offset, mean, and variance.
///   - epsilon: A small float number added to the variance of x.
///   - data_format: The data format for y_backprop, x, x_backprop.
///     Either "NHWC" (default) or "NCHW".
///   - is_training: A bool value to indicate the operation is for training (default)
///     or inference.
///
/// - Outputs:
///   - x_backprop: A 4D Tensor for the gradient with respect to x.
///   - scale_backprop: A 1D Tensor for the gradient with respect to scale.
///   - offset_backprop: A 1D Tensor for the gradient with respect to offset.
///   - reserve_space_3: Unused placeholder to match the mean input in FusedBatchNorm.
///   - reserve_space_4: Unused placeholder to match the variance input
///     in FusedBatchNorm.
@inlinable @inline(__always)
public static func fusedBatchNormGradV2<T: FloatingPoint & TensorFlowScalar, U: FloatingPoint & TensorFlowScalar>(
  yBackprop: Tensor<T>,
  _ x: Tensor<T>,
  scale: Tensor<Float>,
  reserveSpace1: Tensor<U>,
  reserveSpace2: Tensor<U>,
  epsilon: Double = 0.0001,
  dataFormat: DataFormat = .nhwc,
  isTraining: Bool = true
) -> (xBackprop: Tensor<T>, scaleBackprop: Tensor<U>, offsetBackprop: Tensor<U>, reserveSpace3: Tensor<U>, reserveSpace4: Tensor<U>) {
  let ret: (TensorHandle<T>, TensorHandle<U>, TensorHandle<U>, TensorHandle<U>, TensorHandle<U>) = #tfop("FusedBatchNormGradV2",
    yBackprop,
    x,
    scale,
    reserveSpace1,
    reserveSpace2,
    T$dtype: T.tensorFlowDataType,
    U$dtype: U.tensorFlowDataType,
    epsilon: epsilon,
    data_format: dataFormat.cName,
    is_training: isTraining)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2), Tensor(handle: ret.3), Tensor(handle: ret.4))
}

/// Batch normalization.
///
/// Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
/// The size of 1D Tensors matches the dimension C of the 4D Tensors.
///
/// - Parameters:
///   - x: A 4D Tensor for input data.
///   - scale: A 1D Tensor for scaling factor, to scale the normalized x.
///   - offset: A 1D Tensor for offset, to shift to the normalized x.
///   - mean: A 1D Tensor for population mean. Used for inference only;
///     must be empty for training.
///   - variance: A 1D Tensor for population variance. Used for inference only;
///     must be empty for training.
///
/// - Attrs:
///   - T: The data type for the elements of input and output Tensors.
///   - U: The data type for the scale, offset, mean, and variance.
///   - epsilon: A small float number added to the variance of x.
///   - data_format: The data format for x and y. Either "NHWC" (default) or "NCHW".
///   - is_training: A bool value to indicate the operation is for training (default)
///     or inference.
///
/// - Outputs:
///   - y: A 4D Tensor for output data.
///   - batch_mean: A 1D Tensor for the computed batch mean, to be used by TensorFlow
///     to compute the running mean.
///   - batch_variance: A 1D Tensor for the computed batch variance, to be used by
///     TensorFlow to compute the running variance.
///   - reserve_space_1: A 1D Tensor for the computed batch mean, to be reused
///     in the gradient computation.
///   - reserve_space_2: A 1D Tensor for the computed batch variance (inverted variance
///     in the cuDNN case), to be reused in the gradient computation.
@inlinable @inline(__always)
public static func fusedBatchNormV2<T: FloatingPoint & TensorFlowScalar, U: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>,
  scale: Tensor<U>,
  offset: Tensor<U>,
  mean: Tensor<U>,
  variance: Tensor<U>,
  epsilon: Double = 0.0001,
  dataFormat: DataFormat = .nhwc,
  isTraining: Bool = true
) -> (y: Tensor<T>, batchMean: Tensor<U>, batchVariance: Tensor<U>, reserveSpace1: Tensor<U>, reserveSpace2: Tensor<U>) {
  let ret: (TensorHandle<T>, TensorHandle<U>, TensorHandle<U>, TensorHandle<U>, TensorHandle<U>) = #tfop("FusedBatchNormV2",
    x,
    scale,
    offset,
    mean,
    variance,
    T$dtype: T.tensorFlowDataType,
    U$dtype: U.tensorFlowDataType,
    epsilon: epsilon,
    data_format: dataFormat.cName,
    is_training: isTraining)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2), Tensor(handle: ret.3), Tensor(handle: ret.4))
}

/// Performs a padding as a preprocess during a convolution.
///
/// Similar to FusedResizeAndPadConv2d, this op allows for an optimized
/// implementation where the spatial padding transformation stage is fused with the
/// im2col lookup, but in this case without the bilinear filtering required for
/// resizing. Fusing the padding prevents the need to write out the intermediate
/// results as whole tensors, reducing memory pressure, and we can get some latency
/// gains by merging the transformation calculations.
/// The data_format attribute for Conv2D isn't supported by this op, and 'NHWC'
/// order is used instead.
/// Internally this op uses a single per-graph scratch buffer, which means that it
/// will block if multiple versions are being run in parallel. This is because this
/// operator is primarily an optimization to minimize memory usage.
///
/// - Parameters:
///   - input: 4-D with shape `[batch, in_height, in_width, in_channels]`.
///   - paddings: A two-column matrix specifying the padding sizes. The number of
///     rows must be the same as the rank of `input`.
///   - filter: 4-D with shape
///     `[filter_height, filter_width, in_channels, out_channels]`.
///
/// - Attrs:
///   - strides: 1-D of length 4.  The stride of the sliding window for each dimension
///     of `input`. Must be in the same order as the dimension specified with format.
///   - padding: The type of padding algorithm to use.
@inlinable @inline(__always)
public static func fusedPadConv2D<T: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>,
  paddings: Tensor<Int32>,
  filter: Tensor<T>,
  mode: Mode5,
  strides: [Int32],
  padding: Padding
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("FusedPadConv2D",
    input,
    paddings,
    filter,
    T$dtype: T.tensorFlowDataType,
    mode: mode.cName,
    strides: strides,
    padding: padding.cName)
  return Tensor(handle: ret)
}

/// Performs a resize and padding as a preprocess during a convolution.
///
/// It's often possible to do spatial transformations more efficiently as part of
/// the packing stage of a convolution, so this op allows for an optimized
/// implementation where these stages are fused together. This prevents the need to
/// write out the intermediate results as whole tensors, reducing memory pressure,
/// and we can get some latency gains by merging the transformation calculations.
/// The data_format attribute for Conv2D isn't supported by this op, and defaults to
/// 'NHWC' order.
/// Internally this op uses a single per-graph scratch buffer, which means that it
/// will block if multiple versions are being run in parallel. This is because this
/// operator is primarily an optimization to minimize memory usage.
///
/// - Parameters:
///   - input: 4-D with shape `[batch, in_height, in_width, in_channels]`.
///   - size: A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
///     new size for the images.
///   - paddings: A two-column matrix specifying the padding sizes. The number of
///     rows must be the same as the rank of `input`.
///   - filter: 4-D with shape
///     `[filter_height, filter_width, in_channels, out_channels]`.
///
/// - Attrs:
///   - resize_align_corners: If true, the centers of the 4 corner pixels of the input and output tensors are
///     aligned, preserving the values at the corner pixels. Defaults to false.
///   - strides: 1-D of length 4.  The stride of the sliding window for each dimension
///     of `input`. Must be in the same order as the dimension specified with format.
///   - padding: The type of padding algorithm to use.
@inlinable @inline(__always)
public static func fusedResizeAndPadConv2D<T: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>,
  size: Tensor<Int32>,
  paddings: Tensor<Int32>,
  filter: Tensor<T>,
  resizeAlignCorners: Bool = false,
  mode: Mode5,
  strides: [Int32],
  padding: Padding
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("FusedResizeAndPadConv2D",
    input,
    size,
    paddings,
    filter,
    T$dtype: T.tensorFlowDataType,
    resize_align_corners: resizeAlignCorners,
    mode: mode.cName,
    strides: strides,
    padding: padding.cName)
  return Tensor(handle: ret)
}

/// Computes the GRU cell forward propagation for 1 time step.
///
/// Args
///     x: Input to the GRU cell.
///     h_prev: State input from the previous GRU cell.
///     w_ru: Weight matrix for the reset and update gate.
///     w_c: Weight matrix for the cell connection gate.
///     b_ru: Bias vector for the reset and update gate.
///     b_c: Bias vector for the cell connection gate.
///
/// Returns
///     r: Output of the reset gate.
///     u: Output of the update gate.
///     c: Output of the cell connection gate.
///     h: Current state of the GRU cell.
///
/// Note on notation of the variables:
///
/// Concatenation of a and b is represented by a_b
/// Element-wise dot product of a and b is represented by ab
/// Element-wise dot product is represented by \circ
/// Matrix multiplication is represented by *
///
/// Biases are initialized with :
/// `b_ru` - constant_initializer(1.0)
/// `b_c` - constant_initializer(0.0)
///
/// This kernel op implements the following mathematical equations:
///
/// ```
/// x_h_prev = [x, h_prev]
///
/// [r_bar u_bar] = x_h_prev * w_ru + b_ru
///
/// r = sigmoid(r_bar)
/// u = sigmoid(u_bar)
///
/// h_prevr = h_prev \circ r
///
/// x_h_prevr = [x h_prevr]
///
/// c_bar = x_h_prevr * w_c + b_c
/// c = tanh(c_bar)
///
/// h = (1-u) \circ c + u \circ h_prev
/// ```
@inlinable @inline(__always)
public static func gRUBlockCell<T: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>,
  hPrev: Tensor<T>,
  wRu: Tensor<T>,
  wC: Tensor<T>,
  bRu: Tensor<T>,
  bC: Tensor<T>
) -> (r: Tensor<T>, u: Tensor<T>, c: Tensor<T>, h: Tensor<T>) {
  let ret: (TensorHandle<T>, TensorHandle<T>, TensorHandle<T>, TensorHandle<T>) = #tfop("GRUBlockCell",
    x,
    hPrev,
    wRu,
    wC,
    bRu,
    bC,
    T$dtype: T.tensorFlowDataType)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2), Tensor(handle: ret.3))
}

/// Computes the GRU cell back-propagation for 1 time step.
///
/// Args
///     x: Input to the GRU cell.
///     h_prev: State input from the previous GRU cell.
///     w_ru: Weight matrix for the reset and update gate.
///     w_c: Weight matrix for the cell connection gate.
///     b_ru: Bias vector for the reset and update gate.
///     b_c: Bias vector for the cell connection gate.
///     r: Output of the reset gate.
///     u: Output of the update gate.
///     c: Output of the cell connection gate.
///     d_h: Gradients of the h_new wrt to objective function.
///
/// Returns
///     d_x: Gradients of the x wrt to objective function.
///     d_h_prev: Gradients of the h wrt to objective function.
///     d_c_bar Gradients of the c_bar wrt to objective function.
///     d_r_bar_u_bar Gradients of the r_bar & u_bar wrt to objective function.
///
/// This kernel op implements the following mathematical equations:
///
/// Note on notation of the variables:
///
/// Concatenation of a and b is represented by a_b
/// Element-wise dot product of a and b is represented by ab
/// Element-wise dot product is represented by \circ
/// Matrix multiplication is represented by *
///
/// Additional notes for clarity:
///
/// `w_ru` can be segmented into 4 different matrices.
/// ```
/// w_ru = [w_r_x w_u_x
///         w_r_h_prev w_u_h_prev]
/// ```
/// Similarly, `w_c` can be segmented into 2 different matrices.
/// ```
/// w_c = [w_c_x w_c_h_prevr]
/// ```
/// Same goes for biases.
/// ```
/// b_ru = [b_ru_x b_ru_h]
/// b_c = [b_c_x b_c_h]
/// ```
/// Another note on notation:
/// ```
/// d_x = d_x_component_1 + d_x_component_2
///
/// where d_x_component_1 = d_r_bar * w_r_x^T + d_u_bar * w_r_x^T
/// and d_x_component_2 = d_c_bar * w_c_x^T
///
/// d_h_prev = d_h_prev_component_1 + d_h_prevr \circ r + d_h \circ u
/// where d_h_prev_componenet_1 = d_r_bar * w_r_h_prev^T + d_u_bar * w_r_h_prev^T
/// ```
///
/// Mathematics behind the Gradients below:
/// ```
/// d_c_bar = d_h \circ (1-u) \circ (1-c \circ c)
/// d_u_bar = d_h \circ (h-c) \circ u \circ (1-u)
///
/// d_r_bar_u_bar = [d_r_bar d_u_bar]
///
/// [d_x_component_1 d_h_prev_component_1] = d_r_bar_u_bar * w_ru^T
///
/// [d_x_component_2 d_h_prevr] = d_c_bar * w_c^T
///
/// d_x = d_x_component_1 + d_x_component_2
///
/// d_h_prev = d_h_prev_component_1 + d_h_prevr \circ r + u
/// ```
/// Below calculation is performed in the python wrapper for the Gradients
/// (not in the gradient kernel.)
/// ```
/// d_w_ru = x_h_prevr^T * d_c_bar
///
/// d_w_c = x_h_prev^T * d_r_bar_u_bar
///
/// d_b_ru = sum of d_r_bar_u_bar along axis = 0
///
/// d_b_c = sum of d_c_bar along axis = 0
/// ```
@inlinable @inline(__always)
public static func gRUBlockCellGrad<T: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>,
  hPrev: Tensor<T>,
  wRu: Tensor<T>,
  wC: Tensor<T>,
  bRu: Tensor<T>,
  bC: Tensor<T>,
  r: Tensor<T>,
  u: Tensor<T>,
  c: Tensor<T>,
  dH: Tensor<T>
) -> (dX: Tensor<T>, dHPrev: Tensor<T>, dCBar: Tensor<T>, dRBarUBar: Tensor<T>) {
  let ret: (TensorHandle<T>, TensorHandle<T>, TensorHandle<T>, TensorHandle<T>) = #tfop("GRUBlockCellGrad",
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
    T$dtype: T.tensorFlowDataType)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2), Tensor(handle: ret.3))
}

/// Gather slices from `params` according to `indices`.
///
/// `indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
/// Produces an output tensor with shape `indices.shape + params.shape[1:]` where:
///
/// ```python
///     # Scalar indices
///     output[:, ..., :] = params[indices, :, ... :]
///
///     # Vector indices
///     output[i, :, ..., :] = params[indices[i], :, ... :]
///
///     # Higher rank indices
///     output[i, ..., j, :, ... :] = params[indices[i, ..., j], :, ..., :]
/// ```
///
/// If `indices` is a permutation and `len(indices) == params.shape[0]` then
/// this operation will permute `params` accordingly.
///
/// `validate_indices`: DEPRECATED. If this operation is assigned to CPU, values in
/// `indices` are always validated to be within range. If assigned to GPU,
/// out-of-bound indices result in safe but unspecified behavior, which may include
/// raising an error.
///
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/Gather.png" alt>
/// </div>
@inlinable @inline(__always)
public static func gather<Tparams: TensorFlowScalar, Tindices: BinaryInteger & TensorFlowScalar>(
  params: Tensor<Tparams>,
  indices: Tensor<Tindices>,
  validateIndices: Bool = true
) -> Tensor<Tparams> {
  let ret: TensorHandle<Tparams> = #tfop("Gather",
    params,
    indices,
    Tparams$dtype: Tparams.tensorFlowDataType,
    Tindices$dtype: Tindices.tensorFlowDataType,
    validate_indices: validateIndices)
  return Tensor(handle: ret)
}

/// Gather slices from `params` into a Tensor with shape specified by `indices`.
///
/// `indices` is an K-dimensional integer tensor, best thought of as a
/// (K-1)-dimensional tensor of indices into `params`, where each element defines a
/// slice of `params`:
///
///     output[\\(i_0, ..., i_{K-2}\\)] = params[indices[\\(i_0, ..., i_{K-2}\\)]]
///
/// Whereas in `tf.gather` `indices` defines slices into the first
/// dimension of `params`, in `tf.gather_nd`, `indices` defines slices into the
/// first `N` dimensions of `params`, where `N = indices.shape[-1]`.
///
/// The last dimension of `indices` can be at most the rank of
/// `params`:
///
///     indices.shape[-1] <= params.rank
///
/// The last dimension of `indices` corresponds to elements
/// (if `indices.shape[-1] == params.rank`) or slices
/// (if `indices.shape[-1] < params.rank`) along dimension `indices.shape[-1]`
/// of `params`.  The output tensor has shape
///
///     indices.shape[:-1] + params.shape[indices.shape[-1]:]
///
/// Note that on CPU, if an out of bound index is found, an error is returned.
/// On GPU, if an out of bound index is found, a 0 is stored in the
/// corresponding output value.
///
/// Some examples below.
///
/// Simple indexing into a matrix:
///
/// ```python
///     indices = [[0, 0], [1, 1]]
///     params = [['a', 'b'], ['c', 'd']]
///     output = ['a', 'd']
/// ```
///
/// Slice indexing into a matrix:
///
/// ```python
///     indices = [[1], [0]]
///     params = [['a', 'b'], ['c', 'd']]
///     output = [['c', 'd'], ['a', 'b']]
/// ```
///
/// Indexing into a 3-tensor:
///
/// ```python
///     indices = [[1]]
///     params = [[['a0', 'b0'], ['c0', 'd0']],
///               [['a1', 'b1'], ['c1', 'd1']]]
///     output = [[['a1', 'b1'], ['c1', 'd1']]]
///
///
///     indices = [[0, 1], [1, 0]]
///     params = [[['a0', 'b0'], ['c0', 'd0']],
///               [['a1', 'b1'], ['c1', 'd1']]]
///     output = [['c0', 'd0'], ['a1', 'b1']]
///
///
///     indices = [[0, 0, 1], [1, 0, 1]]
///     params = [[['a0', 'b0'], ['c0', 'd0']],
///               [['a1', 'b1'], ['c1', 'd1']]]
///     output = ['b0', 'b1']
/// ```
///
/// Batched indexing into a matrix:
///
/// ```python
///     indices = [[[0, 0]], [[0, 1]]]
///     params = [['a', 'b'], ['c', 'd']]
///     output = [['a'], ['b']]
/// ```
///
/// Batched slice indexing into a matrix:
///
/// ```python
///     indices = [[[1]], [[0]]]
///     params = [['a', 'b'], ['c', 'd']]
///     output = [[['c', 'd']], [['a', 'b']]]
/// ```
///
/// Batched indexing into a 3-tensor:
///
/// ```python
///     indices = [[[1]], [[0]]]
///     params = [[['a0', 'b0'], ['c0', 'd0']],
///               [['a1', 'b1'], ['c1', 'd1']]]
///     output = [[[['a1', 'b1'], ['c1', 'd1']]],
///               [[['a0', 'b0'], ['c0', 'd0']]]]
///
///     indices = [[[0, 1], [1, 0]], [[0, 0], [1, 1]]]
///     params = [[['a0', 'b0'], ['c0', 'd0']],
///               [['a1', 'b1'], ['c1', 'd1']]]
///     output = [[['c0', 'd0'], ['a1', 'b1']],
///               [['a0', 'b0'], ['c1', 'd1']]]
///
///
///     indices = [[[0, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 1, 0]]]
///     params = [[['a0', 'b0'], ['c0', 'd0']],
///               [['a1', 'b1'], ['c1', 'd1']]]
///     output = [['b0', 'b1'], ['d0', 'c1']]
/// ```
///
/// See also `tf.gather` and `tf.batch_gather`.
///
/// - Parameters:
///   - params: The tensor from which to gather values.
///   - indices: Index tensor.
///
/// - Output output: Values from `params` gathered from indices given by `indices`, with
///   shape `indices.shape[:-1] + params.shape[indices.shape[-1]:]`.
@inlinable @inline(__always)
public static func gatherNd<Tparams: TensorFlowScalar, Tindices: BinaryInteger & TensorFlowScalar>(
  params: Tensor<Tparams>,
  indices: Tensor<Tindices>
) -> Tensor<Tparams> {
  let ret: TensorHandle<Tparams> = #tfop("GatherNd",
    params,
    indices,
    Tparams$dtype: Tparams.tensorFlowDataType,
    Tindices$dtype: Tindices.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Gather slices from `params` axis `axis` according to `indices`.
///
/// `indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
/// Produces an output tensor with shape `params.shape[:axis] + indices.shape +
/// params.shape[axis + 1:]` where:
///
/// ```python
///     # Scalar indices (output is rank(params) - 1).
///     output[a_0, ..., a_n, b_0, ..., b_n] =
///       params[a_0, ..., a_n, indices, b_0, ..., b_n]
///
///     # Vector indices (output is rank(params)).
///     output[a_0, ..., a_n, i, b_0, ..., b_n] =
///       params[a_0, ..., a_n, indices[i], b_0, ..., b_n]
///
///     # Higher rank indices (output is rank(params) + rank(indices) - 1).
///     output[a_0, ..., a_n, i, ..., j, b_0, ... b_n] =
///       params[a_0, ..., a_n, indices[i, ..., j], b_0, ..., b_n]
/// ```
///
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/Gather.png" alt>
/// </div>
///
/// Note that on CPU, if an out of bound index is found, an error is returned.
/// On GPU, if an out of bound index is found, a 0 is stored in the
/// corresponding output value.
///
/// See also `tf.batch_gather` and `tf.gather_nd`.
///
/// - Parameters:
///   - params: The tensor from which to gather values. Must be at least rank
///     `axis + 1`.
///   - indices: Index tensor. Must be in range `[0, params.shape[axis])`.
///   - axis: The axis in `params` to gather `indices` from. Defaults to the first
///     dimension. Supports negative indexes.
///
/// - Output output: Values from `params` gathered from indices given by `indices`, with
///   shape `params.shape[:axis] + indices.shape + params.shape[axis + 1:]`.
@inlinable @inline(__always)
public static func gatherV2<Tparams: TensorFlowScalar, Tindices: BinaryInteger & TensorFlowScalar, Taxis: BinaryInteger & TensorFlowScalar>(
  params: Tensor<Tparams>,
  indices: Tensor<Tindices>,
  axis: Tensor<Taxis>
) -> Tensor<Tparams> {
  let ret: TensorHandle<Tparams> = #tfop("GatherV2",
    params,
    indices,
    axis,
    Tparams$dtype: Tparams.tensorFlowDataType,
    Tindices$dtype: Tindices.tensorFlowDataType,
    Taxis$dtype: Taxis.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Re-configures the GCS block cache with the new configuration values.
///
/// If the values are the same as already configured values, this op is a no-op. If
/// they are different, the current contents of the block cache is dropped, and a
/// new block cache is created fresh.
@inlinable @inline(__always)
public static func gcsConfigureBlockCache(
  maxCacheSize: Tensor<UInt64>,
  blockSize: Tensor<UInt64>,
  maxStaleness: Tensor<UInt64>
) {
  return #tfop("GcsConfigureBlockCache",
    maxCacheSize,
    blockSize,
    maxStaleness)
}

/// Configures the credentials used by the GCS client of the local TF runtime.
///
/// The json input can be of the format:
///
/// 1. Refresh Token:
/// {
///   "client_id": "<redacted>",
///   "client_secret": "<redacted>",
///   "refresh_token: "<redacted>",
///   "type": "authorized_user",
/// }
///
/// 2. Service Account:
/// {
///   "type": "service_account",
///   "project_id": "<redacted>",
///   "private_key_id": "<redacted>",
///   "private_key": "------BEGIN PRIVATE KEY-----\n<REDACTED>\n-----END PRIVATE KEY------\n",
///   "client_email": "<REDACTED>@<REDACTED>.iam.gserviceaccount.com",
///   "client_id": "<REDACTED>",
///   # Some additional fields elided
/// }
///
/// Note the credentials established through this method are shared across all
/// sessions run on this runtime.
///
/// Note be sure to feed the inputs to this op to ensure the credentials are not
/// stored in a constant op within the graph that might accidentally be checkpointed
/// or in other ways be persisted or exfiltrated.
@inlinable @inline(__always)
public static func gcsConfigureCredentials(
  json: StringTensor
) {
  return #tfop("GcsConfigureCredentials",
    json)
}

/// Generates serialized partition messages suitable for batch reads.
///
/// This op should not be used directly by clients. Instead, the
/// bigquery_reader_ops.py file defines a clean interface to the reader.
///
/// - Attrs:
///   - project_id: GCP project ID.
///   - dataset_id: BigQuery Dataset ID.
///   - table_id: Table to read.
///   - columns: List of columns to read. Leave empty to read all columns.
///   - timestamp_millis: Table snapshot timestamp in millis since epoch. Relative
///     (negative or zero) snapshot times are not allowed. For more details, see
///     'Table Decorators' in BigQuery docs.
///   - num_partitions: Number of partitions to split the table into.
///   - test_end_point: Do not use. For testing purposes only.
///
/// - Output partitions: Serialized table partitions.
@inlinable @inline(__always)
public static func generateBigQueryReaderPartitions(
  projectId: String,
  datasetId: String,
  tableId: String,
  columns: [String],
  timestampMillis: Int64,
  numPartitions: Int64,
  testEndPoint: String
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("GenerateBigQueryReaderPartitions",
    project_id: projectId,
    dataset_id: datasetId,
    table_id: tableId,
    columns: columns,
    timestamp_millis: timestampMillis,
    num_partitions: numPartitions,
    test_end_point: testEndPoint)
  return StringTensor(handle: ret)
}

/// Given a path to new and old vocabulary files, returns a remapping Tensor of
///
/// length `num_new_vocab`, where `remapping[i]` contains the row number in the old
/// vocabulary that corresponds to row `i` in the new vocabulary (starting at line
/// `new_vocab_offset` and up to `num_new_vocab` entities), or `-1` if entry `i`
/// in the new vocabulary is not in the old vocabulary.  The old vocabulary is
/// constrained to the first `old_vocab_size` entries if `old_vocab_size` is not the
/// default value of -1.
///
/// `num_vocab_offset` enables
/// use in the partitioned variable case, and should generally be set through
/// examining partitioning info.  The format of the files should be a text file,
/// with each line containing a single entity within the vocabulary.
///
/// For example, with `new_vocab_file` a text file containing each of the following
/// elements on a single line: `[f0, f1, f2, f3]`, old_vocab_file = [f1, f0, f3],
/// `num_new_vocab = 3, new_vocab_offset = 1`, the returned remapping would be
/// `[0, -1, 2]`.
///
/// The op also returns a count of how many entries in the new vocabulary
/// were present in the old vocabulary, which is used to calculate the number of
/// values to initialize in a weight matrix remapping
///
/// This functionality can be used to remap both row vocabularies (typically,
/// features) and column vocabularies (typically, classes) from TensorFlow
/// checkpoints.  Note that the partitioning logic relies on contiguous vocabularies
/// corresponding to div-partitioned variables.  Moreover, the underlying remapping
/// uses an IndexTable (as opposed to an inexact CuckooTable), so client code should
/// use the corresponding index_table_from_file() as the FeatureColumn framework
/// does (as opposed to tf.feature_to_id(), which uses a CuckooTable).
///
/// - Parameters:
///   - new_vocab_file: Path to the new vocab file.
///   - old_vocab_file: Path to the old vocab file.
///
/// - Attrs:
///   - new_vocab_offset: How many entries into the new vocab file to start reading.
///   - num_new_vocab: Number of entries in the new vocab file to remap.
///   - old_vocab_size: Number of entries in the old vocab file to consider.  If -1,
///     use the entire old vocabulary.
///
/// - Outputs:
///   - remapping: A Tensor of length num_new_vocab where the element at index i
///     is equal to the old ID that maps to the new ID i.  This element is -1 for any
///     new ID that is not found in the old vocabulary.
///   - num_present: Number of new vocab entries found in old vocab.
@inlinable @inline(__always)
public static func generateVocabRemapping(
  newVocabFile: StringTensor,
  oldVocabFile: StringTensor,
  newVocabOffset: Int64,
  numNewVocab: Int64,
  oldVocabSize: Int64 = -1
) -> (remapping: Tensor<Int64>, numPresent: Tensor<Int32>) {
  let ret: (TensorHandle<Int64>, TensorHandle<Int32>) = #tfop("GenerateVocabRemapping",
    newVocabFile,
    oldVocabFile,
    new_vocab_offset: newVocabOffset,
    num_new_vocab: numNewVocab,
    old_vocab_size: oldVocabSize)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

/// Store the input tensor in the state of the current session.
///
/// - Parameter value: The tensor to be stored.
///
/// - Output handle: The handle for the tensor stored in the session state, represented
///   as a string.
@inlinable @inline(__always)
public static func getSessionHandle<T: TensorFlowScalar>(
  value: Tensor<T>
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("GetSessionHandle",
    value,
    T$dtype: T.tensorFlowDataType)
  return StringTensor(handle: ret)
}

/// Get the value of the tensor specified by its handle.
///
/// - Parameter handle: The handle for a tensor stored in the session state.
///
/// - Attr dtype: The type of the output value.
///
/// - Output value: The tensor for the given handle.
@inlinable @inline(__always)
public static func getSessionTensor<Dtype: TensorFlowScalar>(
  handle: StringTensor
) -> Tensor<Dtype> {
  let ret: TensorHandle<Dtype> = #tfop("GetSessionTensor",
    handle,
    dtype$dtype: Dtype.tensorFlowDataType)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func graphDefVersion(
) -> Tensor<Int32> {
  let ret: TensorHandle<Int32> = #tfop("GraphDefVersion")
  return Tensor(handle: ret)
}

/// Returns the truth value of (x > y) element-wise.
///
/// *NOTE*: `Greater` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@inlinable @inline(__always)
public static func greater<T: Numeric & TensorFlowScalar>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<Bool> {
  let ret: TensorHandle<Bool> = #tfop("Greater",
    x,
    y,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Returns the truth value of (x >= y) element-wise.
///
/// *NOTE*: `GreaterEqual` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@inlinable @inline(__always)
public static func greaterEqual<T: Numeric & TensorFlowScalar>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<Bool> {
  let ret: TensorHandle<Bool> = #tfop("GreaterEqual",
    x,
    y,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Gives a guarantee to the TF runtime that the input tensor is a constant.
///
/// The runtime is then free to make optimizations based on this.
///
/// Only accepts value typed tensors as inputs and rejects resource variable handles
/// as input.
///
/// Returns the input tensor without modification.
@inlinable @inline(__always)
public static func guaranteeConst<T: TensorFlowScalar>(
  _ input: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("GuaranteeConst",
    input,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Convert one or more images from HSV to RGB.
///
/// Outputs a tensor of the same shape as the `images` tensor, containing the RGB
/// value of the pixels. The output is only well defined if the value in `images`
/// are in `[0,1]`.
///
/// See `rgb_to_hsv` for a description of the HSV encoding.
///
/// - Parameter images: 1-D or higher rank. HSV data to convert. Last dimension must be size 3.
///
/// - Output output: `images` converted to RGB.
@inlinable @inline(__always)
public static func hSVToRGB<T: FloatingPoint & TensorFlowScalar>(
  images: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("HSVToRGB",
    images,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Return histogram of values.
///
/// Given the tensor `values`, this operation returns a rank 1 histogram counting
/// the number of entries in `values` that fall into every bin.  The bins are
/// equal width and determined by the arguments `value_range` and `nbins`.
///
/// ```python
/// # Bins will be:  (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
/// nbins = 5
/// value_range = [0.0, 5.0]
/// new_values = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]
///
/// with tf.get_default_session() as sess:
///   hist = tf.histogram_fixed_width(new_values, value_range, nbins=5)
///   variables.global_variables_initializer().run()
///   sess.run(hist) => [2, 1, 1, 0, 2]
/// ```
///
/// - Parameters:
///   - values: Numeric `Tensor`.
///   - value_range: Shape [2] `Tensor` of same `dtype` as `values`.
///     values <= value_range[0] will be mapped to hist[0],
///     values >= value_range[1] will be mapped to hist[-1].
///   - nbins: Scalar `int32 Tensor`.  Number of histogram bins.
///
/// - Output out: A 1-D `Tensor` holding histogram of values.
@inlinable @inline(__always)
public static func histogramFixedWidth<T: Numeric & TensorFlowScalar, Dtype: BinaryInteger & TensorFlowScalar>(
  _ values: Tensor<T>,
  valueRange: Tensor<T>,
  nbins: Tensor<Int32>
) -> Tensor<Dtype> {
  let ret: TensorHandle<Dtype> = #tfop("HistogramFixedWidth",
    values,
    valueRange,
    nbins,
    T$dtype: T.tensorFlowDataType,
    dtype$dtype: Dtype.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Outputs a `Summary` protocol buffer with a histogram.
///
/// The generated
/// [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
/// has one summary value containing a histogram for `values`.
///
/// This op reports an `InvalidArgument` error if any value is not finite.
///
/// - Parameters:
///   - tag: Scalar.  Tag to use for the `Summary.Value`.
///   - values: Any shape. Values to use to build the histogram.
///
/// - Output summary: Scalar. Serialized `Summary` protocol buffer.
@inlinable @inline(__always)
public static func histogramSummary<T: Numeric & TensorFlowScalar>(
  tag: StringTensor,
  _ values: Tensor<T>
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("HistogramSummary",
    tag,
    values,
    T$dtype: T.tensorFlowDataType)
  return StringTensor(handle: ret)
}

/// Inverse fast Fourier transform.
///
/// Computes the inverse 1-dimensional discrete Fourier transform over the
/// inner-most dimension of `input`.
///
/// - Parameter input: A complex tensor.
///
/// - Output output: A complex tensor of the same shape as `input`. The inner-most
///     dimension of `input` is replaced with its inverse 1D Fourier transform.
///
///   @compatibility(numpy)
///   Equivalent to np.fft.ifft
///   @end_compatibility
@inlinable @inline(__always)
public static func iFFT<Tcomplex: TensorFlowScalar>(
  _ input: Tensor<Tcomplex>
) -> Tensor<Tcomplex> {
  let ret: TensorHandle<Tcomplex> = #tfop("IFFT",
    input,
    Tcomplex$dtype: Tcomplex.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Inverse 2D fast Fourier transform.
///
/// Computes the inverse 2-dimensional discrete Fourier transform over the
/// inner-most 2 dimensions of `input`.
///
/// - Parameter input: A complex tensor.
///
/// - Output output: A complex tensor of the same shape as `input`. The inner-most 2
///     dimensions of `input` are replaced with their inverse 2D Fourier transform.
///
///   @compatibility(numpy)
///   Equivalent to np.fft.ifft2
///   @end_compatibility
@inlinable @inline(__always)
public static func iFFT2D<Tcomplex: TensorFlowScalar>(
  _ input: Tensor<Tcomplex>
) -> Tensor<Tcomplex> {
  let ret: TensorHandle<Tcomplex> = #tfop("IFFT2D",
    input,
    Tcomplex$dtype: Tcomplex.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Inverse 3D fast Fourier transform.
///
/// Computes the inverse 3-dimensional discrete Fourier transform over the
/// inner-most 3 dimensions of `input`.
///
/// - Parameter input: A complex64 tensor.
///
/// - Output output: A complex64 tensor of the same shape as `input`. The inner-most 3
///     dimensions of `input` are replaced with their inverse 3D Fourier transform.
///
///   @compatibility(numpy)
///   Equivalent to np.fft.ifftn with 3 dimensions.
///   @end_compatibility
@inlinable @inline(__always)
public static func iFFT3D<Tcomplex: TensorFlowScalar>(
  _ input: Tensor<Tcomplex>
) -> Tensor<Tcomplex> {
  let ret: TensorHandle<Tcomplex> = #tfop("IFFT3D",
    input,
    Tcomplex$dtype: Tcomplex.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Return a tensor with the same shape and contents as the input tensor or value.
@inlinable @inline(__always)
public static func identity<T: TensorFlowScalar>(
  _ input: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Identity",
    input,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Compute the lower regularized incomplete Gamma function `P(a, x)`.
///
/// The lower regularized incomplete Gamma function is defined as:
///
///
/// \\(P(a, x) = gamma(a, x) / Gamma(a) = 1 - Q(a, x)\\)
///
/// where
///
/// \\(gamma(a, x) = \\int_{0}^{x} t^{a-1} exp(-t) dt\\)
///
/// is the lower incomplete Gamma function.
///
/// Note, above `Q(a, x)` (`Igammac`) is the upper regularized complete
/// Gamma function.
@inlinable @inline(__always)
public static func igamma<T: FloatingPoint & TensorFlowScalar>(
  _ a: Tensor<T>,
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Igamma",
    a,
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the gradient of `igamma(a, x)` wrt `a`.
@inlinable @inline(__always)
public static func igammaGradA<T: FloatingPoint & TensorFlowScalar>(
  _ a: Tensor<T>,
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("IgammaGradA",
    a,
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Compute the upper regularized incomplete Gamma function `Q(a, x)`.
///
/// The upper regularized incomplete Gamma function is defined as:
///
/// \\(Q(a, x) = Gamma(a, x) / Gamma(a) = 1 - P(a, x)\\)
///
/// where
///
/// \\(Gamma(a, x) = int_{x}^{\infty} t^{a-1} exp(-t) dt\\)
///
/// is the upper incomplete Gama function.
///
/// Note, above `P(a, x)` (`Igamma`) is the lower regularized complete
/// Gamma function.
@inlinable @inline(__always)
public static func igammac<T: FloatingPoint & TensorFlowScalar>(
  _ a: Tensor<T>,
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Igammac",
    a,
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Returns the imaginary part of a complex number.
///
/// Given a tensor `input` of complex numbers, this operation returns a tensor of
/// type `float` that is the imaginary part of each element in `input`. All
/// elements in `input` must be complex numbers of the form \\(a + bj\\), where *a*
/// is the real part and *b* is the imaginary part returned by this operation.
///
/// For example:
///
/// ```
/// # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
/// tf.imag(input) ==> [4.75, 5.75]
/// ```
@inlinable @inline(__always)
public static func imag<T: TensorFlowScalar, Tout: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>
) -> Tensor<Tout> {
  let ret: TensorHandle<Tout> = #tfop("Imag",
    input,
    T$dtype: T.tensorFlowDataType,
    Tout$dtype: Tout.tensorFlowDataType)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func inPolymorphicTwice<T: TensorFlowScalar>(
  _ a: [Tensor<T>],
  _ b: [Tensor<T>]
) {
  return #tfop("InPolymorphicTwice",
    a,
    b,
    T$dtype: T.tensorFlowDataType)
}

/// Says whether the targets are in the top `K` predictions.
///
/// This outputs a `batch_size` bool array, an entry `out[i]` is `true` if the
/// prediction for the target class is among the top `k` predictions among
/// all predictions for example `i`. Note that the behavior of `InTopK` differs
/// from the `TopK` op in its handling of ties; if multiple classes have the
/// same prediction value and straddle the top-`k` boundary, all of those
/// classes are considered to be in the top `k`.
///
/// More formally, let
///
///   \\(predictions_i\\) be the predictions for all classes for example `i`,
///   \\(targets_i\\) be the target class for example `i`,
///   \\(out_i\\) be the output for example `i`,
///
/// $$out_i = predictions_{i, targets_i} \in TopKIncludingTies(predictions_i)$$
///
/// - Parameters:
///   - predictions: A `batch_size` x `classes` tensor.
///   - targets: A `batch_size` vector of class ids.
///
/// - Attr k: Number of top elements to look at for computing precision.
///
/// - Output precision: Computed Precision at `k` as a `bool Tensor`.
@inlinable @inline(__always)
public static func inTopK<T: BinaryInteger & TensorFlowScalar>(
  predictions: Tensor<Float>,
  targets: Tensor<T>,
  k: Int64
) -> Tensor<Bool> {
  let ret: TensorHandle<Bool> = #tfop("InTopK",
    predictions,
    targets,
    T$dtype: T.tensorFlowDataType,
    k: k)
  return Tensor(handle: ret)
}

/// Says whether the targets are in the top `K` predictions.
///
/// This outputs a `batch_size` bool array, an entry `out[i]` is `true` if the
/// prediction for the target class is among the top `k` predictions among
/// all predictions for example `i`. Note that the behavior of `InTopK` differs
/// from the `TopK` op in its handling of ties; if multiple classes have the
/// same prediction value and straddle the top-`k` boundary, all of those
/// classes are considered to be in the top `k`.
///
/// More formally, let
///
///   \\(predictions_i\\) be the predictions for all classes for example `i`,
///   \\(targets_i\\) be the target class for example `i`,
///   \\(out_i\\) be the output for example `i`,
///
/// $$out_i = predictions_{i, targets_i} \in TopKIncludingTies(predictions_i)$$
///
/// - Parameters:
///   - predictions: A `batch_size` x `classes` tensor.
///   - targets: A `batch_size` vector of class ids.
///   - k: Number of top elements to look at for computing precision.
///
/// - Output precision: Computed precision at `k` as a `bool Tensor`.
@inlinable @inline(__always)
public static func inTopKV2<T: BinaryInteger & TensorFlowScalar>(
  predictions: Tensor<Float>,
  targets: Tensor<T>,
  k: Tensor<T>
) -> Tensor<Bool> {
  let ret: TensorHandle<Bool> = #tfop("InTopKV2",
    predictions,
    targets,
    k,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

///     Adds v into specified rows of x.
///
///     Computes y = x; y[i, :] += v; return y.
///
/// - Parameters:
///   - x: A `Tensor` of type T.
///   - i: A vector. Indices into the left-most dimension of `x`.
///   - v: A `Tensor` of type T. Same dimension sizes as x except the first dimension, which must be the same as i's size.
///
/// - Output y: A `Tensor` of type T. An alias of `x`. The content of `y` is undefined if there are duplicates in `i`.
@inlinable @inline(__always)
public static func inplaceAdd<T: TensorFlowScalar>(
  _ x: Tensor<T>,
  i: Tensor<Int32>,
  v: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("InplaceAdd",
    x,
    i,
    v,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

///     Subtracts `v` into specified rows of `x`.
///
///     Computes y = x; y[i, :] -= v; return y.
///
/// - Parameters:
///   - x: A `Tensor` of type T.
///   - i: A vector. Indices into the left-most dimension of `x`.
///   - v: A `Tensor` of type T. Same dimension sizes as x except the first dimension, which must be the same as i's size.
///
/// - Output y: A `Tensor` of type T. An alias of `x`. The content of `y` is undefined if there are duplicates in `i`.
@inlinable @inline(__always)
public static func inplaceSub<T: TensorFlowScalar>(
  _ x: Tensor<T>,
  i: Tensor<Int32>,
  v: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("InplaceSub",
    x,
    i,
    v,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

///     Updates specified rows with values in `v`.
///
///     Computes `x[i, :] = v; return x`.
///
/// - Parameters:
///   - x: A tensor of type `T`.
///   - i: A vector. Indices into the left-most dimension of `x`.
///   - v: A `Tensor` of type T. Same dimension sizes as x except the first dimension, which must be the same as i's size.
///
/// - Output y: A `Tensor` of type T. An alias of `x`. The content of `y` is undefined if there are duplicates in `i`.
@inlinable @inline(__always)
public static func inplaceUpdate<T: TensorFlowScalar>(
  _ x: Tensor<T>,
  i: Tensor<Int32>,
  v: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("InplaceUpdate",
    x,
    i,
    v,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func int64Output(
) -> Tensor<Int64> {
  let ret: TensorHandle<Int64> = #tfop("Int64Output")
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func intAttr(
  foo: Int64 = 1
) -> Tensor<Int64> {
  let ret: TensorHandle<Int64> = #tfop("IntAttr",
    foo: foo)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func intInput(
  _ a: Tensor<Int32>
) {
  return #tfop("IntInput",
    a)
}

@inlinable @inline(__always)
public static func intInputFloatInput(
  _ a: Tensor<Int32>,
  _ b: Tensor<Float>
) {
  return #tfop("IntInputFloatInput",
    a,
    b)
}

@inlinable @inline(__always)
public static func intInputIntOutput(
  _ a: Tensor<Int32>
) -> Tensor<Int32> {
  let ret: TensorHandle<Int32> = #tfop("IntInputIntOutput",
    a)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func intOutput(
) -> Tensor<Int32> {
  let ret: TensorHandle<Int32> = #tfop("IntOutput")
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func intOutputFloatOutput(
) -> (a: Tensor<Int32>, b: Tensor<Float>) {
  let ret: (TensorHandle<Int32>, TensorHandle<Float>) = #tfop("IntOutputFloatOutput")
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

/// Computes the reciprocal of x element-wise.
///
/// I.e., \\(y = 1 / x\\).
@inlinable @inline(__always)
public static func inv<T: Numeric & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Inv",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the gradient for the inverse of `x` wrt its input.
///
/// Specifically, `grad = -dy * y*y`, where `y = 1/x`, and `dy`
/// is the corresponding input gradient.
@inlinable @inline(__always)
public static func invGrad<T: FloatingPoint & TensorFlowScalar>(
  _ y: Tensor<T>,
  dy: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("InvGrad",
    y,
    dy,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Flips all bits elementwise.
///
/// The result will have exactly those bits set, that are not set in `x`. The
/// computation is performed on the underlying representation of x.
@inlinable @inline(__always)
public static func invert<T: BinaryInteger & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Invert",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the inverse permutation of a tensor.
///
/// This operation computes the inverse of an index permutation. It takes a 1-D
/// integer tensor `x`, which represents the indices of a zero-based array, and
/// swaps each value with its index position. In other words, for an output tensor
/// `y` and an input tensor `x`, this operation computes the following:
///
/// `y[x[i]] = i for i in [0, 1, ..., len(x) - 1]`
///
/// The values must include 0. There can be no duplicate values or negative values.
///
/// For example:
///
/// ```
/// # tensor `x` is [3, 4, 0, 2, 1]
/// invert_permutation(x) ==> [2, 4, 3, 0, 1]
/// ```
///
/// - Parameter x: 1-D.
///
/// - Output y: 1-D.
@inlinable @inline(__always)
public static func invertPermutation<T: BinaryInteger & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("InvertPermutation",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Returns which elements of x are finite.
///
/// @compatibility(numpy)
/// Equivalent to np.isfinite
/// @end_compatibility
@inlinable @inline(__always)
public static func isFinite<T: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<Bool> {
  let ret: TensorHandle<Bool> = #tfop("IsFinite",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Returns which elements of x are Inf.
///
/// @compatibility(numpy)
/// Equivalent to np.isinf
/// @end_compatibility
@inlinable @inline(__always)
public static func isInf<T: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<Bool> {
  let ret: TensorHandle<Bool> = #tfop("IsInf",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Returns which elements of x are NaN.
///
/// @compatibility(numpy)
/// Equivalent to np.isnan
/// @end_compatibility
@inlinable @inline(__always)
public static func isNan<T: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<Bool> {
  let ret: TensorHandle<Bool> = #tfop("IsNan",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Returns the index of a data point that should be added to the seed set.
///
/// Entries in distances are assumed to be squared distances of candidate points to
/// the already sampled centers in the seed set. The op constructs one Markov chain
/// of the k-MC^2 algorithm and returns the index of one candidate point to be added
/// as an additional cluster center.
///
/// - Parameters:
///   - distances: Vector with squared distances to the closest previously sampled cluster center
///     for each candidate point.
///   - seed: Scalar. Seed for initializing the random number generator.
///
/// - Output index: Scalar with the index of the sampled point.
@inlinable @inline(__always)
public static func kMC2ChainInitialization(
  distances: Tensor<Float>,
  seed: Tensor<Int64>
) -> Tensor<Int64> {
  let ret: TensorHandle<Int64> = #tfop("KMC2ChainInitialization",
    distances,
    seed)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func kernelLabel(
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("KernelLabel")
  return StringTensor(handle: ret)
}

@inlinable @inline(__always)
public static func kernelLabelRequired(
  _ input: Tensor<Int32>
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("KernelLabelRequired",
    input)
  return StringTensor(handle: ret)
}

/// Selects num_to_sample rows of input using the KMeans++ criterion.
///
/// Rows of points are assumed to be input points. One row is selected at random.
/// Subsequent rows are sampled with probability proportional to the squared L2
/// distance from the nearest row selected thus far till num_to_sample rows have
/// been sampled.
///
/// - Parameters:
///   - points: Matrix of shape (n, d). Rows are assumed to be input points.
///   - num_to_sample: Scalar. The number of rows to sample. This value must not be larger than n.
///   - seed: Scalar. Seed for initializing the random number generator.
///   - num_retries_per_sample: Scalar. For each row that is sampled, this parameter
///     specifies the number of additional points to draw from the current
///     distribution before selecting the best. If a negative value is specified, a
///     heuristic is used to sample O(log(num_to_sample)) additional points.
///
/// - Output samples: Matrix of shape (num_to_sample, d). The sampled rows.
@inlinable @inline(__always)
public static func kmeansPlusPlusInitialization(
  points: Tensor<Float>,
  numToSample: Tensor<Int64>,
  seed: Tensor<Int64>,
  numRetriesPerSample: Tensor<Int64>
) -> Tensor<Float> {
  let ret: TensorHandle<Float> = #tfop("KmeansPlusPlusInitialization",
    points,
    numToSample,
    seed,
    numRetriesPerSample)
  return Tensor(handle: ret)
}

/// L2 Loss.
///
/// Computes half the L2 norm of a tensor without the `sqrt`:
///
///     output = sum(t ** 2) / 2
///
/// - Parameter t: Typically 2-D, but may have any dimensions.
///
/// - Output output: 0-D.
@inlinable @inline(__always)
public static func l2Loss<T: FloatingPoint & TensorFlowScalar>(
  t: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("L2Loss",
    t,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Local Response Normalization.
///
/// The 4-D `input` tensor is treated as a 3-D array of 1-D vectors (along the last
/// dimension), and each vector is normalized independently.  Within a given vector,
/// each component is divided by the weighted, squared sum of inputs within
/// `depth_radius`.  In detail,
///
///     sqr_sum[a, b, c, d] =
///         sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
///     output = input / (bias + alpha * sqr_sum) ** beta
///
/// For details, see [Krizhevsky et al., ImageNet classification with deep
/// convolutional neural networks (NIPS 2012)](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks).
///
/// - Parameter input: 4-D.
///
/// - Attrs:
///   - depth_radius: 0-D.  Half-width of the 1-D normalization window.
///   - bias: An offset (usually positive to avoid dividing by 0).
///   - alpha: A scale factor, usually positive.
///   - beta: An exponent.
@inlinable @inline(__always)
public static func lRN<T: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>,
  depthRadius: Int64 = 5,
  bias: Double = 1,
  alpha: Double = 1,
  beta: Double = 0.5
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("LRN",
    input,
    T$dtype: T.tensorFlowDataType,
    depth_radius: depthRadius,
    bias: bias,
    alpha: alpha,
    beta: beta)
  return Tensor(handle: ret)
}

/// Gradients for Local Response Normalization.
///
/// - Parameters:
///   - input_grads: 4-D with shape `[batch, height, width, channels]`.
///   - input_image: 4-D with shape `[batch, height, width, channels]`.
///   - output_image: 4-D with shape `[batch, height, width, channels]`.
///
/// - Attrs:
///   - depth_radius: A depth radius.
///   - bias: An offset (usually > 0 to avoid dividing by 0).
///   - alpha: A scale factor, usually positive.
///   - beta: An exponent.
///
/// - Output output: The gradients for LRN.
@inlinable @inline(__always)
public static func lRNGrad<T: FloatingPoint & TensorFlowScalar>(
  inputGrads: Tensor<T>,
  inputImage: Tensor<T>,
  outputImage: Tensor<T>,
  depthRadius: Int64 = 5,
  bias: Double = 1,
  alpha: Double = 1,
  beta: Double = 0.5
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("LRNGrad",
    inputGrads,
    inputImage,
    outputImage,
    T$dtype: T.tensorFlowDataType,
    depth_radius: depthRadius,
    bias: bias,
    alpha: alpha,
    beta: beta)
  return Tensor(handle: ret)
}

/// Computes the LSTM cell forward propagation for 1 time step.
///
/// This implementation uses 1 weight matrix and 1 bias vector, and there's an
/// optional peephole connection.
///
/// This kernel op implements the following mathematical equations:
///
/// ```python
/// xh = [x, h_prev]
/// [i, f, ci, o] = xh * w + b
/// f = f + forget_bias
///
/// if not use_peephole:
///   wci = wcf = wco = 0
///
/// i = sigmoid(cs_prev * wci + i)
/// f = sigmoid(cs_prev * wcf + f)
/// ci = tanh(ci)
///
/// cs = ci .* i + cs_prev .* f
/// cs = clip(cs, cell_clip)
///
/// o = sigmoid(cs * wco + o)
/// co = tanh(cs)
/// h = co .* o
/// ```
///
/// - Parameters:
///   - x: The input to the LSTM cell, shape (batch_size, num_inputs).
///   - cs_prev: Value of the cell state at previous time step.
///   - h_prev: Output of the previous cell at previous time step.
///   - w: The weight matrix.
///   - wci: The weight matrix for input gate peephole connection.
///   - wcf: The weight matrix for forget gate peephole connection.
///   - wco: The weight matrix for output gate peephole connection.
///   - b: The bias vector.
///
/// - Attrs:
///   - forget_bias: The forget gate bias.
///   - cell_clip: Value to clip the 'cs' value to.
///   - use_peephole: Whether to use peephole weights.
///
/// - Outputs:
///   - i: The input gate.
///   - cs: The cell state before the tanh.
///   - f: The forget gate.
///   - o: The output gate.
///   - ci: The cell input.
///   - co: The cell after the tanh.
///   - h: The output h vector.
@inlinable @inline(__always)
public static func lSTMBlockCell<T: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>,
  csPrev: Tensor<T>,
  hPrev: Tensor<T>,
  w: Tensor<T>,
  wci: Tensor<T>,
  wcf: Tensor<T>,
  wco: Tensor<T>,
  _ b: Tensor<T>,
  forgetBias: Double = 1,
  cellClip: Double = 3,
  usePeephole: Bool = false
) -> (i: Tensor<T>, cs: Tensor<T>, f: Tensor<T>, o: Tensor<T>, ci: Tensor<T>, co: Tensor<T>, h: Tensor<T>) {
  let ret: (TensorHandle<T>, TensorHandle<T>, TensorHandle<T>, TensorHandle<T>, TensorHandle<T>, TensorHandle<T>, TensorHandle<T>) = #tfop("LSTMBlockCell",
    x,
    csPrev,
    hPrev,
    w,
    wci,
    wcf,
    wco,
    b,
    T$dtype: T.tensorFlowDataType,
    forget_bias: forgetBias,
    cell_clip: cellClip,
    use_peephole: usePeephole)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2), Tensor(handle: ret.3), Tensor(handle: ret.4), Tensor(handle: ret.5), Tensor(handle: ret.6))
}

/// Computes the LSTM cell backward propagation for 1 timestep.
///
/// This implementation is to be used in conjunction of LSTMBlockCell.
///
/// - Parameters:
///   - x: The input to the LSTM cell, shape (batch_size, num_inputs).
///   - cs_prev: The previous cell state.
///   - h_prev: The previous h state.
///   - w: The weight matrix.
///   - wci: The weight matrix for input gate peephole connection.
///   - wcf: The weight matrix for forget gate peephole connection.
///   - wco: The weight matrix for output gate peephole connection.
///   - b: The bias vector.
///   - i: The input gate.
///   - cs: The cell state before the tanh.
///   - f: The forget gate.
///   - o: The output gate.
///   - ci: The cell input.
///   - co: The cell after the tanh.
///   - cs_grad: The current gradient of cs.
///   - h_grad: The gradient of h vector.
///
/// - Attr use_peephole: Whether the cell uses peephole connections.
///
/// - Outputs:
///   - cs_prev_grad: The gradient of cs to be back-propped.
///   - dicfo: The derivative wrt to [i, cs, f, o].
///   - wci_grad: The gradient for wci to be back-propped.
///   - wcf_grad: The gradient for wcf to be back-propped.
///   - wco_grad: The gradient for wco to be back-propped.
@inlinable @inline(__always)
public static func lSTMBlockCellGrad<T: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>,
  csPrev: Tensor<T>,
  hPrev: Tensor<T>,
  w: Tensor<T>,
  wci: Tensor<T>,
  wcf: Tensor<T>,
  wco: Tensor<T>,
  _ b: Tensor<T>,
  i: Tensor<T>,
  cs: Tensor<T>,
  f: Tensor<T>,
  o: Tensor<T>,
  ci: Tensor<T>,
  co: Tensor<T>,
  csGrad: Tensor<T>,
  hGrad: Tensor<T>,
  usePeephole: Bool
) -> (csPrevGrad: Tensor<T>, dicfo: Tensor<T>, wciGrad: Tensor<T>, wcfGrad: Tensor<T>, wcoGrad: Tensor<T>) {
  let ret: (TensorHandle<T>, TensorHandle<T>, TensorHandle<T>, TensorHandle<T>, TensorHandle<T>) = #tfop("LSTMBlockCellGrad",
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
    T$dtype: T.tensorFlowDataType,
    use_peephole: usePeephole)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2), Tensor(handle: ret.3), Tensor(handle: ret.4))
}

/// Computes rectified linear: `max(features, features * alpha)`.
@inlinable @inline(__always)
public static func leakyRelu<T: FloatingPoint & TensorFlowScalar>(
  features: Tensor<T>,
  alpha: Double = 0.2
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("LeakyRelu",
    features,
    T$dtype: T.tensorFlowDataType,
    alpha: alpha)
  return Tensor(handle: ret)
}

/// Computes rectified linear gradients for a LeakyRelu operation.
///
/// - Parameters:
///   - gradients: The backpropagated gradients to the corresponding LeakyRelu operation.
///   - features: The features passed as input to the corresponding LeakyRelu operation,
///     OR the outputs of that operation (both work equivalently).
///
/// - Output backprops: `gradients * (features > 0) + alpha * gradients * (featurs <= 0)`.
@inlinable @inline(__always)
public static func leakyReluGrad<T: FloatingPoint & TensorFlowScalar>(
  gradients: Tensor<T>,
  features: Tensor<T>,
  alpha: Double = 0.2
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("LeakyReluGrad",
    gradients,
    features,
    T$dtype: T.tensorFlowDataType,
    alpha: alpha)
  return Tensor(handle: ret)
}

/// Generates labels for candidate sampling with a learned unigram distribution.
///
/// See explanations of candidate sampling and the data formats at
/// go/candidate-sampling.
///
/// For each batch, this op picks a single set of sampled candidate labels.
///
/// The advantages of sampling candidates per-batch are simplicity and the
/// possibility of efficient dense matrix multiplication. The disadvantage is that
/// the sampled candidates must be chosen independently of the context and of the
/// true labels.
///
/// - Parameter true_classes: A batch_size * num_true matrix, in which each row contains the
///   IDs of the num_true target_classes in the corresponding original label.
///
/// - Attrs:
///   - num_true: Number of true labels per context.
///   - num_sampled: Number of candidates to randomly sample.
///   - unique: If unique is true, we sample with rejection, so that all sampled
///     candidates in a batch are unique. This requires some approximation to
///     estimate the post-rejection sampling probabilities.
///   - range_max: The sampler will sample integers from the interval [0, range_max).
///   - seed: If either seed or seed2 are set to be non-zero, the random number
///     generator is seeded by the given seed.  Otherwise, it is seeded by a
///     random seed.
///   - seed2: An second seed to avoid seed collision.
///
/// - Outputs:
///   - sampled_candidates: A vector of length num_sampled, in which each element is
///     the ID of a sampled candidate.
///   - true_expected_count: A batch_size * num_true matrix, representing
///     the number of times each candidate is expected to occur in a batch
///     of sampled candidates. If unique=true, then this is a probability.
///   - sampled_expected_count: A vector of length num_sampled, for each sampled
///     candidate representing the number of times the candidate is expected
///     to occur in a batch of sampled candidates.  If unique=true, then this is a
///     probability.
@inlinable @inline(__always)
public static func learnedUnigramCandidateSampler(
  trueClasses: Tensor<Int64>,
  numTrue: Int64,
  numSampled: Int64,
  unique: Bool,
  rangeMax: Int64,
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> (sampledCandidates: Tensor<Int64>, trueExpectedCount: Tensor<Float>, sampledExpectedCount: Tensor<Float>) {
  let ret: (TensorHandle<Int64>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("LearnedUnigramCandidateSampler",
    trueClasses,
    num_true: numTrue,
    num_sampled: numSampled,
    unique: unique,
    range_max: rangeMax,
    seed: seed,
    seed2: seed2)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Elementwise computes the bitwise left-shift of `x` and `y`.
///
/// If `y` is negative, or greater than or equal to the width of `x` in bits the
/// result is implementation defined.
@inlinable @inline(__always)
public static func leftShift<T: BinaryInteger & TensorFlowScalar>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("LeftShift",
    x,
    y,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Returns the truth value of (x < y) element-wise.
///
/// *NOTE*: `Less` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@inlinable @inline(__always)
public static func less<T: Numeric & TensorFlowScalar>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<Bool> {
  let ret: TensorHandle<Bool> = #tfop("Less",
    x,
    y,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Returns the truth value of (x <= y) element-wise.
///
/// *NOTE*: `LessEqual` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@inlinable @inline(__always)
public static func lessEqual<T: Numeric & TensorFlowScalar>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<Bool> {
  let ret: TensorHandle<Bool> = #tfop("LessEqual",
    x,
    y,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the log of the absolute value of `Gamma(x)` element-wise.
@inlinable @inline(__always)
public static func lgamma<T: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Lgamma",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Generates values in an interval.
///
/// A sequence of `num` evenly-spaced values are generated beginning at `start`.
/// If `num > 1`, the values in the sequence increase by `stop - start / num - 1`,
/// so that the last one is exactly `stop`.
///
/// For example:
///
/// ```
/// tf.linspace(10.0, 12.0, 3, name="linspace") => [ 10.0  11.0  12.0]
/// ```
///
/// - Parameters:
///   - start: 0-D tensor. First entry in the range.
///   - stop: 0-D tensor. Last entry in the range.
///   - num: 0-D tensor. Number of values to generate.
///
/// - Output output: 1-D. The generated values.
@inlinable @inline(__always)
public static func linSpace<T: FloatingPoint & TensorFlowScalar, Tidx: BinaryInteger & TensorFlowScalar>(
  start: Tensor<T>,
  stop: Tensor<T>,
  num: Tensor<Tidx>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("LinSpace",
    start,
    stop,
    num,
    T$dtype: T.tensorFlowDataType,
    Tidx$dtype: Tidx.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the difference between two lists of numbers or strings.
///
/// Given a list `x` and a list `y`, this operation returns a list `out` that
/// represents all values that are in `x` but not in `y`. The returned list `out`
/// is sorted in the same order that the numbers appear in `x` (duplicates are
/// preserved). This operation also returns a list `idx` that represents the
/// position of each `out` element in `x`. In other words:
///
/// `out[i] = x[idx[i]] for i in [0, 1, ..., len(out) - 1]`
///
/// For example, given this input:
///
/// ```
/// x = [1, 2, 3, 4, 5, 6]
/// y = [1, 3, 5]
/// ```
///
/// This operation would return:
///
/// ```
/// out ==> [2, 4, 6]
/// idx ==> [1, 3, 5]
/// ```
///
/// - Parameters:
///   - x: 1-D. Values to keep.
///   - y: 1-D. Values to remove.
///
/// - Outputs:
///   - out: 1-D. Values present in `x` but not in `y`.
///   - idx: 1-D. Positions of `x` values preserved in `out`.
@inlinable @inline(__always)
public static func listDiff<T: TensorFlowScalar, OutIdx: BinaryInteger & TensorFlowScalar>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> (out: Tensor<T>, idx: Tensor<OutIdx>) {
  let ret: (TensorHandle<T>, TensorHandle<OutIdx>) = #tfop("ListDiff",
    x,
    y,
    T$dtype: T.tensorFlowDataType,
    out_idx$dtype: OutIdx.tensorFlowDataType)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

@inlinable @inline(__always)
public static func listInput<T: TensorFlowScalar>(
  _ a: [Tensor<T>]
) {
  return #tfop("ListInput",
    a,
    T$dtype: T.tensorFlowDataType)
}

/// Loads a 2-D (matrix) `Tensor` with name `old_tensor_name` from the checkpoint
///
/// at `ckpt_path` and potentially reorders its rows and columns using the
/// specified remappings.
///
/// Most users should use one of the wrapper initializers (such as
/// `tf.contrib.framework.load_and_remap_matrix_initializer`) instead of this
/// function directly.
///
/// The remappings are 1-D tensors with the following properties:
///
/// * `row_remapping` must have exactly `num_rows` entries. Row `i` of the output
///   matrix will be initialized from the row corresponding to index
///   `row_remapping[i]` in the old `Tensor` from the checkpoint.
/// * `col_remapping` must have either 0 entries (indicating that no column
///   reordering is needed) or `num_cols` entries. If specified, column `j` of the
///   output matrix will be initialized from the column corresponding to index
///   `col_remapping[j]` in the old `Tensor` from the checkpoint.
/// * A value of -1 in either of the remappings signifies a "missing" entry. In that
///   case, values from the `initializing_values` tensor will be used to fill that
///   missing row or column. If `row_remapping` has `r` missing entries and
///   `col_remapping` has `c` missing entries, then the following condition must be
///   true:
///
/// `(r * num_cols) + (c * num_rows) - (r * c) == len(initializing_values)`
///
/// The remapping tensors can be generated using the GenerateVocabRemapping op.
///
/// As an example, with row_remapping = [1, 0, -1], col_remapping = [0, 2, -1],
/// initializing_values = [0.5, -0.5, 0.25, -0.25, 42], and w(i, j) representing
/// the value from row i, column j of the old tensor in the checkpoint, the output
/// matrix will look like the following:
///
/// [[w(1, 0),  w(1, 2),  0.5],
///  [w(0, 0),  w(0, 2), -0.5],
///  [0.25,    -0.25,      42]]
///
/// - Parameters:
///   - ckpt_path: Path to the TensorFlow checkpoint (version 2, `TensorBundle`) from
///     which the old matrix `Tensor` will be loaded.
///   - old_tensor_name: Name of the 2-D `Tensor` to load from checkpoint.
///   - row_remapping: An int `Tensor` of row remappings (generally created by
///     `generate_vocab_remapping`).  Even if no row remapping is needed, this must
///     still be an index-valued Tensor (e.g. [0, 1, 2, ...]), or a shifted
///     index-valued `Tensor` (e.g. [8, 9, 10, ...], for partitioned `Variables`).
///   - col_remapping: An int `Tensor` of column remappings (generally created by
///     `generate_vocab_remapping`).  May be a size-0 `Tensor` if only row remapping
///     is to be done (e.g. column ordering is the same).
///   - initializing_values: A float `Tensor` containing  values to fill in for cells
///     in the output matrix that are not loaded from the checkpoint. Length must be
///     exactly the same as the number of missing / new cells.
///
/// - Attrs:
///   - num_rows: Number of rows (length of the 1st dimension) in the output matrix.
///   - num_cols: Number of columns (length of the 2nd dimension) in the output matrix.
///   - max_rows_in_memory: The maximum number of rows to load from the checkpoint at
///     once. If less than or equal to 0, the entire matrix will be loaded into
///     memory. Setting this arg trades increased disk reads for lower memory usage.
///
/// - Output output_matrix: Output matrix containing existing values loaded from the
///   checkpoint, and with any missing values filled in from initializing_values.
@inlinable @inline(__always)
public static func loadAndRemapMatrix(
  ckptPath: StringTensor,
  oldTensorName: StringTensor,
  rowRemapping: Tensor<Int64>,
  colRemapping: Tensor<Int64>,
  initializingValues: Tensor<Float>,
  numRows: Int64,
  numCols: Int64,
  maxRowsInMemory: Int64 = -1
) -> Tensor<Float> {
  let ret: TensorHandle<Float> = #tfop("LoadAndRemapMatrix",
    ckptPath,
    oldTensorName,
    rowRemapping,
    colRemapping,
    initializingValues,
    num_rows: numRows,
    num_cols: numCols,
    max_rows_in_memory: maxRowsInMemory)
  return Tensor(handle: ret)
}

/// Load ADAM embedding parameters.
///
/// An op that loads optimization parameters into HBM for embedding. Must be
/// preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
/// embedding table configuration. For example, this op is used to install
/// parameters that are loaded from a checkpoint before a training loop is
/// executed.
///
/// - Parameters:
///   - parameters: Value of parameters used in the ADAM optimization algorithm.
///   - momenta: Value of momenta used in the ADAM optimization algorithm.
///   - velocities: Value of velocities used in the ADAM optimization algorithm.
@inlinable @inline(__always)
public static func loadTPUEmbeddingADAMParameters(
  parameters: Tensor<Float>,
  momenta: Tensor<Float>,
  velocities: Tensor<Float>,
  tableId: Int64 = -1,
  tableName: String,
  numShards: Int64,
  shardId: Int64
) {
  return #tfop("LoadTPUEmbeddingADAMParameters",
    parameters,
    momenta,
    velocities,
    table_id: tableId,
    table_name: tableName,
    num_shards: numShards,
    shard_id: shardId)
}

/// Load ADAM embedding parameters with debug support.
///
/// An op that loads optimization parameters into HBM for embedding. Must be
/// preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
/// embedding table configuration. For example, this op is used to install
/// parameters that are loaded from a checkpoint before a training loop is
/// executed.
///
/// - Parameters:
///   - parameters: Value of parameters used in the ADAM optimization algorithm.
///   - momenta: Value of momenta used in the ADAM optimization algorithm.
///   - velocities: Value of velocities used in the ADAM optimization algorithm.
///   - gradient_accumulators: Value of gradient_accumulators used in the ADAM optimization algorithm.
@inlinable @inline(__always)
public static func loadTPUEmbeddingADAMParametersGradAccumDebug(
  parameters: Tensor<Float>,
  momenta: Tensor<Float>,
  velocities: Tensor<Float>,
  gradientAccumulators: Tensor<Float>,
  tableId: Int64 = -1,
  tableName: String,
  numShards: Int64,
  shardId: Int64
) {
  return #tfop("LoadTPUEmbeddingADAMParametersGradAccumDebug",
    parameters,
    momenta,
    velocities,
    gradientAccumulators,
    table_id: tableId,
    table_name: tableName,
    num_shards: numShards,
    shard_id: shardId)
}

/// Load Adadelta embedding parameters.
///
/// An op that loads optimization parameters into HBM for embedding. Must be
/// preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
/// embedding table configuration. For example, this op is used to install
/// parameters that are loaded from a checkpoint before a training loop is
/// executed.
///
/// - Parameters:
///   - parameters: Value of parameters used in the Adadelta optimization algorithm.
///   - accumulators: Value of accumulators used in the Adadelta optimization algorithm.
///   - updates: Value of updates used in the Adadelta optimization algorithm.
@inlinable @inline(__always)
public static func loadTPUEmbeddingAdadeltaParameters(
  parameters: Tensor<Float>,
  accumulators: Tensor<Float>,
  updates: Tensor<Float>,
  tableId: Int64 = -1,
  tableName: String,
  numShards: Int64,
  shardId: Int64
) {
  return #tfop("LoadTPUEmbeddingAdadeltaParameters",
    parameters,
    accumulators,
    updates,
    table_id: tableId,
    table_name: tableName,
    num_shards: numShards,
    shard_id: shardId)
}

/// Load Adadelta parameters with debug support.
///
/// An op that loads optimization parameters into HBM for embedding. Must be
/// preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
/// embedding table configuration. For example, this op is used to install
/// parameters that are loaded from a checkpoint before a training loop is
/// executed.
///
/// - Parameters:
///   - parameters: Value of parameters used in the Adadelta optimization algorithm.
///   - accumulators: Value of accumulators used in the Adadelta optimization algorithm.
///   - updates: Value of updates used in the Adadelta optimization algorithm.
///   - gradient_accumulators: Value of gradient_accumulators used in the Adadelta optimization algorithm.
@inlinable @inline(__always)
public static func loadTPUEmbeddingAdadeltaParametersGradAccumDebug(
  parameters: Tensor<Float>,
  accumulators: Tensor<Float>,
  updates: Tensor<Float>,
  gradientAccumulators: Tensor<Float>,
  tableId: Int64 = -1,
  tableName: String,
  numShards: Int64,
  shardId: Int64
) {
  return #tfop("LoadTPUEmbeddingAdadeltaParametersGradAccumDebug",
    parameters,
    accumulators,
    updates,
    gradientAccumulators,
    table_id: tableId,
    table_name: tableName,
    num_shards: numShards,
    shard_id: shardId)
}

/// Load Adagrad embedding parameters.
///
/// An op that loads optimization parameters into HBM for embedding. Must be
/// preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
/// embedding table configuration. For example, this op is used to install
/// parameters that are loaded from a checkpoint before a training loop is
/// executed.
///
/// - Parameters:
///   - parameters: Value of parameters used in the Adagrad optimization algorithm.
///   - accumulators: Value of accumulators used in the Adagrad optimization algorithm.
@inlinable @inline(__always)
public static func loadTPUEmbeddingAdagradParameters(
  parameters: Tensor<Float>,
  accumulators: Tensor<Float>,
  tableId: Int64 = -1,
  tableName: String,
  numShards: Int64,
  shardId: Int64
) {
  return #tfop("LoadTPUEmbeddingAdagradParameters",
    parameters,
    accumulators,
    table_id: tableId,
    table_name: tableName,
    num_shards: numShards,
    shard_id: shardId)
}

/// Load Adagrad embedding parameters with debug support.
///
/// An op that loads optimization parameters into HBM for embedding. Must be
/// preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
/// embedding table configuration. For example, this op is used to install
/// parameters that are loaded from a checkpoint before a training loop is
/// executed.
///
/// - Parameters:
///   - parameters: Value of parameters used in the Adagrad optimization algorithm.
///   - accumulators: Value of accumulators used in the Adagrad optimization algorithm.
///   - gradient_accumulators: Value of gradient_accumulators used in the Adagrad optimization algorithm.
@inlinable @inline(__always)
public static func loadTPUEmbeddingAdagradParametersGradAccumDebug(
  parameters: Tensor<Float>,
  accumulators: Tensor<Float>,
  gradientAccumulators: Tensor<Float>,
  tableId: Int64 = -1,
  tableName: String,
  numShards: Int64,
  shardId: Int64
) {
  return #tfop("LoadTPUEmbeddingAdagradParametersGradAccumDebug",
    parameters,
    accumulators,
    gradientAccumulators,
    table_id: tableId,
    table_name: tableName,
    num_shards: numShards,
    shard_id: shardId)
}

/// Load centered RMSProp embedding parameters.
///
/// An op that loads optimization parameters into HBM for embedding. Must be
/// preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
/// embedding table configuration. For example, this op is used to install
/// parameters that are loaded from a checkpoint before a training loop is
/// executed.
///
/// - Parameters:
///   - parameters: Value of parameters used in the centered RMSProp optimization algorithm.
///   - ms: Value of ms used in the centered RMSProp optimization algorithm.
///   - mom: Value of mom used in the centered RMSProp optimization algorithm.
///   - mg: Value of mg used in the centered RMSProp optimization algorithm.
@inlinable @inline(__always)
public static func loadTPUEmbeddingCenteredRMSPropParameters(
  parameters: Tensor<Float>,
  ms: Tensor<Float>,
  mom: Tensor<Float>,
  mg: Tensor<Float>,
  tableId: Int64 = -1,
  tableName: String,
  numShards: Int64,
  shardId: Int64
) {
  return #tfop("LoadTPUEmbeddingCenteredRMSPropParameters",
    parameters,
    ms,
    mom,
    mg,
    table_id: tableId,
    table_name: tableName,
    num_shards: numShards,
    shard_id: shardId)
}

/// Load FTRL embedding parameters.
///
/// An op that loads optimization parameters into HBM for embedding. Must be
/// preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
/// embedding table configuration. For example, this op is used to install
/// parameters that are loaded from a checkpoint before a training loop is
/// executed.
///
/// - Parameters:
///   - parameters: Value of parameters used in the FTRL optimization algorithm.
///   - accumulators: Value of accumulators used in the FTRL optimization algorithm.
///   - linears: Value of linears used in the FTRL optimization algorithm.
@inlinable @inline(__always)
public static func loadTPUEmbeddingFTRLParameters(
  parameters: Tensor<Float>,
  accumulators: Tensor<Float>,
  linears: Tensor<Float>,
  tableId: Int64 = -1,
  tableName: String,
  numShards: Int64,
  shardId: Int64
) {
  return #tfop("LoadTPUEmbeddingFTRLParameters",
    parameters,
    accumulators,
    linears,
    table_id: tableId,
    table_name: tableName,
    num_shards: numShards,
    shard_id: shardId)
}

/// Load FTRL embedding parameters with debug support.
///
/// An op that loads optimization parameters into HBM for embedding. Must be
/// preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
/// embedding table configuration. For example, this op is used to install
/// parameters that are loaded from a checkpoint before a training loop is
/// executed.
///
/// - Parameters:
///   - parameters: Value of parameters used in the FTRL optimization algorithm.
///   - accumulators: Value of accumulators used in the FTRL optimization algorithm.
///   - linears: Value of linears used in the FTRL optimization algorithm.
///   - gradient_accumulators: Value of gradient_accumulators used in the FTRL optimization algorithm.
@inlinable @inline(__always)
public static func loadTPUEmbeddingFTRLParametersGradAccumDebug(
  parameters: Tensor<Float>,
  accumulators: Tensor<Float>,
  linears: Tensor<Float>,
  gradientAccumulators: Tensor<Float>,
  tableId: Int64 = -1,
  tableName: String,
  numShards: Int64,
  shardId: Int64
) {
  return #tfop("LoadTPUEmbeddingFTRLParametersGradAccumDebug",
    parameters,
    accumulators,
    linears,
    gradientAccumulators,
    table_id: tableId,
    table_name: tableName,
    num_shards: numShards,
    shard_id: shardId)
}

/// Load MDL Adagrad Light embedding parameters.
///
/// An op that loads optimization parameters into HBM for embedding. Must be
/// preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
/// embedding table configuration. For example, this op is used to install
/// parameters that are loaded from a checkpoint before a training loop is
/// executed.
///
/// - Parameters:
///   - parameters: Value of parameters used in the MDL Adagrad Light optimization algorithm.
///   - accumulators: Value of accumulators used in the MDL Adagrad Light optimization algorithm.
///   - weights: Value of weights used in the MDL Adagrad Light optimization algorithm.
///   - benefits: Value of benefits used in the MDL Adagrad Light optimization algorithm.
@inlinable @inline(__always)
public static func loadTPUEmbeddingMDLAdagradLightParameters(
  parameters: Tensor<Float>,
  accumulators: Tensor<Float>,
  weights: Tensor<Float>,
  benefits: Tensor<Float>,
  tableId: Int64 = -1,
  tableName: String,
  numShards: Int64,
  shardId: Int64
) {
  return #tfop("LoadTPUEmbeddingMDLAdagradLightParameters",
    parameters,
    accumulators,
    weights,
    benefits,
    table_id: tableId,
    table_name: tableName,
    num_shards: numShards,
    shard_id: shardId)
}

/// Load Momentum embedding parameters.
///
/// An op that loads optimization parameters into HBM for embedding. Must be
/// preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
/// embedding table configuration. For example, this op is used to install
/// parameters that are loaded from a checkpoint before a training loop is
/// executed.
///
/// - Parameters:
///   - parameters: Value of parameters used in the Momentum optimization algorithm.
///   - momenta: Value of momenta used in the Momentum optimization algorithm.
@inlinable @inline(__always)
public static func loadTPUEmbeddingMomentumParameters(
  parameters: Tensor<Float>,
  momenta: Tensor<Float>,
  tableId: Int64 = -1,
  tableName: String,
  numShards: Int64,
  shardId: Int64
) {
  return #tfop("LoadTPUEmbeddingMomentumParameters",
    parameters,
    momenta,
    table_id: tableId,
    table_name: tableName,
    num_shards: numShards,
    shard_id: shardId)
}

/// Load Momentum embedding parameters with debug support.
///
/// An op that loads optimization parameters into HBM for embedding. Must be
/// preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
/// embedding table configuration. For example, this op is used to install
/// parameters that are loaded from a checkpoint before a training loop is
/// executed.
///
/// - Parameters:
///   - parameters: Value of parameters used in the Momentum optimization algorithm.
///   - momenta: Value of momenta used in the Momentum optimization algorithm.
///   - gradient_accumulators: Value of gradient_accumulators used in the Momentum optimization algorithm.
@inlinable @inline(__always)
public static func loadTPUEmbeddingMomentumParametersGradAccumDebug(
  parameters: Tensor<Float>,
  momenta: Tensor<Float>,
  gradientAccumulators: Tensor<Float>,
  tableId: Int64 = -1,
  tableName: String,
  numShards: Int64,
  shardId: Int64
) {
  return #tfop("LoadTPUEmbeddingMomentumParametersGradAccumDebug",
    parameters,
    momenta,
    gradientAccumulators,
    table_id: tableId,
    table_name: tableName,
    num_shards: numShards,
    shard_id: shardId)
}

/// Load proximal Adagrad embedding parameters.
///
/// An op that loads optimization parameters into HBM for embedding. Must be
/// preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
/// embedding table configuration. For example, this op is used to install
/// parameters that are loaded from a checkpoint before a training loop is
/// executed.
///
/// - Parameters:
///   - parameters: Value of parameters used in the proximal Adagrad optimization algorithm.
///   - accumulators: Value of accumulators used in the proximal Adagrad optimization algorithm.
@inlinable @inline(__always)
public static func loadTPUEmbeddingProximalAdagradParameters(
  parameters: Tensor<Float>,
  accumulators: Tensor<Float>,
  tableId: Int64 = -1,
  tableName: String,
  numShards: Int64,
  shardId: Int64
) {
  return #tfop("LoadTPUEmbeddingProximalAdagradParameters",
    parameters,
    accumulators,
    table_id: tableId,
    table_name: tableName,
    num_shards: numShards,
    shard_id: shardId)
}

/// Load proximal Adagrad embedding parameters with debug support.
///
/// An op that loads optimization parameters into HBM for embedding. Must be
/// preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
/// embedding table configuration. For example, this op is used to install
/// parameters that are loaded from a checkpoint before a training loop is
/// executed.
///
/// - Parameters:
///   - parameters: Value of parameters used in the proximal Adagrad optimization algorithm.
///   - accumulators: Value of accumulators used in the proximal Adagrad optimization algorithm.
///   - gradient_accumulators: Value of gradient_accumulators used in the proximal Adagrad optimization algorithm.
@inlinable @inline(__always)
public static func loadTPUEmbeddingProximalAdagradParametersGradAccumDebug(
  parameters: Tensor<Float>,
  accumulators: Tensor<Float>,
  gradientAccumulators: Tensor<Float>,
  tableId: Int64 = -1,
  tableName: String,
  numShards: Int64,
  shardId: Int64
) {
  return #tfop("LoadTPUEmbeddingProximalAdagradParametersGradAccumDebug",
    parameters,
    accumulators,
    gradientAccumulators,
    table_id: tableId,
    table_name: tableName,
    num_shards: numShards,
    shard_id: shardId)
}

/// Load RMSProp embedding parameters.
///
/// An op that loads optimization parameters into HBM for embedding. Must be
/// preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
/// embedding table configuration. For example, this op is used to install
/// parameters that are loaded from a checkpoint before a training loop is
/// executed.
///
/// - Parameters:
///   - parameters: Value of parameters used in the RMSProp optimization algorithm.
///   - ms: Value of ms used in the RMSProp optimization algorithm.
///   - mom: Value of mom used in the RMSProp optimization algorithm.
@inlinable @inline(__always)
public static func loadTPUEmbeddingRMSPropParameters(
  parameters: Tensor<Float>,
  ms: Tensor<Float>,
  mom: Tensor<Float>,
  tableId: Int64 = -1,
  tableName: String,
  numShards: Int64,
  shardId: Int64
) {
  return #tfop("LoadTPUEmbeddingRMSPropParameters",
    parameters,
    ms,
    mom,
    table_id: tableId,
    table_name: tableName,
    num_shards: numShards,
    shard_id: shardId)
}

/// Load RMSProp embedding parameters with debug support.
///
/// An op that loads optimization parameters into HBM for embedding. Must be
/// preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
/// embedding table configuration. For example, this op is used to install
/// parameters that are loaded from a checkpoint before a training loop is
/// executed.
///
/// - Parameters:
///   - parameters: Value of parameters used in the RMSProp optimization algorithm.
///   - ms: Value of ms used in the RMSProp optimization algorithm.
///   - mom: Value of mom used in the RMSProp optimization algorithm.
///   - gradient_accumulators: Value of gradient_accumulators used in the RMSProp optimization algorithm.
@inlinable @inline(__always)
public static func loadTPUEmbeddingRMSPropParametersGradAccumDebug(
  parameters: Tensor<Float>,
  ms: Tensor<Float>,
  mom: Tensor<Float>,
  gradientAccumulators: Tensor<Float>,
  tableId: Int64 = -1,
  tableName: String,
  numShards: Int64,
  shardId: Int64
) {
  return #tfop("LoadTPUEmbeddingRMSPropParametersGradAccumDebug",
    parameters,
    ms,
    mom,
    gradientAccumulators,
    table_id: tableId,
    table_name: tableName,
    num_shards: numShards,
    shard_id: shardId)
}

/// Load SGD embedding parameters.
///
/// An op that loads optimization parameters into HBM for embedding. Must be
/// preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
/// embedding table configuration. For example, this op is used to install
/// parameters that are loaded from a checkpoint before a training loop is
/// executed.
///
/// - Parameter parameters: Value of parameters used in the stochastic gradient descent optimization algorithm.
@inlinable @inline(__always)
public static func loadTPUEmbeddingStochasticGradientDescentParameters(
  parameters: Tensor<Float>,
  tableId: Int64 = -1,
  tableName: String,
  numShards: Int64,
  shardId: Int64
) {
  return #tfop("LoadTPUEmbeddingStochasticGradientDescentParameters",
    parameters,
    table_id: tableId,
    table_name: tableName,
    num_shards: numShards,
    shard_id: shardId)
}

/// Computes natural logarithm of x element-wise.
///
/// I.e., \\(y = \log_e x\\).
@inlinable @inline(__always)
public static func log<T: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Log",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes natural logarithm of (1 + x) element-wise.
///
/// I.e., \\(y = \log_e (1 + x)\\).
@inlinable @inline(__always)
public static func log1p<T: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Log1p",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the sign and the log of the absolute value of the determinant of
///
/// one or more square matrices.
///
/// The input is a tensor of shape `[N, M, M]` whose inner-most 2 dimensions
/// form square matrices. The outputs are two tensors containing the signs and
/// absolute values of the log determinants for all N input submatrices
/// `[..., :, :]` such that the determinant = sign*exp(log_abs_determinant).
/// The log_abs_determinant is computed as det(P)*sum(log(diag(LU))) where LU
/// is the LU decomposition of the input and P is the corresponding
/// permutation matrix.
///
/// - Parameter input: Shape is `[N, M, M]`.
///
/// - Outputs:
///   - sign: The signs of the log determinants of the inputs. Shape is `[N]`.
///   - log_abs_determinant: The logs of the absolute values of the determinants
///     of the N input matrices.  Shape is `[N]`.
@inlinable @inline(__always)
public static func logMatrixDeterminant<T: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>
) -> (sign: Tensor<T>, logAbsDeterminant: Tensor<T>) {
  let ret: (TensorHandle<T>, TensorHandle<T>) = #tfop("LogMatrixDeterminant",
    input,
    T$dtype: T.tensorFlowDataType)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

/// Computes log softmax activations.
///
/// For each batch `i` and class `j` we have
///
///     logsoftmax[i, j] = logits[i, j] - log(sum(exp(logits[i])))
///
/// - Parameter logits: 2-D with shape `[batch_size, num_classes]`.
///
/// - Output logsoftmax: Same shape as `logits`.
@inlinable @inline(__always)
public static func logSoftmax<T: FloatingPoint & TensorFlowScalar>(
  logits: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("LogSoftmax",
    logits,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Generates labels for candidate sampling with a log-uniform distribution.
///
/// See explanations of candidate sampling and the data formats at
/// go/candidate-sampling.
///
/// For each batch, this op picks a single set of sampled candidate labels.
///
/// The advantages of sampling candidates per-batch are simplicity and the
/// possibility of efficient dense matrix multiplication. The disadvantage is that
/// the sampled candidates must be chosen independently of the context and of the
/// true labels.
///
/// - Parameter true_classes: A batch_size * num_true matrix, in which each row contains the
///   IDs of the num_true target_classes in the corresponding original label.
///
/// - Attrs:
///   - num_true: Number of true labels per context.
///   - num_sampled: Number of candidates to randomly sample.
///   - unique: If unique is true, we sample with rejection, so that all sampled
///     candidates in a batch are unique. This requires some approximation to
///     estimate the post-rejection sampling probabilities.
///   - range_max: The sampler will sample integers from the interval [0, range_max).
///   - seed: If either seed or seed2 are set to be non-zero, the random number
///     generator is seeded by the given seed.  Otherwise, it is seeded by a
///     random seed.
///   - seed2: An second seed to avoid seed collision.
///
/// - Outputs:
///   - sampled_candidates: A vector of length num_sampled, in which each element is
///     the ID of a sampled candidate.
///   - true_expected_count: A batch_size * num_true matrix, representing
///     the number of times each candidate is expected to occur in a batch
///     of sampled candidates. If unique=true, then this is a probability.
///   - sampled_expected_count: A vector of length num_sampled, for each sampled
///     candidate representing the number of times the candidate is expected
///     to occur in a batch of sampled candidates.  If unique=true, then this is a
///     probability.
@inlinable @inline(__always)
public static func logUniformCandidateSampler(
  trueClasses: Tensor<Int64>,
  numTrue: Int64,
  numSampled: Int64,
  unique: Bool,
  rangeMax: Int64,
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> (sampledCandidates: Tensor<Int64>, trueExpectedCount: Tensor<Float>, sampledExpectedCount: Tensor<Float>) {
  let ret: (TensorHandle<Int64>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("LogUniformCandidateSampler",
    trueClasses,
    num_true: numTrue,
    num_sampled: numSampled,
    unique: unique,
    range_max: rangeMax,
    seed: seed,
    seed2: seed2)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Returns the truth value of x AND y element-wise.
///
/// *NOTE*: `LogicalAnd` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@inlinable @inline(__always)
public static func logicalAnd(
  _ x: Tensor<Bool>,
  _ y: Tensor<Bool>
) -> Tensor<Bool> {
  let ret: TensorHandle<Bool> = #tfop("LogicalAnd",
    x,
    y)
  return Tensor(handle: ret)
}

/// Returns the truth value of NOT x element-wise.
@inlinable @inline(__always)
public static func logicalNot(
  _ x: Tensor<Bool>
) -> Tensor<Bool> {
  let ret: TensorHandle<Bool> = #tfop("LogicalNot",
    x)
  return Tensor(handle: ret)
}

/// Returns the truth value of x OR y element-wise.
///
/// *NOTE*: `LogicalOr` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@inlinable @inline(__always)
public static func logicalOr(
  _ x: Tensor<Bool>,
  _ y: Tensor<Bool>
) -> Tensor<Bool> {
  let ret: TensorHandle<Bool> = #tfop("LogicalOr",
    x,
    y)
  return Tensor(handle: ret)
}

/// Forwards the input to the output.
///
/// This operator represents the loop termination condition used by the
/// "pivot" switches of a loop.
///
/// - Parameter input: A boolean scalar, representing the branch predicate of the Switch op.
///
/// - Output output: The same tensor as `input`.
@inlinable @inline(__always)
public static func loopCond(
  _ input: Tensor<Bool>
) -> Tensor<Bool> {
  let ret: TensorHandle<Bool> = #tfop("LoopCond",
    input)
  return Tensor(handle: ret)
}

/// Applies lower_bound(sorted_search_values, values) along each row.
///
/// Each set of rows with the same index in (sorted_inputs, values) is treated
/// independently.  The resulting row is the equivalent of calling
/// `np.searchsorted(sorted_inputs, values, side='left')`.
///
/// The result is not a global index to the entire 
/// `Tensor`, but rather just the index in the last dimension.
///
/// A 2-D example:
///   sorted_sequence = [[0, 3, 9, 9, 10],
///                      [1, 2, 3, 4, 5]]
///   values = [[2, 4, 9],
///             [0, 2, 6]]
///
///   result = LowerBound(sorted_sequence, values)
///
///   result == [[1, 2, 2],
///              [0, 1, 5]]
///
/// - Parameters:
///   - sorted_inputs: 2-D Tensor where each row is ordered.
///   - values: 2-D Tensor with the same numbers of rows as `sorted_search_values`. Contains
///     the values that will be searched for in `sorted_search_values`.
///
/// - Output output: A `Tensor` with the same shape as `values`.  It contains the first scalar index
///   into the last dimension where values can be inserted without changing the
///   ordered property.
@inlinable @inline(__always)
public static func lowerBound<T: TensorFlowScalar, OutType: BinaryInteger & TensorFlowScalar>(
  sortedInputs: Tensor<T>,
  _ values: Tensor<T>
) -> Tensor<OutType> {
  let ret: TensorHandle<OutType> = #tfop("LowerBound",
    sortedInputs,
    values,
    T$dtype: T.tensorFlowDataType,
    out_type$dtype: OutType.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the LU decomposition of one or more square matrices.
///
/// The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
/// form square matrices.
///
/// The input has to be invertible.
///
/// The output consists of two tensors LU and P containing the LU decomposition
/// of all input submatrices `[..., :, :]`. LU encodes the lower triangular and
/// upper triangular factors.
///
/// For each input submatrix of shape `[M, M]`, L is a lower triangular matrix of
/// shape `[M, M]` with unit diagonal whose entries correspond to the strictly lower
/// triangular part of LU. U is a upper triangular matrix of shape `[M, M]` whose
/// entries correspond to the upper triangular part, including the diagonal, of LU.
///
/// P represents a permutation matrix encoded as a list of indices each between `0`
/// and `M-1`, inclusive. If P_mat denotes the permutation matrix corresponding to
/// P, then the L, U and P satisfies P_mat * input = L * U.
///
/// - Parameter input: A tensor of shape `[..., M, M]` whose inner-most 2 dimensions form matrices of
///   size `[M, M]`.
///
/// - Outputs:
///   - lu: A tensor of shape `[..., M, M]` whose strictly lower triangular part denotes the
///     lower triangular factor `L` with unit diagonal, and whose upper triangular part
///     denotes the upper triangular factor `U`.
///   - p: Permutation of the rows encoded as a list of indices in `0..M-1`. Shape is
///     `[..., M]`.
///     @compatibility(scipy)
///     Similar to `scipy.linalg.lu`, except the triangular factors `L` and `U` are
///     packed into a single tensor, the permutation is applied to `input` instead of
///     the right hand side and the permutation `P` is returned as a list of indices
///     instead of a permutation matrix.
///     @end_compatibility
@inlinable @inline(__always)
public static func lu<T: FloatingPoint & TensorFlowScalar, OutputIdxType: BinaryInteger & TensorFlowScalar>(
  _ input: Tensor<T>
) -> (lu: Tensor<T>, p: Tensor<OutputIdxType>) {
  let ret: (TensorHandle<T>, TensorHandle<OutputIdxType>) = #tfop("Lu",
    input,
    T$dtype: T.tensorFlowDataType,
    output_idx_type$dtype: OutputIdxType.tensorFlowDataType)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

/// Op removes all elements in the underlying container.
@inlinable @inline(__always)
public static func mapClear<Dtypes: TensorFlowScalar>(
  capacity: Int64 = 0,
  memoryLimit: Int64 = 0,
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

/// Op returns the number of incomplete elements in the underlying container.
@inlinable @inline(__always)
public static func mapIncompleteSize<Dtypes: TensorFlowScalar>(
  capacity: Int64 = 0,
  memoryLimit: Int64 = 0,
  container: String,
  sharedName: String,
  typeDtypes: Dtypes.Type
) -> Tensor<Int32> {
  let ret: TensorHandle<Int32> = #tfop("MapIncompleteSize",
    capacity: capacity,
    memory_limit: memoryLimit,
    container: container,
    shared_name: sharedName)
  return Tensor(handle: ret)
}

/// Op returns the number of elements in the underlying container.
@inlinable @inline(__always)
public static func mapSize<Dtypes: TensorFlowScalar>(
  capacity: Int64 = 0,
  memoryLimit: Int64 = 0,
  container: String,
  sharedName: String,
  typeDtypes: Dtypes.Type
) -> Tensor<Int32> {
  let ret: TensorHandle<Int32> = #tfop("MapSize",
    capacity: capacity,
    memory_limit: memoryLimit,
    container: container,
    shared_name: sharedName)
  return Tensor(handle: ret)
}

/// Stage (key, values) in the underlying container which behaves like a hashtable.
///
/// - Parameters:
///   - key: int64
///   - values: a list of tensors
///     dtypes A list of data types that inserted values should adhere to.
///
/// - Attrs:
///   - capacity: Maximum number of elements in the Staging Area. If > 0, inserts
///     on the container will block when the capacity is reached.
///   - container: If non-empty, this queue is placed in the given container. Otherwise,
///     a default container is used.
///   - shared_name: It is necessary to match this name to the matching Unstage Op.
@inlinable @inline(__always)
public static func mapStage<Dtypes: TensorFlowScalar, FakeDtypes: TensorFlowScalar>(
  key: Tensor<Int64>,
  indices: Tensor<Int32>,
  _ values: [Tensor<FakeDtypes>],
  capacity: Int64 = 0,
  memoryLimit: Int64 = 0,
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

/// Multiply the matrix "a" by the matrix "b".
///
/// The inputs must be two-dimensional matrices and the inner dimension of
/// "a" (after being transposed if transpose_a is true) must match the
/// outer dimension of "b" (after being transposed if transposed_b is
/// true).
///
/// *Note*: The default kernel implementation for MatMul on GPUs uses
/// cublas.
///
/// - Attrs:
///   - transpose_a: If true, "a" is transposed before multiplication.
///   - transpose_b: If true, "b" is transposed before multiplication.
@inlinable @inline(__always)
public static func matMul<T: Numeric & TensorFlowScalar>(
  _ a: Tensor<T>,
  _ b: Tensor<T>,
  transposeA: Bool = false,
  transposeB: Bool = false
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("MatMul",
    a,
    b,
    T$dtype: T.tensorFlowDataType,
    transpose_a: transposeA,
    transpose_b: transposeB)
  return Tensor(handle: ret)
}

/// Returns the set of files matching one or more glob patterns.
///
/// Note that this routine only supports wildcard characters in the
/// basename portion of the pattern, not in the directory portion.
/// Note also that the order of filenames returned can be non-deterministic.
///
/// - Parameter pattern: Shell wildcard pattern(s). Scalar or vector of type string.
///
/// - Output filenames: A vector of matching filenames.
@inlinable @inline(__always)
public static func matchingFiles(
  pattern: StringTensor
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("MatchingFiles",
    pattern)
  return StringTensor(handle: ret)
}

/// Copy a tensor setting everything outside a central band in each innermost matrix
///
/// to zero.
///
/// The `band` part is computed as follows:
/// Assume `input` has `k` dimensions `[I, J, K, ..., M, N]`, then the output is a
/// tensor with the same shape where
///
/// `band[i, j, k, ..., m, n] = in_band(m, n) * input[i, j, k, ..., m, n]`.
///
/// The indicator function
///
/// `in_band(m, n) = (num_lower < 0 || (m-n) <= num_lower)) &&
///                  (num_upper < 0 || (n-m) <= num_upper)`.
///
/// For example:
///
/// ```
/// # if 'input' is [[ 0,  1,  2, 3]
///                  [-1,  0,  1, 2]
///                  [-2, -1,  0, 1]
///                  [-3, -2, -1, 0]],
///
/// tf.matrix_band_part(input, 1, -1) ==> [[ 0,  1,  2, 3]
///                                        [-1,  0,  1, 2]
///                                        [ 0, -1,  0, 1]
///                                        [ 0,  0, -1, 0]],
///
/// tf.matrix_band_part(input, 2, 1) ==> [[ 0,  1,  0, 0]
///                                       [-1,  0,  1, 0]
///                                       [-2, -1,  0, 1]
///                                       [ 0, -2, -1, 0]]
/// ```
///
/// Useful special cases:
///
/// ```
///  tf.matrix_band_part(input, 0, -1) ==> Upper triangular part.
///  tf.matrix_band_part(input, -1, 0) ==> Lower triangular part.
///  tf.matrix_band_part(input, 0, 0) ==> Diagonal.
/// ```
///
/// - Parameters:
///   - input: Rank `k` tensor.
///   - num_lower: 0-D tensor. Number of subdiagonals to keep. If negative, keep entire
///     lower triangle.
///   - num_upper: 0-D tensor. Number of superdiagonals to keep. If negative, keep
///     entire upper triangle.
///
/// - Output band: Rank `k` tensor of the same shape as input. The extracted banded tensor.
@inlinable @inline(__always)
public static func matrixBandPart<T: TensorFlowScalar, Tindex: BinaryInteger & TensorFlowScalar>(
  _ input: Tensor<T>,
  numLower: Tensor<Tindex>,
  numUpper: Tensor<Tindex>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("MatrixBandPart",
    input,
    numLower,
    numUpper,
    T$dtype: T.tensorFlowDataType,
    Tindex$dtype: Tindex.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the determinant of one or more square matrices.
///
/// The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
/// form square matrices. The output is a tensor containing the determinants
/// for all input submatrices `[..., :, :]`.
///
/// - Parameter input: Shape is `[..., M, M]`.
///
/// - Output output: Shape is `[...]`.
@inlinable @inline(__always)
public static func matrixDeterminant<T: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("MatrixDeterminant",
    input,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Returns a batched diagonal tensor with a given batched diagonal values.
///
/// Given a `diagonal`, this operation returns a tensor with the `diagonal` and
/// everything else padded with zeros. The diagonal is computed as follows:
///
/// Assume `diagonal` has `k` dimensions `[I, J, K, ..., N]`, then the output is a
/// tensor of rank `k+1` with dimensions [I, J, K, ..., N, N]` where:
///
/// `output[i, j, k, ..., m, n] = 1{m=n} * diagonal[i, j, k, ..., n]`.
///
/// For example:
///
/// ```
/// # 'diagonal' is [[1, 2, 3, 4], [5, 6, 7, 8]]
///
/// and diagonal.shape = (2, 4)
///
/// tf.matrix_diag(diagonal) ==> [[[1, 0, 0, 0]
///                                      [0, 2, 0, 0]
///                                      [0, 0, 3, 0]
///                                      [0, 0, 0, 4]],
///                                     [[5, 0, 0, 0]
///                                      [0, 6, 0, 0]
///                                      [0, 0, 7, 0]
///                                      [0, 0, 0, 8]]]
///
/// which has shape (2, 4, 4)
/// ```
///
/// - Parameter diagonal: Rank `k`, where `k >= 1`.
///
/// - Output output: Rank `k+1`, with `output.shape = diagonal.shape + [diagonal.shape[-1]]`.
@inlinable @inline(__always)
public static func matrixDiag<T: TensorFlowScalar>(
  diagonal: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("MatrixDiag",
    diagonal,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Returns the batched diagonal part of a batched tensor.
///
/// This operation returns a tensor with the `diagonal` part
/// of the batched `input`. The `diagonal` part is computed as follows:
///
/// Assume `input` has `k` dimensions `[I, J, K, ..., M, N]`, then the output is a
/// tensor of rank `k - 1` with dimensions `[I, J, K, ..., min(M, N)]` where:
///
/// `diagonal[i, j, k, ..., n] = input[i, j, k, ..., n, n]`.
///
/// The input must be at least a matrix.
///
/// For example:
///
/// ```
/// # 'input' is [[[1, 0, 0, 0]
///                [0, 2, 0, 0]
///                [0, 0, 3, 0]
///                [0, 0, 0, 4]],
///               [[5, 0, 0, 0]
///                [0, 6, 0, 0]
///                [0, 0, 7, 0]
///                [0, 0, 0, 8]]]
///
/// and input.shape = (2, 4, 4)
///
/// tf.matrix_diag_part(input) ==> [[1, 2, 3, 4], [5, 6, 7, 8]]
///
/// which has shape (2, 4)
/// ```
///
/// - Parameter input: Rank `k` tensor where `k >= 2`.
///
/// - Output diagonal: The extracted diagonal(s) having shape
///   `diagonal.shape = input.shape[:-2] + [min(input.shape[-2:])]`.
@inlinable @inline(__always)
public static func matrixDiagPart<T: TensorFlowScalar>(
  _ input: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("MatrixDiagPart",
    input,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Deprecated, use python implementation tf.linalg.matrix_exponential.
@inlinable @inline(__always)
public static func matrixExponential<T: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("MatrixExponential",
    input,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the inverse of one or more square invertible matrices or their
///
/// adjoints (conjugate transposes).
///
/// The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
/// form square matrices. The output is a tensor of the same shape as the input
/// containing the inverse for all input submatrices `[..., :, :]`.
///
/// The op uses LU decomposition with partial pivoting to compute the inverses.
///
/// If a matrix is not invertible there is no guarantee what the op does. It
/// may detect the condition and raise an exception or it may simply return a
/// garbage result.
///
/// - Parameter input: Shape is `[..., M, M]`.
///
/// - Output output: Shape is `[..., M, M]`.
///
///   @compatibility(numpy)
///   Equivalent to np.linalg.inv
///   @end_compatibility
@inlinable @inline(__always)
public static func matrixInverse<T: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>,
  adjoint: Bool = false
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("MatrixInverse",
    input,
    T$dtype: T.tensorFlowDataType,
    adjoint: adjoint)
  return Tensor(handle: ret)
}

/// Computes the matrix logarithm of one or more square matrices:
///
///
/// \\(log(exp(A)) = A\\)
///
/// This op is only defined for complex matrices. If A is positive-definite and
/// real, then casting to a complex matrix, taking the logarithm and casting back
/// to a real matrix will give the correct result.
///
/// This function computes the matrix logarithm using the Schur-Parlett algorithm.
/// Details of the algorithm can be found in Section 11.6.2 of:
/// Nicholas J. Higham, Functions of Matrices: Theory and Computation, SIAM 2008.
/// ISBN 978-0-898716-46-7.
///
/// The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
/// form square matrices. The output is a tensor of the same shape as the input
/// containing the exponential for all input submatrices `[..., :, :]`.
///
/// - Parameter input: Shape is `[..., M, M]`.
///
/// - Output output: Shape is `[..., M, M]`.
///
///   @compatibility(scipy)
///   Equivalent to scipy.linalg.logm
///   @end_compatibility
@inlinable @inline(__always)
public static func matrixLogarithm<T: TensorFlowScalar>(
  _ input: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("MatrixLogarithm",
    input,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Returns a batched matrix tensor with new batched diagonal values.
///
/// Given `input` and `diagonal`, this operation returns a tensor with the
/// same shape and values as `input`, except for the main diagonal of the
/// innermost matrices.  These will be overwritten by the values in `diagonal`.
///
/// The output is computed as follows:
///
/// Assume `input` has `k+1` dimensions `[I, J, K, ..., M, N]` and `diagonal` has
/// `k` dimensions `[I, J, K, ..., min(M, N)]`.  Then the output is a
/// tensor of rank `k+1` with dimensions `[I, J, K, ..., M, N]` where:
///
///   * `output[i, j, k, ..., m, n] = diagonal[i, j, k, ..., n]` for `m == n`.
///   * `output[i, j, k, ..., m, n] = input[i, j, k, ..., m, n]` for `m != n`.
///
/// - Parameters:
///   - input: Rank `k+1`, where `k >= 1`.
///   - diagonal: Rank `k`, where `k >= 1`.
///
/// - Output output: Rank `k+1`, with `output.shape = input.shape`.
@inlinable @inline(__always)
public static func matrixSetDiag<T: TensorFlowScalar>(
  _ input: Tensor<T>,
  diagonal: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("MatrixSetDiag",
    input,
    diagonal,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Solves systems of linear equations.
///
/// `Matrix` is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
/// form square matrices. `Rhs` is a tensor of shape `[..., M, K]`. The `output` is
/// a tensor shape `[..., M, K]`.  If `adjoint` is `False` then each output matrix
/// satisfies `matrix[..., :, :] * output[..., :, :] = rhs[..., :, :]`.
/// If `adjoint` is `True` then each output matrix satisfies
/// `adjoint(matrix[..., :, :]) * output[..., :, :] = rhs[..., :, :]`.
///
/// - Parameters:
///   - matrix: Shape is `[..., M, M]`.
///   - rhs: Shape is `[..., M, K]`.
///
/// - Attr adjoint: Boolean indicating whether to solve with `matrix` or its (block-wise)
///   adjoint.
///
/// - Output output: Shape is `[..., M, K]`.
@inlinable @inline(__always)
public static func matrixSolve<T: FloatingPoint & TensorFlowScalar>(
  matrix: Tensor<T>,
  rhs: Tensor<T>,
  adjoint: Bool = false
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("MatrixSolve",
    matrix,
    rhs,
    T$dtype: T.tensorFlowDataType,
    adjoint: adjoint)
  return Tensor(handle: ret)
}

/// Solves one or more linear least-squares problems.
///
/// `matrix` is a tensor of shape `[..., M, N]` whose inner-most 2 dimensions
/// form real or complex matrices of size `[M, N]`. `Rhs` is a tensor of the same
/// type as `matrix` and shape `[..., M, K]`.
/// The output is a tensor shape `[..., N, K]` where each output matrix solves
/// each of the equations
/// `matrix[..., :, :]` * `output[..., :, :]` = `rhs[..., :, :]`
/// in the least squares sense.
///
/// We use the following notation for (complex) matrix and right-hand sides
/// in the batch:
///
/// `matrix`=\\(A \in \mathbb{C}^{m \times n}\\),
/// `rhs`=\\(B  \in \mathbb{C}^{m \times k}\\),
/// `output`=\\(X  \in \mathbb{C}^{n \times k}\\),
/// `l2_regularizer`=\\(\lambda \in \mathbb{R}\\).
///
/// If `fast` is `True`, then the solution is computed by solving the normal
/// equations using Cholesky decomposition. Specifically, if \\(m \ge n\\) then
/// \\(X = (A^H A + \lambda I)^{-1} A^H B\\), which solves the least-squares
/// problem \\(X = \mathrm{argmin}_{Z \in \Re^{n \times k} } ||A Z - B||_F^2 + \lambda ||Z||_F^2\\). 
/// If \\(m \lt n\\) then `output` is computed as
/// \\(X = A^H (A A^H + \lambda I)^{-1} B\\), which (for \\(\lambda = 0\\)) is the
/// minimum-norm solution to the under-determined linear system, i.e.
/// \\(X = \mathrm{argmin}_{Z \in \mathbb{C}^{n \times k} } ||Z||_F^2 \\),
/// subject to \\(A Z = B\\). Notice that the fast path is only numerically stable
/// when \\(A\\) is numerically full rank and has a condition number
/// \\(\mathrm{cond}(A) \lt \frac{1}{\sqrt{\epsilon_{mach} } }\\) or \\(\lambda\\) is
/// sufficiently large.
///
/// If `fast` is `False` an algorithm based on the numerically robust complete
/// orthogonal decomposition is used. This computes the minimum-norm
/// least-squares solution, even when \\(A\\) is rank deficient. This path is
/// typically 6-7 times slower than the fast path. If `fast` is `False` then
/// `l2_regularizer` is ignored.
///
/// - Parameters:
///   - matrix: Shape is `[..., M, N]`.
///   - rhs: Shape is `[..., M, K]`.
///   - l2_regularizer: Scalar tensor.
///
///     @compatibility(numpy)
///     Equivalent to np.linalg.lstsq
///     @end_compatibility
///
/// - Output output: Shape is `[..., N, K]`.
@inlinable @inline(__always)
public static func matrixSolveLs<T: FloatingPoint & TensorFlowScalar>(
  matrix: Tensor<T>,
  rhs: Tensor<T>,
  l2Regularizer: Tensor<Double>,
  fast: Bool = true
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("MatrixSolveLs",
    matrix,
    rhs,
    l2Regularizer,
    T$dtype: T.tensorFlowDataType,
    fast: fast)
  return Tensor(handle: ret)
}

/// Computes the matrix square root of one or more square matrices:
///
/// matmul(sqrtm(A), sqrtm(A)) = A
///
/// The input matrix should be invertible. If the input matrix is real, it should
/// have no eigenvalues which are real and negative (pairs of complex conjugate
/// eigenvalues are allowed).
///
/// The matrix square root is computed by first reducing the matrix to 
/// quasi-triangular form with the real Schur decomposition. The square root 
/// of the quasi-triangular matrix is then computed directly. Details of 
/// the algorithm can be found in: Nicholas J. Higham, "Computing real 
/// square roots of a real matrix", Linear Algebra Appl., 1987.
///
/// The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
/// form square matrices. The output is a tensor of the same shape as the input
/// containing the matrix square root for all input submatrices `[..., :, :]`.
///
/// - Parameter input: Shape is `[..., M, M]`.
///
/// - Output output: Shape is `[..., M, M]`.
///
///   @compatibility(scipy)
///   Equivalent to scipy.linalg.sqrtm
///   @end_compatibility
@inlinable @inline(__always)
public static func matrixSquareRoot<T: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("MatrixSquareRoot",
    input,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Solves systems of linear equations with upper or lower triangular matrices by
///
/// backsubstitution.
///
/// `matrix` is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions form
/// square matrices. If `lower` is `True` then the strictly upper triangular part
/// of each inner-most matrix is assumed to be zero and not accessed.
/// If `lower` is False then the strictly lower triangular part of each inner-most
/// matrix is assumed to be zero and not accessed.
/// `rhs` is a tensor of shape `[..., M, K]`.
///
/// The output is a tensor of shape `[..., M, K]`. If `adjoint` is
/// `True` then the innermost matrices in `output` satisfy matrix equations
/// `matrix[..., :, :] * output[..., :, :] = rhs[..., :, :]`.
/// If `adjoint` is `False` then the strictly then the  innermost matrices in
/// `output` satisfy matrix equations
/// `adjoint(matrix[..., i, k]) * output[..., k, j] = rhs[..., i, j]`.
///
/// - Parameters:
///   - matrix: Shape is `[..., M, M]`.
///   - rhs: Shape is `[..., M, K]`.
///
/// - Attrs:
///   - lower: Boolean indicating whether the innermost matrices in `matrix` are
///     lower or upper triangular.
///   - adjoint: Boolean indicating whether to solve with `matrix` or its (block-wise)
///              adjoint.
///
///     @compatibility(numpy)
///     Equivalent to scipy.linalg.solve_triangular
///     @end_compatibility
///
/// - Output output: Shape is `[..., M, K]`.
@inlinable @inline(__always)
public static func matrixTriangularSolve<T: FloatingPoint & TensorFlowScalar>(
  matrix: Tensor<T>,
  rhs: Tensor<T>,
  lower: Bool = true,
  adjoint: Bool = false
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("MatrixTriangularSolve",
    matrix,
    rhs,
    T$dtype: T.tensorFlowDataType,
    lower: lower,
    adjoint: adjoint)
  return Tensor(handle: ret)
}

/// Computes the maximum of elements across dimensions of a tensor.
///
/// Reduces `input` along the dimensions given in `axis`. Unless
/// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
/// `axis`. If `keep_dims` is true, the reduced dimensions are
/// retained with length 1.
///
/// - Parameters:
///   - input: The tensor to reduce.
///   - reduction_indices: The dimensions to reduce. Must be in the range
///     `[-rank(input), rank(input))`.
///
/// - Attr keep_dims: If true, retain reduced dimensions with length 1.
///
/// - Output output: The reduced tensor.
@inlinable @inline(__always)
public static func max<T: Numeric & TensorFlowScalar, Tidx: BinaryInteger & TensorFlowScalar>(
  _ input: Tensor<T>,
  reductionIndices: Tensor<Tidx>,
  keepDims: Bool = false
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Max",
    input,
    reductionIndices,
    T$dtype: T.tensorFlowDataType,
    Tidx$dtype: Tidx.tensorFlowDataType,
    keep_dims: keepDims)
  return Tensor(handle: ret)
}

/// Performs max pooling on the input.
///
/// - Parameter input: 4-D input to pool over.
///
/// - Attrs:
///   - ksize: The size of the window for each dimension of the input tensor.
///   - strides: The stride of the sliding window for each dimension of the
///     input tensor.
///   - padding: The type of padding algorithm to use.
///   - data_format: Specify the data format of the input and output data. With the
///     default format "NHWC", the data is stored in the order of:
///         [batch, in_height, in_width, in_channels].
///     Alternatively, the format could be "NCHW", the data storage order of:
///         [batch, in_channels, in_height, in_width].
///
/// - Output output: The max pooled output tensor.
@inlinable @inline(__always)
public static func maxPool<T: Numeric & TensorFlowScalar>(
  _ input: Tensor<T>,
  ksize: [Int32],
  strides: [Int32],
  padding: Padding,
  dataFormat: DataFormat4 = .nhwc
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("MaxPool",
    input,
    T$dtype: T.tensorFlowDataType,
    ksize: ksize,
    strides: strides,
    padding: padding.cName,
    data_format: dataFormat.cName)
  return Tensor(handle: ret)
}

/// Performs 3D max pooling on the input.
///
/// - Parameter input: Shape `[batch, depth, rows, cols, channels]` tensor to pool over.
///
/// - Attrs:
///   - ksize: 1-D tensor of length 5. The size of the window for each dimension of
///     the input tensor. Must have `ksize[0] = ksize[4] = 1`.
///   - strides: 1-D tensor of length 5. The stride of the sliding window for each
///     dimension of `input`. Must have `strides[0] = strides[4] = 1`.
///   - padding: The type of padding algorithm to use.
///   - data_format: The data format of the input and output data. With the
///     default format "NDHWC", the data is stored in the order of:
///         [batch, in_depth, in_height, in_width, in_channels].
///     Alternatively, the format could be "NCDHW", the data storage order is:
///         [batch, in_channels, in_depth, in_height, in_width].
///
/// - Output output: The max pooled output tensor.
@inlinable @inline(__always)
public static func maxPool3D<T: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>,
  ksize: [Int32],
  strides: [Int32],
  padding: Padding,
  dataFormat: DataFormat1 = .ndhwc
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("MaxPool3D",
    input,
    T$dtype: T.tensorFlowDataType,
    ksize: ksize,
    strides: strides,
    padding: padding.cName,
    data_format: dataFormat.cName)
  return Tensor(handle: ret)
}

/// Computes gradients of max pooling function.
///
/// - Parameters:
///   - orig_input: The original input tensor.
///   - orig_output: The original output tensor.
///   - grad: Output backprop of shape `[batch, depth, rows, cols, channels]`.
///
/// - Attrs:
///   - ksize: 1-D tensor of length 5. The size of the window for each dimension of
///     the input tensor. Must have `ksize[0] = ksize[4] = 1`.
///   - strides: 1-D tensor of length 5. The stride of the sliding window for each
///     dimension of `input`. Must have `strides[0] = strides[4] = 1`.
///   - padding: The type of padding algorithm to use.
///   - data_format: The data format of the input and output data. With the
///     default format "NDHWC", the data is stored in the order of:
///         [batch, in_depth, in_height, in_width, in_channels].
///     Alternatively, the format could be "NCDHW", the data storage order is:
///         [batch, in_channels, in_depth, in_height, in_width].
@inlinable @inline(__always)
public static func maxPool3DGrad<T: FloatingPoint & TensorFlowScalar, Tinput: FloatingPoint & TensorFlowScalar>(
  origInput: Tensor<Tinput>,
  origOutput: Tensor<Tinput>,
  grad: Tensor<T>,
  ksize: [Int32],
  strides: [Int32],
  padding: Padding,
  dataFormat: DataFormat1 = .ndhwc
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("MaxPool3DGrad",
    origInput,
    origOutput,
    grad,
    T$dtype: T.tensorFlowDataType,
    TInput$dtype: Tinput.tensorFlowDataType,
    ksize: ksize,
    strides: strides,
    padding: padding.cName,
    data_format: dataFormat.cName)
  return Tensor(handle: ret)
}

/// Computes second-order gradients of the maxpooling function.
///
/// - Parameters:
///   - orig_input: The original input tensor.
///   - orig_output: The original output tensor.
///   - grad: Output backprop of shape `[batch, depth, rows, cols, channels]`.
///
/// - Attrs:
///   - ksize: 1-D tensor of length 5. The size of the window for each dimension of
///     the input tensor. Must have `ksize[0] = ksize[4] = 1`.
///   - strides: 1-D tensor of length 5. The stride of the sliding window for each
///     dimension of `input`. Must have `strides[0] = strides[4] = 1`.
///   - padding: The type of padding algorithm to use.
///   - data_format: The data format of the input and output data. With the
///     default format "NDHWC", the data is stored in the order of:
///         [batch, in_depth, in_height, in_width, in_channels].
///     Alternatively, the format could be "NCDHW", the data storage order is:
///         [batch, in_channels, in_depth, in_height, in_width].
///
/// - Output output: Gradients of gradients w.r.t. the input to `max_pool`.
@inlinable @inline(__always)
public static func maxPool3DGradGrad<T: Numeric & TensorFlowScalar>(
  origInput: Tensor<T>,
  origOutput: Tensor<T>,
  grad: Tensor<T>,
  ksize: [Int32],
  strides: [Int32],
  padding: Padding,
  dataFormat: DataFormat1 = .ndhwc
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("MaxPool3DGradGrad",
    origInput,
    origOutput,
    grad,
    T$dtype: T.tensorFlowDataType,
    ksize: ksize,
    strides: strides,
    padding: padding.cName,
    data_format: dataFormat.cName)
  return Tensor(handle: ret)
}

/// Computes gradients of the maxpooling function.
///
/// - Parameters:
///   - orig_input: The original input tensor.
///   - orig_output: The original output tensor.
///   - grad: 4-D.  Gradients w.r.t. the output of `max_pool`.
///
/// - Attrs:
///   - ksize: The size of the window for each dimension of the input tensor.
///   - strides: The stride of the sliding window for each dimension of the
///     input tensor.
///   - padding: The type of padding algorithm to use.
///   - data_format: Specify the data format of the input and output data. With the
///     default format "NHWC", the data is stored in the order of:
///         [batch, in_height, in_width, in_channels].
///     Alternatively, the format could be "NCHW", the data storage order of:
///         [batch, in_channels, in_height, in_width].
///
/// - Output output: Gradients w.r.t. the input to `max_pool`.
@inlinable @inline(__always)
public static func maxPoolGrad<T: Numeric & TensorFlowScalar>(
  origInput: Tensor<T>,
  origOutput: Tensor<T>,
  grad: Tensor<T>,
  ksize: [Int32],
  strides: [Int32],
  padding: Padding,
  dataFormat: DataFormat = .nhwc
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("MaxPoolGrad",
    origInput,
    origOutput,
    grad,
    T$dtype: T.tensorFlowDataType,
    ksize: ksize,
    strides: strides,
    padding: padding.cName,
    data_format: dataFormat.cName)
  return Tensor(handle: ret)
}

/// Computes second-order gradients of the maxpooling function.
///
/// - Parameters:
///   - orig_input: The original input tensor.
///   - orig_output: The original output tensor.
///   - grad: 4-D.  Gradients of gradients w.r.t. the input of `max_pool`.
///
/// - Attrs:
///   - ksize: The size of the window for each dimension of the input tensor.
///   - strides: The stride of the sliding window for each dimension of the
///     input tensor.
///   - padding: The type of padding algorithm to use.
///   - data_format: Specify the data format of the input and output data. With the
///     default format "NHWC", the data is stored in the order of:
///         [batch, in_height, in_width, in_channels].
///     Alternatively, the format could be "NCHW", the data storage order of:
///         [batch, in_channels, in_height, in_width].
///
/// - Output output: Gradients of gradients w.r.t. the input to `max_pool`.
@inlinable @inline(__always)
public static func maxPoolGradGrad<T: Numeric & TensorFlowScalar>(
  origInput: Tensor<T>,
  origOutput: Tensor<T>,
  grad: Tensor<T>,
  ksize: [Int32],
  strides: [Int32],
  padding: Padding,
  dataFormat: DataFormat = .nhwc
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("MaxPoolGradGrad",
    origInput,
    origOutput,
    grad,
    T$dtype: T.tensorFlowDataType,
    ksize: ksize,
    strides: strides,
    padding: padding.cName,
    data_format: dataFormat.cName)
  return Tensor(handle: ret)
}

/// Computes second-order gradients of the maxpooling function.
///
/// - Parameters:
///   - orig_input: The original input tensor.
///   - orig_output: The original output tensor.
///   - grad: 4-D.  Gradients of gradients w.r.t. the input of `max_pool`.
///   - ksize: The size of the window for each dimension of the input tensor.
///   - strides: The stride of the sliding window for each dimension of the
///     input tensor.
///
/// - Attrs:
///   - padding: The type of padding algorithm to use.
///   - data_format: Specify the data format of the input and output data. With the
///     default format "NHWC", the data is stored in the order of:
///         [batch, in_height, in_width, in_channels].
///     Alternatively, the format could be "NCHW", the data storage order of:
///         [batch, in_channels, in_height, in_width].
///
/// - Output output: Gradients of gradients w.r.t. the input to `max_pool`.
@inlinable @inline(__always)
public static func maxPoolGradGradV2<T: Numeric & TensorFlowScalar>(
  origInput: Tensor<T>,
  origOutput: Tensor<T>,
  grad: Tensor<T>,
  ksize: Tensor<Int32>,
  strides: Tensor<Int32>,
  padding: Padding,
  dataFormat: DataFormat = .nhwc
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("MaxPoolGradGradV2",
    origInput,
    origOutput,
    grad,
    ksize,
    strides,
    T$dtype: T.tensorFlowDataType,
    padding: padding.cName,
    data_format: dataFormat.cName)
  return Tensor(handle: ret)
}

/// Computes second-order gradients of the maxpooling function.
///
/// - Parameters:
///   - input: The original input.
///   - grad: 4-D with shape `[batch, height, width, channels]`.  Gradients w.r.t. the
///     input of `max_pool`.
///   - argmax: The indices of the maximum values chosen for each output of `max_pool`.
///
/// - Attrs:
///   - ksize: The size of the window for each dimension of the input tensor.
///   - strides: The stride of the sliding window for each dimension of the
///     input tensor.
///   - padding: The type of padding algorithm to use.
///   - include_batch_in_index: Whether to include batch dimension in flattened index of `argmax`.
///
/// - Output output: Gradients of gradients w.r.t. the input of `max_pool`.
@inlinable @inline(__always)
public static func maxPoolGradGradWithArgmax<Targmax: BinaryInteger & TensorFlowScalar, T: Numeric & TensorFlowScalar>(
  _ input: Tensor<T>,
  grad: Tensor<T>,
  argmax: Tensor<Targmax>,
  ksize: [Int32],
  strides: [Int32],
  padding: Padding,
  includeBatchInIndex: Bool = false
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("MaxPoolGradGradWithArgmax",
    input,
    grad,
    argmax,
    Targmax$dtype: Targmax.tensorFlowDataType,
    T$dtype: T.tensorFlowDataType,
    ksize: ksize,
    strides: strides,
    padding: padding.cName,
    include_batch_in_index: includeBatchInIndex)
  return Tensor(handle: ret)
}

/// Computes gradients of the maxpooling function.
///
/// - Parameters:
///   - orig_input: The original input tensor.
///   - orig_output: The original output tensor.
///   - grad: 4-D.  Gradients w.r.t. the output of `max_pool`.
///   - ksize: The size of the window for each dimension of the input tensor.
///   - strides: The stride of the sliding window for each dimension of the
///     input tensor.
///
/// - Attrs:
///   - padding: The type of padding algorithm to use.
///   - data_format: Specify the data format of the input and output data. With the
///     default format "NHWC", the data is stored in the order of:
///         [batch, in_height, in_width, in_channels].
///     Alternatively, the format could be "NCHW", the data storage order of:
///         [batch, in_channels, in_height, in_width].
///
/// - Output output: Gradients w.r.t. the input to `max_pool`.
@inlinable @inline(__always)
public static func maxPoolGradV2<T: Numeric & TensorFlowScalar>(
  origInput: Tensor<T>,
  origOutput: Tensor<T>,
  grad: Tensor<T>,
  ksize: Tensor<Int32>,
  strides: Tensor<Int32>,
  padding: Padding,
  dataFormat: DataFormat = .nhwc
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("MaxPoolGradV2",
    origInput,
    origOutput,
    grad,
    ksize,
    strides,
    T$dtype: T.tensorFlowDataType,
    padding: padding.cName,
    data_format: dataFormat.cName)
  return Tensor(handle: ret)
}

/// Computes gradients of the maxpooling function.
///
/// - Parameters:
///   - input: The original input.
///   - grad: 4-D with shape `[batch, height, width, channels]`.  Gradients w.r.t. the
///     output of `max_pool`.
///   - argmax: The indices of the maximum values chosen for each output of `max_pool`.
///
/// - Attrs:
///   - ksize: The size of the window for each dimension of the input tensor.
///   - strides: The stride of the sliding window for each dimension of the
///     input tensor.
///   - padding: The type of padding algorithm to use.
///   - include_batch_in_index: Whether to include batch dimension in flattened index of `argmax`.
///
/// - Output output: Gradients w.r.t. the input of `max_pool`.
@inlinable @inline(__always)
public static func maxPoolGradWithArgmax<Targmax: BinaryInteger & TensorFlowScalar, T: Numeric & TensorFlowScalar>(
  _ input: Tensor<T>,
  grad: Tensor<T>,
  argmax: Tensor<Targmax>,
  ksize: [Int32],
  strides: [Int32],
  padding: Padding,
  includeBatchInIndex: Bool = false
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("MaxPoolGradWithArgmax",
    input,
    grad,
    argmax,
    Targmax$dtype: Targmax.tensorFlowDataType,
    T$dtype: T.tensorFlowDataType,
    ksize: ksize,
    strides: strides,
    padding: padding.cName,
    include_batch_in_index: includeBatchInIndex)
  return Tensor(handle: ret)
}

/// Performs max pooling on the input.
///
/// - Parameters:
///   - input: 4-D input to pool over.
///   - ksize: The size of the window for each dimension of the input tensor.
///   - strides: The stride of the sliding window for each dimension of the
///     input tensor.
///
/// - Attrs:
///   - padding: The type of padding algorithm to use.
///   - data_format: Specify the data format of the input and output data. With the
///     default format "NHWC", the data is stored in the order of:
///         [batch, in_height, in_width, in_channels].
///     Alternatively, the format could be "NCHW", the data storage order of:
///         [batch, in_channels, in_height, in_width].
///
/// - Output output: The max pooled output tensor.
@inlinable @inline(__always)
public static func maxPoolV2<T: Numeric & TensorFlowScalar>(
  _ input: Tensor<T>,
  ksize: Tensor<Int32>,
  strides: Tensor<Int32>,
  padding: Padding,
  dataFormat: DataFormat4 = .nhwc
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("MaxPoolV2",
    input,
    ksize,
    strides,
    T$dtype: T.tensorFlowDataType,
    padding: padding.cName,
    data_format: dataFormat.cName)
  return Tensor(handle: ret)
}

/// Performs max pooling on the input and outputs both max values and indices.
///
/// The indices in `argmax` are flattened, so that a maximum value at position
/// `[b, y, x, c]` becomes flattened index:
/// `(y * width + x) * channels + c` if `include_batch_in_index` is False;
/// `((b * height + y) * width + x) * channels + c` if `include_batch_in_index` is True.
///
/// The indices returned are always in `[0, height) x [0, width)` before flattening,
/// even if padding is involved and the mathematically correct answer is outside
/// (either negative or too large).  This is a bug, but fixing it is difficult to do
/// in a safe backwards compatible way, especially due to flattening.
///
/// - Parameter input: 4-D with shape `[batch, height, width, channels]`.  Input to pool over.
///
/// - Attrs:
///   - ksize: The size of the window for each dimension of the input tensor.
///   - strides: The stride of the sliding window for each dimension of the
///     input tensor.
///   - padding: The type of padding algorithm to use.
///   - include_batch_in_index: Whether to include batch dimension in flattened index of `argmax`.
///
/// - Outputs:
///   - output: The max pooled output tensor.
///   - argmax: 4-D.  The flattened indices of the max values chosen for each output.
@inlinable @inline(__always)
public static func maxPoolWithArgmax<Targmax: BinaryInteger & TensorFlowScalar, T: Numeric & TensorFlowScalar>(
  _ input: Tensor<T>,
  ksize: [Int32],
  strides: [Int32],
  padding: Padding,
  includeBatchInIndex: Bool = false
) -> (output: Tensor<T>, argmax: Tensor<Targmax>) {
  let ret: (TensorHandle<T>, TensorHandle<Targmax>) = #tfop("MaxPoolWithArgmax",
    input,
    Targmax$dtype: Targmax.tensorFlowDataType,
    T$dtype: T.tensorFlowDataType,
    ksize: ksize,
    strides: strides,
    padding: padding.cName,
    include_batch_in_index: includeBatchInIndex)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

/// Returns the max of x and y (i.e. x > y ? x : y) element-wise.
///
/// *NOTE*: `Maximum` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@inlinable @inline(__always)
public static func maximum<T: Numeric & TensorFlowScalar>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Maximum",
    x,
    y,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the mean of elements across dimensions of a tensor.
///
/// Reduces `input` along the dimensions given in `axis`. Unless
/// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
/// `axis`. If `keep_dims` is true, the reduced dimensions are
/// retained with length 1.
///
/// - Parameters:
///   - input: The tensor to reduce.
///   - reduction_indices: The dimensions to reduce. Must be in the range
///     `[-rank(input), rank(input))`.
///
/// - Attr keep_dims: If true, retain reduced dimensions with length 1.
///
/// - Output output: The reduced tensor.
@inlinable @inline(__always)
public static func mean<T: Numeric & TensorFlowScalar, Tidx: BinaryInteger & TensorFlowScalar>(
  _ input: Tensor<T>,
  reductionIndices: Tensor<Tidx>,
  keepDims: Bool = false
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Mean",
    input,
    reductionIndices,
    T$dtype: T.tensorFlowDataType,
    Tidx$dtype: Tidx.tensorFlowDataType,
    keep_dims: keepDims)
  return Tensor(handle: ret)
}

/// Forwards the value of an available tensor from `inputs` to `output`.
///
/// `Merge` waits for at least one of the tensors in `inputs` to become available.
/// It is usually combined with `Switch` to implement branching.
///
/// `Merge` forwards the first tensor to become available to `output`, and sets
/// `value_index` to its index in `inputs`.
///
/// - Parameter inputs: The input tensors, exactly one of which will become available.
///
/// - Outputs:
///   - output: Will be set to the available input tensor.
///   - value_index: The index of the chosen input tensor in `inputs`.
@inlinable @inline(__always)
public static func merge<T: TensorFlowScalar>(
  inputs: [Tensor<T>]
) -> (output: Tensor<T>, valueIndex: Tensor<Int32>) {
  let ret: (TensorHandle<T>, TensorHandle<Int32>) = #tfop("Merge",
    inputs,
    T$dtype: T.tensorFlowDataType)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

/// Merges summaries.
///
/// This op creates a
/// [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
/// protocol buffer that contains the union of all the values in the input
/// summaries.
///
/// When the Op is run, it reports an `InvalidArgument` error if multiple values
/// in the summaries to merge use the same tag.
///
/// - Parameter inputs: Can be of any shape.  Each must contain serialized `Summary` protocol
///   buffers.
///
/// - Output summary: Scalar. Serialized `Summary` protocol buffer.
@inlinable @inline(__always)
public static func mergeSummary(
  inputs: [StringTensor]
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("MergeSummary",
    inputs)
  return StringTensor(handle: ret)
}

/// V2 format specific: merges the metadata files of sharded checkpoints.  The
///
/// result is one logical checkpoint, with one physical metadata file and renamed
/// data files.
///
/// Intended for "grouping" multiple checkpoints in a sharded checkpoint setup.
///
/// If delete_old_dirs is true, attempts to delete recursively the dirname of each
/// path in the input checkpoint_prefixes.  This is useful when those paths are non
/// user-facing temporary locations.
///
/// - Parameters:
///   - checkpoint_prefixes: prefixes of V2 checkpoints to merge.
///   - destination_prefix: scalar.  The desired final prefix.  Allowed to be the same
///     as one of the checkpoint_prefixes.
///
/// - Attr delete_old_dirs: see above.
@inlinable @inline(__always)
public static func mergeV2Checkpoints(
  checkpointPrefixes: StringTensor,
  destinationPrefix: StringTensor,
  deleteOldDirs: Bool = true
) {
  return #tfop("MergeV2Checkpoints",
    checkpointPrefixes,
    destinationPrefix,
    delete_old_dirs: deleteOldDirs)
}

/// Transforms a spectrogram into a form that's useful for speech recognition.
///
/// Mel Frequency Cepstral Coefficients are a way of representing audio data that's
/// been effective as an input feature for machine learning. They are created by
/// taking the spectrum of a spectrogram (a 'cepstrum'), and discarding some of the
/// higher frequencies that are less significant to the human ear. They have a long
/// history in the speech recognition world, and https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
/// is a good resource to learn more.
///
/// - Parameters:
///   - spectrogram: Typically produced by the Spectrogram op, with magnitude_squared
///     set to true.
///   - sample_rate: How many samples per second the source audio used.
///
/// - Attrs:
///   - upper_frequency_limit: The highest frequency to use when calculating the
///     ceptstrum.
///   - lower_frequency_limit: The lowest frequency to use when calculating the
///     ceptstrum.
///   - filterbank_channel_count: Resolution of the Mel bank used internally.
///   - dct_coefficient_count: How many output channels to produce per time slice.
@inlinable @inline(__always)
public static func mfcc(
  spectrogram: Tensor<Float>,
  sampleRate: Tensor<Int32>,
  upperFrequencyLimit: Double = 4000,
  lowerFrequencyLimit: Double = 20,
  filterbankChannelCount: Int64 = 40,
  dctCoefficientCount: Int64 = 13
) -> Tensor<Float> {
  let ret: TensorHandle<Float> = #tfop("Mfcc",
    spectrogram,
    sampleRate,
    upper_frequency_limit: upperFrequencyLimit,
    lower_frequency_limit: lowerFrequencyLimit,
    filterbank_channel_count: filterbankChannelCount,
    dct_coefficient_count: dctCoefficientCount)
  return Tensor(handle: ret)
}

/// Computes the minimum of elements across dimensions of a tensor.
///
/// Reduces `input` along the dimensions given in `axis`. Unless
/// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
/// `axis`. If `keep_dims` is true, the reduced dimensions are
/// retained with length 1.
///
/// - Parameters:
///   - input: The tensor to reduce.
///   - reduction_indices: The dimensions to reduce. Must be in the range
///     `[-rank(input), rank(input))`.
///
/// - Attr keep_dims: If true, retain reduced dimensions with length 1.
///
/// - Output output: The reduced tensor.
@inlinable @inline(__always)
public static func min<T: Numeric & TensorFlowScalar, Tidx: BinaryInteger & TensorFlowScalar>(
  _ input: Tensor<T>,
  reductionIndices: Tensor<Tidx>,
  keepDims: Bool = false
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Min",
    input,
    reductionIndices,
    T$dtype: T.tensorFlowDataType,
    Tidx$dtype: Tidx.tensorFlowDataType,
    keep_dims: keepDims)
  return Tensor(handle: ret)
}

/// Returns the min of x and y (i.e. x < y ? x : y) element-wise.
///
/// *NOTE*: `Minimum` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@inlinable @inline(__always)
public static func minimum<T: Numeric & TensorFlowScalar>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Minimum",
    x,
    y,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Pads a tensor with mirrored values.
///
/// This operation pads a `input` with mirrored values according to the `paddings`
/// you specify. `paddings` is an integer tensor with shape `[n, 2]`, where n is
/// the rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
/// how many values to add before the contents of `input` in that dimension, and
/// `paddings[D, 1]` indicates how many values to add after the contents of `input`
/// in that dimension. Both `paddings[D, 0]` and `paddings[D, 1]` must be no greater
/// than `input.dim_size(D)` (or `input.dim_size(D) - 1`) if `copy_border` is true
/// (if false, respectively).
///
/// The padded size of each dimension D of the output is:
///
/// `paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`
///
/// For example:
///
/// ```
/// # 't' is [[1, 2, 3], [4, 5, 6]].
/// # 'paddings' is [[1, 1]], [2, 2]].
/// # 'mode' is SYMMETRIC.
/// # rank of 't' is 2.
/// pad(t, paddings) ==> [[2, 1, 1, 2, 3, 3, 2]
///                       [2, 1, 1, 2, 3, 3, 2]
///                       [5, 4, 4, 5, 6, 6, 5]
///                       [5, 4, 4, 5, 6, 6, 5]]
/// ```
///
/// - Parameters:
///   - input: The input tensor to be padded.
///   - paddings: A two-column matrix specifying the padding sizes. The number of
///     rows must be the same as the rank of `input`.
///
/// - Attr mode: Either `REFLECT` or `SYMMETRIC`. In reflect mode the padded regions
///   do not include the borders, while in symmetric mode the padded regions
///   do include the borders. For example, if `input` is `[1, 2, 3]` and `paddings`
///   is `[0, 2]`, then the output is `[1, 2, 3, 2, 1]` in reflect mode, and
///   it is `[1, 2, 3, 3, 2]` in symmetric mode.
///
/// - Output output: The padded tensor.
@inlinable @inline(__always)
public static func mirrorPad<T: TensorFlowScalar, Tpaddings: BinaryInteger & TensorFlowScalar>(
  _ input: Tensor<T>,
  paddings: Tensor<Tpaddings>,
  mode: Mode5
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("MirrorPad",
    input,
    paddings,
    T$dtype: T.tensorFlowDataType,
    Tpaddings$dtype: Tpaddings.tensorFlowDataType,
    mode: mode.cName)
  return Tensor(handle: ret)
}

/// Gradient op for `MirrorPad` op. This op folds a mirror-padded tensor.
///
/// This operation folds the padded areas of `input` by `MirrorPad` according to the
/// `paddings` you specify. `paddings` must be the same as `paddings` argument
/// given to the corresponding `MirrorPad` op.
///
/// The folded size of each dimension D of the output is:
///
/// `input.dim_size(D) - paddings(D, 0) - paddings(D, 1)`
///
/// For example:
///
/// ```
/// # 't' is [[1, 2, 3], [4, 5, 6], [7, 8, 9]].
/// # 'paddings' is [[0, 1]], [0, 1]].
/// # 'mode' is SYMMETRIC.
/// # rank of 't' is 2.
/// pad(t, paddings) ==> [[ 1,  5]
///                       [11, 28]]
/// ```
///
/// - Parameters:
///   - input: The input tensor to be folded.
///   - paddings: A two-column matrix specifying the padding sizes. The number of
///     rows must be the same as the rank of `input`.
///
/// - Attr mode: The mode used in the `MirrorPad` op.
///
/// - Output output: The folded tensor.
@inlinable @inline(__always)
public static func mirrorPadGrad<T: TensorFlowScalar, Tpaddings: BinaryInteger & TensorFlowScalar>(
  _ input: Tensor<T>,
  paddings: Tensor<Tpaddings>,
  mode: Mode5
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("MirrorPadGrad",
    input,
    paddings,
    T$dtype: T.tensorFlowDataType,
    Tpaddings$dtype: Tpaddings.tensorFlowDataType,
    mode: mode.cName)
  return Tensor(handle: ret)
}

/// Returns element-wise remainder of division. This emulates C semantics in that
///
/// the result here is consistent with a truncating divide. E.g.
/// `tf.truncatediv(x, y) * y + truncate_mod(x, y) = x`.
///
/// *NOTE*: `Mod` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@inlinable @inline(__always)
public static func mod<T: Numeric & TensorFlowScalar>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Mod",
    x,
    y,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Returns x * y element-wise.
///
/// *NOTE*: `Multiply` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@inlinable @inline(__always)
public static func mul<T: Numeric & TensorFlowScalar>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Mul",
    x,
    y,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Returns x * y element-wise. Returns zero if y is zero, even if x if infinite or NaN.
///
/// *NOTE*: `Mul` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@inlinable @inline(__always)
public static func mulNoNan<T: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("MulNoNan",
    x,
    y,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Draws samples from a multinomial distribution.
///
/// - Parameters:
///   - logits: 2-D Tensor with shape `[batch_size, num_classes]`.  Each slice `[i, :]`
///     represents the unnormalized log probabilities for all classes.
///   - num_samples: 0-D.  Number of independent samples to draw for each row slice.
///
/// - Attrs:
///   - seed: If either seed or seed2 is set to be non-zero, the internal random number
///     generator is seeded by the given seed.  Otherwise, a random seed is used.
///   - seed2: A second seed to avoid seed collision.
///
/// - Output output: 2-D Tensor with shape `[batch_size, num_samples]`.  Each slice `[i, :]`
///   contains the drawn class labels with range `[0, num_classes)`.
@inlinable @inline(__always)
public static func multinomial<T: Numeric & TensorFlowScalar, OutputDtype: BinaryInteger & TensorFlowScalar>(
  logits: Tensor<T>,
  numSamples: Tensor<Int32>,
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> Tensor<OutputDtype> {
  let ret: TensorHandle<OutputDtype> = #tfop("Multinomial",
    logits,
    numSamples,
    T$dtype: T.tensorFlowDataType,
    output_dtype$dtype: OutputDtype.tensorFlowDataType,
    seed: seed,
    seed2: seed2)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func nInPolymorphicTwice<T: TensorFlowScalar>(
  _ a: [Tensor<T>],
  _ b: [Tensor<T>]
) {
  return #tfop("NInPolymorphicTwice",
    a,
    b,
    T$dtype: T.tensorFlowDataType)
}

@inlinable @inline(__always)
public static func nInTwice(
  _ a: [Tensor<Int32>],
  _ b: [StringTensor]
) {
  return #tfop("NInTwice",
    a,
    b)
}

@inlinable @inline(__always)
public static func nInTwoTypeVariables<S: TensorFlowScalar, T: TensorFlowScalar>(
  _ a: [Tensor<S>],
  _ b: [Tensor<T>]
) {
  return #tfop("NInTwoTypeVariables",
    a,
    b,
    S$dtype: S.tensorFlowDataType,
    T$dtype: T.tensorFlowDataType)
}

@inlinable @inline(__always)
public static func nIntsIn(
  _ a: [Tensor<Int32>]
) {
  return #tfop("NIntsIn",
    a)
}

@inlinable @inline(__always)
public static func nPolymorphicIn<T: TensorFlowScalar>(
  _ a: [Tensor<T>]
) {
  return #tfop("NPolymorphicIn",
    a,
    T$dtype: T.tensorFlowDataType)
}

@inlinable @inline(__always)
public static func nPolymorphicRestrictIn<T: TensorFlowScalar>(
  _ a: [Tensor<T>]
) {
  return #tfop("NPolymorphicRestrictIn",
    a,
    T$dtype: T.tensorFlowDataType)
}

/// Outputs a tensor containing the reduction across all input tensors.
///
/// Outputs a tensor containing the reduction across all input tensors passed to ops
/// within the same `shared_name.
///
/// The graph should be constructed so if one op runs with shared_name value `c`,
/// then `num_devices` ops will run with shared_name value `c`.  Failure to do so
/// will cause the graph execution to fail to complete.
///
/// input: the input to the reduction
/// data: the value of the reduction across all `num_devices` devices.
/// reduction: the reduction operation to perform.
/// num_devices: The number of devices participating in this reduction.
/// shared_name: Identifier that shared between ops of the same reduction.
@inlinable @inline(__always)
public static func ncclAllReduce<T: Numeric & TensorFlowScalar>(
  _ input: Tensor<T>,
  reduction: Reduction,
  numDevices: Int64,
  sharedName: String
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("NcclAllReduce",
    input,
    T$dtype: T.tensorFlowDataType,
    reduction: reduction.cName,
    num_devices: numDevices,
    shared_name: sharedName)
  return Tensor(handle: ret)
}

/// Reduces `input` from `num_devices` using `reduction` to a single device.
///
/// Reduces `input` from `num_devices` using `reduction` to a single device.
///
/// The graph should be constructed so that all inputs have a valid device
/// assignment, and the op itself is assigned one of these devices.
///
/// input: The input to the reduction.
/// data: the value of the reduction across all `num_devices` devices.
/// reduction: the reduction operation to perform.
@inlinable @inline(__always)
public static func ncclReduce<T: Numeric & TensorFlowScalar>(
  _ input: [Tensor<T>],
  reduction: Reduction
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("NcclReduce",
    input,
    T$dtype: T.tensorFlowDataType,
    reduction: reduction.cName)
  return Tensor(handle: ret)
}

/// Selects the k nearest centers for each point.
///
/// Rows of points are assumed to be input points. Rows of centers are assumed to be
/// the list of candidate centers. For each point, the k centers that have least L2
/// distance to it are computed.
///
/// - Parameters:
///   - points: Matrix of shape (n, d). Rows are assumed to be input points.
///   - centers: Matrix of shape (m, d). Rows are assumed to be centers.
///   - k: Number of nearest centers to return for each point. If k is larger than m, then
///     only m centers are returned.
///
/// - Outputs:
///   - nearest_center_indices: Matrix of shape (n, min(m, k)). Each row contains the indices of the centers
///     closest to the corresponding point, ordered by increasing distance.
///   - nearest_center_distances: Matrix of shape (n, min(m, k)). Each row contains the squared L2 distance to the
///     corresponding center in nearest_center_indices.
@inlinable @inline(__always)
public static func nearestNeighbors(
  points: Tensor<Float>,
  centers: Tensor<Float>,
  k: Tensor<Int64>
) -> (nearestCenterIndices: Tensor<Int64>, nearestCenterDistances: Tensor<Float>) {
  let ret: (TensorHandle<Int64>, TensorHandle<Float>) = #tfop("NearestNeighbors",
    points,
    centers,
    k)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

/// Computes numerical negative value element-wise.
///
/// I.e., \\(y = -x\\).
@inlinable @inline(__always)
public static func neg<T: Numeric & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Neg",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Returns the next representable value of `x1` in the direction of `x2`, element-wise.
///
/// This operation returns the same result as the C++ std::nextafter function.
///
/// It can also return a subnormal number.
///
/// @compatibility(cpp)
/// Equivalent to C++ std::nextafter function.
/// @end_compatibility
@inlinable @inline(__always)
public static func nextAfter<T: FloatingPoint & TensorFlowScalar>(
  x1: Tensor<T>,
  x2: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("NextAfter",
    x1,
    x2,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Makes its input available to the next iteration.
///
/// - Parameter data: The tensor to be made available to the next iteration.
///
/// - Output output: The same tensor as `data`.
@inlinable @inline(__always)
public static func nextIteration<T: TensorFlowScalar>(
  data: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("NextIteration",
    data,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Does nothing. Only useful as a placeholder for control edges.
@inlinable @inline(__always)
public static func noOp(
) {
  return #tfop("NoOp")
}

/// Non-deterministically generates some integers.
///
/// This op may use some OS-provided source of non-determinism (e.g. an RNG), so each execution will give different results.
///
/// - Parameter shape: The shape of the output tensor.
///
/// - Attr dtype: The type of the output.
///
/// - Output output: Non-deterministic integer values with specified shape.
@inlinable @inline(__always)
public static func nonDeterministicInts<Dtype: TensorFlowScalar, ShapeDtype: TensorFlowScalar>(
  shape: Tensor<ShapeDtype>
) -> Tensor<Dtype> {
  let ret: TensorHandle<Dtype> = #tfop("NonDeterministicInts",
    shape,
    dtype$dtype: Dtype.tensorFlowDataType,
    shape_dtype$dtype: ShapeDtype.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Greedily selects a subset of bounding boxes in descending order of score,
///
/// pruning away boxes that have high intersection-over-union (IOU) overlap
/// with previously selected boxes.  Bounding boxes are supplied as
/// [y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any
/// diagonal pair of box corners and the coordinates can be provided as normalized
/// (i.e., lying in the interval [0, 1]) or absolute.  Note that this algorithm
/// is agnostic to where the origin is in the coordinate system.  Note that this
/// algorithm is invariant to orthogonal transformations and translations
/// of the coordinate system; thus translating or reflections of the coordinate
/// system result in the same boxes being selected by the algorithm.
/// The output of this operation is a set of integers indexing into the input
/// collection of bounding boxes representing the selected boxes.  The bounding
/// box coordinates corresponding to the selected indices can then be obtained
/// using the `tf.gather operation`.  For example:
///   selected_indices = tf.image.non_max_suppression(
///       boxes, scores, max_output_size, iou_threshold)
///   selected_boxes = tf.gather(boxes, selected_indices)
///
/// - Parameters:
///   - boxes: A 2-D float tensor of shape `[num_boxes, 4]`.
///   - scores: A 1-D float tensor of shape `[num_boxes]` representing a single
///     score corresponding to each box (each row of boxes).
///   - max_output_size: A scalar integer tensor representing the maximum number of
///     boxes to be selected by non max suppression.
///
/// - Attr iou_threshold: A float representing the threshold for deciding whether boxes
///   overlap too much with respect to IOU.
///
/// - Output selected_indices: A 1-D integer tensor of shape `[M]` representing the selected
///   indices from the boxes tensor, where `M <= max_output_size`.
@inlinable @inline(__always)
public static func nonMaxSuppression(
  boxes: Tensor<Float>,
  scores: Tensor<Float>,
  maxOutputSize: Tensor<Int32>,
  iouThreshold: Double = 0.5
) -> Tensor<Int32> {
  let ret: TensorHandle<Int32> = #tfop("NonMaxSuppression",
    boxes,
    scores,
    maxOutputSize,
    iou_threshold: iouThreshold)
  return Tensor(handle: ret)
}

/// Greedily selects a subset of bounding boxes in descending order of score,
///
/// pruning away boxes that have high intersection-over-union (IOU) overlap
/// with previously selected boxes.  Bounding boxes are supplied as
/// [y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any
/// diagonal pair of box corners and the coordinates can be provided as normalized
/// (i.e., lying in the interval [0, 1]) or absolute.  Note that this algorithm
/// is agnostic to where the origin is in the coordinate system.  Note that this
/// algorithm is invariant to orthogonal transformations and translations
/// of the coordinate system; thus translating or reflections of the coordinate
/// system result in the same boxes being selected by the algorithm.
///
/// The output of this operation is a set of integers indexing into the input
/// collection of bounding boxes representing the selected boxes.  The bounding
/// box coordinates corresponding to the selected indices can then be obtained
/// using the `tf.gather operation`.  For example:
///
///   selected_indices = tf.image.non_max_suppression_v2(
///       boxes, scores, max_output_size, iou_threshold)
///   selected_boxes = tf.gather(boxes, selected_indices)
///
/// - Parameters:
///   - boxes: A 2-D float tensor of shape `[num_boxes, 4]`.
///   - scores: A 1-D float tensor of shape `[num_boxes]` representing a single
///     score corresponding to each box (each row of boxes).
///   - max_output_size: A scalar integer tensor representing the maximum number of
///     boxes to be selected by non max suppression.
///   - iou_threshold: A 0-D float tensor representing the threshold for deciding whether
///     boxes overlap too much with respect to IOU.
///
/// - Output selected_indices: A 1-D integer tensor of shape `[M]` representing the selected
///   indices from the boxes tensor, where `M <= max_output_size`.
@inlinable @inline(__always)
public static func nonMaxSuppressionV2<T: FloatingPoint & TensorFlowScalar>(
  boxes: Tensor<T>,
  scores: Tensor<T>,
  maxOutputSize: Tensor<Int32>,
  iouThreshold: Tensor<Float>
) -> Tensor<Int32> {
  let ret: TensorHandle<Int32> = #tfop("NonMaxSuppressionV2",
    boxes,
    scores,
    maxOutputSize,
    iouThreshold,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Greedily selects a subset of bounding boxes in descending order of score,
///
/// pruning away boxes that have high intersection-over-union (IOU) overlap
/// with previously selected boxes.  Bounding boxes with score less than
/// `score_threshold` are removed.  Bounding boxes are supplied as
/// [y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any
/// diagonal pair of box corners and the coordinates can be provided as normalized
/// (i.e., lying in the interval [0, 1]) or absolute.  Note that this algorithm
/// is agnostic to where the origin is in the coordinate system and more
/// generally is invariant to orthogonal transformations and translations
/// of the coordinate system; thus translating or reflections of the coordinate
/// system result in the same boxes being selected by the algorithm.
/// The output of this operation is a set of integers indexing into the input
/// collection of bounding boxes representing the selected boxes.  The bounding
/// box coordinates corresponding to the selected indices can then be obtained
/// using the `tf.gather operation`.  For example:
///   selected_indices = tf.image.non_max_suppression_v2(
///       boxes, scores, max_output_size, iou_threshold, score_threshold)
///   selected_boxes = tf.gather(boxes, selected_indices)
///
/// - Parameters:
///   - boxes: A 2-D float tensor of shape `[num_boxes, 4]`.
///   - scores: A 1-D float tensor of shape `[num_boxes]` representing a single
///     score corresponding to each box (each row of boxes).
///   - max_output_size: A scalar integer tensor representing the maximum number of
///     boxes to be selected by non max suppression.
///   - iou_threshold: A 0-D float tensor representing the threshold for deciding whether
///     boxes overlap too much with respect to IOU.
///   - score_threshold: A 0-D float tensor representing the threshold for deciding when to remove
///     boxes based on score.
///
/// - Output selected_indices: A 1-D integer tensor of shape `[M]` representing the selected
///   indices from the boxes tensor, where `M <= max_output_size`.
@inlinable @inline(__always)
public static func nonMaxSuppressionV3<T: FloatingPoint & TensorFlowScalar>(
  boxes: Tensor<T>,
  scores: Tensor<T>,
  maxOutputSize: Tensor<Int32>,
  iouThreshold: Tensor<Float>,
  scoreThreshold: Tensor<Float>
) -> Tensor<Int32> {
  let ret: TensorHandle<Int32> = #tfop("NonMaxSuppressionV3",
    boxes,
    scores,
    maxOutputSize,
    iouThreshold,
    scoreThreshold,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Greedily selects a subset of bounding boxes in descending order of score,
///
/// pruning away boxes that have high intersection-over-union (IOU) overlap
/// with previously selected boxes.  Bounding boxes with score less than
/// `score_threshold` are removed.  Bounding boxes are supplied as
/// [y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any
/// diagonal pair of box corners and the coordinates can be provided as normalized
/// (i.e., lying in the interval [0, 1]) or absolute.  Note that this algorithm
/// is agnostic to where the origin is in the coordinate system and more
/// generally is invariant to orthogonal transformations and translations
/// of the coordinate system; thus translating or reflections of the coordinate
/// system result in the same boxes being selected by the algorithm.
/// The output of this operation is a set of integers indexing into the input
/// collection of bounding boxes representing the selected boxes.  The bounding
/// box coordinates corresponding to the selected indices can then be obtained
/// using the `tf.gather operation`.  For example:
///   selected_indices = tf.image.non_max_suppression_v2(
///       boxes, scores, max_output_size, iou_threshold, score_threshold)
///   selected_boxes = tf.gather(boxes, selected_indices)
///
/// - Parameters:
///   - boxes: A 2-D float tensor of shape `[num_boxes, 4]`.
///   - scores: A 1-D float tensor of shape `[num_boxes]` representing a single
///     score corresponding to each box (each row of boxes).
///   - max_output_size: A scalar integer tensor representing the maximum number of
///     boxes to be selected by non max suppression.
///   - iou_threshold: A 0-D float tensor representing the threshold for deciding whether
///     boxes overlap too much with respect to IOU.
///   - score_threshold: A 0-D float tensor representing the threshold for deciding when to remove
///     boxes based on score.
///
/// - Attr pad_to_max_output_size: If true, the output `selected_indices` is padded to be of length
///   `max_output_size`. Defaults to false.
///
/// - Outputs:
///   - selected_indices: A 1-D integer tensor of shape `[M]` representing the selected
///     indices from the boxes tensor, where `M <= max_output_size`.
///   - valid_outputs: A 0-D integer tensor representing the number of valid elements in
///     `selected_indices`, with the valid elements appearing first.
@inlinable @inline(__always)
public static func nonMaxSuppressionV4<T: FloatingPoint & TensorFlowScalar>(
  boxes: Tensor<T>,
  scores: Tensor<T>,
  maxOutputSize: Tensor<Int32>,
  iouThreshold: Tensor<Float>,
  scoreThreshold: Tensor<Float>,
  padToMaxOutputSize: Bool = false
) -> (selectedIndices: Tensor<Int32>, validOutputs: Tensor<Int32>) {
  let ret: (TensorHandle<Int32>, TensorHandle<Int32>) = #tfop("NonMaxSuppressionV4",
    boxes,
    scores,
    maxOutputSize,
    iouThreshold,
    scoreThreshold,
    T$dtype: T.tensorFlowDataType,
    pad_to_max_output_size: padToMaxOutputSize)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

/// Greedily selects a subset of bounding boxes in descending order of score,
///
/// pruning away boxes that have high overlaps
/// with previously selected boxes.  Bounding boxes with score less than
/// `score_threshold` are removed. N-by-n overlap values are supplied as square matrix,
/// which allows for defining a custom overlap criterium (eg. intersection over union,
/// intersection over area, etc.).
///
/// The output of this operation is a set of integers indexing into the input
/// collection of bounding boxes representing the selected boxes.  The bounding
/// box coordinates corresponding to the selected indices can then be obtained
/// using the `tf.gather operation`.  For example:
///
///   selected_indices = tf.image.non_max_suppression_with_overlaps(
///       overlaps, scores, max_output_size, overlap_threshold, score_threshold)
///   selected_boxes = tf.gather(boxes, selected_indices)
///
/// - Parameters:
///   - overlaps: A 2-D float tensor of shape `[num_boxes, num_boxes]` representing
///     the n-by-n box overlap values.
///   - scores: A 1-D float tensor of shape `[num_boxes]` representing a single
///     score corresponding to each box (each row of boxes).
///   - max_output_size: A scalar integer tensor representing the maximum number of
///     boxes to be selected by non max suppression.
///   - overlap_threshold: A 0-D float tensor representing the threshold for deciding whether
///     boxes overlap too.
///   - score_threshold: A 0-D float tensor representing the threshold for deciding when to remove
///     boxes based on score.
///
/// - Output selected_indices: A 1-D integer tensor of shape `[M]` representing the selected
///   indices from the boxes tensor, where `M <= max_output_size`.
@inlinable @inline(__always)
public static func nonMaxSuppressionWithOverlaps(
  overlaps: Tensor<Float>,
  scores: Tensor<Float>,
  maxOutputSize: Tensor<Int32>,
  overlapThreshold: Tensor<Float>,
  scoreThreshold: Tensor<Float>
) -> Tensor<Int32> {
  let ret: TensorHandle<Int32> = #tfop("NonMaxSuppressionWithOverlaps",
    overlaps,
    scores,
    maxOutputSize,
    overlapThreshold,
    scoreThreshold)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func none(
) {
  return #tfop("None")
}

/// Returns the truth value of (x != y) element-wise.
///
/// *NOTE*: `NotEqual` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@inlinable @inline(__always)
public static func notEqual<T: TensorFlowScalar>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<Bool> {
  let ret: TensorHandle<Bool> = #tfop("NotEqual",
    x,
    y,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Finds values of the `n`-th order statistic for the last dimension.
///
/// If the input is a vector (rank-1), finds the entries which is the nth-smallest
/// value in the vector and outputs their values as scalar tensor.
///
/// For matrices (resp. higher rank input), computes the entries which is the
/// nth-smallest value in each row (resp. vector along the last dimension). Thus,
///
///     values.shape = input.shape[:-1]
///
/// - Parameters:
///   - input: 1-D or higher with last dimension at least `n+1`.
///   - n: 0-D. Position of sorted vector to select along the last dimension (along
///     each row for matrices). Valid range of n is `[0, input.shape[:-1])`
///
/// - Attr reverse: When set to True, find the nth-largest value in the vector and vice
///   versa.
///
/// - Output values: The `n`-th order statistic along each last dimensional slice.
@inlinable @inline(__always)
public static func nthElement<T: Numeric & TensorFlowScalar>(
  _ input: Tensor<T>,
  n: Tensor<Int32>,
  reverse: Bool = false
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("NthElement",
    input,
    n,
    T$dtype: T.tensorFlowDataType,
    reverse: reverse)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func old(
) {
  return #tfop("Old")
}

/// Returns a one-hot tensor.
///
/// The locations represented by indices in `indices` take value `on_value`,
/// while all other locations take value `off_value`.
///
/// If the input `indices` is rank `N`, the output will have rank `N+1`,
/// The new axis is created at dimension `axis` (default: the new axis is
/// appended at the end).
///
/// If `indices` is a scalar the output shape will be a vector of length `depth`.
///
/// If `indices` is a vector of length `features`, the output shape will be:
/// ```
///   features x depth if axis == -1
///   depth x features if axis == 0
/// ```
///
/// If `indices` is a matrix (batch) with shape `[batch, features]`,
/// the output shape will be:
/// ```
///   batch x features x depth if axis == -1
///   batch x depth x features if axis == 1
///   depth x batch x features if axis == 0
/// ```
///
///
/// Examples
/// =========
///
/// Suppose that
/// ```
///   indices = [0, 2, -1, 1]
///   depth = 3
///   on_value = 5.0
///   off_value = 0.0
///   axis = -1
/// ```
///
/// Then output is `[4 x 3]`:
/// ```
/// output =
///   [5.0 0.0 0.0]  // one_hot(0)
///   [0.0 0.0 5.0]  // one_hot(2)
///   [0.0 0.0 0.0]  // one_hot(-1)
///   [0.0 5.0 0.0]  // one_hot(1)
/// ```
///
/// Suppose that
/// ```
///   indices = [0, 2, -1, 1]
///   depth = 3
///   on_value = 0.0
///   off_value = 3.0
///   axis = 0
/// ```
///
/// Then output is `[3 x 4]`:
/// ```
/// output =
///   [0.0 3.0 3.0 3.0]
///   [3.0 3.0 3.0 0.0]
///   [3.0 3.0 3.0 3.0]
///   [3.0 0.0 3.0 3.0]
/// //  ^                one_hot(0)
/// //      ^            one_hot(2)
/// //          ^        one_hot(-1)
/// //              ^    one_hot(1)
/// ```
///
/// Suppose that
/// ```
///   indices = [[0, 2], [1, -1]]
///   depth = 3
///   on_value = 1.0
///   off_value = 0.0
///   axis = -1
/// ```
///
/// Then output is `[2 x 2 x 3]`:
/// ```
/// output =
///   [
///     [1.0, 0.0, 0.0]  // one_hot(0)
///     [0.0, 0.0, 1.0]  // one_hot(2)
///   ][
///     [0.0, 1.0, 0.0]  // one_hot(1)
///     [0.0, 0.0, 0.0]  // one_hot(-1)
///   ]
/// ```
///
/// - Parameters:
///   - indices: A tensor of indices.
///   - depth: A scalar defining the depth of the one hot dimension.
///   - on_value: A scalar defining the value to fill in output when `indices[j] = i`.
///   - off_value: A scalar defining the value to fill in output when `indices[j] != i`.
///
/// - Attr axis: The axis to fill (default: -1, a new inner-most axis).
///
/// - Output output: The one-hot tensor.
@inlinable @inline(__always)
public static func oneHot<T: TensorFlowScalar, Ti: BinaryInteger & TensorFlowScalar>(
  indices: Tensor<Ti>,
  depth: Tensor<Int32>,
  onValue: Tensor<T>,
  offValue: Tensor<T>,
  axis: Int64 = -1
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("OneHot",
    indices,
    depth,
    onValue,
    offValue,
    T$dtype: T.tensorFlowDataType,
    TI$dtype: Ti.tensorFlowDataType,
    axis: axis)
  return Tensor(handle: ret)
}

/// Returns a tensor of ones with the same shape and type as x.
///
/// - Parameter x: a tensor of type T.
///
/// - Output y: a tensor of the same shape and type as x but filled with ones.
@inlinable @inline(__always)
public static func onesLike<T: TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("OnesLike",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func opWithDefaultAttr(
  defaultFloat: Double = 123
) -> Tensor<Int32> {
  let ret: TensorHandle<Int32> = #tfop("OpWithDefaultAttr",
    default_float: defaultFloat)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func opWithFutureDefaultAttr(
) {
  return #tfop("OpWithFutureDefaultAttr")
}

/// Op removes all elements in the underlying container.
@inlinable @inline(__always)
public static func orderedMapClear<Dtypes: TensorFlowScalar>(
  capacity: Int64 = 0,
  memoryLimit: Int64 = 0,
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

/// Op returns the number of incomplete elements in the underlying container.
@inlinable @inline(__always)
public static func orderedMapIncompleteSize<Dtypes: TensorFlowScalar>(
  capacity: Int64 = 0,
  memoryLimit: Int64 = 0,
  container: String,
  sharedName: String,
  typeDtypes: Dtypes.Type
) -> Tensor<Int32> {
  let ret: TensorHandle<Int32> = #tfop("OrderedMapIncompleteSize",
    capacity: capacity,
    memory_limit: memoryLimit,
    container: container,
    shared_name: sharedName)
  return Tensor(handle: ret)
}

/// Op returns the number of elements in the underlying container.
@inlinable @inline(__always)
public static func orderedMapSize<Dtypes: TensorFlowScalar>(
  capacity: Int64 = 0,
  memoryLimit: Int64 = 0,
  container: String,
  sharedName: String,
  typeDtypes: Dtypes.Type
) -> Tensor<Int32> {
  let ret: TensorHandle<Int32> = #tfop("OrderedMapSize",
    capacity: capacity,
    memory_limit: memoryLimit,
    container: container,
    shared_name: sharedName)
  return Tensor(handle: ret)
}

/// Stage (key, values) in the underlying container which behaves like a ordered
///
/// associative container.   Elements are ordered by key.
///
/// - Parameters:
///   - key: int64
///   - values: a list of tensors
///     dtypes A list of data types that inserted values should adhere to.
///
/// - Attrs:
///   - capacity: Maximum number of elements in the Staging Area. If > 0, inserts
///     on the container will block when the capacity is reached.
///   - container: If non-empty, this queue is placed in the given container. Otherwise,
///     a default container is used.
///   - shared_name: It is necessary to match this name to the matching Unstage Op.
@inlinable @inline(__always)
public static func orderedMapStage<Dtypes: TensorFlowScalar, FakeDtypes: TensorFlowScalar>(
  key: Tensor<Int64>,
  indices: Tensor<Int32>,
  _ values: [Tensor<FakeDtypes>],
  capacity: Int64 = 0,
  memoryLimit: Int64 = 0,
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

@inlinable @inline(__always)
public static func outT<T: TensorFlowScalar>(
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("OutT",
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Enqueue a Tensor on the computation outfeed.
///
/// - Parameter input: A tensor that will be inserted into the outfeed queue.
@inlinable @inline(__always)
public static func outfeedEnqueue<Dtype: TensorFlowScalar>(
  _ input: Tensor<Dtype>
) {
  return #tfop("OutfeedEnqueue",
    input,
    dtype$dtype: Dtype.tensorFlowDataType)
}

/// Enqueue multiple Tensor values on the computation outfeed.
///
/// - Parameter inputs: A list of tensors that will be inserted into the outfeed queue as an
///   XLA tuple.
@inlinable @inline(__always)
public static func outfeedEnqueueTuple<Dtypes: TensorFlowScalar>(
  inputs: [Tensor<Dtypes>]
) {
  return #tfop("OutfeedEnqueueTuple",
    inputs)
}

/// Packs a list of `N` rank-`R` tensors into one rank-`(R+1)` tensor.
///
/// Packs the `N` tensors in `values` into a tensor with rank one higher than each
/// tensor in `values`, by packing them along the `axis` dimension.
/// Given a list of tensors of shape `(A, B, C)`;
///
/// if `axis == 0` then the `output` tensor will have the shape `(N, A, B, C)`.
/// if `axis == 1` then the `output` tensor will have the shape `(A, N, B, C)`.
/// Etc.
///
/// For example:
///
/// ```
/// # 'x' is [1, 4]
/// # 'y' is [2, 5]
/// # 'z' is [3, 6]
/// pack([x, y, z]) => [[1, 4], [2, 5], [3, 6]]  # Pack along first dim.
/// pack([x, y, z], axis=1) => [[1, 2, 3], [4, 5, 6]]
/// ```
///
/// This is the opposite of `unpack`.
///
/// - Parameter values: Must be of same shape and type.
///
/// - Attr axis: Dimension along which to pack.  Negative values wrap around, so the
///   valid range is `[-(R+1), R+1)`.
///
/// - Output output: The packed tensor.
@inlinable @inline(__always)
public static func pack<T: TensorFlowScalar>(
  _ values: [Tensor<T>],
  axis: Int64 = 0
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Pack",
    values,
    T$dtype: T.tensorFlowDataType,
    axis: axis)
  return Tensor(handle: ret)
}

/// Pads a tensor with zeros.
///
/// This operation pads a `input` with zeros according to the `paddings` you
/// specify. `paddings` is an integer tensor with shape `[Dn, 2]`, where n is the
/// rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
/// how many zeros to add before the contents of `input` in that dimension, and
/// `paddings[D, 1]` indicates how many zeros to add after the contents of `input`
/// in that dimension.
///
/// The padded size of each dimension D of the output is:
///
/// `paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`
///
/// For example:
///
/// ```
/// # 't' is [[1, 1], [2, 2]]
/// # 'paddings' is [[1, 1], [2, 2]]
/// # rank of 't' is 2
/// pad(t, paddings) ==> [[0, 0, 0, 0, 0, 0]
///                       [0, 0, 1, 1, 0, 0]
///                       [0, 0, 2, 2, 0, 0]
///                       [0, 0, 0, 0, 0, 0]]
/// ```
///
@inlinable @inline(__always)
public static func pad<T: TensorFlowScalar, Tpaddings: BinaryInteger & TensorFlowScalar>(
  _ input: Tensor<T>,
  paddings: Tensor<Tpaddings>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Pad",
    input,
    paddings,
    T$dtype: T.tensorFlowDataType,
    Tpaddings$dtype: Tpaddings.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Pads a tensor.
///
/// This operation pads `input` according to the `paddings` and `constant_values`
/// you specify. `paddings` is an integer tensor with shape `[Dn, 2]`, where n is
/// the rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
/// how many padding values to add before the contents of `input` in that dimension,
/// and `paddings[D, 1]` indicates how many padding values to add after the contents
/// of `input` in that dimension. `constant_values` is a scalar tensor of the same
/// type as `input` that indicates the value to use for padding `input`.
///
/// The padded size of each dimension D of the output is:
///
/// `paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`
///
/// For example:
///
/// ```
/// # 't' is [[1, 1], [2, 2]]
/// # 'paddings' is [[1, 1], [2, 2]]
/// # 'constant_values' is 0
/// # rank of 't' is 2
/// pad(t, paddings) ==> [[0, 0, 0, 0, 0, 0]
///                       [0, 0, 1, 1, 0, 0]
///                       [0, 0, 2, 2, 0, 0]
///                       [0, 0, 0, 0, 0, 0]]
/// ```
@inlinable @inline(__always)
public static func padV2<T: TensorFlowScalar, Tpaddings: BinaryInteger & TensorFlowScalar>(
  _ input: Tensor<T>,
  paddings: Tensor<Tpaddings>,
  constantValues: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("PadV2",
    input,
    paddings,
    constantValues,
    T$dtype: T.tensorFlowDataType,
    Tpaddings$dtype: Tpaddings.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Interleave the values from the `data` tensors into a single tensor.
///
/// Builds a merged tensor such that
///
/// ```python
///     merged[indices[m][i, ..., j], ...] = data[m][i, ..., j, ...]
/// ```
///
/// For example, if each `indices[m]` is scalar or vector, we have
///
/// ```python
///     # Scalar indices:
///     merged[indices[m], ...] = data[m][...]
///
///     # Vector indices:
///     merged[indices[m][i], ...] = data[m][i, ...]
/// ```
///
/// Each `data[i].shape` must start with the corresponding `indices[i].shape`,
/// and the rest of `data[i].shape` must be constant w.r.t. `i`.  That is, we
/// must have `data[i].shape = indices[i].shape + constant`.  In terms of this
/// `constant`, the output shape is
///
///     merged.shape = [max(indices)] + constant
///
/// Values may be merged in parallel, so if an index appears in both `indices[m][i]`
/// and `indices[n][j]`, the result may be invalid. This differs from the normal
/// DynamicStitch operator that defines the behavior in that case.
///
/// For example:
///
/// ```python
///     indices[0] = 6
///     indices[1] = [4, 1]
///     indices[2] = [[5, 2], [0, 3]]
///     data[0] = [61, 62]
///     data[1] = [[41, 42], [11, 12]]
///     data[2] = [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]
///     merged = [[1, 2], [11, 12], [21, 22], [31, 32], [41, 42],
///               [51, 52], [61, 62]]
/// ```
///
/// This method can be used to merge partitions created by `dynamic_partition`
/// as illustrated on the following example:
///
/// ```python
///     # Apply function (increments x_i) on elements for which a certain condition
///     # apply (x_i != -1 in this example).
///     x=tf.constant([0.1, -1., 5.2, 4.3, -1., 7.4])
///     condition_mask=tf.not_equal(x,tf.constant(-1.))
///     partitioned_data = tf.dynamic_partition(
///         x, tf.cast(condition_mask, tf.int32) , 2)
///     partitioned_data[1] = partitioned_data[1] + 1.0
///     condition_indices = tf.dynamic_partition(
///         tf.range(tf.shape(x)[0]), tf.cast(condition_mask, tf.int32) , 2)
///     x = tf.dynamic_stitch(condition_indices, partitioned_data)
///     # Here x=[1.1, -1., 6.2, 5.3, -1, 8.4], the -1. values remain
///     # unchanged.
/// ```
///
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/DynamicStitch.png" alt>
/// </div>
@inlinable @inline(__always)
public static func parallelDynamicStitch<T: TensorFlowScalar>(
  indices: [Tensor<Int32>],
  data: [Tensor<T>]
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("ParallelDynamicStitch",
    indices,
    data,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Outputs random values from a normal distribution. The parameters may each be a
///
/// scalar which applies to the entire output, or a vector of length shape[0] which
/// stores the parameters for each batch.
///
/// - Parameters:
///   - shape: The shape of the output tensor. Batches are indexed by the 0th dimension.
///   - means: The mean parameter of each batch.
///   - stdevs: The standard deviation parameter of each batch. Must be greater than 0.
///   - minvals: The minimum cutoff. May be -infinity.
///   - maxvals: The maximum cutoff. May be +infinity, and must be more than the minval
///     for each batch.
///
/// - Attrs:
///   - seed: If either `seed` or `seed2` are set to be non-zero, the random number
///     generator is seeded by the given seed.  Otherwise, it is seeded by a
///     random seed.
///   - seed2: A second seed to avoid seed collision.
///   - dtype: The type of the output.
///
/// - Output output: A matrix of shape num_batches x samples_per_batch, filled with random
///   truncated normal values using the parameters for each row.
@inlinable @inline(__always)
public static func parameterizedTruncatedNormal<Dtype: FloatingPoint & TensorFlowScalar, T: BinaryInteger & TensorFlowScalar>(
  shape: Tensor<T>,
  means: Tensor<Dtype>,
  stdevs: Tensor<Dtype>,
  minvals: Tensor<Dtype>,
  maxvals: Tensor<Dtype>,
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> Tensor<Dtype> {
  let ret: TensorHandle<Dtype> = #tfop("ParameterizedTruncatedNormal",
    shape,
    means,
    stdevs,
    minvals,
    maxvals,
    dtype$dtype: Dtype.tensorFlowDataType,
    T$dtype: T.tensorFlowDataType,
    seed: seed,
    seed2: seed2)
  return Tensor(handle: ret)
}

/// Transforms a serialized tensorflow.TensorProto proto into a Tensor.
///
/// - Parameter serialized: A scalar string containing a serialized TensorProto proto.
///
/// - Attr out_type: The type of the serialized tensor.  The provided type must match the
///   type of the serialized tensor and no implicit conversion will take place.
///
/// - Output output: A Tensor of type `out_type`.
@inlinable @inline(__always)
public static func parseTensor<OutType: TensorFlowScalar>(
  serialized: StringTensor
) -> Tensor<OutType> {
  let ret: TensorHandle<OutType> = #tfop("ParseTensor",
    serialized,
    out_type$dtype: OutType.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Compute the polygamma function \\(\psi^{(n)}(x)\\).
///
/// The polygamma function is defined as:
///
///
/// \\(\psi^{(a)}(x) = \frac{d^a}{dx^a} \psi(x)\\)
///
/// where \\(\psi(x)\\) is the digamma function.
/// The polygamma function is defined only for non-negative integer orders \\a\\.
@inlinable @inline(__always)
public static func polygamma<T: FloatingPoint & TensorFlowScalar>(
  _ a: Tensor<T>,
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Polygamma",
    a,
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func polymorphic<T: TensorFlowScalar>(
  _ a: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Polymorphic",
    a,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func polymorphicDefaultOut<T: TensorFlowScalar>(
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("PolymorphicDefaultOut",
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func polymorphicOut<T: TensorFlowScalar>(
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("PolymorphicOut",
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes element-wise population count (a.k.a. popcount, bitsum, bitcount).
///
/// For each entry in `x`, calculates the number of `1` (on) bits in the binary
/// representation of that entry.
///
/// **NOTE**: It is more efficient to first `tf.bitcast` your tensors into
/// `int32` or `int64` and perform the bitcount on the result, than to feed in
/// 8- or 16-bit inputs and then aggregate the resulting counts.
@inlinable @inline(__always)
public static func populationCount<T: BinaryInteger & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<UInt8> {
  let ret: TensorHandle<UInt8> = #tfop("PopulationCount",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the power of one value to another.
///
/// Given a tensor `x` and a tensor `y`, this operation computes \\(x^y\\) for
/// corresponding elements in `x` and `y`. For example:
///
/// ```
/// # tensor 'x' is [[2, 2]], [3, 3]]
/// # tensor 'y' is [[8, 16], [2, 3]]
/// tf.pow(x, y) ==> [[256, 65536], [9, 27]]
/// ```
@inlinable @inline(__always)
public static func pow<T: Numeric & TensorFlowScalar>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Pow",
    x,
    y,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// An identity op that triggers an error if a gradient is requested.
///
/// When executed in a graph, this op outputs its input tensor as-is.
///
/// When building ops to compute gradients, the TensorFlow gradient system
/// will return an error when trying to lookup the gradient of this op,
/// because no gradient must ever be registered for this function.  This
/// op exists to prevent subtle bugs from silently returning unimplemented
/// gradients in some corner cases.
///
/// - Parameter input: any tensor.
///
/// - Attr message: Will be printed in the error when anyone tries to differentiate
///   this operation.
///
/// - Output output: the same input tensor.
@inlinable @inline(__always)
public static func preventGradient<T: TensorFlowScalar>(
  _ input: Tensor<T>,
  message: String
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("PreventGradient",
    input,
    T$dtype: T.tensorFlowDataType,
    message: message)
  return Tensor(handle: ret)
}

/// Prints a list of tensors.
///
/// Passes `input` through to `output` and prints `data` when evaluating.
///
/// - Parameters:
///   - input: The tensor passed to `output`
///   - data: A list of tensors to print out when op is evaluated.
///
/// - Attrs:
///   - message: A string, prefix of the error message.
///   - first_n: Only log `first_n` number of times. -1 disables logging.
///   - summarize: Only print this many entries of each tensor.
///
/// - Output output: = The unmodified `input` tensor
@inlinable @inline(__always)
public static func print<T: TensorFlowScalar, U: TensorFlowScalar>(
  _ input: Tensor<T>,
  data: [Tensor<U>],
  message: String,
  firstN: Int64 = -1,
  summarize: Int64 = 3
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Print",
    input,
    data,
    T$dtype: T.tensorFlowDataType,
    message: message,
    first_n: firstN,
    summarize: summarize)
  return Tensor(handle: ret)
}

/// Prints a string scalar.
///
/// Prints a string scalar to the desired output_stream.
///
/// - Parameter input: The string scalar to print.
///
/// - Attr output_stream: A string specifying the output stream or logging level to print to.
@inlinable @inline(__always)
public static func printV2(
  _ input: StringTensor,
  outputStream: String = "stderr"
) {
  return #tfop("PrintV2",
    input,
    output_stream: outputStream)
}

/// Computes the product of elements across dimensions of a tensor.
///
/// Reduces `input` along the dimensions given in `axis`. Unless
/// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
/// `axis`. If `keep_dims` is true, the reduced dimensions are
/// retained with length 1.
///
/// - Parameters:
///   - input: The tensor to reduce.
///   - reduction_indices: The dimensions to reduce. Must be in the range
///     `[-rank(input), rank(input))`.
///
/// - Attr keep_dims: If true, retain reduced dimensions with length 1.
///
/// - Output output: The reduced tensor.
@inlinable @inline(__always)
public static func prod<T: Numeric & TensorFlowScalar, Tidx: BinaryInteger & TensorFlowScalar>(
  _ input: Tensor<T>,
  reductionIndices: Tensor<Tidx>,
  keepDims: Bool = false
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Prod",
    input,
    reductionIndices,
    T$dtype: T.tensorFlowDataType,
    Tidx$dtype: Tidx.tensorFlowDataType,
    keep_dims: keepDims)
  return Tensor(handle: ret)
}

/// Computes the QR decompositions of one or more matrices.
///
/// Computes the QR decomposition of each inner matrix in `tensor` such that
/// `tensor[..., :, :] = q[..., :, :] * r[..., :,:])`
///
/// ```python
/// # a is a tensor.
/// # q is a tensor of orthonormal matrices.
/// # r is a tensor of upper triangular matrices.
/// q, r = qr(a)
/// q_full, r_full = qr(a, full_matrices=True)
/// ```
///
/// - Parameter input: A tensor of shape `[..., M, N]` whose inner-most 2 dimensions
///   form matrices of size `[M, N]`. Let `P` be the minimum of `M` and `N`.
///
/// - Attr full_matrices: If true, compute full-sized `q` and `r`. If false
///   (the default), compute only the leading `P` columns of `q`.
///
/// - Outputs:
///   - q: Orthonormal basis for range of `a`. If `full_matrices` is `False` then
///     shape is `[..., M, P]`; if `full_matrices` is `True` then shape is
///     `[..., M, M]`.
///   - r: Triangular factor. If `full_matrices` is `False` then shape is
///     `[..., P, N]`. If `full_matrices` is `True` then shape is `[..., M, N]`.
@inlinable @inline(__always)
public static func qr<T: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>,
  fullMatrices: Bool = false
) -> (q: Tensor<T>, r: Tensor<T>) {
  let ret: (TensorHandle<T>, TensorHandle<T>) = #tfop("Qr",
    input,
    T$dtype: T.tensorFlowDataType,
    full_matrices: fullMatrices)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

/// Use QuantizeAndDequantizeV2 instead.
@inlinable @inline(__always)
public static func quantizeAndDequantize<T: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>,
  signedInput: Bool = true,
  numBits: Int64 = 8,
  rangeGiven: Bool = false,
  inputMin: Double = 0,
  inputMax: Double = 0
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("QuantizeAndDequantize",
    input,
    T$dtype: T.tensorFlowDataType,
    signed_input: signedInput,
    num_bits: numBits,
    range_given: rangeGiven,
    input_min: inputMin,
    input_max: inputMax)
  return Tensor(handle: ret)
}

/// Quantizes then dequantizes a tensor.
///
/// This op simulates the precision loss from the quantized forward pass by:
///
/// 1. Quantizing the tensor to fixed point numbers, which should match the target
///    quantization method when it is used in inference.
/// 2. Dequantizing it back to floating point numbers for the following ops, most
///    likely matmul.
///
/// There are different ways to quantize. This version uses only scaling, so 0.0
/// maps to 0.
///
/// From the specified 'num_bits' in the quantized output type, it determines
/// minimum and maximum representable quantized values.
///
/// e.g.
///
/// *   [-128, 127] for signed, num_bits = 8, or
/// *   [0, 255] for unsigned, num_bits = 8.
///
/// If range_given == False, the initial input_min, input_max will be determined
/// automatically as the minimum and maximum values in the input tensor, otherwise
/// the specified values of input_min, input_max are used.
///
/// Note: If the input_min, input_max are specified, they do not need to equal the
/// actual minimum and maximum values in the tensor. e.g. in some cases it may be
/// beneficial to specify these values such that the low probability extremes of the
/// input distribution are clipped.
///
/// This op determines the maximum scale_factor that would map the initial
/// [input_min, input_max] range to a range that lies within the representable
/// quantized range.
///
/// It determines the scale from one of input_min and input_max, then updates the
/// other one to maximize the respresentable range.
///
/// e.g.
///
/// *   if the output is signed, num_bits = 8, [input_min, input_max] = [-10.0,
///     5.0]: it would use a scale_factor of -128 / -10.0 = 12.8 In this case, it
///     would update input_max to be 127 / 12.8 = 9.921875
/// *   if the output is signed, num_bits = 8, [input_min, input_max] = [-10.0,
///     10.0]: it would use a scale_factor of 127 / 10.0 = 12.7 In this case, it
///     would update input_min to be 128.0 / 12.7 = -10.07874
/// *   if the output is unsigned, input_min is forced to be 0, and only the
///     specified input_max is used.
///
/// After determining the scale_factor and updating the input range, it applies the
/// following to each value in the 'input' tensor.
///
/// output = round(clamp(value, input_min, input_max) * scale_factor) / scale_factor.
///
/// The above round function rounds the value based on the given round_mode.
///
///
/// - Parameters:
///   - input: Tensor to quantize and then dequantize.
///   - input_min: If `range_given == True`, this specifies the minimum input value that needs to
///     be represented, otherwise it is determined from the min value of the `input`
///     tensor.
///   - input_max: If `range_given == True`, this specifies the maximum input value that needs to
///     be represented, otherwise it is determined from the max value of the `input`
///     tensor.
///
/// - Attrs:
///   - signed_input: Whether the quantization is signed or unsigned. (actually this parameter should
///     have been called <b>`signed_output`</b>)
///   - num_bits: The bitwidth of the quantization.
///   - range_given: Whether the range is given or should be determined from the `input` tensor.
///   - round_mode: The 'round_mode' attribute controls which rounding tie-breaking algorithm is
///     used when rounding float values to their quantized equivalents. The following
///     rounding modes are currently supported:
///
///     *   HALF_TO_EVEN: this is the default round_mode.
///     *   HALF_UP: round towards positive. In this mode 7.5 rounds up to 8 and -7.5
///         rounds up to -7.
///
@inlinable @inline(__always)
public static func quantizeAndDequantizeV2<T: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>,
  inputMin: Tensor<T>,
  inputMax: Tensor<T>,
  signedInput: Bool = true,
  numBits: Int64 = 8,
  rangeGiven: Bool = false,
  roundMode: RoundMode = .halfToEven
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("QuantizeAndDequantizeV2",
    input,
    inputMin,
    inputMax,
    T$dtype: T.tensorFlowDataType,
    signed_input: signedInput,
    num_bits: numBits,
    range_given: rangeGiven,
    round_mode: roundMode.cName)
  return Tensor(handle: ret)
}

/// Quantizes then dequantizes a tensor.
///
/// This is almost identical to QuantizeAndDequantizeV2, except that num_bits is a
/// tensor, so its value can change during training.
@inlinable @inline(__always)
public static func quantizeAndDequantizeV3<T: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>,
  inputMin: Tensor<T>,
  inputMax: Tensor<T>,
  numBits: Tensor<Int32>,
  signedInput: Bool = true,
  rangeGiven: Bool = true
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("QuantizeAndDequantizeV3",
    input,
    inputMin,
    inputMax,
    numBits,
    T$dtype: T.tensorFlowDataType,
    signed_input: signedInput,
    range_given: rangeGiven)
  return Tensor(handle: ret)
}

/// Convert the quantized 'input' tensor into a lower-precision 'output', using the
///
/// actual distribution of the values to maximize the usage of the lower bit depth
/// and adjusting the output min and max ranges accordingly.
///
/// [input_min, input_max] are scalar floats that specify the range for the float
/// interpretation of the 'input' data. For example, if input_min is -1.0f and
/// input_max is 1.0f, and we are dealing with quint16 quantized data, then a 0
/// value in the 16-bit data should be interpreted as -1.0f, and a 65535 means 1.0f.
///
/// This operator tries to squeeze as much precision as possible into an output with
/// a lower bit depth by calculating the actual min and max values found in the
/// data. For example, maybe that quint16 input has no values lower than 16,384 and
/// none higher than 49,152. That means only half the range is actually needed, all
/// the float interpretations are between -0.5f and 0.5f, so if we want to compress
/// the data into a quint8 output, we can use that range rather than the theoretical
/// -1.0f to 1.0f that is suggested by the input min and max.
///
/// In practice, this is most useful for taking output from operations like
/// QuantizedMatMul that can produce higher bit-depth outputs than their inputs and
/// may have large potential output ranges, but in practice have a distribution of
/// input values that only uses a small fraction of the possible range. By feeding
/// that output into this operator, we can reduce it from 32 bits down to 8 with
/// minimal loss of accuracy.
///
/// - Parameters:
///   - input_min: The float value that the minimum quantized input value represents.
///   - input_max: The float value that the maximum quantized input value represents.
///
/// - Attrs:
///   - Tinput: The type of the input.
///   - out_type: The type of the output. Should be a lower bit depth than Tinput.
///
/// - Outputs:
///   - output_min: The float value that the minimum quantized output value represents.
///   - output_max: The float value that the maximum quantized output value represents.
@inlinable @inline(__always)
public static func quantizeDownAndShrinkRange<Tinput: TensorFlowScalar, OutType: TensorFlowScalar>(
  _ input: Tensor<Tinput>,
  inputMin: Tensor<Float>,
  inputMax: Tensor<Float>
) -> (output: Tensor<OutType>, outputMin: Tensor<Float>, outputMax: Tensor<Float>) {
  let ret: (TensorHandle<OutType>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("QuantizeDownAndShrinkRange",
    input,
    inputMin,
    inputMax,
    Tinput$dtype: Tinput.tensorFlowDataType,
    out_type$dtype: OutType.tensorFlowDataType)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Quantize the 'input' tensor of type float to 'output' tensor of type 'T'.
///
/// [min_range, max_range] are scalar floats that specify the range for
/// the 'input' data. The 'mode' attribute controls exactly which calculations are
/// used to convert the float values to their quantized equivalents.  The
/// 'round_mode' attribute controls which rounding tie-breaking algorithm is used
/// when rounding float values to their quantized equivalents.
///
/// In 'MIN_COMBINED' mode, each value of the tensor will undergo the following:
///
/// ```
/// out[i] = (in[i] - min_range) * range(T) / (max_range - min_range)
/// if T == qint8: out[i] -= (range(T) + 1) / 2.0
/// ```
///
/// here `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`
///
/// *MIN_COMBINED Mode Example*
///
/// Assume the input is type float and has a possible range of [0.0, 6.0] and the
/// output type is quint8 ([0, 255]). The min_range and max_range values should be
/// specified as 0.0 and 6.0. Quantizing from float to quint8 will multiply each
/// value of the input by 255/6 and cast to quint8.
///
/// If the output type was qint8 ([-128, 127]), the operation will additionally
/// subtract each value by 128 prior to casting, so that the range of values aligns
/// with the range of qint8.
///
/// If the mode is 'MIN_FIRST', then this approach is used:
///
/// ```
/// num_discrete_values = 1 << (# of bits in T)
/// range_adjust = num_discrete_values / (num_discrete_values - 1)
/// range = (range_max - range_min) * range_adjust
/// range_scale = num_discrete_values / range
/// quantized = round(input * range_scale) - round(range_min * range_scale) +
///   numeric_limits<T>::min()
/// quantized = max(quantized, numeric_limits<T>::min())
/// quantized = min(quantized, numeric_limits<T>::max())
/// ```
///
/// The biggest difference between this and MIN_COMBINED is that the minimum range
/// is rounded first, before it's subtracted from the rounded value. With
/// MIN_COMBINED, a small bias is introduced where repeated iterations of quantizing
/// and dequantizing will introduce a larger and larger error.
///
/// *SCALED mode Example*
///
/// `SCALED` mode matches the quantization approach used in
/// `QuantizeAndDequantize{V2|V3}`.
///
/// If the mode is `SCALED`, we do not use the full range of the output type,
/// choosing to elide the lowest possible value for symmetry (e.g., output range is
/// -127 to 127, not -128 to 127 for signed 8 bit quantization), so that 0.0 maps to
/// 0.
///
/// We first find the range of values in our tensor. The
/// range we use is always centered on 0, so we find m such that
///
/// ```c++
///   m = max(abs(input_min), abs(input_max))
/// ```
///
/// Our input tensor range is then `[-m, m]`.
///
/// Next, we choose our fixed-point quantization buckets, `[min_fixed, max_fixed]`.
/// If T is signed, this is
///
/// ```
///   num_bits = sizeof(T) * 8
///   [min_fixed, max_fixed] =
///       [-(1 << (num_bits - 1) - 1), (1 << (num_bits - 1)) - 1]
/// ```
///
/// Otherwise, if T is unsigned, the fixed-point range is
///
/// ```
///   [min_fixed, max_fixed] = [0, (1 << num_bits) - 1]
/// ```
///
/// From this we compute our scaling factor, s:
///
/// ```c++
///   s = (max_fixed - min_fixed) / (2 * m)
/// ```
///
/// Now we can quantize the elements of our tensor:
///
/// ```c++
/// result = round(input * s)
/// ```
///
/// One thing to watch out for is that the operator may choose to adjust the
/// requested minimum and maximum values slightly during the quantization process,
/// so you should always use the output ports as the range for further calculations.
/// For example, if the requested minimum and maximum values are close to equal,
/// they will be separated by a small epsilon value to prevent ill-formed quantized
/// buffers from being created. Otherwise, you can end up with buffers where all the
/// quantized values map to the same float value, which causes problems for
/// operations that have to perform further calculations on them.
///
/// - Parameters:
///   - min_range: The minimum scalar value possibly produced for the input.
///   - max_range: The maximum scalar value possibly produced for the input.
///
/// - Outputs:
///   - output: The quantized data produced from the float input.
///   - output_min: The actual minimum scalar value used for the output.
///   - output_max: The actual maximum scalar value used for the output.
@inlinable @inline(__always)
public static func quantizeV2<T: TensorFlowScalar>(
  _ input: Tensor<Float>,
  minRange: Tensor<Float>,
  maxRange: Tensor<Float>,
  mode: Mode = .minCombined,
  roundMode: RoundMode6 = .halfAwayFromZero
) -> (output: Tensor<T>, outputMin: Tensor<Float>, outputMax: Tensor<Float>) {
  let ret: (TensorHandle<T>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("QuantizeV2",
    input,
    minRange,
    maxRange,
    T$dtype: T.tensorFlowDataType,
    mode: mode.cName,
    round_mode: roundMode.cName)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Returns x + y element-wise, working on quantized buffers.
///
/// - Parameters:
///   - min_x: The float value that the lowest quantized `x` value represents.
///   - max_x: The float value that the highest quantized `x` value represents.
///   - min_y: The float value that the lowest quantized `y` value represents.
///   - max_y: The float value that the highest quantized `y` value represents.
///
/// - Outputs:
///   - min_z: The float value that the lowest quantized output value represents.
///   - max_z: The float value that the highest quantized output value represents.
///
///     *NOTE*: `QuantizedAdd` supports limited forms of broadcasting. More about
///     broadcasting [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@inlinable @inline(__always)
public static func quantizedAdd<T1: TensorFlowScalar, T2: TensorFlowScalar, Toutput: TensorFlowScalar>(
  _ x: Tensor<T1>,
  _ y: Tensor<T2>,
  minX: Tensor<Float>,
  maxX: Tensor<Float>,
  minY: Tensor<Float>,
  maxY: Tensor<Float>
) -> (z: Tensor<Toutput>, minZ: Tensor<Float>, maxZ: Tensor<Float>) {
  let ret: (TensorHandle<Toutput>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("QuantizedAdd",
    x,
    y,
    minX,
    maxX,
    minY,
    maxY,
    T1$dtype: T1.tensorFlowDataType,
    T2$dtype: T2.tensorFlowDataType,
    Toutput$dtype: Toutput.tensorFlowDataType)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Produces the average pool of the input tensor for quantized types.
///
/// - Parameters:
///   - input: 4-D with shape `[batch, height, width, channels]`.
///   - min_input: The float value that the lowest quantized input value represents.
///   - max_input: The float value that the highest quantized input value represents.
///
/// - Attrs:
///   - ksize: The size of the window for each dimension of the input tensor.
///     The length must be 4 to match the number of dimensions of the input.
///   - strides: The stride of the sliding window for each dimension of the input
///     tensor.  The length must be 4 to match the number of dimensions of the input.
///   - padding: The type of padding algorithm to use.
///
/// - Outputs:
///   - min_output: The float value that the lowest quantized output value represents.
///   - max_output: The float value that the highest quantized output value represents.
@inlinable @inline(__always)
public static func quantizedAvgPool<T: TensorFlowScalar>(
  _ input: Tensor<T>,
  minInput: Tensor<Float>,
  maxInput: Tensor<Float>,
  ksize: [Int32],
  strides: [Int32],
  padding: Padding
) -> (output: Tensor<T>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>) {
  let ret: (TensorHandle<T>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("QuantizedAvgPool",
    input,
    minInput,
    maxInput,
    T$dtype: T.tensorFlowDataType,
    ksize: ksize,
    strides: strides,
    padding: padding.cName)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Quantized Batch normalization.
///
/// This op is deprecated and will be removed in the future. Prefer
/// `tf.nn.batch_normalization`.
///
/// - Parameters:
///   - t: A 4D input Tensor.
///   - t_min: The value represented by the lowest quantized input.
///   - t_max: The value represented by the highest quantized input.
///   - m: A 1D mean Tensor with size matching the last dimension of t.
///     This is the first output from tf.nn.moments,
///     or a saved moving average thereof.
///   - m_min: The value represented by the lowest quantized mean.
///   - m_max: The value represented by the highest quantized mean.
///   - v: A 1D variance Tensor with size matching the last dimension of t.
///     This is the second output from tf.nn.moments,
///     or a saved moving average thereof.
///   - v_min: The value represented by the lowest quantized variance.
///   - v_max: The value represented by the highest quantized variance.
///   - beta: A 1D beta Tensor with size matching the last dimension of t.
///     An offset to be added to the normalized tensor.
///   - beta_min: The value represented by the lowest quantized offset.
///   - beta_max: The value represented by the highest quantized offset.
///   - gamma: A 1D gamma Tensor with size matching the last dimension of t.
///     If "scale_after_normalization" is true, this tensor will be multiplied
///     with the normalized tensor.
///   - gamma_min: The value represented by the lowest quantized gamma.
///   - gamma_max: The value represented by the highest quantized gamma.
///
/// - Attrs:
///   - variance_epsilon: A small float number to avoid dividing by 0.
///   - scale_after_normalization: A bool indicating whether the resulted tensor
///     needs to be multiplied with gamma.
@inlinable @inline(__always)
public static func quantizedBatchNormWithGlobalNormalization<Tinput: TensorFlowScalar, OutType: TensorFlowScalar>(
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
) -> (result: Tensor<OutType>, resultMin: Tensor<Float>, resultMax: Tensor<Float>) {
  let ret: (TensorHandle<OutType>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("QuantizedBatchNormWithGlobalNormalization",
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
    Tinput$dtype: Tinput.tensorFlowDataType,
    out_type$dtype: OutType.tensorFlowDataType,
    variance_epsilon: varianceEpsilon,
    scale_after_normalization: scaleAfterNormalization)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Adds Tensor 'bias' to Tensor 'input' for Quantized types.
///
/// Broadcasts the values of bias on dimensions 0..N-2 of 'input'.
///
/// - Parameters:
///   - bias: A 1D bias Tensor with size matching the last dimension of 'input'.
///   - min_input: The float value that the lowest quantized input value represents.
///   - max_input: The float value that the highest quantized input value represents.
///   - min_bias: The float value that the lowest quantized bias value represents.
///   - max_bias: The float value that the highest quantized bias value represents.
///
/// - Outputs:
///   - min_out: The float value that the lowest quantized output value represents.
///   - max_out: The float value that the highest quantized output value represents.
@inlinable @inline(__always)
public static func quantizedBiasAdd<T1: TensorFlowScalar, T2: TensorFlowScalar, OutType: TensorFlowScalar>(
  _ input: Tensor<T1>,
  bias: Tensor<T2>,
  minInput: Tensor<Float>,
  maxInput: Tensor<Float>,
  minBias: Tensor<Float>,
  maxBias: Tensor<Float>
) -> (output: Tensor<OutType>, minOut: Tensor<Float>, maxOut: Tensor<Float>) {
  let ret: (TensorHandle<OutType>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("QuantizedBiasAdd",
    input,
    bias,
    minInput,
    maxInput,
    minBias,
    maxBias,
    T1$dtype: T1.tensorFlowDataType,
    T2$dtype: T2.tensorFlowDataType,
    out_type$dtype: OutType.tensorFlowDataType)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Concatenates quantized tensors along one dimension.
///
/// - Parameters:
///   - concat_dim: 0-D.  The dimension along which to concatenate.  Must be in the
///     range [0, rank(values)).
///   - values: The `N` Tensors to concatenate. Their ranks and types must match,
///     and their sizes must match in all dimensions except `concat_dim`.
///   - input_mins: The minimum scalar values for each of the input tensors.
///   - input_maxes: The maximum scalar values for each of the input tensors.
///
/// - Outputs:
///   - output: A `Tensor` with the concatenation of values stacked along the
///     `concat_dim` dimension.  This tensor's shape matches that of `values` except
///     in `concat_dim` where it has the sum of the sizes.
///   - output_min: The float value that the minimum quantized output value represents.
///   - output_max: The float value that the maximum quantized output value represents.
@inlinable @inline(__always)
public static func quantizedConcat<T: TensorFlowScalar>(
  concatDim: Tensor<Int32>,
  _ values: [Tensor<T>],
  inputMins: [Tensor<Float>],
  inputMaxes: [Tensor<Float>]
) -> (output: Tensor<T>, outputMin: Tensor<Float>, outputMax: Tensor<Float>) {
  let ret: (TensorHandle<T>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("QuantizedConcat",
    concatDim,
    values,
    inputMins,
    inputMaxes,
    T$dtype: T.tensorFlowDataType)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Computes a 2D convolution given quantized 4D input and filter tensors.
///
/// The inputs are quantized tensors where the lowest value represents the real
/// number of the associated minimum, and the highest represents the maximum.
/// This means that you can only interpret the quantized output in the same way, by
/// taking the returned minimum and maximum values into account.
///
/// - Parameters:
///   - filter: filter's input_depth dimension must match input's depth dimensions.
///   - min_input: The float value that the lowest quantized input value represents.
///   - max_input: The float value that the highest quantized input value represents.
///   - min_filter: The float value that the lowest quantized filter value represents.
///   - max_filter: The float value that the highest quantized filter value represents.
///
/// - Attrs:
///   - strides: The stride of the sliding window for each dimension of the input
///     tensor.
///   - padding: The type of padding algorithm to use.
///   - dilations: 1-D tensor of length 4.  The dilation factor for each dimension of
///     `input`. If set to k > 1, there will be k-1 skipped cells between each
///     filter element on that dimension. The dimension order is determined by the
///     value of `data_format`, see above for details. Dilations in the batch and
///     depth dimensions must be 1.
///
/// - Outputs:
///   - min_output: The float value that the lowest quantized output value represents.
///   - max_output: The float value that the highest quantized output value represents.
@inlinable @inline(__always)
public static func quantizedConv2D<Tinput: TensorFlowScalar, Tfilter: TensorFlowScalar, OutType: TensorFlowScalar>(
  _ input: Tensor<Tinput>,
  filter: Tensor<Tfilter>,
  minInput: Tensor<Float>,
  maxInput: Tensor<Float>,
  minFilter: Tensor<Float>,
  maxFilter: Tensor<Float>,
  strides: [Int32],
  padding: Padding,
  dilations: [Int32] = [1, 1, 1, 1]
) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>) {
  let ret: (TensorHandle<OutType>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("QuantizedConv2D",
    input,
    filter,
    minInput,
    maxInput,
    minFilter,
    maxFilter,
    Tinput$dtype: Tinput.tensorFlowDataType,
    Tfilter$dtype: Tfilter.tensorFlowDataType,
    out_type$dtype: OutType.tensorFlowDataType,
    strides: strides,
    padding: padding.cName,
    dilations: dilations)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

@inlinable @inline(__always)
public static func quantizedConv2DAndRelu<Tinput: TensorFlowScalar, Tfilter: TensorFlowScalar, OutType: TensorFlowScalar>(
  _ input: Tensor<Tinput>,
  filter: Tensor<Tfilter>,
  minInput: Tensor<Float>,
  maxInput: Tensor<Float>,
  minFilter: Tensor<Float>,
  maxFilter: Tensor<Float>,
  strides: [Int32],
  padding: Padding,
  dilations: [Int32] = [1, 1, 1, 1],
  paddingList: [Int32]
) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>) {
  let ret: (TensorHandle<OutType>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("QuantizedConv2DAndRelu",
    input,
    filter,
    minInput,
    maxInput,
    minFilter,
    maxFilter,
    Tinput$dtype: Tinput.tensorFlowDataType,
    Tfilter$dtype: Tfilter.tensorFlowDataType,
    out_type$dtype: OutType.tensorFlowDataType,
    strides: strides,
    padding: padding.cName,
    dilations: dilations,
    padding_list: paddingList)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

@inlinable @inline(__always)
public static func quantizedConv2DAndReluAndRequantize<Tinput: TensorFlowScalar, Tfilter: TensorFlowScalar, OutType: TensorFlowScalar>(
  _ input: Tensor<Tinput>,
  filter: Tensor<Tfilter>,
  minInput: Tensor<Float>,
  maxInput: Tensor<Float>,
  minFilter: Tensor<Float>,
  maxFilter: Tensor<Float>,
  minFreezedOutput: Tensor<Float>,
  maxFreezedOutput: Tensor<Float>,
  strides: [Int32],
  padding: Padding,
  dilations: [Int32] = [1, 1, 1, 1],
  paddingList: [Int32]
) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>) {
  let ret: (TensorHandle<OutType>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("QuantizedConv2DAndReluAndRequantize",
    input,
    filter,
    minInput,
    maxInput,
    minFilter,
    maxFilter,
    minFreezedOutput,
    maxFreezedOutput,
    Tinput$dtype: Tinput.tensorFlowDataType,
    Tfilter$dtype: Tfilter.tensorFlowDataType,
    out_type$dtype: OutType.tensorFlowDataType,
    strides: strides,
    padding: padding.cName,
    dilations: dilations,
    padding_list: paddingList)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

@inlinable @inline(__always)
public static func quantizedConv2DAndRequantize<Tinput: TensorFlowScalar, Tfilter: TensorFlowScalar, OutType: TensorFlowScalar>(
  _ input: Tensor<Tinput>,
  filter: Tensor<Tfilter>,
  minInput: Tensor<Float>,
  maxInput: Tensor<Float>,
  minFilter: Tensor<Float>,
  maxFilter: Tensor<Float>,
  minFreezedOutput: Tensor<Float>,
  maxFreezedOutput: Tensor<Float>,
  strides: [Int32],
  padding: Padding,
  dilations: [Int32] = [1, 1, 1, 1],
  paddingList: [Int32]
) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>) {
  let ret: (TensorHandle<OutType>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("QuantizedConv2DAndRequantize",
    input,
    filter,
    minInput,
    maxInput,
    minFilter,
    maxFilter,
    minFreezedOutput,
    maxFreezedOutput,
    Tinput$dtype: Tinput.tensorFlowDataType,
    Tfilter$dtype: Tfilter.tensorFlowDataType,
    out_type$dtype: OutType.tensorFlowDataType,
    strides: strides,
    padding: padding.cName,
    dilations: dilations,
    padding_list: paddingList)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Computes QuantizedConv2D per channel.
///
/// - Parameters:
///   - input: The original input tensor.
///   - filter: The original filter tensor.
///   - min_input: The minimum value of the input tensor
///   - max_input: The maximum value of the input tensor.
///   - min_filter: The minimum value of the filter tensor.
///   - max_filter: The maximum value of the filter tensor.
///
/// - Attrs:
///   - Tinput: The quantized type of input tensor that needs to be converted. 
///   - Tfilter: The quantized type of filter tensor that needs to be converted. 
///   - out_type: The quantized type of output tensor that needs to be converted.
///   - strides: list of stride values.
///   - dilations: list of dilation values.
///
/// - Outputs:
///   - output: The output tensor.
///   - min_output: The minimum value of the final output tensor.
///   - max_output: The maximum value of the final output tensor.
@inlinable @inline(__always)
public static func quantizedConv2DPerChannel<Tinput: TensorFlowScalar, Tfilter: TensorFlowScalar, OutType: TensorFlowScalar>(
  _ input: Tensor<Tinput>,
  filter: Tensor<Tfilter>,
  minInput: Tensor<Float>,
  maxInput: Tensor<Float>,
  minFilter: Tensor<Float>,
  maxFilter: Tensor<Float>,
  strides: [Int32],
  padding: Padding,
  dilations: [Int32] = [1, 1, 1, 1]
) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>) {
  let ret: (TensorHandle<OutType>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("QuantizedConv2DPerChannel",
    input,
    filter,
    minInput,
    maxInput,
    minFilter,
    maxFilter,
    Tinput$dtype: Tinput.tensorFlowDataType,
    Tfilter$dtype: Tfilter.tensorFlowDataType,
    out_type$dtype: OutType.tensorFlowDataType,
    strides: strides,
    padding: padding.cName,
    dilations: dilations)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

@inlinable @inline(__always)
public static func quantizedConv2DWithBias<Tinput: TensorFlowScalar, Tfilter: TensorFlowScalar, OutType: TensorFlowScalar>(
  _ input: Tensor<Tinput>,
  filter: Tensor<Tfilter>,
  bias: Tensor<Float>,
  minInput: Tensor<Float>,
  maxInput: Tensor<Float>,
  minFilter: Tensor<Float>,
  maxFilter: Tensor<Float>,
  strides: [Int32],
  padding: Padding,
  dilations: [Int32] = [1, 1, 1, 1],
  paddingList: [Int32]
) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>) {
  let ret: (TensorHandle<OutType>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("QuantizedConv2DWithBias",
    input,
    filter,
    bias,
    minInput,
    maxInput,
    minFilter,
    maxFilter,
    Tinput$dtype: Tinput.tensorFlowDataType,
    Tfilter$dtype: Tfilter.tensorFlowDataType,
    out_type$dtype: OutType.tensorFlowDataType,
    strides: strides,
    padding: padding.cName,
    dilations: dilations,
    padding_list: paddingList)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

@inlinable @inline(__always)
public static func quantizedConv2DWithBiasAndRelu<Tinput: TensorFlowScalar, Tfilter: TensorFlowScalar, OutType: TensorFlowScalar>(
  _ input: Tensor<Tinput>,
  filter: Tensor<Tfilter>,
  bias: Tensor<Float>,
  minInput: Tensor<Float>,
  maxInput: Tensor<Float>,
  minFilter: Tensor<Float>,
  maxFilter: Tensor<Float>,
  strides: [Int32],
  padding: Padding,
  dilations: [Int32] = [1, 1, 1, 1],
  paddingList: [Int32]
) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>) {
  let ret: (TensorHandle<OutType>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("QuantizedConv2DWithBiasAndRelu",
    input,
    filter,
    bias,
    minInput,
    maxInput,
    minFilter,
    maxFilter,
    Tinput$dtype: Tinput.tensorFlowDataType,
    Tfilter$dtype: Tfilter.tensorFlowDataType,
    out_type$dtype: OutType.tensorFlowDataType,
    strides: strides,
    padding: padding.cName,
    dilations: dilations,
    padding_list: paddingList)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

@inlinable @inline(__always)
public static func quantizedConv2DWithBiasAndReluAndRequantize<Tinput: TensorFlowScalar, Tfilter: TensorFlowScalar, Tbias: FloatingPoint & TensorFlowScalar, OutType: TensorFlowScalar>(
  _ input: Tensor<Tinput>,
  filter: Tensor<Tfilter>,
  bias: Tensor<Tbias>,
  minInput: Tensor<Float>,
  maxInput: Tensor<Float>,
  minFilter: Tensor<Float>,
  maxFilter: Tensor<Float>,
  minFreezedOutput: Tensor<Float>,
  maxFreezedOutput: Tensor<Float>,
  strides: [Int32],
  padding: Padding,
  dilations: [Int32] = [1, 1, 1, 1],
  paddingList: [Int32]
) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>) {
  let ret: (TensorHandle<OutType>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("QuantizedConv2DWithBiasAndReluAndRequantize",
    input,
    filter,
    bias,
    minInput,
    maxInput,
    minFilter,
    maxFilter,
    minFreezedOutput,
    maxFreezedOutput,
    Tinput$dtype: Tinput.tensorFlowDataType,
    Tfilter$dtype: Tfilter.tensorFlowDataType,
    Tbias$dtype: Tbias.tensorFlowDataType,
    out_type$dtype: OutType.tensorFlowDataType,
    strides: strides,
    padding: padding.cName,
    dilations: dilations,
    padding_list: paddingList)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

@inlinable @inline(__always)
public static func quantizedConv2DWithBiasAndRequantize<Tinput: TensorFlowScalar, Tfilter: TensorFlowScalar, Tbias: FloatingPoint & TensorFlowScalar, OutType: TensorFlowScalar>(
  _ input: Tensor<Tinput>,
  filter: Tensor<Tfilter>,
  bias: Tensor<Tbias>,
  minInput: Tensor<Float>,
  maxInput: Tensor<Float>,
  minFilter: Tensor<Float>,
  maxFilter: Tensor<Float>,
  minFreezedOutput: Tensor<Float>,
  maxFreezedOutput: Tensor<Float>,
  strides: [Int32],
  padding: Padding,
  dilations: [Int32] = [1, 1, 1, 1],
  paddingList: [Int32]
) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>) {
  let ret: (TensorHandle<OutType>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("QuantizedConv2DWithBiasAndRequantize",
    input,
    filter,
    bias,
    minInput,
    maxInput,
    minFilter,
    maxFilter,
    minFreezedOutput,
    maxFreezedOutput,
    Tinput$dtype: Tinput.tensorFlowDataType,
    Tfilter$dtype: Tfilter.tensorFlowDataType,
    Tbias$dtype: Tbias.tensorFlowDataType,
    out_type$dtype: OutType.tensorFlowDataType,
    strides: strides,
    padding: padding.cName,
    dilations: dilations,
    padding_list: paddingList)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

@inlinable @inline(__always)
public static func quantizedConv2DWithBiasSignedSumAndReluAndRequantize<Tinput: TensorFlowScalar, Tfilter: TensorFlowScalar, Tbias: FloatingPoint & TensorFlowScalar, Tsummand: TensorFlowScalar, OutType: TensorFlowScalar>(
  _ input: Tensor<Tinput>,
  filter: Tensor<Tfilter>,
  bias: Tensor<Tbias>,
  minInput: Tensor<Float>,
  maxInput: Tensor<Float>,
  minFilter: Tensor<Float>,
  maxFilter: Tensor<Float>,
  minFreezedOutput: Tensor<Float>,
  maxFreezedOutput: Tensor<Float>,
  summand: Tensor<Tsummand>,
  minSummand: Tensor<Float>,
  maxSummand: Tensor<Float>,
  strides: [Int32],
  padding: Padding,
  dilations: [Int32] = [1, 1, 1, 1],
  paddingList: [Int32]
) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>) {
  let ret: (TensorHandle<OutType>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("QuantizedConv2DWithBiasSignedSumAndReluAndRequantize",
    input,
    filter,
    bias,
    minInput,
    maxInput,
    minFilter,
    maxFilter,
    minFreezedOutput,
    maxFreezedOutput,
    summand,
    minSummand,
    maxSummand,
    Tinput$dtype: Tinput.tensorFlowDataType,
    Tfilter$dtype: Tfilter.tensorFlowDataType,
    Tbias$dtype: Tbias.tensorFlowDataType,
    Tsummand$dtype: Tsummand.tensorFlowDataType,
    out_type$dtype: OutType.tensorFlowDataType,
    strides: strides,
    padding: padding.cName,
    dilations: dilations,
    padding_list: paddingList)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

@inlinable @inline(__always)
public static func quantizedConv2DWithBiasSumAndRelu<Tinput: TensorFlowScalar, Tfilter: TensorFlowScalar, OutType: TensorFlowScalar>(
  _ input: Tensor<Tinput>,
  filter: Tensor<Tfilter>,
  bias: Tensor<Float>,
  minInput: Tensor<Float>,
  maxInput: Tensor<Float>,
  minFilter: Tensor<Float>,
  maxFilter: Tensor<Float>,
  summand: Tensor<Float>,
  strides: [Int32],
  padding: Padding,
  dilations: [Int32] = [1, 1, 1, 1],
  paddingList: [Int32]
) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>) {
  let ret: (TensorHandle<OutType>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("QuantizedConv2DWithBiasSumAndRelu",
    input,
    filter,
    bias,
    minInput,
    maxInput,
    minFilter,
    maxFilter,
    summand,
    Tinput$dtype: Tinput.tensorFlowDataType,
    Tfilter$dtype: Tfilter.tensorFlowDataType,
    out_type$dtype: OutType.tensorFlowDataType,
    strides: strides,
    padding: padding.cName,
    dilations: dilations,
    padding_list: paddingList)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

@inlinable @inline(__always)
public static func quantizedConv2DWithBiasSumAndReluAndRequantize<Tinput: TensorFlowScalar, Tfilter: TensorFlowScalar, Tbias: FloatingPoint & TensorFlowScalar, Tsummand: TensorFlowScalar, OutType: TensorFlowScalar>(
  _ input: Tensor<Tinput>,
  filter: Tensor<Tfilter>,
  bias: Tensor<Tbias>,
  minInput: Tensor<Float>,
  maxInput: Tensor<Float>,
  minFilter: Tensor<Float>,
  maxFilter: Tensor<Float>,
  minFreezedOutput: Tensor<Float>,
  maxFreezedOutput: Tensor<Float>,
  summand: Tensor<Tsummand>,
  minSummand: Tensor<Float>,
  maxSummand: Tensor<Float>,
  strides: [Int32],
  padding: Padding,
  dilations: [Int32] = [1, 1, 1, 1],
  paddingList: [Int32]
) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>) {
  let ret: (TensorHandle<OutType>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("QuantizedConv2DWithBiasSumAndReluAndRequantize",
    input,
    filter,
    bias,
    minInput,
    maxInput,
    minFilter,
    maxFilter,
    minFreezedOutput,
    maxFreezedOutput,
    summand,
    minSummand,
    maxSummand,
    Tinput$dtype: Tinput.tensorFlowDataType,
    Tfilter$dtype: Tfilter.tensorFlowDataType,
    Tbias$dtype: Tbias.tensorFlowDataType,
    Tsummand$dtype: Tsummand.tensorFlowDataType,
    out_type$dtype: OutType.tensorFlowDataType,
    strides: strides,
    padding: padding.cName,
    dilations: dilations,
    padding_list: paddingList)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Quantized Instance normalization.
///
/// - Parameters:
///   - x: A 4D input Tensor.
///   - x_min: The value represented by the lowest quantized input.
///   - x_max: The value represented by the highest quantized input.
///
/// - Attrs:
///   - output_range_given: If True, `given_y_min` and `given_y_min`
///     and `given_y_max` are used as the output range. Otherwise,
///     the implementation computes the output range.
///   - given_y_min: Output in `y_min` if `output_range_given` is True.
///   - given_y_max: Output in `y_max` if `output_range_given` is True.
///   - variance_epsilon: A small float number to avoid dividing by 0.
///   - min_separation: Minimum value of `y_max - y_min`
///
/// - Outputs:
///   - y: A 4D Tensor.
///   - y_min: The value represented by the lowest quantized output.
///   - y_max: The value represented by the highest quantized output.
@inlinable @inline(__always)
public static func quantizedInstanceNorm<T: TensorFlowScalar>(
  _ x: Tensor<T>,
  xMin: Tensor<Float>,
  xMax: Tensor<Float>,
  outputRangeGiven: Bool = false,
  givenYMin: Double = 0,
  givenYMax: Double = 0,
  varianceEpsilon: Double = 1e-05,
  minSeparation: Double = 0.001
) -> (y: Tensor<T>, yMin: Tensor<Float>, yMax: Tensor<Float>) {
  let ret: (TensorHandle<T>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("QuantizedInstanceNorm",
    x,
    xMin,
    xMax,
    T$dtype: T.tensorFlowDataType,
    output_range_given: outputRangeGiven,
    given_y_min: givenYMin,
    given_y_max: givenYMax,
    variance_epsilon: varianceEpsilon,
    min_separation: minSeparation)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Perform a quantized matrix multiplication of  `a` by the matrix `b`.
///
/// The inputs must be two-dimensional matrices and the inner dimension of
/// `a` (after being transposed if `transpose_a` is non-zero) must match the
/// outer dimension of `b` (after being transposed if `transposed_b` is
/// non-zero).
///
/// - Parameters:
///   - a: Must be a two-dimensional tensor.
///   - b: Must be a two-dimensional tensor.
///   - min_a: The float value that the lowest quantized `a` value represents.
///   - max_a: The float value that the highest quantized `a` value represents.
///   - min_b: The float value that the lowest quantized `b` value represents.
///   - max_b: The float value that the highest quantized `b` value represents.
///
/// - Attrs:
///   - transpose_a: If true, `a` is transposed before multiplication.
///   - transpose_b: If true, `b` is transposed before multiplication.
///   - Tactivation: The type of output produced by activation function
///     following this operation.
///
/// - Outputs:
///   - min_out: The float value that the lowest quantized output value represents.
///   - max_out: The float value that the highest quantized output value represents.
@inlinable @inline(__always)
public static func quantizedMatMul<T1: TensorFlowScalar, T2: TensorFlowScalar, Toutput: TensorFlowScalar, Tactivation: TensorFlowScalar>(
  _ a: Tensor<T1>,
  _ b: Tensor<T2>,
  minA: Tensor<Float>,
  maxA: Tensor<Float>,
  minB: Tensor<Float>,
  maxB: Tensor<Float>,
  transposeA: Bool = false,
  transposeB: Bool = false,
  typeTactivation: Tactivation.Type
) -> (out: Tensor<Toutput>, minOut: Tensor<Float>, maxOut: Tensor<Float>) {
  let ret: (TensorHandle<Toutput>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("QuantizedMatMul",
    a,
    b,
    minA,
    maxA,
    minB,
    maxB,
    T1$dtype: T1.tensorFlowDataType,
    T2$dtype: T2.tensorFlowDataType,
    Toutput$dtype: Toutput.tensorFlowDataType,
    Tactivation$dtype: Tactivation.tensorFlowDataType,
    transpose_a: transposeA,
    transpose_b: transposeB)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Produces the max pool of the input tensor for quantized types.
///
/// - Parameters:
///   - input: The 4D (batch x rows x cols x depth) Tensor to MaxReduce over.
///   - min_input: The float value that the lowest quantized input value represents.
///   - max_input: The float value that the highest quantized input value represents.
///
/// - Attrs:
///   - ksize: The size of the window for each dimension of the input tensor.
///     The length must be 4 to match the number of dimensions of the input.
///   - strides: The stride of the sliding window for each dimension of the input
///     tensor. The length must be 4 to match the number of dimensions of the input.
///   - padding: The type of padding algorithm to use.
///
/// - Outputs:
///   - min_output: The float value that the lowest quantized output value represents.
///   - max_output: The float value that the highest quantized output value represents.
@inlinable @inline(__always)
public static func quantizedMaxPool<T: TensorFlowScalar>(
  _ input: Tensor<T>,
  minInput: Tensor<Float>,
  maxInput: Tensor<Float>,
  ksize: [Int32],
  strides: [Int32],
  padding: Padding
) -> (output: Tensor<T>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>) {
  let ret: (TensorHandle<T>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("QuantizedMaxPool",
    input,
    minInput,
    maxInput,
    T$dtype: T.tensorFlowDataType,
    ksize: ksize,
    strides: strides,
    padding: padding.cName)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Returns x * y element-wise, working on quantized buffers.
///
/// - Parameters:
///   - min_x: The float value that the lowest quantized `x` value represents.
///   - max_x: The float value that the highest quantized `x` value represents.
///   - min_y: The float value that the lowest quantized `y` value represents.
///   - max_y: The float value that the highest quantized `y` value represents.
///
/// - Outputs:
///   - min_z: The float value that the lowest quantized output value represents.
///   - max_z: The float value that the highest quantized output value represents.
///
///     *NOTE*: `QuantizedMul` supports limited forms of broadcasting. More about
///     broadcasting [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@inlinable @inline(__always)
public static func quantizedMul<T1: TensorFlowScalar, T2: TensorFlowScalar, Toutput: TensorFlowScalar>(
  _ x: Tensor<T1>,
  _ y: Tensor<T2>,
  minX: Tensor<Float>,
  maxX: Tensor<Float>,
  minY: Tensor<Float>,
  maxY: Tensor<Float>
) -> (z: Tensor<Toutput>, minZ: Tensor<Float>, maxZ: Tensor<Float>) {
  let ret: (TensorHandle<Toutput>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("QuantizedMul",
    x,
    y,
    minX,
    maxX,
    minY,
    maxY,
    T1$dtype: T1.tensorFlowDataType,
    T2$dtype: T2.tensorFlowDataType,
    Toutput$dtype: Toutput.tensorFlowDataType)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Computes Quantized Rectified Linear: `max(features, 0)`
///
/// - Parameters:
///   - min_features: The float value that the lowest quantized value represents.
///   - max_features: The float value that the highest quantized value represents.
///
/// - Outputs:
///   - activations: Has the same output shape as "features".
///   - min_activations: The float value that the lowest quantized value represents.
///   - max_activations: The float value that the highest quantized value represents.
@inlinable @inline(__always)
public static func quantizedRelu<Tinput: TensorFlowScalar, OutType: TensorFlowScalar>(
  features: Tensor<Tinput>,
  minFeatures: Tensor<Float>,
  maxFeatures: Tensor<Float>
) -> (activations: Tensor<OutType>, minActivations: Tensor<Float>, maxActivations: Tensor<Float>) {
  let ret: (TensorHandle<OutType>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("QuantizedRelu",
    features,
    minFeatures,
    maxFeatures,
    Tinput$dtype: Tinput.tensorFlowDataType,
    out_type$dtype: OutType.tensorFlowDataType)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Computes Quantized Rectified Linear 6: `min(max(features, 0), 6)`
///
/// - Parameters:
///   - min_features: The float value that the lowest quantized value represents.
///   - max_features: The float value that the highest quantized value represents.
///
/// - Outputs:
///   - activations: Has the same output shape as "features".
///   - min_activations: The float value that the lowest quantized value represents.
///   - max_activations: The float value that the highest quantized value represents.
@inlinable @inline(__always)
public static func quantizedRelu6<Tinput: TensorFlowScalar, OutType: TensorFlowScalar>(
  features: Tensor<Tinput>,
  minFeatures: Tensor<Float>,
  maxFeatures: Tensor<Float>
) -> (activations: Tensor<OutType>, minActivations: Tensor<Float>, maxActivations: Tensor<Float>) {
  let ret: (TensorHandle<OutType>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("QuantizedRelu6",
    features,
    minFeatures,
    maxFeatures,
    Tinput$dtype: Tinput.tensorFlowDataType,
    out_type$dtype: OutType.tensorFlowDataType)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Computes Quantized Rectified Linear X: `min(max(features, 0), max_value)`
///
/// - Parameters:
///   - min_features: The float value that the lowest quantized value represents.
///   - max_features: The float value that the highest quantized value represents.
///
/// - Outputs:
///   - activations: Has the same output shape as "features".
///   - min_activations: The float value that the lowest quantized value represents.
///   - max_activations: The float value that the highest quantized value represents.
@inlinable @inline(__always)
public static func quantizedReluX<Tinput: TensorFlowScalar, OutType: TensorFlowScalar>(
  features: Tensor<Tinput>,
  maxValue: Tensor<Float>,
  minFeatures: Tensor<Float>,
  maxFeatures: Tensor<Float>
) -> (activations: Tensor<OutType>, minActivations: Tensor<Float>, maxActivations: Tensor<Float>) {
  let ret: (TensorHandle<OutType>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("QuantizedReluX",
    features,
    maxValue,
    minFeatures,
    maxFeatures,
    Tinput$dtype: Tinput.tensorFlowDataType,
    out_type$dtype: OutType.tensorFlowDataType)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Reshapes a quantized tensor as per the Reshape op.
///
/// ```
///
/// - Parameters:
///   - shape: Defines the shape of the output tensor.
///   - input_min: The minimum value of the input.
///   - input_max: The maximum value of the input.
///
/// - Outputs:
///   - output_min: This value is copied from input_min.
///   - output_max: This value is copied from input_max.
@inlinable @inline(__always)
public static func quantizedReshape<T: TensorFlowScalar, Tshape: BinaryInteger & TensorFlowScalar>(
  _ tensor: Tensor<T>,
  shape: Tensor<Tshape>,
  inputMin: Tensor<Float>,
  inputMax: Tensor<Float>
) -> (output: Tensor<T>, outputMin: Tensor<Float>, outputMax: Tensor<Float>) {
  let ret: (TensorHandle<T>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("QuantizedReshape",
    tensor,
    shape,
    inputMin,
    inputMax,
    T$dtype: T.tensorFlowDataType,
    Tshape$dtype: Tshape.tensorFlowDataType)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Resize quantized `images` to `size` using quantized bilinear interpolation.
///
/// Input images and output images must be quantized types.
///
/// - Parameters:
///   - images: 4-D with shape `[batch, height, width, channels]`.
///   - size: = A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
///     new size for the images.
///
/// - Attr align_corners: If true, the centers of the 4 corner pixels of the input and output tensors are
///   aligned, preserving the values at the corner pixels. Defaults to false.
///
/// - Output resized_images: 4-D with shape
///   `[batch, new_height, new_width, channels]`.
@inlinable @inline(__always)
public static func quantizedResizeBilinear<T: FloatingPoint & TensorFlowScalar>(
  images: Tensor<T>,
  size: Tensor<Int32>,
  min: Tensor<Float>,
  max: Tensor<Float>,
  alignCorners: Bool = false,
  halfPixelCenters: Bool = false
) -> (resizedImages: Tensor<T>, outMin: Tensor<Float>, outMax: Tensor<Float>) {
  let ret: (TensorHandle<T>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("QuantizedResizeBilinear",
    images,
    size,
    min,
    max,
    T$dtype: T.tensorFlowDataType,
    align_corners: alignCorners,
    half_pixel_centers: halfPixelCenters)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Converts one or more images from RGB to HSV.
///
/// Outputs a tensor of the same shape as the `images` tensor, containing the HSV
/// value of the pixels. The output is only well defined if the value in `images`
/// are in `[0,1]`.
///
/// `output[..., 0]` contains hue, `output[..., 1]` contains saturation, and
/// `output[..., 2]` contains value. All HSV values are in `[0,1]`. A hue of 0
/// corresponds to pure red, hue 1/3 is pure green, and 2/3 is pure blue.
///
/// - Parameter images: 1-D or higher rank. RGB data to convert. Last dimension must be size 3.
///
/// - Output output: `images` converted to HSV.
@inlinable @inline(__always)
public static func rGBToHSV<T: FloatingPoint & TensorFlowScalar>(
  images: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("RGBToHSV",
    images,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Returns a `RaggedTensor` containing the specified sequences of numbers.
///
///
/// Returns a `RaggedTensor` `result` composed from `rt_dense_values` and
/// `rt_nested_splits`, such that
/// `result[i] = range(starts[i], limits[i], deltas[i])`.
///
/// ```python
/// >>> (rt_nested_splits, rt_dense_values) = gen_ragged_ops.ragged_range(
/// ...     starts=[2, 5, 8], limits=[3, 5, 12], deltas=1)
/// >>> result = ragged.from_nested_row_splits(rt_dense_values, rt_nested_splits)
/// >>> print result.eval().tolist()
/// [[2],               # result[0] = range(2, 3)
///  [],                # result[1] = range(5, 5)
///  [8, 9, 10, 11]]    # result[2] = range(8, 12)
/// ```
///
/// The input tensors `starts`, `limits`, and `deltas` may be scalars or vectors.
/// The vector inputs must all have the same size.  Scalar inputs are broadcast
/// to match the size of the vector inputs.
///
/// - Parameters:
///   - starts: The starts of each range.
///   - limits: The limits of each range.
///   - deltas: The deltas of each range.
///
/// - Outputs:
///   - rt_nested_splits: The `row_splits` for the returned `RaggedTensor`.
///   - rt_dense_values: The `flat_values` for the returned `RaggedTensor`.
@inlinable @inline(__always)
public static func raggedRange<T: Numeric & TensorFlowScalar>(
  starts: Tensor<T>,
  limits: Tensor<T>,
  deltas: Tensor<T>
) -> (rtNestedSplits: Tensor<Int64>, rtDenseValues: Tensor<T>) {
  let ret: (TensorHandle<Int64>, TensorHandle<T>) = #tfop("RaggedRange",
    starts,
    limits,
    deltas,
    T$dtype: T.tensorFlowDataType)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

/// Converts a `RaggedTensor` into a `SparseTensor` with the same values.
///
/// input=ragged.from_nested_row_splits(rt_dense_values, rt_nested_splits)
/// output=SparseTensor(indices=sparse_indices, values=sparse_values,
///                     dense_shape=sparse_dense_shape)
///
/// - Parameters:
///   - rt_nested_splits: The `row_splits` for the `RaggedTensor`.
///   - rt_dense_values: The `flat_values` for the `RaggedTensor`.
///
/// - Attr RAGGED_RANK: The ragged rank of the input RaggedTensor.  `rt_nested_splits` should contain
///   this number of ragged-splits tensors.  This value should equal
///   `input.ragged_rank`.
///
/// - Outputs:
///   - sparse_indices: The indices for the `SparseTensor`.
///   - sparse_values: The values of the `SparseTensor`.
///   - sparse_dense_shape: `sparse_dense_shape` is a tight bounding box of the input `RaggedTensor`.
@inlinable @inline(__always)
public static func raggedTensorToSparse<T: TensorFlowScalar>(
  rtNestedSplits: [Tensor<Int64>],
  rtDenseValues: Tensor<T>
) -> (sparseIndices: Tensor<Int64>, sparseValues: Tensor<T>, sparseDenseShape: Tensor<Int64>) {
  let ret: (TensorHandle<Int64>, TensorHandle<T>, TensorHandle<Int64>) = #tfop("RaggedTensorToSparse",
    rtNestedSplits,
    rtDenseValues,
    T$dtype: T.tensorFlowDataType)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Randomly crop `image`.
///
/// `size` is a 1-D int64 tensor with 2 elements representing the crop height and
/// width.  The values must be non negative.
///
/// This Op picks a random location in `image` and crops a `height` by `width`
/// rectangle from that location.  The random location is picked so the cropped
/// area will fit inside the original image.
///
/// - Parameters:
///   - image: 3-D of shape `[height, width, channels]`.
///   - size: 1-D of length 2 containing: `crop_height`, `crop_width`..
///
/// - Attrs:
///   - seed: If either seed or seed2 are set to be non-zero, the random number
///     generator is seeded by the given seed.  Otherwise, it is seeded by a
///     random seed.
///   - seed2: An second seed to avoid seed collision.
///
/// - Output output: 3-D of shape `[crop_height, crop_width, channels].`
@inlinable @inline(__always)
public static func randomCrop<T: Numeric & TensorFlowScalar>(
  image: Tensor<T>,
  size: Tensor<Int64>,
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("RandomCrop",
    image,
    size,
    T$dtype: T.tensorFlowDataType,
    seed: seed,
    seed2: seed2)
  return Tensor(handle: ret)
}

/// Outputs random values from the Gamma distribution(s) described by alpha.
///
/// This op uses the algorithm by Marsaglia et al. to acquire samples via
/// transformation-rejection from pairs of uniform and normal random variables.
/// See http://dl.acm.org/citation.cfm?id=358414
///
/// - Parameters:
///   - shape: 1-D integer tensor. Shape of independent samples to draw from each
///     distribution described by the shape parameters given in alpha.
///   - alpha: A tensor in which each scalar is a "shape" parameter describing the
///     associated gamma distribution.
///
/// - Attrs:
///   - seed: If either `seed` or `seed2` are set to be non-zero, the random number
///     generator is seeded by the given seed.  Otherwise, it is seeded by a
///     random seed.
///   - seed2: A second seed to avoid seed collision.
///
/// - Output output: A tensor with shape `shape + shape(alpha)`. Each slice
///   `[:, ..., :, i0, i1, ...iN]` contains the samples drawn for
///   `alpha[i0, i1, ...iN]`. The dtype of the output matches the dtype of alpha.
@inlinable @inline(__always)
public static func randomGamma<S: BinaryInteger & TensorFlowScalar, T: FloatingPoint & TensorFlowScalar>(
  shape: Tensor<S>,
  alpha: Tensor<T>,
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("RandomGamma",
    shape,
    alpha,
    S$dtype: S.tensorFlowDataType,
    T$dtype: T.tensorFlowDataType,
    seed: seed,
    seed2: seed2)
  return Tensor(handle: ret)
}

/// Computes the derivative of a Gamma random sample w.r.t. `alpha`.
@inlinable @inline(__always)
public static func randomGammaGrad<T: FloatingPoint & TensorFlowScalar>(
  alpha: Tensor<T>,
  sample: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("RandomGammaGrad",
    alpha,
    sample,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Use RandomPoissonV2 instead.
@inlinable @inline(__always)
public static func randomPoisson<S: BinaryInteger & TensorFlowScalar, Dtype: FloatingPoint & TensorFlowScalar>(
  shape: Tensor<S>,
  rate: Tensor<Dtype>,
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> Tensor<Dtype> {
  let ret: TensorHandle<Dtype> = #tfop("RandomPoisson",
    shape,
    rate,
    S$dtype: S.tensorFlowDataType,
    dtype$dtype: Dtype.tensorFlowDataType,
    seed: seed,
    seed2: seed2)
  return Tensor(handle: ret)
}

/// Outputs random values from the Poisson distribution(s) described by rate.
///
/// This op uses two algorithms, depending on rate. If rate >= 10, then
/// the algorithm by Hormann is used to acquire samples via
/// transformation-rejection.
/// See http://www.sciencedirect.com/science/article/pii/0167668793909974.
///
/// Otherwise, Knuth's algorithm is used to acquire samples via multiplying uniform
/// random variables.
/// See Donald E. Knuth (1969). Seminumerical Algorithms. The Art of Computer
/// Programming, Volume 2. Addison Wesley
///
/// - Parameters:
///   - shape: 1-D integer tensor. Shape of independent samples to draw from each
///     distribution described by the shape parameters given in rate.
///   - rate: A tensor in which each scalar is a "rate" parameter describing the
///     associated poisson distribution.
///
/// - Attrs:
///   - seed: If either `seed` or `seed2` are set to be non-zero, the random number
///     generator is seeded by the given seed.  Otherwise, it is seeded by a
///     random seed.
///   - seed2: A second seed to avoid seed collision.
///
/// - Output output: A tensor with shape `shape + shape(rate)`. Each slice
///   `[:, ..., :, i0, i1, ...iN]` contains the samples drawn for
///   `rate[i0, i1, ...iN]`.
@inlinable @inline(__always)
public static func randomPoissonV2<S: BinaryInteger & TensorFlowScalar, R: Numeric & TensorFlowScalar, Dtype: Numeric & TensorFlowScalar>(
  shape: Tensor<S>,
  rate: Tensor<R>,
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> Tensor<Dtype> {
  let ret: TensorHandle<Dtype> = #tfop("RandomPoissonV2",
    shape,
    rate,
    S$dtype: S.tensorFlowDataType,
    R$dtype: R.tensorFlowDataType,
    dtype$dtype: Dtype.tensorFlowDataType,
    seed: seed,
    seed2: seed2)
  return Tensor(handle: ret)
}

/// Randomly shuffles a tensor along its first dimension.
///
///   The tensor is shuffled along dimension 0, such that each `value[j]` is mapped
///   to one and only one `output[i]`. For example, a mapping that might occur for a
///   3x2 tensor is:
///
/// ```
/// [[1, 2],       [[5, 6],
///  [3, 4],  ==>   [1, 2],
///  [5, 6]]        [3, 4]]
/// ```
///
/// - Parameter value: The tensor to be shuffled.
///
/// - Attrs:
///   - seed: If either `seed` or `seed2` are set to be non-zero, the random number
///     generator is seeded by the given seed.  Otherwise, it is seeded by a
///     random seed.
///   - seed2: A second seed to avoid seed collision.
///
/// - Output output: A tensor of same shape and type as `value`, shuffled along its first
///   dimension.
@inlinable @inline(__always)
public static func randomShuffle<T: TensorFlowScalar>(
  value: Tensor<T>,
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("RandomShuffle",
    value,
    T$dtype: T.tensorFlowDataType,
    seed: seed,
    seed2: seed2)
  return Tensor(handle: ret)
}

/// Outputs random values from a normal distribution.
///
/// The generated values will have mean 0 and standard deviation 1.
///
/// - Parameter shape: The shape of the output tensor.
///
/// - Attrs:
///   - seed: If either `seed` or `seed2` are set to be non-zero, the random number
///     generator is seeded by the given seed.  Otherwise, it is seeded by a
///     random seed.
///   - seed2: A second seed to avoid seed collision.
///   - dtype: The type of the output.
///
/// - Output output: A tensor of the specified shape filled with random normal values.
@inlinable @inline(__always)
public static func randomStandardNormal<Dtype: FloatingPoint & TensorFlowScalar, T: BinaryInteger & TensorFlowScalar>(
  shape: Tensor<T>,
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> Tensor<Dtype> {
  let ret: TensorHandle<Dtype> = #tfop("RandomStandardNormal",
    shape,
    dtype$dtype: Dtype.tensorFlowDataType,
    T$dtype: T.tensorFlowDataType,
    seed: seed,
    seed2: seed2)
  return Tensor(handle: ret)
}

/// Outputs random values from a uniform distribution.
///
/// The generated values follow a uniform distribution in the range `[0, 1)`. The
/// lower bound 0 is included in the range, while the upper bound 1 is excluded.
///
/// - Parameter shape: The shape of the output tensor.
///
/// - Attrs:
///   - seed: If either `seed` or `seed2` are set to be non-zero, the random number
///     generator is seeded by the given seed.  Otherwise, it is seeded by a
///     random seed.
///   - seed2: A second seed to avoid seed collision.
///   - dtype: The type of the output.
///
/// - Output output: A tensor of the specified shape filled with uniform random values.
@inlinable @inline(__always)
public static func randomUniform<Dtype: FloatingPoint & TensorFlowScalar, T: BinaryInteger & TensorFlowScalar>(
  shape: Tensor<T>,
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> Tensor<Dtype> {
  let ret: TensorHandle<Dtype> = #tfop("RandomUniform",
    shape,
    dtype$dtype: Dtype.tensorFlowDataType,
    T$dtype: T.tensorFlowDataType,
    seed: seed,
    seed2: seed2)
  return Tensor(handle: ret)
}

/// Outputs random integers from a uniform distribution.
///
/// The generated values are uniform integers in the range `[minval, maxval)`.
/// The lower bound `minval` is included in the range, while the upper bound
/// `maxval` is excluded.
///
/// The random integers are slightly biased unless `maxval - minval` is an exact
/// power of two.  The bias is small for values of `maxval - minval` significantly
/// smaller than the range of the output (either `2^32` or `2^64`).
///
/// - Parameters:
///   - shape: The shape of the output tensor.
///   - minval: 0-D.  Inclusive lower bound on the generated integers.
///   - maxval: 0-D.  Exclusive upper bound on the generated integers.
///
/// - Attrs:
///   - seed: If either `seed` or `seed2` are set to be non-zero, the random number
///     generator is seeded by the given seed.  Otherwise, it is seeded by a
///     random seed.
///   - seed2: A second seed to avoid seed collision.
///
/// - Output output: A tensor of the specified shape filled with uniform random integers.
@inlinable @inline(__always)
public static func randomUniformInt<Tout: BinaryInteger & TensorFlowScalar, T: BinaryInteger & TensorFlowScalar>(
  shape: Tensor<T>,
  minval: Tensor<Tout>,
  maxval: Tensor<Tout>,
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> Tensor<Tout> {
  let ret: TensorHandle<Tout> = #tfop("RandomUniformInt",
    shape,
    minval,
    maxval,
    Tout$dtype: Tout.tensorFlowDataType,
    T$dtype: T.tensorFlowDataType,
    seed: seed,
    seed2: seed2)
  return Tensor(handle: ret)
}

/// Creates a sequence of numbers.
///
/// This operation creates a sequence of numbers that begins at `start` and
/// extends by increments of `delta` up to but not including `limit`.
///
/// For example:
///
/// ```
/// # 'start' is 3
/// # 'limit' is 18
/// # 'delta' is 3
/// tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15]
/// ```
///
/// - Parameters:
///   - start: 0-D (scalar). First entry in the sequence.
///   - limit: 0-D (scalar). Upper limit of sequence, exclusive.
///   - delta: 0-D (scalar). Optional. Default is 1. Number that increments `start`.
///
/// - Output output: 1-D.
@inlinable @inline(__always)
public static func range<Tidx: Numeric & TensorFlowScalar>(
  start: Tensor<Tidx>,
  limit: Tensor<Tidx>,
  delta: Tensor<Tidx>
) -> Tensor<Tidx> {
  let ret: TensorHandle<Tidx> = #tfop("Range",
    start,
    limit,
    delta,
    Tidx$dtype: Tidx.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Returns the rank of a tensor.
///
/// This operation returns an integer representing the rank of `input`.
///
/// For example:
///
/// ```
/// # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
/// # shape of tensor 't' is [2, 2, 3]
/// rank(t) ==> 3
/// ```
///
/// **Note**: The rank of a tensor is not the same as the rank of a matrix. The rank
/// of a tensor is the number of indices required to uniquely select each element
/// of the tensor. Rank is also known as "order", "degree", or "ndims."
@inlinable @inline(__always)
public static func rank<T: TensorFlowScalar>(
  _ input: Tensor<T>
) -> Tensor<Int32> {
  let ret: TensorHandle<Int32> = #tfop("Rank",
    input,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Reads and outputs the entire contents of the input filename.
@inlinable @inline(__always)
public static func readFile(
  filename: StringTensor
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("ReadFile",
    filename)
  return StringTensor(handle: ret)
}

/// Returns the real part of a complex number.
///
/// Given a tensor `input` of complex numbers, this operation returns a tensor of
/// type `float` that is the real part of each element in `input`. All elements in
/// `input` must be complex numbers of the form \\(a + bj\\), where *a* is the real
///  part returned by this operation and *b* is the imaginary part.
///
/// For example:
///
/// ```
/// # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
/// tf.real(input) ==> [-2.25, 3.25]
/// ```
@inlinable @inline(__always)
public static func real<T: TensorFlowScalar, Tout: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>
) -> Tensor<Tout> {
  let ret: TensorHandle<Tout> = #tfop("Real",
    input,
    T$dtype: T.tensorFlowDataType,
    Tout$dtype: Tout.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Returns x / y element-wise for real types.
///
/// If `x` and `y` are reals, this will return the floating-point division.
///
/// *NOTE*: `Div` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@inlinable @inline(__always)
public static func realDiv<T: Numeric & TensorFlowScalar>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("RealDiv",
    x,
    y,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the reciprocal of x element-wise.
///
/// I.e., \\(y = 1 / x\\).
@inlinable @inline(__always)
public static func reciprocal<T: Numeric & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Reciprocal",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the gradient for the inverse of `x` wrt its input.
///
/// Specifically, `grad = -dy * y*y`, where `y = 1/x`, and `dy`
/// is the corresponding input gradient.
@inlinable @inline(__always)
public static func reciprocalGrad<T: FloatingPoint & TensorFlowScalar>(
  _ y: Tensor<T>,
  dy: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("ReciprocalGrad",
    y,
    dy,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Emits randomized records.
///
/// - Attrs:
///   - file_pattern: Glob pattern for the data files.
///   - file_random_seed: Random seeds used to produce randomized records.
///   - file_shuffle_shift_ratio: Shifts the list of files after the list is randomly
///     shuffled.
///   - file_buffer_size: The randomization shuffling buffer.
///   - file_parallelism: How many sstables are opened and concurrently iterated over.
///   - batch_size: The batch size.
///   - compression_type: The type of compression for the file. Currently ZLIB and
///     GZIP are supported. Defaults to none.
///
/// - Output records: A tensor of shape [batch_size].
@inlinable @inline(__always)
public static func recordInput(
  filePattern: String,
  fileRandomSeed: Int64 = 301,
  fileShuffleShiftRatio: Double = 0,
  fileBufferSize: Int64 = 10000,
  fileParallelism: Int64 = 16,
  batchSize: Int64 = 32,
  compressionType: String
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("RecordInput",
    file_pattern: filePattern,
    file_random_seed: fileRandomSeed,
    file_shuffle_shift_ratio: fileShuffleShiftRatio,
    file_buffer_size: fileBufferSize,
    file_parallelism: fileParallelism,
    batch_size: batchSize,
    compression_type: compressionType)
  return StringTensor(handle: ret)
}

/// Joins a string Tensor across the given dimensions.
///
/// Computes the string join across dimensions in the given string Tensor of shape
/// `[\\(d_0, d_1, ..., d_{n-1}\\)]`.  Returns a new Tensor created by joining the input
/// strings with the given separator (default: empty string).  Negative indices are
/// counted backwards from the end, with `-1` being equivalent to `n - 1`.  If
/// indices are not specified, joins across all dimensions beginning from `n - 1`
/// through `0`.
///
/// For example:
///
/// ```python
/// # tensor `a` is [["a", "b"], ["c", "d"]]
/// tf.reduce_join(a, 0) ==> ["ac", "bd"]
/// tf.reduce_join(a, 1) ==> ["ab", "cd"]
/// tf.reduce_join(a, -2) = tf.reduce_join(a, 0) ==> ["ac", "bd"]
/// tf.reduce_join(a, -1) = tf.reduce_join(a, 1) ==> ["ab", "cd"]
/// tf.reduce_join(a, 0, keep_dims=True) ==> [["ac", "bd"]]
/// tf.reduce_join(a, 1, keep_dims=True) ==> [["ab"], ["cd"]]
/// tf.reduce_join(a, 0, separator=".") ==> ["a.c", "b.d"]
/// tf.reduce_join(a, [0, 1]) ==> "acbd"
/// tf.reduce_join(a, [1, 0]) ==> "abcd"
/// tf.reduce_join(a, []) ==> [["a", "b"], ["c", "d"]]
/// tf.reduce_join(a) = tf.reduce_join(a, [1, 0]) ==> "abcd"
/// ```
///
/// - Parameters:
///   - inputs: The input to be joined.  All reduced indices must have non-zero size.
///   - reduction_indices: The dimensions to reduce over.  Dimensions are reduced in the
///     order specified.  Omitting `reduction_indices` is equivalent to passing
///     `[n-1, n-2, ..., 0]`.  Negative indices from `-n` to `-1` are supported.
///
/// - Attrs:
///   - keep_dims: If `True`, retain reduced dimensions with length `1`.
///   - separator: The separator to use when joining.
///
/// - Output output: Has shape equal to that of the input with reduced dimensions removed or
///   set to `1` depending on `keep_dims`.
@inlinable @inline(__always)
public static func reduceJoin(
  inputs: StringTensor,
  reductionIndices: Tensor<Int32>,
  keepDims: Bool = false,
  separator: String
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("ReduceJoin",
    inputs,
    reductionIndices,
    keep_dims: keepDims,
    separator: separator)
  return StringTensor(handle: ret)
}

/// Check if the input matches the regex pattern.
///
/// The input is a string tensor of any shape. The pattern is a scalar
/// string tensor which is applied to every element of the input tensor.
/// The boolean values (True or False) of the output tensor indicate
/// if the input matches the regex pattern provided.
///
/// The pattern follows the re2 syntax (https://github.com/google/re2/wiki/Syntax)
///
/// - Parameters:
///   - input: A string tensor of the text to be processed.
///   - pattern: A scalar string tensor containing the regular expression to match the input.
///
/// - Output output: A bool tensor with the same shape as `input`.
@inlinable @inline(__always)
public static func regexFullMatch(
  _ input: StringTensor,
  pattern: StringTensor
) -> Tensor<Bool> {
  let ret: TensorHandle<Bool> = #tfop("RegexFullMatch",
    input,
    pattern)
  return Tensor(handle: ret)
}

/// Replaces matches of the `pattern` regular expression in `input` with the
/// replacement string provided in `rewrite`.
///
/// It follows the re2 syntax (https://github.com/google/re2/wiki/Syntax)
///
/// - Parameters:
///   - input: The text to be processed.
///   - pattern: The regular expression to be matched in the `input` strings.
///   - rewrite: The rewrite string to be substituted for the `pattern` expression where it is
///     matched in the `input` strings.
///
/// - Attr replace_global: If True, the replacement is global (that is, all matches of the `pattern` regular
///   expression in each input string are rewritten), otherwise the `rewrite`
///   substitution is only made for the first `pattern` match.
///
/// - Output output: The text after applying pattern match and rewrite substitution.
@inlinable @inline(__always)
public static func regexReplace(
  _ input: StringTensor,
  pattern: StringTensor,
  rewrite: StringTensor,
  replaceGlobal: Bool = true
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("RegexReplace",
    input,
    pattern,
    rewrite,
    replace_global: replaceGlobal)
  return StringTensor(handle: ret)
}

/// Computes rectified linear: `max(features, 0)`.
@inlinable @inline(__always)
public static func relu<T: Numeric & TensorFlowScalar>(
  features: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Relu",
    features,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes rectified linear 6: `min(max(features, 0), 6)`.
@inlinable @inline(__always)
public static func relu6<T: Numeric & TensorFlowScalar>(
  features: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Relu6",
    features,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes rectified linear 6 gradients for a Relu6 operation.
///
/// - Parameters:
///   - gradients: The backpropagated gradients to the corresponding Relu6 operation.
///   - features: The features passed as input to the corresponding Relu6 operation, or
///     its output; using either one produces the same result.
///
/// - Output backprops: The gradients:
///   `gradients * (features > 0) * (features < 6)`.
@inlinable @inline(__always)
public static func relu6Grad<T: Numeric & TensorFlowScalar>(
  gradients: Tensor<T>,
  features: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Relu6Grad",
    gradients,
    features,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes rectified linear gradients for a Relu operation.
///
/// - Parameters:
///   - gradients: The backpropagated gradients to the corresponding Relu operation.
///   - features: The features passed as input to the corresponding Relu operation, OR
///     the outputs of that operation (both work equivalently).
///
/// - Output backprops: `gradients * (features > 0)`.
@inlinable @inline(__always)
public static func reluGrad<T: Numeric & TensorFlowScalar>(
  gradients: Tensor<T>,
  features: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("ReluGrad",
    gradients,
    features,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes a range that covers the actual values present in a quantized tensor.
///
/// Given a quantized tensor described by `(input, input_min, input_max)`, outputs a
/// range that covers the actual values present in that tensor. This op is typically
/// used to produce the `requested_output_min` and `requested_output_max` for
/// `Requantize`.
///
/// - Parameters:
///   - input_min: The float value that the minimum quantized input value represents.
///   - input_max: The float value that the maximum quantized input value represents.
///
/// - Attr Tinput: The type of the input.
///
/// - Outputs:
///   - output_min: The computed min output.
///   - output_max: the computed max output.
@inlinable @inline(__always)
public static func requantizationRange<Tinput: TensorFlowScalar>(
  _ input: Tensor<Tinput>,
  inputMin: Tensor<Float>,
  inputMax: Tensor<Float>
) -> (outputMin: Tensor<Float>, outputMax: Tensor<Float>) {
  let ret: (TensorHandle<Float>, TensorHandle<Float>) = #tfop("RequantizationRange",
    input,
    inputMin,
    inputMax,
    Tinput$dtype: Tinput.tensorFlowDataType)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

/// Computes requantization range per channel.
///
/// - Parameters:
///   - input: The original input tensor.
///   - input_min: The minimum value of the input tensor
///   - input_max: The maximum value of the input tensor.
///
/// - Attrs:
///   - T: The quantized type of input tensor that needs to be converted. 
///   - clip_value_max: The maximum value of the output that needs to be clipped.
///     Example: set this to 6 for Relu6. 
///
/// - Outputs:
///   - output_min: The minimum value of the final output tensor
///   - output_max: The maximum value of the final output tensor.
@inlinable @inline(__always)
public static func requantizationRangePerChannel<T: TensorFlowScalar>(
  _ input: Tensor<T>,
  inputMin: Tensor<Float>,
  inputMax: Tensor<Float>,
  clipValueMax: Double
) -> (outputMin: Tensor<Float>, outputMax: Tensor<Float>) {
  let ret: (TensorHandle<Float>, TensorHandle<Float>) = #tfop("RequantizationRangePerChannel",
    input,
    inputMin,
    inputMax,
    T$dtype: T.tensorFlowDataType,
    clip_value_max: clipValueMax)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

/// Converts the quantized `input` tensor into a lower-precision `output`.
///
/// Converts the quantized `input` tensor into a lower-precision `output`, using the
/// output range specified with `requested_output_min` and `requested_output_max`.
///
/// `[input_min, input_max]` are scalar floats that specify the range for the float
/// interpretation of the `input` data. For example, if `input_min` is -1.0f and
/// `input_max` is 1.0f, and we are dealing with `quint16` quantized data, then a 0
/// value in the 16-bit data should be interpreted as -1.0f, and a 65535 means 1.0f.
///
/// - Parameters:
///   - input_min: The float value that the minimum quantized input value represents.
///   - input_max: The float value that the maximum quantized input value represents.
///   - requested_output_min: The float value that the minimum quantized output value represents.
///   - requested_output_max: The float value that the maximum quantized output value represents.
///
/// - Attrs:
///   - Tinput: The type of the input.
///   - out_type: The type of the output. Should be a lower bit depth than Tinput.
///
/// - Outputs:
///   - output_min: The requested_output_min value is copied into this output.
///   - output_max: The requested_output_max value is copied into this output.
@inlinable @inline(__always)
public static func requantize<Tinput: TensorFlowScalar, OutType: TensorFlowScalar>(
  _ input: Tensor<Tinput>,
  inputMin: Tensor<Float>,
  inputMax: Tensor<Float>,
  requestedOutputMin: Tensor<Float>,
  requestedOutputMax: Tensor<Float>
) -> (output: Tensor<OutType>, outputMin: Tensor<Float>, outputMax: Tensor<Float>) {
  let ret: (TensorHandle<OutType>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("Requantize",
    input,
    inputMin,
    inputMax,
    requestedOutputMin,
    requestedOutputMax,
    Tinput$dtype: Tinput.tensorFlowDataType,
    out_type$dtype: OutType.tensorFlowDataType)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Requantizes input with min and max values known per channel.
///
/// - Parameters:
///   - input: The original input tensor.
///   - input_min: The minimum value of the input tensor
///   - input_max: The maximum value of the input tensor.
///   - requested_output_min: The minimum value of the output tensor requested.
///   - requested_output_max: The maximum value of the output tensor requested.
///
/// - Attrs:
///   - T: The quantized type of input tensor that needs to be converted. 
///   - out_type: The quantized type of output tensor that needs to be converted.
///
/// - Outputs:
///   - output: Output tensor.
///   - output_min: The minimum value of the final output tensor
///   - output_max: The maximum value of the final output tensor.
@inlinable @inline(__always)
public static func requantizePerChannel<T: TensorFlowScalar, OutType: TensorFlowScalar>(
  _ input: Tensor<T>,
  inputMin: Tensor<Float>,
  inputMax: Tensor<Float>,
  requestedOutputMin: Tensor<Float>,
  requestedOutputMax: Tensor<Float>
) -> (output: Tensor<OutType>, outputMin: Tensor<Float>, outputMax: Tensor<Float>) {
  let ret: (TensorHandle<OutType>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("RequantizePerChannel",
    input,
    inputMin,
    inputMax,
    requestedOutputMin,
    requestedOutputMax,
    T$dtype: T.tensorFlowDataType,
    out_type$dtype: OutType.tensorFlowDataType)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

@inlinable @inline(__always)
public static func requiresOlderGraphVersion(
) -> Tensor<Int32> {
  let ret: TensorHandle<Int32> = #tfop("RequiresOlderGraphVersion")
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func reservedAttr(
  range: Int64
) {
  return #tfop("ReservedAttr",
    range: range)
}

@inlinable @inline(__always)
public static func reservedInput(
  _ input: Tensor<Int32>
) {
  return #tfop("ReservedInput",
    input)
}

/// Reshapes a tensor.
///
/// Given `tensor`, this operation returns a tensor that has the same values
/// as `tensor` with shape `shape`.
///
/// If one component of `shape` is the special value -1, the size of that dimension
/// is computed so that the total size remains constant.  In particular, a `shape`
/// of `[-1]` flattens into 1-D.  At most one component of `shape` can be -1.
///
/// If `shape` is 1-D or higher, then the operation returns a tensor with shape
/// `shape` filled with the values of `tensor`. In this case, the number of elements
/// implied by `shape` must be the same as the number of elements in `tensor`.
///
/// For example:
///
/// ```
/// # tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
/// # tensor 't' has shape [9]
/// reshape(t, [3, 3]) ==> [[1, 2, 3],
///                         [4, 5, 6],
///                         [7, 8, 9]]
///
/// # tensor 't' is [[[1, 1], [2, 2]],
/// #                [[3, 3], [4, 4]]]
/// # tensor 't' has shape [2, 2, 2]
/// reshape(t, [2, 4]) ==> [[1, 1, 2, 2],
///                         [3, 3, 4, 4]]
///
/// # tensor 't' is [[[1, 1, 1],
/// #                 [2, 2, 2]],
/// #                [[3, 3, 3],
/// #                 [4, 4, 4]],
/// #                [[5, 5, 5],
/// #                 [6, 6, 6]]]
/// # tensor 't' has shape [3, 2, 3]
/// # pass '[-1]' to flatten 't'
/// reshape(t, [-1]) ==> [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]
///
/// # -1 can also be used to infer the shape
///
/// # -1 is inferred to be 9:
/// reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
///                          [4, 4, 4, 5, 5, 5, 6, 6, 6]]
/// # -1 is inferred to be 2:
/// reshape(t, [-1, 9]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
///                          [4, 4, 4, 5, 5, 5, 6, 6, 6]]
/// # -1 is inferred to be 3:
/// reshape(t, [ 2, -1, 3]) ==> [[[1, 1, 1],
///                               [2, 2, 2],
///                               [3, 3, 3]],
///                              [[4, 4, 4],
///                               [5, 5, 5],
///                               [6, 6, 6]]]
///
/// # tensor 't' is [7]
/// # shape `[]` reshapes to a scalar
/// reshape(t, []) ==> 7
/// ```
///
/// - Parameter shape: Defines the shape of the output tensor.
@inlinable @inline(__always)
public static func reshape<T: TensorFlowScalar, Tshape: BinaryInteger & TensorFlowScalar>(
  _ tensor: Tensor<T>,
  shape: Tensor<Tshape>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Reshape",
    tensor,
    shape,
    T$dtype: T.tensorFlowDataType,
    Tshape$dtype: Tshape.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Resize `images` to `size` using area interpolation.
///
/// Input images can be of different types but output images are always float.
///
/// The range of pixel values for the output image might be slightly different
/// from the range for the input image because of limited numerical precision.
/// To guarantee an output range, for example `[0.0, 1.0]`, apply
/// `tf.clip_by_value` to the output.
///
/// Each output pixel is computed by first transforming the pixel's footprint into
/// the input tensor and then averaging the pixels that intersect the footprint. An
/// input pixel's contribution to the average is weighted by the fraction of its
/// area that intersects the footprint.  This is the same as OpenCV's INTER_AREA.
///
/// - Parameters:
///   - images: 4-D with shape `[batch, height, width, channels]`.
///   - size: = A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
///     new size for the images.
///
/// - Attr align_corners: If true, the centers of the 4 corner pixels of the input and output tensors are
///   aligned, preserving the values at the corner pixels. Defaults to false.
///
/// - Output resized_images: 4-D with shape
///   `[batch, new_height, new_width, channels]`.
@inlinable @inline(__always)
public static func resizeArea<T: Numeric & TensorFlowScalar>(
  images: Tensor<T>,
  size: Tensor<Int32>,
  alignCorners: Bool = false
) -> Tensor<Float> {
  let ret: TensorHandle<Float> = #tfop("ResizeArea",
    images,
    size,
    T$dtype: T.tensorFlowDataType,
    align_corners: alignCorners)
  return Tensor(handle: ret)
}

/// Resize `images` to `size` using bicubic interpolation.
///
/// Input images can be of different types but output images are always float.
///
/// - Parameters:
///   - images: 4-D with shape `[batch, height, width, channels]`.
///   - size: = A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
///     new size for the images.
///
/// - Attr align_corners: If true, the centers of the 4 corner pixels of the input and output tensors are
///   aligned, preserving the values at the corner pixels. Defaults to false.
///
/// - Output resized_images: 4-D with shape
///   `[batch, new_height, new_width, channels]`.
@inlinable @inline(__always)
public static func resizeBicubic<T: Numeric & TensorFlowScalar>(
  images: Tensor<T>,
  size: Tensor<Int32>,
  alignCorners: Bool = false,
  halfPixelCenters: Bool = false
) -> Tensor<Float> {
  let ret: TensorHandle<Float> = #tfop("ResizeBicubic",
    images,
    size,
    T$dtype: T.tensorFlowDataType,
    align_corners: alignCorners,
    half_pixel_centers: halfPixelCenters)
  return Tensor(handle: ret)
}

/// Computes the gradient of bicubic interpolation.
///
/// - Parameters:
///   - grads: 4-D with shape `[batch, height, width, channels]`.
///   - original_image: 4-D with shape `[batch, orig_height, orig_width, channels]`,
///     The image tensor that was resized.
///
/// - Attr align_corners: If true, the centers of the 4 corner pixels of the input and grad tensors are
///   aligned. Defaults to false.
///
/// - Output output: 4-D with shape `[batch, orig_height, orig_width, channels]`.
///   Gradients with respect to the input image. Input image must have been
///   float or double.
@inlinable @inline(__always)
public static func resizeBicubicGrad<T: FloatingPoint & TensorFlowScalar>(
  grads: Tensor<Float>,
  originalImage: Tensor<T>,
  alignCorners: Bool = false,
  halfPixelCenters: Bool = false
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("ResizeBicubicGrad",
    grads,
    originalImage,
    T$dtype: T.tensorFlowDataType,
    align_corners: alignCorners,
    half_pixel_centers: halfPixelCenters)
  return Tensor(handle: ret)
}

/// Resize `images` to `size` using bilinear interpolation.
///
/// Input images can be of different types but output images are always float.
///
/// - Parameters:
///   - images: 4-D with shape `[batch, height, width, channels]`.
///   - size: = A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
///     new size for the images.
///
/// - Attr align_corners: If true, the centers of the 4 corner pixels of the input and output tensors are
///   aligned, preserving the values at the corner pixels. Defaults to false.
///
/// - Output resized_images: 4-D with shape
///   `[batch, new_height, new_width, channels]`.
@inlinable @inline(__always)
public static func resizeBilinear<T: Numeric & TensorFlowScalar>(
  images: Tensor<T>,
  size: Tensor<Int32>,
  alignCorners: Bool = false,
  halfPixelCenters: Bool = false
) -> Tensor<Float> {
  let ret: TensorHandle<Float> = #tfop("ResizeBilinear",
    images,
    size,
    T$dtype: T.tensorFlowDataType,
    align_corners: alignCorners,
    half_pixel_centers: halfPixelCenters)
  return Tensor(handle: ret)
}

/// Computes the gradient of bilinear interpolation.
///
/// - Parameters:
///   - grads: 4-D with shape `[batch, height, width, channels]`.
///   - original_image: 4-D with shape `[batch, orig_height, orig_width, channels]`,
///     The image tensor that was resized.
///
/// - Attr align_corners: If true, the centers of the 4 corner pixels of the input and grad tensors are
///   aligned. Defaults to false.
///
/// - Output output: 4-D with shape `[batch, orig_height, orig_width, channels]`.
///   Gradients with respect to the input image. Input image must have been
///   float or double.
@inlinable @inline(__always)
public static func resizeBilinearGrad<T: FloatingPoint & TensorFlowScalar>(
  grads: Tensor<Float>,
  originalImage: Tensor<T>,
  alignCorners: Bool = false,
  halfPixelCenters: Bool = false
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("ResizeBilinearGrad",
    grads,
    originalImage,
    T$dtype: T.tensorFlowDataType,
    align_corners: alignCorners,
    half_pixel_centers: halfPixelCenters)
  return Tensor(handle: ret)
}

/// Resize `images` to `size` using nearest neighbor interpolation.
///
/// - Parameters:
///   - images: 4-D with shape `[batch, height, width, channels]`.
///   - size: = A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
///     new size for the images.
///
/// - Attr align_corners: If true, the centers of the 4 corner pixels of the input and output tensors are
///   aligned, preserving the values at the corner pixels. Defaults to false.
///
/// - Output resized_images: 4-D with shape
///   `[batch, new_height, new_width, channels]`.
@inlinable @inline(__always)
public static func resizeNearestNeighbor<T: Numeric & TensorFlowScalar>(
  images: Tensor<T>,
  size: Tensor<Int32>,
  alignCorners: Bool = false,
  halfPixelCenters: Bool = false
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("ResizeNearestNeighbor",
    images,
    size,
    T$dtype: T.tensorFlowDataType,
    align_corners: alignCorners,
    half_pixel_centers: halfPixelCenters)
  return Tensor(handle: ret)
}

/// Computes the gradient of nearest neighbor interpolation.
///
/// - Parameters:
///   - grads: 4-D with shape `[batch, height, width, channels]`.
///   - size: = A 1-D int32 Tensor of 2 elements: `orig_height, orig_width`. The
///     original input size.
///
/// - Attr align_corners: If true, the centers of the 4 corner pixels of the input and grad tensors are
///   aligned. Defaults to false.
///
/// - Output output: 4-D with shape `[batch, orig_height, orig_width, channels]`. Gradients
///   with respect to the input image.
@inlinable @inline(__always)
public static func resizeNearestNeighborGrad<T: Numeric & TensorFlowScalar>(
  grads: Tensor<T>,
  size: Tensor<Int32>,
  alignCorners: Bool = false,
  halfPixelCenters: Bool = false
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("ResizeNearestNeighborGrad",
    grads,
    size,
    T$dtype: T.tensorFlowDataType,
    align_corners: alignCorners,
    half_pixel_centers: halfPixelCenters)
  return Tensor(handle: ret)
}

/// Restores a tensor from checkpoint files.
///
/// Reads a tensor stored in one or several files. If there are several files (for
/// instance because a tensor was saved as slices), `file_pattern` may contain
/// wildcard symbols (`*` and `?`) in the filename portion only, not in the
/// directory portion.
///
/// If a `file_pattern` matches several files, `preferred_shard` can be used to hint
/// in which file the requested tensor is likely to be found. This op will first
/// open the file at index `preferred_shard` in the list of matching files and try
/// to restore tensors from that file.  Only if some tensors or tensor slices are
/// not found in that first file, then the Op opens all the files. Setting
/// `preferred_shard` to match the value passed as the `shard` input
/// of a matching `Save` Op may speed up Restore.  This attribute only affects
/// performance, not correctness.  The default value -1 means files are processed in
/// order.
///
/// See also `RestoreSlice`.
///
/// - Parameters:
///   - file_pattern: Must have a single element. The pattern of the files from
///     which we read the tensor.
///   - tensor_name: Must have a single element. The name of the tensor to be
///     restored.
///
/// - Attrs:
///   - dt: The type of the tensor to be restored.
///   - preferred_shard: Index of file to open first if multiple files match
///     `file_pattern`.
///
/// - Output tensor: The restored tensor.
@inlinable @inline(__always)
public static func restore<Dt: TensorFlowScalar>(
  filePattern: StringTensor,
  tensorName: StringTensor,
  preferredShard: Int64 = -1
) -> Tensor<Dt> {
  let ret: TensorHandle<Dt> = #tfop("Restore",
    filePattern,
    tensorName,
    dt$dtype: Dt.tensorFlowDataType,
    preferred_shard: preferredShard)
  return Tensor(handle: ret)
}

/// Restores a tensor from checkpoint files.
///
/// This is like `Restore` except that restored tensor can be listed as filling
/// only a slice of a larger tensor.  `shape_and_slice` specifies the shape of the
/// larger tensor and the slice that the restored tensor covers.
///
/// The `shape_and_slice` input has the same format as the
/// elements of the `shapes_and_slices` input of the `SaveSlices` op.
///
/// - Parameters:
///   - file_pattern: Must have a single element. The pattern of the files from
///     which we read the tensor.
///   - tensor_name: Must have a single element. The name of the tensor to be
///     restored.
///   - shape_and_slice: Scalar. The shapes and slice specifications to use when
///     restoring a tensors.
///
/// - Attrs:
///   - dt: The type of the tensor to be restored.
///   - preferred_shard: Index of file to open first if multiple files match
///     `file_pattern`. See the documentation for `Restore`.
///
/// - Output tensor: The restored tensor.
@inlinable @inline(__always)
public static func restoreSlice<Dt: TensorFlowScalar>(
  filePattern: StringTensor,
  tensorName: StringTensor,
  shapeAndSlice: StringTensor,
  preferredShard: Int64 = -1
) -> Tensor<Dt> {
  let ret: TensorHandle<Dt> = #tfop("RestoreSlice",
    filePattern,
    tensorName,
    shapeAndSlice,
    dt$dtype: Dt.tensorFlowDataType,
    preferred_shard: preferredShard)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func restrict<T: TensorFlowScalar>(
  _ a: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Restrict",
    a,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Retrieve ADAM embedding parameters.
///
/// An op that retrieves optimization parameters from embedding to host
/// memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
/// the correct embedding table configuration. For example, this op is
/// used to retrieve updated parameters before saving a checkpoint.
///
/// - Outputs:
///   - parameters: Parameter parameters updated by the ADAM optimization algorithm.
///   - momenta: Parameter momenta updated by the ADAM optimization algorithm.
///   - velocities: Parameter velocities updated by the ADAM optimization algorithm.
@inlinable @inline(__always)
public static func retrieveTPUEmbeddingADAMParameters(
  tableId: Int64 = -1,
  tableName: String,
  numShards: Int64,
  shardId: Int64
) -> (parameters: Tensor<Float>, momenta: Tensor<Float>, velocities: Tensor<Float>) {
  let ret: (TensorHandle<Float>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("RetrieveTPUEmbeddingADAMParameters",
    table_id: tableId,
    table_name: tableName,
    num_shards: numShards,
    shard_id: shardId)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Retrieve ADAM embedding parameters with debug support.
///
/// An op that retrieves optimization parameters from embedding to host
/// memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
/// the correct embedding table configuration. For example, this op is
/// used to retrieve updated parameters before saving a checkpoint.
///
/// - Outputs:
///   - parameters: Parameter parameters updated by the ADAM optimization algorithm.
///   - momenta: Parameter momenta updated by the ADAM optimization algorithm.
///   - velocities: Parameter velocities updated by the ADAM optimization algorithm.
///   - gradient_accumulators: Parameter gradient_accumulators updated by the ADAM optimization algorithm.
@inlinable @inline(__always)
public static func retrieveTPUEmbeddingADAMParametersGradAccumDebug(
  tableId: Int64 = -1,
  tableName: String,
  numShards: Int64,
  shardId: Int64
) -> (parameters: Tensor<Float>, momenta: Tensor<Float>, velocities: Tensor<Float>, gradientAccumulators: Tensor<Float>) {
  let ret: (TensorHandle<Float>, TensorHandle<Float>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("RetrieveTPUEmbeddingADAMParametersGradAccumDebug",
    table_id: tableId,
    table_name: tableName,
    num_shards: numShards,
    shard_id: shardId)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2), Tensor(handle: ret.3))
}

/// Retrieve Adadelta embedding parameters.
///
/// An op that retrieves optimization parameters from embedding to host
/// memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
/// the correct embedding table configuration. For example, this op is
/// used to retrieve updated parameters before saving a checkpoint.
///
/// - Outputs:
///   - parameters: Parameter parameters updated by the Adadelta optimization algorithm.
///   - accumulators: Parameter accumulators updated by the Adadelta optimization algorithm.
///   - updates: Parameter updates updated by the Adadelta optimization algorithm.
@inlinable @inline(__always)
public static func retrieveTPUEmbeddingAdadeltaParameters(
  tableId: Int64 = -1,
  tableName: String,
  numShards: Int64,
  shardId: Int64
) -> (parameters: Tensor<Float>, accumulators: Tensor<Float>, updates: Tensor<Float>) {
  let ret: (TensorHandle<Float>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("RetrieveTPUEmbeddingAdadeltaParameters",
    table_id: tableId,
    table_name: tableName,
    num_shards: numShards,
    shard_id: shardId)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Retrieve Adadelta embedding parameters with debug support.
///
/// An op that retrieves optimization parameters from embedding to host
/// memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
/// the correct embedding table configuration. For example, this op is
/// used to retrieve updated parameters before saving a checkpoint.
///
/// - Outputs:
///   - parameters: Parameter parameters updated by the Adadelta optimization algorithm.
///   - accumulators: Parameter accumulators updated by the Adadelta optimization algorithm.
///   - updates: Parameter updates updated by the Adadelta optimization algorithm.
///   - gradient_accumulators: Parameter gradient_accumulators updated by the Adadelta optimization algorithm.
@inlinable @inline(__always)
public static func retrieveTPUEmbeddingAdadeltaParametersGradAccumDebug(
  tableId: Int64 = -1,
  tableName: String,
  numShards: Int64,
  shardId: Int64
) -> (parameters: Tensor<Float>, accumulators: Tensor<Float>, updates: Tensor<Float>, gradientAccumulators: Tensor<Float>) {
  let ret: (TensorHandle<Float>, TensorHandle<Float>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("RetrieveTPUEmbeddingAdadeltaParametersGradAccumDebug",
    table_id: tableId,
    table_name: tableName,
    num_shards: numShards,
    shard_id: shardId)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2), Tensor(handle: ret.3))
}

/// Retrieve Adagrad embedding parameters.
///
/// An op that retrieves optimization parameters from embedding to host
/// memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
/// the correct embedding table configuration. For example, this op is
/// used to retrieve updated parameters before saving a checkpoint.
///
/// - Outputs:
///   - parameters: Parameter parameters updated by the Adagrad optimization algorithm.
///   - accumulators: Parameter accumulators updated by the Adagrad optimization algorithm.
@inlinable @inline(__always)
public static func retrieveTPUEmbeddingAdagradParameters(
  tableId: Int64 = -1,
  tableName: String,
  numShards: Int64,
  shardId: Int64
) -> (parameters: Tensor<Float>, accumulators: Tensor<Float>) {
  let ret: (TensorHandle<Float>, TensorHandle<Float>) = #tfop("RetrieveTPUEmbeddingAdagradParameters",
    table_id: tableId,
    table_name: tableName,
    num_shards: numShards,
    shard_id: shardId)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

/// Retrieve Adagrad embedding parameters with debug support.
///
/// An op that retrieves optimization parameters from embedding to host
/// memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
/// the correct embedding table configuration. For example, this op is
/// used to retrieve updated parameters before saving a checkpoint.
///
/// - Outputs:
///   - parameters: Parameter parameters updated by the Adagrad optimization algorithm.
///   - accumulators: Parameter accumulators updated by the Adagrad optimization algorithm.
///   - gradient_accumulators: Parameter gradient_accumulators updated by the Adagrad optimization algorithm.
@inlinable @inline(__always)
public static func retrieveTPUEmbeddingAdagradParametersGradAccumDebug(
  tableId: Int64 = -1,
  tableName: String,
  numShards: Int64,
  shardId: Int64
) -> (parameters: Tensor<Float>, accumulators: Tensor<Float>, gradientAccumulators: Tensor<Float>) {
  let ret: (TensorHandle<Float>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("RetrieveTPUEmbeddingAdagradParametersGradAccumDebug",
    table_id: tableId,
    table_name: tableName,
    num_shards: numShards,
    shard_id: shardId)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Retrieve centered RMSProp embedding parameters.
///
/// An op that retrieves optimization parameters from embedding to host
/// memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
/// the correct embedding table configuration. For example, this op is
/// used to retrieve updated parameters before saving a checkpoint.
///
/// - Outputs:
///   - parameters: Parameter parameters updated by the centered RMSProp optimization algorithm.
///   - ms: Parameter ms updated by the centered RMSProp optimization algorithm.
///   - mom: Parameter mom updated by the centered RMSProp optimization algorithm.
///   - mg: Parameter mg updated by the centered RMSProp optimization algorithm.
@inlinable @inline(__always)
public static func retrieveTPUEmbeddingCenteredRMSPropParameters(
  tableId: Int64 = -1,
  tableName: String,
  numShards: Int64,
  shardId: Int64
) -> (parameters: Tensor<Float>, ms: Tensor<Float>, mom: Tensor<Float>, mg: Tensor<Float>) {
  let ret: (TensorHandle<Float>, TensorHandle<Float>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("RetrieveTPUEmbeddingCenteredRMSPropParameters",
    table_id: tableId,
    table_name: tableName,
    num_shards: numShards,
    shard_id: shardId)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2), Tensor(handle: ret.3))
}

/// Retrieve FTRL embedding parameters.
///
/// An op that retrieves optimization parameters from embedding to host
/// memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
/// the correct embedding table configuration. For example, this op is
/// used to retrieve updated parameters before saving a checkpoint.
///
/// - Outputs:
///   - parameters: Parameter parameters updated by the FTRL optimization algorithm.
///   - accumulators: Parameter accumulators updated by the FTRL optimization algorithm.
///   - linears: Parameter linears updated by the FTRL optimization algorithm.
@inlinable @inline(__always)
public static func retrieveTPUEmbeddingFTRLParameters(
  tableId: Int64 = -1,
  tableName: String,
  numShards: Int64,
  shardId: Int64
) -> (parameters: Tensor<Float>, accumulators: Tensor<Float>, linears: Tensor<Float>) {
  let ret: (TensorHandle<Float>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("RetrieveTPUEmbeddingFTRLParameters",
    table_id: tableId,
    table_name: tableName,
    num_shards: numShards,
    shard_id: shardId)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Retrieve FTRL embedding parameters with debug support.
///
/// An op that retrieves optimization parameters from embedding to host
/// memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
/// the correct embedding table configuration. For example, this op is
/// used to retrieve updated parameters before saving a checkpoint.
///
/// - Outputs:
///   - parameters: Parameter parameters updated by the FTRL optimization algorithm.
///   - accumulators: Parameter accumulators updated by the FTRL optimization algorithm.
///   - linears: Parameter linears updated by the FTRL optimization algorithm.
///   - gradient_accumulators: Parameter gradient_accumulators updated by the FTRL optimization algorithm.
@inlinable @inline(__always)
public static func retrieveTPUEmbeddingFTRLParametersGradAccumDebug(
  tableId: Int64 = -1,
  tableName: String,
  numShards: Int64,
  shardId: Int64
) -> (parameters: Tensor<Float>, accumulators: Tensor<Float>, linears: Tensor<Float>, gradientAccumulators: Tensor<Float>) {
  let ret: (TensorHandle<Float>, TensorHandle<Float>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("RetrieveTPUEmbeddingFTRLParametersGradAccumDebug",
    table_id: tableId,
    table_name: tableName,
    num_shards: numShards,
    shard_id: shardId)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2), Tensor(handle: ret.3))
}

/// Retrieve MDL Adagrad Light embedding parameters.
///
/// An op that retrieves optimization parameters from embedding to host
/// memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
/// the correct embedding table configuration. For example, this op is
/// used to retrieve updated parameters before saving a checkpoint.
///
/// - Outputs:
///   - parameters: Parameter parameters updated by the MDL Adagrad Light optimization algorithm.
///   - accumulators: Parameter accumulators updated by the MDL Adagrad Light optimization algorithm.
///   - weights: Parameter weights updated by the MDL Adagrad Light optimization algorithm.
///   - benefits: Parameter benefits updated by the MDL Adagrad Light optimization algorithm.
@inlinable @inline(__always)
public static func retrieveTPUEmbeddingMDLAdagradLightParameters(
  tableId: Int64 = -1,
  tableName: String,
  numShards: Int64,
  shardId: Int64
) -> (parameters: Tensor<Float>, accumulators: Tensor<Float>, weights: Tensor<Float>, benefits: Tensor<Float>) {
  let ret: (TensorHandle<Float>, TensorHandle<Float>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("RetrieveTPUEmbeddingMDLAdagradLightParameters",
    table_id: tableId,
    table_name: tableName,
    num_shards: numShards,
    shard_id: shardId)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2), Tensor(handle: ret.3))
}

/// Retrieve Momentum embedding parameters.
///
/// An op that retrieves optimization parameters from embedding to host
/// memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
/// the correct embedding table configuration. For example, this op is
/// used to retrieve updated parameters before saving a checkpoint.
///
/// - Outputs:
///   - parameters: Parameter parameters updated by the Momentum optimization algorithm.
///   - momenta: Parameter momenta updated by the Momentum optimization algorithm.
@inlinable @inline(__always)
public static func retrieveTPUEmbeddingMomentumParameters(
  tableId: Int64 = -1,
  tableName: String,
  numShards: Int64,
  shardId: Int64
) -> (parameters: Tensor<Float>, momenta: Tensor<Float>) {
  let ret: (TensorHandle<Float>, TensorHandle<Float>) = #tfop("RetrieveTPUEmbeddingMomentumParameters",
    table_id: tableId,
    table_name: tableName,
    num_shards: numShards,
    shard_id: shardId)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

/// Retrieve Momentum embedding parameters with debug support.
///
/// An op that retrieves optimization parameters from embedding to host
/// memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
/// the correct embedding table configuration. For example, this op is
/// used to retrieve updated parameters before saving a checkpoint.
///
/// - Outputs:
///   - parameters: Parameter parameters updated by the Momentum optimization algorithm.
///   - momenta: Parameter momenta updated by the Momentum optimization algorithm.
///   - gradient_accumulators: Parameter gradient_accumulators updated by the Momentum optimization algorithm.
@inlinable @inline(__always)
public static func retrieveTPUEmbeddingMomentumParametersGradAccumDebug(
  tableId: Int64 = -1,
  tableName: String,
  numShards: Int64,
  shardId: Int64
) -> (parameters: Tensor<Float>, momenta: Tensor<Float>, gradientAccumulators: Tensor<Float>) {
  let ret: (TensorHandle<Float>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("RetrieveTPUEmbeddingMomentumParametersGradAccumDebug",
    table_id: tableId,
    table_name: tableName,
    num_shards: numShards,
    shard_id: shardId)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Retrieve proximal Adagrad embedding parameters.
///
/// An op that retrieves optimization parameters from embedding to host
/// memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
/// the correct embedding table configuration. For example, this op is
/// used to retrieve updated parameters before saving a checkpoint.
///
/// - Outputs:
///   - parameters: Parameter parameters updated by the proximal Adagrad optimization algorithm.
///   - accumulators: Parameter accumulators updated by the proximal Adagrad optimization algorithm.
@inlinable @inline(__always)
public static func retrieveTPUEmbeddingProximalAdagradParameters(
  tableId: Int64 = -1,
  tableName: String,
  numShards: Int64,
  shardId: Int64
) -> (parameters: Tensor<Float>, accumulators: Tensor<Float>) {
  let ret: (TensorHandle<Float>, TensorHandle<Float>) = #tfop("RetrieveTPUEmbeddingProximalAdagradParameters",
    table_id: tableId,
    table_name: tableName,
    num_shards: numShards,
    shard_id: shardId)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

/// Retrieve proximal Adagrad embedding parameters with debug support.
///
/// An op that retrieves optimization parameters from embedding to host
/// memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
/// the correct embedding table configuration. For example, this op is
/// used to retrieve updated parameters before saving a checkpoint.
///
/// - Outputs:
///   - parameters: Parameter parameters updated by the proximal Adagrad optimization algorithm.
///   - accumulators: Parameter accumulators updated by the proximal Adagrad optimization algorithm.
///   - gradient_accumulators: Parameter gradient_accumulators updated by the proximal Adagrad optimization algorithm.
@inlinable @inline(__always)
public static func retrieveTPUEmbeddingProximalAdagradParametersGradAccumDebug(
  tableId: Int64 = -1,
  tableName: String,
  numShards: Int64,
  shardId: Int64
) -> (parameters: Tensor<Float>, accumulators: Tensor<Float>, gradientAccumulators: Tensor<Float>) {
  let ret: (TensorHandle<Float>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("RetrieveTPUEmbeddingProximalAdagradParametersGradAccumDebug",
    table_id: tableId,
    table_name: tableName,
    num_shards: numShards,
    shard_id: shardId)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Retrieve RMSProp embedding parameters.
///
/// An op that retrieves optimization parameters from embedding to host
/// memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
/// the correct embedding table configuration. For example, this op is
/// used to retrieve updated parameters before saving a checkpoint.
///
/// - Outputs:
///   - parameters: Parameter parameters updated by the RMSProp optimization algorithm.
///   - ms: Parameter ms updated by the RMSProp optimization algorithm.
///   - mom: Parameter mom updated by the RMSProp optimization algorithm.
@inlinable @inline(__always)
public static func retrieveTPUEmbeddingRMSPropParameters(
  tableId: Int64 = -1,
  tableName: String,
  numShards: Int64,
  shardId: Int64
) -> (parameters: Tensor<Float>, ms: Tensor<Float>, mom: Tensor<Float>) {
  let ret: (TensorHandle<Float>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("RetrieveTPUEmbeddingRMSPropParameters",
    table_id: tableId,
    table_name: tableName,
    num_shards: numShards,
    shard_id: shardId)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Retrieve RMSProp embedding parameters with debug support.
///
/// An op that retrieves optimization parameters from embedding to host
/// memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
/// the correct embedding table configuration. For example, this op is
/// used to retrieve updated parameters before saving a checkpoint.
///
/// - Outputs:
///   - parameters: Parameter parameters updated by the RMSProp optimization algorithm.
///   - ms: Parameter ms updated by the RMSProp optimization algorithm.
///   - mom: Parameter mom updated by the RMSProp optimization algorithm.
///   - gradient_accumulators: Parameter gradient_accumulators updated by the RMSProp optimization algorithm.
@inlinable @inline(__always)
public static func retrieveTPUEmbeddingRMSPropParametersGradAccumDebug(
  tableId: Int64 = -1,
  tableName: String,
  numShards: Int64,
  shardId: Int64
) -> (parameters: Tensor<Float>, ms: Tensor<Float>, mom: Tensor<Float>, gradientAccumulators: Tensor<Float>) {
  let ret: (TensorHandle<Float>, TensorHandle<Float>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("RetrieveTPUEmbeddingRMSPropParametersGradAccumDebug",
    table_id: tableId,
    table_name: tableName,
    num_shards: numShards,
    shard_id: shardId)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2), Tensor(handle: ret.3))
}

/// Retrieve SGD embedding parameters.
///
/// An op that retrieves optimization parameters from embedding to host
/// memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
/// the correct embedding table configuration. For example, this op is
/// used to retrieve updated parameters before saving a checkpoint.
///
/// - Output parameters: Parameter parameters updated by the stochastic gradient descent optimization algorithm.
@inlinable @inline(__always)
public static func retrieveTPUEmbeddingStochasticGradientDescentParameters(
  tableId: Int64 = -1,
  tableName: String,
  numShards: Int64,
  shardId: Int64
) -> Tensor<Float> {
  let ret: TensorHandle<Float> = #tfop("RetrieveTPUEmbeddingStochasticGradientDescentParameters",
    table_id: tableId,
    table_name: tableName,
    num_shards: numShards,
    shard_id: shardId)
  return Tensor(handle: ret)
}

/// Reverses specific dimensions of a tensor.
///
/// Given a `tensor`, and a `bool` tensor `dims` representing the dimensions
/// of `tensor`, this operation reverses each dimension i of `tensor` where
/// `dims[i]` is `True`.
///
/// `tensor` can have up to 8 dimensions. The number of dimensions
/// of `tensor` must equal the number of elements in `dims`. In other words:
///
/// `rank(tensor) = size(dims)`
///
/// For example:
///
/// ```
/// # tensor 't' is [[[[ 0,  1,  2,  3],
/// #                  [ 4,  5,  6,  7],
/// #                  [ 8,  9, 10, 11]],
/// #                 [[12, 13, 14, 15],
/// #                  [16, 17, 18, 19],
/// #                  [20, 21, 22, 23]]]]
/// # tensor 't' shape is [1, 2, 3, 4]
///
/// # 'dims' is [False, False, False, True]
/// reverse(t, dims) ==> [[[[ 3,  2,  1,  0],
///                         [ 7,  6,  5,  4],
///                         [ 11, 10, 9, 8]],
///                        [[15, 14, 13, 12],
///                         [19, 18, 17, 16],
///                         [23, 22, 21, 20]]]]
///
/// # 'dims' is [False, True, False, False]
/// reverse(t, dims) ==> [[[[12, 13, 14, 15],
///                         [16, 17, 18, 19],
///                         [20, 21, 22, 23]
///                        [[ 0,  1,  2,  3],
///                         [ 4,  5,  6,  7],
///                         [ 8,  9, 10, 11]]]]
///
/// # 'dims' is [False, False, True, False]
/// reverse(t, dims) ==> [[[[8, 9, 10, 11],
///                         [4, 5, 6, 7],
///                         [0, 1, 2, 3]]
///                        [[20, 21, 22, 23],
///                         [16, 17, 18, 19],
///                         [12, 13, 14, 15]]]]
/// ```
///
/// - Parameters:
///   - tensor: Up to 8-D.
///   - dims: 1-D. The dimensions to reverse.
///
/// - Output output: The same shape as `tensor`.
@inlinable @inline(__always)
public static func reverse<T: TensorFlowScalar>(
  _ tensor: Tensor<T>,
  dims: Tensor<Bool>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Reverse",
    tensor,
    dims,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Reverses variable length slices.
///
/// This op first slices `input` along the dimension `batch_dim`, and for each
/// slice `i`, reverses the first `seq_lengths[i]` elements along
/// the dimension `seq_dim`.
///
/// The elements of `seq_lengths` must obey `seq_lengths[i] <= input.dims[seq_dim]`,
/// and `seq_lengths` must be a vector of length `input.dims[batch_dim]`.
///
/// The output slice `i` along dimension `batch_dim` is then given by input
/// slice `i`, with the first `seq_lengths[i]` slices along dimension
/// `seq_dim` reversed.
///
/// For example:
///
/// ```
/// # Given this:
/// batch_dim = 0
/// seq_dim = 1
/// input.dims = (4, 8, ...)
/// seq_lengths = [7, 2, 3, 5]
///
/// # then slices of input are reversed on seq_dim, but only up to seq_lengths:
/// output[0, 0:7, :, ...] = input[0, 7:0:-1, :, ...]
/// output[1, 0:2, :, ...] = input[1, 2:0:-1, :, ...]
/// output[2, 0:3, :, ...] = input[2, 3:0:-1, :, ...]
/// output[3, 0:5, :, ...] = input[3, 5:0:-1, :, ...]
///
/// # while entries past seq_lens are copied through:
/// output[0, 7:, :, ...] = input[0, 7:, :, ...]
/// output[1, 2:, :, ...] = input[1, 2:, :, ...]
/// output[2, 3:, :, ...] = input[2, 3:, :, ...]
/// output[3, 2:, :, ...] = input[3, 2:, :, ...]
/// ```
///
/// In contrast, if:
///
/// ```
/// # Given this:
/// batch_dim = 2
/// seq_dim = 0
/// input.dims = (8, ?, 4, ...)
/// seq_lengths = [7, 2, 3, 5]
///
/// # then slices of input are reversed on seq_dim, but only up to seq_lengths:
/// output[0:7, :, 0, :, ...] = input[7:0:-1, :, 0, :, ...]
/// output[0:2, :, 1, :, ...] = input[2:0:-1, :, 1, :, ...]
/// output[0:3, :, 2, :, ...] = input[3:0:-1, :, 2, :, ...]
/// output[0:5, :, 3, :, ...] = input[5:0:-1, :, 3, :, ...]
///
/// # while entries past seq_lens are copied through:
/// output[7:, :, 0, :, ...] = input[7:, :, 0, :, ...]
/// output[2:, :, 1, :, ...] = input[2:, :, 1, :, ...]
/// output[3:, :, 2, :, ...] = input[3:, :, 2, :, ...]
/// output[2:, :, 3, :, ...] = input[2:, :, 3, :, ...]
/// ```
///
/// - Parameters:
///   - input: The input to reverse.
///   - seq_lengths: 1-D with length `input.dims(batch_dim)` and
///     `max(seq_lengths) <= input.dims(seq_dim)`
///
/// - Attrs:
///   - seq_dim: The dimension which is partially reversed.
///   - batch_dim: The dimension along which reversal is performed.
///
/// - Output output: The partially reversed input. It has the same shape as `input`.
@inlinable @inline(__always)
public static func reverseSequence<T: TensorFlowScalar, Tlen: BinaryInteger & TensorFlowScalar>(
  _ input: Tensor<T>,
  seqLengths: Tensor<Tlen>,
  seqDim: Int64,
  batchDim: Int64 = 0
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("ReverseSequence",
    input,
    seqLengths,
    T$dtype: T.tensorFlowDataType,
    Tlen$dtype: Tlen.tensorFlowDataType,
    seq_dim: seqDim,
    batch_dim: batchDim)
  return Tensor(handle: ret)
}

/// Reverses specific dimensions of a tensor.
///
/// NOTE `tf.reverse` has now changed behavior in preparation for 1.0.
/// `tf.reverse_v2` is currently an alias that will be deprecated before TF 1.0.
///
/// Given a `tensor`, and a `int32` tensor `axis` representing the set of
/// dimensions of `tensor` to reverse. This operation reverses each dimension
/// `i` for which there exists `j` s.t. `axis[j] == i`.
///
/// `tensor` can have up to 8 dimensions. The number of dimensions specified
/// in `axis` may be 0 or more entries. If an index is specified more than
/// once, a InvalidArgument error is raised.
///
/// For example:
///
/// ```
/// # tensor 't' is [[[[ 0,  1,  2,  3],
/// #                  [ 4,  5,  6,  7],
/// #                  [ 8,  9, 10, 11]],
/// #                 [[12, 13, 14, 15],
/// #                  [16, 17, 18, 19],
/// #                  [20, 21, 22, 23]]]]
/// # tensor 't' shape is [1, 2, 3, 4]
///
/// # 'dims' is [3] or 'dims' is [-1]
/// reverse(t, dims) ==> [[[[ 3,  2,  1,  0],
///                         [ 7,  6,  5,  4],
///                         [ 11, 10, 9, 8]],
///                        [[15, 14, 13, 12],
///                         [19, 18, 17, 16],
///                         [23, 22, 21, 20]]]]
///
/// # 'dims' is '[1]' (or 'dims' is '[-3]')
/// reverse(t, dims) ==> [[[[12, 13, 14, 15],
///                         [16, 17, 18, 19],
///                         [20, 21, 22, 23]
///                        [[ 0,  1,  2,  3],
///                         [ 4,  5,  6,  7],
///                         [ 8,  9, 10, 11]]]]
///
/// # 'dims' is '[2]' (or 'dims' is '[-2]')
/// reverse(t, dims) ==> [[[[8, 9, 10, 11],
///                         [4, 5, 6, 7],
///                         [0, 1, 2, 3]]
///                        [[20, 21, 22, 23],
///                         [16, 17, 18, 19],
///                         [12, 13, 14, 15]]]]
/// ```
///
/// - Parameters:
///   - tensor: Up to 8-D.
///   - axis: 1-D. The indices of the dimensions to reverse. Must be in the range
///     `[-rank(tensor), rank(tensor))`.
///
/// - Output output: The same shape as `tensor`.
@inlinable @inline(__always)
public static func reverseV2<Tidx: BinaryInteger & TensorFlowScalar, T: TensorFlowScalar>(
  _ tensor: Tensor<T>,
  axis: Tensor<Tidx>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("ReverseV2",
    tensor,
    axis,
    Tidx$dtype: Tidx.tensorFlowDataType,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Elementwise computes the bitwise right-shift of `x` and `y`.
///
/// Performs a logical shift for unsigned integer types, and an arithmetic shift
/// for signed integer types.
///
/// If `y` is negative, or greater than or equal to than the width of `x` in bits
/// the result is implementation defined.
@inlinable @inline(__always)
public static func rightShift<T: BinaryInteger & TensorFlowScalar>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("RightShift",
    x,
    y,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Returns element-wise integer closest to x.
///
/// If the result is midway between two representable values,
/// the even representable is chosen.
/// For example:
///
/// ```
/// rint(-1.5) ==> -2.0
/// rint(0.5000001) ==> 1.0
/// rint([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]) ==> [-2., -2., -0., 0., 2., 2., 2.]
/// ```
@inlinable @inline(__always)
public static func rint<T: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Rint",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Rolls the elements of a tensor along an axis.
///
/// The elements are shifted positively (towards larger indices) by the offset of
/// `shift` along the dimension of `axis`. Negative `shift` values will shift
/// elements in the opposite direction. Elements that roll passed the last position
/// will wrap around to the first and vice versa. Multiple shifts along multiple
/// axes may be specified.
///
/// For example:
///
/// ```
/// # 't' is [0, 1, 2, 3, 4]
/// roll(t, shift=2, axis=0) ==> [3, 4, 0, 1, 2]
///
/// # shifting along multiple dimensions
/// # 't' is [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
/// roll(t, shift=[1, -2], axis=[0, 1]) ==> [[7, 8, 9, 5, 6], [2, 3, 4, 0, 1]]
///
/// # shifting along the same axis multiple times
/// # 't' is [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
/// roll(t, shift=[2, -3], axis=[1, 1]) ==> [[1, 2, 3, 4, 0], [6, 7, 8, 9, 5]]
/// ```
///
/// - Parameters:
///   - shift: Dimension must be 0-D or 1-D. `shift[i]` specifies the number of places by which
///     elements are shifted positively (towards larger indices) along the dimension
///     specified by `axis[i]`. Negative shifts will roll the elements in the opposite
///     direction.
///   - axis: Dimension must be 0-D or 1-D. `axis[i]` specifies the dimension that the shift
///     `shift[i]` should occur. If the same axis is referenced more than once, the
///     total shift for that axis will be the sum of all the shifts that belong to that
///     axis.
///
/// - Output output: Has the same shape and size as the input. The elements are shifted
///   positively (towards larger indices) by the offsets of `shift` along the
///   dimensions of `axis`.
@inlinable @inline(__always)
public static func roll<T: TensorFlowScalar, Tshift: BinaryInteger & TensorFlowScalar, Taxis: BinaryInteger & TensorFlowScalar>(
  _ input: Tensor<T>,
  shift: Tensor<Tshift>,
  axis: Tensor<Taxis>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Roll",
    input,
    shift,
    axis,
    T$dtype: T.tensorFlowDataType,
    Tshift$dtype: Tshift.tensorFlowDataType,
    Taxis$dtype: Taxis.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Rounds the values of a tensor to the nearest integer, element-wise.
///
/// Rounds half to even.  Also known as bankers rounding. If you want to round
/// according to the current system rounding mode use std::cint.
@inlinable @inline(__always)
public static func round<T: Numeric & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Round",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Perform batches of RPC requests.
///
/// This op asynchronously performs either a single RPC request, or a batch
/// of requests.  RPC requests are defined by three main parameters:
///
///   - `address` (the host+port or BNS address of the request)
///   - `method` (the RPC method name for the request)
///   - `request` (the serialized proto string, or vector of strings,
///      of the RPC request argument).
///
/// For example, if you have an RPC service running on port localhost:2345,
/// and its interface is configured with the following proto declaration:
///
/// ```
/// service MyService {
///   rpc MyMethod(MyRequestProto) returns (MyResponseProto) {
///   }
/// };
/// ```
///
/// then call this op with arguments:
///
/// ```
/// address = "localhost:2345"
/// method = "MyService/MyMethod"
/// ```
///
/// The `request` tensor is a string tensor representing serialized `MyRequestProto`
/// strings; and the output string tensor `response` will have the same shape
/// and contain (upon successful completion) corresponding serialized
/// `MyResponseProto` strings.
///
/// For example, to send a single, empty, `MyRequestProto`, call
/// this op with `request = ""`.  To send 5 **parallel** empty requests,
/// call this op with `request = ["", "", "", "", ""]`.
///
/// More generally, one can create a batch of `MyRequestProto` serialized protos
/// from regular batched tensors using the `encode_proto` op, and convert
/// the response `MyResponseProto` serialized protos to batched tensors
/// using the `decode_proto` op.
///
/// **NOTE** Working with serialized proto strings is faster than instantiating
/// actual proto objects in memory, so no performance degradation is expected
/// compared to writing custom kernels for this workflow.
///
/// If the connection fails or the remote worker returns an error
/// status, the op reraises this exception locally.
///
/// See the `TryRpc` op if you prefer to handle RPC failures manually in the graph.
///
/// - Parameters:
///   - address: `0-D` or `1-D`.  The address (i.e. host_name:port) of the RPC server.
///     If this tensor has more than 1 element, then multiple parallel rpc requests
///     are sent.  This argument broadcasts with `method` and `request`.
///   - method: `0-D` or `1-D`.  The method address on the RPC server.
///     If this tensor has more than 1 element, then multiple parallel rpc requests
///     are sent.  This argument broadcasts with `address` and `request`.
///   - request: `0-D` or `1-D`.  Serialized proto strings: the rpc request argument.
///     If this tensor has more than 1 element, then multiple parallel rpc requests
///     are sent.  This argument broadcasts with `address` and `method`.
///
/// - Attrs:
///   - protocol: RPC protocol to use.  Empty string means use the default protocol.
///     Options include 'grpc'.
///   - fail_fast: `boolean`. If `true` (default), then failures to connect
///     (i.e., the server does not immediately respond) cause an RPC failure.
///   - timeout_in_ms: `int`. If `0` (default), then the kernel will run the RPC
///     request and only time out if the RPC deadline passes or the session times out.
///     If this value is greater than `0`, then the op will raise an exception if
///     the RPC takes longer than `timeout_in_ms`.
///
/// - Output response: Same shape as `request`. Serialized proto strings: the rpc responses.
@inlinable @inline(__always)
public static func rpc(
  address: StringTensor,
  method: StringTensor,
  request: StringTensor,
  protocol_: String,
  failFast: Bool = true,
  timeoutInMs: Int64 = 0
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("Rpc",
    address,
    method,
    request,
    protocol: protocol_,
    fail_fast: failFast,
    timeout_in_ms: timeoutInMs)
  return StringTensor(handle: ret)
}

/// Computes reciprocal of square root of x element-wise.
///
/// I.e., \\(y = 1 / \sqrt{x}\\).
@inlinable @inline(__always)
public static func rsqrt<T: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Rsqrt",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the gradient for the rsqrt of `x` wrt its input.
///
/// Specifically, `grad = dy * -0.5 * y^3`, where `y = rsqrt(x)`, and `dy`
/// is the corresponding input gradient.
@inlinable @inline(__always)
public static func rsqrtGrad<T: FloatingPoint & TensorFlowScalar>(
  _ y: Tensor<T>,
  dy: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("RsqrtGrad",
    y,
    dy,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Generate a single randomly distorted bounding box for an image.
///
/// Bounding box annotations are often supplied in addition to ground-truth labels
/// in image recognition or object localization tasks. A common technique for
/// training such a system is to randomly distort an image while preserving
/// its content, i.e. *data augmentation*. This Op outputs a randomly distorted
/// localization of an object, i.e. bounding box, given an `image_size`,
/// `bounding_boxes` and a series of constraints.
///
/// The output of this Op is a single bounding box that may be used to crop the
/// original image. The output is returned as 3 tensors: `begin`, `size` and
/// `bboxes`. The first 2 tensors can be fed directly into `tf.slice` to crop the
/// image. The latter may be supplied to `tf.image.draw_bounding_boxes` to visualize
/// what the bounding box looks like.
///
/// Bounding boxes are supplied and returned as `[y_min, x_min, y_max, x_max]`. The
/// bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
/// height of the underlying image.
///
/// For example,
///
/// ```python
///     # Generate a single distorted bounding box.
///     begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
///         tf.shape(image),
///         bounding_boxes=bounding_boxes)
///
///     # Draw the bounding box in an image summary.
///     image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
///                                                   bbox_for_draw)
///     tf.summary.image('images_with_box', image_with_box)
///
///     # Employ the bounding box to distort the image.
///     distorted_image = tf.slice(image, begin, size)
/// ```
///
/// Note that if no bounding box information is available, setting
/// `use_image_if_no_bounding_boxes = true` will assume there is a single implicit
/// bounding box covering the whole image. If `use_image_if_no_bounding_boxes` is
/// false and no bounding boxes are supplied, an error is raised.
///
/// - Parameters:
///   - image_size: 1-D, containing `[height, width, channels]`.
///   - bounding_boxes: 3-D with shape `[batch, N, 4]` describing the N bounding boxes
///     associated with the image.
///
/// - Attrs:
///   - seed: If either `seed` or `seed2` are set to non-zero, the random number
///     generator is seeded by the given `seed`.  Otherwise, it is seeded by a random
///     seed.
///   - seed2: A second seed to avoid seed collision.
///   - min_object_covered: The cropped area of the image must contain at least this
///     fraction of any bounding box supplied. The value of this parameter should be
///     non-negative. In the case of 0, the cropped area does not need to overlap
///     any of the bounding boxes supplied.
///   - aspect_ratio_range: The cropped area of the image must have an aspect ratio =
///     width / height within this range.
///   - area_range: The cropped area of the image must contain a fraction of the
///     supplied image within this range.
///   - max_attempts: Number of attempts at generating a cropped region of the image
///     of the specified constraints. After `max_attempts` failures, return the entire
///     image.
///   - use_image_if_no_bounding_boxes: Controls behavior if no bounding boxes supplied.
///     If true, assume an implicit bounding box covering the whole input. If false,
///     raise an error.
///
/// - Outputs:
///   - begin: 1-D, containing `[offset_height, offset_width, 0]`. Provide as input to
///     `tf.slice`.
///   - size: 1-D, containing `[target_height, target_width, -1]`. Provide as input to
///     `tf.slice`.
///   - bboxes: 3-D with shape `[1, 1, 4]` containing the distorted bounding box.
///     Provide as input to `tf.image.draw_bounding_boxes`.
@inlinable @inline(__always)
public static func sampleDistortedBoundingBox<T: BinaryInteger & TensorFlowScalar>(
  imageSize: Tensor<T>,
  boundingBoxes: Tensor<Float>,
  seed: Int64 = 0,
  seed2: Int64 = 0,
  minObjectCovered: Double = 0.1,
  aspectRatioRange: [Double] = [0.75, 1.33],
  areaRange: [Double] = [0.05, 1],
  maxAttempts: Int64 = 100,
  useImageIfNoBoundingBoxes: Bool = false
) -> (begin: Tensor<T>, size: Tensor<T>, bboxes: Tensor<Float>) {
  let ret: (TensorHandle<T>, TensorHandle<T>, TensorHandle<Float>) = #tfop("SampleDistortedBoundingBox",
    imageSize,
    boundingBoxes,
    T$dtype: T.tensorFlowDataType,
    seed: seed,
    seed2: seed2,
    min_object_covered: minObjectCovered,
    aspect_ratio_range: aspectRatioRange,
    area_range: areaRange,
    max_attempts: maxAttempts,
    use_image_if_no_bounding_boxes: useImageIfNoBoundingBoxes)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Generate a single randomly distorted bounding box for an image.
///
/// Bounding box annotations are often supplied in addition to ground-truth labels
/// in image recognition or object localization tasks. A common technique for
/// training such a system is to randomly distort an image while preserving
/// its content, i.e. *data augmentation*. This Op outputs a randomly distorted
/// localization of an object, i.e. bounding box, given an `image_size`,
/// `bounding_boxes` and a series of constraints.
///
/// The output of this Op is a single bounding box that may be used to crop the
/// original image. The output is returned as 3 tensors: `begin`, `size` and
/// `bboxes`. The first 2 tensors can be fed directly into `tf.slice` to crop the
/// image. The latter may be supplied to `tf.image.draw_bounding_boxes` to visualize
/// what the bounding box looks like.
///
/// Bounding boxes are supplied and returned as `[y_min, x_min, y_max, x_max]`. The
/// bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
/// height of the underlying image.
///
/// For example,
///
/// ```python
///     # Generate a single distorted bounding box.
///     begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
///         tf.shape(image),
///         bounding_boxes=bounding_boxes)
///
///     # Draw the bounding box in an image summary.
///     image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
///                                                   bbox_for_draw)
///     tf.summary.image('images_with_box', image_with_box)
///
///     # Employ the bounding box to distort the image.
///     distorted_image = tf.slice(image, begin, size)
/// ```
///
/// Note that if no bounding box information is available, setting
/// `use_image_if_no_bounding_boxes = true` will assume there is a single implicit
/// bounding box covering the whole image. If `use_image_if_no_bounding_boxes` is
/// false and no bounding boxes are supplied, an error is raised.
///
/// - Parameters:
///   - image_size: 1-D, containing `[height, width, channels]`.
///   - bounding_boxes: 3-D with shape `[batch, N, 4]` describing the N bounding boxes
///     associated with the image.
///   - min_object_covered: The cropped area of the image must contain at least this
///     fraction of any bounding box supplied. The value of this parameter should be
///     non-negative. In the case of 0, the cropped area does not need to overlap
///     any of the bounding boxes supplied.
///
/// - Attrs:
///   - seed: If either `seed` or `seed2` are set to non-zero, the random number
///     generator is seeded by the given `seed`.  Otherwise, it is seeded by a random
///     seed.
///   - seed2: A second seed to avoid seed collision.
///   - aspect_ratio_range: The cropped area of the image must have an aspect ratio =
///     width / height within this range.
///   - area_range: The cropped area of the image must contain a fraction of the
///     supplied image within this range.
///   - max_attempts: Number of attempts at generating a cropped region of the image
///     of the specified constraints. After `max_attempts` failures, return the entire
///     image.
///   - use_image_if_no_bounding_boxes: Controls behavior if no bounding boxes supplied.
///     If true, assume an implicit bounding box covering the whole input. If false,
///     raise an error.
///
/// - Outputs:
///   - begin: 1-D, containing `[offset_height, offset_width, 0]`. Provide as input to
///     `tf.slice`.
///   - size: 1-D, containing `[target_height, target_width, -1]`. Provide as input to
///     `tf.slice`.
///   - bboxes: 3-D with shape `[1, 1, 4]` containing the distorted bounding box.
///     Provide as input to `tf.image.draw_bounding_boxes`.
@inlinable @inline(__always)
public static func sampleDistortedBoundingBoxV2<T: BinaryInteger & TensorFlowScalar>(
  imageSize: Tensor<T>,
  boundingBoxes: Tensor<Float>,
  minObjectCovered: Tensor<Float>,
  seed: Int64 = 0,
  seed2: Int64 = 0,
  aspectRatioRange: [Double] = [0.75, 1.33],
  areaRange: [Double] = [0.05, 1],
  maxAttempts: Int64 = 100,
  useImageIfNoBoundingBoxes: Bool = false
) -> (begin: Tensor<T>, size: Tensor<T>, bboxes: Tensor<Float>) {
  let ret: (TensorHandle<T>, TensorHandle<T>, TensorHandle<Float>) = #tfop("SampleDistortedBoundingBoxV2",
    imageSize,
    boundingBoxes,
    minObjectCovered,
    T$dtype: T.tensorFlowDataType,
    seed: seed,
    seed2: seed2,
    aspect_ratio_range: aspectRatioRange,
    area_range: areaRange,
    max_attempts: maxAttempts,
    use_image_if_no_bounding_boxes: useImageIfNoBoundingBoxes)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Saves the input tensors to disk.
///
/// The size of `tensor_names` must match the number of tensors in `data`. `data[i]`
/// is written to `filename` with name `tensor_names[i]`.
///
/// See also `SaveSlices`.
///
/// - Parameters:
///   - filename: Must have a single element. The name of the file to which we write
///     the tensor.
///   - tensor_names: Shape `[N]`. The names of the tensors to be saved.
///   - data: `N` tensors to save.
@inlinable @inline(__always)
public static func save<T: TensorFlowScalar>(
  filename: StringTensor,
  tensorNames: StringTensor,
  data: [Tensor<T>]
) {
  return #tfop("Save",
    filename,
    tensorNames,
    data)
}

/// Saves input tensors slices to disk.
///
/// This is like `Save` except that tensors can be listed in the saved file as being
/// a slice of a larger tensor.  `shapes_and_slices` specifies the shape of the
/// larger tensor and the slice that this tensor covers. `shapes_and_slices` must
/// have as many elements as `tensor_names`.
///
/// Elements of the `shapes_and_slices` input must either be:
///
/// *  The empty string, in which case the corresponding tensor is
///    saved normally.
/// *  A string of the form `dim0 dim1 ... dimN-1 slice-spec` where the
///    `dimI` are the dimensions of the larger tensor and `slice-spec`
///    specifies what part is covered by the tensor to save.
///
/// `slice-spec` itself is a `:`-separated list: `slice0:slice1:...:sliceN-1`
/// where each `sliceI` is either:
///
/// *  The string `-` meaning that the slice covers all indices of this dimension
/// *  `start,length` where `start` and `length` are integers.  In that
///    case the slice covers `length` indices starting at `start`.
///
/// See also `Save`.
///
/// - Parameters:
///   - filename: Must have a single element. The name of the file to which we write the
///     tensor.
///   - tensor_names: Shape `[N]`. The names of the tensors to be saved.
///   - shapes_and_slices: Shape `[N]`.  The shapes and slice specifications to use when
///     saving the tensors.
///   - data: `N` tensors to save.
@inlinable @inline(__always)
public static func saveSlices<T: TensorFlowScalar>(
  filename: StringTensor,
  tensorNames: StringTensor,
  shapesAndSlices: StringTensor,
  data: [Tensor<T>]
) {
  return #tfop("SaveSlices",
    filename,
    tensorNames,
    shapesAndSlices,
    data)
}

/// Saves tensors in V2 checkpoint format.
///
/// By default, saves the named tensors in full.  If the caller wishes to save
/// specific slices of full tensors, "shape_and_slices" should be non-empty strings
/// and correspondingly well-formed.
///
/// - Parameters:
///   - prefix: Must have a single element. The prefix of the V2 checkpoint to which we
///     write the tensors.
///   - tensor_names: shape {N}. The names of the tensors to be saved.
///   - shape_and_slices: shape {N}.  The slice specs of the tensors to be saved.
///     Empty strings indicate that they are non-partitioned tensors.
///   - tensors: `N` tensors to save.
@inlinable @inline(__always)
public static func saveV2<Dtypes: TensorFlowScalar>(
  prefix: StringTensor,
  tensorNames: StringTensor,
  shapeAndSlices: StringTensor,
  tensors: [Tensor<Dtypes>]
) {
  return #tfop("SaveV2",
    prefix,
    tensorNames,
    shapeAndSlices,
    tensors)
}

/// Outputs a `Summary` protocol buffer with scalar values.
///
/// The input `tags` and `values` must have the same shape.  The generated summary
/// has a summary value for each tag-value pair in `tags` and `values`.
///
/// - Parameters:
///   - tags: Tags for the summary.
///   - values: Same shape as `tags.  Values for the summary.
///
/// - Output summary: Scalar.  Serialized `Summary` protocol buffer.
@inlinable @inline(__always)
public static func scalarSummary<T: Numeric & TensorFlowScalar>(
  tags: StringTensor,
  _ values: Tensor<T>
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("ScalarSummary",
    tags,
    values,
    T$dtype: T.tensorFlowDataType)
  return StringTensor(handle: ret)
}

@inlinable @inline(__always)
public static func scaleAndTranslate<T: Numeric & TensorFlowScalar>(
  images: Tensor<T>,
  size: Tensor<Int32>,
  scale: Tensor<Float>,
  translation: Tensor<Float>,
  kernelType: String = "lanczos3",
  antialias: Bool = true
) -> Tensor<Float> {
  let ret: TensorHandle<Float> = #tfop("ScaleAndTranslate",
    images,
    size,
    scale,
    translation,
    T$dtype: T.tensorFlowDataType,
    kernel_type: kernelType,
    antialias: antialias)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func scaleAndTranslateGrad<T: FloatingPoint & TensorFlowScalar>(
  grads: Tensor<T>,
  originalImage: Tensor<T>,
  scale: Tensor<Float>,
  translation: Tensor<Float>,
  kernelType: String = "lanczos3",
  antialias: Bool = true
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("ScaleAndTranslateGrad",
    grads,
    originalImage,
    scale,
    translation,
    T$dtype: T.tensorFlowDataType,
    kernel_type: kernelType,
    antialias: antialias)
  return Tensor(handle: ret)
}

/// Scatter `updates` into a new tensor according to `indices`.
///
/// Creates a new tensor by applying sparse `updates` to individual values or
/// slices within a tensor (initially zero for numeric, empty for string) of
/// the given `shape` according to indices.  This operator is the inverse of the
/// `tf.gather_nd` operator which extracts values or slices from a given tensor.
///
/// This operation is similar to tensor_scatter_add, except that the tensor is
/// zero-initialized. Calling `tf.scatter_nd(indices, values, shape)` is identical
/// to `tensor_scatter_add(tf.zeros(shape, values.dtype), indices, values)`
///
/// If `indices` contains duplicates, then their updates are accumulated (summed).
///
/// **WARNING**: The order in which updates are applied is nondeterministic, so the
/// output will be nondeterministic if `indices` contains duplicates -- because
/// of some numerical approximation issues, numbers summed in different order
/// may yield different results.
///
/// `indices` is an integer tensor containing indices into a new tensor of shape
/// `shape`.  The last dimension of `indices` can be at most the rank of `shape`:
///
///     indices.shape[-1] <= shape.rank
///
/// The last dimension of `indices` corresponds to indices into elements
/// (if `indices.shape[-1] = shape.rank`) or slices
/// (if `indices.shape[-1] < shape.rank`) along dimension `indices.shape[-1]` of
/// `shape`.  `updates` is a tensor with shape
///
///     indices.shape[:-1] + shape[indices.shape[-1]:]
///
/// The simplest form of scatter is to insert individual elements in a tensor by
/// index. For example, say we want to insert 4 scattered elements in a rank-1
/// tensor with 8 elements.
///
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/ScatterNd1.png" alt>
/// </div>
///
/// In Python, this scatter operation would look like this:
///
/// ```python
///     indices = tf.constant([[4], [3], [1], [7]])
///     updates = tf.constant([9, 10, 11, 12])
///     shape = tf.constant([8])
///     scatter = tf.scatter_nd(indices, updates, shape)
///     with tf.Session() as sess:
///       print(sess.run(scatter))
/// ```
///
/// The resulting tensor would look like this:
///
///     [0, 11, 0, 10, 9, 0, 0, 12]
///
/// We can also, insert entire slices of a higher rank tensor all at once. For
/// example, if we wanted to insert two slices in the first dimension of a
/// rank-3 tensor with two matrices of new values.
///
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/ScatterNd2.png" alt>
/// </div>
///
/// In Python, this scatter operation would look like this:
///
/// ```python
///     indices = tf.constant([[0], [2]])
///     updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
///                             [7, 7, 7, 7], [8, 8, 8, 8]],
///                            [[5, 5, 5, 5], [6, 6, 6, 6],
///                             [7, 7, 7, 7], [8, 8, 8, 8]]])
///     shape = tf.constant([4, 4, 4])
///     scatter = tf.scatter_nd(indices, updates, shape)
///     with tf.Session() as sess:
///       print(sess.run(scatter))
/// ```
///
/// The resulting tensor would look like this:
///
///     [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
///      [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
///      [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
///      [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
///
/// Note that on CPU, if an out of bound index is found, an error is returned.
/// On GPU, if an out of bound index is found, the index is ignored.
///
/// - Parameters:
///   - indices: Index tensor.
///   - updates: Updates to scatter into output.
///   - shape: 1-D. The shape of the resulting tensor.
///
/// - Output output: A new tensor with the given shape and updates applied according
///   to the indices.
@inlinable @inline(__always)
public static func scatterNd<T: TensorFlowScalar, Tindices: BinaryInteger & TensorFlowScalar>(
  indices: Tensor<Tindices>,
  updates: Tensor<T>,
  shape: Tensor<Tindices>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("ScatterNd",
    indices,
    updates,
    shape,
    T$dtype: T.tensorFlowDataType,
    Tindices$dtype: Tindices.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Applies sparse addition to `input` using individual values or slices
///
/// from `updates` according to indices `indices`.  The updates are non-aliasing:
/// `input` is only modified in-place if no other operations will use it.
/// Otherwise, a copy of `input` is made.  This operation has a gradient with
/// respect to both `input` and `updates`.
///
/// `input` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.
///
/// `indices` must be integer tensor, containing indices into `input`.
/// It must be shape \\([d_0, ..., d_{Q-2}, K]\\) where `0 < K <= P`.
///
/// The innermost dimension of `indices` (with length `K`) corresponds to
/// indices into elements (if `K = P`) or `(P-K)`-dimensional slices
/// (if `K < P`) along the `K`th dimension of `input`.
///
/// `updates` is `Tensor` of rank `Q-1+P-K` with shape:
///
/// $$[d_0, ..., d_{Q-2}, input.shape[K], ..., input.shape[P-1]].$$
///
/// For example, say we want to add 4 scattered elements to a rank-1 tensor to 8
/// elements. In Python, that addition would look like this:
///
///     input = tf.constant([1, 2, 3, 4, 5, 6, 7, 8])
///     indices = tf.constant([[4], [3], [1], [7]])
///     updates = tf.constant([9, 10, 11, 12])
///     output = tf.scatter_nd_non_aliasing_add(input, indices, updates)
///     with tf.Session() as sess:
///       print(sess.run(output))
///
/// The resulting value `output` would look like this:
///
///     [1, 13, 3, 14, 14, 6, 7, 20]
///
/// See `tf.scatter_nd` for more details about how to make updates to slices.
///
/// - Parameters:
///   - input: A Tensor.
///   - indices: A Tensor. Must be one of the following types: `int32`, `int64`.
///     A tensor of indices into `input`.
///   - updates: A Tensor. Must have the same type as ref. A tensor of updated values
///     to add to `input`.
///
/// - Output output: A `Tensor` with the same shape as `input`, containing values of `input`
///   updated with `updates`.
@inlinable @inline(__always)
public static func scatterNdNonAliasingAdd<T: TensorFlowScalar, Tindices: BinaryInteger & TensorFlowScalar>(
  _ input: Tensor<T>,
  indices: Tensor<Tindices>,
  updates: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("ScatterNdNonAliasingAdd",
    input,
    indices,
    updates,
    T$dtype: T.tensorFlowDataType,
    Tindices$dtype: Tindices.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes fingerprints of the input strings.
///
/// - Parameter input: vector of strings to compute fingerprints on.
///
/// - Output output: a (N,2) shaped matrix where N is the number of elements in the input
///   vector. Each row contains the low and high parts of the fingerprint.
@inlinable @inline(__always)
public static func sdcaFprint(
  _ input: StringTensor
) -> Tensor<Int64> {
  let ret: TensorHandle<Int64> = #tfop("SdcaFprint",
    input)
  return Tensor(handle: ret)
}

/// Computes the maximum along segments of a tensor.
///
/// Read
/// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
/// for an explanation of segments.
///
/// Computes a tensor such that
/// \\(output_i = \max_j(data_j)\\) where `max` is over `j` such
/// that `segment_ids[j] == i`.
///
/// If the max is empty for a given segment ID `i`, `output[i] = 0`.
///
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/SegmentMax.png" alt>
/// </div>
///
/// For example:
///
/// ```
/// c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
/// tf.segment_max(c, tf.constant([0, 0, 1]))
/// # ==> [[4, 3, 3, 4],
/// #      [5, 6, 7, 8]]
/// ```
///
///
/// - Parameter segment_ids: A 1-D tensor whose size is equal to the size of `data`'s
///   first dimension.  Values should be sorted and can be repeated.
///
/// - Output output: Has same shape as data, except for dimension 0 which
///   has size `k`, the number of segments.
@inlinable @inline(__always)
public static func segmentMax<T: Numeric & TensorFlowScalar, Tindices: BinaryInteger & TensorFlowScalar>(
  data: Tensor<T>,
  segmentIds: Tensor<Tindices>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("SegmentMax",
    data,
    segmentIds,
    T$dtype: T.tensorFlowDataType,
    Tindices$dtype: Tindices.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the mean along segments of a tensor.
///
/// Read
/// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
/// for an explanation of segments.
///
/// Computes a tensor such that
/// \\(output_i = \frac{\sum_j data_j}{N}\\) where `mean` is
/// over `j` such that `segment_ids[j] == i` and `N` is the total number of
/// values summed.
///
/// If the mean is empty for a given segment ID `i`, `output[i] = 0`.
///
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/SegmentMean.png" alt>
/// </div>
///
/// For example:
///
/// ```
/// c = tf.constant([[1.0,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
/// tf.segment_mean(c, tf.constant([0, 0, 1]))
/// # ==> [[2.5, 2.5, 2.5, 2.5],
/// #      [5, 6, 7, 8]]
/// ```
///
///
/// - Parameter segment_ids: A 1-D tensor whose size is equal to the size of `data`'s
///   first dimension.  Values should be sorted and can be repeated.
///
/// - Output output: Has same shape as data, except for dimension 0 which
///   has size `k`, the number of segments.
@inlinable @inline(__always)
public static func segmentMean<T: Numeric & TensorFlowScalar, Tindices: BinaryInteger & TensorFlowScalar>(
  data: Tensor<T>,
  segmentIds: Tensor<Tindices>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("SegmentMean",
    data,
    segmentIds,
    T$dtype: T.tensorFlowDataType,
    Tindices$dtype: Tindices.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the minimum along segments of a tensor.
///
/// Read
/// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
/// for an explanation of segments.
///
/// Computes a tensor such that
/// \\(output_i = \min_j(data_j)\\) where `min` is over `j` such
/// that `segment_ids[j] == i`.
///
/// If the min is empty for a given segment ID `i`, `output[i] = 0`.
///
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/SegmentMin.png" alt>
/// </div>
///
/// For example:
///
/// ```
/// c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
/// tf.segment_min(c, tf.constant([0, 0, 1]))
/// # ==> [[1, 2, 2, 1],
/// #      [5, 6, 7, 8]]
/// ```
///
/// - Parameter segment_ids: A 1-D tensor whose size is equal to the size of `data`'s
///   first dimension.  Values should be sorted and can be repeated.
///
/// - Output output: Has same shape as data, except for dimension 0 which
///   has size `k`, the number of segments.
@inlinable @inline(__always)
public static func segmentMin<T: Numeric & TensorFlowScalar, Tindices: BinaryInteger & TensorFlowScalar>(
  data: Tensor<T>,
  segmentIds: Tensor<Tindices>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("SegmentMin",
    data,
    segmentIds,
    T$dtype: T.tensorFlowDataType,
    Tindices$dtype: Tindices.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the product along segments of a tensor.
///
/// Read
/// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
/// for an explanation of segments.
///
/// Computes a tensor such that
/// \\(output_i = \prod_j data_j\\) where the product is over `j` such
/// that `segment_ids[j] == i`.
///
/// If the product is empty for a given segment ID `i`, `output[i] = 1`.
///
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/SegmentProd.png" alt>
/// </div>
///
/// For example:
///
/// ```
/// c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
/// tf.segment_prod(c, tf.constant([0, 0, 1]))
/// # ==> [[4, 6, 6, 4],
/// #      [5, 6, 7, 8]]
/// ```
///
///
/// - Parameter segment_ids: A 1-D tensor whose size is equal to the size of `data`'s
///   first dimension.  Values should be sorted and can be repeated.
///
/// - Output output: Has same shape as data, except for dimension 0 which
///   has size `k`, the number of segments.
@inlinable @inline(__always)
public static func segmentProd<T: Numeric & TensorFlowScalar, Tindices: BinaryInteger & TensorFlowScalar>(
  data: Tensor<T>,
  segmentIds: Tensor<Tindices>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("SegmentProd",
    data,
    segmentIds,
    T$dtype: T.tensorFlowDataType,
    Tindices$dtype: Tindices.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the sum along segments of a tensor.
///
/// Read
/// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
/// for an explanation of segments.
///
/// Computes a tensor such that
/// \\(output_i = \sum_j data_j\\) where sum is over `j` such
/// that `segment_ids[j] == i`.
///
/// If the sum is empty for a given segment ID `i`, `output[i] = 0`.
///
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/SegmentSum.png" alt>
/// </div>
///
/// For example:
///
/// ```
/// c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
/// tf.segment_sum(c, tf.constant([0, 0, 1]))
/// # ==> [[5, 5, 5, 5],
/// #      [5, 6, 7, 8]]
/// ```
///
///
/// - Parameter segment_ids: A 1-D tensor whose size is equal to the size of `data`'s
///   first dimension.  Values should be sorted and can be repeated.
///
/// - Output output: Has same shape as data, except for dimension 0 which
///   has size `k`, the number of segments.
@inlinable @inline(__always)
public static func segmentSum<T: Numeric & TensorFlowScalar, Tindices: BinaryInteger & TensorFlowScalar>(
  data: Tensor<T>,
  segmentIds: Tensor<Tindices>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("SegmentSum",
    data,
    segmentIds,
    T$dtype: T.tensorFlowDataType,
    Tindices$dtype: Tindices.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Selects elements from `x` or `y`, depending on `condition`.
///
/// The `x`, and `y` tensors must all have the same shape, and the
/// output will also have that shape.
///
/// The `condition` tensor must be a scalar if `x` and `y` are scalars.
/// If `x` and `y` are vectors or higher rank, then `condition` must be either a
/// scalar, a vector with size matching the first dimension of `x`, or must have
/// the same shape as `x`.
///
/// The `condition` tensor acts as a mask that chooses, based on the value at each
/// element, whether the corresponding element / row in the output should be
/// taken from `x` (if true) or `y` (if false).
///
/// If `condition` is a vector and `x` and `y` are higher rank matrices, then
/// it chooses which row (outer dimension) to copy from `x` and `y`.
/// If `condition` has the same shape as `x` and `y`, then it chooses which
/// element to copy from `x` and `y`.
///
/// For example:
///
/// ```python
/// # 'condition' tensor is [[True,  False]
/// #                        [False, True]]
/// # 't' is [[1, 2],
/// #         [3, 4]]
/// # 'e' is [[5, 6],
/// #         [7, 8]]
/// select(condition, t, e)  # => [[1, 6], [7, 4]]
///
///
/// # 'condition' tensor is [True, False]
/// # 't' is [[1, 2],
/// #         [3, 4]]
/// # 'e' is [[5, 6],
/// #         [7, 8]]
/// select(condition, t, e) ==> [[1, 2],
///                              [7, 8]]
///
/// ```
///
/// - Parameters:
///   - t: = A `Tensor` which may have the same shape as `condition`.
///     If `condition` is rank 1, `x` may have higher rank,
///     but its first dimension must match the size of `condition`.
///   - e: = A `Tensor` with the same type and shape as `x`.
///
/// - Output output: = A `Tensor` with the same type and shape as `x` and `y`.
@inlinable @inline(__always)
public static func select<T: TensorFlowScalar>(
  condition: Tensor<Bool>,
  t: Tensor<T>,
  e: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Select",
    condition,
    t,
    e,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the Eigen Decomposition of a batch of square self-adjoint matrices.
///
/// The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
/// form square matrices, with the same constraints as the single matrix
/// SelfAdjointEig.
///
/// The result is a [..., M+1, M] matrix with [..., 0,:] containing the
/// eigenvalues, and subsequent [...,1:, :] containing the eigenvectors. The eigenvalues
/// are sorted in non-decreasing order.
///
/// - Parameter input: Shape is `[..., M, M]`.
///
/// - Output output: Shape is `[..., M+1, M]`.
@inlinable @inline(__always)
public static func selfAdjointEig<T: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("SelfAdjointEig",
    input,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the eigen decomposition of one or more square self-adjoint matrices.
///
/// Computes the eigenvalues and (optionally) eigenvectors of each inner matrix in
/// `input` such that `input[..., :, :] = v[..., :, :] * diag(e[..., :])`. The eigenvalues
/// are sorted in non-decreasing order.
///
/// ```python
/// # a is a tensor.
/// # e is a tensor of eigenvalues.
/// # v is a tensor of eigenvectors.
/// e, v = self_adjoint_eig(a)
/// e = self_adjoint_eig(a, compute_v=False)
/// ```
///
/// - Parameter input: `Tensor` input of shape `[N, N]`.
///
/// - Attr compute_v: If `True` then eigenvectors will be computed and returned in `v`.
///   Otherwise, only the eigenvalues will be computed.
///
/// - Outputs:
///   - e: Eigenvalues. Shape is `[N]`.
///   - v: Eigenvectors. Shape is `[N, N]`.
@inlinable @inline(__always)
public static func selfAdjointEigV2<T: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>,
  computeV: Bool = true
) -> (e: Tensor<T>, v: Tensor<T>) {
  let ret: (TensorHandle<T>, TensorHandle<T>) = #tfop("SelfAdjointEigV2",
    input,
    T$dtype: T.tensorFlowDataType,
    compute_v: computeV)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

/// Computes scaled exponential linear: `scale * alpha * (exp(features) - 1)`
///
/// if < 0, `scale * features` otherwise.
///
/// To be used together with
/// `initializer = tf.variance_scaling_initializer(factor=1.0, mode='FAN_IN')`.
/// For correct dropout, use `tf.contrib.nn.alpha_dropout`.
///
/// See [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
@inlinable @inline(__always)
public static func selu<T: FloatingPoint & TensorFlowScalar>(
  features: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Selu",
    features,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes gradients for the scaled exponential linear (Selu) operation.
///
/// - Parameters:
///   - gradients: The backpropagated gradients to the corresponding Selu operation.
///   - outputs: The outputs of the corresponding Selu operation.
///
/// - Output backprops: The gradients: `gradients * (outputs + scale * alpha)`
///   if outputs < 0, `scale * gradients` otherwise.
@inlinable @inline(__always)
public static func seluGrad<T: FloatingPoint & TensorFlowScalar>(
  gradients: Tensor<T>,
  outputs: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("SeluGrad",
    gradients,
    outputs,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Performs gradient updates of embedding tables.
///
/// - Parameters:
///   - inputs: A TensorList of gradients with which to update embedding tables.
///     This argument has the same length and shapes as the return value of
///     RecvTPUEmbeddingActivations, but contains gradients of the model's loss
///     with respect to the embedding activations. The embedding tables are updated
///     from these gradients via the optimizer specified in the TPU embedding
///     configuration given to tpu.initialize_system.
///   - learning_rates: A TensorList of float32 scalars, one for each dynamic learning
///     rate tag: see the comments in
///     //third_party/tensorflow/core/protobuf/tpu/optimization_parameters.proto.
///     Multiple tables can share the same dynamic learning rate tag as specified
///     in the configuration. If the learning rates for all tables are constant,
///     this list should be empty.
///
/// - Attr config: Serialized TPUEmbeddingConfiguration proto.
@inlinable @inline(__always)
public static func sendTPUEmbeddingGradients(
  inputs: [Tensor<Float>],
  learningRates: [Tensor<Float>],
  config: String
) {
  return #tfop("SendTPUEmbeddingGradients",
    inputs,
    learningRates,
    config: config)
}

/// Serialize an `N`-minibatch `SparseTensor` into an `[N, 3]` `Tensor` object.
///
/// The `SparseTensor` must have rank `R` greater than 1, and the first dimension
/// is treated as the minibatch dimension.  Elements of the `SparseTensor`
/// must be sorted in increasing order of this first dimension.  The serialized
/// `SparseTensor` objects going into each row of `serialized_sparse` will have
/// rank `R-1`.
///
/// The minibatch size `N` is extracted from `sparse_shape[0]`.
///
/// - Parameters:
///   - sparse_indices: 2-D.  The `indices` of the minibatch `SparseTensor`.
///   - sparse_values: 1-D.  The `values` of the minibatch `SparseTensor`.
///   - sparse_shape: 1-D.  The `shape` of the minibatch `SparseTensor`.
///
/// - Attr out_type: The `dtype` to use for serialization; the supported types are `string`
///   (default) and `variant`.
@inlinable @inline(__always)
public static func serializeManySparse<T: TensorFlowScalar, OutType: TensorFlowScalar>(
  sparseIndices: Tensor<Int64>,
  sparseValues: Tensor<T>,
  sparseShape: Tensor<Int64>
) -> Tensor<OutType> {
  let ret: TensorHandle<OutType> = #tfop("SerializeManySparse",
    sparseIndices,
    sparseValues,
    sparseShape,
    T$dtype: T.tensorFlowDataType,
    out_type$dtype: OutType.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Serialize a `SparseTensor` into a `[3]` `Tensor` object.
///
/// - Parameters:
///   - sparse_indices: 2-D.  The `indices` of the `SparseTensor`.
///   - sparse_values: 1-D.  The `values` of the `SparseTensor`.
///   - sparse_shape: 1-D.  The `shape` of the `SparseTensor`.
///
/// - Attr out_type: The `dtype` to use for serialization; the supported types are `string`
///   (default) and `variant`.
@inlinable @inline(__always)
public static func serializeSparse<T: TensorFlowScalar, OutType: TensorFlowScalar>(
  sparseIndices: Tensor<Int64>,
  sparseValues: Tensor<T>,
  sparseShape: Tensor<Int64>
) -> Tensor<OutType> {
  let ret: TensorHandle<OutType> = #tfop("SerializeSparse",
    sparseIndices,
    sparseValues,
    sparseShape,
    T$dtype: T.tensorFlowDataType,
    out_type$dtype: OutType.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Transforms a Tensor into a serialized TensorProto proto.
///
/// - Parameter tensor: A Tensor of type `T`.
///
/// - Attr T: The type of the input tensor.
///
/// - Output serialized: A serialized TensorProto proto of the input tensor.
@inlinable @inline(__always)
public static func serializeTensor<T: TensorFlowScalar>(
  _ tensor: Tensor<T>
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("SerializeTensor",
    tensor,
    T$dtype: T.tensorFlowDataType)
  return StringTensor(handle: ret)
}

/// Number of unique elements along last dimension of input `set`.
///
/// Input `set` is a `SparseTensor` represented by `set_indices`, `set_values`,
/// and `set_shape`. The last dimension contains values in a set, duplicates are
/// allowed but ignored.
///
/// If `validate_indices` is `True`, this op validates the order and range of `set`
/// indices.
///
/// - Parameters:
///   - set_indices: 2D `Tensor`, indices of a `SparseTensor`.
///   - set_values: 1D `Tensor`, values of a `SparseTensor`.
///   - set_shape: 1D `Tensor`, shape of a `SparseTensor`.
///
/// - Output size: For `set` ranked `n`, this is a `Tensor` with rank `n-1`, and the same 1st
///   `n-1` dimensions as `set`. Each value is the number of unique elements in
///   the corresponding `[0...n-1]` dimension of `set`.
@inlinable @inline(__always)
public static func setSize<T: BinaryInteger & TensorFlowScalar>(
  setIndices: Tensor<Int64>,
  setValues: Tensor<T>,
  setShape: Tensor<Int64>,
  validateIndices: Bool = true
) -> Tensor<Int32> {
  let ret: TensorHandle<Int32> = #tfop("SetSize",
    setIndices,
    setValues,
    setShape,
    T$dtype: T.tensorFlowDataType,
    validate_indices: validateIndices)
  return Tensor(handle: ret)
}

/// Returns the shape of a tensor.
///
/// This operation returns a 1-D integer tensor representing the shape of `input`.
///
/// For example:
///
/// ```
/// # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
/// shape(t) ==> [2, 2, 3]
/// ```
@inlinable @inline(__always)
public static func shape<T: TensorFlowScalar, OutType: BinaryInteger & TensorFlowScalar>(
  _ input: Tensor<T>
) -> Tensor<OutType> {
  let ret: TensorHandle<OutType> = #tfop("Shape",
    input,
    T$dtype: T.tensorFlowDataType,
    out_type$dtype: OutType.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Generate a sharded filename. The filename is printf formatted as
///
///    %s-%05d-of-%05d, basename, shard, num_shards.
@inlinable @inline(__always)
public static func shardedFilename(
  basename: StringTensor,
  shard: Tensor<Int32>,
  numShards: Tensor<Int32>
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("ShardedFilename",
    basename,
    shard,
    numShards)
  return StringTensor(handle: ret)
}

/// Generate a glob pattern matching all sharded file names.
@inlinable @inline(__always)
public static func shardedFilespec(
  basename: StringTensor,
  numShards: Tensor<Int32>
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("ShardedFilespec",
    basename,
    numShards)
  return StringTensor(handle: ret)
}

/// Shuts down a running distributed TPU system.
///
/// The op returns an error if no system is running.
@inlinable @inline(__always)
public static func shutdownDistributedTPU(
) {
  return #tfop("ShutdownDistributedTPU")
}

/// Computes sigmoid of `x` element-wise.
///
/// Specifically, `y = 1 / (1 + exp(-x))`.
@inlinable @inline(__always)
public static func sigmoid<T: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Sigmoid",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the gradient of the sigmoid of `x` wrt its input.
///
/// Specifically, `grad = dy * y * (1 - y)`, where `y = sigmoid(x)`, and
/// `dy` is the corresponding input gradient.
@inlinable @inline(__always)
public static func sigmoidGrad<T: FloatingPoint & TensorFlowScalar>(
  _ y: Tensor<T>,
  dy: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("SigmoidGrad",
    y,
    dy,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Returns an element-wise indication of the sign of a number.
///
/// `y = sign(x) = -1` if `x < 0`; 0 if `x == 0`; 1 if `x > 0`.
///
/// For complex numbers, `y = sign(x) = x / |x|` if `x != 0`, otherwise `y = 0`.
@inlinable @inline(__always)
public static func sign<T: Numeric & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Sign",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func simple(
  _ a: Tensor<Int32>
) -> Tensor<Float> {
  let ret: TensorHandle<Float> = #tfop("Simple",
    a)
  return Tensor(handle: ret)
}

/// Computes sin of x element-wise.
@inlinable @inline(__always)
public static func sin<T: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Sin",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes hyperbolic sine of x element-wise.
@inlinable @inline(__always)
public static func sinh<T: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Sinh",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Returns the size of a tensor.
///
/// This operation returns an integer representing the number of elements in
/// `input`.
///
/// For example:
///
/// ```
/// # 't' is [[[1, 1,, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]]
/// size(t) ==> 12
/// ```
@inlinable @inline(__always)
public static func size<T: TensorFlowScalar, OutType: BinaryInteger & TensorFlowScalar>(
  _ input: Tensor<T>
) -> Tensor<OutType> {
  let ret: TensorHandle<OutType> = #tfop("Size",
    input,
    T$dtype: T.tensorFlowDataType,
    out_type$dtype: OutType.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Parses a text file and creates a batch of examples.
///
/// - Attrs:
///   - filename: The corpus's text file name.
///   - batch_size: The size of produced batch.
///   - window_size: The number of words to predict to the left and right of the target.
///   - min_count: The minimum number of word occurrences for it to be included in the
///     vocabulary.
///   - subsample: Threshold for word occurrence. Words that appear with higher
///     frequency will be randomly down-sampled. Set to 0 to disable.
///
/// - Outputs:
///   - vocab_word: A vector of words in the corpus.
///   - vocab_freq: Frequencies of words. Sorted in the non-ascending order.
///   - words_per_epoch: Number of words per epoch in the data file.
///   - current_epoch: The current epoch number.
///   - total_words_processed: The total number of words processed so far.
///   - examples: A vector of word ids.
///   - labels: A vector of word ids.
@inlinable @inline(__always)
public static func skipgram(
  filename: String,
  batchSize: Int64,
  windowSize: Int64 = 5,
  minCount: Int64 = 5,
  subsample: Double = 0.001
) -> (vocabWord: StringTensor, vocabFreq: Tensor<Int32>, wordsPerEpoch: Tensor<Int64>, currentEpoch: Tensor<Int32>, totalWordsProcessed: Tensor<Int64>, examples: Tensor<Int32>, labels: Tensor<Int32>) {
  let ret: (TensorHandle<String>, TensorHandle<Int32>, TensorHandle<Int64>, TensorHandle<Int32>, TensorHandle<Int64>, TensorHandle<Int32>, TensorHandle<Int32>) = #tfop("Skipgram",
    filename: filename,
    batch_size: batchSize,
    window_size: windowSize,
    min_count: minCount,
    subsample: subsample)
  return (StringTensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2), Tensor(handle: ret.3), Tensor(handle: ret.4), Tensor(handle: ret.5), Tensor(handle: ret.6))
}

/// Return a slice from 'input'.
///
/// The output tensor is a tensor with dimensions described by 'size'
/// whose values are extracted from 'input' starting at the offsets in
/// 'begin'.
///
/// *Requirements*:
///   0 <= begin[i] <= begin[i] + size[i] <= Di  for i in [0, n)
///
/// - Parameters:
///   - begin: begin[i] specifies the offset into the 'i'th dimension of
///     'input' to slice from.
///   - size: size[i] specifies the number of elements of the 'i'th dimension
///     of 'input' to slice. If size[i] is -1, all remaining elements in dimension
///     i are included in the slice (i.e. this is equivalent to setting
///     size[i] = input.dim_size(i) - begin[i]).
@inlinable @inline(__always)
public static func slice<T: TensorFlowScalar, Index: BinaryInteger & TensorFlowScalar>(
  _ input: Tensor<T>,
  begin: Tensor<Index>,
  size: Tensor<Index>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Slice",
    input,
    begin,
    size,
    T$dtype: T.tensorFlowDataType,
    Index$dtype: Index.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Returns a copy of the input tensor.
@inlinable @inline(__always)
public static func snapshot<T: TensorFlowScalar>(
  _ input: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Snapshot",
    input,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes softmax activations.
///
/// For each batch `i` and class `j` we have
///
///     $$softmax[i, j] = exp(logits[i, j]) / sum_j(exp(logits[i, j]))$$
///
/// - Parameter logits: 2-D with shape `[batch_size, num_classes]`.
///
/// - Output softmax: Same shape as `logits`.
@inlinable @inline(__always)
public static func softmax<T: FloatingPoint & TensorFlowScalar>(
  logits: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Softmax",
    logits,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes softmax cross entropy cost and gradients to backpropagate.
///
/// Inputs are the logits, not probabilities.
///
/// - Parameters:
///   - features: batch_size x num_classes matrix
///   - labels: batch_size x num_classes matrix
///     The caller must ensure that each batch of labels represents a valid
///     probability distribution.
///
/// - Outputs:
///   - loss: Per example loss (batch_size vector).
///   - backprop: backpropagated gradients (batch_size x num_classes matrix).
@inlinable @inline(__always)
public static func softmaxCrossEntropyWithLogits<T: FloatingPoint & TensorFlowScalar>(
  features: Tensor<T>,
  labels: Tensor<T>
) -> (loss: Tensor<T>, backprop: Tensor<T>) {
  let ret: (TensorHandle<T>, TensorHandle<T>) = #tfop("SoftmaxCrossEntropyWithLogits",
    features,
    labels,
    T$dtype: T.tensorFlowDataType)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

/// Computes softplus: `log(exp(features) + 1)`.
@inlinable @inline(__always)
public static func softplus<T: FloatingPoint & TensorFlowScalar>(
  features: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Softplus",
    features,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes softplus gradients for a softplus operation.
///
/// - Parameters:
///   - gradients: The backpropagated gradients to the corresponding softplus operation.
///   - features: The features passed as input to the corresponding softplus operation.
///
/// - Output backprops: The gradients: `gradients / (1 + exp(-features))`.
@inlinable @inline(__always)
public static func softplusGrad<T: FloatingPoint & TensorFlowScalar>(
  gradients: Tensor<T>,
  features: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("SoftplusGrad",
    gradients,
    features,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes softsign: `features / (abs(features) + 1)`.
@inlinable @inline(__always)
public static func softsign<T: FloatingPoint & TensorFlowScalar>(
  features: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Softsign",
    features,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes softsign gradients for a softsign operation.
///
/// - Parameters:
///   - gradients: The backpropagated gradients to the corresponding softsign operation.
///   - features: The features passed as input to the corresponding softsign operation.
///
/// - Output backprops: The gradients: `gradients / (1 + abs(features)) ** 2`.
@inlinable @inline(__always)
public static func softsignGrad<T: FloatingPoint & TensorFlowScalar>(
  gradients: Tensor<T>,
  features: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("SoftsignGrad",
    gradients,
    features,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// SpaceToBatch for 4-D tensors of type T.
///
/// This is a legacy version of the more general SpaceToBatchND.
///
/// Zero-pads and then rearranges (permutes) blocks of spatial data into batch.
/// More specifically, this op outputs a copy of the input tensor where values from
/// the `height` and `width` dimensions are moved to the `batch` dimension. After
/// the zero-padding, both `height` and `width` of the input must be divisible by the
/// block size.
///
/// - Parameters:
///   - input: 4-D with shape `[batch, height, width, depth]`.
///   - paddings: 2-D tensor of non-negative integers with shape `[2, 2]`. It specifies
///       the padding of the input with zeros across the spatial dimensions as follows:
///
///           paddings = [[pad_top, pad_bottom], [pad_left, pad_right]]
///
///       The effective spatial dimensions of the zero-padded input tensor will be:
///
///           height_pad = pad_top + height + pad_bottom
///           width_pad = pad_left + width + pad_right
///
///     The attr `block_size` must be greater than one. It indicates the block size.
///
///       * Non-overlapping blocks of size `block_size x block size` in the height and
///         width dimensions are rearranged into the batch dimension at each location.
///       * The batch of the output tensor is `batch * block_size * block_size`.
///       * Both height_pad and width_pad must be divisible by block_size.
///
///     The shape of the output will be:
///
///         [batch*block_size*block_size, height_pad/block_size, width_pad/block_size,
///          depth]
///
///     Some examples:
///
///     (1) For the following input of shape `[1, 2, 2, 1]` and block_size of 2:
///
///     ```
///     x = [[[[1], [2]], [[3], [4]]]]
///     ```
///
///     The output tensor has shape `[4, 1, 1, 1]` and value:
///
///     ```
///     [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
///     ```
///
///     (2) For the following input of shape `[1, 2, 2, 3]` and block_size of 2:
///
///     ```
///     x = [[[[1, 2, 3], [4, 5, 6]],
///           [[7, 8, 9], [10, 11, 12]]]]
///     ```
///
///     The output tensor has shape `[4, 1, 1, 3]` and value:
///
///     ```
///     [[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]]
///     ```
///
///     (3) For the following input of shape `[1, 4, 4, 1]` and block_size of 2:
///
///     ```
///     x = [[[[1],   [2],  [3],  [4]],
///           [[5],   [6],  [7],  [8]],
///           [[9],  [10], [11],  [12]],
///           [[13], [14], [15],  [16]]]]
///     ```
///
///     The output tensor has shape `[4, 2, 2, 1]` and value:
///
///     ```
///     x = [[[[1], [3]], [[9], [11]]],
///          [[[2], [4]], [[10], [12]]],
///          [[[5], [7]], [[13], [15]]],
///          [[[6], [8]], [[14], [16]]]]
///     ```
///
///     (4) For the following input of shape `[2, 2, 4, 1]` and block_size of 2:
///
///     ```
///     x = [[[[1],   [2],  [3],  [4]],
///           [[5],   [6],  [7],  [8]]],
///          [[[9],  [10], [11],  [12]],
///           [[13], [14], [15],  [16]]]]
///     ```
///
///     The output tensor has shape `[8, 1, 2, 1]` and value:
///
///     ```
///     x = [[[[1], [3]]], [[[9], [11]]], [[[2], [4]]], [[[10], [12]]],
///          [[[5], [7]]], [[[13], [15]]], [[[6], [8]]], [[[14], [16]]]]
///     ```
///
///     Among others, this operation is useful for reducing atrous convolution into
///     regular convolution.
@inlinable @inline(__always)
public static func spaceToBatch<T: TensorFlowScalar, Tpaddings: BinaryInteger & TensorFlowScalar>(
  _ input: Tensor<T>,
  paddings: Tensor<Tpaddings>,
  blockSize: Int64
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("SpaceToBatch",
    input,
    paddings,
    T$dtype: T.tensorFlowDataType,
    Tpaddings$dtype: Tpaddings.tensorFlowDataType,
    block_size: blockSize)
  return Tensor(handle: ret)
}

/// SpaceToBatch for N-D tensors of type T.
///
/// This operation divides "spatial" dimensions `[1, ..., M]` of the input into a
/// grid of blocks of shape `block_shape`, and interleaves these blocks with the
/// "batch" dimension (0) such that in the output, the spatial dimensions
/// `[1, ..., M]` correspond to the position within the grid, and the batch
/// dimension combines both the position within a spatial block and the original
/// batch position.  Prior to division into blocks, the spatial dimensions of the
/// input are optionally zero padded according to `paddings`.  See below for a
/// precise description.
///
/// - Parameters:
///   - input: N-D with shape `input_shape = [batch] + spatial_shape + remaining_shape`,
///     where spatial_shape has `M` dimensions.
///   - block_shape: 1-D with shape `[M]`, all values must be >= 1.
///   - paddings: 2-D with shape `[M, 2]`, all values must be >= 0.
///       `paddings[i] = [pad_start, pad_end]` specifies the padding for input dimension
///       `i + 1`, which corresponds to spatial dimension `i`.  It is required that
///       `block_shape[i]` divides `input_shape[i + 1] + pad_start + pad_end`.
///
///     This operation is equivalent to the following steps:
///
///     1. Zero-pad the start and end of dimensions `[1, ..., M]` of the
///        input according to `paddings` to produce `padded` of shape `padded_shape`.
///
///     2. Reshape `padded` to `reshaped_padded` of shape:
///
///          [batch] +
///          [padded_shape[1] / block_shape[0],
///            block_shape[0],
///           ...,
///           padded_shape[M] / block_shape[M-1],
///           block_shape[M-1]] +
///          remaining_shape
///
///     3. Permute dimensions of `reshaped_padded` to produce
///        `permuted_reshaped_padded` of shape:
///
///          block_shape +
///          [batch] +
///          [padded_shape[1] / block_shape[0],
///           ...,
///           padded_shape[M] / block_shape[M-1]] +
///          remaining_shape
///
///     4. Reshape `permuted_reshaped_padded` to flatten `block_shape` into the batch
///        dimension, producing an output tensor of shape:
///
///          [batch * prod(block_shape)] +
///          [padded_shape[1] / block_shape[0],
///           ...,
///           padded_shape[M] / block_shape[M-1]] +
///          remaining_shape
///
///     Some examples:
///
///     (1) For the following input of shape `[1, 2, 2, 1]`, `block_shape = [2, 2]`, and
///         `paddings = [[0, 0], [0, 0]]`:
///
///     ```
///     x = [[[[1], [2]], [[3], [4]]]]
///     ```
///
///     The output tensor has shape `[4, 1, 1, 1]` and value:
///
///     ```
///     [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
///     ```
///
///     (2) For the following input of shape `[1, 2, 2, 3]`, `block_shape = [2, 2]`, and
///         `paddings = [[0, 0], [0, 0]]`:
///
///     ```
///     x = [[[[1, 2, 3], [4, 5, 6]],
///           [[7, 8, 9], [10, 11, 12]]]]
///     ```
///
///     The output tensor has shape `[4, 1, 1, 3]` and value:
///
///     ```
///     [[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]]
///     ```
///
///     (3) For the following input of shape `[1, 4, 4, 1]`, `block_shape = [2, 2]`, and
///         `paddings = [[0, 0], [0, 0]]`:
///
///     ```
///     x = [[[[1],   [2],  [3],  [4]],
///           [[5],   [6],  [7],  [8]],
///           [[9],  [10], [11],  [12]],
///           [[13], [14], [15],  [16]]]]
///     ```
///
///     The output tensor has shape `[4, 2, 2, 1]` and value:
///
///     ```
///     x = [[[[1], [3]], [[9], [11]]],
///          [[[2], [4]], [[10], [12]]],
///          [[[5], [7]], [[13], [15]]],
///          [[[6], [8]], [[14], [16]]]]
///     ```
///
///     (4) For the following input of shape `[2, 2, 4, 1]`, block_shape = `[2, 2]`, and
///         paddings = `[[0, 0], [2, 0]]`:
///
///     ```
///     x = [[[[1],   [2],  [3],  [4]],
///           [[5],   [6],  [7],  [8]]],
///          [[[9],  [10], [11],  [12]],
///           [[13], [14], [15],  [16]]]]
///     ```
///
///     The output tensor has shape `[8, 1, 3, 1]` and value:
///
///     ```
///     x = [[[[0], [1], [3]]], [[[0], [9], [11]]],
///          [[[0], [2], [4]]], [[[0], [10], [12]]],
///          [[[0], [5], [7]]], [[[0], [13], [15]]],
///          [[[0], [6], [8]]], [[[0], [14], [16]]]]
///     ```
///
///     Among others, this operation is useful for reducing atrous convolution into
///     regular convolution.
@inlinable @inline(__always)
public static func spaceToBatchND<T: TensorFlowScalar, TblockShape: BinaryInteger & TensorFlowScalar, Tpaddings: BinaryInteger & TensorFlowScalar>(
  _ input: Tensor<T>,
  blockShape: Tensor<TblockShape>,
  paddings: Tensor<Tpaddings>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("SpaceToBatchND",
    input,
    blockShape,
    paddings,
    T$dtype: T.tensorFlowDataType,
    Tblock_shape$dtype: TblockShape.tensorFlowDataType,
    Tpaddings$dtype: Tpaddings.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// SpaceToDepth for tensors of type T.
///
/// Rearranges blocks of spatial data, into depth. More specifically,
/// this op outputs a copy of the input tensor where values from the `height`
/// and `width` dimensions are moved to the `depth` dimension.
/// The attr `block_size` indicates the input block size.
///
///   * Non-overlapping blocks of size `block_size x block size` are rearranged
///     into depth at each location.
///   * The depth of the output tensor is `block_size * block_size * input_depth`.
///   * The Y, X coordinates within each block of the input become the high order
///     component of the output channel index.
///   * The input tensor's height and width must be divisible by block_size.
///
/// The `data_format` attr specifies the layout of the input and output tensors
/// with the following options:
///   "NHWC": `[ batch, height, width, channels ]`
///   "NCHW": `[ batch, channels, height, width ]`
///   "NCHW_VECT_C":
///       `qint8 [ batch, channels / 4, height, width, 4 ]`
///
/// It is useful to consider the operation as transforming a 6-D Tensor.
/// e.g. for data_format = NHWC,
///      Each element in the input tensor can be specified via 6 coordinates,
///      ordered by decreasing memory layout significance as:
///      n,oY,bY,oX,bX,iC  (where n=batch index, oX, oY means X or Y coordinates
///                         within the output image, bX, bY means coordinates
///                         within the input block, iC means input channels).
///      The output would be a transpose to the following layout:
///      n,oY,oX,bY,bX,iC
///
/// This operation is useful for resizing the activations between convolutions
/// (but keeping all data), e.g. instead of pooling. It is also useful for training
/// purely convolutional models.
///
/// For example, given an input of shape `[1, 2, 2, 1]`, data_format = "NHWC" and
/// block_size = 2:
///
/// ```
/// x = [[[[1], [2]],
///       [[3], [4]]]]
/// ```
///
/// This operation will output a tensor of shape `[1, 1, 1, 4]`:
///
/// ```
/// [[[[1, 2, 3, 4]]]]
/// ```
///
/// Here, the input has a batch of 1 and each batch element has shape `[2, 2, 1]`,
/// the corresponding output will have a single element (i.e. width and height are
/// both 1) and will have a depth of 4 channels (1 * block_size * block_size).
/// The output element shape is `[1, 1, 4]`.
///
/// For an input tensor with larger depth, here of shape `[1, 2, 2, 3]`, e.g.
///
/// ```
/// x = [[[[1, 2, 3], [4, 5, 6]],
///       [[7, 8, 9], [10, 11, 12]]]]
/// ```
///
/// This operation, for block_size of 2, will return the following tensor of shape
/// `[1, 1, 1, 12]`
///
/// ```
/// [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
/// ```
///
/// Similarly, for the following input of shape `[1 4 4 1]`, and a block size of 2:
///
/// ```
/// x = [[[[1],   [2],  [5],  [6]],
///       [[3],   [4],  [7],  [8]],
///       [[9],  [10], [13],  [14]],
///       [[11], [12], [15],  [16]]]]
/// ```
///
/// the operator will return the following tensor of shape `[1 2 2 4]`:
///
/// ```
/// x = [[[[1, 2, 3, 4],
///        [5, 6, 7, 8]],
///       [[9, 10, 11, 12],
///        [13, 14, 15, 16]]]]
/// ```
///
/// - Attr block_size: The size of the spatial block.
@inlinable @inline(__always)
public static func spaceToDepth<T: TensorFlowScalar>(
  _ input: Tensor<T>,
  blockSize: Int64,
  dataFormat: DataFormat4 = .nhwc
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("SpaceToDepth",
    input,
    T$dtype: T.tensorFlowDataType,
    block_size: blockSize,
    data_format: dataFormat.cName)
  return Tensor(handle: ret)
}

/// Adds two `SparseTensor` objects to produce another `SparseTensor`.
///
/// The input `SparseTensor` objects' indices are assumed ordered in standard
/// lexicographic order.  If this is not the case, before this step run
/// `SparseReorder` to restore index ordering.
///
/// By default, if two values sum to zero at some index, the output `SparseTensor`
/// would still include that particular location in its index, storing a zero in the
/// corresponding value slot.  To override this, callers can specify `thresh`,
/// indicating that if the sum has a magnitude strictly smaller than `thresh`, its
/// corresponding value and index would then not be included.  In particular,
/// `thresh == 0` (default) means everything is kept and actual thresholding happens
/// only for a positive value.
///
/// In the following shapes, `nnz` is the count after taking `thresh` into account.
///
/// - Parameters:
///   - a_indices: 2-D.  The `indices` of the first `SparseTensor`, size `[nnz, ndims]` Matrix.
///   - a_values: 1-D.  The `values` of the first `SparseTensor`, size `[nnz]` Vector.
///   - a_shape: 1-D.  The `shape` of the first `SparseTensor`, size `[ndims]` Vector.
///   - b_indices: 2-D.  The `indices` of the second `SparseTensor`, size `[nnz, ndims]` Matrix.
///   - b_values: 1-D.  The `values` of the second `SparseTensor`, size `[nnz]` Vector.
///   - b_shape: 1-D.  The `shape` of the second `SparseTensor`, size `[ndims]` Vector.
///   - thresh: 0-D.  The magnitude threshold that determines if an output value/index
///     pair takes space.
@inlinable @inline(__always)
public static func sparseAdd<T: Numeric & TensorFlowScalar, Treal: Numeric & TensorFlowScalar>(
  aIndices: Tensor<Int64>,
  aValues: Tensor<T>,
  aShape: Tensor<Int64>,
  bIndices: Tensor<Int64>,
  bValues: Tensor<T>,
  bShape: Tensor<Int64>,
  thresh: Tensor<Treal>
) -> (sumIndices: Tensor<Int64>, sumValues: Tensor<T>, sumShape: Tensor<Int64>) {
  let ret: (TensorHandle<Int64>, TensorHandle<T>, TensorHandle<Int64>) = #tfop("SparseAdd",
    aIndices,
    aValues,
    aShape,
    bIndices,
    bValues,
    bShape,
    thresh,
    T$dtype: T.tensorFlowDataType,
    Treal$dtype: Treal.tensorFlowDataType)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// The gradient operator for the SparseAdd op.
///
/// The SparseAdd op calculates A + B, where A, B, and the sum are all represented
/// as `SparseTensor` objects.  This op takes in the upstream gradient w.r.t.
/// non-empty values of the sum, and outputs the gradients w.r.t. the non-empty
/// values of A and B.
///
/// - Parameters:
///   - backprop_val_grad: 1-D with shape `[nnz(sum)]`.  The gradient with respect to
///     the non-empty values of the sum.
///   - a_indices: 2-D.  The `indices` of the `SparseTensor` A, size `[nnz(A), ndims]`.
///   - b_indices: 2-D.  The `indices` of the `SparseTensor` B, size `[nnz(B), ndims]`.
///   - sum_indices: 2-D.  The `indices` of the sum `SparseTensor`, size
///     `[nnz(sum), ndims]`.
///
/// - Outputs:
///   - a_val_grad: 1-D with shape `[nnz(A)]`. The gradient with respect to the
///     non-empty values of A.
///   - b_val_grad: 1-D with shape `[nnz(B)]`. The gradient with respect to the
///     non-empty values of B.
@inlinable @inline(__always)
public static func sparseAddGrad<T: Numeric & TensorFlowScalar>(
  backpropValGrad: Tensor<T>,
  aIndices: Tensor<Int64>,
  bIndices: Tensor<Int64>,
  sumIndices: Tensor<Int64>
) -> (aValGrad: Tensor<T>, bValGrad: Tensor<T>) {
  let ret: (TensorHandle<T>, TensorHandle<T>) = #tfop("SparseAddGrad",
    backpropValGrad,
    aIndices,
    bIndices,
    sumIndices,
    T$dtype: T.tensorFlowDataType)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

/// Concatenates a list of `SparseTensor` along the specified dimension.
///
/// Concatenation is with respect to the dense versions of these sparse tensors.
/// It is assumed that each input is a `SparseTensor` whose elements are ordered
/// along increasing dimension number.
///
/// All inputs' shapes must match, except for the concat dimension.  The
/// `indices`, `values`, and `shapes` lists must have the same length.
///
/// The output shape is identical to the inputs', except along the concat
/// dimension, where it is the sum of the inputs' sizes along that dimension.
///
/// The output elements will be resorted to preserve the sort order along
/// increasing dimension number.
///
/// This op runs in `O(M log M)` time, where `M` is the total number of non-empty
/// values across all inputs. This is due to the need for an internal sort in
/// order to concatenate efficiently across an arbitrary dimension.
///
/// For example, if `concat_dim = 1` and the inputs are
///
///     sp_inputs[0]: shape = [2, 3]
///     [0, 2]: "a"
///     [1, 0]: "b"
///     [1, 1]: "c"
///
///     sp_inputs[1]: shape = [2, 4]
///     [0, 1]: "d"
///     [0, 2]: "e"
///
/// then the output will be
///
///     shape = [2, 7]
///     [0, 2]: "a"
///     [0, 4]: "d"
///     [0, 5]: "e"
///     [1, 0]: "b"
///     [1, 1]: "c"
///
/// Graphically this is equivalent to doing
///
///     [    a] concat [  d e  ] = [    a   d e  ]
///     [b c  ]        [       ]   [b c          ]
///
/// - Parameters:
///   - indices: 2-D.  Indices of each input `SparseTensor`.
///   - values: 1-D.  Non-empty values of each `SparseTensor`.
///   - shapes: 1-D.  Shapes of each `SparseTensor`.
///
/// - Attr concat_dim: Dimension to concatenate along. Must be in range [-rank, rank),
///   where rank is the number of dimensions in each input `SparseTensor`.
///
/// - Outputs:
///   - output_indices: 2-D.  Indices of the concatenated `SparseTensor`.
///   - output_values: 1-D.  Non-empty values of the concatenated `SparseTensor`.
///   - output_shape: 1-D.  Shape of the concatenated `SparseTensor`.
@inlinable @inline(__always)
public static func sparseConcat<T: TensorFlowScalar>(
  indices: [Tensor<Int64>],
  _ values: [Tensor<T>],
  shapes: [Tensor<Int64>],
  concatDim: Int64
) -> (outputIndices: Tensor<Int64>, outputValues: Tensor<T>, outputShape: Tensor<Int64>) {
  let ret: (TensorHandle<Int64>, TensorHandle<T>, TensorHandle<Int64>) = #tfop("SparseConcat",
    indices,
    values,
    shapes,
    T$dtype: T.tensorFlowDataType,
    concat_dim: concatDim)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Generates sparse cross from a list of sparse and dense tensors.
///
/// The op takes two lists, one of 2D `SparseTensor` and one of 2D `Tensor`, each
/// representing features of one feature column. It outputs a 2D `SparseTensor` with
/// the batchwise crosses of these features.
///
/// For example, if the inputs are
///
///     inputs[0]: SparseTensor with shape = [2, 2]
///     [0, 0]: "a"
///     [1, 0]: "b"
///     [1, 1]: "c"
///
///     inputs[1]: SparseTensor with shape = [2, 1]
///     [0, 0]: "d"
///     [1, 0]: "e"
///
///     inputs[2]: Tensor [["f"], ["g"]]
///
/// then the output will be
///
///     shape = [2, 2]
///     [0, 0]: "a_X_d_X_f"
///     [1, 0]: "b_X_e_X_g"
///     [1, 1]: "c_X_e_X_g"
///
/// if hashed_output=true then the output will be
///
///     shape = [2, 2]
///     [0, 0]: FingerprintCat64(
///                 Fingerprint64("f"), FingerprintCat64(
///                     Fingerprint64("d"), Fingerprint64("a")))
///     [1, 0]: FingerprintCat64(
///                 Fingerprint64("g"), FingerprintCat64(
///                     Fingerprint64("e"), Fingerprint64("b")))
///     [1, 1]: FingerprintCat64(
///                 Fingerprint64("g"), FingerprintCat64(
///                     Fingerprint64("e"), Fingerprint64("c")))
///
/// - Parameters:
///   - indices: 2-D.  Indices of each input `SparseTensor`.
///   - values: 1-D.   values of each `SparseTensor`.
///   - shapes: 1-D.   Shapes of each `SparseTensor`.
///   - dense_inputs: 2-D.    Columns represented by dense `Tensor`.
///
/// - Attrs:
///   - hashed_output: If true, returns the hash of the cross instead of the string.
///     This will allow us avoiding string manipulations.
///   - num_buckets: It is used if hashed_output is true.
///     output = hashed_value%num_buckets if num_buckets > 0 else hashed_value.
///   - hash_key: Specify the hash_key that will be used by the `FingerprintCat64`
///     function to combine the crosses fingerprints.
///
/// - Outputs:
///   - output_indices: 2-D.  Indices of the concatenated `SparseTensor`.
///   - output_values: 1-D.  Non-empty values of the concatenated or hashed
///     `SparseTensor`.
///   - output_shape: 1-D.  Shape of the concatenated `SparseTensor`.
@inlinable @inline(__always)
public static func sparseCross<SparseTypes: BinaryInteger & TensorFlowScalar, DenseTypes: BinaryInteger & TensorFlowScalar, OutType: BinaryInteger & TensorFlowScalar, InternalType: BinaryInteger & TensorFlowScalar>(
  indices: [Tensor<Int64>],
  _ values: [Tensor<SparseTypes>],
  shapes: [Tensor<Int64>],
  denseInputs: [Tensor<DenseTypes>],
  hashedOutput: Bool,
  numBuckets: Int64,
  hashKey: Int64,
  typeInternalType: InternalType.Type
) -> (outputIndices: Tensor<Int64>, outputValues: Tensor<OutType>, outputShape: Tensor<Int64>) {
  let ret: (TensorHandle<Int64>, TensorHandle<OutType>, TensorHandle<Int64>) = #tfop("SparseCross",
    indices,
    values,
    shapes,
    denseInputs,
    out_type$dtype: OutType.tensorFlowDataType,
    internal_type$dtype: InternalType.tensorFlowDataType,
    hashed_output: hashedOutput,
    num_buckets: numBuckets,
    hash_key: hashKey)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Adds up a SparseTensor and a dense Tensor, using these special rules:
///
/// (1) Broadcasts the dense side to have the same shape as the sparse side, if
///     eligible;
/// (2) Then, only the dense values pointed to by the indices of the SparseTensor
///     participate in the cwise addition.
///
/// By these rules, the result is a logical SparseTensor with exactly the same
/// indices and shape, but possibly with different non-zero values.  The output of
/// this Op is the resultant non-zero values.
///
/// - Parameters:
///   - sp_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
///     SparseTensor, possibly not in canonical ordering.
///   - sp_values: 1-D.  `N` non-empty values corresponding to `sp_indices`.
///   - sp_shape: 1-D.  Shape of the input SparseTensor.
///   - dense: `R`-D.  The dense Tensor operand.
///
/// - Output output: 1-D.  The `N` values that are operated on.
@inlinable @inline(__always)
public static func sparseDenseCwiseAdd<T: Numeric & TensorFlowScalar>(
  spIndices: Tensor<Int64>,
  spValues: Tensor<T>,
  spShape: Tensor<Int64>,
  dense: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("SparseDenseCwiseAdd",
    spIndices,
    spValues,
    spShape,
    dense,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Component-wise divides a SparseTensor by a dense Tensor.
///
/// *Limitation*: this Op only broadcasts the dense side to the sparse side, but not
/// the other direction.
///
/// - Parameters:
///   - sp_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
///     SparseTensor, possibly not in canonical ordering.
///   - sp_values: 1-D.  `N` non-empty values corresponding to `sp_indices`.
///   - sp_shape: 1-D.  Shape of the input SparseTensor.
///   - dense: `R`-D.  The dense Tensor operand.
///
/// - Output output: 1-D.  The `N` values that are operated on.
@inlinable @inline(__always)
public static func sparseDenseCwiseDiv<T: Numeric & TensorFlowScalar>(
  spIndices: Tensor<Int64>,
  spValues: Tensor<T>,
  spShape: Tensor<Int64>,
  dense: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("SparseDenseCwiseDiv",
    spIndices,
    spValues,
    spShape,
    dense,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Component-wise multiplies a SparseTensor by a dense Tensor.
///
/// The output locations corresponding to the implicitly zero elements in the sparse
/// tensor will be zero (i.e., will not take up storage space), regardless of the
/// contents of the dense tensor (even if it's +/-INF and that INF*0 == NaN).
///
/// *Limitation*: this Op only broadcasts the dense side to the sparse side, but not
/// the other direction.
///
/// - Parameters:
///   - sp_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
///     SparseTensor, possibly not in canonical ordering.
///   - sp_values: 1-D.  `N` non-empty values corresponding to `sp_indices`.
///   - sp_shape: 1-D.  Shape of the input SparseTensor.
///   - dense: `R`-D.  The dense Tensor operand.
///
/// - Output output: 1-D.  The `N` values that are operated on.
@inlinable @inline(__always)
public static func sparseDenseCwiseMul<T: Numeric & TensorFlowScalar>(
  spIndices: Tensor<Int64>,
  spValues: Tensor<T>,
  spShape: Tensor<Int64>,
  dense: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("SparseDenseCwiseMul",
    spIndices,
    spValues,
    spShape,
    dense,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Fills empty rows in the input 2-D `SparseTensor` with a default value.
///
/// The input `SparseTensor` is represented via the tuple of inputs
/// (`indices`, `values`, `dense_shape`).  The output `SparseTensor` has the
/// same `dense_shape` but with indices `output_indices` and values
/// `output_values`.
///
/// This op inserts a single entry for every row that doesn't have any values.
/// The index is created as `[row, 0, ..., 0]` and the inserted value
/// is `default_value`.
///
/// For example, suppose `sp_input` has shape `[5, 6]` and non-empty values:
///
///     [0, 1]: a
///     [0, 3]: b
///     [2, 0]: c
///     [3, 1]: d
///
/// Rows 1 and 4 are empty, so the output will be of shape `[5, 6]` with values:
///
///     [0, 1]: a
///     [0, 3]: b
///     [1, 0]: default_value
///     [2, 0]: c
///     [3, 1]: d
///     [4, 0]: default_value
///
/// The output `SparseTensor` will be in row-major order and will have the
/// same shape as the input.
///
/// This op also returns an indicator vector shaped `[dense_shape[0]]` such that
///
///     empty_row_indicator[i] = True iff row i was an empty row.
///
/// And a reverse index map vector shaped `[indices.shape[0]]` that is used during
/// backpropagation,
///
///     reverse_index_map[j] = out_j s.t. indices[j, :] == output_indices[out_j, :]
///
/// - Parameters:
///   - indices: 2-D. the indices of the sparse tensor.
///   - values: 1-D. the values of the sparse tensor.
///   - dense_shape: 1-D. the shape of the sparse tensor.
///   - default_value: 0-D. default value to insert into location `[row, 0, ..., 0]`
///       for rows missing from the input sparse tensor.
///     output indices: 2-D. the indices of the filled sparse tensor.
///
/// - Outputs:
///   - output_values: 1-D. the values of the filled sparse tensor.
///   - empty_row_indicator: 1-D. whether the dense row was missing in the
///     input sparse tensor.
///   - reverse_index_map: 1-D. a map from the input indices to the output indices.
@inlinable @inline(__always)
public static func sparseFillEmptyRows<T: TensorFlowScalar>(
  indices: Tensor<Int64>,
  _ values: Tensor<T>,
  denseShape: Tensor<Int64>,
  defaultValue: Tensor<T>
) -> (outputIndices: Tensor<Int64>, outputValues: Tensor<T>, emptyRowIndicator: Tensor<Bool>, reverseIndexMap: Tensor<Int64>) {
  let ret: (TensorHandle<Int64>, TensorHandle<T>, TensorHandle<Bool>, TensorHandle<Int64>) = #tfop("SparseFillEmptyRows",
    indices,
    values,
    denseShape,
    defaultValue,
    T$dtype: T.tensorFlowDataType)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2), Tensor(handle: ret.3))
}

/// The gradient of SparseFillEmptyRows.
///
/// Takes vectors reverse_index_map, shaped `[N]`, and grad_values,
/// shaped `[N_full]`, where `N_full >= N` and copies data into either
/// `d_values` or `d_default_value`.  Here `d_values` is shaped `[N]` and
/// `d_default_value` is a scalar.
///
///   d_values[j] = grad_values[reverse_index_map[j]]
///   d_default_value = sum_{k : 0 .. N_full - 1} (
///      grad_values[k] * 1{k not in reverse_index_map})
///
/// - Parameters:
///   - reverse_index_map: 1-D.  The reverse index map from SparseFillEmptyRows.
///   - grad_values: 1-D.  The gradients from backprop.
///
/// - Outputs:
///   - d_values: 1-D.  The backprop into values.
///   - d_default_value: 0-D.  The backprop into default_value.
@inlinable @inline(__always)
public static func sparseFillEmptyRowsGrad<T: TensorFlowScalar>(
  reverseIndexMap: Tensor<Int64>,
  gradValues: Tensor<T>
) -> (dValues: Tensor<T>, dDefaultValue: Tensor<T>) {
  let ret: (TensorHandle<T>, TensorHandle<T>) = #tfop("SparseFillEmptyRowsGrad",
    reverseIndexMap,
    gradValues,
    T$dtype: T.tensorFlowDataType)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

/// Multiply matrix "a" by matrix "b".
///
/// The inputs must be two-dimensional matrices and the inner dimension of "a" must
/// match the outer dimension of "b". Both "a" and "b" must be `Tensor`s not
/// `SparseTensor`s.  This op is optimized for the case where at least one of "a" or
/// "b" is sparse, in the sense that they have a large proportion of zero values.
/// The breakeven for using this versus a dense matrix multiply on one platform was
/// 30% zero values in the sparse matrix.
///
/// The gradient computation of this operation will only take advantage of sparsity
/// in the input gradient when that gradient comes from a Relu.
@inlinable @inline(__always)
public static func sparseMatMul<Ta: FloatingPoint & TensorFlowScalar, Tb: FloatingPoint & TensorFlowScalar>(
  _ a: Tensor<Ta>,
  _ b: Tensor<Tb>,
  transposeA: Bool = false,
  transposeB: Bool = false,
  aIsSparse: Bool = false,
  bIsSparse: Bool = false
) -> Tensor<Float> {
  let ret: TensorHandle<Float> = #tfop("SparseMatMul",
    a,
    b,
    Ta$dtype: Ta.tensorFlowDataType,
    Tb$dtype: Tb.tensorFlowDataType,
    transpose_a: transposeA,
    transpose_b: transposeB,
    a_is_sparse: aIsSparse,
    b_is_sparse: bIsSparse)
  return Tensor(handle: ret)
}

/// Computes the max of elements across dimensions of a SparseTensor.
///
/// This Op takes a SparseTensor and is the sparse counterpart to
/// `tf.reduce_max()`.  In particular, this Op also returns a dense `Tensor`
/// instead of a sparse one.
///
/// Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
/// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
/// `reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
/// with length 1.
///
/// If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
/// with a single element is returned.  Additionally, the axes can be negative,
/// which are interpreted according to the indexing rules in Python.
///
/// - Parameters:
///   - input_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
///     SparseTensor, possibly not in canonical ordering.
///   - input_values: 1-D.  `N` non-empty values corresponding to `input_indices`.
///   - input_shape: 1-D.  Shape of the input SparseTensor.
///   - reduction_axes: 1-D.  Length-`K` vector containing the reduction axes.
///
/// - Attr keep_dims: If true, retain reduced dimensions with length 1.
///
/// - Output output: `R-K`-D.  The reduced Tensor.
@inlinable @inline(__always)
public static func sparseReduceMax<T: Numeric & TensorFlowScalar>(
  inputIndices: Tensor<Int64>,
  inputValues: Tensor<T>,
  inputShape: Tensor<Int64>,
  reductionAxes: Tensor<Int32>,
  keepDims: Bool = false
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("SparseReduceMax",
    inputIndices,
    inputValues,
    inputShape,
    reductionAxes,
    T$dtype: T.tensorFlowDataType,
    keep_dims: keepDims)
  return Tensor(handle: ret)
}

/// Computes the max of elements across dimensions of a SparseTensor.
///
/// This Op takes a SparseTensor and is the sparse counterpart to
/// `tf.reduce_max()`.  In contrast to SparseReduceMax, this Op returns a
/// SparseTensor.
///
/// Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
/// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
/// `reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
/// with length 1.
///
/// If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
/// with a single element is returned.  Additionally, the axes can be negative,
/// which are interpreted according to the indexing rules in Python.
///
/// - Parameters:
///   - input_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
///     SparseTensor, possibly not in canonical ordering.
///   - input_values: 1-D.  `N` non-empty values corresponding to `input_indices`.
///   - input_shape: 1-D.  Shape of the input SparseTensor.
///   - reduction_axes: 1-D.  Length-`K` vector containing the reduction axes.
///
/// - Attr keep_dims: If true, retain reduced dimensions with length 1.
@inlinable @inline(__always)
public static func sparseReduceMaxSparse<T: Numeric & TensorFlowScalar>(
  inputIndices: Tensor<Int64>,
  inputValues: Tensor<T>,
  inputShape: Tensor<Int64>,
  reductionAxes: Tensor<Int32>,
  keepDims: Bool = false
) -> (outputIndices: Tensor<Int64>, outputValues: Tensor<T>, outputShape: Tensor<Int64>) {
  let ret: (TensorHandle<Int64>, TensorHandle<T>, TensorHandle<Int64>) = #tfop("SparseReduceMaxSparse",
    inputIndices,
    inputValues,
    inputShape,
    reductionAxes,
    T$dtype: T.tensorFlowDataType,
    keep_dims: keepDims)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Computes the sum of elements across dimensions of a SparseTensor.
///
/// This Op takes a SparseTensor and is the sparse counterpart to
/// `tf.reduce_sum()`.  In particular, this Op also returns a dense `Tensor`
/// instead of a sparse one.
///
/// Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
/// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
/// `reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
/// with length 1.
///
/// If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
/// with a single element is returned.  Additionally, the axes can be negative,
/// which are interpreted according to the indexing rules in Python.
///
/// - Parameters:
///   - input_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
///     SparseTensor, possibly not in canonical ordering.
///   - input_values: 1-D.  `N` non-empty values corresponding to `input_indices`.
///   - input_shape: 1-D.  Shape of the input SparseTensor.
///   - reduction_axes: 1-D.  Length-`K` vector containing the reduction axes.
///
/// - Attr keep_dims: If true, retain reduced dimensions with length 1.
///
/// - Output output: `R-K`-D.  The reduced Tensor.
@inlinable @inline(__always)
public static func sparseReduceSum<T: Numeric & TensorFlowScalar>(
  inputIndices: Tensor<Int64>,
  inputValues: Tensor<T>,
  inputShape: Tensor<Int64>,
  reductionAxes: Tensor<Int32>,
  keepDims: Bool = false
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("SparseReduceSum",
    inputIndices,
    inputValues,
    inputShape,
    reductionAxes,
    T$dtype: T.tensorFlowDataType,
    keep_dims: keepDims)
  return Tensor(handle: ret)
}

/// Computes the sum of elements across dimensions of a SparseTensor.
///
/// This Op takes a SparseTensor and is the sparse counterpart to
/// `tf.reduce_sum()`.  In contrast to SparseReduceSum, this Op returns a
/// SparseTensor.
///
/// Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
/// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
/// `reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
/// with length 1.
///
/// If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
/// with a single element is returned.  Additionally, the axes can be negative,
/// which are interpreted according to the indexing rules in Python.
///
/// - Parameters:
///   - input_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
///     SparseTensor, possibly not in canonical ordering.
///   - input_values: 1-D.  `N` non-empty values corresponding to `input_indices`.
///   - input_shape: 1-D.  Shape of the input SparseTensor.
///   - reduction_axes: 1-D.  Length-`K` vector containing the reduction axes.
///
/// - Attr keep_dims: If true, retain reduced dimensions with length 1.
@inlinable @inline(__always)
public static func sparseReduceSumSparse<T: Numeric & TensorFlowScalar>(
  inputIndices: Tensor<Int64>,
  inputValues: Tensor<T>,
  inputShape: Tensor<Int64>,
  reductionAxes: Tensor<Int32>,
  keepDims: Bool = false
) -> (outputIndices: Tensor<Int64>, outputValues: Tensor<T>, outputShape: Tensor<Int64>) {
  let ret: (TensorHandle<Int64>, TensorHandle<T>, TensorHandle<Int64>) = #tfop("SparseReduceSumSparse",
    inputIndices,
    inputValues,
    inputShape,
    reductionAxes,
    T$dtype: T.tensorFlowDataType,
    keep_dims: keepDims)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Reorders a SparseTensor into the canonical, row-major ordering.
///
/// Note that by convention, all sparse ops preserve the canonical ordering along
/// increasing dimension number. The only time ordering can be violated is during
/// manual manipulation of the indices and values vectors to add entries.
///
/// Reordering does not affect the shape of the SparseTensor.
///
/// If the tensor has rank `R` and `N` non-empty values, `input_indices` has
/// shape `[N, R]`, input_values has length `N`, and input_shape has length `R`.
///
/// - Parameters:
///   - input_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
///     SparseTensor, possibly not in canonical ordering.
///   - input_values: 1-D.  `N` non-empty values corresponding to `input_indices`.
///   - input_shape: 1-D.  Shape of the input SparseTensor.
///
/// - Outputs:
///   - output_indices: 2-D.  `N x R` matrix with the same indices as input_indices, but
///     in canonical row-major ordering.
///   - output_values: 1-D.  `N` non-empty values corresponding to `output_indices`.
@inlinable @inline(__always)
public static func sparseReorder<T: TensorFlowScalar>(
  inputIndices: Tensor<Int64>,
  inputValues: Tensor<T>,
  inputShape: Tensor<Int64>
) -> (outputIndices: Tensor<Int64>, outputValues: Tensor<T>) {
  let ret: (TensorHandle<Int64>, TensorHandle<T>) = #tfop("SparseReorder",
    inputIndices,
    inputValues,
    inputShape,
    T$dtype: T.tensorFlowDataType)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

/// Reshapes a SparseTensor to represent values in a new dense shape.
///
/// This operation has the same semantics as reshape on the represented dense
/// tensor.  The `input_indices` are recomputed based on the requested `new_shape`.
///
/// If one component of `new_shape` is the special value -1, the size of that
/// dimension is computed so that the total dense size remains constant.  At
/// most one component of `new_shape` can be -1.  The number of dense elements
/// implied by `new_shape` must be the same as the number of dense elements
/// originally implied by `input_shape`.
///
/// Reshaping does not affect the order of values in the SparseTensor.
///
/// If the input tensor has rank `R_in` and `N` non-empty values, and `new_shape`
/// has length `R_out`, then `input_indices` has shape `[N, R_in]`,
/// `input_shape` has length `R_in`, `output_indices` has shape `[N, R_out]`, and
/// `output_shape` has length `R_out`.
///
/// - Parameters:
///   - input_indices: 2-D.  `N x R_in` matrix with the indices of non-empty values in a
///     SparseTensor.
///   - input_shape: 1-D.  `R_in` vector with the input SparseTensor's dense shape.
///   - new_shape: 1-D.  `R_out` vector with the requested new dense shape.
///
/// - Outputs:
///   - output_indices: 2-D.  `N x R_out` matrix with the updated indices of non-empty
///     values in the output SparseTensor.
///   - output_shape: 1-D.  `R_out` vector with the full dense shape of the output
///     SparseTensor.  This is the same as `new_shape` but with any -1 dimensions
///     filled in.
@inlinable @inline(__always)
public static func sparseReshape(
  inputIndices: Tensor<Int64>,
  inputShape: Tensor<Int64>,
  newShape: Tensor<Int64>
) -> (outputIndices: Tensor<Int64>, outputShape: Tensor<Int64>) {
  let ret: (TensorHandle<Int64>, TensorHandle<Int64>) = #tfop("SparseReshape",
    inputIndices,
    inputShape,
    newShape)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

/// Computes the mean along sparse segments of a tensor.
///
/// See `tf.sparse.segment_sum` for usage examples.
///
/// Like `SegmentMean`, but `segment_ids` can have rank less than `data`'s first
/// dimension, selecting a subset of dimension 0, specified by `indices`.
///
/// - Parameters:
///   - indices: A 1-D tensor. Has same rank as `segment_ids`.
///   - segment_ids: A 1-D tensor. Values should be sorted and can be repeated.
///
/// - Output output: Has same shape as data, except for dimension 0 which
///   has size `k`, the number of segments.
@inlinable @inline(__always)
public static func sparseSegmentMean<T: FloatingPoint & TensorFlowScalar, Tidx: BinaryInteger & TensorFlowScalar>(
  data: Tensor<T>,
  indices: Tensor<Tidx>,
  segmentIds: Tensor<Int32>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("SparseSegmentMean",
    data,
    indices,
    segmentIds,
    T$dtype: T.tensorFlowDataType,
    Tidx$dtype: Tidx.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes gradients for SparseSegmentMean.
///
/// Returns tensor "output" with same shape as grad, except for dimension 0 whose
/// value is output_dim0.
///
/// - Parameters:
///   - grad: gradient propagated to the SparseSegmentMean op.
///   - indices: indices passed to the corresponding SparseSegmentMean op.
///   - segment_ids: segment_ids passed to the corresponding SparseSegmentMean op.
///   - output_dim0: dimension 0 of "data" passed to SparseSegmentMean op.
@inlinable @inline(__always)
public static func sparseSegmentMeanGrad<T: FloatingPoint & TensorFlowScalar, Tidx: BinaryInteger & TensorFlowScalar>(
  grad: Tensor<T>,
  indices: Tensor<Tidx>,
  segmentIds: Tensor<Int32>,
  outputDim0: Tensor<Int32>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("SparseSegmentMeanGrad",
    grad,
    indices,
    segmentIds,
    outputDim0,
    T$dtype: T.tensorFlowDataType,
    Tidx$dtype: Tidx.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the mean along sparse segments of a tensor.
///
/// Like `SparseSegmentMean`, but allows missing ids in `segment_ids`. If an id is
/// misisng, the `output` tensor at that position will be zeroed.
///
/// Read
/// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
/// for an explanation of segments.
///
/// - Parameters:
///   - indices: A 1-D tensor. Has same rank as `segment_ids`.
///   - segment_ids: A 1-D tensor. Values should be sorted and can be repeated.
///   - num_segments: Should equal the number of distinct segment IDs.
///
/// - Output output: Has same shape as data, except for dimension 0 which has size
///   `num_segments`.
@inlinable @inline(__always)
public static func sparseSegmentMeanWithNumSegments<T: FloatingPoint & TensorFlowScalar, Tidx: BinaryInteger & TensorFlowScalar, Tnumsegments: BinaryInteger & TensorFlowScalar>(
  data: Tensor<T>,
  indices: Tensor<Tidx>,
  segmentIds: Tensor<Int32>,
  numSegments: Tensor<Tnumsegments>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("SparseSegmentMeanWithNumSegments",
    data,
    indices,
    segmentIds,
    numSegments,
    T$dtype: T.tensorFlowDataType,
    Tidx$dtype: Tidx.tensorFlowDataType,
    Tnumsegments$dtype: Tnumsegments.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the sum along sparse segments of a tensor divided by the sqrt of N.
///
/// N is the size of the segment being reduced.
///
/// See `tf.sparse.segment_sum` for usage examples.
///
///
/// - Parameters:
///   - indices: A 1-D tensor. Has same rank as `segment_ids`.
///   - segment_ids: A 1-D tensor. Values should be sorted and can be repeated.
///
/// - Output output: Has same shape as data, except for dimension 0 which
///   has size `k`, the number of segments.
@inlinable @inline(__always)
public static func sparseSegmentSqrtN<T: FloatingPoint & TensorFlowScalar, Tidx: BinaryInteger & TensorFlowScalar>(
  data: Tensor<T>,
  indices: Tensor<Tidx>,
  segmentIds: Tensor<Int32>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("SparseSegmentSqrtN",
    data,
    indices,
    segmentIds,
    T$dtype: T.tensorFlowDataType,
    Tidx$dtype: Tidx.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes gradients for SparseSegmentSqrtN.
///
/// Returns tensor "output" with same shape as grad, except for dimension 0 whose
/// value is output_dim0.
///
/// - Parameters:
///   - grad: gradient propagated to the SparseSegmentSqrtN op.
///   - indices: indices passed to the corresponding SparseSegmentSqrtN op.
///   - segment_ids: segment_ids passed to the corresponding SparseSegmentSqrtN op.
///   - output_dim0: dimension 0 of "data" passed to SparseSegmentSqrtN op.
@inlinable @inline(__always)
public static func sparseSegmentSqrtNGrad<T: FloatingPoint & TensorFlowScalar, Tidx: BinaryInteger & TensorFlowScalar>(
  grad: Tensor<T>,
  indices: Tensor<Tidx>,
  segmentIds: Tensor<Int32>,
  outputDim0: Tensor<Int32>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("SparseSegmentSqrtNGrad",
    grad,
    indices,
    segmentIds,
    outputDim0,
    T$dtype: T.tensorFlowDataType,
    Tidx$dtype: Tidx.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the sum along sparse segments of a tensor divided by the sqrt of N.
///
/// N is the size of the segment being reduced.
///
/// Like `SparseSegmentSqrtN`, but allows missing ids in `segment_ids`. If an id is
/// misisng, the `output` tensor at that position will be zeroed.
///
/// Read
/// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
/// for an explanation of segments.
///
/// - Parameters:
///   - indices: A 1-D tensor. Has same rank as `segment_ids`.
///   - segment_ids: A 1-D tensor. Values should be sorted and can be repeated.
///   - num_segments: Should equal the number of distinct segment IDs.
///
/// - Output output: Has same shape as data, except for dimension 0 which
///   has size `k`, the number of segments.
@inlinable @inline(__always)
public static func sparseSegmentSqrtNWithNumSegments<T: FloatingPoint & TensorFlowScalar, Tidx: BinaryInteger & TensorFlowScalar, Tnumsegments: BinaryInteger & TensorFlowScalar>(
  data: Tensor<T>,
  indices: Tensor<Tidx>,
  segmentIds: Tensor<Int32>,
  numSegments: Tensor<Tnumsegments>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("SparseSegmentSqrtNWithNumSegments",
    data,
    indices,
    segmentIds,
    numSegments,
    T$dtype: T.tensorFlowDataType,
    Tidx$dtype: Tidx.tensorFlowDataType,
    Tnumsegments$dtype: Tnumsegments.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the sum along sparse segments of a tensor.
///
/// Read
/// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
/// for an explanation of segments.
///
/// Like `SegmentSum`, but `segment_ids` can have rank less than `data`'s first
/// dimension, selecting a subset of dimension 0, specified by `indices`.
///
/// For example:
///
/// ```python
/// c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])
///
/// # Select two rows, one segment.
/// tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 0]))
/// # => [[0 0 0 0]]
///
/// # Select two rows, two segment.
/// tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 1]))
/// # => [[ 1  2  3  4]
/// #     [-1 -2 -3 -4]]
///
/// # Select all rows, two segments.
/// tf.sparse_segment_sum(c, tf.constant([0, 1, 2]), tf.constant([0, 0, 1]))
/// # => [[0 0 0 0]
/// #     [5 6 7 8]]
///
/// # Which is equivalent to:
/// tf.segment_sum(c, tf.constant([0, 0, 1]))
/// ```
///
/// - Parameters:
///   - indices: A 1-D tensor. Has same rank as `segment_ids`.
///   - segment_ids: A 1-D tensor. Values should be sorted and can be repeated.
///
/// - Output output: Has same shape as data, except for dimension 0 which
///   has size `k`, the number of segments.
@inlinable @inline(__always)
public static func sparseSegmentSum<T: Numeric & TensorFlowScalar, Tidx: BinaryInteger & TensorFlowScalar>(
  data: Tensor<T>,
  indices: Tensor<Tidx>,
  segmentIds: Tensor<Int32>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("SparseSegmentSum",
    data,
    indices,
    segmentIds,
    T$dtype: T.tensorFlowDataType,
    Tidx$dtype: Tidx.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the sum along sparse segments of a tensor.
///
/// Like `SparseSegmentSum`, but allows missing ids in `segment_ids`. If an id is
/// misisng, the `output` tensor at that position will be zeroed.
///
/// Read
/// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/sparse#Segmentation)
/// for an explanation of segments.
///
/// For example:
///
/// ```python
/// c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])
///
/// tf.sparse_segment_sum_with_num_segments(
///     c, tf.constant([0, 1]), tf.constant([0, 0]), num_segments=3)
/// # => [[0 0 0 0]
/// #     [0 0 0 0]
/// #     [0 0 0 0]]
///
/// tf.sparse_segment_sum_with_num_segments(c,
///                                         tf.constant([0, 1]),
///                                         tf.constant([0, 2],
///                                         num_segments=4))
/// # => [[ 1  2  3  4]
/// #     [ 0  0  0  0]
/// #     [-1 -2 -3 -4]
/// #     [ 0  0  0  0]]
/// ```
///
/// - Parameters:
///   - indices: A 1-D tensor. Has same rank as `segment_ids`.
///   - segment_ids: A 1-D tensor. Values should be sorted and can be repeated.
///   - num_segments: Should equal the number of distinct segment IDs.
///
/// - Output output: Has same shape as data, except for dimension 0 which
///   has size `num_segments`.
@inlinable @inline(__always)
public static func sparseSegmentSumWithNumSegments<T: Numeric & TensorFlowScalar, Tidx: BinaryInteger & TensorFlowScalar, Tnumsegments: BinaryInteger & TensorFlowScalar>(
  data: Tensor<T>,
  indices: Tensor<Tidx>,
  segmentIds: Tensor<Int32>,
  numSegments: Tensor<Tnumsegments>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("SparseSegmentSumWithNumSegments",
    data,
    indices,
    segmentIds,
    numSegments,
    T$dtype: T.tensorFlowDataType,
    Tidx$dtype: Tidx.tensorFlowDataType,
    Tnumsegments$dtype: Tnumsegments.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Slice a `SparseTensor` based on the `start` and `size`.
///
/// For example, if the input is
///
///     input_tensor = shape = [2, 7]
///     [    a   d e  ]
///     [b c          ]
///
/// Graphically the output tensors are:
///
///     sparse_slice([0, 0], [2, 4]) = shape = [2, 4]
///     [    a  ]
///     [b c    ]
///
///     sparse_slice([0, 4], [2, 3]) = shape = [2, 3]
///     [ d e  ]
///     [      ]
///
/// - Parameters:
///   - indices: 2-D tensor represents the indices of the sparse tensor.
///   - values: 1-D tensor represents the values of the sparse tensor.
///   - shape: 1-D. tensor represents the shape of the sparse tensor.
///   - start: 1-D. tensor represents the start of the slice.
///   - size: 1-D. tensor represents the size of the slice.
///     output indices: A list of 1-D tensors represents the indices of the output
///     sparse tensors.
///
/// - Outputs:
///   - output_values: A list of 1-D tensors represents the values of the output sparse
///     tensors.
///   - output_shape: A list of 1-D tensors represents the shape of the output sparse
///     tensors.
@inlinable @inline(__always)
public static func sparseSlice<T: TensorFlowScalar>(
  indices: Tensor<Int64>,
  _ values: Tensor<T>,
  shape: Tensor<Int64>,
  start: Tensor<Int64>,
  size: Tensor<Int64>
) -> (outputIndices: Tensor<Int64>, outputValues: Tensor<T>, outputShape: Tensor<Int64>) {
  let ret: (TensorHandle<Int64>, TensorHandle<T>, TensorHandle<Int64>) = #tfop("SparseSlice",
    indices,
    values,
    shape,
    start,
    size,
    T$dtype: T.tensorFlowDataType)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// The gradient operator for the SparseSlice op.
///
/// This op takes in the upstream gradient w.r.t. non-empty values of
/// the sliced `SparseTensor`, and outputs the gradients w.r.t.
/// the non-empty values of input `SparseTensor`.
///
/// - Parameters:
///   - backprop_val_grad: 1-D. The gradient with respect to
///     the non-empty values of the sliced `SparseTensor`.
///   - input_indices: 2-D.  The `indices` of the input `SparseTensor`.
///   - input_start: 1-D. tensor represents the start of the slice.
///   - output_indices: 2-D.  The `indices` of the sliced `SparseTensor`.
///
/// - Output val_grad: 1-D. The gradient with respect to the non-empty values of input `SparseTensor`.
@inlinable @inline(__always)
public static func sparseSliceGrad<T: Numeric & TensorFlowScalar>(
  backpropValGrad: Tensor<T>,
  inputIndices: Tensor<Int64>,
  inputStart: Tensor<Int64>,
  outputIndices: Tensor<Int64>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("SparseSliceGrad",
    backpropValGrad,
    inputIndices,
    inputStart,
    outputIndices,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Applies softmax to a batched N-D `SparseTensor`.
///
/// The inputs represent an N-D SparseTensor  with logical shape `[..., B, C]`
/// (where `N >= 2`), and with indices sorted in the canonical lexicographic order.
///
/// This op is equivalent to applying the normal `tf.nn.softmax()` to each innermost
/// logical submatrix with shape `[B, C]`, but with the catch that *the implicitly
/// zero elements do not participate*.  Specifically, the algorithm is equivalent
/// to the following:
///
///   (1) Applies `tf.nn.softmax()` to a densified view of each innermost submatrix
///       with shape `[B, C]`, along the size-C dimension;
///   (2) Masks out the original implicitly-zero locations;
///   (3) Renormalizes the remaining elements.
///
/// Hence, the `SparseTensor` result has exactly the same non-zero indices and
/// shape.
///
/// - Parameters:
///   - sp_indices: 2-D.  `NNZ x R` matrix with the indices of non-empty values in a
///     SparseTensor, in canonical ordering.
///   - sp_values: 1-D.  `NNZ` non-empty values corresponding to `sp_indices`.
///   - sp_shape: 1-D.  Shape of the input SparseTensor.
///
/// - Output output: 1-D.  The `NNZ` values for the result `SparseTensor`.
@inlinable @inline(__always)
public static func sparseSoftmax<T: FloatingPoint & TensorFlowScalar>(
  spIndices: Tensor<Int64>,
  spValues: Tensor<T>,
  spShape: Tensor<Int64>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("SparseSoftmax",
    spIndices,
    spValues,
    spShape,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes softmax cross entropy cost and gradients to backpropagate.
///
/// Unlike `SoftmaxCrossEntropyWithLogits`, this operation does not accept
/// a matrix of label probabilities, but rather a single label per row
/// of features.  This label is considered to have probability 1.0 for the
/// given row.
///
/// Inputs are the logits, not probabilities.
///
/// - Parameters:
///   - features: batch_size x num_classes matrix
///   - labels: batch_size vector with values in [0, num_classes).
///     This is the label for the given minibatch entry.
///
/// - Outputs:
///   - loss: Per example loss (batch_size vector).
///   - backprop: backpropagated gradients (batch_size x num_classes matrix).
@inlinable @inline(__always)
public static func sparseSoftmaxCrossEntropyWithLogits<T: FloatingPoint & TensorFlowScalar, Tlabels: BinaryInteger & TensorFlowScalar>(
  features: Tensor<T>,
  labels: Tensor<Tlabels>
) -> (loss: Tensor<T>, backprop: Tensor<T>) {
  let ret: (TensorHandle<T>, TensorHandle<T>) = #tfop("SparseSoftmaxCrossEntropyWithLogits",
    features,
    labels,
    T$dtype: T.tensorFlowDataType,
    Tlabels$dtype: Tlabels.tensorFlowDataType)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

/// Returns the element-wise max of two SparseTensors.
///
/// Assumes the two SparseTensors have the same shape, i.e., no broadcasting.
///
/// - Parameters:
///   - a_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
///     SparseTensor, in the canonical lexicographic ordering.
///   - a_values: 1-D.  `N` non-empty values corresponding to `a_indices`.
///   - a_shape: 1-D.  Shape of the input SparseTensor.
///   - b_indices: counterpart to `a_indices` for the other operand.
///   - b_values: counterpart to `a_values` for the other operand; must be of the same dtype.
///   - b_shape: counterpart to `a_shape` for the other operand; the two shapes must be equal.
///
/// - Outputs:
///   - output_indices: 2-D.  The indices of the output SparseTensor.
///   - output_values: 1-D.  The values of the output SparseTensor.
@inlinable @inline(__always)
public static func sparseSparseMaximum<T: Numeric & TensorFlowScalar>(
  aIndices: Tensor<Int64>,
  aValues: Tensor<T>,
  aShape: Tensor<Int64>,
  bIndices: Tensor<Int64>,
  bValues: Tensor<T>,
  bShape: Tensor<Int64>
) -> (outputIndices: Tensor<Int64>, outputValues: Tensor<T>) {
  let ret: (TensorHandle<Int64>, TensorHandle<T>) = #tfop("SparseSparseMaximum",
    aIndices,
    aValues,
    aShape,
    bIndices,
    bValues,
    bShape,
    T$dtype: T.tensorFlowDataType)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

/// Returns the element-wise min of two SparseTensors.
///
/// Assumes the two SparseTensors have the same shape, i.e., no broadcasting.
///
/// - Parameters:
///   - a_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
///     SparseTensor, in the canonical lexicographic ordering.
///   - a_values: 1-D.  `N` non-empty values corresponding to `a_indices`.
///   - a_shape: 1-D.  Shape of the input SparseTensor.
///   - b_indices: counterpart to `a_indices` for the other operand.
///   - b_values: counterpart to `a_values` for the other operand; must be of the same dtype.
///   - b_shape: counterpart to `a_shape` for the other operand; the two shapes must be equal.
///
/// - Outputs:
///   - output_indices: 2-D.  The indices of the output SparseTensor.
///   - output_values: 1-D.  The values of the output SparseTensor.
@inlinable @inline(__always)
public static func sparseSparseMinimum<T: Numeric & TensorFlowScalar>(
  aIndices: Tensor<Int64>,
  aValues: Tensor<T>,
  aShape: Tensor<Int64>,
  bIndices: Tensor<Int64>,
  bValues: Tensor<T>,
  bShape: Tensor<Int64>
) -> (outputIndices: Tensor<Int64>, outputValues: Tensor<T>) {
  let ret: (TensorHandle<Int64>, TensorHandle<T>) = #tfop("SparseSparseMinimum",
    aIndices,
    aValues,
    aShape,
    bIndices,
    bValues,
    bShape,
    T$dtype: T.tensorFlowDataType)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

/// Adds up a `SparseTensor` and a dense `Tensor`, producing a dense `Tensor`.
///
/// This Op does not require `a_indices` be sorted in standard lexicographic order.
///
/// - Parameters:
///   - a_indices: 2-D.  The `indices` of the `SparseTensor`, with shape `[nnz, ndims]`.
///   - a_values: 1-D.  The `values` of the `SparseTensor`, with shape `[nnz]`.
///   - a_shape: 1-D.  The `shape` of the `SparseTensor`, with shape `[ndims]`.
///   - b: `ndims`-D Tensor.  With shape `a_shape`.
@inlinable @inline(__always)
public static func sparseTensorDenseAdd<T: Numeric & TensorFlowScalar, Tindices: BinaryInteger & TensorFlowScalar>(
  aIndices: Tensor<Tindices>,
  aValues: Tensor<T>,
  aShape: Tensor<Tindices>,
  _ b: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("SparseTensorDenseAdd",
    aIndices,
    aValues,
    aShape,
    b,
    T$dtype: T.tensorFlowDataType,
    Tindices$dtype: Tindices.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Multiply SparseTensor (of rank 2) "A" by dense matrix "B".
///
/// No validity checking is performed on the indices of A.  However, the following
/// input format is recommended for optimal behavior:
///
/// if adjoint_a == false:
///   A should be sorted in lexicographically increasing order.  Use SparseReorder
///   if you're not sure.
/// if adjoint_a == true:
///   A should be sorted in order of increasing dimension 1 (i.e., "column major"
///   order instead of "row major" order).
///
/// - Parameters:
///   - a_indices: 2-D.  The `indices` of the `SparseTensor`, size `[nnz, 2]` Matrix.
///   - a_values: 1-D.  The `values` of the `SparseTensor`, size `[nnz]` Vector.
///   - a_shape: 1-D.  The `shape` of the `SparseTensor`, size `[2]` Vector.
///   - b: 2-D.  A dense Matrix.
///
/// - Attrs:
///   - adjoint_a: Use the adjoint of A in the matrix multiply.  If A is complex, this
///     is transpose(conj(A)).  Otherwise it's transpose(A).
///   - adjoint_b: Use the adjoint of B in the matrix multiply.  If B is complex, this
///     is transpose(conj(B)).  Otherwise it's transpose(B).
@inlinable @inline(__always)
public static func sparseTensorDenseMatMul<T: TensorFlowScalar, Tindices: BinaryInteger & TensorFlowScalar>(
  aIndices: Tensor<Tindices>,
  aValues: Tensor<T>,
  aShape: Tensor<Int64>,
  _ b: Tensor<T>,
  adjointA: Bool = false,
  adjointB: Bool = false
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("SparseTensorDenseMatMul",
    aIndices,
    aValues,
    aShape,
    b,
    T$dtype: T.tensorFlowDataType,
    Tindices$dtype: Tindices.tensorFlowDataType,
    adjoint_a: adjointA,
    adjoint_b: adjointB)
  return Tensor(handle: ret)
}

/// Converts a sparse representation into a dense tensor.
///
/// Builds an array `dense` with shape `output_shape` such that
///
/// ```
/// # If sparse_indices is scalar
/// dense[i] = (i == sparse_indices ? sparse_values : default_value)
///
/// # If sparse_indices is a vector, then for each i
/// dense[sparse_indices[i]] = sparse_values[i]
///
/// # If sparse_indices is an n by d matrix, then for each i in [0, n)
/// dense[sparse_indices[i][0], ..., sparse_indices[i][d-1]] = sparse_values[i]
/// ```
///
/// All other values in `dense` are set to `default_value`.  If `sparse_values` is a
/// scalar, all sparse indices are set to this single value.
///
/// Indices should be sorted in lexicographic order, and indices must not
/// contain any repeats. If `validate_indices` is true, these properties
/// are checked during execution.
///
/// - Parameters:
///   - sparse_indices: 0-D, 1-D, or 2-D.  `sparse_indices[i]` contains the complete
///     index where `sparse_values[i]` will be placed.
///   - output_shape: 1-D.  Shape of the dense output tensor.
///   - sparse_values: 1-D.  Values corresponding to each row of `sparse_indices`,
///     or a scalar value to be used for all sparse indices.
///   - default_value: Scalar value to set for indices not specified in
///     `sparse_indices`.
///
/// - Attr validate_indices: If true, indices are checked to make sure they are sorted in
///   lexicographic order and that there are no repeats.
///
/// - Output dense: Dense output tensor of shape `output_shape`.
@inlinable @inline(__always)
public static func sparseToDense<T: TensorFlowScalar, Tindices: BinaryInteger & TensorFlowScalar>(
  sparseIndices: Tensor<Tindices>,
  outputShape: Tensor<Tindices>,
  sparseValues: Tensor<T>,
  defaultValue: Tensor<T>,
  validateIndices: Bool = true
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("SparseToDense",
    sparseIndices,
    outputShape,
    sparseValues,
    defaultValue,
    T$dtype: T.tensorFlowDataType,
    Tindices$dtype: Tindices.tensorFlowDataType,
    validate_indices: validateIndices)
  return Tensor(handle: ret)
}

/// Applies set operation along last dimension of 2 `SparseTensor` inputs.
///
/// See SetOperationOp::SetOperationFromContext for values of `set_operation`.
///
/// If `validate_indices` is `True`, `SparseToSparseSetOperation` validates the
/// order and range of `set1` and `set2` indices.
///
/// Input `set1` is a `SparseTensor` represented by `set1_indices`, `set1_values`,
/// and `set1_shape`. For `set1` ranked `n`, 1st `n-1` dimensions must be the same
/// as `set2`. Dimension `n` contains values in a set, duplicates are allowed but
/// ignored.
///
/// Input `set2` is a `SparseTensor` represented by `set2_indices`, `set2_values`,
/// and `set2_shape`. For `set2` ranked `n`, 1st `n-1` dimensions must be the same
/// as `set1`. Dimension `n` contains values in a set, duplicates are allowed but
/// ignored.
///
/// If `validate_indices` is `True`, this op validates the order and range of `set1`
/// and `set2` indices.
///
/// Output `result` is a `SparseTensor` represented by `result_indices`,
/// `result_values`, and `result_shape`. For `set1` and `set2` ranked `n`, this
/// has rank `n` and the same 1st `n-1` dimensions as `set1` and `set2`. The `nth`
/// dimension contains the result of `set_operation` applied to the corresponding
/// `[0...n-1]` dimension of `set`.
///
/// - Parameters:
///   - set1_indices: 2D `Tensor`, indices of a `SparseTensor`. Must be in row-major
///     order.
///   - set1_values: 1D `Tensor`, values of a `SparseTensor`. Must be in row-major
///     order.
///   - set1_shape: 1D `Tensor`, shape of a `SparseTensor`. `set1_shape[0...n-1]` must
///     be the same as `set2_shape[0...n-1]`, `set1_shape[n]` is the
///     max set size across `0...n-1` dimensions.
///   - set2_indices: 2D `Tensor`, indices of a `SparseTensor`. Must be in row-major
///     order.
///   - set2_values: 1D `Tensor`, values of a `SparseTensor`. Must be in row-major
///     order.
///   - set2_shape: 1D `Tensor`, shape of a `SparseTensor`. `set2_shape[0...n-1]` must
///     be the same as `set1_shape[0...n-1]`, `set2_shape[n]` is the
///     max set size across `0...n-1` dimensions.
///
/// - Outputs:
///   - result_indices: 2D indices of a `SparseTensor`.
///   - result_values: 1D values of a `SparseTensor`.
///   - result_shape: 1D `Tensor` shape of a `SparseTensor`. `result_shape[0...n-1]` is
///     the same as the 1st `n-1` dimensions of `set1` and `set2`, `result_shape[n]`
///     is the max result set size across all `0...n-1` dimensions.
@inlinable @inline(__always)
public static func sparseToSparseSetOperation<T: BinaryInteger & TensorFlowScalar>(
  set1Indices: Tensor<Int64>,
  set1Values: Tensor<T>,
  set1Shape: Tensor<Int64>,
  set2Indices: Tensor<Int64>,
  set2Values: Tensor<T>,
  set2Shape: Tensor<Int64>,
  setOperation: String,
  validateIndices: Bool = true
) -> (resultIndices: Tensor<Int64>, resultValues: Tensor<T>, resultShape: Tensor<Int64>) {
  let ret: (TensorHandle<Int64>, TensorHandle<T>, TensorHandle<Int64>) = #tfop("SparseToSparseSetOperation",
    set1Indices,
    set1Values,
    set1Shape,
    set2Indices,
    set2Values,
    set2Shape,
    T$dtype: T.tensorFlowDataType,
    set_operation: setOperation,
    validate_indices: validateIndices)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Computes square root of x element-wise.
///
/// I.e., \\(y = \sqrt{x} = x^{1/2}\\).
@inlinable @inline(__always)
public static func sqrt<T: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Sqrt",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the gradient for the sqrt of `x` wrt its input.
///
/// Specifically, `grad = dy * 0.5 / y`, where `y = sqrt(x)`, and `dy`
/// is the corresponding input gradient.
@inlinable @inline(__always)
public static func sqrtGrad<T: FloatingPoint & TensorFlowScalar>(
  _ y: Tensor<T>,
  dy: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("SqrtGrad",
    y,
    dy,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes square of x element-wise.
///
/// I.e., \\(y = x * x = x^2\\).
@inlinable @inline(__always)
public static func square<T: Numeric & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Square",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Returns (x - y)(x - y) element-wise.
///
/// *NOTE*: `SquaredDifference` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@inlinable @inline(__always)
public static func squaredDifference<T: Numeric & TensorFlowScalar>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("SquaredDifference",
    x,
    y,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Removes dimensions of size 1 from the shape of a tensor.
///
/// Given a tensor `input`, this operation returns a tensor of the same type with
/// all dimensions of size 1 removed. If you don't want to remove all size 1
/// dimensions, you can remove specific size 1 dimensions by specifying
/// `axis`.
///
/// For example:
///
/// ```
/// # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
/// shape(squeeze(t)) ==> [2, 3]
/// ```
///
/// Or, to remove specific size 1 dimensions:
///
/// ```
/// # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
/// shape(squeeze(t, [2, 4])) ==> [1, 2, 3, 1]
/// ```
///
/// - Parameter input: The `input` to squeeze.
///
/// - Attr squeeze_dims: If specified, only squeezes the dimensions listed. The dimension
///   index starts at 0. It is an error to squeeze a dimension that is not 1. Must
///   be in the range `[-rank(input), rank(input))`.
///
/// - Output output: Contains the same data as `input`, but has one or more dimensions of
///   size 1 removed.
@inlinable @inline(__always)
public static func squeeze<T: TensorFlowScalar>(
  _ input: Tensor<T>,
  squeezeDims: [Int32]
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Squeeze",
    input,
    T$dtype: T.tensorFlowDataType,
    squeeze_dims: squeezeDims)
  return Tensor(handle: ret)
}

/// Stage values similar to a lightweight Enqueue.
///
/// The basic functionality of this Op is similar to a queue with many
/// fewer capabilities and options.  This Op is optimized for performance.
///
/// - Parameter values: a list of tensors
///   dtypes A list of data types that inserted values should adhere to.
///
/// - Attrs:
///   - capacity: Maximum number of elements in the Staging Area. If > 0, inserts
///     on the container will block when the capacity is reached.
///   - memory_limit: The maximum number of bytes allowed for Tensors in the Staging Area.
///     If > 0, inserts will block until sufficient space is available.
///   - container: If non-empty, this queue is placed in the given container. Otherwise,
///     a default container is used.
///   - shared_name: It is necessary to match this name to the matching Unstage Op.
@inlinable @inline(__always)
public static func stage<Dtypes: TensorFlowScalar>(
  _ values: [Tensor<Dtypes>],
  capacity: Int64 = 0,
  memoryLimit: Int64 = 0,
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

/// Op removes all elements in the underlying container.
@inlinable @inline(__always)
public static func stageClear<Dtypes: TensorFlowScalar>(
  capacity: Int64 = 0,
  memoryLimit: Int64 = 0,
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

/// Op returns the number of elements in the underlying container.
@inlinable @inline(__always)
public static func stageSize<Dtypes: TensorFlowScalar>(
  capacity: Int64 = 0,
  memoryLimit: Int64 = 0,
  container: String,
  sharedName: String,
  typeDtypes: Dtypes.Type
) -> Tensor<Int32> {
  let ret: TensorHandle<Int32> = #tfop("StageSize",
    capacity: capacity,
    memory_limit: memoryLimit,
    container: container,
    shared_name: sharedName)
  return Tensor(handle: ret)
}

/// Draws samples from a multinomial distribution.
///
/// - Parameters:
///   - logits: 2-D Tensor with shape `[batch_size, num_classes]`.  Each slice `[i, :]`
///     represents the unnormalized log probabilities for all classes.
///   - num_samples: 0-D.  Number of independent samples to draw for each row slice.
///   - seed: 2 seeds (shape [2]).
///
/// - Output output: 2-D Tensor with shape `[batch_size, num_samples]`.  Each slice `[i, :]`
///   contains the drawn class labels with range `[0, num_classes)`.
@inlinable @inline(__always)
public static func statelessMultinomial<T: Numeric & TensorFlowScalar, Tseed: BinaryInteger & TensorFlowScalar, OutputDtype: BinaryInteger & TensorFlowScalar>(
  logits: Tensor<T>,
  numSamples: Tensor<Int32>,
  seed: Tensor<Tseed>
) -> Tensor<OutputDtype> {
  let ret: TensorHandle<OutputDtype> = #tfop("StatelessMultinomial",
    logits,
    numSamples,
    seed,
    T$dtype: T.tensorFlowDataType,
    Tseed$dtype: Tseed.tensorFlowDataType,
    output_dtype$dtype: OutputDtype.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Outputs deterministic pseudorandom values from a normal distribution.
///
/// The generated values will have mean 0 and standard deviation 1.
///
/// The outputs are a deterministic function of `shape` and `seed`.
///
/// - Parameters:
///   - shape: The shape of the output tensor.
///   - seed: 2 seeds (shape [2]).
///
/// - Attr dtype: The type of the output.
///
/// - Output output: Random values with specified shape.
@inlinable @inline(__always)
public static func statelessRandomNormal<Dtype: FloatingPoint & TensorFlowScalar, T: BinaryInteger & TensorFlowScalar, Tseed: BinaryInteger & TensorFlowScalar>(
  shape: Tensor<T>,
  seed: Tensor<Tseed>
) -> Tensor<Dtype> {
  let ret: TensorHandle<Dtype> = #tfop("StatelessRandomNormal",
    shape,
    seed,
    dtype$dtype: Dtype.tensorFlowDataType,
    T$dtype: T.tensorFlowDataType,
    Tseed$dtype: Tseed.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Outputs deterministic pseudorandom random values from a uniform distribution.
///
/// The generated values follow a uniform distribution in the range `[0, 1)`. The
/// lower bound 0 is included in the range, while the upper bound 1 is excluded.
///
/// The outputs are a deterministic function of `shape` and `seed`.
///
/// - Parameters:
///   - shape: The shape of the output tensor.
///   - seed: 2 seeds (shape [2]).
///
/// - Attr dtype: The type of the output.
///
/// - Output output: Random values with specified shape.
@inlinable @inline(__always)
public static func statelessRandomUniform<Dtype: FloatingPoint & TensorFlowScalar, T: BinaryInteger & TensorFlowScalar, Tseed: BinaryInteger & TensorFlowScalar>(
  shape: Tensor<T>,
  seed: Tensor<Tseed>
) -> Tensor<Dtype> {
  let ret: TensorHandle<Dtype> = #tfop("StatelessRandomUniform",
    shape,
    seed,
    dtype$dtype: Dtype.tensorFlowDataType,
    T$dtype: T.tensorFlowDataType,
    Tseed$dtype: Tseed.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Outputs deterministic pseudorandom random integers from a uniform distribution.
///
/// The generated values follow a uniform distribution in the range `[minval, maxval)`.
///
/// The outputs are a deterministic function of `shape`, `seed`, `minval`, and `maxval`.
///
/// - Parameters:
///   - shape: The shape of the output tensor.
///   - seed: 2 seeds (shape [2]).
///   - minval: Minimum value (inclusive, scalar).
///   - maxval: Maximum value (exclusive, scalar).
///
/// - Attr dtype: The type of the output.
///
/// - Output output: Random values with specified shape.
@inlinable @inline(__always)
public static func statelessRandomUniformInt<Dtype: BinaryInteger & TensorFlowScalar, T: BinaryInteger & TensorFlowScalar, Tseed: BinaryInteger & TensorFlowScalar>(
  shape: Tensor<T>,
  seed: Tensor<Tseed>,
  minval: Tensor<Dtype>,
  maxval: Tensor<Dtype>
) -> Tensor<Dtype> {
  let ret: TensorHandle<Dtype> = #tfop("StatelessRandomUniformInt",
    shape,
    seed,
    minval,
    maxval,
    dtype$dtype: Dtype.tensorFlowDataType,
    T$dtype: T.tensorFlowDataType,
    Tseed$dtype: Tseed.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Outputs deterministic pseudorandom values from a truncated normal distribution.
///
/// The generated values follow a normal distribution with mean 0 and standard
/// deviation 1, except that values whose magnitude is more than 2 standard
/// deviations from the mean are dropped and re-picked.
///
/// The outputs are a deterministic function of `shape` and `seed`.
///
/// - Parameters:
///   - shape: The shape of the output tensor.
///   - seed: 2 seeds (shape [2]).
///
/// - Attr dtype: The type of the output.
///
/// - Output output: Random values with specified shape.
@inlinable @inline(__always)
public static func statelessTruncatedNormal<Dtype: FloatingPoint & TensorFlowScalar, T: BinaryInteger & TensorFlowScalar, Tseed: BinaryInteger & TensorFlowScalar>(
  shape: Tensor<T>,
  seed: Tensor<Tseed>
) -> Tensor<Dtype> {
  let ret: TensorHandle<Dtype> = #tfop("StatelessTruncatedNormal",
    shape,
    seed,
    dtype$dtype: Dtype.tensorFlowDataType,
    T$dtype: T.tensorFlowDataType,
    Tseed$dtype: Tseed.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Check if the input matches the regex pattern.
///
/// The input is a string tensor of any shape. The pattern is the
/// regular expression to be matched with every element of the input tensor.
/// The boolean values (True or False) of the output tensor indicate
/// if the input matches the regex pattern provided.
///
/// The pattern follows the re2 syntax (https://github.com/google/re2/wiki/Syntax)
///
/// - Parameter input: A string tensor of the text to be processed.
///
/// - Attr pattern: The regular expression to match the input.
///
/// - Output output: A bool tensor with the same shape as `input`.
@inlinable @inline(__always)
public static func staticRegexFullMatch(
  _ input: StringTensor,
  pattern: String
) -> Tensor<Bool> {
  let ret: TensorHandle<Bool> = #tfop("StaticRegexFullMatch",
    input,
    pattern: pattern)
  return Tensor(handle: ret)
}

/// Replaces the match of pattern in input with rewrite.
///
/// It follows the re2 syntax (https://github.com/google/re2/wiki/Syntax)
///
/// - Parameter input: The text to be processed.
///
/// - Attrs:
///   - pattern: The regular expression to match the input.
///   - rewrite: The rewrite to be applied to the matched expression.
///   - replace_global: If True, the replacement is global, otherwise the replacement
///     is done only on the first match.
///
/// - Output output: The text after applying pattern and rewrite.
@inlinable @inline(__always)
public static func staticRegexReplace(
  _ input: StringTensor,
  pattern: String,
  rewrite: String,
  replaceGlobal: Bool = true
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("StaticRegexReplace",
    input,
    pattern: pattern,
    rewrite: rewrite,
    replace_global: replaceGlobal)
  return StringTensor(handle: ret)
}

/// Stops gradient computation.
///
/// When executed in a graph, this op outputs its input tensor as-is.
///
/// When building ops to compute gradients, this op prevents the contribution of
/// its inputs to be taken into account.  Normally, the gradient generator adds ops
/// to a graph to compute the derivatives of a specified 'loss' by recursively
/// finding out inputs that contributed to its computation.  If you insert this op
/// in the graph it inputs are masked from the gradient generator.  They are not
/// taken into account for computing gradients.
///
/// This is useful any time you want to compute a value with TensorFlow but need
/// to pretend that the value was a constant. Some examples include:
///
/// *  The *EM* algorithm where the *M-step* should not involve backpropagation
///    through the output of the *E-step*.
/// *  Contrastive divergence training of Boltzmann machines where, when
///    differentiating the energy function, the training must not backpropagate
///    through the graph that generated the samples from the model.
/// *  Adversarial training, where no backprop should happen through the adversarial
///    example generation process.
@inlinable @inline(__always)
public static func stopGradient<T: TensorFlowScalar>(
  _ input: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("StopGradient",
    input,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Return a strided slice from `input`.
///
/// Note, most python users will want to use the Python `Tensor.__getitem__`
/// or `Variable.__getitem__` rather than this op directly.
///
/// The goal of this op is to produce a new tensor with a subset of
/// the elements from the `n` dimensional `input` tensor. The subset is chosen using
/// a sequence of `m` sparse range specifications encoded into the arguments
/// of this function. Note, in some cases
/// `m` could be equal to `n`, but this need not be the case. Each
/// range specification entry can be one of the following:
///
/// - An ellipsis (...). Ellipses are used to imply zero or more
///   dimensions of full-dimension selection and are produced using
///   `ellipsis_mask`. For example, `foo[...]` is the identity slice.
///
/// - A new axis. This is used to insert a new shape=1 dimension and is
///   produced using `new_axis_mask`. For example, `foo[:, ...]` where
///   `foo` is shape `(3, 4)` produces a `(1, 3, 4)` tensor.
///
///
/// - A range `begin:end:stride`. This is used to specify how much to choose from
///   a given dimension. `stride` can be any integer but 0.  `begin` is an integer
///   which represents the index of the first value to select while `end` represents
///   the index of the last value to select. The number of values selected in each
///   dimension is `end - begin` if `stride > 0` and `begin - end` if `stride < 0`.
///   `begin` and `end` can be negative where `-1` is the last element, `-2` is
///   the second to last. `begin_mask` controls whether to replace the explicitly
///   given `begin` with an implicit effective value of `0` if `stride > 0` and
///   `-1` if `stride < 0`. `end_mask` is analogous but produces the number
///   required to create the largest open interval. For example, given a shape
///   `(3,)` tensor `foo[:]`, the effective `begin` and `end` are `0` and `3`. Do
///   not assume this is equivalent to `foo[0:-1]` which has an effective `begin`
///   and `end` of `0` and `2`. Another example is `foo[-2::-1]` which reverses the
///   first dimension of a tensor while dropping the last two (in the original
///   order elements). For example `foo = [1,2,3,4]; foo[-2::-1]` is `[4,3]`.
///
/// - A single index. This is used to keep only elements that have a given
///   index. For example (`foo[2, :]` on a shape `(5,6)` tensor produces a
///   shape `(6,)` tensor. This is encoded in `begin` and `end` and
///   `shrink_axis_mask`.
///
/// Each conceptual range specification is encoded in the op's argument. This
/// encoding is best understand by considering a non-trivial example. In
/// particular,
/// `foo[1, 2:4, None, ..., :-3:-1, :]` will be encoded as
///
/// ```
/// begin = [1, 2, x, x, 0, x] # x denotes don't care (usually 0)
/// end = [2, 4, x, x, -3, x]
/// strides = [1, 1, x, x, -1, 1]
/// begin_mask = 1<<4 | 1 << 5 = 48
/// end_mask = 1<<5 = 32
/// ellipsis_mask = 1<<3 = 8
/// new_axis_mask = 1<<2 4
/// shrink_axis_mask = 1<<0
/// ```
///
/// In this case if `foo.shape` is (5, 5, 5, 5, 5, 5) the final shape of
/// the slice becomes (2, 1, 5, 5, 2, 5).
/// Let us walk step by step through each argument specification.
///
/// 1.  The first argument in the example slice is turned into `begin = 1` and
/// `end = begin + 1 = 2`. To disambiguate from the original spec `2:4` we
/// also set the appropriate bit in `shrink_axis_mask`.
///
/// 2. `2:4` is contributes 2, 4, 1 to begin, end, and stride. All masks have
/// zero bits contributed.
///
/// 3. None is a synonym for `tf.newaxis`. This means insert a dimension of size 1
/// dimension in the final shape. Dummy values are contributed to begin,
/// end and stride, while the new_axis_mask bit is set.
///
/// 4. `...` grab the full ranges from as many dimensions as needed to
/// fully specify a slice for every dimension of the input shape.
///
/// 5. `:-3:-1` shows the use of negative indices. A negative index `i` associated
/// with a dimension that has shape `s` is converted to a positive index
/// `s + i`. So `-1` becomes `s-1` (i.e. the last element). This conversion
/// is done internally so begin, end and strides receive x, -3, and -1.
/// The appropriate begin_mask bit is set to indicate the start range is the
/// full range (ignoring the x).
///
/// 6. `:` indicates that the entire contents of the corresponding dimension
/// is selected. This is equivalent to `::` or `0::1`. begin, end, and strides
/// receive 0, 0, and 1, respectively. The appropriate bits in `begin_mask` and
/// `end_mask` are also set.
///
/// *Requirements*:
///   `0 != strides[i] for i in [0, m)`
///   `ellipsis_mask must be a power of two (only one ellipsis)`
///
/// - Parameters:
///   - begin: `begin[k]` specifies the offset into the `k`th range specification.
///     The exact dimension this corresponds to will be determined by context.
///     Out-of-bounds values will be silently clamped. If the `k`th bit of
///     `begin_mask` then `begin[k]` is ignored and the full range of the
///     appropriate dimension is used instead. Negative values causes indexing
///     to start from the highest element e.g. If `foo==[1,2,3]` then `foo[-1]==3`.
///   - end: `end[i]` is like `begin` with the exception that `end_mask` is
///     used to determine full ranges.
///   - strides: `strides[i]` specifies the increment in the `i`th specification
///     after extracting a given element. Negative indices will reverse
///     the original order. Out or range values are
///     clamped to `[0,dim[i]) if slice[i]>0` or `[-1,dim[i]-1] if slice[i] < 0`
///
/// - Attrs:
///   - begin_mask: a bitmask where a bit i being 1 means to ignore the begin
///     value and instead use the largest interval possible. At runtime
///     begin[i] will be replaced with `[0, n-1)` if `stride[i] > 0` or
///     `[-1, n-1]` if `stride[i] < 0`
///   - end_mask: analogous to `begin_mask`
///   - ellipsis_mask: a bitmask where bit `i` being 1 means the `i`th
///     position is actually an ellipsis. One bit at most can be 1.
///     If `ellipsis_mask == 0`, then an implicit ellipsis mask of `1 << (m+1)`
///     is provided. This means that `foo[3:5] == foo[3:5, ...]`. An ellipsis
///     implicitly creates as many range specifications as necessary to fully
///     specify the sliced range for every dimension. For example for a 4-dimensional
///     tensor `foo` the slice `foo[2, ..., 5:8]` implies `foo[2, :, :, 5:8]`.
///   - new_axis_mask: a bitmask where bit `i` being 1 means the `i`th
///     specification creates a new shape 1 dimension. For example
///     `foo[:4, tf.newaxis, :2]` would produce a shape `(4, 1, 2)` tensor.
///   - shrink_axis_mask: a bitmask where bit `i` implies that the `i`th
///     specification should shrink the dimensionality. begin and end
///     must imply a slice of size 1 in the dimension. For example in
///     python one might do `foo[:, 3, :]` which would result in
///     `shrink_axis_mask` being 2.
@inlinable @inline(__always)
public static func stridedSlice<T: TensorFlowScalar, Index: BinaryInteger & TensorFlowScalar>(
  _ input: Tensor<T>,
  begin: Tensor<Index>,
  end: Tensor<Index>,
  strides: Tensor<Index>,
  beginMask: Int64 = 0,
  endMask: Int64 = 0,
  ellipsisMask: Int64 = 0,
  newAxisMask: Int64 = 0,
  shrinkAxisMask: Int64 = 0
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("StridedSlice",
    input,
    begin,
    end,
    strides,
    T$dtype: T.tensorFlowDataType,
    Index$dtype: Index.tensorFlowDataType,
    begin_mask: beginMask,
    end_mask: endMask,
    ellipsis_mask: ellipsisMask,
    new_axis_mask: newAxisMask,
    shrink_axis_mask: shrinkAxisMask)
  return Tensor(handle: ret)
}

/// Returns the gradient of `StridedSlice`.
///
/// Since `StridedSlice` cuts out pieces of its `input` which is size
/// `shape`, its gradient will have the same shape (which is passed here
/// as `shape`). The gradient will be zero in any element that the slice
/// does not select.
///
/// Arguments are the same as StridedSliceGrad with the exception that
/// `dy` is the input gradient to be propagated and `shape` is the
/// shape of `StridedSlice`'s `input`.
@inlinable @inline(__always)
public static func stridedSliceGrad<T: TensorFlowScalar, Index: BinaryInteger & TensorFlowScalar>(
  shape: Tensor<Index>,
  begin: Tensor<Index>,
  end: Tensor<Index>,
  strides: Tensor<Index>,
  dy: Tensor<T>,
  beginMask: Int64 = 0,
  endMask: Int64 = 0,
  ellipsisMask: Int64 = 0,
  newAxisMask: Int64 = 0,
  shrinkAxisMask: Int64 = 0
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("StridedSliceGrad",
    shape,
    begin,
    end,
    strides,
    dy,
    T$dtype: T.tensorFlowDataType,
    Index$dtype: Index.tensorFlowDataType,
    begin_mask: beginMask,
    end_mask: endMask,
    ellipsis_mask: ellipsisMask,
    new_axis_mask: newAxisMask,
    shrink_axis_mask: shrinkAxisMask)
  return Tensor(handle: ret)
}

/// Formats a string template using a list of tensors.
///
/// Formats a string template using a list of tensors, pretty-printing tensor summaries.
///
/// - Parameter inputs: The list of tensors to format into the placeholder string.
///
/// - Attrs:
///   - template: A string, the template to format tensor summaries into.
///   - placeholder: A string, at each placeholder in the template a subsequent tensor summary will be inserted.
///   - summarize: When formatting the tensor summaries print the first and last summarize entries of each tensor dimension.
///
/// - Output output: = The resulting string scalar.
@inlinable @inline(__always)
public static func stringFormat<T: TensorFlowScalar>(
  inputs: [Tensor<T>],
  template: String = "%s",
  placeholder: String = "%s",
  summarize: Int64 = 3
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("StringFormat",
    inputs,
    template: template,
    placeholder: placeholder,
    summarize: summarize)
  return StringTensor(handle: ret)
}

/// Joins the strings in the given list of string tensors into one tensor;
///
/// with the given separator (default is an empty separator).
///
/// - Parameter inputs: A list of string tensors.  The tensors must all have the same shape,
///   or be scalars.  Scalars may be mixed in; these will be broadcast to the shape
///   of non-scalar inputs.
///
/// - Attr separator: string, an optional join separator.
@inlinable @inline(__always)
public static func stringJoin(
  inputs: [StringTensor],
  separator: String
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("StringJoin",
    inputs,
    separator: separator)
  return StringTensor(handle: ret)
}

/// String lengths of `input`.
///
/// Computes the length of each string given in the input tensor.
///
/// - Parameter input: The string for which to compute the length.
///
/// - Attr unit: The unit that is counted to compute string length.  One of: `"BYTE"` (for
///   the number of bytes in each string) or `"UTF8_CHAR"` (for the number of UTF-8
///   encoded Unicode code points in each string).  Results are undefined
///   if `unit=UTF8_CHAR` and the `input` strings do not contain structurally
///   valid UTF-8.
///
/// - Output output: Integer tensor that has the same shape as `input`. The output contains the
///   element-wise string lengths of `input`.
@inlinable @inline(__always)
public static func stringLength(
  _ input: StringTensor,
  unit: Unit = .byte
) -> Tensor<Int32> {
  let ret: TensorHandle<Int32> = #tfop("StringLength",
    input,
    unit: unit.cName)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func stringListAttr(
  _ a: [String],
  _ b: String
) {
  return #tfop("StringListAttr",
    a: a,
    b: b)
}

/// Split elements of `input` based on `delimiter` into a `SparseTensor`.
///
/// Let N be the size of source (typically N will be the batch size). Split each
/// element of `input` based on `delimiter` and return a `SparseTensor`
/// containing the splitted tokens. Empty tokens are ignored.
///
/// `delimiter` can be empty, or a string of split characters. If `delimiter` is an
///  empty string, each element of `input` is split into individual single-byte
///  character strings, including splitting of UTF-8 multibyte sequences. Otherwise
///  every character of `delimiter` is a potential split point.
///
/// For example:
///   N = 2, input[0] is 'hello world' and input[1] is 'a b c', then the output
///   will be
///
///   indices = [0, 0;
///              0, 1;
///              1, 0;
///              1, 1;
///              1, 2]
///   shape = [2, 3]
///   values = ['hello', 'world', 'a', 'b', 'c']
///
/// - Parameters:
///   - input: 1-D. Strings to split.
///   - delimiter: 0-D. Delimiter characters (bytes), or empty string.
///
/// - Attr skip_empty: A `bool`. If `True`, skip the empty strings from the result.
///
/// - Outputs:
///   - indices: A dense matrix of int64 representing the indices of the sparse tensor.
///   - values: A vector of strings corresponding to the splited values.
///   - shape: a length-2 vector of int64 representing the shape of the sparse
///     tensor, where the first value is N and the second value is the maximum number
///     of tokens in a single input entry.
@inlinable @inline(__always)
public static func stringSplit(
  _ input: StringTensor,
  delimiter: StringTensor,
  skipEmpty: Bool = true
) -> (indices: Tensor<Int64>, values: StringTensor, shape: Tensor<Int64>) {
  let ret: (TensorHandle<Int64>, TensorHandle<String>, TensorHandle<Int64>) = #tfop("StringSplit",
    input,
    delimiter,
    skip_empty: skipEmpty)
  return (Tensor(handle: ret.0), StringTensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Split elements of `source` based on `sep` into a `SparseTensor`.
///
/// Let N be the size of source (typically N will be the batch size). Split each
/// element of `source` based on `sep` and return a `SparseTensor`
/// containing the split tokens. Empty tokens are ignored.
///
/// For example, N = 2, source[0] is 'hello world' and source[1] is 'a b c',
/// then the output will be
/// ```
/// st.indices = [0, 0;
///               0, 1;
///               1, 0;
///               1, 1;
///               1, 2]
/// st.shape = [2, 3]
/// st.values = ['hello', 'world', 'a', 'b', 'c']
/// ```
///
/// If `sep` is given, consecutive delimiters are not grouped together and are
/// deemed to delimit empty strings. For example, source of `"1<>2<><>3"` and
/// sep of `"<>"` returns `["1", "2", "", "3"]`. If `sep` is None or an empty
/// string, consecutive whitespace are regarded as a single separator, and the
/// result will contain no empty strings at the startor end if the string has
/// leading or trailing whitespace.
///
/// Note that the above mentioned behavior matches python's str.split.
///
/// - Parameters:
///   - input: `1-D` string `Tensor`, the strings to split.
///   - sep: `0-D` string `Tensor`, the delimiter character.
///
/// - Attr maxsplit: An `int`. If `maxsplit > 0`, limit of the split of the result.
@inlinable @inline(__always)
public static func stringSplitV2(
  _ input: StringTensor,
  sep: StringTensor,
  maxsplit: Int64 = -1
) -> (indices: Tensor<Int64>, values: StringTensor, shape: Tensor<Int64>) {
  let ret: (TensorHandle<Int64>, TensorHandle<String>, TensorHandle<Int64>) = #tfop("StringSplitV2",
    input,
    sep,
    maxsplit: maxsplit)
  return (Tensor(handle: ret.0), StringTensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Strip leading and trailing whitespaces from the Tensor.
///
/// - Parameter input: A string `Tensor` of any shape.
///
/// - Output output: A string `Tensor` of the same shape as the input.
@inlinable @inline(__always)
public static func stringStrip(
  _ input: StringTensor
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("StringStrip",
    input)
  return StringTensor(handle: ret)
}

/// Converts each string in the input Tensor to its hash mod by a number of buckets.
///
/// The hash function is deterministic on the content of the string within the
/// process.
///
/// Note that the hash function may change from time to time.
/// This functionality will be deprecated and it's recommended to use
/// `tf.string_to_hash_bucket_fast()` or `tf.string_to_hash_bucket_strong()`.
///
/// - Attr num_buckets: The number of buckets.
///
/// - Output output: A Tensor of the same shape as the input `string_tensor`.
@inlinable @inline(__always)
public static func stringToHashBucket(
  stringTensor: StringTensor,
  numBuckets: Int64
) -> Tensor<Int64> {
  let ret: TensorHandle<Int64> = #tfop("StringToHashBucket",
    stringTensor,
    num_buckets: numBuckets)
  return Tensor(handle: ret)
}

/// Converts each string in the input Tensor to its hash mod by a number of buckets.
///
/// The hash function is deterministic on the content of the string within the
/// process and will never change. However, it is not suitable for cryptography.
/// This function may be used when CPU time is scarce and inputs are trusted or
/// unimportant. There is a risk of adversaries constructing inputs that all hash
/// to the same bucket. To prevent this problem, use a strong hash function with
/// `tf.string_to_hash_bucket_strong`.
///
/// - Parameter input: The strings to assign a hash bucket.
///
/// - Attr num_buckets: The number of buckets.
///
/// - Output output: A Tensor of the same shape as the input `string_tensor`.
@inlinable @inline(__always)
public static func stringToHashBucketFast(
  _ input: StringTensor,
  numBuckets: Int64
) -> Tensor<Int64> {
  let ret: TensorHandle<Int64> = #tfop("StringToHashBucketFast",
    input,
    num_buckets: numBuckets)
  return Tensor(handle: ret)
}

/// Converts each string in the input Tensor to its hash mod by a number of buckets.
///
/// The hash function is deterministic on the content of the string within the
/// process. The hash function is a keyed hash function, where attribute `key`
/// defines the key of the hash function. `key` is an array of 2 elements.
///
/// A strong hash is important when inputs may be malicious, e.g. URLs with
/// additional components. Adversaries could try to make their inputs hash to the
/// same bucket for a denial-of-service attack or to skew the results. A strong
/// hash prevents this by making it difficult, if not infeasible, to compute inputs
/// that hash to the same bucket. This comes at a cost of roughly 4x higher compute
/// time than `tf.string_to_hash_bucket_fast`.
///
/// - Parameter input: The strings to assign a hash bucket.
///
/// - Attrs:
///   - num_buckets: The number of buckets.
///   - key: The key for the keyed hash function passed as a list of two uint64
///     elements.
///
/// - Output output: A Tensor of the same shape as the input `string_tensor`.
@inlinable @inline(__always)
public static func stringToHashBucketStrong(
  _ input: StringTensor,
  numBuckets: Int64,
  key: [Int32]
) -> Tensor<Int64> {
  let ret: TensorHandle<Int64> = #tfop("StringToHashBucketStrong",
    input,
    num_buckets: numBuckets,
    key: key)
  return Tensor(handle: ret)
}

/// Converts each string in the input Tensor to the specified numeric type.
///
/// (Note that int32 overflow results in an error while float overflow
/// results in a rounded value.)
///
/// - Attr out_type: The numeric type to interpret each string in `string_tensor` as.
///
/// - Output output: A Tensor of the same shape as the input `string_tensor`.
@inlinable @inline(__always)
public static func stringToNumber<OutType: Numeric & TensorFlowScalar>(
  stringTensor: StringTensor
) -> Tensor<OutType> {
  let ret: TensorHandle<OutType> = #tfop("StringToNumber",
    stringTensor,
    out_type$dtype: OutType.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Returns x - y element-wise.
///
/// *NOTE*: `Subtract` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@inlinable @inline(__always)
public static func sub<T: Numeric & TensorFlowScalar>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Sub",
    x,
    y,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Return substrings from `Tensor` of strings.
///
/// For each string in the input `Tensor`, creates a substring starting at index
/// `pos` with a total length of `len`.
///
/// If `len` defines a substring that would extend beyond the length of the input
/// string, then as many characters as possible are used.
///
/// A negative `pos` indicates distance within the string backwards from the end.
///
/// If `pos` specifies an index which is out of range for any of the input strings,
/// then an `InvalidArgumentError` is thrown.
///
/// `pos` and `len` must have the same shape, otherwise a `ValueError` is thrown on
/// Op creation.
///
/// *NOTE*: `Substr` supports broadcasting up to two dimensions. More about
/// broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
///
/// ---
///
/// Examples
///
/// Using scalar `pos` and `len`:
///
/// ```python
/// input = [b'Hello', b'World']
/// position = 1
/// length = 3
///
/// output = [b'ell', b'orl']
/// ```
///
/// Using `pos` and `len` with same shape as `input`:
///
/// ```python
/// input = [[b'ten', b'eleven', b'twelve'],
///          [b'thirteen', b'fourteen', b'fifteen'],
///          [b'sixteen', b'seventeen', b'eighteen']]
/// position = [[1, 2, 3],
///             [1, 2, 3],
///             [1, 2, 3]]
/// length =   [[2, 3, 4],
///             [4, 3, 2],
///             [5, 5, 5]]
///
/// output = [[b'en', b'eve', b'lve'],
///           [b'hirt', b'urt', b'te'],
///           [b'ixtee', b'vente', b'hteen']]
/// ```
///
/// Broadcasting `pos` and `len` onto `input`:
///
/// ```
/// input = [[b'ten', b'eleven', b'twelve'],
///          [b'thirteen', b'fourteen', b'fifteen'],
///          [b'sixteen', b'seventeen', b'eighteen'],
///          [b'nineteen', b'twenty', b'twentyone']]
/// position = [1, 2, 3]
/// length =   [1, 2, 3]
///
/// output = [[b'e', b'ev', b'lve'],
///           [b'h', b'ur', b'tee'],
///           [b'i', b've', b'hte'],
///           [b'i', b'en', b'nty']]
/// ```
///
/// Broadcasting `input` onto `pos` and `len`:
///
/// ```
/// input = b'thirteen'
/// position = [1, 5, 7]
/// length =   [3, 2, 1]
///
/// output = [b'hir', b'ee', b'n']
/// ```
///
/// - Parameters:
///   - input: Tensor of strings
///   - pos: Scalar defining the position of first character in each substring
///   - len: Scalar defining the number of characters to include in each substring
///
/// - Attr unit: The unit that is used to create the substring.  One of: `"BYTE"` (for
///   defining position and length by bytes) or `"UTF8_CHAR"` (for the UTF-8
///   encoded Unicode code points).  The default is `"BYTE"`. Results are undefined if
///   `unit=UTF8_CHAR` and the `input` strings do not contain structurally valid
///   UTF-8.
///
/// - Output output: Tensor of substrings
@inlinable @inline(__always)
public static func substr<T: BinaryInteger & TensorFlowScalar>(
  _ input: StringTensor,
  pos: Tensor<T>,
  len: Tensor<T>,
  unit: Unit = .byte
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("Substr",
    input,
    pos,
    len,
    T$dtype: T.tensorFlowDataType,
    unit: unit.cName)
  return StringTensor(handle: ret)
}

/// Computes the sum of elements across dimensions of a tensor.
///
/// Reduces `input` along the dimensions given in `axis`. Unless
/// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
/// `axis`. If `keep_dims` is true, the reduced dimensions are
/// retained with length 1.
///
/// - Parameters:
///   - input: The tensor to reduce.
///   - reduction_indices: The dimensions to reduce. Must be in the range
///     `[-rank(input), rank(input))`.
///
/// - Attr keep_dims: If true, retain reduced dimensions with length 1.
///
/// - Output output: The reduced tensor.
@inlinable @inline(__always)
public static func sum<T: Numeric & TensorFlowScalar, Tidx: BinaryInteger & TensorFlowScalar>(
  _ input: Tensor<T>,
  reductionIndices: Tensor<Tidx>,
  keepDims: Bool = false
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Sum",
    input,
    reductionIndices,
    T$dtype: T.tensorFlowDataType,
    Tidx$dtype: Tidx.tensorFlowDataType,
    keep_dims: keepDims)
  return Tensor(handle: ret)
}

/// Computes the singular value decompositions of one or more matrices.
///
/// Computes the SVD of each inner matrix in `input` such that
/// `input[..., :, :] = u[..., :, :] * diag(s[..., :, :]) * transpose(v[..., :, :])`
///
/// ```python
/// # a is a tensor containing a batch of matrices.
/// # s is a tensor of singular values for each matrix.
/// # u is the tensor containing of left singular vectors for each matrix.
/// # v is the tensor containing of right singular vectors for each matrix.
/// s, u, v = svd(a)
/// s, _, _ = svd(a, compute_uv=False)
/// ```
///
/// - Parameter input: A tensor of shape `[..., M, N]` whose inner-most 2 dimensions
///   form matrices of size `[M, N]`. Let `P` be the minimum of `M` and `N`.
///
/// - Attrs:
///   - compute_uv: If true, left and right singular vectors will be
///     computed and returned in `u` and `v`, respectively.
///     If false, `u` and `v` are not set and should never referenced.
///   - full_matrices: If true, compute full-sized `u` and `v`. If false
///     (the default), compute only the leading `P` singular vectors.
///     Ignored if `compute_uv` is `False`.
///
/// - Outputs:
///   - s: Singular values. Shape is `[..., P]`.
///   - u: Left singular vectors. If `full_matrices` is `False` then shape is
///     `[..., M, P]`; if `full_matrices` is `True` then shape is
///     `[..., M, M]`. Undefined if `compute_uv` is `False`.
///   - v: Left singular vectors. If `full_matrices` is `False` then shape is
///     `[..., N, P]`. If `full_matrices` is `True` then shape is `[..., N, N]`.
///     Undefined if `compute_uv` is false.
@inlinable @inline(__always)
public static func svd<T: FloatingPoint & TensorFlowScalar>(
  _ input: Tensor<T>,
  computeUv: Bool = true,
  fullMatrices: Bool = false
) -> (s: Tensor<T>, u: Tensor<T>, v: Tensor<T>) {
  let ret: (TensorHandle<T>, TensorHandle<T>, TensorHandle<T>) = #tfop("Svd",
    input,
    T$dtype: T.tensorFlowDataType,
    compute_uv: computeUv,
    full_matrices: fullMatrices)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Forwards `data` to the output port determined by `pred`.
///
/// If `pred` is true, the `data` input is forwarded to `output_true`. Otherwise,
/// the data goes to `output_false`.
///
/// See also `RefSwitch` and `Merge`.
///
/// - Parameters:
///   - data: The tensor to be forwarded to the appropriate output.
///   - pred: A scalar that specifies which output port will receive data.
///
/// - Outputs:
///   - output_false: If `pred` is false, data will be forwarded to this output.
///   - output_true: If `pred` is true, data will be forwarded to this output.
@inlinable @inline(__always)
public static func switch_<T: TensorFlowScalar>(
  data: Tensor<T>,
  pred: Tensor<Bool>
) -> (outputFalse: Tensor<T>, outputTrue: Tensor<T>) {
  let ret: (TensorHandle<T>, TensorHandle<T>) = #tfop("Switch",
    data,
    pred,
    T$dtype: T.tensorFlowDataType)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

/// CompilationResultProto indicating the status of the TPU compilation.
@inlinable @inline(__always)
public static func tPUCompilationResult(
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("TPUCompilationResult")
  return StringTensor(handle: ret)
}

/// An op enabling differentiation of TPU Embeddings.
///
/// This op simply returns its first input, which is assumed to have been sliced
/// from the Tensors returned by TPUEmbeddingDequeueActivations. The presence of
/// this op, and its first argument being a trainable Variable, enables automatic
/// differentiation of graphs containing embeddings via the TPU Embedding Python
/// libraries.
///
/// - Parameters:
///   - embedding_variable: A trainable variable, enabling optimizers to find this op.
///   - sliced_activations: The embedding activations Tensor to return.
///
/// - Attrs:
///   - table_id: The id of the table in the embedding layer configuration from which
///     these activations were computed.
///   - lookup_id: Identifier of the set of embedding indices which produced these
///     activations.
@inlinable @inline(__always)
public static func tPUEmbeddingActivations(
  embeddingVariable: Tensor<Float>,
  slicedActivations: Tensor<Float>,
  tableId: Int64,
  lookupId: Int64
) -> Tensor<Float> {
  let ret: TensorHandle<Float> = #tfop("TPUEmbeddingActivations",
    embeddingVariable,
    slicedActivations,
    table_id: tableId,
    lookup_id: lookupId)
  return Tensor(handle: ret)
}

/// A TPU core selector Op.
///
/// This Op produces a set of TPU cores (for warm-up) or a single TPU core
/// (for regular inference) to execute the TPU program on. The output is
/// consumed by TPUPartitionedCall.
///
/// - Output device_ordinals: A vector 1 or more TPU cores.
@inlinable @inline(__always)
public static func tPUOrdinalSelector(
) -> Tensor<Int32> {
  let ret: TensorHandle<Int32> = #tfop("TPUOrdinalSelector")
  return Tensor(handle: ret)
}

/// Metadata indicaitng how the TPU computation should be replicated.
///
/// - Attrs:
///   - num_replicas: Number of replicas of the computation
///   - num_cores_per_replica: Number of cores per replica. Used for model parallelism.
///   - topology: TopologyProto indicating the topology of the TPU pod slice.
///   - use_tpu: Whether to place the computation on the TPU.
///   - device_assignment: The assignment of devices for the computation.
///   - computation_shape: DEPRECATED. Use num_cores_per_replica instead.
@inlinable @inline(__always)
public static func tPUReplicateMetadata(
  numReplicas: Int64,
  numCoresPerReplica: Int64 = 1,
  topology: String,
  useTpu: Bool = true,
  deviceAssignment: [Int32],
  computationShape: [Int32],
  hostComputeCore: [String],
  paddingMap: [String],
  stepMarkerLocation: String = "STEP_MARK_AT_ENTRY"
) {
  return #tfop("TPUReplicateMetadata",
    num_replicas: numReplicas,
    num_cores_per_replica: numCoresPerReplica,
    topology: topology,
    use_tpu: useTpu,
    device_assignment: deviceAssignment,
    computation_shape: computationShape,
    host_compute_core: hostComputeCore,
    padding_map: paddingMap,
    step_marker_location: stepMarkerLocation)
}

/// Connects N inputs to an N-way replicated TPU computation.
@inlinable @inline(__always)
public static func tPUReplicatedInput<T: TensorFlowScalar>(
  inputs: [Tensor<T>]
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("TPUReplicatedInput",
    inputs,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Read `SparseTensors` from a `SparseTensorsMap` and concatenate them.
///
/// The input `sparse_handles` must be an `int64` matrix of shape `[N, 1]` where
/// `N` is the minibatch size and the rows correspond to the output handles of
/// `AddSparseToTensorsMap` or `AddManySparseToTensorsMap`.  The ranks of the
/// original `SparseTensor` objects that went into the given input ops must all
/// match.  When the final `SparseTensor` is created, it has rank one
/// higher than the ranks of the incoming `SparseTensor` objects
/// (they have been concatenated along a new row dimension on the left).
///
/// The output `SparseTensor` object's shape values for all dimensions but the
/// first are the max across the input `SparseTensor` objects' shape values
/// for the corresponding dimensions.  Its first shape value is `N`, the minibatch
/// size.
///
/// The input `SparseTensor` objects' indices are assumed ordered in
/// standard lexicographic order.  If this is not the case, after this
/// step run `SparseReorder` to restore index ordering.
///
/// For example, if the handles represent an input, which is a `[2, 3]` matrix
/// representing two original `SparseTensor` objects:
///
/// ```
///     index = [ 0]
///             [10]
///             [20]
///     values = [1, 2, 3]
///     shape = [50]
/// ```
///
/// and
///
/// ```
///     index = [ 2]
///             [10]
///     values = [4, 5]
///     shape = [30]
/// ```
///
/// then the final `SparseTensor` will be:
///
/// ```
///     index = [0  0]
///             [0 10]
///             [0 20]
///             [1  2]
///             [1 10]
///     values = [1, 2, 3, 4, 5]
///     shape = [2 50]
/// ```
///
/// - Parameter sparse_handles: 1-D, The `N` serialized `SparseTensor` objects.
///   Shape: `[N]`.
///
/// - Attrs:
///   - dtype: The `dtype` of the `SparseTensor` objects stored in the
///     `SparseTensorsMap`.
///   - container: The container name for the `SparseTensorsMap` read by this op.
///   - shared_name: The shared name for the `SparseTensorsMap` read by this op.
///     It should not be blank; rather the `shared_name` or unique Operation name
///     of the Op that created the original `SparseTensorsMap` should be used.
///
/// - Outputs:
///   - sparse_indices: 2-D.  The `indices` of the minibatch `SparseTensor`.
///   - sparse_values: 1-D.  The `values` of the minibatch `SparseTensor`.
///   - sparse_shape: 1-D.  The `shape` of the minibatch `SparseTensor`.
@inlinable @inline(__always)
public static func takeManySparseFromTensorsMap<Dtype: TensorFlowScalar>(
  sparseHandles: Tensor<Int64>,
  container: String,
  sharedName: String
) -> (sparseIndices: Tensor<Int64>, sparseValues: Tensor<Dtype>, sparseShape: Tensor<Int64>) {
  let ret: (TensorHandle<Int64>, TensorHandle<Dtype>, TensorHandle<Int64>) = #tfop("TakeManySparseFromTensorsMap",
    sparseHandles,
    dtype$dtype: Dtype.tensorFlowDataType,
    container: container,
    shared_name: sharedName)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Computes tan of x element-wise.
@inlinable @inline(__always)
public static func tan<T: Numeric & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Tan",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes hyperbolic tangent of `x` element-wise.
@inlinable @inline(__always)
public static func tanh<T: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Tanh",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the gradient for the tanh of `x` wrt its input.
///
/// Specifically, `grad = dy * (1 - y*y)`, where `y = tanh(x)`, and `dy`
/// is the corresponding input gradient.
@inlinable @inline(__always)
public static func tanhGrad<T: FloatingPoint & TensorFlowScalar>(
  _ y: Tensor<T>,
  dy: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("TanhGrad",
    y,
    dy,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Deprecated. Use TensorArrayCloseV3
@inlinable @inline(__always)
public static func tensorArrayCloseV2(
  handle: StringTensor
) {
  return #tfop("TensorArrayCloseV2",
    handle)
}

/// Deprecated. Use TensorArrayGradV3
@inlinable @inline(__always)
public static func tensorArrayGradV2(
  handle: StringTensor,
  flowIn: Tensor<Float>,
  source: String
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("TensorArrayGradV2",
    handle,
    flowIn,
    source: source)
  return StringTensor(handle: ret)
}

/// Deprecated. Use TensorArrayReadV3
@inlinable @inline(__always)
public static func tensorArrayReadV2<Dtype: TensorFlowScalar>(
  handle: StringTensor,
  index: Tensor<Int32>,
  flowIn: Tensor<Float>
) -> Tensor<Dtype> {
  let ret: TensorHandle<Dtype> = #tfop("TensorArrayReadV2",
    handle,
    index,
    flowIn,
    dtype$dtype: Dtype.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Deprecated. Use TensorArrayScatterV3
@inlinable @inline(__always)
public static func tensorArrayScatterV2<T: TensorFlowScalar>(
  handle: StringTensor,
  indices: Tensor<Int32>,
  value: Tensor<T>,
  flowIn: Tensor<Float>
) -> Tensor<Float> {
  let ret: TensorHandle<Float> = #tfop("TensorArrayScatterV2",
    handle,
    indices,
    value,
    flowIn,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Deprecated. Use TensorArraySizeV3
@inlinable @inline(__always)
public static func tensorArraySizeV2(
  handle: StringTensor,
  flowIn: Tensor<Float>
) -> Tensor<Int32> {
  let ret: TensorHandle<Int32> = #tfop("TensorArraySizeV2",
    handle,
    flowIn)
  return Tensor(handle: ret)
}

/// Deprecated. Use TensorArraySplitV3
@inlinable @inline(__always)
public static func tensorArraySplitV2<T: TensorFlowScalar>(
  handle: StringTensor,
  value: Tensor<T>,
  lengths: Tensor<Int64>,
  flowIn: Tensor<Float>
) -> Tensor<Float> {
  let ret: TensorHandle<Float> = #tfop("TensorArraySplitV2",
    handle,
    value,
    lengths,
    flowIn,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Deprecated. Use TensorArrayGradV3
@inlinable @inline(__always)
public static func tensorArrayWriteV2<T: TensorFlowScalar>(
  handle: StringTensor,
  index: Tensor<Int32>,
  value: Tensor<T>,
  flowIn: Tensor<Float>
) -> Tensor<Float> {
  let ret: TensorHandle<Float> = #tfop("TensorArrayWriteV2",
    handle,
    index,
    value,
    flowIn,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Adds sparse `updates` to an existing tensor according to `indices`.
///
/// This operation creates a new tensor by adding sparse `updates` to the passed
/// in `tensor`.
/// This operation is very similar to `tf.scatter_nd_add`, except that the updates
/// are added onto an existing tensor (as opposed to a variable). If the memory
/// for the existing tensor cannot be re-used, a copy is made and updated.
///
/// `indices` is an integer tensor containing indices into a new tensor of shape
/// `shape`.  The last dimension of `indices` can be at most the rank of `shape`:
///
///     indices.shape[-1] <= shape.rank
///
/// The last dimension of `indices` corresponds to indices into elements
/// (if `indices.shape[-1] = shape.rank`) or slices
/// (if `indices.shape[-1] < shape.rank`) along dimension `indices.shape[-1]` of
/// `shape`.  `updates` is a tensor with shape
///
///     indices.shape[:-1] + shape[indices.shape[-1]:]
///
/// The simplest form of tensor_scatter_add is to add individual elements to a
/// tensor by index. For example, say we want to add 4 elements in a rank-1
/// tensor with 8 elements.
///
/// In Python, this scatter add operation would look like this:
///
/// ```python
///     indices = tf.constant([[4], [3], [1], [7]])
///     updates = tf.constant([9, 10, 11, 12])
///     tensor = tf.ones([8], dtype=tf.int32)
///     updated = tf.tensor_scatter_add(tensor, indices, updates)
///     with tf.Session() as sess:
///       print(sess.run(scatter))
/// ```
///
/// The resulting tensor would look like this:
///
///     [1, 12, 1, 11, 10, 1, 1, 13]
///
/// We can also, insert entire slices of a higher rank tensor all at once. For
/// example, if we wanted to insert two slices in the first dimension of a
/// rank-3 tensor with two matrices of new values.
///
/// In Python, this scatter add operation would look like this:
///
/// ```python
///     indices = tf.constant([[0], [2]])
///     updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
///                             [7, 7, 7, 7], [8, 8, 8, 8]],
///                            [[5, 5, 5, 5], [6, 6, 6, 6],
///                             [7, 7, 7, 7], [8, 8, 8, 8]]])
///     tensor = tf.ones([4, 4, 4])
///     updated = tf.tensor_scatter_add(tensor, indices, updates)
///     with tf.Session() as sess:
///       print(sess.run(scatter))
/// ```
///
/// The resulting tensor would look like this:
///
///     [[[6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9]],
///      [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
///      [[6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9]],
///      [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]]
///
/// Note that on CPU, if an out of bound index is found, an error is returned.
/// On GPU, if an out of bound index is found, the index is ignored.
///
/// - Parameters:
///   - tensor: Tensor to copy/update.
///   - indices: Index tensor.
///   - updates: Updates to scatter into output.
///
/// - Output output: A new tensor copied from tensor and updates added according to the indices.
@inlinable @inline(__always)
public static func tensorScatterAdd<T: TensorFlowScalar, Tindices: BinaryInteger & TensorFlowScalar>(
  _ tensor: Tensor<T>,
  indices: Tensor<Tindices>,
  updates: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("TensorScatterAdd",
    tensor,
    indices,
    updates,
    T$dtype: T.tensorFlowDataType,
    Tindices$dtype: Tindices.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Subtracts sparse `updates` from an existing tensor according to `indices`.
///
/// This operation creates a new tensor by subtracting sparse `updates` from the
/// passed in `tensor`.
/// This operation is very similar to `tf.scatter_nd_sub`, except that the updates
/// are subtracted from an existing tensor (as opposed to a variable). If the memory
/// for the existing tensor cannot be re-used, a copy is made and updated.
///
/// `indices` is an integer tensor containing indices into a new tensor of shape
/// `shape`.  The last dimension of `indices` can be at most the rank of `shape`:
///
///     indices.shape[-1] <= shape.rank
///
/// The last dimension of `indices` corresponds to indices into elements
/// (if `indices.shape[-1] = shape.rank`) or slices
/// (if `indices.shape[-1] < shape.rank`) along dimension `indices.shape[-1]` of
/// `shape`.  `updates` is a tensor with shape
///
///     indices.shape[:-1] + shape[indices.shape[-1]:]
///
/// The simplest form of tensor_scatter_sub is to subtract individual elements
/// from a tensor by index. For example, say we want to insert 4 scattered elements
/// in a rank-1 tensor with 8 elements.
///
/// In Python, this scatter subtract operation would look like this:
///
/// ```python
///     indices = tf.constant([[4], [3], [1], [7]])
///     updates = tf.constant([9, 10, 11, 12])
///     tensor = tf.ones([8], dtype=tf.int32)
///     updated = tf.tensor_scatter_sub(tensor, indices, updates)
///     with tf.Session() as sess:
///       print(sess.run(scatter))
/// ```
///
/// The resulting tensor would look like this:
///
///     [1, -10, 1, -9, -8, 1, 1, -11]
///
/// We can also, insert entire slices of a higher rank tensor all at once. For
/// example, if we wanted to insert two slices in the first dimension of a
/// rank-3 tensor with two matrices of new values.
///
/// In Python, this scatter add operation would look like this:
///
/// ```python
///     indices = tf.constant([[0], [2]])
///     updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
///                             [7, 7, 7, 7], [8, 8, 8, 8]],
///                            [[5, 5, 5, 5], [6, 6, 6, 6],
///                             [7, 7, 7, 7], [8, 8, 8, 8]]])
///     tensor = tf.ones([4, 4, 4])
///     updated = tf.tensor_scatter_sub(tensor, indices, updates)
///     with tf.Session() as sess:
///       print(sess.run(scatter))
/// ```
///
/// The resulting tensor would look like this:
///
///     [[[-4, -4, -4, -4], [-5, -5, -5, -5], [-6, -6, -6, -6], [-7, -7, -7, -7]],
///      [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
///      [[-4, -4, -4, -4], [-5, -5, -5, -5], [-6, -6, -6, -6], [-7, -7, -7, -7]],
///      [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]]
///
/// Note that on CPU, if an out of bound index is found, an error is returned.
/// On GPU, if an out of bound index is found, the index is ignored.
///
/// - Parameters:
///   - tensor: Tensor to copy/update.
///   - indices: Index tensor.
///   - updates: Updates to scatter into output.
///
/// - Output output: A new tensor copied from tensor and updates subtracted according to the indices.
@inlinable @inline(__always)
public static func tensorScatterSub<T: TensorFlowScalar, Tindices: BinaryInteger & TensorFlowScalar>(
  _ tensor: Tensor<T>,
  indices: Tensor<Tindices>,
  updates: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("TensorScatterSub",
    tensor,
    indices,
    updates,
    T$dtype: T.tensorFlowDataType,
    Tindices$dtype: Tindices.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Scatter `updates` into an existing tensor according to `indices`.
///
/// This operation creates a new tensor by applying sparse `updates` to the passed
/// in `tensor`.
/// This operation is very similar to `tf.scatter_nd`, except that the updates are
/// scattered onto an existing tensor (as opposed to a zero-tensor). If the memory
/// for the existing tensor cannot be re-used, a copy is made and updated.
///
/// If `indices` contains duplicates, then their updates are accumulated (summed).
///
/// **WARNING**: The order in which updates are applied is nondeterministic, so the
/// output will be nondeterministic if `indices` contains duplicates -- because
/// of some numerical approximation issues, numbers summed in different order
/// may yield different results.
///
/// `indices` is an integer tensor containing indices into a new tensor of shape
/// `shape`.  The last dimension of `indices` can be at most the rank of `shape`:
///
///     indices.shape[-1] <= shape.rank
///
/// The last dimension of `indices` corresponds to indices into elements
/// (if `indices.shape[-1] = shape.rank`) or slices
/// (if `indices.shape[-1] < shape.rank`) along dimension `indices.shape[-1]` of
/// `shape`.  `updates` is a tensor with shape
///
///     indices.shape[:-1] + shape[indices.shape[-1]:]
///
/// The simplest form of scatter is to insert individual elements in a tensor by
/// index. For example, say we want to insert 4 scattered elements in a rank-1
/// tensor with 8 elements.
///
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/ScatterNd1.png" alt>
/// </div>
///
/// In Python, this scatter operation would look like this:
///
/// ```python
///     indices = tf.constant([[4], [3], [1], [7]])
///     updates = tf.constant([9, 10, 11, 12])
///     tensor = tf.ones([8], dtype=tf.int32)
///     updated = tf.tensor_scatter_update(tensor, indices, updates)
///     with tf.Session() as sess:
///       print(sess.run(scatter))
/// ```
///
/// The resulting tensor would look like this:
///
///     [1, 11, 1, 10, 9, 1, 1, 12]
///
/// We can also, insert entire slices of a higher rank tensor all at once. For
/// example, if we wanted to insert two slices in the first dimension of a
/// rank-3 tensor with two matrices of new values.
///
/// In Python, this scatter operation would look like this:
///
/// ```python
///     indices = tf.constant([[0], [2]])
///     updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
///                             [7, 7, 7, 7], [8, 8, 8, 8]],
///                            [[5, 5, 5, 5], [6, 6, 6, 6],
///                             [7, 7, 7, 7], [8, 8, 8, 8]]])
///     tensor = tf.ones([4, 4, 4])
///     updated = tf.tensor_scatter_update(tensor, indices, updates)
///     with tf.Session() as sess:
///       print(sess.run(scatter))
/// ```
///
/// The resulting tensor would look like this:
///
///     [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
///      [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
///      [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
///      [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]]
///
/// Note that on CPU, if an out of bound index is found, an error is returned.
/// On GPU, if an out of bound index is found, the index is ignored.
///
/// - Parameters:
///   - tensor: Tensor to copy/update.
///   - indices: Index tensor.
///   - updates: Updates to scatter into output.
///
/// - Output output: A new tensor with the given shape and updates applied according
///   to the indices.
@inlinable @inline(__always)
public static func tensorScatterUpdate<T: TensorFlowScalar, Tindices: BinaryInteger & TensorFlowScalar>(
  _ tensor: Tensor<T>,
  indices: Tensor<Tindices>,
  updates: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("TensorScatterUpdate",
    tensor,
    indices,
    updates,
    T$dtype: T.tensorFlowDataType,
    Tindices$dtype: Tindices.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Outputs a `Summary` protocol buffer with a tensor.
///
/// This op is being phased out in favor of TensorSummaryV2, which lets callers pass
/// a tag as well as a serialized SummaryMetadata proto string that contains
/// plugin-specific data. We will keep this op to maintain backwards compatibility.
///
/// - Parameter tensor: A tensor to serialize.
///
/// - Attrs:
///   - description: A json-encoded SummaryDescription proto.
///   - labels: An unused list of strings.
///   - display_name: An unused string.
@inlinable @inline(__always)
public static func tensorSummary<T: TensorFlowScalar>(
  _ tensor: Tensor<T>,
  description: String,
  labels: [String],
  displayName: String
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("TensorSummary",
    tensor,
    T$dtype: T.tensorFlowDataType,
    description: description,
    labels: labels,
    display_name: displayName)
  return StringTensor(handle: ret)
}

/// Outputs a `Summary` protocol buffer with a tensor and per-plugin data.
///
/// - Parameters:
///   - tag: A string attached to this summary. Used for organization in TensorBoard.
///   - tensor: A tensor to serialize.
///   - serialized_summary_metadata: A serialized SummaryMetadata proto. Contains plugin
///     data.
@inlinable @inline(__always)
public static func tensorSummaryV2<T: TensorFlowScalar>(
  tag: StringTensor,
  _ tensor: Tensor<T>,
  serializedSummaryMetadata: StringTensor
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("TensorSummaryV2",
    tag,
    tensor,
    serializedSummaryMetadata,
    T$dtype: T.tensorFlowDataType)
  return StringTensor(handle: ret)
}

@inlinable @inline(__always)
public static func testAttr<T: FloatingPoint & TensorFlowScalar>(
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("TestAttr",
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func testStringOutput(
  _ input: Tensor<Float>
) -> (output1: Tensor<Float>, output2: StringTensor) {
  let ret: (TensorHandle<Float>, TensorHandle<String>) = #tfop("TestStringOutput",
    input)
  return (Tensor(handle: ret.0), StringTensor(handle: ret.1))
}

/// Generates labels for candidate sampling with a learned unigram distribution.
///
/// See explanations of candidate sampling and the data formats at
/// go/candidate-sampling.
///
/// For each batch, this op picks a single set of sampled candidate labels.
///
/// The advantages of sampling candidates per-batch are simplicity and the
/// possibility of efficient dense matrix multiplication. The disadvantage is that
/// the sampled candidates must be chosen independently of the context and of the
/// true labels.
///
/// - Parameter true_classes: A batch_size * num_true matrix, in which each row contains the
///   IDs of the num_true target_classes in the corresponding original label.
///
/// - Attrs:
///   - num_true: Number of true labels per context.
///   - num_sampled: Number of candidates to randomly sample.
///   - unique: If unique is true, we sample with rejection, so that all sampled
///     candidates in a batch are unique. This requires some approximation to
///     estimate the post-rejection sampling probabilities.
///   - range_max: The sampler will sample integers from the interval [0, range_max).
///   - seed: If either seed or seed2 are set to be non-zero, the random number
///     generator is seeded by the given seed.  Otherwise, it is seeded by a
///     random seed.
///   - seed2: An second seed to avoid seed collision.
///
/// - Outputs:
///   - sampled_candidates: A vector of length num_sampled, in which each element is
///     the ID of a sampled candidate.
///   - true_expected_count: A batch_size * num_true matrix, representing
///     the number of times each candidate is expected to occur in a batch
///     of sampled candidates. If unique=true, then this is a probability.
///   - sampled_expected_count: A vector of length num_sampled, for each sampled
///     candidate representing the number of times the candidate is expected
///     to occur in a batch of sampled candidates.  If unique=true, then this is a
///     probability.
@inlinable @inline(__always)
public static func threadUnsafeUnigramCandidateSampler(
  trueClasses: Tensor<Int64>,
  numTrue: Int64,
  numSampled: Int64,
  unique: Bool,
  rangeMax: Int64,
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> (sampledCandidates: Tensor<Int64>, trueExpectedCount: Tensor<Float>, sampledExpectedCount: Tensor<Float>) {
  let ret: (TensorHandle<Int64>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("ThreadUnsafeUnigramCandidateSampler",
    trueClasses,
    num_true: numTrue,
    num_sampled: numSampled,
    unique: unique,
    range_max: rangeMax,
    seed: seed,
    seed2: seed2)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Constructs a tensor by tiling a given tensor.
///
/// This operation creates a new tensor by replicating `input` `multiples` times.
/// The output tensor's i'th dimension has `input.dims(i) * multiples[i]` elements,
/// and the values of `input` are replicated `multiples[i]` times along the 'i'th
/// dimension. For example, tiling `[a b c d]` by `[2]` produces
/// `[a b c d a b c d]`.
///
/// - Parameters:
///   - input: 1-D or higher.
///   - multiples: 1-D. Length must be the same as the number of dimensions in `input`
@inlinable @inline(__always)
public static func tile<T: TensorFlowScalar, Tmultiples: BinaryInteger & TensorFlowScalar>(
  _ input: Tensor<T>,
  multiples: Tensor<Tmultiples>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Tile",
    input,
    multiples,
    T$dtype: T.tensorFlowDataType,
    Tmultiples$dtype: Tmultiples.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Returns the gradient of `Tile`.
///
/// Since `Tile` takes an input and repeats the input `multiples` times
/// along each dimension, `TileGrad` takes in `multiples` and aggregates
/// each repeated tile of `input` into `output`.
@inlinable @inline(__always)
public static func tileGrad<T: TensorFlowScalar>(
  _ input: Tensor<T>,
  multiples: Tensor<Int32>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("TileGrad",
    input,
    multiples,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Provides the time since epoch in seconds.
///
/// Returns the timestamp as a `float64` for seconds since the Unix epoch.
///
/// Note: the timestamp is computed when the op is executed, not when it is added
/// to the graph.
@inlinable @inline(__always)
public static func timestamp(
) -> Tensor<Double> {
  let ret: TensorHandle<Double> = #tfop("Timestamp")
  return Tensor(handle: ret)
}

/// Finds values and indices of the `k` largest elements for the last dimension.
///
/// If the input is a vector (rank-1), finds the `k` largest entries in the vector
/// and outputs their values and indices as vectors.  Thus `values[j]` is the
/// `j`-th largest entry in `input`, and its index is `indices[j]`.
///
/// For matrices (resp. higher rank input), computes the top `k` entries in each
/// row (resp. vector along the last dimension).  Thus,
///
///     values.shape = indices.shape = input.shape[:-1] + [k]
///
/// If two elements are equal, the lower-index element appears first.
///
/// If `k` varies dynamically, use `TopKV2` below.
///
/// - Parameter input: 1-D or higher with last dimension at least `k`.
///
/// - Attrs:
///   - k: Number of top elements to look for along the last dimension (along each
///     row for matrices).
///   - sorted: If true the resulting `k` elements will be sorted by the values in
///     descending order.
///
/// - Outputs:
///   - values: The `k` largest elements along each last dimensional slice.
///   - indices: The indices of `values` within the last dimension of `input`.
@inlinable @inline(__always)
public static func topK<T: Numeric & TensorFlowScalar>(
  _ input: Tensor<T>,
  k: Int64,
  sorted: Bool = true
) -> (values: Tensor<T>, indices: Tensor<Int32>) {
  let ret: (TensorHandle<T>, TensorHandle<Int32>) = #tfop("TopK",
    input,
    T$dtype: T.tensorFlowDataType,
    k: k,
    sorted: sorted)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

/// Finds values and indices of the `k` largest elements for the last dimension.
///
/// If the input is a vector (rank-1), finds the `k` largest entries in the vector
/// and outputs their values and indices as vectors.  Thus `values[j]` is the
/// `j`-th largest entry in `input`, and its index is `indices[j]`.
///
/// For matrices (resp. higher rank input), computes the top `k` entries in each
/// row (resp. vector along the last dimension).  Thus,
///
///     values.shape = indices.shape = input.shape[:-1] + [k]
///
/// If two elements are equal, the lower-index element appears first.
///
/// - Parameters:
///   - input: 1-D or higher with last dimension at least `k`.
///   - k: 0-D.  Number of top elements to look for along the last dimension (along each
///     row for matrices).
///
/// - Attr sorted: If true the resulting `k` elements will be sorted by the values in
///   descending order.
///
/// - Outputs:
///   - values: The `k` largest elements along each last dimensional slice.
///   - indices: The indices of `values` within the last dimension of `input`.
@inlinable @inline(__always)
public static func topKV2<T: Numeric & TensorFlowScalar>(
  _ input: Tensor<T>,
  k: Tensor<Int32>,
  sorted: Bool = true
) -> (values: Tensor<T>, indices: Tensor<Int32>) {
  let ret: (TensorHandle<T>, TensorHandle<Int32>) = #tfop("TopKV2",
    input,
    k,
    T$dtype: T.tensorFlowDataType,
    sorted: sorted)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

/// Shuffle dimensions of x according to a permutation.
///
/// The output `y` has the same rank as `x`. The shapes of `x` and `y` satisfy:
///   `y.shape[i] == x.shape[perm[i]] for i in [0, 1, ..., rank(x) - 1]`
@inlinable @inline(__always)
public static func transpose<T: TensorFlowScalar, Tperm: BinaryInteger & TensorFlowScalar>(
  _ x: Tensor<T>,
  perm: Tensor<Tperm>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Transpose",
    x,
    perm,
    T$dtype: T.tensorFlowDataType,
    Tperm$dtype: Tperm.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Solves tridiagonal systems of equations.
///
/// `diagonals` is a tensor of shape `[..., 3, M]` whose inner-most 2 dimensions
/// represent matrices with three rows being the superdiagonal, diagonals, and
/// subdiagonals, in order. The last element of the superdiagonal and the first
/// element of the subdiagonal is ignored.
/// `rhs` is a tensor of shape `[..., M, K]`, representing K right-hand sides per
/// each left-hand side.
/// The output is a tensor of shape `[..., M, K]` containing the solutions.
///
/// - Parameters:
///   - diagonals: Shape is `[..., 3, M]`.
///   - rhs: Shape is `[..., M, K]`.
///
/// - Output output: Shape is `[..., M, K]`.
@inlinable @inline(__always)
public static func tridiagonalSolve<T: FloatingPoint & TensorFlowScalar>(
  diagonals: Tensor<T>,
  rhs: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("TridiagonalSolve",
    diagonals,
    rhs,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Returns x / y element-wise for integer types.
///
/// Truncation designates that negative numbers will round fractional quantities
/// toward zero. I.e. -7 / 5 = -1. This matches C semantics but it is different
/// than Python semantics. See `FloorDiv` for a division function that matches
/// Python Semantics.
///
/// *NOTE*: `TruncateDiv` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@inlinable @inline(__always)
public static func truncateDiv<T: Numeric & TensorFlowScalar>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("TruncateDiv",
    x,
    y,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Returns element-wise remainder of division. This emulates C semantics in that
///
/// the result here is consistent with a truncating divide. E.g. `truncate(x / y) *
/// y + truncate_mod(x, y) = x`.
///
/// *NOTE*: `TruncateMod` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
@inlinable @inline(__always)
public static func truncateMod<T: Numeric & TensorFlowScalar>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("TruncateMod",
    x,
    y,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Outputs random values from a truncated normal distribution.
///
/// The generated values follow a normal distribution with mean 0 and standard
/// deviation 1, except that values whose magnitude is more than 2 standard
/// deviations from the mean are dropped and re-picked.
///
/// - Parameter shape: The shape of the output tensor.
///
/// - Attrs:
///   - seed: If either `seed` or `seed2` are set to be non-zero, the random number
///     generator is seeded by the given seed.  Otherwise, it is seeded by a
///     random seed.
///   - seed2: A second seed to avoid seed collision.
///   - dtype: The type of the output.
///
/// - Output output: A tensor of the specified shape filled with random truncated normal
///   values.
@inlinable @inline(__always)
public static func truncatedNormal<Dtype: FloatingPoint & TensorFlowScalar, T: BinaryInteger & TensorFlowScalar>(
  shape: Tensor<T>,
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> Tensor<Dtype> {
  let ret: TensorHandle<Dtype> = #tfop("TruncatedNormal",
    shape,
    dtype$dtype: Dtype.tensorFlowDataType,
    T$dtype: T.tensorFlowDataType,
    seed: seed,
    seed2: seed2)
  return Tensor(handle: ret)
}

/// Perform batches of RPC requests.
///
/// This op asynchronously performs either a single RPC request, or a batch
/// of requests.  RPC requests are defined by three main parameters:
///
///   - `address` (the host+port or BNS address of the request)
///   - `method` (the method name for the request)
///   - `request` (the serialized proto string, or vector of strings,
///      of the RPC request argument).
///
/// For example, if you have an RPC service running on port localhost:2345,
/// and its interface is configured with the following proto declaration:
///
/// ```
/// service MyService {
///   rpc MyMethod(MyRequestProto) returns (MyResponseProto) {
///   }
/// };
/// ```
///
/// then call this op with arguments:
///
/// ```
/// address = "localhost:2345"
/// method = "MyService/MyMethod"
/// ```
///
/// The `request` tensor is a string tensor representing serialized `MyRequestProto`
/// strings; and the output string tensor `response` will have the same shape
/// and contain (upon successful completion) corresponding serialized
/// `MyResponseProto` strings.
///
/// For example, to send a single, empty, `MyRequestProto`, call
/// this op with `request = ""`.  To send 5 **parallel** empty requests,
/// call this op with `request = ["", "", "", "", ""]`.
///
/// More generally, one can create a batch of `MyRequestProto` serialized protos
/// from regular batched tensors using the `encode_proto` op, and convert
/// the response `MyResponseProto` serialized protos to batched tensors
/// using the `decode_proto` op.
///
/// **NOTE** Working with serialized proto strings is faster than instantiating
/// actual proto objects in memory, so no performance degradation is expected
/// compared to writing custom kernels for this workflow.
///
/// Unlike the standard `Rpc` op, if the connection fails or the remote worker
/// returns an error status, this op does **not** reraise the exception.
/// Instead, the `status_code` and `status_message` entry for the corresponding RPC
/// call is set with the error returned from the RPC call.  The `response` tensor
/// will contain valid response values for those minibatch entries whose RPCs did
/// not fail; the rest of the entries will have empty strings.
///
/// - Parameters:
///   - address: `0-D` or `1-D`.  The address (i.e. host_name:port) of the RPC server.
///     If this tensor has more than 1 element, then multiple parallel rpc requests
///     are sent.  This argument broadcasts with `method` and `request`.
///   - method: `0-D` or `1-D`.  The method address on the RPC server.
///     If this tensor has more than 1 element, then multiple parallel rpc requests
///     are sent.  This argument broadcasts with `address` and `request`.
///   - request: `0-D` or `1-D`.  Serialized proto strings: the rpc request argument.
///     If this tensor has more than 1 element, then multiple parallel rpc requests
///     are sent.  This argument broadcasts with `address` and `method`.
///
/// - Attrs:
///   - protocol: RPC protocol to use.  Empty string means use the default protocol.
///     Options include 'grpc'.
///   - fail_fast: `boolean`. If `true` (default), then failures to connect
///     (i.e., the server does not immediately respond) cause an RPC failure.
///   - timeout_in_ms: `int`. If `0` (default), then the kernel will run the RPC
///     request and only time out if the RPC deadline passes or the session times out.
///     If this value is greater than `0`, then the op will raise an exception if
///     the RPC takes longer than `timeout_in_ms`.
///
/// - Outputs:
///   - response: Same shape as `request`. Serialized proto strings: the rpc responses.
///   - status_code: Same shape as `request`.  Values correspond to tensorflow Status enum codes.
///   - status_message: Same shape as `request`.  Values correspond to Status messages
///     returned from the RPC calls.
@inlinable @inline(__always)
public static func tryRpc(
  address: StringTensor,
  method: StringTensor,
  request: StringTensor,
  protocol_: String,
  failFast: Bool = true,
  timeoutInMs: Int64 = 0
) -> (response: StringTensor, statusCode: Tensor<Int32>, statusMessage: StringTensor) {
  let ret: (TensorHandle<String>, TensorHandle<Int32>, TensorHandle<String>) = #tfop("TryRpc",
    address,
    method,
    request,
    protocol: protocol_,
    fail_fast: failFast,
    timeout_in_ms: timeoutInMs)
  return (StringTensor(handle: ret.0), Tensor(handle: ret.1), StringTensor(handle: ret.2))
}

@inlinable @inline(__always)
public static func twoFloatInputs(
  _ a: Tensor<Float>,
  _ b: Tensor<Float>
) {
  return #tfop("TwoFloatInputs",
    a,
    b)
}

@inlinable @inline(__always)
public static func twoFloatInputsFloatOutput(
  _ a: Tensor<Float>,
  _ b: Tensor<Float>
) -> Tensor<Float> {
  let ret: TensorHandle<Float> = #tfop("TwoFloatInputsFloatOutput",
    a,
    b)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func twoFloatInputsIntOutput(
  _ a: Tensor<Float>,
  _ b: Tensor<Float>
) -> Tensor<Int32> {
  let ret: TensorHandle<Int32> = #tfop("TwoFloatInputsIntOutput",
    a,
    b)
  return Tensor(handle: ret)
}

@inlinable @inline(__always)
public static func twoFloatOutputs(
) -> (a: Tensor<Float>, b: Tensor<Float>) {
  let ret: (TensorHandle<Float>, TensorHandle<Float>) = #tfop("TwoFloatOutputs")
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

@inlinable @inline(__always)
public static func twoIntInputs(
  _ a: Tensor<Int32>,
  _ b: Tensor<Int32>
) {
  return #tfop("TwoIntInputs",
    a,
    b)
}

@inlinable @inline(__always)
public static func twoIntOutputs(
) -> (a: Tensor<Int32>, b: Tensor<Int32>) {
  let ret: (TensorHandle<Int32>, TensorHandle<Int32>) = #tfop("TwoIntOutputs")
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

@inlinable @inline(__always)
public static func typeList<T: TensorFlowScalar>(
  _ a: [Tensor<T>]
) {
  return #tfop("TypeList",
    a)
}

@inlinable @inline(__always)
public static func typeListRestrict<T: TensorFlowScalar>(
  _ a: [Tensor<T>]
) {
  return #tfop("TypeListRestrict",
    a)
}

@inlinable @inline(__always)
public static func typeListTwice<T: TensorFlowScalar>(
  _ a: [Tensor<T>],
  _ b: [Tensor<T>]
) {
  return #tfop("TypeListTwice",
    a,
    b)
}

@inlinable @inline(__always)
public static func unary<T: TensorFlowScalar>(
  _ a: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Unary",
    a,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Reverses the operation of Batch for a single output Tensor.
///
/// An instance of Unbatch either receives an empty batched_tensor, in which case it
/// asynchronously waits until the values become available from a concurrently
/// running instance of Unbatch with the same container and shared_name, or receives
/// a non-empty batched_tensor in which case it finalizes all other concurrently
/// running instances and outputs its own element from the batch.
///
/// batched_tensor: The possibly transformed output of Batch. The size of the first
///  dimension should remain unchanged by the transformations for the operation to
///  work.
/// batch_index: The matching batch_index obtained from Batch.
/// id: The id scalar emitted by Batch.
/// unbatched_tensor: The Tensor corresponding to this execution.
/// timeout_micros: Maximum amount of time (in microseconds) to wait to receive the
///  batched input tensor associated with a given invocation of the op.
/// container: Container to control resource sharing.
/// shared_name: Instances of Unbatch with the same container and shared_name are
///  assumed to possibly belong to the same batch. If left empty, the op name will
///  be used as the shared name.
@inlinable @inline(__always)
public static func unbatch<T: TensorFlowScalar>(
  batchedTensor: Tensor<T>,
  batchIndex: Tensor<Int64>,
  id: Tensor<Int64>,
  timeoutMicros: Int64,
  container: String,
  sharedName: String
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Unbatch",
    batchedTensor,
    batchIndex,
    id,
    T$dtype: T.tensorFlowDataType,
    timeout_micros: timeoutMicros,
    container: container,
    shared_name: sharedName)
  return Tensor(handle: ret)
}

/// Gradient of Unbatch.
///
/// Acts like Batch but using the given batch_index index of batching things as they
/// become available. This ensures that the gradients are propagated back in the
/// same session which did the forward pass.
///
/// original_input: The input to the Unbatch operation this is the gradient of.
/// batch_index: The batch_index given to the Unbatch operation this is the gradient
/// of.
/// grad: The downstream gradient.
/// id: The id scalar emitted by Batch.
/// batched_grad: The return value, either an empty tensor or the batched gradient.
/// container: Container to control resource sharing.
/// shared_name: Instances of UnbatchGrad with the same container and shared_name
///  are assumed to possibly belong to the same batch. If left empty, the op name
///  will be used as the shared name.
@inlinable @inline(__always)
public static func unbatchGrad<T: TensorFlowScalar>(
  originalInput: Tensor<T>,
  batchIndex: Tensor<Int64>,
  grad: Tensor<T>,
  id: Tensor<Int64>,
  container: String,
  sharedName: String
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("UnbatchGrad",
    originalInput,
    batchIndex,
    grad,
    id,
    T$dtype: T.tensorFlowDataType,
    container: container,
    shared_name: sharedName)
  return Tensor(handle: ret)
}

/// Decodes each string in `input` into a sequence of Unicode code points.
///
/// The character codepoints for all strings are returned using a single vector
/// `char_values`, with strings expanded to characters in row-major order.
///
/// The `row_splits` tensor indicates where the codepoints for
/// each input string begin and end within the `char_values` tensor.
/// In particular, the values for the `i`th
/// string (in row-major order) are stored in the slice
/// `[row_splits[i]:row_splits[i+1]]`. Thus:
///
/// * `char_values[row_splits[i]+j]` is the Unicode codepoint for the `j`th
///   character in the `i`th string (in row-major order).
/// * `row_splits[i+1] - row_splits[i]` is the number of characters in the `i`th
///   string (in row-major order).
///
/// - Parameter input: The text to be decoded. Can have any shape. Note that the output is flattened
///   to a vector of char values.
///
/// - Attrs:
///   - input_encoding: Text encoding of the input strings. This is any of the encodings supported
///     by ICU ucnv algorithmic converters. Examples: `"UTF-16", "US ASCII", "UTF-8"`.
///   - errors: Error handling policy when there is invalid formatting found in the input.
///     The value of 'strict' will cause the operation to produce a InvalidArgument
///     error on any invalid input formatting. A value of 'replace' (the default) will
///     cause the operation to replace any invalid formatting in the input with the
///     `replacement_char` codepoint. A value of 'ignore' will cause the operation to
///     skip any invalid formatting in the input and produce no corresponding output
///     character.
///   - replacement_char: The replacement character codepoint to be used in place of any invalid
///     formatting in the input when `errors='replace'`. Any valid unicode codepoint may
///     be used. The default value is the default unicode replacement character is
///     0xFFFD or U+65533.)
///   - replace_control_characters: Whether to replace the C0 control characters (00-1F) with the
///     `replacement_char`. Default is false.
///
/// - Outputs:
///   - row_splits: A 1D int32 tensor containing the row splits.
///   - char_values: A 1D int32 Tensor containing the decoded codepoints.
@inlinable @inline(__always)
public static func unicodeDecode(
  _ input: StringTensor,
  inputEncoding: String,
  errors: Errors = .replace,
  replacementChar: Int64 = 65533,
  replaceControlCharacters: Bool = false
) -> (rowSplits: Tensor<Int64>, charValues: Tensor<Int32>) {
  let ret: (TensorHandle<Int64>, TensorHandle<Int32>) = #tfop("UnicodeDecode",
    input,
    input_encoding: inputEncoding,
    errors: errors.cName,
    replacement_char: replacementChar,
    replace_control_characters: replaceControlCharacters)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

/// Decodes each string in `input` into a sequence of Unicode code points.
///
/// The character codepoints for all strings are returned using a single vector
/// `char_values`, with strings expanded to characters in row-major order.
/// Similarly, the character start byte offsets are returned using a single vector
/// `char_to_byte_starts`, with strings expanded in row-major order.
///
/// The `row_splits` tensor indicates where the codepoints and start offsets for
/// each input string begin and end within the `char_values` and
/// `char_to_byte_starts` tensors.  In particular, the values for the `i`th
/// string (in row-major order) are stored in the slice
/// `[row_splits[i]:row_splits[i+1]]`. Thus:
///
/// * `char_values[row_splits[i]+j]` is the Unicode codepoint for the `j`th
///   character in the `i`th string (in row-major order).
/// * `char_to_bytes_starts[row_splits[i]+j]` is the start byte offset for the `j`th
///   character in the `i`th string (in row-major order).
/// * `row_splits[i+1] - row_splits[i]` is the number of characters in the `i`th
///   string (in row-major order).
///
/// - Parameter input: The text to be decoded. Can have any shape. Note that the output is flattened
///   to a vector of char values.
///
/// - Attrs:
///   - input_encoding: Text encoding of the input strings. This is any of the encodings supported
///     by ICU ucnv algorithmic converters. Examples: `"UTF-16", "US ASCII", "UTF-8"`.
///   - errors: Error handling policy when there is invalid formatting found in the input.
///     The value of 'strict' will cause the operation to produce a InvalidArgument
///     error on any invalid input formatting. A value of 'replace' (the default) will
///     cause the operation to replace any invalid formatting in the input with the
///     `replacement_char` codepoint. A value of 'ignore' will cause the operation to
///     skip any invalid formatting in the input and produce no corresponding output
///     character.
///   - replacement_char: The replacement character codepoint to be used in place of any invalid
///     formatting in the input when `errors='replace'`. Any valid unicode codepoint may
///     be used. The default value is the default unicode replacement character is
///     0xFFFD or U+65533.)
///   - replace_control_characters: Whether to replace the C0 control characters (00-1F) with the
///     `replacement_char`. Default is false.
///
/// - Outputs:
///   - row_splits: A 1D int32 tensor containing the row splits.
///   - char_values: A 1D int32 Tensor containing the decoded codepoints.
///   - char_to_byte_starts: A 1D int32 Tensor containing the byte index in the input string where each
///     character in `char_values` starts.
@inlinable @inline(__always)
public static func unicodeDecodeWithOffsets(
  _ input: StringTensor,
  inputEncoding: String,
  errors: Errors = .replace,
  replacementChar: Int64 = 65533,
  replaceControlCharacters: Bool = false
) -> (rowSplits: Tensor<Int64>, charValues: Tensor<Int32>, charToByteStarts: Tensor<Int64>) {
  let ret: (TensorHandle<Int64>, TensorHandle<Int32>, TensorHandle<Int64>) = #tfop("UnicodeDecodeWithOffsets",
    input,
    input_encoding: inputEncoding,
    errors: errors.cName,
    replacement_char: replacementChar,
    replace_control_characters: replaceControlCharacters)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Encode a tensor of ints into unicode strings.
///
/// Returns a vector of strings, where `output[i]` is constructed by encoding the
/// Unicode codepoints in `input_values[input_splits[i]:input_splits[i+1]]`
/// using `output_encoding`.
///
/// ---
///
/// Example:
///
/// ```
/// input_values = [72, 101, 108, 108, 111, 87, 111, 114, 108, 100]
/// input_splits = [0, 5, 10]
/// output_encoding = 'UTF-8'
///
/// output = ['Hello', 'World']
/// ```
///
/// - Parameters:
///   - input_values: A 1D tensor containing the unicode codepoints that should be encoded.
///   - input_splits: A 1D tensor specifying how the unicode codepoints should be split into strings.
///     In particular, `output[i]` is constructed by encoding the codepoints in the
///     slice `input_values[input_splits[i]:input_splits[i+1]]`.
///
/// - Attrs:
///   - errors: Error handling policy when there is invalid formatting found in the input.
///     The value of 'strict' will cause the operation to produce a InvalidArgument
///     error on any invalid input formatting. A value of 'replace' (the default) will
///     cause the operation to replace any invalid formatting in the input with the
///     `replacement_char` codepoint. A value of 'ignore' will cause the operation to
///     skip any invalid formatting in the input and produce no corresponding output
///     character.
///   - output_encoding: Unicode encoding of the output strings. Valid encodings are: `"UTF-8",
///     "UTF-16-BE", and "UTF-32-BE"`.
///   - replacement_char: The replacement character codepoint to be used in place of any invalid
///     formatting in the input when `errors='replace'`. Any valid unicode codepoint may
///     be used. The default value is the default unicode replacement character is
///     0xFFFD (U+65533).
///
/// - Output output: The 1-D Tensor of strings encoded from the provided unicode codepoints.
@inlinable @inline(__always)
public static func unicodeEncode(
  inputValues: Tensor<Int32>,
  inputSplits: Tensor<Int64>,
  errors: Errors = .replace,
  outputEncoding: OutputEncoding,
  replacementChar: Int64 = 65533
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("UnicodeEncode",
    inputValues,
    inputSplits,
    errors: errors.cName,
    output_encoding: outputEncoding.cName,
    replacement_char: replacementChar)
  return StringTensor(handle: ret)
}

/// Determine the script codes of a given tensor of Unicode integer code points.
///
/// This operation converts Unicode code points to script codes corresponding to
/// each code point. Script codes correspond to International Components for
/// Unicode (ICU) UScriptCode values. See http://icu-project.org/apiref/icu4c/uscript_8h.html.
/// Returns -1 (USCRIPT_INVALID_CODE) for invalid codepoints. Output shape will
/// match input shape.
///
/// - Parameter input: A Tensor of int32 Unicode code points.
///
/// - Output output: A Tensor of int32 script codes corresponding to each input code point.
@inlinable @inline(__always)
public static func unicodeScript(
  _ input: Tensor<Int32>
) -> Tensor<Int32> {
  let ret: TensorHandle<Int32> = #tfop("UnicodeScript",
    input)
  return Tensor(handle: ret)
}

/// Transcode the input text from a source encoding to a destination encoding.
///
/// The input is a string tensor of any shape. The output is a string tensor of
/// the same shape containing the transcoded strings. Output strings are always
/// valid unicode. If the input contains invalid encoding positions, the
/// `errors` attribute sets the policy for how to deal with them. If the default
/// error-handling policy is used, invalid formatting will be substituted in the
/// output by the `replacement_char`. If the errors policy is to `ignore`, any
/// invalid encoding positions in the input are skipped and not included in the
/// output. If it set to `strict` then any invalid formatting will result in an
/// InvalidArgument error.
///
/// This operation can be used with `output_encoding = input_encoding` to enforce
/// correct formatting for inputs even if they are already in the desired encoding.
///
/// If the input is prefixed by a Byte Order Mark needed to determine encoding
/// (e.g. if the encoding is UTF-16 and the BOM indicates big-endian), then that
/// BOM will be consumed and not emitted into the output. If the input encoding
/// is marked with an explicit endianness (e.g. UTF-16-BE), then the BOM is
/// interpreted as a non-breaking-space and is preserved in the output (including
/// always for UTF-8).
///
/// The end result is that if the input is marked as an explicit endianness the
/// transcoding is faithful to all codepoints in the source. If it is not marked
/// with an explicit endianness, the BOM is not considered part of the string itself
/// but as metadata, and so is not preserved in the output.
///
/// - Parameter input: The text to be processed. Can have any shape.
///
/// - Attrs:
///   - input_encoding: Text encoding of the input strings. This is any of the encodings supported
///     by ICU ucnv algorithmic converters. Examples: `"UTF-16", "US ASCII", "UTF-8"`.
///   - output_encoding: The unicode encoding to use in the output. Must be one of
///     `"UTF-8", "UTF-16-BE", "UTF-32-BE"`. Multi-byte encodings will be big-endian.
///   - errors: Error handling policy when there is invalid formatting found in the input.
///     The value of 'strict' will cause the operation to produce a InvalidArgument
///     error on any invalid input formatting. A value of 'replace' (the default) will
///     cause the operation to replace any invalid formatting in the input with the
///     `replacement_char` codepoint. A value of 'ignore' will cause the operation to
///     skip any invalid formatting in the input and produce no corresponding output
///     character.
///   - replacement_char: The replacement character codepoint to be used in place of any invalid
///     formatting in the input when `errors='replace'`. Any valid unicode codepoint may
///     be used. The default value is the default unicode replacement character is
///     0xFFFD or U+65533.)
///
///     Note that for UTF-8, passing a replacement character expressible in 1 byte, such
///     as ' ', will preserve string alignment to the source since invalid bytes will be
///     replaced with a 1-byte replacement. For UTF-16-BE and UTF-16-LE, any 1 or 2 byte
///     replacement character will preserve byte alignment to the source.
///   - replace_control_characters: Whether to replace the C0 control characters (00-1F) with the
///     `replacement_char`. Default is false.
///
/// - Output output: A string tensor containing unicode text encoded using `output_encoding`.
@inlinable @inline(__always)
public static func unicodeTranscode(
  _ input: StringTensor,
  inputEncoding: String,
  outputEncoding: OutputEncoding,
  errors: Errors = .replace,
  replacementChar: Int64 = 65533,
  replaceControlCharacters: Bool = false
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("UnicodeTranscode",
    input,
    input_encoding: inputEncoding,
    output_encoding: outputEncoding.cName,
    errors: errors.cName,
    replacement_char: replacementChar,
    replace_control_characters: replaceControlCharacters)
  return StringTensor(handle: ret)
}

/// Generates labels for candidate sampling with a uniform distribution.
///
/// See explanations of candidate sampling and the data formats at
/// go/candidate-sampling.
///
/// For each batch, this op picks a single set of sampled candidate labels.
///
/// The advantages of sampling candidates per-batch are simplicity and the
/// possibility of efficient dense matrix multiplication. The disadvantage is that
/// the sampled candidates must be chosen independently of the context and of the
/// true labels.
///
/// - Parameter true_classes: A batch_size * num_true matrix, in which each row contains the
///   IDs of the num_true target_classes in the corresponding original label.
///
/// - Attrs:
///   - num_true: Number of true labels per context.
///   - num_sampled: Number of candidates to randomly sample.
///   - unique: If unique is true, we sample with rejection, so that all sampled
///     candidates in a batch are unique. This requires some approximation to
///     estimate the post-rejection sampling probabilities.
///   - range_max: The sampler will sample integers from the interval [0, range_max).
///   - seed: If either seed or seed2 are set to be non-zero, the random number
///     generator is seeded by the given seed.  Otherwise, it is seeded by a
///     random seed.
///   - seed2: An second seed to avoid seed collision.
///
/// - Outputs:
///   - sampled_candidates: A vector of length num_sampled, in which each element is
///     the ID of a sampled candidate.
///   - true_expected_count: A batch_size * num_true matrix, representing
///     the number of times each candidate is expected to occur in a batch
///     of sampled candidates. If unique=true, then this is a probability.
///   - sampled_expected_count: A vector of length num_sampled, for each sampled
///     candidate representing the number of times the candidate is expected
///     to occur in a batch of sampled candidates.  If unique=true, then this is a
///     probability.
@inlinable @inline(__always)
public static func uniformCandidateSampler(
  trueClasses: Tensor<Int64>,
  numTrue: Int64,
  numSampled: Int64,
  unique: Bool,
  rangeMax: Int64,
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> (sampledCandidates: Tensor<Int64>, trueExpectedCount: Tensor<Float>, sampledExpectedCount: Tensor<Float>) {
  let ret: (TensorHandle<Int64>, TensorHandle<Float>, TensorHandle<Float>) = #tfop("UniformCandidateSampler",
    trueClasses,
    num_true: numTrue,
    num_sampled: numSampled,
    unique: unique,
    range_max: rangeMax,
    seed: seed,
    seed2: seed2)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Finds unique elements in a 1-D tensor.
///
/// This operation returns a tensor `y` containing all of the unique elements of `x`
/// sorted in the same order that they occur in `x`. This operation also returns a
/// tensor `idx` the same size as `x` that contains the index of each value of `x`
/// in the unique output `y`. In other words:
///
/// `y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`
///
/// For example:
///
/// ```
/// # tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
/// y, idx = unique(x)
/// y ==> [1, 2, 4, 7, 8]
/// idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
/// ```
///
/// - Parameter x: 1-D.
///
/// - Outputs:
///   - y: 1-D.
///   - idx: 1-D.
@inlinable @inline(__always)
public static func unique<T: TensorFlowScalar, OutIdx: BinaryInteger & TensorFlowScalar>(
  _ x: Tensor<T>
) -> (y: Tensor<T>, idx: Tensor<OutIdx>) {
  let ret: (TensorHandle<T>, TensorHandle<OutIdx>) = #tfop("Unique",
    x,
    T$dtype: T.tensorFlowDataType,
    out_idx$dtype: OutIdx.tensorFlowDataType)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

/// Finds unique elements along an axis of a tensor.
///
/// This operation either returns a tensor `y` containing unique elements
/// along the `axis` of a tensor. The returned unique elements is sorted
/// in the same order as they occur along `axis` in `x`.
/// This operation also returns a tensor `idx` that is the same size as
/// the number of the elements in `x` along the `axis` dimension. It
/// contains the index in the unique output `y`.
/// In other words, for an `1-D` tensor `x` with `axis = None:
///
/// `y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`
///
/// For example:
///
/// ```
/// # tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
/// y, idx = unique(x)
/// y ==> [1, 2, 4, 7, 8]
/// idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
/// ```
///
/// For an `2-D` tensor `x` with `axis = 0`:
///
/// ```
/// # tensor 'x' is [[1, 0, 0],
/// #                [1, 0, 0],
/// #                [2, 0, 0]]
/// y, idx = unique(x, axis=0)
/// y ==> [[1, 0, 0],
///        [2, 0, 0]]
/// idx ==> [0, 0, 1]
/// ```
///
/// For an `2-D` tensor `x` with `axis = 1`:
///
/// ```
/// # tensor 'x' is [[1, 0, 0],
/// #                [1, 0, 0],
/// #                [2, 0, 0]]
/// y, idx = unique(x, axis=1)
/// y ==> [[1, 0],
///        [1, 0],
///        [2, 0]]
/// idx ==> [0, 1, 1]
/// ```
///
/// - Parameters:
///   - x: A `Tensor`.
///   - axis: A `Tensor` of type `int32` (default: None). The axis of the Tensor to
///     find the unique elements.
///
/// - Outputs:
///   - y: A `Tensor`. Unique elements along the `axis` of `Tensor` x.
///   - idx: A 1-D Tensor. Has the same type as x that contains the index of each
///     value of x in the output y.
@inlinable @inline(__always)
public static func uniqueV2<T: TensorFlowScalar, Taxis: BinaryInteger & TensorFlowScalar, OutIdx: BinaryInteger & TensorFlowScalar>(
  _ x: Tensor<T>,
  axis: Tensor<Taxis>
) -> (y: Tensor<T>, idx: Tensor<OutIdx>) {
  let ret: (TensorHandle<T>, TensorHandle<OutIdx>) = #tfop("UniqueV2",
    x,
    axis,
    T$dtype: T.tensorFlowDataType,
    Taxis$dtype: Taxis.tensorFlowDataType,
    out_idx$dtype: OutIdx.tensorFlowDataType)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1))
}

/// Finds unique elements in a 1-D tensor.
///
/// This operation returns a tensor `y` containing all of the unique elements of `x`
/// sorted in the same order that they occur in `x`. This operation also returns a
/// tensor `idx` the same size as `x` that contains the index of each value of `x`
/// in the unique output `y`. Finally, it returns a third tensor `count` that
/// contains the count of each element of `y` in `x`. In other words:
///
/// `y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`
///
/// For example:
///
/// ```
/// # tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
/// y, idx, count = unique_with_counts(x)
/// y ==> [1, 2, 4, 7, 8]
/// idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
/// count ==> [2, 1, 3, 1, 2]
/// ```
///
/// - Parameter x: 1-D.
///
/// - Outputs:
///   - y: 1-D.
///   - idx: 1-D.
///   - count: 1-D.
@inlinable @inline(__always)
public static func uniqueWithCounts<T: TensorFlowScalar, OutIdx: BinaryInteger & TensorFlowScalar>(
  _ x: Tensor<T>
) -> (y: Tensor<T>, idx: Tensor<OutIdx>, count: Tensor<OutIdx>) {
  let ret: (TensorHandle<T>, TensorHandle<OutIdx>, TensorHandle<OutIdx>) = #tfop("UniqueWithCounts",
    x,
    T$dtype: T.tensorFlowDataType,
    out_idx$dtype: OutIdx.tensorFlowDataType)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Finds unique elements along an axis of a tensor.
///
/// This operation either returns a tensor `y` containing unique elements
/// along the `axis` of a tensor. The returned unique elements is sorted
/// in the same order as they occur along `axis` in `x`.
/// This operation also returns a tensor `idx` and a tensor `count`
/// that are the same size as the number of the elements in `x` along the
/// `axis` dimension. The `idx` contains the index in the unique output `y`
/// and the `count` contains the count in the unique output `y`.
/// In other words, for an `1-D` tensor `x` with `axis = None:
///
/// `y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`
///
/// For example:
///
/// ```
/// # tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
/// y, idx, count = unique_with_counts(x)
/// y ==> [1, 2, 4, 7, 8]
/// idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
/// count ==> [2, 1, 3, 1, 2]
/// ```
///
/// For an `2-D` tensor `x` with `axis = 0`:
///
/// ```
/// # tensor 'x' is [[1, 0, 0],
/// #                [1, 0, 0],
/// #                [2, 0, 0]]
/// y, idx, count = unique_with_counts(x, axis=0)
/// y ==> [[1, 0, 0],
///        [2, 0, 0]]
/// idx ==> [0, 0, 1]
/// count ==> [2, 1]
/// ```
///
/// For an `2-D` tensor `x` with `axis = 1`:
///
/// ```
/// # tensor 'x' is [[1, 0, 0],
/// #                [1, 0, 0],
/// #                [2, 0, 0]]
/// y, idx, count = unique_with_counts(x, axis=1)
/// y ==> [[1, 0],
///        [1, 0],
///        [2, 0]]
/// idx ==> [0, 1, 1]
/// count ==> [1, 2]
/// ```
///
/// - Parameters:
///   - x: A `Tensor`.
///   - axis: A `Tensor` of type `int32` (default: None). The axis of the Tensor to
///     find the unique elements.
///
/// - Outputs:
///   - y: A `Tensor`. Unique elements along the `axis` of `Tensor` x.
///   - idx: A 1-D Tensor. Has the same type as x that contains the index of each
///     value of x in the output y.
///   - count: A 1-D Tensor. The count of each value of x in the output y.
@inlinable @inline(__always)
public static func uniqueWithCountsV2<T: TensorFlowScalar, Taxis: BinaryInteger & TensorFlowScalar, OutIdx: BinaryInteger & TensorFlowScalar>(
  _ x: Tensor<T>,
  axis: Tensor<Taxis>
) -> (y: Tensor<T>, idx: Tensor<OutIdx>, count: Tensor<OutIdx>) {
  let ret: (TensorHandle<T>, TensorHandle<OutIdx>, TensorHandle<OutIdx>) = #tfop("UniqueWithCountsV2",
    x,
    axis,
    T$dtype: T.tensorFlowDataType,
    Taxis$dtype: Taxis.tensorFlowDataType,
    out_idx$dtype: OutIdx.tensorFlowDataType)
  return (Tensor(handle: ret.0), Tensor(handle: ret.1), Tensor(handle: ret.2))
}

/// Converts a flat index or array of flat indices into a tuple of
///
/// coordinate arrays.
///
/// @compatibility(numpy)
/// Equivalent to np.unravel_index
/// @end_compatibility
///
/// - Parameters:
///   - indices: An 0-D or 1-D `int` Tensor whose elements are indices into the
///     flattened version of an array of dimensions dims.
///   - dims: An 1-D `int` Tensor. The shape of the array to use for unraveling
///     indices.
///
/// - Output output: An 2-D (or 1-D if indices is 0-D) tensor where each row has the
///   same shape as the indices array.
@inlinable @inline(__always)
public static func unravelIndex<Tidx: BinaryInteger & TensorFlowScalar>(
  indices: Tensor<Tidx>,
  dims: Tensor<Tidx>
) -> Tensor<Tidx> {
  let ret: TensorHandle<Tidx> = #tfop("UnravelIndex",
    indices,
    dims,
    Tidx$dtype: Tidx.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the maximum along segments of a tensor.
///
/// Read
/// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
/// for an explanation of segments.
///
/// This operator is similar to the unsorted segment sum operator found
/// [(here)](../../../api_docs/python/math_ops.md#UnsortedSegmentSum).
/// Instead of computing the sum over segments, it computes the maximum such that:
///
/// \\(output_i = \max_{j...} data[j...]\\) where max is over tuples `j...` such
/// that `segment_ids[j...] == i`.
///
/// If the maximum is empty for a given segment ID `i`, it outputs the smallest
/// possible value for the specific numeric type,
/// `output[i] = numeric_limits<T>::lowest()`.
///
/// If the given segment ID `i` is negative, then the corresponding value is
/// dropped, and will not be included in the result.
///
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/UnsortedSegmentMax.png" alt>
/// </div>
///
/// For example:
///
/// ``` python
/// c = tf.constant([[1,2,3,4], [5,6,7,8], [4,3,2,1]])
/// tf.unsorted_segment_max(c, tf.constant([0, 1, 0]), num_segments=2)
/// # ==> [[ 4,  3, 3, 4],
/// #       [5,  6, 7, 8]]
/// ```
///
///
/// - Parameter segment_ids: A tensor whose shape is a prefix of `data.shape`.
///
/// - Output output: Has same shape as data, except for the first `segment_ids.rank`
///   dimensions, which are replaced with a single dimension which has size
///   `num_segments`.
@inlinable @inline(__always)
public static func unsortedSegmentMax<T: Numeric & TensorFlowScalar, Tindices: BinaryInteger & TensorFlowScalar, Tnumsegments: BinaryInteger & TensorFlowScalar>(
  data: Tensor<T>,
  segmentIds: Tensor<Tindices>,
  numSegments: Tensor<Tnumsegments>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("UnsortedSegmentMax",
    data,
    segmentIds,
    numSegments,
    T$dtype: T.tensorFlowDataType,
    Tindices$dtype: Tindices.tensorFlowDataType,
    Tnumsegments$dtype: Tnumsegments.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the minimum along segments of a tensor.
///
/// Read
/// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
/// for an explanation of segments.
///
/// This operator is similar to the unsorted segment sum operator found
/// [(here)](../../../api_docs/python/math_ops.md#UnsortedSegmentSum).
/// Instead of computing the sum over segments, it computes the minimum such that:
///
/// \\(output_i = \min_{j...} data_[j...]\\) where min is over tuples `j...` such
/// that `segment_ids[j...] == i`.
///
/// If the minimum is empty for a given segment ID `i`, it outputs the largest
/// possible value for the specific numeric type,
/// `output[i] = numeric_limits<T>::max()`.
///
/// For example:
///
/// ``` python
/// c = tf.constant([[1,2,3,4], [5,6,7,8], [4,3,2,1]])
/// tf.unsorted_segment_min(c, tf.constant([0, 1, 0]), num_segments=2)
/// # ==> [[ 1,  2, 2, 1],
/// #       [5,  6, 7, 8]]
/// ```
///
/// If the given segment ID `i` is negative, then the corresponding value is
/// dropped, and will not be included in the result.
///
/// - Parameter segment_ids: A tensor whose shape is a prefix of `data.shape`.
///
/// - Output output: Has same shape as data, except for the first `segment_ids.rank`
///   dimensions, which are replaced with a single dimension which has size
///   `num_segments`.
@inlinable @inline(__always)
public static func unsortedSegmentMin<T: Numeric & TensorFlowScalar, Tindices: BinaryInteger & TensorFlowScalar, Tnumsegments: BinaryInteger & TensorFlowScalar>(
  data: Tensor<T>,
  segmentIds: Tensor<Tindices>,
  numSegments: Tensor<Tnumsegments>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("UnsortedSegmentMin",
    data,
    segmentIds,
    numSegments,
    T$dtype: T.tensorFlowDataType,
    Tindices$dtype: Tindices.tensorFlowDataType,
    Tnumsegments$dtype: Tnumsegments.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the product along segments of a tensor.
///
/// Read
/// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
/// for an explanation of segments.
///
/// This operator is similar to the unsorted segment sum operator found
/// [(here)](../../../api_docs/python/math_ops.md#UnsortedSegmentSum).
/// Instead of computing the sum over segments, it computes the product of all
/// entries belonging to a segment such that:
///
/// \\(output_i = \prod_{j...} data[j...]\\) where the product is over tuples
/// `j...` such that `segment_ids[j...] == i`.
///
/// For example:
///
/// ``` python
/// c = tf.constant([[1,2,3,4], [5,6,7,8], [4,3,2,1]])
/// tf.unsorted_segment_prod(c, tf.constant([0, 1, 0]), num_segments=2)
/// # ==> [[ 4,  6, 6, 4],
/// #       [5,  6, 7, 8]]
/// ```
///
/// If there is no entry for a given segment ID `i`, it outputs 1.
///
/// If the given segment ID `i` is negative, then the corresponding value is
/// dropped, and will not be included in the result.
///
/// - Parameter segment_ids: A tensor whose shape is a prefix of `data.shape`.
///
/// - Output output: Has same shape as data, except for the first `segment_ids.rank`
///   dimensions, which are replaced with a single dimension which has size
///   `num_segments`.
@inlinable @inline(__always)
public static func unsortedSegmentProd<T: Numeric & TensorFlowScalar, Tindices: BinaryInteger & TensorFlowScalar, Tnumsegments: BinaryInteger & TensorFlowScalar>(
  data: Tensor<T>,
  segmentIds: Tensor<Tindices>,
  numSegments: Tensor<Tnumsegments>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("UnsortedSegmentProd",
    data,
    segmentIds,
    numSegments,
    T$dtype: T.tensorFlowDataType,
    Tindices$dtype: Tindices.tensorFlowDataType,
    Tnumsegments$dtype: Tnumsegments.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Computes the sum along segments of a tensor.
///
/// Read
/// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
/// for an explanation of segments.
///
/// Computes a tensor such that
/// \\(output[i] = \sum_{j...} data[j...]\\) where the sum is over tuples `j...` such
/// that `segment_ids[j...] == i`.  Unlike `SegmentSum`, `segment_ids`
/// need not be sorted and need not cover all values in the full
/// range of valid values.
///
/// If the sum is empty for a given segment ID `i`, `output[i] = 0`.
/// If the given segment ID `i` is negative, the value is dropped and will not be
/// added to the sum of the segment.
///
/// `num_segments` should equal the number of distinct segment IDs.
///
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/UnsortedSegmentSum.png" alt>
/// </div>
///
/// ``` python
/// c = tf.constant([[1,2,3,4], [5,6,7,8], [4,3,2,1]])
/// tf.unsorted_segment_sum(c, tf.constant([0, 1, 0]), num_segments=2)
/// # ==> [[ 5,  5, 5, 5],
/// #       [5,  6, 7, 8]]
/// ```
///
///
/// - Parameter segment_ids: A tensor whose shape is a prefix of `data.shape`.
///
/// - Output output: Has same shape as data, except for the first `segment_ids.rank`
///   dimensions, which are replaced with a single dimension which has size
///   `num_segments`.
@inlinable @inline(__always)
public static func unsortedSegmentSum<T: Numeric & TensorFlowScalar, Tindices: BinaryInteger & TensorFlowScalar, Tnumsegments: BinaryInteger & TensorFlowScalar>(
  data: Tensor<T>,
  segmentIds: Tensor<Tindices>,
  numSegments: Tensor<Tnumsegments>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("UnsortedSegmentSum",
    data,
    segmentIds,
    numSegments,
    T$dtype: T.tensorFlowDataType,
    Tindices$dtype: Tindices.tensorFlowDataType,
    Tnumsegments$dtype: Tnumsegments.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Applies upper_bound(sorted_search_values, values) along each row.
///
/// Each set of rows with the same index in (sorted_inputs, values) is treated
/// independently.  The resulting row is the equivalent of calling
/// `np.searchsorted(sorted_inputs, values, side='right')`.
///
/// The result is not a global index to the entire 
/// `Tensor`, but rather just the index in the last dimension.
///
/// A 2-D example:
///   sorted_sequence = [[0, 3, 9, 9, 10],
///                      [1, 2, 3, 4, 5]]
///   values = [[2, 4, 9],
///             [0, 2, 6]]
///
///   result = UpperBound(sorted_sequence, values)
///
///   result == [[1, 2, 4],
///              [0, 2, 5]]
///
/// - Parameters:
///   - sorted_inputs: 2-D Tensor where each row is ordered.
///   - values: 2-D Tensor with the same numbers of rows as `sorted_search_values`. Contains
///     the values that will be searched for in `sorted_search_values`.
///
/// - Output output: A `Tensor` with the same shape as `values`.  It contains the last scalar index
///   into the last dimension where values can be inserted without changing the
///   ordered property.
@inlinable @inline(__always)
public static func upperBound<T: TensorFlowScalar, OutType: BinaryInteger & TensorFlowScalar>(
  sortedInputs: Tensor<T>,
  _ values: Tensor<T>
) -> Tensor<OutType> {
  let ret: TensorHandle<OutType> = #tfop("UpperBound",
    sortedInputs,
    values,
    T$dtype: T.tensorFlowDataType,
    out_type$dtype: OutType.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Returns locations of nonzero / true values in a tensor.
///
/// This operation returns the coordinates of true elements in `condition`. The
/// coordinates are returned in a 2-D tensor where the first dimension (rows)
/// represents the number of true elements, and the second dimension (columns)
/// represents the coordinates of the true elements. Keep in mind, the shape of
/// the output tensor can vary depending on how many true values there are in
/// `condition`. Indices are output in row-major order.
///
/// For example:
///
/// ```
/// # 'input' tensor is [[True, False]
/// #                    [True, False]]
/// # 'input' has two true values, so output has two coordinates.
/// # 'input' has rank of 2, so coordinates have two indices.
/// where(input) ==> [[0, 0],
///                   [1, 0]]
///
/// # `condition` tensor is [[[True, False]
/// #                     [True, False]]
/// #                    [[False, True]
/// #                     [False, True]]
/// #                    [[False, False]
/// #                     [False, True]]]
/// # 'input' has 5 true values, so output has 5 coordinates.
/// # 'input' has rank of 3, so coordinates have three indices.
/// where(input) ==> [[0, 0, 0],
///                   [0, 1, 0],
///                   [1, 0, 1],
///                   [1, 1, 1],
///                   [2, 1, 1]]
///
/// # `condition` tensor is [[[1.5,  0.0]
/// #                     [-0.5, 0.0]]
/// #                    [[0.0,  0.25]
/// #                     [0.0,  0.75]]
/// #                    [[0.0,  0.0]
/// #                     [0.0,  0.01]]]
/// # 'input' has 5 nonzero values, so output has 5 coordinates.
/// # 'input' has rank of 3, so coordinates have three indices.
/// where(input) ==> [[0, 0, 0],
///                   [0, 1, 0],
///                   [1, 0, 1],
///                   [1, 1, 1],
///                   [2, 1, 1]]
///
/// # `condition` tensor is [[[1.5 + 0.0j, 0.0  + 0.0j]
/// #                     [0.0 + 0.5j, 0.0  + 0.0j]]
/// #                    [[0.0 + 0.0j, 0.25 + 1.5j]
/// #                     [0.0 + 0.0j, 0.75 + 0.0j]]
/// #                    [[0.0 + 0.0j, 0.0  + 0.0j]
/// #                     [0.0 + 0.0j, 0.01 + 0.0j]]]
/// # 'input' has 5 nonzero magnitude values, so output has 5 coordinates.
/// # 'input' has rank of 3, so coordinates have three indices.
/// where(input) ==> [[0, 0, 0],
///                   [0, 1, 0],
///                   [1, 0, 1],
///                   [1, 1, 1],
///                   [2, 1, 1]]
/// ```
@inlinable @inline(__always)
public static func where_<T: TensorFlowScalar>(
  _ input: Tensor<T>
) -> Tensor<Int64> {
  let ret: TensorHandle<Int64> = #tfop("Where",
    input,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Worker heartbeat op.
///
/// Heartbeats may be sent periodically to indicate the coordinator is still active,
/// to retrieve the current worker status and to expedite shutdown when necessary.
///
/// - Parameter request: A string tensor containing a serialized WorkerHeartbeatRequest
///
/// - Output response: A string tensor containing a serialized WorkerHeartbeatResponse
@inlinable @inline(__always)
public static func workerHeartbeat(
  request: StringTensor
) -> StringTensor {
  let ret: TensorHandle<String> = #tfop("WorkerHeartbeat",
    request)
  return StringTensor(handle: ret)
}

/// Writes contents to the file at input filename. Creates file and recursively
///
/// creates directory if not existing.
///
/// - Parameters:
///   - filename: scalar. The name of the file to which we write the contents.
///   - contents: scalar. The content to be written to the output file.
@inlinable @inline(__always)
public static func writeFile(
  filename: StringTensor,
  contents: StringTensor
) {
  return #tfop("WriteFile",
    filename,
    contents)
}

/// Returns 0 if x == 0, and x / y otherwise, elementwise.
@inlinable @inline(__always)
public static func xdivy<T: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Xdivy",
    x,
    y,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Returns 0 if x == 0, and x * log(y) otherwise, elementwise.
@inlinable @inline(__always)
public static func xlogy<T: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Xlogy",
    x,
    y,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Returns a tensor of zeros with the same shape and type as x.
///
/// - Parameter x: a tensor of type T.
///
/// - Output y: a tensor of the same shape and type as x but filled with zeros.
@inlinable @inline(__always)
public static func zerosLike<T: TensorFlowScalar>(
  _ x: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("ZerosLike",
    x,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

/// Compute the Hurwitz zeta function \\(\zeta(x, q)\\).
///
/// The Hurwitz zeta function is defined as:
///
///
/// \\(\zeta(x, q) = \sum_{n=0}^{\infty} (q + n)^{-x}\\)
@inlinable @inline(__always)
public static func zeta<T: FloatingPoint & TensorFlowScalar>(
  _ x: Tensor<T>,
  q: Tensor<T>
) -> Tensor<T> {
  let ret: TensorHandle<T> = #tfop("Zeta",
    x,
    q,
    T$dtype: T.tensorFlowDataType)
  return Tensor(handle: ret)
}

}