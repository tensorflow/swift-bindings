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

static let generatedTensorFlowVersion = "1.9.0-rc0"
static let generatedTensorFlowGitVersion = "v1.8.0-3901-g9590c4c32d"

@_frozen
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

@_frozen
public enum DataFormat {
  case hwcn
  case hwnc
  case nchw
  case nhwc

  @inlinable
  var cName: String {
    @inline(__always)
    get {
      switch self {
      case .hwcn: return "HWCN"
      case .hwnc: return "HWNC"
      case .nchw: return "NCHW"
      case .nhwc: return "NHWC"
      }
    }
  }
}

@_frozen
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

@_frozen
public enum DataFormat3 {
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

@_frozen
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

@_frozen
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

@_frozen
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

@_frozen
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

@_frozen
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

@_frozen
public enum LossType {
  case hingeLoss
  case logisticLoss
  case smoothHingeLoss
  case squaredLoss

  @inlinable
  var cName: String {
    @inline(__always)
    get {
      switch self {
      case .hingeLoss: return "hinge_loss"
      case .logisticLoss: return "logistic_loss"
      case .smoothHingeLoss: return "smooth_hinge_loss"
      case .squaredLoss: return "squared_loss"
      }
    }
  }
}

@_frozen
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

@_frozen
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

@_frozen
public enum Method2 {
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

@_frozen
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

@_frozen
public enum Mode4 {
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

@_frozen
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

@_frozen
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

@_frozen
public enum RoundMode {
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

@inlinable @inline(__always)
public static func a(
) -> Tensor<Float> {
  return #tfop("A")
}

@inlinable @inline(__always)
public static func abort(
  errorMsg: String,
  exitWithoutError: Bool = false
) {
  return #tfop("Abort",
    error_msg: errorMsg,
    exit_without_error: exitWithoutError)
}

@inlinable @inline(__always)
public static func abs<T: Numeric>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Abs",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func acos<T: Numeric>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Acos",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func acosh<T: BinaryFloatingPoint>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Acosh",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func add<T: Numeric>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  return #tfop("Add",
    x,
    y,
    T: T.self)
}

@inlinable @inline(__always)
public static func addManySparseToTensorsMap<T: AccelerableByTensorFlow>(
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

@inlinable @inline(__always)
public static func addN<T: Numeric>(
  inputs: [Tensor<T>]
) -> Tensor<T> {
  return #tfop("AddN",
    inputs,
    T: T.self)
}

@inlinable @inline(__always)
public static func addSparseToTensorsMap<T: AccelerableByTensorFlow>(
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

@inlinable @inline(__always)
public static func addV2<T: Numeric>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  return #tfop("AddV2",
    x,
    y,
    T: T.self)
}

@inlinable @inline(__always)
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

@inlinable @inline(__always)
public static func adjustContrastv2(
  images: Tensor<Float>,
  contrastFactor: Tensor<Float>
) -> Tensor<Float> {
  return #tfop("AdjustContrastv2",
    images,
    contrastFactor)
}

@inlinable @inline(__always)
public static func adjustHue(
  images: Tensor<Float>,
  delta: Tensor<Float>
) -> Tensor<Float> {
  return #tfop("AdjustHue",
    images,
    delta)
}

@inlinable @inline(__always)
public static func adjustSaturation(
  images: Tensor<Float>,
  scale: Tensor<Float>
) -> Tensor<Float> {
  return #tfop("AdjustSaturation",
    images,
    scale)
}

@inlinable @inline(__always)
public static func all<Tidx: BinaryInteger>(
  _ input: Tensor<Bool>,
  reductionIndices: Tensor<Tidx>,
  keepDims: Bool = false
) -> Tensor<Bool> {
  return #tfop("All",
    input,
    reductionIndices,
    Tidx: Tidx.self,
    keep_dims: keepDims)
}

@inlinable @inline(__always)
public static func allCandidateSampler(
  trueClasses: Tensor<Int64>,
  numTrue: Int64,
  numSampled: Int64,
  unique: Bool,
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> (Tensor<Int64>, Tensor<Float>, Tensor<Float>) {
  return #tfop("AllCandidateSampler",
    trueClasses,
    num_true: numTrue,
    num_sampled: numSampled,
    unique: unique,
    seed: seed,
    seed2: seed2)
}

@inlinable @inline(__always)
public static func angle<T: AccelerableByTensorFlow, Tout: BinaryFloatingPoint>(
  _ input: Tensor<T>
) -> Tensor<Tout> {
  return #tfop("Angle",
    input,
    T: T.self,
    Tout: Tout.self)
}

@inlinable @inline(__always)
public static func any<Tidx: BinaryInteger>(
  _ input: Tensor<Bool>,
  reductionIndices: Tensor<Tidx>,
  keepDims: Bool = false
) -> Tensor<Bool> {
  return #tfop("Any",
    input,
    reductionIndices,
    Tidx: Tidx.self,
    keep_dims: keepDims)
}

@inlinable @inline(__always)
public static func applyAdaMax<T: Numeric>(
  var_: Tensor<T>,
  m: Tensor<T>,
  v: Tensor<T>,
  beta1Power: Tensor<T>,
  lr: Tensor<T>,
  beta1: Tensor<T>,
  beta2: Tensor<T>,
  epsilon: Tensor<T>,
  grad: Tensor<T>,
  useLocking: Bool = false
) -> Tensor<T> {
  return #tfop("ApplyAdaMax",
    var_,
    m,
    v,
    beta1Power,
    lr,
    beta1,
    beta2,
    epsilon,
    grad,
    T: T.self,
    use_locking: useLocking)
}

@inlinable @inline(__always)
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

@inlinable @inline(__always)
public static func applyAdagrad<T: Numeric>(
  var_: Tensor<T>,
  accum: Tensor<T>,
  lr: Tensor<T>,
  grad: Tensor<T>,
  useLocking: Bool = false,
  updateSlots: Bool = true
) -> Tensor<T> {
  return #tfop("ApplyAdagrad",
    var_,
    accum,
    lr,
    grad,
    T: T.self,
    use_locking: useLocking,
    update_slots: updateSlots)
}

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
public static func approximateEqual<T: Numeric>(
  _ x: Tensor<T>,
  _ y: Tensor<T>,
  tolerance: Double = 1e-05
) -> Tensor<Bool> {
  return #tfop("ApproximateEqual",
    x,
    y,
    T: T.self,
    tolerance: tolerance)
}

@inlinable @inline(__always)
public static func argMax<T: Numeric, Tidx: BinaryInteger, Output_type: BinaryInteger>(
  _ input: Tensor<T>,
  dimension: Tensor<Tidx>
) -> Tensor<Output_type> {
  return #tfop("ArgMax",
    input,
    dimension,
    T: T.self,
    Tidx: Tidx.self,
    output_type: Output_type.self)
}

@inlinable @inline(__always)
public static func argMin<T: Numeric, Tidx: BinaryInteger, Output_type: BinaryInteger>(
  _ input: Tensor<T>,
  dimension: Tensor<Tidx>
) -> Tensor<Output_type> {
  return #tfop("ArgMin",
    input,
    dimension,
    T: T.self,
    Tidx: Tidx.self,
    output_type: Output_type.self)
}

@inlinable @inline(__always)
public static func asin<T: Numeric>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Asin",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func asinh<T: BinaryFloatingPoint>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Asinh",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func assert<T: AccelerableByTensorFlow>(
  condition: Tensor<Bool>,
  data: [Tensor<T>],
  summarize: Int64 = 3
) {
  return #tfop("Assert",
    condition,
    data,
    summarize: summarize)
}

@inlinable @inline(__always)
public static func assign<T: AccelerableByTensorFlow>(
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
public static func atan<T: Numeric>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Atan",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func atan2<T: BinaryFloatingPoint>(
  _ y: Tensor<T>,
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Atan2",
    y,
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func atanh<T: BinaryFloatingPoint>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Atanh",
    x,
    T: T.self)
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
public static func attrListTypeDefault<T: AccelerableByTensorFlow>(
  _ a: [Tensor<T>],
  _ b: [Tensor<T>]
) {
  return #tfop("AttrListTypeDefault",
    a,
    b,
    T: T.self)
}

@inlinable @inline(__always)
public static func attrMin(
  _ a: Int64
) {
  return #tfop("AttrMin",
    a: a)
}

@inlinable @inline(__always)
public static func attrTypeDefault<T: AccelerableByTensorFlow>(
  _ a: Tensor<T>
) {
  return #tfop("AttrTypeDefault",
    a,
    T: T.self)
}

@inlinable @inline(__always)
public static func audioSpectrogram(
  _ input: Tensor<Float>,
  windowSize: Int64,
  stride: Int64,
  magnitudeSquared: Bool = false
) -> Tensor<Float> {
  return #tfop("AudioSpectrogram",
    input,
    window_size: windowSize,
    stride: stride,
    magnitude_squared: magnitudeSquared)
}

@inlinable @inline(__always)
public static func avgPool<T: BinaryFloatingPoint>(
  value: Tensor<T>,
  ksize: [Int32],
  strides: [Int32],
  padding: Padding,
  dataFormat: DataFormat = .nhwc
) -> Tensor<T> {
  return #tfop("AvgPool",
    value,
    T: T.self,
    ksize: ksize,
    strides: strides,
    padding: padding.cName,
    data_format: dataFormat.cName)
}

@inlinable @inline(__always)
public static func avgPool3D<T: BinaryFloatingPoint>(
  _ input: Tensor<T>,
  ksize: [Int32],
  strides: [Int32],
  padding: Padding,
  dataFormat: DataFormat1 = .ndhwc
) -> Tensor<T> {
  return #tfop("AvgPool3D",
    input,
    T: T.self,
    ksize: ksize,
    strides: strides,
    padding: padding.cName,
    data_format: dataFormat.cName)
}

@inlinable @inline(__always)
public static func avgPool3DGrad<T: BinaryFloatingPoint>(
  origInputShape: Tensor<Int32>,
  grad: Tensor<T>,
  ksize: [Int32],
  strides: [Int32],
  padding: Padding,
  dataFormat: DataFormat1 = .ndhwc
) -> Tensor<T> {
  return #tfop("AvgPool3DGrad",
    origInputShape,
    grad,
    T: T.self,
    ksize: ksize,
    strides: strides,
    padding: padding.cName,
    data_format: dataFormat.cName)
}

@inlinable @inline(__always)
public static func avgPoolGrad<T: BinaryFloatingPoint>(
  origInputShape: Tensor<Int32>,
  grad: Tensor<T>,
  ksize: [Int32],
  strides: [Int32],
  padding: Padding,
  dataFormat: DataFormat = .nhwc
) -> Tensor<T> {
  return #tfop("AvgPoolGrad",
    origInputShape,
    grad,
    T: T.self,
    ksize: ksize,
    strides: strides,
    padding: padding.cName,
    data_format: dataFormat.cName)
}

@inlinable @inline(__always)
public static func b(
) -> Tensor<Float> {
  return #tfop("B")
}

@inlinable @inline(__always)
public static func batch<T: AccelerableByTensorFlow>(
  inTensors: [Tensor<T>],
  numBatchThreads: Int64,
  maxBatchSize: Int64,
  maxEnqueuedBatches: Int64 = 10,
  batchTimeoutMicros: Int64,
  allowedBatchSizes: [Int32],
  gradTimeoutMicros: Int64,
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

@inlinable @inline(__always)
public static func batchCholesky<T: BinaryFloatingPoint>(
  _ input: Tensor<T>
) -> Tensor<T> {
  return #tfop("BatchCholesky",
    input,
    T: T.self)
}

@inlinable @inline(__always)
public static func batchCholeskyGrad<T: BinaryFloatingPoint>(
  l: Tensor<T>,
  grad: Tensor<T>
) -> Tensor<T> {
  return #tfop("BatchCholeskyGrad",
    l,
    grad,
    T: T.self)
}

@inlinable @inline(__always)
public static func batchMatMul<T: Numeric>(
  _ x: Tensor<T>,
  _ y: Tensor<T>,
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

@inlinable @inline(__always)
public static func batchMatrixBandPart<T: AccelerableByTensorFlow>(
  _ input: Tensor<T>,
  numLower: Tensor<Int64>,
  numUpper: Tensor<Int64>
) -> Tensor<T> {
  return #tfop("BatchMatrixBandPart",
    input,
    numLower,
    numUpper,
    T: T.self)
}

@inlinable @inline(__always)
public static func batchMatrixDeterminant<T: BinaryFloatingPoint>(
  _ input: Tensor<T>
) -> Tensor<T> {
  return #tfop("BatchMatrixDeterminant",
    input,
    T: T.self)
}

@inlinable @inline(__always)
public static func batchMatrixDiag<T: AccelerableByTensorFlow>(
  diagonal: Tensor<T>
) -> Tensor<T> {
  return #tfop("BatchMatrixDiag",
    diagonal,
    T: T.self)
}

@inlinable @inline(__always)
public static func batchMatrixDiagPart<T: AccelerableByTensorFlow>(
  _ input: Tensor<T>
) -> Tensor<T> {
  return #tfop("BatchMatrixDiagPart",
    input,
    T: T.self)
}

@inlinable @inline(__always)
public static func batchMatrixInverse<T: BinaryFloatingPoint>(
  _ input: Tensor<T>,
  adjoint: Bool = false
) -> Tensor<T> {
  return #tfop("BatchMatrixInverse",
    input,
    T: T.self,
    adjoint: adjoint)
}

@inlinable @inline(__always)
public static func batchMatrixSetDiag<T: AccelerableByTensorFlow>(
  _ input: Tensor<T>,
  diagonal: Tensor<T>
) -> Tensor<T> {
  return #tfop("BatchMatrixSetDiag",
    input,
    diagonal,
    T: T.self)
}

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
public static func batchSelfAdjointEig<T: BinaryFloatingPoint>(
  _ input: Tensor<T>
) -> Tensor<T> {
  return #tfop("BatchSelfAdjointEig",
    input,
    T: T.self)
}

@inlinable @inline(__always)
public static func batchSelfAdjointEigV2<T: BinaryFloatingPoint>(
  _ input: Tensor<T>,
  computeV: Bool = true
) -> (Tensor<T>, Tensor<T>) {
  return #tfop("BatchSelfAdjointEigV2",
    input,
    T: T.self,
    compute_v: computeV)
}

@inlinable @inline(__always)
public static func batchSvd<T: BinaryFloatingPoint>(
  _ input: Tensor<T>,
  computeUv: Bool = true,
  fullMatrices: Bool = false
) -> (Tensor<T>, Tensor<T>, Tensor<T>) {
  return #tfop("BatchSvd",
    input,
    T: T.self,
    compute_uv: computeUv,
    full_matrices: fullMatrices)
}

@inlinable @inline(__always)
public static func batchToSpace<T: AccelerableByTensorFlow, Tidx: BinaryInteger>(
  _ input: Tensor<T>,
  crops: Tensor<Tidx>,
  blockSize: Int64
) -> Tensor<T> {
  return #tfop("BatchToSpace",
    input,
    crops,
    T: T.self,
    Tidx: Tidx.self,
    block_size: blockSize)
}

@inlinable @inline(__always)
public static func batchToSpaceND<T: AccelerableByTensorFlow, Tblock_shape: BinaryInteger, Tcrops: BinaryInteger>(
  _ input: Tensor<T>,
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

@inlinable @inline(__always)
public static func besselI0e<T: BinaryFloatingPoint>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("BesselI0e",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func besselI1e<T: BinaryFloatingPoint>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("BesselI1e",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func betainc<T: BinaryFloatingPoint>(
  _ a: Tensor<T>,
  _ b: Tensor<T>,
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Betainc",
    a,
    b,
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func biasAdd<T: Numeric>(
  value: Tensor<T>,
  bias: Tensor<T>,
  dataFormat: DataFormat = .nhwc
) -> Tensor<T> {
  return #tfop("BiasAdd",
    value,
    bias,
    T: T.self,
    data_format: dataFormat.cName)
}

@inlinable @inline(__always)
public static func biasAddGrad<T: Numeric>(
  outBackprop: Tensor<T>,
  dataFormat: DataFormat = .nhwc
) -> Tensor<T> {
  return #tfop("BiasAddGrad",
    outBackprop,
    T: T.self,
    data_format: dataFormat.cName)
}

@inlinable @inline(__always)
public static func biasAddV1<T: Numeric>(
  value: Tensor<T>,
  bias: Tensor<T>
) -> Tensor<T> {
  return #tfop("BiasAddV1",
    value,
    bias,
    T: T.self)
}

@inlinable @inline(__always)
public static func binary<T: AccelerableByTensorFlow>(
  _ a: Tensor<T>,
  _ b: Tensor<T>
) -> Tensor<T> {
  return #tfop("Binary",
    a,
    b,
    T: T.self)
}

@inlinable @inline(__always)
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

@inlinable @inline(__always)
public static func bitcast<T: Numeric, Type: Numeric>(
  _ input: Tensor<T>
) -> Tensor<Type> {
  return #tfop("Bitcast",
    input,
    T: T.self,
    type: Type.self)
}

@inlinable @inline(__always)
public static func bitwiseAnd<T: BinaryInteger>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  return #tfop("BitwiseAnd",
    x,
    y,
    T: T.self)
}

@inlinable @inline(__always)
public static func bitwiseOr<T: BinaryInteger>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  return #tfop("BitwiseOr",
    x,
    y,
    T: T.self)
}

@inlinable @inline(__always)
public static func bitwiseXor<T: BinaryInteger>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  return #tfop("BitwiseXor",
    x,
    y,
    T: T.self)
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
public static func blockLSTM<T: BinaryFloatingPoint>(
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
public static func blockLSTMGrad<T: BinaryFloatingPoint>(
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

@inlinable @inline(__always)
public static func boostedTreesCalculateBestGainsPerFeature(
  nodeIdRange: Tensor<Int32>,
  statsSummaryList: [Tensor<Float>],
  l1: Tensor<Float>,
  l2: Tensor<Float>,
  treeComplexity: Tensor<Float>,
  minNodeWeight: Tensor<Float>,
  maxSplits: Int64
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

@inlinable @inline(__always)
public static func boostedTreesMakeStatsSummary(
  nodeIds: Tensor<Int32>,
  gradients: Tensor<Float>,
  hessians: Tensor<Float>,
  bucketizedFeaturesList: [Tensor<Int32>],
  maxSplits: Int64,
  numBuckets: Int64
) -> Tensor<Float> {
  return #tfop("BoostedTreesMakeStatsSummary",
    nodeIds,
    gradients,
    hessians,
    bucketizedFeaturesList,
    max_splits: maxSplits,
    num_buckets: numBuckets)
}

@inlinable @inline(__always)
public static func broadcastArgs<T: BinaryInteger>(
  s0: Tensor<T>,
  s1: Tensor<T>
) -> Tensor<T> {
  return #tfop("BroadcastArgs",
    s0,
    s1,
    T: T.self)
}

@inlinable @inline(__always)
public static func broadcastGradientArgs<T: BinaryInteger>(
  s0: Tensor<T>,
  s1: Tensor<T>
) -> (Tensor<T>, Tensor<T>) {
  return #tfop("BroadcastGradientArgs",
    s0,
    s1,
    T: T.self)
}

@inlinable @inline(__always)
public static func broadcastTo<T: AccelerableByTensorFlow, Tidx: BinaryInteger>(
  _ input: Tensor<T>,
  shape: Tensor<Tidx>
) -> Tensor<T> {
  return #tfop("BroadcastTo",
    input,
    shape,
    T: T.self,
    Tidx: Tidx.self)
}

@inlinable @inline(__always)
public static func bucketize<T: Numeric>(
  _ input: Tensor<T>,
  boundaries: [Double]
) -> Tensor<Int32> {
  return #tfop("Bucketize",
    input,
    T: T.self,
    boundaries: boundaries)
}

@inlinable @inline(__always)
public static func cTCBeamSearchDecoder(
  inputs: Tensor<Float>,
  sequenceLength: Tensor<Int32>,
  beamWidth: Int64,
  topPaths: Int64,
  mergeRepeated: Bool = true
) -> ([Tensor<Int64>], [Tensor<Int64>], [Tensor<Int64>], Tensor<Float>) {
  return #tfop("CTCBeamSearchDecoder",
    inputs,
    sequenceLength,
    beam_width: beamWidth,
    top_paths: topPaths,
    merge_repeated: mergeRepeated)
}

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
public static func cast<Srct: AccelerableByTensorFlow, Dstt: AccelerableByTensorFlow>(
  _ x: Tensor<Srct>
) -> Tensor<Dstt> {
  return #tfop("Cast",
    x,
    SrcT: Srct.self,
    DstT: Dstt.self)
}

@inlinable @inline(__always)
public static func ceil<T: BinaryFloatingPoint>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Ceil",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func checkNumerics<T: BinaryFloatingPoint>(
  _ tensor: Tensor<T>,
  message: String
) -> Tensor<T> {
  return #tfop("CheckNumerics",
    tensor,
    T: T.self,
    message: message)
}

@inlinable @inline(__always)
public static func cholesky<T: BinaryFloatingPoint>(
  _ input: Tensor<T>
) -> Tensor<T> {
  return #tfop("Cholesky",
    input,
    T: T.self)
}

@inlinable @inline(__always)
public static func choleskyGrad<T: BinaryFloatingPoint>(
  l: Tensor<T>,
  grad: Tensor<T>
) -> Tensor<T> {
  return #tfop("CholeskyGrad",
    l,
    grad,
    T: T.self)
}

@inlinable @inline(__always)
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

@inlinable @inline(__always)
public static func collectiveReduce<T: Numeric>(
  _ input: Tensor<T>,
  groupSize: Int64,
  groupKey: Int64,
  instanceKey: Int64,
  mergeOp: MergeOp,
  finalOp: FinalOp,
  subdivOffsets: [Int32]
) -> Tensor<T> {
  return #tfop("CollectiveReduce",
    input,
    T: T.self,
    group_size: groupSize,
    group_key: groupKey,
    instance_key: instanceKey,
    merge_op: mergeOp.cName,
    final_op: finalOp.cName,
    subdiv_offsets: subdivOffsets)
}

@inlinable @inline(__always)
public static func compareAndBitpack<T: AccelerableByTensorFlow>(
  _ input: Tensor<T>,
  threshold: Tensor<T>
) -> Tensor<UInt8> {
  return #tfop("CompareAndBitpack",
    input,
    threshold,
    T: T.self)
}

@inlinable @inline(__always)
public static func complex<T: BinaryFloatingPoint, Tout: AccelerableByTensorFlow>(
  real: Tensor<T>,
  imag: Tensor<T>
) -> Tensor<Tout> {
  return #tfop("Complex",
    real,
    imag,
    T: T.self,
    Tout: Tout.self)
}

@inlinable @inline(__always)
public static func complexAbs<T: AccelerableByTensorFlow, Tout: BinaryFloatingPoint>(
  _ x: Tensor<T>
) -> Tensor<Tout> {
  return #tfop("ComplexAbs",
    x,
    T: T.self,
    Tout: Tout.self)
}

@inlinable @inline(__always)
public static func complexStruct<T_c: AccelerableByTensorFlow>(
  nA: Int64,
  nB: Int64
) -> ([Tensor<Int32>], [Tensor<Int64>], [Tensor<T_c>]) {
  return #tfop("ComplexStruct",
    n_a: nA,
    n_b: nB)
}

@inlinable @inline(__always)
public static func computeAccidentalHits(
  trueClasses: Tensor<Int64>,
  sampledCandidates: Tensor<Int64>,
  numTrue: Int64,
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> (Tensor<Int32>, Tensor<Int64>, Tensor<Float>) {
  return #tfop("ComputeAccidentalHits",
    trueClasses,
    sampledCandidates,
    num_true: numTrue,
    seed: seed,
    seed2: seed2)
}

@inlinable @inline(__always)
public static func concat<T: AccelerableByTensorFlow>(
  concatDim: Tensor<Int32>,
  _ values: [Tensor<T>]
) -> Tensor<T> {
  return #tfop("Concat",
    concatDim,
    values,
    T: T.self)
}

@inlinable @inline(__always)
public static func concatOffset(
  concatDim: Tensor<Int32>,
  shape: [Tensor<Int32>]
) -> [Tensor<Int32>] {
  return #tfop("ConcatOffset",
    concatDim,
    shape)
}

@inlinable @inline(__always)
public static func concatV2<T: AccelerableByTensorFlow, Tidx: BinaryInteger>(
  _ values: [Tensor<T>],
  axis: Tensor<Tidx>
) -> Tensor<T> {
  return #tfop("ConcatV2",
    values,
    axis,
    T: T.self,
    Tidx: Tidx.self)
}

@inlinable @inline(__always)
public static func conj<T: AccelerableByTensorFlow>(
  _ input: Tensor<T>
) -> Tensor<T> {
  return #tfop("Conj",
    input,
    T: T.self)
}

@inlinable @inline(__always)
public static func conjugateTranspose<T: AccelerableByTensorFlow, Tperm: BinaryInteger>(
  _ x: Tensor<T>,
  perm: Tensor<Tperm>
) -> Tensor<T> {
  return #tfop("ConjugateTranspose",
    x,
    perm,
    T: T.self,
    Tperm: Tperm.self)
}

@inlinable @inline(__always)
public static func constructionFails(
) {
  return #tfop("ConstructionFails")
}

@inlinable @inline(__always)
public static func controlTrigger(
) {
  return #tfop("ControlTrigger")
}

@inlinable @inline(__always)
public static func conv2D<T: BinaryFloatingPoint>(
  _ input: Tensor<T>,
  filter: Tensor<T>,
  strides: [Int32],
  useCudnnOnGpu: Bool = true,
  padding: Padding,
  dataFormat: DataFormat = .nhwc,
  dilations: [Int32] = [1, 1, 1, 1]
) -> Tensor<T> {
  return #tfop("Conv2D",
    input,
    filter,
    T: T.self,
    strides: strides,
    use_cudnn_on_gpu: useCudnnOnGpu,
    padding: padding.cName,
    data_format: dataFormat.cName,
    dilations: dilations)
}

@inlinable @inline(__always)
public static func conv2DBackpropFilter<T: BinaryFloatingPoint>(
  _ input: Tensor<T>,
  filterSizes: Tensor<Int32>,
  outBackprop: Tensor<T>,
  strides: [Int32],
  useCudnnOnGpu: Bool = true,
  padding: Padding,
  dataFormat: DataFormat = .nhwc,
  dilations: [Int32] = [1, 1, 1, 1]
) -> Tensor<T> {
  return #tfop("Conv2DBackpropFilter",
    input,
    filterSizes,
    outBackprop,
    T: T.self,
    strides: strides,
    use_cudnn_on_gpu: useCudnnOnGpu,
    padding: padding.cName,
    data_format: dataFormat.cName,
    dilations: dilations)
}

@inlinable @inline(__always)
public static func conv2DBackpropInput<T: BinaryFloatingPoint>(
  inputSizes: Tensor<Int32>,
  filter: Tensor<T>,
  outBackprop: Tensor<T>,
  strides: [Int32],
  useCudnnOnGpu: Bool = true,
  padding: Padding,
  dataFormat: DataFormat = .nhwc,
  dilations: [Int32] = [1, 1, 1, 1]
) -> Tensor<T> {
  return #tfop("Conv2DBackpropInput",
    inputSizes,
    filter,
    outBackprop,
    T: T.self,
    strides: strides,
    use_cudnn_on_gpu: useCudnnOnGpu,
    padding: padding.cName,
    data_format: dataFormat.cName,
    dilations: dilations)
}

@inlinable @inline(__always)
public static func conv3D<T: BinaryFloatingPoint>(
  _ input: Tensor<T>,
  filter: Tensor<T>,
  strides: [Int32],
  padding: Padding,
  dataFormat: DataFormat1 = .ndhwc,
  dilations: [Int32] = [1, 1, 1, 1, 1]
) -> Tensor<T> {
  return #tfop("Conv3D",
    input,
    filter,
    T: T.self,
    strides: strides,
    padding: padding.cName,
    data_format: dataFormat.cName,
    dilations: dilations)
}

@inlinable @inline(__always)
public static func conv3DBackpropFilter<T: BinaryFloatingPoint>(
  _ input: Tensor<T>,
  filter: Tensor<T>,
  outBackprop: Tensor<T>,
  strides: [Int32],
  padding: Padding,
  dilations: [Int32] = [1, 1, 1, 1, 1]
) -> Tensor<T> {
  return #tfop("Conv3DBackpropFilter",
    input,
    filter,
    outBackprop,
    T: T.self,
    strides: strides,
    padding: padding.cName,
    dilations: dilations)
}

@inlinable @inline(__always)
public static func conv3DBackpropFilterV2<T: BinaryFloatingPoint>(
  _ input: Tensor<T>,
  filterSizes: Tensor<Int32>,
  outBackprop: Tensor<T>,
  strides: [Int32],
  padding: Padding,
  dataFormat: DataFormat1 = .ndhwc,
  dilations: [Int32] = [1, 1, 1, 1, 1]
) -> Tensor<T> {
  return #tfop("Conv3DBackpropFilterV2",
    input,
    filterSizes,
    outBackprop,
    T: T.self,
    strides: strides,
    padding: padding.cName,
    data_format: dataFormat.cName,
    dilations: dilations)
}

@inlinable @inline(__always)
public static func conv3DBackpropInput<T: BinaryFloatingPoint>(
  _ input: Tensor<T>,
  filter: Tensor<T>,
  outBackprop: Tensor<T>,
  strides: [Int32],
  padding: Padding,
  dilations: [Int32] = [1, 1, 1, 1, 1]
) -> Tensor<T> {
  return #tfop("Conv3DBackpropInput",
    input,
    filter,
    outBackprop,
    T: T.self,
    strides: strides,
    padding: padding.cName,
    dilations: dilations)
}

@inlinable @inline(__always)
public static func conv3DBackpropInputV2<T: BinaryFloatingPoint, Tshape: BinaryInteger>(
  inputSizes: Tensor<Tshape>,
  filter: Tensor<T>,
  outBackprop: Tensor<T>,
  strides: [Int32],
  padding: Padding,
  dataFormat: DataFormat1 = .ndhwc,
  dilations: [Int32] = [1, 1, 1, 1, 1]
) -> Tensor<T> {
  return #tfop("Conv3DBackpropInputV2",
    inputSizes,
    filter,
    outBackprop,
    T: T.self,
    Tshape: Tshape.self,
    strides: strides,
    padding: padding.cName,
    data_format: dataFormat.cName,
    dilations: dilations)
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
public static func copy<T: AccelerableByTensorFlow>(
  _ input: Tensor<T>,
  tensorName: String,
  debugOpsSpec: [String]
) -> Tensor<T> {
  return #tfop("Copy",
    input,
    T: T.self,
    tensor_name: tensorName,
    debug_ops_spec: debugOpsSpec)
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
public static func copyHost<T: AccelerableByTensorFlow>(
  _ input: Tensor<T>,
  tensorName: String,
  debugOpsSpec: [String]
) -> Tensor<T> {
  return #tfop("CopyHost",
    input,
    T: T.self,
    tensor_name: tensorName,
    debug_ops_spec: debugOpsSpec)
}

@inlinable @inline(__always)
public static func copyOp<T: AccelerableByTensorFlow>(
  _ a: Tensor<T>
) -> Tensor<T> {
  return #tfop("CopyOp",
    a,
    T: T.self)
}

@inlinable @inline(__always)
public static func cos<T: BinaryFloatingPoint>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Cos",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func cosh<T: BinaryFloatingPoint>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Cosh",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func countUpTo<T: BinaryInteger>(
  ref: Tensor<T>,
  limit: Int64
) -> Tensor<T> {
  return #tfop("CountUpTo",
    ref,
    T: T.self,
    limit: limit)
}

@inlinable @inline(__always)
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
    method: method.cName,
    extrapolation_value: extrapolationValue)
}

@inlinable @inline(__always)
public static func cropAndResizeGradBoxes<T: Numeric>(
  grads: Tensor<Float>,
  image: Tensor<T>,
  boxes: Tensor<Float>,
  boxInd: Tensor<Int32>,
  method: Method2 = .bilinear
) -> Tensor<Float> {
  return #tfop("CropAndResizeGradBoxes",
    grads,
    image,
    boxes,
    boxInd,
    T: T.self,
    method: method.cName)
}

@inlinable @inline(__always)
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
    method: method.cName)
}

@inlinable @inline(__always)
public static func cross<T: Numeric>(
  _ a: Tensor<T>,
  _ b: Tensor<T>
) -> Tensor<T> {
  return #tfop("Cross",
    a,
    b,
    T: T.self)
}

@inlinable @inline(__always)
public static func cudnnRNN<T: BinaryFloatingPoint>(
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
) -> (Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>) {
  return #tfop("CudnnRNN",
    input,
    inputH,
    inputC,
    params,
    T: T.self,
    rnn_mode: rnnMode.cName,
    input_mode: inputMode.cName,
    direction: direction.cName,
    dropout: dropout,
    seed: seed,
    seed2: seed2,
    is_training: isTraining)
}

@inlinable @inline(__always)
public static func cudnnRNNBackprop<T: BinaryFloatingPoint>(
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
    rnn_mode: rnnMode.cName,
    input_mode: inputMode.cName,
    direction: direction.cName,
    dropout: dropout,
    seed: seed,
    seed2: seed2)
}

@inlinable @inline(__always)
public static func cudnnRNNBackpropV2<T: BinaryFloatingPoint>(
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
) -> (Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>) {
  return #tfop("CudnnRNNBackpropV2",
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
    T: T.self,
    rnn_mode: rnnMode.cName,
    input_mode: inputMode.cName,
    direction: direction.cName,
    dropout: dropout,
    seed: seed,
    seed2: seed2)
}

@inlinable @inline(__always)
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
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> Tensor<T> {
  return #tfop("CudnnRNNCanonicalToParams",
    numLayers,
    numUnits,
    inputSize,
    weights,
    biases,
    T: T.self,
    rnn_mode: rnnMode.cName,
    input_mode: inputMode.cName,
    direction: direction.cName,
    dropout: dropout,
    seed: seed,
    seed2: seed2)
}

@inlinable @inline(__always)
public static func cudnnRNNParamsSize<T: BinaryFloatingPoint, S: BinaryInteger>(
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
  return #tfop("CudnnRNNParamsSize",
    numLayers,
    numUnits,
    inputSize,
    T: T.self,
    S: S.self,
    rnn_mode: rnnMode.cName,
    input_mode: inputMode.cName,
    direction: direction.cName,
    dropout: dropout,
    seed: seed,
    seed2: seed2)
}

@inlinable @inline(__always)
public static func cudnnRNNParamsToCanonical<T: BinaryFloatingPoint>(
  numLayers: Tensor<Int32>,
  numUnits: Tensor<Int32>,
  inputSize: Tensor<Int32>,
  params: Tensor<T>,
  numParams: Int64,
  rnnMode: RnnMode = .lstm,
  inputMode: InputMode = .linearInput,
  direction: Direction = .unidirectional,
  dropout: Double = 0,
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> ([Tensor<T>], [Tensor<T>]) {
  return #tfop("CudnnRNNParamsToCanonical",
    numLayers,
    numUnits,
    inputSize,
    params,
    T: T.self,
    num_params: numParams,
    rnn_mode: rnnMode.cName,
    input_mode: inputMode.cName,
    direction: direction.cName,
    dropout: dropout,
    seed: seed,
    seed2: seed2)
}

@inlinable @inline(__always)
public static func cudnnRNNV2<T: BinaryFloatingPoint>(
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
) -> (Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<Int8>) {
  return #tfop("CudnnRNNV2",
    input,
    inputH,
    inputC,
    params,
    T: T.self,
    rnn_mode: rnnMode.cName,
    input_mode: inputMode.cName,
    direction: direction.cName,
    dropout: dropout,
    seed: seed,
    seed2: seed2,
    is_training: isTraining)
}

@inlinable @inline(__always)
public static func cumprod<T: Numeric, Tidx: BinaryInteger>(
  _ x: Tensor<T>,
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

@inlinable @inline(__always)
public static func cumsum<T: Numeric, Tidx: BinaryInteger>(
  _ x: Tensor<T>,
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

@inlinable @inline(__always)
public static func dataFormatDimMap<T: BinaryInteger>(
  _ x: Tensor<T>,
  srcFormat: String = "NHWC",
  dstFormat: String = "NCHW"
) -> Tensor<T> {
  return #tfop("DataFormatDimMap",
    x,
    T: T.self,
    src_format: srcFormat,
    dst_format: dstFormat)
}

@inlinable @inline(__always)
public static func dataFormatVecPermute<T: BinaryInteger>(
  _ x: Tensor<T>,
  srcFormat: String = "NHWC",
  dstFormat: String = "NCHW"
) -> Tensor<T> {
  return #tfop("DataFormatVecPermute",
    x,
    T: T.self,
    src_format: srcFormat,
    dst_format: dstFormat)
}

@inlinable @inline(__always)
public static func debugGradientIdentity<T: AccelerableByTensorFlow>(
  _ input: Tensor<T>
) -> Tensor<T> {
  return #tfop("DebugGradientIdentity",
    input,
    T: T.self)
}

@inlinable @inline(__always)
public static func debugGradientRefIdentity<T: AccelerableByTensorFlow>(
  _ input: Tensor<T>
) -> Tensor<T> {
  return #tfop("DebugGradientRefIdentity",
    input,
    T: T.self)
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
public static func debugIdentity<T: AccelerableByTensorFlow>(
  _ input: Tensor<T>,
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
public static func debugNanCount<T: AccelerableByTensorFlow>(
  _ input: Tensor<T>,
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
public static func debugNumericSummary<T: AccelerableByTensorFlow>(
  _ input: Tensor<T>,
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

@inlinable @inline(__always)
public static func deepCopy<T: AccelerableByTensorFlow>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("DeepCopy",
    x,
    T: T.self)
}

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
public static func depthToSpace<T: AccelerableByTensorFlow>(
  _ input: Tensor<T>,
  blockSize: Int64,
  dataFormat: DataFormat3 = .nhwc
) -> Tensor<T> {
  return #tfop("DepthToSpace",
    input,
    T: T.self,
    block_size: blockSize,
    data_format: dataFormat.cName)
}

@inlinable @inline(__always)
public static func depthwiseConv2dNative<T: BinaryFloatingPoint>(
  _ input: Tensor<T>,
  filter: Tensor<T>,
  strides: [Int32],
  padding: Padding,
  dataFormat: DataFormat = .nhwc,
  dilations: [Int32] = [1, 1, 1, 1]
) -> Tensor<T> {
  return #tfop("DepthwiseConv2dNative",
    input,
    filter,
    T: T.self,
    strides: strides,
    padding: padding.cName,
    data_format: dataFormat.cName,
    dilations: dilations)
}

@inlinable @inline(__always)
public static func depthwiseConv2dNativeBackpropFilter<T: BinaryFloatingPoint>(
  _ input: Tensor<T>,
  filterSizes: Tensor<Int32>,
  outBackprop: Tensor<T>,
  strides: [Int32],
  padding: Padding,
  dataFormat: DataFormat = .nhwc,
  dilations: [Int32] = [1, 1, 1, 1]
) -> Tensor<T> {
  return #tfop("DepthwiseConv2dNativeBackpropFilter",
    input,
    filterSizes,
    outBackprop,
    T: T.self,
    strides: strides,
    padding: padding.cName,
    data_format: dataFormat.cName,
    dilations: dilations)
}

@inlinable @inline(__always)
public static func depthwiseConv2dNativeBackpropInput<T: BinaryFloatingPoint>(
  inputSizes: Tensor<Int32>,
  filter: Tensor<T>,
  outBackprop: Tensor<T>,
  strides: [Int32],
  padding: Padding,
  dataFormat: DataFormat = .nhwc,
  dilations: [Int32] = [1, 1, 1, 1]
) -> Tensor<T> {
  return #tfop("DepthwiseConv2dNativeBackpropInput",
    inputSizes,
    filter,
    outBackprop,
    T: T.self,
    strides: strides,
    padding: padding.cName,
    data_format: dataFormat.cName,
    dilations: dilations)
}

@inlinable @inline(__always)
public static func dequantize<T: AccelerableByTensorFlow>(
  _ input: Tensor<T>,
  minRange: Tensor<Float>,
  maxRange: Tensor<Float>,
  mode: Mode = .minCombined
) -> Tensor<Float> {
  return #tfop("Dequantize",
    input,
    minRange,
    maxRange,
    T: T.self,
    mode: mode.cName)
}

@inlinable @inline(__always)
public static func deserializeSparse<Dtype: AccelerableByTensorFlow, Tserialized: AccelerableByTensorFlow>(
  serializedSparse: Tensor<Tserialized>
) -> (Tensor<Int64>, Tensor<Dtype>, Tensor<Int64>) {
  return #tfop("DeserializeSparse",
    serializedSparse,
    dtype: Dtype.self,
    Tserialized: Tserialized.self)
}

@inlinable @inline(__always)
public static func destroyTemporaryVariable<T: AccelerableByTensorFlow>(
  ref: Tensor<T>,
  varName: String
) -> Tensor<T> {
  return #tfop("DestroyTemporaryVariable",
    ref,
    T: T.self,
    var_name: varName)
}

@inlinable @inline(__always)
public static func diag<T: Numeric>(
  diagonal: Tensor<T>
) -> Tensor<T> {
  return #tfop("Diag",
    diagonal,
    T: T.self)
}

@inlinable @inline(__always)
public static func diagPart<T: Numeric>(
  _ input: Tensor<T>
) -> Tensor<T> {
  return #tfop("DiagPart",
    input,
    T: T.self)
}

@inlinable @inline(__always)
public static func digamma<T: BinaryFloatingPoint>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Digamma",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func dilation2D<T: Numeric>(
  _ input: Tensor<T>,
  filter: Tensor<T>,
  strides: [Int32],
  rates: [Int32],
  padding: Padding
) -> Tensor<T> {
  return #tfop("Dilation2D",
    input,
    filter,
    T: T.self,
    strides: strides,
    rates: rates,
    padding: padding.cName)
}

@inlinable @inline(__always)
public static func dilation2DBackpropFilter<T: Numeric>(
  _ input: Tensor<T>,
  filter: Tensor<T>,
  outBackprop: Tensor<T>,
  strides: [Int32],
  rates: [Int32],
  padding: Padding
) -> Tensor<T> {
  return #tfop("Dilation2DBackpropFilter",
    input,
    filter,
    outBackprop,
    T: T.self,
    strides: strides,
    rates: rates,
    padding: padding.cName)
}

@inlinable @inline(__always)
public static func dilation2DBackpropInput<T: Numeric>(
  _ input: Tensor<T>,
  filter: Tensor<T>,
  outBackprop: Tensor<T>,
  strides: [Int32],
  rates: [Int32],
  padding: Padding
) -> Tensor<T> {
  return #tfop("Dilation2DBackpropInput",
    input,
    filter,
    outBackprop,
    T: T.self,
    strides: strides,
    rates: rates,
    padding: padding.cName)
}

@inlinable @inline(__always)
public static func div<T: Numeric>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  return #tfop("Div",
    x,
    y,
    T: T.self)
}

@inlinable @inline(__always)
public static func drawBoundingBoxes<T: BinaryFloatingPoint>(
  images: Tensor<T>,
  boxes: Tensor<Float>
) -> Tensor<T> {
  return #tfop("DrawBoundingBoxes",
    images,
    boxes,
    T: T.self)
}

@inlinable @inline(__always)
public static func dynamicPartition<T: AccelerableByTensorFlow>(
  data: Tensor<T>,
  partitions: Tensor<Int32>,
  numPartitions: Int64
) -> [Tensor<T>] {
  return #tfop("DynamicPartition",
    data,
    partitions,
    T: T.self,
    num_partitions: numPartitions)
}

@inlinable @inline(__always)
public static func dynamicStitch<T: AccelerableByTensorFlow>(
  indices: [Tensor<Int32>],
  data: [Tensor<T>]
) -> Tensor<T> {
  return #tfop("DynamicStitch",
    indices,
    data,
    T: T.self)
}

@inlinable @inline(__always)
public static func eagerPyFunc<Tin: AccelerableByTensorFlow, Tout: AccelerableByTensorFlow>(
  _ input: [Tensor<Tin>],
  token: String
) -> [Tensor<Tout>] {
  return #tfop("EagerPyFunc",
    input,
    token: token)
}

@inlinable @inline(__always)
public static func editDistance<T: AccelerableByTensorFlow>(
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

@inlinable @inline(__always)
public static func elu<T: BinaryFloatingPoint>(
  features: Tensor<T>
) -> Tensor<T> {
  return #tfop("Elu",
    features,
    T: T.self)
}

@inlinable @inline(__always)
public static func eluGrad<T: BinaryFloatingPoint>(
  gradients: Tensor<T>,
  outputs: Tensor<T>
) -> Tensor<T> {
  return #tfop("EluGrad",
    gradients,
    outputs,
    T: T.self)
}

@inlinable @inline(__always)
public static func empty<Dtype: AccelerableByTensorFlow>(
  shape: Tensor<Int32>,
  init_: Bool = false
) -> Tensor<Dtype> {
  return #tfop("Empty",
    shape,
    dtype: Dtype.self,
    init: init_)
}

@inlinable @inline(__always)
public static func enter<T: AccelerableByTensorFlow>(
  data: Tensor<T>,
  frameName: String,
  isConstant: Bool = false,
  parallelIterations: Int64 = 10
) -> Tensor<T> {
  return #tfop("Enter",
    data,
    T: T.self,
    frame_name: frameName,
    is_constant: isConstant,
    parallel_iterations: parallelIterations)
}

@inlinable @inline(__always)
public static func equal<T: AccelerableByTensorFlow>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<Bool> {
  return #tfop("Equal",
    x,
    y,
    T: T.self)
}

@inlinable @inline(__always)
public static func erf<T: BinaryFloatingPoint>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Erf",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func erfc<T: BinaryFloatingPoint>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Erfc",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func exit<T: AccelerableByTensorFlow>(
  data: Tensor<T>
) -> Tensor<T> {
  return #tfop("Exit",
    data,
    T: T.self)
}

@inlinable @inline(__always)
public static func exp<T: BinaryFloatingPoint>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Exp",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func expandDims<T: AccelerableByTensorFlow, Tdim: BinaryInteger>(
  _ input: Tensor<T>,
  dim: Tensor<Tdim>
) -> Tensor<T> {
  return #tfop("ExpandDims",
    input,
    dim,
    T: T.self,
    Tdim: Tdim.self)
}

@inlinable @inline(__always)
public static func expm1<T: BinaryFloatingPoint>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Expm1",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func extractGlimpse(
  _ input: Tensor<Float>,
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

@inlinable @inline(__always)
public static func extractImagePatches<T: Numeric>(
  images: Tensor<T>,
  ksizes: [Int32],
  strides: [Int32],
  rates: [Int32],
  padding: Padding
) -> Tensor<T> {
  return #tfop("ExtractImagePatches",
    images,
    T: T.self,
    ksizes: ksizes,
    strides: strides,
    rates: rates,
    padding: padding.cName)
}

@inlinable @inline(__always)
public static func fFT<Tcomplex: AccelerableByTensorFlow>(
  _ input: Tensor<Tcomplex>
) -> Tensor<Tcomplex> {
  return #tfop("FFT",
    input,
    Tcomplex: Tcomplex.self)
}

@inlinable @inline(__always)
public static func fFT2D<Tcomplex: AccelerableByTensorFlow>(
  _ input: Tensor<Tcomplex>
) -> Tensor<Tcomplex> {
  return #tfop("FFT2D",
    input,
    Tcomplex: Tcomplex.self)
}

@inlinable @inline(__always)
public static func fFT3D<Tcomplex: AccelerableByTensorFlow>(
  _ input: Tensor<Tcomplex>
) -> Tensor<Tcomplex> {
  return #tfop("FFT3D",
    input,
    Tcomplex: Tcomplex.self)
}

@inlinable @inline(__always)
public static func fakeQuantWithMinMaxArgs(
  inputs: Tensor<Float>,
  min: Double = -6,
  max: Double = 6,
  numBits: Int64 = 8,
  narrowRange: Bool = false
) -> Tensor<Float> {
  return #tfop("FakeQuantWithMinMaxArgs",
    inputs,
    min: min,
    max: max,
    num_bits: numBits,
    narrow_range: narrowRange)
}

@inlinable @inline(__always)
public static func fakeQuantWithMinMaxArgsGradient(
  gradients: Tensor<Float>,
  inputs: Tensor<Float>,
  min: Double = -6,
  max: Double = 6,
  numBits: Int64 = 8,
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

@inlinable @inline(__always)
public static func fakeQuantWithMinMaxVars(
  inputs: Tensor<Float>,
  min: Tensor<Float>,
  max: Tensor<Float>,
  numBits: Int64 = 8,
  narrowRange: Bool = false
) -> Tensor<Float> {
  return #tfop("FakeQuantWithMinMaxVars",
    inputs,
    min,
    max,
    num_bits: numBits,
    narrow_range: narrowRange)
}

@inlinable @inline(__always)
public static func fakeQuantWithMinMaxVarsGradient(
  gradients: Tensor<Float>,
  inputs: Tensor<Float>,
  min: Tensor<Float>,
  max: Tensor<Float>,
  numBits: Int64 = 8,
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

@inlinable @inline(__always)
public static func fakeQuantWithMinMaxVarsPerChannel(
  inputs: Tensor<Float>,
  min: Tensor<Float>,
  max: Tensor<Float>,
  numBits: Int64 = 8,
  narrowRange: Bool = false
) -> Tensor<Float> {
  return #tfop("FakeQuantWithMinMaxVarsPerChannel",
    inputs,
    min,
    max,
    num_bits: numBits,
    narrow_range: narrowRange)
}

@inlinable @inline(__always)
public static func fakeQuantWithMinMaxVarsPerChannelGradient(
  gradients: Tensor<Float>,
  inputs: Tensor<Float>,
  min: Tensor<Float>,
  max: Tensor<Float>,
  numBits: Int64 = 8,
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

@inlinable @inline(__always)
public static func fill<T: AccelerableByTensorFlow, Index_type: BinaryInteger>(
  dims: Tensor<Index_type>,
  value: Tensor<T>
) -> Tensor<T> {
  return #tfop("Fill",
    dims,
    value,
    T: T.self,
    index_type: Index_type.self)
}

@inlinable @inline(__always)
public static func fiveFloatOutputs(
) -> (Tensor<Float>, Tensor<Float>, Tensor<Float>, Tensor<Float>, Tensor<Float>) {
  return #tfop("FiveFloatOutputs")
}

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
  return #tfop("FloatOutput")
}

@inlinable @inline(__always)
public static func floor<T: BinaryFloatingPoint>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Floor",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func floorDiv<T: Numeric>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  return #tfop("FloorDiv",
    x,
    y,
    T: T.self)
}

@inlinable @inline(__always)
public static func floorMod<T: Numeric>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  return #tfop("FloorMod",
    x,
    y,
    T: T.self)
}

@inlinable @inline(__always)
public static func foo1(
  _ a: Tensor<Float>,
  _ b: Tensor<Int32>,
  c: Tensor<Int32>
) -> (Tensor<Float>, Tensor<Int32>) {
  return #tfop("Foo1",
    a,
    b,
    c)
}

@inlinable @inline(__always)
public static func fractionalAvgPool<T: Numeric>(
  value: Tensor<T>,
  poolingRatio: [Double],
  pseudoRandom: Bool = false,
  overlapping: Bool = false,
  deterministic: Bool = false,
  seed: Int64 = 0,
  seed2: Int64 = 0
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
public static func fractionalMaxPool<T: Numeric>(
  value: Tensor<T>,
  poolingRatio: [Double],
  pseudoRandom: Bool = false,
  overlapping: Bool = false,
  deterministic: Bool = false,
  seed: Int64 = 0,
  seed2: Int64 = 0
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
public static func fusedBatchNorm<T: BinaryFloatingPoint>(
  _ x: Tensor<T>,
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

@inlinable @inline(__always)
public static func fusedBatchNormGrad<T: BinaryFloatingPoint>(
  yBackprop: Tensor<T>,
  _ x: Tensor<T>,
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

@inlinable @inline(__always)
public static func fusedBatchNormGradV2<T: BinaryFloatingPoint, U: BinaryFloatingPoint>(
  yBackprop: Tensor<T>,
  _ x: Tensor<T>,
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

@inlinable @inline(__always)
public static func fusedBatchNormV2<T: BinaryFloatingPoint, U: BinaryFloatingPoint>(
  _ x: Tensor<T>,
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

@inlinable @inline(__always)
public static func fusedPadConv2D<T: BinaryFloatingPoint>(
  _ input: Tensor<T>,
  paddings: Tensor<Int32>,
  filter: Tensor<T>,
  mode: Mode4,
  strides: [Int32],
  padding: Padding
) -> Tensor<T> {
  return #tfop("FusedPadConv2D",
    input,
    paddings,
    filter,
    T: T.self,
    mode: mode.cName,
    strides: strides,
    padding: padding.cName)
}

@inlinable @inline(__always)
public static func fusedResizeAndPadConv2D<T: BinaryFloatingPoint>(
  _ input: Tensor<T>,
  size: Tensor<Int32>,
  paddings: Tensor<Int32>,
  filter: Tensor<T>,
  resizeAlignCorners: Bool = false,
  mode: Mode4,
  strides: [Int32],
  padding: Padding
) -> Tensor<T> {
  return #tfop("FusedResizeAndPadConv2D",
    input,
    size,
    paddings,
    filter,
    T: T.self,
    resize_align_corners: resizeAlignCorners,
    mode: mode.cName,
    strides: strides,
    padding: padding.cName)
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
public static func gRUBlockCell<T: BinaryFloatingPoint>(
  _ x: Tensor<T>,
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
public static func gRUBlockCellGrad<T: BinaryFloatingPoint>(
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

@inlinable @inline(__always)
public static func gather<Tparams: AccelerableByTensorFlow, Tindices: BinaryInteger>(
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

@inlinable @inline(__always)
public static func gatherNd<Tparams: AccelerableByTensorFlow, Tindices: BinaryInteger>(
  params: Tensor<Tparams>,
  indices: Tensor<Tindices>
) -> Tensor<Tparams> {
  return #tfop("GatherNd",
    params,
    indices,
    Tparams: Tparams.self,
    Tindices: Tindices.self)
}

@inlinable @inline(__always)
public static func gatherV2<Tparams: AccelerableByTensorFlow, Tindices: BinaryInteger, Taxis: BinaryInteger>(
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

@inlinable @inline(__always)
public static func graphDefVersion(
) -> Tensor<Int32> {
  return #tfop("GraphDefVersion")
}

@inlinable @inline(__always)
public static func greater<T: Numeric>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<Bool> {
  return #tfop("Greater",
    x,
    y,
    T: T.self)
}

@inlinable @inline(__always)
public static func greaterEqual<T: Numeric>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<Bool> {
  return #tfop("GreaterEqual",
    x,
    y,
    T: T.self)
}

@inlinable @inline(__always)
public static func guaranteeConst<T: AccelerableByTensorFlow>(
  _ input: Tensor<T>
) -> Tensor<T> {
  return #tfop("GuaranteeConst",
    input,
    T: T.self)
}

@inlinable @inline(__always)
public static func hSVToRGB<T: BinaryFloatingPoint>(
  images: Tensor<T>
) -> Tensor<T> {
  return #tfop("HSVToRGB",
    images,
    T: T.self)
}

@inlinable @inline(__always)
public static func histogramFixedWidth<T: Numeric, Dtype: BinaryInteger>(
  _ values: Tensor<T>,
  valueRange: Tensor<T>,
  nbins: Tensor<Int32>
) -> Tensor<Dtype> {
  return #tfop("HistogramFixedWidth",
    values,
    valueRange,
    nbins,
    T: T.self,
    dtype: Dtype.self)
}

@inlinable @inline(__always)
public static func iFFT<Tcomplex: AccelerableByTensorFlow>(
  _ input: Tensor<Tcomplex>
) -> Tensor<Tcomplex> {
  return #tfop("IFFT",
    input,
    Tcomplex: Tcomplex.self)
}

@inlinable @inline(__always)
public static func iFFT2D<Tcomplex: AccelerableByTensorFlow>(
  _ input: Tensor<Tcomplex>
) -> Tensor<Tcomplex> {
  return #tfop("IFFT2D",
    input,
    Tcomplex: Tcomplex.self)
}

@inlinable @inline(__always)
public static func iFFT3D<Tcomplex: AccelerableByTensorFlow>(
  _ input: Tensor<Tcomplex>
) -> Tensor<Tcomplex> {
  return #tfop("IFFT3D",
    input,
    Tcomplex: Tcomplex.self)
}

@inlinable @inline(__always)
public static func identity<T: AccelerableByTensorFlow>(
  _ input: Tensor<T>
) -> Tensor<T> {
  return #tfop("Identity",
    input,
    T: T.self)
}

@inlinable @inline(__always)
public static func identityN<T: AccelerableByTensorFlow>(
  _ input: [Tensor<T>]
) -> [Tensor<T>] {
  return #tfop("IdentityN",
    input)
}

@inlinable @inline(__always)
public static func igamma<T: BinaryFloatingPoint>(
  _ a: Tensor<T>,
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Igamma",
    a,
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func igammaGradA<T: BinaryFloatingPoint>(
  _ a: Tensor<T>,
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("IgammaGradA",
    a,
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func igammac<T: BinaryFloatingPoint>(
  _ a: Tensor<T>,
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Igammac",
    a,
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func imag<T: AccelerableByTensorFlow, Tout: BinaryFloatingPoint>(
  _ input: Tensor<T>
) -> Tensor<Tout> {
  return #tfop("Imag",
    input,
    T: T.self,
    Tout: Tout.self)
}

@inlinable @inline(__always)
public static func inPolymorphicTwice<T: AccelerableByTensorFlow>(
  _ a: [Tensor<T>],
  _ b: [Tensor<T>]
) {
  return #tfop("InPolymorphicTwice",
    a,
    b,
    T: T.self)
}

@inlinable @inline(__always)
public static func inTopK<T: BinaryInteger>(
  predictions: Tensor<Float>,
  targets: Tensor<T>,
  k: Int64
) -> Tensor<Bool> {
  return #tfop("InTopK",
    predictions,
    targets,
    T: T.self,
    k: k)
}

@inlinable @inline(__always)
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

@inlinable @inline(__always)
public static func inplaceAdd<T: AccelerableByTensorFlow>(
  _ x: Tensor<T>,
  i: Tensor<Int32>,
  v: Tensor<T>
) -> Tensor<T> {
  return #tfop("InplaceAdd",
    x,
    i,
    v,
    T: T.self)
}

@inlinable @inline(__always)
public static func inplaceSub<T: AccelerableByTensorFlow>(
  _ x: Tensor<T>,
  i: Tensor<Int32>,
  v: Tensor<T>
) -> Tensor<T> {
  return #tfop("InplaceSub",
    x,
    i,
    v,
    T: T.self)
}

@inlinable @inline(__always)
public static func inplaceUpdate<T: AccelerableByTensorFlow>(
  _ x: Tensor<T>,
  i: Tensor<Int32>,
  v: Tensor<T>
) -> Tensor<T> {
  return #tfop("InplaceUpdate",
    x,
    i,
    v,
    T: T.self)
}

@inlinable @inline(__always)
public static func int64Output(
) -> Tensor<Int64> {
  return #tfop("Int64Output")
}

@inlinable @inline(__always)
public static func intAttr(
  foo: Int64 = 1
) -> Tensor<Int64> {
  return #tfop("IntAttr",
    foo: foo)
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
  return #tfop("IntInputIntOutput",
    a)
}

@inlinable @inline(__always)
public static func intOutput(
) -> Tensor<Int32> {
  return #tfop("IntOutput")
}

@inlinable @inline(__always)
public static func intOutputFloatOutput(
) -> (Tensor<Int32>, Tensor<Float>) {
  return #tfop("IntOutputFloatOutput")
}

@inlinable @inline(__always)
public static func inv<T: Numeric>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Inv",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func invGrad<T: BinaryFloatingPoint>(
  _ y: Tensor<T>,
  dy: Tensor<T>
) -> Tensor<T> {
  return #tfop("InvGrad",
    y,
    dy,
    T: T.self)
}

@inlinable @inline(__always)
public static func invert<T: BinaryInteger>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Invert",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func invertPermutation<T: BinaryInteger>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("InvertPermutation",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func isFinite<T: BinaryFloatingPoint>(
  _ x: Tensor<T>
) -> Tensor<Bool> {
  return #tfop("IsFinite",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func isInf<T: BinaryFloatingPoint>(
  _ x: Tensor<T>
) -> Tensor<Bool> {
  return #tfop("IsInf",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func isNan<T: BinaryFloatingPoint>(
  _ x: Tensor<T>
) -> Tensor<Bool> {
  return #tfop("IsNan",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func isVariableInitialized<Dtype: AccelerableByTensorFlow>(
  ref: Tensor<Dtype>
) -> Tensor<Bool> {
  return #tfop("IsVariableInitialized",
    ref,
    dtype: Dtype.self)
}

@inlinable @inline(__always)
public static func l2Loss<T: BinaryFloatingPoint>(
  t: Tensor<T>
) -> Tensor<T> {
  return #tfop("L2Loss",
    t,
    T: T.self)
}

@inlinable @inline(__always)
public static func lRN<T: BinaryFloatingPoint>(
  _ input: Tensor<T>,
  depthRadius: Int64 = 5,
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

@inlinable @inline(__always)
public static func lRNGrad<T: BinaryFloatingPoint>(
  inputGrads: Tensor<T>,
  inputImage: Tensor<T>,
  outputImage: Tensor<T>,
  depthRadius: Int64 = 5,
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
public static func lSTMBlockCell<T: BinaryFloatingPoint>(
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
public static func lSTMBlockCellGrad<T: BinaryFloatingPoint>(
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

@inlinable @inline(__always)
public static func learnedUnigramCandidateSampler(
  trueClasses: Tensor<Int64>,
  numTrue: Int64,
  numSampled: Int64,
  unique: Bool,
  rangeMax: Int64,
  seed: Int64 = 0,
  seed2: Int64 = 0
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

@inlinable @inline(__always)
public static func leftShift<T: BinaryInteger>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  return #tfop("LeftShift",
    x,
    y,
    T: T.self)
}

@inlinable @inline(__always)
public static func less<T: Numeric>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<Bool> {
  return #tfop("Less",
    x,
    y,
    T: T.self)
}

@inlinable @inline(__always)
public static func lessEqual<T: Numeric>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<Bool> {
  return #tfop("LessEqual",
    x,
    y,
    T: T.self)
}

@inlinable @inline(__always)
public static func lgamma<T: BinaryFloatingPoint>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Lgamma",
    x,
    T: T.self)
}

@inlinable @inline(__always)
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

@inlinable @inline(__always)
public static func listDiff<T: AccelerableByTensorFlow, Out_idx: BinaryInteger>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> (Tensor<T>, Tensor<Out_idx>) {
  return #tfop("ListDiff",
    x,
    y,
    T: T.self,
    out_idx: Out_idx.self)
}

@inlinable @inline(__always)
public static func listInput<T: AccelerableByTensorFlow>(
  _ a: [Tensor<T>]
) {
  return #tfop("ListInput",
    a,
    T: T.self)
}

@inlinable @inline(__always)
public static func listOutput<T: AccelerableByTensorFlow>(
) -> [Tensor<T>] {
  return #tfop("ListOutput")
}

@inlinable @inline(__always)
public static func log<T: BinaryFloatingPoint>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Log",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func log1p<T: BinaryFloatingPoint>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Log1p",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func logMatrixDeterminant<T: BinaryFloatingPoint>(
  _ input: Tensor<T>
) -> (Tensor<T>, Tensor<T>) {
  return #tfop("LogMatrixDeterminant",
    input,
    T: T.self)
}

@inlinable @inline(__always)
public static func logSoftmax<T: BinaryFloatingPoint>(
  logits: Tensor<T>
) -> Tensor<T> {
  return #tfop("LogSoftmax",
    logits,
    T: T.self)
}

@inlinable @inline(__always)
public static func logUniformCandidateSampler(
  trueClasses: Tensor<Int64>,
  numTrue: Int64,
  numSampled: Int64,
  unique: Bool,
  rangeMax: Int64,
  seed: Int64 = 0,
  seed2: Int64 = 0
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

@inlinable @inline(__always)
public static func logicalAnd(
  _ x: Tensor<Bool>,
  _ y: Tensor<Bool>
) -> Tensor<Bool> {
  return #tfop("LogicalAnd",
    x,
    y)
}

@inlinable @inline(__always)
public static func logicalNot(
  _ x: Tensor<Bool>
) -> Tensor<Bool> {
  return #tfop("LogicalNot",
    x)
}

@inlinable @inline(__always)
public static func logicalOr(
  _ x: Tensor<Bool>,
  _ y: Tensor<Bool>
) -> Tensor<Bool> {
  return #tfop("LogicalOr",
    x,
    y)
}

@inlinable @inline(__always)
public static func loopCond(
  _ input: Tensor<Bool>
) -> Tensor<Bool> {
  return #tfop("LoopCond",
    input)
}

@inlinable @inline(__always)
public static func mapClear<Dtypes: AccelerableByTensorFlow>(
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

@inlinable @inline(__always)
public static func mapIncompleteSize<Dtypes: AccelerableByTensorFlow>(
  capacity: Int64 = 0,
  memoryLimit: Int64 = 0,
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

@inlinable @inline(__always)
public static func mapPeek<Dtypes: AccelerableByTensorFlow>(
  key: Tensor<Int64>,
  indices: Tensor<Int32>,
  capacity: Int64 = 0,
  memoryLimit: Int64 = 0,
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

@inlinable @inline(__always)
public static func mapSize<Dtypes: AccelerableByTensorFlow>(
  capacity: Int64 = 0,
  memoryLimit: Int64 = 0,
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

@inlinable @inline(__always)
public static func mapStage<Dtypes: AccelerableByTensorFlow, Fake_dtypes: AccelerableByTensorFlow>(
  key: Tensor<Int64>,
  indices: Tensor<Int32>,
  _ values: [Tensor<Fake_dtypes>],
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

@inlinable @inline(__always)
public static func mapUnstage<Dtypes: AccelerableByTensorFlow>(
  key: Tensor<Int64>,
  indices: Tensor<Int32>,
  capacity: Int64 = 0,
  memoryLimit: Int64 = 0,
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

@inlinable @inline(__always)
public static func mapUnstageNoKey<Dtypes: AccelerableByTensorFlow>(
  indices: Tensor<Int32>,
  capacity: Int64 = 0,
  memoryLimit: Int64 = 0,
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

@inlinable @inline(__always)
public static func matMul<T: Numeric>(
  _ a: Tensor<T>,
  _ b: Tensor<T>,
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

@inlinable @inline(__always)
public static func matrixBandPart<T: AccelerableByTensorFlow, Tindex: BinaryInteger>(
  _ input: Tensor<T>,
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

@inlinable @inline(__always)
public static func matrixDeterminant<T: BinaryFloatingPoint>(
  _ input: Tensor<T>
) -> Tensor<T> {
  return #tfop("MatrixDeterminant",
    input,
    T: T.self)
}

@inlinable @inline(__always)
public static func matrixDiag<T: AccelerableByTensorFlow>(
  diagonal: Tensor<T>
) -> Tensor<T> {
  return #tfop("MatrixDiag",
    diagonal,
    T: T.self)
}

@inlinable @inline(__always)
public static func matrixDiagPart<T: AccelerableByTensorFlow>(
  _ input: Tensor<T>
) -> Tensor<T> {
  return #tfop("MatrixDiagPart",
    input,
    T: T.self)
}

@inlinable @inline(__always)
public static func matrixExponential<T: BinaryFloatingPoint>(
  _ input: Tensor<T>
) -> Tensor<T> {
  return #tfop("MatrixExponential",
    input,
    T: T.self)
}

@inlinable @inline(__always)
public static func matrixInverse<T: BinaryFloatingPoint>(
  _ input: Tensor<T>,
  adjoint: Bool = false
) -> Tensor<T> {
  return #tfop("MatrixInverse",
    input,
    T: T.self,
    adjoint: adjoint)
}

@inlinable @inline(__always)
public static func matrixLogarithm<T: AccelerableByTensorFlow>(
  _ input: Tensor<T>
) -> Tensor<T> {
  return #tfop("MatrixLogarithm",
    input,
    T: T.self)
}

@inlinable @inline(__always)
public static func matrixSetDiag<T: AccelerableByTensorFlow>(
  _ input: Tensor<T>,
  diagonal: Tensor<T>
) -> Tensor<T> {
  return #tfop("MatrixSetDiag",
    input,
    diagonal,
    T: T.self)
}

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
public static func max<T: Numeric, Tidx: BinaryInteger>(
  _ input: Tensor<T>,
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

@inlinable @inline(__always)
public static func maxPool<T: Numeric>(
  _ input: Tensor<T>,
  ksize: [Int32],
  strides: [Int32],
  padding: Padding,
  dataFormat: DataFormat3 = .nhwc
) -> Tensor<T> {
  return #tfop("MaxPool",
    input,
    T: T.self,
    ksize: ksize,
    strides: strides,
    padding: padding.cName,
    data_format: dataFormat.cName)
}

@inlinable @inline(__always)
public static func maxPool3D<T: BinaryFloatingPoint>(
  _ input: Tensor<T>,
  ksize: [Int32],
  strides: [Int32],
  padding: Padding,
  dataFormat: DataFormat1 = .ndhwc
) -> Tensor<T> {
  return #tfop("MaxPool3D",
    input,
    T: T.self,
    ksize: ksize,
    strides: strides,
    padding: padding.cName,
    data_format: dataFormat.cName)
}

@inlinable @inline(__always)
public static func maxPool3DGrad<T: BinaryFloatingPoint, Tinput: BinaryFloatingPoint>(
  origInput: Tensor<Tinput>,
  origOutput: Tensor<Tinput>,
  grad: Tensor<T>,
  ksize: [Int32],
  strides: [Int32],
  padding: Padding,
  dataFormat: DataFormat1 = .ndhwc
) -> Tensor<T> {
  return #tfop("MaxPool3DGrad",
    origInput,
    origOutput,
    grad,
    T: T.self,
    TInput: Tinput.self,
    ksize: ksize,
    strides: strides,
    padding: padding.cName,
    data_format: dataFormat.cName)
}

@inlinable @inline(__always)
public static func maxPool3DGradGrad<T: Numeric>(
  origInput: Tensor<T>,
  origOutput: Tensor<T>,
  grad: Tensor<T>,
  ksize: [Int32],
  strides: [Int32],
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
    padding: padding.cName,
    data_format: dataFormat.cName)
}

@inlinable @inline(__always)
public static func maxPoolGrad<T: Numeric>(
  origInput: Tensor<T>,
  origOutput: Tensor<T>,
  grad: Tensor<T>,
  ksize: [Int32],
  strides: [Int32],
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
    padding: padding.cName,
    data_format: dataFormat.cName)
}

@inlinable @inline(__always)
public static func maxPoolGradGrad<T: Numeric>(
  origInput: Tensor<T>,
  origOutput: Tensor<T>,
  grad: Tensor<T>,
  ksize: [Int32],
  strides: [Int32],
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
    padding: padding.cName,
    data_format: dataFormat.cName)
}

@inlinable @inline(__always)
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
    padding: padding.cName,
    data_format: dataFormat.cName)
}

@inlinable @inline(__always)
public static func maxPoolGradGradWithArgmax<Targmax: BinaryInteger, T: Numeric>(
  _ input: Tensor<T>,
  grad: Tensor<T>,
  argmax: Tensor<Targmax>,
  ksize: [Int32],
  strides: [Int32],
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
    padding: padding.cName)
}

@inlinable @inline(__always)
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
    padding: padding.cName,
    data_format: dataFormat.cName)
}

@inlinable @inline(__always)
public static func maxPoolGradWithArgmax<Targmax: BinaryInteger, T: Numeric>(
  _ input: Tensor<T>,
  grad: Tensor<T>,
  argmax: Tensor<Targmax>,
  ksize: [Int32],
  strides: [Int32],
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
    padding: padding.cName)
}

@inlinable @inline(__always)
public static func maxPoolV2<T: Numeric>(
  _ input: Tensor<T>,
  ksize: Tensor<Int32>,
  strides: Tensor<Int32>,
  padding: Padding,
  dataFormat: DataFormat3 = .nhwc
) -> Tensor<T> {
  return #tfop("MaxPoolV2",
    input,
    ksize,
    strides,
    T: T.self,
    padding: padding.cName,
    data_format: dataFormat.cName)
}

@inlinable @inline(__always)
public static func maxPoolWithArgmax<Targmax: BinaryInteger, T: Numeric>(
  _ input: Tensor<T>,
  ksize: [Int32],
  strides: [Int32],
  padding: Padding
) -> (Tensor<T>, Tensor<Targmax>) {
  return #tfop("MaxPoolWithArgmax",
    input,
    Targmax: Targmax.self,
    T: T.self,
    ksize: ksize,
    strides: strides,
    padding: padding.cName)
}

@inlinable @inline(__always)
public static func maximum<T: Numeric>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  return #tfop("Maximum",
    x,
    y,
    T: T.self)
}

@inlinable @inline(__always)
public static func mean<T: Numeric, Tidx: BinaryInteger>(
  _ input: Tensor<T>,
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

@inlinable @inline(__always)
public static func merge<T: AccelerableByTensorFlow>(
  inputs: [Tensor<T>]
) -> (Tensor<T>, Tensor<Int32>) {
  return #tfop("Merge",
    inputs,
    T: T.self)
}

@inlinable @inline(__always)
public static func mfcc(
  spectrogram: Tensor<Float>,
  sampleRate: Tensor<Int32>,
  upperFrequencyLimit: Double = 4000,
  lowerFrequencyLimit: Double = 20,
  filterbankChannelCount: Int64 = 40,
  dctCoefficientCount: Int64 = 13
) -> Tensor<Float> {
  return #tfop("Mfcc",
    spectrogram,
    sampleRate,
    upper_frequency_limit: upperFrequencyLimit,
    lower_frequency_limit: lowerFrequencyLimit,
    filterbank_channel_count: filterbankChannelCount,
    dct_coefficient_count: dctCoefficientCount)
}

@inlinable @inline(__always)
public static func min<T: Numeric, Tidx: BinaryInteger>(
  _ input: Tensor<T>,
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

@inlinable @inline(__always)
public static func minimum<T: Numeric>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  return #tfop("Minimum",
    x,
    y,
    T: T.self)
}

@inlinable @inline(__always)
public static func mirrorPad<T: AccelerableByTensorFlow, Tpaddings: BinaryInteger>(
  _ input: Tensor<T>,
  paddings: Tensor<Tpaddings>,
  mode: Mode4
) -> Tensor<T> {
  return #tfop("MirrorPad",
    input,
    paddings,
    T: T.self,
    Tpaddings: Tpaddings.self,
    mode: mode.cName)
}

@inlinable @inline(__always)
public static func mirrorPadGrad<T: AccelerableByTensorFlow, Tpaddings: BinaryInteger>(
  _ input: Tensor<T>,
  paddings: Tensor<Tpaddings>,
  mode: Mode4
) -> Tensor<T> {
  return #tfop("MirrorPadGrad",
    input,
    paddings,
    T: T.self,
    Tpaddings: Tpaddings.self,
    mode: mode.cName)
}

@inlinable @inline(__always)
public static func mixedStruct(
  nA: Int64
) -> ([Tensor<Int32>], Tensor<Float>) {
  return #tfop("MixedStruct",
    n_a: nA)
}

@inlinable @inline(__always)
public static func mod<T: Numeric>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  return #tfop("Mod",
    x,
    y,
    T: T.self)
}

@inlinable @inline(__always)
public static func mul<T: Numeric>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  return #tfop("Mul",
    x,
    y,
    T: T.self)
}

@inlinable @inline(__always)
public static func multinomial<T: Numeric, Output_dtype: BinaryInteger>(
  logits: Tensor<T>,
  numSamples: Tensor<Int32>,
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> Tensor<Output_dtype> {
  return #tfop("Multinomial",
    logits,
    numSamples,
    T: T.self,
    output_dtype: Output_dtype.self,
    seed: seed,
    seed2: seed2)
}

@inlinable @inline(__always)
public static func nInPolymorphicTwice<T: AccelerableByTensorFlow>(
  _ a: [Tensor<T>],
  _ b: [Tensor<T>]
) {
  return #tfop("NInPolymorphicTwice",
    a,
    b,
    T: T.self)
}

@inlinable @inline(__always)
public static func nInTwoTypeVariables<S: AccelerableByTensorFlow, T: AccelerableByTensorFlow>(
  _ a: [Tensor<S>],
  _ b: [Tensor<T>]
) {
  return #tfop("NInTwoTypeVariables",
    a,
    b,
    S: S.self,
    T: T.self)
}

@inlinable @inline(__always)
public static func nIntsIn(
  _ a: [Tensor<Int32>]
) {
  return #tfop("NIntsIn",
    a)
}

@inlinable @inline(__always)
public static func nIntsOut(
  n: Int64
) -> [Tensor<Int32>] {
  return #tfop("NIntsOut",
    N: n)
}

@inlinable @inline(__always)
public static func nIntsOutDefault(
  n: Int64 = 3
) -> [Tensor<Int32>] {
  return #tfop("NIntsOutDefault",
    N: n)
}

@inlinable @inline(__always)
public static func nPolymorphicIn<T: AccelerableByTensorFlow>(
  _ a: [Tensor<T>]
) {
  return #tfop("NPolymorphicIn",
    a,
    T: T.self)
}

@inlinable @inline(__always)
public static func nPolymorphicOut<T: AccelerableByTensorFlow>(
  n: Int64
) -> [Tensor<T>] {
  return #tfop("NPolymorphicOut",
    T: T.self,
    N: n)
}

@inlinable @inline(__always)
public static func nPolymorphicOutDefault<T: AccelerableByTensorFlow>(
  n: Int64 = 2
) -> [Tensor<T>] {
  return #tfop("NPolymorphicOutDefault",
    T: T.self,
    N: n)
}

@inlinable @inline(__always)
public static func nPolymorphicRestrictIn<T: AccelerableByTensorFlow>(
  _ a: [Tensor<T>]
) {
  return #tfop("NPolymorphicRestrictIn",
    a,
    T: T.self)
}

@inlinable @inline(__always)
public static func nPolymorphicRestrictOut<T: AccelerableByTensorFlow>(
  n: Int64
) -> [Tensor<T>] {
  return #tfop("NPolymorphicRestrictOut",
    T: T.self,
    N: n)
}

@inlinable @inline(__always)
public static func neg<T: Numeric>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Neg",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func negTrain(
  wIn: Tensor<Float>,
  wOut: Tensor<Float>,
  examples: Tensor<Int32>,
  labels: Tensor<Int32>,
  lr: Tensor<Float>,
  vocabCount: [Int32],
  numNegativeSamples: Int64
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

@inlinable @inline(__always)
public static func nextIteration<T: AccelerableByTensorFlow>(
  data: Tensor<T>
) -> Tensor<T> {
  return #tfop("NextIteration",
    data,
    T: T.self)
}

@inlinable @inline(__always)
public static func noOp(
) {
  return #tfop("NoOp")
}

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
public static func nonMaxSuppressionV3(
  boxes: Tensor<Float>,
  scores: Tensor<Float>,
  maxOutputSize: Tensor<Int32>,
  iouThreshold: Tensor<Float>,
  scoreThreshold: Tensor<Float>
) -> Tensor<Int32> {
  return #tfop("NonMaxSuppressionV3",
    boxes,
    scores,
    maxOutputSize,
    iouThreshold,
    scoreThreshold)
}

@inlinable @inline(__always)
public static func none(
) {
  return #tfop("None")
}

@inlinable @inline(__always)
public static func notEqual<T: AccelerableByTensorFlow>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<Bool> {
  return #tfop("NotEqual",
    x,
    y,
    T: T.self)
}

@inlinable @inline(__always)
public static func nthElement<T: Numeric>(
  _ input: Tensor<T>,
  n: Tensor<Int32>,
  reverse: Bool = false
) -> Tensor<T> {
  return #tfop("NthElement",
    input,
    n,
    T: T.self,
    reverse: reverse)
}

@inlinable @inline(__always)
public static func old(
) {
  return #tfop("Old")
}

@inlinable @inline(__always)
public static func oneHot<T: AccelerableByTensorFlow, Ti: BinaryInteger>(
  indices: Tensor<Ti>,
  depth: Tensor<Int32>,
  onValue: Tensor<T>,
  offValue: Tensor<T>,
  axis: Int64 = -1
) -> Tensor<T> {
  return #tfop("OneHot",
    indices,
    depth,
    onValue,
    offValue,
    T: T.self,
    TI: Ti.self,
    axis: axis)
}

@inlinable @inline(__always)
public static func onesLike<T: AccelerableByTensorFlow>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("OnesLike",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func opWithDefaultAttr(
  defaultFloat: Double = 123
) -> Tensor<Int32> {
  return #tfop("OpWithDefaultAttr",
    default_float: defaultFloat)
}

@inlinable @inline(__always)
public static func opWithFutureDefaultAttr(
) {
  return #tfop("OpWithFutureDefaultAttr")
}

@inlinable @inline(__always)
public static func orderedMapClear<Dtypes: AccelerableByTensorFlow>(
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

@inlinable @inline(__always)
public static func orderedMapIncompleteSize<Dtypes: AccelerableByTensorFlow>(
  capacity: Int64 = 0,
  memoryLimit: Int64 = 0,
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

@inlinable @inline(__always)
public static func orderedMapPeek<Dtypes: AccelerableByTensorFlow>(
  key: Tensor<Int64>,
  indices: Tensor<Int32>,
  capacity: Int64 = 0,
  memoryLimit: Int64 = 0,
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

@inlinable @inline(__always)
public static func orderedMapSize<Dtypes: AccelerableByTensorFlow>(
  capacity: Int64 = 0,
  memoryLimit: Int64 = 0,
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

@inlinable @inline(__always)
public static func orderedMapStage<Dtypes: AccelerableByTensorFlow, Fake_dtypes: AccelerableByTensorFlow>(
  key: Tensor<Int64>,
  indices: Tensor<Int32>,
  _ values: [Tensor<Fake_dtypes>],
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
public static func orderedMapUnstage<Dtypes: AccelerableByTensorFlow>(
  key: Tensor<Int64>,
  indices: Tensor<Int32>,
  capacity: Int64 = 0,
  memoryLimit: Int64 = 0,
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

@inlinable @inline(__always)
public static func orderedMapUnstageNoKey<Dtypes: AccelerableByTensorFlow>(
  indices: Tensor<Int32>,
  capacity: Int64 = 0,
  memoryLimit: Int64 = 0,
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

@inlinable @inline(__always)
public static func outT<T: AccelerableByTensorFlow>(
) -> Tensor<T> {
  return #tfop("OutT",
    T: T.self)
}

@inlinable @inline(__always)
public static func outTypeList<T: AccelerableByTensorFlow>(
) -> [Tensor<T>] {
  return #tfop("OutTypeList")
}

@inlinable @inline(__always)
public static func outTypeListRestrict<T: AccelerableByTensorFlow>(
) -> [Tensor<T>] {
  return #tfop("OutTypeListRestrict")
}

@inlinable @inline(__always)
public static func pack<T: AccelerableByTensorFlow>(
  _ values: [Tensor<T>],
  axis: Int64 = 0
) -> Tensor<T> {
  return #tfop("Pack",
    values,
    T: T.self,
    axis: axis)
}

@inlinable @inline(__always)
public static func pad<T: AccelerableByTensorFlow, Tpaddings: BinaryInteger>(
  _ input: Tensor<T>,
  paddings: Tensor<Tpaddings>
) -> Tensor<T> {
  return #tfop("Pad",
    input,
    paddings,
    T: T.self,
    Tpaddings: Tpaddings.self)
}

@inlinable @inline(__always)
public static func padV2<T: AccelerableByTensorFlow, Tpaddings: BinaryInteger>(
  _ input: Tensor<T>,
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

@inlinable @inline(__always)
public static func parallelDynamicStitch<T: AccelerableByTensorFlow>(
  indices: [Tensor<Int32>],
  data: [Tensor<T>]
) -> Tensor<T> {
  return #tfop("ParallelDynamicStitch",
    indices,
    data,
    T: T.self)
}

@inlinable @inline(__always)
public static func parameterizedTruncatedNormal<Dtype: BinaryFloatingPoint, T: BinaryInteger>(
  shape: Tensor<T>,
  means: Tensor<Dtype>,
  stdevs: Tensor<Dtype>,
  minvals: Tensor<Dtype>,
  maxvals: Tensor<Dtype>,
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> Tensor<Dtype> {
  return #tfop("ParameterizedTruncatedNormal",
    shape,
    means,
    stdevs,
    minvals,
    maxvals,
    dtype: Dtype.self,
    T: T.self,
    seed: seed,
    seed2: seed2)
}

@inlinable @inline(__always)
public static func polygamma<T: BinaryFloatingPoint>(
  _ a: Tensor<T>,
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Polygamma",
    a,
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func polymorphic<T: AccelerableByTensorFlow>(
  _ a: Tensor<T>
) -> Tensor<T> {
  return #tfop("Polymorphic",
    a,
    T: T.self)
}

@inlinable @inline(__always)
public static func polymorphicDefaultOut<T: AccelerableByTensorFlow>(
) -> Tensor<T> {
  return #tfop("PolymorphicDefaultOut",
    T: T.self)
}

@inlinable @inline(__always)
public static func polymorphicOut<T: AccelerableByTensorFlow>(
) -> Tensor<T> {
  return #tfop("PolymorphicOut",
    T: T.self)
}

@inlinable @inline(__always)
public static func populationCount<T: BinaryInteger>(
  _ x: Tensor<T>
) -> Tensor<UInt8> {
  return #tfop("PopulationCount",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func pow<T: Numeric>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  return #tfop("Pow",
    x,
    y,
    T: T.self)
}

@inlinable @inline(__always)
public static func preventGradient<T: AccelerableByTensorFlow>(
  _ input: Tensor<T>,
  message: String
) -> Tensor<T> {
  return #tfop("PreventGradient",
    input,
    T: T.self,
    message: message)
}

@inlinable @inline(__always)
public static func print<T: AccelerableByTensorFlow, U: AccelerableByTensorFlow>(
  _ input: Tensor<T>,
  data: [Tensor<U>],
  message: String,
  firstN: Int64 = -1,
  summarize: Int64 = 3
) -> Tensor<T> {
  return #tfop("Print",
    input,
    data,
    T: T.self,
    message: message,
    first_n: firstN,
    summarize: summarize)
}

@inlinable @inline(__always)
public static func prod<T: Numeric, Tidx: BinaryInteger>(
  _ input: Tensor<T>,
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

@inlinable @inline(__always)
public static func pyFunc<Tin: AccelerableByTensorFlow, Tout: AccelerableByTensorFlow>(
  _ input: [Tensor<Tin>],
  token: String
) -> [Tensor<Tout>] {
  return #tfop("PyFunc",
    input,
    token: token)
}

@inlinable @inline(__always)
public static func pyFuncStateless<Tin: AccelerableByTensorFlow, Tout: AccelerableByTensorFlow>(
  _ input: [Tensor<Tin>],
  token: String
) -> [Tensor<Tout>] {
  return #tfop("PyFuncStateless",
    input,
    token: token)
}

@inlinable @inline(__always)
public static func qr<T: BinaryFloatingPoint>(
  _ input: Tensor<T>,
  fullMatrices: Bool = false
) -> (Tensor<T>, Tensor<T>) {
  return #tfop("Qr",
    input,
    T: T.self,
    full_matrices: fullMatrices)
}

@inlinable @inline(__always)
public static func quantizeAndDequantize<T: BinaryFloatingPoint>(
  _ input: Tensor<T>,
  signedInput: Bool = true,
  numBits: Int64 = 8,
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

@inlinable @inline(__always)
public static func quantizeAndDequantizeV2<T: BinaryFloatingPoint>(
  _ input: Tensor<T>,
  inputMin: Tensor<T>,
  inputMax: Tensor<T>,
  signedInput: Bool = true,
  numBits: Int64 = 8,
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

@inlinable @inline(__always)
public static func quantizeAndDequantizeV3<T: BinaryFloatingPoint>(
  _ input: Tensor<T>,
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

@inlinable @inline(__always)
public static func quantizeDownAndShrinkRange<Tinput: AccelerableByTensorFlow, Out_type: AccelerableByTensorFlow>(
  _ input: Tensor<Tinput>,
  inputMin: Tensor<Float>,
  inputMax: Tensor<Float>
) -> (Tensor<Out_type>, Tensor<Float>, Tensor<Float>) {
  return #tfop("QuantizeDownAndShrinkRange",
    input,
    inputMin,
    inputMax,
    Tinput: Tinput.self,
    out_type: Out_type.self)
}

@inlinable @inline(__always)
public static func quantizeV2<T: AccelerableByTensorFlow>(
  _ input: Tensor<Float>,
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
    mode: mode.cName,
    round_mode: roundMode.cName)
}

@inlinable @inline(__always)
public static func quantizedAdd<T1: AccelerableByTensorFlow, T2: AccelerableByTensorFlow, Toutput: AccelerableByTensorFlow>(
  _ x: Tensor<T1>,
  _ y: Tensor<T2>,
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

@inlinable @inline(__always)
public static func quantizedAvgPool<T: AccelerableByTensorFlow>(
  _ input: Tensor<T>,
  minInput: Tensor<Float>,
  maxInput: Tensor<Float>,
  ksize: [Int32],
  strides: [Int32],
  padding: Padding
) -> (Tensor<T>, Tensor<Float>, Tensor<Float>) {
  return #tfop("QuantizedAvgPool",
    input,
    minInput,
    maxInput,
    T: T.self,
    ksize: ksize,
    strides: strides,
    padding: padding.cName)
}

@inlinable @inline(__always)
public static func quantizedBatchNormWithGlobalNormalization<Tinput: AccelerableByTensorFlow, Out_type: AccelerableByTensorFlow>(
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
    out_type: Out_type.self,
    variance_epsilon: varianceEpsilon,
    scale_after_normalization: scaleAfterNormalization)
}

@inlinable @inline(__always)
public static func quantizedBiasAdd<T1: AccelerableByTensorFlow, T2: AccelerableByTensorFlow, Out_type: AccelerableByTensorFlow>(
  _ input: Tensor<T1>,
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
    out_type: Out_type.self)
}

@inlinable @inline(__always)
public static func quantizedConcat<T: AccelerableByTensorFlow>(
  concatDim: Tensor<Int32>,
  _ values: [Tensor<T>],
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

@inlinable @inline(__always)
public static func quantizedConv2D<Tinput: AccelerableByTensorFlow, Tfilter: AccelerableByTensorFlow, Out_type: AccelerableByTensorFlow>(
  _ input: Tensor<Tinput>,
  filter: Tensor<Tfilter>,
  minInput: Tensor<Float>,
  maxInput: Tensor<Float>,
  minFilter: Tensor<Float>,
  maxFilter: Tensor<Float>,
  strides: [Int32],
  padding: Padding,
  dilations: [Int32] = [1, 1, 1, 1]
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
    out_type: Out_type.self,
    strides: strides,
    padding: padding.cName,
    dilations: dilations)
}

@inlinable @inline(__always)
public static func quantizedInstanceNorm<T: AccelerableByTensorFlow>(
  _ x: Tensor<T>,
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

@inlinable @inline(__always)
public static func quantizedMatMul<T1: AccelerableByTensorFlow, T2: AccelerableByTensorFlow, Toutput: AccelerableByTensorFlow, Tactivation: AccelerableByTensorFlow>(
  _ a: Tensor<T1>,
  _ b: Tensor<T2>,
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

@inlinable @inline(__always)
public static func quantizedMaxPool<T: AccelerableByTensorFlow>(
  _ input: Tensor<T>,
  minInput: Tensor<Float>,
  maxInput: Tensor<Float>,
  ksize: [Int32],
  strides: [Int32],
  padding: Padding
) -> (Tensor<T>, Tensor<Float>, Tensor<Float>) {
  return #tfop("QuantizedMaxPool",
    input,
    minInput,
    maxInput,
    T: T.self,
    ksize: ksize,
    strides: strides,
    padding: padding.cName)
}

@inlinable @inline(__always)
public static func quantizedMul<T1: AccelerableByTensorFlow, T2: AccelerableByTensorFlow, Toutput: AccelerableByTensorFlow>(
  _ x: Tensor<T1>,
  _ y: Tensor<T2>,
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

@inlinable @inline(__always)
public static func quantizedRelu<Tinput: AccelerableByTensorFlow, Out_type: AccelerableByTensorFlow>(
  features: Tensor<Tinput>,
  minFeatures: Tensor<Float>,
  maxFeatures: Tensor<Float>
) -> (Tensor<Out_type>, Tensor<Float>, Tensor<Float>) {
  return #tfop("QuantizedRelu",
    features,
    minFeatures,
    maxFeatures,
    Tinput: Tinput.self,
    out_type: Out_type.self)
}

@inlinable @inline(__always)
public static func quantizedRelu6<Tinput: AccelerableByTensorFlow, Out_type: AccelerableByTensorFlow>(
  features: Tensor<Tinput>,
  minFeatures: Tensor<Float>,
  maxFeatures: Tensor<Float>
) -> (Tensor<Out_type>, Tensor<Float>, Tensor<Float>) {
  return #tfop("QuantizedRelu6",
    features,
    minFeatures,
    maxFeatures,
    Tinput: Tinput.self,
    out_type: Out_type.self)
}

@inlinable @inline(__always)
public static func quantizedReluX<Tinput: AccelerableByTensorFlow, Out_type: AccelerableByTensorFlow>(
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
    out_type: Out_type.self)
}

@inlinable @inline(__always)
public static func quantizedReshape<T: AccelerableByTensorFlow, Tshape: BinaryInteger>(
  _ tensor: Tensor<T>,
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
public static func rGBToHSV<T: BinaryFloatingPoint>(
  images: Tensor<T>
) -> Tensor<T> {
  return #tfop("RGBToHSV",
    images,
    T: T.self)
}

@inlinable @inline(__always)
public static func randomCrop<T: Numeric>(
  image: Tensor<T>,
  size: Tensor<Int64>,
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> Tensor<T> {
  return #tfop("RandomCrop",
    image,
    size,
    T: T.self,
    seed: seed,
    seed2: seed2)
}

@inlinable @inline(__always)
public static func randomGamma<S: BinaryInteger, T: BinaryFloatingPoint>(
  shape: Tensor<S>,
  alpha: Tensor<T>,
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> Tensor<T> {
  return #tfop("RandomGamma",
    shape,
    alpha,
    S: S.self,
    T: T.self,
    seed: seed,
    seed2: seed2)
}

@inlinable @inline(__always)
public static func randomGammaGrad<T: BinaryFloatingPoint>(
  alpha: Tensor<T>,
  sample: Tensor<T>
) -> Tensor<T> {
  return #tfop("RandomGammaGrad",
    alpha,
    sample,
    T: T.self)
}

@inlinable @inline(__always)
public static func randomPoisson<S: BinaryInteger, Dtype: BinaryFloatingPoint>(
  shape: Tensor<S>,
  rate: Tensor<Dtype>,
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> Tensor<Dtype> {
  return #tfop("RandomPoisson",
    shape,
    rate,
    S: S.self,
    dtype: Dtype.self,
    seed: seed,
    seed2: seed2)
}

@inlinable @inline(__always)
public static func randomPoissonV2<S: BinaryInteger, R: Numeric, Dtype: Numeric>(
  shape: Tensor<S>,
  rate: Tensor<R>,
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> Tensor<Dtype> {
  return #tfop("RandomPoissonV2",
    shape,
    rate,
    S: S.self,
    R: R.self,
    dtype: Dtype.self,
    seed: seed,
    seed2: seed2)
}

@inlinable @inline(__always)
public static func randomShuffle<T: AccelerableByTensorFlow>(
  value: Tensor<T>,
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> Tensor<T> {
  return #tfop("RandomShuffle",
    value,
    T: T.self,
    seed: seed,
    seed2: seed2)
}

@inlinable @inline(__always)
public static func randomStandardNormal<Dtype: BinaryFloatingPoint, T: BinaryInteger>(
  shape: Tensor<T>,
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> Tensor<Dtype> {
  return #tfop("RandomStandardNormal",
    shape,
    dtype: Dtype.self,
    T: T.self,
    seed: seed,
    seed2: seed2)
}

@inlinable @inline(__always)
public static func randomUniform<Dtype: BinaryFloatingPoint, T: BinaryInteger>(
  shape: Tensor<T>,
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> Tensor<Dtype> {
  return #tfop("RandomUniform",
    shape,
    dtype: Dtype.self,
    T: T.self,
    seed: seed,
    seed2: seed2)
}

@inlinable @inline(__always)
public static func randomUniformInt<Tout: BinaryInteger, T: BinaryInteger>(
  shape: Tensor<T>,
  minval: Tensor<Tout>,
  maxval: Tensor<Tout>,
  seed: Int64 = 0,
  seed2: Int64 = 0
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
public static func rank<T: AccelerableByTensorFlow>(
  _ input: Tensor<T>
) -> Tensor<Int32> {
  return #tfop("Rank",
    input,
    T: T.self)
}

@inlinable @inline(__always)
public static func real<T: AccelerableByTensorFlow, Tout: BinaryFloatingPoint>(
  _ input: Tensor<T>
) -> Tensor<Tout> {
  return #tfop("Real",
    input,
    T: T.self,
    Tout: Tout.self)
}

@inlinable @inline(__always)
public static func realDiv<T: Numeric>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  return #tfop("RealDiv",
    x,
    y,
    T: T.self)
}

@inlinable @inline(__always)
public static func reciprocal<T: Numeric>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Reciprocal",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func reciprocalGrad<T: BinaryFloatingPoint>(
  _ y: Tensor<T>,
  dy: Tensor<T>
) -> Tensor<T> {
  return #tfop("ReciprocalGrad",
    y,
    dy,
    T: T.self)
}

@inlinable @inline(__always)
public static func refEnter<T: AccelerableByTensorFlow>(
  data: Tensor<T>,
  frameName: String,
  isConstant: Bool = false,
  parallelIterations: Int64 = 10
) -> Tensor<T> {
  return #tfop("RefEnter",
    data,
    T: T.self,
    frame_name: frameName,
    is_constant: isConstant,
    parallel_iterations: parallelIterations)
}

@inlinable @inline(__always)
public static func refExit<T: AccelerableByTensorFlow>(
  data: Tensor<T>
) -> Tensor<T> {
  return #tfop("RefExit",
    data,
    T: T.self)
}

@inlinable @inline(__always)
public static func refIdentity<T: AccelerableByTensorFlow>(
  _ input: Tensor<T>
) -> Tensor<T> {
  return #tfop("RefIdentity",
    input,
    T: T.self)
}

@inlinable @inline(__always)
public static func refIn<T: AccelerableByTensorFlow>(
  _ a: Tensor<T>
) {
  return #tfop("RefIn",
    a,
    T: T.self)
}

@inlinable @inline(__always)
public static func refInputFloatInput(
  _ a: Tensor<Float>,
  _ b: Tensor<Float>
) {
  return #tfop("RefInputFloatInput",
    a,
    b)
}

@inlinable @inline(__always)
public static func refInputFloatInputIntOutput(
  _ a: Tensor<Float>,
  _ b: Tensor<Float>
) -> Tensor<Int32> {
  return #tfop("RefInputFloatInputIntOutput",
    a,
    b)
}

@inlinable @inline(__always)
public static func refInputIntInput(
  _ a: Tensor<Int32>,
  _ b: Tensor<Int32>
) {
  return #tfop("RefInputIntInput",
    a,
    b)
}

@inlinable @inline(__always)
public static func refMerge<T: AccelerableByTensorFlow>(
  inputs: [Tensor<T>]
) -> (Tensor<T>, Tensor<Int32>) {
  return #tfop("RefMerge",
    inputs,
    T: T.self)
}

@inlinable @inline(__always)
public static func refNextIteration<T: AccelerableByTensorFlow>(
  data: Tensor<T>
) -> Tensor<T> {
  return #tfop("RefNextIteration",
    data,
    T: T.self)
}

@inlinable @inline(__always)
public static func refOut<T: AccelerableByTensorFlow>(
) -> Tensor<T> {
  return #tfop("RefOut",
    T: T.self)
}

@inlinable @inline(__always)
public static func refOutput(
) -> Tensor<Int32> {
  return #tfop("RefOutput")
}

@inlinable @inline(__always)
public static func refOutputFloatOutput(
) -> (Tensor<Float>, Tensor<Float>) {
  return #tfop("RefOutputFloatOutput")
}

@inlinable @inline(__always)
public static func refSelect<T: AccelerableByTensorFlow>(
  index: Tensor<Int32>,
  inputs: [Tensor<T>]
) -> Tensor<T> {
  return #tfop("RefSelect",
    index,
    inputs,
    T: T.self)
}

@inlinable @inline(__always)
public static func refSwitch<T: AccelerableByTensorFlow>(
  data: Tensor<T>,
  pred: Tensor<Bool>
) -> (Tensor<T>, Tensor<T>) {
  return #tfop("RefSwitch",
    data,
    pred,
    T: T.self)
}

@inlinable @inline(__always)
public static func relu<T: Numeric>(
  features: Tensor<T>
) -> Tensor<T> {
  return #tfop("Relu",
    features,
    T: T.self)
}

@inlinable @inline(__always)
public static func relu6<T: Numeric>(
  features: Tensor<T>
) -> Tensor<T> {
  return #tfop("Relu6",
    features,
    T: T.self)
}

@inlinable @inline(__always)
public static func relu6Grad<T: Numeric>(
  gradients: Tensor<T>,
  features: Tensor<T>
) -> Tensor<T> {
  return #tfop("Relu6Grad",
    gradients,
    features,
    T: T.self)
}

@inlinable @inline(__always)
public static func reluGrad<T: Numeric>(
  gradients: Tensor<T>,
  features: Tensor<T>
) -> Tensor<T> {
  return #tfop("ReluGrad",
    gradients,
    features,
    T: T.self)
}

@inlinable @inline(__always)
public static func remoteFusedGraphExecute<Tinputs: AccelerableByTensorFlow, Toutputs: AccelerableByTensorFlow>(
  inputs: [Tensor<Tinputs>],
  serializedRemoteFusedGraphExecuteInfo: String
) -> [Tensor<Toutputs>] {
  return #tfop("RemoteFusedGraphExecute",
    inputs,
    serialized_remote_fused_graph_execute_info: serializedRemoteFusedGraphExecuteInfo)
}

@inlinable @inline(__always)
public static func requantizationRange<Tinput: AccelerableByTensorFlow>(
  _ input: Tensor<Tinput>,
  inputMin: Tensor<Float>,
  inputMax: Tensor<Float>
) -> (Tensor<Float>, Tensor<Float>) {
  return #tfop("RequantizationRange",
    input,
    inputMin,
    inputMax,
    Tinput: Tinput.self)
}

@inlinable @inline(__always)
public static func requantize<Tinput: AccelerableByTensorFlow, Out_type: AccelerableByTensorFlow>(
  _ input: Tensor<Tinput>,
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
    out_type: Out_type.self)
}

@inlinable @inline(__always)
public static func requiresOlderGraphVersion(
) -> Tensor<Int32> {
  return #tfop("RequiresOlderGraphVersion")
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

@inlinable @inline(__always)
public static func reshape<T: AccelerableByTensorFlow, Tshape: BinaryInteger>(
  _ tensor: Tensor<T>,
  shape: Tensor<Tshape>
) -> Tensor<T> {
  return #tfop("Reshape",
    tensor,
    shape,
    T: T.self,
    Tshape: Tshape.self)
}

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
public static func restrict<T: AccelerableByTensorFlow>(
  _ a: Tensor<T>
) -> Tensor<T> {
  return #tfop("Restrict",
    a,
    T: T.self)
}

@inlinable @inline(__always)
public static func reverse<T: AccelerableByTensorFlow>(
  _ tensor: Tensor<T>,
  dims: Tensor<Bool>
) -> Tensor<T> {
  return #tfop("Reverse",
    tensor,
    dims,
    T: T.self)
}

@inlinable @inline(__always)
public static func reverseSequence<T: AccelerableByTensorFlow, Tlen: BinaryInteger>(
  _ input: Tensor<T>,
  seqLengths: Tensor<Tlen>,
  seqDim: Int64,
  batchDim: Int64 = 0
) -> Tensor<T> {
  return #tfop("ReverseSequence",
    input,
    seqLengths,
    T: T.self,
    Tlen: Tlen.self,
    seq_dim: seqDim,
    batch_dim: batchDim)
}

@inlinable @inline(__always)
public static func reverseV2<Tidx: BinaryInteger, T: AccelerableByTensorFlow>(
  _ tensor: Tensor<T>,
  axis: Tensor<Tidx>
) -> Tensor<T> {
  return #tfop("ReverseV2",
    tensor,
    axis,
    Tidx: Tidx.self,
    T: T.self)
}

@inlinable @inline(__always)
public static func rightShift<T: BinaryInteger>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  return #tfop("RightShift",
    x,
    y,
    T: T.self)
}

@inlinable @inline(__always)
public static func rint<T: BinaryFloatingPoint>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Rint",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func roll<T: AccelerableByTensorFlow, Tshift: BinaryInteger, Taxis: BinaryInteger>(
  _ input: Tensor<T>,
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

@inlinable @inline(__always)
public static func round<T: Numeric>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Round",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func rsqrt<T: BinaryFloatingPoint>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Rsqrt",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func rsqrtGrad<T: BinaryFloatingPoint>(
  _ y: Tensor<T>,
  dy: Tensor<T>
) -> Tensor<T> {
  return #tfop("RsqrtGrad",
    y,
    dy,
    T: T.self)
}

@inlinable @inline(__always)
public static func sampleDistortedBoundingBox<T: BinaryInteger>(
  imageSize: Tensor<T>,
  boundingBoxes: Tensor<Float>,
  seed: Int64 = 0,
  seed2: Int64 = 0,
  minObjectCovered: Double = 0.1,
  aspectRatioRange: [Double] = [0.75, 1.33],
  areaRange: [Double] = [0.05, 1],
  maxAttempts: Int64 = 100,
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

@inlinable @inline(__always)
public static func sampleDistortedBoundingBoxV2<T: BinaryInteger>(
  imageSize: Tensor<T>,
  boundingBoxes: Tensor<Float>,
  minObjectCovered: Tensor<Float>,
  seed: Int64 = 0,
  seed2: Int64 = 0,
  aspectRatioRange: [Double] = [0.75, 1.33],
  areaRange: [Double] = [0.05, 1],
  maxAttempts: Int64 = 100,
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
public static func scatterNd<T: AccelerableByTensorFlow, Tindices: BinaryInteger>(
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
public static func scatterNdNonAliasingAdd<T: Numeric, Tindices: BinaryInteger>(
  _ input: Tensor<T>,
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
public static func scatterNdUpdate<T: AccelerableByTensorFlow, Tindices: BinaryInteger>(
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
public static func scatterUpdate<T: AccelerableByTensorFlow, Tindices: BinaryInteger>(
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

@inlinable @inline(__always)
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
  numLossPartitions: Int64,
  numInnerIterations: Int64
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
    loss_type: lossType.cName,
    adaptative: adaptative,
    l1: l1,
    l2: l2,
    num_loss_partitions: numLossPartitions,
    num_inner_iterations: numInnerIterations)
}

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
public static func select<T: AccelerableByTensorFlow>(
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

@inlinable @inline(__always)
public static func selfAdjointEig<T: BinaryFloatingPoint>(
  _ input: Tensor<T>
) -> Tensor<T> {
  return #tfop("SelfAdjointEig",
    input,
    T: T.self)
}

@inlinable @inline(__always)
public static func selfAdjointEigV2<T: BinaryFloatingPoint>(
  _ input: Tensor<T>,
  computeV: Bool = true
) -> (Tensor<T>, Tensor<T>) {
  return #tfop("SelfAdjointEigV2",
    input,
    T: T.self,
    compute_v: computeV)
}

@inlinable @inline(__always)
public static func selu<T: BinaryFloatingPoint>(
  features: Tensor<T>
) -> Tensor<T> {
  return #tfop("Selu",
    features,
    T: T.self)
}

@inlinable @inline(__always)
public static func seluGrad<T: BinaryFloatingPoint>(
  gradients: Tensor<T>,
  outputs: Tensor<T>
) -> Tensor<T> {
  return #tfop("SeluGrad",
    gradients,
    outputs,
    T: T.self)
}

@inlinable @inline(__always)
public static func serializeManySparse<T: AccelerableByTensorFlow, Out_type: AccelerableByTensorFlow>(
  sparseIndices: Tensor<Int64>,
  sparseValues: Tensor<T>,
  sparseShape: Tensor<Int64>
) -> Tensor<Out_type> {
  return #tfop("SerializeManySparse",
    sparseIndices,
    sparseValues,
    sparseShape,
    T: T.self,
    out_type: Out_type.self)
}

@inlinable @inline(__always)
public static func serializeSparse<T: AccelerableByTensorFlow, Out_type: AccelerableByTensorFlow>(
  sparseIndices: Tensor<Int64>,
  sparseValues: Tensor<T>,
  sparseShape: Tensor<Int64>
) -> Tensor<Out_type> {
  return #tfop("SerializeSparse",
    sparseIndices,
    sparseValues,
    sparseShape,
    T: T.self,
    out_type: Out_type.self)
}

@inlinable @inline(__always)
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

@inlinable @inline(__always)
public static func shape<T: AccelerableByTensorFlow, Out_type: BinaryInteger>(
  _ input: Tensor<T>
) -> Tensor<Out_type> {
  return #tfop("Shape",
    input,
    T: T.self,
    out_type: Out_type.self)
}

@inlinable @inline(__always)
public static func shapeN<T: AccelerableByTensorFlow, Out_type: BinaryInteger>(
  _ input: [Tensor<T>]
) -> [Tensor<Out_type>] {
  return #tfop("ShapeN",
    input,
    T: T.self,
    out_type: Out_type.self)
}

@inlinable @inline(__always)
public static func sigmoid<T: BinaryFloatingPoint>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Sigmoid",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func sigmoidGrad<T: BinaryFloatingPoint>(
  _ y: Tensor<T>,
  dy: Tensor<T>
) -> Tensor<T> {
  return #tfop("SigmoidGrad",
    y,
    dy,
    T: T.self)
}

@inlinable @inline(__always)
public static func sign<T: Numeric>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Sign",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func simple(
  _ a: Tensor<Int32>
) -> Tensor<Float> {
  return #tfop("Simple",
    a)
}

@inlinable @inline(__always)
public static func simpleStruct(
  nA: Int64
) -> [Tensor<Int32>] {
  return #tfop("SimpleStruct",
    n_a: nA)
}

@inlinable @inline(__always)
public static func sin<T: BinaryFloatingPoint>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Sin",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func sinh<T: BinaryFloatingPoint>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Sinh",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func size<T: AccelerableByTensorFlow, Out_type: BinaryInteger>(
  _ input: Tensor<T>
) -> Tensor<Out_type> {
  return #tfop("Size",
    input,
    T: T.self,
    out_type: Out_type.self)
}

@inlinable @inline(__always)
public static func slice<T: AccelerableByTensorFlow, Index: BinaryInteger>(
  _ input: Tensor<T>,
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

@inlinable @inline(__always)
public static func snapshot<T: AccelerableByTensorFlow>(
  _ input: Tensor<T>
) -> Tensor<T> {
  return #tfop("Snapshot",
    input,
    T: T.self)
}

@inlinable @inline(__always)
public static func softmax<T: BinaryFloatingPoint>(
  logits: Tensor<T>
) -> Tensor<T> {
  return #tfop("Softmax",
    logits,
    T: T.self)
}

@inlinable @inline(__always)
public static func softmaxCrossEntropyWithLogits<T: BinaryFloatingPoint>(
  features: Tensor<T>,
  labels: Tensor<T>
) -> (Tensor<T>, Tensor<T>) {
  return #tfop("SoftmaxCrossEntropyWithLogits",
    features,
    labels,
    T: T.self)
}

@inlinable @inline(__always)
public static func softplus<T: Numeric>(
  features: Tensor<T>
) -> Tensor<T> {
  return #tfop("Softplus",
    features,
    T: T.self)
}

@inlinable @inline(__always)
public static func softplusGrad<T: Numeric>(
  gradients: Tensor<T>,
  features: Tensor<T>
) -> Tensor<T> {
  return #tfop("SoftplusGrad",
    gradients,
    features,
    T: T.self)
}

@inlinable @inline(__always)
public static func softsign<T: Numeric>(
  features: Tensor<T>
) -> Tensor<T> {
  return #tfop("Softsign",
    features,
    T: T.self)
}

@inlinable @inline(__always)
public static func softsignGrad<T: Numeric>(
  gradients: Tensor<T>,
  features: Tensor<T>
) -> Tensor<T> {
  return #tfop("SoftsignGrad",
    gradients,
    features,
    T: T.self)
}

@inlinable @inline(__always)
public static func spaceToBatch<T: AccelerableByTensorFlow, Tpaddings: BinaryInteger>(
  _ input: Tensor<T>,
  paddings: Tensor<Tpaddings>,
  blockSize: Int64
) -> Tensor<T> {
  return #tfop("SpaceToBatch",
    input,
    paddings,
    T: T.self,
    Tpaddings: Tpaddings.self,
    block_size: blockSize)
}

@inlinable @inline(__always)
public static func spaceToBatchND<T: AccelerableByTensorFlow, Tblock_shape: BinaryInteger, Tpaddings: BinaryInteger>(
  _ input: Tensor<T>,
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

@inlinable @inline(__always)
public static func spaceToDepth<T: AccelerableByTensorFlow>(
  _ input: Tensor<T>,
  blockSize: Int64,
  dataFormat: DataFormat3 = .nhwc
) -> Tensor<T> {
  return #tfop("SpaceToDepth",
    input,
    T: T.self,
    block_size: blockSize,
    data_format: dataFormat.cName)
}

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
public static func sparseApplyAdagrad<T: Numeric, Tindices: BinaryInteger>(
  var_: Tensor<T>,
  accum: Tensor<T>,
  lr: Tensor<T>,
  grad: Tensor<T>,
  indices: Tensor<Tindices>,
  useLocking: Bool = false,
  updateSlots: Bool = true
) -> Tensor<T> {
  return #tfop("SparseApplyAdagrad",
    var_,
    accum,
    lr,
    grad,
    indices,
    T: T.self,
    Tindices: Tindices.self,
    use_locking: useLocking,
    update_slots: updateSlots)
}

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
public static func sparseConcat<T: AccelerableByTensorFlow>(
  indices: [Tensor<Int64>],
  _ values: [Tensor<T>],
  shapes: [Tensor<Int64>],
  concatDim: Int64
) -> (Tensor<Int64>, Tensor<T>, Tensor<Int64>) {
  return #tfop("SparseConcat",
    indices,
    values,
    shapes,
    T: T.self,
    concat_dim: concatDim)
}

@inlinable @inline(__always)
public static func sparseCross<Sparse_types: BinaryInteger, Dense_types: BinaryInteger, Out_type: BinaryInteger, Internal_type: BinaryInteger>(
  indices: [Tensor<Int64>],
  _ values: [Tensor<Sparse_types>],
  shapes: [Tensor<Int64>],
  denseInputs: [Tensor<Dense_types>],
  hashedOutput: Bool,
  numBuckets: Int64,
  hashKey: Int64,
  typeInternal_type: Internal_type.Type
) -> (Tensor<Int64>, Tensor<Out_type>, Tensor<Int64>) {
  return #tfop("SparseCross",
    indices,
    values,
    shapes,
    denseInputs,
    out_type: Out_type.self,
    internal_type: Internal_type.self,
    hashed_output: hashedOutput,
    num_buckets: numBuckets,
    hash_key: hashKey)
}

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
public static func sparseFillEmptyRows<T: AccelerableByTensorFlow>(
  indices: Tensor<Int64>,
  _ values: Tensor<T>,
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

@inlinable @inline(__always)
public static func sparseFillEmptyRowsGrad<T: AccelerableByTensorFlow>(
  reverseIndexMap: Tensor<Int64>,
  gradValues: Tensor<T>
) -> (Tensor<T>, Tensor<T>) {
  return #tfop("SparseFillEmptyRowsGrad",
    reverseIndexMap,
    gradValues,
    T: T.self)
}

@inlinable @inline(__always)
public static func sparseMatMul<Ta: BinaryFloatingPoint, Tb: BinaryFloatingPoint>(
  _ a: Tensor<Ta>,
  _ b: Tensor<Tb>,
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
public static func sparseReorder<T: AccelerableByTensorFlow>(
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
public static func sparseSlice<T: AccelerableByTensorFlow>(
  indices: Tensor<Int64>,
  _ values: Tensor<T>,
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

@inlinable @inline(__always)
public static func sparseSliceGrad<T: Numeric>(
  backpropValGrad: Tensor<T>,
  inputIndices: Tensor<Int64>,
  inputStart: Tensor<Int64>,
  outputIndices: Tensor<Int64>
) -> Tensor<T> {
  return #tfop("SparseSliceGrad",
    backpropValGrad,
    inputIndices,
    inputStart,
    outputIndices,
    T: T.self)
}

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
public static func sparseSplit<T: AccelerableByTensorFlow>(
  splitDim: Tensor<Int64>,
  indices: Tensor<Int64>,
  _ values: Tensor<T>,
  shape: Tensor<Int64>,
  numSplit: Int64
) -> ([Tensor<Int64>], [Tensor<T>], [Tensor<Int64>]) {
  return #tfop("SparseSplit",
    splitDim,
    indices,
    values,
    shape,
    T: T.self,
    num_split: numSplit)
}

@inlinable @inline(__always)
public static func sparseTensorDenseAdd<T: Numeric, Tindices: BinaryInteger>(
  aIndices: Tensor<Tindices>,
  aValues: Tensor<T>,
  aShape: Tensor<Tindices>,
  _ b: Tensor<T>
) -> Tensor<T> {
  return #tfop("SparseTensorDenseAdd",
    aIndices,
    aValues,
    aShape,
    b,
    T: T.self,
    Tindices: Tindices.self)
}

@inlinable @inline(__always)
public static func sparseTensorDenseMatMul<T: AccelerableByTensorFlow, Tindices: BinaryInteger>(
  aIndices: Tensor<Tindices>,
  aValues: Tensor<T>,
  aShape: Tensor<Int64>,
  _ b: Tensor<T>,
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

@inlinable @inline(__always)
public static func sparseToDense<T: AccelerableByTensorFlow, Tindices: BinaryInteger>(
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
public static func split<T: AccelerableByTensorFlow>(
  splitDim: Tensor<Int32>,
  value: Tensor<T>,
  numSplit: Int64
) -> [Tensor<T>] {
  return #tfop("Split",
    splitDim,
    value,
    T: T.self,
    num_split: numSplit)
}

@inlinable @inline(__always)
public static func splitV<T: AccelerableByTensorFlow, Tlen: BinaryInteger>(
  value: Tensor<T>,
  sizeSplits: Tensor<Tlen>,
  splitDim: Tensor<Int32>,
  numSplit: Int64
) -> [Tensor<T>] {
  return #tfop("SplitV",
    value,
    sizeSplits,
    splitDim,
    T: T.self,
    Tlen: Tlen.self,
    num_split: numSplit)
}

@inlinable @inline(__always)
public static func sqrt<T: BinaryFloatingPoint>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Sqrt",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func sqrtGrad<T: BinaryFloatingPoint>(
  _ y: Tensor<T>,
  dy: Tensor<T>
) -> Tensor<T> {
  return #tfop("SqrtGrad",
    y,
    dy,
    T: T.self)
}

@inlinable @inline(__always)
public static func square<T: Numeric>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Square",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func squaredDifference<T: Numeric>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  return #tfop("SquaredDifference",
    x,
    y,
    T: T.self)
}

@inlinable @inline(__always)
public static func squeeze<T: AccelerableByTensorFlow>(
  _ input: Tensor<T>,
  squeezeDims: [Int32]
) -> Tensor<T> {
  return #tfop("Squeeze",
    input,
    T: T.self,
    squeeze_dims: squeezeDims)
}

@inlinable @inline(__always)
public static func stage<Dtypes: AccelerableByTensorFlow>(
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

@inlinable @inline(__always)
public static func stageClear<Dtypes: AccelerableByTensorFlow>(
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

@inlinable @inline(__always)
public static func stagePeek<Dtypes: AccelerableByTensorFlow>(
  index: Tensor<Int32>,
  capacity: Int64 = 0,
  memoryLimit: Int64 = 0,
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

@inlinable @inline(__always)
public static func stageSize<Dtypes: AccelerableByTensorFlow>(
  capacity: Int64 = 0,
  memoryLimit: Int64 = 0,
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

@inlinable @inline(__always)
public static func statelessMultinomial<T: Numeric, Tseed: BinaryInteger, Output_dtype: BinaryInteger>(
  logits: Tensor<T>,
  numSamples: Tensor<Int32>,
  seed: Tensor<Tseed>
) -> Tensor<Output_dtype> {
  return #tfop("StatelessMultinomial",
    logits,
    numSamples,
    seed,
    T: T.self,
    Tseed: Tseed.self,
    output_dtype: Output_dtype.self)
}

@inlinable @inline(__always)
public static func statelessRandomNormal<Dtype: BinaryFloatingPoint, T: BinaryInteger, Tseed: BinaryInteger>(
  shape: Tensor<T>,
  seed: Tensor<Tseed>
) -> Tensor<Dtype> {
  return #tfop("StatelessRandomNormal",
    shape,
    seed,
    dtype: Dtype.self,
    T: T.self,
    Tseed: Tseed.self)
}

@inlinable @inline(__always)
public static func statelessRandomUniform<Dtype: BinaryFloatingPoint, T: BinaryInteger, Tseed: BinaryInteger>(
  shape: Tensor<T>,
  seed: Tensor<Tseed>
) -> Tensor<Dtype> {
  return #tfop("StatelessRandomUniform",
    shape,
    seed,
    dtype: Dtype.self,
    T: T.self,
    Tseed: Tseed.self)
}

@inlinable @inline(__always)
public static func statelessTruncatedNormal<Dtype: BinaryFloatingPoint, T: BinaryInteger, Tseed: BinaryInteger>(
  shape: Tensor<T>,
  seed: Tensor<Tseed>
) -> Tensor<Dtype> {
  return #tfop("StatelessTruncatedNormal",
    shape,
    seed,
    dtype: Dtype.self,
    T: T.self,
    Tseed: Tseed.self)
}

@inlinable @inline(__always)
public static func stopGradient<T: AccelerableByTensorFlow>(
  _ input: Tensor<T>
) -> Tensor<T> {
  return #tfop("StopGradient",
    input,
    T: T.self)
}

@inlinable @inline(__always)
public static func stridedSlice<T: AccelerableByTensorFlow, Index: BinaryInteger>(
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

@inlinable @inline(__always)
public static func stridedSliceAssign<T: AccelerableByTensorFlow, Index: BinaryInteger>(
  ref: Tensor<T>,
  begin: Tensor<Index>,
  end: Tensor<Index>,
  strides: Tensor<Index>,
  value: Tensor<T>,
  beginMask: Int64 = 0,
  endMask: Int64 = 0,
  ellipsisMask: Int64 = 0,
  newAxisMask: Int64 = 0,
  shrinkAxisMask: Int64 = 0
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

@inlinable @inline(__always)
public static func stridedSliceGrad<T: AccelerableByTensorFlow, Index: BinaryInteger>(
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

@inlinable @inline(__always)
public static func stringListAttr(
  _ a: [String],
  _ b: String
) {
  return #tfop("StringListAttr",
    a: a,
    b: b)
}

@inlinable @inline(__always)
public static func sub<T: Numeric>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  return #tfop("Sub",
    x,
    y,
    T: T.self)
}

@inlinable @inline(__always)
public static func sum<T: Numeric, Tidx: BinaryInteger>(
  _ input: Tensor<T>,
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

@inlinable @inline(__always)
public static func svd<T: BinaryFloatingPoint>(
  _ input: Tensor<T>,
  computeUv: Bool = true,
  fullMatrices: Bool = false
) -> (Tensor<T>, Tensor<T>, Tensor<T>) {
  return #tfop("Svd",
    input,
    T: T.self,
    compute_uv: computeUv,
    full_matrices: fullMatrices)
}

@inlinable @inline(__always)
public static func switch_<T: AccelerableByTensorFlow>(
  data: Tensor<T>,
  pred: Tensor<Bool>
) -> (Tensor<T>, Tensor<T>) {
  return #tfop("Switch",
    data,
    pred,
    T: T.self)
}

@inlinable @inline(__always)
public static func takeManySparseFromTensorsMap<Dtype: AccelerableByTensorFlow>(
  sparseHandles: Tensor<Int64>,
  container: String,
  sharedName: String
) -> (Tensor<Int64>, Tensor<Dtype>, Tensor<Int64>) {
  return #tfop("TakeManySparseFromTensorsMap",
    sparseHandles,
    dtype: Dtype.self,
    container: container,
    shared_name: sharedName)
}

@inlinable @inline(__always)
public static func tan<T: Numeric>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Tan",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func tanh<T: BinaryFloatingPoint>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("Tanh",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func tanhGrad<T: BinaryFloatingPoint>(
  _ y: Tensor<T>,
  dy: Tensor<T>
) -> Tensor<T> {
  return #tfop("TanhGrad",
    y,
    dy,
    T: T.self)
}

@inlinable @inline(__always)
public static func testAttr<T: BinaryFloatingPoint>(
) -> Tensor<T> {
  return #tfop("TestAttr",
    T: T.self)
}

@inlinable @inline(__always)
public static func threadUnsafeUnigramCandidateSampler(
  trueClasses: Tensor<Int64>,
  numTrue: Int64,
  numSampled: Int64,
  unique: Bool,
  rangeMax: Int64,
  seed: Int64 = 0,
  seed2: Int64 = 0
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

@inlinable @inline(__always)
public static func tile<T: AccelerableByTensorFlow, Tmultiples: BinaryInteger>(
  _ input: Tensor<T>,
  multiples: Tensor<Tmultiples>
) -> Tensor<T> {
  return #tfop("Tile",
    input,
    multiples,
    T: T.self,
    Tmultiples: Tmultiples.self)
}

@inlinable @inline(__always)
public static func tileGrad<T: AccelerableByTensorFlow>(
  _ input: Tensor<T>,
  multiples: Tensor<Int32>
) -> Tensor<T> {
  return #tfop("TileGrad",
    input,
    multiples,
    T: T.self)
}

@inlinable @inline(__always)
public static func timestamp(
) -> Tensor<Double> {
  return #tfop("Timestamp")
}

@inlinable @inline(__always)
public static func topK<T: Numeric>(
  _ input: Tensor<T>,
  k: Int64,
  sorted: Bool = true
) -> (Tensor<T>, Tensor<Int32>) {
  return #tfop("TopK",
    input,
    T: T.self,
    k: k,
    sorted: sorted)
}

@inlinable @inline(__always)
public static func topKV2<T: Numeric>(
  _ input: Tensor<T>,
  k: Tensor<Int32>,
  sorted: Bool = true
) -> (Tensor<T>, Tensor<Int32>) {
  return #tfop("TopKV2",
    input,
    k,
    T: T.self,
    sorted: sorted)
}

@inlinable @inline(__always)
public static func transpose<T: AccelerableByTensorFlow, Tperm: BinaryInteger>(
  _ x: Tensor<T>,
  perm: Tensor<Tperm>
) -> Tensor<T> {
  return #tfop("Transpose",
    x,
    perm,
    T: T.self,
    Tperm: Tperm.self)
}

@inlinable @inline(__always)
public static func truncateDiv<T: Numeric>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  return #tfop("TruncateDiv",
    x,
    y,
    T: T.self)
}

@inlinable @inline(__always)
public static func truncateMod<T: Numeric>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> Tensor<T> {
  return #tfop("TruncateMod",
    x,
    y,
    T: T.self)
}

@inlinable @inline(__always)
public static func truncatedNormal<Dtype: BinaryFloatingPoint, T: BinaryInteger>(
  shape: Tensor<T>,
  seed: Int64 = 0,
  seed2: Int64 = 0
) -> Tensor<Dtype> {
  return #tfop("TruncatedNormal",
    shape,
    dtype: Dtype.self,
    T: T.self,
    seed: seed,
    seed2: seed2)
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
  return #tfop("TwoFloatInputsFloatOutput",
    a,
    b)
}

@inlinable @inline(__always)
public static func twoFloatInputsIntOutput(
  _ a: Tensor<Float>,
  _ b: Tensor<Float>
) -> Tensor<Int32> {
  return #tfop("TwoFloatInputsIntOutput",
    a,
    b)
}

@inlinable @inline(__always)
public static func twoFloatOutputs(
) -> (Tensor<Float>, Tensor<Float>) {
  return #tfop("TwoFloatOutputs")
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
) -> (Tensor<Int32>, Tensor<Int32>) {
  return #tfop("TwoIntOutputs")
}

@inlinable @inline(__always)
public static func twoRefsIn<T: AccelerableByTensorFlow>(
  _ a: Tensor<T>,
  _ b: Tensor<T>
) {
  return #tfop("TwoRefsIn",
    a,
    b,
    T: T.self)
}

@inlinable @inline(__always)
public static func typeList<T: AccelerableByTensorFlow>(
  _ a: [Tensor<T>]
) {
  return #tfop("TypeList",
    a)
}

@inlinable @inline(__always)
public static func typeListRestrict<T: AccelerableByTensorFlow>(
  _ a: [Tensor<T>]
) {
  return #tfop("TypeListRestrict",
    a)
}

@inlinable @inline(__always)
public static func typeListTwice<T: AccelerableByTensorFlow>(
  _ a: [Tensor<T>],
  _ b: [Tensor<T>]
) {
  return #tfop("TypeListTwice",
    a,
    b)
}

@inlinable @inline(__always)
public static func unary<T: AccelerableByTensorFlow>(
  _ a: Tensor<T>
) -> Tensor<T> {
  return #tfop("Unary",
    a,
    T: T.self)
}

@inlinable @inline(__always)
public static func unbatch<T: AccelerableByTensorFlow>(
  batchedTensor: Tensor<T>,
  batchIndex: Tensor<Int64>,
  id: Tensor<Int64>,
  timeoutMicros: Int64,
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

@inlinable @inline(__always)
public static func unbatchGrad<T: AccelerableByTensorFlow>(
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

@inlinable @inline(__always)
public static func uniformCandidateSampler(
  trueClasses: Tensor<Int64>,
  numTrue: Int64,
  numSampled: Int64,
  unique: Bool,
  rangeMax: Int64,
  seed: Int64 = 0,
  seed2: Int64 = 0
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

@inlinable @inline(__always)
public static func unique<T: AccelerableByTensorFlow, Out_idx: BinaryInteger>(
  _ x: Tensor<T>
) -> (Tensor<T>, Tensor<Out_idx>) {
  return #tfop("Unique",
    x,
    T: T.self,
    out_idx: Out_idx.self)
}

@inlinable @inline(__always)
public static func uniqueV2<T: AccelerableByTensorFlow, Taxis: BinaryInteger, Out_idx: BinaryInteger>(
  _ x: Tensor<T>,
  axis: Tensor<Taxis>
) -> (Tensor<T>, Tensor<Out_idx>) {
  return #tfop("UniqueV2",
    x,
    axis,
    T: T.self,
    Taxis: Taxis.self,
    out_idx: Out_idx.self)
}

@inlinable @inline(__always)
public static func uniqueWithCounts<T: AccelerableByTensorFlow, Out_idx: BinaryInteger>(
  _ x: Tensor<T>
) -> (Tensor<T>, Tensor<Out_idx>, Tensor<Out_idx>) {
  return #tfop("UniqueWithCounts",
    x,
    T: T.self,
    out_idx: Out_idx.self)
}

@inlinable @inline(__always)
public static func uniqueWithCountsV2<T: AccelerableByTensorFlow, Taxis: BinaryInteger, Out_idx: BinaryInteger>(
  _ x: Tensor<T>,
  axis: Tensor<Taxis>
) -> (Tensor<T>, Tensor<Out_idx>, Tensor<Out_idx>) {
  return #tfop("UniqueWithCountsV2",
    x,
    axis,
    T: T.self,
    Taxis: Taxis.self,
    out_idx: Out_idx.self)
}

@inlinable @inline(__always)
public static func unpack<T: AccelerableByTensorFlow>(
  value: Tensor<T>,
  num: Int64,
  axis: Int64 = 0
) -> [Tensor<T>] {
  return #tfop("Unpack",
    value,
    T: T.self,
    num: num,
    axis: axis)
}

@inlinable @inline(__always)
public static func unravelIndex<Tidx: BinaryInteger>(
  indices: Tensor<Tidx>,
  dims: Tensor<Tidx>
) -> Tensor<Tidx> {
  return #tfop("UnravelIndex",
    indices,
    dims,
    Tidx: Tidx.self)
}

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
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

@inlinable @inline(__always)
public static func unstage<Dtypes: AccelerableByTensorFlow>(
  capacity: Int64 = 0,
  memoryLimit: Int64 = 0,
  container: String,
  sharedName: String
) -> [Tensor<Dtypes>] {
  return #tfop("Unstage",
    capacity: capacity,
    memory_limit: memoryLimit,
    container: container,
    shared_name: sharedName)
}

@inlinable @inline(__always)
public static func where_<T: AccelerableByTensorFlow>(
  _ input: Tensor<T>
) -> Tensor<Int64> {
  return #tfop("Where",
    input,
    T: T.self)
}

@inlinable @inline(__always)
public static func zerosLike<T: AccelerableByTensorFlow>(
  _ x: Tensor<T>
) -> Tensor<T> {
  return #tfop("ZerosLike",
    x,
    T: T.self)
}

@inlinable @inline(__always)
public static func zeta<T: BinaryFloatingPoint>(
  _ x: Tensor<T>,
  q: Tensor<T>
) -> Tensor<T> {
  return #tfop("Zeta",
    x,
    q,
    T: T.self)
}

}