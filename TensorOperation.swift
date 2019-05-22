public protocol TensorOperation {
  // TODO(bgogul): Enable this when opaque return types are
  /// generally available
  // associatedtype TensorValueHandle

  // We use functions instead of fields to give freedom in the
  // representation for the conforming types.
  init(_ name: String, _ outputCount: Int)

  // TODO(bgogul): Enable this when opaque return types are
  /// generally available
  // func addInput(_ input : TensorValueHandle)

  func updateAttribute(_ name: String, _ value: Bool)
  func updateAttribute(_ name: String, _ value: Int)
  func updateAttribute(_ name: String, _ value: Int32)
  func updateAttribute(_ name: String, _ value: Int64)
  func updateAttribute(_ name: String, _ value: Float)
  func updateAttribute(_ name: String, _ value: Double)
  func updateAttribute(_ name: String, _ value: String)
  func updateAttribute(_ name: String, _ value: [Bool])
  func updateAttribute(_ name: String, _ value: [Int])
  func updateAttribute(_ name: String, _ value: [Int32])
  func updateAttribute(_ name: String, _ value: [Int64])
  func updateAttribute(_ name: String, _ value: [Float])
  func updateAttribute(_ name: String, _ value: [Double])
  func updateAttribute(_ name: String, _ value: [String])

  // TODO(bgogul): Enable this when opaque return types are
  /// generally available
  // func evaluate() -> ([TensorValueHandle])
}
