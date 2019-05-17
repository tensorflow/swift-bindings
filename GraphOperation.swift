public protocol GraphOperation {
  associatedtype TensorValueHandle
  // We use functions instead of fields to give freedom in the
  // representation for the conforming types.
  init(_ name: String, _ nOutputs: Int)

  // TODO: addInput does not need to return a value, but leaving it
  // for now so that it is compatible with the generator script.
  // We can change `generated_wrappers.py` once we reach a consensus.
  func addInput(_ input : TensorValueHandle) -> Int
  func setAttr(_ name: String, _ value: Bool)
  func setAttr(_ name: String, _ value: Int)
  func setAttr(_ name: String, _ value: Int32)
  func setAttr(_ name: String, _ value: Int64)
  func setAttr(_ name: String, _ value: Float)
  func setAttr(_ name: String, _ value: Double)
  func setAttr(_ name: String, _ value: String)
  func setAttr(_ name: String, _ value: [Bool])
  func setAttr(_ name: String, _ value: [Int])
  func setAttr(_ name: String, _ value: [Int32])
  func setAttr(_ name: String, _ value: [Int64])
  func setAttr(_ name: String, _ value: [Float])
  func setAttr(_ name: String, _ value: [Double])
  func setAttr(_ name: String, _ value: [String])

  func evaluate() -> ([TensorValueHandle])
}
