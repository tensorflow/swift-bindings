extension _ExecutionContext {
  // The execution mode is effectively encoded in the GraphOperation.
  // We can use this to switch switch between different execution modes.
  // TODO: Can we interop between modes?
  public static func makeOp(_ name: String, _ nOutputs: Int) -> some TensorFlowGraphOperation {
    return TFE_Op(name, nOutputs)
  }
}

/// A handle that is compatible with the TensorFlow library.
public protocol TensorFlowHandle {
  var _cTensorHandle: CTensorHandle {get}
}

extension _AnyTensorHandle : TensorFlowHandle {}

/// A graph operation that is compatible with the Tensor library.
public protocol TensorFlowGraphOperation : GraphOperation
where TensorValueHandle : TensorFlowHandle {
  // TODO: addInput does not need to return a value, but leaving it
  // for now so that it is compatible with the generator script.
  // We can change `generated_wrappers.py` once we reach a consensus.
  func addInput<Scalar: TensorFlowScalar>(_ input: Tensor<Scalar>) -> Int
  func addInput(_ input: StringTensor) -> Int
  func addInput(_ input: VariantHandle) -> Int
  func addInput(_ input: ResourceHandle) -> Int
  func addInputList<T: TensorArrayProtocol>(_ input: T) -> Int

  func setAttr(_ name: String, _ value: TensorDataType)
  func setAttr(_ name: String, _ value: TensorShape)
  func setAttr(_ name: String, _ value: TensorShape?)
  func setAttr(_ name: String, _ value: [TensorDataType])
  func setAttr(_ name: String, _ value: [TensorShape])
  func setAttr(_ name: String, _ value: [TensorShape?])
  func setAttr<In: TensorGroup, Out: TensorGroup>(_ name: String, _ value: (In) -> Out)

  func execute()

  func execute<T0 : TensorArrayProtocol>(
    _ count0: Int
  ) -> (T0)

  func execute<T0 : TensorArrayProtocol, T1 : TensorArrayProtocol>(
    _ count0: Int,
    _ count1: Int
  ) -> (T0, T1)

  func execute<T0 : TensorArrayProtocol, T1 : TensorArrayProtocol, T2 : TensorArrayProtocol>(
    _ count0: Int,
    _ count1: Int,
    _ count2: Int
  ) -> (T0, T1, T2)

  func execute<T0 : TensorArrayProtocol, T1 : TensorArrayProtocol, T2 : TensorArrayProtocol, T3 : TensorArrayProtocol>(
    _ count0: Int,
    _ count1: Int,
    _ count2: Int,
    _ count3: Int
  ) -> (T0, T1, T2, T3)

  func execute<T0 : TensorArrayProtocol, T1 : TensorArrayProtocol, T2 : TensorArrayProtocol, T3 : TensorArrayProtocol, T4 : TensorArrayProtocol>(
    _ count0: Int,
    _ count1: Int,
    _ count2: Int,
    _ count3: Int,
    _ count4: Int
  ) -> (T0, T1, T2, T3, T4)

  func execute<T0 : TensorArrayProtocol, T1 : TensorArrayProtocol, T2 : TensorArrayProtocol, T3 : TensorArrayProtocol, T4 : TensorArrayProtocol, T5 : TensorArrayProtocol>(
    _ count0: Int,
    _ count1: Int,
    _ count2: Int,
    _ count3: Int,
    _ count4: Int,
    _ count5: Int
  ) -> (T0, T1, T2, T3, T4, T5)

  func execute<T0 : TensorArrayProtocol, T1 : TensorArrayProtocol, T2 : TensorArrayProtocol, T3 : TensorArrayProtocol, T4 : TensorArrayProtocol, T5 : TensorArrayProtocol, T6 : TensorArrayProtocol>(
    _ count0: Int,
    _ count1: Int,
    _ count2: Int,
    _ count3: Int,
    _ count4: Int,
    _ count5: Int,
    _ count6: Int
  ) -> (T0, T1, T2, T3, T4, T5, T6)
    
  func execute<T0 : TensorArrayProtocol, T1 : TensorArrayProtocol, T2 : TensorArrayProtocol, T3 : TensorArrayProtocol, T4 : TensorArrayProtocol, T5 : TensorArrayProtocol, T6 : TensorArrayProtocol, T7 : TensorArrayProtocol>(
    _ count0: Int,
    _ count1: Int,
    _ count2: Int,
    _ count3: Int,
    _ count4: Int,
    _ count5: Int,
    _ count6: Int,
    _ count7: Int
  ) -> (T0, T1, T2, T3, T4, T5, T6, T7)

  func execute<T0 : TensorArrayProtocol, T1 : TensorArrayProtocol, T2 : TensorArrayProtocol, T3 : TensorArrayProtocol, T4 : TensorArrayProtocol, T5 : TensorArrayProtocol, T6 : TensorArrayProtocol, T7 : TensorArrayProtocol, T8 : TensorArrayProtocol>(
    _ count0: Int,
    _ count1: Int,
    _ count2: Int,
    _ count3: Int,
    _ count4: Int,
    _ count5: Int,
    _ count6: Int,
    _ count7: Int,
    _ count8: Int
  ) -> (T0, T1, T2, T3, T4, T5, T6, T7, T8)
  
  func execute<T0 : TensorArrayProtocol, T1 : TensorArrayProtocol, T2 : TensorArrayProtocol, T3 : TensorArrayProtocol, T4 : TensorArrayProtocol, T5 : TensorArrayProtocol, T6 : TensorArrayProtocol, T7 : TensorArrayProtocol, T8 : TensorArrayProtocol, T9 : TensorArrayProtocol>(
    _ count0: Int,
    _ count1: Int,
    _ count2: Int,
    _ count3: Int,
    _ count4: Int,
    _ count5: Int,
    _ count6: Int,
    _ count7: Int,
    _ count8: Int,
    _ count9: Int
  ) -> (T0, T1, T2, T3, T4, T5, T6, T7, T8, T9)

  // The following should work as well.
  // We can change `generated_wrappers.py` once we reach a consensus.
  // 
  // func execute0()
  // func execute1() -> (TensorArrayProtocol)
  // func execute2() -> (TensorArrayProtocol, TensorArrayProtocol)
  // func execute3() -> (
  //   TensorArrayProtocol, TensorArrayProtocol, TensorArrayProtocol)
  // func execute4() -> (
  //   TensorArrayProtocol, TensorArrayProtocol,
  //   TensorArrayProtocol, TensorArrayProtocol)
  // func execute5() -> (
  //   TensorArrayProtocol, TensorArrayProtocol,
  //   TensorArrayProtocol, TensorArrayProtocol,
  //   TensorArrayProtocol
  // )
  // func execute6() -> (
  //   TensorArrayProtocol, TensorArrayProtocol,
  //   TensorArrayProtocol, TensorArrayProtocol,
  //   TensorArrayProtocol, TensorArrayProtocol)
  // func execute7() -> (
  //   TensorArrayProtocol, TensorArrayProtocol,
  //   TensorArrayProtocol, TensorArrayProtocol,
  //   TensorArrayProtocol, TensorArrayProtocol,
  //   TensorArrayProtocol
  // )
  // func execute8() -> (
  //   TensorArrayProtocol, TensorArrayProtocol,
  //   TensorArrayProtocol, TensorArrayProtocol,
  //   TensorArrayProtocol, TensorArrayProtocol,
  //   TensorArrayProtocol, TensorArrayProtocol)
  // func execute9() -> (
  //   TensorArrayProtocol, TensorArrayProtocol,
  //   TensorArrayProtocol, TensorArrayProtocol,
  //   TensorArrayProtocol, TensorArrayProtocol,
  //   TensorArrayProtocol, TensorArrayProtocol,
  //   TensorArrayProtocol
  // )
  // func execute10() -> (
  //   TensorArrayProtocol, TensorArrayProtocol,
  //   TensorArrayProtocol, TensorArrayProtocol,
  //   TensorArrayProtocol, TensorArrayProtocol,
  //   TensorArrayProtocol, TensorArrayProtocol,
  //   TensorArrayProtocol, TensorArrayProtocol)
}
