// !!! THIS CODE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND !!!
//
// Copyright 2018-19 Google LLC
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

import CTensorFlow

#if !COMPILING_TENSORFLOW_MODULE
import TensorFlow
#endif

internal struct TFE_Op {
  internal let status: CTFStatus
  internal let op: CTFEOp

  @inlinable
  internal init(_ name: String) {
    self.status = TF_NewStatus()
    self.op = TFE_NewOp(_ExecutionContext.global.eagerContext, name, status)
  }

  @inlinable
  internal func addInput<T: TensorArrayProtocol>(input: T) -> Int {
    let count = input._tensorHandleCount
    let buffer = UnsafeMutableBufferPointer<CTensorHandle>.allocate(capacity: Int(count))
    t._unpackTensorHandles(into: buffer.baseAddress)
    for handle in buffer {
      TFE_OpAddInput(op, handle, status)
      guard TF_GetCode(status) == TF_OK else {
        return 0
      }
    }
    buffer.deallocate()
    return Int(count)
  }

  @inlinable
  internal func setAttr(_ name: String, _ value: Bool) {
    TFE_OpSetAttrBool(op, name, value ? 1 : 0)
  }

  @inlinable
  internal func setAttr(_ name: String, _ value: Int) {
    TFE_OpSetAttrInt(op, name, Int64(value))
  }

  @inlinable
  internal func setAttr(_ name: String, _ value: Int32) {
    TFE_OpSetAttrInt(op, name, Int64(value))
  }

  @inlinable
  internal func setAttr(_ name: String, _ value: Int64) {
    TFE_OpSetAttrInt(op, name, value)
  }

  @inlinable
  internal func setAttr(_ name: String, _ value: Float) {
    TFE_OpSetAttrFloat(op, name, value)
  }

  @inlinable
  internal func setAttr(_ name: String, _ value: Double) {
    TFE_OpSetAttrFloat(op, name, Float(value))
  }

  @inlinable
  internal func setAttr(_ name: String, _ value: String) {
    value.utf8CString.withUnsafeBufferPointer { buffer in
      // utf8CString is null-terminated; TFE_OpSetAttrString wants
      // non-null-terminated.
      TFE_OpSetAttrString(op, name, buffer.baseAddress, buffer.count - 1)
    }
  }

  @inlinable
  internal func setAttr(_ name: String, _ value: TensorDataType) {
    TFE_OpSetAttrType(op, name, value._cDataType)
  }

  @inlinable
  internal func setAttr(_ name: String, _ value: TensorShape) {
    let dimensions: [Int64] = value.dimensions.map(Int64.init)
    dimensions.withUnsafeBufferPointer { buffer in
      TFE_OpSetAttrShape(op, name, buffer.baseAddress, Int32(buffer.count), status)
    }
  }

  @inlinable
  internal func setAttr(_ name: String, _ value: TensorShape?) {
    guard let shape = value else {
      TFE_OpSetAttrShape(op, name, nil, -1, status)
      return
    }
    setAttr(name, shape)
  }

  @inlinable
  internal func setAttr(_ name: String, _ value: [Bool]) {
    value.map({ $0 ? UInt8(1) : UInt8(0) }).withUnsafeBufferPointer { buffer in
      TFE_OpSetAttrBoolList(op, name, buffer.baseAddress, Int32(buffer.count))
    }
  }

  @inlinable
  internal func setAttr(_ name: String, _ value: [Int]) {
    setAttr(name, value.map(Int64.init))
  }

  @inlinable
  internal func setAttr(_ name: String, _ value: [Int32]) {
    setAttr(name, value.map(Int64.init))
  }

  @inlinable
  internal func setAttr(_ name: String, _ value: [Int64]) {
    value.withUnsafeBufferPointer { buffer in
      TFE_OpSetAttrIntList(op, name, buffer.baseAddress, Int32(buffer.count))
    }
  }

  @inlinable
  internal func setAttr(_ name: String, _ value: [Float]) {
    value.withUnsafeBufferPointer { buffer in
      TFE_OpSetAttrFloatList(op, name, buffer.baseAddress, Int32(buffer.count))
    }
  }

  @inlinable
  internal func setAttr(_ name: String, _ value: [Double]) {
    setAttr(name, value.map(Float.init))
  }

  @inlinable
  internal func setAttr(_ name: String, _ value: [String]) {
    // Collect all the strings' utf8 bytes into a single array so that we can
    // address all the strings with a single
    // `flattenedStringBytes.withUnsafeBufferPointer`.
    var flattenedStringBytes: [CChar] = []
    var lengths: [Int] = []
    for string in value {
      // Don't include the null-terminator because TFE_OpSetAttrStringList uses
      // lengths instead of null-terminators.
      let stringBytes = string.utf8CString.dropLast()
      flattenedStringBytes.append(contentsOf: stringBytes)
      lengths.append(stringBytes.count)
    }

    // Calculate the addresses of all the strings within our single buffer, and
    // then call TFE_OpSetAttrStringList.
    flattenedStringBytes.withUnsafeBufferPointer { flattenedStringBytesBuffer in
      var stringAddrs: [UnsafeRawPointer?] = []
      var currentStringAddr =
        flattenedStringBytesBuffer.baseAddress.map(UnsafeRawPointer.init)
      for length in lengths {
        stringAddrs.append(currentStringAddr)
        currentStringAddr = currentStringAddr?.advanced(by: length)
      }

      stringAddrs.withUnsafeBufferPointer { stringAddrsBuffer in
        lengths.withUnsafeBufferPointer { lengthsBuffer in
          TFE_OpSetAttrStringList(op, name, stringAddrsBuffer.baseAddress,
            lengthsBuffer.baseAddress, Int32(value.count))
        }
      }
    }
  }

  @inlinable
  internal func setAttr(_ name: String, _ value: [TensorDataType]) {
    value.withUnsafeBufferPointer { buffer in
      buffer.withMemoryRebound(to: TF_DataType.self) { reboundBuffer in
        TFE_OpSetAttrTypeList(op, name, reboundBuffer.baseAddress, Int32(reboundBuffer.count))
      }
    }
  }

  @inlinable
  internal func setAttr(_ name: String, _ value: [TensorShape]) {
    let flattenedDims = value.flatMap { $0.dimensions.map(Int64.init) }
    let ranks = value.map { Int32($0.rank) }
    flattenedDims.withUnsafeBufferPointer { flattenedDimsBuffer in
      var dimsPtr: UnsafePointer<Int64>? = flattenedDimsBuffer.baseAddress
      var dims: [UnsafePointer<Int64>?] = []
      for rank in ranks {
        dims.append(dimsPtr)
        if rank >= 0 {
          dimsPtr = dimsPtr.map { $0.advanced(by: Int(rank)) }
        }
      }
      dims.withUnsafeMutableBufferPointer { dimsBuffer in
        ranks.withUnsafeBufferPointer { ranksBuffer in
          TFE_OpSetAttrShapeList(
            op, name, dimsBuffer.baseAddress, ranksBuffer.baseAddress,
            Int32(ranksBuffer.count), status)
        }
      }
    }
  }

  @inlinable
  internal func setAttr(_ name: String, _ value: [TensorShape?]) {
    let flattenedDims = value.flatMap { (tensorShapeOpt) -> [Int64] in
      if let tensorShape = tensorShapeOpt {
        return tensorShape.dimensions.map(Int64.init)
      }
      return []
    }
    let ranks = value.map { shape in (shape?.rank).map(Int32.init) ?? -1 }
    flattenedDims.withUnsafeBufferPointer { flattenedDimsBuffer in
      var dimsPtr: UnsafePointer<Int64>? = flattenedDimsBuffer.baseAddress
      var dims: [UnsafePointer<Int64>?] = []
      for rank in ranks {
        dims.append(dimsPtr)
        if rank >= 0 {
          dimsPtr = dimsPtr.map { $0.advanced(by: Int(rank)) }
        }
      }
      dims.withUnsafeMutableBufferPointer { dimsBuffer in
        ranks.withUnsafeBufferPointer { ranksBuffer in
          TFE_OpSetAttrShapeList(
            op, name, dimsBuffer.baseAddress, ranksBuffer.baseAddress,
            Int32(ranksBuffer.count), status)
        }
      }
    }
  }

  @inlinable
  internal func setAttr<In: TensorGroup, Out: TensorGroup>(_ name: String, _ value: (In) -> Out) {
    _tffunc(value).utf8CString.withUnsafeBufferPointer { buffer in
      // utf8CString is null-terminated; TFE_OpSetAttrFunctionName wants
      // non-null-terminated.
      TFE_OpSetAttrFunctionName(op, name, buffer.baseAddress, buffer.count - 1)
    }
  }

  @inlinable
  internal func execute() {
    var count: Int32 = 0
    var unused: CTensorHandle?
    _TFCEagerExecute(op, &unused, &count, status)
    checkOk(status)
    buffer.deallocate()
    TFE_DeleteOp(op)
    TF_DeleteStatus(status)
  }

  /// NOTE: Any of the following functions can only be executed once.
  @inlinable
  internal func execute<T0 : TensorArrayProtocol>(
    count0: Int
  ) -> (T0) {
    var count = Int32(count0)
    let buffer: UnsafeMutablePointer<CTensorHandle> =
      UnsafeMutablePointer.allocate(capacity: Int(count))
    _TFCEagerExecute(op, UnsafeMutablePointer<CTensorHandle?>(buffer), &count, status)
    checkOk(status)
    let offset0 = 0
    let result = (
      T0.init(_owning: buffer.advanced(by: Int(offset0)), count: count0))
    buffer.deallocate()
    TFE_DeleteOp(op)
    TF_DeleteStatus(status)
    return result
  }

  @inlinable
  internal func execute<T0 : TensorArrayProtocol, T1 : TensorArrayProtocol>(
    count0: Int,
    count1: Int
  ) -> (T0, T1) {
    var count = Int32(count0) + Int32(count1)
    let buffer: UnsafeMutablePointer<CTensorHandle> =
      UnsafeMutablePointer.allocate(capacity: Int(count))
    _TFCEagerExecute(op, UnsafeMutablePointer<CTensorHandle?>(buffer), &count, status)
    checkOk(status)
    let offset0 = 0
    let offset1 = offset0 + Int32(count0)
    let result = (
      T0.init(_owning: buffer.advanced(by: Int(offset0)), count: count0),
      T1.init(_owning: buffer.advanced(by: Int(offset1)), count: count1))
    buffer.deallocate()
    TFE_DeleteOp(op)
    TF_DeleteStatus(status)
    return result
  }

  @inlinable
  internal func execute<T0 : TensorArrayProtocol, T1 : TensorArrayProtocol, T2 : TensorArrayProtocol>(
    count0: Int,
    count1: Int,
    count2: Int
  ) -> (T0, T1, T2) {
    var count = Int32(count0) + Int32(count1) + Int32(count2)
    let buffer: UnsafeMutablePointer<CTensorHandle> =
      UnsafeMutablePointer.allocate(capacity: Int(count))
    _TFCEagerExecute(op, UnsafeMutablePointer<CTensorHandle?>(buffer), &count, status)
    checkOk(status)
    let offset0 = 0
    let offset1 = offset0 + Int32(count0)
    let offset2 = offset1 + Int32(count1)
    let result = (
      T0.init(_owning: buffer.advanced(by: Int(offset0)), count: count0),
      T1.init(_owning: buffer.advanced(by: Int(offset1)), count: count1),
      T2.init(_owning: buffer.advanced(by: Int(offset2)), count: count2))
    buffer.deallocate()
    TFE_DeleteOp(op)
    TF_DeleteStatus(status)
    return result
  }

  @inlinable
  internal func execute<T0 : TensorArrayProtocol, T1 : TensorArrayProtocol, T2 : TensorArrayProtocol, T3 : TensorArrayProtocol>(
    count0: Int,
    count1: Int,
    count2: Int,
    count3: Int
  ) -> (T0, T1, T2, T3) {
    var count = Int32(count0) + Int32(count1) + Int32(count2) + Int32(count3)
    let buffer: UnsafeMutablePointer<CTensorHandle> =
      UnsafeMutablePointer.allocate(capacity: Int(count))
    _TFCEagerExecute(op, UnsafeMutablePointer<CTensorHandle?>(buffer), &count, status)
    checkOk(status)
    let offset0 = 0
    let offset1 = offset0 + Int32(count0)
    let offset2 = offset1 + Int32(count1)
    let offset3 = offset2 + Int32(count2)
    let result = (
      T0.init(_owning: buffer.advanced(by: Int(offset0)), count: count0),
      T1.init(_owning: buffer.advanced(by: Int(offset1)), count: count1),
      T2.init(_owning: buffer.advanced(by: Int(offset2)), count: count2),
      T3.init(_owning: buffer.advanced(by: Int(offset3)), count: count3))
    buffer.deallocate()
    TFE_DeleteOp(op)
    TF_DeleteStatus(status)
    return result
  }

  @inlinable
  internal func execute<T0 : TensorArrayProtocol, T1 : TensorArrayProtocol, T2 : TensorArrayProtocol, T3 : TensorArrayProtocol, T4 : TensorArrayProtocol>(
    count0: Int,
    count1: Int,
    count2: Int,
    count3: Int,
    count4: Int
  ) -> (T0, T1, T2, T3, T4) {
    var count = Int32(count0) + Int32(count1) + Int32(count2) + Int32(count3) + Int32(count4)
    let buffer: UnsafeMutablePointer<CTensorHandle> =
      UnsafeMutablePointer.allocate(capacity: Int(count))
    _TFCEagerExecute(op, UnsafeMutablePointer<CTensorHandle?>(buffer), &count, status)
    checkOk(status)
    let offset0 = 0
    let offset1 = offset0 + Int32(count0)
    let offset2 = offset1 + Int32(count1)
    let offset3 = offset2 + Int32(count2)
    let offset4 = offset3 + Int32(count3)
    let result = (
      T0.init(_owning: buffer.advanced(by: Int(offset0)), count: count0),
      T1.init(_owning: buffer.advanced(by: Int(offset1)), count: count1),
      T2.init(_owning: buffer.advanced(by: Int(offset2)), count: count2),
      T3.init(_owning: buffer.advanced(by: Int(offset3)), count: count3),
      T4.init(_owning: buffer.advanced(by: Int(offset4)), count: count4))
    buffer.deallocate()
    TFE_DeleteOp(op)
    TF_DeleteStatus(status)
    return result
  }

  @inlinable
  internal func execute<T0 : TensorArrayProtocol, T1 : TensorArrayProtocol, T2 : TensorArrayProtocol, T3 : TensorArrayProtocol, T4 : TensorArrayProtocol, T5 : TensorArrayProtocol>(
    count0: Int,
    count1: Int,
    count2: Int,
    count3: Int,
    count4: Int,
    count5: Int
  ) -> (T0, T1, T2, T3, T4, T5) {
    var count = Int32(count0) + Int32(count1) + Int32(count2) + Int32(count3) + Int32(count4) + Int32(count5)
    let buffer: UnsafeMutablePointer<CTensorHandle> =
      UnsafeMutablePointer.allocate(capacity: Int(count))
    _TFCEagerExecute(op, UnsafeMutablePointer<CTensorHandle?>(buffer), &count, status)
    checkOk(status)
    let offset0 = 0
    let offset1 = offset0 + Int32(count0)
    let offset2 = offset1 + Int32(count1)
    let offset3 = offset2 + Int32(count2)
    let offset4 = offset3 + Int32(count3)
    let offset5 = offset4 + Int32(count4)
    let result = (
      T0.init(_owning: buffer.advanced(by: Int(offset0)), count: count0),
      T1.init(_owning: buffer.advanced(by: Int(offset1)), count: count1),
      T2.init(_owning: buffer.advanced(by: Int(offset2)), count: count2),
      T3.init(_owning: buffer.advanced(by: Int(offset3)), count: count3),
      T4.init(_owning: buffer.advanced(by: Int(offset4)), count: count4),
      T5.init(_owning: buffer.advanced(by: Int(offset5)), count: count5))
    buffer.deallocate()
    TFE_DeleteOp(op)
    TF_DeleteStatus(status)
    return result
  }

  @inlinable
  internal func execute<T0 : TensorArrayProtocol, T1 : TensorArrayProtocol, T2 : TensorArrayProtocol, T3 : TensorArrayProtocol, T4 : TensorArrayProtocol, T5 : TensorArrayProtocol, T6 : TensorArrayProtocol>(
    count0: Int,
    count1: Int,
    count2: Int,
    count3: Int,
    count4: Int,
    count5: Int,
    count6: Int
  ) -> (T0, T1, T2, T3, T4, T5, T6) {
    var count = Int32(count0) + Int32(count1) + Int32(count2) + Int32(count3) + Int32(count4) + Int32(count5) + Int32(count6)
    let buffer: UnsafeMutablePointer<CTensorHandle> =
      UnsafeMutablePointer.allocate(capacity: Int(count))
    _TFCEagerExecute(op, UnsafeMutablePointer<CTensorHandle?>(buffer), &count, status)
    checkOk(status)
    let offset0 = 0
    let offset1 = offset0 + Int32(count0)
    let offset2 = offset1 + Int32(count1)
    let offset3 = offset2 + Int32(count2)
    let offset4 = offset3 + Int32(count3)
    let offset5 = offset4 + Int32(count4)
    let offset6 = offset5 + Int32(count5)
    let result = (
      T0.init(_owning: buffer.advanced(by: Int(offset0)), count: count0),
      T1.init(_owning: buffer.advanced(by: Int(offset1)), count: count1),
      T2.init(_owning: buffer.advanced(by: Int(offset2)), count: count2),
      T3.init(_owning: buffer.advanced(by: Int(offset3)), count: count3),
      T4.init(_owning: buffer.advanced(by: Int(offset4)), count: count4),
      T5.init(_owning: buffer.advanced(by: Int(offset5)), count: count5),
      T6.init(_owning: buffer.advanced(by: Int(offset6)), count: count6))
    buffer.deallocate()
    TFE_DeleteOp(op)
    TF_DeleteStatus(status)
    return result
  }

  @inlinable
  internal func execute<T0 : TensorArrayProtocol, T1 : TensorArrayProtocol, T2 : TensorArrayProtocol, T3 : TensorArrayProtocol, T4 : TensorArrayProtocol, T5 : TensorArrayProtocol, T6 : TensorArrayProtocol, T7 : TensorArrayProtocol>(
    count0: Int,
    count1: Int,
    count2: Int,
    count3: Int,
    count4: Int,
    count5: Int,
    count6: Int,
    count7: Int
  ) -> (T0, T1, T2, T3, T4, T5, T6, T7) {
    var count = Int32(count0) + Int32(count1) + Int32(count2) + Int32(count3) + Int32(count4) + Int32(count5) + Int32(count6) + Int32(count7)
    let buffer: UnsafeMutablePointer<CTensorHandle> =
      UnsafeMutablePointer.allocate(capacity: Int(count))
    _TFCEagerExecute(op, UnsafeMutablePointer<CTensorHandle?>(buffer), &count, status)
    checkOk(status)
    let offset0 = 0
    let offset1 = offset0 + Int32(count0)
    let offset2 = offset1 + Int32(count1)
    let offset3 = offset2 + Int32(count2)
    let offset4 = offset3 + Int32(count3)
    let offset5 = offset4 + Int32(count4)
    let offset6 = offset5 + Int32(count5)
    let offset7 = offset6 + Int32(count6)
    let result = (
      T0.init(_owning: buffer.advanced(by: Int(offset0)), count: count0),
      T1.init(_owning: buffer.advanced(by: Int(offset1)), count: count1),
      T2.init(_owning: buffer.advanced(by: Int(offset2)), count: count2),
      T3.init(_owning: buffer.advanced(by: Int(offset3)), count: count3),
      T4.init(_owning: buffer.advanced(by: Int(offset4)), count: count4),
      T5.init(_owning: buffer.advanced(by: Int(offset5)), count: count5),
      T6.init(_owning: buffer.advanced(by: Int(offset6)), count: count6),
      T7.init(_owning: buffer.advanced(by: Int(offset7)), count: count7))
    buffer.deallocate()
    TFE_DeleteOp(op)
    TF_DeleteStatus(status)
    return result
  }

  @inlinable
  internal func execute<T0 : TensorArrayProtocol, T1 : TensorArrayProtocol, T2 : TensorArrayProtocol, T3 : TensorArrayProtocol, T4 : TensorArrayProtocol, T5 : TensorArrayProtocol, T6 : TensorArrayProtocol, T7 : TensorArrayProtocol, T8 : TensorArrayProtocol>(
    count0: Int,
    count1: Int,
    count2: Int,
    count3: Int,
    count4: Int,
    count5: Int,
    count6: Int,
    count7: Int,
    count8: Int
  ) -> (T0, T1, T2, T3, T4, T5, T6, T7, T8) {
    var count = Int32(count0) + Int32(count1) + Int32(count2) + Int32(count3) + Int32(count4) + Int32(count5) + Int32(count6) + Int32(count7) + Int32(count8)
    let buffer: UnsafeMutablePointer<CTensorHandle> =
      UnsafeMutablePointer.allocate(capacity: Int(count))
    _TFCEagerExecute(op, UnsafeMutablePointer<CTensorHandle?>(buffer), &count, status)
    checkOk(status)
    let offset0 = 0
    let offset1 = offset0 + Int32(count0)
    let offset2 = offset1 + Int32(count1)
    let offset3 = offset2 + Int32(count2)
    let offset4 = offset3 + Int32(count3)
    let offset5 = offset4 + Int32(count4)
    let offset6 = offset5 + Int32(count5)
    let offset7 = offset6 + Int32(count6)
    let offset8 = offset7 + Int32(count7)
    let result = (
      T0.init(_owning: buffer.advanced(by: Int(offset0)), count: count0),
      T1.init(_owning: buffer.advanced(by: Int(offset1)), count: count1),
      T2.init(_owning: buffer.advanced(by: Int(offset2)), count: count2),
      T3.init(_owning: buffer.advanced(by: Int(offset3)), count: count3),
      T4.init(_owning: buffer.advanced(by: Int(offset4)), count: count4),
      T5.init(_owning: buffer.advanced(by: Int(offset5)), count: count5),
      T6.init(_owning: buffer.advanced(by: Int(offset6)), count: count6),
      T7.init(_owning: buffer.advanced(by: Int(offset7)), count: count7),
      T8.init(_owning: buffer.advanced(by: Int(offset8)), count: count8))
    buffer.deallocate()
    TFE_DeleteOp(op)
    TF_DeleteStatus(status)
    return result
  }

  @inlinable
  internal func execute<T0 : TensorArrayProtocol, T1 : TensorArrayProtocol, T2 : TensorArrayProtocol, T3 : TensorArrayProtocol, T4 : TensorArrayProtocol, T5 : TensorArrayProtocol, T6 : TensorArrayProtocol, T7 : TensorArrayProtocol, T8 : TensorArrayProtocol, T9 : TensorArrayProtocol>(
    count0: Int,
    count1: Int,
    count2: Int,
    count3: Int,
    count4: Int,
    count5: Int,
    count6: Int,
    count7: Int,
    count8: Int,
    count9: Int
  ) -> (T0, T1, T2, T3, T4, T5, T6, T7, T8, T9) {
    var count = Int32(count0) + Int32(count1) + Int32(count2) + Int32(count3) + Int32(count4) + Int32(count5) + Int32(count6) + Int32(count7) + Int32(count8) + Int32(count9)
    let buffer: UnsafeMutablePointer<CTensorHandle> =
      UnsafeMutablePointer.allocate(capacity: Int(count))
    _TFCEagerExecute(op, UnsafeMutablePointer<CTensorHandle?>(buffer), &count, status)
    checkOk(status)
    let offset0 = 0
    let offset1 = offset0 + Int32(count0)
    let offset2 = offset1 + Int32(count1)
    let offset3 = offset2 + Int32(count2)
    let offset4 = offset3 + Int32(count3)
    let offset5 = offset4 + Int32(count4)
    let offset6 = offset5 + Int32(count5)
    let offset7 = offset6 + Int32(count6)
    let offset8 = offset7 + Int32(count7)
    let offset9 = offset8 + Int32(count8)
    let result = (
      T0.init(_owning: buffer.advanced(by: Int(offset0)), count: count0),
      T1.init(_owning: buffer.advanced(by: Int(offset1)), count: count1),
      T2.init(_owning: buffer.advanced(by: Int(offset2)), count: count2),
      T3.init(_owning: buffer.advanced(by: Int(offset3)), count: count3),
      T4.init(_owning: buffer.advanced(by: Int(offset4)), count: count4),
      T5.init(_owning: buffer.advanced(by: Int(offset5)), count: count5),
      T6.init(_owning: buffer.advanced(by: Int(offset6)), count: count6),
      T7.init(_owning: buffer.advanced(by: Int(offset7)), count: count7),
      T8.init(_owning: buffer.advanced(by: Int(offset8)), count: count8),
      T9.init(_owning: buffer.advanced(by: Int(offset9)), count: count9))
    buffer.deallocate()
    TFE_DeleteOp(op)
    TF_DeleteStatus(status)
    return result
  }

}