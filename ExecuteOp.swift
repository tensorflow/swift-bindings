// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 1)
@usableFromInline
func TensorGroupExecuteOp(_ op: CTFEOp, _ s: CTFStatus) {
  var count: Int32 = 0
  var unused: CTensorHandle?
  _TFCEagerExecute(op, &unused, &count, s)
  checkOk(s)
}

// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 10)
@usableFromInline
func TensorGroupExecuteOp<T0 : TensorGroup>
  (_ op: CTFEOp, _ s: CTFStatus)
  -> (T0) {

  var count: Int32 = T0._tensorHandleCount
  let buffer: UnsafeMutablePointer<CTensorHandle> =
    UnsafeMutablePointer.allocate(capacity: Int(count))
  defer { buffer.deallocate() }
  _TFCEagerExecute(op, UnsafeMutablePointer<CTensorHandle?>(buffer), &count, s)
  checkOk(s)
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off0: Int32 = 0
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 24)
  return (T0.init(_owning: buffer.advanced(by: Int(off0))))
}
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 10)
@usableFromInline
func TensorGroupExecuteOp<T0 : TensorGroup, T1 : TensorGroup>
  (_ op: CTFEOp, _ s: CTFStatus)
  -> (T0, T1) {

  var count: Int32 = T0._tensorHandleCount + T1._tensorHandleCount
  let buffer: UnsafeMutablePointer<CTensorHandle> =
    UnsafeMutablePointer.allocate(capacity: Int(count))
  defer { buffer.deallocate() }
  _TFCEagerExecute(op, UnsafeMutablePointer<CTensorHandle?>(buffer), &count, s)
  checkOk(s)
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off0: Int32 = 0
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off1: Int32 = off0 + T0._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 24)
  return (T0.init(_owning: buffer.advanced(by: Int(off0))), T1.init(_owning: buffer.advanced(by: Int(off1))))
}
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 10)
@usableFromInline
func TensorGroupExecuteOp<T0 : TensorGroup, T1 : TensorGroup, T2 : TensorGroup>
  (_ op: CTFEOp, _ s: CTFStatus)
  -> (T0, T1, T2) {

  var count: Int32 = T0._tensorHandleCount + T1._tensorHandleCount + T2._tensorHandleCount
  let buffer: UnsafeMutablePointer<CTensorHandle> =
    UnsafeMutablePointer.allocate(capacity: Int(count))
  defer { buffer.deallocate() }
  _TFCEagerExecute(op, UnsafeMutablePointer<CTensorHandle?>(buffer), &count, s)
  checkOk(s)
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off0: Int32 = 0
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off1: Int32 = off0 + T0._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off2: Int32 = off1 + T1._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 24)
  return (T0.init(_owning: buffer.advanced(by: Int(off0))), T1.init(_owning: buffer.advanced(by: Int(off1))), T2.init(_owning: buffer.advanced(by: Int(off2))))
}
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 10)
@usableFromInline
func TensorGroupExecuteOp<T0 : TensorGroup, T1 : TensorGroup, T2 : TensorGroup, T3 : TensorGroup>
  (_ op: CTFEOp, _ s: CTFStatus)
  -> (T0, T1, T2, T3) {

  var count: Int32 = T0._tensorHandleCount + T1._tensorHandleCount + T2._tensorHandleCount + T3._tensorHandleCount
  let buffer: UnsafeMutablePointer<CTensorHandle> =
    UnsafeMutablePointer.allocate(capacity: Int(count))
  defer { buffer.deallocate() }
  _TFCEagerExecute(op, UnsafeMutablePointer<CTensorHandle?>(buffer), &count, s)
  checkOk(s)
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off0: Int32 = 0
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off1: Int32 = off0 + T0._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off2: Int32 = off1 + T1._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off3: Int32 = off2 + T2._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 24)
  return (T0.init(_owning: buffer.advanced(by: Int(off0))), T1.init(_owning: buffer.advanced(by: Int(off1))), T2.init(_owning: buffer.advanced(by: Int(off2))), T3.init(_owning: buffer.advanced(by: Int(off3))))
}
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 10)
@usableFromInline
func TensorGroupExecuteOp<T0 : TensorGroup, T1 : TensorGroup, T2 : TensorGroup, T3 : TensorGroup, T4 : TensorGroup>
  (_ op: CTFEOp, _ s: CTFStatus)
  -> (T0, T1, T2, T3, T4) {

  var count: Int32 = T0._tensorHandleCount + T1._tensorHandleCount + T2._tensorHandleCount + T3._tensorHandleCount + T4._tensorHandleCount
  let buffer: UnsafeMutablePointer<CTensorHandle> =
    UnsafeMutablePointer.allocate(capacity: Int(count))
  defer { buffer.deallocate() }
  _TFCEagerExecute(op, UnsafeMutablePointer<CTensorHandle?>(buffer), &count, s)
  checkOk(s)
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off0: Int32 = 0
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off1: Int32 = off0 + T0._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off2: Int32 = off1 + T1._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off3: Int32 = off2 + T2._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off4: Int32 = off3 + T3._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 24)
  return (T0.init(_owning: buffer.advanced(by: Int(off0))), T1.init(_owning: buffer.advanced(by: Int(off1))), T2.init(_owning: buffer.advanced(by: Int(off2))), T3.init(_owning: buffer.advanced(by: Int(off3))), T4.init(_owning: buffer.advanced(by: Int(off4))))
}
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 10)
@usableFromInline
func TensorGroupExecuteOp<T0 : TensorGroup, T1 : TensorGroup, T2 : TensorGroup, T3 : TensorGroup, T4 : TensorGroup, T5 : TensorGroup>
  (_ op: CTFEOp, _ s: CTFStatus)
  -> (T0, T1, T2, T3, T4, T5) {

  var count: Int32 = T0._tensorHandleCount + T1._tensorHandleCount + T2._tensorHandleCount + T3._tensorHandleCount + T4._tensorHandleCount + T5._tensorHandleCount
  let buffer: UnsafeMutablePointer<CTensorHandle> =
    UnsafeMutablePointer.allocate(capacity: Int(count))
  defer { buffer.deallocate() }
  _TFCEagerExecute(op, UnsafeMutablePointer<CTensorHandle?>(buffer), &count, s)
  checkOk(s)
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off0: Int32 = 0
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off1: Int32 = off0 + T0._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off2: Int32 = off1 + T1._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off3: Int32 = off2 + T2._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off4: Int32 = off3 + T3._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off5: Int32 = off4 + T4._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 24)
  return (T0.init(_owning: buffer.advanced(by: Int(off0))), T1.init(_owning: buffer.advanced(by: Int(off1))), T2.init(_owning: buffer.advanced(by: Int(off2))), T3.init(_owning: buffer.advanced(by: Int(off3))), T4.init(_owning: buffer.advanced(by: Int(off4))), T5.init(_owning: buffer.advanced(by: Int(off5))))
}
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 10)
@usableFromInline
func TensorGroupExecuteOp<T0 : TensorGroup, T1 : TensorGroup, T2 : TensorGroup, T3 : TensorGroup, T4 : TensorGroup, T5 : TensorGroup, T6 : TensorGroup>
  (_ op: CTFEOp, _ s: CTFStatus)
  -> (T0, T1, T2, T3, T4, T5, T6) {

  var count: Int32 = T0._tensorHandleCount + T1._tensorHandleCount + T2._tensorHandleCount + T3._tensorHandleCount + T4._tensorHandleCount + T5._tensorHandleCount + T6._tensorHandleCount
  let buffer: UnsafeMutablePointer<CTensorHandle> =
    UnsafeMutablePointer.allocate(capacity: Int(count))
  defer { buffer.deallocate() }
  _TFCEagerExecute(op, UnsafeMutablePointer<CTensorHandle?>(buffer), &count, s)
  checkOk(s)
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off0: Int32 = 0
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off1: Int32 = off0 + T0._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off2: Int32 = off1 + T1._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off3: Int32 = off2 + T2._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off4: Int32 = off3 + T3._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off5: Int32 = off4 + T4._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off6: Int32 = off5 + T5._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 24)
  return (T0.init(_owning: buffer.advanced(by: Int(off0))), T1.init(_owning: buffer.advanced(by: Int(off1))), T2.init(_owning: buffer.advanced(by: Int(off2))), T3.init(_owning: buffer.advanced(by: Int(off3))), T4.init(_owning: buffer.advanced(by: Int(off4))), T5.init(_owning: buffer.advanced(by: Int(off5))), T6.init(_owning: buffer.advanced(by: Int(off6))))
}
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 10)
@usableFromInline
func TensorGroupExecuteOp<T0 : TensorGroup, T1 : TensorGroup, T2 : TensorGroup, T3 : TensorGroup, T4 : TensorGroup, T5 : TensorGroup, T6 : TensorGroup, T7 : TensorGroup>
  (_ op: CTFEOp, _ s: CTFStatus)
  -> (T0, T1, T2, T3, T4, T5, T6, T7) {

  var count: Int32 = T0._tensorHandleCount + T1._tensorHandleCount + T2._tensorHandleCount + T3._tensorHandleCount + T4._tensorHandleCount + T5._tensorHandleCount + T6._tensorHandleCount + T7._tensorHandleCount
  let buffer: UnsafeMutablePointer<CTensorHandle> =
    UnsafeMutablePointer.allocate(capacity: Int(count))
  defer { buffer.deallocate() }
  _TFCEagerExecute(op, UnsafeMutablePointer<CTensorHandle?>(buffer), &count, s)
  checkOk(s)
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off0: Int32 = 0
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off1: Int32 = off0 + T0._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off2: Int32 = off1 + T1._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off3: Int32 = off2 + T2._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off4: Int32 = off3 + T3._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off5: Int32 = off4 + T4._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off6: Int32 = off5 + T5._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off7: Int32 = off6 + T6._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 24)
  return (T0.init(_owning: buffer.advanced(by: Int(off0))), T1.init(_owning: buffer.advanced(by: Int(off1))), T2.init(_owning: buffer.advanced(by: Int(off2))), T3.init(_owning: buffer.advanced(by: Int(off3))), T4.init(_owning: buffer.advanced(by: Int(off4))), T5.init(_owning: buffer.advanced(by: Int(off5))), T6.init(_owning: buffer.advanced(by: Int(off6))), T7.init(_owning: buffer.advanced(by: Int(off7))))
}
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 10)
@usableFromInline
func TensorGroupExecuteOp<T0 : TensorGroup, T1 : TensorGroup, T2 : TensorGroup, T3 : TensorGroup, T4 : TensorGroup, T5 : TensorGroup, T6 : TensorGroup, T7 : TensorGroup, T8 : TensorGroup>
  (_ op: CTFEOp, _ s: CTFStatus)
  -> (T0, T1, T2, T3, T4, T5, T6, T7, T8) {

  var count: Int32 = T0._tensorHandleCount + T1._tensorHandleCount + T2._tensorHandleCount + T3._tensorHandleCount + T4._tensorHandleCount + T5._tensorHandleCount + T6._tensorHandleCount + T7._tensorHandleCount + T8._tensorHandleCount
  let buffer: UnsafeMutablePointer<CTensorHandle> =
    UnsafeMutablePointer.allocate(capacity: Int(count))
  defer { buffer.deallocate() }
  _TFCEagerExecute(op, UnsafeMutablePointer<CTensorHandle?>(buffer), &count, s)
  checkOk(s)
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off0: Int32 = 0
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off1: Int32 = off0 + T0._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off2: Int32 = off1 + T1._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off3: Int32 = off2 + T2._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off4: Int32 = off3 + T3._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off5: Int32 = off4 + T4._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off6: Int32 = off5 + T5._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off7: Int32 = off6 + T6._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off8: Int32 = off7 + T7._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 24)
  return (T0.init(_owning: buffer.advanced(by: Int(off0))), T1.init(_owning: buffer.advanced(by: Int(off1))), T2.init(_owning: buffer.advanced(by: Int(off2))), T3.init(_owning: buffer.advanced(by: Int(off3))), T4.init(_owning: buffer.advanced(by: Int(off4))), T5.init(_owning: buffer.advanced(by: Int(off5))), T6.init(_owning: buffer.advanced(by: Int(off6))), T7.init(_owning: buffer.advanced(by: Int(off7))), T8.init(_owning: buffer.advanced(by: Int(off8))))
}
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 10)
@usableFromInline
func TensorGroupExecuteOp<T0 : TensorGroup, T1 : TensorGroup, T2 : TensorGroup, T3 : TensorGroup, T4 : TensorGroup, T5 : TensorGroup, T6 : TensorGroup, T7 : TensorGroup, T8 : TensorGroup, T9 : TensorGroup>
  (_ op: CTFEOp, _ s: CTFStatus)
  -> (T0, T1, T2, T3, T4, T5, T6, T7, T8, T9) {

  var count: Int32 = T0._tensorHandleCount + T1._tensorHandleCount + T2._tensorHandleCount + T3._tensorHandleCount + T4._tensorHandleCount + T5._tensorHandleCount + T6._tensorHandleCount + T7._tensorHandleCount + T8._tensorHandleCount + T9._tensorHandleCount
  let buffer: UnsafeMutablePointer<CTensorHandle> =
    UnsafeMutablePointer.allocate(capacity: Int(count))
  defer { buffer.deallocate() }
  _TFCEagerExecute(op, UnsafeMutablePointer<CTensorHandle?>(buffer), &count, s)
  checkOk(s)
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off0: Int32 = 0
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off1: Int32 = off0 + T0._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off2: Int32 = off1 + T1._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off3: Int32 = off2 + T2._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off4: Int32 = off3 + T3._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off5: Int32 = off4 + T4._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off6: Int32 = off5 + T5._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off7: Int32 = off6 + T6._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off8: Int32 = off7 + T7._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 22)
let off9: Int32 = off8 + T8._tensorHandleCount
// ###sourceLocation(file: "/usr/local/google/home/parkers/swift-source/tensorflow-swift-bindings/ExecuteOp.swift.gyb", line: 24)
  return (T0.init(_owning: buffer.advanced(by: Int(off0))), T1.init(_owning: buffer.advanced(by: Int(off1))), T2.init(_owning: buffer.advanced(by: Int(off2))), T3.init(_owning: buffer.advanced(by: Int(off3))), T4.init(_owning: buffer.advanced(by: Int(off4))), T5.init(_owning: buffer.advanced(by: Int(off5))), T6.init(_owning: buffer.advanced(by: Int(off6))), T7.init(_owning: buffer.advanced(by: Int(off7))), T8.init(_owning: buffer.advanced(by: Int(off8))), T9.init(_owning: buffer.advanced(by: Int(off9))))
}
