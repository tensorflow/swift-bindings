# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generates some swift wrapper from some ops description protobuf."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import six
import tensorflow as tf

from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import c_api_util

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
  'api_def_path',
  None,
  'path to the api_def directory, e.g. tensorflow/core/api_def/base_api')

flags.DEFINE_string(
  'output_path',
  None,
  'path for the generated swift file')

flags.DEFINE_string(
  'mode',
  'eager',
  'Code generation mode that can be either "tfop" or "eager".')

_WARNING = """// !!! THIS CODE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND !!!
//
"""

_HEADER = """// Copyright 2018 Google LLC
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
"""

_OUTPUT_FILE = 'RawOpsGenerated.swift'
_RENAMED_KEYWORDS = {
  '': 'empty',
  'in': 'in_',
  'var': 'var_',
  'where': 'where_',
  'switch': 'switch_',
  'protocol': 'protocol_',
  'init': 'init_'}

_TYPE_PROTOCOLS = [
  (set(), 'TensorFlowScalar'),
  ({types_pb2.DT_UINT8,
    types_pb2.DT_UINT16,
    types_pb2.DT_UINT32,
    types_pb2.DT_UINT64}, 'TensorFlowUnsignedInteger'),
  ({types_pb2.DT_UINT8,
    types_pb2.DT_UINT16,
    types_pb2.DT_UINT32,
    types_pb2.DT_UINT64,
    types_pb2.DT_INT8,
    types_pb2.DT_INT16,
    types_pb2.DT_INT32,
    types_pb2.DT_INT64}, 'TensorFlowInteger'),
  ({types_pb2.DT_FLOAT,
    types_pb2.DT_DOUBLE,
    types_pb2.DT_HALF,
    types_pb2.DT_BFLOAT16}, 'TensorFlowFloatingPoint'),
  ({types_pb2.DT_UINT8,
    types_pb2.DT_UINT16,
    types_pb2.DT_UINT32,
    types_pb2.DT_UINT64,
    types_pb2.DT_INT8,
    types_pb2.DT_INT16,
    types_pb2.DT_INT32,
    types_pb2.DT_INT64,
    types_pb2.DT_FLOAT,
    types_pb2.DT_DOUBLE,
    types_pb2.DT_HALF,
    types_pb2.DT_BFLOAT16}, 'TensorFlowNumeric')]

_SWIFTIFIED_TYPES = {
  types_pb2.DT_FLOAT: 'Float',
  types_pb2.DT_DOUBLE: 'Double',
  types_pb2.DT_INT32: 'Int32',
  types_pb2.DT_UINT8: 'UInt8',
  types_pb2.DT_INT16: 'Int16',
  types_pb2.DT_INT8: 'Int8',
  types_pb2.DT_INT64: 'Int64',
  types_pb2.DT_BOOL: 'Bool',
  types_pb2.DT_UINT16: 'UInt16',
  types_pb2.DT_UINT32: 'UInt32',
  types_pb2.DT_UINT64: 'UInt64'}

_SWIFTIFIED_ATTR_TYPES = {
  'int': 'Int64',
  'float': 'Double',
  'bool': 'Bool',
  'string': 'String',
  'type': 'TensorDataType',
  'shape': 'TensorShape?',
  'list(int)': '[Int32]',
  'list(float)': '[Double]',
  'list(bool)': '[Bool]',
  'list(string)': '[String]',
  'list(type)': '[TensorDataType]',
  'list(shape)': '[TensorShape?]'}

_OMITTED_PARAMETER_NAMES = {
  'x', 'y', 'a', 'b', 'input', 'tensor', 'values'}

_START_COMMENT = '///'


class UnableToGenerateCodeError(Exception):
  def __init__(self, details):
    self.details = details
    super(UnableToGenerateCodeError, self).__init__()

  def __str__(self):
    return self.details


class Op(object):
  def __init__(self, op_def, api_def, enum_store):
    self.op_def = op_def
    self.api_def = api_def
    self.enum_store = enum_store

    # Collect all the attributes that need to be provided
    # as inputs. Note that we do not generate input
    # arguments for number attributes as these are always
    # inferred from provided arrays, and also for type
    # attributes whose values can be inferred from the rest
    # of the inputs.
    excluded_attrs = set([
      attr.number_attr for attr in op_def.input_arg])
    self.attrs = [
      Attribute(attr, op=self)
      for attr in op_def.attr
      if attr.name not in excluded_attrs]
    self.type_attrs = [
      attr for attr in self.attrs
      if attr.is_type_attr]

    # Collect all the input and output arguments.
    self.input_args = [
      Argument(arg_def, op=self)
      for arg_def in self.op_def.input_arg]
    self.output_args = [
      Argument(arg_def, op=self)
      for arg_def in self.op_def.output_arg]

  def swift_function(self, mode):
    return '''
{documentation}@inlinable @inline(__always)
public static func {name}{generics}({input_args}
){return_type} {{
  {body}
}}'''.format(
      documentation=self._swift_documentation(),
      name=self._swift_name(),
      generics=self._swift_generics(),
      input_args=self._swift_input_args(),
      return_type=self._swift_return_type(),
      body=self._swift_body(mode))

  def _swift_documentation(self):
    def comment_block(text, indent_level):
      """Returns a commented block of text with some specified indentation."""
      def indent(line_index):
        if indent_level == 0:
          return ''
        if line_index:
          return '  ' * indent_level
        return '  ' * (indent_level - 1) + '- '

      return ''.join([
        (_START_COMMENT + ' ' + indent(line_index) + line + '\n'
         if line else _START_COMMENT + '\n')
        for line_index, line in enumerate(text.split('\n'))
      ])

    def append_list(doc, args, arg_type):
      """Returns the documentation for lists of inputs/outputs/attributes."""
      args = [arg for arg in args if arg.description]
      if len(args) == 1:
        block = '%s %s: %s' % (arg_type, args[0].name, args[0].description)
        doc += _START_COMMENT + '\n'
        doc += comment_block(block, indent_level=1)
      elif len(args) > 1:
        doc += '%s\n%s - %ss:\n' % (_START_COMMENT, _START_COMMENT, arg_type)
        for arg in args:
          block = '%s: %s' % (arg.name, arg.description)
          doc += comment_block(block, indent_level=2)
      return doc

    doc = ''
    if self.api_def.summary:
      doc = comment_block(self.api_def.summary, indent_level=0)
    if self.api_def.description:
      doc += _START_COMMENT + '\n'
      doc += comment_block(self.api_def.description, indent_level=0)
    doc = append_list(doc, self.api_def.in_arg, 'Parameter')
    doc = append_list(doc, self.api_def.attr, 'Attr')
    doc = append_list(doc, self.api_def.out_arg, 'Output')
    if doc and not doc.endswith('\n'):
      doc = doc + '\n'
    return doc

  def _swift_name(self):
    return swift_compatible_identifier(
      self.op_def.name[0].lower() + self.op_def.name[1:])

  def _swift_generics(self):
    arg_type_attrs = [
      attr for attr in self.type_attrs
      if attr.is_inferred_type_attr]
    if arg_type_attrs:
      return '<' + ', '.join([
        attr.swift_name + ': ' + attr.protocol
        for attr in arg_type_attrs]) + '>'
    return ''

  def _swift_input_args(self):
    args = ''
    for arg in self.input_args:
      args += '\n  %s: %s,' % (arg.swift_arg_name, str(arg.swift_type))
    for attr in self.attrs:
      if not attr.is_inferred_type_attr:
        args += '\n  %s: %s%s,' % (attr.swift_arg_name, attr.swift_type, attr.swift_default)
    if args != '':
      args = args[:-1]
    return args

  def _swift_return_type(self):
    # Do not generate ops with output lists.
    # TODO: We could support output lists by giving the outputs generic type that
    # conforms to TensorGroup.
    for output_arg in self.output_args:
      if output_arg.is_list:
        raise UnableToGenerateCodeError('Output lists not supported.')

    return_type = ''
    if len(self.output_args) == 1:
      return_type = ' -> ' + str(self.output_args[0].swift_type)
    elif len(self.output_args) > 1:
      named_types = [
        arg.swift_name + ': ' + str(arg.swift_type)
        for arg in self.output_args]
      return_type = ' -> (' + ', '.join(named_types) + ')'
    return return_type

  def _swift_body(self, mode):
    if mode == 'tfop':
      tfop_args = ['"' + self.op_def.name + '"']
      for arg in self.input_args:
        tfop_args.append(arg.swift_setter(mode))
      for attr in self.attrs:
        setter = attr.swift_setter(mode)
        if setter != '':
          tfop_args.append(setter)
      tfop_args = ',\n    '.join(tfop_args)
      if not self.output_args:
        return 'return #tfop({})'.format(tfop_args)
      else:
        # Example body with 1 return tensor:
        # let ret: [TensorHandle<Int32>] = #tfop("ConcatOffset",
        #   concatDim,
        #   shape)
        # return ret.0.map(Tensor.init)
        #
        # Example body with 2 return tensors:
        # let ret: (TensorHandle<T>, TensorHandle<T>) = #tfop("SoftmaxCrossEntr...",
        #   features,
        #   labels,
        #   T: T.self)
        # return (Tensor(handle: ret.0), Tensor(handle: ret.1))
        # if ret.0 is [TensorHandle<T>], then we construct ret.0.map(Tensor.init) to
        # convert it to [Tensor<T>]
        handle_types = [arg.swift_handle_type for arg in self.output_args]
        if len(self.output_args) > 1:
          return_handle_type = '(' + ', '.join(handle_types) + ')'
        else:
          return_handle_type = handle_types[0]
        body = 'let ret: {} = #tfop({})'.format(
          return_handle_type, tfop_args)
        body += '\n  return '

        def convert_for_return(arg, value):
          if arg.type.kind == 'Tensor' and arg.type.base_type == 'String':
            return 'StringTensor(handle: ' + value + ')'
          elif arg.type.kind == 'Tensor':
            return 'Tensor(handle: ' + value + ')'
          return value

        if len(self.output_args) > 1:
          returned_tuple = [
            convert_for_return(arg, 'ret.' + str(i))
            for i, arg in enumerate(self.output_args)]
          body += '(' + ', '.join(returned_tuple) + ')'
        else:
          body += convert_for_return(self.output_args[0], 'ret')
        return body
    elif mode == 'eager':
      body = '''let s: CTFStatus = TF_NewStatus()
  defer { TF_DeleteStatus(s) }
  let op: CTFEOp = TFE_NewOp(_ExecutionContext.global.eagerContext, "%s", s)
  defer { TFE_DeleteOp(op) }''' % self.op_def.name
      setters = []
      for arg in self.input_args:
        setters.append(arg.swift_setter(mode))
      for attr in self.attrs:
        setters.append(attr.swift_setter(mode))
      body += '\n  '.join(setters)
      return body + '\n  return TensorGroupExecuteOp(op, s)'

    # `mode` was neither "tfop" nor "eager".
    raise UnableToGenerateCodeError(
      'Invalid mode "%s" provided (only "tfop" and "eager" are supported).'
      % mode)


class Argument(object):
  def __init__(self, arg_def, op):
    self.arg_def = arg_def
    self.op = op
    self.is_list = arg_def.number_attr or arg_def.type_list_attr

  @property
  def name(self):
    return self.arg_def.name

  @property
  def swift_name(self):
    return swift_compatible_identifier(
      self.name[0].lower() + self.name[1:])

  @property
  def swift_arg_name(self):
    name = self.swift_name
    if name in _OMITTED_PARAMETER_NAMES:
      name = '_ ' + name
    return name

  @property
  def swift_type(self):
    return self.type.swift_type

  @property
  def swift_handle_type(self):
    return self.type.swift_handle_type

  def swift_setter(self, mode):
    if mode == 'tfop':
      return self.swift_name
    elif mode == 'eager':
      if self.arg_def.number_attr:
        return ('let {name}Count = _TFCOpAddInputFromTensorGroup(op, {name}, s)' +
                'TFE_OpSetAttrInt(op, "{number_attr}", Int64({name}Count))'
                ).format(name=self.swift_name, number_attr=self.arg_def.number_attr)
      else:
        return 'let _ = _TFCOpAddInputFromTensorGroup(op, {name}, s)'.format(name=self.swift_name)

    # `mode` was neither "tfop" nor "eager".
    raise UnableToGenerateCodeError(
      'Invalid mode "%s" provided (only "tfop" and "eager" are supported).'
      % mode)

  @property
  def type(self):
    array = self.arg_def.number_attr
    if self.arg_def.type_attr:
      type_attr = next(
        attr for attr in self.op.type_attrs
        if attr.name == self.arg_def.type_attr)
      return Type('Tensor', base_type=type_attr.swift_name, array=array)
    if self.arg_def.type_list_attr:
      type_attr = next(
        attr for attr in self.op.type_attrs
        if attr.name == self.arg_def.type_list_attr)
      # There are never any numbered type lists.
      return Type(type_attr.swift_name)
    if self.arg_def.type in _SWIFTIFIED_TYPES:
      base_type = _SWIFTIFIED_TYPES[self.arg_def.type]
      return Type('Tensor', base_type=base_type, array=array)
    if self.arg_def.type == types_pb2.DT_STRING:
      return Type('Tensor', base_type='String', array=array)
    if self.arg_def.type == types_pb2.DT_RESOURCE:
      return Type('ResourceHandle', array=array)
    if self.arg_def.type == types_pb2.DT_VARIANT:
      return Type('VariantHandle', array=array)
    raise UnableToGenerateCodeError(
      'Unsupported type for argument "%s".' % self.name)


class Type(object):
  def __init__(self, kind, base_type=None, array=False):
    self.kind = kind
    self.base_type = base_type
    self.array = array

  @property
  def swift_type(self):
    if self.kind == 'Tensor':
      if self.base_type == 'String':
        name = 'StringTensor'
      else:
        name = 'Tensor<' + self.base_type + '>'
    elif self.kind == 'TensorHandle':
      name = 'TensorHandle<' + self.base_type + '>'
    elif self.kind == 'ResourceHandle':
      name = 'ResourceHandle'
    elif self.kind == 'VariantHandle':
      name = 'VariantHandle'
    else:
      name = self.kind
    return ('[%s]' % name) if self.array else name

  @property
  def swift_handle_type(self):
    if self.kind == 'Tensor' or self.kind == 'TensorHandle':
      name = 'TensorHandle<' + self.base_type + '>'
    elif self.kind == 'ResourceHandle':
      name = 'ResourceHandle'
    elif self.kind == 'VariantHandle':
      name = 'VariantHandle'
    else:
      # TODO: [tfop]
      raise UnableToGenerateCodeError('Unsupported handle type "%s".' % self.kind)
    return ('[%s]' % name) if self.array else name


class Attribute(object):
  """Represents information extracted from op `type` and `list(type)` attributes."""

  def __init__(self, attr_def, op):
    self.attr_def = attr_def
    self.op = op
    self.is_type_attr = attr_def.type in ['type', 'list(type)']

    # Check whether the value of this attribute can be
    # inferred automatically (this only applies to
    # type-valued attributes).
    args = list(op.op_def.input_arg) + list(op.op_def.output_arg)
    arg_type_attrs = set(
      [arg.type_attr for arg in args] +
      [arg.type_list_attr for arg in args])
    self.is_inferred_type_attr = self.is_type_attr and attr_def.name in arg_type_attrs

    # The following properties are only relevant for
    # non-inferred-type-valued attributes.
    self._swift_type = ''
    self._use_enum = False
    if not self.is_inferred_type_attr:
      if self.attr_def.type not in _SWIFTIFIED_ATTR_TYPES:
        raise UnableToGenerateCodeError(
          'Unsupported type for attribute "%s".'
          % self.attr_def.name)

      # Get the arg type.
      self._swift_type = _SWIFTIFIED_ATTR_TYPES[self.attr_def.type]

      # Check if the arg is an enum type.
      self._use_enum = False
      if self.attr_def.type == 'string':
        allowed_values = tuple(sorted(self.attr_def.allowed_values.list.s))
        if allowed_values:
          self._swift_type = self.op.enum_store.maybe_add(
            allowed_values, self.attr_def.name)
          self._use_enum = True

  @property
  def name(self):
    return self.attr_def.name

  @property
  def swift_name(self):
    if self.is_inferred_type_attr:
      return swift_compatible_identifier(
        self.name, capitalize=True)
    return swift_compatible_identifier(
      self.name[0].lower() + self.name[1:])

  @property
  def swift_arg_name(self):
    name = self.swift_name
    if name in _OMITTED_PARAMETER_NAMES:
      name = '_ ' + name
    return name

  @property
  def swift_type(self):
    if self.is_inferred_type_attr:
      return ''
    return self._swift_type

  @property
  def swift_default(self):
    def swift_float(f):
      if f == float('inf'): return 'Double.infinity'
      if f == float('-inf'): return '-Double.infinity'
      return '%g' % f

    if not self.is_inferred_type_attr and self.attr_def.default_value:
      default_value = self.attr_def.default_value
      if default_value.HasField('b'):
        default_value = str(default_value.b).lower()
      elif default_value.HasField('i'):
        default_value = str(default_value.i)
      elif default_value.HasField('f'):
        default_value = swift_float(default_value.f)
      elif default_value.HasField('s') and default_value.s:
        s = str(default_value.s, encoding='utf-8')
        default_value = '.' + swift_compatible_identifier(s.lower()) \
          if self._use_enum else '"' + s + '"'
      elif default_value.HasField('list'):
        if default_value.list.i:
          default_values = [str(s) for s in default_value.list.i]
          default_value = '[' + ', '.join(default_values) + ']'
        elif default_value.list.f:
          default_values = [swift_float(s) for s in default_value.list.f]
          default_value = '[' + ', '.join(default_values) + ']'
        else:
          default_value = None
      else:
        default_value = None
      if default_value is not None:
        default_value = default_value.replace("\t", "\\t")
        return ' = ' + default_value
    return ''

  def swift_setter(self, mode):
    if mode == 'tfop':
      # First, we handle inferred-type-valued attributes.
      if self.is_inferred_type_attr:
        if self.attr_def.type == 'list(type)':
          # TODO: [tfop]
          raise UnableToGenerateCodeError('Unsupported type for attribute "%s" in "tfop" mode.' % self.attr_def.name)
        return self.name + '$dtype: ' + self.swift_name + '.tensorFlowDataType'

      # The following is for non-inferred-type-valued attributes.
      value = self.swift_name + '.cName' if self._use_enum else self.swift_name
      return '{name}: {value}'.format(name=self.name, value=value)
    elif mode == 'eager':
      # First, we handle inferred-type-valued attributes.
      if self.is_inferred_type_attr:
        if self.attr_def.type == 'list(type)':
          return '_TFCOpSetAttrTypeArray(op, "' + self.name + '", ' + self.swift_name + '._typeList)'
        return 'TFE_OpSetAttrType(op, "' + self.name + '", ' + self.swift_name + '.tensorFlowDataType._cDataType)'

      # The following is for non-inferred-type-valued attributes.
      value = self.swift_name + '.cName' if self._use_enum else self.swift_name
      if self.attr_def.type == 'bool':
        setter_fn = 'TFE_OpSetAttrBool'
        value = '(' + value + ') ? 1 : 0'
      elif self.attr_def.type == 'int':
        setter_fn = 'TFE_OpSetAttrInt'
      elif self.attr_def.type == 'float':
        setter_fn = 'TFE_OpSetAttrFloat'
        value = 'Float(' + value + ')'
      elif self.attr_def.type == 'string':
        setter_fn = '_TFCOpSetAttrString'
      elif self.attr_def.type == 'type':
        setter_fn = 'TFE_OpSetAttrType'
        value = value + '._cDataType'
      elif self.attr_def.type == 'shape':
        setter_fn = '_TFCOpSetAttrOptionalTensorShape'
        value = value + ', s'
      elif self.attr_def.type == 'list(float)':
        setter_fn = '_TFCOpSetAttrDoubleArray'
      elif self.attr_def.type == 'list(string)':
        setter_fn = '_TFCOpSetAttrStringArray'
      elif self.attr_def.type == 'list(bool)':
        setter_fn = '_TFCOpSetAttrBoolArray'
      elif self.attr_def.type == 'list(int)':
        setter_fn = '_TFCOpSetAttrInt32Array'
      elif self.attr_def.type == 'list(type)':
        setter_fn = '_TFCOpSetAttrTypeArray'
      elif self.attr_def.type == 'list(shape)':
        setter_fn = '_TFCOpSetAttrOptionalTensorShapeArray'
        value = value + ', s'
      else:
        raise UnableToGenerateCodeError(
          'Unsupported type "%s" for attribute "%s".'
          % (self.attr_def.type, self.name))
      return setter_fn + '(op, "' + self.name + '", ' + value + ')'

    # `mode` was neither "tfop" nor "eager".
    raise UnableToGenerateCodeError(
      'Invalid mode "%s" provided (only "tfop" and "eager" are supported).'
      % mode)

  @property
  def protocol(self):
    protocol = None
    if self.attr_def.type == 'list(type)':
      protocol = 'TensorGroup'
    elif self.attr_def.type == 'type':
      protocol = 'TensorFlowScalar'
      allowed_types = set(self.attr_def.allowed_values.list.type)
      allowed_types &= set(_SWIFTIFIED_TYPES.keys())
      for types, protocol_name in _TYPE_PROTOCOLS:
        if allowed_types.issubset(types):
          protocol = protocol_name
          break
    return protocol


def swift_compatible_identifier(s, capitalize=False):
  """Transforms an identifier to be more swift idiomatic."""
  if s in _RENAMED_KEYWORDS:
    return _RENAMED_KEYWORDS[s]
  if capitalize:
    s = s.capitalize()
  without_underscores = []
  capitalize_next_char = False
  for c in s:
    if c == '-' or c == '_' or c == '(' or c == ')':
      capitalize_next_char = True
    elif capitalize_next_char:
      capitalize_next_char = False
      without_underscores.append(c.upper())
    else:
      without_underscores.append(c)
  return ''.join(without_underscores)


class EnumStore(object):
  """Stores details on string attributes represented as swift enums."""

  def __init__(self):
    self._entries = {}
    self._type_names = set()
    self._counter = 1

  def enum_codes(self):
    """Generates the swift code for enums."""
    codes = []
    entries = list(six.iteritems(self._entries))
    for allowed_values, type_name in sorted(entries, key=lambda x: x[1]):
      allowed_values = [str(a, encoding='utf-8') for a in allowed_values]
      codes.append(
          # FIXME: Re-add `@_frozen` after SR-9739 is resolved.
          # https://bugs.swift.org/browse/SR-9739
          # '@_frozen\n' +
          '// @_frozen // SR-9739\n' +
          'public enum {} {{\n'.format(type_name) +
          '\n'.join(['  case {}'.format(
            swift_compatible_identifier(a.lower()))
            for a in allowed_values]) +
          '\n\n' +
          '  @inlinable\n' +
          '  var cName: String {\n' +
          '    @inline(__always)\n' +
          '    get {\n' +
          '      switch self {\n' +
          '\n'.join(['      case .{}: return "{}"'.format(
            swift_compatible_identifier(a.lower()), a)
            for a in allowed_values]) +
          '\n' +
          '      }\n' +
          '    }\n' +
          '  }\n' +
          '}')
    return codes

  def maybe_add(self, allowed_values, attr_def_name):
    if allowed_values in self._entries:
      return self._entries[allowed_values]
    type_name = swift_compatible_identifier(attr_def_name, capitalize=True)
    while type_name in self._type_names:
      type_name += str(self._counter)
      self._counter += 1
    self._type_names.add(type_name)
    self._entries[allowed_values] = type_name
    return type_name


def main(argv):
  del argv  # Unused.
  if FLAGS.output_path is None:
    raise ValueError('No output_path has been set')

  api_def_map = c_api_util.ApiDefMap()

  op_codes = []
  enum_store = EnumStore()
  op_names = api_def_map.op_names()
  if FLAGS.api_def_path is not None:
    for op_name in op_names:
      path = os.path.join(FLAGS.api_def_path, 'api_def_%s.pbtxt' % op_name)
      if not tf.gfile.Exists(path):
        continue
      with tf.gfile.Open(path, 'r') as fobj:
        data = fobj.read()
      try:
        api_def_map.put_api_def(data)
      except Exception as e:
        print('Cannot load api def for %s: %s' % (op_name, str(e)))

  for op_name in sorted(op_names):
    try:
      if op_name[0] == '_': continue
      op_def = api_def_map.get_op_def(op_name)
      if any(a.is_ref for a in op_def.input_arg):
        raise UnableToGenerateCodeError('has ref-valued input')
      if any(a.is_ref for a in op_def.output_arg):
        raise UnableToGenerateCodeError('has ref-valued output')
      api_def = api_def_map.get_api_def(bytes(op_name, 'utf8'))
      op = Op(op_def, api_def, enum_store)
      op_codes.append(op.swift_function(mode=FLAGS.mode))
    except UnableToGenerateCodeError as e:
      print('Cannot generate code for %s: %s' % (op_name, e.details))
  print('Generated code for %d/%d ops.' % (len(op_codes), len(op_names)))

  version_codes = [
      'static let generatedTensorFlowVersion = "%s"' % tf.__version__,
      'static let generatedTensorFlowGitVersion = "%s"' % tf.__git_version__]

  swift_code = (
      _WARNING +
      _HEADER +
      ('import CTensorFlow\n\n' if FLAGS.mode == 'eager' else '') +
      '\npublic enum Raw {\n\n' +
      '\n'.join(version_codes) +
      '\n\n' +
      '\n\n'.join(enum_store.enum_codes()) +
      '\n\n' +
      '\n'.join(op_codes) +
      '\n\n}\n')
  with tf.gfile.Open(FLAGS.output_path, 'w') as f:
    f.write(swift_code)


if __name__ == '__main__':
  tf.app.run(main)
