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

_WARNING = """//
// !!!THIS CODE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND!!!
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
    'init': 'init_',
}
_TYPE_PROTOCOLS = [
    (set([]), 'AccelerableByTensorFlow'),
    (set([types_pb2.DT_UINT8,
          types_pb2.DT_UINT16,
          types_pb2.DT_UINT32,
          types_pb2.DT_UINT64]), 'UnsignedInteger'),
    (set([types_pb2.DT_UINT8,
          types_pb2.DT_UINT16,
          types_pb2.DT_UINT32,
          types_pb2.DT_UINT64,
          types_pb2.DT_INT8,
          types_pb2.DT_INT16,
          types_pb2.DT_INT32,
          types_pb2.DT_INT64]), 'BinaryInteger'),
    (set([types_pb2.DT_FLOAT,
          types_pb2.DT_DOUBLE,
          types_pb2.DT_HALF,
          types_pb2.DT_BFLOAT16]), 'BinaryFloatingPoint'),
    (set([types_pb2.DT_UINT8,
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
          types_pb2.DT_BFLOAT16]), 'Numeric'),
]

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
    types_pb2.DT_UINT64: 'UInt64',
}

_SWIFTIFIED_ATTR_TYPES = {
    'int': 'Int64',
    'float': 'Double',
    'bool': 'Bool',
    'string': 'String',
    'list(int)': '[Int32]',
    'list(float)': '[Double]',
    'list(bool)': '[Bool]',
    'list(string)': '[String]',
}

_OMITTED_PARAMETER_NAMES = set(
    ['x', 'y', 'a', 'b', 'input', 'tensor', 'values'])

_START_COMMENT = '///'


class UnableToGenerateCodeError(Exception):

  def __init__(self, details):
    self.details = details
    super(UnableToGenerateCodeError, self).__init__()

  def __str__(self):
    return self.details


def swift_compatible(s, capitalize=False):
  """Transforms an identifier to be more swift idiomatic."""
  if s in _RENAMED_KEYWORDS:
    return _RENAMED_KEYWORDS[s]
  if capitalize:
    s = s.capitalize()
  without_underscores = []
  prev_char_was_underscore = False
  for c in s:
    if c == '_':
      prev_char_was_underscore = True
    elif prev_char_was_underscore:
      prev_char_was_underscore = False
      without_underscores.append(c.upper())
    else:
      without_underscores.append(c)
  return ''.join(without_underscores)


def swiftified_name(name):
  return swift_compatible(name[0].lower() + name[1:])


def swiftified_name_for_enums(name):
  return swift_compatible(name.lower())


class EnumStore(object):
  """Stores details on string attributes represented as swift enums."""

  def __init__(self):
    self._entries = {}
    self._type_names = set()
    self._counter = 1

  def enum_codes(self):
    """Generates the swift code for enums."""
    codes = []
    entries = list(self._entries.iteritems())
    for allowed_values, type_name in sorted(entries, key=lambda x: x[1]):
      codes.append(
          '@_frozen\n' +
          'public enum {} {{\n'.format(type_name) +
          '\n'.join(['  case {}'.format(
              swiftified_name_for_enums(a)) for a in allowed_values]) +
          '\n\n' +
          '  @inlinable\n' +
          '  var cName: String {\n' +
          '    @inline(__always)\n' +
          '    get {\n' +
          '      switch self {\n' +
          '\n'.join(['      case .{}: return "{}"'.format(
              swiftified_name_for_enums(a), a) for a in allowed_values]) +
          '\n' +
          '      }\n' +
          '    }\n' +
          '  }\n' +
          '}')
    return codes

  def maybe_add(self, allowed_values, attr_def_name):
    if allowed_values in self._entries:
      return self._entries[allowed_values]
    type_name = swift_compatible(attr_def_name, capitalize=True)
    while type_name in self._type_names:
      type_name += str(self._counter)
      self._counter += 1
    self._type_names.add(type_name)
    self._entries[allowed_values] = type_name
    return type_name


class Types(object):
  """Extracts some type information from a type or list(type) attr."""

  def __init__(self, attr_def):
    self._is_list_attr = attr_def.type == 'list(type)'
    self.swift_name = swift_compatible(attr_def.name, capitalize=True)
    self.attr_def_name = attr_def.name
    allowed_types = set(attr_def.allowed_values.list.type)
    allowed_types &= set(_SWIFTIFIED_TYPES.keys())
    self._protocol_name = 'AccelerableByTensorFlow'
    for handled_types, protocol_name in _TYPE_PROTOCOLS:
      if allowed_types.issubset(handled_types):
        self._protocol_name = protocol_name
        break

  def generics(self):
    return self.swift_name + ': ' + self._protocol_name

  def op_arg(self):
    # Do not pass list(type) attr as these have to use an array of types.
    if self._is_list_attr:
      return None
    return self.attr_def_name + ': ' + self.swift_name + '.self'


def swift_float(f):
  if f == float('inf'): return 'Double.infinity'
  if f == float('-inf'): return '-Double.infinity'
  return '%g' % f


def swift_default_value(attr_value, use_enum):
  """Converts the default value for an attr to a swift value."""
  if attr_value.HasField('b'):
    return str(attr_value.b).lower()
  if attr_value.HasField('i'):
    return str(attr_value.i)
  if attr_value.HasField('f'):
    return swift_float(attr_value.f)
  if attr_value.HasField('s') and attr_value.s:
    s = str(attr_value.s)
    return '.' + swiftified_name_for_enums(s) if use_enum else '"' + s + '"'
  if attr_value.HasField('list'):
    if attr_value.list.i:
      default_values = [str(s) for s in attr_value.list.i]
      return '[' + ', '.join(default_values) + ']'
    if attr_value.list.f:
      default_values = [swift_float(s) for s in attr_value.list.f]
      return '[' + ', '.join(default_values) + ']'
    return None
  return None


class AttributeAsInput(object):
  """Extracts from an attr_def some swift related fields."""

  def __init__(self, attr_def, enum_store):
    if attr_def.type not in _SWIFTIFIED_ATTR_TYPES:
      raise UnableToGenerateCodeError('unsupported type for ' + attr_def.name)
    self.swift_type_and_default_value = _SWIFTIFIED_ATTR_TYPES[attr_def.type]
    use_enum = False
    if attr_def.type == 'string':
      allowed_values = tuple(sorted(attr_def.allowed_values.list.s))
      if allowed_values:
        self.swift_type_and_default_value = enum_store.maybe_add(
            allowed_values, attr_def.name)
        use_enum = True

    if attr_def.default_value:
      default_value = swift_default_value(
          attr_def.default_value, use_enum=use_enum)
      if default_value is not None:
        self.swift_type_and_default_value += ' = ' + default_value

    self.swift_name = swiftified_name(attr_def.name)
    self.tfop_name = attr_def.name
    self.swift_value = (
        self.swift_name if not use_enum
        else self.swift_name + '.cName')


def attr_def_defines_a_type(attr_def):
  return attr_def.type in ['type', 'list(type)']


def arg_def_type_as_string(arg_def, handle=False):
  """Returns the tensor type for the provided input/output argument."""
  if arg_def.type_attr:
    base_type = swift_compatible(arg_def.type_attr, capitalize=True)
  elif arg_def.type_list_attr:
    base_type = swift_compatible(arg_def.type_list_attr, capitalize=True)
  elif arg_def.type in _SWIFTIFIED_TYPES:
    base_type = _SWIFTIFIED_TYPES[arg_def.type]
  else:
    raise UnableToGenerateCodeError('unsupported type for ' + arg_def.name)
  tensor_type = 'TensorHandle' if handle else 'Tensor'
  tensor_type += '<' + base_type + '>'
  if arg_def.number_attr or arg_def.type_list_attr:
    return '[' + tensor_type + ']'
  return tensor_type


def arg_def_type_is_list(arg_def):
  return arg_def.number_attr or arg_def.type_list_attr


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


def documentation(api_def):
  """Generates some documentation comment for a given op api def."""
  def append_list(doc, args, arg_type):
    """Returns the documentation for lists of inputs/outputs/attrs."""
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
  if api_def.summary:
    doc = comment_block(api_def.summary, indent_level=0)
  if api_def.description:
    doc += _START_COMMENT + '\n'
    doc += comment_block(api_def.description, indent_level=0)
  doc = append_list(doc, api_def.in_arg, 'Parameter')
  doc = append_list(doc, api_def.attr, 'Attr')
  doc = append_list(doc, api_def.out_arg, 'Output')
  if doc and not doc.endswith('\n'):
    doc += '\n'
  return doc


def maybe_named(name):
  if name in _OMITTED_PARAMETER_NAMES:
    return '_ ' + name
  return name


# Two cases based on if handle_var is TensorHandle or [TensorHandle]:
# 1. Case TensorHandle: h -> Tensor(handle: h)
# 2. Case [TensorHandle]: h -> h.map(Tensor.init)
def convert_handle_to_tensor(handle_var, is_list):
  if is_list:
    return handle_var + '.map(Tensor.init)'
  return 'Tensor(handle: ' + handle_var + ')'


def generate_code(op, api_def, enum_store):
  """Generates some swift code for a given op."""
  types = [Types(a) for a in op.attr if attr_def_defines_a_type(a)]
  generics_type = ''
  if types:
    generics_type = '<' + ', '.join([t.generics() for t in types]) + '>'

  input_names_and_types = [
      (swiftified_name(a.name), arg_def_type_as_string(a))
      for a in op.input_arg]

  # Do not generate an input parameter for numberAttr as these are inferred from
  # some array length.
  excluded_attributes = set([a.number_attr for a in op.input_arg])
  attributes_as_input = [
      AttributeAsInput(a, enum_store)
      for a in op.attr
      if not attr_def_defines_a_type(a) and a.name not in excluded_attributes
  ]

  # An example triplet entry can be:
  # ('batch', '[Tensor<Float>]', True)
  # The third component is a bool indicating if the entry is a tensor list.
  return_name_and_types = [
      (swiftified_name(a.name),
       arg_def_type_as_string(a),
       arg_def_type_is_list(a))
      for a in op.output_arg]
  return_type = ''
  if len(return_name_and_types) == 1:
    return_type = ' -> ' + return_name_and_types[0][1]
  elif len(return_name_and_types) > 1:
    named_types = [n + ': ' + t for n, t, _ in return_name_and_types]
    return_type = ' -> (' + ', '.join(named_types) + ')'

  tfop_args = ',\n    '.join(
      ['"' + op.name + '"'] +
      [name for name, _ in input_names_and_types] +
      filter(None, [t.op_arg() for t in types]) +
      [a.tfop_name + ': ' + a.swift_value for a in attributes_as_input]
  )

  attr_names_and_types = [
      (a.swift_name, a.swift_type_and_default_value)
      for a in attributes_as_input
  ]
  all_args = list(op.input_arg) + list(op.output_arg)
  arg_types = set(
      [a.type_attr for a in all_args] +
      [a.type_list_attr for a in all_args])
  missing_types = [
      ('type' + t.swift_name, t.swift_name + '.Type')
      for t in types if t.attr_def_name not in arg_types]
  all_inputs = [
      '\n  ' + maybe_named(name) + ': ' + type_and_default_value
      for name, type_and_default_value in (
          input_names_and_types +
          attr_names_and_types +
          missing_types)]
  body = ''
  if not return_name_and_types:
    body = 'return #tfop({tfop_args})'.format(tfop_args=tfop_args)
  elif len(return_name_and_types) >= 1:
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
    return_name_and_handle_types = [
        (swiftified_name(a.name), arg_def_type_as_string(a, handle=True))
        for a in op.output_arg]
    return_handle_type = ''
    if len(return_name_and_handle_types) > 1:
      handle_types = [t for n, t in return_name_and_handle_types]
      return_handle_type = '(' + ', '.join(handle_types) + ')'
    else:
      return_handle_type = return_name_and_handle_types[0][1]
    body = 'let ret: {return_handle_type} = #tfop({tfop_args})'.format(
        return_handle_type=return_handle_type, tfop_args=tfop_args)
    body += '\n  return '
    if len(return_name_and_types) > 1:
      returned_tuple = [
          convert_handle_to_tensor('ret.' + str(ind), is_list=named_type[2])
          for ind, named_type in enumerate(return_name_and_types)]
      body += '(' + ', '.join(returned_tuple) + ')'
    else:
      body += convert_handle_to_tensor(
          'ret', is_list=return_name_and_types[0][2])
  return (
      """{documentation}@inlinable @inline(__always)
public static func {function_name}{generics_type}({joined_inputs}
){return_type} {{
  {body}
}}""".format(documentation=documentation(api_def),
             function_name=swiftified_name(op.name),
             generics_type=generics_type,
             joined_inputs=','.join(all_inputs),
             return_type=return_type,
             body=body))


def main(argv):
  del argv  # Unused.
  if FLAGS.output_path is None:
    raise ValueError('no output_path has been set')

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
      api_def_map.put_api_def(data)

  for op_name in sorted(op_names):
    try:
      if op_name[0] == '_': continue
      op = api_def_map.get_op_def(op_name)
      api_def = api_def_map.get_api_def(bytes(op_name))
      op_codes.append(generate_code(op, api_def, enum_store))
    except UnableToGenerateCodeError as e:
      print('Cannot generate code for %s: %s' % (op.name, e.details))
  print('Generated code for %d/%d ops.'  % (len(op_codes), len(op_names)))

  version_codes = [
      'static let generatedTensorFlowVersion = "%s"' % tf.__version__,
      'static let generatedTensorFlowGitVersion = "%s"' % tf.__git_version__,
  ]

  swift_code = (
      _WARNING +
      _HEADER +
      '\npublic enum Raw {\n\n' +
      '\n'.join(version_codes) +
      '\n\n' +
      '\n\n'.join(enum_store.enum_codes()) +
      '\n\n' +
      '\n\n'.join(op_codes) +
      '\n\n}')
  with tf.gfile.Open(FLAGS.output_path, 'w') as fobj:
    fobj.write(swift_code)


if __name__ == '__main__':
  tf.app.run(main)
