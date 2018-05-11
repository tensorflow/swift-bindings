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

from google.protobuf import text_format
from tensorflow.core.framework import op_def_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.python import pywrap_tensorflow as c_api

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'ops_path',
    None,
    'path to the ops.pbtxt file')

flags.DEFINE_string(
    'src_dir',
    None,
    'path to the source directory')

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
import TensorFlow
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
    (set([]), 'Numeric'),
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
    'int': 'Int',
    'float': 'Double',
    'bool': 'Bool',
    'string': 'String',
    'list(int)': '[Int]',
    'list(float)': '[Double]',
    'list(bool)': '[Bool]',
    'list(string)': '[String]',
}


class UnableToGenerateCodeError(Exception):

  def __init__(self, details):
    self.details = details
    super(UnableToGenerateCodeError, self).__init__()

  def __str__(self):
    return self.details


def swift_compatible(s):
  """Transforms an identifier to be more swift idiomatic."""
  if s in _RENAMED_KEYWORDS:
    return _RENAMED_KEYWORDS[s]
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
    codes = []
    entries = list(self._entries.iteritems())
    for allowed_values, type_name in sorted(entries, key=lambda x: x[1]):
      codes.append(
          'public enum {}: String {{\n'.format(type_name) +
          '\n'.join(['  case {} = "{}"'.format(
              swiftified_name_for_enums(a), a) for a in allowed_values]) +
          '\n}')
    return codes

  def maybe_add(self, allowed_values, attr_def_name):
    if allowed_values in self._entries:
      return self._entries[allowed_values]
    type_name = swift_compatible(attr_def_name.capitalize())
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
    self.swift_name = attr_def.name.capitalize()
    self.attr_def_name = attr_def.name
    allowed_types = set(attr_def.allowed_values.list.type)
    allowed_types &= set(_SWIFTIFIED_TYPES.keys())
    self._protocol_name = 'Numeric'
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
    return self.swift_name + ': ' + self.swift_name + '.self'


def swift_default_value(attr_value, use_enum):
  """Converts the default value for an attr to a swift value."""
  if attr_value.HasField('b'):
    return str(attr_value.b).lower()
  if attr_value.HasField('i'):
    return str(attr_value.i)
  if attr_value.HasField('f'):
    if attr_value.f == float('inf'): return 'Double.infinity'
    if attr_value.f == float('-inf'): return '-Double.infinity'
    return '%g' % attr_value.f
  if attr_value.HasField('s') and attr_value.s:
    s = str(attr_value.s)
    return '.' + swiftified_name_for_enums(s) if use_enum else '"' + s + '"'
  return None


class AttributeAsInput(object):
  """Extract from an attr_def some swift related fields."""

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
        else self.swift_name + '.rawValue')


def attr_def_defines_a_type(attr_def):
  return attr_def.type in ['type', 'list(type)']


def arg_def_type_as_string(arg_def):
  """Returns the tensor type for the provided input/output argument."""
  if arg_def.type_attr:
    base_type = arg_def.type_attr.capitalize()
  elif arg_def.type_list_attr:
    base_type = arg_def.type_list_attr.capitalize()
  elif arg_def.type in _SWIFTIFIED_TYPES:
    base_type = _SWIFTIFIED_TYPES[arg_def.type]
  else:
    raise UnableToGenerateCodeError('unsupported type for ' + arg_def.name)
  tensor_type = 'Tensor<' + base_type + '>'
  if arg_def.number_attr or arg_def.type_list_attr:
    return '[' + tensor_type + ']'
  return tensor_type


def generate_code(op, enum_store):
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

  return_types = [arg_def_type_as_string(a) for a in op.output_arg]
  return_type = ''
  if len(return_types) == 1:
    return_type = ' -> ' + return_types[0]
  elif len(return_types) > 1:
    return_type = ' -> (' + ', '.join(return_types) + ')'

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
      '\n  ' + name + ': ' + type_and_default_value
      for name, type_and_default_value in (
          input_names_and_types +
          attr_names_and_types +
          missing_types)]
  return (
      """@_inlineable @inline(__always)
public static func {function_name}{generics_type}({joined_inputs}
){return_type} {{
  return #tfop({tfop_args})
}}""".format(function_name=swiftified_name(op.name),
             generics_type=generics_type,
             joined_inputs=','.join(all_inputs),
             return_type=return_type,
             tfop_args=tfop_args))


def main(argv):
  del argv  # Unused.

  proto = op_def_pb2.OpList()
  if FLAGS.ops_path is not None:
    with tf.gfile.Open(FLAGS.ops_path, 'r') as fobj:
      data = fobj.read()
    text_format.Parse(data, proto)
  else:
    tf_buffer = c_api.TF_GetAllOpList()
    proto.ParseFromString(c_api.TF_GetBuffer(tf_buffer))


  op_codes = []
  enum_store = EnumStore()
  for op in sorted(proto.op, key=lambda x: x.name):
    try:
      if op.name[0] == '_': continue
      op_codes.append(generate_code(op, enum_store))
    except UnableToGenerateCodeError as e:
      print('Cannot generate code for %s: %s' % (op.name, e.details))
  print('Generated code for %d/%d ops.'  % (len(op_codes), len(proto.op)))

  swift_code = (
      _WARNING +
      _HEADER +
      '\npublic enum Raw {\n\n' +
      '\n\n'.join(enum_store.enum_codes()) +
      '\n\n' +
      '\n\n'.join(op_codes) +
      '\n\n}')
  output_path = os.path.join(FLAGS.src_dir, _OUTPUT_FILE)
  with tf.gfile.Open(output_path, 'w') as fobj:
    fobj.write(swift_code)


if __name__ == '__main__':
  tf.app.run(main)
