# generated from rosidl_generator_py/resource/_idl.py.em
# with input from vg_control_interfaces:srv/VacuumSet.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_VacuumSet_Request(type):
    """Metaclass of message 'VacuumSet_Request'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('vg_control_interfaces')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'vg_control_interfaces.srv.VacuumSet_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__vacuum_set__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__vacuum_set__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__vacuum_set__request
            cls._TYPE_SUPPORT = module.type_support_msg__srv__vacuum_set__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__vacuum_set__request

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class VacuumSet_Request(metaclass=Metaclass_VacuumSet_Request):
    """Message class 'VacuumSet_Request'."""

    __slots__ = [
        '_channel_a',
        '_channel_b',
    ]

    _fields_and_field_types = {
        'channel_a': 'int32',
        'channel_b': 'int32',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.channel_a = kwargs.get('channel_a', int())
        self.channel_b = kwargs.get('channel_b', int())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.channel_a != other.channel_a:
            return False
        if self.channel_b != other.channel_b:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def channel_a(self):
        """Message field 'channel_a'."""
        return self._channel_a

    @channel_a.setter
    def channel_a(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'channel_a' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'channel_a' field must be an integer in [-2147483648, 2147483647]"
        self._channel_a = value

    @builtins.property
    def channel_b(self):
        """Message field 'channel_b'."""
        return self._channel_b

    @channel_b.setter
    def channel_b(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'channel_b' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'channel_b' field must be an integer in [-2147483648, 2147483647]"
        self._channel_b = value


# Import statements for member types

# already imported above
# import builtins

# already imported above
# import rosidl_parser.definition


class Metaclass_VacuumSet_Response(type):
    """Metaclass of message 'VacuumSet_Response'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('vg_control_interfaces')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'vg_control_interfaces.srv.VacuumSet_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__vacuum_set__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__vacuum_set__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__vacuum_set__response
            cls._TYPE_SUPPORT = module.type_support_msg__srv__vacuum_set__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__vacuum_set__response

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class VacuumSet_Response(metaclass=Metaclass_VacuumSet_Response):
    """Message class 'VacuumSet_Response'."""

    __slots__ = [
        '_success',
        '_message',
    ]

    _fields_and_field_types = {
        'success': 'boolean',
        'message': 'string',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.success = kwargs.get('success', bool())
        self.message = kwargs.get('message', str())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.success != other.success:
            return False
        if self.message != other.message:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def success(self):
        """Message field 'success'."""
        return self._success

    @success.setter
    def success(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'success' field must be of type 'bool'"
        self._success = value

    @builtins.property
    def message(self):
        """Message field 'message'."""
        return self._message

    @message.setter
    def message(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'message' field must be of type 'str'"
        self._message = value


class Metaclass_VacuumSet(type):
    """Metaclass of service 'VacuumSet'."""

    _TYPE_SUPPORT = None

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('vg_control_interfaces')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'vg_control_interfaces.srv.VacuumSet')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__srv__vacuum_set

            from vg_control_interfaces.srv import _vacuum_set
            if _vacuum_set.Metaclass_VacuumSet_Request._TYPE_SUPPORT is None:
                _vacuum_set.Metaclass_VacuumSet_Request.__import_type_support__()
            if _vacuum_set.Metaclass_VacuumSet_Response._TYPE_SUPPORT is None:
                _vacuum_set.Metaclass_VacuumSet_Response.__import_type_support__()


class VacuumSet(metaclass=Metaclass_VacuumSet):
    from vg_control_interfaces.srv._vacuum_set import VacuumSet_Request as Request
    from vg_control_interfaces.srv._vacuum_set import VacuumSet_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')
