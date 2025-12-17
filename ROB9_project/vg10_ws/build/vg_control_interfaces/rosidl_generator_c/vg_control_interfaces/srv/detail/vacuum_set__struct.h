// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from vg_control_interfaces:srv/VacuumSet.idl
// generated code does not contain a copyright notice

#ifndef VG_CONTROL_INTERFACES__SRV__DETAIL__VACUUM_SET__STRUCT_H_
#define VG_CONTROL_INTERFACES__SRV__DETAIL__VACUUM_SET__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/VacuumSet in the package vg_control_interfaces.
typedef struct vg_control_interfaces__srv__VacuumSet_Request
{
  /// Vacuum level for channel A (0-255)
  int32_t channel_a;
  /// Vacuum level for channel B (0-255)
  int32_t channel_b;
} vg_control_interfaces__srv__VacuumSet_Request;

// Struct for a sequence of vg_control_interfaces__srv__VacuumSet_Request.
typedef struct vg_control_interfaces__srv__VacuumSet_Request__Sequence
{
  vg_control_interfaces__srv__VacuumSet_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} vg_control_interfaces__srv__VacuumSet_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'message'
#include "rosidl_runtime_c/string.h"

/// Struct defined in srv/VacuumSet in the package vg_control_interfaces.
typedef struct vg_control_interfaces__srv__VacuumSet_Response
{
  bool success;
  rosidl_runtime_c__String message;
} vg_control_interfaces__srv__VacuumSet_Response;

// Struct for a sequence of vg_control_interfaces__srv__VacuumSet_Response.
typedef struct vg_control_interfaces__srv__VacuumSet_Response__Sequence
{
  vg_control_interfaces__srv__VacuumSet_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} vg_control_interfaces__srv__VacuumSet_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // VG_CONTROL_INTERFACES__SRV__DETAIL__VACUUM_SET__STRUCT_H_
