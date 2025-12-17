// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from vg_control_interfaces:srv/VacuumSet.idl
// generated code does not contain a copyright notice
#include "vg_control_interfaces/srv/detail/vacuum_set__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"

bool
vg_control_interfaces__srv__VacuumSet_Request__init(vg_control_interfaces__srv__VacuumSet_Request * msg)
{
  if (!msg) {
    return false;
  }
  // channel_a
  // channel_b
  return true;
}

void
vg_control_interfaces__srv__VacuumSet_Request__fini(vg_control_interfaces__srv__VacuumSet_Request * msg)
{
  if (!msg) {
    return;
  }
  // channel_a
  // channel_b
}

bool
vg_control_interfaces__srv__VacuumSet_Request__are_equal(const vg_control_interfaces__srv__VacuumSet_Request * lhs, const vg_control_interfaces__srv__VacuumSet_Request * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // channel_a
  if (lhs->channel_a != rhs->channel_a) {
    return false;
  }
  // channel_b
  if (lhs->channel_b != rhs->channel_b) {
    return false;
  }
  return true;
}

bool
vg_control_interfaces__srv__VacuumSet_Request__copy(
  const vg_control_interfaces__srv__VacuumSet_Request * input,
  vg_control_interfaces__srv__VacuumSet_Request * output)
{
  if (!input || !output) {
    return false;
  }
  // channel_a
  output->channel_a = input->channel_a;
  // channel_b
  output->channel_b = input->channel_b;
  return true;
}

vg_control_interfaces__srv__VacuumSet_Request *
vg_control_interfaces__srv__VacuumSet_Request__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  vg_control_interfaces__srv__VacuumSet_Request * msg = (vg_control_interfaces__srv__VacuumSet_Request *)allocator.allocate(sizeof(vg_control_interfaces__srv__VacuumSet_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(vg_control_interfaces__srv__VacuumSet_Request));
  bool success = vg_control_interfaces__srv__VacuumSet_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
vg_control_interfaces__srv__VacuumSet_Request__destroy(vg_control_interfaces__srv__VacuumSet_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    vg_control_interfaces__srv__VacuumSet_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
vg_control_interfaces__srv__VacuumSet_Request__Sequence__init(vg_control_interfaces__srv__VacuumSet_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  vg_control_interfaces__srv__VacuumSet_Request * data = NULL;

  if (size) {
    data = (vg_control_interfaces__srv__VacuumSet_Request *)allocator.zero_allocate(size, sizeof(vg_control_interfaces__srv__VacuumSet_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = vg_control_interfaces__srv__VacuumSet_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        vg_control_interfaces__srv__VacuumSet_Request__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
vg_control_interfaces__srv__VacuumSet_Request__Sequence__fini(vg_control_interfaces__srv__VacuumSet_Request__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      vg_control_interfaces__srv__VacuumSet_Request__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

vg_control_interfaces__srv__VacuumSet_Request__Sequence *
vg_control_interfaces__srv__VacuumSet_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  vg_control_interfaces__srv__VacuumSet_Request__Sequence * array = (vg_control_interfaces__srv__VacuumSet_Request__Sequence *)allocator.allocate(sizeof(vg_control_interfaces__srv__VacuumSet_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = vg_control_interfaces__srv__VacuumSet_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
vg_control_interfaces__srv__VacuumSet_Request__Sequence__destroy(vg_control_interfaces__srv__VacuumSet_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    vg_control_interfaces__srv__VacuumSet_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
vg_control_interfaces__srv__VacuumSet_Request__Sequence__are_equal(const vg_control_interfaces__srv__VacuumSet_Request__Sequence * lhs, const vg_control_interfaces__srv__VacuumSet_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!vg_control_interfaces__srv__VacuumSet_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
vg_control_interfaces__srv__VacuumSet_Request__Sequence__copy(
  const vg_control_interfaces__srv__VacuumSet_Request__Sequence * input,
  vg_control_interfaces__srv__VacuumSet_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(vg_control_interfaces__srv__VacuumSet_Request);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    vg_control_interfaces__srv__VacuumSet_Request * data =
      (vg_control_interfaces__srv__VacuumSet_Request *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!vg_control_interfaces__srv__VacuumSet_Request__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          vg_control_interfaces__srv__VacuumSet_Request__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!vg_control_interfaces__srv__VacuumSet_Request__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `message`
#include "rosidl_runtime_c/string_functions.h"

bool
vg_control_interfaces__srv__VacuumSet_Response__init(vg_control_interfaces__srv__VacuumSet_Response * msg)
{
  if (!msg) {
    return false;
  }
  // success
  // message
  if (!rosidl_runtime_c__String__init(&msg->message)) {
    vg_control_interfaces__srv__VacuumSet_Response__fini(msg);
    return false;
  }
  return true;
}

void
vg_control_interfaces__srv__VacuumSet_Response__fini(vg_control_interfaces__srv__VacuumSet_Response * msg)
{
  if (!msg) {
    return;
  }
  // success
  // message
  rosidl_runtime_c__String__fini(&msg->message);
}

bool
vg_control_interfaces__srv__VacuumSet_Response__are_equal(const vg_control_interfaces__srv__VacuumSet_Response * lhs, const vg_control_interfaces__srv__VacuumSet_Response * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // success
  if (lhs->success != rhs->success) {
    return false;
  }
  // message
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->message), &(rhs->message)))
  {
    return false;
  }
  return true;
}

bool
vg_control_interfaces__srv__VacuumSet_Response__copy(
  const vg_control_interfaces__srv__VacuumSet_Response * input,
  vg_control_interfaces__srv__VacuumSet_Response * output)
{
  if (!input || !output) {
    return false;
  }
  // success
  output->success = input->success;
  // message
  if (!rosidl_runtime_c__String__copy(
      &(input->message), &(output->message)))
  {
    return false;
  }
  return true;
}

vg_control_interfaces__srv__VacuumSet_Response *
vg_control_interfaces__srv__VacuumSet_Response__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  vg_control_interfaces__srv__VacuumSet_Response * msg = (vg_control_interfaces__srv__VacuumSet_Response *)allocator.allocate(sizeof(vg_control_interfaces__srv__VacuumSet_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(vg_control_interfaces__srv__VacuumSet_Response));
  bool success = vg_control_interfaces__srv__VacuumSet_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
vg_control_interfaces__srv__VacuumSet_Response__destroy(vg_control_interfaces__srv__VacuumSet_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    vg_control_interfaces__srv__VacuumSet_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
vg_control_interfaces__srv__VacuumSet_Response__Sequence__init(vg_control_interfaces__srv__VacuumSet_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  vg_control_interfaces__srv__VacuumSet_Response * data = NULL;

  if (size) {
    data = (vg_control_interfaces__srv__VacuumSet_Response *)allocator.zero_allocate(size, sizeof(vg_control_interfaces__srv__VacuumSet_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = vg_control_interfaces__srv__VacuumSet_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        vg_control_interfaces__srv__VacuumSet_Response__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
vg_control_interfaces__srv__VacuumSet_Response__Sequence__fini(vg_control_interfaces__srv__VacuumSet_Response__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      vg_control_interfaces__srv__VacuumSet_Response__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

vg_control_interfaces__srv__VacuumSet_Response__Sequence *
vg_control_interfaces__srv__VacuumSet_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  vg_control_interfaces__srv__VacuumSet_Response__Sequence * array = (vg_control_interfaces__srv__VacuumSet_Response__Sequence *)allocator.allocate(sizeof(vg_control_interfaces__srv__VacuumSet_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = vg_control_interfaces__srv__VacuumSet_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
vg_control_interfaces__srv__VacuumSet_Response__Sequence__destroy(vg_control_interfaces__srv__VacuumSet_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    vg_control_interfaces__srv__VacuumSet_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
vg_control_interfaces__srv__VacuumSet_Response__Sequence__are_equal(const vg_control_interfaces__srv__VacuumSet_Response__Sequence * lhs, const vg_control_interfaces__srv__VacuumSet_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!vg_control_interfaces__srv__VacuumSet_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
vg_control_interfaces__srv__VacuumSet_Response__Sequence__copy(
  const vg_control_interfaces__srv__VacuumSet_Response__Sequence * input,
  vg_control_interfaces__srv__VacuumSet_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(vg_control_interfaces__srv__VacuumSet_Response);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    vg_control_interfaces__srv__VacuumSet_Response * data =
      (vg_control_interfaces__srv__VacuumSet_Response *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!vg_control_interfaces__srv__VacuumSet_Response__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          vg_control_interfaces__srv__VacuumSet_Response__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!vg_control_interfaces__srv__VacuumSet_Response__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
