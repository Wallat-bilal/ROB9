// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from vg_control_interfaces:srv/VacuumRelease.idl
// generated code does not contain a copyright notice

#ifndef VG_CONTROL_INTERFACES__SRV__DETAIL__VACUUM_RELEASE__TRAITS_HPP_
#define VG_CONTROL_INTERFACES__SRV__DETAIL__VACUUM_RELEASE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "vg_control_interfaces/srv/detail/vacuum_release__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace vg_control_interfaces
{

namespace srv
{

inline void to_flow_style_yaml(
  const VacuumRelease_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: release_vacuum
  {
    out << "release_vacuum: ";
    rosidl_generator_traits::value_to_yaml(msg.release_vacuum, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const VacuumRelease_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: release_vacuum
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "release_vacuum: ";
    rosidl_generator_traits::value_to_yaml(msg.release_vacuum, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const VacuumRelease_Request & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace vg_control_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use vg_control_interfaces::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const vg_control_interfaces::srv::VacuumRelease_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  vg_control_interfaces::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use vg_control_interfaces::srv::to_yaml() instead")]]
inline std::string to_yaml(const vg_control_interfaces::srv::VacuumRelease_Request & msg)
{
  return vg_control_interfaces::srv::to_yaml(msg);
}

template<>
inline const char * data_type<vg_control_interfaces::srv::VacuumRelease_Request>()
{
  return "vg_control_interfaces::srv::VacuumRelease_Request";
}

template<>
inline const char * name<vg_control_interfaces::srv::VacuumRelease_Request>()
{
  return "vg_control_interfaces/srv/VacuumRelease_Request";
}

template<>
struct has_fixed_size<vg_control_interfaces::srv::VacuumRelease_Request>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<vg_control_interfaces::srv::VacuumRelease_Request>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<vg_control_interfaces::srv::VacuumRelease_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace vg_control_interfaces
{

namespace srv
{

inline void to_flow_style_yaml(
  const VacuumRelease_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: success
  {
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << ", ";
  }

  // member: message
  {
    out << "message: ";
    rosidl_generator_traits::value_to_yaml(msg.message, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const VacuumRelease_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: success
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << "\n";
  }

  // member: message
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "message: ";
    rosidl_generator_traits::value_to_yaml(msg.message, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const VacuumRelease_Response & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace vg_control_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use vg_control_interfaces::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const vg_control_interfaces::srv::VacuumRelease_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  vg_control_interfaces::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use vg_control_interfaces::srv::to_yaml() instead")]]
inline std::string to_yaml(const vg_control_interfaces::srv::VacuumRelease_Response & msg)
{
  return vg_control_interfaces::srv::to_yaml(msg);
}

template<>
inline const char * data_type<vg_control_interfaces::srv::VacuumRelease_Response>()
{
  return "vg_control_interfaces::srv::VacuumRelease_Response";
}

template<>
inline const char * name<vg_control_interfaces::srv::VacuumRelease_Response>()
{
  return "vg_control_interfaces/srv/VacuumRelease_Response";
}

template<>
struct has_fixed_size<vg_control_interfaces::srv::VacuumRelease_Response>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<vg_control_interfaces::srv::VacuumRelease_Response>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<vg_control_interfaces::srv::VacuumRelease_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<vg_control_interfaces::srv::VacuumRelease>()
{
  return "vg_control_interfaces::srv::VacuumRelease";
}

template<>
inline const char * name<vg_control_interfaces::srv::VacuumRelease>()
{
  return "vg_control_interfaces/srv/VacuumRelease";
}

template<>
struct has_fixed_size<vg_control_interfaces::srv::VacuumRelease>
  : std::integral_constant<
    bool,
    has_fixed_size<vg_control_interfaces::srv::VacuumRelease_Request>::value &&
    has_fixed_size<vg_control_interfaces::srv::VacuumRelease_Response>::value
  >
{
};

template<>
struct has_bounded_size<vg_control_interfaces::srv::VacuumRelease>
  : std::integral_constant<
    bool,
    has_bounded_size<vg_control_interfaces::srv::VacuumRelease_Request>::value &&
    has_bounded_size<vg_control_interfaces::srv::VacuumRelease_Response>::value
  >
{
};

template<>
struct is_service<vg_control_interfaces::srv::VacuumRelease>
  : std::true_type
{
};

template<>
struct is_service_request<vg_control_interfaces::srv::VacuumRelease_Request>
  : std::true_type
{
};

template<>
struct is_service_response<vg_control_interfaces::srv::VacuumRelease_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // VG_CONTROL_INTERFACES__SRV__DETAIL__VACUUM_RELEASE__TRAITS_HPP_
