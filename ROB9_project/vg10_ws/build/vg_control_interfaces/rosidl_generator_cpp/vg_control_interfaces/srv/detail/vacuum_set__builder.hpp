// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from vg_control_interfaces:srv/VacuumSet.idl
// generated code does not contain a copyright notice

#ifndef VG_CONTROL_INTERFACES__SRV__DETAIL__VACUUM_SET__BUILDER_HPP_
#define VG_CONTROL_INTERFACES__SRV__DETAIL__VACUUM_SET__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "vg_control_interfaces/srv/detail/vacuum_set__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace vg_control_interfaces
{

namespace srv
{

namespace builder
{

class Init_VacuumSet_Request_channel_b
{
public:
  explicit Init_VacuumSet_Request_channel_b(::vg_control_interfaces::srv::VacuumSet_Request & msg)
  : msg_(msg)
  {}
  ::vg_control_interfaces::srv::VacuumSet_Request channel_b(::vg_control_interfaces::srv::VacuumSet_Request::_channel_b_type arg)
  {
    msg_.channel_b = std::move(arg);
    return std::move(msg_);
  }

private:
  ::vg_control_interfaces::srv::VacuumSet_Request msg_;
};

class Init_VacuumSet_Request_channel_a
{
public:
  Init_VacuumSet_Request_channel_a()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_VacuumSet_Request_channel_b channel_a(::vg_control_interfaces::srv::VacuumSet_Request::_channel_a_type arg)
  {
    msg_.channel_a = std::move(arg);
    return Init_VacuumSet_Request_channel_b(msg_);
  }

private:
  ::vg_control_interfaces::srv::VacuumSet_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::vg_control_interfaces::srv::VacuumSet_Request>()
{
  return vg_control_interfaces::srv::builder::Init_VacuumSet_Request_channel_a();
}

}  // namespace vg_control_interfaces


namespace vg_control_interfaces
{

namespace srv
{

namespace builder
{

class Init_VacuumSet_Response_message
{
public:
  explicit Init_VacuumSet_Response_message(::vg_control_interfaces::srv::VacuumSet_Response & msg)
  : msg_(msg)
  {}
  ::vg_control_interfaces::srv::VacuumSet_Response message(::vg_control_interfaces::srv::VacuumSet_Response::_message_type arg)
  {
    msg_.message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::vg_control_interfaces::srv::VacuumSet_Response msg_;
};

class Init_VacuumSet_Response_success
{
public:
  Init_VacuumSet_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_VacuumSet_Response_message success(::vg_control_interfaces::srv::VacuumSet_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_VacuumSet_Response_message(msg_);
  }

private:
  ::vg_control_interfaces::srv::VacuumSet_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::vg_control_interfaces::srv::VacuumSet_Response>()
{
  return vg_control_interfaces::srv::builder::Init_VacuumSet_Response_success();
}

}  // namespace vg_control_interfaces

#endif  // VG_CONTROL_INTERFACES__SRV__DETAIL__VACUUM_SET__BUILDER_HPP_
