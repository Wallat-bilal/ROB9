// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from vg_control_interfaces:srv/VacuumRelease.idl
// generated code does not contain a copyright notice

#ifndef VG_CONTROL_INTERFACES__SRV__DETAIL__VACUUM_RELEASE__BUILDER_HPP_
#define VG_CONTROL_INTERFACES__SRV__DETAIL__VACUUM_RELEASE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "vg_control_interfaces/srv/detail/vacuum_release__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace vg_control_interfaces
{

namespace srv
{

namespace builder
{

class Init_VacuumRelease_Request_release_vacuum
{
public:
  Init_VacuumRelease_Request_release_vacuum()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::vg_control_interfaces::srv::VacuumRelease_Request release_vacuum(::vg_control_interfaces::srv::VacuumRelease_Request::_release_vacuum_type arg)
  {
    msg_.release_vacuum = std::move(arg);
    return std::move(msg_);
  }

private:
  ::vg_control_interfaces::srv::VacuumRelease_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::vg_control_interfaces::srv::VacuumRelease_Request>()
{
  return vg_control_interfaces::srv::builder::Init_VacuumRelease_Request_release_vacuum();
}

}  // namespace vg_control_interfaces


namespace vg_control_interfaces
{

namespace srv
{

namespace builder
{

class Init_VacuumRelease_Response_message
{
public:
  explicit Init_VacuumRelease_Response_message(::vg_control_interfaces::srv::VacuumRelease_Response & msg)
  : msg_(msg)
  {}
  ::vg_control_interfaces::srv::VacuumRelease_Response message(::vg_control_interfaces::srv::VacuumRelease_Response::_message_type arg)
  {
    msg_.message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::vg_control_interfaces::srv::VacuumRelease_Response msg_;
};

class Init_VacuumRelease_Response_success
{
public:
  Init_VacuumRelease_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_VacuumRelease_Response_message success(::vg_control_interfaces::srv::VacuumRelease_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_VacuumRelease_Response_message(msg_);
  }

private:
  ::vg_control_interfaces::srv::VacuumRelease_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::vg_control_interfaces::srv::VacuumRelease_Response>()
{
  return vg_control_interfaces::srv::builder::Init_VacuumRelease_Response_success();
}

}  // namespace vg_control_interfaces

#endif  // VG_CONTROL_INTERFACES__SRV__DETAIL__VACUUM_RELEASE__BUILDER_HPP_
