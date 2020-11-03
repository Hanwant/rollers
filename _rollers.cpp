#include <vector>

#include "include/rollers.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
namespace py = pybind11;


PYBIND11_MODULE(_rollers, m){
  m.doc() = "Rollers";

  // m.def("movingWindowMean", py::overload_cast<Matrix<time_t, Dynamic, 1>, Matrix<double, Dynamic, 1>, int>(&movingWindowMean), "discrete window moving mean - useful for testing I/O",
  //       py::arg("timestamps"), py::arg("price"), py::arg("window"), py::return_value_policy::copy);

  py::class_<outType>(m, "outType", py::buffer_protocol())
    .def_buffer([](outType &m) -> py::buffer_info{
        py::ssize_t element_size = sizeof(double);
        py::ssize_t rows = m.shape()[0], nfeats = m.shape()[1], ntimeframes = m.shape()[2];
        return py::buffer_info(m.data(),
                               element_size,
                               py::format_descriptor<double>::format(),
                               3,
                               py::detail::any_container<py::ssize_t>({rows, nfeats, ntimeframes}),
                               py::detail::any_container<py::ssize_t>({element_size * nfeats * ntimeframes, element_size * ntimeframes, element_size})
                               );
      });
  py::class_<outBoolType>(m, "outBoolType", py::buffer_protocol())
    .def_buffer([](outBoolType &m) -> py::buffer_info{
        py::ssize_t element_size = sizeof(bool);
        py::ssize_t rows = m.shape()[0], nfeats = m.shape()[1], ntimeframes = m.shape()[2];
        return py::buffer_info(m.data(),
                               element_size,
                               py::format_descriptor<bool>::format(),
                               3,
                               py::detail::any_container<py::ssize_t>({rows, nfeats, ntimeframes}),
                               py::detail::any_container<py::ssize_t>({element_size * nfeats * ntimeframes, element_size * ntimeframes, element_size})
                               );
      });

  py::class_<RollerX>(m, "RollerX")
    .def(py::init<vector<uint64_t>, int> (), py::arg("timeframes"), py::arg("nzones")=int(0))
    .def_readonly("I", &RollerX::I)
    .def_readonly("N", &RollerX::N)
    .def_readonly("nfeats", &RollerX::nfeats)
    .def_readonly("ntimeframes", &RollerX::ntimeframes)
    .def("roll", (outType (RollerX::*)(py::EigenDRef<arrType>, py::EigenDRef<timestampsType>, bool, string, int)) &RollerX::roll,
         "Apply rolling functions to inputs - continuous time frames only",
         py::arg("arr"), py::arg("timestamps"), py::arg("sample")=false,
         py::arg("sample_condition")="", py::arg("sampling_tf_idx")=0,
         // py::return_value_policy::take_ownership)
         py::return_value_policy::copy)
    .def("roll", (outType (RollerX::*)(py::EigenDRef<arrType>, py::EigenDRef<timestampsType>, py::EigenDRef<zoneBoolType>, bool, string, int)) &RollerX::roll,
         "Apply rolling functions to inputs - continuous and discrete time frames",
         py::arg("arr"), py::arg("timestamps"), py::arg("zones"), py::arg("sample")=false,
         py::arg("sample_condition")="", py::arg("sampling_tf_idx")=0,
         py::return_value_policy::take_ownership)
    ;

  py::class_<RollerY>(m, "RollerY")
    .def(py::init<vector<uint64_t>, int> (), py::arg("timeframes"), py::arg("nzones")=int(0))
    .def_readonly("I", &RollerY::I)
    .def_readonly("N", &RollerY::N)
    .def_readonly("nfeats", &RollerY::nfeats)
    .def_readonly("nlabels", &RollerY::nlabels)
    .def_readonly("ntimeframes", &RollerY::ntimeframes)
    .def("roll", (outBoolType (RollerY::*)(py::EigenDRef<arrType>, py::array_t<double>, py::array_t<double>, py::EigenDRef<timearrType> timestamps)) &RollerY::roll,
         "Apply rolling functions to inputs - continuous time frames only",
         py::arg("priceArr"), py::arg("xFeats"), py::arg("yFeats"), py::arg("timestamps"),
         py::return_value_policy::take_ownership)
    .def("roll", (outBoolType (RollerY::*)(py::EigenDRef<arrType> arr, py::array_t<double>, py::EigenDRef<timearrType> timestamps)) &RollerY::roll,
         "Apply rolling functions to inputs - continuous and discrete time frames",
         py::arg("priceArr"), py::arg("xFeats"), py::arg("timestamps"),
         py::return_value_policy::take_ownership)
    .def("shift", (outType (RollerY::*)(py::array_t<double>, py::EigenDRef<timestampsType>)) &RollerY::shift,
         "Shift output of RollerX to get future regression targets",
         py::arg("xfeatures"), py::arg("timestamps"),
         py::return_value_policy::take_ownership)
    ;
}
