#ifndef _ROLLERS_
#define _ROLLERS_
#define BOOST_DISABLE_ASSERTS
/* #include "crud.h" */
#include <cmath>
#include<iostream>
#include<string>
#include<vector>
#include <chrono>

#include <boost/range/adaptors.hpp>
#include <boost/date_time/gregorian/gregorian_types.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/multi_array.hpp>

// #include <H5Cpp.h>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

// #include "../movingWindow.h"

// #define GNUPLOT_ENABLE_PTY
// #include "gnuplot-iostream.h"
using namespace std;

using Eigen::MatrixXd;
using Eigen::Matrix;
using Eigen::Vector;
using Eigen::Dynamic;
using Eigen::RowMajor;

typedef vector< tuple<time_t, double> > dataType;
/* typedef tuple<vector<time_t>, vector<double>> dataType; */
typedef tuple<vector<time_t>, vector<double>> dataType1;
typedef Matrix<double, Dynamic, Dynamic> EigOHLC;
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

using namespace std;
namespace py = pybind11;
using boost::multi_array;
using boost::extents;

const auto now = std::chrono::high_resolution_clock::now;
/* typedef vector<vector<time_t>, vector<double>> dataType; */
typedef deque<pair<double, uint64_t>> dequeType;
/* typedef Matrix<double, Dynamic, 1> arrType; */
/* typedef Matrix<uint64_t, Dynamic, 1> timestampsType; */
/* typedef Matrix<uint64_t, Dynamic, 1> timearrType; */
typedef Vector<double, Dynamic> arrType;
typedef Vector<uint64_t, Dynamic> timestampsType;
typedef Vector<uint64_t, Dynamic> timearrType;
typedef multi_array<double, 3> outType;
typedef multi_array<bool, 3> outBoolType;
typedef Eigen::Array<bool, Dynamic, Dynamic, Eigen::RowMajor> zoneBoolType;
typedef multi_array<bool, 2> zoneType;
typedef boost::multi_array_types::index_range range;

struct timePeriod{
  int start;
  int end;
  timePeriod(int start, int end): start(start), end(end) {};
};


// dataType movingWindowMean(dataType rawPrice, int window);
// vector<double> movingWindowMean(vector<time_t> timeStamps, vector<double> price, int window);
// Matrix<double, Dynamic, 1> movingWindowMean(Matrix<time_t, Dynamic, 1> timeStamps, Matrix<double, Dynamic, 1> price, int window);
// vector<double> movingWindowMax(vector<double> arr, int window);


inline auto timeFuncInvocation =
  [](auto&& func, auto&&... params) {
  // get time before function invocation
  const auto& start = std::chrono::high_resolution_clock::now();
  // function invocation using perfect forwarding
  std::forward<decltype(func)>(func)(std::forward<decltype(params)>(params)...);
  // get time after function invocation
  const auto& stop = std::chrono::high_resolution_clock::now();
  return chrono::duration_cast<chrono::nanoseconds>(stop - start).count();
  // return (stop - start).count();
};
/* dataType movingWindowMean(dataType rawPrice, int window); */
/* /\* dataType movingWindowMean(vector<time_t> timeStamps, vector<double> price, int window); *\/ */
/* /\* void movingWindowMean(vector<time_t> timeStamps, vector<double> price, int window, vector<float> &out); *\/ */
/* /\* dataType1 movingWindowMean(vector<time_t> timeStamps, vector<double> price, int window); *\/ */
/* vector<double> movingWindowMean(vector<time_t> timeStamps, vector<double> price, int window); */
/* Matrix<double, Dynamic, 1> movingWindowMean(Matrix<time_t, Dynamic, 1> timeStamps, Matrix<double, Dynamic, 1> price, int window); */
/* vector<double> movingWindowMax(vector<double> arr, int window); */


class RollerX{
 public:
  bool initialized;
  uint64_t N, I, left_idx;
  int ntimeframes, ntimeframes_cont, nzones, sampling_tf_idx, nfeats=8;

 public:

  RollerX(vector<uint64_t> timeframes, int nzones=0);
  outType roll(py::EigenDRef<arrType> arr, py::EigenDRef<timestampsType> timestamps, bool sample=false, string sample_condition="", int sampling_tf_idx=0);
  outType roll(py::EigenDRef<arrType> arr, py::EigenDRef<timestampsType> timestamps, py::EigenDRef<zoneBoolType> zones, bool sample=false, string sample_condition="", int sampling_tf_idx=0);
  void highlowSample();

 private: // methods
  void ingest(const arrType& arr, const timearrType& timestamps, outType& out, zoneBoolType& zones);

  void _roll(const arrType& arr, const timearrType& timestamps, outType& out);
  void _roll(const arrType& arr, const timearrType& timearr, outType& out, zoneBoolType& zones);
  void _roll_cont(const arrType& arr, const timearrType& timearr, outType& out);

  void _step(const arrType& arr, const timearrType& timearr, outType& out, const Eigen::Ref<const zoneBoolType>& zones);
  void _step(const arrType& arr, const timearrType& timearr, outType& out);
  void _step(const arrType& arr, const timearrType& timearr, const Eigen::Ref<const zoneBoolType>& zones);
  void _step(const arrType& arr, const timearrType& timearr);

  void head_next(double price, int tf);
  void tail_update(const arrType& arr, const timearrType& timearr, int tf);
  void adjust_minmaxdeques(double price, int tf);

  void head_next_zones(double price, int tf, bool zone_bool, bool zone_bool_prev);
  void tail_update_zones(const arrType& arr, const timearrType& timearr, int tf, bool zone_bool, bool zone_bool_prev);
  void adjust_minmaxdeques_zones(double price, int tf, bool zone_bool, bool zone_bool_prev);

  void update_out(const arrType& arr, uint64_t idx, int tf, outType& out);

  bool highlow_condition(double val);

 private: // attributes

  vector <uint64_t> counts, timeframes, timeframes_lids, timeout_view, idxout_view;

  /* boost::multi_array<bool, 2> session_bool; */
  /* boost::multi_array<uint8_t, 2> current_hours, session_hours; */
  /* Matrix<bool, Dynamic, Dynamic> session_bool; */
  /* Matrix<uint8_t, Dynamic, Dynamic> current_hours, session_hours; */
  /* vector<timePeriod> zones; */

  /* map<int, dequeType> maxques; */
  /* map<int, dequeType> minques; */
  vector<dequeType> maxques;
  vector<dequeType> minques;
  vector<double> means;
  vector<double> delt;
  vector<double> vol_sum;
  vector<double> vols;
  vector<double> skew;
  vector<double> kurt;
  vector<double> ssqs;
  vector<double> stds;
  /* boost::multi_array<double, 2> rets; */
  Matrix<double, Dynamic, Dynamic> rets;
  arrType arr_memory;
  timearrType timearr_memory;

  /* boost::multi_array<uint8_t, 2> current_hours_mem; */
  /* boost::multi_array<bool, 2> session_bool_mem; */
  /* boost::multi_array<double, 2> rets_memory; */
  /* Matrix<uint8_t, Dynamic, Dynamic> current_hours_memory; */
  /* Matrix<bool, Dynamic, Dynamic> session_bool_memory; */
  zoneBoolType zones_memory;
  Matrix<double, Dynamic, Dynamic> rets_memory;

  /* outType out; */
  timearrType timeout;
  arrType close;

  size_t step_cont_time=0;
  size_t head_next_time=0;
  size_t tail_update_time=0;
  size_t minmax_time=0;
  size_t update_time=0;
  /* chrono::duration<size_t, ratio<1, 1000000000>> head_next_time; */
  /* chrono::duration<size_t, ratio<1, 1000000000>> tail_update_time; */
  /* chrono::duration<size_t, ratio<1, 1000000000>> minmax_time; */
  /* timePeriod SYDNEY(7, 16); */
  /* timePeriod TOKYO (9, 18); */
  /* timePeriod LONDON(8, 16); */
  /* timePeriod NY(8, 17); */
};


class RollerY{
 public:
  bool initialized;
  uint64_t N, I, left_idx;
  int ntimeframes, ntimeframes_cont, nzones, sampling_tf_idx, nfeats=8, nlabels=11;

 public:

  RollerY(vector<uint64_t> timeframes, int nzones=0, double eps_thresh=0., double range_multipler=2.);
  outBoolType roll(py::EigenDRef<arrType> arr, py::array_t<double> _xFeats, py::array_t<double> _yFeats, py::EigenDRef<timearrType> timestamps);
  outBoolType roll(py::EigenDRef<arrType> arr, py::array_t<double> _xFeats, outType& _yFeats, py::EigenDRef<timearrType> timestamps);
  outBoolType roll(py::EigenDRef<arrType> arr, outType& _xFeats, outType& _yFeats, py::EigenDRef<timearrType> timestamps);
  outBoolType roll(py::EigenDRef<arrType> arr, py::array_t<double> _xFeats, py::EigenDRef<timearrType> timestamps);
  outBoolType roll(py::EigenDRef<arrType> arr, py::EigenDRef<timestampsType> timestamps, py::array_t<double> prevOuts, py::EigenDRef<zoneBoolType> zones, bool sample=false, string sample_condition="", int sampling_tf_idx=0);
  outType shift(py::array_t<double> _prevOuts, py::EigenDRef<timearrType> timestamps);
  outType shift(outType& xFeats, py::EigenDRef<timearrType> timestamps);
  void highlowSample();

 private: // methods
  void ingest(const outType& prevOuts, const timearrType& timestamps, const outBoolType& outLabels);
  void ingest(const outType& prevOuts, const timearrType& timestamps, const outType& outCont);
  /* void ingest(const arrType& arr, const timearrType& timestamps, outType& outCont, outBoolType& outLabels); */

  void _roll(const arrType& arr, const timearrType& timestamps, outType& out);
  void _roll(const arrType& arr, const timearrType& timearr, outType& out, zoneBoolType& zones);
  void _roll_cont(const arrType& arr, const timearrType& timearr, outType& out);
  void _step(const arrType& arr, const timearrType& timearr, const outType& prevOuts, outType& outCont, outBoolType& outLabels);
  void _step(const arrType& arr, const timearrType& timearr, bool sample);

  void head_next(const arrType& arr, int tf);
  void tail_update(const arrType& arr, const timearrType& timearr, int tf);
  void adjust_minmaxdeques(double price, int tf);

  void head_next_zones(double price, int tf, bool zone_bool, bool zone_bool_prev);
  void tail_update_zones(const arrType& arr, const timearrType& timearr, int tf, bool zone_bool, bool zone_bool_prev);
  void adjust_minmaxdeques_zones(double price, int tf, bool zone_bool, bool zone_bool_prev);

  template<int>
  void crawl_path(py::EigenDRef<arrType> arr, const outType& prevOut, const outType& outCont, outBoolType& outLabels, uint64_t idx, int tf);
  void update_out(const arrType& arr, const timearrType& timearr, uint64_t idx, int tf, const outType& prevOut, outType& outCont, outBoolType& outLabels);

  bool highlow_condition(double val);

 private: // attributes

  vector <uint64_t> counts, timeframes, timeframes_rids, timeout_view, idxout_view;
  double eps, range_multiplier;

  /* boost::multi_array<bool, 2> session_bool; */
  /* boost::multi_array<uint8_t, 2> current_hours, session_hours; */
  /* Matrix<bool, Dynamic, Dynamic> session_bool; */
  /* Matrix<uint8_t, Dynamic, Dynamic> current_hours, session_hours; */
  /* vector<timePeriod> zones; */

  vector<dequeType> maxques;
  vector<dequeType> minques;
  vector<double> means;
  vector<double> delt;
  vector<double> vol_sum;
  vector<double> vols;
  vector<double> skew;
  vector<double> kurt;
  vector<double> ssqs;
  vector<double> stds;
  /* boost::multi_array<double, 2> rets; */
  Matrix<double, Dynamic, Dynamic> rets;
  arrType arr_memory;
  timearrType timearr_memory;
  outType prevOuts_memory;
  outType yprevOuts_memory;
  /* boost::multi_array<uint8_t, 2> current_hours_mem; */
  /* boost::multi_array<bool, 2> session_bool_mem; */
  /* boost::multi_array<double, 2> rets_memory; */
  /* Matrix<uint8_t, Dynamic, Dynamic> current_hours_memory; */
  /* Matrix<bool, Dynamic, Dynamic> session_bool_memory; */
  zoneBoolType zones_memory;
  Matrix<double, Dynamic, Dynamic> rets_memory;

  /* outType out; */
  timearrType timeout;
  arrType close;

  size_t step_cont_time=0;
  size_t head_next_time=0;
  size_t tail_update_time=0;
  size_t minmax_time=0;
  size_t update_time=0;
  /* chrono::duration<size_t, ratio<1, 1000000000>> head_next_time; */
  /* chrono::duration<size_t, ratio<1, 1000000000>> tail_update_time; */
  /* chrono::duration<size_t, ratio<1, 1000000000>> minmax_time; */
  /* timePeriod SYDNEY(7, 16); */
  /* timePeriod TOKYO (9, 18); */
  /* timePeriod LONDON(8, 16); */
  /* timePeriod NY(8, 17); */
};

#endif
