#include "../include/rollers.h"
#include <exception>
#include <chrono>
#include <boost/bind/bind.hpp>
// #include <valgrind/callgrind.h>

// const auto now = std::chrono::high_resolution_clock::now;
// namespace py = pybind11;

// timePeriod SYDNEY{7, 16};
// timePeriod TOKYO {9, 18};
// timePeriod LONDON{8, 16};
// timePeriod NY{8, 17};
// string SAMPLING_HIGHLOW = "SAMPLING_HIGHLOW";
// string SAMPLING_STD = "SAMPLING_STD";
// vector<string> SAMPLING_METHODS{SAMPLING_HIGHLOW, SAMPLING_STD};


// inline auto timeFuncInvocation =
//   [](auto&& func, auto&&... params) {
//   // get time before function invocation
//   const auto& start = std::chrono::high_resolution_clock::now();
//   // function invocation using perfect forwarding
//   std::forward<decltype(func)>(func)(std::forward<decltype(params)>(params)...);
//   // get time after function invocation
//   const auto& stop = std::chrono::high_resolution_clock::now();
//   return chrono::duration_cast<chrono::nanoseconds>(stop - start).count();
//   // return (stop - start).count();
// };


RollerX::RollerX(vector<uint64_t> timeframes, int nzones): timeframes(timeframes), nzones(nzones){ // ntimesessions refers to discrete blocks of time I.e the hour between
  // cout << "Constructor Called" << endl;                          // 12 and 1 or trading session.
  initialized = false;
  ntimeframes_cont = timeframes.size();
  ntimeframes = ntimeframes_cont + nzones;

  for (int tf=0; tf < ntimeframes; tf++){
    maxques.push_back(dequeType());
    minques.push_back(dequeType());
  }
  counts.resize(ntimeframes);
  delt.resize(ntimeframes);
  means.resize(ntimeframes);
  ssqs.resize(ntimeframes);
  stds.resize(ntimeframes);
  vols.resize(ntimeframes);
  vol_sum.resize(ntimeframes);
  timeframes_lids.resize(ntimeframes);
}

void RollerX::ingest(const arrType& arr, const timearrType& timestamps, outType& out, zoneBoolType& zones){
  N = arr.size();

  if (!initialized){
    initialized = true;
    out.resize(extents[N][nfeats][ntimeframes]);
    // INITIALIZE VALUES FOR CONTINUOUS TIMEFRAMES
    for (int tf = 0; tf < ntimeframes_cont; tf++){
      maxques[tf].push_back(make_pair(arr(0), 0));
      minques[tf].push_back(make_pair(arr(0), 0));
      rets(0, tf) = 0.;
      out[0][0][tf] = arr(0);
      out[0][1][tf] = arr(0);
      out[0][2][tf] = arr(0);
      out[0][3][tf] = NAN;
      out[0][4][tf] = arr(0);
      out[0][5][tf] = NAN;
      out[0][6][tf] = NAN;
      out[0][7][tf] = NAN;
    }
    // INITIALIZE VALUES FOR DISCRETE TIMEFRAMES
    for (int tf = ntimeframes_cont; tf < ntimeframes; tf++){
      bool sess_bool = zones(0, tf - ntimeframes_cont);
      // FOR FEATS WHICH CAN BE INITIALIZED (I.e Mean)
      // ONLY INITIALIZE IF CURRENTLY IN ZONE
      if(sess_bool){
        maxques[tf].push_back(make_pair(arr(0), 0));
        minques[tf].push_back(make_pair(arr(0), 0));
        rets(0, tf) = 0.;
        out[0][0][tf] = arr(0);
        out[0][1][tf] = arr(0);
        out[0][2][tf] = arr(0);
        out[0][3][tf] = NAN;
        out[0][4][tf] = arr(0);
        out[0][5][tf] = NAN;
        out[0][6][tf] = NAN;
        out[0][7][tf] = NAN;
      }
      // OTHERWISE, INITIALIZE AS NAN
      else{
        for (int feat=0; feat<nfeats; feat++){
          out[0][feat][tf] = NAN;
        }
      }
    }
    I = 0;
  }
  else{
    out.resize(extents[N][nfeats][ntimeframes]);
    for (int tf=0; tf<ntimeframes; tf++){
      timeframes_lids[tf] -= left_idx;
      int queue_size = maxques[tf].size();
      for(int i=0; i<queue_size; i++){
        pair<double, uint64_t> &vals = maxques[tf].front();
        maxques[tf].push_back(make_pair(vals.first, vals.second - left_idx));
        maxques[tf].pop_front();
      }
      queue_size = minques[tf].size();
      for(int i=0; i<queue_size; i++){
        pair<double, uint64_t> &vals = minques[tf].front();
        minques[tf].push_back(make_pair(vals.first, vals.second - left_idx));
        minques[tf].pop_front();
      }
    }
    I = arr_memory.rows();
  }
}

outType RollerX::roll(py::EigenDRef<arrType> arr_in, py::EigenDRef<timearrType> timestamps, bool sample, string sample_condition, int sampling_idx){
  zoneBoolType zones(0, 0);
  return roll(arr_in, timestamps, zones, sample, sample_condition, sampling_idx);

}

outType RollerX::roll(py::EigenDRef<arrType> arr_in, py::EigenDRef<timearrType> timestamps, py::EigenDRef<zoneBoolType> zones_in, bool sample, string sample_condition, int sampling_idx){
  sampling_tf_idx = sampling_idx;
  // SANITY CHECKS
  if(arr_in.rows() != timestamps.rows()){
    throw invalid_argument("arr_in and timestamps don't have equal length!");
  }

  if(zones_in.rows() > 0){
    if((zones_in.rows() == arr_in.rows()) && (zones_in.cols() == nzones)){
      // cout << "Doing continuous and discrete time operations" << endl;
    }
    else{
      string message = "Dimensions mismatch, nrows for price/time arrays: " + std::to_string(arr_in.rows()) + " nzones: " + std::to_string(nzones) + "\n";
      message += "Dimensions for zones array provided (rows, cols): (" + std::to_string(zones_in.rows()) + ", " + std::to_string(zones_in.cols()) +  ")\n";
      throw invalid_argument(message);
    }
  }
  else{
    if(nzones > 0){
      cout << "WARNING" << endl;
      cout << "Initialized with " << nzones << " zones but no zone data provided" << endl;
      throw std::invalid_argument("No Zone Data Provided");
      // cout << "Only returning continuous-timeframe states." << endl;
    }
    else{
      // cout << "Doing continuous time operations only" << endl;
    }
  }

  if(sample){
    if (sample_condition == "highlow"){
    }
    else{
      throw invalid_argument("Invalid sample_condition. Choices are: highlow");
    }
  }

  // Memory initialization for main input Arrays
  size_t start = arr_memory.rows(); // Used at end to output array
  arrType arr(arr_in.rows() + arr_memory.rows(), 1);
  timearrType timearr(timearr_memory.rows() + timestamps.rows(), 1);
  zoneBoolType zones(zones_memory.rows() + zones_in.rows(), zones_in.cols());
  rets.resize(rets_memory.rows() + arr_in.rows(), ntimeframes);
  if (start > 0){
    arr << arr_memory, arr_in;
    timearr << timearr_memory, timestamps;
    zones << zones_memory, zones_in;
    rets.topRows(start) << rets_memory;
  }
  else{
    arr << arr_in;
    timearr << timestamps;
    zones << zones_in;
  }
  outType out;

  // Memory init for internal + output arrays
  ingest(arr, timearr, out, zones);

  uint64_t idx(0); // for both rolling and sampling mode - used as indicator of output array length
  if(!sample){
    if(nzones>0){
      // cout << "Rolling with cont + zones" << endl;
      while(I <= N-1){
        _step(arr, timearr, out, zones);
        if (I > N-1) break;
        I++;
      }
      idx = I;
    }
    else{
      // cout << "Rolling with only cont" << endl;
      while(I <= N-1){
        _step(arr, timearr, out);
        if (I > N-1) break;
        I++;
      }
      idx = I;
      // cout << "Done Roll" << endl;
    }
    // TIMING FOR DEBUGGING
    // Hooks still need to be added manually before/after each function every time this is used
    // cout << "Time taken by functions (total ns): " << endl;
    // cout << "step_cont (ns, s)" << step_cont_time << ", " << (double)step_cont_time/1000000000 << endl;
    // cout << "head (ns, s): " << head_next_time<< ", " << (double)head_next_time/ 1000000000 << endl;
    // cout << "tail (ns, s): " << tail_update_time<< ", " << (double)tail_update_time/ 1000000000 <<  endl;
    // cout << "minmax (ns, s): " << minmax_time<< ", " << (double)minmax_time/ 1000000000 <<  endl;
    // cout << "update out (ns, s): " << update_time<< ", " << (double)update_time/ 1000000000 <<  endl;
  }
  else{
    idx = timearr_memory.rows();
    // cout << "sampling" << endl;
    bool (RollerX::*pSampleCondition)(double) = NULL;
    if(sample_condition == "highlow"){
      pSampleCondition = &RollerX::highlow_condition;
    }
    if(nzones>0){
       while(I <= N-1){

        _step(arr, timearr, zones); //"true" routes to overloaded func that doesn't update_out

        if ((this->*pSampleCondition)(arr(I))){
          for(int tf=0; tf<ntimeframes; tf++){
            update_out(arr, idx, tf, out);
          }
          idx++;
        }

        // if (I > N-1) break;
        I++;
      }
    }
    else{
       while(I <= N-1){

        _step(arr, timearr); // "true" routes to overloaded func that doesn't update_out

        if ((this->*pSampleCondition)(arr(I))){
          for(int tf=0; tf<ntimeframes; tf++){
            update_out(arr, idx, tf, out);
          }
          idx++;
        }
          // if (I > N-1) break;
        I++;
      }
    }
  }

  // Book-keeping for memory-arrays to be preserved for sequential stateful rolling.
  left_idx = *min_element(timeframes_lids.begin(), timeframes_lids.end());
  arr_memory.resize(N-left_idx);
  timearr_memory.resize(N-left_idx);
  rets_memory.resize(N-left_idx, ntimeframes);

  arr_memory = arr.bottomRows(N-left_idx);
  timearr_memory = timearr.bottomRows(N-left_idx);
  rets_memory = rets.bottomRows(N-left_idx); // rets.columns == ntimeframes

  if (nzones > 0){
    zones_memory.resize(N-left_idx, zones.cols());
    zones_memory = zones.bottomRows(N-left_idx);
  }
  // py::array_t<double> out_np({N, nfeats, ntimeframes},{nfeats*ntimeframes*8, ntimeframes*8, 8}, &out);
  // cout << "OUTPUTTING start: " << start << endl;
  // cout << "OUTPUTTING idx: " << idx << endl;
  // cout << "OUTPUTTING full length: " << out.shape()[0]<< endl;
  // if (sample){
  //   start=0;
  // }
  return out[ boost::indices[range(start, idx)][range()][range()]];
}


void RollerX::_step(const arrType& arr, const timearrType& timearr, outType& out, const Eigen::Ref<const zoneBoolType>& zones){
  // overloaded with zones matrix - does continuous + discrete timeframes
  bool zone_bool;
  bool zone_bool_prev;

  // first do continuous timeframes (probably inlined so no overhead of function call)
  _step(arr, timearr, out);

  for(int tf=ntimeframes_cont; tf<ntimeframes; tf++){
    zone_bool_prev = (I>0)? zones(I-1, tf-ntimeframes_cont): false;
    zone_bool = zones(I, tf-ntimeframes_cont);

    head_next_zones(arr(I), tf, zone_bool, zone_bool_prev);
    tail_update_zones(arr, timearr, tf, zone_bool, zone_bool_prev);
    adjust_minmaxdeques_zones(arr(I), tf, zone_bool, zone_bool_prev);

    update_out(arr, I, tf, out);
  }
}

void RollerX::_step(const arrType& arr, const timearrType& timearr, outType& out){
  // overloaded without zones matrix - only continuous timeframes
  for(int tf=0; tf<ntimeframes_cont; tf++){
    head_next(arr(I), tf);
    tail_update(arr, timearr, tf);
    adjust_minmaxdeques(arr(I), tf);
    update_out(arr, I, tf, out);
  }
}

void RollerX::_step(const arrType& arr, const timearrType& timearr, const Eigen::Ref<const zoneBoolType>& zones){
  // overloaded with zones matrix - does continuous + discrete timeframes
  bool zone_bool;
  bool zone_bool_prev;

  // first do continuous timeframes (probably inlined so no overhead of function call)
  _step(arr, timearr);

  for(int tf=ntimeframes_cont; tf<ntimeframes; tf++){
    zone_bool_prev = (I>0)? zones(I-1, tf-ntimeframes_cont): false;
    zone_bool = zones(I, tf-ntimeframes_cont);

    head_next_zones(arr(I), tf, zone_bool, zone_bool_prev);
    tail_update_zones(arr, timearr, tf, zone_bool, zone_bool_prev);
    adjust_minmaxdeques_zones(arr(I), tf, zone_bool, zone_bool_prev);
  }
}

void RollerX::_step(const arrType& arr, const timearrType& timearr){
  // overloaded without zones matrix - only continuous timeframes
  for(int tf=0; tf<ntimeframes_cont; tf++){
    head_next(arr(I), tf);
    tail_update(arr, timearr, tf);
    adjust_minmaxdeques(arr(I), tf);
  }
}

void RollerX::head_next(double price, int tf){
  counts[tf] += 1;
  double delt = price - means[tf];
  means[tf] += delt / counts[tf];
  ssqs[tf] += abs(delt * (price - means[tf]));
}

void RollerX::head_next_zones(double price, int tf, bool zone_bool, bool zone_bool_prev){
  if (zone_bool){
    if (zone_bool_prev){
      counts[tf] += 1;
      double delt = price - means[tf];
      means[tf] += delt / counts[tf];
      ssqs[tf] += abs(delt * (price - means[tf]));
    }
    else{
      counts[tf] = 1;
      means[tf] = price;
      ssqs[tf] = 0.;
      vol_sum[tf] = 0.;
    }
  }
}

void RollerX::tail_update(const arrType& arr, const timearrType& timearr, int tf){
  while (timearr(timeframes_lids[tf]) <= timearr(I) - timeframes[tf]){
    if (counts[tf] > 1){
      counts[tf] -= 1;
      double delt2 = arr(timeframes_lids[tf]) - means[tf];
      means[tf] -= delt2 / counts[tf];
      ssqs[tf] -= abs(delt2 * (arr(timeframes_lids[tf]) - means[tf]));
      vol_sum[tf] -= (rets(timeframes_lids[tf], tf) * rets(timeframes_lids[tf], tf));
    }
    timeframes_lids[tf] += 1;
  }
  rets(I, tf) = arr(I) - arr(timeframes_lids[tf]);
  vol_sum[tf] += rets(I, tf) * rets(I, tf);
  if (counts[tf] <=1){
    stds[tf] = NAN;
    vols[tf] = NAN;
  }
  else{
    stds[tf] = sqrt(ssqs[tf] / (counts[tf] - 1));
    vols[tf] = sqrt(vol_sum[tf] / (counts[tf] -1));
  }
}

void RollerX::tail_update_zones(const arrType& arr, const timearrType& timearr, int tf, bool zone_bool, bool zone_bool_prev){
  if (zone_bool){
    if(!zone_bool_prev){
      timeframes_lids[tf] = I;
      counts[tf] = 1;
    }
    if (counts[tf] <=1){
      stds[tf] = NAN;
      vols[tf] = NAN;
    }
    else{
      stds[tf] = sqrt(ssqs[tf]/(counts[tf]-1));
      rets(I,tf) = arr(I) - arr(timeframes_lids[tf]);
      vol_sum[tf] += rets(I,tf) * rets(I,tf);
      vols[tf] = sqrt(vol_sum[tf] / (counts[tf] - 1));
    }
  }
}

void RollerX::adjust_minmaxdeques(double price, int tf){

  while ( (!maxques[tf].empty()) && (price > maxques[tf].back().first)){
    maxques[tf].pop_back();
  }
  maxques[tf].push_back(make_pair(price, I));
  while ( (maxques[tf].front().second < timeframes_lids[tf])){
    maxques[tf].pop_front();
  }

  while( (!minques[tf].empty()) && (price < minques[tf].back().first)){
    minques[tf].pop_back();
  }
  minques[tf].push_back(make_pair(price, I));
  while ( (minques[tf].front().second < timeframes_lids[tf])){
    minques[tf].pop_front();
  }
}

void RollerX::adjust_minmaxdeques_zones(double price, int tf, bool zone_bool, bool zone_bool_prev){
  if (zone_bool){
    if (zone_bool_prev){
      // adjust_minmaxdeques(price, tf);
      while (! maxques[tf].empty() && (price > maxques[tf].back().first)){
          maxques[tf].pop_back();
        }
      maxques[tf].push_back(make_pair(price, I));
      while (! minques[tf].empty() && (price < minques[tf].back().first)){
        minques[tf].pop_back();
      }
      minques[tf].push_back(make_pair(price, I));
    }
    else{
      maxques[tf].clear();
      maxques[tf].push_back(make_pair(price, I));
      minques[tf].clear();
      minques[tf].push_back(make_pair(price, I));
    }
  }
}

void RollerX::update_out(const arrType& arr, uint64_t idx, int tf, outType& out){
  out[idx][0][tf] = arr(timeframes_lids[tf]);
  out[idx][1][tf] = maxques[tf].front().first;
  out[idx][2][tf] = minques[tf].front().first;
  out[idx][3][tf] = means[tf];
  out[idx][4][tf] = (arr(I) / out[idx][0][tf])  - 1;
  out[idx][5][tf] = log(arr(I) / out[idx][0][tf]);
  out[idx][6][tf] = stds[tf];
  out[idx][7][tf] = vols[tf];
}

bool RollerX::highlow_condition(double val){
  // cout << "Doing Condition" << endl;
  // cout << "minmax empty: " << this->minques[sampling_tf_idx].empty() << ", " << this->maxques[sampling_tf_idx].empty() << endl;
  // cout << "minmax sizes: " << this->minques[sampling_tf_idx].size() << ", " << this->maxques[sampling_tf_idx].size() << endl;
  // cout << "first elements min, max" << this->minques[sampling_tf_idx].front().first << ", " << this->maxques[sampling_tf_idx].front().first << endl;
  if (minques[sampling_tf_idx].empty() || maxques[sampling_tf_idx].empty()){
    return true;
  }
  else{
    return ((val <= minques[sampling_tf_idx].front().first) || (val >= maxques[sampling_tf_idx].front().first));
  }
}

// bool std_condition(outType &out){
//   return (arr[I] < out[])
// }
