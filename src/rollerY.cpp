#include "../include/rollers.h"
#include <exception>
#include <chrono>
#include <boost/bind/bind.hpp>
// #include <valgrind/callgrind.h>

static const int LABELTYPE = 1;

RollerY::RollerY(vector<uint64_t> timeframes, int nzones, double eps_thresh, double range_multiplier): timeframes(timeframes), nzones(nzones), eps(eps_thresh), range_multiplier(range_multiplier){ // ntimesessions refers to discrete blocks of time I.e the hour between trading sessions
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
  timeframes_rids.resize(ntimeframes);
}

// void RollerY::ingest(const arrType& arr, const timearrType& timestamps, outType& outCont, outBoolType& outLabels){
// }

void RollerY::ingest(const outType& prevOuts, const timearrType& timestamps, const outType& outCont){
  N = prevOuts.shape()[0];
  // rets.resize(N, ntimeframes);

  if (!initialized){
    initialized = true;
    // outCont.resize(extents[N][nfeats][ntimeframes]);
    // outLabels.resize(extents[N][nfeats][ntimeframes]);
  }
  else{
    for (int tf=0; tf<ntimeframes; tf++){
      timeframes_rids[tf] -= I;
    }
    // cout << "resizing initalized before" << endl;
    // cout << "N, nfeats, ntimeframes: " << N << ", " << nfeats << ", " << ntimeframes << endl;
    // outCont.resize(extents[N][nfeats][ntimeframes]);
    // cout << "resized initalizd before" << endl;
    // outLabels.resize(extents[N][nfeats][ntimeframes]);
    // for (int tf=0; tf<ntimeframes; tf++){
    //   timeframes_rids[tf] -= left_idx;
    //   int queue_size = maxques[tf].size();
    //   for(int i=0; i<queue_size; i++){
    //     pair<double, uint64_t> &vals = maxques[tf].front();
    //     maxques[tf].push_back(make_pair(vals.first, vals.second - left_idx));
    //     maxques[tf].pop_front();
    //   }
    //   queue_size = minques[tf].size();
    //   for(int i=0; i<queue_size; i++){
    //     pair<double, uint64_t> &vals = minques[tf].front();
    //     minques[tf].push_back(make_pair(vals.first, vals.second - left_idx));
    //     minques[tf].pop_front();
    //   }
    // }
    // I = arr_memory.rows();
  }
}

// outType RollerY::roll(py::EigenDRef<arrType> arr_in, py::EigenDRef<timearrType> timestamps, py::array_t<double> _prevOuts, bool sample, string sample_condition, int sampling_tf_idx){
//   zoneBoolType zones(0, 0);
//   return roll(arr_in, timestamps, _prevOuts, zones, sample, sample_condition, sampling_tf_idx);
// }

template<>
void RollerY::crawl_path<0>(py::EigenDRef<arrType> arr, const outType& prevOuts, const outType& yprevOuts, outBoolType& outLabels, uint64_t idx, int tf){

  double current_price, diff, maxdiff, mindiff, maxpeak, minpeak, hl_range, vol_range;
  bool maxhit=false, minhit=false;
  current_price = arr(idx);
  maxdiff = yprevOuts[idx][1][tf] - current_price; // yprevOuts[idx][0][tf] == OPEN for future == close at current
  mindiff = yprevOuts[idx][2][tf] - current_price;
  hl_range = abs(prevOuts[I][1][tf] - prevOuts[I][2][tf]);
  vol_range = prevOuts[I][7][tf];
  if(maxdiff+mindiff > eps){ // Relative to current price, high is further away than low
    // yprevOuts[idx][1][tf](idx, 5, tf) = abs(maxdiff);
    outLabels[idx][0][tf] = true;
  }
  else if(maxdiff+mindiff < -eps){ // Relative to current price, low is further away than high
    // yprevOuts[idx][1][tf](idx, 5, tf) = abs(mindiff);
    outLabels[idx][1][tf] = true;
  }
  else{ // Neither are far away enough (minimum distance = eps)
    // yprevOuts[idx][1][tf](idx, 5, tf) = 0.;
    outLabels[idx][2][tf] = true;
  }
  if(maxdiff > (range_multiplier * hl_range)){ // Price touches top of highlow range barrier
    outLabels[idx][3][tf] = true;
    maxhit=true;
  }
  if(mindiff < (-range_multiplier * hl_range)){// Price touches bottom of highlow range barrier
    outLabels[idx][4][tf] = true;
    minhit=true;
  }
  if( !maxhit && !minhit){
    outLabels[idx][5][tf] = true; // No Barrier touch - should be most of the time
  }
  maxhit=false;
  minhit=false;
  if(maxdiff > (range_multiplier * vol_range)){ // Price touches top of highlow range barrier
    outLabels[idx][6][tf] = true;
    maxhit=true;
  }
  if(mindiff < (-range_multiplier * vol_range)){// Price touches bottom of highlow range barrier
    outLabels[idx][7][tf] = true;
    minhit=true;
  }
  if( !maxhit && !minhit){
    outLabels[idx][8][tf] = true; // No Barrier touch - should be most of the time
  }

  if(yprevOuts[idx][0][tf] > (eps + current_price)){ // If close is +ve at end of tf period
    outLabels[idx][9][tf] = true;
  }
  else if (yprevOuts[idx][0][tf] < (-eps + current_price)){ // if close is -ve
    outLabels[idx][10][tf] = true;
  }
  else{
    outLabels[idx][11][tf] = true;
  }
}

template<>
void RollerY::crawl_path<1>(py::EigenDRef<arrType> arr, const outType& prevOuts, const outType& yprevOuts, outBoolType& outLabels, uint64_t idx, int tf){

  double current_price, diff, maxdiff, mindiff, maxpeak, minpeak, hl_range, vol_range;
  bool maxhit=false, minhit=false;
  current_price = arr(idx);
  maxdiff = yprevOuts[idx][1][tf] - current_price; // yprevOuts[idx][0][tf] == OPEN for future == close at current
  mindiff = yprevOuts[idx][2][tf] - current_price;
  hl_range = abs(prevOuts[I][1][tf] - prevOuts[I][2][tf]);
  vol_range = prevOuts[I][7][tf];
  if(maxdiff+mindiff > eps){ // Relative to current price, high is further away than low
    // yprevOuts[idx][1][tf](idx, 5, tf) = abs(maxdiff);
    outLabels[idx][0][tf] = true;
  }
  else if(maxdiff+mindiff < -eps){ // Relative to current price, low is further away than high
    // yprevOuts[idx][1][tf](idx, 5, tf) = abs(mindiff);
    outLabels[idx][1][tf] = true;
  }
  else{
    outLabels[idx][2][tf] = true;
  }
}
template<>
void RollerY::crawl_path<2>(py::EigenDRef<arrType> arr, const outType& prevOuts, const outType& yprevOuts, outBoolType& outLabels, uint64_t idx, int tf){

  double current_price, diff, maxdiff, mindiff, maxpeak, minpeak, hl_range, vol_range;
  bool maxhit=false, minhit=false;
  current_price = arr(idx);
  hl_range = abs(prevOuts[I][1][tf] - prevOuts[I][2][tf]);
  vol_range = prevOuts[I][7][tf];
  if(maxdiff > (range_multiplier * hl_range)){ // Price touches top of highlow range barrier
    outLabels[idx][0][tf] = true;
    maxhit=true;
  }
  if(mindiff < (-range_multiplier * hl_range)){// Price touches bottom of highlow range barrier
    outLabels[idx][1][tf] = true;
    minhit=true;
  }
  if( !maxhit && !minhit){
    outLabels[idx][2][tf] = true; // No Barrier touch - should be most of the time
  }
}
template<>
void RollerY::crawl_path<3>(py::EigenDRef<arrType> arr, const outType& prevOuts, const outType& yprevOuts, outBoolType& outLabels, uint64_t idx, int tf){

  double current_price, diff, maxdiff, mindiff, maxpeak, minpeak, hl_range, vol_range;
  bool maxhit=false, minhit=false;
  current_price = arr(idx);
  maxdiff = yprevOuts[idx][1][tf] - current_price; // yprevOuts[idx][0][tf] == OPEN for future == close at current
  mindiff = yprevOuts[idx][2][tf] - current_price;
  hl_range = abs(prevOuts[I][1][tf] - prevOuts[I][2][tf]);
  vol_range = prevOuts[I][7][tf];
  if(maxdiff+mindiff > eps){ // Relative to current price, high is further away than low
    // yprevOuts[idx][1][tf](idx, 5, tf) = abs(maxdiff);
    outLabels[idx][0][tf] = true;
  }
  else if(maxdiff+mindiff < -eps){ // Relative to current price, low is further away than high
    // yprevOuts[idx][1][tf](idx, 5, tf) = abs(mindiff);
    outLabels[idx][1][tf] = true;
  }
  else{
    outLabels[idx][2][tf] = true;
  }
}
outType RollerY::shift(py::array_t<double> _prevOuts, py::EigenDRef<timearrType> timestamps){
  // Initialize boost multi_array reference to ndarray buffer
  py::buffer_info buff_info = _prevOuts.request();
  vector<long int> shape = buff_info.shape;
  boost::multi_array_ref<double, 3> xFeats((double*)buff_info.ptr, shape);

  // SANITY CHECKS
  if(xFeats.shape()[0] != timestamps.rows()){
    string message = "xFeats and timestamps don't have equal length!\n";
    message += "length of xFeats, timestamps: " + std::to_string(xFeats.shape()[0]) + ", " + std::to_string(timestamps.rows());
    throw invalid_argument(message);
  }
  if(xFeats.shape()[1] != nfeats){
    string message = "xFeats doesn't have the number of features that this Roller was initialized with\n";
    message += "xFeats.shape()[1], this.nfeats: " + std::to_string(xFeats.shape()[1]) + ", " + std::to_string(nfeats);
    throw invalid_argument(message);
  }
  if(xFeats.shape()[2] != ntimeframes){
    string message = "xFeats doesn't have the number of timeframes that this Roller was initialized with\n";
    message += "xFeats.shape()[2], this.ntimeframes: " + std::to_string(xFeats.shape()[2]) + ", " + std::to_string(ntimeframes);
    throw invalid_argument(message);
  }

  // Memory Initialization for output array and stateful behaviour for sequential rolling
  size_t _N = timearr_memory.rows() + timestamps.rows();
  timearrType timearr(_N, 1);
  timearr << timearr_memory, timestamps;
  // zoneBoolType zones(zones_memory.rows() + zones_in.rows(), zones_in.cols());
  // zones << zones_memory, zones_in;
  outType prevOuts(extents[_N][nfeats][ntimeframes]);
  std::copy(prevOuts_memory.begin(), prevOuts_memory.end(), prevOuts.begin());
  std::copy(xFeats.begin(), xFeats.end(), prevOuts.begin() + prevOuts_memory.shape()[0]);

  outType outCont(extents[prevOuts.shape()[0]][nfeats][ntimeframes]);

  ingest(prevOuts, timearr, outCont);

  // ROLL
  uint64_t max_tf = *std::max_element(timeframes.begin(), timeframes.end());
  uint64_t max_time = timearr(N-1) - max_tf;
  // cout << "prevOut shape: " << prevOut.shape()[0] << ", " << prevOut.shape()[1] << ", " << prevOut.shape()[2] << endl;
  // I=0; MUST BE DONE AFTER INGEST SO THAT ON SECOND INIT, TIMEFRANE_RIDS CAN BE ADJUSTED BASED ON PREV I
  I=0;
  while(true){
    for (int tf=0; tf<ntimeframes; tf++){
      while((timearr(timeframes_rids[tf]) <= timearr(I) + timeframes[tf])){
        timeframes_rids[tf] += 1;
      }
      for (int feat=0; feat<nfeats; feat++){
        outCont[I][feat][tf] = prevOuts[timeframes_rids[tf]][feat][tf];
      }
    }
    if (I == N-1 || timearr(I+1) >= max_time) break;
    I++;
  }
  // Save Memory for stateful behaviour
  timearr_memory.resize(N-I);
  timearr_memory = timearr.bottomRows(N-I);
  prevOuts_memory.resize(extents[N-I][nfeats][ntimeframes]);
  std::copy(prevOuts.begin()+I, prevOuts.end(), prevOuts_memory.begin());
  // if (nzones > 0){
  //   zones_memory.resize(N-left_idx, zones.cols());
  //   zones_memory = zones.bottomRows(N-left_idx);
  // }
  // py::array_t<double> out_np({N, nfeats, ntimeframes},{nfeats*ntimeframes*8, ntimeframes*8, 8}, &out);
  return outCont[boost::indices[range(0, I)][range()][range()]];
}

outType RollerY::shift(outType& xFeats, py::EigenDRef<timearrType> timestamps){
  // Initialize boost multi_array reference to ndarray buffer

  // SANITY CHECKS
  if(xFeats.shape()[0] != timestamps.rows()){
    string message = "xFeats and timestamps don't have equal length!\n";
    message += "length of xFeats, timestamps: " + std::to_string(xFeats.shape()[0]) + ", " + std::to_string(timestamps.rows());
    throw invalid_argument(message);
  }
  if(xFeats.shape()[1] != nfeats){
    string message = "xFeats doesn't have the number of features that this Roller was initialized with\n";
    message += "xFeats.shape()[1], this.nfeats: " + std::to_string(xFeats.shape()[1]) + ", " + std::to_string(nfeats);
    throw invalid_argument(message);
  }
  if(xFeats.shape()[2] != ntimeframes){
    string message = "xFeats doesn't have the number of timeframes that this Roller was initialized with\n";
    message += "xFeats.shape()[2], this.ntimeframes: " + std::to_string(xFeats.shape()[2]) + ", " + std::to_string(ntimeframes);
    throw invalid_argument(message);
  }

  // Memory Initialization for output array and stateful behaviour for sequential rolling
  size_t _N = timearr_memory.rows() + timestamps.rows();
  timearrType timearr(_N, 1);
  timearr << timearr_memory, timestamps;
  // zoneBoolType zones(zones_memory.rows() + zones_in.rows(), zones_in.cols());
  // zones << zones_memory, zones_in;
  outType prevOuts(extents[_N][nfeats][ntimeframes]);
  std::copy(prevOuts_memory.begin(), prevOuts_memory.end(), prevOuts.begin());
  std::copy(xFeats.begin(), xFeats.end(), prevOuts.begin() + prevOuts_memory.shape()[0]);

  outType outCont(extents[prevOuts.shape()[0]][nfeats][ntimeframes]);

  ingest(prevOuts, timearr, outCont);

  // ROLL
  uint64_t max_tf = *std::max_element(timeframes.begin(), timeframes.end());
  uint64_t max_time = timearr(N-1) - max_tf;
  // cout << "prevOut shape: " << prevOut.shape()[0] << ", " << prevOut.shape()[1] << ", " << prevOut.shape()[2] << endl;
  // I=0; MUST BE DONE AFTER INGEST SO THAT ON SECOND INIT, TIMEFRANE_RIDS CAN BE ADJUSTED BASED ON PREV I
  I=0;
  while(true){
    for (int tf=0; tf<ntimeframes; tf++){
      while((timearr(timeframes_rids[tf]) <= timearr(I) + timeframes[tf])){
        timeframes_rids[tf] += 1;
      }
      for (int feat=0; feat<nfeats; feat++){
        outCont[I][feat][tf] = prevOuts[timeframes_rids[tf]][feat][tf];
      }
    }
    if (I == N-1 || timearr(I+1) >= max_time) break;
    I++;
  }
  // Save Memory for stateful behaviour
  timearr_memory.resize(N-I);
  timearr_memory = timearr.bottomRows(N-I);
  prevOuts_memory.resize(extents[N-I][nfeats][ntimeframes]);
  std::copy(prevOuts.begin()+I, prevOuts.end(), prevOuts_memory.begin());
  outCont.resize(extents[I][nfeats][ntimeframes]);
  // outType out(outCont[boost::indices[range(0, I)][range()][range()]]);
  // if (nzones > 0){
  //   zones_memory.resize(N-left_idx, zones.cols());
  //   zones_memory = zones.bottomRows(N-left_idx);
  // }
  // py::array_t<double> out_np({N, nfeats, ntimeframes},{nfeats*ntimeframes*8, ntimeframes*8, 8}, &out);
  // return out;
  // return outCont[boost::indices[range(0, I)][range()][range()]];
  return outCont;
}


outBoolType RollerY::roll(py::EigenDRef<arrType> arr, py::array_t<double> _xFeats, py::array_t<double> _yFeats, py::EigenDRef<timearrType> timestamps){
  // Init buffers from python (ndarrays)
  py::buffer_info xbuff_info = _xFeats.request();
  vector<long int> xshape = xbuff_info.shape;
  boost::multi_array_ref<double, 3> xFeats((double*)xbuff_info.ptr, xshape);
  // boost::multi_array_ref<double, 3> prevOuts((double*)xbuff_info.ptr, xshape);
  py::buffer_info ybuff_info = _yFeats.request();
  vector<long int> yshape = ybuff_info.shape;
  boost::multi_array_ref<double, 3> yFeats((double*)ybuff_info.ptr, yshape);
  // boost::multi_array_ref<double, 3> yprevOuts((double*)ybuff_info.ptr, yshape);

  cout << "BUFFER INIT Done" << endl;
  // SANITY CHECKS
  if(xFeats.shape()[0] != timestamps.rows()){
    throw invalid_argument("xFeats and timestamps don't have equal length!");
  }
  if(xFeats.shape()[1] != yFeats.shape()[1]){
    throw invalid_argument("xFeats and yFeats don't have the same number of features!");
  }
  if(xFeats.shape()[2] != yFeats.shape()[2]){
    throw invalid_argument("xFeats and yFeats don't have the same number of timeframes!");
  }

  // Memory initialization for main input Arrays
  // size_t _N = timestamps.rows() + timearr_memory.rows();
  size_t _N = timestamps.rows();
  timearrType timearr(_N, 1);
  timearr << timestamps;
  outType prevOuts(extents[_N][nfeats][ntimeframes]);
  std::copy(xFeats.begin(), xFeats.end(), prevOuts.begin());
  // std::copy(prevOuts_memory.begin(), prevOuts_memory.end(), prevOuts.begin());
  // std::copy(xFeats.begin(), xFeats.end(), prevOuts.begin() + prevOuts_memory.shape()[0]);
  outType yprevOuts(extents[yFeats.shape()[0]][nfeats][ntimeframes]);
  std::copy(yFeats.begin(), yFeats.end(), yprevOuts.begin());
  // std::copy(yprevOuts_memory.begin(), yprevOuts_memory.end(), yprevOuts.begin());
  // std::copy(yFeats.begin(), yFeats.end(), yprevOuts.begin() + yprevOuts_memory.shape()[0]);
  outBoolType outLabels(extents[yprevOuts.shape()[0]][nlabels][ntimeframes]);

  cout << "MEMORY INIT Done" << endl;
  cout << "N, I, timearr_memory.rows(): " << N << ", "  << I << ", "<< timearr_memory.rows() << endl;
  cout << "yPrevOuts shape: " << yprevOuts.shape()[0] << ", " << yprevOuts.shape()[1] << ", " << yprevOuts.shape()[2] << endl;
  cout << "outLabels shape: " << outLabels.shape()[0] << ", " << outLabels.shape()[1] << ", " << outLabels.shape()[2] << endl;
  ingest(prevOuts, timearr, yprevOuts);

  // ROLL
  uint64_t max_tf = *std::max_element(timeframes.begin(), timeframes.end());
  uint64_t max_time = timearr(_N-1) - max_tf;
  // cout << "prevOut shape: " << prevOut.shape()[0] << ", " << prevOut.shape()[1] << ", " << prevOut.shape()[2] << endl;
  // I=0; MUST BE DONE AFTER INGEST SO THAT ON SECOND INIT, TIMEFRANE_RIDS CAN BE ADJUSTED BASED ON PREV I
  I=0;
  size_t stop = yprevOuts.shape()[0];
  while(true){
    for(int tf=0; tf<ntimeframes; tf++){
      crawl_path<LABELTYPE>(arr, prevOuts, yprevOuts, outLabels, I, tf);
    }
    // if(yprevOuts[I][0][0] != yprevOuts[I][0][2]){
    //   cout << "DOESN'T EQUAL at: " << I << endl;
    //   cout << yprevOuts[I][0][0] << ", " <<  yprevOuts[I][0][2] << endl;
    // }
    if (I == stop-1) {
      break;
    }
    I++;
  }

  cout << "ROLL Done" << endl;
  boost::multi_array_ref<bool, 3> outLabelsOUT(outLabels[boost::indices[range(0, I)][range()][range()]].origin(), yshape);
  cout << "Assigned" << endl;
  // Save Memory for stateful behaviour
  // timearr_memory.resize(_N-I);
  // timearr_memory = timearr.bottomRows(_N-I);
  // prevOuts_memory.resize(extents[_N-I][nfeats][ntimeframes]);
  // std::copy(prevOuts.begin()+I, prevOuts.end(), prevOuts_memory.begin());
  // yprevOuts_memory.resize(extents[_N-I][nfeats][ntimeframes]);
  // std::copy(yprevOuts.begin()+I, yprevOuts.end(), yprevOuts_memory.begin());
  // cout << "MEMORY SAVING DONE" << endl;
  // cout << "outLabels shape: " << outLabels.shape()[0] << ", " << outLabels.shape()[1] << ", " << outLabels.shape()[2] << endl;
  // outBoolType outLabels2 = outLabels[boost::indices[range(0, I)][range()][range()]];
  // cout << "out shape: " << outLabels2.shape()[0] << ", " << outLabels2.shape()[1] << ", " << outLabels2.shape()[2] << endl;
  // if (nzones > 0){
  //   zones_memory.resize(N-left_idx, zones.cols());
  //   zones_memory = zones.bottomRows(N-left_idx);
  // }
  // py::array_t<double> out_np({N, nfeats, ntimeframes},{nfeats*ntimeframes*8, ntimeframes*8, 8}, &out);
  // return outLabels[boost::indices[range(0, I)][range()][range()]];
  return outLabelsOUT;
}

outBoolType RollerY::roll(py::EigenDRef<arrType> arr, outType& prevOuts, outType& yprevOuts, py::EigenDRef<timearrType> timestamps){
  // Init buffers from python (ndarrays)
  // py::buffer_info xbuff_info = _xFeats.request();
  // vector<long int> xshape = xbuff_info.shape;
  // boost::multi_array_ref<double, 3> xFeats((double*)xbuff_info.ptr, xshape);
  // // boost::multi_array_ref<double, 3> prevOuts((double*)xbuff_info.ptr, xshape);
  // py::buffer_info ybuff_info = _yFeats.request();
  // vector<long int> yshape = ybuff_info.shape;
  // boost::multi_array_ref<double, 3> yFeats((double*)ybuff_info.ptr, yshape);
  // boost::multi_array_ref<double, 3> yprevOuts((double*)ybuff_info.ptr, yshape);

  cout << "BUFFER INIT Done" << endl;
  // SANITY CHECKS
  if(prevOuts.shape()[0] != timestamps.rows()){
    throw invalid_argument("xFeats and timestamps don't have equal length!");
  }
  if(prevOuts.shape()[1] != yprevOuts.shape()[1]){
    throw invalid_argument("xFeats and yFeats don't have the same number of features!");
  }
  if(prevOuts.shape()[2] != yprevOuts.shape()[2]){
    throw invalid_argument("xFeats and yFeats don't have the same number of timeframes!");
  }

  // Memory initialization for main input Arrays
  // size_t _N = timestamps.rows() + timearr_memory.rows();
  size_t _N = timestamps.rows();
  timearrType timearr(_N, 1);
  timearr << timestamps;
  // outType prevOuts(extents[_N][nfeats][ntimeframes]);
  // std::copy(xFeats.begin(), xFeats.end(), prevOuts.begin());
  // std::copy(prevOuts_memory.begin(), prevOuts_memory.end(), prevOuts.begin());
  // std::copy(xFeats.begin(), xFeats.end(), prevOuts.begin() + prevOuts_memory.shape()[0]);
  // outType yprevOuts(extents[yFeats.shape()[0]][nfeats][ntimeframes]);
  // std::copy(yFeats.begin(), yFeats.end(), yprevOuts.begin());
  // std::copy(yprevOuts_memory.begin(), yprevOuts_memory.end(), yprevOuts.begin());
  // std::copy(yFeats.begin(), yFeats.end(), yprevOuts.begin() + yprevOuts_memory.shape()[0]);
  outBoolType outLabels(extents[yprevOuts.shape()[0]][nlabels][ntimeframes]);

  cout << "MEMORY INIT Done" << endl;
  cout << "N, I, timearr_memory.rows(): " << N << ", "  << I << ", "<< timearr_memory.rows() << endl;
  cout << "yPrevOuts shape: " << yprevOuts.shape()[0] << ", " << yprevOuts.shape()[1] << ", " << yprevOuts.shape()[2] << endl;
  cout << "outLabels shape: " << outLabels.shape()[0] << ", " << outLabels.shape()[1] << ", " << outLabels.shape()[2] << endl;
  ingest(prevOuts, timearr, yprevOuts);

  // ROLL
  uint64_t max_tf = *std::max_element(timeframes.begin(), timeframes.end());
  uint64_t max_time = timearr(_N-1) - max_tf;
  // cout << "prevOut shape: " << prevOut.shape()[0] << ", " << prevOut.shape()[1] << ", " << prevOut.shape()[2] << endl;
  // I=0; MUST BE DONE AFTER INGEST SO THAT ON SECOND INIT, TIMEFRANE_RIDS CAN BE ADJUSTED BASED ON PREV I
  I=0;
  size_t stop = yprevOuts.shape()[0];
  while(true){
    for(int tf=0; tf<ntimeframes; tf++){
      crawl_path<LABELTYPE>(arr, prevOuts, yprevOuts, outLabels, I, tf);
    }
    // if(yprevOuts[I][0][0] != yprevOuts[I][0][2]){
    //   cout << "DOESN'T EQUAL at: " << I << endl;
    //   cout << yprevOuts[I][0][0] << ", " <<  yprevOuts[I][0][2] << endl;
    // }
    if (I == stop-1) {
      cout << "STOPPING AT I: " << I << endl;
      break;
    }
    I++;
  }

  cout << "ROLL Done" << endl;
  // boost::multi_array_ref<bool, 3> outLabelsOUT(outLabels[boost::indices[range(0, I)][range()][range()]].origin(), yprevOuts.shape());
  // boost::multi_array_ref<bool, 3> outLabelsOUT(extents[I][outLabels.shape()[1]][outLabels.shape()[2]]);
  // outBoolType outLabelsOUT(extents[I][outLabels.shape()[1]][outLabels.shape()[2]]);
  // for (int i=0; i<I; i++){
  //   for (int feat=0; feat<outLabels.shape()[1]; feat++){
  //     for (int tf=0; tf<outLabels.shape()[2]; tf++){
  //       outLabelsOUT[i][feat][tf] = outLabels[i][feat][tf];
  //     }
  //   }
  // }

  cout << "Assigned" << endl;
  // py::array_t<double> out_np({N, nfeats, ntimeframes},{nfeats*ntimeframes*8, ntimeframes*8, 8}, &out);
  return outLabels[boost::indices[range(0, I)][range()][range()]];
  // return outLabelsOUT;
}

outBoolType RollerY::roll(py::EigenDRef<arrType> arr, py::array_t<double> _xFeats, py::EigenDRef<timearrType> timestamps){
  outType _yFeats(shift(_xFeats, timestamps));
  return roll(arr, _xFeats, _yFeats, timestamps);
}

outBoolType RollerY::roll(py::EigenDRef<arrType> arr, py::array_t<double> _xFeats, outType& _yFeats, py::EigenDRef<timearrType> timestamps){
  // Overloaded with outType& _yFeats instead of py::array_t<double> _yFeats
  // If called, this means that shift has already been called and memory arrays have already been updated
  // So memories don't need to be reallocated (end of function)
  // Init buffers from python (ndarrays)
  py::buffer_info xbuff_info = _xFeats.request();
  vector<long int> xshape = xbuff_info.shape;
  boost::multi_array_ref<double, 3> xFeats((double*)xbuff_info.ptr, xshape);
  boost::multi_array_ref<double, 3> yFeats(_yFeats);

  // SANITY CHECKS
  if(xFeats.shape()[0] != timestamps.rows()){
    throw invalid_argument("xFeats and timestamps don't have equal length!");
  }
  if(xFeats.shape()[1] != yFeats.shape()[1]){
    throw invalid_argument("xFeats and yFeats don't have the same number of features!");
  }
  if(xFeats.shape()[2] != yFeats.shape()[2]){
    throw invalid_argument("xFeats and yFeats don't have the same number of timeframes!");
  }

  // if(sample){
  //   if (sample_condition == "highlow"){
  //   }
  //   else{
  //     throw invalid_argument("Invalid sample_condition. Choices are: highlow");
  //   }
  // }

  // Memory initialization for main input Arrays
  size_t _N = timearr_memory.rows() + timestamps.rows();
  timearrType timearr(_N, 1);
  timearr << timearr_memory, timestamps;
  // zoneBoolType zones(zones_memory.rows() + zones_in.rows(), zones_in.cols());
  // zones << zones_memory, zones_in;
  outType prevOuts(extents[_N][nfeats][ntimeframes]);
  std::copy(prevOuts_memory.begin(), prevOuts_memory.end(), prevOuts.begin());
  std::copy(xFeats.begin(), xFeats.end(), prevOuts.begin() + prevOuts_memory.shape()[0]);
  outType yprevOuts(extents[_N][nfeats][ntimeframes]);
  std::copy(yprevOuts_memory.begin(), yprevOuts_memory.end(), yprevOuts.begin());
  std::copy(yFeats.begin(), yFeats.end(), yprevOuts.begin() + yprevOuts_memory.shape()[0]);

  outBoolType outLabels(extents[yprevOuts.shape()[0]][nlabels][ntimeframes]);

  ingest(prevOuts, timearr, yprevOuts);

  // ROLL
  uint64_t max_tf = *std::max_element(timeframes.begin(), timeframes.end());
  uint64_t max_time = timearr(N-1) - max_tf;
  // cout << "prevOut shape: " << prevOut.shape()[0] << ", " << prevOut.shape()[1] << ", " << prevOut.shape()[2] << endl;
  // I=0; MUST BE DONE AFTER INGEST SO THAT ON SECOND INIT, TIMEFRANE_RIDS CAN BE ADJUSTED BASED ON PREV I
  I=0;
  size_t stop = yFeats.shape()[0];
  while(true){
    for(int tf=0; tf<ntimeframes; tf++){
      crawl_path<LABELTYPE>(arr, prevOuts, yprevOuts, outLabels, I, tf);
    }
    // if(yprevOuts[I][0][0] != yprevOuts[I][0][2]){
    //   cout << "DOESN'T EQUAL" << endl;
    // }
    if (I == stop || timearr(I+1) >= max_time) break;
    I++;
  }
  // Save Memory for stateful behaviour
  // timearr_memory.resize(N-I);
  // timearr_memory = timearr.bottomRows(N-I);
  // prevOuts_memory.resize(extents[N-I][nfeats][ntimeframes]);
  // std::copy(prevOuts.begin()+I, prevOuts.end(), prevOuts_memory.begin());
  // yprevOuts_memory.resize(extents[N-I][nfeats][ntimeframes]);
  // std::copy(yprevOuts.begin()+I, yprevOuts.end(), yprevOuts_memory.begin());
  // if (nzones > 0){
  //   zones_memory.resize(N-left_idx, zones.cols());
  //   zones_memory = zones.bottomRows(N-left_idx);
  // }
  // py::array_t<double> out_np({N, nfeats, ntimeframes},{nfeats*ntimeframes*8, ntimeframes*8, 8}, &out);
  return outLabels[boost::indices[range(0, I)][range()][range()]];
}


// void RollerY::_step(const arrType& arr, const timearrType& timearr, const outType& prevOut, outType& outCont, outBoolType& outLabels){
//   // only continuous timeframes
//   for(int tf=0; tf<ntimeframes_cont; tf++){
//     while(timearr(timeframes_rids[tf]) < timearr(I) + timeframes[tf]){
//       head_next(arr, tf);
//       timeframes_rids[tf] += 1;
//     }
//     tail_update(arr, timearr, tf);
//     update_out(arr, timearr, I, tf, prevOut, outCont, outLabels);
//   }
// }

// void RollerY::_step(const arrType& arr, const timearrType& timearr, bool sample){
//   // overloaded to not include update_out - for sampling
//   for(int tf=0; tf<ntimeframes_cont; tf++){
//     while(timearr(timeframes_rids[tf]) < timearr(I) + timeframes[tf]){
//       head_next(arr, tf);
//       timeframes_rids[tf] += 1;
//     }
//     tail_update(arr, timearr, tf);
//   }
// }

// void RollerY::head_next(const arrType& arr, int tf){
//   double new_val = arr(timeframes_rids[tf]);
//   double delt = new_val - means[tf];
//   counts[tf] += 1;
//   means[tf] += delt / counts[tf];
//   ssqs[tf] += abs(delt * (new_val - means[tf]));
//   rets(I, tf) = arr(I) - new_val;
//   vol_sum[tf] += rets(I, tf) * rets(I, tf);
//   // rets_lids[timeframes_rids[tf]] = I;
//   while ( !(maxques[tf].empty()) && (new_val > maxques[tf].back().first)){
//     maxques[tf].pop_back();
//   }
//   maxques[tf].push_back(make_pair(new_val, timeframes_rids[tf]));
//   while( !(minques[tf].empty()) && (new_val < minques[tf].back().first)){
//     minques[tf].pop_back();
//   }
//   minques[tf].push_back(make_pair(new_val, timeframes_rids[tf]));
// }

// void RollerY::head_next_zones(double price, int tf, bool zone_bool, bool zone_bool_prev){
//   if (zone_bool){
//     if (zone_bool_prev){
//       counts[tf] += 1;
//       double delt = price - means[tf];
//       means[tf] += delt / counts[tf];
//       ssqs[tf] += abs(delt * (price - means[tf]));
//     }
//     else{
//       counts[tf] = 1;
//       means[tf] = price;
//       ssqs[tf] = 0.;
//       vol_sum[tf] = 0.;
//     }
//   }
// }

// void RollerY::tail_update(const arrType& arr, const timearrType& timearr, int tf){
//   if(I > 0){
//     if (counts[tf] > 1){
//       counts[tf] -= 1;
//       // rets_lids[timeframes_rids[tf]] = I;
//       double delt = arr(I-1) - means[tf];
//       means[tf] -= delt / counts[tf];
//       ssqs[tf] -= abs(delt * (arr(I-1) - means[tf]));
//       // vol_sum[tf] -= (rets(I-1, tf) * rets(timeframes_rids[tf], tf));
//       double prev_ret = rets(timeframes_rids[tf], tf);
//       vol_sum[tf] -= prev_ret * prev_ret;
//     }
//     vol_sum[tf] += rets(I, tf) * rets(I, tf);
//   }
//   stds[tf] = sqrt(ssqs[tf]/(counts[tf] - 1));
//   if (counts[tf] <=1){
//     stds[tf] = NAN;
//     vols[tf] = NAN;
//   }
//   else{
//     stds[tf] = sqrt(ssqs[tf] / (counts[tf] - 1));
//     vols[tf] = sqrt(vol_sum[tf] / (counts[tf] -1));
//   }
//   while ( !maxques[tf].empty() && (maxques[tf].front().second < I)){
//     maxques[tf].pop_front();
//   }
//   while ( !minques[tf].empty() && (minques[tf].front().second < I)){
//     minques[tf].pop_front();
//   }
// }

// void RollerY::tail_update_zones(const arrType& arr, const timearrType& timearr, int tf, bool zone_bool, bool zone_bool_prev){
//   if (zone_bool){
//     if(!zone_bool_prev){
//       timeframes_rids[tf] = I;
//       counts[tf] = 1;
//     }
//     if (counts[tf] <=1){
//       stds[tf] = NAN;
//       vols[tf] = NAN;
//     }
//     else{
//       stds[tf] = sqrt(ssqs[tf]/(counts[tf]-1));
//       rets(I,tf) = arr(I) - arr(timeframes_rids[tf]);
//       vol_sum[tf] += rets(I,tf) * rets(I,tf);
//       vols[tf] = sqrt(vol_sum[tf] / (counts[tf] - 1));
//     }
//   }
// }

// void RollerY::adjust_minmaxdeques(double price, int tf){

//   while ( !(maxques[tf].empty()) && (price > maxques[tf].back().first)){
//     maxques[tf].pop_back();
//   }
//   maxques[tf].push_back(make_pair(price, I));
//   while ( (maxques[tf].front().second < timeframes_rids[tf])){
//     maxques[tf].pop_front();
//   }

//   while( !(minques[tf].empty()) && (price < minques[tf].back().first)){
//     minques[tf].pop_back();
//   }
//   minques[tf].push_back(make_pair(price, I));
//   while ( (minques[tf].front().second < timeframes_rids[tf])){
//     minques[tf].pop_front();
//   }
// }

// void RollerY::adjust_minmaxdeques_zones(double price, int tf, bool zone_bool, bool zone_bool_prev){
//   if (zone_bool){
//     if (zone_bool_prev){
//       // adjust_minmaxdeques(price, tf);
//       while (! maxques[tf].empty() && (price > maxques[tf].back().first)){
//           maxques[tf].pop_back();
//         }
//       maxques[tf].push_back(make_pair(price, I));
//       while (! minques[tf].empty() && (price < minques[tf].back().first)){
//         minques[tf].pop_back();
//       }
//       minques[tf].push_back(make_pair(price, I));
//     }
//     else{
//       maxques[tf].clear();
//       maxques[tf].push_back(make_pair(price, I));
//       minques[tf].clear();
//       minques[tf].push_back(make_pair(price, I));
//     }
//   }
// }



// void RollerY::update_out(const arrType& arr, const timearrType& timearr, uint64_t idx, int tf, const outType& prevOut, outType& outCont, outBoolType& outLabels){
//   // cout << "Updating idx, tf:  " << idx << ", " << tf <<  endl;
//   outCont[idx][0][tf] = arr(timeframes_rids[tf]-1);
//   outCont[idx][1][tf] = maxques[tf].front().first;
//   outCont[idx][2][tf] = minques[tf].front().first;
//   outCont[idx][3][tf] = means[tf];
//   outCont[idx][4][tf] = (arr(I) / outCont[idx][0][tf])  - 1;
//   outCont[idx][5][tf] = log(arr(I) / outCont[idx][0][tf]);
//   outCont[idx][6][tf] = stds[tf];
//   outCont[idx][7][tf] = vols[tf];
//   crawl_path(arr, timearr, prevOut, outCont, outLabels, idx, tf);
// }

// bool RollerY::highlow_condition(double val){
//   return (val < minques[sampling_tf_idx].front().first || val > maxques[sampling_tf_idx].front().first);
// }

// bool std_condition(outType &out){
//   return (arr[I] < out[])
// }
