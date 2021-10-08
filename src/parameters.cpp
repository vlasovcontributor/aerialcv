#include "parameters.h"

#include <cstdarg>
#include <stdexcept>

void ParameterIterator::AddParamMeta(std::vector<Param>& p){
  if (p.size() != _params->size()) {
    throw std::invalid_argument("len of params and param meta does not match");
  }
  for (size_t i = 0; i < p.size(); ++i) {
    _param_info.push_back(p[i]);
    (*_params)[i] = p[i].low;
    if (_total == 0) {
      _total = (p[i].high - p[i].low) / p[i].step;
    }    else {
      _total *= (p[i].high - p[i].low) / p[i].step;
    }
  }
}
void ParameterIterator::LinkParams(std::vector<int>* params) {
  _params = params;
}
void ParameterIterator::RecieveFeedback(std::vector<double> assessments){
  if (assessments.size() != _assessments.size()) {
    throw std::invalid_argument("len of assessments does not match");
  }
  for (size_t i = 0; i < assessments.size(); ++i) {
    if (assessments[i] > _assessments[i].grade) {
      _assessments[i].grade = assessments[i];
      _assessments[i].params = *_params;
    }
  }
}
bool ParameterIterator::Iterate(){
  for (size_t i = 0; i < _params->size(); ++i) {
    if ((*_params)[i] + _param_info[i].step <= _param_info[i].high) {
      (*_params)[i] += _param_info[i].step;
      break;
    }else {
      if (i == _params->size() - 1) {
        return false;
      }
      (*_params)[i] = _param_info[i].low;
    }
  }
  _progress += 1;
  return true;
}

double ParameterIterator::Progress(){
  return (double)_progress / _total;
}
ParameterIterator::ParameterIterator(std::vector<int>* params, std::vector<Param>& p_meta, int assessment_count){
  LinkParams(params);
  AddParamMeta(p_meta);
  _assessments = std::vector<Assessment>(assessment_count);
}