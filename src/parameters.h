#ifndef PARAMETERS_03092021
#define CURVES_03092021

#include <vector>

struct Param {
 public:
   int low, high, step;
};
struct Assessment {
 public:
   double grade{ 0 };
   std::vector<int> params;
};

class ParameterIterator {
 public:
   ParameterIterator(std::vector<int>* params, std::vector<Param>& p_meta, int assessment_count);

   void AddParamMeta(std::vector<Param>& p);
   bool Iterate();
   double Progress();
   void LinkParams(std::vector<int>*);

   void RecieveFeedback(std::vector<double> assessments);

   std::vector<Assessment> _assessments;
 private:
   std::vector<Param> _param_info;
   std::vector<int>* _params;
   long long  _progress{ 0 }, _total{ 0 };
};

#endif