#ifndef CURVES_24052021
#define CURVES_24052021

#include <opencv2/opencv.hpp>
#include <vector>

cv::Mat SelectCurves(cv::Mat img, int r, int l);

class Finder {
 public:
  Finder(cv::Mat& image, cv::Point2i point, int r, int l);
  void SelectCurve();
  void Extrapolate();

 private:
  void search();
  bool isBent(cv::Point2i point);
  void bfs(cv::Point2i point, const std::vector<uint8_t>& available,
           uint8_t visited);
  void extrapolate(cv::Point2i point);
  std::vector<cv::Point2i> pollDirection(cv::Point2i point);
  void getArea(cv::Point2i center, double l, std::vector<cv::Point2i>& area);
  void checkElongation(cv::Point2i start, cv::Point2i end);

  void findEnds(std::vector<cv::Point2i>& ends);
  int extrBfs();
  // void extrSearch(cv::Point2i start);
  void extrSearchBfs();

  void selectBest();

  cv::Mat* image_;
  int r_;
  int l_;

  cv::Mat img_copy_;
  cv::Point2i start_;

  std::list<cv::Point2i> q_;
  cv::Mat prev_;
  cv::Mat path_lens_;

  std::vector<cv::Point2i> interesting_;
};

#endif