#include "curves.h"

#define _USE_MATH_DEFINES
#include <math.h>

#include <list>
#include <map>
#include <opencv2/ximgproc.hpp>

double distance(cv::Point2i a, cv::Point2i b) {
  return std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}
Finder::Finder(cv::Mat& image, cv::Point2i point, int r, int l)
    : image_(&image), start_(point), r_(r), l_(l) {
  img_copy_ = image.clone();

  prev_ = cv::Mat(image.rows, image.cols, CV_32SC2);
  prev_.at<cv::Point2i>(point) = point;
  path_lens_ = cv::Mat(image.rows, image.cols, CV_64F,
                       cv::Scalar(std::numeric_limits<double>::max()));
  path_lens_.at<double>(point) = 0;
  q_.push_back(point);
  img_copy_.at<uint8_t>(point) = 128;

  interesting_.resize(0);
}
void Finder::bfs(cv::Point2i point, const std::vector<uint8_t>& available,
                 uint8_t visited) {
  for (int i = -1; i <= 1; ++i) {
    for (int j = -1; j <= 1; ++j) {
      auto cur = cv::Point2i(point.x + j, point.y + i);
      if (cur.x < 0 || cur.x >= img_copy_.cols || cur.y < 0 ||
          cur.y >= img_copy_.rows) {
        continue;
      }

      uint8_t& pixel = img_copy_.at<uint8_t>(cur);
      if (std::find(available.begin(), available.end(), pixel) !=
          available.end()) {
        // if (pixel == 255 || pixel == 64){
        pixel = visited;
        q_.push_back(cur);
        path_lens_.at<double>(cur) =
            path_lens_.at<double>(point) + distance(cur, point);
        prev_.at<cv::Point2i>(cur) = point;
      }
    }
  }
}

bool Finder::isBent(cv::Point2i point) {
  if (point.x == 45 && point.y == 11) {
    int x = 0;
  }
  double length(0);
  auto prev_point = point;
  while (length < l_) {  // too little length
    if (prev_.at<cv::Point2i>(prev_point) == prev_point) {
      return false;
    }
    length += distance(prev_.at<cv::Point2i>(prev_point), prev_point);
    prev_point = prev_.at<cv::Point2i>(prev_point);
  }
  auto d = distance(prev_point, point);
  // line too bent
  return d < r_;
}

void Finder::SelectCurve() {
  search();
  // extrapolate();
  // cv::imshow("copy", img_copy_);
  // cv::waitKey();
  selectBest();
}
void Finder::search() {
  while (!q_.empty()) {
    auto point = q_.front();
    q_.pop_front();
    bool bent = isBent(point);
    if (!bent) {
      bfs(point, std::vector<uint8_t>{255, 64}, 128);
    }
    double len = path_lens_.at<double>(point);
    double dist = distance(point, start_);
    if (!bent && path_lens_.at<double>(point) >= l_) {
      if (std::find(interesting_.begin(), interesting_.end(), point) ==
          interesting_.end()) {
        interesting_.push_back(point);
      }
    }
  }
}
// void distribute(std::vector<std::pair<cv::Point2i, uint8_t>>& ordered,
//                std::vector<cv::Point2i>& pattern) {
//  std::sort(ordered.begin(), ordered.end(),
//            [](const std::pair<cv::Point2i, uint8_t>& a,
//               const std::pair<cv::Point2i, uint8_t>& b) -> bool {
//              return a.second > b.second;
//            });
//
//  int a = ordered[0].second;
//  int b = ordered[1].second;
//  if (b == 0) {
//    for (int i = 0; i < a; ++i) {
//      pattern.push_back(ordered[0].first);
//    }
//  } else {
//    double k = (double)a / b;
//    double have_a(0);
//    int have_b(1);
//    pattern.push_back(ordered[1].first);
//
//    for (int i = 1; i < a + b; ++i) {
//      if (have_a / have_b > k) {
//        pattern.push_back(ordered[1].first);
//        have_b += 1;
//      } else {
//        pattern.push_back(ordered[0].first);
//        have_a += 1;
//      }
//    }
//  }
//}
// std::vector<cv::Point2i> Finder::pollDirection(cv::Point2i point) {
//  auto less = [](const cv::Point2i& a, const cv::Point2i& b) -> bool {
//    if (a.y == b.y) {
//      return a.x < b.x;
//    }
//    return a.y < b.y;
//  };
//  std::map<cv::Point2i, uint8_t, decltype(less)> options(less);
//  for (int i = -1; i <= 1; ++i) {
//    for (int j = -1; j <= 1; ++j) {
//      options.insert(std::pair(cv::Point2i(i, j), 0));
//    }
//  }
//  auto cur = point;
//  double l(0);
//  while (l < l_) {
//    cv::Point2i dir = cur - prev_.at<cv::Point2i>(cur);
//    if (dir.x == 0 && dir.y == 0) {
//      break;
//    }
//    options.at(dir) += 1;
//    l += distance(cur, prev_.at<cv::Point2i>(cur));
//    cur = prev_.at<cv::Point2i>(cur);
//  }
//  std::vector<std::pair<cv::Point2i, uint8_t>> ordered(options.begin(),
//                                                       options.end());
//  std::vector<cv::Point2i> pattern;
//  distribute(ordered, pattern);
//  return pattern;
//}
// void Finder::getArea(cv::Point2i center, double l,
//                     std::vector<cv::Point2i>& area) {
//  double alpha = M_PI / 24;
//  double r = l * std::tan(alpha);
//  for (double y = -r; y <= r; y += 1) {
//    for (double x = -r; x <= r; x += 1) {
//      auto point = cv::Point2i(center.x + x, center.y + y);
//      if (point.x < 0 || point.x >= img_copy_.cols || point.y < 0 ||
//          point.y >= img_copy_.rows) {
//        continue;
//      }
//      if ((center.x - point.x) * (center.x - point.x) +
//              (center.y - point.y) * (center.y - point.y) <=
//          r * r) {
//        area.push_back(point);
//      }
//    }
//  }
//}
// double getAngle(cv::Point2i start, cv::Point2i main, cv::Point2i near) {
//  cv::Point2i v1 = main - start;
//  cv::Point2i v2 = main - near;
//  float len1 = sqrt(v1.x * v1.x + v1.y * v1.y);
//  float len2 = sqrt(v2.x * v2.x + v2.y * v2.y);
//
//  float dot = v1.x * v2.x + v1.y * v2.y;
//
//  float a = dot / (len1 * len2);
//
//  if (a >= 1.0)
//    return 0.0;
//  else if (a <= -1.0)
//    return M_PI;
//  else
//    return acos(a);  // 0..PI
//}
// void Finder::extrapolate(cv::Point2i point) {
//  auto pattern = pollDirection(point);
//  if (pattern.size() == 0) {
//    return;
//  }
//  double l(0);
//  int i(0);
//  auto cur = point;
//  // std::vector<cv::Point2i> extrapolation;
//  // extrapolation.push_back(point);
//  while (l < path_lens_.at<double>(point)) {
//    cv::Point2i step = cur + pattern[i % pattern.size()];
//    i += 1;
//    if (step.x < 0 || step.x >= img_copy_.cols || step.y < 0 ||
//        step.y >= img_copy_.rows) {
//      break;
//    }
//    // extrapolation.push_back(step);
//    l += distance(step, cur);
//    std::vector<cv::Point2i> area;
//    getArea(step, l, area);
//    for (int j = 0; j < area.size(); ++j) {
//      auto color = img_copy_.at<uint8_t>(area[j]);
//      if (color == 255 || color == 64) {
//        // auto angle = getAngle(point, step, area[j]);
//        checkElongation(point, area[j]);
//      }
//    }
//    cur = step;
//  }
//}
// cv::Point2i rotate(cv::Point2i center, cv::Point2i point, double angle) {
//  cv::Point2i v = point - center;
//  auto rotated = cv::Point2i(v.x * cos(angle) - v.y * sin(angle),
//                             v.x * sin(angle) + v.y * cos(angle));
//  return rotated + center;
//}
// void Finder::checkElongation(cv::Point2i start, cv::Point2i end) {
//  cv::LineIterator li(start, end);
//  ++li;
//  auto prev = start;
//  for (int i = 1; i < li.count; ++i, ++li) {
//    prev_.at<cv::Point2i>(li.pos()) = prev;
//    path_lens_.at<double>(li.pos()) =
//        path_lens_.at<double>(prev) + distance(prev, li.pos());
//    bool bent = isBent(li.pos());
//    if (bent) {
//      return;
//    }
//    prev = li.pos();
//  }
//  // cv::imshow("line", img_copy_);
//  // cv::waitKey();
//  // std::cout << prev << ' ' << end << '\n';
//  q_.push_back(prev);
//  search();
//}
// void Finder::extrapolate() {
//  for (int i = 0; i < interesting_.size(); ++i) {
//    extrapolate(interesting_[i]);
//    // std::cout << interesting_.size() << '\n';
//  }
//}

void Finder::selectBest() {
  if (interesting_.size() != 0) {
    cv::Point2i max_point;
    double max(-1);
    for (int i = 0; i < interesting_.size(); ++i) {
      if (distance(interesting_[i], start_) > max) {
        max = distance(interesting_[i], start_);
        max_point = interesting_[i];
      }
    }
    auto cur = max_point;
    while (prev_.at<cv::Point2i>(cur) != cur) {
      image_->at<uint8_t>(cur) = 64;
      // img_copy_.at<uint8_t>(cur) = 64;
      cur = prev_.at<cv::Point2i>(cur);
    }
    image_->at<uint8_t>(cur) = 64;
    // img_copy_.at<uint8_t>(cur) = 64;
  }
}
cv::Mat SelectCurves(cv::Mat img, int r, int l) {
  auto temp = cv::Mat(img.rows, img.cols, CV_8U);
  cv::ximgproc::thinning(img, temp, cv::ximgproc::THINNING_GUOHALL);

  for (int y = 0; y < temp.rows; ++y) {
    for (int x = 0; x < temp.cols; ++x) {
      if (temp.at<uint8_t>(y, x) == 255) {
        auto finder = Finder(temp, cv::Point2i(x, y), r, l);

        finder.SelectCurve();
      }
    }
  }

  return temp;
}

void distribute(std::vector<std::pair<cv::Point2i, uint8_t>>& ordered,
                std::vector<cv::Point2i>& pattern) {
  std::sort(ordered.begin(), ordered.end(),
            [](const std::pair<cv::Point2i, uint8_t>& a,
               const std::pair<cv::Point2i, uint8_t>& b) -> bool {
              return a.second > b.second;
            });

  int a = ordered[0].second;
  int b = ordered[1].second;
  if (b == 0) {
    for (int i = 0; i < a; ++i) {
      pattern.push_back(ordered[0].first);
    }
  } else {
    double k = (double)a / b;
    double have_a(0);
    int have_b(1);
    pattern.push_back(ordered[1].first);

    for (int i = 1; i < a + b; ++i) {
      if (have_a / have_b > k) {
        pattern.push_back(ordered[1].first);
        have_b += 1;
      } else {
        pattern.push_back(ordered[0].first);
        have_a += 1;
      }
    }
  }
}
std::vector<cv::Point2i> Finder::pollDirection(cv::Point2i point) {
  auto less = [](const cv::Point2i& a, const cv::Point2i& b) -> bool {
    if (a.y == b.y) {
      return a.x < b.x;
    }
    return a.y < b.y;
  };
  std::map<cv::Point2i, uint8_t, decltype(less)> options(less);
  for (int i = -1; i <= 1; ++i) {
    for (int j = -1; j <= 1; ++j) {
      options.insert(std::pair(cv::Point2i(i, j), 0));
    }
  }
  auto cur = point;
  double l(0);
  while (l < l_) {
    cv::Point2i dir = cur - prev_.at<cv::Point2i>(cur);
    if (dir.x == 0 && dir.y == 0) {
      break;
    }
    options.at(dir) += 1;
    l += distance(cur, prev_.at<cv::Point2i>(cur));
    cur = prev_.at<cv::Point2i>(cur);
  }
  std::vector<std::pair<cv::Point2i, uint8_t>> ordered(options.begin(),
                                                       options.end());
  std::vector<cv::Point2i> pattern;
  distribute(ordered, pattern);
  return pattern;
}
void Finder::getArea(cv::Point2i center, double l,
                     std::vector<cv::Point2i>& area) {
  double alpha = M_PI / 24;
  double r = l * std::tan(alpha);
  for (double y = -r; y <= r; y += 1) {
    for (double x = -r; x <= r; x += 1) {
      auto point = cv::Point2i(center.x + x, center.y + y);
      if (point.x < 0 || point.x >= img_copy_.cols || point.y < 0 ||
          point.y >= img_copy_.rows) {
        continue;
      }
      if ((center.x - point.x) * (center.x - point.x) +
              (center.y - point.y) * (center.y - point.y) <=
          r * r) {
        area.push_back(point);
      }
    }
  }
}
double getAngle(cv::Point2i start, cv::Point2i main, cv::Point2i near) {
  cv::Point2i v1 = main - start;
  cv::Point2i v2 = main - near;
  float len1 = sqrt(v1.x * v1.x + v1.y * v1.y);
  float len2 = sqrt(v2.x * v2.x + v2.y * v2.y);

  float dot = v1.x * v2.x + v1.y * v2.y;

  float a = dot / (len1 * len2);

  if (a >= 1.0)
    return 0.0;
  else if (a <= -1.0)
    return M_PI;
  else
    return acos(a);  // 0..PI
}

// void Finder::extrSearchBfs() {
//  auto point = q_.front();
//  int count;
//  for (int i = -1; i <= 1; ++i) {
//    for (int j = -1; j <= 1; ++j) {
//      auto cur = cv::Point2i(point.x + j, point.y + i);
//      if (cur.x < 0 || cur.x >= img_copy_.cols || cur.y < 0 ||
//          cur.y >= img_copy_.rows) {
//        break;
//      }
//      count = 0;
//
//      uint8_t& pixel = img_copy_.at<uint8_t>(cur);
//      if (pixel == 32) {
//        pixel = 16;
//        count += 1;
//        path_lens_.at<double>(cur) =
//            path_lens_.at<double>(point) + distance(cur, point);
//        prev_.at<cv::Point2i>(cur) = point;
//        q_.push_back(cur);
//      } else if (pixel == 32) {
//        count += 1;
//      }
//    }
//  }
//}

// cv::Point2i rotate(cv::Point2i center, cv::Point2i point, double angle) {
//  cv::Point2i v = point - center;
//  auto rotated = cv::Point2i(v.x * cos(angle) - v.y * sin(angle),
//                             v.x * sin(angle) + v.y * cos(angle));
//  return rotated + center;
//}
// void Finder::checkElongation(cv::Point2i start, cv::Point2i end) {
//  cv::LineIterator li(start, end);
//  ++li;
//  auto prev = start;
//  for (int i = 1; i < li.count; ++i, ++li) {
//    prev_.at<cv::Point2i>(li.pos()) = prev;
//    path_lens_.at<double>(li.pos()) =
//        path_lens_.at<double>(prev) + distance(prev, li.pos());
//    bool bent = isBent(li.pos());
//    if (bent) {
//      return;
//    }
//    prev = li.pos();
//  }
//  // cv::imshow("line", img_copy_);
//  // cv::waitKey();
//  // std::cout << prev << ' ' << end << '\n';
//  q_.push_back(prev);
//  search();
//}
// void Finder::extrapolate(cv::Point2i point) {
//  auto pattern = pollDirection(point);
//  if (pattern.size() == 0) {
//    return;
//  }
//  double l(0);
//  int i(0);
//  auto cur = point;
//  // std::vector<cv::Point2i> extrapolation;
//  // extrapolation.push_back(point);
//  while (l < path_lens_.at<double>(point)) {
//    cv::Point2i step = cur + pattern[i % pattern.size()];
//    i += 1;
//    if (step.x < 0 || step.x >= img_copy_.cols || step.y < 0 ||
//        step.y >= img_copy_.rows) {
//      break;
//    }
//    // extrapolation.push_back(step);
//    l += distance(step, cur);
//    std::vector<cv::Point2i> area;
//    getArea(step, l, area);
//    for (int j = 0; j < area.size(); ++j) {
//      auto color = img_copy_.at<uint8_t>(area[j]);
//      if (color == 255 || color == 64) {
//        // auto angle = getAngle(point, step, area[j]);
//        checkElongation(point, area[j]);
//      }
//    }
//    cur = step;
//  }
//}
// void Finder::findEnds(std::vector<cv::Point2i>& ends) {
//  // std::vector<cv::Point2i> ends;
//  for (int i = 0; i < img_copy_.rows; ++i) {
//    for (int j = 0; j < img_copy_.cols; ++j) {
//      auto point = cv::Point2i(j, i);
//      if (img_copy_.at<uint8_t>(point) == 64) {
//        q_.push_back(point);
//        img_copy_.at<uint8_t>(point) = 32;
//        while (!q_.empty()) {
//          point = q_.front();
//          q_.pop_front();
//          auto had = q_.size();
//          bfs(point, std::vector<uint8_t>{64}, 32);
//          if ((had == 0 && q_.size() < 2) || q_.size() == had) {
//            ends.push_back(q_.front());
//          }
//          // auto count = extrBfs();
//          // if (count == 1) {
//          //  ends.push_back(q_.front());
//          //}
//          // q_.pop_front();
//        }
//      }
//    }
//  }
//}
// void Finder::Extrapolate() {
//  std::vector<cv::Point2i> ends;
//  findEnds(ends);
//  for (int i = 0; i < ends.size(); ++i) {
//    extrapolate(ends[i]);
//  }
//}