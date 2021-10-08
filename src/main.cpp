#include <math.h>

#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

#include "bigint.h"
#include "curves.h"
#include "parameters.h"

cv::Mat getU();
cv::Mat U = getU();

std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch (depth) {
    case CV_8U:
      r = "8U";
      break;
    case CV_8S:
      r = "8S";
      break;
    case CV_16U:
      r = "16U";
      break;
    case CV_16S:
      r = "16S";
      break;
    case CV_32S:
      r = "32S";
      break;
    case CV_32F:
      r = "32F";
      break;
    case CV_64F:
      r = "64F";
      break;
    default:
      r = "User";
      break;
  }

  r += "C";
  r += (chans + '0');

  return r;
}
void getKoefs(cv::Mat& img, double& k, double& b) {
  cv::Mat temp;
  img.copyTo(temp);
  temp.reshape(1);
  double min, max;
  cv::minMaxIdx(temp, &min, &max);
  k = 255.0 / (max - min);
  b = 0 - min * k;
}
cv::Mat getMapped(cv::Mat& img) {
  double k, b;
  getKoefs(img, k, b);

  if (img.type() == CV_64FC3) {
    cv::Mat temp;
    img.convertTo(temp, CV_8UC3);
    cv::convertScaleAbs(img, temp, k, b);
    return temp;
  }
  throw std::exception("only for CV_64FC3");
}
void showMapped(std::string window_name, cv::Mat& img) {
  auto temp = getMapped(img);
  cv::imshow(window_name, temp);
}

double geometricMeanDouble(cv::Vec3d& val) {
  auto sum = val[0] * val[0] + val[1] * val[1] + val[2] * val[2];
  return pow(sum, 1.0 / 3);
}
double geometricMeanByte(cv::Vec3b& val) {
  auto sum = val[0] * val[0] + val[1] * val[1] + val[2] * val[2];
  return pow(sum, 1.0 / 3);
}
cv::Mat geometricMeanInvariantLog(cv::Mat img) {
  cv::Mat res = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_64FC3);
  img.forEach<cv::Vec3d>([&](cv::Vec3d& pix, const int* pos) -> void {
    auto mean = geometricMeanDouble(pix);
    for (int i = 0; i < 3; ++i) {
      double logarithmized =
          log(std::max(1.0, (double)pix[i]) / std::max(1.0, mean));
      res.at<cv::Vec3d>(pos[0], pos[1])[i] = logarithmized;
    }
  });
  return res;
}
cv::Mat getU() {
  float Iflat[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  float uflat[] = {1.0 / sqrt(3), 1.0 / sqrt(3), 1.0 / sqrt(3)};
  cv::Mat I = cv::Mat(cv::Size(3, 3), CV_32F, Iflat);
  cv::Mat u = cv::Mat(cv::Size(1, 3), CV_32F, uflat);

  cv::Mat P = I - u * u.t();
  // idk how to solve P = U.t*U;
  cv::Mat U = (cv::Mat_<double>(2, 3) << 0.70710678, -0.70710678, 0, 0.40824829,
               0.40824829, -0.81649658);
  return U;
}
cv::Mat to2DPlane(cv::Mat& src) {
  cv::Mat2d dst = cv::Mat(cv::Size(src.rows, src.cols), CV_64FC2);
  src.forEach<cv::Vec3d>([&](cv::Vec3d& pix, const int* pos) -> void {
    auto chi = cv::Mat(U * pix);
    dst.at<cv::Vec2d>(pos[0], pos[1])[0] = chi.at<double>(0, 0);
    dst.at<cv::Vec2d>(pos[0], pos[1])[1] = chi.at<double>(1, 0);
  });
  return dst;
}
cv::Mat projection(cv::Mat chi, double theta) {
  cv::Mat proj = cv::Mat(cv::Size(chi.rows, chi.cols), CV_64F);
  chi.forEach<cv::Vec2d>([&](cv::Vec2d& pix, const int* pos) -> void {
    proj.at<double>(pos[0], pos[1]) = pix[0] * cos(theta) + pix[1] * sin(theta);
  });
  double k, b;
  getKoefs(proj, k, b);
  proj.convertTo(proj, CV_8U, k, b);
  return proj;
}
int binWidthScott(std::vector<uint8_t> data) {
  std::vector<double> means, deviations;
  cv::meanStdDev(data, means, deviations);
  int width = 3.5 * deviations[0] * pow(data.size(), 1.0 / 3);
  return width;
}
cv::Mat plotHist(std::vector<int>& hist) {
  auto plot = cv::Mat(cv::Size(200, 200), CV_8U, cv::Scalar(0, 0, 0));
  int max = *std::max_element(hist.begin(), hist.end());
  int width = plot.cols / hist.size();
  for (int i = 0; i < hist.size(); ++i) {
    int height = (double)hist[i] / max * plot.rows;
    cv::rectangle(plot, cv::Rect(i * width, plot.rows - height, width, height),
                  cv::Scalar(255, 0, 0));
  }
  return plot;
}
std::vector<int> calcHist(cv::Mat img) {
  std::vector<uint8_t> data = img.clone().reshape(1, 1);
  std::sort(data.begin(), data.end());
  data = std::vector(&data[data.size() * 0.05], &data[data.size() * 0.95]);
  auto bin_width = binWidthScott(data);
  auto step = (double)(data.back() - data.front()) / (data.size() / bin_width);
  double thresh = data[0] + step;
  std::vector<int> hist(data.size() / bin_width);
  int j = 0;
  for (int i = 0; i < hist.size(); ++i) {
    for (; j < data.size(); ++j) {
      if (data[j] <= thresh) {
        hist[i] += 1;
      } else {
        thresh += step;
        break;
      }
    }
  }
  return hist;
}
Dodecahedron::Bigint zero_minus_entropy(cv::Mat img) {
  auto hist = calcHist(img);
  // for (int i = 0; i < hist.size(); ++i) {
  //   std::cout << hist[i] << ", ";
  // }
  // std::cout << "\n\n";
  Dodecahedron::Bigint zero_minus_entropy = 0;
  for (int i = 0; i < hist.size(); ++i) {
    if (hist[i] != 0) {
      zero_minus_entropy += hist[i] * log(hist[i]);
    }
  }
  // auto plot = plotHist(hist);
  // cv::imshow("hist", plot);
  // cv::imshow("gray", img);
  // cv::waitKey();
  return zero_minus_entropy;
}
cv::Mat plotScatter(std::vector<double>& values) {
  auto plot = cv::Mat(cv::Size(200, 200), CV_8U, cv::Scalar(0, 0, 0));
  auto minmax = std::minmax_element(values.begin(), values.end());
  int step = plot.cols / values.size();
  double x(0);
  double y = plot.rows - (values[0] - *minmax.first) /
                             (*minmax.second - *minmax.first) * plot.rows;
  cv::Point2d p1(y, x);
  for (int i = 1; i < values.size(); ++i) {
    y = plot.rows - (values[i] - *minmax.first) /
                        (*minmax.second - *minmax.first) * plot.rows;
    x += step;
    cv::line(plot, cv::Point2d(x, y), p1, cv::Scalar(255, 0, 0));
    p1 = cv::Point2d(x, y);
  }
  return plot;
}
cv::Mat findBestProjection(cv::Mat chi) {
  cv::Mat proj;
  // std::vector<double> entropies(180);
  Dodecahedron::Bigint max = 0;
  double max_theta = -1;
  for (int theta = 0; theta < 180; ++theta) {
    proj = projection(chi, theta);
    Dodecahedron::Bigint e = zero_minus_entropy(proj);
    // entropies[theta] = e;
    // std::cout << e << ' ';
    if (e > max) {
      max = e;
      // std::cout << max << ' ' << theta << '\n';
      max_theta = theta;
    }
    // cv::imshow("gray", projection(chi, theta));
    // std::cout << theta << '\n';
    // cv::waitKey();
  }
  // std::cout << '\n';
  // auto e_plot = plotScatter(entropies);
  // cv::imshow("entropies", e_plot);
  // std::cout << min;

  return projection(chi, 97);
}
cv::Mat removeShadows(cv::Mat img) {
  cv::Mat temp;
  img.copyTo(temp);
  cv::GaussianBlur(img, temp, cv::Size(5, 5), 0);
  auto rho = geometricMeanInvariantLog(temp);
  showMapped("c3", rho);
  auto chi = to2DPlane(rho);

  temp = findBestProjection(chi);
  return temp;
}
void ThreshConnectedComponents(cv::Mat& src, cv::Mat& dst, int thresh) {
  cv::Mat labels, stats, centroids;
  src.copyTo(dst);
  cv::connectedComponentsWithStats(src, labels, stats, centroids);
  for (int y = 0; y < labels.rows; ++y) {
    for (int x = 0; x < labels.cols; ++x) {
      auto label = labels.at<int>(y, x);
      if (label == 0) {
        dst.at<uint8_t>(y, x) = 0;
      } else {
        auto size = stats.at<int32_t>(label, cv::CC_STAT_AREA);
        if (size < thresh) {
          dst.at<uint8_t>(y, x) = 0;
        }
      }
    }
  }
}
cv::Mat removeNoise(cv::Mat img) {
  cv::Mat filtered;
  filtered = img.clone();
  // cv::morphologyEx(
  //     img, filtered, cv::MORPH_CLOSE,
  //     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
  // cv::imshow("filtered", filtered);
  // cv::waitKey();
  auto noiseless = cv::Mat(filtered.rows, filtered.cols, CV_8U);
  ThreshConnectedComponents(filtered, noiseless, 20);
  return noiseless;
}
void drawRoads(cv::Mat img, cv::Mat curves) {
  curves.forEach<uint8_t>([&](uint8_t& pix, const int* pos) -> void {
    if (pix == 255) {
      img.at<cv::Vec3b>(pos[0], pos[1]) = cv::Vec3b(0, 0, 255);
    }
    if (pix == 128) {
      img.at<cv::Vec3b>(pos[0], pos[1]) = cv::Vec3b(0, 255, 0);
    }
  });
}
int bfs(cv::Mat img, std::list<cv::Point2i>& q, cv::Mat labels, int label) {
  int count(0);
  auto point = q.front();
  for (int i = -1; i <= 1; ++i) {
    for (int j = -1; j <= 1; ++j) {
      if (i == 0 && j == 0) {
        continue;
      }
      auto cur = cv::Point2i(point.x + j, point.y + i);
      if (cur.x < 0 || cur.x >= img.cols || cur.y < 0 || cur.y >= img.rows) {
        continue;
      }
      if (img.at<uint8_t>(cur) > 0 && labels.at<int32_t>(cur) == 0) {
        q.push_back(cur);
        count += 1;
        labels.at<int32_t>(cur) = label;
      }
    }
  }
  q.pop_front();
  return count;
}
double distance(const cv::Point2i& a, const cv::Point2i& b) {
  return std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}
void leaveTwoFurthest(std::vector<cv::Point2i>& points) {
  double maxd(-1);
  int ind1, ind2;
  for (int i = 0; i < points.size(); ++i) {
    for (int j = i + 1; j < points.size(); ++j) {
      double d = distance(points[i], points[j]);
      if (d > maxd) {
        maxd = d;
        ind1 = i;
        ind2 = j;
      }
    }
  }
  auto temp = points[0];
  points[0] = points[ind1];
  if (ind2 == 0) {
    points[1] = temp;
  } else {
    points[1] = points[ind2];
  }
  points.resize(2);
}
int findEnds(cv::Mat img,
             std::vector<std::pair<cv::Point2i, cv::Point2i>>& ends,
             std::vector<std::vector<cv::Point2i>>& lines, cv::Mat labels) {
  cv::Mat vis_labels = cv::Mat(labels.rows, labels.cols, CV_8U);
  cv::namedWindow("ends", cv::WINDOW_NORMAL);
  cv::resizeWindow("ends", cv::Size(400, 400));
  cv::namedWindow("labels", cv::WINDOW_NORMAL);
  cv::resizeWindow("labels", cv::Size(400, 400));
  std::list<cv::Point2i> q;
  int label(1);
  for (int y = 0; y < img.rows; ++y) {
    for (int x = 0; x < img.cols; ++x) {
      if (img.at<uint8_t>(y, x) > 0 && labels.at<int32_t>(y, x) == 0) {
        std::vector<cv::Point2i> two_ends;
        lines.push_back(std::vector<cv::Point2i>());
        q.push_back(cv::Point2i(x, y));
        lines[label - 1].push_back(cv::Point2i(x, y));
        labels.at<int32_t>(y, x) = label;
        int count = bfs(img, q, labels, label);
        if (count == 1) {
          two_ends.push_back(cv::Point2i(x, y));
          // if (label == 8) {
          // vis_labels = labels == label;
          // cv::imshow("labels", vis_labels);
          // std::cout << count << '\n';
          // cv::waitKey();
          //}
        }
        while (!q.empty()) {
          auto cur = q.front();
          lines[label - 1].push_back(cur);
          // labels.at<int32_t>(cur) = label;
          count = bfs(img, q, labels, label);
          if (count == 0) {
            two_ends.push_back(cur);
            // if (label == 8) {
            // vis_labels = labels == label;
            // cv::imshow("labels", vis_labels);
            // std::cout << count << '\n';
            // cv::waitKey();
            //}
          }
        }
        if (two_ends.size() != 2) {
          // workaround fix
          leaveTwoFurthest(two_ends);
        }
        ends.push_back(std::pair(two_ends[0], two_ends[1]));

        // img.at<uint8_t>(two_ends[0]) = 128;
        // img.at<uint8_t>(two_ends[1]) = 128;
        // std::cout << label << ' ' << two_ends[0] << ' ' << two_ends[1] <<
        // vis_labels = labels != 0;
        // cv::imshow("labels", vis_labels);
        // cv::imshow("ends", img);
        // cv::waitKey();
        //
        // cv::Mat vis_labels = cv::Mat(labels.rows, labels.cols, CV_8U);
        // vis_labels = labels == label;
        // std::cout << label << '\n';
        // cv::imshow("labels", vis_labels);
        // cv::waitKey();
        label += 1;
      }
    }
  }
  return label;
}
cv::Point2i findClosest(cv::Point2i point,
                        const std::vector<cv::Point2i>& line) {
  double min_value(100000);
  cv::Point2i min_point;
  for (int i = 0; i < line.size(); ++i) {
    auto dist = distance(point, line[i]);
    if (dist < min_value) {
      min_value = dist;
      min_point = line[i];
    }
  }
  return min_point;
}
void visualizeClosest(cv::Point2i a, cv::Point2i b,
                      std::vector<std::vector<cv::Point2i>> lines) {
  cv::Mat vis = cv::Mat::zeros(200, 200, CV_8U);
  for (int l = 0; l < lines.size(); ++l) {
    auto line = lines[l];
    for (int i = 0; i < line.size(); ++i) {
      vis.at<uint8_t>(line[i]) = 255;
    }
  }
  cv::line(vis, a, b, cv::Scalar(128));
  cv::imshow("closest", vis);
  cv::waitKey();
}
void selectMostParallel(std::vector<std::pair<cv::Point2i, cv::Point2i>>& ends,
                        std::vector<std::pair<int, int>>& dest, cv::Mat labels,
                        const std::vector<std::vector<cv::Point2i>>& lines,
                        double d, double l) {
  for (int i = 0; i < ends.size(); ++i) {
    for (int j = 0; j < ends.size(); ++j) {
      if (i == j) {
        continue;
      }
      auto closest1 = findClosest(ends[i].first, lines[j]);
      // visualizeClosest(ends[i].first, closest1, lines);
      // std::cout << "d1: " << distance(closest1, ends[i].first) << '\n';
      if (distance(closest1, ends[i].first) > l) {
        continue;
      }
      auto closest2 = findClosest(ends[i].second, lines[j]);
      // visualizeClosest(ends[i].second, closest2, lines);
      // std::cout << "d2: " << distance(closest2, ends[i].second) << '\n';
      if (distance(closest2, ends[i].second) > l) {
        continue;
      }
      auto v1 = ends[i].first - closest1;
      auto v2 = ends[i].second - closest2;
      if (cv::norm(v1 - v2) == 0) {
        std::cout << ends[i].first << ' ' << ends[i].second << '\n';
        // visualizeClosest(ends[i].first, closest1, lines);
        // std::cout << "d1: " << distance(closest1, ends[i].first) << '\n';
        // visualizeClosest(ends[i].second, closest2, lines);
        // std::cout << "d2: " << distance(closest2, ends[i].second) << '\n';
      }
      if (cv::norm(v1 - v2) <= d) {
        // std::cout << cv::norm(v1 - v2) << '\n';
        dest.push_back(std::pair(i, j));
      }
      // if (distance(ends[i].first, ends[j].second) <= l &&
      //    distance(ends[i].second, ends[j].first) <= l) {
      //  std::swap(ends[j].first, ends[j].second);
      //}
      // if (distance(ends[i].first, ends[j].first) <= l &&
      //    distance(ends[i].second, ends[j].second) <= l) {
      //  auto v1 = ends[j].first - ends[i].first;
      //  auto v2 = ends[j].second - ends[i].second;
      //  std::cout << cv::norm(v1 - v2) << '\n';
      //  if (cv::norm(v1 - v2) <= d) {
      //    dest.push_back(std::pair(ends[i], ends[j]));
      //  }
      //}
    }
  }
}
std::vector<cv::Point2i> getLine(cv::Point2i a, cv::Point2i b) {
  std::vector<cv::Point2i> pointsOfLine;

  int dx = abs(b.x - a.x), sx = a.x < b.x ? 1 : -1;
  int dy = abs(b.y - a.y), sy = a.y < b.y ? 1 : -1;
  int err = (dx > dy ? dx : -dy) / 2, e2;

  for (;;) {
    pointsOfLine.push_back(a);
    if (a == b) break;
    e2 = err;
    if (e2 > -dx) {
      err -= dy;
      a.x += sx;
    }
    if (e2 < dy) {
      err += dx;
      a.y += sy;
    }
  }
  return pointsOfLine;
}
void drawMiddle(std::vector<std::pair<int, int>>& parallel,
                std::vector<std::vector<cv::Point2i>>& lines, cv::Mat& dest) {
  dest = cv::Mat::zeros(200, 200, CV_8U);
  for (int i = 0; i < parallel.size(); ++i) {
    std::vector<cv::Point2i>& line1 = lines[parallel[i].first];
    std::vector<cv::Point2i>& line2 = lines[parallel[i].second];
    // for (int j = 0; j < line1.size(); ++j) {
    //  dest.at<uint8_t>(line1[j]) = 255;
    //}
    // for (int j = 0; j < line2.size(); ++j) {
    //  dest.at<uint8_t>(line2[j]) = 255;
    //}
    std::vector<cv::Point2i> prev;
    for (int j = 0; j < line1.size(); ++j) {
      auto closest = findClosest(line1[j], line2);
      auto line = getLine(line1[j], closest);

      for (int k = 0; k < line.size(); ++k) {
        dest.at<uint8_t>(line[k]) = 255;
      }

      for (int k = 0; k < prev.size(); ++k) {
        for (int ii = 0; ii < line.size(); ++ii) {
          cv::line(dest, prev[k], line[ii], cv::Scalar(255));
        }
      }

      prev = line;
      // cv::line(dest, line1[j], closest, cv::Scalar(255));
      // auto middle = cv::Point2i((closest.x + line1[j].x) / 2,
      //                          (closest.y + line1[j].y) / 2);
      // if (j == 0) {
      //  dest.at<uint8_t>(middle) = 255;
      //} else {
      //  cv::line(dest, prev, middle, cv::Scalar(255));
      //}
      // prev = middle;
    }
    // cv::imshow("parallel", dest);
    // cv::waitKey();
    // dest = cv::Mat::zeros(200, 200, CV_8U);
  }

  // cv::ximgproc::thinning(dest, dest, cv::ximgproc::THINNING_GUOHALL);
}
cv::Mat middleFromParallel(cv::Mat img, double d, double l) {
  cv::Mat middles = cv::Mat(img.rows, img.cols, CV_8U);

  cv::Mat labels = cv::Mat::zeros(cv::Size(img.rows, img.cols), CV_32S);
  std::vector<std::pair<cv::Point2i, cv::Point2i>> ends;
  std::vector<std ::vector<cv::Point2i>> lines;
  int n_labels = findEnds(img, ends, lines, labels);

  std::vector<std::pair<int, int>> mostParallel;
  selectMostParallel(ends, mostParallel, labels, lines, d, l);

  drawMiddle(mostParallel, lines, middles);
  return middles;
}
void addBorder(cv::Mat& img, int width) {
  for (int i = 0; i < width; ++i) {
    for (int j = 0; j < img.cols; ++j) {
      img.at<uint8_t>(i, j) = 0;
    }
  }
  for (int i = img.rows - width; i < img.rows; ++i) {
    for (int j = 0; j < img.cols; ++j) {
      img.at<uint8_t>(i, j) = 0;
    }
  }
  for (int i = 0; i < img.rows; ++i) {
    for (int j = 0; j < width; ++j) {
      img.at<uint8_t>(i, j) = 0;
    }
  }
  for (int i = 0; i < img.rows; ++i) {
    for (int j = img.cols - width; j < img.cols; ++j) {
      img.at<uint8_t>(i, j) = 0;
    }
  }
}
void prepareReference() {
  using directory_iterator = std::filesystem::directory_iterator;
  // std::cout << std::filesystem::current_path();
  for (const auto& dirEntry :
       directory_iterator("../../data/rural/selected_demo/reference")) {
    auto img = cv::imread(dirEntry.path().string(), cv::IMREAD_COLOR);
    // auto img = cv::imread("../data/rural/roads_100/rural_001500.png",

    // cv::IMREAD_COLOR);
    if (!img.data) {
      return;
    }
    cv::imshow("img", img);

    cv::Mat centers = cv::Mat::zeros(img.rows, img.cols, CV_8U);
    img.forEach<cv::Vec3b>([&](cv::Vec3b& pix, const int* pos) -> void {
      if (pix == cv::Vec3b(0, 0, 255)) {
        centers.at<uint8_t>(pos[0], pos[1]) = 255;
      }
    });

    addBorder(centers, 1);
    cv::ximgproc::thinning(centers, centers, cv::ximgproc::THINNING_ZHANGSUEN);
    cv::ximgproc::thinning(centers, centers, cv::ximgproc::THINNING_GUOHALL);
    cv::imwrite(dirEntry.path().string(), centers);

    // cv::imshow("centers", centers);

    // cv::waitKey();
  }
}
void selectPoints(cv::Mat& img, std::vector<cv::Point2i>& points) {
  for (int i = 0; i < img.rows; ++i) {
    for (int j = 0; j < img.cols; ++j) {
      if (img.at<uint8_t>(i, j) == 255) {
        points.push_back(cv::Point2i(j, i));
      }
    }
  }
}
std::vector<double> assess(cv::Mat& img, cv::Mat& ref, cv::Mat vis_dest) {
  int tp(0), tn(0), fp(0), fn(0);
  for (int i = 0; i < img.rows; ++i) {
    for (int j = 0; j < img.cols; ++j) {
      auto ref_current = ref.at<cv::Vec3b>(i, j)[0];
      auto img_current = img.at<uint8_t>(i, j);
      if (ref_current == 255) {
        if (img_current == 255) {
          tp += 1;
          vis_dest.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 255, 0);
        } else {
          fn += 1;
          vis_dest.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 0, 0);
        }
      } else {
        if (img_current == 255) {
          fp += 1;
          vis_dest.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);
        } else {
          tn += 1;
        }
      }
    }
  }
  auto accuracy = (double)(tp + tn) / (tp + tn + fp + fn);
  auto precision = (double)tp / (tp + fp);
  auto recall = (double)tp / (tp + fn);
  if (tp == 0) {
    precision = 0;
  }

  return std::vector<double>{accuracy, precision, recall};
}

cv::Mat sRGBToLinearRGB(cv::Mat img) {
  cv::Mat res = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_64FC3);
  img.forEach<cv::Vec3b>([&](cv::Vec3b& pix, const int* pos) -> void {
    auto mean = geometricMeanByte(pix);
    for (int i = 0; i < 3; ++i) {
      float s = pix[i];
      float linear;
      if (s <= 0.04045)
        linear = s / 12.92;
      else
        linear = pow((s + 0.055) / 1.055, 2.4);
      res.at<cv::Vec3d>(pos[0], pos[1])[i] = linear;
    }
  });
  return res;
}
void saveEnlarged(std::string path, cv::Mat img, double scale) {
  cv::resize(img, img, cv::Size(img.rows * scale, img.cols * scale));
  cv::imwrite(path, img);
}
int main() {
  // // //prepareReference();
  std::cout << "started" << '\n';
  cv::namedWindow("shadowless", cv::WINDOW_NORMAL);
  cv::resizeWindow("shadowless", cv::Size(400, 400));
  cv::namedWindow("edges_from_shadowless", cv::WINDOW_NORMAL);
  cv::resizeWindow("edges_from_shadowless", cv::Size(400, 400));
  cv::namedWindow("edges_from_original", cv::WINDOW_NORMAL);
  cv::resizeWindow("edges_from_original", cv::Size(400, 400));
  cv::namedWindow("curves", cv::WINDOW_NORMAL);
  cv::resizeWindow("curves", cv::Size(400, 400));
  cv::namedWindow("roads", cv::WINDOW_NORMAL);
  cv::resizeWindow("roads", cv::Size(400, 400));
  using directory_iterator = std::filesystem::directory_iterator;
  // std::cout << std::filesystem::current_path();
  for (const auto& dirEntry :
       directory_iterator("../data/rural/selected_demo/orig")) {
    auto img = cv::imread(dirEntry.path().string(), cv::IMREAD_COLOR);
    // auto img = cv::imread("../data/rural/roads_100/rural_001500.png",
    // cv::IMREAD_COLOR);
    if (!img.data) {
      return 1;
    }
    cv::imshow("img", img);
    auto ref =
        cv::imread(dirEntry.path().parent_path().parent_path().string() +
                       "/reference/" + dirEntry.path().filename().string(),
                   cv::IMREAD_COLOR);
    if (!img.data) {
      return 1;
    }
    cv::imshow("ref", ref);

    auto linear = sRGBToLinearRGB(img);
    showMapped("linear", linear);

    auto shadowless = removeShadows(linear);
    cv::GaussianBlur(shadowless, shadowless, cv::Size(3, 3), 0);
    cv::imshow("shadowless", shadowless);

    // auto params = std::vector<int>{185, 250, 32, 50, 19, 14};
    // auto params = std::vector<int>{97, 40, 5, 5, 50, 50};
    auto params = std::vector<int>{24, 35, 28, 30, 20, 40};
    // auto params = std::vector<int>{130, 153, 1, 34, 11, 12};
    int& canny_t1{params[0]};
    int& canny_t2{params[1]};
    int& sc_r{params[2]};
    int& sc_l{params[3]};
    int& mfp_d{params[4]};
    int& mfp_l{params[5]};

    cv::Mat edges_from_original;
    cv::Canny(img, edges_from_original, canny_t1, canny_t2);
    cv::imshow("edges_from_original", edges_from_original);
    cv::Mat nl_orig;
    nl_orig = removeNoise(edges_from_original);
    cv::imshow("nl_orig", nl_orig);

    cv::Mat edges_from_shadowless;
    cv::Canny(shadowless, edges_from_shadowless, canny_t1, canny_t2);
    cv::imshow("edges_from_shadowless", edges_from_shadowless);

    cv::Mat nl_sl;
    nl_sl = removeNoise(edges_from_shadowless);
    cv::imshow("nl_shadowless", nl_sl);

    auto curves = SelectCurves(nl_sl, sc_r, sc_l);
    cv::imshow("curves", curves);

    auto roads_bin = middleFromParallel(curves == 64, mfp_d, mfp_l);
    cv::imshow("road_fill", roads_bin);

    auto roads = img.clone();
    auto assessment = assess(roads_bin, ref, roads);
    cv::imshow("refed", ref);
    std::cout << dirEntry.path().string() << ' ' << assessment[0] << ' '
              << assessment[1] << ' ' << assessment[2] << '\n';

    cv::imshow("roads", roads);

    auto key = cv::waitKey();
    if (key == 'y') {
      std::string path = "../data/rural/selected_demo/refined/";
      auto filename = dirEntry.path().filename().string();

      saveEnlarged(path + "orig/" + filename, img, 1.7);
      saveEnlarged(path + "edges_original/" + filename, edges_from_original,
                   1.7);
      saveEnlarged(path + "original_nl/" + filename, edges_from_original, 1.7);

      cv::Mat linear_for_write = getMapped(linear);
      saveEnlarged(path + "linear/" + filename, linear_for_write, 1.7);

      saveEnlarged(path + "edges_shadowless/" + filename, edges_from_shadowless,
                   1.7);
      saveEnlarged(path + "shadowless/" + filename, shadowless, 1.7);
      saveEnlarged(path + "shadowless_nl/" + filename, nl_sl, 1.7);
      saveEnlarged(path + "curves/" + filename, curves, 1.7);
      saveEnlarged(path + "centers/" + filename, roads_bin, 1.7);
      saveEnlarged(path + "detected/" + filename, roads, 1.7);
    }
  }
  std::cout << "ended\n";
}