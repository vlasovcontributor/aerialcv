#include <iostream>
#include <opencv2/opencv.hpp>

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
int subdivide(const cv::Mat &img, const int rowDivisor, const int colDivisor,
              std::vector<cv::Mat> &blocks) {
  /* Checking if the image was passed correctly */
  if (!img.data || img.empty())
    std::cerr << "Problem Loading Image" << std::endl;

  // check if divisors fit to image dimensions
  if (img.cols % colDivisor == 0 && img.rows % rowDivisor == 0) {
    for (int y = 0; y < img.cols; y += img.cols / colDivisor) {
      for (int x = 0; x < img.rows; x += img.rows / rowDivisor) {
        blocks.push_back(img(cv::Rect(y, x, (img.cols / colDivisor),
                                      (img.rows / rowDivisor)))
                             .clone());
      }
    }
  } else if (img.cols % colDivisor != 0) {
    std::cerr << "Error: Please use another divisor for the column split.\n";
    exit(1);
  } else if (img.rows % rowDivisor != 0) {
    std::cerr << "Error: Please use another divisor for the row split.\n";
    exit(1);
  }
  return EXIT_SUCCESS;
}
cv::Vec3b black(0, 0, 0);
cv::Vec3b white(255, 255, 255);
int main() {
  auto img = cv::imread("../data/rural/flat.png", cv::IMREAD_COLOR);
  if (!img.data) {
    return 1;
  }
  cv::resize(img, img, cv::Size(22400, 19000));
  std::vector<cv::Mat> tiles;
  subdivide(img, 95, 112, tiles);
  for (int i = 0; i < tiles.size(); ++i) {
    auto pixel = tiles[i].at<cv::Vec3b>(0, 0);
    if (pixel != black && pixel != white) {
      char str[255];
      snprintf(str, 255, "../data/rural/tiles/rural_%06d.png", i);
      std::string name(str);
      cv::imwrite(std::string(str), tiles[i]);
      // cv::imshow("tile", tiles[i]);
      // cv::waitKey();
    }
  }
}