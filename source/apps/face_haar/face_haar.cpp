#include <cstdio>
#include <opencv2/opencv.hpp>

#include "encoder.hpp"

#include <iomanip>
#include <sstream>

#include <chrono>
#include <ctime>
using std::chrono::system_clock;
std::string get_fname_from_timestamp() {
  //現在日時を取得する
  system_clock::time_point p = system_clock::now();
  //形式を変換する
  std::time_t t     = system_clock::to_time_t(p);
  const std::tm *lt = localtime(&t);
  // sに独自フォーマットになるように連結していく
  std::stringstream s;
  s << "20";
  s << lt->tm_year - 100;  // 100を引くことで20xxのxxの部分になる
  s << "-";
  s << lt->tm_mon + 1;  //月を0からカウントしているため
  s << "-";
  s << lt->tm_mday;  //そのまま
  s << "_";
  s << lt->tm_hour;
  s << "-";
  s << lt->tm_min;
  s << "-";
  s << lt->tm_sec;
  s << ".j2c";
  // result = "2015-5-19_hh-mm-ss"
  return s.str();
}

int32_t log2i32(int32_t x) {
  if (x <= 0) {
    printf("ERROR: cannot compute log2 of negative value.\n");
    exit(EXIT_FAILURE);
  }
  int32_t y = 0;
  while (x > 1) {
    y++;
    x >>= 1;
  }
  return y;
}

int main() {
  cv::Mat frame;
  cv::Mat output;
  cv::CascadeClassifier detector_face, detector_eye;
  // path to XML files might be different. Check your install of OpenCV.
  detector_face.load(
      "/usr/share/opencv/haarcascades/"
      "haarcascade_frontalface_alt.xml");
  detector_eye.load(
      "/usr/share/opencv/haarcascades/"
      "haarcascade_eye.xml");
  const int cap_width  = 640;
  const int cap_height = 480;
  cv::VideoCapture camera(0);
  if (camera.isOpened() == false) {
    printf("ERROR: cannot open the camera.\n");
    return EXIT_FAILURE;
  }
  camera.set(cv::CAP_PROP_FRAME_WIDTH, cap_width);
  camera.set(cv::CAP_PROP_FRAME_HEIGHT, cap_height);

  open_htj2k::siz_params siz;  // information of input image
  siz.Rsiz   = 0;
  siz.Xsiz   = cap_width;
  siz.Ysiz   = cap_height;
  siz.XOsiz  = 0;
  siz.YOsiz  = 0;
  siz.XTsiz  = cap_width;
  siz.YTsiz  = cap_height;
  siz.XTOsiz = 0;
  siz.YTOsiz = 0;
  siz.Csiz   = 3;
  for (auto c = 0; c < siz.Csiz; ++c) {
    siz.Ssiz.push_back(8 - 1);
    // auto compw = cap_width;
    // auto comph = cap_height;
    siz.XRsiz.push_back(1);
    siz.YRsiz.push_back(1);
  }

  open_htj2k::cod_params cod;  // parameters related to COD marker

  cod.blkwidth          = log2i32(64) - 2;
  cod.blkheight         = log2i32(64) - 2;
  cod.is_max_precincts  = true;
  cod.use_SOP           = false;
  cod.use_EPH           = false;
  cod.progression_order = 0;
  cod.number_of_layers  = 1;
  cod.use_color_trafo   = 1;  // 1: RGB->YCbCr ON
  cod.dwt_levels        = 5;
  cod.codeblock_style   = 0x040;
  cod.transformation    = 0;  // 0: lossy, 1:lossless
  // std::vector<element_siz_local> PP = args.get_prct_size();
  // for (auto &i : PP) {
  //   cod.PPx.push_back(i.x);
  //   cod.PPy.push_back(i.y);
  // }

  open_htj2k::qcd_params qcd;  // parameters related to QCD marker
  qcd.is_derived          = false;
  qcd.number_of_guardbits = 1;
  qcd.base_step           = 1.0f / 256.0f;
  if (qcd.base_step == 0.0) {
    qcd.base_step = 1.0f / static_cast<float>(1 << 8);
  }
  const uint8_t color_space = 0;  // 0: sRGB
  if (camera.read(frame) == false) {
    printf("ERROR: cannot grab a frame\n");
    exit(EXIT_FAILURE);
  }
  std::vector<cv::Mat> BGR;
  std::vector<int32_t *> input_buf;
  for (size_t c = 0; c < frame.channels(); ++c) {
    size_t length = static_cast<size_t>(frame.cols * frame.rows);
    input_buf.emplace_back(new int32_t[length]);
  }
  while (true) {
    if (camera.read(frame) == false) {
      printf("ERROR: cannot grab a frame\n");
      break;
    }
    std::vector<cv::Rect> faces, eyes;
    detector_face.detectMultiScale(frame, faces, 1.1, 3, 0, cv::Size(20, 20));
    detector_eye.detectMultiScale(frame, eyes, 1.1, 3, 0, cv::Size(20, 20));
    bool flag = false;
    for (size_t i = 0; i < faces.size(); ++i) {
      // 顔の周辺を長方形で囲む
      // rectangle(frame, faces[i], cv::Scalar(0, 0, 255), 3);
      for (size_t j = 0; j < eyes.size(); ++j) {
        if (eyes[j].x > faces[i].x && eyes[j].x + eyes[j].width < faces[i].x + faces[i].width
            && eyes[j].y > faces[i].y && eyes[j].y + eyes[j].height < faces[i].y + faces[i].height) {
          flag = true;
        }
      }
    }

    if (flag) {
      printf("Face is detected: start encoding");
      cv::split(frame, BGR);
      for (size_t c = BGR.size(); c > 0; --c) {
        size_t length = static_cast<size_t>(BGR[c - 1].cols * BGR[c - 1].rows);
        int32_t *dp   = input_buf[BGR.size() - c];
        uint8_t *sp   = BGR[c - 1].data;
        for (size_t i = 0; i < length; ++i) {
          dp[i] = sp[i];
        }
        printf(".");
      }

      // encode begin

      open_htj2k::openhtj2k_encoder encoder(get_fname_from_timestamp().c_str(), input_buf, siz, cod, qcd,
                                            90, false, color_space, 0);
      size_t total_size = encoder.invoke();
      printf("done.\n");
      // encode end
    } else {
      printf("no face\n");
    }
    // for (int i = 0; i < eyes.size(); ++i) {
    //   rectangle(
    //       frame, cv::Point(eyes[i].x, eyes[i].y),
    //       cv::Point(eyes[i].x + eyes[i].width, eyes[i].y + eyes[i].height),
    //       cv::Scalar(0, 255, 0), 3);
    // }
    cv::flip(frame, output, 1);
    cv::imshow("face detection", output);
    int keycode = cv::waitKey(1);
    if (keycode == 'q') {
      break;
    }
    // if (tmp) {
    //   break;
    // }
  }
  // delete memory
  for (size_t c = 0; c < BGR.size(); ++c) {
    delete[] input_buf[c];
  }
  cv::destroyAllWindows();

  return EXIT_SUCCESS;
}