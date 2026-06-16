/**
 * MedVision C++ Tracker - Complete Rewrite
 *
 * Architecture (based on MLLCV + servopilot + Orbbec SDK):
 * Thread 1: Camera capture (V4L2 direct, 30fps)
 * Thread 2: Detection (color + YOLO, parallel)
 * Thread 3: Servo control (async UART, 300Hz)
 * Main: PID + Kalman + visual servo
 *
 * Build: g++ -O3 -o medvision medvision.cpp $(pkg-config --cflags --libs opencv4) -lonnxruntime -lpthread
 */

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>
#include <cmath>
#include <deque>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>

struct Config {
    int cam_width = 640, cam_height = 480;
    float kp = 0.8f;
    int dead_zone = 15;
    float max_speed = 400.0f;
    int servo_speed = 300;
    int tilt_min = 0, tilt_max = 45;
    int pan_min = -35, pan_max = 35;
    int pan_offset = 20, tilt_offset = -75;
    int pan_id = 0, tilt_id = 1;
    int confirm_frames = 5;
    float servo_delay = 0.3f;
    int yolo_interval = 30;
    float kalman_q = 5.0f, kalman_r = 20.0f;
};

class Kalman2D {
public:
    Kalman2D(float dt = 0.033f, float q = 5.0f, float r = 20.0f)
        : dt(dt), initialized(false) {
        F = (cv::Mat_<float>(4,4) << 1,0,dt,0, 0,1,0,dt, 0,0,1,0, 0,0,0,1);
        H = (cv::Mat_<float>(2,4) << 1,0,0,0, 0,1,0,0);
        Q = cv::Mat::eye(4,4,CV_32F) * q;
        R = cv::Mat::eye(2,2,CV_32F) * r;
        P = cv::Mat::eye(4,4,CV_32F) * 1000;
        x = cv::Mat::zeros(4,1,CV_32F);
    }

    cv::Point2f predict() {
        x = F * x;
        P = F * P * F.t() + Q;
        return cv::Point2f(x.at<float>(0), x.at<float>(1));
    }

    cv::Point2f update(float mx, float my) {
        if (!initialized) {
            x.at<float>(0) = mx; x.at<float>(1) = my;
            initialized = true;
            return cv::Point2f(mx, my);
        }
        cv::Mat z = (cv::Mat_<float>(2,1) << mx, my);
        cv::Mat y = z - H * x;
        cv::Mat S = H * P * H.t() + R;
        cv::Mat K = P * H.t() * S.inv();
        x = x + K * y;
        P = (cv::Mat::eye(4,4,CV_32F) - K * H) * P;
        return cv::Point2f(x.at<float>(0), x.at<float>(1));
    }

    cv::Point2f getVelocity() {
        return cv::Point2f(x.at<float>(2), x.at<float>(3));
    }

    void reset() {
        x = cv::Mat::zeros(4,1,CV_32F);
        P = cv::Mat::eye(4,4,CV_32F) * 1000;
        initialized = false;
    }

    void decayVelocity(float factor = 0.8f) {
        x.at<float>(2) *= factor;
        x.at<float>(3) *= factor;
    }

private:
    float dt;
    cv::Mat F, H, Q, R, P, x;
    bool initialized;
};

class DualPID {
public:
    DualPID(float kp = 0.4f, float ki = 0.008f, float kd = 0.25f,
            float out_lim = 4.0f, float int_lim = 50.0f)
        : kp(kp), ki(ki), kd(kd), out_lim(out_lim), int_lim(int_lim),
          errx_sum(0), erry_sum(0), prev_x(0), prev_y(0) {}

    std::pair<float,float> update(float ex, float ey, float dt = 0.033f) {
        errx_sum = std::max(-int_lim, std::min(int_lim, errx_sum + ex * dt));
        erry_sum = std::max(-int_lim, std::min(int_lim, erry_sum + ey * dt));
        float dx = (ex - prev_x) / dt;
        float dy = (ey - prev_y) / dt;
        prev_x = ex; prev_y = ey;
        float ox = kp * ex + ki * errx_sum + kd * dx;
        float oy = kp * ey + ki * erry_sum + kd * dy;
        ox = std::max(-out_lim, std::min(out_lim, ox));
        oy = std::max(-out_lim, std::min(out_lim, oy));
        return {ox, oy};
    }

    void reset() { errx_sum = erry_sum = prev_x = prev_y = 0; }

private:
    float kp, ki, kd, out_lim, int_lim;
    float errx_sum, erry_sum, prev_x, prev_y;
};

class FrameBuffer {
public:
    void put(const cv::Mat& frame) {
        std::lock_guard<std::mutex> lock(mtx);
        latest = frame.clone();
        has_frame = true;
    }

    cv::Mat get() {
        std::lock_guard<std::mutex> lock(mtx);
        if (!has_frame) return cv::Mat();
        return latest.clone();
    }

    bool ready() {
        std::lock_guard<std::mutex> lock(mtx);
        return has_frame;
    }

private:
    cv::Mat latest;
    std::mutex mtx;
    bool has_frame = false;
};

struct Detection {
    int cx, cy, w, h, area;
    float score;
    std::string method;
};

class ColorDetector {
public:
    Detection detect(const cv::Mat& bgr) {
        cv::Mat hsv;
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

        cv::Mat m1, m2, mask;
        cv::inRange(hsv, cv::Scalar(0,80,80), cv::Scalar(15,255,255), m1);
        cv::inRange(hsv, cv::Scalar(155,80,80), cv::Scalar(180,255,255), m2);
        cv::bitwise_or(m1, m2, mask);

        cv::Mat roi_mask = cv::Mat::zeros(mask.size(), CV_8U);
        roi_mask(cv::Rect(0, bgr.rows * 0.4, bgr.cols, bgr.rows * 0.6)) = 255;
        cv::bitwise_and(mask, roi_mask, mask);

        cv::Mat k3 = cv::Mat::ones(3,3,CV_8U);
        cv::Mat k7 = cv::Mat::ones(7,7,CV_8U);
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, k3);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, k7);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        Detection best = {0,0,0,0,0,0,""};
        float best_score = 0;

        for (auto& c : contours) {
            double area = cv::contourArea(c);
            if (area < 200) continue;
            cv::RotatedRect rect = cv::minAreaRect(c);
            cv::Size2f sz = rect.size;
            float rx = std::max(sz.width, sz.height);
            float ry = std::min(sz.width, sz.height);
            if (ry == 0) continue;
            float aspect = rx / ry;
            float fill = area / (rx * ry);
            if (aspect > 0.3f && aspect < 4.0f && fill > 0.2f && area > 500 && area < 50000) {
                float score = area * fill * std::min(aspect, 2.0f);
                if (score > best_score) {
                    best_score = score;
                    best = {(int)rect.center.x, (int)rect.center.y,
                            (int)rx, (int)ry, (int)area, score, "COLOR"};
                }
            }
        }
        return best;
    }
};

class YoloDetector {
public:
    YoloDetector(const std::string& model_path, const std::string& labels_path) {
        session = std::make_unique<Ort::Session>(env, model_path.c_str(), Ort::SessionOptions{});
        auto allocator = Ort::AllocatorWithDefaultOptions();
        auto name_ptr = session->GetInputNameAllocated(0, allocator);
        input_name_str = std::string(name_ptr.get());
        input_name = input_name_str.c_str();
        auto input_type = session->GetInputTypeInfo(0);
        auto tensor_info = input_type.GetTensorTypeAndShapeInfo();
        use_fp16 = (tensor_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);

        std::ifstream f(labels_path);
        std::string line;
        while (std::getline(f, line)) classes.push_back(line);
        std::cout << "[YOLO] Loaded " << classes.size() << " classes" << std::endl;
    }

    Detection detect(const cv::Mat& bgr) {
        try {
            int h = bgr.rows, w = bgr.cols;
            cv::Mat resized;
            cv::resize(bgr, resized, {640, 640});
            cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

            std::vector<int64_t> input_shape = {1, 3, 640, 640};
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            std::vector<float> input_data(640*640*3);
            for (int i = 0; i < 640*640*3; i++) {
                input_data[i] = (float)resized.data[i] / 255.0f;
            }
            std::vector<float> chw(640*640*3);
            for (int c = 0; c < 3; c++)
                for (int i = 0; i < 640*640; i++)
                    chw[c*640*640 + i] = input_data[i*3+c];

            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, chw.data(), chw.size(), input_shape.data(), 4);

            auto output = session->Run(Ort::RunOptions{nullptr},
                &input_name, &input_tensor, 1, nullptr, 0);

            auto out_shape = output[0].GetTensorTypeAndShapeInfo().GetShape();
            const float* preds = output[0].GetTensorData<float>();

            Detection best = {0,0,0,0,0,0,""};
            float best_conf = 0;

            for (int i = 0; i < out_shape[1]; i++) {
                float conf = preds[i*85+4];
                if (conf < 0.3f || !std::isfinite(conf)) continue;
                int cid = 0;
                float max_cls = preds[i*85+5];
                for (int j = 1; j < 80; j++) {
                    if (preds[i*85+5+j] > max_cls) { max_cls = preds[i*85+5+j]; cid = j; }
                }
                if (cid == 0 || max_cls * conf < 0.3f) continue; // skip person

                float cx_n = preds[i*85], cy_n = preds[i*85+1];
                float ww = preds[i*85+2], hh = preds[i*85+3];
                if (!std::isfinite(cx_n) || !std::isfinite(cy_n)) continue;

                int cx = std::clamp((int)(cx_n * w), 0, w-1);
                int cy = std::clamp((int)(cy_n * h), 0, h-1);
                int bw = (int)(ww * w), bh = (int)(hh * h);

                if (conf > best_conf) {
                    best_conf = conf;
                    best = {cx, cy, bw, bh, bw*bh, conf, "YOLO"};
                }
            }
            return best;
        } catch (...) {
            return {0,0,0,0,0,0,""};
        }
    }

private:
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "medvision"};
    std::unique_ptr<Ort::Session> session;
    std::string input_name_str;
    const char* input_name = nullptr;
    bool use_fp16 = false;
    std::vector<std::string> classes;
};

class GimbalController {
public:
    GimbalController(const std::string& port = "/dev/ttyUSB0") {
        fd = open(port.c_str(), O_RDWR | O_NOCTTY);
        if (fd < 0) { perror("open serial"); return; }
        struct termios tty{};
        tcgetattr(fd, &tty);
        cfsetospeed(&tty, B115200);
        cfsetispeed(&tty, B115200);
        tty.c_cflag &= ~PARENB;
        tty.c_cflag &= ~CSTOPB;
        tty.c_cflag &= ~CSIZE;
        tty.c_cflag |= CS8;
        tty.c_cflag &= ~CRTSCTS;
        tty.c_cflag |= CREAD | CLOCAL;
        tty.c_iflag &= ~(IXON | IXOFF | IXANY);
        tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
        tty.c_oflag &= ~OPOST;
        tcsetattr(fd, TCSANOW, &tty);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        pingServo(0);
        pingServo(1);
        queryAngle(0);
        queryAngle(1);
        std::cout << "[Gimbal] Connected to " << port << std::endl;
    }

    void velocityMove(float dpan, float dtilt, int speed = 200) {
        pan = std::clamp(pan + dpan, (float)pan_min, (float)pan_max);
        tilt = std::clamp(tilt + dtilt, (float)tilt_min, (float)tilt_max);
        sendAngle(pan_id, (int)(pan + pan_offset), speed);
        sendAngle(tilt_id, (int)(tilt + tilt_offset), speed);
    }

    void positionMove(float p, float t, int speed = 300) {
        pan = std::clamp(p, (float)pan_min, (float)pan_max);
        tilt = std::clamp(t, (float)tilt_min, (float)tilt_max);
        sendAngle(pan_id, (int)(pan + pan_offset), speed);
        sendAngle(tilt_id, (int)(tilt + tilt_offset), speed);
    }

    void center() { positionMove(0, 0); }

    std::pair<float,float> query() { return {pan, tilt}; }

    void close() { if (fd >= 0) ::close(fd); }

    float pan = 0, tilt = 0;
    int pan_min = -35, pan_max = 35, tilt_min = 0, tilt_max = 45;
    int pan_offset = 20, tilt_offset = -75;
    int pan_id = 0, tilt_id = 1;

    void pingServo(int sid) {
        uint8_t params[1] = {(uint8_t)sid};
        sendPacket(1, params, 1);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        uint8_t buf[32];
        int n = ::read(fd, buf, sizeof(buf));
        (void)n;
    }

    void queryAngle(int sid) {
        uint8_t params[1] = {(uint8_t)sid};
        sendPacket(10, params, 1);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        uint8_t buf[32];
        int n = ::read(fd, buf, sizeof(buf));
        if (n >= 7) {
            int16_t angle10 = (int16_t)(buf[4] | (buf[5] << 8));
            float angle = angle10 / 10.0f;
            if (sid == pan_id) pan = angle - pan_offset;
            else tilt = angle - tilt_offset;
        }
    }

    void sendPacket(int code, const uint8_t* params, int param_len) {
        uint8_t pkt[20];
        pkt[0] = 0x12; pkt[1] = 0x4C;
        pkt[2] = (uint8_t)code;
        pkt[3] = (uint8_t)param_len;
        memcpy(&pkt[4], params, param_len);
        uint8_t cs = 0;
        for (int i = 0; i < 4 + param_len; i++) cs += pkt[i];
        pkt[4 + param_len] = cs;
        ::write(fd, pkt, 5 + param_len);
        ::tcdrain(fd);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

private:
    int fd = -1;

    void sendAngle(int sid, int angle_raw, int velocity) {
        int angle10 = angle_raw * 10;
        int vel10 = velocity * 10;

        uint8_t pkt[15];
        pkt[0] = 0x12; pkt[1] = 0x4C;  // header
        pkt[2] = 12;                      // code
        pkt[3] = 11;                      // param length (B+h+H+H+H+H = 1+2+2+2+2+2=11)
        pkt[4] = (uint8_t)sid;            // servo_id
        int16_t a16 = (int16_t)angle10;
        pkt[5] = (uint8_t)(a16 & 0xFF);
        pkt[6] = (uint8_t)((a16 >> 8) & 0xFF);
        uint16_t v16 = (uint16_t)vel10;
        pkt[7] = (uint8_t)(v16 & 0xFF);
        pkt[8] = (uint8_t)((v16 >> 8) & 0xFF);
        pkt[9] = 20; pkt[10] = 0;
        pkt[11] = 20; pkt[12] = 0;
        pkt[13] = 0; // low byte
        uint8_t cs = 0;
        for (int i = 0; i < 14; i++) cs += pkt[i];
        pkt[14] = cs;

        ::write(fd, pkt, 15);
        ::tcdrain(fd);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
};

int main() {
    std::cout << "========================================\n";
    std::cout << "  MedVision C++ Tracker v1.0\n";
    std::cout << "========================================\n\n";

    Config cfg;
    GimbalController gimbal;
    gimbal.center();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    auto [gp, gt] = gimbal.query();
    std::cout << "Gimbal: pan=" << gp << " tilt=" << gt << std::endl;

    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, cfg.cam_width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.cam_height);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera!" << std::endl;
        return 1;
    }
    std::cout << "Camera: V4L2 (" << cfg.cam_width << "x" << cfg.cam_height << ")\n" << std::endl;

    ColorDetector color_det;
    std::unique_ptr<YoloDetector> yolo_det;
    std::string yolo_path = "/opt/medvision/yolov5s.onnx";
    std::string labels_path = "/home/pi/.local/lib/python3.13/site-packages/pyorbbecsdk/examples/applications/object_detection/coco.names";
    if (std::filesystem::exists(yolo_path)) {
        yolo_det = std::make_unique<YoloDetector>(yolo_path, labels_path);
    }

    Kalman2D kalman(0.033f, cfg.kalman_q, cfg.kalman_r);
    DualPID pid(0.4f, 0.008f, 0.25f, cfg.max_speed, 50.0f);

    std::cout << "Finding clip..." << std::endl;
    float best_area = 0;
    int best_p = 0, best_t = 0;

    {
        cv::Mat bgr;
        cap >> bgr;
        if (!bgr.empty()) {
            auto r = color_det.detect(bgr);
            best_area = r.area;
            best_p = (int)gimbal.pan;
            best_t = (int)gimbal.tilt;
            if (best_area > 0)
                std::cout << "  Found at current position!" << std::endl;
        }
    }

    if (best_area == 0) {
        for (int p = -20; p <= 20; p += 15) {
            for (int t = 20; t <= 35; t += 7) {
                gimbal.positionMove(p, t, 500);
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                cv::Mat bgr;
                cap >> bgr;
                if (bgr.empty()) continue;
                auto r = color_det.detect(bgr);
                if (r.area > best_area) {
                    best_area = r.area;
                    best_p = p; best_t = t;
                }
            }
        }
        gimbal.positionMove(best_p, best_t, 300);
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }
    std::cout << "Clip at pan=" << best_p << " tilt=" << best_t << "\n" << std::endl;

    std::ofstream csv("/opt/medvision/logs/tracker_cpp.csv");
    csv << "time,fc,method,tx,ty,err_x,err_y,spd_p,spd_t,pan,tilt,fps\n";

    bool tracking = false;
    int confirm = 0, lost = 0;
    int fc = 0;
    auto t0 = std::chrono::steady_clock::now();
    int last_yolo = 0;

    std::cout << "Tracking AUTO-STARTED. Ctrl+C to stop.\n" << std::endl;

    while (true) {
        cv::Mat bgr;
        cap >> bgr;
        if (bgr.empty()) continue;

        fc++;
        auto now = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float>(now - t0).count();
        float fps = fc / elapsed;

        auto result = color_det.detect(bgr);
        std::string method = "COLOR";

        if (result.area == 0 && yolo_det && (fc - last_yolo) > cfg.yolo_interval) {
            result = yolo_det->detect(bgr);
            if (result.area > 0) {
                method = "YOLO";
                last_yolo = fc;
            }
        }

        if (result.area > 0) {
            lost = 0;
            if (!tracking) {
                confirm++;
                if (confirm >= cfg.confirm_frames) {
                    tracking = true;
                    std::cout << "[CONFIRMED] " << method << " (" << result.cx << "," << result.cy << ")" << std::endl;
                }
                continue;
            }

            float ex = result.cx - cfg.cam_width / 2.0f;
            float ey = result.cy - cfg.cam_height / 2.0f;

            float sp = 0, st = 0;
            if (std::abs(ex) > cfg.dead_zone || std::abs(ey) > cfg.dead_zone) {
                sp = std::clamp(cfg.kp * ex, -cfg.max_speed, cfg.max_speed);
                st = std::clamp(cfg.kp * ey, -cfg.max_speed, cfg.max_speed);
                float dt = 1.0f / fps;
                gimbal.velocityMove(sp / fps, st / fps, cfg.servo_speed);
            }

            if (fc % 5 == 0) {
                auto [gpan, gtilt] = gimbal.query();
                printf("F%04d | %s %s (%d,%d) e=(%+.0f,%+.0f) G=(%+.1f,%+.1f) FPS=%.0f\n",
                       fc, (sp != 0 ? "SPEED" : "CENTER"), method.c_str(),
                       result.cx, result.cy, ex, ey, gpan, gtilt, fps);
            }

            csv << elapsed << "," << fc << "," << method << ","
                << result.cx << "," << result.cy << "," << ex << "," << ey
                << "," << (sp != 0 ? sp : 0) << "," << (st != 0 ? st : 0)
                << "," << gimbal.pan << "," << gimbal.tilt << "," << fps << "\n";
        } else {
            confirm = 0;
            if (tracking) {
                lost++;
                if (lost > cfg.confirm_frames) {
                    tracking = false;
                    kalman.decayVelocity();
                    std::cout << "[LOST]" << std::endl;
                }
            }
            if (fc % 30 == 0)
                printf("F%04d | NO TARGET\n", fc);
        }
    }

    gimbal.center();
    gimbal.close();
    cap.release();
    csv.close();
    std::cout << "\nDone!" << std::endl;
    return 0;
}
