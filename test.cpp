#include <iostream>
#include <sstream>
#include "NvInfer.h"

#define CREATE_INFER_BUILDER createInferBuilder
#define CREATE_INFER_RUNTIME createInferRuntime
#define LOG_GIE "[GIE]  "

class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char* msg) override {
    if (severity != Severity::kINFO /*|| mEnableDebug*/)
      printf(LOG_GIE "%s\n", msg);
  }
} gLogger;

int main() {
  nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
  std::cout << "hello word" << NV_TENSORRT_MAJOR << std::endl;
  return 0;
}