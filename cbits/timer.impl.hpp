
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

#include "json.hpp"
#include "timer.h"

using json = nlohmann::json;

using shapes_t = std::vector<std::vector<int>>;

using timestamp_t = std::chrono::time_point<std::chrono::system_clock>;

static timestamp_t now() { return std::chrono::system_clock::now(); }

static double elapsed_time(timestamp_t start, timestamp_t end) {
  const auto elapsed =
      std::chrono::duration<double, std::milli>(end - start).count();
  return elapsed;
}

static uint64_t to_nanoseconds(timestamp_t t) {
  const auto duration = t.time_since_epoch();
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count());
}

struct profile_entry {
  profile_entry(int layer_sequence_index, std::string name,
                std::string metadata, shapes_t shapes)
      : layer_sequence_index_(layer_sequence_index),
        name_(name),
        metadata_(metadata),
        shapes_(shapes) {
    start();
  }
  ~profile_entry() {}

  error_t start() {
    start_ = now();
    return success;
  }

  error_t end() {
    end_ = now();
    return success;
  }

  json to_json() {
    const auto start_ns = to_nanoseconds(start_);
    const auto end_ns = to_nanoseconds(end_);
    uint64_t id = std::hash<std::thread::id>()(std::this_thread::get_id());
    return json{
        {"name", name_},
        {"metadata", metadata_},
        {"start", start_ns},
        {"end", end_ns},
        {"layer_sequence_index", layer_sequence_index_},
        {"shapes", shapes_},
        {"thread_id", id},
    };
  }

  void dump() {
    const auto j = this->to_json();
    std::cout << j.dump(2) << "\n";
  }

 private:
  int layer_sequence_index_{0};
  std::string name_{""};
  std::string metadata_{""};
  shapes_t shapes_{};
  timestamp_t start_{}, end_{};
};

struct profile {
  profile(std::string name = "", std::string metadata = "")
      : name_(name), metadata_(metadata) {
    start();
  }
  ~profile() { this->reset(); }

  error_t start() {
    start_ = now();
    return success;
  }

  error_t end() {
    end_ = now();
    return success;
  }

  error_t reset() {
    std::lock_guard<std::mutex> lock(mut_);
    for (auto e : entries_) {
      if (e != nullptr) {
        delete e;
      }
    }
    entries_.clear();
    return success;
  }

  error_t add(int ii, profile_entry *entry) {
    std::lock_guard<std::mutex> lock(mut_);
    if (ii >= entries_.size()) {
      entries_.reserve(2 * ii);
    }
    auto begin = entries_.begin();
    if (ii > entries_.size() + 1) {
      for (int jj = entries_.size(); jj < ii; jj++) {
        entries_.insert(begin + jj, nullptr);
      }
    }
    entries_.insert(begin + ii, entry);
    return success;
  }

  profile_entry *get(int ii) {
    std::lock_guard<std::mutex> lock(mut_);
    if (ii >= entries_.size()) {
      return nullptr;
    }
    return entries_[ii];
  }

  json to_json() {
    std::lock_guard<std::mutex> lock(mut_);

    const auto start_ns = to_nanoseconds(start_);
    const auto end_ns = to_nanoseconds(end_);

    json elements = json::array();
    for (const auto e : entries_) {
      if (e == nullptr) {
        continue;
      }
      elements.emplace_back(e->to_json());
    }
    return json{
        {"name", name_}, {"metadata", metadata_}, {"start", start_ns},
        {"end", end_ns}, {"elements", elements},
    };
  }

  void dump() {
    const auto j = this->to_json();
    std::cout << j.dump(2) << "\n";
  }

  std::string read() {
    const auto j = this->to_json();
    return j.dump();
  }

 private:
  std::string name_{""};
  std::string metadata_{""};
  std::vector<profile_entry *> entries_{};
  timestamp_t start_{}, end_{};
  std::mutex mut_;
};
