// XOR with manual gradients in C++
// Produces normalized output: loss.csv, timings.json, stdout.txt
// Build: g++ -O3 -std=c++17 -o xor xor.cpp

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <sys/resource.h>

// Simple JSON parser for init_weights.json
#include <string>
#include <map>

struct Matrix {
    std::vector<double> data;
    int rows, cols;
    
    Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0) {}
    Matrix() : rows(0), cols(0) {}
    
    double& operator()(int i, int j) { return data[i * cols + j]; }
    double operator()(int i, int j) const { return data[i * cols + j]; }
    
    Matrix T() const {
        Matrix result(cols, rows);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result(j, i) = (*this)(i, j);
        return result;
    }
};

Matrix matmul(const Matrix& A, const Matrix& B) {
    Matrix C(A.rows, B.cols);
    for (int i = 0; i < A.rows; i++)
        for (int j = 0; j < B.cols; j++)
            for (int k = 0; k < A.cols; k++)
                C(i, j) += A(i, k) * B(k, j);
    return C;
}

Matrix sigmoid(const Matrix& A) {
    Matrix result(A.rows, A.cols);
    for (size_t i = 0; i < A.data.size(); i++) {
        double x = std::max(-500.0, std::min(500.0, A.data[i]));
        result.data[i] = 1.0 / (1.0 + std::exp(-x));
    }
    return result;
}

Matrix operator-(const Matrix& A, const Matrix& B) {
    Matrix result(A.rows, A.cols);
    for (size_t i = 0; i < A.data.size(); i++)
        result.data[i] = A.data[i] - B.data[i];
    return result;
}

Matrix operator*(const Matrix& A, const Matrix& B) {
    Matrix result(A.rows, A.cols);
    for (size_t i = 0; i < A.data.size(); i++)
        result.data[i] = A.data[i] * B.data[i];
    return result;
}

Matrix operator*(const Matrix& A, double s) {
    Matrix result(A.rows, A.cols);
    for (size_t i = 0; i < A.data.size(); i++)
        result.data[i] = A.data[i] * s;
    return result;
}

double mean(const Matrix& A) {
    double sum = 0.0;
    for (double v : A.data) sum += v;
    return sum / A.data.size();
}

double compute_accuracy(const Matrix& pred, const Matrix& target) {
    int correct = 0;
    for (size_t i = 0; i < pred.data.size(); i++) {
        if ((pred.data[i] > 0.5) == (target.data[i] > 0.5)) correct++;
    }
    return (double)correct / pred.data.size();
}

struct AdamState {
    Matrix m, v;
    int t = 0;
    
    AdamState(int rows, int cols) : m(rows, cols), v(rows, cols) {}
    AdamState() {}
    
    void update(Matrix& param, const Matrix& grad, double lr, 
                double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8) {
        t++;
        for (size_t i = 0; i < param.data.size(); i++) {
            m.data[i] = beta1 * m.data[i] + (1.0 - beta1) * grad.data[i];
            v.data[i] = beta2 * v.data[i] + (1.0 - beta2) * grad.data[i] * grad.data[i];
            double m_hat = m.data[i] / (1.0 - std::pow(beta1, t));
            double v_hat = v.data[i] / (1.0 - std::pow(beta2, t));
            param.data[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
        }
    }
    
    void reset(int rows, int cols) {
        m = Matrix(rows, cols);
        v = Matrix(rows, cols);
        t = 0;
    }
};

// Minimal JSON array parser
std::vector<std::vector<double>> parse_2d_array(const std::string& content, const std::string& key) {
    std::vector<std::vector<double>> result;
    size_t pos = content.find("\"" + key + "\"");
    if (pos == std::string::npos) return result;
    
    pos = content.find('[', pos);
    if (pos == std::string::npos) return result;
    pos++; // Skip first [
    
    while (true) {
        pos = content.find('[', pos);
        if (pos == std::string::npos) break;
        
        size_t end = content.find(']', pos);
        if (end == std::string::npos) break;
        
        std::string row_str = content.substr(pos + 1, end - pos - 1);
        std::vector<double> row;
        std::stringstream ss(row_str);
        std::string token;
        while (std::getline(ss, token, ',')) {
            row.push_back(std::stod(token));
        }
        result.push_back(row);
        pos = end + 1;
        
        // Check if we hit the closing ]]
        size_t next_bracket = content.find('[', pos);
        size_t close_bracket = content.find(']', pos);
        if (close_bracket < next_bracket || next_bracket == std::string::npos) break;
    }
    return result;
}

int main(int argc, char* argv[]) {
    std::string out_dir = argc > 1 ? argv[1] : "out/runs/latest/cpp_manual";
    
    // Create output directory (simplified - assumes parent exists)
    std::string mkdir_cmd = "mkdir -p " + out_dir;
    system(mkdir_cmd.c_str());
    
    // Config (hardcoded to match config.json)
    const double lr_phase1 = 0.05;
    const double lr_phase2 = 0.12;
    const double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
    const int max_iter_phase1 = 200, max_iter_phase2 = 400;
    const int hidden_initial = 2, hidden_final = 16;
    const int growth_step = 200;
    const double threshold_final = 0.002;
    const int warmup_iters = 50;
    
    // XOR dataset
    Matrix X(4, 2), Y(4, 1);
    X(0,0)=0; X(0,1)=0; Y(0,0)=0;
    X(1,0)=0; X(1,1)=1; Y(1,0)=1;
    X(2,0)=1; X(2,1)=0; Y(2,0)=1;
    X(3,0)=1; X(3,1)=1; Y(3,0)=0;
    
    // Load init weights
    // Try to find init_weights.json relative to output dir or in common locations
    std::string init_path;
    std::vector<std::string> search_paths = {
        out_dir + "/../../../init/init_weights.json",
        "out/init/init_weights.json",
        "../out/init/init_weights.json",
        "../../out/init/init_weights.json",
        "/workspaces/Noma/demo_self_growing_xor/out/init/init_weights.json"
    };
    
    std::ifstream init_file;
    for (const auto& path : search_paths) {
        init_file.open(path);
        if (init_file.good()) {
            init_path = path;
            break;
        }
        init_file.close();
    }
    
    if (!init_file.good()) {
        std::cerr << "ERROR: Cannot find init_weights.json" << std::endl;
        return 1;
    }
    
    std::stringstream buffer;
    buffer << init_file.rdbuf();
    std::string init_content = buffer.str();
    init_file.close();
    
    auto W1_init = parse_2d_array(init_content, "W1");
    auto W2_init = parse_2d_array(init_content, "W2");
    auto W1_extra_init = parse_2d_array(init_content, "W1_extra");
    auto W2_extra_init = parse_2d_array(init_content, "W2_extra");
    
    Matrix W1(2, hidden_initial), W2(hidden_initial, 1);
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < hidden_initial; j++)
            W1(i, j) = W1_init[i][j];
    for (int i = 0; i < hidden_initial; i++)
        W2(i, 0) = W2_init[i][0];
    
    AdamState adam_W1(2, hidden_initial), adam_W2(hidden_initial, 1);
    
    auto start_total = std::chrono::high_resolution_clock::now();
    
    std::vector<std::tuple<int, double, double, int, int>> loss_records;
    std::vector<std::string> stdout_lines;
    std::vector<double> steady_times;
    
    int step = 0;
    int phase = 1;
    int hidden = hidden_initial;
    double lr = lr_phase1;
    
    auto warmup_start = start_total;
    auto warmup_end = start_total;
    auto steady_start = start_total;
    auto growth_start = start_total;
    auto growth_end = start_total;
    double time_to_threshold = -1;
    
    int max_iter = max_iter_phase1 + max_iter_phase2;
    
    while (step < max_iter) {
        auto iter_start = std::chrono::high_resolution_clock::now();
        
        // Forward
        Matrix h = sigmoid(matmul(X, W1));
        Matrix pred = sigmoid(matmul(h, W2));
        Matrix err = pred - Y;
        double loss = mean(err * err);
        double acc = compute_accuracy(pred, Y);
        
        loss_records.push_back({step, loss, acc, hidden, phase});
        
        std::ostringstream line;
        line << "step=" << step << " loss=" << std::fixed << std::setprecision(6) << loss 
             << " acc=" << std::setprecision(2) << acc << " hidden=" << hidden;
        stdout_lines.push_back(line.str());
        
        if (loss < threshold_final && time_to_threshold < 0) {
            auto now = std::chrono::high_resolution_clock::now();
            time_to_threshold = std::chrono::duration<double, std::milli>(now - start_total).count();
        }
        
        // Growth check
        if (phase == 1 && step >= growth_step - 1) {
            growth_start = std::chrono::high_resolution_clock::now();
            
            std::ostringstream growth_line;
            growth_line << "GROWTH TRIGGERED: hidden " << hidden_initial << " -> " << hidden_final;
            stdout_lines.push_back(growth_line.str());
            
            // Extend weights
            Matrix W1_new(2, hidden_final), W2_new(hidden_final, 1);
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < hidden_initial; j++)
                    W1_new(i, j) = W1(i, j);
                for (int j = 0; j < hidden_final - hidden_initial; j++)
                    W1_new(i, hidden_initial + j) = W1_extra_init[i][j];
            }
            for (int i = 0; i < hidden_initial; i++)
                W2_new(i, 0) = W2(i, 0);
            for (int i = 0; i < hidden_final - hidden_initial; i++)
                W2_new(hidden_initial + i, 0) = W2_extra_init[i][0];
            
            W1 = W1_new;
            W2 = W2_new;
            adam_W1.reset(2, hidden_final);
            adam_W2.reset(hidden_final, 1);
            
            hidden = hidden_final;
            phase = 2;
            lr = lr_phase2;
            growth_end = std::chrono::high_resolution_clock::now();
            step++;
            continue;
        }
        
        if (phase == 2 && loss < threshold_final) break;
        
        // Backward
        Matrix dloss = err * (2.0 / X.rows);
        Matrix dout(pred.rows, pred.cols);
        for (size_t i = 0; i < pred.data.size(); i++)
            dout.data[i] = dloss.data[i] * pred.data[i] * (1.0 - pred.data[i]);
        Matrix dW2 = matmul(h.T(), dout);
        
        Matrix dh = matmul(dout, W2.T());
        Matrix dh_act(h.rows, h.cols);
        for (size_t i = 0; i < h.data.size(); i++)
            dh_act.data[i] = dh.data[i] * h.data[i] * (1.0 - h.data[i]);
        Matrix dW1 = matmul(X.T(), dh_act);
        
        adam_W1.update(W1, dW1, lr, beta1, beta2, eps);
        adam_W2.update(W2, dW2, lr, beta1, beta2, eps);
        
        auto iter_end = std::chrono::high_resolution_clock::now();
        
        if (step == 0) warmup_start = iter_start;
        if (step == warmup_iters - 1) {
            warmup_end = iter_end;
            steady_start = iter_end;
        }
        if (step >= warmup_iters) {
            double iter_us = std::chrono::duration<double, std::micro>(iter_end - iter_start).count();
            steady_times.push_back(iter_us);
        }
        
        step++;
    }
    
    auto end_total = std::chrono::high_resolution_clock::now();
    
    double total_ms = std::chrono::duration<double, std::milli>(end_total - start_total).count();
    double warmup_ms = std::chrono::duration<double, std::milli>(warmup_end - warmup_start).count();
    double steady_ms = std::chrono::duration<double, std::milli>(end_total - steady_start).count();
    double growth_ms = std::chrono::duration<double, std::milli>(growth_end - growth_start).count();
    
    // Compute median and p95
    std::sort(steady_times.begin(), steady_times.end());
    double steady_median = steady_times.empty() ? 0 : steady_times[steady_times.size() / 2];
    double steady_p95 = steady_times.empty() ? 0 : steady_times[(size_t)(steady_times.size() * 0.95)];
    
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    long peak_rss = usage.ru_maxrss;
    
    // Write loss.csv
    std::ofstream csv(out_dir + "/loss.csv");
    csv << "step,loss,accuracy,hidden,phase\n";
    for (const auto& [s, l, a, h, p] : loss_records) {
        csv << s << "," << std::fixed << std::setprecision(6) << l << ","
            << std::setprecision(2) << a << "," << h << "," << p << "\n";
    }
    csv.close();
    
    // Write timings.json
    std::ofstream timings(out_dir + "/timings.json");
    timings << "{\n";
    timings << "  \"cold_start_ms\": 0,\n";
    timings << "  \"compile_overhead_ms\": 0,\n";
    timings << "  \"train_warmup_ms\": " << warmup_ms << ",\n";
    timings << "  \"train_steady_ms\": " << steady_ms << ",\n";
    timings << "  \"steady_step_us_median\": " << steady_median << ",\n";
    timings << "  \"steady_step_us_p95\": " << steady_p95 << ",\n";
    timings << "  \"growth_event_ms\": " << growth_ms << ",\n";
    timings << "  \"total_ms\": " << total_ms << ",\n";
    timings << "  \"iters_total\": " << step << ",\n";
    timings << "  \"time_to_threshold_ms\": " << (time_to_threshold < 0 ? "null" : std::to_string(time_to_threshold)) << ",\n";
    timings << "  \"final_loss\": " << std::get<1>(loss_records.back()) << ",\n";
    timings << "  \"final_accuracy\": " << std::get<2>(loss_records.back()) << ",\n";
    timings << "  \"peak_rss_kb\": " << peak_rss << ",\n";
    timings << "  \"precision\": \"f64\",\n";
    timings << "  \"device\": \"cpu\",\n";
    timings << "  \"impl\": \"cpp_manual\"\n";
    timings << "}\n";
    timings.close();
    
    // Write stdout.txt
    std::ofstream stdout_file(out_dir + "/stdout.txt");
    for (const auto& line : stdout_lines) stdout_file << line << "\n";
    stdout_file.close();
    
    std::cout << "[cpp_manual] " << step << " iters, final_loss=" << std::fixed 
              << std::setprecision(6) << std::get<1>(loss_records.back()) 
              << ", total=" << std::setprecision(2) << total_ms << "ms\n";
    
    return 0;
}
