#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <immintrin.h> 
#include <chrono> 


class SimpleMCPricer {
private:
    double variance;
    double root_Variance;
    double itoCorr;
    double movedSpot;
    double runningSum;
    double mean;
    int paths;

public:
    SimpleMCPricer(double expiry, double strike, double spot, double vol, double r, int paths)
            : paths(paths) {
        variance = vol * vol * expiry;
        root_Variance = std::sqrt(variance);
        itoCorr = -0.5 * variance;
        movedSpot = spot * std::exp(r * expiry + itoCorr);
        runningSum = 0;

        // Generate Gaussian random numbers
        std::vector<double> gaussians(paths);
        generateGaussian(gaussians);

        // SIMD processing with unrolling. Shaved a tiny amount of time away
        __m256d sum_vec = _mm256_setzero_pd();
        for (int i {0}; i < paths; i += 8) { // Unroll the loop by processing 8 elements per iteration
            __m256d gauss1 = _mm256_loadu_pd(&gaussians[i]);
            __m256d gauss2 = _mm256_loadu_pd(&gaussians[i + 4]);
            __m256d spotPrices1 = _mm256_mul_pd(_mm256_set1_pd(movedSpot), exp_ps(_mm256_mul_pd(_mm256_set1_pd(root_Variance), gauss1)));
            __m256d spotPrices2 = _mm256_mul_pd(_mm256_set1_pd(movedSpot), exp_ps(_mm256_mul_pd(_mm256_set1_pd(root_Variance), gauss2)));
            __m256d strikePrices = _mm256_set1_pd(strike);
            __m256d payoffs1 = _mm256_sub_pd(spotPrices1, strikePrices);
            __m256d payoffs2 = _mm256_sub_pd(spotPrices2, strikePrices);
            __m256d zero = _mm256_setzero_pd();
            __m256d result1 = _mm256_max_pd(payoffs1, zero);
            __m256d result2 = _mm256_max_pd(payoffs2, zero);
            sum_vec = _mm256_add_pd(sum_vec, result1);
            sum_vec = _mm256_add_pd(sum_vec, result2);
        }
        runningSum = _mm256_reduce_add_pd(sum_vec);

        mean = runningSum / paths;
        mean *= std::exp(-r * expiry);
    }

    double getMean() const {
        return round(mean * 100) / 100.0;
    }

private:
    static void generateGaussian(std::vector<double>& gaussians) {
        static std::random_device rd;
        static std::mt19937 gen(rd()); //mersenne
        static std::normal_distribution<double> dist(0, 1);
        std::generate(gaussians.begin(), gaussians.end(), [&]() { return dist(gen); });
    }

    // Fast exp approximation function for SIMD (slowest part without SIMD)
    // https://github.com/gnuradio/volk/blob/39f04949a91ea4f80565e9dda6ee5fc08e6c700b/kernels/volk/volk_32f_exp_32f.h#L110
    // https://github.com/usi-dag/intel-vectorized-benchmark-suite/blob/283e87bd913b15bf36587e134dd4df9b45c085f6/common/size_256/__exp.h#L46
    // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
    // (arguments from here, I probably could have found a more efficient way of benchmark but just bruteforced going through these gits and stuck with the most performative ones

    // Best video I've ever watched on SIMD.
    // https://www.youtube.com/watch?v=x9Scb5Mku1g
    // TO:DO: Buy C++ concurrency in action - Anthony Williams (after interview prep done) to fully understand/figure out how to use intrinsics in other projects
    static __m256d exp_ps(__m256d x) {
        x = _mm256_min_pd(x, _mm256_set1_pd(88.3762626647949)); // Clamp the values to the maximum input for exp (VOLT)
        x = _mm256_max_pd(x, _mm256_set1_pd(-88.3762626647949)); // Clamp the values to the minimum input for exp (VOlT)

        __m256d fx = _mm256_fmadd_pd(x, _mm256_set1_pd(1.44269504088896341), _mm256_set1_pd(0.5)); // Approximate log2(e) * x + 0.5 (VOLK)
        __m128i emm0 = _mm256_cvttpd_epi32(fx); // Convert to integer for exponent
        fx = _mm256_cvtepi32_pd(emm0); // Convert back to double

        __m256d tmp = _mm256_fmsub_pd(fx, _mm256_set1_pd(0.693359375), x); // Subtract log2(e) * 0.693359375 (VOLK)
        x = _mm256_fmsub_pd(fx, _mm256_set1_pd(-2.12194440e-4), tmp); // Subtract small correction factor

        __m256d z = _mm256_fmadd_pd(_mm256_set1_pd(1.9875691500e-4), x, _mm256_set1_pd(1.3981999507e-3)); // Polynomial approximation (VOLK)
        z = _mm256_fmadd_pd(z, x, _mm256_set1_pd(1.0)); // Continue polynomial approximation with x

        __m256i emm0_256 = _mm256_cvtepi32_epi64(emm0); // Convert back to integer for exponent
        __m256i pow2n = _mm256_slli_epi64(_mm256_add_epi32(emm0_256, _mm256_set1_epi32(1023)), 52); // Adjust exponent (VOLK)
        return _mm256_mul_pd(z, _mm256_castsi256_pd(pow2n)); // Multiply by power of 2
    }


    static double _mm256_reduce_add_pd(__m256d x) {
        __m256d hi = _mm256_permute2f128_pd(x, x, 0x1);
        x = _mm256_add_pd(x, hi);
        __m128d lo = _mm_add_pd(_mm256_castpd256_pd128(x), _mm256_extractf128_pd(x, 1));
        lo = _mm_hadd_pd(lo, lo);
        return _mm_cvtsd_f64(lo);
    }
};

template<typename T>
void testPerformance(T func, const std::string& testName) {
    double expiry {1.0};
    double strike {100};
    double spot {100};
    double vol {0.2};
    double r {0.05};
    int paths {1000000};

    auto start = std::chrono::high_resolution_clock::now();
    func(expiry, strike, spot, vol, r, paths);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << testName << " - Time taken: " << elapsed.count() << " seconds" << std::endl;
}

int main() {
    auto testFunc = [](double expiry, double strike, double spot, double vol, double r, int paths) {
        SimpleMCPricer pricer(expiry, strike, spot, vol, r, paths);
        double mean = pricer.getMean();
        std::cout << "Monte Carlo Mean Price: " << mean << std::endl;
    };

    testPerformance(testFunc, "Monte Carlo Options Pricer");

    return 0;
}
