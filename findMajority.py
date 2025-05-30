import time
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple
import sys

# Increase recursion limit for large datasets
sys.setrecursionlimit(10000)

def generate_array(alg_name: str, scenario: str, size: int,
                   domain_size: int = None,
                   p_majority: float = 0.5) -> List[int]:
    """
    Unified test-case generator for any algorithm and scenario:
      - scenario in {"best", "worst", "average"}
      - alg_name matters only for best/worst patterns where needed
    """
    if domain_size is None:
        domain_size = size * 5

    if scenario == "best":
        if alg_name == "Brute Force":
            return [1] * size
        if alg_name == "Insertion Sort":
            return list(range(1, size + 1))
        if alg_name in ["Merge Sort", "Quick Sort"]:
            return random.sample(range(size * 2), size)
        if alg_name == "Divide and Conquer":
            return [1] * size
        if alg_name == "Hash-based":
            return [1] * size
        if alg_name == "Boyer-Moore":
            maj = 1
            arr = [maj] * (size // 2 + 1) + list(range(2, size + 1))
            random.shuffle(arr)
            return arr
        if alg_name == "Random Selection":
            return [1] * size

    if scenario == "worst":
        if alg_name == "Brute Force":
            return list(range(1, size + 1))
        if alg_name == "Insertion Sort":
            return list(range(size, 0, -1))
        if alg_name == "Merge Sort":
            return random.sample(range(size * 2), size)
        if alg_name == "Quick Sort":
            return list(range(1, size + 1))
        if alg_name == "Divide and Conquer":
            return [(i % 2) + 1 for i in range(size)]
        if alg_name == "Hash-based":
            return list(range(1, size + 1))
        if alg_name == "Boyer-Moore":
            return [(i % 2) + 1 for i in range(size)]
        if alg_name == "Random Selection":
            return list(range(1, size + 1))

    # average case
    if random.random() < p_majority:
        maj = random.randint(1, domain_size)
        maj_cnt = random.randint(size // 2 + 1, size)
        arr = [maj] * maj_cnt
        arr += random.choices(
            [x for x in range(1, domain_size + 1) if x != maj],
            k=size - maj_cnt
        )
    else:
        arr = random.choices(range(1, domain_size + 1), k=size)
    random.shuffle(arr)
    return arr


class FindMajorityExperiment:
    def __init__(self):
        self.algorithms = {
            "Brute Force": self.brute_force,
            "Insertion Sort": self.insertion_sort_majority,
            "Merge Sort": self.merge_sort_majority,
            "Quick Sort": self.quick_sort_majority,
            "Divide and Conquer": self.divide_and_conquer_majority,
            "Hash-based": self.hash_based_majority,
            "Boyer-Moore": self.boyer_moore_majority,
            "Random Selection": self.random_selection_majority
        }
        self.comparison_count = 0

    def reset_comparison_count(self):
        self.comparison_count = 0

    def brute_force(self, arr: List[int]) -> int:
        n = len(arr)
        threshold = n // 2
        for i in range(n):
            count = 0
            for j in range(n):
                self.comparison_count += 1
                if arr[i] == arr[j]:
                    count += 1
            self.comparison_count += 1
            if count > threshold:
                return arr[i]
        return -1

    def insertion_sort(self, arr: List[int]) -> List[int]:
        n = len(arr)
        sorted_arr = arr.copy()
        for i in range(1, n):
            key = sorted_arr[i]
            j = i - 1
            while j >= 0 and sorted_arr[j] > key:
                self.comparison_count += 1
                sorted_arr[j + 1] = sorted_arr[j]
                j -= 1
            if j >= 0:
                self.comparison_count += 1
            sorted_arr[j + 1] = key
        return sorted_arr

    def insertion_sort_majority(self, arr: List[int]) -> int:
        if not arr:
            return -1
        sorted_arr = self.insertion_sort(arr)
        n = len(arr)
        candidate = sorted_arr[n // 2]
        count = 0
        for x in sorted_arr:
            self.comparison_count += 1
            if x == candidate:
                count += 1
        self.comparison_count += 1
        return candidate if count > n // 2 else -1

    def merge(self, left: List[int], right: List[int]) -> List[int]:
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            self.comparison_count += 1
            if left[i] <= right[j]:
                self.comparison_count += 1
                result.append(left[i]); i += 1
            else:
                result.append(right[j]); j += 1
        result.extend(left[i:]); result.extend(right[j:])
        return result

    def merge_sort(self, arr: List[int]) -> List[int]:
        if len(arr) <= 1:
            self.comparison_count += 1
            return arr
        mid = len(arr) // 2
        return self.merge(self.merge_sort(arr[:mid]), self.merge_sort(arr[mid:]))

    def merge_sort_majority(self, arr: List[int]) -> int:
        if not arr:
            return -1
        sorted_arr = self.merge_sort(arr)
        n = len(arr)
        candidate = sorted_arr[n // 2]
        count = 0
        for x in sorted_arr:
            self.comparison_count += 1
            if x == candidate:
                count += 1
        self.comparison_count += 1
        return candidate if count > n // 2 else -1

    def partition(self, arr: List[int], low: int, high: int) -> int:
        pivot = arr[low]
        i = low + 1
        for j in range(low + 1, high + 1):
            self.comparison_count += 1
            if arr[j] < pivot:
                arr[i], arr[j] = arr[j], arr[i]; i += 1
        arr[low], arr[i - 1] = arr[i - 1], arr[low]
        return i - 1

    def quick_sort_helper(self, arr: List[int], low: int, high: int) -> None:
        while low < high:
            self.comparison_count += 1
            pi = self.partition(arr, low, high)
            if pi - low < high - pi:
                self.quick_sort_helper(arr, low, pi - 1)
                low = pi + 1
            else:
                self.quick_sort_helper(arr, pi + 1, high)
                high = pi - 1

    def quick_sort(self, arr: List[int]) -> List[int]:
        a = arr.copy()
        if len(a) > 1:
            self.quick_sort_helper(a, 0, len(a) - 1)
        return a

    def quick_sort_majority(self, arr: List[int]) -> int:
        if not arr:
            return -1
        sa = self.quick_sort(arr)
        n = len(arr); cand = sa[n // 2]
        count = 0
        for x in sa:
            self.comparison_count += 1
            if x == cand:
                count += 1
        self.comparison_count += 1
        return cand if count > n // 2 else -1

    def divide_and_conquer_majority(self, array: List[int],
                                    low: int = 0, high: int = None) -> int:
        if high is None:
            high = len(array) - 1
        if low == high:
            return array[low]
        mid = (low + high) // 2
        left = self.divide_and_conquer_majority(array, low, mid)
        right = self.divide_and_conquer_majority(array, mid + 1, high)
        self.comparison_count += 1
        if left == right:
            return left
        lc = rc = 0
        for i in range(low, high + 1):
            self.comparison_count += 1
            if array[i] == left:
                lc += 1
            else:
                self.comparison_count += 1
                if array[i] == right:
                    rc += 1
        size = high - low + 1
        self.comparison_count += 1
        if lc > size // 2:
            return left
        self.comparison_count += 1
        if rc > size // 2:
            return right
        return -1

    def hash_based_majority(self, arr: List[int]) -> int:
        if not arr:
            return -1
        freq = Counter(arr)
        threshold = len(arr) // 2
        for num, cnt in freq.items():
            self.comparison_count += 1
            if cnt > threshold:
                return num
        return -1

    def boyer_moore_majority(self, arr: List[int]) -> int:
        if not arr:
            return -1
        cand = None; cnt = 0
        for x in arr:
            self.comparison_count += 1
            if cnt == 0:
                cand, cnt = x, 1
            elif cand == x:
                self.comparison_count += 1; cnt += 1
            else:
                self.comparison_count += 1; cnt -= 1
        # Verification step: count how many times cand appears, updating comparison_count for each comparison
        cnt_verify = 0
        for x in arr:
            self.comparison_count += 1
            if x == cand:
                cnt_verify += 1
        self.comparison_count += 1
        return cand if cnt_verify > len(arr) // 2 else -1

    def random_selection_majority(self, arr: List[int]) -> int:
        if not arr:
            return -1
        counts = Counter(arr)
        working = arr.copy()
        while working:
            self.comparison_count += 1
            idx = random.randrange(len(working))
            cand = working[idx]
            if counts[cand] > len(arr) // 2:
                return cand
            self.comparison_count += 1
            new_working = []
            for x in working:
                self.comparison_count += 1
                if x != cand:
                    new_working.append(x)
            working = new_working
        return -1

    def run_experiment(self, array_sizes: List[int], num_trials: int = 5):
        results = {"time": {}, "comparisons": {}}
        for alg in self.algorithms:
            results["time"][alg] = {}
            results["comparisons"][alg] = {}
            for n in array_sizes:
                results["time"][alg][n] = {s: 0 for s in ["best", "worst", "average"]}
                results["comparisons"][alg][n] = {s: 0 for s in ["best", "worst", "average"]}
        for n in array_sizes:
            print(f"\nRunning Experiment for array size {n}:")
            for scenario in ["best", "worst", "average"]:
                print(f"  Scenario: {scenario}")
                for alg, func in self.algorithms.items():
                    total_t = total_c = 0
                    runs = 10 if alg == "Random Selection" else 1
                    for _ in range(num_trials):
                        arr = generate_array(alg, scenario, n)
                        self.reset_comparison_count()
                        start = time.time()
                        for __ in range(runs):
                            func(arr)
                        end = time.time()
                        total_t += (end - start) * 1000 / runs
                        total_c += self.comparison_count / runs
                    results["time"][alg][n][scenario] = total_t / num_trials
                    results["comparisons"][alg][n][scenario] = total_c / num_trials
        return results

    def plot_time_comparison(self, results, array_sizes, scenario="average"):
        plt.figure(figsize=(12, 8))

        for alg_name in results["time"].keys():
            times = [results["time"][alg_name][size][scenario] for size in array_sizes
                     if results["time"][alg_name][size][scenario] != float('inf')]

            if times:
                valid_sizes = [size for size in array_sizes
                               if results["time"][alg_name][size][scenario] != float('inf')]
                plt.plot(valid_sizes, times, marker='o', label=alg_name)

        plt.title(f'Time Comparison ({scenario} case)')
        plt.xlabel('Array Size')
        plt.ylabel('Time (ms)')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.savefig(f'time_comparison_{scenario}.png')
        plt.close()

    def plot_comparisons_comparison(self, results, array_sizes, scenario="average"):
        plt.figure(figsize=(12, 8))

        for alg_name in results["comparisons"].keys():
            comps = [results["comparisons"][alg_name][size][scenario] for size in array_sizes
                     if results["comparisons"][alg_name][size][scenario] != float('inf')]

            if comps:
                valid_sizes = [size for size in array_sizes
                               if results["comparisons"][alg_name][size][scenario] != float('inf')]
                plt.plot(valid_sizes, comps, marker='o', label=alg_name)

        plt.title(f'Comparison Count ({scenario} case)')
        plt.xlabel('Array Size')
        plt.ylabel('Comparison Count')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.savefig(f'comparison_count_{scenario}.png')
        plt.close()

    def plot_all_results(self, results, array_sizes):
        for scenario in ["best", "worst", "average"]:
            self.plot_time_comparison(results, array_sizes, scenario)
            self.plot_comparisons_comparison(results, array_sizes, scenario)

        self.plot_theoretical_comparison(results, array_sizes)

    def theoretical_complexity(self, algorithm_name, n):
        """
        Return precise theoretical complexity values with quadratic and logarithmic behaviors
        """
        if algorithm_name in ["Brute Force", "Insertion Sort"]:
            return n ** 2  # O(n²) - quadratic time, now explicitly squared
        elif algorithm_name in ["Merge Sort", "Quick Sort", "Divide and Conquer"]:
            return n * np.log2(n + 1)  # O(n log n) - linearithmic time
        elif algorithm_name in ["Hash-based", "Boyer-Moore", "Random Selection"]:
            return n  # O(n) - linear time
        else:
            return n  # Default linear time

    def plot_theoretical_comparison(self, results, array_sizes):
        """
        For each algorithm, plot the actual running time versus the theoretical
        complexity on the same graph.
        The theoretical curve is scaled by a single factor that minimises the
        mean error in log–log space, so shape differences (slope) are easier to
        see even when magnitudes differ.
        """
        import numpy as np
        scenario = "average"

        for alg_name in results["time"].keys():
            # Collect finite (non-inf) timing values
            times = [
                results["time"][alg_name][n][scenario]
                for n in array_sizes
                if results["time"][alg_name][n][scenario] != float("inf")
            ]

            if not times:
                continue  # Skip if no data for this algorithm

            valid_sizes = [
                n
                for n in array_sizes
                if results["time"][alg_name][n][scenario] != float("inf")
            ]

            # --- Plot actual performance ---
            plt.figure(figsize=(10, 6))
            plt.plot(valid_sizes, times, "o-", label=f"{alg_name} (Actual)")

            # --- Compute theoretical values and scale factor ---
            theo = np.array(
                [self.theoretical_complexity(alg_name, n) for n in valid_sizes],
                dtype=float,
            )
            real = np.array(times, dtype=float)

            # Robust scaling approach
            try:
                # Use median of ratios for more robust scaling
                scaling_ratios = real / (theo + 1e-10)  # Avoid division by zero
                scale = np.median(scaling_ratios)
            except Exception:
                # Fallback to simple mean-based scaling
                scale = np.mean(real) / np.mean(theo)

            # Ensure scale is reasonable
            scale = max(1e-5, min(scale, 1e5))

            # Scale theoretical values
            theo_scaled = theo * scale

            # --- Plot scaled theoretical curve ---
            plt.plot(
                valid_sizes,
                theo_scaled,
                "--",
                label=f"{alg_name} (Theoretical, scaled)",
            )

            # Axes setup
            plt.xscale("log")
            plt.yscale("log")

            plt.title(f"{alg_name}: Theoretical vs Actual ({scenario} case)")
            plt.xlabel("Array Size (n)")
            plt.ylabel("Time (ms)")
            plt.grid(True, which="both", ls="--")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                f'theoretical_comparison_{alg_name.replace(" ", "_").lower()}.png'
            )
            plt.close()

    def summary_table(self, results, array_sizes):
        """Creates a summary table of algorithm performance for average case"""
        print("\nAlgorithm Performance Summary (Average Case):")
        print("-" * 100)
        header = "Algorithm".ljust(20)
        for size in array_sizes:
            header += f"| n={size}".ljust(15)
        print(header)
        print("-" * 100)

        # Time results
        print("Execution Time (ms):")
        for alg_name in self.algorithms.keys():
            row = alg_name.ljust(20)
            for size in array_sizes:
                time_val = results["time"][alg_name][size]["average"]
                time_str = f"{time_val:.4f}" if time_val != float('inf') else "N/A"
                row += f"| {time_str}".ljust(15)
            print(row)

        print("-" * 100)

        # Comparison count results
        print("Comparison Count:")
        for alg_name in self.algorithms.keys():
            row = alg_name.ljust(20)
            for size in array_sizes:
                comp_val = results["comparisons"][alg_name][size]["average"]
                comp_str = f"{comp_val:.1f}" if comp_val != float('inf') else "N/A"
                row += f"| {comp_str}".ljust(15)
            print(row)

        print("-" * 100)

        # Best and worst case analysis
        print("\nBest/Worst Case Analysis:")
        for alg_name in self.algorithms.keys():
            print(f"\n{alg_name}:")
            for size in array_sizes:
                if size <= 10000:  # Show sizes up to 10000
                    best_time = results["time"][alg_name][size]["best"]
                    worst_time = results["time"][alg_name][size]["worst"]
                    avg_time = results["time"][alg_name][size]["average"]

                    if best_time != float('inf'):
                        best_comp = results["comparisons"][alg_name][size]["best"]
                        worst_comp = results["comparisons"][alg_name][size]["worst"]
                        avg_comp = results["comparisons"][alg_name][size]["average"]

                        print(
                            f"  Size {size}: Best/Avg/Worst Time = {best_time:.4f}/{avg_time:.4f}/{worst_time:.4f} ms")
                        print(
                            f"  Size {size}: Best/Avg/Worst Comparisons = {best_comp:.1f}/{avg_comp:.1f}/{worst_comp:.1f}")


if __name__ == "__main__":
    experiment = FindMajorityExperiment()

    # Use logarithmically spaced sizes up to 10000 for even spacing on a log–log plot
    array_sizes = [int(x) for x in np.logspace(2, np.log10(10000), num=30)]

    print("Running Experiments...")
    results = experiment.run_experiment(array_sizes, num_trials=3)

    print("Constructing Graphs...")
    experiment.plot_all_results(results, array_sizes)

    # Print summary table
    experiment.summary_table(results, array_sizes)

    print("Experiment Completed Successfully!")