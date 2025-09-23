import re
import matplotlib.pyplot as plt

def parse_output(filename):
    data = {"basic": {}, "blocked": {}, "blas": {}}
    current = None
    with open(filename) as f:
        for line in f:
            if "Basic implementation" in line:
                current = "basic"
            elif "Blocked dgemm" in line:
                current = "blocked"
            elif "Reference dgemm" in line:
                current = "blas"

            # Basic / BLAS
            m = re.match(r"N=(\d+).*time=(\d+\.\d+)", line)
            if m and current in ["basic", "blas"]:
                n, t = int(m.group(1)), float(m.group(2))
                data[current][n] = t

            # Blocked
            m = re.match(r"N=(\d+).*block=(\d+).*time=(\d+\.\d+)", line)
            if m and current == "blocked":
                n, b, t = int(m.group(1)), int(m.group(2)), float(m.group(3))
                if n not in data["blocked"]:
                    data["blocked"][n] = {}
                data["blocked"][n][b] = t
    return data

def compute_mflops(n, t):
    return (2 * (n**3)) / (t * 1e6) if t > 0 else 0

def plot_results(data):
    # 1: Basic vs BLAS
    Ns = sorted(data["basic"].keys())
    plt.figure()
    plt.plot(Ns, [compute_mflops(N, data["basic"][N]) for N in Ns], "o-", label="basic")
    plt.plot(Ns, [compute_mflops(N, data["blas"][N]) for N in Ns], "s-", label="blas")
    plt.xlabel("Problem size N")
    plt.ylabel("MFLOP/s")
    plt.title("Basic vs BLAS")
    plt.legend()
    plt.grid(True)
    plt.savefig("basic_vs_blas.png")

    # 2: Blocked vs BLAS
    Ns = sorted(data["blocked"].keys())
    plt.figure()
    for b in sorted(next(iter(data["blocked"].values())).keys()):
        plt.plot(Ns, [compute_mflops(N, data["blocked"][N][b]) for N in Ns], "o-", label=f"blocked B={b}")
    plt.plot(sorted(data["blas"].keys()), [compute_mflops(N, data["blas"][N]) for N in sorted(data["blas"].keys())], "s-", label="blas")
    plt.xlabel("Problem size N")
    plt.ylabel("MFLOP/s")
    plt.title("Blocked vs BLAS")
    plt.legend()
    plt.grid(True)
    plt.savefig("blocked_vs_blas.png")

    # 3: Basic vs Blocked
    Ns = sorted(data["blocked"].keys())
    plt.figure()
    plt.plot(sorted(data["basic"].keys()), 
             [compute_mflops(N, data["basic"][N]) for N in sorted(data["basic"].keys())], "o-", label="basic")
    for b in sorted(next(iter(data["blocked"].values())).keys()):
        plt.plot(Ns, [compute_mflops(N, data["blocked"][N][b]) for N in Ns], "o-", label=f"blocked B={b}")
    plt.xlabel("Problem size N")
    plt.ylabel("MFLOP/s")
    plt.title("Basic vs Blocked")
    plt.legend()
    plt.grid(True)
    plt.savefig("basic_vs_blocked.png")

    # 4: All Implementations
    Ns = sorted(set(data["basic"].keys()) & set(data["blas"].keys()))
    plt.figure()
    plt.plot(Ns, [compute_mflops(N, data["basic"][N]) for N in Ns], "o-", label="basic")
    plt.plot(Ns, [compute_mflops(N, data["blas"][N]) for N in Ns], "s-", label="blas")
    for b in sorted(next(iter(data["blocked"].values())).keys()):
        plt.plot(Ns, [compute_mflops(N, data["blocked"][N][b]) for N in Ns], "^-", label=f"blocked B={b}")
    plt.xlabel("Problem size N")
    plt.ylabel("MFLOP/s")
    plt.title("All Implementations")
    plt.ylim(0, 60000)
    plt.legend()
    plt.grid(True)
    plt.savefig("all_impl.png")

def main():
    data = parse_output("mmul-43006909.o")
    plot_results(data)
    print("✅ Image saved: basic_vs_blas.png, blocked_vs_blas.png, basic_vs_blocked.png, all_impl.png")

if __name__ == "__main__":
    main()
