int Factor(int N) {
    return (N <= 0) ? 1 : (N * Factor(N - 1));
}

