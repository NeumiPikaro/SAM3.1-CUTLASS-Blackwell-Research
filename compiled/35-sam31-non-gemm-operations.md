# Non-GEMM Operations in SAM 3.1 — CUDA Implementation

## 1. LayerNorm

### Operation
```
y = (x - mean) / sqrt(var + eps) * gamma + beta
mean = mean(x), var = variance(x)
```

### SAM 3.1 Usage
- 2 per ViT block (pre-attention, pre-MLP) × 32 blocks = 64 calls
- 2 per DETR layer × 12 = 24 calls
- Total: ~88 LayerNorm calls

### CUTLASS Implementation
```cuda
__global__ void layernorm_kernel(
    bfloat16_t* __restrict__ output,
    const bfloat16_t* __restrict__ input,
    const bfloat16_t* __restrict__ gamma,
    const bfloat16_t* __restrict__ beta,
    int N, int D, float eps = 1e-6f) 
{
    int row = blockIdx.x;
    if (row >= N) return;
    
    extern __shared__ float smem[];
    float* s_sum = smem;
    float* s_sum_sq = smem + blockDim.x;
    
    // Pass 1: compute mean
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float val = static_cast<float>(input[row * D + i]);
        local_sum += val;
    }
    s_sum[threadIdx.x] = local_sum;
    __syncthreads();
    
    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
        __syncthreads();
    }
    float mean = s_sum[0] / D;
    
    // Pass 2: compute variance
    float local_var = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float val = static_cast<float>(input[row * D + i]) - mean;
        local_var += val * val;
    }
    s_sum[threadIdx.x] = local_var;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
        __syncthreads();
    }
    float variance = s_sum[0] / D;
    float inv_std = rsqrtf(variance + eps);
    
    // Pass 3: normalize
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float val = static_cast<float>(input[row * D + i]);
        float normed = (val - mean) * inv_std;
        float g = static_cast<float>(gamma[i]);
        float b = static_cast<float>(beta[i]);
        output[row * D + i] = static_cast<bfloat16_t>(normed * g + b);
    }
}
```

### Performance
- Memory: 3 passes over data (mean, variance, normalize)
- Bandwidth: 3 × D × 2 bytes per row
- For D=1024: 6 KB per row — memory bound
- Optimization: fuse with preceding residual add (see Section 34)

## 2. GELU Activation

### Operation
```
GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
```
Or the approximate version:
```
GELU(x) ≈ x * Φ(x) ≈ x * 0.5 * (1 + tanh(...))
```

### CUTLASS Integration
GELU is available as a CUTLASS epilogue compute operation:
```cpp
Sm90Compute<cutlass::epilogue::thread::GELU, float, ElementC>
```
Used in EVT for addmm_act fusion.

### Standalone GELU Kernel (if needed)
```cuda
__global__ void gelu_kernel(bfloat16_t* out, const bfloat16_t* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = static_cast<float>(in[idx]);
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        out[idx] = static_cast<bfloat16_t>(x * cdf);
    }
}
```

## 3. Softmax

### Operation
```
softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
```

### In Flash Attention
Softmax is fused into the Flash Attention mainloop — no separate kernel.

### Standalone Softmax (for DETR attention)
```cuda
__global__ void softmax_kernel(float* out, const float* in, int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return;
    
    extern __shared__ float smem[];
    
    // Find max
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        max_val = fmaxf(max_val, in[row * N + i]);
    smem[threadIdx.x] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }
    max_val = smem[0];
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float val = expf(in[row * N + i] - max_val);
        out[row * N + i] = val;
        sum += val;
    }
    smem[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    sum = smem[0];
    
    // Normalize
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        out[row * N + i] /= sum;
}
```

## 4. RoPE (Rotary Position Embeddings)

### Operation (Real-Only Form)
```
For position p, dimension i:
  freq = base^(-2i/d) where base=10000
  Q[2i]   = Q[2i] * cos(freq*p) - Q[2i+1] * sin(freq*p)
  Q[2i+1] = Q[2i] * sin(freq*p) + Q[2i+1] * cos(freq*p)
```

### SAM 3.1 2D RoPE
For 2D patches at position (row, col):
```
Q_rot = Q ⊙ exp(i * (θ_row + θ_col))
```
Separate frequencies for row and column dimensions.

### CUDA Implementation
```cuda
__global__ void rope_2d_kernel(
    bfloat16_t* Q, const bfloat16_t* Q_in,
    int seq_len, int head_dim, int H, int W) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * head_dim;
    if (idx >= total) return;
    
    int patch = idx / head_dim;
    int dim = idx % head_dim;
    int row = patch / W;
    int col = patch % W;
    
    float theta = 10000.0f;
    int pair = dim / 2;
    float freq_row = powf(theta, -2.0f * pair / head_dim);
    float freq_col = powf(theta, -2.0f * (pair + head_dim/2) / head_dim);
    float angle = freq_row * row + freq_col * col;
    
    float val = static_cast<float>(Q_in[idx]);
    if (dim % 2 == 0) {
        int next_idx = idx + 1;
        float val_next = (next_idx < total) ? static_cast<float>(Q_in[next_idx]) : 0.0f;
        Q[idx] = static_cast<bfloat16_t>(val * cosf(angle) - val_next * sinf(angle));
    } else {
        int prev_idx = idx - 1;
        float val_prev = static_cast<float>(Q_in[prev_idx]);
        Q[idx] = static_cast<bfloat16_t>(val_prev * sinf(angle) + val * cosf(angle));
    }
}
```

## 5. Upsampling (Bilinear)

### Operation
```cuda
__global__ void bilinear_upsample_2x_kernel(
    bfloat16_t* out, const bfloat16_t* in,
    int H_in, int W_in, int C) 
{
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = H_in * 2, W_out = W_in * 2;
    
    if (h_out >= H_out || w_out >= W_out) return;
    
    float h_in = (h_out + 0.5f) * 0.5f - 0.5f;
    float w_in = (w_out + 0.5f) * 0.5f - 0.5f;
    
    int h0 = max(0, min((int)h_in, H_in - 1));
    int h1 = min(h0 + 1, H_in - 1);
    int w0 = max(0, min((int)w_in, W_in - 1));
    int w1 = min(w0 + 1, W_in - 1);
    
    float dh = h_in - h0, dw = w_in - w0;
    
    for (int c = 0; c < C; c++) {
        float v00 = in[h0 * W_in * C + w0 * C + c];
        float v01 = in[h0 * W_in * C + w1 * C + c];
        float v10 = in[h1 * W_in * C + w0 * C + c];
        float v11 = in[h1 * W_in * C + w1 * C + c];
        
        float val = (1-dh)*(1-dw)*v00 + (1-dh)*dw*v01 + dh*(1-dw)*v10 + dh*dw*v11;
        out[h_out * W_out * C + w_out * C + c] = val;
    }
}
```

