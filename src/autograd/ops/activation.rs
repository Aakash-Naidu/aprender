// ============================================================================
// Activation Functions
// ============================================================================
use crate::autograd::tensor::TensorId;
impl Tensor {
    /// `ReLU` activation: z = max(0, self)
    #[must_use]
    pub fn relu(&self) -> Tensor {
        let a = if self.is_transposed() {
            self.contiguous()
        } else {
            self.clone()
        };
        // Delegate to trueno's AVX2 SIMD relu with zero-copy allocation.
        // Contract: provable-contracts/contracts/activation-kernel-v1.yaml
        let data = trueno::blis::elementwise::relu_alloc(a.data());
        let mut result = Tensor::from_vec(data, self.shape());

        if is_grad_enabled() && self.requires_grad_enabled() {
            result.requires_grad_(true);
            let grad_fn = Arc::new(ReluBackward { x: self.clone() });
            result.set_grad_fn(grad_fn.clone());

            with_graph(|graph| {
                graph.register_tensor(self.clone());
                graph.record(result.id(), grad_fn, vec![self.id()]);
            });
        }

        result
    }

    /// Sigmoid activation: z = 1 / (1 + exp(-self))
    #[must_use]
    pub fn sigmoid(&self) -> Tensor {
        let a = if self.is_transposed() {
            self.contiguous()
        } else {
            self.clone()
        };
        let src = a.data();
        let n = src.len();
        let mut data = vec![0.0f32; n];
        for i in 0..n {
            data[i] = 1.0 / (1.0 + (-src[i]).exp());
        }
        let mut result = Tensor::from_vec(data, self.shape());

        if is_grad_enabled() && self.requires_grad_enabled() {
            result.requires_grad_(true);
            let grad_fn = Arc::new(SigmoidBackward {
                output: result.clone(),
            });
            result.set_grad_fn(grad_fn.clone());

            with_graph(|graph| {
                graph.register_tensor(self.clone());
                graph.record(result.id(), grad_fn, vec![self.id()]);
            });
        }

        result
    }

    /// Tanh activation
    #[must_use]
    pub fn tanh_(&self) -> Tensor {
        let a = if self.is_transposed() {
            self.contiguous()
        } else {
            self.clone()
        };
        let data: Vec<f32> = a.data().iter().map(|&a| a.tanh()).collect();
        let mut result = Tensor::from_vec(data, self.shape());

        if is_grad_enabled() && self.requires_grad_enabled() {
            result.requires_grad_(true);
            let grad_fn = Arc::new(TanhBackward {
                output: result.clone(),
            });
            result.set_grad_fn(grad_fn.clone());

            with_graph(|graph| {
                graph.register_tensor(self.clone());
                graph.record(result.id(), grad_fn, vec![self.id()]);
            });
        }

        result
    }

    /// Leaky `ReLU` activation: z = `max(negative_slope` * x, x)
    ///
    /// # Arguments
    ///
    /// * `negative_slope` - Controls the angle of the negative slope (default: 0.01)
    #[must_use]
    pub fn leaky_relu(&self, negative_slope: f32) -> Tensor {
        let a = if self.is_transposed() {
            self.contiguous()
        } else {
            self.clone()
        };
        let src = a.data();
        let n = src.len();
        let mut data = vec![0.0f32; n];
        for i in 0..n {
            data[i] = if src[i] > 0.0 {
                src[i]
            } else {
                negative_slope * src[i]
            };
        }
        let mut result = Tensor::from_vec(data, self.shape());

        if is_grad_enabled() && self.requires_grad_enabled() {
            result.requires_grad_(true);
            let grad_fn = Arc::new(LeakyReluBackward {
                x: self.clone(),
                negative_slope,
            });
            result.set_grad_fn(grad_fn.clone());

            with_graph(|graph| {
                graph.register_tensor(self.clone());
                graph.record(result.id(), grad_fn, vec![self.id()]);
            });
        }

        result
    }

    /// GELU (Gaussian Error Linear Unit) activation.
    ///
    /// Uses the tanh approximation:
    /// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    #[must_use]
    pub fn gelu(&self) -> Tensor {
        let a = if self.is_transposed() {
            self.contiguous()
        } else {
            self.clone()
        };
        let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();

        let src = a.data();
        let n = src.len();
        let mut data = vec![0.0f32; n];
        for i in 0..n {
            let x = src[i];
            let inner = sqrt_2_over_pi * (x + 0.044715 * x.powi(3));
            data[i] = 0.5 * x * (1.0 + inner.tanh());
        }
        let mut result = Tensor::from_vec(data, self.shape());

        if is_grad_enabled() && self.requires_grad_enabled() {
            result.requires_grad_(true);
            let grad_fn = Arc::new(GeluBackward { x: self.clone() });
            result.set_grad_fn(grad_fn.clone());

            with_graph(|graph| {
                graph.register_tensor(self.clone());
                graph.record(result.id(), grad_fn, vec![self.id()]);
            });
        }

        result
    }

    /// Softmax activation over the last dimension of a 2D tensor.
    ///
    /// softmax(x)_i = `exp(x_i)` / `Σ_j` `exp(x_j)`
    ///
    /// Uses numerically stable computation with max subtraction.
    #[must_use]
    pub fn softmax(&self) -> Tensor {
        // Softmax operates along a dimension, so layout matters.
        // Ensure data is contiguous for correct mapping.
        if self.is_transposed() {
            return self.contiguous().softmax();
        }

        // ONE PATH: Computation delegates to nn::functional::softmax (UCBD §4).
        // Gradient tracking is handled here (autograd layer).
        let computed = crate::nn::functional::softmax(self, -1);
        let mut result = Tensor::from_vec(computed.data().to_vec(), self.shape());

        if is_grad_enabled() && self.requires_grad_enabled() {
            result.requires_grad_(true);
            let grad_fn = Arc::new(SoftmaxBackward {
                output: result.clone(),
            });
            result.set_grad_fn(grad_fn.clone());

            with_graph(|graph| {
                graph.register_tensor(self.clone());
                graph.record(result.id(), grad_fn, vec![self.id()]);
            });
        }

        result
    }
}

// ============================================================================
// Linear Algebra
// ============================================================================

impl Tensor {
    /// Matrix multiplication: z = self @ other
    ///
    /// Currently supports 2D tensors only. Batched matmul (3D+ tensors) can be
    /// added by iterating over batch dimensions and calling 2D matmul.
    ///
    /// Contract: matmul-kernel-v1, equation "matmul"
    #[provable_contracts_macros::contract("matmul-kernel-v1", equation = "matmul")]
    #[must_use]
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.ndim(), 2, "matmul requires 2D tensors");
        assert_eq!(other.ndim(), 2, "matmul requires 2D tensors");

        // Matmul requires contiguous memory for optimal performance with BLIS.
        // If either tensor is transposed, make it contiguous.
        let a = if self.is_transposed() {
            self.contiguous()
        } else {
            self.clone()
        };
        let b = if other.is_transposed() {
            other.contiguous()
        } else {
            other.clone()
        };

        let (m, k1) = (a.shape()[0], a.shape()[1]);
        let (k2, n) = (b.shape()[0], b.shape()[1]);
        assert_eq!(k1, k2, "matmul dimension mismatch: {k1} vs {k2}");

        let data = if m == 1 {
            // GEMV fast path: call trueno's SIMD gemv directly on borrowed slices.
            // Avoids copying the K×N weight matrix (172MB at LLM scale).
            let mut c = vec![0.0f32; n];
            trueno::blis::gemv::gemv(k1, n, a.data(), b.data(), &mut c);
            c
        } else {
            // General matmul via trueno Matrix (copies data for Matrix ownership)
            let a_matrix = trueno::Matrix::from_vec(m, k1, a.data().to_vec())
                .expect("valid matrix dimensions");
            let b_matrix = trueno::Matrix::from_vec(k2, n, b.data().to_vec())
                .expect("valid matrix dimensions");
            let result_matrix = a_matrix.matmul(&b_matrix).expect("matmul should succeed");
            result_matrix.as_slice().to_vec()
        };

        let mut result = Tensor::from_vec(data, &[m, n]);

        if is_grad_enabled() && (self.requires_grad_enabled() || other.requires_grad_enabled()) {
            result.requires_grad_(true);
            let grad_fn = Arc::new(MatmulBackward {
                x: self.clone(),
                y: other.clone(),
            });
            result.set_grad_fn(grad_fn.clone());

            with_graph(|graph| {
                graph.register_tensor(self.clone());
                graph.register_tensor(other.clone());
                graph.record(result.id(), grad_fn, vec![self.id(), other.id()]);
            });
        }

        result
    }

    /// Transpose a 2D tensor.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    /// let a_t = a.transpose();
    /// // a_t = [[1, 3], [2, 4]]
    /// ```
    #[must_use]
    pub fn transpose(&self) -> Tensor {
        assert!(self.ndim() >= 2, "transpose requires at least 2D tensor");

        let mut result = self.clone();

        // Lazy Transpose: swap last two dimensions and flip flag
        let ndim = result.shape.len();
        result.shape.swap(ndim - 1, ndim - 2);
        result.is_transposed = !self.is_transposed;
        result.id = TensorId::new();

        if is_grad_enabled() && self.requires_grad_enabled() {
            result.requires_grad_(true);
            let grad_fn = Arc::new(TransposeBackward);
            result.set_grad_fn(grad_fn.clone());

            with_graph(|graph| {
                graph.register_tensor(self.clone());
                graph.record(result.id(), grad_fn, vec![self.id()]);
            });
        }

        result
    }

    /// Broadcast addition: z = matrix + vector (broadcasts over rows).
    ///
    /// The vector is broadcast to match the matrix's second dimension.
    /// This is useful for adding biases in neural networks.
    ///
    /// # Shape
    ///
    /// - self: `[N, M]` (2D matrix)
    /// - other: `[M]` (1D vector)
    /// - output: `[N, M]`
    ///
    /// # Example
    ///
    /// ```ignore
    /// let matrix = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    /// let bias = Tensor::new(&[10.0, 20.0], &[2]);
    /// let result = matrix.broadcast_add(&bias);
    /// // result = [[11, 22], [13, 24]]
    /// ```
    #[must_use]
    pub fn broadcast_add(&self, other: &Tensor) -> Tensor {
        let a = if self.is_transposed() {
            self.contiguous()
        } else {
            self.clone()
        };
        let b = if other.is_transposed() {
            other.contiguous()
        } else {
            other.clone()
        };

        assert_eq!(a.ndim(), 2, "broadcast_add requires 2D matrix");
        assert_eq!(b.ndim(), 1, "broadcast_add requires 1D vector");
        assert_eq!(
            a.shape()[1],
            b.shape()[0],
            "Matrix columns {} must match vector length {}",
            a.shape()[1],
            b.shape()[0]
        );

        let (rows, cols) = (a.shape()[0], a.shape()[1]);
        let mut data = vec![0.0; rows * cols];

        for i in 0..rows {
            for j in 0..cols {
                data[i * cols + j] = a.data()[i * cols + j] + b.data()[j];
            }
        }

        let mut result = Tensor::from_vec(data, self.shape());

        if is_grad_enabled() && (self.requires_grad_enabled() || other.requires_grad_enabled()) {
            result.requires_grad_(true);
            let grad_fn = Arc::new(BroadcastAddBackward {
                x_shape: self.shape().to_vec(),
                y_shape: other.shape().to_vec(),
            });
            result.set_grad_fn(grad_fn.clone());

            with_graph(|graph| {
                graph.register_tensor(self.clone());
                graph.register_tensor(other.clone());
                graph.record(result.id(), grad_fn, vec![self.id(), other.id()]);
            });
        }

        result
    }

    /// Reshape tensor to a new shape (view).
    ///
    /// The total number of elements must remain the same.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    /// let b = a.view(&[3, 2]);
    /// // b = [[1, 2], [3, 4], [5, 6]]
    /// ```
    #[must_use]
    pub fn view(&self, new_shape: &[usize]) -> Tensor {
        let a = if self.is_transposed() {
            self.contiguous()
        } else {
            self.clone()
        };
        let old_numel: usize = a.shape().iter().product();
        let new_numel: usize = new_shape.iter().product();
        assert_eq!(
            old_numel, new_numel,
            "view: number of elements must match ({old_numel} vs {new_numel})"
        );

        let mut result = Tensor::new(self.data(), new_shape);

        if is_grad_enabled() && self.requires_grad_enabled() {
            result.requires_grad_(true);
            let grad_fn = Arc::new(ViewBackward {
                input_shape: self.shape().to_vec(),
            });
            result.set_grad_fn(grad_fn.clone());

            with_graph(|graph| {
                graph.register_tensor(self.clone());
                graph.record(result.id(), grad_fn, vec![self.id()]);
            });
        }

        result
    }
}

#[cfg(test)]
mod tests;
