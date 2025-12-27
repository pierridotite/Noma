use crate::graph::{ComputationalGraph, NodeId, NodeType, Value, Tensor};
use std::collections::{BTreeSet, HashMap};

/// Represents a value in LLVM IR - either a scalar SSA value or a tensor descriptor
#[derive(Debug, Clone)]
enum LLVMValue {
    /// Scalar value: the SSA variable name (e.g., "%5")
    Scalar(String),
    /// Tensor value: (data_ptr, shape_ptr, rank, total_size)
    /// data_ptr: double* pointing to the tensor data
    /// shape_ptr: i64* pointing to the shape array (stored as global constant)
    /// rank: number of dimensions
    /// total_size: total number of elements
    Tensor {
        data_ptr: String,
        shape: Vec<usize>,
    },
}

/// LLVM IR code generator with tensor support
/// Converts a computational graph to LLVM Intermediate Representation
pub struct LLVMCodegen {
    counter: usize,
    label_counter: usize,
    fast_math: bool,
    extern_decls: BTreeSet<String>,
    /// Global constants for tensor data
    global_constants: Vec<String>,
    /// Track which tensors need to be freed
    allocated_tensors: Vec<String>,
}

impl LLVMCodegen {
    pub fn new() -> Self {
        Self { 
            counter: 0, 
            label_counter: 0,
            fast_math: false, 
            extern_decls: BTreeSet::new(),
            global_constants: Vec::new(),
            allocated_tensors: Vec::new(),
        }
    }

    pub fn with_fast_math(mut self, enabled: bool) -> Self {
        self.fast_math = enabled;
        self
    }

    fn fmt_f64(&self, v: f64) -> String {
        format!("{:.16e}", v)
    }

    fn fresh_var(&mut self) -> String {
        let var = format!("%v{}", self.counter);
        self.counter += 1;
        var
    }

    fn fresh_label(&mut self, prefix: &str) -> String {
        let label = format!("{}{}", prefix, self.label_counter);
        self.label_counter += 1;
        label
    }

    /// Create a global constant array for tensor data
    fn create_tensor_global(&mut self, data: &[f64], name_hint: &str) -> String {
        let global_name = format!("@tensor_data_{}", self.global_constants.len());
        let values: Vec<String> = data.iter().map(|v| format!("double {}", self.fmt_f64(*v))).collect();
        let global_def = format!(
            "{} = private unnamed_addr constant [{} x double] [{}], align 8\n",
            global_name,
            data.len(),
            values.join(", ")
        );
        self.global_constants.push(global_def);
        global_name
    }

    /// Generate code to allocate a tensor on the heap
    fn gen_tensor_alloc(&mut self, ir: &mut String, size: usize) -> String {
        self.extern_decls.insert("declare i8* @malloc(i64)".to_string());
        
        let bytes = size * 8; // sizeof(double)
        let malloc_result = self.fresh_var();
        let data_ptr = self.fresh_var();
        
        ir.push_str(&format!("  {} = call i8* @malloc(i64 {})\n", malloc_result, bytes));
        ir.push_str(&format!("  {} = bitcast i8* {} to double*\n", data_ptr, malloc_result));
        
        self.allocated_tensors.push(data_ptr.clone());
        data_ptr
    }

    /// Generate code to copy tensor data from a global constant to a heap allocation
    fn gen_tensor_copy_from_global(&mut self, ir: &mut String, global_name: &str, size: usize) -> String {
        let data_ptr = self.gen_tensor_alloc(ir, size);
        
        // Copy data from global to heap
        let global_ptr = self.fresh_var();
        ir.push_str(&format!(
            "  {} = getelementptr [{} x double], [{} x double]* {}, i64 0, i64 0\n",
            global_ptr, size, size, global_name
        ));
        
        // Use memcpy for efficiency
        self.extern_decls.insert("declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i1)".to_string());
        let dest_i8 = self.fresh_var();
        let src_i8 = self.fresh_var();
        ir.push_str(&format!("  {} = bitcast double* {} to i8*\n", dest_i8, data_ptr));
        ir.push_str(&format!("  {} = bitcast double* {} to i8*\n", src_i8, global_ptr));
        ir.push_str(&format!(
            "  call void @llvm.memcpy.p0i8.p0i8.i64(i8* {}, i8* {}, i64 {}, i1 false)\n",
            dest_i8, src_i8, size * 8
        ));
        
        data_ptr
    }

    /// Generate element-wise unary operation on a tensor
    fn gen_tensor_unary_op(&mut self, ir: &mut String, input: &LLVMValue, op: &str) -> Result<LLVMValue, String> {
        let (in_ptr, shape) = match input {
            LLVMValue::Tensor { data_ptr, shape } => (data_ptr.clone(), shape.clone()),
            LLVMValue::Scalar(_) => return Err("Expected tensor for unary op".to_string()),
        };
        
        let size: usize = shape.iter().product();
        let out_ptr = self.gen_tensor_alloc(ir, size);
        
        // Generate loop
        let loop_var = self.fresh_var();
        let loop_header = self.fresh_label("loop_header_");
        let loop_body = self.fresh_label("loop_body_");
        let loop_end = self.fresh_label("loop_end_");
        
        // Initialize loop counter
        let loop_alloca = self.fresh_var();
        ir.push_str(&format!("  {} = alloca i64\n", loop_alloca));
        ir.push_str(&format!("  store i64 0, i64* {}\n", loop_alloca));
        ir.push_str(&format!("  br label %{}\n", loop_header));
        
        // Loop header: check condition
        ir.push_str(&format!("{}:\n", loop_header));
        let idx = self.fresh_var();
        ir.push_str(&format!("  {} = load i64, i64* {}\n", idx, loop_alloca));
        let cond = self.fresh_var();
        ir.push_str(&format!("  {} = icmp slt i64 {}, {}\n", cond, idx, size));
        ir.push_str(&format!("  br i1 {}, label %{}, label %{}\n", cond, loop_body, loop_end));
        
        // Loop body: perform operation
        ir.push_str(&format!("{}:\n", loop_body));
        let in_elem_ptr = self.fresh_var();
        ir.push_str(&format!("  {} = getelementptr double, double* {}, i64 {}\n", in_elem_ptr, in_ptr, idx));
        let in_val = self.fresh_var();
        ir.push_str(&format!("  {} = load double, double* {}\n", in_val, in_elem_ptr));
        
        // Apply operation
        let out_val = match op {
            "neg" => {
                let v = self.fresh_var();
                ir.push_str(&format!("  {} = fsub double 0.0, {}\n", v, in_val));
                v
            }
            "sigmoid" => {
                let neg = self.fresh_var();
                ir.push_str(&format!("  {} = fsub double 0.0, {}\n", neg, in_val));
                let exp_val = self.fresh_var();
                ir.push_str(&format!("  {} = call double @llvm.exp.f64(double {})\n", exp_val, neg));
                let one_plus = self.fresh_var();
                ir.push_str(&format!("  {} = fadd double 1.0, {}\n", one_plus, exp_val));
                let result = self.fresh_var();
                ir.push_str(&format!("  {} = fdiv double 1.0, {}\n", result, one_plus));
                result
            }
            "relu" => {
                let result = self.fresh_var();
                ir.push_str(&format!("  {} = call double @llvm.maxnum.f64(double {}, double 0.0)\n", result, in_val));
                result
            }
            "tanh" => {
                let result = self.fresh_var();
                ir.push_str(&format!("  {} = call double @llvm.tanh.f64(double {})\n", result, in_val));
                result
            }
            "exp" => {
                let result = self.fresh_var();
                ir.push_str(&format!("  {} = call double @llvm.exp.f64(double {})\n", result, in_val));
                result
            }
            "log" => {
                let result = self.fresh_var();
                ir.push_str(&format!("  {} = call double @llvm.log.f64(double {})\n", result, in_val));
                result
            }
            "sqrt" => {
                let result = self.fresh_var();
                ir.push_str(&format!("  {} = call double @llvm.sqrt.f64(double {})\n", result, in_val));
                result
            }
            "sin" => {
                let result = self.fresh_var();
                ir.push_str(&format!("  {} = call double @llvm.sin.f64(double {})\n", result, in_val));
                result
            }
            "cos" => {
                let result = self.fresh_var();
                ir.push_str(&format!("  {} = call double @llvm.cos.f64(double {})\n", result, in_val));
                result
            }
            _ => return Err(format!("Unsupported unary tensor op: {}", op)),
        };
        
        // Store result
        let out_elem_ptr = self.fresh_var();
        ir.push_str(&format!("  {} = getelementptr double, double* {}, i64 {}\n", out_elem_ptr, out_ptr, idx));
        ir.push_str(&format!("  store double {}, double* {}\n", out_val, out_elem_ptr));
        
        // Increment and branch back
        let next_idx = self.fresh_var();
        ir.push_str(&format!("  {} = add i64 {}, 1\n", next_idx, idx));
        ir.push_str(&format!("  store i64 {}, i64* {}\n", next_idx, loop_alloca));
        ir.push_str(&format!("  br label %{}\n", loop_header));
        
        // Loop end
        ir.push_str(&format!("{}:\n", loop_end));
        
        Ok(LLVMValue::Tensor { data_ptr: out_ptr, shape })
    }

    /// Generate element-wise binary operation on tensors (with broadcasting support for scalar)
    fn gen_tensor_binary_op(&mut self, ir: &mut String, left: &LLVMValue, right: &LLVMValue, op: &str) -> Result<LLVMValue, String> {
        let fmf = if self.fast_math { " fast" } else { "" };
        
        match (left, right) {
            // Tensor op Tensor (same shape)
            (LLVMValue::Tensor { data_ptr: l_ptr, shape: l_shape }, 
             LLVMValue::Tensor { data_ptr: r_ptr, shape: r_shape }) => {
                if l_shape != r_shape {
                    return Err(format!("Tensor shape mismatch: {:?} vs {:?}", l_shape, r_shape));
                }
                
                let size: usize = l_shape.iter().product();
                let out_ptr = self.gen_tensor_alloc(ir, size);
                
                // Generate loop
                let loop_alloca = self.fresh_var();
                let loop_header = self.fresh_label("binop_header_");
                let loop_body = self.fresh_label("binop_body_");
                let loop_end = self.fresh_label("binop_end_");
                
                ir.push_str(&format!("  {} = alloca i64\n", loop_alloca));
                ir.push_str(&format!("  store i64 0, i64* {}\n", loop_alloca));
                ir.push_str(&format!("  br label %{}\n", loop_header));
                
                ir.push_str(&format!("{}:\n", loop_header));
                let idx = self.fresh_var();
                ir.push_str(&format!("  {} = load i64, i64* {}\n", idx, loop_alloca));
                let cond = self.fresh_var();
                ir.push_str(&format!("  {} = icmp slt i64 {}, {}\n", cond, idx, size));
                ir.push_str(&format!("  br i1 {}, label %{}, label %{}\n", cond, loop_body, loop_end));
                
                ir.push_str(&format!("{}:\n", loop_body));
                let l_elem_ptr = self.fresh_var();
                ir.push_str(&format!("  {} = getelementptr double, double* {}, i64 {}\n", l_elem_ptr, l_ptr, idx));
                let l_val = self.fresh_var();
                ir.push_str(&format!("  {} = load double, double* {}\n", l_val, l_elem_ptr));
                let r_elem_ptr = self.fresh_var();
                ir.push_str(&format!("  {} = getelementptr double, double* {}, i64 {}\n", r_elem_ptr, r_ptr, idx));
                let r_val = self.fresh_var();
                ir.push_str(&format!("  {} = load double, double* {}\n", r_val, r_elem_ptr));
                
                let result = self.fresh_var();
                match op {
                    "add" => ir.push_str(&format!("  {} = fadd{} double {}, {}\n", result, fmf, l_val, r_val)),
                    "sub" => ir.push_str(&format!("  {} = fsub{} double {}, {}\n", result, fmf, l_val, r_val)),
                    "mul" => ir.push_str(&format!("  {} = fmul{} double {}, {}\n", result, fmf, l_val, r_val)),
                    "div" => ir.push_str(&format!("  {} = fdiv{} double {}, {}\n", result, fmf, l_val, r_val)),
                    "pow" => ir.push_str(&format!("  {} = call double @llvm.pow.f64(double {}, double {})\n", result, l_val, r_val)),
                    _ => return Err(format!("Unsupported binary tensor op: {}", op)),
                }
                
                let out_elem_ptr = self.fresh_var();
                ir.push_str(&format!("  {} = getelementptr double, double* {}, i64 {}\n", out_elem_ptr, out_ptr, idx));
                ir.push_str(&format!("  store double {}, double* {}\n", result, out_elem_ptr));
                
                let next_idx = self.fresh_var();
                ir.push_str(&format!("  {} = add i64 {}, 1\n", next_idx, idx));
                ir.push_str(&format!("  store i64 {}, i64* {}\n", next_idx, loop_alloca));
                ir.push_str(&format!("  br label %{}\n", loop_header));
                
                ir.push_str(&format!("{}:\n", loop_end));
                
                Ok(LLVMValue::Tensor { data_ptr: out_ptr, shape: l_shape.clone() })
            }
            // Scalar op Tensor (broadcast scalar)
            (LLVMValue::Scalar(s_var), LLVMValue::Tensor { data_ptr: t_ptr, shape }) |
            (LLVMValue::Tensor { data_ptr: t_ptr, shape }, LLVMValue::Scalar(s_var)) => {
                let is_left_scalar = matches!(left, LLVMValue::Scalar(_));
                let size: usize = shape.iter().product();
                let out_ptr = self.gen_tensor_alloc(ir, size);
                
                let loop_alloca = self.fresh_var();
                let loop_header = self.fresh_label("broadcast_header_");
                let loop_body = self.fresh_label("broadcast_body_");
                let loop_end = self.fresh_label("broadcast_end_");
                
                ir.push_str(&format!("  {} = alloca i64\n", loop_alloca));
                ir.push_str(&format!("  store i64 0, i64* {}\n", loop_alloca));
                ir.push_str(&format!("  br label %{}\n", loop_header));
                
                ir.push_str(&format!("{}:\n", loop_header));
                let idx = self.fresh_var();
                ir.push_str(&format!("  {} = load i64, i64* {}\n", idx, loop_alloca));
                let cond = self.fresh_var();
                ir.push_str(&format!("  {} = icmp slt i64 {}, {}\n", cond, idx, size));
                ir.push_str(&format!("  br i1 {}, label %{}, label %{}\n", cond, loop_body, loop_end));
                
                ir.push_str(&format!("{}:\n", loop_body));
                let t_elem_ptr = self.fresh_var();
                ir.push_str(&format!("  {} = getelementptr double, double* {}, i64 {}\n", t_elem_ptr, t_ptr, idx));
                let t_val = self.fresh_var();
                ir.push_str(&format!("  {} = load double, double* {}\n", t_val, t_elem_ptr));
                
                let (l_val, r_val) = if is_left_scalar {
                    (s_var.clone(), t_val)
                } else {
                    (t_val, s_var.clone())
                };
                
                let result = self.fresh_var();
                match op {
                    "add" => ir.push_str(&format!("  {} = fadd{} double {}, {}\n", result, fmf, l_val, r_val)),
                    "sub" => ir.push_str(&format!("  {} = fsub{} double {}, {}\n", result, fmf, l_val, r_val)),
                    "mul" => ir.push_str(&format!("  {} = fmul{} double {}, {}\n", result, fmf, l_val, r_val)),
                    "div" => ir.push_str(&format!("  {} = fdiv{} double {}, {}\n", result, fmf, l_val, r_val)),
                    "pow" => ir.push_str(&format!("  {} = call double @llvm.pow.f64(double {}, double {})\n", result, l_val, r_val)),
                    _ => return Err(format!("Unsupported binary tensor op: {}", op)),
                }
                
                let out_elem_ptr = self.fresh_var();
                ir.push_str(&format!("  {} = getelementptr double, double* {}, i64 {}\n", out_elem_ptr, out_ptr, idx));
                ir.push_str(&format!("  store double {}, double* {}\n", result, out_elem_ptr));
                
                let next_idx = self.fresh_var();
                ir.push_str(&format!("  {} = add i64 {}, 1\n", next_idx, idx));
                ir.push_str(&format!("  store i64 {}, i64* {}\n", next_idx, loop_alloca));
                ir.push_str(&format!("  br label %{}\n", loop_header));
                
                ir.push_str(&format!("{}:\n", loop_end));
                
                Ok(LLVMValue::Tensor { data_ptr: out_ptr, shape: shape.clone() })
            }
            // Scalar op Scalar - should not reach here normally
            (LLVMValue::Scalar(l), LLVMValue::Scalar(r)) => {
                let result = self.fresh_var();
                match op {
                    "add" => ir.push_str(&format!("  {} = fadd{} double {}, {}\n", result, fmf, l, r)),
                    "sub" => ir.push_str(&format!("  {} = fsub{} double {}, {}\n", result, fmf, l, r)),
                    "mul" => ir.push_str(&format!("  {} = fmul{} double {}, {}\n", result, fmf, l, r)),
                    "div" => ir.push_str(&format!("  {} = fdiv{} double {}, {}\n", result, fmf, l, r)),
                    "pow" => ir.push_str(&format!("  {} = call double @llvm.pow.f64(double {}, double {})\n", result, l, r)),
                    _ => return Err(format!("Unsupported binary op: {}", op)),
                }
                Ok(LLVMValue::Scalar(result))
            }
        }
    }

    /// Generate sum reduction on a tensor
    fn gen_tensor_sum(&mut self, ir: &mut String, input: &LLVMValue) -> Result<LLVMValue, String> {
        let (in_ptr, shape) = match input {
            LLVMValue::Tensor { data_ptr, shape } => (data_ptr.clone(), shape.clone()),
            LLVMValue::Scalar(s) => return Ok(LLVMValue::Scalar(s.clone())),
        };
        
        let size: usize = shape.iter().product();
        
        let sum_alloca = self.fresh_var();
        let loop_alloca = self.fresh_var();
        let loop_header = self.fresh_label("sum_header_");
        let loop_body = self.fresh_label("sum_body_");
        let loop_end = self.fresh_label("sum_end_");
        
        ir.push_str(&format!("  {} = alloca double\n", sum_alloca));
        ir.push_str(&format!("  store double 0.0, double* {}\n", sum_alloca));
        ir.push_str(&format!("  {} = alloca i64\n", loop_alloca));
        ir.push_str(&format!("  store i64 0, i64* {}\n", loop_alloca));
        ir.push_str(&format!("  br label %{}\n", loop_header));
        
        ir.push_str(&format!("{}:\n", loop_header));
        let idx = self.fresh_var();
        ir.push_str(&format!("  {} = load i64, i64* {}\n", idx, loop_alloca));
        let cond = self.fresh_var();
        ir.push_str(&format!("  {} = icmp slt i64 {}, {}\n", cond, idx, size));
        ir.push_str(&format!("  br i1 {}, label %{}, label %{}\n", cond, loop_body, loop_end));
        
        ir.push_str(&format!("{}:\n", loop_body));
        let elem_ptr = self.fresh_var();
        ir.push_str(&format!("  {} = getelementptr double, double* {}, i64 {}\n", elem_ptr, in_ptr, idx));
        let val = self.fresh_var();
        ir.push_str(&format!("  {} = load double, double* {}\n", val, elem_ptr));
        let cur_sum = self.fresh_var();
        ir.push_str(&format!("  {} = load double, double* {}\n", cur_sum, sum_alloca));
        let new_sum = self.fresh_var();
        ir.push_str(&format!("  {} = fadd double {}, {}\n", new_sum, cur_sum, val));
        ir.push_str(&format!("  store double {}, double* {}\n", new_sum, sum_alloca));
        
        let next_idx = self.fresh_var();
        ir.push_str(&format!("  {} = add i64 {}, 1\n", next_idx, idx));
        ir.push_str(&format!("  store i64 {}, i64* {}\n", next_idx, loop_alloca));
        ir.push_str(&format!("  br label %{}\n", loop_header));
        
        ir.push_str(&format!("{}:\n", loop_end));
        let result = self.fresh_var();
        ir.push_str(&format!("  {} = load double, double* {}\n", result, sum_alloca));
        
        Ok(LLVMValue::Scalar(result))
    }

    /// Generate mean reduction on a tensor
    fn gen_tensor_mean(&mut self, ir: &mut String, input: &LLVMValue) -> Result<LLVMValue, String> {
        let size = match input {
            LLVMValue::Tensor { shape, .. } => shape.iter().product::<usize>(),
            LLVMValue::Scalar(s) => return Ok(LLVMValue::Scalar(s.clone())),
        };
        
        let sum = self.gen_tensor_sum(ir, input)?;
        let sum_var = match sum {
            LLVMValue::Scalar(s) => s,
            _ => return Err("Expected scalar from sum".to_string()),
        };
        
        let result = self.fresh_var();
        ir.push_str(&format!("  {} = fdiv double {}, {}\n", result, sum_var, self.fmt_f64(size as f64)));
        
        Ok(LLVMValue::Scalar(result))
    }

    /// Generate matrix multiplication: C = A @ B where A is [M, K] and B is [K, N]
    fn gen_matmul(&mut self, ir: &mut String, a: &LLVMValue, b: &LLVMValue) -> Result<LLVMValue, String> {
        let (a_ptr, a_shape) = match a {
            LLVMValue::Tensor { data_ptr, shape } => (data_ptr.clone(), shape.clone()),
            _ => return Err("matmul expects tensor input".to_string()),
        };
        let (b_ptr, b_shape) = match b {
            LLVMValue::Tensor { data_ptr, shape } => (data_ptr.clone(), shape.clone()),
            _ => return Err("matmul expects tensor input".to_string()),
        };
        
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err("matmul expects rank-2 tensors".to_string());
        }
        
        let m = a_shape[0];
        let k = a_shape[1];
        let k2 = b_shape[0];
        let n = b_shape[1];
        
        if k != k2 {
            return Err(format!("matmul inner dimensions mismatch: {} vs {}", k, k2));
        }
        
        let out_size = m * n;
        let out_ptr = self.gen_tensor_alloc(ir, out_size);
        
        // Triple nested loop: for i in 0..M, for j in 0..N, for p in 0..K
        let i_alloca = self.fresh_var();
        let j_alloca = self.fresh_var();
        let p_alloca = self.fresh_var();
        let sum_alloca = self.fresh_var();
        
        ir.push_str(&format!("  {} = alloca i64\n", i_alloca));
        ir.push_str(&format!("  {} = alloca i64\n", j_alloca));
        ir.push_str(&format!("  {} = alloca i64\n", p_alloca));
        ir.push_str(&format!("  {} = alloca double\n", sum_alloca));
        
        let i_header = self.fresh_label("mm_i_header_");
        let i_body = self.fresh_label("mm_i_body_");
        let i_end = self.fresh_label("mm_i_end_");
        let j_header = self.fresh_label("mm_j_header_");
        let j_body = self.fresh_label("mm_j_body_");
        let j_end = self.fresh_label("mm_j_end_");
        let p_header = self.fresh_label("mm_p_header_");
        let p_body = self.fresh_label("mm_p_body_");
        let p_end = self.fresh_label("mm_p_end_");
        
        // Initialize i = 0
        ir.push_str(&format!("  store i64 0, i64* {}\n", i_alloca));
        ir.push_str(&format!("  br label %{}\n", i_header));
        
        // i loop header
        ir.push_str(&format!("{}:\n", i_header));
        let i_val = self.fresh_var();
        ir.push_str(&format!("  {} = load i64, i64* {}\n", i_val, i_alloca));
        let i_cond = self.fresh_var();
        ir.push_str(&format!("  {} = icmp slt i64 {}, {}\n", i_cond, i_val, m));
        ir.push_str(&format!("  br i1 {}, label %{}, label %{}\n", i_cond, i_body, i_end));
        
        // i loop body - initialize j = 0
        ir.push_str(&format!("{}:\n", i_body));
        ir.push_str(&format!("  store i64 0, i64* {}\n", j_alloca));
        ir.push_str(&format!("  br label %{}\n", j_header));
        
        // j loop header
        ir.push_str(&format!("{}:\n", j_header));
        let i_val2 = self.fresh_var();
        ir.push_str(&format!("  {} = load i64, i64* {}\n", i_val2, i_alloca));
        let j_val = self.fresh_var();
        ir.push_str(&format!("  {} = load i64, i64* {}\n", j_val, j_alloca));
        let j_cond = self.fresh_var();
        ir.push_str(&format!("  {} = icmp slt i64 {}, {}\n", j_cond, j_val, n));
        ir.push_str(&format!("  br i1 {}, label %{}, label %{}\n", j_cond, j_body, j_end));
        
        // j loop body - initialize p = 0 and sum = 0
        ir.push_str(&format!("{}:\n", j_body));
        ir.push_str(&format!("  store i64 0, i64* {}\n", p_alloca));
        ir.push_str(&format!("  store double 0.0, double* {}\n", sum_alloca));
        ir.push_str(&format!("  br label %{}\n", p_header));
        
        // p loop header
        ir.push_str(&format!("{}:\n", p_header));
        let i_val3 = self.fresh_var();
        ir.push_str(&format!("  {} = load i64, i64* {}\n", i_val3, i_alloca));
        let j_val2 = self.fresh_var();
        ir.push_str(&format!("  {} = load i64, i64* {}\n", j_val2, j_alloca));
        let p_val = self.fresh_var();
        ir.push_str(&format!("  {} = load i64, i64* {}\n", p_val, p_alloca));
        let p_cond = self.fresh_var();
        ir.push_str(&format!("  {} = icmp slt i64 {}, {}\n", p_cond, p_val, k));
        ir.push_str(&format!("  br i1 {}, label %{}, label %{}\n", p_cond, p_body, p_end));
        
        // p loop body: sum += A[i,p] * B[p,j]
        ir.push_str(&format!("{}:\n", p_body));
        let i_val4 = self.fresh_var();
        ir.push_str(&format!("  {} = load i64, i64* {}\n", i_val4, i_alloca));
        let j_val3 = self.fresh_var();
        ir.push_str(&format!("  {} = load i64, i64* {}\n", j_val3, j_alloca));
        let p_val2 = self.fresh_var();
        ir.push_str(&format!("  {} = load i64, i64* {}\n", p_val2, p_alloca));
        
        // A index = i * K + p
        let a_idx1 = self.fresh_var();
        ir.push_str(&format!("  {} = mul i64 {}, {}\n", a_idx1, i_val4, k));
        let a_idx = self.fresh_var();
        ir.push_str(&format!("  {} = add i64 {}, {}\n", a_idx, a_idx1, p_val2));
        let a_elem_ptr = self.fresh_var();
        ir.push_str(&format!("  {} = getelementptr double, double* {}, i64 {}\n", a_elem_ptr, a_ptr, a_idx));
        let a_val = self.fresh_var();
        ir.push_str(&format!("  {} = load double, double* {}\n", a_val, a_elem_ptr));
        
        // B index = p * N + j
        let b_idx1 = self.fresh_var();
        ir.push_str(&format!("  {} = mul i64 {}, {}\n", b_idx1, p_val2, n));
        let b_idx = self.fresh_var();
        ir.push_str(&format!("  {} = add i64 {}, {}\n", b_idx, b_idx1, j_val3));
        let b_elem_ptr = self.fresh_var();
        ir.push_str(&format!("  {} = getelementptr double, double* {}, i64 {}\n", b_elem_ptr, b_ptr, b_idx));
        let b_val = self.fresh_var();
        ir.push_str(&format!("  {} = load double, double* {}\n", b_val, b_elem_ptr));
        
        // product and accumulate
        let prod = self.fresh_var();
        ir.push_str(&format!("  {} = fmul double {}, {}\n", prod, a_val, b_val));
        let cur_sum = self.fresh_var();
        ir.push_str(&format!("  {} = load double, double* {}\n", cur_sum, sum_alloca));
        let new_sum = self.fresh_var();
        ir.push_str(&format!("  {} = fadd double {}, {}\n", new_sum, cur_sum, prod));
        ir.push_str(&format!("  store double {}, double* {}\n", new_sum, sum_alloca));
        
        // increment p
        let p_next = self.fresh_var();
        ir.push_str(&format!("  {} = add i64 {}, 1\n", p_next, p_val2));
        ir.push_str(&format!("  store i64 {}, i64* {}\n", p_next, p_alloca));
        ir.push_str(&format!("  br label %{}\n", p_header));
        
        // p loop end - store result to C[i,j]
        ir.push_str(&format!("{}:\n", p_end));
        let i_val5 = self.fresh_var();
        ir.push_str(&format!("  {} = load i64, i64* {}\n", i_val5, i_alloca));
        let j_val4 = self.fresh_var();
        ir.push_str(&format!("  {} = load i64, i64* {}\n", j_val4, j_alloca));
        let c_idx1 = self.fresh_var();
        ir.push_str(&format!("  {} = mul i64 {}, {}\n", c_idx1, i_val5, n));
        let c_idx = self.fresh_var();
        ir.push_str(&format!("  {} = add i64 {}, {}\n", c_idx, c_idx1, j_val4));
        let c_elem_ptr = self.fresh_var();
        ir.push_str(&format!("  {} = getelementptr double, double* {}, i64 {}\n", c_elem_ptr, out_ptr, c_idx));
        let final_sum = self.fresh_var();
        ir.push_str(&format!("  {} = load double, double* {}\n", final_sum, sum_alloca));
        ir.push_str(&format!("  store double {}, double* {}\n", final_sum, c_elem_ptr));
        
        // increment j
        let j_next = self.fresh_var();
        ir.push_str(&format!("  {} = add i64 {}, 1\n", j_next, j_val4));
        ir.push_str(&format!("  store i64 {}, i64* {}\n", j_next, j_alloca));
        ir.push_str(&format!("  br label %{}\n", j_header));
        
        // j loop end
        ir.push_str(&format!("{}:\n", j_end));
        let i_val6 = self.fresh_var();
        ir.push_str(&format!("  {} = load i64, i64* {}\n", i_val6, i_alloca));
        let i_next = self.fresh_var();
        ir.push_str(&format!("  {} = add i64 {}, 1\n", i_next, i_val6));
        ir.push_str(&format!("  store i64 {}, i64* {}\n", i_next, i_alloca));
        ir.push_str(&format!("  br label %{}\n", i_header));
        
        // i loop end
        ir.push_str(&format!("{}:\n", i_end));
        
        Ok(LLVMValue::Tensor { data_ptr: out_ptr, shape: vec![m, n] })
    }

    /// Generate LLVM IR for a computational graph, returning the value of a specific node
    pub fn generate_with_return(&mut self, graph: &ComputationalGraph, return_node: Option<NodeId>) -> Result<String, String> {
        self.generate_internal(graph, return_node)
    }

    /// Generate LLVM IR for a computational graph
    pub fn generate(&mut self, graph: &ComputationalGraph) -> Result<String, String> {
        self.generate_internal(graph, None)
    }
    
    fn generate_internal(&mut self, graph: &ComputationalGraph, return_node: Option<NodeId>) -> Result<String, String> {
        self.extern_decls.clear();
        self.global_constants.clear();
        self.allocated_tensors.clear();
        
        let mut body_ir = String::new();
        let mut var_map: HashMap<NodeId, LLVMValue> = HashMap::new();
        let nodes = graph.nodes();

        // Sort nodes by their numeric id for deterministic order
        let mut node_ids: Vec<NodeId> = nodes.keys().copied().collect();
        node_ids.sort_by_key(|id| id.index());

        let mut last_value: Option<LLVMValue> = None;

        // Process nodes in deterministic order
        for node_id in node_ids {
            let node = &nodes[&node_id];

            match &node.node_type {
                NodeType::Constant(Value::Scalar(val)) => {
                    let var = self.fresh_var();
                    body_ir.push_str(&format!("  {} = fadd double {}, 0.0\n", var, self.fmt_f64(*val)));
                    let llvm_val = LLVMValue::Scalar(var);
                    var_map.insert(node_id, llvm_val.clone());
                    last_value = Some(llvm_val);
                }
                NodeType::Constant(Value::Tensor(tensor)) => {
                    let global_name = self.create_tensor_global(&tensor.data, "const");
                    let data_ptr = self.gen_tensor_copy_from_global(&mut body_ir, &global_name, tensor.data.len());
                    let llvm_val = LLVMValue::Tensor { data_ptr, shape: tensor.shape.clone() };
                    var_map.insert(node_id, llvm_val.clone());
                    last_value = Some(llvm_val);
                }
                NodeType::Learnable(_) | NodeType::Variable(_) => {
                    let llvm_val = match node.value.clone() {
                        Some(Value::Scalar(v)) => {
                            let var = self.fresh_var();
                            body_ir.push_str(&format!("  {} = fadd double {}, 0.0\n", var, self.fmt_f64(v)));
                            LLVMValue::Scalar(var)
                        }
                        Some(Value::Tensor(tensor)) => {
                            let global_name = self.create_tensor_global(&tensor.data, "param");
                            let data_ptr = self.gen_tensor_copy_from_global(&mut body_ir, &global_name, tensor.data.len());
                            LLVMValue::Tensor { data_ptr, shape: tensor.shape.clone() }
                        }
                        None => {
                            let var = self.fresh_var();
                            body_ir.push_str(&format!("  {} = fadd double 0.0, 0.0\n", var));
                            LLVMValue::Scalar(var)
                        }
                    };
                    var_map.insert(node_id, llvm_val.clone());
                    last_value = Some(llvm_val);
                }
                NodeType::BinaryOp(op_str) => {
                    if node.inputs.len() != 2 {
                        return Err("Binary operation requires 2 inputs".to_string());
                    }
                    let left_val = var_map.get(&node.inputs[0]).ok_or("Left operand not found")?.clone();
                    let right_val = var_map.get(&node.inputs[1]).ok_or("Right operand not found")?.clone();

                    // Check if we need tensor operations
                    let needs_tensor_op = matches!((&left_val, &right_val), 
                        (LLVMValue::Tensor { .. }, _) | (_, LLVMValue::Tensor { .. }));
                    
                    if needs_tensor_op && matches!(op_str.as_str(), "add" | "sub" | "mul" | "div" | "pow") {
                        let result = self.gen_tensor_binary_op(&mut body_ir, &left_val, &right_val, op_str)?;
                        var_map.insert(node_id, result.clone());
                        last_value = Some(result);
                    } else {
                        // Scalar operations
                        let left_var = match &left_val {
                            LLVMValue::Scalar(s) => s.clone(),
                            _ => return Err("Expected scalar for comparison op".to_string()),
                        };
                        let right_var = match &right_val {
                            LLVMValue::Scalar(s) => s.clone(),
                            _ => return Err("Expected scalar for comparison op".to_string()),
                        };
                        
                        let fmf = if self.fast_math { " fast" } else { "" };
                        let result = match op_str.as_str() {
                            "add" => {
                                let v = self.fresh_var();
                                body_ir.push_str(&format!("  {} = fadd{} double {}, {}\n", v, fmf, left_var, right_var));
                                LLVMValue::Scalar(v)
                            }
                            "sub" => {
                                let v = self.fresh_var();
                                body_ir.push_str(&format!("  {} = fsub{} double {}, {}\n", v, fmf, left_var, right_var));
                                LLVMValue::Scalar(v)
                            }
                            "mul" => {
                                let v = self.fresh_var();
                                body_ir.push_str(&format!("  {} = fmul{} double {}, {}\n", v, fmf, left_var, right_var));
                                LLVMValue::Scalar(v)
                            }
                            "div" => {
                                let v = self.fresh_var();
                                body_ir.push_str(&format!("  {} = fdiv{} double {}, {}\n", v, fmf, left_var, right_var));
                                LLVMValue::Scalar(v)
                            }
                            "mod" => {
                                let v = self.fresh_var();
                                body_ir.push_str(&format!("  {} = frem double {}, {}\n", v, left_var, right_var));
                                LLVMValue::Scalar(v)
                            }
                            "pow" => {
                                let v = self.fresh_var();
                                body_ir.push_str(&format!("  {} = call double @llvm.pow.f64(double {}, double {})\n", v, left_var, right_var));
                                LLVMValue::Scalar(v)
                            }
                            "eq" | "ne" | "lt" | "gt" | "le" | "ge" => {
                                let pred = match op_str.as_str() {
                                    "eq" => "oeq", "ne" => "one", "lt" => "olt",
                                    "gt" => "ogt", "le" => "ole", "ge" => "oge",
                                    _ => unreachable!(),
                                };
                                let cmp = self.fresh_var();
                                body_ir.push_str(&format!("  {} = fcmp {} double {}, {}\n", cmp, pred, left_var, right_var));
                                let result_var = self.fresh_var();
                                body_ir.push_str(&format!("  {} = uitofp i1 {} to double\n", result_var, cmp));
                                LLVMValue::Scalar(result_var)
                            }
                            "and" => {
                                let l0 = self.fresh_var();
                                body_ir.push_str(&format!("  {} = fcmp one double {}, 0.0\n", l0, left_var));
                                let r0 = self.fresh_var();
                                body_ir.push_str(&format!("  {} = fcmp one double {}, 0.0\n", r0, right_var));
                                let pred = self.fresh_var();
                                body_ir.push_str(&format!("  {} = and i1 {}, {}\n", pred, l0, r0));
                                let result_var = self.fresh_var();
                                body_ir.push_str(&format!("  {} = uitofp i1 {} to double\n", result_var, pred));
                                LLVMValue::Scalar(result_var)
                            }
                            "or" => {
                                let l0 = self.fresh_var();
                                body_ir.push_str(&format!("  {} = fcmp one double {}, 0.0\n", l0, left_var));
                                let r0 = self.fresh_var();
                                body_ir.push_str(&format!("  {} = fcmp one double {}, 0.0\n", r0, right_var));
                                let pred = self.fresh_var();
                                body_ir.push_str(&format!("  {} = or i1 {}, {}\n", pred, l0, r0));
                                let result_var = self.fresh_var();
                                body_ir.push_str(&format!("  {} = uitofp i1 {} to double\n", result_var, pred));
                                LLVMValue::Scalar(result_var)
                            }
                            _ => return Err(format!("Unsupported binary operator: {}", op_str)),
                        };
                        var_map.insert(node_id, result.clone());
                        last_value = Some(result);
                    }
                }
                NodeType::UnaryOp(op_str) => {
                    if node.inputs.len() != 1 {
                        return Err("Unary operation requires 1 input".to_string());
                    }
                    let arg_val = var_map.get(&node.inputs[0]).ok_or("Argument not found")?.clone();

                    let result = match &arg_val {
                        LLVMValue::Tensor { .. } => {
                            self.gen_tensor_unary_op(&mut body_ir, &arg_val, op_str)?
                        }
                        LLVMValue::Scalar(arg_var) => {
                            match op_str.as_str() {
                                "neg" => {
                                    let v = self.fresh_var();
                                    body_ir.push_str(&format!("  {} = fsub double 0.0, {}\n", v, arg_var));
                                    LLVMValue::Scalar(v)
                                }
                                "not" => {
                                    return Err("NOT operator not supported in numeric code generation".to_string());
                                }
                                _ => return Err(format!("Unsupported unary operator: {}", op_str)),
                            }
                        }
                    };
                    var_map.insert(node_id, result.clone());
                    last_value = Some(result);
                }
                NodeType::FunctionCall(func_name) => {
                    let result = match func_name.as_str() {
                        "sigmoid" | "relu" | "tanh" | "exp" | "log" | "sqrt" | "sin" | "cos" => {
                            if node.inputs.len() != 1 {
                                return Err(format!("{} expects 1 argument", func_name));
                            }
                            let arg_val = var_map.get(&node.inputs[0]).ok_or("Argument not found")?.clone();
                            
                            match &arg_val {
                                LLVMValue::Tensor { .. } => {
                                    self.gen_tensor_unary_op(&mut body_ir, &arg_val, func_name)?
                                }
                                LLVMValue::Scalar(arg_var) => {
                                    match func_name.as_str() {
                                        "sigmoid" => {
                                            let neg_var = self.fresh_var();
                                            body_ir.push_str(&format!("  {} = fsub double 0.0, {}\n", neg_var, arg_var));
                                            let exp_var = self.fresh_var();
                                            body_ir.push_str(&format!("  {} = call double @llvm.exp.f64(double {})\n", exp_var, neg_var));
                                            let one_add = self.fresh_var();
                                            body_ir.push_str(&format!("  {} = fadd double 1.0, {}\n", one_add, exp_var));
                                            let result_var = self.fresh_var();
                                            body_ir.push_str(&format!("  {} = fdiv double 1.0, {}\n", result_var, one_add));
                                            LLVMValue::Scalar(result_var)
                                        }
                                        "relu" => {
                                            let v = self.fresh_var();
                                            body_ir.push_str(&format!("  {} = call double @llvm.maxnum.f64(double {}, double 0.0)\n", v, arg_var));
                                            LLVMValue::Scalar(v)
                                        }
                                        _ => {
                                            let intrinsic = match func_name.as_str() {
                                                "sin" => "llvm.sin.f64",
                                                "cos" => "llvm.cos.f64",
                                                "exp" => "llvm.exp.f64",
                                                "log" => "llvm.log.f64",
                                                "sqrt" => "llvm.sqrt.f64",
                                                "tanh" => "llvm.tanh.f64",
                                                _ => unreachable!(),
                                            };
                                            let v = self.fresh_var();
                                            body_ir.push_str(&format!("  {} = call double @{}(double {})\n", v, intrinsic, arg_var));
                                            LLVMValue::Scalar(v)
                                        }
                                    }
                                }
                            }
                        }
                        "sum" => {
                            if node.inputs.len() != 1 {
                                return Err("sum expects 1 argument".to_string());
                            }
                            let arg_val = var_map.get(&node.inputs[0]).ok_or("Argument not found")?.clone();
                            self.gen_tensor_sum(&mut body_ir, &arg_val)?
                        }
                        "mean" => {
                            if node.inputs.len() != 1 {
                                return Err("mean expects 1 argument".to_string());
                            }
                            let arg_val = var_map.get(&node.inputs[0]).ok_or("Argument not found")?.clone();
                            self.gen_tensor_mean(&mut body_ir, &arg_val)?
                        }
                        "matmul" => {
                            if node.inputs.len() != 2 {
                                return Err("matmul expects 2 arguments".to_string());
                            }
                            let a_val = var_map.get(&node.inputs[0]).ok_or("Arg a not found")?.clone();
                            let b_val = var_map.get(&node.inputs[1]).ok_or("Arg b not found")?.clone();
                            self.gen_matmul(&mut body_ir, &a_val, &b_val)?
                        }
                        "abs" => {
                            if node.inputs.len() != 1 { return Err("abs expects 1 argument".to_string()); }
                            let arg_val = var_map.get(&node.inputs[0]).ok_or("Argument not found")?.clone();
                            match &arg_val {
                                LLVMValue::Scalar(arg_var) => {
                                    let neg = self.fresh_var();
                                    body_ir.push_str(&format!("  {} = fsub double 0.0, {}\n", neg, arg_var));
                                    let v = self.fresh_var();
                                    body_ir.push_str(&format!("  {} = call double @llvm.maxnum.f64(double {}, double {})\n", v, arg_var, neg));
                                    LLVMValue::Scalar(v)
                                }
                                LLVMValue::Tensor { .. } => {
                                    // TODO: implement tensor abs
                                    return Err("abs on tensor not yet supported".to_string());
                                }
                            }
                        }
                        "floor" | "ceil" => {
                            if node.inputs.len() != 1 { return Err(format!("{} expects 1 argument", func_name)); }
                            let arg_val = var_map.get(&node.inputs[0]).ok_or("Argument not found")?.clone();
                            match &arg_val {
                                LLVMValue::Scalar(arg_var) => {
                                    let intrinsic = if func_name == "floor" { "llvm.floor.f64" } else { "llvm.ceil.f64" };
                                    let v = self.fresh_var();
                                    body_ir.push_str(&format!("  {} = call double @{}(double {})\n", v, intrinsic, arg_var));
                                    LLVMValue::Scalar(v)
                                }
                                _ => return Err(format!("{} on tensor not yet supported", func_name)),
                            }
                        }
                        "rand" => {
                            if node.inputs.len() != 0 { return Err("rand expects 0 arguments".to_string()); }
                            self.extern_decls.insert("declare double @drand48()".to_string());
                            let v = self.fresh_var();
                            body_ir.push_str(&format!("  {} = call double @drand48()\n", v));
                            LLVMValue::Scalar(v)
                        }
                        "print" => {
                            // print is a no-op in compiled code for now
                            if node.inputs.len() != 1 { return Err("print expects 1 argument".to_string()); }
                            let arg_val = var_map.get(&node.inputs[0]).ok_or("Argument not found")?.clone();
                            arg_val
                        }
                        _ => {
                            // First check if this node has a pre-computed value (e.g., rand_normal_tensor, he_init, etc.)
                            // These are evaluated during forward_pass() and their results are stored in node.value
                            if let Some(val) = &node.value {
                                match val {
                                    Value::Tensor(tensor) => {
                                        let global_name = self.create_tensor_global(&tensor.data, "precomputed");
                                        let data_ptr = self.gen_tensor_copy_from_global(&mut body_ir, &global_name, tensor.data.len());
                                        LLVMValue::Tensor { data_ptr, shape: tensor.shape.clone() }
                                    }
                                    Value::Scalar(v) => {
                                        let var = self.fresh_var();
                                        body_ir.push_str(&format!("  {} = fadd double {}, 0.0\n", var, self.fmt_f64(*v)));
                                        LLVMValue::Scalar(var)
                                    }
                                }
                            } else {
                                // Unknown function without pre-computed value - try to call as external
                                let mut arg_scalars = Vec::new();
                                for inp in &node.inputs {
                                    let val = var_map.get(inp).ok_or("Argument not found")?;
                                    match val {
                                        LLVMValue::Scalar(s) => arg_scalars.push(s.clone()),
                                        _ => return Err(format!("Cannot pass tensor to external function {}", func_name)),
                                    }
                                }
                                let params_decl: Vec<&str> = (0..arg_scalars.len()).map(|_| "double").collect();
                                self.extern_decls.insert(format!("declare double @{}({})", func_name, params_decl.join(", ")));
                                let params: Vec<String> = arg_scalars.iter().map(|a| format!("double {}", a)).collect();
                                let v = self.fresh_var();
                                body_ir.push_str(&format!("  {} = call double @{}({})\n", v, func_name, params.join(", ")));
                                LLVMValue::Scalar(v)
                            }
                        }
                    };
                    var_map.insert(node_id, result.clone());
                    last_value = Some(result);
                }
                NodeType::HeapTensor(_name) => {
                    let llvm_val = match node.value.clone() {
                        Some(Value::Tensor(tensor)) => {
                            let global_name = self.create_tensor_global(&tensor.data, "heap");
                            let data_ptr = self.gen_tensor_copy_from_global(&mut body_ir, &global_name, tensor.data.len());
                            LLVMValue::Tensor { data_ptr, shape: tensor.shape.clone() }
                        }
                        Some(Value::Scalar(v)) => {
                            let var = self.fresh_var();
                            body_ir.push_str(&format!("  {} = fadd double {}, 0.0\n", var, self.fmt_f64(v)));
                            LLVMValue::Scalar(var)
                        }
                        None => {
                            let var = self.fresh_var();
                            body_ir.push_str(&format!("  {} = fadd double 0.0, 0.0\n", var));
                            LLVMValue::Scalar(var)
                        }
                    };
                    var_map.insert(node_id, llvm_val.clone());
                    last_value = Some(llvm_val);
                }
                NodeType::FreedTensor(name) => {
                    return Err(format!("Cannot generate code for freed tensor '{}'", name));
                }
            }
        }

        // Build the final IR
        let mut ir = String::new();
        
        // LLVM module header
        ir.push_str("; NOMA LLVM IR Generated Code (with tensor support)\n");
        ir.push_str("source_filename = \"noma_generated\"\n");
        ir.push_str("target datalayout = \"e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128\"\n");
        ir.push_str("target triple = \"x86_64-unknown-linux-gnu\"\n\n");
        
        // Global tensor constants
        for global in &self.global_constants {
            ir.push_str(global);
        }
        if !self.global_constants.is_empty() {
            ir.push_str("\n");
        }
        
        // Declare external functions
        ir.push_str("declare i32 @printf(i8*, ...)\n");
        ir.push_str("@.str = private unnamed_addr constant [4 x i8] c\"%f\\0A\\00\", align 1\n\n");

        // Generate compute function
        ir.push_str("define double @compute() {\nentry:\n");
        ir.push_str(&body_ir);

        // Return value - extract scalar if it's a tensor (e.g., return first element or sum)
        let ret_val = if let Some(ret_node) = return_node {
            var_map.get(&ret_node).cloned()
        } else {
            last_value.clone()
        };
        
        match ret_val {
            Some(LLVMValue::Scalar(s)) => {
                ir.push_str(&format!("  ret double {}\n", s));
            }
            Some(LLVMValue::Tensor { data_ptr, shape }) => {
                // For tensor return, we return the first element (or could return sum)
                let first_elem_ptr = self.fresh_var();
                ir.push_str(&format!("  {} = getelementptr double, double* {}, i64 0\n", first_elem_ptr, data_ptr));
                let first_elem = self.fresh_var();
                ir.push_str(&format!("  {} = load double, double* {}\n", first_elem, first_elem_ptr));
                ir.push_str(&format!("  ret double {}\n", first_elem));
            }
            None => {
                ir.push_str("  ret double 0.0\n");
            }
        }

        ir.push_str("}\n\n");

        // Declare LLVM intrinsics
        ir.push_str("declare double @llvm.pow.f64(double, double)\n");
        ir.push_str("declare double @llvm.exp.f64(double)\n");
        ir.push_str("declare double @llvm.maxnum.f64(double, double)\n");
        ir.push_str("declare double @llvm.sin.f64(double)\n");
        ir.push_str("declare double @llvm.cos.f64(double)\n");
        ir.push_str("declare double @llvm.log.f64(double)\n");
        ir.push_str("declare double @llvm.sqrt.f64(double)\n");
        ir.push_str("declare double @llvm.tanh.f64(double)\n");
        ir.push_str("declare double @llvm.floor.f64(double)\n");
        ir.push_str("declare double @llvm.ceil.f64(double)\n");

        // External function declarations
        for decl in &self.extern_decls {
            ir.push_str(decl);
            ir.push_str("\n");
        }

        Ok(ir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llvm_constant() {
        let graph = ComputationalGraph::new();
        let mut codegen = LLVMCodegen::new();
        let ir = codegen.generate(&graph).expect("IR generation failed");
        
        assert!(ir.contains("source_filename = \"noma_generated\""));
        assert!(ir.contains("define double @compute()"));
        assert!(ir.contains("ret double"));
    }

    #[test]
    fn test_llvm_scalar_ops() {
        let mut graph = ComputationalGraph::new();
        let a = graph.add_constant(5.0);
        let b = graph.add_constant(2.0);
        let _m = graph.add_binary_op("mod", a, b);
        let _o = graph.add_binary_op("or", a, b);
        let _a = graph.add_binary_op("and", a, b);

        let mut codegen = LLVMCodegen::new();
        let ir = codegen.generate(&graph).expect("IR generation failed");

        assert!(ir.contains("frem double"));
        assert!(ir.contains("and i1"));
        assert!(ir.contains("or i1"));
    }

    #[test]
    fn test_llvm_tensor_constant() {
        let mut graph = ComputationalGraph::new();
        let _t = graph.add_constant_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        
        let mut codegen = LLVMCodegen::new();
        let ir = codegen.generate(&graph).expect("IR generation failed");
        
        assert!(ir.contains("@tensor_data_"));
        assert!(ir.contains("@malloc"));
    }
}
