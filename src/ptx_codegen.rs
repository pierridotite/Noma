use crate::graph::{ComputationalGraph, NodeId, NodeType, Value};
use std::collections::HashMap;

/// PTX code generator (minimal)
/// Emits a simple kernel that computes the last node into out[0]. Supports add/mul/div/pow.
pub struct PTXCodegen {
    reg_count: usize,
}

impl PTXCodegen {
    pub fn new() -> Self {
        PTXCodegen { reg_count: 0 }
    }

    fn fresh(&mut self) -> String {
        let r = format!("%f{}", self.reg_count);
        self.reg_count += 1;
        r
    }

    fn fmt_f64(&self, v: f64) -> String {
        format!("{:.16e}", v)
    }

    pub fn generate(&mut self, graph: &ComputationalGraph) -> Result<String, String> {
        let nodes = graph.nodes();
        let mut ids: Vec<NodeId> = nodes.keys().copied().collect();
        ids.sort_by_key(|id| id.index());

        // Fanout counts to detect outputs
        let mut fanout: HashMap<NodeId, usize> = HashMap::new();
        for id in &ids {
            fanout.insert(*id, 0);
        }
        for id in &ids {
            if let Some(node) = nodes.get(id) {
                for inp in &node.inputs {
                    *fanout.entry(*inp).or_insert(0) += 1;
                }
            }
        }

        let outputs: Vec<NodeId> = ids
            .iter()
            .copied()
            .filter(|id| {
                fanout.get(id).copied().unwrap_or(0) == 0
                    && matches!(nodes[id].node_type, NodeType::BinaryOp(_) | NodeType::UnaryOp(_) | NodeType::FunctionCall(_))
            })
            .collect();

        // Map variables/learnables to input offsets and lengths (scalars => len=1)
        #[derive(Clone, Copy, Debug)]
        struct VarInfo { base_bytes: u64, len: u64 }
        let mut var_offsets: HashMap<NodeId, VarInfo> = HashMap::new();
        let mut next_in_elems = 0u64;
        for id in &ids {
            if let Some(node) = nodes.get(id) {
                match node.node_type {
                    NodeType::Variable(_) | NodeType::Learnable(_) => {
                        let len = match &node.value {
                            Some(Value::Scalar(_)) => 1u64,
                            Some(Value::Tensor(t)) => t.data.len() as u64,
                            None => 1u64,
                        };
                        let info = VarInfo { base_bytes: next_in_elems * 8, len };
                        var_offsets.insert(*id, info);
                        next_in_elems += len;
                    }
                    _ => {}
                }
            }
        }

        let mut reg_map: HashMap<NodeId, String> = HashMap::new();
        self.reg_count = 0;

        // Detect elementwise mode: any variable/learnable with tensor value
        let mut elementwise_len: Option<u64> = None;
        for (_id, info) in &var_offsets {
            if info.len > 1 {
                if let Some(cur) = elementwise_len {
                    if cur != info.len { return Err("PTX: mismatched tensor lengths among inputs".to_string()); }
                } else {
                    elementwise_len = Some(info.len);
                }
            }
        }

        // Pre-count registers (worst case: 5x nodes to cover pow+sigmoid temps)
        let reg_decl = format!(".reg .f64 %f<{}>;\n", ids.len().saturating_mul(5).max(12));

        let mut out = String::new();
        out.push_str("// NOMA PTX backend (minimal)\n");
        out.push_str(".version 7.0\n.target sm_70\n.address_size 64\n\n");
        if elementwise_len.is_some() {
            out.push_str(".visible .entry compute(\n    .param .u64 in_ptr,\n    .param .u64 out_ptr,\n    .param .u32 n_elems\n) {\n");
        } else {
            out.push_str(".visible .entry compute(\n    .param .u64 in_ptr,\n    .param .u64 out_ptr\n) {\n");
        }
            out.push_str("    .reg .pred %p<3>;\n");
        out.push_str("    .reg .u64 %rd<4>;\n");
        if elementwise_len.is_some() {
            out.push_str("    .reg .u64 %rd_idx;\n    .reg .u32 %r_tid;\n    .reg .u32 %r_n;\n");
        }
        out.push_str(&format!("    {}", reg_decl));
        out.push_str("\n    ld.param.u64 %rd0, [in_ptr];\n");
        out.push_str("    ld.param.u64 %rd1, [out_ptr];\n");
        if elementwise_len.is_some() {
            out.push_str("    ld.param.u32 %r_n, [n_elems];\n");
            out.push_str("    mov.u32 %r_tid, %tid.x;\n");
            out.push_str("    setp.ge.u32 %p0, %r_tid, %r_n;\n");
            out.push_str("    @%p0 ret;\n");
            out.push_str("    mul.wide.u32 %rd_idx, %r_tid, 8;\n");
        }

        let mut last: Option<NodeId> = None;

        for id in ids {
            let node = &nodes[&id];
            let dest = self.fresh();
            reg_map.insert(id, dest.clone());
            last = Some(id);

            match &node.node_type {
                NodeType::Constant(val) => {
                    match val {
                        Value::Scalar(v) => {
                            out.push_str(&format!("    mov.f64 {}, {}d0; // const {}\n", dest, self.fmt_f64(*v), self.fmt_f64(*v)));
                        }
                        Value::Tensor(_) => {
                            return Err("Tensor constants are not yet supported in PTX codegen".to_string());
                        }
                    }
                }
                NodeType::Learnable(_) | NodeType::Variable(_) => {
                    let info = *var_offsets.get(&id).ok_or("missing var offset")?;
                    if elementwise_len.is_some() && info.len > 1 {
                        out.push_str(&format!("    add.u64 %rd2, %rd0, {};\n", info.base_bytes));
                        out.push_str("    add.u64 %rd2, %rd2, %rd_idx;\n");
                        out.push_str(&format!("    ld.global.f64 {}, [%rd2]; // load tensor elem\n", dest));
                    } else {
                        out.push_str(&format!("    add.u64 %rd2, %rd0, {};\n", info.base_bytes));
                        out.push_str(&format!("    ld.global.f64 {}, [%rd2]; // load var\n", dest));
                    }
                }
                NodeType::BinaryOp(op) => {
                    if node.inputs.len() != 2 {
                        return Err("Binary op expects 2 inputs".to_string());
                    }
                    let a = reg_map.get(&node.inputs[0]).ok_or("missing left")?;
                    let b = reg_map.get(&node.inputs[1]).ok_or("missing right")?;
                    match op.as_str() {
                        "add" => out.push_str(&format!("    add.f64 {}, {}, {};\n", dest, a, b)),
                        "sub" => out.push_str(&format!("    sub.f64 {}, {}, {};\n", dest, a, b)),
                        "mul" => out.push_str(&format!("    mul.f64 {}, {}, {};\n", dest, a, b)),
                        "div" => out.push_str(&format!("    div.rn.f64 {}, {}, {};\n", dest, a, b)),
                        "mod" => out.push_str(&format!("    rem.rn.f64 {}, {}, {}\n", dest, a, b)),
                        "pow" => {
                            // pow(x,y) ≈ exp2(y * log2(x))
                            let t1 = self.fresh();
                            let t2 = self.fresh();
                            out.push_str(&format!("    lg2.rn.f64 {}, {};\n", t1, a));
                            out.push_str(&format!("    mul.f64 {}, {}, {};\n", t2, t1, b));
                            out.push_str(&format!("    ex2.rn.f64 {}, {};\n", dest, t2));
                        }
                        "and" => {
                            // dest = (a!=0 && b!=0) ? 1.0 : 0.0
                            out.push_str(&format!("    setp.ne.f64 %p0, {}, 0d0;\n", a));
                            out.push_str(&format!("    setp.ne.f64 %p1, {}, 0d0;\n", b));
                            out.push_str("    and.pred %p2, %p0, %p1;\n");
                            out.push_str(&format!("    selp.f64 {}, 1d0, 0d0, %p2;\n", dest));
                        }
                        "or" => {
                            // dest = (a!=0 || b!=0) ? 1.0 : 0.0
                            out.push_str(&format!("    setp.ne.f64 %p0, {}, 0d0;\n", a));
                            out.push_str(&format!("    setp.ne.f64 %p1, {}, 0d0;\n", b));
                            out.push_str("    or.pred %p2, %p0, %p1;\n");
                            out.push_str(&format!("    selp.f64 {}, 1d0, 0d0, %p2;\n", dest));
                        }
                        _ => return Err(format!("Unsupported binary op: {}", op)),
                    }
                }
                NodeType::UnaryOp(op) => {
                    if node.inputs.len() != 1 {
                        return Err("Unary op expects 1 input".to_string());
                    }
                    let a = reg_map.get(&node.inputs[0]).ok_or("missing arg")?;
                    match op.as_str() {
                        "neg" => out.push_str(&format!("    neg.f64 {}, {};\n", dest, a)),
                        _ => return Err(format!("Unsupported unary op: {}", op)),
                    }
                }
                NodeType::FunctionCall(name) => {
                    if node.inputs.len() != 1 {
                        return Err("Function expects 1 input".to_string());
                    }
                    let a = reg_map.get(&node.inputs[0]).ok_or("missing arg")?;
                    match name.as_str() {
                        "relu" => {
                            out.push_str(&format!("    max.f64 {}, {}, 0d0;\n", dest, a));
                        }
                        "sigmoid" => {
                            // sigmoid(x) = 1 / (1 + exp(-x)) using exp2 approximation
                            let t1 = self.fresh();
                            let t2 = self.fresh();
                            let t3 = self.fresh();
                            out.push_str(&format!("    neg.f64 {}, {};\n", t1, a));
                            out.push_str(&format!("    mul.f64 {}, {}, 1.4426950408889634d0;\n", t2, t1)); // log2(e)
                            out.push_str(&format!("    ex2.rn.f64 {}, {};\n", t3, t2));
                            out.push_str(&format!("    add.f64 {}, {}, 1d0;\n", t3, t3));
                            out.push_str(&format!("    div.rn.f64 {}, 1d0, {};\n", dest, t3));
                        }
                        _ => return Err(format!("Unsupported function in PTX backend: {}", name)),
                    }
                }
                NodeType::HeapTensor(name) => {
                    // HeapTensors are not supported in PTX codegen (GPU)
                    return Err(format!("HeapTensor '{}' not supported in PTX codegen", name));
                }
                NodeType::FreedTensor(name) => {
                    return Err(format!("Cannot generate PTX for freed tensor '{}'", name));
                }
            }
        }

        let outs = if outputs.is_empty() { last.into_iter().collect() } else { outputs.clone() };
        if elementwise_len.is_some() {
            if let Some(oid) = outs.last() {
                let reg = reg_map.get(oid).ok_or("missing output reg")?;
                out.push_str("    add.u64 %rd3, %rd1, 0;\n");
                out.push_str("    add.u64 %rd3, %rd3, %rd_idx;\n");
                out.push_str(&format!("    st.global.f64 [%rd3], {};\n", reg));
            }
        } else {
            for (i, oid) in outs.iter().enumerate() {
                let reg = reg_map.get(oid).ok_or("missing output reg")?;
                out.push_str(&format!("    add.u64 %rd3, %rd1, {};\n", (i as u64) * 8));
                out.push_str(&format!("    st.global.f64 [%rd3], {};\n", reg));
            }
        }
        out.push_str("    ret;\n}\n");
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ptx_basic_kernel() {
        let mut graph = ComputationalGraph::new();
        let a = graph.add_constant(2.0);
        let b = graph.add_constant(3.0);
        let mul = graph.add_binary_op("mul", a, b);
        let _pow = graph.add_binary_op("pow", mul, b);

        let mut gen = PTXCodegen::new();
        let code = gen.generate(&graph).expect("ptx generation");
        assert!(code.contains(".entry compute"));
        assert!(code.contains("mul.f64"));
        assert!(code.contains("ex2.rn.f64"));
    }

    #[test]
    fn test_ptx_mod_and_or() {
        let mut graph = ComputationalGraph::new();
        let a = graph.add_constant(5.0);
        let b = graph.add_constant(2.0);
        let _m = graph.add_binary_op("mod", a, b);
        let _o = graph.add_binary_op("or", a, b);

        let mut gen = PTXCodegen::new();
        let code = gen.generate(&graph).expect("ptx generation");
        assert!(code.contains("rem.rn.f64"));
        assert!(code.contains("or.pred"));
    }

    #[test]
    fn test_ptx_tensor_elementwise_header() {
        let mut graph = ComputationalGraph::new();
        let w = graph.add_learnable_tensor("w".to_string(), vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let c = graph.add_constant(2.0);
        let _m = graph.add_binary_op("mul", w, c);

        let mut gen = PTXCodegen::new();
        let code = gen.generate(&graph).expect("ptx generation");
        assert!(code.contains(".param .u32 n_elems"));
        assert!(code.contains("add.u64 %rd2, %rd2, %rd_idx"));
    }
}
